import os
import time
import json
import tempfile
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


def get_qdrant_client():
    """Get Qdrant client instance."""
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


def get_embeddings():
    """Get OpenAI embeddings instance."""
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def create_collection_if_not_exists():
    """Create Qdrant collection if it doesn't exist."""
    client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    collections = client.get_collections().collections
    if not any(col.name == collection_name for col in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )
        print(f"Created collection: {collection_name}")


def generate_embeddings_batch(
    texts: list[str], model: str = "text-embedding-3-large"
) -> list[list[float]]:
    """
    Generate embeddings using OpenAI's batch API for efficiency.

    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model to use

    Returns:
        List of embedding vectors in the same order as input texts
    """
    if not texts:
        return []

    # For small batches, use regular API
    if len(texts) <= 5:
        print(f"  Using standard API for {len(texts)} embeddings (small batch)...")
        embeddings = get_embeddings()
        vectors = embeddings.embed_documents(texts)
        print(f"  ✓ Standard embeddings completed")
        return vectors

    print(f"\n{'='*60}")
    print(f"BATCH EMBEDDING PROCESS")
    print(f"{'='*60}")
    print(f"Total texts to embed: {len(texts)}")
    print(f"Model: {model}")

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create temporary JSONL file for batch input
    print(f"\n[1/5] Creating batch input file...")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as f:
        input_file_path = f.name
        for idx, text in enumerate(texts):
            payload = {
                "custom_id": f"req_{idx}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": model, "input": text, "encoding_format": "float"},
            }
            f.write(json.dumps(payload) + "\n")
    print(f"      ✓ Created JSONL file with {len(texts)} requests")

    try:
        # Upload the file
        print(f"\n[2/5] Uploading batch file to OpenAI...")
        with open(input_file_path, "rb") as f:
            file_obj = openai_client.files.create(file=f, purpose="batch")
        input_file_id = file_obj.id
        print(f"      ✓ File uploaded: {input_file_id}")

        # Create batch job
        print(f"\n[3/5] Creating batch job...")
        batch_job = openai_client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )
        batch_id = batch_job.id
        print(f"      ✓ Batch job created: {batch_id}")

        # Poll until completion
        print(f"\n[4/5] Processing batch (polling every 10s)...")
        start_time = time.time()
        last_completed = 0

        while True:
            job = openai_client.batches.retrieve(batch_id)
            completed = job.request_counts.completed or 0
            failed = job.request_counts.failed or 0
            total = job.request_counts.total or 0

            elapsed = int(time.time() - start_time)
            progress_pct = (completed / total * 100) if total > 0 else 0

            # Show progress bar
            bar_length = 30
            filled = int(bar_length * completed / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_length - filled)

            print(
                f"      [{bar}] {progress_pct:.1f}% | {completed}/{total} | {elapsed}s elapsed | Status: {job.status}",
                end="\r",
            )

            if completed > last_completed:
                last_completed = completed

            if job.status in ("completed", "failed", "cancelled"):
                print()  # New line after progress bar
                break

            time.sleep(10)  # Check every 10 seconds

        if job.status != "completed":
            print(f"      ✗ Batch job failed with status: {job.status}")
            raise RuntimeError(f"Batch job failed with status: {job.status}")

        print(f"      ✓ Batch processing completed in {elapsed}s")
        if failed > 0:
            print(f"      ⚠ Warning: {failed} requests failed")

        # Download results
        print(f"\n[5/5] Downloading and parsing results...")
        output_file_id = job.output_file_id
        output_content = openai_client.files.content(output_file_id).text

        # Parse results and sort by custom_id to maintain order
        results = []
        for line in output_content.splitlines():
            if not line:
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id")
            idx = int(custom_id.split("_")[1])
            embedding = entry["response"]["body"]["data"][0]["embedding"]
            results.append((idx, embedding))

        # Sort by index and return embeddings in original order
        results.sort(key=lambda x: x[0])
        embeddings = [emb for _, emb in results]

        print(f"      ✓ Retrieved {len(embeddings)} embeddings")
        print(f"\n{'='*60}")
        print(f"BATCH EMBEDDING COMPLETE")
        print(f"{'='*60}\n")

        return embeddings

    finally:
        # Clean up temp file
        if os.path.exists(input_file_path):
            os.unlink(input_file_path)


def add_documents_to_qdrant(documents: list[dict]):
    """
    Add documents to Qdrant with custom structure.

    Expected document structure:
    {
        "text": str,
        "file_name": str,
        "file_ext": str,
        "tags": list[str],
        "chunk_id": int,
        "total_chunks": int,
        "created": str (ISO timestamp)
    }
    """
    client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    # Generate embeddings using batch API
    texts = [doc["text"] for doc in documents]
    vectors = generate_embeddings_batch(texts)

    # Create points
    points = []
    for i, (doc, vector) in enumerate(zip(documents, vectors)):
        point = PointStruct(
            id=i,  # Will be auto-generated by Qdrant if not provided
            vector=vector,
            payload={
                "text": doc["text"],
                "file_name": doc["file_name"],
                "file_ext": doc["file_ext"],
                "tags": doc["tags"],
                "chunk_id": doc["chunk_id"],
                "total_chunks": doc["total_chunks"],
                "created": doc["created"],
            },
        )
        points.append(point)

    # Upload to Qdrant
    client.upload_points(collection_name=collection_name, points=points)

    return len(points)


def search_documents(query: str, limit: int = 3) -> list[dict]:
    """
    Search for documents in Qdrant.

    Returns list of documents with structure:
    {
        "text": str,
        "file_name": str,
        "file_ext": str,
        "tags": list[str],
        "chunk_id": int,
        "total_chunks": int,
        "created": str,
        "score": float
    }
    """
    client = get_qdrant_client()
    embeddings = get_embeddings()
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")

    # Generate query embedding
    query_vector = embeddings.embed_query(query)

    # Search
    results = client.search(
        collection_name=collection_name, query_vector=query_vector, limit=limit
    )

    # Format results
    documents = []
    for result in results:
        doc = {
            "text": result.payload["text"],
            "file_name": result.payload["file_name"],
            "file_ext": result.payload["file_ext"],
            "tags": result.payload["tags"],
            "chunk_id": result.payload["chunk_id"],
            "total_chunks": result.payload["total_chunks"],
            "created": result.payload["created"],
            "score": result.score,
        }
        documents.append(doc)

    return documents
