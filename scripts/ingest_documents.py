"""Script to ingest various document types (PDF, DOCX, MD, TXT) into Qdrant."""

import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.vectorstore import create_collection_if_not_exists, add_documents_to_qdrant
from src.tag_generator import generate_tags_batch


def load_document(file_path: str):
    """Load a single document based on its extension."""

    file_extension = Path(file_path).suffix.lower()

    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".md": UnstructuredMarkdownLoader,
        ".txt": TextLoader,
    }

    loader_class = loaders.get(file_extension)
    if not loader_class:
        print(f"Unsupported file type: {file_extension}")
        return []

    try:
        loader = loader_class(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


@traceable(name="ingest_documents", metadata={"operation": "document_ingestion"})
def ingest_documents(directory_path: str = "data"):
    """Ingest documents from a directory into Qdrant."""

    # Create collection if it doesn't exist
    print("Creating Qdrant collection if not exists...")
    create_collection_if_not_exists()

    # Supported file extensions
    supported_extensions = [".pdf", ".docx", ".doc", ".md", ".txt"]

    # Find all supported files
    data_path = Path(directory_path)
    if not data_path.exists():
        print(f"Directory {directory_path} does not exist!")
        return

    all_files = []
    for ext in supported_extensions:
        all_files.extend(data_path.rglob(f"*{ext}"))

    if not all_files:
        print(f"No supported files found in {directory_path}")
        return

    print(f"Found {len(all_files)} files to process\n")

    # Process each file
    all_documents = []

    for file_path in all_files:
        print(f"Processing: {file_path.name}")

        # Load document
        docs = load_document(str(file_path))
        if not docs:
            continue

        # Combine all pages/sections into one text
        full_text = "\n\n".join([doc.page_content for doc in docs])

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(full_text)

        print(f"  Created {len(chunks)} chunks")

        # Generate tags for all chunks
        print(f"  Generating tags...")
        all_tags = generate_tags_batch(chunks)

        # Create document structure for each chunk
        file_name = file_path.name
        file_ext = file_path.suffix.lower().replace(".", "")
        created = datetime.now(timezone(timedelta(hours=6))).isoformat()

        for i, (chunk, tags) in enumerate(zip(chunks, all_tags)):
            doc = {
                "text": chunk,
                "file_name": file_name,
                "file_ext": file_ext,
                "tags": tags,
                "chunk_id": i + 1,
                "total_chunks": len(chunks),
                "created": created,
            }
            all_documents.append(doc)

        print(f"  ✓ Processed {file_name}\n")

    if not all_documents:
        print("No documents were processed successfully")
        return

    # Add to Qdrant
    print(f"Adding {len(all_documents)} chunks to Qdrant...")
    count = add_documents_to_qdrant(all_documents)

    print(f"\n✓ Successfully ingested {count} document chunks into Qdrant")
    print(f"  Collection: {os.getenv('QDRANT_COLLECTION_NAME', 'rag')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument(
        "--dir",
        type=str,
        default="data",
        help="Directory containing documents to ingest (default: data)",
    )

    args = parser.parse_args()
    ingest_documents(args.dir)
