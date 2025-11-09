"""RAG tool for retrieving and answering questions from documents."""

from langchain_core.tools import tool
from src.vectorstore import search_documents


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for relevant information and return context.

    Use this tool when you need to answer questions about documents in the knowledge base.

    Args:
        query: The search query or question to find relevant information for

    Returns:
        Relevant context from the knowledge base
    """
    # Retrieve relevant documents
    docs = search_documents(query, limit=3)

    if not docs:
        return "No relevant information found in the knowledge base."

    # Format context
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(
            f"[Source {i}] (from {doc['file_name']}, tags: {', '.join(doc['tags'])})\n{doc['text']}"
        )

    return "\n\n---\n\n".join(context_parts)
