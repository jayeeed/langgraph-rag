from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from src.state import State
from src.vectorstore import search_documents
from src.agent_init import get_llm
import os


@traceable(name="retrieve_documents")
def retrieve_documents(state: State) -> dict:
    """Retrieve relevant documents from Qdrant."""

    query = state["query"]

    # Retrieve top 3 relevant documents
    docs = search_documents(query, limit=3)
    retrieved_docs = [doc["text"] for doc in docs]

    return {"retrieved_docs": retrieved_docs}


@traceable(name="generate_answer", metadata={"model": os.getenv("MODEL_NAME")})
def generate_answer(state: State) -> dict:
    """Generate answer using retrieved documents and LLM."""

    query = state["query"]
    retrieved_docs = state.get("retrieved_docs", [])

    # Initialize LLM with OpenRouter
    llm = get_llm()

    # Format context from retrieved documents
    context = "\n\n".join(
        [f"Document {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)]
    )

    # Create prompt
    messages = [
        SystemMessage(
            content="You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain relevant information, say so."
        ),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"),
    ]

    # Generate response
    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "messages": [HumanMessage(content=query), response],
    }
