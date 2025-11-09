from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from src.state import State
from src.vectorstore import search_documents
import os


@traceable(name="retrieve_documents")
def retrieve_documents(state: State) -> dict:
    """Retrieve relevant documents from Qdrant."""

    query = state["query"]

    # Retrieve top 3 relevant documents
    docs = search_documents(query, limit=3)
    retrieved_docs = [doc["text"] for doc in docs]

    return {"retrieved_docs": retrieved_docs}


@traceable(
    name="generate_answer", metadata={"model": os.getenv("MODEL_NAME", "unknown")}
)
def generate_answer(state: State) -> dict:
    """Generate answer using retrieved documents and LLM."""

    query = state["query"]
    retrieved_docs = state.get("retrieved_docs", [])

    # Initialize LLM with OpenRouter
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        temperature=0.1,
    )

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
