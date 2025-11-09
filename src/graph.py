from langgraph.graph import StateGraph, START, END
from src.state import State
from src.nodes import retrieve_documents, generate_answer


def create_graph():
    """Create and compile the RAG graph."""

    # Initialize graph
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)

    # Add edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile graph
    return workflow.compile()


# Export the compiled graph
graph = create_graph()
