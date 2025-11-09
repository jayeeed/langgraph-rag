from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class State(TypedDict):
    """Graph state for RAG application."""
    messages: Annotated[list, add_messages]
    query: str
    retrieved_docs: list[str]
    answer: str
