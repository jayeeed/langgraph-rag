"""State definition for the agent."""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State for the agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]
