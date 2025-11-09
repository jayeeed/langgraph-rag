"""Tools for the agent."""

from src.tools.rag_tool import search_knowledge_base
from src.tools.stock_tool import stock_info

# List of all available tools
ALL_TOOLS = [
    search_knowledge_base,
    stock_info,
]

__all__ = ["ALL_TOOLS", "search_knowledge_base", "stock_info"]
