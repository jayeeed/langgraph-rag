"""Agent graph with tool calling capabilities."""

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from src.agent_state import AgentState
from src.agent_init import get_llm
from src.tools import ALL_TOOLS


def create_agent_graph():
    """Create and compile the agent graph with tools."""

    # Initialize LLM with tools
    llm = get_llm()

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Define chat initialization node
    def chat_init(state: AgentState) -> dict:
        """Initialize chat with system message."""
        messages = state["messages"]

        # Add system message at the beginning
        system_message = SystemMessage(
            content="""You are a helpful AI assistant with access to a knowledge base.

When users ask questions:
1. Use the search_knowledge_base tool to find relevant information from documents
2. Use the stock_info tool when users ask about stock prices or market data
3. Provide clear, accurate answers based on the retrieved context
4. If the knowledge base doesn't contain relevant information, say so clearly
5. Cite sources when providing information

Be conversational and helpful."""
        )

        return {"messages": [system_message] + messages}

    # Define agent node
    def agent(state: AgentState) -> dict:
        """Agent node that decides whether to use tools or respond."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Create tool node
    tool_node = ToolNode(ALL_TOOLS)

    # Build graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("chat_init", chat_init)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge(START, "chat_init")
    workflow.add_edge("chat_init", "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")

    # Compile
    return workflow.compile()


# Export the compiled graph
agent_graph = create_agent_graph()
