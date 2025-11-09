"""Example tool - template for adding new tools."""

from langchain_core.tools import tool


@tool
def example_calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Use this tool when you need to perform calculations.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")

    Returns:
        The result of the calculation
    """
    try:
        # WARNING: eval() is used here for demonstration only
        # In production, use a proper math parser like sympy
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


# To add this tool to the agent:
# 1. Import it in src/tools/__init__.py
# 2. Add it to the ALL_TOOLS list
