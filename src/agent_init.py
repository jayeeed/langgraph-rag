"""Initialize LLM agent with OpenRouter configuration."""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm() -> ChatOpenAI:
    """
    Initialize and return ChatOpenAI instance with OpenRouter configuration.

    Returns:
        Configured ChatOpenAI instance
    """
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv("MODEL_NAME"),
        temperature=float(os.getenv("TEMPERATURE")),
        max_tokens=int(os.getenv("MAX_TOKENS")),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    )
    return llm
