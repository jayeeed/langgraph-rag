"""Generate tags for document chunks using LLM."""

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json


def generate_tags(text: str) -> list[str]:
    """
    Generate 3 relevant tags for a text chunk using LLM.

    Args:
        text: The text chunk to generate tags for

    Returns:
        List of 3 tags
    """
    # Initialize LLM with OpenRouter
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
        temperature=0.1,
    )

    # Create prompt
    messages = [
        SystemMessage(
            content="""You are a tagging assistant. Generate exactly 3 relevant, concise tags for the given text.
Tags should be:
- Single words only
- Lowercase
- Descriptive of the main topics/concepts
- Separated by commas

Respond ONLY with the 3 tags separated by commas, nothing else.
Example: machine learning, python, data science"""
        ),
        HumanMessage(content=f"Generate 3 tags for this text:\n\n{text[:500]}"),
    ]

    try:
        # Generate tags
        response = llm.invoke(messages)
        tags_text = response.content.strip()

        # Parse tags
        tags = [tag.strip() for tag in tags_text.split(",")]

        # Ensure we have exactly 3 tags
        if len(tags) < 3:
            tags.extend(["general"] * (3 - len(tags)))
        elif len(tags) > 3:
            tags = tags[:3]

        return tags

    except Exception as e:
        print(f"Error generating tags: {e}")
        return ["general", "document", "content"]


def generate_tags_batch(texts: list[str], batch_size: int = 10) -> list[list[str]]:
    """
    Generate tags for multiple text chunks in batches.

    Args:
        texts: List of text chunks
        batch_size: Number of texts to process at once

    Returns:
        List of tag lists
    """
    all_tags = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"Generating tags for chunks {i+1}-{min(i+batch_size, len(texts))}...")

        for text in batch:
            tags = generate_tags(text)
            all_tags.append(tags)

    return all_tags
