"""Simple test script for the RAG API."""

import requests
import json


def test_health():
    """Test health check endpoint."""
    response = requests.get("http://localhost:8000/")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()


def test_chat(query: str):
    """Test chat endpoint - retrieves docs from Qdrant and uses LLM to generate answer."""
    response = requests.post("http://localhost:8000/chat", json={"query": query})

    if response.status_code == 200:
        result = response.json()
        print(f"Query: {result['query']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\n(Used {result['sources_count']} source documents)")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    print("Testing RAG API\n" + "=" * 50 + "\n")

    # Test health
    test_health()

    # Test chat
    test_chat("jayed's skills?")
