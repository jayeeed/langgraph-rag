"""FastAPI server for the RAG application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from src.graph import graph
from src.vectorstore import create_collection_if_not_exists
from langsmith import traceable, Client
from langsmith.run_helpers import get_current_run_tree
import uvicorn
import os

# Load environment variables
load_dotenv()

# Initialize LangSmith client for feedback
langsmith_client = Client()

# Initialize FastAPI app
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    """Initialize Qdrant collection on startup."""
    print("Checking Qdrant collection...")
    try:
        create_collection_if_not_exists()
        collection_name = os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")
        print(f"✓ Qdrant collection '{collection_name}' is ready")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize Qdrant collection: {e}")


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources_count: int
    retrieved_docs: list[str] = []
    run_id: str = None


class FeedbackRequest(BaseModel):
    run_id: str
    score: float  # 1.0 for thumbs up, 0.0 for thumbs down
    comment: str = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "RAG API is running"}


@app.post("/chat", response_model=ChatResponse)
@traceable(name="Chat", metadata={"endpoint": "/chat"})
async def chat(request: ChatRequest):
    """Chat endpoint that processes queries using RAG."""

    try:
        # Get current run tree to capture run_id
        run_tree = get_current_run_tree()
        run_id = str(run_tree.id) if run_tree else None

        # Run the graph
        result = graph.invoke(
            {
                "query": request.query,
                "messages": [],
                "retrieved_docs": [],
                "answer": "",
            },
            {"metadata": {"user_query": request.query}},
        )

        return {
            "query": request.query,
            "answer": result["answer"],
            "sources_count": len(result["retrieved_docs"]),
            "retrieved_docs": result["retrieved_docs"],
            "run_id": run_id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for a specific run."""

    try:
        langsmith_client.create_feedback(
            run_id=feedback.run_id,
            key="user_feedback",
            score=feedback.score,
            comment=feedback.comment,
        )

        return {"status": "success", "message": "Feedback submitted"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
