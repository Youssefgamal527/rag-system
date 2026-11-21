from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.application import RAGApplication


rag_app = RAGApplication()


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_app.initialize()
    yield


app = FastAPI(
    title="RAG System API",
    description="Document Question Answering API",
    version="1.0.0",
    lifespan=lifespan
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "RAG API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "initialized": rag_app.initialized,
        "documents_count": rag_app.documents_count
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:
        result = rag_app.ask(request.question)

        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "budget_estimates": result.get("budget_estimates"),
            "tuition_estimate": result.get("tuition_estimate")
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )