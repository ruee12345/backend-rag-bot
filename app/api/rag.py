from fastapi import APIRouter, HTTPException
from app.services.rag_service import RAGService
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/rag", tags=["rag"])

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

@router.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        rag_service = RAGService()
        result = rag_service.ask(request.question)
        
        # Check if result is a dict with 'answer' key
        if isinstance(result, dict) and "answer" in result:
            return {
                "answer": result["answer"]
            }
        else:
            # If result is just a string, return it directly
            return {
                "answer": str(result)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/stats")
async def get_stats():
    try:
        rag_service = RAGService()
        total_documents = rag_service.get_document_count()
        return {
            "total_documents": total_documents,
            "vector_store_loaded": total_documents > 0
        }
    except Exception as e:
        return {
            "total_documents": 0,
            "vector_store_loaded": False,
            "error": str(e)
        }