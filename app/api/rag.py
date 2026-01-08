from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.services.rag_service import RAGService
import os

router = APIRouter()
rag_service = RAGService()

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class QueryResponse(BaseModel):
    success: bool
    answer: str
    sources: list
    relevant_chunks: int
    error: Optional[str] = None

@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a question using RAG"""
    try:
        result = rag_service.ask_question(
            question=request.question,
            session_id=request.session_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@router.get("/stats")
async def get_stats():
    """Get RAG system statistics"""
    try:
        # Get document count
        total_documents = rag_service.get_document_count()
        
        # Get total chunks
        total_chunks = 0
        if hasattr(rag_service.vector_store, 'documents') and rag_service.vector_store.documents:
            total_chunks = len(rag_service.vector_store.documents)
        
        # Check if vector store is loaded (has index)
        vector_store_loaded = hasattr(rag_service.vector_store, 'index') and rag_service.vector_store.index is not None
        
        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "vector_store_loaded": vector_store_loaded
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "vector_store_loaded": False
        }

@router.get("/documents/stats")
async def get_documents_stats():
    """Alias for /stats for frontend compatibility"""
    return await get_stats()
