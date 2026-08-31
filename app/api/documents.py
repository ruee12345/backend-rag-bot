from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.rag_service import RAGService
import os

# ✅ Add prefix and tags here
router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, DOCX, or TXT) to the RAG system"""
    try:
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        rag_service = RAGService()
        chunk_count = rag_service.add_document(file_path)
        os.remove(file_path)
        
        return {
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "chunks_created": chunk_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/count")
async def get_document_count():
    try:
        rag_service = RAGService()
        count = rag_service.get_document_count()
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.rag_service import RAGService
import os

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, DOCX, or TXT) to the RAG system"""
    try:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save the uploaded file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the document using RAG service
        rag_service = RAGService()
        chunk_count = rag_service.add_document(file_path)
        
        # Clean up
        os.remove(file_path)
        
        return {
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "chunks_created": chunk_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/count")
async def get_document_count():
    """Get the total number of documents in the vector store"""
    try:
        rag_service = RAGService()
        count = rag_service.get_document_count()
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
