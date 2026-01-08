from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, List
import shutil
import os
from datetime import datetime
from app.services.rag_service import RAGService
from app.core.config import settings

router = APIRouter()
rag_service = RAGService()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None)
):
    """Upload a document"""
    try:
        # Save file temporarily
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process file
        file_type = file.filename.split('.')[-1].lower()
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        result = rag_service.upload_document(file_content, file.filename, file_type)
        
        # Cleanup
        os.remove(file_path)
        
        if result["success"]:
            return {
                "message": "Document uploaded successfully",
                "filename": result["filename"],
                "category": category,
                "total_chunks": result["total_chunks"],
                "file_size": result["total_characters"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        total_documents = rag_service.get_document_count()
        total_chunks = len(rag_service.vector_store.documents) if rag_service.vector_store.documents else 0
        
        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "vector_store_loaded": total_documents > 0,
            "upload_folder": settings.upload_folder,
            "vector_store_path": settings.vector_store_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def list_documents():
    """List all uploaded documents with details"""
    try:
        documents = []
        
        if rag_service.vector_store.documents:
            # Get unique filenames
            filenames = set()
            for doc in rag_service.vector_store.documents:
                filename = doc.get("filename")
                if filename:
                    filenames.add(filename)
            
            # Get details for each document
            for filename in filenames:
                # Count chunks for this document
                chunks = 0
                file_type = ""
                file_path = ""
                for doc in rag_service.vector_store.documents:
                    if doc.get("filename") == filename:
                        chunks += 1
                        file_type = doc.get("file_type", file_type)
                        file_path = doc.get("file_path", file_path)
                
                # Get file info
                file_size = 0
                uploaded_at = None
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    uploaded_at = datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                
                documents.append({
                    "filename": filename,
                    "chunks": chunks,
                    "file_type": file_type,
                    "file_size": file_size,
                    "uploaded_at": uploaded_at,
                    "file_path": file_path
                })
        
        return {
            "total_documents": len(documents),
            "documents": sorted(documents, key=lambda x: x.get("uploaded_at", ""), reverse=True)
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{filename}")
async def delete_document(filename: str):
    """Delete a document by filename"""
    try:
        import urllib.parse
        filename = urllib.parse.unquote(filename)
        
        print(f"Attempting to delete document: {filename}")
        
        # Use selective deletion
        success = rag_service.delete_document(filename)
        
        if success:
            return {
                "success": True,
                "message": f"Document '{filename}' deleted successfully",
                "note": "Document was selectively removed from vector store"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found or could not be deleted")
            
    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))@router.post("/clear")
async def clear_documents():
    """Clear all documents"""
    try:
        success = rag_service.clear_documents()
        if success:
            return {"message": "All documents cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear documents")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
