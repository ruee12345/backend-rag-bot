import os
from typing import List, Dict, Any
from app.services.vector_store import VectorStore
from app.services.pdf_processor import PDFProcessor
from groq import Groq

class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.pdf_processor = PDFProcessor()
        
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )
        self.model = os.environ.get("GROQ_MODEL", "groq/compound")

    def add_document(self, file_path: str) -> int:
        chunks = self.pdf_processor.process_document(file_path)
        if chunks:
            self.vector_store.add_documents(chunks)
        return len(chunks)
    
    def ask(self, query: str, k: int = 3) -> Dict[str, Any]:
        results = self.vector_store.search(query, k)
        
        if not results:
            return {
                "success": False,
                "answer": "I don't have enough information to answer this question.",
                "sources": []
            }
        
        context_parts = []
        sources = []
        total_chars = 0
        max_context_chars = 2000
        
        for result in results:
            text = result['text']
            if total_chars + len(text) > max_context_chars:
                remaining = max_context_chars - total_chars
                if remaining > 100:
                    context_parts.append(text[:remaining] + "...")
                    sources.append({
                        "text": text[:200] + "...",
                        "filename": result['filename'],
                        "chunk_id": result['chunk_id']
                    })
                break
            context_parts.append(text)
            sources.append({
                "text": text[:200] + "...",
                "filename": result['filename'],
                "chunk_id": result['chunk_id']
            })
            total_chars += len(text)
        
        if not context_parts and results:
            text = results[0]['text']
            context_parts.append(text[:1500])
            sources.append({
                "text": text[:200] + "...",
                "filename": results[0]['filename'],
                "chunk_id": results[0]['chunk_id']
            })
        
        context = "\n\n".join(context_parts)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an HR compliance assistant. Answer based ONLY on the context provided. Be concise and accurate."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}"
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return {
                "success": True,
                "answer": response.choices[0].message.content,
                "sources": sources
            }
        except Exception as e:
            return {
                "success": False,
                "answer": f"Error: {str(e)}",
                "sources": sources
            }
    
    def get_document_count(self) -> int:
        return self.vector_store.get_document_count()
