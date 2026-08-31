import os
from typing import List, Dict, Any
from app.services.vector_store import VectorStore
from app.services.pdf_processor import PDFProcessor
import ollama

class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.pdf_processor = PDFProcessor()
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama2")
    
    def add_document(self, file_path: str) -> int:
        chunks = self.pdf_processor.process_document(file_path)
        if chunks:
            self.vector_store.add_documents(chunks)
        return len(chunks)
    
    def ask(self, query: str, k: int = 5) -> Dict[str, Any]:
        results = self.vector_store.search(query, k)
        
        if not results:
            return {
                "answer": "I don't have enough information to answer this question. Please upload relevant documents first.",
                "sources": []
            }
        
        context = "\n\n".join([result['text'] for result in results])
        sources = [
            {
                "text": result['text'][:200] + "...",
                "filename": result['filename'],
                "chunk_id": result['chunk_id']
            }
            for result in results
        ]
        
        try:
            prompt = f"""You are an HR compliance assistant. Answer the following question based ONLY on the provided context.

Context:
{context}

Question: {query}

Answer:"""
            
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt
            )
            
            return {
                "answer": response['response'],
                "sources": sources
            }
        except Exception as e:
            print(f"Ollama error: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}. Please check if Ollama is running.",
                "sources": sources
            }
    
    def get_document_count(self) -> int:
        return self.vector_store.get_document_count()
