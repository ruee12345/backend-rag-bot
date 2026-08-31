import os
from typing import List, Dict, Any
from app.services.vector_store import VectorStore
from app.services.pdf_processor import PDFProcessor
from openai import OpenAI

class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.pdf_processor = PDFProcessor()
        
        # Initialize DeepSeek client (OpenAI-compatible)
        self.client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    
    def add_document(self, file_path: str) -> int:
        chunks = self.pdf_processor.process_document(file_path)
        if chunks:
            self.vector_store.add_documents(chunks)
        return len(chunks)
    
    def ask(self, query: str, k: int = 5) -> Dict[str, Any]:
        # Search for relevant documents
        results = self.vector_store.search(query, k)
        
        if not results:
            return {
                "answer": "I don't have enough information to answer this question. Please upload relevant documents first.",
                "sources": []
            }
        
        # Build context from search results
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
            # Generate answer using DeepSeek
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an HR compliance assistant. Answer the following question based ONLY on the provided context.
                        
                        Rules:
                        1. Only use information from the context provided.
                        2. If the answer is not in the context, say "I don't have enough information to answer this."
                        3. Be concise and professional.
                        4. Cite specific sources when possible."""
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
                "answer": response.choices[0].message.content,
                "sources": sources
            }
        except Exception as e:
            print(f"DeepSeek error: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}. Please check your DeepSeek API key.",
                "sources": sources
            }
    
    def get_document_count(self) -> int:
        return self.vector_store.get_document_count()
