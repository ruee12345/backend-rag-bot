import os
from typing import List, Dict, Any, Optional
from datetime import date, datetime
from app.services.vector_store import VectorStore
from app.services.pdf_processor import PDFProcessor
from app.core.config import settings
import ollama

# Simple conversation memory store
conversation_store = {}

class RAGService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.pdf_processor = PDFProcessor()
        self.upload_folder = settings.upload_folder
        
        # Create upload folder if it doesn't exist
        os.makedirs(self.upload_folder, exist_ok=True)
    
    def upload_document(self, file_content: bytes, filename: str, file_type: str) -> Dict[str, Any]:
        """Upload and process a document"""
        try:
            # Save file
            file_path = os.path.join(self.upload_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Process document
            result = self.pdf_processor.process_document(file_path, file_type)
            
            # Prepare documents for vector store
            documents_for_store = []
            for chunk in result["chunks"]:
                documents_for_store.append({
                    "text": chunk["text"],
                    "filename": filename,
                    "file_type": file_type,
                    "file_path": file_path,
                    "chunk_id": chunk["chunk_id"],
                    "total_chunks": chunk["total_chunks"]
                })
            
            # Add to vector store
            self.vector_store.add_documents(documents_for_store)
            
            return {
                "success": True,
                "filename": filename,
                "total_chunks": result["total_chunks"],
                "total_characters": result["total_characters"],
                "file_path": file_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def ask_question(self, question: str, k: int = 10, session_id: str = "default") -> Dict[str, Any]:
        """Answer question using RAG with conversation memory"""
        try:
            # Enhanced search for follow-up questions
            search_query = question
            
            # If question is vague, use keywords from conversation history
            vague_indicators = ["those", "they", "them", "it", "this", "that"]
            if any(indicator in question.lower() for indicator in vague_indicators) and session_id in conversation_store:
                # Get last question from history
                history = conversation_store[session_id]
                if history:
                    last_question = history[-1]["question"]
                    # Combine with current question for better search
                    search_query = f"{last_question} {question}"
                    print(f"DEBUG: Enhanced search query: {search_query}")
            
            # Search for relevant documents
            search_results = self.vector_store.search(search_query, k)
            
            if not search_results:
                return {
                    "success": False,
                    "answer": "I couldn't find relevant information in the documents.",
                    "sources": [],
                    "relevant_chunks": 0,
                    "error": "No relevant documents found"
                }

            # Prepare context from search results - GROUPED BY FILENAME
            context_parts = []
            sources_by_file = {}  # Group by filename
            
            for result in search_results:
                context_parts.append(result["document"]["text"])
                
                filename = result["document"].get("filename", "Unknown")
                if filename not in sources_by_file:
                    sources_by_file[filename] = {
                        "filename": filename,
                        "matches": []
                    }
                
                sources_by_file[filename]["matches"].append({
                    "text": result["document"]["text"][:200] + "...",
                    "chunk_id": result["document"].get("chunk_id")
                })
            
            # Convert to list format for response
            sources = []
            for filename, file_data in sources_by_file.items():
                # Take top 3 matches per file
                top_matches = file_data["matches"][:3]
                
                sources.append({
                    "filename": filename,
                    "matches": top_matches
                })
            
            context = "\n\n".join(context_parts)
            
            # Get conversation history
            history_text = ""
            if session_id in conversation_store:
                # Get last 2 Q&A pairs
                for msg in conversation_store[session_id][-2:]:
                    history_text += f"Previous Question: {msg['question']}\nPrevious Answer: {msg['answer'][:150]}...\n\n"
            
            # Generate answer using Ollama with conversation history
            answer = self._generate_answer_with_llm(question, context, history_text)
            
            # Store in conversation memory
            if session_id not in conversation_store:
                conversation_store[session_id] = []
            
            conversation_store[session_id].append({
                "question": question,
                "answer": answer,
                "sources": sources
            })
            
            # Keep only last 5 conversations per session
            if len(conversation_store[session_id]) > 5:
                conversation_store[session_id] = conversation_store[session_id][-5:]
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources,
                "relevant_chunks": len(search_results),
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": [],
                "relevant_chunks": 0,
                "error": str(e)
            }
    
    def _generate_answer_with_llm(self, question: str, context: str, history: str = "") -> str:
        """Generate answer using Ollama LLM with conversation history"""
        from datetime import date
        
        # Get current date for date-sensitive questions
        today = date.today()
        current_date = today.strftime("%B %d, %Y")
        current_day = today.strftime("%A")
        current_year = today.year
        
        prompt = f"""You are an HR Policy Assistant. Answer based STRICTLY on the context below.

IMPORTANT DATE CONTEXT:
- Today is {current_date} ({current_day})
- Current year is {current_year}

CONVERSATION HISTORY:
{history}

CONTEXT FROM HR POLICIES:
{context}

CURRENT QUESTION: {question}

INSTRUCTIONS:
1. Use today's date ({current_date}) to answer date-sensitive questions like holidays
2. For questions about office timings:
   - Look for information about "shift duration", "break", and "login hours"
   - The context mentions: "The total shift duration is of 9.30 hours. There would be a break of 1 hour. Hence, minimum login hours & timesheet required on KEKA portal will be of 8 hours"
   - Company allows flexible login: Employees may clock in at any time but have to complete 8 hours of login every day
   - Week-offs are observed on all Sundays and Saturdays
3. For questions about specific policies, extract the exact details from the context
4. If context explicitly states something is prohibited, mention that clearly
5. If question refers to previous conversation (like "those", "they", "them", "it"), check the HISTORY
6. Answer ONLY using information from context or history
7. If context doesn't contain the answer, say "Based on the provided documents, I don't have information about this"
8. For holiday questions: Check if {current_date} matches any holiday dates in the context

ANSWER:"""
        
        try:
            response = ollama.generate(
                model="llama3.2",
                prompt=prompt,
                options={
                    'temperature': 0.1,
                    'num_predict': 512
                }
            )
            return response['response']
        except Exception as e:
            print(f"Ollama error: {e}")
            return f"Based on the HR policies, here's what I found:\n\n{context[:1000]}"
    
    def get_document_count(self) -> int:
        """Get number of UNIQUE documents in vector store"""
        if not self.vector_store.documents:
            return 0
        
        # Count unique filenames
        filenames = set()
        for doc in self.vector_store.documents:
            filename = doc.get("filename")
            if filename:
                filenames.add(filename)
        
        return len(filenames)
    
    def clear_documents(self) -> bool:
        """Clear all documents from vector store"""
        try:
            self.vector_store.clear()
            return True
        except Exception as e:
            print(f"Error clearing documents: {e}")
            return False

    def delete_document(self, filename: str) -> bool:
        """Delete a specific document by filename"""
        try:
            # Use vector store's selective deletion
            success = self.vector_store.remove_document(filename)
            
            # Also delete the physical file
            file_path = os.path.join(self.upload_folder, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted physical file: {file_path}")
            
            return success
        except Exception as e:
            print(f"Error deleting document {filename}: {e}")
            return False
