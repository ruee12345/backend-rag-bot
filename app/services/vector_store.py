import os
import uuid
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStore:
    def __init__(self, persist_dir: str = "data/vector_store"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.client.get_or_create_collection(
                name="hr_documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Load embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"✅ VectorStore ready with {self.collection.count()} documents")
        except Exception as e:
            print(f"❌ VectorStore initialization error: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        if not documents:
            return
        
        try:
            ids = []
            texts = []
            metadatas = []
            
            for doc in documents:
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                texts.append(doc['text'])
                metadatas.append({
                    'filename': doc.get('filename', 'unknown'),
                    'chunk_id': doc.get('chunk_id', 0),
                })
            
            embeddings = self.embedding_model.encode(texts).tolist()
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            print(f"✅ Added {len(documents)} documents")
        except Exception as e:
            print(f"❌ Error adding documents: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            
            formatted = []
            if results['documents']:
                for i in range(len(results['documents'][0])):
                    formatted.append({
                        'text': results['documents'][0][i],
                        'filename': results['metadatas'][0][i]['filename'],
                        'chunk_id': results['metadatas'][0][i]['chunk_id'],
                        'distance': results['distances'][0][i]
                    })
            return formatted
        except Exception as e:
            print(f"❌ Error searching: {e}")
            return []
    
    def get_document_count(self) -> int:
        try:
            return self.collection.count()
        except:
            return 0
    
    def delete_all(self):
        try:
            self.client.delete_collection("hr_documents")
            self.collection = self.client.create_collection("hr_documents")
            print("✅ All documents deleted")
        except Exception as e:
            print(f"❌ Error deleting: {e}")
