import os
import uuid
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, persist_dir: str = "data/vector_store"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name="hr_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✅ VectorStore ready with {self.collection.count()} documents")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        if not documents:
            return
        
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
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
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
    
    def get_document_count(self) -> int:
        return self.collection.count()
    
    def delete_all(self):
        self.client.delete_collection("hr_documents")
        self.collection = self.client.create_collection("hr_documents")
    
    def remove_document(self, filename: str) -> bool:
        results = self.collection.get(where={"filename": filename})
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            return True
        return False
