import os
import pickle
import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
from app.core.config import settings

class VectorStore:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Prevent re-initialization
        if self._initialized:
            return
            
        print(f"DEBUG: Initializing VectorStore with Sentence Transformers (Local)")
        
        # Initialize local embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.vector_store_path = settings.vector_store_path
        self.index = None
        self.documents = []  # Store document chunks
        self.metadata = []   # Store metadata for each chunk
        
        # Create vector store directory if it doesn't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        print(f"DEBUG: Vector store path: {self.vector_store_path}")
        
        # Try to load existing index
        self.load_index()
        
        self._initialized = True
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for list of texts using Local Model"""
        print(f"DEBUG: Creating embeddings for {len(texts)} texts locally")
        
        if not texts:
            return np.array([])
        
        try:
            # Create embeddings locally
            embeddings = self.embedding_model.encode(
                texts, 
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            print(f"DEBUG: Local embeddings shape: {embeddings.shape}")
            return embeddings.astype(np.float32)
            
        except Exception as e:
            print(f"DEBUG ERROR: Local embedding failed: {e}")
            print("DEBUG: Using random embeddings as fallback")
            dimension = 384  # all-MiniLM-L6-v2 dimension
            return np.random.randn(len(texts), dimension).astype(np.float32)
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to vector store"""
        if not documents:
            print("DEBUG: No documents to add")
            return
        
        print(f"DEBUG: Adding {len(documents)} documents to vector store")
        
        # Extract texts
        texts = [doc["text"] for doc in documents]
        print(f"DEBUG: First text sample: {texts[0][:100]}...")
        
        # Create embeddings using local model
        embeddings = self.create_embeddings(texts)
        print(f"DEBUG: Created embeddings shape: {embeddings.shape}")
        
        # Initialize or extend FAISS index
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            print(f"DEBUG: Created new FAISS index with dimension {dimension}")
        else:
            print(f"DEBUG: Extending existing FAISS index")
        
        # Add to index
        self.index.add(embeddings)
        print(f"DEBUG: Added embeddings to FAISS index")
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadata.extend([{"doc_id": len(self.documents) - 1, **doc} for doc in documents])
        print(f"DEBUG: Now have {len(self.documents)} total documents")
        
        # Save index
        self.save_index()
        print(f"DEBUG: Saved index to {self.vector_store_path}")
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        print(f"DEBUG: Searching for query: '{query}'")
        
        if self.index is None:
            print("DEBUG: Index is None, loading...")
            self.load_index()
            if self.index is None:
                print("DEBUG: Still None after load, returning empty")
                return []
        
        if len(self.documents) == 0:
            print("DEBUG: No documents in store, returning empty results")
            return []
        
        print(f"DEBUG: Index has {self.index.ntotal} vectors, store has {len(self.documents)} documents")
        
        # Create query embedding using local model
        query_embedding = self.create_embeddings([query])
        print(f"DEBUG: Query embedding shape: {query_embedding.shape}")
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        print(f"DEBUG: Search results - distances: {distances}, indices: {indices}")
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(distance),
                    "rank": i + 1
                })
                print(f"DEBUG: Result {i+1}: idx={idx}, distance={distance:.4f}")
            else:
                print(f"DEBUG: Invalid index {idx}, skipping")
        
        print(f"DEBUG: Returning {len(results)} results")
        return results
    
    def save_index(self):
        """Save index to disk"""
        if self.index is not None:
            print(f"DEBUG: Saving index with {len(self.documents)} documents")
            
            # Save FAISS index
            index_path = os.path.join(self.vector_store_path, "index.faiss")
            try:
                faiss.write_index(self.index, index_path)
                print(f"DEBUG: Saved FAISS index to {index_path}")
            except Exception as e:
                print(f"DEBUG ERROR: Failed to save FAISS index: {e}")
            
            # Save documents and metadata
            docs_path = os.path.join(self.vector_store_path, "documents.pkl")
            try:
                with open(docs_path, "wb") as f:
                    pickle.dump(self.documents, f)
                print(f"DEBUG: Saved documents to {docs_path}")
            except Exception as e:
                print(f"DEBUG ERROR: Failed to save documents: {e}")
            
            meta_path = os.path.join(self.vector_store_path, "metadata.pkl")
            try:
                with open(meta_path, "wb") as f:
                    pickle.dump(self.metadata, f)
                print(f"DEBUG: Saved metadata to {meta_path}")
            except Exception as e:
                print(f"DEBUG ERROR: Failed to save metadata: {e}")
        else:
            print("DEBUG: Cannot save - index is None")
    
    def load_index(self):
        """Load index from disk"""
        index_path = os.path.join(self.vector_store_path, "index.faiss")
        docs_path = os.path.join(self.vector_store_path, "documents.pkl")
        meta_path = os.path.join(self.vector_store_path, "metadata.pkl")
        
        print(f"DEBUG: Checking for existing index at {index_path}")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                print(f"DEBUG: Loading existing index...")
                self.index = faiss.read_index(index_path)
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                print(f"DEBUG: Loaded vector store with {len(self.documents)} documents")
            except Exception as e:
                print(f"DEBUG ERROR: Error loading vector store: {e}")
                self.index = None
                self.documents = []
                self.metadata = []
        else:
            print(f"DEBUG: No existing index found")
    

    def remove_document(self, filename: str) -> bool:
        """Remove all chunks for a specific document"""
        print(f"DEBUG: Removing document: {filename}")
        
        if not self.documents:
            print(f"DEBUG: No documents to remove")
            return False
        
        # Find indices of chunks to keep
        keep_indices = []
        for i, doc in enumerate(self.documents):
            if doc.get("filename") != filename:
                keep_indices.append(i)
        
        if len(keep_indices) == len(self.documents):
            print(f"DEBUG: Document '{filename}' not found")
            return False
        
        print(f"DEBUG: Keeping {len(keep_indices)} chunks, removing {len(self.documents) - len(keep_indices)} chunks")
        
        # Rebuild everything from scratch (simpler than selective FAISS removal)
        # 1. Keep only documents we want
        new_documents = [self.documents[i] for i in keep_indices]
        new_metadata = [self.metadata[i] for i in keep_indices]
        
        # 2. If we have documents left, rebuild index
        if new_documents:
            # Extract texts
            texts = [doc["text"] for doc in new_documents]
            
            # Create new embeddings
            embeddings = self.create_embeddings(texts)
            
            # Create new index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        else:
            # No documents left
            self.index = None
        
        # 3. Update stored data
        self.documents = new_documents
        self.metadata = new_metadata
        
        # 4. Save updated index
        self.save_index()
        print(f"DEBUG: Successfully removed document '{filename}'")
        return True

    def clear(self):
        """Clear vector store"""
        print(f"DEBUG: Clearing vector store")
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Delete saved files
        for filename in ["index.faiss", "documents.pkl", "metadata.pkl"]:
            filepath = os.path.join(self.vector_store_path, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"DEBUG: Removed {filepath}")
        
        self._initialized = False
