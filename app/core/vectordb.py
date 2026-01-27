"""
Vector Database Connection and Operations

This module provides a unified interface for vector database operations,
supporting Pinecone, Weaviate, and FAISS.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result from vector database"""
    id: str
    score: float
    metadata: Dict[str, Any]
    text: str


class VectorDB(ABC):
    """Abstract base class for vector database operations"""
    
    @abstractmethod
    def connect(self) -> None:
        """Connect to vector database"""
        pass
    
    @abstractmethod
    def create_index(self, dimension: int, metric: str = "cosine") -> None:
        """Create a new index"""
        pass
    
    @abstractmethod
    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Insert or update vectors"""
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete vectors by IDs"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass


class PineconeVectorDB(VectorDB):
    """Pinecone vector database implementation"""
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.index = None
        self._client = None
    
    def connect(self) -> None:
        """Connect to Pinecone"""
        try:
            import pinecone
            
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                raise ValueError(f"Index '{self.index_name}' does not exist")
            
            self.index = pinecone.Index(self.index_name)
            print(f"✅ Connected to Pinecone index: {self.index_name}")
            
        except ImportError:
            raise ImportError("pinecone-client not installed. Run: pip install pinecone-client")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Pinecone: {e}")
    
    def create_index(self, dimension: int, metric: str = "cosine") -> None:
        """Create a new Pinecone index"""
        try:
            import pinecone
            
            if self.index_name in pinecone.list_indexes():
                print(f"⚠️  Index '{self.index_name}' already exists")
                return
            
            pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                pods=1,
                pod_type="p1.x1"
            )
            print(f"✅ Created Pinecone index: {self.index_name}")
            self.connect()
            
        except Exception as e:
            raise RuntimeError(f"Failed to create index: {e}")
    
    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Insert or update vectors in Pinecone"""
        if not self.index:
            raise RuntimeError("Not connected to Pinecone. Call connect() first.")
        
        # Prepare data for upsert
        items = [
            (id_, vector, meta)
            for id_, vector, meta in zip(ids, vectors, metadata)
        ]
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"✅ Upserted {len(items)} vectors")
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors in Pinecone"""
        if not self.index:
            raise RuntimeError("Not connected to Pinecone. Call connect() first.")
        
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        search_results = []
        for match in results.matches:
            search_results.append(SearchResult(
                id=match.id,
                score=match.score,
                metadata=match.metadata,
                text=match.metadata.get("text", "")
            ))
        
        return search_results
    
    def delete(self, ids: List[str]) -> None:
        """Delete vectors from Pinecone"""
        if not self.index:
            raise RuntimeError("Not connected to Pinecone. Call connect() first.")
        
        self.index.delete(ids=ids)
        print(f"✅ Deleted {len(ids)} vectors")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        if not self.index:
            raise RuntimeError("Not connected to Pinecone. Call connect() first.")
        
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness
        }


class FAISSVectorDB(VectorDB):
    """FAISS vector database implementation (for local development)"""
    
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path
        self.index = None
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
    
    def connect(self) -> None:
        """Load FAISS index from disk"""
        try:
            import faiss
            
            if self.index_path and os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                print(f"✅ Loaded FAISS index from: {self.index_path}")
            else:
                print("⚠️  No existing FAISS index found")
        
        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")
    
    def create_index(self, dimension: int, metric: str = "cosine") -> None:
        """Create a new FAISS index"""
        try:
            import faiss
            
            if metric == "cosine":
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
            elif metric == "euclidean":
                self.index = faiss.IndexFlatL2(dimension)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            print(f"✅ Created FAISS index with dimension: {dimension}")
        
        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")
    
    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """Insert or update vectors in FAISS"""
        if self.index is None:
            raise RuntimeError("Index not created. Call create_index() first.")
        
        import numpy as np
        
        vectors_np = np.array(vectors, dtype=np.float32)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors_np)
        
        start_idx = self.index.ntotal
        self.index.add(vectors_np)
        
        # Store metadata
        for i, (id_, meta) in enumerate(zip(ids, metadata)):
            idx = start_idx + i
            self.id_to_idx[id_] = idx
            self.idx_to_id[idx] = id_
            self.metadata_store[id_] = meta
        
        print(f"✅ Upserted {len(vectors)} vectors")
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar vectors in FAISS"""
        if self.index is None:
            raise RuntimeError("Index not created. Call create_index() first.")
        
        import numpy as np
        import faiss
        
        query_np = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_np)
        
        distances, indices = self.index.search(query_np, top_k)
        
        search_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No result
                continue
            
            id_ = self.idx_to_id.get(idx)
            if id_:
                metadata = self.metadata_store.get(id_, {})
                search_results.append(SearchResult(
                    id=id_,
                    score=float(dist),
                    metadata=metadata,
                    text=metadata.get("text", "")
                ))
        
        return search_results
    
    def delete(self, ids: List[str]) -> None:
        """Delete vectors from FAISS (not directly supported, requires rebuild)"""
        print("⚠️  FAISS does not support direct deletion. Index needs to be rebuilt.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics"""
        if self.index is None:
            return {"total_vectors": 0}
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d
        }
    
    def save(self, path: str) -> None:
        """Save FAISS index to disk"""
        if self.index is None:
            raise RuntimeError("No index to save")
        
        import faiss
        import pickle
        
        faiss.write_index(self.index, path)
        
        # Save metadata
        metadata_path = path + ".metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata_store': self.metadata_store,
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id
            }, f)
        
        print(f"✅ Saved FAISS index to: {path}")


def get_vector_db(db_type: str = "faiss", **kwargs) -> VectorDB:
    """Factory function to get vector database instance"""
    if db_type.lower() == "pinecone":
        return PineconeVectorDB(**kwargs)
    elif db_type.lower() == "faiss":
        return FAISSVectorDB(**kwargs)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
