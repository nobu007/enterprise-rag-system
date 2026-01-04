"""
Vector Database Connection and Operations

This module provides a unified interface for vector database operations,
supporting Pinecone, Weaviate, and FAISS.
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

from app.core.logging_config import get_logger


logger = get_logger(__name__)


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
            logger.info(f"Connected to Pinecone index: {self.index_name}")

        except ImportError:
            raise ImportError("pinecone-client not installed. Run: pip install pinecone-client")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Pinecone: {e}")
    
    def create_index(self, dimension: int, metric: str = "cosine") -> None:
        """Create a new Pinecone index"""
        try:
            import pinecone
            
            if self.index_name in pinecone.list_indexes():
                logger.warning(f"Index '{self.index_name}' already exists")
                return

            pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                pods=1,
                pod_type="p1.x1"
            )
            logger.info(f"Created Pinecone index: {self.index_name}")
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

        logger.info(f"Upserted {len(items)} vectors")
    
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
        logger.info(f"Deleted {len(ids)} vectors")
    
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
        # Support multiple collections: {collection_name: index}
        self.indices: Dict[str, Any] = {}
        # Support multiple metadata stores: {collection_name: {doc_id: metadata}}
        self.metadata_stores: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # Support multiple ID mappings: {collection_name: {doc_id: idx}}
        self.id_to_idx_mappings: Dict[str, Dict[str, int]] = {}
        # Support multiple reverse ID mappings: {collection_name: {idx: doc_id}}
        self.idx_to_id_mappings: Dict[str, Dict[int, str]] = {}
        # Keep a reference to the current default index for backward compatibility
        self.index = None
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}

    def _get_collection_index(self, collection: str) -> Optional[Any]:
        """Get index for a specific collection"""
        return self.indices.get(collection)

    def _get_or_create_collection(self, collection: str, dimension: int = None, metric: str = "cosine") -> Any:
        """Get existing collection or create a new one"""
        if collection not in self.indices:
            # If dimension is not provided, try to infer from existing indices
            if dimension is None:
                if self.indices:
                    # Use dimension from first existing collection
                    first_index = next(iter(self.indices.values()))
                    dimension = first_index.d
                else:
                    raise ValueError(f"Collection '{collection}' does not exist and cannot infer dimension")
            self._create_collection_index(collection, dimension, metric)
        return self.indices[collection]

    def _create_collection_index(self, collection: str, dimension: int, metric: str = "cosine") -> None:
        """Create a new index for a collection"""
        try:
            import faiss

            if metric == "cosine":
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
            elif metric == "euclidean":
                index = faiss.IndexFlatL2(dimension)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            self.indices[collection] = index
            self.metadata_stores[collection] = {}
            self.id_to_idx_mappings[collection] = {}
            self.idx_to_id_mappings[collection] = {}

            # Set as current index if it's 'default'
            if collection == "default":
                self.index = index
                self.metadata_store = self.metadata_stores[collection]
                self.id_to_idx = self.id_to_idx_mappings[collection]
                self.idx_to_id = self.idx_to_id_mappings[collection]

            logger.info(f"Created FAISS index for collection '{collection}' with dimension: {dimension}")

        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")
    
    def connect(self) -> None:
        """Load FAISS index from disk"""
        try:
            import faiss
            import os
            import pickle
            import glob

            if self.index_path and os.path.exists(self.index_path):
                # Load the default index
                self.index = faiss.read_index(self.index_path)
                self.indices["default"] = self.index

                # Try to load metadata for default collection
                metadata_path = self.index_path + ".metadata.pkl"
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        data = pickle.load(f)
                        self.metadata_store = data['metadata_store']
                        self.id_to_idx = data['id_to_idx']
                        self.idx_to_id = data['idx_to_id']

                    # Initialize metadata stores for default collection
                    self.metadata_stores["default"] = self.metadata_store
                    self.id_to_idx_mappings["default"] = self.id_to_idx
                    self.idx_to_id_mappings["default"] = self.idx_to_id

                # Look for other collection indexes (index_path.<collection_name>)
                base_dir = os.path.dirname(self.index_path)
                base_name = os.path.basename(self.index_path)
                pattern = os.path.join(base_dir, f"{base_name}.*")
                for collection_file in glob.glob(pattern):
                    # Extract collection name from filename
                    collection_name = collection_file.split('.')[-1]
                    if collection_name == "metadata" or collection_name == "pkl":
                        continue

                    # Load collection index
                    try:
                        collection_index = faiss.read_index(collection_file)
                        self.indices[collection_name] = collection_index

                        # Load metadata for this collection
                        collection_metadata_path = collection_file + ".metadata.pkl"
                        if os.path.exists(collection_metadata_path):
                            with open(collection_metadata_path, 'rb') as f:
                                data = pickle.load(f)
                                self.metadata_stores[collection_name] = data['metadata_store']
                                self.id_to_idx_mappings[collection_name] = data['id_to_idx']
                                self.idx_to_id_mappings[collection_name] = data['idx_to_id']

                        logger.info(f"Loaded FAISS index for collection '{collection_name}' from: {collection_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load collection '{collection_name}': {e}")

                logger.info(f"Loaded FAISS index from: {self.index_path}")
            else:
                logger.warning("No existing FAISS index found")

        except ImportError:
            raise ImportError("faiss not installed. Run: pip install faiss-cpu")
    
    def create_index(self, dimension: int, metric: str = "cosine", collection: str = "default") -> None:
        """Create a new FAISS index for a collection"""
        self._create_collection_index(collection, dimension, metric)
    
    def upsert(
        self,
        vectors: List[List[float]],
        ids: List[str],
        metadata: List[Dict[str, Any]],
        collection: str = "default"
    ) -> None:
        """Insert or update vectors in FAISS for a specific collection"""
        # Get or create collection index
        index = self._get_or_create_collection(collection)
        if index is None:
            raise RuntimeError(f"Index for collection '{collection}' not created. Call create_index() first.")

        import numpy as np
        import faiss

        vectors_np = np.array(vectors, dtype=np.float32)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors_np)

        start_idx = index.ntotal
        index.add(vectors_np)

        # Store metadata for this collection
        metadata_store = self.metadata_stores[collection]
        id_to_idx = self.id_to_idx_mappings[collection]
        idx_to_id = self.idx_to_id_mappings[collection]

        for i, (id_, meta) in enumerate(zip(ids, metadata)):
            idx = start_idx + i
            id_to_idx[id_] = idx
            idx_to_id[idx] = id_
            metadata_store[id_] = meta

        logger.info(f"Upserted {len(vectors)} vectors into collection '{collection}'")
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        collection: str = "default"
    ) -> List[SearchResult]:
        """Search for similar vectors in FAISS within a specific collection"""
        index = self._get_collection_index(collection)
        if index is None:
            logger.warning(f"Collection '{collection}' not found")
            return []

        import numpy as np
        import faiss

        query_np = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_np)

        distances, indices = index.search(query_np, top_k)

        search_results = []
        metadata_store = self.metadata_stores.get(collection, {})
        idx_to_id = self.idx_to_id_mappings.get(collection, {})

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No result
                continue

            id_ = idx_to_id.get(idx)
            if id_:
                metadata = metadata_store.get(id_, {})
                search_results.append(SearchResult(
                    id=id_,
                    score=float(dist),
                    metadata=metadata,
                    text=metadata.get("text", "")
                ))

        return search_results
    
    def delete(self, ids: List[str], collection: str = "default") -> None:
        """Delete vectors from FAISS (not directly supported, requires rebuild)"""
        logger.warning(f"FAISS does not support direct deletion from collection '{collection}'. Index needs to be rebuilt.")

    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS index statistics for all collections"""
        if not self.indices:
            return {"total_vectors": 0, "collections": {}}

        collection_stats = {}
        total_vectors = 0

        for collection_name, index in self.indices.items():
            count = index.ntotal
            collection_stats[collection_name] = {
                "total_vectors": count,
                "dimension": index.d
            }
            total_vectors += count

        return {
            "total_vectors": total_vectors,
            "collections": collection_stats
        }
    
    def save(self, path: str, collection: str = None) -> None:
        """Save FAISS index to disk for a specific collection or all collections"""
        import faiss
        import pickle

        if collection is not None:
            # Save a specific collection
            index = self._get_collection_index(collection)
            if index is None:
                raise RuntimeError(f"No index to save for collection '{collection}'")

            # Save index with collection suffix if not default
            index_path = path if collection == "default" else f"{path}.{collection}"
            faiss.write_index(index, index_path)

            # Save metadata for this collection
            metadata_path = index_path + ".metadata.pkl"
            metadata_store = self.metadata_stores.get(collection, {})
            id_to_idx = self.id_to_idx_mappings.get(collection, {})
            idx_to_id = self.idx_to_id_mappings.get(collection, {})

            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata_store': metadata_store,
                    'id_to_idx': id_to_idx,
                    'idx_to_id': idx_to_id,
                    'collection': collection
                }, f)

            logger.info(f"Saved FAISS index for collection '{collection}' to: {index_path}")
        else:
            # Save all collections
            for collection_name, index in self.indices.items():
                # Save index with collection suffix if not default
                index_path = path if collection_name == "default" else f"{path}.{collection_name}"
                faiss.write_index(index, index_path)

                # Save metadata for this collection
                metadata_path = index_path + ".metadata.pkl"
                metadata_store = self.metadata_stores.get(collection_name, {})
                id_to_idx = self.id_to_idx_mappings.get(collection_name, {})
                idx_to_id = self.idx_to_id_mappings.get(collection_name, {})

                with open(metadata_path, 'wb') as f:
                    pickle.dump({
                        'metadata_store': metadata_store,
                        'id_to_idx': id_to_idx,
                        'idx_to_id': idx_to_id,
                        'collection': collection_name
                    }, f)

                logger.info(f"Saved FAISS index for collection '{collection_name}' to: {index_path}")


def get_vector_db(db_type: str = "faiss", **kwargs) -> VectorDB:
    """Factory function to get vector database instance"""
    if db_type.lower() == "pinecone":
        return PineconeVectorDB(**kwargs)
    elif db_type.lower() == "faiss":
        return FAISSVectorDB(**kwargs)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")
