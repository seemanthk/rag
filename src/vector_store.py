"""
Vector store module for embedding and retrieval
Supports FAISS and ChromaDB backends
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store for semantic search over product documents"""

    def __init__(self,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: str = "data/vector_index",
                 use_faiss: bool = True):
        """
        Initialize vector store

        Args:
            embedding_model_name: Name of sentence transformer model
            index_path: Path to save/load index
            use_faiss: Whether to use FAISS (True) or ChromaDB (False)
        """
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self.use_faiss = use_faiss

        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.documents = []
        self.embeddings = None
        self.index = None

        if use_faiss:
            try:
                import faiss
                self.faiss = faiss
            except ImportError:
                raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        else:
            try:
                import chromadb
                self.chromadb = chromadb
                self.client = None
                self.collection = None
            except ImportError:
                raise ImportError("ChromaDB not installed. Install with: pip install chromadb")

    def create_embeddings(self, documents: List[Dict], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for documents

        Args:
            documents: List of document dictionaries with 'text' field
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Creating embeddings for {len(documents)} documents...")

        texts = [doc['text'] for doc in documents]
        self.documents = documents

        # Encode in batches
        self.embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
        return self.embeddings

    def build_index(self, embeddings: Optional[np.ndarray] = None):
        """
        Build vector index

        Args:
            embeddings: Optional precomputed embeddings
        """
        if embeddings is not None:
            self.embeddings = embeddings

        if self.embeddings is None:
            raise ValueError("No embeddings available. Call create_embeddings() first.")

        if self.use_faiss:
            self._build_faiss_index()
        else:
            self._build_chromadb_index()

    def _build_faiss_index(self):
        """Build FAISS index"""
        logger.info("Building FAISS index...")

        dimension = self.embeddings.shape[1]

        # Use IndexFlatIP for cosine similarity (normalized vectors)
        self.index = self.faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

        self.index.add(normalized_embeddings.astype('float32'))

        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

    def _build_chromadb_index(self):
        """Build ChromaDB collection"""
        logger.info("Building ChromaDB collection...")

        self.client = self.chromadb.Client()

        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name="amazon_products",
                metadata={"hnsw:space": "cosine"}
            )
        except:
            # Collection might already exist
            self.collection = self.client.get_collection("amazon_products")

        # Add documents
        ids = [str(i) for i in range(len(self.documents))]
        texts = [doc['text'] for doc in self.documents]
        metadatas = [doc['metadata'] for doc in self.documents]
        embeddings = self.embeddings.tolist()

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        logger.info(f"ChromaDB collection built with {len(self.documents)} documents")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of document dictionaries with similarity scores
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)

        if self.use_faiss:
            return self._search_faiss(query_embedding, top_k)
        else:
            return self._search_chromadb(query, top_k)

    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Search using FAISS"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Normalize query
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                result = {
                    'document': self.documents[idx],
                    'score': float(score),
                    'index': int(idx)
                }
                results.append(result)

        return results

    def _search_chromadb(self, query: str, top_k: int) -> List[Dict]:
        """Search using ChromaDB"""
        if self.collection is None:
            raise ValueError("Collection not built. Call build_index() first.")

        results_dict = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        results = []
        for i in range(len(results_dict['ids'][0])):
            doc_idx = int(results_dict['ids'][0][i])
            result = {
                'document': self.documents[doc_idx],
                'score': 1.0 - results_dict['distances'][0][i],  # Convert distance to similarity
                'index': doc_idx
            }
            results.append(result)

        return results

    def save_index(self, path: Optional[str] = None):
        """Save index to disk"""
        save_path = path or self.index_path
        os.makedirs(save_path, exist_ok=True)

        logger.info(f"Saving index to {save_path}...")

        if self.use_faiss:
            # Save FAISS index
            self.faiss.write_index(
                self.index,
                os.path.join(save_path, "faiss.index")
            )

        # Save documents and embeddings
        with open(os.path.join(save_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

        with open(os.path.join(save_path, "embeddings.pkl"), "wb") as f:
            pickle.dump(self.embeddings, f)

        logger.info("Index saved successfully")

    def load_index(self, path: Optional[str] = None):
        """Load index from disk"""
        load_path = path or self.index_path

        logger.info(f"Loading index from {load_path}...")

        if self.use_faiss:
            # Load FAISS index
            index_file = os.path.join(load_path, "faiss.index")
            if os.path.exists(index_file):
                self.index = self.faiss.read_index(index_file)
            else:
                raise FileNotFoundError(f"FAISS index not found at {index_file}")

        # Load documents and embeddings
        with open(os.path.join(load_path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)

        with open(os.path.join(load_path, "embeddings.pkl"), "rb") as f:
            self.embeddings = pickle.load(f)

        logger.info(f"Loaded index with {len(self.documents)} documents")
