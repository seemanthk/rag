"""
RAG System module that combines vector retrieval with LLM generation
"""

from typing import List, Dict, Optional
import logging
from .vector_store import VectorStore
from .llm_handler import MultiLLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Complete RAG system for Amazon product Q&A"""

    def __init__(self,
                 vector_store: VectorStore,
                 llm_manager: MultiLLMManager,
                 top_k: int = 5,
                 system_prompt: Optional[str] = None,
                 qa_template: Optional[str] = None):
        """
        Initialize RAG system

        Args:
            vector_store: Vector store for retrieval
            llm_manager: LLM manager for generation
            top_k: Number of documents to retrieve
            system_prompt: System prompt for the LLM
            qa_template: Template for formatting query and context
        """
        self.vector_store = vector_store
        self.llm_manager = llm_manager
        self.top_k = top_k

        self.system_prompt = system_prompt or """You are a helpful AI assistant specialized in answering questions about Amazon products.
Use the provided context to answer the user's question accurately and concisely.
If you cannot find the answer in the context, say so clearly."""

        self.qa_template = qa_template or """Context information is below:
---------------------
{context}
---------------------

Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:"""

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query

        Args:
            query: User query
            top_k: Number of documents to retrieve (uses default if None)

        Returns:
            List of retrieved documents with scores
        """
        k = top_k or self.top_k
        return self.vector_store.search(query, top_k=k)

    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Format retrieved documents into context string

        Args:
            retrieved_docs: List of retrieved document dictionaries

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc_result in enumerate(retrieved_docs):
            doc = doc_result['document']
            score = doc_result['score']
            context_parts.append(f"[Document {i+1}] (Relevance: {score:.3f})")
            context_parts.append(doc['text'])
            context_parts.append("")  # Empty line between documents

        return "\n".join(context_parts)

    def create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for LLM

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        # Format the QA template
        qa_prompt = self.qa_template.format(context=context, query=query)

        # Combine system prompt and query
        full_prompt = f"{self.system_prompt}\n\n{qa_prompt}"

        return full_prompt

    def query(self,
              query: str,
              llm_name: Optional[str] = None,
              top_k: Optional[int] = None,
              return_context: bool = False,
              **generation_kwargs) -> Dict:
        """
        Run complete RAG pipeline for a query

        Args:
            query: User query
            llm_name: Specific LLM to use (uses current if None)
            top_k: Number of documents to retrieve
            return_context: Whether to return retrieved context
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary with answer and optional context
        """
        # Retrieve relevant documents
        logger.info(f"Retrieving documents for: {query}")
        retrieved_docs = self.retrieve(query, top_k=top_k)

        # Format context
        context = self.format_context(retrieved_docs)

        # Create prompt
        prompt = self.create_prompt(query, context)

        # Generate answer
        logger.info(f"Generating answer with {llm_name or 'current LLM'}...")
        answer = self.llm_manager.generate(prompt, llm_name=llm_name, **generation_kwargs)

        result = {
            "query": query,
            "answer": answer,
            "num_retrieved": len(retrieved_docs),
        }

        if return_context:
            result["retrieved_docs"] = retrieved_docs
            result["context"] = context
            result["prompt"] = prompt

        return result

    def query_all_llms(self,
                      query: str,
                      top_k: Optional[int] = None,
                      return_context: bool = False,
                      **generation_kwargs) -> Dict[str, Dict]:
        """
        Query using all loaded LLMs

        Args:
            query: User query
            top_k: Number of documents to retrieve
            return_context: Whether to return retrieved context
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary mapping LLM names to results
        """
        # Retrieve once for all LLMs
        logger.info(f"Retrieving documents for: {query}")
        retrieved_docs = self.retrieve(query, top_k=top_k)
        context = self.format_context(retrieved_docs)
        prompt = self.create_prompt(query, context)

        # Generate with all LLMs
        results = {}
        for llm_name in self.llm_manager.get_loaded_llms():
            logger.info(f"Generating answer with {llm_name}...")
            answer = self.llm_manager.generate(prompt, llm_name=llm_name, **generation_kwargs)

            result = {
                "query": query,
                "answer": answer,
                "num_retrieved": len(retrieved_docs),
            }

            if return_context:
                result["retrieved_docs"] = retrieved_docs
                result["context"] = context
                result["prompt"] = prompt

            results[llm_name] = result

        return results

    def batch_query(self,
                   queries: List[str],
                   llm_name: Optional[str] = None,
                   **kwargs) -> List[Dict]:
        """
        Process multiple queries

        Args:
            queries: List of user queries
            llm_name: Specific LLM to use
            **kwargs: Additional parameters for query()

        Returns:
            List of result dictionaries
        """
        results = []
        for query in queries:
            result = self.query(query, llm_name=llm_name, **kwargs)
            results.append(result)
        return results

    def evaluate_retrieval(self, query: str, top_k: Optional[int] = None) -> Dict:
        """
        Evaluate retrieval quality for a query

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with retrieval metrics
        """
        retrieved_docs = self.retrieve(query, top_k=top_k)

        scores = [doc['score'] for doc in retrieved_docs]

        metrics = {
            "num_retrieved": len(retrieved_docs),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "score_variance": self._calculate_variance(scores),
        }

        return metrics

    @staticmethod
    def _calculate_variance(values: List[float]) -> float:
        """Calculate variance of values"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
