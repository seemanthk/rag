"""
Simple demo script for testing the RAG system with a single question
"""

import yaml
import logging
import os

from src.data_loader import AmazonProductDataLoader
from src.vector_store import VectorStore
from src.llm_handler import MultiLLMManager
from src.rag_system import RAGSystem
from src.evaluation import RAGEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "=" * 80)
    print("AMAZON PRODUCT RAG SYSTEM - DEMO")
    print("=" * 80)

    # Check if index exists
    index_path = config['vector_db']['index_path']

    if not os.path.exists(index_path):
        print("\nVector index not found. Building index from dataset...")
        print("This is a one-time process and may take a few minutes.\n")

        # Load and process data
        data_loader = AmazonProductDataLoader(config['dataset']['path'])
        df = data_loader.load_data()
        df = data_loader.preprocess_data()
        documents = data_loader.create_documents()

        # Build vector store
        vector_store = VectorStore(
            embedding_model_name=config['embedding']['model_name'],
            index_path=index_path,
            use_faiss=True
        )
        vector_store.create_embeddings(documents, batch_size=32)
        vector_store.build_index()
        vector_store.save_index()

        print("Vector index built and saved!\n")
    else:
        print("\nLoading existing vector index...")
        vector_store = VectorStore(
            embedding_model_name=config['embedding']['model_name'],
            index_path=index_path,
            use_faiss=True
        )
        vector_store.load_index()
        print("Vector index loaded!\n")

    # Load a single LLM for demo (use smallest for speed)
    print("Loading LLM (this may take a minute)...")
    print("Using Phi-3 model for demo (smallest/fastest)...\n")

    llm_configs = {
        'phi3': config['llms']['phi3']
    }
    llm_manager = MultiLLMManager(llm_configs)
    llm_manager.load_llm('phi3')

    # Create RAG system
    rag_system = RAGSystem(
        vector_store=vector_store,
        llm_manager=llm_manager,
        top_k=config['rag']['top_k'],
        system_prompt=config['prompts']['system_prompt'],
        qa_template=config['prompts']['qa_template']
    )

    # Demo question
    question = "What are the top-rated electronics products with a rating above 4.5 stars?"

    print("=" * 80)
    print(f"Demo Question: {question}")
    print("=" * 80)

    # Query the system
    print("\nRetrieving relevant documents...")
    result = rag_system.query(
        query=question,
        llm_name='phi3',
        return_context=True
    )

    # Display results
    print("\n" + "-" * 80)
    print("RETRIEVED CONTEXT (Top 3):")
    print("-" * 80)
    for i, doc in enumerate(result['retrieved_docs'][:3]):
        print(f"\n[Document {i+1}] Score: {doc['score']:.3f}")
        print(doc['document']['text'][:200] + "...")

    print("\n" + "=" * 80)
    print("GENERATED ANSWER:")
    print("=" * 80)
    print(result['answer'])

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION METRICS:")
    print("=" * 80)

    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_answer(
        answer=result['answer'],
        context=result['context'],
        query=question
    )

    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.3f}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nTo run full evaluation with all 3 LLMs, use:")
    print("  python main.py --build-index")
    print("\n")


if __name__ == "__main__":
    main()
