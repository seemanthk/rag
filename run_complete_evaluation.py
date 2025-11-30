"""
Complete Evaluation Script with Enhanced Metrics and HTML Dashboard
Run this to get comprehensive evaluation with visualization
"""

import yaml
import logging
import json
from datetime import datetime
from pathlib import Path
import time

from src.data_loader import AmazonProductDataLoader
from src.vector_store import VectorStore
from src.ollama_handler import OllamaMultiLLMManager
from src.rag_system import RAGSystem
from src.evaluation import RAGEvaluator
from src.questions import get_all_questions
from src.html_visualizer import HTMLVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("="*80)
    logger.info("SHOPSMART RAG COMPLETE EVALUATION")
    logger.info("="*80)

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Setup data
    logger.info("\n1. Loading data...")
    data_loader = AmazonProductDataLoader(config['dataset']['path'])
    df = data_loader.load_data()
    df = data_loader.preprocess_data()
    documents = data_loader.create_documents()
    logger.info(f"   Loaded {len(documents)} products")

    # Setup vector store
    logger.info("\n2. Loading vector store...")
    vector_store = VectorStore(
        embedding_model_name=config['embedding']['model_name'],
        index_path=config['vector_db']['index_path'],
        use_faiss=True
    )

    try:
        vector_store.load_index()
        logger.info("   Loaded existing index")
    except:
        logger.info("   Building new index...")
        embeddings = vector_store.create_embeddings(documents, batch_size=32)
        vector_store.build_index()
        vector_store.save_index()
        logger.info("   Index created and saved")

    # Setup LLMs
    logger.info("\n3. Loading Ollama models...")
    llm_manager = OllamaMultiLLMManager(
        llm_configs=config['llms'],
        base_url=config['ollama']['base_url']
    )

    # Load all models
    for model_name in ['phi3', 'llama3', 'gemma2']:
        try:
            llm_manager.load_llm(model_name)
            logger.info(f"   ✓ {model_name} loaded")
        except Exception as e:
            logger.error(f"   ✗ {model_name} failed: {e}")

    # Setup RAG system
    logger.info("\n4. Setting up RAG system...")
    rag_system = RAGSystem(
        vector_store=vector_store,
        llm_manager=llm_manager,
        top_k=3
    )

    # Setup evaluator
    evaluator = RAGEvaluator()

    # Get questions
    questions = get_all_questions()
    logger.info(f"\n5. Running evaluation on {len(questions)} questions...")

    all_results = []

    for i, q in enumerate(questions):
        logger.info(f"\n   Question {i+1}/{len(questions)}: {q['question']}")

        question_result = {
            'question_id': q['id'],
            'question': q['question'],
            'category': q['category'],
            'difficulty': q['difficulty'],
            'type': q.get('type', ''),
            'model_results': {}
        }

        # Query all models
        try:
            results = rag_system.query_all_llms(
                query=q['question'],
                return_context=True
            )

            for model_name, result in results.items():
                # Evaluate with enhanced metrics
                metrics = evaluator.evaluate_answer(
                    answer=result['answer'],
                    context=result.get('context', ''),
                    query=q['question'],
                    retrieved_docs=result.get('retrieved_docs', [])
                )

                question_result['model_results'][model_name] = {
                    'answer': result['answer'],
                    'num_retrieved': result['num_retrieved'],
                    'metrics': metrics,
                    'response_time': 0  # Can add timing if needed
                }

                logger.info(f"      {model_name}: {metrics['answer_length']} words, "
                          f"accuracy={metrics['factual_accuracy']:.2f}, "
                          f"relevancy={metrics['query_overlap']:.2f}")

        except Exception as e:
            logger.error(f"   Error: {e}")
            continue

        all_results.append(question_result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Save JSON
    json_file = output_dir / f"evaluation_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✓ JSON saved: {json_file}")

    # Generate HTML Dashboard
    logger.info("\n6. Generating HTML dashboard...")
    visualizer = HTMLVisualizer()
    html_file = output_dir / f"dashboard_{timestamp}.html"
    visualizer.generate_dashboard(all_results, str(html_file))
    logger.info(f"✓ HTML dashboard: {html_file}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Questions evaluated: {len(all_results)}")
    logger.info(f"Models compared: {len(llm_manager.get_loaded_llms())}")
    logger.info(f"\nResults:")
    logger.info(f"  - JSON: {json_file}")
    logger.info(f"  - HTML: {html_file}")
    logger.info(f"\nOpen the HTML file in your browser to view the dashboard!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
