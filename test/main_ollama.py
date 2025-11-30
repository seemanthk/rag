"""
Main script for running the Air Conditioner RAG system with Ollama models
Supports phi3, llama3, and gemma2
"""

import argparse
import yaml
import logging
import json
import os
from datetime import datetime
from pathlib import Path

from src.data_loader import AmazonProductDataLoader
from src.vector_store import VectorStore
from src.ollama_handler import OllamaMultiLLMManager
from src.rag_system import RAGSystem
from src.evaluation import RAGEvaluator
from src.questions import get_all_questions, get_questions_by_category

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_data(config):
    """Load and preprocess data"""
    logger.info("Loading data...")
    data_loader = AmazonProductDataLoader(config['dataset']['path'])
    df = data_loader.load_data()

    logger.info("Preprocessing data...")
    df = data_loader.preprocess_data()

    logger.info("Creating documents...")
    documents = data_loader.create_documents()

    logger.info(f"Created {len(documents)} documents")

    return data_loader, documents


def build_vector_index(config, documents, force_rebuild=False):
    """Build or load vector index"""
    index_path = config['vector_db']['index_path']

    vector_store = VectorStore(
        embedding_model_name=config['embedding']['model_name'],
        index_path=index_path,
        use_faiss=True
    )

    # Check if index exists
    index_file = f"{index_path}.index"
    if os.path.exists(index_file) and not force_rebuild:
        logger.info("Loading existing vector index...")
        vector_store.load_index()
    else:
        logger.info("Building new vector index...")
        embeddings = vector_store.create_embeddings(
            documents,
            batch_size=config['embedding'].get('batch_size', 32)
        )
        logger.info(f"Created embeddings with shape: {embeddings.shape}")

        vector_store.build_index()
        vector_store.save_index()
        logger.info("Vector index built and saved!")

    return vector_store


def setup_llms(config, models=None):
    """Setup Ollama LLM manager"""
    logger.info("Initializing Ollama LLM manager...")

    llm_manager = OllamaMultiLLMManager(
        llm_configs=config['llms'],
        base_url=config['ollama']['base_url']
    )

    # Load specified models or all models
    models_to_load = models if models else list(config['llms'].keys())

    for model_name in models_to_load:
        try:
            logger.info(f"Loading {model_name}...")
            llm_manager.load_llm(model_name)
            logger.info(f"✓ {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load {model_name}: {e}")

    loaded = llm_manager.get_loaded_llms()
    logger.info(f"Loaded models: {loaded}")

    return llm_manager


def setup_rag_system(config, vector_store, llm_manager):
    """Create RAG system"""
    logger.info("Setting up RAG system...")

    rag_system = RAGSystem(
        vector_store=vector_store,
        llm_manager=llm_manager,
        top_k=config['rag']['top_k'],
        system_prompt=config['prompts']['system_prompt'],
        qa_template=config['prompts']['qa_template']
    )

    logger.info("RAG system ready!")
    return rag_system


def run_single_query(rag_system, question, models=None, save_results=True):
    """Run a single query with specified models"""
    logger.info(f"Query: {question}")

    results = {}

    if models:
        # Query specific models
        for model_name in models:
            try:
                logger.info(f"Generating answer with {model_name}...")
                result = rag_system.query(
                    query=question,
                    llm_name=model_name,
                    return_context=True
                )
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
                results[model_name] = {'error': str(e)}
    else:
        # Query all loaded models
        try:
            results = rag_system.query_all_llms(
                query=question,
                return_context=True
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            return None

    # Display results
    print("\n" + "="*80)
    print(f"QUESTION: {question}")
    print("="*80)

    for model_name, result in results.items():
        if 'error' in result:
            print(f"\n{model_name.upper()}: ERROR")
            print(f"  {result['error']}")
        else:
            print(f"\n{model_name.upper()}:")
            print("-"*80)
            print(result['answer'])
            print(f"\n(Retrieved {result['num_retrieved']} documents)")

    print("="*80)

    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"query_result_{timestamp}.json"

        # Prepare data for JSON serialization
        save_data = {
            'question': question,
            'timestamp': timestamp,
            'results': {}
        }

        for model_name, result in results.items():
            if 'error' not in result:
                save_data['results'][model_name] = {
                    'answer': result['answer'],
                    'num_retrieved': result['num_retrieved']
                }
            else:
                save_data['results'][model_name] = result

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_file}")

    return results


def run_evaluation(rag_system, evaluator, questions=None, models=None, save_results=True):
    """Run evaluation on multiple questions"""

    if questions is None:
        questions = get_all_questions()

    logger.info(f"Running evaluation on {len(questions)} questions...")

    all_results = []

    for i, q in enumerate(questions):
        logger.info(f"\nQuestion {i+1}/{len(questions)}: {q['question']}")

        question_results = {
            'question_id': q['id'],
            'question': q['question'],
            'category': q['category'],
            'difficulty': q['difficulty'],
            'model_results': {}
        }

        # Get answers from all models
        try:
            if models:
                for model_name in models:
                    result = rag_system.query(
                        query=q['question'],
                        llm_name=model_name
                    )
                    question_results['model_results'][model_name] = {
                        'answer': result['answer'],
                        'num_retrieved': result['num_retrieved']
                    }
            else:
                results = rag_system.query_all_llms(query=q['question'])
                for model_name, result in results.items():
                    question_results['model_results'][model_name] = {
                        'answer': result['answer'],
                        'num_retrieved': result['num_retrieved']
                    }
        except Exception as e:
            logger.error(f"Error processing question {q['id']}: {e}")
            continue

        all_results.append(question_results)

    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"evaluation_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {output_file}")

        # Generate summary report
        generate_summary_report(all_results, output_dir / f"summary_{timestamp}.txt")

    return all_results


def generate_summary_report(results, output_file):
    """Generate a text summary report"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("AIR CONDITIONER RAG SYSTEM - EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total Questions Evaluated: {len(results)}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Organize by category
        categories = {}
        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)

        f.write(f"Questions by Category:\n")
        for cat, items in categories.items():
            f.write(f"  - {cat}: {len(items)} questions\n")

        f.write("\n" + "-"*80 + "\n\n")

        # Detailed results
        for result in results:
            f.write(f"Q{result['question_id']}: {result['question']}\n")
            f.write(f"Category: {result['category']} | Difficulty: {result['difficulty']}\n\n")

            for model_name, model_result in result['model_results'].items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"{model_result['answer']}\n")
                f.write(f"(Retrieved: {model_result['num_retrieved']} docs)\n\n")

            f.write("-"*80 + "\n\n")

    logger.info(f"Summary report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Air Conditioner RAG System with Ollama")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--build-index', action='store_true', help='Build vector index')
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild index')
    parser.add_argument('--query', type=str, help='Run a single query')
    parser.add_argument('--evaluate', action='store_true', help='Run full evaluation')
    parser.add_argument('--evaluate-category', type=str, help='Evaluate specific category')
    parser.add_argument('--models', nargs='+', help='Models to use (e.g., phi3 llama3)')
    parser.add_argument('--num-questions', type=int, help='Number of questions to evaluate')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info("Configuration loaded")

    # Setup data and vector store
    data_loader, documents = setup_data(config)

    if args.build_index or args.force_rebuild:
        vector_store = build_vector_index(config, documents, force_rebuild=args.force_rebuild)
    else:
        # Try to load existing index
        try:
            vector_store = build_vector_index(config, documents, force_rebuild=False)
        except:
            logger.warning("No existing index found. Building new one...")
            vector_store = build_vector_index(config, documents, force_rebuild=True)

    # Setup LLMs
    llm_manager = setup_llms(config, models=args.models)

    if not llm_manager.get_loaded_llms():
        logger.error("No models loaded! Please ensure Ollama is running and models are available.")
        logger.error("Run: ollama pull phi3; ollama pull llama3; ollama pull gemma2")
        return

    # Setup RAG system
    rag_system = setup_rag_system(config, vector_store, llm_manager)

    # Setup evaluator
    evaluator = RAGEvaluator()

    # Run based on arguments
    if args.query:
        run_single_query(rag_system, args.query, models=args.models)

    elif args.evaluate:
        questions = get_all_questions()
        if args.num_questions:
            questions = questions[:args.num_questions]
        run_evaluation(rag_system, evaluator, questions=questions, models=args.models)

    elif args.evaluate_category:
        questions = get_questions_by_category(args.evaluate_category)
        run_evaluation(rag_system, evaluator, questions=questions, models=args.models)

    else:
        # Interactive mode
        logger.info("\nInteractive mode - No arguments provided")
        logger.info("Example commands:")
        logger.info("  python main_ollama.py --query 'What are the best ACs under ₹30,000?'")
        logger.info("  python main_ollama.py --evaluate --num-questions 5")
        logger.info("  python main_ollama.py --evaluate --models phi3 llama3")
        logger.info("  python main_ollama.py --build-index")

        # Run a demo query
        demo_question = "What are the top-rated 1.5 ton inverter air conditioners?"
        logger.info(f"\nRunning demo query: {demo_question}")
        run_single_query(rag_system, demo_question, models=args.models)


if __name__ == "__main__":
    main()
