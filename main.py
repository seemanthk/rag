"""
Main script for building and running the RAG system
"""

import yaml
import argparse
import logging
import os
from pathlib import Path
import json
from datetime import datetime

from src.data_loader import AmazonProductDataLoader
from src.vector_store import VectorStore
from src.llm_handler import MultiLLMManager
from src.rag_system import RAGSystem
from src.evaluation import RAGEvaluator, BatchEvaluator
from src.questions import get_all_questions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_vector_index(config):
    """Build vector index from dataset"""
    logger.info("=" * 80)
    logger.info("STEP 1: LOADING AND PREPROCESSING DATA")
    logger.info("=" * 80)

    # Load data
    data_loader = AmazonProductDataLoader(config['dataset']['path'])
    df = data_loader.load_data()
    df = data_loader.preprocess_data()

    # Print statistics
    stats = data_loader.get_statistics()
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total products: {stats['total_products']}")
    logger.info(f"  Columns: {stats['columns']}")

    # Create documents
    documents = data_loader.create_documents()

    logger.info("=" * 80)
    logger.info("STEP 2: BUILDING VECTOR INDEX")
    logger.info("=" * 80)

    # Create vector store
    vector_store = VectorStore(
        embedding_model_name=config['embedding']['model_name'],
        index_path=config['vector_db']['index_path'],
        use_faiss=(config['vector_db']['type'] == 'faiss')
    )

    # Create embeddings and build index
    vector_store.create_embeddings(documents, batch_size=config['embedding']['batch_size'])
    vector_store.build_index()

    # Save index
    vector_store.save_index()
    logger.info("Vector index saved successfully")

    return vector_store


def load_vector_index(config):
    """Load existing vector index"""
    logger.info("Loading existing vector index...")

    vector_store = VectorStore(
        embedding_model_name=config['embedding']['model_name'],
        index_path=config['vector_db']['index_path'],
        use_faiss=(config['vector_db']['type'] == 'faiss')
    )

    vector_store.load_index()
    logger.info("Vector index loaded successfully")

    return vector_store


def setup_llms(config, llm_names=None):
    """Set up LLM manager"""
    logger.info("=" * 80)
    logger.info("STEP 3: LOADING LANGUAGE MODELS")
    logger.info("=" * 80)

    # Filter LLMs if specific ones requested
    llm_configs = config['llms']
    if llm_names:
        llm_configs = {k: v for k, v in llm_configs.items() if k in llm_names}

    llm_manager = MultiLLMManager(llm_configs)

    # Load all LLMs
    for llm_name in llm_configs.keys():
        logger.info(f"\nLoading {llm_name}...")
        llm_manager.load_llm(llm_name)

    return llm_manager


def run_evaluation(config, vector_store, llm_manager, questions=None, output_dir="outputs"):
    """Run evaluation on questions"""
    logger.info("=" * 80)
    logger.info("STEP 4: RUNNING EVALUATION")
    logger.info("=" * 80)

    # Create RAG system
    rag_system = RAGSystem(
        vector_store=vector_store,
        llm_manager=llm_manager,
        top_k=config['rag']['top_k'],
        system_prompt=config['prompts']['system_prompt'],
        qa_template=config['prompts']['qa_template']
    )

    # Get questions
    if questions is None:
        questions = get_all_questions()

    # Create evaluator
    evaluator = RAGEvaluator()
    batch_evaluator = BatchEvaluator()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []
    all_comparisons = []

    # Process each question
    for i, q in enumerate(questions):
        logger.info(f"\n{'='*80}")
        logger.info(f"Question {i+1}/{len(questions)} (ID: {q['id']})")
        logger.info(f"Category: {q['category']} | Difficulty: {q['difficulty']}")
        logger.info(f"{'='*80}")
        logger.info(f"Q: {q['question']}")
        logger.info(f"{'='*80}\n")

        # Query all LLMs
        results = rag_system.query_all_llms(
            query=q['question'],
            return_context=True
        )

        # Print answers
        for llm_name, result in results.items():
            logger.info(f"\n{llm_name} Answer:")
            logger.info(f"{'-'*80}")
            logger.info(result['answer'])
            logger.info(f"{'-'*80}")

        # Evaluate
        comparison = evaluator.compare_llm_outputs(results)
        comparison['question_id'] = q['id']
        comparison['question'] = q['question']
        comparison['category'] = q['category']
        comparison['difficulty'] = q['difficulty']

        # Generate report
        report = evaluator.generate_report(comparison, q['question'])
        logger.info(f"\n{report}")

        # Store results
        all_results.append({
            'question': q,
            'results': results,
            'comparison': comparison
        })
        all_comparisons.append(comparison)

    # Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATED RESULTS")
    logger.info("=" * 80)

    # Save detailed results
    output_file = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(output_file, 'w') as f:
        # Convert to serializable format
        serializable_results = []
        for r in all_results:
            serializable_r = {
                'question': r['question'],
                'results': {
                    llm: {
                        'query': res['query'],
                        'answer': res['answer'],
                        'num_retrieved': res['num_retrieved']
                    }
                    for llm, res in r['results'].items()
                },
                'comparison': {
                    'llm_metrics': r['comparison']['llm_metrics'],
                    'diversity_score': r['comparison']['diversity_score'],
                    'agreement_score': r['comparison']['agreement_score']
                }
            }
            serializable_results.append(serializable_r)

        json.dump(serializable_results, f, indent=2)

    logger.info(f"\nDetailed results saved to: {output_file}")

    # Generate summary report
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RAG SYSTEM EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Number of questions: {len(questions)}\n")
        f.write(f"LLMs evaluated: {', '.join(llm_manager.get_loaded_llms())}\n\n")

        # Calculate average metrics per LLM
        llm_names = llm_manager.get_loaded_llms()
        for llm_name in llm_names:
            f.write(f"\n{llm_name} Performance:\n")
            f.write("-" * 80 + "\n")

            # Collect metrics
            metrics_list = [c['llm_metrics'][llm_name] for c in all_comparisons]

            # Calculate averages
            avg_metrics = {}
            for metric_name in metrics_list[0].keys():
                values = [m[metric_name] for m in metrics_list]
                avg_metrics[metric_name] = sum(values) / len(values)

            for metric_name, avg_value in avg_metrics.items():
                f.write(f"  {metric_name}: {avg_value:.3f}\n")

        # Overall statistics
        f.write(f"\n{'='*80}\n")
        f.write("Overall Statistics:\n")
        f.write("-" * 80 + "\n")
        avg_diversity = sum(c['diversity_score'] for c in all_comparisons) / len(all_comparisons)
        avg_agreement = sum(c['agreement_score'] for c in all_comparisons) / len(all_comparisons)
        f.write(f"Average Diversity Score: {avg_diversity:.3f}\n")
        f.write(f"Average Agreement Score: {avg_agreement:.3f}\n")

    logger.info(f"Summary report saved to: {summary_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Amazon Product RAG System")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--build-index', action='store_true', help='Build new vector index')
    parser.add_argument('--llms', nargs='+', help='Specific LLMs to use (e.g., llama3 mistral)')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--num-questions', type=int, help='Number of questions to evaluate')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Build or load vector index
    if args.build_index or not os.path.exists(config['vector_db']['index_path']):
        vector_store = build_vector_index(config)
    else:
        vector_store = load_vector_index(config)

    # Setup LLMs
    llm_manager = setup_llms(config, llm_names=args.llms)

    # Get questions
    questions = get_all_questions()
    if args.num_questions:
        questions = questions[:args.num_questions]

    # Run evaluation
    results = run_evaluation(config, vector_store, llm_manager, questions, args.output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
