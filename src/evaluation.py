"""
Evaluation module for comparing LLM outputs in RAG system
"""

import numpy as np
from typing import Optional, Dict, Any
from typing import List, Dict, Any
import logging
from collections import Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluator for RAG system and LLM comparison"""

    def __init__(self):
        """Initialize evaluator"""
        self.metrics = {}

    def evaluate_answer(self, answer: str, context: str, query: str) -> Dict[str, float]:
        """
        Evaluate a single answer

        Args:
            answer: Generated answer
            context: Retrieved context
            query: Original query

        Returns:
            Dictionary of metric scores
        """
        metrics = {}

        # Length metrics
        metrics['answer_length'] = len(answer.split())
        metrics['answer_char_length'] = len(answer)

        # Answer relevancy (keyword overlap with query)
        metrics['query_overlap'] = self._calculate_query_overlap(answer, query)

        # Faithfulness (context grounding)
        metrics['context_overlap'] = self._calculate_context_overlap(answer, context)

        # Readability
        metrics['avg_sentence_length'] = self._calculate_avg_sentence_length(answer)

        # Specificity (presence of numbers, product names, etc.)
        metrics['specificity_score'] = self._calculate_specificity(answer)

        return metrics

    def compare_llm_outputs(self,
                           results: Dict[str, Dict],
                           ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare outputs from multiple LLMs

        Args:
            results: Dictionary mapping LLM names to result dicts
            ground_truth: Optional ground truth answer

        Returns:
            Comparison metrics
        """
        comparison = {
            'llm_metrics': {},
            'summary': {}
        }

        # Evaluate each LLM
        for llm_name, result in results.items():
            answer = result['answer']
            context = result.get('context', '')
            query = result['query']

            metrics = self.evaluate_answer(answer, context, query)
            comparison['llm_metrics'][llm_name] = metrics

        # Calculate summary statistics
        comparison['summary'] = self._calculate_summary_stats(comparison['llm_metrics'])

        # Compare diversity
        answers = [result['answer'] for result in results.values()]
        comparison['diversity_score'] = self._calculate_diversity(answers)

        # Compare agreement
        comparison['agreement_score'] = self._calculate_agreement(answers)

        return comparison

    def _calculate_query_overlap(self, answer: str, query: str) -> float:
        """Calculate keyword overlap between answer and query"""
        answer_words = set(self._tokenize(answer.lower()))
        query_words = set(self._tokenize(query.lower()))

        if not query_words:
            return 0.0

        overlap = len(answer_words & query_words)
        return overlap / len(query_words)

    def _calculate_context_overlap(self, answer: str, context: str) -> float:
        """Calculate overlap between answer and context"""
        answer_words = set(self._tokenize(answer.lower()))
        context_words = set(self._tokenize(context.lower()))

        if not answer_words:
            return 0.0

        overlap = len(answer_words & context_words)
        return overlap / len(answer_words)

    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)

    def _calculate_specificity(self, text: str) -> float:
        """
        Calculate specificity score based on presence of:
        - Numbers
        - Dollar amounts
        - Percentages
        - Specific product terms
        """
        score = 0.0

        # Check for numbers
        if re.search(r'\d+', text):
            score += 0.25

        # Check for dollar amounts
        if re.search(r'\$\d+', text):
            score += 0.25

        # Check for percentages
        if re.search(r'\d+%', text):
            score += 0.25

        # Check for ratings
        if re.search(r'\d+\.?\d*\s*(star|rating)', text, re.IGNORECASE):
            score += 0.25

        return score

    def _calculate_diversity(self, texts: List[str]) -> float:
        """
        Calculate diversity between multiple texts
        Using unique n-gram ratio
        """
        all_words = []
        for text in texts:
            words = self._tokenize(text.lower())
            all_words.extend(words)

        if not all_words:
            return 0.0

        unique_words = len(set(all_words))
        total_words = len(all_words)

        return unique_words / total_words

    def _calculate_agreement(self, texts: List[str]) -> float:
        """
        Calculate agreement between texts using word overlap
        """
        if len(texts) < 2:
            return 1.0

        word_sets = [set(self._tokenize(text.lower())) for text in texts]

        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                intersection = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                if union > 0:
                    similarities.append(intersection / union)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_summary_stats(self, llm_metrics: Dict[str, Dict]) -> Dict:
        """Calculate summary statistics across LLMs"""
        summary = {}

        # Get all metric names
        if not llm_metrics:
            return summary

        first_llm = list(llm_metrics.values())[0]
        metric_names = first_llm.keys()

        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in llm_metrics.values()]

            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }

            # Rank LLMs by this metric
            llm_scores = [(llm, metrics[metric_name]) for llm, metrics in llm_metrics.items()]
            llm_scores.sort(key=lambda x: x[1], reverse=True)
            summary[metric_name]['ranking'] = [llm for llm, _ in llm_scores]

        return summary

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def generate_report(self, comparison: Dict, query: str) -> str:
        """
        Generate a human-readable comparison report

        Args:
            comparison: Comparison results from compare_llm_outputs
            query: Original query

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append(f"RAG EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"\nQuery: {query}\n")

        # LLM-specific metrics
        report.append("-" * 80)
        report.append("INDIVIDUAL LLM METRICS")
        report.append("-" * 80)

        for llm_name, metrics in comparison['llm_metrics'].items():
            report.append(f"\n{llm_name}:")
            for metric_name, value in metrics.items():
                report.append(f"  {metric_name}: {value:.3f}")

        # Summary statistics
        report.append("\n" + "-" * 80)
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)

        for metric_name, stats in comparison['summary'].items():
            report.append(f"\n{metric_name}:")
            report.append(f"  Mean: {stats['mean']:.3f}")
            report.append(f"  Std:  {stats['std']:.3f}")
            report.append(f"  Ranking: {', '.join(stats['ranking'])}")

        # Overall metrics
        report.append("\n" + "-" * 80)
        report.append("OVERALL COMPARISON")
        report.append("-" * 80)
        report.append(f"\nDiversity Score: {comparison['diversity_score']:.3f}")
        report.append(f"Agreement Score: {comparison['agreement_score']:.3f}")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


class BatchEvaluator:
    """Evaluator for multiple queries and results"""

    def __init__(self):
        """Initialize batch evaluator"""
        self.evaluator = RAGEvaluator()
        self.results = []

    def evaluate_batch(self, batch_results: List[Dict[str, Dict]]) -> Dict:
        """
        Evaluate a batch of query results

        Args:
            batch_results: List of result dictionaries (one per query)

        Returns:
            Aggregated evaluation metrics
        """
        all_comparisons = []

        for result_dict in batch_results:
            # Assume result_dict maps LLM names to results for a single query
            query = list(result_dict.values())[0]['query']
            comparison = self.evaluator.compare_llm_outputs(result_dict)
            comparison['query'] = query
            all_comparisons.append(comparison)

        # Aggregate metrics across all queries
        aggregated = self._aggregate_comparisons(all_comparisons)

        return aggregated

    def _aggregate_comparisons(self, comparisons: List[Dict]) -> Dict:
        """Aggregate multiple comparison results"""
        aggregated = {
            'num_queries': len(comparisons),
            'avg_metrics': {},
            'llm_rankings': {},
        }

        # Get LLM names
        if not comparisons:
            return aggregated

        llm_names = list(comparisons[0]['llm_metrics'].keys())

        # Aggregate metrics for each LLM
        for llm_name in llm_names:
            llm_metrics_list = []
            for comp in comparisons:
                if llm_name in comp['llm_metrics']:
                    llm_metrics_list.append(comp['llm_metrics'][llm_name])

            # Calculate averages
            if llm_metrics_list:
                avg_metrics = {}
                metric_names = llm_metrics_list[0].keys()
                for metric in metric_names:
                    values = [m[metric] for m in llm_metrics_list]
                    avg_metrics[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
                aggregated['avg_metrics'][llm_name] = avg_metrics

        # Calculate overall rankings
        aggregated['diversity_scores'] = [c['diversity_score'] for c in comparisons]
        aggregated['agreement_scores'] = [c['agreement_score'] for c in comparisons]

        aggregated['avg_diversity'] = np.mean(aggregated['diversity_scores'])
        aggregated['avg_agreement'] = np.mean(aggregated['agreement_scores'])

        return aggregated
