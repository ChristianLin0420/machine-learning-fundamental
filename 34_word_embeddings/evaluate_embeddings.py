"""
Word Embedding Evaluation Tools
===============================

Comprehensive evaluation suite for word embeddings including:
- Analogy tests (semantic and syntactic)
- Word similarity evaluations
- Embedding quality metrics
- Comparative analysis between different models

Standard evaluation benchmarks:
- Google analogy dataset format
- Word similarity correlation tests
- Semantic coherence measures
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
from typing import List, Tuple, Dict, Optional, Union
import pandas as pd
from collections import defaultdict
import os

class EmbeddingEvaluator:
    """Comprehensive evaluation toolkit for word embeddings"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_analogies(self, model, analogy_questions: List[Tuple[str, str, str, str]], 
                          name: str = "model") -> Dict:
        """
        Evaluate model on analogy questions
        
        Args:
            model: Model with get_word_vector() and analogy() methods
            analogy_questions: List of (word_a, word_b, word_c, expected) tuples
            name: Model name for results
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating {name} on {len(analogy_questions)} analogy questions...")
        
        correct = 0
        total = 0
        failed_retrievals = 0
        results_per_category = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        detailed_results = []
        
        for word_a, word_b, word_c, expected in analogy_questions:
            # Check if all words are in vocabulary
            if hasattr(model, 'vocab'):
                if not all(word in model.vocab for word in [word_a, word_b, word_c, expected]):
                    failed_retrievals += 1
                    continue
            
            # Get analogy predictions
            try:
                predictions = model.analogy(word_a, word_b, word_c, top_k=10)
                
                if predictions:
                    predicted_words = [pred[0] for pred in predictions]
                    is_correct = expected in predicted_words
                    
                    if is_correct:
                        correct += 1
                        rank = predicted_words.index(expected) + 1
                    else:
                        rank = None
                    
                    # Determine category (basic heuristic)
                    if any(word in ['king', 'queen', 'man', 'woman', 'prince', 'princess'] 
                           for word in [word_a, word_b, word_c, expected]):
                        category = 'semantic_gender'
                    elif any(word in ['big', 'small', 'large', 'tiny', 'huge'] 
                            for word in [word_a, word_b, word_c, expected]):
                        category = 'semantic_size'
                    else:
                        category = 'other'
                    
                    results_per_category[category]['total'] += 1
                    if is_correct:
                        results_per_category[category]['correct'] += 1
                    
                    detailed_results.append({
                        'word_a': word_a,
                        'word_b': word_b,
                        'word_c': word_c,
                        'expected': expected,
                        'predicted': predicted_words[0] if predicted_words else None,
                        'correct': is_correct,
                        'rank': rank,
                        'category': category
                    })
                    
                    total += 1
                else:
                    failed_retrievals += 1
                    
            except Exception as e:
                print(f"Error processing analogy {word_a}-{word_b}+{word_c}={expected}: {e}")
                failed_retrievals += 1
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        coverage = (total) / len(analogy_questions)
        
        # Category-wise accuracies
        category_accuracies = {}
        for cat, results in results_per_category.items():
            category_accuracies[cat] = results['correct'] / results['total'] if results['total'] > 0 else 0
        
        results = {
            'model_name': name,
            'total_questions': len(analogy_questions),
            'answered_questions': total,
            'failed_retrievals': failed_retrievals,
            'correct_answers': correct,
            'accuracy': accuracy,
            'coverage': coverage,
            'category_accuracies': category_accuracies,
            'detailed_results': detailed_results
        }
        
        print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"Coverage: {coverage:.3f} ({total}/{len(analogy_questions)})")
        
        return results
    
    def evaluate_word_similarity(self, model, word_pairs: List[Tuple[str, str, float]], 
                                name: str = "model") -> Dict:
        """
        Evaluate model on word similarity tasks
        
        Args:
            model: Model with get_word_vector() method
            word_pairs: List of (word1, word2, human_similarity) tuples
            name: Model name for results
        
        Returns:
            Dictionary with correlation results
        """
        print(f"Evaluating {name} on {len(word_pairs)} word similarity pairs...")
        
        model_similarities = []
        human_similarities = []
        valid_pairs = []
        
        for word1, word2, human_sim in word_pairs:
            # Check if both words are in vocabulary
            vec1 = model.get_word_vector(word1)
            vec2 = model.get_word_vector(word2)
            
            if vec1 is not None and vec2 is not None:
                # Calculate cosine similarity
                cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                model_similarities.append(cosine_sim)
                human_similarities.append(human_sim)
                valid_pairs.append((word1, word2, human_sim, cosine_sim))
        
        if len(model_similarities) == 0:
            return {
                'model_name': name,
                'total_pairs': len(word_pairs),
                'valid_pairs': 0,
                'coverage': 0,
                'spearman_correlation': 0,
                'pearson_correlation': 0
            }
        
        # Calculate correlations
        spearman_corr, spearman_p = spearmanr(human_similarities, model_similarities)
        pearson_corr, pearson_p = pearsonr(human_similarities, model_similarities)
        
        coverage = len(valid_pairs) / len(word_pairs)
        
        results = {
            'model_name': name,
            'total_pairs': len(word_pairs),
            'valid_pairs': len(valid_pairs),
            'coverage': coverage,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'detailed_pairs': valid_pairs
        }
        
        print(f"Spearman correlation: {spearman_corr:.3f}")
        print(f"Pearson correlation: {pearson_corr:.3f}")
        print(f"Coverage: {coverage:.3f} ({len(valid_pairs)}/{len(word_pairs)})")
        
        return results
    
    def evaluate_semantic_coherence(self, model, word_groups: Dict[str, List[str]], 
                                   name: str = "model") -> Dict:
        """
        Evaluate semantic coherence within word groups
        
        Args:
            model: Model with get_word_vector() method
            word_groups: Dictionary of {category: [words]} 
            name: Model name for results
        
        Returns:
            Dictionary with coherence scores
        """
        print(f"Evaluating {name} on semantic coherence...")
        
        coherence_scores = {}
        
        for category, words in word_groups.items():
            # Get vectors for all words in the group
            vectors = []
            valid_words = []
            
            for word in words:
                vec = model.get_word_vector(word)
                if vec is not None:
                    vectors.append(vec)
                    valid_words.append(word)
            
            if len(vectors) < 2:
                coherence_scores[category] = 0
                continue
            
            # Calculate pairwise similarities within group
            similarities = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    vec1, vec2 = vectors[i], vectors[j]
                    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    similarities.append(sim)
            
            # Average similarity as coherence score
            coherence_scores[category] = np.mean(similarities)
        
        overall_coherence = np.mean(list(coherence_scores.values()))
        
        results = {
            'model_name': name,
            'category_coherence': coherence_scores,
            'overall_coherence': overall_coherence
        }
        
        print(f"Overall coherence: {overall_coherence:.3f}")
        for category, score in coherence_scores.items():
            print(f"  {category}: {score:.3f}")
        
        return results
    
    def compare_models(self, models: Dict[str, object], 
                      analogy_questions: List[Tuple[str, str, str, str]],
                      word_pairs: List[Tuple[str, str, float]] = None,
                      word_groups: Dict[str, List[str]] = None) -> Dict:
        """
        Compare multiple models on various evaluation tasks
        
        Args:
            models: Dictionary of {name: model} 
            analogy_questions: Analogy test questions
            word_pairs: Word similarity pairs (optional)
            word_groups: Semantic coherence groups (optional)
        
        Returns:
            Comprehensive comparison results
        """
        print("Comparing models...")
        
        comparison_results = {
            'analogy_results': {},
            'similarity_results': {},
            'coherence_results': {}
        }
        
        # Evaluate analogies for all models
        for name, model in models.items():
            analogy_result = self.evaluate_analogies(model, analogy_questions, name)
            comparison_results['analogy_results'][name] = analogy_result
        
        # Evaluate word similarity if provided
        if word_pairs:
            for name, model in models.items():
                similarity_result = self.evaluate_word_similarity(model, word_pairs, name)
                comparison_results['similarity_results'][name] = similarity_result
        
        # Evaluate semantic coherence if provided
        if word_groups:
            for name, model in models.items():
                coherence_result = self.evaluate_semantic_coherence(model, word_groups, name)
                comparison_results['coherence_results'][name] = coherence_result
        
        return comparison_results
    
    def plot_comparison_results(self, comparison_results: Dict, save_path: str = None):
        """Plot comprehensive comparison results"""
        
        # Extract model names
        model_names = list(comparison_results['analogy_results'].keys())
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Analogy Accuracy Comparison
        analogy_accuracies = []
        analogy_coverages = []
        
        for name in model_names:
            result = comparison_results['analogy_results'][name]
            analogy_accuracies.append(result['accuracy'])
            analogy_coverages.append(result['coverage'])
        
        axes[0, 0].bar(model_names, analogy_accuracies, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Analogy Task Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(analogy_accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 2. Coverage Comparison
        axes[0, 1].bar(model_names, analogy_coverages, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Vocabulary Coverage')
        axes[0, 1].set_ylabel('Coverage')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(analogy_coverages):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # 3. Word Similarity Correlations (if available)
        if comparison_results['similarity_results']:
            spearman_corrs = []
            pearson_corrs = []
            
            for name in model_names:
                result = comparison_results['similarity_results'][name]
                spearman_corrs.append(result['spearman_correlation'])
                pearson_corrs.append(result['pearson_correlation'])
            
            x = np.arange(len(model_names))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, spearman_corrs, width, label='Spearman', alpha=0.7)
            axes[1, 0].bar(x + width/2, pearson_corrs, width, label='Pearson', alpha=0.7)
            axes[1, 0].set_title('Word Similarity Correlations')
            axes[1, 0].set_ylabel('Correlation')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(model_names)
            axes[1, 0].legend()
            axes[1, 0].set_ylim(-1, 1)
        else:
            axes[1, 0].text(0.5, 0.5, 'No similarity data', ha='center', va='center', 
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Word Similarity Correlations')
        
        # 4. Semantic Coherence (if available)
        if comparison_results['coherence_results']:
            coherence_scores = []
            
            for name in model_names:
                result = comparison_results['coherence_results'][name]
                coherence_scores.append(result['overall_coherence'])
            
            axes[1, 1].bar(model_names, coherence_scores, alpha=0.7, color='lightgreen')
            axes[1, 1].set_title('Semantic Coherence')
            axes[1, 1].set_ylabel('Average Coherence')
            axes[1, 1].set_ylim(0, 1)
            for i, v in enumerate(coherence_scores):
                axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        else:
            axes[1, 1].text(0.5, 0.5, 'No coherence data', ha='center', va='center', 
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Semantic Coherence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.close()
    
    def generate_evaluation_report(self, comparison_results: Dict, save_path: str = None) -> str:
        """Generate a detailed evaluation report"""
        
        report = "Word Embedding Evaluation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Analogy Results
        report += "ANALOGY EVALUATION\n"
        report += "-" * 20 + "\n"
        
        for name, result in comparison_results['analogy_results'].items():
            report += f"\nModel: {name}\n"
            report += f"  Accuracy: {result['accuracy']:.3f} ({result['correct_answers']}/{result['answered_questions']})\n"
            report += f"  Coverage: {result['coverage']:.3f} ({result['answered_questions']}/{result['total_questions']})\n"
            report += f"  Failed retrievals: {result['failed_retrievals']}\n"
            
            if result['category_accuracies']:
                report += "  Category accuracies:\n"
                for cat, acc in result['category_accuracies'].items():
                    report += f"    {cat}: {acc:.3f}\n"
        
        # Similarity Results
        if comparison_results['similarity_results']:
            report += "\nWORD SIMILARITY EVALUATION\n"
            report += "-" * 25 + "\n"
            
            for name, result in comparison_results['similarity_results'].items():
                report += f"\nModel: {name}\n"
                report += f"  Spearman correlation: {result['spearman_correlation']:.3f}\n"
                report += f"  Pearson correlation: {result['pearson_correlation']:.3f}\n"
                report += f"  Coverage: {result['coverage']:.3f}\n"
        
        # Coherence Results
        if comparison_results['coherence_results']:
            report += "\nSEMANTIC COHERENCE EVALUATION\n"
            report += "-" * 30 + "\n"
            
            for name, result in comparison_results['coherence_results'].items():
                report += f"\nModel: {name}\n"
                report += f"  Overall coherence: {result['overall_coherence']:.3f}\n"
                report += "  Category coherence:\n"
                for cat, score in result['category_coherence'].items():
                    report += f"    {cat}: {score:.3f}\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Evaluation report saved to {save_path}")
        
        return report


def create_evaluation_datasets():
    """Create evaluation datasets for testing"""
    
    # Analogy questions (word_a, word_b, word_c, expected)
    analogy_questions = [
        # Gender analogies
        ('king', 'man', 'queen', 'woman'),
        ('prince', 'king', 'princess', 'queen'),
        ('man', 'king', 'woman', 'queen'),
        
        # Size analogies
        ('big', 'small', 'large', 'tiny'),
        ('huge', 'big', 'tiny', 'small'),
        ('large', 'huge', 'small', 'tiny'),
        
        # Animal analogies
        ('dog', 'puppy', 'cat', 'kitten'),
        ('cat', 'kitten', 'dog', 'puppy'),
        
        # Color analogies (if available)
        ('red', 'fire', 'blue', 'sky'),
        ('green', 'grass', 'blue', 'sky'),
    ]
    
    # Word similarity pairs (word1, word2, similarity_score)
    # Similarity scores are on a scale of 0-1
    word_pairs = [
        ('king', 'queen', 0.8),
        ('man', 'woman', 0.7),
        ('dog', 'cat', 0.6),
        ('big', 'large', 0.9),
        ('small', 'tiny', 0.9),
        ('prince', 'princess', 0.8),
        ('fire', 'water', 0.1),
        ('sky', 'blue', 0.5),
        ('tree', 'forest', 0.7),
        ('flower', 'garden', 0.6)
    ]
    
    # Semantic coherence groups
    word_groups = {
        'royalty': ['king', 'queen', 'prince', 'princess'],
        'gender': ['man', 'woman'],
        'animals': ['dog', 'cat', 'bird', 'fish'],
        'sizes': ['big', 'large', 'huge', 'small', 'tiny'],
        'nature': ['tree', 'flower', 'forest', 'garden']
    }
    
    return analogy_questions, word_pairs, word_groups


def demonstrate_evaluation():
    """Demonstrate evaluation functionality"""
    print("Word Embedding Evaluation Demonstration")
    print("=" * 50)
    
    # Create evaluation datasets
    analogy_questions, word_pairs, word_groups = create_evaluation_datasets()
    
    print(f"Created evaluation datasets:")
    print(f"  Analogy questions: {len(analogy_questions)}")
    print(f"  Word similarity pairs: {len(word_pairs)}")
    print(f"  Semantic groups: {len(word_groups)}")
    
    # Note: This would typically load trained models
    # For demonstration, we'll show the evaluation framework
    print("\nEvaluation framework ready!")
    print("To use with trained models:")
    print("1. Load your trained Word2Vec/GloVe models")
    print("2. Pass them to EmbeddingEvaluator.compare_models()")
    print("3. Generate plots and reports")
    
    return analogy_questions, word_pairs, word_groups


if __name__ == "__main__":
    demonstrate_evaluation()