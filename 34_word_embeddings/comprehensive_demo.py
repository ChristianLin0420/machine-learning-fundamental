"""
Comprehensive Word Embeddings Demo
==================================

Complete demonstration of Word2Vec and GloVe implementations including:
- Training both models on the same corpus
- Comparative evaluation on multiple tasks
- Comprehensive visualizations
- Performance analysis and insights

This script runs the full pipeline from training to evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from word2vec_from_scratch import Word2Vec, create_sample_corpus
from glove_from_scratch import GloVe, create_extended_corpus
from evaluate_embeddings import EmbeddingEvaluator, create_evaluation_datasets
from visualize_embeddings import EmbeddingVisualizer, create_visualization_categories


def create_comprehensive_corpus():
    """Create a comprehensive corpus combining both Word2Vec and GloVe corpora"""
    
    # Get base corpora
    w2v_corpus = create_sample_corpus()
    glove_corpus = create_extended_corpus()
    
    # Additional domain-specific content
    additional_content = [
        # Technology
        "computer processes data quickly and efficiently",
        "software runs on hardware systems",
        "internet connects people around world",
        "phone calls friends and family",
        
        # Food
        "apple is red sweet fruit",
        "banana is yellow curved fruit", 
        "bread is baked from wheat",
        "milk comes from cow",
        
        # Weather
        "snow falls in winter season",
        "sun shines in summer heat",
        "rain drops from cloudy sky",
        "wind blows leaves from tree",
        
        # Transportation
        "car drives on road fast",
        "train travels on railway track",
        "plane flies high in sky",
        "boat sails on water ocean",
        
        # Education
        "student learns in school building",
        "teacher teaches important lessons daily",
        "book contains knowledge and wisdom",
        "pen writes words on paper",
        
        # Sports
        "ball bounces on ground surface",
        "player runs fast in game",
        "team wins important match today",
        "goal scores points for victory"
    ]
    
    # Combine all corpora with repetition for training stability
    combined_corpus = (w2v_corpus + glove_corpus + additional_content * 10)
    
    print(f"Created comprehensive corpus with {len(combined_corpus)} sentences")
    return combined_corpus


def train_all_models(corpus, save_models=True):
    """Train Word2Vec and GloVe models on the same corpus"""
    
    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)
    
    models = {}
    training_times = {}
    
    # Common parameters for fair comparison
    embedding_dim = 50
    window_size = 5
    min_count = 3
    
    # Train Skip-gram Word2Vec
    print(f"\n{'Training Skip-gram Word2Vec':-^60}")
    start_time = time.time()
    
    skipgram_model = Word2Vec(
        embedding_dim=embedding_dim,
        window_size=window_size,
        min_count=min_count,
        negative_samples=5,
        learning_rate=0.025,
        epochs=15,
        model_type='skipgram'
    )
    
    skipgram_model.fit(corpus)
    training_times['Skip-gram'] = time.time() - start_time
    models['Skip-gram'] = skipgram_model
    
    if save_models:
        skipgram_model.save_model('plots/skipgram_comprehensive.pkl')
    
    # Train CBOW Word2Vec
    print(f"\n{'Training CBOW Word2Vec':-^60}")
    start_time = time.time()
    
    cbow_model = Word2Vec(
        embedding_dim=embedding_dim,
        window_size=window_size,
        min_count=min_count,
        negative_samples=5,
        learning_rate=0.025,
        epochs=15,
        model_type='cbow'
    )
    
    cbow_model.fit(corpus)
    training_times['CBOW'] = time.time() - start_time
    models['CBOW'] = cbow_model
    
    if save_models:
        cbow_model.save_model('plots/cbow_comprehensive.pkl')
    
    # Train GloVe
    print(f"\n{'Training GloVe':-^60}")
    start_time = time.time()
    
    glove_model = GloVe(
        embedding_dim=embedding_dim,
        window_size=window_size,
        min_count=min_count,
        learning_rate=0.05,
        max_iter=150,
        x_max=100.0,
        alpha=0.75
    )
    
    glove_model.fit(corpus)
    training_times['GloVe'] = time.time() - start_time
    models['GloVe'] = glove_model
    
    if save_models:
        glove_model.save_model('plots/glove_comprehensive.pkl')
    
    # Print training summary
    print(f"\n{'Training Summary':-^60}")
    for model_name, train_time in training_times.items():
        vocab_size = len(models[model_name].vocab)
        print(f"{model_name:>12}: {train_time:6.1f}s, Vocab: {vocab_size:4d} words")
    
    return models, training_times


def comprehensive_evaluation(models):
    """Run comprehensive evaluation on all models"""
    
    print(f"\n{'COMPREHENSIVE EVALUATION':-^60}")
    print("=" * 60)
    
    # Create evaluation datasets
    analogy_questions, word_pairs, word_groups = create_evaluation_datasets()
    
    # Initialize evaluator
    evaluator = EmbeddingEvaluator()
    
    # Run comparative evaluation
    comparison_results = evaluator.compare_models(
        models=models,
        analogy_questions=analogy_questions,
        word_pairs=word_pairs,
        word_groups=word_groups
    )
    
    # Generate comprehensive plots
    evaluator.plot_comparison_results(
        comparison_results, 
        save_path='plots/comprehensive_evaluation.png'
    )
    
    # Generate detailed report
    report = evaluator.generate_evaluation_report(
        comparison_results, 
        save_path='plots/evaluation_report.txt'
    )
    
    print("\nEvaluation completed! Check plots/ for detailed results.")
    
    return comparison_results


def comprehensive_visualization(models):
    """Create comprehensive visualizations for all models"""
    
    print(f"\n{'COMPREHENSIVE VISUALIZATION':-^60}")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer()
    
    # Get visualization categories
    categories = create_visualization_categories()
    
    # Get all words that exist in all models for fair comparison
    common_words = set()
    for model in models.values():
        if hasattr(model, 'vocab'):
            if not common_words:
                common_words = set(model.vocab.keys())
            else:
                common_words = common_words.intersection(set(model.vocab.keys()))
    
    # Select representative words from each category
    viz_words = []
    for category, words in categories.items():
        available_words = [w for w in words if w in common_words]
        viz_words.extend(available_words[:3])  # Max 3 words per category
    
    viz_words = list(set(viz_words))[:30]  # Limit total words for clarity
    
    print(f"Visualizing {len(viz_words)} common words across all models")
    
    # Create individual model visualizations
    for model_name, model in models.items():
        print(f"Creating t-SNE visualization for {model_name}...")
        
        visualizer.plot_tsne_embeddings(
            model=model,
            words=viz_words,
            categories=categories,
            save_path=f'plots/tsne_{model_name.lower().replace("-", "_")}.png',
            title=f't-SNE Visualization: {model_name}'
        )
        
        visualizer.plot_pca_embeddings(
            model=model,
            words=viz_words,
            categories=categories,
            save_path=f'plots/pca_{model_name.lower().replace("-", "_")}.png',
            title=f'PCA Visualization: {model_name}'
        )
    
    # Create comparative visualization
    print("Creating comparative visualization...")
    visualizer.compare_embeddings_2d(
        models=models,
        words=viz_words[:20],  # Limit for clarity
        method='tsne',
        save_path='plots/model_comparison_tsne.png'
    )
    
    visualizer.compare_embeddings_2d(
        models=models,
        words=viz_words[:20],
        method='pca', 
        save_path='plots/model_comparison_pca.png'
    )
    
    # Create analogy visualizations for each model
    analogy_tests = [
        ('king', 'man', 'queen'),
        ('big', 'small', 'huge'),
        ('dog', 'puppy', 'cat')
    ]
    
    for word_a, word_b, word_c in analogy_tests:
        if all(word in common_words for word in [word_a, word_b, word_c]):
            analogy_name = f"{word_a}_{word_b}_{word_c}"
            
            for model_name, model in models.items():
                print(f"Creating analogy visualization: {word_a}-{word_b}+{word_c} ({model_name})")
                
                visualizer.plot_analogy_visualization(
                    model=model,
                    word_a=word_a,
                    word_b=word_b,
                    word_c=word_c,
                    save_path=f'plots/analogy_{analogy_name}_{model_name.lower().replace("-", "_")}.png'
                )
    
    # Create similarity heatmaps
    heatmap_words = ['king', 'queen', 'man', 'woman', 'prince', 'princess']
    available_heatmap_words = [w for w in heatmap_words if w in common_words]
    
    if len(available_heatmap_words) >= 4:
        for model_name, model in models.items():
            print(f"Creating similarity heatmap for {model_name}...")
            
            visualizer.plot_similarity_heatmap(
                model=model,
                words=available_heatmap_words,
                save_path=f'plots/similarity_heatmap_{model_name.lower().replace("-", "_")}.png',
                title=f'Word Similarity Heatmap: {model_name}'
            )
    
    print("Visualization completed! Check plots/ for all visualizations.")


def analyze_training_dynamics(models):
    """Analyze and visualize training dynamics"""
    
    print(f"\n{'TRAINING DYNAMICS ANALYSIS':-^60}")
    print("=" * 60)
    
    # Plot training curves comparison
    plt.figure(figsize=(15, 5))
    
    # Individual training curves
    plt.subplot(1, 3, 1)
    if 'Skip-gram' in models and hasattr(models['Skip-gram'], 'loss_history'):
        plt.plot(models['Skip-gram'].loss_history, label='Skip-gram', linewidth=2)
    plt.title('Skip-gram Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    if 'CBOW' in models and hasattr(models['CBOW'], 'loss_history'):
        plt.plot(models['CBOW'].loss_history, label='CBOW', linewidth=2, color='orange')
    plt.title('CBOW Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 3)
    if 'GloVe' in models and hasattr(models['GloVe'], 'loss_history'):
        plt.plot(models['GloVe'].loss_history, label='GloVe', linewidth=2, color='green')
        plt.yscale('log')  # Log scale for GloVe loss
    plt.title('GloVe Training Loss (Log Scale)')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/training_dynamics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined comparison (normalize for comparison)
    plt.figure(figsize=(10, 6))
    
    if 'Skip-gram' in models and hasattr(models['Skip-gram'], 'loss_history'):
        sg_loss = np.array(models['Skip-gram'].loss_history)
        sg_loss_norm = (sg_loss - sg_loss.min()) / (sg_loss.max() - sg_loss.min())
        plt.plot(sg_loss_norm, label='Skip-gram', linewidth=2)
    
    if 'CBOW' in models and hasattr(models['CBOW'], 'loss_history'):
        cbow_loss = np.array(models['CBOW'].loss_history)
        cbow_loss_norm = (cbow_loss - cbow_loss.min()) / (cbow_loss.max() - cbow_loss.min())
        plt.plot(cbow_loss_norm, label='CBOW', linewidth=2)
    
    plt.title('Normalized Training Loss Comparison (Word2Vec)')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/normalized_training_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training dynamics analysis completed!")


def demonstrate_real_world_analogies(models):
    """Test models on interesting real-world analogies"""
    
    print(f"\n{'REAL-WORLD ANALOGIES':-^60}")
    print("=" * 60)
    
    # Define interesting analogy tests
    real_world_analogies = [
        # Basic semantic relationships
        ('king', 'man', 'queen', 'Expected: woman'),
        ('prince', 'king', 'princess', 'Expected: queen'),
        
        # Size relationships
        ('big', 'small', 'huge', 'Expected: tiny'),
        ('large', 'small', 'big', 'Expected: tiny/small'),
        
        # Animal relationships
        ('dog', 'puppy', 'cat', 'Expected: kitten'),
        ('cat', 'kitten', 'dog', 'Expected: puppy'),
        
        # Nature relationships
        ('tree', 'forest', 'flower', 'Expected: garden'),
        ('sun', 'day', 'moon', 'Expected: night'),
    ]
    
    for word_a, word_b, word_c, expected in real_world_analogies:
        print(f"\nAnalogy: {word_a} - {word_b} + {word_c} = ? ({expected})")
        print("-" * 50)
        
        for model_name, model in models.items():
            try:
                # Check if all words are in vocabulary
                if hasattr(model, 'vocab'):
                    if not all(word in model.vocab for word in [word_a, word_b, word_c]):
                        print(f"{model_name:>12}: Missing words in vocabulary")
                        continue
                
                results = model.analogy(word_a, word_b, word_c, top_k=3)
                
                if results:
                    print(f"{model_name:>12}: ", end="")
                    top_predictions = [f"{word} ({score:.3f})" for word, score in results]
                    print(" | ".join(top_predictions))
                else:
                    print(f"{model_name:>12}: No predictions")
                    
            except Exception as e:
                print(f"{model_name:>12}: Error - {e}")


def run_comprehensive_demo():
    """Run the complete comprehensive demonstration"""
    
    print("=" * 80)
    print("COMPREHENSIVE WORD EMBEDDINGS DEMONSTRATION")
    print("=" * 80)
    print("This demo will:")
    print("1. Create a comprehensive training corpus")
    print("2. Train Skip-gram, CBOW, and GloVe models")
    print("3. Run comparative evaluation on multiple tasks")
    print("4. Generate comprehensive visualizations")
    print("5. Analyze training dynamics")
    print("6. Test real-world analogies")
    print("=" * 80)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Step 1: Create comprehensive corpus
    print(f"\n{'STEP 1: CORPUS CREATION':-^60}")
    corpus = create_comprehensive_corpus()
    
    # Step 2: Train all models
    print(f"\n{'STEP 2: MODEL TRAINING':-^60}")
    models, training_times = train_all_models(corpus)
    
    # Step 3: Comprehensive evaluation
    print(f"\n{'STEP 3: EVALUATION':-^60}")
    evaluation_results = comprehensive_evaluation(models)
    
    # Step 4: Comprehensive visualization  
    print(f"\n{'STEP 4: VISUALIZATION':-^60}")
    comprehensive_visualization(models)
    
    # Step 5: Training dynamics analysis
    print(f"\n{'STEP 5: TRAINING ANALYSIS':-^60}")
    analyze_training_dynamics(models)
    
    # Step 6: Real-world analogies
    print(f"\n{'STEP 6: REAL-WORLD TESTS':-^60}")
    demonstrate_real_world_analogies(models)
    
    # Final summary
    print(f"\n{'DEMONSTRATION COMPLETE':-^60}")
    print("=" * 60)
    print("Generated files in plots/ directory:")
    print("- Model training curves and comparisons")
    print("- Comprehensive evaluation plots and report")
    print("- t-SNE and PCA visualizations for each model")
    print("- Model comparison visualizations")
    print("- Analogy visualizations")
    print("- Similarity heatmaps")
    print("- Training dynamics analysis")
    print()
    print("Key findings:")
    
    # Print key insights
    if evaluation_results and 'analogy_results' in evaluation_results:
        print("\nAnalogy Task Performance:")
        for model_name, result in evaluation_results['analogy_results'].items():
            print(f"  {model_name}: {result['accuracy']:.3f} accuracy")
    
    print(f"\nTraining Times:")
    for model_name, train_time in training_times.items():
        print(f"  {model_name}: {train_time:.1f} seconds")
    
    print(f"\nVocabulary Sizes:")
    for model_name, model in models.items():
        if hasattr(model, 'vocab_size'):
            print(f"  {model_name}: {model.vocab_size} words")
    
    print("\nRecommendations:")
    print("- Skip-gram: Good for rare words, captures detailed relationships")
    print("- CBOW: Faster training, good for frequent words")
    print("- GloVe: Global statistics, consistent performance")
    
    return models, evaluation_results


if __name__ == "__main__":
    models, results = run_comprehensive_demo()