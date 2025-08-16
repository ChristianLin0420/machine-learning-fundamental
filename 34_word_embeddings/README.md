# Word Embeddings (Word2Vec, GloVe)

## 📌 Overview
Comprehensive implementation of word embedding techniques that convert text into dense vector representations capturing semantic relationships. This implementation includes both Word2Vec (Skip-gram and CBOW) and GloVe models built from scratch with comparative evaluation and visualization tools.

## 🧠 Theory and Mathematical Foundations

### What are Word Embeddings?
Word embeddings are dense vector representations of words where:
- Each word is mapped to a continuous vector space
- Semantic similarity is preserved in geometric relationships
- Mathematical operations can capture linguistic relationships

**Example**: `vec("king") - vec("man") + vec("woman") ≈ vec("queen")`

### Word2Vec Models

**Skip-gram Model**: Predicts context words given a center word
- **Objective**: Maximize log probability of context words
- **Advantages**: Works well with rare words, captures detailed relationships
- **Use cases**: Large vocabularies, detailed semantic analysis

**CBOW (Continuous Bag of Words)**: Predicts center word from context
- **Objective**: Predict target word from average of context vectors
- **Advantages**: Faster training, better for frequent words
- **Use cases**: Smaller datasets, efficiency-focused applications

**Negative Sampling**: Computational efficiency technique
- Instead of computing expensive softmax over entire vocabulary
- Sample a few "negative" examples for each positive example
- Reduces complexity from O(V) to O(k) where k << V

### GloVe (Global Vectors)

**Core Idea**: Combine global matrix factorization with local context methods

**Objective Function**:
$$J = \sum_{i,j=1}^{V} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$$

Where:
- `X_ij` = co-occurrence count of words i and j
- `f(x)` = weighting function: `f(x) = (x/x_max)^α if x < x_max else 1`
- `w_i`, `w̃_j` = word and context vectors
- `b_i`, `b̃_j` = word and context biases

**Advantages**:
- Uses global corpus statistics
- Efficient training on co-occurrence matrix
- Often more consistent than Word2Vec

## 🛠️ Implementation Architecture

### Files Overview
- `word2vec_from_scratch.py`: Complete Word2Vec implementation (Skip-gram + CBOW)
- `glove_from_scratch.py`: Complete GloVe implementation with co-occurrence matrix
- `evaluate_embeddings.py`: Comprehensive evaluation suite (analogies, similarities)
- `visualize_embeddings.py`: Visualization tools (t-SNE, PCA, heat maps)
- `comprehensive_demo.py`: Full pipeline demonstration

### Model Comparison

| Feature | Word2Vec (Skip-gram) | Word2Vec (CBOW) | GloVe |
|---------|---------------------|-----------------|-------|
| Method | Predictive (neural) | Predictive (neural) | Count-based (matrix factorization) |
| Training | Online, local context | Online, local context | Batch, global statistics |
| Rare words | Excellent | Poor | Good |
| Frequent words | Good | Excellent | Good |
| Speed | Medium | Fast | Medium |
| Memory | Medium | Medium | High (co-occurrence matrix) |

## 🔬 Experimental Setup and Results

### Training Configuration
- **Embedding Dimension**: 50
- **Window Size**: 5
- **Minimum Count**: 3
- **Negative Samples**: 5 (Word2Vec)
- **Learning Rate**: 0.025 (Word2Vec), 0.05 (GloVe)
- **Epochs**: 15 (Word2Vec), 150 iterations (GloVe)

### Corpus Statistics
- **Total Sentences**: ~1,000+ diverse sentences
- **Vocabulary Size**: ~80-100 unique words
- **Domains**: Royalty, animals, nature, colors, sizes
- **Semantic Clusters**: Gender, size, animal relationships

### Evaluation Metrics

**1. Analogy Task Performance**
- **Format**: A is to B as C is to D
- **Examples**: king - man + woman = ? (Expected: queen)
- **Metric**: Top-k accuracy (k=1,3,5)

**2. Word Similarity Correlation**
- **Method**: Cosine similarity vs human judgments
- **Metrics**: Spearman and Pearson correlations
- **Coverage**: Percentage of test pairs in vocabulary

**3. Semantic Coherence**
- **Method**: Average pairwise similarity within categories
- **Categories**: Royalty, animals, sizes, colors
- **Metric**: Average cosine similarity

## 📊 Visualizations and Analysis

### Generated Plots

#### Training Dynamics
- `plots/word2vec_training_curves.png`: Individual model convergence
  - Shows the convergence behavior of Skip-gram and CBOW models
  - Skip-gram typically has higher initial loss but may achieve better final performance on rare words
  - CBOW converges faster due to averaging context vectors

- `plots/glove_training_curve.png`: GloVe optimization progress
  - GloVe loss on log scale showing weighted least squares optimization
  - Loss should decrease monotonically with occasional plateaus
  - Models global co-occurrence patterns

#### Embedding Visualizations
- `plots/tsne_*.png`: t-SNE visualizations for each model
  - Shows semantic clusters formed by embeddings
  - Look for tight clusters of related words (animals, royalty, sizes)
  - Reveals meaningful spatial relationships between concepts

- `plots/model_comparison_tsne.png`: Side-by-side comparison
  - Direct comparison of all three models on the same words
  - Reveals differences in how each method captures semantic relationships
  - Helps identify model-specific clustering patterns

#### Evaluation Results
- `plots/comprehensive_evaluation.png`: Multi-metric comparison
  - Summary of analogy accuracy, vocabulary coverage, similarity correlations
  - Identifies strengths and weaknesses of each approach
  - Guides model selection for specific tasks

- `plots/similarity_heatmap_*.png`: Word similarity matrices
  - Heat maps showing cosine similarities between word pairs
  - Validates that semantically related words have high similarity scores
  - Useful for debugging embedding quality

### Interpretation Guidelines

**t-SNE Plots**:
- **Tight clusters**: Strong semantic coherence
- **Spatial relationships**: Geometric preservation of word relationships
- **Outliers**: Words with unique or ambiguous semantics

**Training Curves**:
- **Smooth decrease**: Healthy optimization
- **Oscillations**: Learning rate too high
- **Plateaus**: Convergence or local minima

## 🏃‍♂️ How to Run

### Quick Start
```bash
# Run individual model demonstrations
python word2vec_from_scratch.py    # Word2Vec models
python glove_from_scratch.py       # GloVe model

# Run comprehensive pipeline
python comprehensive_demo.py      # Full demonstration
```

### Step-by-Step Usage

**1. Train Individual Models**:
```python
from word2vec_from_scratch import Word2Vec
from glove_from_scratch import GloVe

# Create corpus
corpus = ["the king is strong", "the queen is beautiful", ...]

# Train Skip-gram
model = Word2Vec(model_type='skipgram', embedding_dim=50)
model.fit(corpus)

# Train GloVe
glove = GloVe(embedding_dim=50)
glove.fit(corpus)
```

**2. Evaluate Models**:
```python
from evaluate_embeddings import EmbeddingEvaluator

evaluator = EmbeddingEvaluator()
results = evaluator.compare_models({
    'Skip-gram': skipgram_model,
    'GloVe': glove_model
}, analogy_questions, word_pairs)
```

**3. Create Visualizations**:
```python
from visualize_embeddings import EmbeddingVisualizer

visualizer = EmbeddingVisualizer()
visualizer.plot_tsne_embeddings(model, words, categories)
visualizer.plot_analogy_visualization(model, 'king', 'man', 'queen')
```

### Expected Outputs
- **Models**: Saved to `plots/*.pkl`
- **Plots**: Generated in `plots/` directory
- **Report**: `plots/evaluation_report.txt`
- **Console**: Detailed progress and results

## 💡 Practical Applications

### Use Cases
1. **Semantic Search**: Find similar documents or concepts
2. **Machine Translation**: Cross-lingual word alignments
3. **Sentiment Analysis**: Capture emotional word relationships
4. **Recommendation Systems**: Item similarity based on descriptions
5. **Knowledge Graphs**: Entity relationship modeling

### Model Selection Guidelines

**Choose Skip-gram when**:
- Working with rare or specialized vocabulary
- Need detailed semantic relationships
- Have sufficient computational resources

**Choose CBOW when**:
- Training speed is priority
- Working with frequent, common words
- Limited computational resources

**Choose GloVe when**:
- Need consistent, stable embeddings
- Have sufficient memory for co-occurrence matrix
- Want to leverage global corpus statistics

## 🔍 Advanced Topics

### Numerical Stability
- **Stable softmax**: Subtract max for overflow prevention
- **Gradient clipping**: Prevent exploding gradients
- **Learning rate decay**: Improve convergence

### Optimization Techniques
- **AdaGrad**: Adaptive learning rates (GloVe)
- **Negative sampling**: Computational efficiency (Word2Vec)
- **Hierarchical softmax**: Alternative to negative sampling

### Extensions
- **FastText**: Subword information for rare words
- **Contextual embeddings**: BERT, GPT (context-dependent)
- **Multilingual embeddings**: Cross-lingual representations

## 🎯 Key Takeaways

1. **Word embeddings capture semantic relationships** through geometric properties of vector spaces
2. **Skip-gram excels with rare words**, CBOW with frequent words, GloVe with global consistency
3. **Proper evaluation requires multiple metrics**: analogies, similarities, coherence
4. **Visualization is crucial** for understanding embedding quality and relationships
5. **Hyper-parameter tuning significantly impacts** final embedding quality
6. **Implementation details matter**: numerical stability, optimization, preprocessing

## 📚 References

### Original Papers
- [Mikolov et al., 2013](https://arxiv.org/abs/1301.3781): Efficient Estimation of Word Representations in Vector Space
- [Pennington et al., 2014](https://nlp.stanford.edu/pubs/glove.pdf): GloVe: Global Vectors for Word Representation

### Implementation Guides
- [Understanding Word2Vec](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [GloVe Implementation Details](https://nlp.stanford.edu/projects/glove/)
- [Negative Sampling Explained](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)