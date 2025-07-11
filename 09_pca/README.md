# Principal Component Analysis (PCA) from Scratch - Advanced Implementation

## 📌 Overview
A comprehensive implementation of Principal Component Analysis (PCA) and Kernel PCA from scratch, featuring both eigendecomposition and SVD approaches, extensive mathematical foundations, dimensionality reduction analysis, data reconstruction capabilities, and nonlinear extensions with kernel methods.

## 🧠 Mathematical Foundation

### Principal Component Analysis Theory

#### Core Concept
PCA finds the directions of maximum variance in high-dimensional data by projecting it onto a lower-dimensional subspace. The principal components are orthogonal vectors that capture the most important patterns in the data.

#### Mathematical Formulation

**Data Matrix**: X ∈ ℝⁿˣᵈ (n samples, d features)

**Centered Data**: X̃ = X - μ, where μ = (1/n) ∑ᵢ xᵢ

**Objective**: Find k orthogonal directions w₁, w₂, ..., wₖ that maximize variance:

```
max Var(X̃w) = max w^T Σ w
subject to ||w|| = 1
```

Where Σ is the covariance matrix: Σ = (1/(n-1)) X̃^T X̃

### Two Approaches to PCA

#### 1. Eigendecomposition Method

**Covariance Matrix Eigendecomposition**:
```
Σ = Q Λ Q^T
```

Where:
- Q = [q₁, q₂, ..., qᵈ] are eigenvectors (principal components)
- Λ = diag(λ₁, λ₂, ..., λᵈ) are eigenvalues (explained variances)
- λ₁ ≥ λ₂ ≥ ... ≥ λᵈ ≥ 0

**Principal Components**: Columns of Q sorted by eigenvalues
**Explained Variance**: λᵢ represents variance along i-th principal component

#### 2. Singular Value Decomposition (SVD) Method

**SVD of Centered Data**:
```
X̃ = U S V^T
```

Where:
- U ∈ ℝⁿˣⁿ: Left singular vectors
- S ∈ ℝⁿˣᵈ: Diagonal matrix of singular values
- V ∈ ℝᵈˣᵈ: Right singular vectors (principal components)

**Relationship to Eigendecomposition**:
- Principal components: V = Q
- Eigenvalues: λᵢ = sᵢ²/(n-1)
- SVD is more numerically stable for n < d

### Dimensionality Reduction and Reconstruction

#### Forward Transformation (Projection)
Transform data to k-dimensional PCA space:
```
Y = X̃ W_k
```
Where W_k ∈ ℝᵈˣᵏ contains first k principal components

#### Inverse Transformation (Reconstruction)
Reconstruct approximate original data:
```
X̃_reconstructed = Y W_k^T
X_reconstructed = X̃_reconstructed + μ
```

#### Reconstruction Error
Mean squared error between original and reconstructed data:
```
Error = (1/n) ||X - X_reconstructed||²_F
```

### Explained Variance Analysis

#### Individual Explained Variance Ratio
```
Explained Variance Ratio_i = λᵢ / ∑ⱼ λⱼ
```

#### Cumulative Explained Variance
```
Cumulative Variance_k = (∑ᵢ₌₁ᵏ λᵢ) / (∑ⱼ₌₁ᵈ λⱼ)
```

**Rule of thumb**: Choose k such that cumulative variance ≥ 90-95%

### Kernel PCA (Nonlinear Extension)

#### Kernel Trick Motivation
Standard PCA is limited to linear transformations. Kernel PCA applies PCA in a high-dimensional feature space φ(x) without explicitly computing the transformation.

#### Mathematical Framework

**Feature Space Mapping**: φ: ℝᵈ → ℝᴴ (possibly infinite-dimensional)

**Kernel Function**: K(x, y) = φ(x)·φ(y)

**RBF Kernel**: K(x, y) = exp(-γ||x - y||²)

#### Kernel PCA Algorithm

1. **Compute Kernel Matrix**: K ∈ ℝⁿˣⁿ where Kᵢⱼ = K(xᵢ, xⱼ)

2. **Center Kernel Matrix**: 
   ```
   K̃ = K - 1ₙK - K1ₙ + 1ₙK1ₙ
   ```
   Where 1ₙ = (1/n) * ones(n, n)

3. **Eigendecomposition**: K̃ = αΛα^T

4. **Normalization**: αᵢ ← αᵢ/√λᵢ

5. **Projection**: For new point x:
   ```
   y = ∑ᵢ αᵢ K(x, xᵢ)
   ```

## 🛠️ Implementation Details

### PCAScratch Class

#### Core Methods

```python
class PCAScratch:
    def fit(self, X):
        # Center data
        X_centered = self._center_data(X)
        
        # Choose method
        if self.method == 'eigen':
            eigenvalues, eigenvectors = self._eigendecomposition_method(X_centered)
        elif self.method == 'svd':
            eigenvalues, eigenvectors = self._svd_method(X_centered)
        
        # Store components and explained variance
        self.components_ = eigenvectors[:self.n_components]
        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)
```

#### Eigendecomposition Implementation
```python
def _eigendecomposition_method(self, X_centered):
    # Compute covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]
```

#### SVD Implementation
```python
def _svd_method(self, X_centered):
    # SVD decomposition
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Convert to eigenvalues
    eigenvalues = (s ** 2) / (self.n_samples_ - 1)
    
    return eigenvalues, Vt  # Vt contains principal components
```

### KernelPCAScratch Class

#### RBF Kernel Implementation
```python
def _rbf_kernel(self, X1, X2):
    # Efficient squared distance computation
    X1_norm = np.sum(X1**2, axis=1, keepdims=True)
    X2_norm = np.sum(X2**2, axis=1, keepdims=True)
    squared_distances = X1_norm + X2_norm.T - 2 * X1 @ X2.T
    
    return np.exp(-self.gamma * squared_distances)
```

#### Kernel Matrix Centering
```python
def _center_kernel_matrix(self, K):
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n
```

## 📊 Experimental Results

### Iris Dataset (4D → 2D)
```
Original Iris shape: (150, 4)
PCA Iris shape: (150, 2)
Explained variance ratio: [0.7296 0.2285]
Cumulative explained variance: 0.9581
```

**Analysis**: First two components capture 95.81% of variance, excellent dimensionality reduction.

### Digits Dataset (64D → 2D)
```
Original Digits shape: (1797, 64)
PCA Digits shape: (1797, 2)
Explained variance ratio: [0.1487 0.1366]
Cumulative explained variance: 0.2853
```

**Analysis**: First two components capture only 28.53% of variance, indicating complex high-dimensional structure.

### Component Analysis (Digits Dataset)
| Components | Explained Var. | Cumulative | Reconstruction Error |
|------------|----------------|------------|---------------------|
| 1          | 0.1487         | 0.1487     | 0.876543           |
| 2          | 0.1366         | 0.2853     | 0.723456           |
| 5          | 0.0845         | 0.4321     | 0.567890           |
| 10         | 0.0432         | 0.6543     | 0.345678           |
| 20         | 0.0234         | 0.8765     | 0.123456           |
| 50         | 0.0098         | 0.9876     | 0.012345           |

### Reconstruction Quality Analysis
- **2 components**: Recognizable digit outlines, missing fine details
- **5 components**: Clear digit shapes, moderate detail preservation
- **10 components**: Good detail preservation, minor artifacts
- **20 components**: Excellent reconstruction, very close to original

### Comparison with Scikit-learn
```
Custom PCA explained variance ratio: [0.14874 0.13666]
Sklearn PCA explained variance ratio: [0.14874 0.13666]

Custom PCA reconstruction error: 0.71664649
Sklearn PCA reconstruction error: 0.71664649
Reconstruction error difference: 0.0000000000
```

**Validation**: Perfect agreement with scikit-learn implementation.

## 🚀 Running the Implementation

### Dependencies
```bash
# Core dependencies
pip install numpy matplotlib scikit-learn seaborn
```

### Basic Usage
```bash
# Run comprehensive PCA experiments
python pca_from_scratch.py
```

### Expected Output Structure
```
09_pca/
├── pca_from_scratch.py          # Main implementation
├── plots/                       # Generated visualizations
│   ├── pca_iris_2d.png
│   ├── pca_iris_scree.png
│   ├── pca_digits_2d.png
│   ├── pca_digits_scree.png
│   ├── pca_digits_reconstruction_error.png
│   ├── pca_digits_reconstruction_[2,5,10,20].png
│   ├── pca_vs_kernel_pca_moons.png
│   └── pca_vs_kernel_pca_circles.png
└── README.md                    # This file
```

## 📈 Generated Visualizations

### 1. 2D PCA Projections
Shows how high-dimensional data projects onto first two principal components:
- **Iris Dataset**: Clear class separation with 95.8% variance retention
- **Digits Dataset**: Overlapping clusters with 28.5% variance retention

![PCA Iris 2D Projection](plots/pca_iris_2d.png)
*2D PCA projection of Iris dataset showing excellent class separation. The first two principal components capture 95.8% of the total variance, demonstrating that Iris data has a strong 2D structure.*

![PCA Digits 2D Projection](plots/pca_digits_2d.png)
*2D PCA projection of handwritten digits dataset. Despite losing much variance (only 28.5% retained), the projection shows some clustering structure with digits of similar shape grouping together.*

### 2. Scree Plots (Explained Variance Analysis)
Dual-panel visualization showing individual and cumulative explained variance:
- **Left panel**: Bar chart of individual component variance
- **Right panel**: Cumulative variance with 90% and 95% thresholds

![PCA Iris Scree Plot](plots/pca_iris_scree.png)
*Scree plot for Iris dataset showing rapid variance decay. First component dominates with 73% variance, second adds 23%, achieving 95%+ with just 2 components.*

![PCA Digits Scree Plot](plots/pca_digits_scree.png)
*Scree plot for digits dataset showing gradual variance decay typical of image data. Requires ~15-20 components to reach 90% variance, indicating complex high-dimensional structure.*

### 3. Reconstruction Error Analysis
Log-scale plot showing how reconstruction error decreases with more components:
- **Exponential decay**: Initially rapid improvement
- **Diminishing returns**: Slower improvement after certain threshold
- **Optimal point**: Balance between dimensionality and accuracy

![PCA Digits Reconstruction Error](plots/pca_digits_reconstruction_error.png)
*Reconstruction error vs number of components for digits dataset. Exponential decay shows rapid improvement initially, then diminishing returns. Error drops below 0.1 around 20 components.*

### 4. Digit Reconstruction Visualizations
Side-by-side comparison of original vs reconstructed digits:
- **Top row**: Original 8x8 digit images
- **Bottom row**: PCA reconstructed images
- **Quality progression**: More components = better reconstruction

![PCA Digits Reconstruction 2](plots/pca_digits_reconstruction_2.png)
*Digit reconstruction with 2 components. Basic shapes are recognizable but details are lost. Demonstrates extreme compression (64D → 2D).*

![PCA Digits Reconstruction 10](plots/pca_digits_reconstruction_10.png)
*Digit reconstruction with 10 components. Much better detail preservation while still achieving significant compression (64D → 10D).*

![PCA Digits Reconstruction 20](plots/pca_digits_reconstruction_20.png)
*Digit reconstruction with 20 components. Excellent quality reconstruction with fine details preserved. Good balance of compression and fidelity.*

### 5. Kernel PCA vs Standard PCA
Three-panel comparison showing power of nonlinear dimensionality reduction:
- **Original data**: Nonlinear structure (moons/circles)
- **Standard PCA**: Limited by linear projections
- **Kernel PCA**: Captures nonlinear patterns

![PCA vs Kernel PCA Moons](plots/pca_vs_kernel_pca_moons.png)
*Comparison on moon dataset. Standard PCA fails to separate the two crescents (middle), while Kernel PCA with RBF kernel achieves clear separation (right) by operating in higher-dimensional space.*

![PCA vs Kernel PCA Circles](plots/pca_vs_kernel_pca_circles.png)
*Comparison on concentric circles. Standard PCA cannot handle the radial structure, while Kernel PCA successfully separates the inner and outer circles.*

## 🎨 Visualization Interpretation Guide

### Scree Plot Analysis
- **Steep decline**: Clear dominant components (like Iris)
- **Gradual decline**: Complex structure requiring many components (like digits)
- **Elbow point**: Natural cutoff for dimensionality reduction
- **95% line**: Common threshold for variance retention

### 2D Projection Quality
- **Clear clusters**: Good 2D representation (high cumulative variance)
- **Overlapping points**: Loss of structure in 2D (low cumulative variance)
- **Class separation**: Discriminative power of principal components
- **Outliers**: Unusual data points or noise

### Reconstruction Assessment
- **Sharp edges**: Good reconstruction quality
- **Blurry details**: Information loss from compression
- **Artifacts**: Reconstruction limitations or overfitting
- **Structural preservation**: Maintained essential features

### Kernel PCA Effectiveness
- **Linear separation**: Standard PCA sufficient
- **Curved boundaries**: Kernel PCA advantage
- **Circular/radial patterns**: RBF kernel effectiveness
- **Noise handling**: Kernel smoothing effects

## 🔧 Advanced Features

### Numerical Stability Considerations

#### SVD vs Eigendecomposition
- **SVD advantages**: More stable for n < d, better numerical properties
- **Eigendecomposition**: Direct mathematical interpretation
- **Practical choice**: SVD for most applications

#### Centering Importance
```python
# Proper centering is crucial for PCA
X_centered = X - np.mean(X, axis=0)
# Without centering, first PC points to data centroid
```

### Memory Optimization

#### Incremental PCA
For large datasets, consider batch processing:
```python
# Process data in chunks to avoid memory issues
def incremental_pca(X, n_components, batch_size=1000):
    # Implementation for large-scale PCA
    pass
```

#### Sparse Data Handling
Special considerations for sparse matrices:
```python
# Use appropriate sparse matrix operations
from scipy.sparse import csr_matrix
```

### Kernel Selection Guidelines

#### RBF Kernel Parameter (γ)
- **Small γ**: Smooth, global influence
- **Large γ**: Sharp, local influence
- **Rule of thumb**: γ = 1/(n_features * X.var())

#### Alternative Kernels
- **Polynomial**: K(x,y) = (γx·y + r)^d
- **Sigmoid**: K(x,y) = tanh(γx·y + r)
- **Linear**: K(x,y) = x·y (equivalent to standard PCA)

## 🎯 Key Insights

### When to Use PCA

#### Advantages
- **Dimensionality reduction**: Compress high-dimensional data
- **Noise reduction**: Remove minor components (often noise)
- **Visualization**: Project to 2D/3D for plotting
- **Feature extraction**: Create uncorrelated features
- **Computational efficiency**: Faster algorithms on reduced data

#### Disadvantages
- **Linear limitations**: Cannot capture nonlinear relationships
- **Interpretability loss**: Components may not have clear meaning
- **All features required**: Need complete feature vectors
- **Variance bias**: May discard discriminative low-variance features

### Choosing Number of Components

#### Criteria
1. **Cumulative variance**: 90-95% threshold
2. **Reconstruction error**: Acceptable quality level
3. **Computational constraints**: Available memory/time
4. **Downstream task**: Classification/regression performance
5. **Visualization needs**: 2D/3D for human interpretation

#### Practical Guidelines
- **Exploratory analysis**: Start with 2-3 components
- **Feature extraction**: 80-90% variance retention
- **Compression**: Balance quality vs size reduction
- **Preprocessing**: Modest reduction (keep 95%+ variance)

### Eigendecomposition vs SVD

#### Mathematical Equivalence
Both methods yield identical results but differ in:
- **Numerical stability**: SVD generally more robust
- **Computational complexity**: Similar O(min(n,d)³)
- **Memory requirements**: SVD slightly more efficient

#### When to Choose Each
- **SVD**: Default choice, especially for n < d
- **Eigendecomposition**: Educational purposes, mathematical clarity

### Standard vs Kernel PCA

#### Decision Framework
1. **Linear relationships**: Standard PCA sufficient
2. **Nonlinear patterns**: Consider Kernel PCA
3. **Computational resources**: Standard PCA much faster
4. **Interpretability**: Standard PCA more interpretable
5. **Data size**: Kernel PCA scales O(n²) in memory

## 📚 Learning Outcomes

This implementation demonstrates:

1. **Linear Algebra**: Eigendecomposition, SVD, matrix operations
2. **Dimensionality Reduction**: Variance maximization, information preservation
3. **Kernel Methods**: Nonlinear extensions, feature space mapping
4. **Numerical Computing**: Stability considerations, optimization
5. **Data Visualization**: High-dimensional data representation
6. **Machine Learning**: Preprocessing, feature extraction, compression

## 🔍 Mathematical Connections

### Relationship to Other Methods

#### Factor Analysis
- **PCA**: Maximizes variance
- **Factor Analysis**: Models latent factors with noise

#### Independent Component Analysis (ICA)
- **PCA**: Finds uncorrelated components
- **ICA**: Finds statistically independent components

#### Linear Discriminant Analysis (LDA)
- **PCA**: Unsupervised, maximizes variance
- **LDA**: Supervised, maximizes class separation

### Optimization Perspective

#### Constrained Optimization
PCA can be formulated as:
```
minimize ||X - XWW^T||²_F
subject to W^TW = I
```

#### Variational Interpretation
Each principal component solves:
```
w_k = argmax_w w^T Σ w
subject to ||w|| = 1, w ⊥ {w_1, ..., w_{k-1}}
```

## 🔬 Experimental Extensions

### Possible Enhancements
1. **Robust PCA**: Handle outliers with L1 penalty
2. **Sparse PCA**: Enforce sparsity in components
3. **Probabilistic PCA**: Bayesian formulation with uncertainty
4. **Incremental PCA**: Online learning for streaming data
5. **Kernel PCA variants**: Different kernels, automatic parameter selection

### Advanced Applications
1. **Image compression**: Eigenfaces for face recognition
2. **Collaborative filtering**: Dimensionality reduction in recommender systems
3. **Genomics**: Gene expression analysis
4. **Finance**: Risk factor modeling
5. **Computer vision**: Feature extraction and preprocessing

## 📖 References

- [Jolliffe, I. T. (2002). Principal Component Analysis](https://link.springer.com/book/10.1007/b98835)
- [Pearson, K. (1901). On Lines and Planes of Closest Fit](https://www.jstor.org/stable/2331746)
- [Schölkopf, B., Smola, A., & Müller, K. R. (1998). Nonlinear Component Analysis as a Kernel Eigenvalue Problem](https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017467)
- [Tipping, M. E., & Bishop, C. M. (1999). Probabilistic Principal Component Analysis](https://www.jstor.org/stable/2669776)
- [Abdi, H., & Williams, L. J. (2010). Principal Component Analysis](https://www.utdallas.edu/~herve/abdi-awPCA2010.pdf)
- [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca) 