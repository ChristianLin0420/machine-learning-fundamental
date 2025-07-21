# Bias-Variance Decomposition: Mathematical Theory

A comprehensive mathematical treatment of the bias-variance tradeoff in machine learning, including formal derivations and intuitive explanations.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Framework](#mathematical-framework)
3. [Bias-Variance Decomposition for Regression](#bias-variance-decomposition-for-regression)
4. [Detailed Mathematical Derivation](#detailed-mathematical-derivation)
5. [Bias-Variance Decomposition for Classification](#bias-variance-decomposition-for-classification)
6. [Sources of Bias and Variance](#sources-of-bias-and-variance)
7. [Model Complexity and the Tradeoff](#model-complexity-and-the-tradeoff)
8. [Practical Implications](#practical-implications)
9. [Examples and Applications](#examples-and-applications)
10. [References](#references)

## Introduction

The bias-variance tradeoff is one of the most fundamental concepts in machine learning and statistical learning theory. It provides a framework for understanding the sources of prediction error and guides us in selecting appropriate model complexity.

### Key Concepts

- **Bias**: Error introduced by approximating a real-world problem with a simplified model
- **Variance**: Error introduced by sensitivity to small fluctuations in the training set
- **Irreducible Error**: Error that cannot be reduced regardless of the algorithm used

## Mathematical Framework

### Notation

Let's establish our mathematical notation:

- $X$: Input features (random variable)
- $Y$: True target values (random variable)
- $f(x)$: True underlying function, where $Y = f(X) + \epsilon$
- $\epsilon$: Irreducible noise, $\epsilon \sim \mathcal{N}(0, \sigma^2)$
- $\mathcal{D}$: Training dataset
- $\hat{f}_{\mathcal{D}}(x)$: Learned function from dataset $\mathcal{D}$
- $\hat{y}$: Prediction made by $\hat{f}_{\mathcal{D}}(x)$

### The Learning Problem

We want to learn a function $\hat{f}$ that minimizes the expected prediction error:

$$\text{Expected Error} = \mathbb{E}[(Y - \hat{f}(X))^2]$$

where the expectation is taken over all possible values of $X$, $Y$, and training datasets $\mathcal{D}$.

## Bias-Variance Decomposition for Regression

### The Main Result

For a fixed point $x_0$, the expected mean squared error can be decomposed as:

$$\mathbb{E}[(Y - \hat{f}(x_0))^2] = \text{Bias}^2[\hat{f}(x_0)] + \text{Var}[\hat{f}(x_0)] + \sigma^2$$

where:

$$\text{Bias}[\hat{f}(x_0)] = \mathbb{E}[\hat{f}(x_0)] - f(x_0)$$

$$\text{Var}[\hat{f}(x_0)] = \mathbb{E}[(\hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)])^2]$$

$$\sigma^2 = \text{Var}[\epsilon] = \text{Irreducible Error}$$

### Intuitive Interpretation

1. **Bias²**: How far off is our model on average?
2. **Variance**: How much do our predictions vary across different training sets?
3. **Irreducible Error**: Noise in the data that cannot be eliminated

## Detailed Mathematical Derivation

### Step 1: Setup

Consider a fixed input point $x_0$. We want to analyze the expected squared error:

$$\mathbb{E}[(Y - \hat{f}(x_0))^2]$$

where:
- $Y = f(x_0) + \epsilon$ (true target with noise)
- $\hat{f}(x_0)$ is our prediction
- Expectation is over noise $\epsilon$ and training sets $\mathcal{D}$

### Step 2: Expand the Error

$$\mathbb{E}[(Y - \hat{f}(x_0))^2] = \mathbb{E}[(f(x_0) + \epsilon - \hat{f}(x_0))^2]$$

### Step 3: Add and Subtract the Expected Prediction

Add and subtract $\mathbb{E}[\hat{f}(x_0)]$:

$$\mathbb{E}[(f(x_0) + \epsilon - \hat{f}(x_0))^2] = \mathbb{E}[(f(x_0) + \epsilon - \mathbb{E}[\hat{f}(x_0)] + \mathbb{E}[\hat{f}(x_0)] - \hat{f}(x_0))^2]$$

### Step 4: Rearrange Terms

$$= \mathbb{E}[((f(x_0) - \mathbb{E}[\hat{f}(x_0)]) + \epsilon + (\mathbb{E}[\hat{f}(x_0)] - \hat{f}(x_0)))^2]$$

### Step 5: Expand the Square

Let:
- $A = f(x_0) - \mathbb{E}[\hat{f}(x_0)]$ (deterministic)
- $B = \epsilon$ (random, mean 0)
- $C = \mathbb{E}[\hat{f}(x_0)] - \hat{f}(x_0)$ (random, mean 0)

$$\mathbb{E}[(A + B + C)^2] = \mathbb{E}[A^2 + B^2 + C^2 + 2AB + 2AC + 2BC]$$

### Step 6: Apply Expectation

Since $\mathbb{E}[B] = \mathbb{E}[C] = 0$ and $A$ is deterministic:

$$= A^2 + \mathbb{E}[B^2] + \mathbb{E}[C^2] + 2A\mathbb{E}[B] + 2A\mathbb{E}[C] + 2\mathbb{E}[BC]$$

$$= A^2 + \mathbb{E}[B^2] + \mathbb{E}[C^2] + 0 + 0 + 2\mathbb{E}[BC]$$

### Step 7: Independence Assumption

Assuming $\epsilon$ (noise) is independent of the training process:

$$\mathbb{E}[BC] = \mathbb{E}[\epsilon(\mathbb{E}[\hat{f}(x_0)] - \hat{f}(x_0))] = 0$$

### Step 8: Final Result

$$\mathbb{E}[(Y - \hat{f}(x_0))^2] = (f(x_0) - \mathbb{E}[\hat{f}(x_0)])^2 + \mathbb{E}[\epsilon^2] + \mathbb{E}[(\mathbb{E}[\hat{f}(x_0)] - \hat{f}(x_0))^2]$$

$$= \text{Bias}^2[\hat{f}(x_0)] + \sigma^2 + \text{Var}[\hat{f}(x_0)]$$

### Mathematical Verification

Each component:

1. **Bias²**: 
   $$\text{Bias}^2[\hat{f}(x_0)] = (f(x_0) - \mathbb{E}[\hat{f}(x_0)])^2$$

2. **Variance**: 
   $$\text{Var}[\hat{f}(x_0)] = \mathbb{E}[(\hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)])^2]$$

3. **Irreducible Error**: 
   $$\sigma^2 = \mathbb{E}[\epsilon^2] = \text{Var}[\epsilon]$$

## Bias-Variance Decomposition for Classification

For classification problems, the decomposition is more complex due to the discrete nature of the output.

### 0-1 Loss Decomposition

For binary classification with 0-1 loss, the decomposition becomes:

$$\mathbb{E}[L] = \text{Noise} + \text{Bias} + \text{Variance}$$

where:

$$\text{Bias} = (P^*(x) - \mathbb{E}[P_{\mathcal{D}}(x)])^2$$

$$\text{Variance} = \mathbb{E}[P_{\mathcal{D}}(x)] - (\mathbb{E}[P_{\mathcal{D}}(x)])^2$$

$$\text{Noise} = P^*(x)(1 - P^*(x))$$

Here:
- $P^*(x)$ is the true conditional probability $P(Y=1|X=x)$
- $P_{\mathcal{D}}(x)$ is the estimated probability from dataset $\mathcal{D}$

### Squared Loss for Classification

When using squared loss for classification (treating it as regression):

$$\mathbb{E}[(Y - \hat{P}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Bayes Error}$$

This follows the same decomposition as regression.

## Sources of Bias and Variance

### Sources of High Bias

1. **Model Undercomplexity**
   - Linear models for nonlinear relationships
   - Low-capacity models (shallow trees, small neural networks)

2. **Strong Assumptions**
   - Assuming linearity when relationship is nonlinear
   - Feature independence assumptions when features are correlated

3. **Mathematical Form**
   For a model with parameters $\theta$:
   $$\text{Bias} = f(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x)]$$

### Sources of High Variance

1. **Model Overcomplexity**
   - High-degree polynomials
   - Deep decision trees
   - Large neural networks with insufficient data

2. **Small Training Sets**
   - Insufficient data to constrain the model
   - Random sampling variations have large impact

3. **Mathematical Form**
   $$\text{Variance} = \mathbb{E}_{\mathcal{D}}[(\hat{f}_{\mathcal{D}}(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x)])^2]$$

### Irreducible Error

$$\sigma^2 = \text{Var}[\epsilon]$$

Sources:
- Measurement noise
- Unobserved variables
- Inherent randomness in the process

## Model Complexity and the Tradeoff

### The Fundamental Tradeoff

As model complexity increases:

$$\text{Bias} \downarrow \quad \text{Variance} \uparrow$$

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$$

### Mathematical Analysis of Polynomial Models

For polynomial regression of degree $d$:

$$\hat{f}(x) = \sum_{i=0}^{d} \hat{\beta}_i x^i$$

**Bias Analysis:**
- Low degree $d$: High bias (cannot capture true complexity)
- High degree $d$: Low bias (can approximate any smooth function)

**Variance Analysis:**
- Low degree $d$: Low variance (few parameters to estimate)
- High degree $d$: High variance (many parameters, sensitive to training data)

### Optimal Complexity

The optimal model complexity $d^*$ minimizes:

$$\text{Risk}(d) = \text{Bias}^2(d) + \text{Variance}(d) + \sigma^2$$

Solved by:
$$\frac{d}{dd}\text{Risk}(d) = 0$$

This typically results in the characteristic U-shaped curve for test error.

## Practical Implications

### Model Selection Guidelines

1. **High Bias Indicators**
   - Low training accuracy
   - Low validation accuracy
   - Training and validation errors are similar

2. **High Variance Indicators**
   - High training accuracy
   - Low validation accuracy
   - Large gap between training and validation errors

### Bias-Variance Tradeoff Strategies

#### Reducing Bias
1. **Increase Model Complexity**
   - Add polynomial features
   - Use deeper decision trees
   - Increase neural network capacity

2. **Add More Features**
   - Feature engineering
   - Polynomial interactions
   - Domain-specific features

#### Reducing Variance
1. **Regularization**
   - L1/L2 regularization: $\lambda \sum |\theta_i|$ or $\lambda \sum \theta_i^2$
   - Early stopping
   - Dropout in neural networks

2. **Ensemble Methods**
   - Bagging: $\hat{f}_{\text{bag}}(x) = \frac{1}{B}\sum_{b=1}^{B} \hat{f}_b(x)$
   - Reduces variance by factor of $B$ for independent models

3. **Increase Training Data**
   - More data generally reduces variance
   - Asymptotically: $\text{Variance} \propto \frac{1}{n}$

## Examples and Applications

### Example 1: Polynomial Regression

Consider fitting polynomials of degree $d$ to data generated from:
$$y = \sin(2\pi x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.1^2)$$

**Degree 1 (Linear)**:
- High bias: Cannot capture sinusoidal relationship
- Low variance: Only 2 parameters to estimate
- $\text{Bias}^2 \approx 0.5, \text{Variance} \approx 0.01$

**Degree 10**:
- Low bias: Can approximate sine function well
- High variance: 11 parameters, sensitive to noise
- $\text{Bias}^2 \approx 0.01, \text{Variance} \approx 0.3$

**Degree 3-4 (Optimal)**:
- Moderate bias and variance
- $\text{Bias}^2 \approx 0.05, \text{Variance} \approx 0.05$

### Example 2: k-Nearest Neighbors

For k-NN with $k$ neighbors:

**Small k (k=1)**:
- Low bias: Can fit any pattern locally
- High variance: Very sensitive to individual points
- $\text{Bias} \to 0$ as $n \to \infty$
- $\text{Variance} = O(1/k)$

**Large k (k=n)**:
- High bias: Essentially constant prediction
- Low variance: Stable across datasets
- Becomes global average

**Optimal k**: 
$$k^* = O(n^{4/(4+d)})$$
where $d$ is the intrinsic dimension.

### Example 3: Decision Trees

**Shallow Trees (max_depth = 2)**:
- High bias: Cannot capture complex patterns
- Low variance: Simple structure, stable
- Underfitting behavior

**Deep Trees (max_depth = None)**:
- Low bias: Can memorize training data
- High variance: Different trees for different samples
- Overfitting behavior

## Mathematical Properties

### Decomposition Properties

1. **Additivity**: The three components sum exactly
   $$\mathbb{E}[\text{MSE}] = \text{Bias}^2 + \text{Variance} + \sigma^2$$

2. **Non-negativity**: Each component is non-negative
   $$\text{Bias}^2 \geq 0, \quad \text{Variance} \geq 0, \quad \sigma^2 \geq 0$$

3. **Independence**: Often (but not always) there's a tradeoff
   $$\frac{\partial \text{Bias}^2}{\partial \text{complexity}} \cdot \frac{\partial \text{Variance}}{\partial \text{complexity}} < 0$$

### Asymptotic Behavior

As sample size $n \to \infty$:

1. **Variance**: Generally decreases as $O(1/n)$
2. **Bias**: Remains constant (depends only on model class)
3. **Optimal complexity**: Often increases with $n$

### Bounds and Inequalities

**Jensen's Inequality Application**:
$$\text{Bias}^2 = (\mathbb{E}[\hat{f}] - f)^2 \leq \mathbb{E}[(\hat{f} - f)^2]$$

**Decomposition Bound**:
$$\text{MSE} \geq \sigma^2$$

The irreducible error provides a lower bound on achievable performance.

## Practical Estimation

### Empirical Bias-Variance Estimation

Given multiple datasets $\{\mathcal{D}_1, \mathcal{D}_2, \ldots, \mathcal{D}_B\}$:

1. **Train models**: $\hat{f}_1, \hat{f}_2, \ldots, \hat{f}_B$

2. **Estimate bias**:
   $$\widehat{\text{Bias}}^2 = \left(\frac{1}{B}\sum_{b=1}^{B} \hat{f}_b(x) - f(x)\right)^2$$

3. **Estimate variance**:
   $$\widehat{\text{Variance}} = \frac{1}{B}\sum_{b=1}^{B} \left(\hat{f}_b(x) - \frac{1}{B}\sum_{b'=1}^{B} \hat{f}_{b'}(x)\right)^2$$

### Bootstrap Estimation

When true function is unknown, use bootstrap:

1. Generate bootstrap samples from original dataset
2. Train models on each bootstrap sample
3. Estimate variance directly
4. Estimate bias using validation set

## References

### Foundational Papers
1. Geman, S., Bienenstock, E., & Doursat, R. (1992). "Neural networks and the bias/variance dilemma"
2. Breiman, L. (1996). "Bias, variance, and arcing classifiers"
3. Domingos, P. (2000). "A unified bias-variance decomposition for zero-one and squared loss"

### Books
1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"
2. Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"
3. Murphy, K. P. (2012). "Machine Learning: A Probabilistic Perspective"

### Mathematical Foundations
1. The derivation assumes independence between noise and model predictions
2. Extensions exist for correlated errors and non-i.i.d. data
3. Information-theoretic perspectives provide additional insights

---

This mathematical framework provides the theoretical foundation for understanding why certain models work better in different scenarios and guides practical decisions in model selection and hyperparameter tuning. 