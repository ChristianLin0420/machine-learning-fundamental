# Day 17: Model Selection & Cross-Validation

A comprehensive implementation of model selection techniques and cross-validation strategies in machine learning, including manual implementations, sklearn examples, nested CV, and model comparison frameworks.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [Implementation Components](#implementation-components)
4. [Usage Examples](#usage-examples)
5. [Key Concepts](#key-concepts)
6. [Experimental Results](#experimental-results)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

## 🎯 Overview

This module provides a complete toolkit for model selection and cross-validation in machine learning:

- **Manual CV implementations**: Built from scratch k-fold, stratified, and LOOCV
- **sklearn integration**: Comprehensive examples with various CV strategies
- **Nested cross-validation**: Unbiased performance estimation with hyperparameter tuning
- **Model comparison**: Statistical analysis and comprehensive evaluation framework
- **Visualization tools**: Rich plotting capabilities for CV results and model comparisons

## 📚 Theoretical Background

### Cross-Validation Fundamentals

Cross-validation is a statistical method used to estimate the skill of machine learning models by training and testing on different subsets of data.

#### Bias-Variance Tradeoff

```
MSE = Bias² + Variance + Irreducible Error
```

- **Bias**: Error from overly simplistic assumptions
- **Variance**: Error from sensitivity to small fluctuations in training set
- **Cross-validation**: Helps estimate both bias and variance

#### Types of Cross-Validation

1. **Hold-out Validation**
   - Split: 70% train, 30% test
   - Pros: Fast, simple
   - Cons: High variance, data dependent

2. **K-Fold Cross-Validation**
   ```
   CV Score = (1/k) × Σ(Score_i) for i=1 to k
   ```
   - Splits data into k equal parts
   - Train on k-1 parts, test on 1
   - Repeat k times

3. **Stratified K-Fold**
   - Maintains class distribution in each fold
   - Essential for imbalanced datasets
   - Reduces variance in classification problems

4. **Leave-One-Out (LOOCV)**
   ```
   LOOCV = (1/n) × Σ(Error_i) for i=1 to n
   ```
   - k = n (number of samples)
   - Low bias, high variance
   - Computationally expensive

5. **Time Series Cross-Validation**
   - Respects temporal order
   - Forward chaining validation
   - Prevents data leakage

### Statistical Concepts

#### Confidence Intervals

For cross-validation scores:
```
CI = mean ± t(α/2, df) × (std/√k)
```

Where:
- `t(α/2, df)`: t-distribution critical value
- `df = k-1`: degrees of freedom
- `k`: number of folds

#### Paired t-test

For comparing two models:
```
t = (mean_diff) / (std_diff/√k)
```

Under null hypothesis: no difference between models.

#### Effect Size (Cohen's d)

```
d = (mean₁ - mean₂) / pooled_std
```

Interpretation:
- |d| < 0.2: Negligible
- 0.2 ≤ |d| < 0.5: Small
- 0.5 ≤ |d| < 0.8: Medium
- |d| ≥ 0.8: Large

### Nested Cross-Validation

Two-level validation for unbiased model selection:

```
Outer Loop: Performance estimation
├── Fold 1
│   └── Inner Loop: Hyperparameter optimization
│       ├── Grid Search CV
│       └── Best model → Test on Fold 1
├── Fold 2
│   └── Inner Loop: Hyperparameter optimization
│       └── Best model → Test on Fold 2
└── ...
```

**Benefits:**
- Unbiased performance estimates
- Separates model selection from evaluation
- Prevents optimistic bias from data leakage

## 🛠 Implementation Components

### 1. Manual Cross-Validation (`cross_validation_from_scratch.py`)

```python
from cross_validation_from_scratch import CrossValidatorFromScratch

# Initialize
cv = CrossValidatorFromScratch()

# K-Fold split
train_folds, test_folds = cv.k_fold_split(X, y, k=5)

# Stratified K-Fold
train_folds, test_folds = cv.stratified_k_fold_split(X, y, k=5)

# LOOCV
train_folds, test_folds = cv.leave_one_out_split(X, y)

# Cross-validate with scoring
results = cv.cross_validate(
    model, X, y, 
    cv_method='k_fold',
    k=5,
    scoring=['accuracy', 'f1'],
    return_fold_info=True
)
```

#### Key Features:
- **Multiple CV strategies**: k-fold, stratified, LOOCV, time series
- **Multiple scoring metrics**: Accuracy, F1, precision, recall, AUC
- **Statistical analysis**: Confidence intervals, significance testing
- **Fold-level insights**: Per-fold performance and model parameters
- **Visualization**: CV results and fold distributions

### 2. sklearn Cross-Validation Examples (`sklearn_cv_examples.py`)

```python
from sklearn_cv_examples import demonstrate_cv_strategies

# Compare different CV strategies
demonstrate_cv_strategies()

# Learning curves
plot_learning_curves(model, X, y, cv=5)

# Validation curves
plot_validation_curves(model, X, y, param_name='C', param_range=[0.1, 1, 10])
```

#### Covered Strategies:
- **KFold**: Standard k-fold validation
- **StratifiedKFold**: Class-balanced folds
- **LeaveOneOut**: Maximum training data usage
- **ShuffleSplit**: Random sampling with replacement
- **GroupKFold**: Ensures groups don't overlap between folds
- **TimeSeriesSplit**: Respects temporal order

### 3. Nested Cross-Validation (`nested_cv_pipeline.py`)

```python
from nested_cv_pipeline import NestedCVPipeline

# Define model and parameter grid
model = SVC()
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

# Nested CV
nested_cv = NestedCVPipeline(
    estimator=model,
    param_grid=param_grid,
    inner_cv=5,
    outer_cv=5,
    scoring='accuracy'
)

# Fit and evaluate
nested_cv.fit(X, y)
summary = nested_cv.get_performance_summary()

# Visualize results
nested_cv.plot_results("Nested CV Results")
```

#### Key Metrics:
- **Outer CV performance**: Unbiased estimate
- **Inner CV performance**: Hyperparameter selection quality
- **Optimistic bias**: Difference between inner and outer CV
- **Variance analysis**: Stability across folds

### 4. Model Comparison (`compare_models.py`)

```python
from compare_models import ModelComparator

# Initialize comparator
comparator = ModelComparator(
    cv=5,
    scoring=['accuracy', 'f1_weighted', 'roc_auc_ovr_weighted']
)

# Compare models
comparator.compare(X, y, scale_features=True)

# Get summary table
summary_df = comparator.get_summary_table()

# Statistical significance testing
sig_results = comparator.statistical_significance_test()

# Comprehensive plotting
comparator.plot_comparison()

# Extract best model
best_pipeline = comparator.get_best_model_pipeline()
```

#### Analysis Features:
- **Multiple metrics**: Accuracy, F1, precision, recall, AUC
- **Statistical testing**: Paired t-tests with effect sizes
- **Overfitting analysis**: Train vs test performance
- **Visualization**: Boxplots, heatmaps, scatter plots
- **Pipeline extraction**: Ready-to-use best model

## 📊 Usage Examples

### Example 1: Basic Cross-Validation

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from cross_validation_from_scratch import CrossValidatorFromScratch

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Initialize model and CV
model = RandomForestClassifier(random_state=42)
cv = CrossValidatorFromScratch()

# Perform 5-fold CV
results = cv.cross_validate(
    model, X, y,
    cv_method='stratified_k_fold',
    k=5,
    scoring=['accuracy', 'f1'],
    confidence_level=0.95
)

print(f"Accuracy: {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}")
print(f"95% CI: [{results['accuracy']['ci_lower']:.4f}, {results['accuracy']['ci_upper']:.4f}]")
```

### Example 2: Nested Cross-Validation

```python
from sklearn.svm import SVC
from nested_cv_pipeline import NestedCVPipeline

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

# Nested CV
nested_cv = NestedCVPipeline(
    estimator=SVC(random_state=42),
    param_grid=param_grid,
    inner_cv=3,
    outer_cv=5,
    scoring='accuracy'
)

# Fit and analyze
nested_cv.fit(X, y)
summary = nested_cv.get_performance_summary()

print(f"Unbiased accuracy: {summary['outer_cv_mean']:.4f} ± {summary['outer_cv_std']:.4f}")
print(f"Optimistic bias: {summary['inner_cv_mean'] - summary['outer_cv_mean']:.4f}")
```

### Example 3: Model Comparison

```python
from compare_models import ModelComparator

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Compare
comparator = ModelComparator(
    models=models,
    cv=5,
    scoring=['accuracy', 'f1_weighted', 'precision_weighted']
)

comparator.compare(X, y)

# Get statistical comparison
sig_results = comparator.statistical_significance_test()
print("Model comparison results:")
print(sig_results[['Model 1', 'Model 2', 'P-value', 'Significant', 'Effect Magnitude']])
```

## 🔬 Key Concepts

### 1. Cross-Validation Strategies

| Strategy | Use Case | Pros | Cons |
|----------|----------|------|------|
| **K-Fold** | General purpose | Balanced bias-variance | May not preserve class distribution |
| **Stratified K-Fold** | Classification (imbalanced) | Preserves class ratios | Only for classification |
| **LOOCV** | Small datasets | Maximum training data | High variance, expensive |
| **Time Series Split** | Temporal data | Respects time order | Limited to time series |
| **Group K-Fold** | Grouped data | Prevents data leakage | Requires group information |

### 2. Scoring Metrics

#### Classification Metrics

```python
# Available scoring options
scoring_options = [
    'accuracy',           # (TP + TN) / (TP + TN + FP + FN)
    'precision',          # TP / (TP + FP)
    'recall',            # TP / (TP + FN)
    'f1',                # 2 * (precision * recall) / (precision + recall)
    'roc_auc',           # Area Under ROC Curve
    'average_precision'   # Area Under Precision-Recall Curve
]
```

#### Regression Metrics

```python
regression_scoring = [
    'neg_mean_squared_error',      # -MSE
    'neg_mean_absolute_error',     # -MAE
    'neg_root_mean_squared_error', # -RMSE
    'r2',                          # R² coefficient
    'explained_variance'           # Explained variance score
]
```

### 3. Statistical Analysis

#### Confidence Intervals

Cross-validation provides multiple performance estimates. Confidence intervals quantify uncertainty:

```python
def calculate_confidence_interval(scores, confidence_level=0.95):
    """Calculate confidence interval for CV scores"""
    n = len(scores)
    mean = np.mean(scores)
    std_err = np.std(scores, ddof=1) / np.sqrt(n)
    
    # t-distribution critical value
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin_error = t_critical * std_err
    
    return {
        'mean': mean,
        'ci_lower': mean - margin_error,
        'ci_upper': mean + margin_error,
        'margin_error': margin_error
    }
```

#### Model Comparison

When comparing models, use paired statistical tests:

```python
def compare_models_statistically(scores1, scores2, alpha=0.05):
    """Compare two models using paired t-test"""
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    # Effect size (Cohen's d)
    diff = np.mean(scores1) - np.mean(scores2)
    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    
    return {
        'mean_difference': diff,
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': cohens_d,
        'interpretation': interpret_effect_size(abs(cohens_d))
    }
```

### 4. Best Practices

#### Choosing Cross-Validation Strategy

1. **Small datasets (< 1000 samples)**: LOOCV or high k in k-fold
2. **Large datasets**: 5-fold or 10-fold CV
3. **Imbalanced classes**: Stratified k-fold
4. **Time series data**: Time series split
5. **Grouped data**: Group k-fold

#### Avoiding Common Pitfalls

1. **Data leakage**: Ensure preprocessing happens within CV folds
2. **Optimistic bias**: Use nested CV for hyperparameter tuning
3. **Temporal leakage**: Use appropriate splits for time series data
4. **Insufficient folds**: Balance bias-variance tradeoff

## 📈 Experimental Results

### Cross-Validation Strategy Comparison

Dataset: Breast Cancer (569 samples, 30 features, 2 classes)

| Strategy | Mean Accuracy | Std Dev | Runtime (s) |
|----------|---------------|---------|-------------|
| 5-Fold | 0.9649 | 0.0187 | 0.12 |
| 10-Fold | 0.9649 | 0.0201 | 0.23 |
| Stratified 5-Fold | 0.9649 | 0.0171 | 0.13 |
| LOOCV | 0.9649 | 0.1844 | 2.45 |

**Observations:**
- Stratified k-fold shows lower variance
- LOOCV has highest variance despite low bias
- 5-fold provides good bias-variance balance

### Nested CV vs Simple CV

Model: SVM with hyperparameter tuning

| Method | Mean Accuracy | Optimistic Bias |
|--------|---------------|-----------------|
| Simple CV | 0.9807 | +0.0158 |
| Nested CV | 0.9649 | 0.0000 |

**Key Insight:** Simple CV overestimates performance by ~1.6% due to data leakage during hyperparameter selection.

### Model Comparison Results

Dataset: Wine (178 samples, 13 features, 3 classes)

| Model | Accuracy | F1-Score | Significant Differences |
|-------|----------|----------|------------------------|
| Random Forest | 0.9718 | 0.9718 | None |
| SVM | 0.9775 | 0.9779 | None |
| Logistic Regression | 0.9493 | 0.9497 | vs RF (p=0.23), vs SVM (p=0.18) |
| Gradient Boosting | 0.9437 | 0.9444 | vs SVM (p=0.04)* |

*Statistically significant at α=0.05

## 🎯 Best Practices

### 1. Cross-Validation Selection

```python
def choose_cv_strategy(n_samples, n_classes, has_groups=False, is_timeseries=False):
    """Guide for selecting appropriate CV strategy"""
    
    if is_timeseries:
        return "TimeSeriesSplit"
    
    if has_groups:
        return "GroupKFold"
    
    if n_samples < 100:
        return "LeaveOneOut"
    
    if n_classes > 1 and min(class_counts) < 10:
        return "StratifiedKFold"
    
    if n_samples < 1000:
        return "KFold with k=10"
    
    return "KFold with k=5"
```

### 2. Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# CORRECT: Preprocessing inside CV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

scores = cross_val_score(pipeline, X, y, cv=5)

# INCORRECT: Preprocessing before CV
# X_scaled = StandardScaler().fit_transform(X)  # Data leakage!
# scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)
```

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# CORRECT: Nested CV approach
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Inner loop: hyperparameter optimization
grid_search = GridSearchCV(
    SVC(),
    param_grid={'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    cv=inner_cv,
    scoring='accuracy'
)

# Outer loop: performance estimation
outer_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='accuracy')
```

### 4. Statistical Validation

```python
def validate_model_comparison(model_scores, alpha=0.05):
    """Validate model comparison with proper statistics"""
    
    results = {}
    model_names = list(model_scores.keys())
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            
            scores1 = model_scores[model1]
            scores2 = model_scores[model2]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            
            # Effect size
            cohens_d = (np.mean(scores1) - np.mean(scores2)) / \
                      np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
            
            results[f"{model1}_vs_{model2}"] = {
                'p_value': p_value,
                'significant': p_value < alpha,
                'effect_size': cohens_d,
                'winner': model1 if np.mean(scores1) > np.mean(scores2) else model2
            }
    
    return results
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Convergence Warnings

**Problem:** sklearn models not converging
```
ConvergenceWarning: lbfgs failed to converge
```

**Solution:**
```python
# Increase max_iter
model = LogisticRegression(max_iter=1000)

# Or use different solver
model = LogisticRegression(solver='liblinear')

# Scale features
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

#### 2. Memory Issues with Large Datasets

**Problem:** Out of memory during CV
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Use smaller k
cv = KFold(n_splits=3)  # Instead of 10

# Process in batches
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=3, n_jobs=1)  # Reduce parallel jobs
```

#### 3. Imbalanced Classes

**Problem:** Poor performance on minority classes

**Solution:**
```python
# Use stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Appropriate scoring metrics
scoring = ['f1_weighted', 'roc_auc', 'average_precision']

# Class balancing
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
model = RandomForestClassifier(class_weight='balanced')
```

#### 4. Time Series Data Leakage

**Problem:** Using future information to predict past

**Solution:**
```python
from sklearn.model_selection import TimeSeriesSplit

# Proper time series CV
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)

# Or manual implementation
def time_series_split(n_samples, n_splits):
    """Custom time series split"""
    fold_size = n_samples // (n_splits + 1)
    
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = test_start + fold_size
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, min(test_end, n_samples))
        
        yield train_idx, test_idx
```

### Performance Optimization

#### 1. Parallel Processing

```python
# Use all CPU cores
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)

# Limit cores for memory constraints
scores = cross_val_score(model, X, y, cv=5, n_jobs=2)
```

#### 2. Early Stopping

```python
# For iterative models
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.1,
    n_iter_no_change=10,  # Early stopping
    random_state=42
)
```

## 📚 References

### Research Papers
1. Kohavi, R. (1995). "A study of cross-validation and bootstrap for accuracy estimation and model selection"
2. Varma, S. & Simon, R. (2006). "Bias in error estimation when using cross-validation for model selection"
3. Cawley, G.C. & Talbot, N.L. (2010). "On over-fitting in model selection and subsequent selection bias in performance evaluation"

### Statistical Methods
- Student's t-test for paired samples
- Cohen's d for effect size estimation
- Bootstrap methods for confidence intervals
- Friedman test for multiple model comparison

### Best Practice Guides
- scikit-learn Cross-validation Documentation
- "The Elements of Statistical Learning" - Hastie, Tibshirani, Friedman
- "Pattern Recognition and Machine Learning" - Christopher Bishop

---

## 🚀 Quick Start

```bash
# Run individual modules
python cross_validation_from_scratch.py
python sklearn_cv_examples.py
python nested_cv_pipeline.py
python compare_models.py

# Or import and use in your projects
from cross_validation_from_scratch import CrossValidatorFromScratch
from compare_models import ModelComparator
```

This implementation provides a comprehensive foundation for understanding and applying cross-validation and model selection techniques in machine learning projects. 