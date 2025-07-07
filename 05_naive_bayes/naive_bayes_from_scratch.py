"""
Naive Bayes Implementation from Scratch

This module implements three variants of Naive Bayes classifiers:
- Gaussian Naive Bayes: For continuous features with normal distribution
- Multinomial Naive Bayes: For discrete features like text/word counts
- Bernoulli Naive Bayes: For binary features

Key Features:
- Maximum Likelihood Estimation (MLE) for parameters
- Log-space computations to avoid numerical underflow
- Laplace (additive) smoothing for zero-count handling
- Vectorized operations for efficiency
- Comprehensive evaluation metrics
- Decision boundary visualization
- Feature importance analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_20newsgroups, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
import re
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes implementation for continuous features.
    
    Assumes features follow normal distribution within each class.
    Uses Maximum Likelihood Estimation for mean and variance.
    """
    
    def __init__(self, var_smoothing=1e-9):
        """
        Initialize Gaussian Naive Bayes.
        
        Args:
            var_smoothing (float): Portion of the largest variance added to variances
                                 for calculation stability
        """
        self.var_smoothing = var_smoothing
        
        # Model parameters
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None  # Mean of each feature per class
        self.sigma_ = None  # Variance of each feature per class
        self.n_features_ = None
    
    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes classifier.
        
        Args:
            X (np.array): Training features (n_samples, n_features)
            y (np.array): Training targets (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, self.n_features_ = X.shape
        
        # Initialize parameters
        self.theta_ = np.zeros((n_classes, self.n_features_))
        self.sigma_ = np.zeros((n_classes, self.n_features_))
        self.class_prior_ = np.zeros(n_classes)
        
        # Calculate parameters for each class
        for i, class_label in enumerate(self.classes_):
            # Get samples for this class
            X_class = X[y == class_label]
            
            # Class prior: P(y)
            self.class_prior_[i] = len(X_class) / n_samples
            
            # Mean: μ_y,i = E[X_i | y]
            self.theta_[i] = np.mean(X_class, axis=0)
            
            # Variance: σ²_y,i = Var[X_i | y]
            self.sigma_[i] = np.var(X_class, axis=0)
        
        # Add smoothing to avoid zero variances
        self.sigma_ += self.var_smoothing
        
        return self
    
    def _calculate_log_likelihood(self, X):
        """
        Calculate log-likelihood for each class.
        
        Args:
            X (np.array): Test features
        
        Returns:
            np.array: Log-likelihood matrix (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Initialize log-likelihood matrix
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for i in range(n_classes):
            # Calculate log P(x_i | y) for each feature
            # log P(x_i | y) = -0.5 * log(2π * σ²) - (x_i - μ)² / (2σ²)
            
            diff = X - self.theta_[i]
            log_prob_features = (
                -0.5 * np.log(2 * np.pi * self.sigma_[i]) -
                (diff ** 2) / (2 * self.sigma_[i])
            )
            
            # Sum log probabilities across features (naive assumption)
            log_likelihood[:, i] = np.sum(log_prob_features, axis=1)
        
        return log_likelihood
    
    def predict_log_proba(self, X):
        """
        Calculate log posterior probabilities.
        
        Args:
            X (np.array): Test features
        
        Returns:
            np.array: Log posterior probabilities
        """
        X = np.array(X)
        
        # Calculate log-likelihood
        log_likelihood = self._calculate_log_likelihood(X)
        
        # Add log prior: log P(y | x) ∝ log P(y) + log P(x | y)
        log_posterior = log_likelihood + np.log(self.class_prior_)
        
        # Normalize to get proper log probabilities
        log_prob_norm = np.log(np.sum(np.exp(log_posterior), axis=1, keepdims=True))
        log_proba = log_posterior - log_prob_norm
        
        return log_proba
    
    def predict_proba(self, X):
        """
        Calculate posterior probabilities.
        
        Args:
            X (np.array): Test features
        
        Returns:
            np.array: Posterior probabilities
        """
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (np.array): Test features
        
        Returns:
            np.array: Predicted class labels
        """
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
    
    def get_params(self):
        """Get model parameters."""
        return {
            'classes': self.classes_,
            'class_prior': self.class_prior_,
            'means': self.theta_,
            'variances': self.sigma_
        }


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes implementation for discrete features.
    
    Commonly used for text classification with word count features.
    Includes Laplace smoothing for handling zero counts.
    """
    
    def __init__(self, alpha=1.0):
        """
        Initialize Multinomial Naive Bayes.
        
        Args:
            alpha (float): Laplace smoothing parameter
        """
        self.alpha = alpha
        
        # Model parameters
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None
        self.n_features_ = None
    
    def fit(self, X, y):
        """
        Fit Multinomial Naive Bayes classifier.
        
        Args:
            X (np.array): Training features (n_samples, n_features)
            y (np.array): Training targets (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, self.n_features_ = X.shape
        
        # Initialize parameters
        self.class_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, self.n_features_))
        
        # Calculate parameters for each class
        for i, class_label in enumerate(self.classes_):
            # Get samples for this class
            X_class = X[y == class_label]
            
            # Class prior: P(y)
            self.class_prior_[i] = len(X_class) / n_samples
            
            # Feature probabilities with Laplace smoothing
            # P(x_i | y) = (count(x_i, y) + α) / (sum(count(all features, y)) + α * n_features)
            
            # Sum of feature counts for this class
            feature_counts = np.sum(X_class, axis=0)
            
            # Total count for this class (sum across all features)
            total_count = np.sum(feature_counts)
            
            # Apply Laplace smoothing
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = total_count + self.alpha * self.n_features_
            
            # Calculate log probabilities
            self.feature_log_prob_[i] = np.log(smoothed_counts / smoothed_total)
        
        return self
    
    def predict_log_proba(self, X):
        """
        Calculate log posterior probabilities.
        
        Args:
            X (np.array): Test features
        
        Returns:
            np.array: Log posterior probabilities
        """
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Initialize log posterior matrix
        log_posterior = np.zeros((n_samples, n_classes))
        
        for i in range(n_classes):
            # Calculate log P(x | y) = Σ x_j * log P(x_j | y)
            log_likelihood = np.sum(X * self.feature_log_prob_[i], axis=1)
            
            # Add log prior: log P(y | x) ∝ log P(y) + log P(x | y)
            log_posterior[:, i] = np.log(self.class_prior_[i]) + log_likelihood
        
        # Normalize to get proper log probabilities
        log_prob_norm = np.log(np.sum(np.exp(log_posterior), axis=1, keepdims=True))
        log_proba = log_posterior - log_prob_norm
        
        return log_proba
    
    def predict_proba(self, X):
        """Calculate posterior probabilities."""
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X):
        """Make predictions."""
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
    
    def get_feature_importance(self, feature_names=None, top_k=10):
        """
        Get most important features for each class.
        
        Args:
            feature_names (list): Names of features
            top_k (int): Number of top features to return
        
        Returns:
            dict: Feature importance for each class
        """
        importance = {}
        
        for i, class_label in enumerate(self.classes_):
            # Get feature log probabilities for this class
            log_probs = self.feature_log_prob_[i]
            
            # Get top k features
            top_indices = np.argsort(log_probs)[-top_k:][::-1]
            
            if feature_names is not None:
                top_features = [(feature_names[idx], log_probs[idx]) for idx in top_indices]
            else:
                top_features = [(f"feature_{idx}", log_probs[idx]) for idx in top_indices]
            
            importance[class_label] = top_features
        
        return importance


class BernoulliNaiveBayes:
    """
    Bernoulli Naive Bayes implementation for binary features.
    
    Each feature is treated as a binary variable (present/absent).
    Useful for text classification with binary word occurrence features.
    """
    
    def __init__(self, alpha=1.0, binarize=0.0):
        """
        Initialize Bernoulli Naive Bayes.
        
        Args:
            alpha (float): Laplace smoothing parameter
            binarize (float): Threshold for binarizing features
        """
        self.alpha = alpha
        self.binarize = binarize
        
        # Model parameters
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None
        self.n_features_ = None
    
    def _binarize_features(self, X):
        """Binarize features based on threshold."""
        return (X > self.binarize).astype(float)
    
    def fit(self, X, y):
        """
        Fit Bernoulli Naive Bayes classifier.
        
        Args:
            X (np.array): Training features (n_samples, n_features)
            y (np.array): Training targets (n_samples,)
        """
        X = np.array(X)
        y = np.array(y)
        
        # Binarize features
        X = self._binarize_features(X)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, self.n_features_ = X.shape
        
        # Initialize parameters
        self.class_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, self.n_features_))
        
        # Calculate parameters for each class
        for i, class_label in enumerate(self.classes_):
            # Get samples for this class
            X_class = X[y == class_label]
            n_class_samples = len(X_class)
            
            # Class prior: P(y)
            self.class_prior_[i] = n_class_samples / n_samples
            
            # Feature probabilities with Laplace smoothing
            # P(x_i = 1 | y) = (count(x_i = 1, y) + α) / (count(y) + 2α)
            
            # Count of feature = 1 for this class
            feature_counts = np.sum(X_class, axis=0)
            
            # Apply Laplace smoothing
            smoothed_counts = feature_counts + self.alpha
            smoothed_total = n_class_samples + 2 * self.alpha
            
            # Calculate log probabilities for feature = 1
            prob_feature_1 = smoothed_counts / smoothed_total
            
            # Store log probabilities: log P(x_i = 1 | y) and log P(x_i = 0 | y)
            self.feature_log_prob_[i] = np.log(prob_feature_1)
        
        return self
    
    def predict_log_proba(self, X):
        """Calculate log posterior probabilities."""
        X = np.array(X)
        X = self._binarize_features(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Initialize log posterior matrix
        log_posterior = np.zeros((n_samples, n_classes))
        
        for i in range(n_classes):
            # Calculate log P(x | y) for Bernoulli features
            # log P(x | y) = Σ [x_j * log P(x_j = 1 | y) + (1 - x_j) * log P(x_j = 0 | y)]
            
            log_prob_1 = self.feature_log_prob_[i]
            log_prob_0 = np.log(1 - np.exp(log_prob_1))
            
            # Calculate likelihood for each sample
            log_likelihood = np.sum(
                X * log_prob_1 + (1 - X) * log_prob_0, axis=1
            )
            
            # Add log prior
            log_posterior[:, i] = np.log(self.class_prior_[i]) + log_likelihood
        
        # Normalize
        log_prob_norm = np.log(np.sum(np.exp(log_posterior), axis=1, keepdims=True))
        log_proba = log_posterior - log_prob_norm
        
        return log_proba
    
    def predict_proba(self, X):
        """Calculate posterior probabilities."""
        return np.exp(self.predict_log_proba(X))
    
    def predict(self, X):
        """Make predictions."""
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]


def create_text_dataset():
    """
    Create a simple text classification dataset for demonstration.
    
    Returns:
        tuple: (texts, labels, vectorizer)
    """
    # Simple spam/ham classification dataset
    texts = [
        "Buy now! Limited time offer! Click here!",
        "Free money! You won a million dollars!",
        "Urgent! Your account will be closed!",
        "Make money fast! No experience needed!",
        "Congratulations! You have won a prize!",
        "Hi John, how are you doing today?",
        "Meeting scheduled for tomorrow at 2pm",
        "Please review the attached document",
        "Thanks for your help with the project",
        "Looking forward to our lunch meeting",
        "The weather is nice today",
        "Can you send me the report?",
        "Happy birthday! Hope you have a great day",
        "Reminder: dentist appointment tomorrow",
        "Great job on the presentation!"
    ]
    
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 1 = spam, 0 = ham
    
    # Vectorize text
    vectorizer = CountVectorizer(stop_words='english', lowercase=True)
    X = vectorizer.fit_transform(texts).toarray()
    
    return X, np.array(labels), vectorizer, texts


def plot_gaussian_distributions(model, X, y, feature_names=None):
    """
    Plot Gaussian distributions for each feature and class.
    
    Args:
        model: Fitted Gaussian Naive Bayes model
        X: Training features
        y: Training labels
        feature_names: Names of features
    """
    n_features = model.n_features_
    n_classes = len(model.classes_)
    
    # Determine subplot layout
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    for i in range(n_features):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Plot distributions for each class
        for j, class_label in enumerate(model.classes_):
            mean = model.theta_[j, i]
            std = np.sqrt(model.sigma_[j, i])
            
            # Generate x values for plotting
            x_min, x_max = X[:, i].min() - 2*std, X[:, i].max() + 2*std
            x = np.linspace(x_min, x_max, 100)
            
            # Calculate Gaussian PDF
            pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
            
            ax.plot(x, pdf, color=colors[j], label=f'Class {class_label}', linewidth=2)
            ax.axvline(mean, color=colors[j], linestyle='--', alpha=0.7)
        
        # Add data points
        for j, class_label in enumerate(model.classes_):
            class_data = X[y == class_label, i]
            ax.scatter(class_data, np.zeros_like(class_data) - 0.01, 
                      color=colors[j], alpha=0.6, s=20)
        
        title = feature_names[i] if feature_names else f'Feature {i}'
        ax.set_title(f'Distribution: {title}')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(n_features, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        if n_rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('plots/gaussian_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_decision_boundary_2d(X, y, model, title="Naive Bayes Decision Boundary", resolution=100):
    """
    Plot decision boundary for 2D data.
    
    Args:
        X: 2D feature matrix
        y: Target labels
        model: Fitted Naive Bayes model
        title: Plot title
        resolution: Grid resolution
    """
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Make predictions on the mesh grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, 
                         edgecolors='black', s=50)
    plt.colorbar(scatter)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/nb_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(importance_dict, top_k=10):
    """
    Plot feature importance for each class.
    
    Args:
        importance_dict: Dictionary with class-wise feature importance
        top_k: Number of top features to show
    """
    n_classes = len(importance_dict)
    
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 8))
    if n_classes == 1:
        axes = [axes]
    
    for i, (class_label, features) in enumerate(importance_dict.items()):
        # Extract feature names and scores
        feature_names = [item[0] for item in features[:top_k]]
        scores = [item[1] for item in features[:top_k]]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        axes[i].barh(y_pos, scores, align='center')
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(feature_names)
        axes[i].invert_yaxis()  # Top features at the top
        axes[i].set_xlabel('Log Probability')
        axes[i].set_title(f'Top Features - Class {class_label}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrices(models_results):
    """
    Plot confusion matrices for different models.
    
    Args:
        models_results: Dictionary with model results
    """
    n_models = len(models_results)
    
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for i, (model_name, results) in enumerate(models_results.items()):
        cm = confusion_matrix(results['y_true'], results['y_pred'])
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   ax=axes[i], cbar=True)
        axes[i].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.3f}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_laplace_smoothing(X_train, y_train, X_test, y_test, alpha_values):
    """
    Evaluate the impact of different Laplace smoothing values.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        alpha_values: List of alpha values to test
    
    Returns:
        dict: Results for different alpha values
    """
    results = {
        'alpha_values': alpha_values,
        'train_accuracy': [],
        'test_accuracy': []
    }
    
    print("Evaluating Laplace smoothing impact:")
    print("=" * 50)
    
    for alpha in alpha_values:
        # Train Multinomial Naive Bayes with current alpha
        model = MultinomialNaiveBayes(alpha=alpha)
        model.fit(X_train, y_train)
        
        # Calculate accuracies
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        results['train_accuracy'].append(train_acc)
        results['test_accuracy'].append(test_acc)
        
        print(f"α = {alpha:6.3f}: Train = {train_acc:.4f}, Test = {test_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(alpha_values, results['train_accuracy'], 'b-o', 
                 label='Training Accuracy', markersize=6)
    plt.semilogx(alpha_values, results['test_accuracy'], 'r-s', 
                 label='Test Accuracy', markersize=6)
    plt.xlabel('Laplace Smoothing Parameter (α)')
    plt.ylabel('Accuracy')
    plt.title('Impact of Laplace Smoothing on Multinomial Naive Bayes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/laplace_smoothing_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def compare_with_sklearn(X_train, y_train, X_test, y_test, model_type='gaussian'):
    """
    Compare custom implementation with scikit-learn.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_type: Type of model ('gaussian', 'multinomial', 'bernoulli')
    """
    print(f"\nComparison with sklearn - {model_type.capitalize()} Naive Bayes:")
    print("=" * 60)
    
    if model_type == 'gaussian':
        # Custom implementation
        custom_model = GaussianNaiveBayes()
        custom_model.fit(X_train, y_train)
        custom_pred = custom_model.predict(X_test)
        custom_proba = custom_model.predict_proba(X_test)
        
        # Sklearn implementation
        sklearn_model = GaussianNB()
        sklearn_model.fit(X_train, y_train)
        sklearn_pred = sklearn_model.predict(X_test)
        sklearn_proba = sklearn_model.predict_proba(X_test)
        
    elif model_type == 'multinomial':
        # Custom implementation
        custom_model = MultinomialNaiveBayes(alpha=1.0)
        custom_model.fit(X_train, y_train)
        custom_pred = custom_model.predict(X_test)
        custom_proba = custom_model.predict_proba(X_test)
        
        # Sklearn implementation
        sklearn_model = MultinomialNB(alpha=1.0)
        sklearn_model.fit(X_train, y_train)
        sklearn_pred = sklearn_model.predict(X_test)
        sklearn_proba = sklearn_model.predict_proba(X_test)
        
    elif model_type == 'bernoulli':
        # Custom implementation
        custom_model = BernoulliNaiveBayes(alpha=1.0)
        custom_model.fit(X_train, y_train)
        custom_pred = custom_model.predict(X_test)
        custom_proba = custom_model.predict_proba(X_test)
        
        # Sklearn implementation
        sklearn_model = BernoulliNB(alpha=1.0)
        sklearn_model.fit(X_train, y_train)
        sklearn_pred = sklearn_model.predict(X_test)
        sklearn_proba = sklearn_model.predict_proba(X_test)
    
    # Calculate accuracies
    custom_accuracy = accuracy_score(y_test, custom_pred)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    # Calculate probability differences
    prob_diff = np.mean(np.abs(custom_proba - sklearn_proba))
    
    print(f"Custom Implementation Accuracy:  {custom_accuracy:.6f}")
    print(f"Sklearn Implementation Accuracy: {sklearn_accuracy:.6f}")
    print(f"Accuracy Difference:             {abs(custom_accuracy - sklearn_accuracy):.8f}")
    print(f"Mean Probability Difference:     {prob_diff:.8f}")
    
    return {
        'custom_accuracy': custom_accuracy,
        'sklearn_accuracy': sklearn_accuracy,
        'custom_predictions': custom_pred,
        'sklearn_predictions': sklearn_pred
    }


def main():
    """
    Main function to run Naive Bayes experiments.
    """
    print("="*70)
    print("NAIVE BAYES IMPLEMENTATION FROM SCRATCH")
    print("="*70)
    
    # 1. Gaussian Naive Bayes with Iris Dataset
    print("\n1. GAUSSIAN NAIVE BAYES - IRIS DATASET")
    print("-" * 50)
    
    # Load Iris dataset
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Split data
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
    )
    
    # Train Gaussian Naive Bayes
    gaussian_nb = GaussianNaiveBayes()
    gaussian_nb.fit(X_train_iris, y_train_iris)
    
    # Make predictions
    y_pred_iris = gaussian_nb.predict(X_test_iris)
    y_proba_iris = gaussian_nb.predict_proba(X_test_iris)
    
    # Evaluate
    accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)
    print(f"Gaussian NB Accuracy: {accuracy_iris:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_iris, y_pred_iris, target_names=iris.target_names))
    
    # Plot Gaussian distributions
    plot_gaussian_distributions(gaussian_nb, X_train_iris, y_train_iris, iris.feature_names)
    
    # 2D Decision boundary (using PCA)
    print("\n2. GAUSSIAN NB DECISION BOUNDARY (PCA PROJECTION)")
    print("-" * 50)
    
    # Project to 2D using PCA
    pca = PCA(n_components=2)
    X_iris_2d = pca.fit_transform(X_iris)
    
    # Train Gaussian NB on 2D data
    gaussian_nb_2d = GaussianNaiveBayes()
    gaussian_nb_2d.fit(X_iris_2d, y_iris)
    
    # Plot decision boundary
    plot_decision_boundary_2d(X_iris_2d, y_iris, gaussian_nb_2d, 
                             "Gaussian Naive Bayes Decision Boundary (Iris PCA)")
    
    # 2. Multinomial Naive Bayes with Text Data
    print("\n3. MULTINOMIAL NAIVE BAYES - TEXT CLASSIFICATION")
    print("-" * 50)
    
    # Create text dataset
    X_text, y_text, vectorizer, texts = create_text_dataset()
    
    # Split data
    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
        X_text, y_text, test_size=0.3, random_state=42
    )
    
    # Train Multinomial Naive Bayes
    multinomial_nb = MultinomialNaiveBayes(alpha=1.0)
    multinomial_nb.fit(X_train_text, y_train_text)
    
    # Make predictions
    y_pred_text = multinomial_nb.predict(X_test_text)
    
    # Evaluate
    accuracy_text = accuracy_score(y_test_text, y_pred_text)
    print(f"Multinomial NB Accuracy: {accuracy_text:.4f}")
    
    # Feature importance analysis
    feature_names = vectorizer.get_feature_names_out()
    importance = multinomial_nb.get_feature_importance(feature_names, top_k=10)
    
    print("\nMost important words for each class:")
    for class_label, features in importance.items():
        class_name = "Spam" if class_label == 1 else "Ham"
        print(f"\n{class_name}:")
        for word, score in features[:5]:
            print(f"  {word}: {score:.4f}")
    
    # Plot feature importance
    plot_feature_importance(importance, top_k=8)
    
    # 3. Bernoulli Naive Bayes
    print("\n4. BERNOULLI NAIVE BAYES - BINARY FEATURES")
    print("-" * 50)
    
    # Use binarized text data
    X_binary = (X_text > 0).astype(float)
    X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
        X_binary, y_text, test_size=0.3, random_state=42
    )
    
    # Train Bernoulli Naive Bayes
    bernoulli_nb = BernoulliNaiveBayes(alpha=1.0)
    bernoulli_nb.fit(X_train_binary, y_train_binary)
    
    # Make predictions
    y_pred_binary = bernoulli_nb.predict(X_test_binary)
    
    # Evaluate
    accuracy_binary = accuracy_score(y_test_binary, y_pred_binary)
    print(f"Bernoulli NB Accuracy: {accuracy_binary:.4f}")
    
    # 4. Laplace Smoothing Analysis
    print("\n5. LAPLACE SMOOTHING IMPACT ANALYSIS")
    print("-" * 50)
    
    alpha_values = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    smoothing_results = evaluate_laplace_smoothing(
        X_train_text, y_train_text, X_test_text, y_test_text, alpha_values
    )
    
    # 5. Model Comparison
    print("\n6. MODEL COMPARISON")
    print("-" * 50)
    
    # Prepare results for confusion matrix plotting
    models_results = {
        'Gaussian NB': {
            'y_true': y_test_iris,
            'y_pred': y_pred_iris,
            'accuracy': accuracy_iris
        },
        'Multinomial NB': {
            'y_true': y_test_text,
            'y_pred': y_pred_text,
            'accuracy': accuracy_text
        },
        'Bernoulli NB': {
            'y_true': y_test_binary,
            'y_pred': y_pred_binary,
            'accuracy': accuracy_binary
        }
    }
    
    plot_confusion_matrices(models_results)
    
    # 6. Comparison with Scikit-learn
    print("\n7. COMPARISON WITH SCIKIT-LEARN")
    print("-" * 50)
    
    # Compare Gaussian NB
    gaussian_comparison = compare_with_sklearn(
        X_train_iris, y_train_iris, X_test_iris, y_test_iris, 'gaussian'
    )
    
    # Compare Multinomial NB
    multinomial_comparison = compare_with_sklearn(
        X_train_text, y_train_text, X_test_text, y_test_text, 'multinomial'
    )
    
    # Compare Bernoulli NB
    bernoulli_comparison = compare_with_sklearn(
        X_train_binary, y_train_binary, X_test_binary, y_test_binary, 'bernoulli'
    )
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Generated visualizations:")
    print("- gaussian_distributions.png")
    print("- nb_decision_boundary.png")
    print("- feature_importance.png")
    print("- confusion_matrices.png")
    print("- laplace_smoothing_impact.png")


if __name__ == "__main__":
    main() 