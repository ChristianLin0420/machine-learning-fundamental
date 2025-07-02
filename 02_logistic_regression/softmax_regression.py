"""
Softmax Regression (Multinomial Logistic Regression) Implementation from Scratch

This module implements softmax regression with:
- Vectorized softmax function and gradient computation
- Cross-entropy loss with L2 regularization
- Gradient descent optimization
- Early stopping
- Comprehensive evaluation and comparison with scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Load and preprocess the Iris dataset from sklearn.
    
    Args:
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, class_names)
    """
    # Load iris dataset from sklearn
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, iris.feature_names, iris.target_names

def one_hot_encode(y, num_classes):
    """
    Convert integer labels to one-hot encoding.
    
    Args:
        y (np.array): Integer labels
        num_classes (int): Number of classes
    
    Returns:
        np.array: One-hot encoded labels
    """
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
    
    Returns:
        dict: Dictionary containing various metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): List of class names
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_loss_curve(losses, save_path=None):
    """
    Plot training loss curve.
    
    Args:
        losses (list): List of loss values over epochs
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_decision_boundary_2d(X, y, model, feature_names, class_names, save_path=None):
    """
    Plot decision boundary for 2D data (using first two features).
    
    Args:
        X (np.array): Feature data
        y (np.array): Labels
        model: Trained model with predict method
        feature_names (list): Names of features
        class_names (list): Names of classes
        save_path (str): Path to save the plot
    """
    # Use only first two features
    X_2d = X[:, :2]
    
    # Create mesh
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Create prediction grid
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    # Pad with zeros for remaining features if needed
    if X.shape[1] > 2:
        mesh_points = np.column_stack([mesh_points, 
                                     np.zeros((mesh_points.shape[0], X.shape[1] - 2))])
    
    # Predict on mesh
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = ['red', 'green', 'blue']
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    
    # Plot data points
    for i, class_name in enumerate(class_names):
        idx = y == i
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], 
                   c=colors[i], label=class_name, s=50, edgecolors='black')
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Decision Boundary (2D Projection)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def compare_models_metrics(metrics1, metrics2, model1_name="Custom", model2_name="Scikit-learn"):
    """
    Compare metrics between two models.
    
    Args:
        metrics1 (dict): Metrics from first model
        metrics2 (dict): Metrics from second model
        model1_name (str): Name of first model
        model2_name (str): Name of second model
    """
    comparison_df = pd.DataFrame({
        model1_name: [metrics1['accuracy'], metrics1['f1_macro'], 
                     metrics1['f1_micro'], metrics1['f1_weighted']],
        model2_name: [metrics2['accuracy'], metrics2['f1_macro'], 
                     metrics2['f1_micro'], metrics2['f1_weighted']]
    }, index=['Accuracy', 'F1-Macro', 'F1-Micro', 'F1-Weighted'])
    
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(comparison_df.round(4))
    print("="*50)
    
    return comparison_df

def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification report.
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        class_names (list): Names of classes
    """
    print("\nClassification Report:")
    print("-" * 50)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

class SoftmaxRegression:
    """
    Softmax Regression (Multinomial Logistic Regression) implementation from scratch.
    
    This class implements multi-class classification using the softmax function
    and cross-entropy loss with L2 regularization.
    """
    
    def __init__(self, learning_rate=0.01, max_epochs=1000, regularization_strength=0.01, 
                 tolerance=1e-6, early_stopping_patience=50, random_state=42):
        """
        Initialize the Softmax Regression model.
        
        Args:
            learning_rate (float): Learning rate for gradient descent
            max_epochs (int): Maximum number of training epochs
            regularization_strength (float): L2 regularization parameter (lambda)
            tolerance (float): Tolerance for early stopping based on loss change
            early_stopping_patience (int): Number of epochs to wait before early stopping
            random_state (int): Random seed for weight initialization
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.regularization_strength = regularization_strength
        self.tolerance = tolerance
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        # Model parameters (to be initialized in fit)
        self.weights = None
        self.bias = None
        self.num_classes = None
        self.num_features = None
        
        # Training history
        self.loss_history = []
        self.epochs_trained = 0
    
    def _initialize_parameters(self, num_features, num_classes):
        """
        Initialize model parameters using Xavier initialization.
        
        Args:
            num_features (int): Number of input features
            num_classes (int): Number of output classes
        """
        np.random.seed(self.random_state)
        
        # Xavier initialization for weights
        xavier_bound = np.sqrt(6.0 / (num_features + num_classes))
        self.weights = np.random.uniform(
            -xavier_bound, xavier_bound, size=(num_features, num_classes)
        )
        
        # Initialize bias to zeros
        self.bias = np.zeros((1, num_classes))
        
        self.num_features = num_features
        self.num_classes = num_classes
    
    def _softmax(self, logits):
        """
        Compute softmax probabilities with numerical stability.
        
        Args:
            logits (np.array): Raw model outputs of shape (batch_size, num_classes)
        
        Returns:
            np.array: Softmax probabilities of shape (batch_size, num_classes)
        """
        # Subtract max for numerical stability
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def _compute_loss(self, y_true_onehot, y_pred_proba):
        """
        Compute cross-entropy loss with L2 regularization.
        
        Args:
            y_true_onehot (np.array): One-hot encoded true labels
            y_pred_proba (np.array): Predicted probabilities
        
        Returns:
            float: Total loss (cross-entropy + L2 regularization)
        """
        batch_size = y_true_onehot.shape[0]
        
        # Cross-entropy loss
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred_proba_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        cross_entropy = -np.sum(y_true_onehot * np.log(y_pred_proba_clipped)) / batch_size
        
        # L2 regularization term
        l2_penalty = self.regularization_strength * np.sum(self.weights ** 2) / 2
        
        return cross_entropy + l2_penalty
    
    def _compute_gradients(self, X, y_true_onehot, y_pred_proba):
        """
        Compute gradients of loss w.r.t weights and bias using vectorized operations.
        
        Args:
            X (np.array): Input features of shape (batch_size, num_features)
            y_true_onehot (np.array): One-hot encoded true labels
            y_pred_proba (np.array): Predicted probabilities
        
        Returns:
            tuple: (grad_weights, grad_bias)
        """
        batch_size = X.shape[0]
        
        # Error term: (y_pred - y_true)
        error = y_pred_proba - y_true_onehot
        
        # Gradient w.r.t weights: X^T * error / batch_size + L2 regularization
        grad_weights = np.dot(X.T, error) / batch_size + self.regularization_strength * self.weights
        
        # Gradient w.r.t bias: mean of error
        grad_bias = np.mean(error, axis=0, keepdims=True)
        
        return grad_weights, grad_bias
    
    def fit(self, X, y, verbose=True):
        """
        Train the softmax regression model using gradient descent.
        
        Args:
            X (np.array): Training features of shape (num_samples, num_features)
            y (np.array): Training labels of shape (num_samples,)
            verbose (bool): Whether to print training progress
        
        Returns:
            self: Returns the instance itself
        """
        # Get data dimensions
        num_samples, num_features = X.shape
        self.num_classes = len(np.unique(y))
        
        # Initialize parameters
        self._initialize_parameters(num_features, self.num_classes)
        
        # Convert labels to one-hot encoding
        y_onehot = one_hot_encode(y, self.num_classes)
        
        # Training loop
        self.loss_history = []
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            # Forward pass
            logits = np.dot(X, self.weights) + self.bias
            y_pred_proba = self._softmax(logits)
            
            # Compute loss
            current_loss = self._compute_loss(y_onehot, y_pred_proba)
            self.loss_history.append(current_loss)
            
            # Compute gradients
            grad_weights, grad_bias = self._compute_gradients(X, y_onehot, y_pred_proba)
            
            # Update parameters
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias
            
            # Early stopping check
            if current_loss < best_loss - self.tolerance:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                accuracy = self._compute_accuracy(X, y)
                print(f"Epoch {epoch + 1:4d} | Loss: {current_loss:.6f} | Accuracy: {accuracy:.4f}")
        
        self.epochs_trained = epoch + 1
        
        if verbose:
            final_accuracy = self._compute_accuracy(X, y)
            print(f"\nTraining completed!")
            print(f"Final training accuracy: {final_accuracy:.4f}")
            print(f"Total epochs: {self.epochs_trained}")
        
        return self
    
    def _compute_accuracy(self, X, y):
        """
        Compute accuracy on given data.
        
        Args:
            X (np.array): Features
            y (np.array): True labels
        
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (np.array): Features of shape (num_samples, num_features)
        
        Returns:
            np.array: Predicted probabilities of shape (num_samples, num_classes)
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        logits = np.dot(X, self.weights) + self.bias
        return self._softmax(logits)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X (np.array): Features of shape (num_samples, num_features)
        
        Returns:
            np.array: Predicted class labels of shape (num_samples,)
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


def main():
    """
    Main function to run the softmax regression experiment.
    """
    print("="*60)
    print("SOFTMAX REGRESSION FROM SCRATCH")
    print("="*60)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, class_names = load_and_preprocess_data(
        test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(class_names)} - {list(class_names)}")
    
    # Train custom softmax regression
    print("\n2. Training custom Softmax Regression...")
    custom_model = SoftmaxRegression(
        learning_rate=0.1,
        max_epochs=1000,
        regularization_strength=0.01,
        early_stopping_patience=50,
        random_state=42
    )
    
    custom_model.fit(X_train, y_train, verbose=True)
    
    # Train scikit-learn model for comparison
    print("\n3. Training Scikit-learn Logistic Regression for comparison...")
    sklearn_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=1.0/0.01,  # C = 1/lambda
        max_iter=1000,
        random_state=42
    )
    sklearn_model.fit(X_train, y_train)
    
    # Make predictions
    print("\n4. Making predictions...")
    
    # Custom model predictions
    custom_train_pred = custom_model.predict(X_train)
    custom_test_pred = custom_model.predict(X_test)
    custom_train_proba = custom_model.predict_proba(X_train)
    custom_test_proba = custom_model.predict_proba(X_test)
    
    # Scikit-learn predictions
    sklearn_train_pred = sklearn_model.predict(X_train)
    sklearn_test_pred = sklearn_model.predict(X_test)
    sklearn_train_proba = sklearn_model.predict_proba(X_train)
    sklearn_test_proba = sklearn_model.predict_proba(X_test)
    
    # Calculate metrics
    print("\n5. Evaluating models...")
    
    # Custom model metrics
    custom_train_metrics = calculate_metrics(y_train, custom_train_pred)
    custom_test_metrics = calculate_metrics(y_test, custom_test_pred)
    
    # Scikit-learn metrics
    sklearn_train_metrics = calculate_metrics(y_train, sklearn_train_pred)
    sklearn_test_metrics = calculate_metrics(y_test, sklearn_test_pred)
    
    # Print results
    print("\nCUSTOM MODEL RESULTS:")
    print(f"Training Accuracy: {custom_train_metrics['accuracy']:.4f}")
    print(f"Test Accuracy: {custom_test_metrics['accuracy']:.4f}")
    print(f"Test F1-Score (macro): {custom_test_metrics['f1_macro']:.4f}")
    
    print("\nSCIKIT-LEARN MODEL RESULTS:")
    print(f"Training Accuracy: {sklearn_train_metrics['accuracy']:.4f}")
    print(f"Test Accuracy: {sklearn_test_metrics['accuracy']:.4f}")
    print(f"Test F1-Score (macro): {sklearn_test_metrics['f1_macro']:.4f}")
    
    # Compare models
    compare_models_metrics(custom_test_metrics, sklearn_test_metrics, 
                          "Custom Softmax", "Scikit-learn")
    
    # Print detailed classification reports
    print("\n6. Detailed Classification Reports...")
    print("\nCustom Model:")
    print_classification_report(y_test, custom_test_pred, class_names)
    
    print("\nScikit-learn Model:")
    print_classification_report(y_test, sklearn_test_pred, class_names)
    
    # Create visualizations
    print("\n7. Creating visualizations...")
    
    # Plot training loss curve
    plot_loss_curve(custom_model.loss_history, save_path='plots/loss_curve.png')
    
    # Plot confusion matrices
    plot_confusion_matrix(y_test, custom_test_pred, class_names, 
                         save_path='plots/confusion_matrix.png')
    
    # Plot decision boundary (2D projection)
    plot_decision_boundary_2d(X_test, y_test, custom_model, feature_names, class_names,
                             save_path='plots/decision_boundary.png')
    
    print("\n8. Model Analysis...")
    print(f"Custom model converged in {custom_model.epochs_trained} epochs")
    print(f"Final training loss: {custom_model.loss_history[-1]:.6f}")
    
    # Compare weights (if possible)
    print("\nWeight matrix comparison:")
    print("Custom model weights shape:", custom_model.weights.shape)
    print("Scikit-learn weights shape:", sklearn_model.coef_.shape)
    
    # Weight magnitudes
    custom_weight_norm = np.linalg.norm(custom_model.weights)
    sklearn_weight_norm = np.linalg.norm(sklearn_model.coef_)
    print(f"Custom model weight L2 norm: {custom_weight_norm:.4f}")
    print(f"Scikit-learn weight L2 norm: {sklearn_weight_norm:.4f}")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Check the 'plots/' directory for visualizations.")


if __name__ == "__main__":
    main() 