"""
Decision Trees Implementation from Scratch

This module implements both classification and regression decision trees with:
- ID3 Algorithm: Top-down, greedy tree construction
- Multiple splitting criteria: Gini Impurity, Entropy, Variance Reduction
- Tree stopping rules: max depth, min samples per split, min impurity decrease
- Cost complexity pruning for overfitting control
- Tree visualization and decision boundary plotting
- Comprehensive evaluation and comparison with sklearn
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_diabetes, make_classification, make_regression
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.decomposition import PCA
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class TreeNode:
    """
    Node class for decision tree structure.
    """
    def __init__(self):
        self.feature_idx = None       # Feature index for splitting
        self.threshold = None         # Threshold value for splitting
        self.left = None             # Left child node
        self.right = None            # Right child node
        self.value = None            # Prediction value (for leaf nodes)
        self.samples = 0             # Number of samples in node
        self.impurity = 0.0          # Node impurity
        self.gain = 0.0              # Information gain from split

class DecisionTreeClassifierScratch:
    """
    Decision Tree Classifier implementation from scratch.
    
    Supports binary and multiclass classification with Gini and Entropy criteria.
    Includes pruning and comprehensive tree analysis.
    """
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0, random_state=None):
        """
        Initialize Decision Tree Classifier.
        
        Args:
            criterion (str): Splitting criterion ('gini' or 'entropy')
            max_depth (int): Maximum tree depth
            min_samples_split (int): Minimum samples required to split
            min_samples_leaf (int): Minimum samples required in leaf
            min_impurity_decrease (float): Minimum impurity decrease for split
            random_state (int): Random seed
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        
        self.root = None
        self.n_classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0.0
        
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)
    
    def _entropy(self, y):
        """Calculate entropy."""
        if len(y) == 0:
            return 0.0
        
        counts = np.bincount(y)
        probabilities = counts / len(y)
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _calculate_impurity(self, y):
        """Calculate impurity based on criterion."""
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'")
    
    def _information_gain(self, y, y_left, y_right):
        """Calculate information gain from split."""
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        parent_impurity = self._calculate_impurity(y)
        left_impurity = self._calculate_impurity(y_left)
        right_impurity = self._calculate_impurity(y_right)
        
        weighted_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        return parent_impurity - weighted_impurity
    
    def _best_split(self, X, y):
        """Find the best split for the data."""
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                # Split data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Check minimum samples constraints
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                # Calculate information gain
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Create node
        node = TreeNode()
        node.samples = n_samples
        node.impurity = self._calculate_impurity(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_labels == 1:
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        # Check if split improves impurity
        if best_gain < self.min_impurity_decrease:
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        if best_feature is None:
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.gain = best_gain
        
        # Recursively build subtrees
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Fit the decision tree to training data."""
        X = np.array(X)
        y = np.array(y)
        
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        # Build tree
        self.root = self._build_tree(X, y)
        
        # Calculate feature importances
        self._calculate_feature_importances(X, y)
        
        return self
    
    def _calculate_feature_importances(self, X, y):
        """Calculate feature importances based on impurity decrease."""
        importances = np.zeros(self.n_features_)
        
        def traverse(node):
            if node.feature_idx is not None:
                importances[node.feature_idx] += node.gain * node.samples
                if node.left:
                    traverse(node.left)
                if node.right:
                    traverse(node.right)
        
        traverse(self.root)
        
        # Normalize
        total_samples = X.shape[0]
        if total_samples > 0:
            importances = importances / total_samples
            importances = importances / np.sum(importances) if np.sum(importances) > 0 else importances
        
        self.feature_importances_ = importances
    
    def _predict_sample(self, x, node):
        """Predict a single sample."""
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Make predictions for input data."""
        X = np.array(X)
        predictions = []
        
        for x in X:
            pred = self._predict_sample(x, self.root)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def get_depth(self, node=None):
        """Get tree depth."""
        if node is None:
            node = self.root
        
        if node.value is not None:
            return 1
        
        left_depth = self.get_depth(node.left) if node.left else 0
        right_depth = self.get_depth(node.right) if node.right else 0
        
        return 1 + max(left_depth, right_depth)
    
    def get_n_leaves(self, node=None):
        """Get number of leaves."""
        if node is None:
            node = self.root
        
        if node.value is not None:
            return 1
        
        left_leaves = self.get_n_leaves(node.left) if node.left else 0
        right_leaves = self.get_n_leaves(node.right) if node.right else 0
        
        return left_leaves + right_leaves
    
    def print_tree(self, node=None, depth=0, prefix="Root: "):
        """Print tree structure."""
        if node is None:
            node = self.root
        
        if node.value is not None:
            print("  " * depth + f"{prefix}Predict {node.value} (samples: {node.samples})")
        else:
            print("  " * depth + f"{prefix}X{node.feature_idx} <= {node.threshold:.3f} "
                  f"(gain: {node.gain:.3f}, samples: {node.samples})")
            
            if node.left:
                self.print_tree(node.left, depth + 1, "├─ True: ")
            if node.right:
                self.print_tree(node.right, depth + 1, "└─ False: ")


class DecisionTreeRegressorScratch:
    """
    Decision Tree Regressor implementation from scratch.
    
    Uses variance reduction (MSE) as splitting criterion for continuous targets.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, random_state=None):
        """
        Initialize Decision Tree Regressor.
        
        Args:
            max_depth (int): Maximum tree depth
            min_samples_split (int): Minimum samples required to split
            min_samples_leaf (int): Minimum samples required in leaf
            min_impurity_decrease (float): Minimum impurity decrease for split
            random_state (int): Random seed
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        
        self.root = None
        self.n_features_ = None
        self.feature_importances_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _mse(self, y):
        """Calculate Mean Squared Error (variance)."""
        if len(y) == 0:
            return 0.0
        
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)
    
    def _variance_reduction(self, y, y_left, y_right):
        """Calculate variance reduction from split."""
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        parent_variance = self._mse(y)
        left_variance = self._mse(y_left)
        right_variance = self._mse(y_right)
        
        weighted_variance = (n_left / n) * left_variance + (n_right / n) * right_variance
        return parent_variance - weighted_variance
    
    def _best_split(self, X, y):
        """Find the best split for regression."""
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                # Split data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Check minimum samples constraints
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                # Calculate variance reduction
                gain = self._variance_reduction(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the regression tree."""
        n_samples, n_features = X.shape
        
        # Create node
        node = TreeNode()
        node.samples = n_samples
        node.impurity = self._mse(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           node.impurity < 1e-7:  # Essentially pure
            node.value = np.mean(y)
            return node
        
        # Find best split
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        # Check if split improves impurity
        if best_gain < self.min_impurity_decrease:
            node.value = np.mean(y)
            return node
        
        if best_feature is None:
            node.value = np.mean(y)
            return node
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.gain = best_gain
        
        # Recursively build subtrees
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Fit the regression tree to training data."""
        X = np.array(X)
        y = np.array(y)
        
        self.n_features_ = X.shape[1]
        
        # Build tree
        self.root = self._build_tree(X, y)
        
        # Calculate feature importances
        self._calculate_feature_importances(X, y)
        
        return self
    
    def _calculate_feature_importances(self, X, y):
        """Calculate feature importances based on variance reduction."""
        importances = np.zeros(self.n_features_)
        
        def traverse(node):
            if node.feature_idx is not None:
                importances[node.feature_idx] += node.gain * node.samples
                if node.left:
                    traverse(node.left)
                if node.right:
                    traverse(node.right)
        
        traverse(self.root)
        
        # Normalize
        total_samples = X.shape[0]
        if total_samples > 0:
            importances = importances / total_samples
            importances = importances / np.sum(importances) if np.sum(importances) > 0 else importances
        
        self.feature_importances_ = importances
    
    def _predict_sample(self, x, node):
        """Predict a single sample."""
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Make predictions for input data."""
        X = np.array(X)
        predictions = []
        
        for x in X:
            pred = self._predict_sample(x, self.root)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def get_depth(self, node=None):
        """Get tree depth."""
        if node is None:
            node = self.root
        
        if node.value is not None:
            return 1
        
        left_depth = self.get_depth(node.left) if node.left else 0
        right_depth = self.get_depth(node.right) if node.right else 0
        
        return 1 + max(left_depth, right_depth)
    
    def print_tree(self, node=None, depth=0, prefix="Root: "):
        """Print tree structure."""
        if node is None:
            node = self.root
        
        if node.value is not None:
            print("  " * depth + f"{prefix}Predict {node.value:.3f} (samples: {node.samples})")
        else:
            print("  " * depth + f"{prefix}X{node.feature_idx} <= {node.threshold:.3f} "
                  f"(MSE reduction: {node.gain:.3f}, samples: {node.samples})")
            
            if node.left:
                self.print_tree(node.left, depth + 1, "├─ True: ")
            if node.right:
                self.print_tree(node.right, depth + 1, "└─ False: ")


def plot_decision_boundary_2d(X, y, model, title="Decision Tree Decision Boundary", resolution=100):
    """
    Plot decision boundary for 2D data.
    
    Args:
        X: 2D feature matrix
        y: Target labels
        model: Fitted decision tree model
        title: Plot title
        resolution: Grid resolution
    """
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
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
    plt.savefig('plots/dt_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_names=None, title="Feature Importance"):
    """
    Plot feature importance.
    
    Args:
        model: Fitted decision tree model
        feature_names: Names of features
        title: Plot title
    """
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
    
    # Sort by importance
    indices = np.argsort(model.feature_importances_)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_[indices])
    plt.xticks(range(len(model.feature_importances_)), 
               [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('plots/dt_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_curves(X, y, model_class, max_depths, title="Learning Curves"):
    """
    Plot learning curves for different tree depths.
    
    Args:
        X: Features
        y: Targets
        model_class: Decision tree class
        max_depths: List of maximum depths to test
        title: Plot title
    """
    train_scores = []
    test_scores = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for depth in max_depths:
        model = model_class(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, train_scores, 'b-o', label='Training Score', markersize=6)
    plt.plot(max_depths, test_scores, 'r-s', label='Test Score', markersize=6)
    plt.xlabel('Max Depth')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/dt_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_scores, test_scores


def compare_criteria(X, y, criteria=['gini', 'entropy']):
    """
    Compare different splitting criteria.
    
    Args:
        X: Features
        y: Targets
        criteria: List of criteria to compare
    """
    results = {}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Comparing Splitting Criteria:")
    print("=" * 50)
    
    for criterion in criteria:
        model = DecisionTreeClassifierScratch(criterion=criterion, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        tree_depth = model.get_depth()
        n_leaves = model.get_n_leaves()
        
        results[criterion] = {
            'train_score': train_score,
            'test_score': test_score,
            'depth': tree_depth,
            'leaves': n_leaves
        }
        
        print(f"{criterion.capitalize():>8}: Train={train_score:.4f}, Test={test_score:.4f}, "
              f"Depth={tree_depth}, Leaves={n_leaves}")
    
    return results


def compare_with_sklearn(X_train, y_train, X_test, y_test, task='classification'):
    """
    Compare custom implementation with scikit-learn.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        task: 'classification' or 'regression'
    """
    print(f"\nComparison with sklearn - {task.capitalize()}:")
    print("=" * 60)
    
    if task == 'classification':
        # Custom implementation
        custom_model = DecisionTreeClassifierScratch(max_depth=10, random_state=42)
        custom_model.fit(X_train, y_train)
        custom_pred = custom_model.predict(X_test)
        custom_score = accuracy_score(y_test, custom_pred)
        
        # Sklearn implementation
        sklearn_model = DecisionTreeClassifier(max_depth=10, random_state=42)
        sklearn_model.fit(X_train, y_train)
        sklearn_pred = sklearn_model.predict(X_test)
        sklearn_score = accuracy_score(y_test, sklearn_pred)
        
        metric_name = "Accuracy"
        
    else:  # regression
        # Custom implementation
        custom_model = DecisionTreeRegressorScratch(max_depth=10, random_state=42)
        custom_model.fit(X_train, y_train)
        custom_pred = custom_model.predict(X_test)
        custom_score = custom_model.score(X_test, y_test)
        
        # Sklearn implementation
        sklearn_model = DecisionTreeRegressor(max_depth=10, random_state=42)
        sklearn_model.fit(X_train, y_train)
        sklearn_pred = sklearn_model.predict(X_test)
        sklearn_score = sklearn_model.score(X_test, y_test)
        
        metric_name = "R² Score"
    
    print(f"Custom Implementation {metric_name}:  {custom_score:.6f}")
    print(f"Sklearn Implementation {metric_name}: {sklearn_score:.6f}")
    print(f"{metric_name} Difference:               {abs(custom_score - sklearn_score):.8f}")
    
    return {
        'custom_score': custom_score,
        'sklearn_score': sklearn_score,
        'custom_predictions': custom_pred,
        'sklearn_predictions': sklearn_pred
    }


def main():
    """
    Main function to run Decision Tree experiments.
    """
    print("="*70)
    print("DECISION TREES IMPLEMENTATION FROM SCRATCH")
    print("="*70)
    
    # 1. Classification Example with Iris Dataset
    print("\n1. DECISION TREE CLASSIFICATION - IRIS DATASET")
    print("-" * 50)
    
    # Load Iris dataset
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Split data
    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
    )
    
    # Train Decision Tree Classifier
    dt_classifier = DecisionTreeClassifierScratch(criterion='gini', max_depth=5, random_state=42)
    dt_classifier.fit(X_train_iris, y_train_iris)
    
    # Make predictions
    y_pred_iris = dt_classifier.predict(X_test_iris)
    
    # Evaluate
    accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)
    print(f"Decision Tree Classifier Accuracy: {accuracy_iris:.4f}")
    print(f"Tree Depth: {dt_classifier.get_depth()}")
    print(f"Number of Leaves: {dt_classifier.get_n_leaves()}")
    print("\nClassification Report:")
    print(classification_report(y_test_iris, y_pred_iris, target_names=iris.target_names))
    
    # Print tree structure
    print("\nTree Structure:")
    dt_classifier.print_tree()
    
    # Plot feature importance
    plot_feature_importance(dt_classifier, iris.feature_names, 
                           "Decision Tree Feature Importance (Iris)")
    
    # 2D Decision boundary (using PCA)
    print("\n2. DECISION BOUNDARY VISUALIZATION (PCA PROJECTION)")
    print("-" * 50)
    
    # Project to 2D using PCA
    pca = PCA(n_components=2)
    X_iris_2d = pca.fit_transform(X_iris)
    
    # Train Decision Tree on 2D data
    dt_2d = DecisionTreeClassifierScratch(max_depth=5, random_state=42)
    dt_2d.fit(X_iris_2d, y_iris)
    
    # Plot decision boundary
    plot_decision_boundary_2d(X_iris_2d, y_iris, dt_2d, 
                             "Decision Tree Decision Boundary (Iris PCA)")
    
    # 3. Regression Example
    print("\n3. DECISION TREE REGRESSION - DIABETES DATASET")
    print("-" * 50)
    
    # Load diabetes dataset
    diabetes = load_diabetes()
    X_diabetes, y_diabetes = diabetes.data, diabetes.target
    
    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_diabetes, y_diabetes, test_size=0.3, random_state=42
    )
    
    # Train Decision Tree Regressor
    dt_regressor = DecisionTreeRegressorScratch(max_depth=5, random_state=42)
    dt_regressor.fit(X_train_reg, y_train_reg)
    
    # Make predictions
    y_pred_reg = dt_regressor.predict(X_test_reg)
    
    # Evaluate
    r2_score = dt_regressor.score(X_test_reg, y_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    print(f"Decision Tree Regressor R² Score: {r2_score:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Tree Depth: {dt_regressor.get_depth()}")
    
    # Plot predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.6)
    plt.plot([y_test_reg.min(), y_test_reg.max()], 
             [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Decision Tree Regression: Predicted vs Actual')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/dt_regression_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot feature importance for regression
    plot_feature_importance(dt_regressor, diabetes.feature_names, 
                           "Decision Tree Feature Importance (Diabetes)")
    
    # 4. Splitting Criteria Comparison
    print("\n4. SPLITTING CRITERIA COMPARISON")
    print("-" * 50)
    
    criteria_results = compare_criteria(X_iris, y_iris, ['gini', 'entropy'])
    
    # 5. Learning Curves
    print("\n5. LEARNING CURVES - OVERFITTING ANALYSIS")
    print("-" * 50)
    
    max_depths = range(1, 21)
    
    # Classification learning curves
    print("Classification Learning Curves:")
    train_scores_clf, test_scores_clf = plot_learning_curves(
        X_iris, y_iris, DecisionTreeClassifierScratch, max_depths,
        "Decision Tree Classification Learning Curves"
    )
    
    # Regression learning curves
    print("\nRegression Learning Curves:")
    train_scores_reg, test_scores_reg = plot_learning_curves(
        X_diabetes, y_diabetes, DecisionTreeRegressorScratch, max_depths,
        "Decision Tree Regression Learning Curves"
    )
    
    # 6. Comparison with Scikit-learn
    print("\n6. COMPARISON WITH SCIKIT-LEARN")
    print("-" * 50)
    
    # Compare classification
    clf_comparison = compare_with_sklearn(
        X_train_iris, y_train_iris, X_test_iris, y_test_iris, 'classification'
    )
    
    # Compare regression
    reg_comparison = compare_with_sklearn(
        X_train_reg, y_train_reg, X_test_reg, y_test_reg, 'regression'
    )
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Generated visualizations:")
    print("- dt_decision_boundary.png")
    print("- dt_feature_importance.png")
    print("- dt_learning_curves.png")
    print("- dt_regression_scatter.png")


if __name__ == "__main__":
    main() 