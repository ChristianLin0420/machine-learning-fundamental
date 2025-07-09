"""
Random Forest Implementation from Scratch

This module implements Random Forest ensemble methods for both classification and regression:
- Bagging (Bootstrap Aggregating) of training data
- Random subspace method (feature bagging)
- Out-of-Bag (OOB) error estimation
- Feature importance aggregation
- Variance reduction through ensemble averaging
- Comprehensive evaluation and comparison with sklearn
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_diabetes, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
from sklearn.decomposition import PCA
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import or recreate the decision tree classes from Day 6
class TreeNode:
    """Node class for decision tree structure."""
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
    """Decision Tree Classifier for use in Random Forest."""
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0, random_state=None,
                 max_features=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.max_features = max_features
        
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
        probabilities = probabilities[probabilities > 0]
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
        """Find the best split for the data with feature randomization."""
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Feature randomization for Random Forest
        if self.max_features is not None:
            max_features = min(self.max_features, n_features)
            feature_indices = np.random.choice(n_features, max_features, replace=False)
        else:
            feature_indices = range(n_features)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
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
        
        node = TreeNode()
        node.samples = n_samples
        node.impurity = self._calculate_impurity(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or n_labels == 1:
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain < self.min_impurity_decrease or best_feature is None:
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.gain = best_gain
        
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Fit the decision tree to training data."""
        X = np.array(X)
        y = np.array(y)
        
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        self.root = self._build_tree(X, y)
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

class DecisionTreeRegressorScratch:
    """Decision Tree Regressor for use in Random Forest."""
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, random_state=None, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.max_features = max_features
        
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
        """Find the best split for regression with feature randomization."""
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Feature randomization for Random Forest
        if self.max_features is not None:
            max_features = min(self.max_features, n_features)
            feature_indices = np.random.choice(n_features, max_features, replace=False)
        else:
            feature_indices = range(n_features)
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                
                gain = self._variance_reduction(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the regression tree."""
        n_samples, n_features = X.shape
        
        node = TreeNode()
        node.samples = n_samples
        node.impurity = self._mse(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or node.impurity < 1e-7:
            node.value = np.mean(y)
            return node
        
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain < self.min_impurity_decrease or best_feature is None:
            node.value = np.mean(y)
            return node
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node.feature_idx = best_feature
        node.threshold = best_threshold
        node.gain = best_gain
        
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def fit(self, X, y):
        """Fit the regression tree to training data."""
        X = np.array(X)
        y = np.array(y)
        
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y)
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

class RandomForestClassifierScratch:
    """
    Random Forest Classifier implementation from scratch.
    
    Implements ensemble learning with bagging, feature randomization, and OOB estimation.
    """
    
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_impurity_decrease=0.0,
                 max_features='sqrt', bootstrap=True, oob_score=False, random_state=None):
        """
        Initialize Random Forest Classifier.
        
        Args:
            n_estimators (int): Number of trees in the forest
            criterion (str): Splitting criterion ('gini' or 'entropy')
            max_depth (int): Maximum tree depth
            min_samples_split (int): Minimum samples required to split
            min_samples_leaf (int): Minimum samples required in leaf
            min_impurity_decrease (float): Minimum impurity decrease for split
            max_features (str/int): Number of features to consider ('sqrt', 'log2', int)
            bootstrap (bool): Whether to use bootstrap sampling
            oob_score (bool): Whether to calculate out-of-bag score
            random_state (int): Random seed
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        
        self.estimators_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.oob_decision_function_ = None
        self.n_features_ = None
        self.n_classes_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_max_features(self, n_features):
        """Calculate max_features based on input."""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif self.max_features is None:
            return n_features
        else:
            raise ValueError("max_features must be 'sqrt', 'log2', int, or None")
    
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample of the data."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # Track out-of-bag samples
        oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        
        return X[indices], y[indices], indices, oob_indices
    
    def fit(self, X, y):
        """
        Fit Random Forest classifier.
        
        Args:
            X (np.array): Training features
            y (np.array): Training targets
        """
        X = np.array(X)
        y = np.array(y)
        
        n_samples, self.n_features_ = X.shape
        self.n_classes_ = len(np.unique(y))
        max_features = self._get_max_features(self.n_features_)
        
        self.estimators_ = []
        oob_predictions = np.zeros((n_samples, self.n_classes_))
        oob_counts = np.zeros(n_samples)
        
        # Build ensemble of trees
        for i in range(self.n_estimators):
            # Set random seed for reproducibility
            tree_random_state = None
            if self.random_state is not None:
                tree_random_state = self.random_state + i
            
            # Create decision tree
            tree = DecisionTreeClassifierScratch(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=max_features,
                random_state=tree_random_state
            )
            
            # Bootstrap sampling
            if self.bootstrap:
                X_bootstrap, y_bootstrap, bootstrap_indices, oob_indices = self._bootstrap_sample(X, y)
            else:
                X_bootstrap, y_bootstrap = X, y
                oob_indices = []
            
            # Fit tree
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)
            
            # Calculate OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                for j, idx in enumerate(oob_indices):
                    oob_predictions[idx, oob_pred[j]] += 1
                    oob_counts[idx] += 1
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        # Calculate OOB score
        if self.oob_score:
            self._calculate_oob_score(y, oob_predictions, oob_counts)
        
        return self
    
    def _calculate_feature_importances(self):
        """Calculate aggregated feature importances."""
        importances = np.zeros(self.n_features_)
        
        for tree in self.estimators_:
            importances += tree.feature_importances_
        
        # Average importances across trees
        importances = importances / len(self.estimators_)
        
        # Normalize
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        self.feature_importances_ = importances
    
    def _calculate_oob_score(self, y_true, oob_predictions, oob_counts):
        """Calculate out-of-bag score."""
        # Only consider samples that were out-of-bag at least once
        valid_oob = oob_counts > 0
        
        if np.sum(valid_oob) == 0:
            self.oob_score_ = 0.0
            return
        
        # Get OOB predictions
        oob_pred_classes = np.argmax(oob_predictions[valid_oob], axis=1)
        
        # Calculate accuracy
        self.oob_score_ = accuracy_score(y_true[valid_oob], oob_pred_classes)
        self.oob_decision_function_ = oob_predictions
    
    def predict(self, X):
        """Make predictions using majority voting."""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Collect predictions from all trees
        predictions = np.zeros((n_samples, len(self.estimators_)))
        
        for i, tree in enumerate(self.estimators_):
            predictions[:, i] = tree.predict(X)
        
        # Majority voting
        final_predictions = []
        for i in range(n_samples):
            vote_counts = Counter(predictions[i, :])
            final_predictions.append(vote_counts.most_common(1)[0][0])
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Collect predictions from all trees
        predictions = np.zeros((n_samples, len(self.estimators_)))
        
        for i, tree in enumerate(self.estimators_):
            predictions[:, i] = tree.predict(X)
        
        # Calculate probabilities based on votes
        probabilities = np.zeros((n_samples, self.n_classes_))
        
        for i in range(n_samples):
            for j in range(len(self.estimators_)):
                class_idx = int(predictions[i, j])
                probabilities[i, class_idx] += 1
            
            # Normalize
            probabilities[i] = probabilities[i] / len(self.estimators_)
        
        return probabilities
    
    def score(self, X, y):
        """Calculate accuracy score."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class RandomForestRegressorScratch:
    """
    Random Forest Regressor implementation from scratch.
    
    Implements ensemble learning with bagging, feature randomization, and averaging.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0, max_features='sqrt',
                 bootstrap=True, oob_score=False, random_state=None):
        """Initialize Random Forest Regressor."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        
        self.estimators_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.n_features_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_max_features(self, n_features):
        """Calculate max_features based on input."""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif self.max_features is None:
            return n_features
        else:
            return int(n_features / 3)  # Default for regression
    
    def _bootstrap_sample(self, X, y):
        """Create bootstrap sample of the data."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        return X[indices], y[indices], indices, oob_indices
    
    def fit(self, X, y):
        """Fit Random Forest regressor."""
        X = np.array(X)
        y = np.array(y)
        
        n_samples, self.n_features_ = X.shape
        max_features = self._get_max_features(self.n_features_)
        
        self.estimators_ = []
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)
        
        # Build ensemble of trees
        for i in range(self.n_estimators):
            tree_random_state = None
            if self.random_state is not None:
                tree_random_state = self.random_state + i
            
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                max_features=max_features,
                random_state=tree_random_state
            )
            
            # Bootstrap sampling
            if self.bootstrap:
                X_bootstrap, y_bootstrap, bootstrap_indices, oob_indices = self._bootstrap_sample(X, y)
            else:
                X_bootstrap, y_bootstrap = X, y
                oob_indices = []
            
            # Fit tree
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)
            
            # Calculate OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                for j, idx in enumerate(oob_indices):
                    oob_predictions[idx] += oob_pred[j]
                    oob_counts[idx] += 1
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        # Calculate OOB score
        if self.oob_score:
            self._calculate_oob_score(y, oob_predictions, oob_counts)
        
        return self
    
    def _calculate_feature_importances(self):
        """Calculate aggregated feature importances."""
        importances = np.zeros(self.n_features_)
        
        for tree in self.estimators_:
            importances += tree.feature_importances_
        
        importances = importances / len(self.estimators_)
        
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        self.feature_importances_ = importances
    
    def _calculate_oob_score(self, y_true, oob_predictions, oob_counts):
        """Calculate out-of-bag R² score."""
        valid_oob = oob_counts > 0
        
        if np.sum(valid_oob) == 0:
            self.oob_score_ = 0.0
            return
        
        # Average OOB predictions
        oob_pred_avg = oob_predictions[valid_oob] / oob_counts[valid_oob]
        
        # Calculate R² score
        ss_res = np.sum((y_true[valid_oob] - oob_pred_avg) ** 2)
        ss_tot = np.sum((y_true[valid_oob] - np.mean(y_true[valid_oob])) ** 2)
        self.oob_score_ = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def predict(self, X):
        """Make predictions using averaging."""
        X = np.array(X)
        n_samples = X.shape[0]
        
        # Collect predictions from all trees
        predictions = np.zeros((n_samples, len(self.estimators_)))
        
        for i, tree in enumerate(self.estimators_):
            predictions[:, i] = tree.predict(X)
        
        # Average predictions
        return np.mean(predictions, axis=1)
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def plot_n_estimators_performance(X, y, n_estimators_range, model_class, title):
    """
    Plot performance vs number of estimators.
    
    Args:
        X: Features
        y: Targets
        n_estimators_range: Range of n_estimators to test
        model_class: RandomForest class
        title: Plot title
    """
    train_scores = []
    test_scores = []
    oob_scores = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for n_est in n_estimators_range:
        model = model_class(n_estimators=n_est, oob_score=True, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        oob_score = model.oob_score_
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        oob_scores.append(oob_score)
    
    plt.figure(figsize=(12, 6))
    plt.plot(n_estimators_range, train_scores, 'b-o', label='Training Score', markersize=6)
    plt.plot(n_estimators_range, test_scores, 'r-s', label='Test Score', markersize=6)
    plt.plot(n_estimators_range, oob_scores, 'g-^', label='OOB Score', markersize=6)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/rf_n_estimators_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_scores, test_scores, oob_scores

def plot_feature_importance(model, feature_names=None, title="Random Forest Feature Importance", top_k=10):
    """Plot feature importance from Random Forest."""
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
    
    # Sort by importance
    indices = np.argsort(model.feature_importances_)[::-1][:top_k]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(indices)), model.feature_importances_[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('plots/rf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_oob_error_evolution(X, y, max_estimators=100, model_class=RandomForestClassifierScratch):
    """Plot OOB error evolution as trees are added."""
    model = model_class(n_estimators=1, oob_score=True, random_state=42)
    
    oob_errors = []
    n_estimators_list = []
    
    for i in range(1, max_estimators + 1):
        model.n_estimators = i
        model.fit(X, y)
        
        if model.oob_score_ is not None:
            oob_error = 1 - model.oob_score_
            oob_errors.append(oob_error)
            n_estimators_list.append(i)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_list, oob_errors, 'b-', linewidth=2)
    plt.xlabel('Number of Trees')
    plt.ylabel('OOB Error')
    plt.title('Out-of-Bag Error Evolution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/rf_oob_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return n_estimators_list, oob_errors

def compare_with_sklearn(X_train, y_train, X_test, y_test, task='classification'):
    """Compare custom implementation with scikit-learn."""
    print(f"\nComparison with sklearn - Random Forest {task.capitalize()}:")
    print("=" * 70)
    
    if task == 'classification':
        # Custom implementation
        custom_model = RandomForestClassifierScratch(n_estimators=50, random_state=42)
        custom_model.fit(X_train, y_train)
        custom_pred = custom_model.predict(X_test)
        custom_score = accuracy_score(y_test, custom_pred)
        
        # Sklearn implementation
        sklearn_model = RandomForestClassifier(n_estimators=50, random_state=42)
        sklearn_model.fit(X_train, y_train)
        sklearn_pred = sklearn_model.predict(X_test)
        sklearn_score = accuracy_score(y_test, sklearn_pred)
        
        metric_name = "Accuracy"
        
    else:  # regression
        # Custom implementation
        custom_model = RandomForestRegressorScratch(n_estimators=50, random_state=42)
        custom_model.fit(X_train, y_train)
        custom_pred = custom_model.predict(X_test)
        custom_score = custom_model.score(X_test, y_test)
        
        # Sklearn implementation
        sklearn_model = RandomForestRegressor(n_estimators=50, random_state=42)
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
    """Main function to run Random Forest experiments."""
    print("="*70)
    print("RANDOM FOREST IMPLEMENTATION FROM SCRATCH")
    print("="*70)
    
    # 1. Classification Example with Wine Dataset
    print("\n1. RANDOM FOREST CLASSIFICATION - WINE DATASET")
    print("-" * 50)
    
    # Load Wine dataset
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target
    
    # Split data
    X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
        X_wine, y_wine, test_size=0.3, random_state=42, stratify=y_wine
    )
    
    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifierScratch(
        n_estimators=50, max_depth=10, oob_score=True, random_state=42
    )
    rf_classifier.fit(X_train_wine, y_train_wine)
    
    # Make predictions
    y_pred_wine = rf_classifier.predict(X_test_wine)
    
    # Evaluate
    accuracy_wine = accuracy_score(y_test_wine, y_pred_wine)
    print(f"Random Forest Classifier Accuracy: {accuracy_wine:.4f}")
    print(f"Out-of-Bag Score: {rf_classifier.oob_score_:.4f}")
    print(f"Number of Trees: {rf_classifier.n_estimators}")
    print("\nClassification Report:")
    print(classification_report(y_test_wine, y_pred_wine, target_names=wine.target_names))
    
    # Plot feature importance
    plot_feature_importance(rf_classifier, wine.feature_names, 
                           "Random Forest Feature Importance (Wine)")
    
    # 2. Regression Example with California Housing
    print("\n2. RANDOM FOREST REGRESSION - CALIFORNIA HOUSING")
    print("-" * 50)
    
    # Load California housing dataset
    california = fetch_california_housing()
    X_housing, y_housing = california.data, california.target
    
    # Use subset for faster computation
    n_samples = 5000
    indices = np.random.choice(len(X_housing), n_samples, replace=False)
    X_housing = X_housing[indices]
    y_housing = y_housing[indices]
    
    # Split data
    X_train_housing, X_test_housing, y_train_housing, y_test_housing = train_test_split(
        X_housing, y_housing, test_size=0.3, random_state=42
    )
    
    # Train Random Forest Regressor
    rf_regressor = RandomForestRegressorScratch(
        n_estimators=50, max_depth=10, oob_score=True, random_state=42
    )
    rf_regressor.fit(X_train_housing, y_train_housing)
    
    # Make predictions
    y_pred_housing = rf_regressor.predict(X_test_housing)
    
    # Evaluate
    r2_housing = rf_regressor.score(X_test_housing, y_test_housing)
    rmse_housing = np.sqrt(mean_squared_error(y_test_housing, y_pred_housing))
    print(f"Random Forest Regressor R² Score: {r2_housing:.4f}")
    print(f"Root Mean Squared Error: {rmse_housing:.4f}")
    print(f"Out-of-Bag Score: {rf_regressor.oob_score_:.4f}")
    
    # Plot predicted vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_housing, y_pred_housing, alpha=0.6)
    plt.plot([y_test_housing.min(), y_test_housing.max()], 
             [y_test_housing.min(), y_test_housing.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest Regression: Predicted vs Actual (California Housing)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/rf_regression_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot feature importance for regression
    plot_feature_importance(rf_regressor, california.feature_names, 
                           "Random Forest Feature Importance (California Housing)")
    
    # 3. Number of Estimators Analysis
    print("\n3. NUMBER OF ESTIMATORS ANALYSIS")
    print("-" * 50)
    
    n_estimators_range = [1, 5, 10, 20, 30, 50, 75, 100]
    
    # Classification performance vs n_estimators
    print("Classification Performance vs Number of Trees:")
    train_scores_clf, test_scores_clf, oob_scores_clf = plot_n_estimators_performance(
        X_wine, y_wine, n_estimators_range, RandomForestClassifierScratch,
        "Random Forest Classification: Performance vs Number of Trees"
    )
    
    # 4. OOB Error Evolution
    print("\n4. OUT-OF-BAG ERROR EVOLUTION")
    print("-" * 50)
    
    # Use Iris dataset for faster computation
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    print("Tracking OOB error as trees are added...")
    n_trees, oob_errors = plot_oob_error_evolution(X_iris, y_iris, max_estimators=50)
    
    # 5. Feature Importance Analysis
    print("\n5. FEATURE IMPORTANCE COMPARISON")
    print("-" * 50)
    
    print("Top 5 most important features (Wine dataset):")
    importance_indices = np.argsort(rf_classifier.feature_importances_)[::-1]
    for i in range(5):
        idx = importance_indices[i]
        print(f"  {wine.feature_names[idx]}: {rf_classifier.feature_importances_[idx]:.4f}")
    
    # 6. Comparison with Scikit-learn
    print("\n6. COMPARISON WITH SCIKIT-LEARN")
    print("-" * 50)
    
    # Compare classification
    clf_comparison = compare_with_sklearn(
        X_train_wine, y_train_wine, X_test_wine, y_test_wine, 'classification'
    )
    
    # Compare regression
    reg_comparison = compare_with_sklearn(
        X_train_housing, y_train_housing, X_test_housing, y_test_housing, 'regression'
    )
    
    # 7. Bias-Variance Analysis
    print("\n7. BIAS-VARIANCE TRADEOFF DEMONSTRATION")
    print("-" * 50)
    
    # Compare single tree vs Random Forest
    single_tree = DecisionTreeClassifierScratch(max_depth=10, random_state=42)
    single_tree.fit(X_train_wine, y_train_wine)
    single_tree_score = single_tree.score(X_test_wine, y_test_wine)
    
    print(f"Single Decision Tree Accuracy: {single_tree_score:.4f}")
    print(f"Random Forest Accuracy:        {accuracy_wine:.4f}")
    print(f"Improvement from Ensemble:     {accuracy_wine - single_tree_score:.4f}")
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Generated visualizations:")
    print("- rf_feature_importance.png")
    print("- rf_n_estimators_performance.png") 
    print("- rf_oob_evolution.png")
    print("- rf_regression_scatter.png")

if __name__ == "__main__":
    main() 