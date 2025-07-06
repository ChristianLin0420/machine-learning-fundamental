"""
K-Nearest Neighbors Implementation from Scratch

This module implements:
- KNN Classification with majority voting
- KNN Regression with neighbor averaging  
- Multiple distance metrics (Euclidean, Manhattan, Cosine)
- Vectorized distance computations for efficiency
- Cross-validation for hyperparameter tuning
- Decision boundary visualization
- Optional KD-Tree for efficient neighbor search
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class KNNFromScratch:
    """
    K-Nearest Neighbors implementation from scratch supporting both classification and regression.
    """
    
    def __init__(self, k=5, distance_metric='euclidean', task='classification', weights='uniform'):
        """
        Initialize KNN model.
        
        Args:
            k (int): Number of neighbors to consider
            distance_metric (str): Distance metric ('euclidean', 'manhattan', 'cosine')
            task (str): Task type ('classification' or 'regression')
            weights (str): Weighting scheme ('uniform' or 'distance')
        """
        self.k = k
        self.distance_metric = distance_metric.lower()
        self.task = task.lower()
        self.weights = weights
        
        # Training data storage
        self.X_train = None
        self.y_train = None
        self.n_classes = None
        
        # Validation
        if self.distance_metric not in ['euclidean', 'manhattan', 'cosine']:
            raise ValueError("Distance metric must be 'euclidean', 'manhattan', or 'cosine'")
        if self.task not in ['classification', 'regression']:
            raise ValueError("Task must be 'classification' or 'regression'")
        if self.weights not in ['uniform', 'distance']:
            raise ValueError("Weights must be 'uniform' or 'distance'")
    
    def _compute_distances(self, X_test, X_train):
        """
        Compute distances between test and training points using vectorized operations.
        
        Args:
            X_test (np.array): Test samples (n_test, n_features)
            X_train (np.array): Training samples (n_train, n_features)
        
        Returns:
            np.array: Distance matrix (n_test, n_train)
        """
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(X_test, X_train)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(X_test, X_train)
        elif self.distance_metric == 'cosine':
            return self._cosine_distance(X_test, X_train)
    
    def _euclidean_distance(self, X_test, X_train):
        """Vectorized Euclidean distance computation."""
        # Broadcasting: (n_test, 1, n_features) - (1, n_train, n_features)
        diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    
    def _manhattan_distance(self, X_test, X_train):
        """Vectorized Manhattan distance computation."""
        # Broadcasting: (n_test, 1, n_features) - (1, n_train, n_features)
        diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        return np.sum(np.abs(diff), axis=2)
    
    def _cosine_distance(self, X_test, X_train):
        """Vectorized Cosine distance computation."""
        # Normalize vectors
        X_test_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)
        X_train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        cosine_sim = np.dot(X_test_norm, X_train_norm.T)
        
        # Cosine distance = 1 - cosine similarity
        return 1 - cosine_sim
    
    def fit(self, X, y):
        """
        Fit KNN model (store training data).
        
        Args:
            X (np.array): Training features
            y (np.array): Training targets
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        if self.task == 'classification':
            self.n_classes = len(np.unique(y))
        
        return self
    
    def _get_neighbors(self, X_test):
        """
        Get k nearest neighbors for each test point.
        
        Args:
            X_test (np.array): Test samples
        
        Returns:
            tuple: (neighbor_indices, distances)
        """
        distances = self._compute_distances(X_test, self.X_train)
        
        # Get indices of k nearest neighbors for each test point
        neighbor_indices = np.argsort(distances, axis=1)[:, :self.k]
        
        # Get corresponding distances
        neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
        
        return neighbor_indices, neighbor_distances
    
    def predict(self, X):
        """
        Make predictions for test data.
        
        Args:
            X (np.array): Test samples
        
        Returns:
            np.array: Predictions
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_test = np.array(X)
        neighbor_indices, neighbor_distances = self._get_neighbors(X_test)
        
        if self.task == 'classification':
            return self._predict_classification(neighbor_indices, neighbor_distances)
        else:
            return self._predict_regression(neighbor_indices, neighbor_distances)
    
    def _predict_classification(self, neighbor_indices, neighbor_distances):
        """Predict class labels using majority voting."""
        predictions = []
        
        for i in range(len(neighbor_indices)):
            # Get neighbor labels
            neighbor_labels = self.y_train[neighbor_indices[i]]
            
            if self.weights == 'uniform':
                # Simple majority vote
                vote_counts = Counter(neighbor_labels)
                predicted_class = vote_counts.most_common(1)[0][0]
            else:
                # Distance-weighted vote
                weights = 1 / (neighbor_distances[i] + 1e-8)  # Avoid division by zero
                weighted_votes = {}
                
                for label, weight in zip(neighbor_labels, weights):
                    if label not in weighted_votes:
                        weighted_votes[label] = 0
                    weighted_votes[label] += weight
                
                predicted_class = max(weighted_votes, key=weighted_votes.get)
            
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def _predict_regression(self, neighbor_indices, neighbor_distances):
        """Predict continuous values using neighbor averaging."""
        predictions = []
        
        for i in range(len(neighbor_indices)):
            # Get neighbor values
            neighbor_values = self.y_train[neighbor_indices[i]]
            
            if self.weights == 'uniform':
                # Simple average
                prediction = np.mean(neighbor_values)
            else:
                # Distance-weighted average
                weights = 1 / (neighbor_distances[i] + 1e-8)  # Avoid division by zero
                weights = weights / np.sum(weights)  # Normalize weights
                prediction = np.sum(neighbor_values * weights)
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for classification tasks.
        
        Args:
            X (np.array): Test samples
        
        Returns:
            np.array: Class probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        X_test = np.array(X)
        neighbor_indices, neighbor_distances = self._get_neighbors(X_test)
        
        probabilities = []
        
        for i in range(len(neighbor_indices)):
            # Get neighbor labels
            neighbor_labels = self.y_train[neighbor_indices[i]]
            
            # Initialize probability array
            class_probs = np.zeros(self.n_classes)
            
            if self.weights == 'uniform':
                # Count votes for each class
                for label in neighbor_labels:
                    class_probs[int(label)] += 1
                class_probs /= self.k
            else:
                # Distance-weighted probabilities
                weights = 1 / (neighbor_distances[i] + 1e-8)
                for label, weight in zip(neighbor_labels, weights):
                    class_probs[int(label)] += weight
                class_probs /= np.sum(class_probs)
            
            probabilities.append(class_probs)
        
        return np.array(probabilities)


class KDTree:
    """
    KD-Tree implementation for efficient nearest neighbor search.
    """
    
    def __init__(self, X, y, leaf_size=10):
        """
        Initialize KD-Tree.
        
        Args:
            X (np.array): Training features
            y (np.array): Training targets
            leaf_size (int): Maximum number of points in leaf nodes
        """
        self.X = np.array(X)
        self.y = np.array(y)
        self.leaf_size = leaf_size
        self.n_features = X.shape[1]
        self.tree = self._build_tree(np.arange(len(X)), 0)
    
    def _build_tree(self, indices, depth):
        """Recursively build KD-Tree."""
        if len(indices) <= self.leaf_size:
            return {
                'leaf': True,
                'indices': indices,
                'points': self.X[indices],
                'targets': self.y[indices]
            }
        
        # Choose splitting dimension (cycle through features)
        dim = depth % self.n_features
        
        # Sort indices by the chosen dimension
        sorted_indices = indices[np.argsort(self.X[indices, dim])]
        
        # Find median
        median_idx = len(sorted_indices) // 2
        median_point = sorted_indices[median_idx]
        
        # Split into left and right subtrees
        left_indices = sorted_indices[:median_idx]
        right_indices = sorted_indices[median_idx + 1:]
        
        return {
            'leaf': False,
            'dim': dim,
            'point': median_point,
            'value': self.X[median_point, dim],
            'left': self._build_tree(left_indices, depth + 1),
            'right': self._build_tree(right_indices, depth + 1)
        }
    
    def query(self, point, k=1):
        """
        Find k nearest neighbors to a query point.
        
        Args:
            point (np.array): Query point
            k (int): Number of neighbors to find
        
        Returns:
            tuple: (neighbor_indices, distances)
        """
        self.best_neighbors = []
        self.k = k
        self._search(self.tree, point, 0)
        
        # Sort by distance and return top k
        self.best_neighbors.sort(key=lambda x: x[1])
        neighbors = self.best_neighbors[:k]
        
        indices = [n[0] for n in neighbors]
        distances = [n[1] for n in neighbors]
        
        return np.array(indices), np.array(distances)
    
    def _search(self, node, point, depth):
        """Recursively search KD-Tree."""
        if node['leaf']:
            # Check all points in leaf
            for i, idx in enumerate(node['indices']):
                dist = np.linalg.norm(point - self.X[idx])
                self._add_neighbor(idx, dist)
        else:
            dim = node['dim']
            
            # Decide which subtree to search first
            if point[dim] < node['value']:
                near_subtree = node['left']
                far_subtree = node['right']
            else:
                near_subtree = node['right']
                far_subtree = node['left']
            
            # Search near subtree
            self._search(near_subtree, point, depth + 1)
            
            # Check if we need to search far subtree
            if (len(self.best_neighbors) < self.k or 
                abs(point[dim] - node['value']) < self.best_neighbors[-1][1]):
                self._search(far_subtree, point, depth + 1)
            
            # Check the splitting point
            dist = np.linalg.norm(point - self.X[node['point']])
            self._add_neighbor(node['point'], dist)
    
    def _add_neighbor(self, idx, dist):
        """Add a neighbor to the best neighbors list."""
        self.best_neighbors.append((idx, dist))
        self.best_neighbors.sort(key=lambda x: x[1])
        
        # Keep only k best neighbors
        if len(self.best_neighbors) > self.k:
            self.best_neighbors = self.best_neighbors[:self.k]


class KNNWithKDTree(KNNFromScratch):
    """
    KNN implementation using KD-Tree for efficient neighbor search.
    """
    
    def __init__(self, k=5, distance_metric='euclidean', task='classification', 
                 weights='uniform', leaf_size=10):
        super().__init__(k, distance_metric, task, weights)
        self.leaf_size = leaf_size
        self.kdtree = None
    
    def fit(self, X, y):
        """Fit KNN model and build KD-Tree."""
        super().fit(X, y)
        if self.distance_metric == 'euclidean':  # KD-Tree works best with Euclidean distance
            self.kdtree = KDTree(X, y, self.leaf_size)
        return self
    
    def predict(self, X):
        """Make predictions using KD-Tree for neighbor search."""
        if self.kdtree is None:
            # Fall back to brute force if KD-Tree not available
            return super().predict(X)
        
        X_test = np.array(X)
        predictions = []
        
        for point in X_test:
            # Use KD-Tree to find neighbors
            neighbor_indices, neighbor_distances = self.kdtree.query(point, self.k)
            
            if self.task == 'classification':
                neighbor_labels = self.y_train[neighbor_indices]
                
                if self.weights == 'uniform':
                    vote_counts = Counter(neighbor_labels)
                    predicted_class = vote_counts.most_common(1)[0][0]
                else:
                    weights = 1 / (neighbor_distances + 1e-8)
                    weighted_votes = {}
                    for label, weight in zip(neighbor_labels, weights):
                        if label not in weighted_votes:
                            weighted_votes[label] = 0
                        weighted_votes[label] += weight
                    predicted_class = max(weighted_votes, key=weighted_votes.get)
                
                predictions.append(predicted_class)
            else:
                neighbor_values = self.y_train[neighbor_indices]
                
                if self.weights == 'uniform':
                    prediction = np.mean(neighbor_values)
                else:
                    weights = 1 / (neighbor_distances + 1e-8)
                    weights = weights / np.sum(weights)
                    prediction = np.sum(neighbor_values * weights)
                
                predictions.append(prediction)
        
        return np.array(predictions)


def cross_validate_knn(X, y, k_range=range(1, 21), cv_folds=5, task='classification', 
                      distance_metric='euclidean', random_state=42):
    """
    Perform cross-validation to find optimal k for KNN.
    
    Args:
        X (np.array): Features
        y (np.array): Targets
        k_range (range): Range of k values to test
        cv_folds (int): Number of cross-validation folds
        task (str): Task type ('classification' or 'regression')
        distance_metric (str): Distance metric to use
        random_state (int): Random seed
    
    Returns:
        dict: Results containing k values and corresponding scores
    """
    np.random.seed(random_state)
    
    # Create folds
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // cv_folds
    
    results = {
        'k_values': list(k_range),
        'mean_scores': [],
        'std_scores': [],
        'all_scores': []
    }
    
    print(f"Running {cv_folds}-fold cross-validation for k in {list(k_range)}")
    
    for k in k_range:
        fold_scores = []
        
        for fold in range(cv_folds):
            # Create train/validation split
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < cv_folds - 1 else n_samples
            
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            X_train_fold, X_val_fold = X[train_indices], X[val_indices]
            y_train_fold, y_val_fold = y[train_indices], y[val_indices]
            
            # Train and evaluate KNN
            knn = KNNFromScratch(k=k, distance_metric=distance_metric, task=task)
            knn.fit(X_train_fold, y_train_fold)
            y_pred = knn.predict(X_val_fold)
            
            # Calculate score
            if task == 'classification':
                score = accuracy_score(y_val_fold, y_pred)
            else:
                score = -mean_squared_error(y_val_fold, y_pred)  # Negative MSE for maximization
            
            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        results['mean_scores'].append(mean_score)
        results['std_scores'].append(std_score)
        results['all_scores'].append(fold_scores)
        
        print(f"  k={k:2d}: {mean_score:.4f} (±{std_score:.4f})")
    
    # Find best k
    best_idx = np.argmax(results['mean_scores'])
    best_k = results['k_values'][best_idx]
    best_score = results['mean_scores'][best_idx]
    
    print(f"\nBest k = {best_k} with score = {best_score:.4f}")
    
    return results, best_k


def plot_cross_validation_results(cv_results, task='classification'):
    """
    Plot cross-validation results.
    
    Args:
        cv_results (dict): Results from cross_validate_knn
        task (str): Task type for y-axis label
    """
    k_values = cv_results['k_values']
    mean_scores = cv_results['mean_scores']
    std_scores = cv_results['std_scores']
    
    plt.figure(figsize=(12, 8))
    
    # Plot mean scores with error bars
    plt.subplot(2, 1, 1)
    plt.errorbar(k_values, mean_scores, yerr=std_scores, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=6)
    plt.xlabel('k (Number of Neighbors)')
    
    if task == 'classification':
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('KNN Cross-Validation: Accuracy vs k')
    else:
        plt.ylabel('Cross-Validation Score (-MSE)')
        plt.title('KNN Cross-Validation: -MSE vs k')
    
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # Plot individual fold scores
    plt.subplot(2, 1, 2)
    all_scores = np.array(cv_results['all_scores'])
    
    # Box plot of scores for each k
    plt.boxplot(all_scores.T, positions=k_values, widths=0.6)
    plt.xlabel('k (Number of Neighbors)')
    
    if task == 'classification':
        plt.ylabel('Fold Accuracy')
        plt.title('Distribution of Fold Accuracies')
    else:
        plt.ylabel('Fold Score (-MSE)')
        plt.title('Distribution of Fold Scores')
    
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(f'plots/knn_cross_validation_{task}.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_decision_boundary_2d(X, y, knn_model, title="KNN Decision Boundary", resolution=100):
    """
    Plot decision boundary for 2D data.
    
    Args:
        X (np.array): 2D features
        y (np.array): Target labels
        knn_model: Fitted KNN model
        title (str): Plot title
        resolution (int): Grid resolution for decision boundary
    """
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # Make predictions on the mesh grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn_model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{title} (k={knn_model.k})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/decision_boundary_{knn_model.k}.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_distance_metrics(X, y, k=5, task='classification'):
    """
    Compare performance of different distance metrics.
    
    Args:
        X (np.array): Features
        y (np.array): Targets
        k (int): Number of neighbors
        task (str): Task type
    """
    metrics = ['euclidean', 'manhattan', 'cosine']
    results = {}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Comparing Distance Metrics:")
    print("=" * 50)
    
    for metric in metrics:
        knn = KNNFromScratch(k=k, distance_metric=metric, task=task)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        if task == 'classification':
            score = accuracy_score(y_test, y_pred)
            print(f"{metric.capitalize():>10}: Accuracy = {score:.4f}")
        else:
            score = mean_squared_error(y_test, y_pred)
            print(f"{metric.capitalize():>10}: MSE = {score:.4f}")
        
        results[metric] = score
    
    return results


def performance_comparison_brute_vs_kdtree(n_samples_range=[100, 500, 1000, 2000], 
                                         n_features=10, k=5, random_state=42):
    """
    Compare performance of brute force vs KD-Tree for different dataset sizes.
    
    Args:
        n_samples_range (list): Range of sample sizes to test
        n_features (int): Number of features
        k (int): Number of neighbors
        random_state (int): Random seed
    """
    import time
    
    results = {
        'n_samples': [],
        'brute_force_time': [],
        'kdtree_time': [],
        'speedup': []
    }
    
    print("Performance Comparison: Brute Force vs KD-Tree")
    print("=" * 50)
    
    for n_samples in n_samples_range:
        # Generate synthetic data
        X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                 n_redundant=0, n_informative=n_features,
                                 random_state=random_state)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Brute force KNN
        start_time = time.time()
        knn_brute = KNNFromScratch(k=k, distance_metric='euclidean')
        knn_brute.fit(X_train, y_train)
        y_pred_brute = knn_brute.predict(X_test)
        brute_time = time.time() - start_time
        
        # KD-Tree KNN
        start_time = time.time()
        knn_kdtree = KNNWithKDTree(k=k, distance_metric='euclidean')
        knn_kdtree.fit(X_train, y_train)
        y_pred_kdtree = knn_kdtree.predict(X_test)
        kdtree_time = time.time() - start_time
        
        # Calculate speedup
        speedup = brute_time / kdtree_time if kdtree_time > 0 else 0
        
        results['n_samples'].append(n_samples)
        results['brute_force_time'].append(brute_time)
        results['kdtree_time'].append(kdtree_time)
        results['speedup'].append(speedup)
        
        print(f"n_samples={n_samples:4d}: Brute={brute_time:.4f}s, KD-Tree={kdtree_time:.4f}s, Speedup={speedup:.2f}x")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(results['n_samples'], results['brute_force_time'], 'b-o', label='Brute Force')
    plt.plot(results['n_samples'], results['kdtree_time'], 'r-s', label='KD-Tree')
    plt.xlabel('Number of Samples')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(results['n_samples'], results['speedup'], 'g-^', linewidth=2, markersize=8)
    plt.xlabel('Number of Samples')
    plt.ylabel('Speedup (Brute/KD-Tree)')
    plt.title('KD-Tree Speedup')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.loglog(results['n_samples'], results['brute_force_time'], 'b-o', label='Brute Force')
    plt.loglog(results['n_samples'], results['kdtree_time'], 'r-s', label='KD-Tree')
    plt.xlabel('Number of Samples')
    plt.ylabel('Time (seconds)')
    plt.title('Log-Log Scale Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def main():
    """
    Main function to run KNN experiments.
    """
    print("="*60)
    print("K-NEAREST NEIGHBORS FROM SCRATCH")
    print("="*60)
    
    # 1. Classification Example with Iris Dataset
    print("\n1. CLASSIFICATION EXAMPLE: IRIS DATASET")
    print("-" * 50)
    
    # Load and prepare data
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Standardize features
    scaler = StandardScaler()
    X_iris_scaled = scaler.fit_transform(X_iris)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_iris_scaled, y_iris, test_size=0.3, random_state=42, stratify=y_iris
    )
    
    # Train KNN classifier
    knn_classifier = KNNFromScratch(k=5, task='classification')
    knn_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = knn_classifier.predict(X_test)
    y_proba = knn_classifier.predict_proba(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN Classifier Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Cross-validation for k selection
    print("\n2. CROSS-VALIDATION FOR OPTIMAL K")
    print("-" * 50)
    
    cv_results, best_k = cross_validate_knn(X_iris_scaled, y_iris, k_range=range(1, 21), 
                                          task='classification')
    plot_cross_validation_results(cv_results, task='classification')
    
    # Decision boundary visualization (2D PCA projection)
    print("\n3. DECISION BOUNDARY VISUALIZATION")
    print("-" * 50)
    
    # Project to 2D using PCA
    pca = PCA(n_components=2)
    X_iris_2d = pca.fit_transform(X_iris_scaled)
    
    # Train KNN on 2D data
    knn_2d = KNNFromScratch(k=best_k, task='classification')
    knn_2d.fit(X_iris_2d, y_iris)
    
    # Plot decision boundary
    plot_decision_boundary_2d(X_iris_2d, y_iris, knn_2d, 
                             title="KNN Decision Boundary (Iris Dataset - PCA)")
    
    # 4. Regression Example
    print("\n4. REGRESSION EXAMPLE: SYNTHETIC DATA")
    print("-" * 50)
    
    # Generate synthetic regression data
    X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=0.1, random_state=42)
    
    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    
    # Train KNN regressor
    knn_regressor = KNNFromScratch(k=5, task='regression')
    knn_regressor.fit(X_train_reg, y_train_reg)
    
    # Make predictions
    y_pred_reg = knn_regressor.predict(X_test_reg)
    
    # Evaluate
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    print(f"KNN Regressor MSE: {mse:.4f}")
    
    # Plot regression results
    plt.figure(figsize=(12, 8))
    
    # Sort for plotting
    sort_idx = np.argsort(X_test_reg.ravel())
    X_test_sorted = X_test_reg[sort_idx]
    y_test_sorted = y_test_reg[sort_idx]
    y_pred_sorted = y_pred_reg[sort_idx]
    
    plt.scatter(X_train_reg, y_train_reg, alpha=0.6, label='Training Data', s=50)
    plt.scatter(X_test_reg, y_test_reg, alpha=0.6, label='Test Data', s=50)
    plt.plot(X_test_sorted, y_pred_sorted, 'r-', linewidth=2, label='KNN Predictions')
    
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title(f'KNN Regression (k={knn_regressor.k})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/knn_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Distance Metrics Comparison
    print("\n5. DISTANCE METRICS COMPARISON")
    print("-" * 50)
    
    distance_results = compare_distance_metrics(X_iris_scaled, y_iris, k=5, task='classification')
    
    # 6. Performance Comparison
    print("\n6. PERFORMANCE COMPARISON: BRUTE FORCE vs KD-TREE")
    print("-" * 50)
    
    perf_results = performance_comparison_brute_vs_kdtree(
        n_samples_range=[100, 500, 1000, 2000], n_features=10, k=5
    )
    
    # 7. Comparison with sklearn
    print("\n7. COMPARISON WITH SCIKIT-LEARN")
    print("-" * 50)
    
    # Our implementation
    our_knn = KNNFromScratch(k=best_k, task='classification')
    our_knn.fit(X_train, y_train)
    our_pred = our_knn.predict(X_test)
    our_accuracy = accuracy_score(y_test, our_pred)
    
    # sklearn implementation
    sklearn_knn = KNeighborsClassifier(n_neighbors=best_k)
    sklearn_knn.fit(X_train, y_train)
    sklearn_pred = sklearn_knn.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    print(f"Our KNN Accuracy:      {our_accuracy:.4f}")
    print(f"Scikit-learn Accuracy: {sklearn_accuracy:.4f}")
    print(f"Accuracy Difference:   {abs(our_accuracy - sklearn_accuracy):.6f}")
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Generated visualizations:")
    print("- knn_cross_validation_classification.png")
    print("- decision_boundary_X.png")
    print("- knn_regression.png")
    print("- performance_comparison.png")


if __name__ == "__main__":
    main() 