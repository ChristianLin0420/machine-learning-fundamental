"""
Support Vector Machine (SVM) Implementation from Scratch

This module implements Support Vector Machines from scratch with:
- Primal formulation (gradient descent on hinge loss)
- Dual formulation (quadratic programming with cvxopt)
- Linear and kernelized SVMs
- Multiple kernel types (Linear, Polynomial, RBF)
- Soft margin with slack variables
- Support vector identification
- Comprehensive visualization and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Try to import cvxopt for quadratic programming
try:
    import cvxopt
    import cvxopt.solvers
    cvxopt.solvers.options['show_progress'] = False
    CVXOPT_AVAILABLE = True
except ImportError:
    CVXOPT_AVAILABLE = False
    print("Warning: cvxopt not available. Dual formulation will use simplified approach.")

class SVMPrimal:
    """
    SVM implementation using primal formulation with gradient descent on hinge loss.
    
    Solves the primal optimization problem:
    min (1/2)||w||² + C * Σ max(0, 1 - yᵢ(w·xᵢ + b))
    """
    
    def __init__(self, C=1.0, learning_rate=0.01, max_iters=1000, tol=1e-6):
        """
        Initialize SVM with primal formulation.
        
        Args:
            C (float): Regularization parameter (inverse of regularization strength)
            learning_rate (float): Learning rate for gradient descent
            max_iters (int): Maximum number of iterations
            tol (float): Tolerance for convergence
        """
        self.C = C
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        
        self.w = None
        self.b = None
        self.support_vectors_ = None
        self.support_vector_indices_ = None
        self.n_support_ = None
        self.training_history_ = []
    
    def _hinge_loss(self, X, y):
        """
        Calculate hinge loss and its gradient.
        
        Hinge loss: L(w,b) = (1/2)||w||² + C * Σ max(0, 1 - yᵢ(w·xᵢ + b))
        """
        n_samples = X.shape[0]
        
        # Calculate predictions
        distances = y * (X.dot(self.w) + self.b)
        
        # Hinge loss
        hinge_loss = np.maximum(0, 1 - distances)
        loss = 0.5 * np.dot(self.w, self.w) + self.C * np.sum(hinge_loss)
        
        # Gradient calculation
        dw = self.w.copy()
        db = 0
        
        for i in range(n_samples):
            if distances[i] < 1:  # Violating samples
                dw -= self.C * y[i] * X[i]
                db -= self.C * y[i]
        
        return loss, dw, db
    
    def fit(self, X, y):
        """
        Fit SVM using gradient descent on hinge loss.
        
        Args:
            X (np.array): Training features
            y (np.array): Training labels (-1 or +1)
        """
        # Convert labels to -1, +1
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM is for binary classification only")
        
        y_binary = np.where(y == unique_labels[0], -1, 1)
        
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.w = np.random.normal(0, 0.01, n_features)
        self.b = 0
        
        prev_loss = float('inf')
        
        # Gradient descent
        for iteration in range(self.max_iters):
            loss, dw, db = self._hinge_loss(X, y_binary)
            
            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Store training history
            self.training_history_.append({
                'iteration': iteration,
                'loss': loss,
                'w_norm': np.linalg.norm(self.w)
            })
            
            # Check for convergence
            if abs(prev_loss - loss) < self.tol:
                break
            
            prev_loss = loss
        
        # Identify support vectors (samples with margin <= 1)
        distances = y_binary * (X.dot(self.w) + self.b)
        support_mask = distances <= 1.01  # Small tolerance for numerical stability
        
        self.support_vector_indices_ = np.where(support_mask)[0]
        self.support_vectors_ = X[support_mask]
        self.n_support_ = np.sum(support_mask)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        scores = X.dot(self.w) + self.b
        return np.where(scores >= 0, 1, -1)
    
    def decision_function(self, X):
        """Calculate decision function values."""
        return X.dot(self.w) + self.b
    
    def score(self, X, y):
        """Calculate accuracy."""
        y_pred = self.predict(X)
        # Convert y to -1, +1 format
        unique_labels = np.unique(y)
        y_binary = np.where(y == unique_labels[0], -1, 1)
        return accuracy_score(y_binary, y_pred)

class SVMDual:
    """
    SVM implementation using dual formulation with quadratic programming.
    
    Solves the dual optimization problem:
    max Σ αᵢ - (1/2) Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ)
    subject to: 0 ≤ αᵢ ≤ C and Σ αᵢ yᵢ = 0
    """
    
    def __init__(self, C=1.0, kernel='linear', gamma='scale', degree=3, coef0=0.0, tol=1e-3):
        """
        Initialize SVM with dual formulation.
        
        Args:
            C (float): Regularization parameter
            kernel (str): Kernel type ('linear', 'poly', 'rbf')
            gamma (str/float): Kernel coefficient ('scale', 'auto', or float)
            degree (int): Polynomial kernel degree
            coef0 (float): Independent term in polynomial/sigmoid kernels
            tol (float): Tolerance for optimization
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        
        self.alpha_ = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.support_vector_indices_ = None
        self.n_support_ = None
        self.intercept_ = None
        self._gamma = None
        
        # Store training data for kernelized predictions
        self.X_train_ = None
        self.y_train_ = None
    
    def _compute_gamma(self, X):
        """Compute gamma parameter for RBF and polynomial kernels."""
        if self.gamma == 'scale':
            return 1 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            return 1 / X.shape[1]
        else:
            return self.gamma
    
    def _kernel_function(self, X1, X2):
        """
        Compute kernel matrix K(X1, X2).
        
        Returns:
            kernel_matrix (np.array): K[i,j] = K(X1[i], X2[j])
        """
        if self.kernel == 'linear':
            return X1.dot(X2.T)
        
        elif self.kernel == 'poly':
            return (self._gamma * X1.dot(X2.T) + self.coef0) ** self.degree
        
        elif self.kernel == 'rbf':
            # Efficient computation of RBF kernel
            X1_norm = np.sum(X1**2, axis=1, keepdims=True)
            X2_norm = np.sum(X2**2, axis=1, keepdims=True)
            
            # ||x1 - x2||² = ||x1||² + ||x2||² - 2*x1·x2
            squared_distances = X1_norm + X2_norm.T - 2 * X1.dot(X2.T)
            return np.exp(-self._gamma * squared_distances)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _solve_qp_cvxopt(self, K, y):
        """
        Solve quadratic programming problem using cvxopt.
        
        Minimize: (1/2) x^T P x + q^T x
        Subject to: G x <= h and A x = b
        """
        n_samples = len(y)
        
        # Quadratic term: P = y_i * y_j * K(x_i, x_j)
        P = cvxopt.matrix(np.outer(y, y) * K)
        
        # Linear term: q = -1 (maximize Σ α_i)
        q = cvxopt.matrix(-np.ones(n_samples))
        
        # Inequality constraints: -α_i <= 0 and α_i <= C
        G = cvxopt.matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))
        h = cvxopt.matrix(np.hstack([np.zeros(n_samples), np.full(n_samples, self.C)]))
        
        # Equality constraint: Σ α_i y_i = 0
        A = cvxopt.matrix(y.reshape(1, -1).astype(float))
        b = cvxopt.matrix(0.0)
        
        # Solve QP
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Extract alphas
        alpha = np.ravel(solution['x'])
        
        return alpha
    
    def _solve_qp_simplified(self, K, y):
        """
        Simplified QP solver for when cvxopt is not available.
        Uses coordinate descent approach (SMO-like algorithm).
        """
        n_samples = len(y)
        alpha = np.random.random(n_samples) * 0.001
        
        # Simple coordinate descent
        for iteration in range(1000):
            alpha_old = alpha.copy()
            
            for i in range(n_samples):
                # Calculate error
                E_i = np.sum(alpha * y * K[i, :]) - y[i]
                
                # Update alpha[i]
                if y[i] * E_i < -self.tol and alpha[i] < self.C:
                    alpha[i] = min(self.C, alpha[i] + 0.01)
                elif y[i] * E_i > self.tol and alpha[i] > 0:
                    alpha[i] = max(0, alpha[i] - 0.01)
            
            # Check convergence
            if np.linalg.norm(alpha - alpha_old) < self.tol:
                break
        
        return alpha
    
    def fit(self, X, y):
        """
        Fit SVM using dual formulation.
        
        Args:
            X (np.array): Training features
            y (np.array): Training labels
        """
        # Convert labels to -1, +1
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM is for binary classification only")
        
        y_binary = np.where(y == unique_labels[0], -1, 1)
        
        n_samples, n_features = X.shape
        
        # Store training data
        self.X_train_ = X.copy()
        self.y_train_ = y_binary.copy()
        
        # Compute gamma parameter
        self._gamma = self._compute_gamma(X)
        
        # Compute kernel matrix
        K = self._kernel_function(X, X)
        
        # Solve quadratic programming problem
        if CVXOPT_AVAILABLE:
            alpha = self._solve_qp_cvxopt(K, y_binary)
        else:
            alpha = self._solve_qp_simplified(K, y_binary)
        
        # Identify support vectors (α > tolerance)
        support_vector_indices = alpha > self.tol
        self.alpha_ = alpha[support_vector_indices]
        self.support_vectors_ = X[support_vector_indices]
        self.support_vector_labels_ = y_binary[support_vector_indices]
        self.support_vector_indices_ = np.where(support_vector_indices)[0]
        self.n_support_ = len(self.alpha_)
        
        # Calculate intercept (bias term)
        # Use support vectors with 0 < α < C for numerical stability
        margin_sv_mask = (alpha > self.tol) & (alpha < self.C - self.tol)
        if np.any(margin_sv_mask):
            margin_sv_indices = np.where(margin_sv_mask)[0]
            intercepts = []
            
            for idx in margin_sv_indices:
                intercept = y_binary[idx]
                for i, sv_idx in enumerate(self.support_vector_indices_):
                    intercept -= self.alpha_[i] * self.support_vector_labels_[i] * K[idx, sv_idx]
                intercepts.append(intercept)
            
            self.intercept_ = np.mean(intercepts)
        else:
            # Fallback: use all support vectors
            self.intercept_ = np.mean(y_binary[support_vector_indices])
        
        return self
    
    def decision_function(self, X):
        """
        Calculate decision function values.
        
        f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b
        """
        if self.alpha_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Compute kernel between test points and support vectors
        K = self._kernel_function(X, self.support_vectors_)
        
        # Calculate decision function: K has shape (n_test, n_support)
        # alpha_ * support_vector_labels_ has shape (n_support,)
        # We want K @ (alpha_ * support_vector_labels_) -> shape (n_test,)
        decision = K.dot(self.alpha_ * self.support_vector_labels_) + self.intercept_
        
        return decision
    
    def predict(self, X):
        """Make predictions."""
        return np.where(self.decision_function(X) >= 0, 1, -1)
    
    def score(self, X, y):
        """Calculate accuracy."""
        y_pred = self.predict(X)
        # Convert y to -1, +1 format
        unique_labels = np.unique(y)
        y_binary = np.where(y == unique_labels[0], -1, 1)
        return accuracy_score(y_binary, y_pred)

def plot_decision_boundary(X, y, model, title="SVM Decision Boundary", 
                          support_vectors=None, filename=None):
    """
    Plot decision boundary with support vectors highlighted.
    
    Args:
        X (np.array): 2D feature data
        y (np.array): Labels
        model: Trained SVM model
        title (str): Plot title
        support_vectors (np.array): Support vector coordinates
        filename (str): Save filename
    """
    plt.figure(figsize=(12, 8))
    
    # Create a mesh for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Calculate decision function on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5, 
               linestyles=['--', '-', '--'], colors=['red', 'black', 'blue'])
    plt.contourf(xx, yy, Z, levels=[-np.inf, -1, 1, np.inf], 
                alpha=0.1, colors=['red', 'blue'])
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=50, alpha=0.8)
    
    # Highlight support vectors
    if support_vectors is not None and len(support_vectors) > 0:
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                   s=200, facecolors='none', edgecolors='black', linewidth=2,
                   label=f'Support Vectors ({len(support_vectors)})')
        plt.legend()
    
    plt.colorbar(scatter)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_regularization_analysis(X, y, C_range, title="SVM Regularization Analysis"):
    """
    Plot the effect of regularization parameter C on model complexity.
    
    Args:
        X (np.array): Training features
        y (np.array): Training labels
        C_range (list): Range of C values to test
        title (str): Plot title
    """
    train_scores = []
    val_scores = []
    n_support_vectors = []
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for C in C_range:
        # Train SVM with current C
        svm = SVMDual(C=C, kernel='rbf', gamma='scale')
        svm.fit(X_train, y_train)
        
        # Calculate scores
        train_score = svm.score(X_train, y_train)
        val_score = svm.score(X_val, y_val)
        n_sv = svm.n_support_
        
        train_scores.append(train_score)
        val_scores.append(val_score)
        n_support_vectors.append(n_sv)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Accuracy vs C
    ax1.semilogx(C_range, train_scores, 'b-o', label='Training Accuracy', markersize=6)
    ax1.semilogx(C_range, val_scores, 'r-s', label='Validation Accuracy', markersize=6)
    ax1.set_xlabel('Regularization Parameter C')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Regularization Strength')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of Support Vectors vs C
    ax2.semilogx(C_range, n_support_vectors, 'g-^', label='Support Vectors', markersize=6)
    ax2.set_xlabel('Regularization Parameter C')
    ax2.set_ylabel('Number of Support Vectors')
    ax2.set_title('Model Complexity vs Regularization Strength')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('plots/svm_regularization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_scores, val_scores, n_support_vectors

def plot_kernel_comparison(X, y, kernels=['linear', 'poly', 'rbf'], 
                          title="SVM Kernel Comparison"):
    """
    Compare different SVM kernels on the same dataset.
    
    Args:
        X (np.array): 2D feature data
        y (np.array): Labels
        kernels (list): List of kernel types to compare
        title (str): Overall title
    """
    fig, axes = plt.subplots(1, len(kernels), figsize=(5*len(kernels), 4))
    if len(kernels) == 1:
        axes = [axes]
    
    for i, kernel in enumerate(kernels):
        # Train SVM with current kernel
        svm = SVMDual(C=1.0, kernel=kernel, gamma='scale')
        svm.fit(X, y)
        
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Calculate decision function
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        axes[i].contour(xx, yy, Z, levels=[-1, 0, 1], alpha=0.5,
                       linestyles=['--', '-', '--'], colors=['red', 'black', 'blue'])
        axes[i].contourf(xx, yy, Z, levels=[-np.inf, -1, 1, np.inf],
                        alpha=0.1, colors=['red', 'blue'])
        
        # Plot data points
        scatter = axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=50, alpha=0.8)
        
        # Highlight support vectors
        if hasattr(svm, 'support_vectors_') and svm.support_vectors_ is not None:
            axes[i].scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                           s=200, facecolors='none', edgecolors='black', linewidth=2)
        
        # Calculate accuracy
        accuracy = svm.score(X, y)
        axes[i].set_title(f'{kernel.upper()} Kernel\nAccuracy: {accuracy:.3f}\nSVs: {svm.n_support_}')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('plots/svm_kernel_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_with_sklearn(X_train, y_train, X_test, y_test, kernel='rbf'):
    """Compare custom SVM implementation with scikit-learn."""
    print(f"\nComparison with sklearn SVM (kernel='{kernel}'):")
    print("=" * 60)
    
    # Custom implementation
    custom_svm = SVMDual(C=1.0, kernel=kernel, gamma='scale')
    custom_svm.fit(X_train, y_train)
    custom_pred = custom_svm.predict(X_test)
    
    # Convert labels for sklearn comparison
    unique_labels = np.unique(y_test)
    y_train_sklearn = np.where(y_train == unique_labels[0], 0, 1)
    y_test_sklearn = np.where(y_test == unique_labels[0], 0, 1)
    custom_pred_sklearn = np.where(custom_pred == -1, 0, 1)
    
    custom_accuracy = accuracy_score(y_test_sklearn, custom_pred_sklearn)
    
    # Sklearn implementation
    sklearn_svm = SVC(C=1.0, kernel=kernel, gamma='scale')
    sklearn_svm.fit(X_train, y_train_sklearn)  # Use training labels, not test labels
    sklearn_pred = sklearn_svm.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test_sklearn, sklearn_pred)
    
    print(f"Custom SVM Accuracy:   {custom_accuracy:.6f}")
    print(f"Sklearn SVM Accuracy:  {sklearn_accuracy:.6f}")
    print(f"Accuracy Difference:   {abs(custom_accuracy - sklearn_accuracy):.6f}")
    print(f"Custom Support Vectors: {custom_svm.n_support_}")
    print(f"Sklearn Support Vectors: {len(sklearn_svm.support_vectors_)}")
    
    return {
        'custom_accuracy': custom_accuracy,
        'sklearn_accuracy': sklearn_accuracy,
        'custom_n_support': custom_svm.n_support_,
        'sklearn_n_support': len(sklearn_svm.support_vectors_)
    }

def main():
    """Main function to run SVM experiments."""
    print("="*70)
    print("SUPPORT VECTOR MACHINE IMPLEMENTATION FROM SCRATCH")
    print("="*70)
    
    # 1. Linear SVM on Linearly Separable Data
    print("\n1. LINEAR SVM - LINEARLY SEPARABLE DATA")
    print("-" * 50)
    
    # Generate linearly separable data
    X_linear, y_linear = make_blobs(n_samples=100, centers=2, n_features=2,
                                   cluster_std=1.0, random_state=42)
    
    # Train both primal and dual formulations
    svm_primal = SVMPrimal(C=1.0, learning_rate=0.01, max_iters=1000)
    svm_primal.fit(X_linear, y_linear)
    
    svm_dual = SVMDual(C=1.0, kernel='linear')
    svm_dual.fit(X_linear, y_linear)
    
    print(f"Primal SVM Accuracy: {svm_primal.score(X_linear, y_linear):.4f}")
    print(f"Dual SVM Accuracy: {svm_dual.score(X_linear, y_linear):.4f}")
    print(f"Primal Support Vectors: {svm_primal.n_support_}")
    print(f"Dual Support Vectors: {svm_dual.n_support_}")
    
    # Visualize primal formulation
    plot_decision_boundary(X_linear, y_linear, svm_primal,
                          "Linear SVM - Primal Formulation (Linearly Separable)",
                          svm_primal.support_vectors_, "svm_linear_primal.png")
    
    # Visualize dual formulation
    plot_decision_boundary(X_linear, y_linear, svm_dual,
                          "Linear SVM - Dual Formulation (Linearly Separable)",
                          svm_dual.support_vectors_, "svm_linear_dual.png")
    
    # 2. RBF SVM on Non-linear Data (Moons)
    print("\n2. RBF SVM - NON-LINEAR DATA (MOONS)")
    print("-" * 50)
    
    # Generate moon-shaped data
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    # Train RBF SVM
    svm_rbf = SVMDual(C=1.0, kernel='rbf', gamma='scale')
    svm_rbf.fit(X_moons, y_moons)
    
    print(f"RBF SVM Accuracy: {svm_rbf.score(X_moons, y_moons):.4f}")
    print(f"Support Vectors: {svm_rbf.n_support_}")
    
    # Visualize RBF SVM
    plot_decision_boundary(X_moons, y_moons, svm_rbf,
                          "RBF SVM - Non-linear Data (Moons)",
                          svm_rbf.support_vectors_, "svm_rbf_moons.png")
    
    # 3. RBF SVM on Circular Data
    print("\n3. RBF SVM - CIRCULAR DATA")
    print("-" * 50)
    
    # Generate circular data
    X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.6, random_state=42)
    
    # Train RBF SVM
    svm_circles = SVMDual(C=1.0, kernel='rbf', gamma='scale')
    svm_circles.fit(X_circles, y_circles)
    
    print(f"RBF SVM Accuracy: {svm_circles.score(X_circles, y_circles):.4f}")
    print(f"Support Vectors: {svm_circles.n_support_}")
    
    # Visualize circular data
    plot_decision_boundary(X_circles, y_circles, svm_circles,
                          "RBF SVM - Circular Data",
                          svm_circles.support_vectors_, "svm_rbf_circles.png")
    
    # 4. Kernel Comparison
    print("\n4. KERNEL COMPARISON")
    print("-" * 50)
    
    print("Comparing Linear, Polynomial, and RBF kernels on moon data:")
    plot_kernel_comparison(X_moons, y_moons, ['linear', 'poly', 'rbf'],
                          "SVM Kernel Comparison (Moon Dataset)")
    
    # 5. Regularization Analysis
    print("\n5. REGULARIZATION PARAMETER ANALYSIS")
    print("-" * 50)
    
    C_range = [0.01, 0.1, 1, 10, 100, 1000]
    print("Analyzing effect of regularization parameter C...")
    
    train_scores, val_scores, n_svs = plot_regularization_analysis(
        X_moons, y_moons, C_range, "SVM Regularization Analysis (Moon Dataset)"
    )
    
    # Print regularization results
    print("\nRegularization Results:")
    print("C\tTrain Acc\tVal Acc\t\tSupport Vectors")
    print("-" * 50)
    for i, C in enumerate(C_range):
        print(f"{C}\t{train_scores[i]:.3f}\t\t{val_scores[i]:.3f}\t\t{n_svs[i]}")
    
    # 6. Real Dataset - Breast Cancer
    print("\n6. REAL DATASET - BREAST CANCER")
    print("-" * 50)
    
    # Load breast cancer dataset
    cancer = load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target
    
    # Use first 2 features for visualization
    X_cancer_2d = X_cancer[:, :2]
    
    # Standardize features
    scaler = StandardScaler()
    X_cancer_2d_scaled = scaler.fit_transform(X_cancer_2d)
    
    # Split data
    X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
        X_cancer_2d_scaled, y_cancer, test_size=0.3, random_state=42, stratify=y_cancer
    )
    
    # Train SVM
    svm_cancer = SVMDual(C=1.0, kernel='rbf', gamma='scale')
    svm_cancer.fit(X_train_cancer, y_train_cancer)
    
    print(f"Breast Cancer SVM Accuracy: {svm_cancer.score(X_test_cancer, y_test_cancer):.4f}")
    print(f"Support Vectors: {svm_cancer.n_support_}")
    
    # Visualize on full training data
    plot_decision_boundary(X_cancer_2d_scaled, y_cancer, svm_cancer,
                          "RBF SVM - Breast Cancer Dataset (2D projection)",
                          svm_cancer.support_vectors_, "svm_breast_cancer.png")
    
    # 7. Comparison with Scikit-learn
    print("\n7. COMPARISON WITH SCIKIT-LEARN")
    print("-" * 50)
    
    # Compare on moon dataset
    X_train_moons, X_test_moons, y_train_moons, y_test_moons = train_test_split(
        X_moons, y_moons, test_size=0.3, random_state=42
    )
    
    comparison_results = compare_with_sklearn(
        X_train_moons, y_train_moons, X_test_moons, y_test_moons, kernel='rbf'
    )
    
    # 8. Training History Visualization (Primal)
    print("\n8. TRAINING CONVERGENCE ANALYSIS")
    print("-" * 50)
    
    if svm_primal.training_history_:
        iterations = [h['iteration'] for h in svm_primal.training_history_]
        losses = [h['loss'] for h in svm_primal.training_history_]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, losses, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Hinge Loss')
        plt.title('SVM Training Convergence (Primal Formulation)')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('plots/svm_training_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training converged in {len(iterations)} iterations")
        print(f"Final loss: {losses[-1]:.6f}")
    
    print("\n" + "="*70)
    print("SVM EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Generated visualizations:")
    print("- svm_linear_primal.png")
    print("- svm_linear_dual.png")
    print("- svm_rbf_moons.png")
    print("- svm_rbf_circles.png")
    print("- svm_kernel_comparison.png")
    print("- svm_regularization_analysis.png")
    print("- svm_breast_cancer.png")
    print("- svm_training_convergence.png")

if __name__ == "__main__":
    main() 