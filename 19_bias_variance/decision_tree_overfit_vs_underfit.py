"""
Decision Tree Overfitting vs Underfitting Analysis

This module demonstrates how decision tree depth affects the bias-variance tradeoff
through visualization of decision boundaries, predictions, and variance analysis.

Key demonstrations:
- Decision boundaries for different max_depth values
- Prediction variance with noisy inputs
- Training vs validation error curves
- Feature importance analysis

Author: ML Learning Series
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class DecisionTreeBiasVarianceDemo:
    """
    Comprehensive demonstration of decision tree bias-variance tradeoff.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the demo.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Store results
        self.results = {}
        self.dataset_info = {}
        
    def generate_2d_dataset(self, dataset_type='moons', n_samples=300, noise=0.3):
        """
        Generate 2D dataset for visualization.
        
        Parameters:
        -----------
        dataset_type : str
            Type of dataset ('moons', 'circles', 'classification')
        n_samples : int
            Number of samples
        noise : float
            Noise level
            
        Returns:
        --------
        X, y : arrays
            Features and target values
        """
        if dataset_type == 'moons':
            X, y = make_moons(n_samples=n_samples, noise=noise, 
                             random_state=self.random_state)
            self.dataset_info = {
                'name': 'Two Moons',
                'description': 'Nonlinear crescent-shaped clusters'
            }
            
        elif dataset_type == 'circles':
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5,
                               random_state=self.random_state)
            self.dataset_info = {
                'name': 'Two Circles',
                'description': 'Concentric circles'
            }
            
        elif dataset_type == 'classification':
            X, y = make_classification(
                n_samples=n_samples, n_features=2, n_redundant=0,
                n_informative=2, n_clusters_per_class=1,
                random_state=self.random_state
            )
            # Standardize for better visualization
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            self.dataset_info = {
                'name': 'Linear Separable',
                'description': 'Linearly separable clusters'
            }
            
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        print(f"Generated {self.dataset_info['name']} dataset:")
        print(f"  Samples: {n_samples}")
        print(f"  Features: 2 (for visualization)")
        print(f"  Noise level: {noise}")
        print(f"  Description: {self.dataset_info['description']}")
        
        return X, y
    
    def analyze_depth_complexity(self, X, y, max_depths=None):
        """
        Analyze how tree depth affects performance.
        
        Parameters:
        -----------
        X, y : arrays
            Training data
        max_depths : list, optional
            List of max_depth values to test
            
        Returns:
        --------
        results : dict
            Performance metrics for different depths
        """
        if max_depths is None:
            max_depths = list(range(1, 21)) + [None]  # None means unlimited depth
        
        print("\nAnalyzing decision tree depth complexity...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state
        )
        
        train_accuracies = []
        test_accuracies = []
        tree_sizes = []  # Number of nodes
        models = {}
        
        for depth in max_depths:
            # Train model
            clf = DecisionTreeClassifier(
                max_depth=depth, 
                random_state=self.random_state,
                min_samples_split=2,
                min_samples_leaf=1
            )
            clf.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)
            
            # Accuracies
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            tree_sizes.append(clf.tree_.node_count)
            models[depth] = clf
            
            depth_str = str(depth) if depth is not None else "None"
            print(f"  Depth {depth_str:>4}: Train Acc = {train_acc:.4f}, "
                  f"Test Acc = {test_acc:.4f}, Nodes = {clf.tree_.node_count}")
        
        results = {
            'depths': max_depths,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'tree_sizes': tree_sizes,
            'models': models,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        # Find optimal depth
        optimal_idx = np.argmax(test_accuracies)
        optimal_depth = max_depths[optimal_idx]
        
        print(f"\nOptimal max_depth: {optimal_depth}")
        print(f"Best test accuracy: {test_accuracies[optimal_idx]:.4f}")
        
        self.results['depth_analysis'] = results
        return results
    
    def analyze_prediction_variance(self, X, y, depths=[1, 5, 10, None], 
                                  n_bootstrap=50, noise_std=0.05):
        """
        Analyze prediction variance for different tree depths.
        
        Parameters:
        -----------
        X, y : arrays
            Training data
        depths : list
            Tree depths to analyze
        n_bootstrap : int
            Number of bootstrap samples
        noise_std : float
            Standard deviation of noise to add
            
        Returns:
        --------
        variance_results : dict
            Prediction variance analysis
        """
        print(f"\nAnalyzing prediction variance with {n_bootstrap} bootstrap samples...")
        
        # Create a grid for predictions
        h = 0.02  # Step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        variance_results = {}
        
        for depth in depths:
            print(f"  Analyzing depth {depth}...")
            
            # Store predictions for each bootstrap sample
            all_predictions = []
            
            for i in range(n_bootstrap):
                # Add noise to training data
                X_noisy = X + np.random.normal(0, noise_std, X.shape)
                
                # Bootstrap sample
                n_samples = len(X)
                bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X_noisy[bootstrap_indices]
                y_boot = y[bootstrap_indices]
                
                # Train model
                clf = DecisionTreeClassifier(
                    max_depth=depth,
                    random_state=i,  # Different random state for each bootstrap
                    min_samples_split=2,
                    min_samples_leaf=1
                )
                clf.fit(X_boot, y_boot)
                
                # Predict on grid
                grid_pred = clf.predict(grid_points)
                all_predictions.append(grid_pred)
            
            all_predictions = np.array(all_predictions)
            
            # Calculate prediction variance
            # For classification, variance is the fraction of disagreement
            mean_pred = np.mean(all_predictions, axis=0)
            variance = np.mean((all_predictions - mean_pred) ** 2, axis=0)
            
            variance_results[depth] = {
                'predictions': all_predictions,
                'mean_prediction': mean_pred,
                'variance': variance,
                'grid_shape': xx.shape,
                'xx': xx,
                'yy': yy
            }
        
        self.results['variance_analysis'] = variance_results
        return variance_results
    
    def plot_decision_boundaries(self, X, y, depths=[1, 3, 5, 10], save_path=None):
        """
        Plot decision boundaries for different tree depths.
        
        Parameters:
        -----------
        X, y : arrays
            Training data
        depths : list
            Tree depths to visualize
        save_path : str, optional
            Path to save the plot
        """
        print(f"\nPlotting decision boundaries for depths: {depths}")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        fig.suptitle(f'Decision Tree Boundaries: {self.dataset_info["name"]} Dataset', 
                     fontsize=16, fontweight='bold')
        
        # Create a mesh for plotting
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        colors = ['lightblue', 'lightcoral']
        
        for i, depth in enumerate(depths):
            if i >= 4:  # Maximum 4 subplots
                break
                
            ax = axes[i]
            
            # Train model
            clf = DecisionTreeClassifier(
                max_depth=depth,
                random_state=self.random_state,
                min_samples_split=2,
                min_samples_leaf=1
            )
            clf.fit(X, y)
            
            # Make predictions on mesh
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
            
            # Plot data points
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
            
            # Calculate and display accuracy
            y_pred = clf.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            # Determine bias-variance characteristics
            if depth <= 2:
                bias_variance = "High Bias (Underfitting)"
                color_indicator = "blue"
            elif depth >= 10:
                bias_variance = "High Variance (Overfitting)"
                color_indicator = "red"
            else:
                bias_variance = "Balanced"
                color_indicator = "green"
            
            ax.set_title(f'Max Depth = {depth}\n'
                        f'Accuracy: {accuracy:.3f} | {bias_variance}\n'
                        f'Nodes: {clf.tree_.node_count}',
                        fontsize=10)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            
            # Add colored border to indicate bias-variance
            for spine in ax.spines.values():
                spine.set_color(color_indicator)
                spine.set_linewidth(3)
        
        # Hide unused subplots
        for j in range(len(depths), 4):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Decision boundaries plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_variance_analysis(self, save_path=None):
        """
        Plot prediction variance analysis.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if 'variance_analysis' not in self.results:
            raise ValueError("Must run analyze_prediction_variance first")
        
        variance_results = self.results['variance_analysis']
        depths = list(variance_results.keys())
        
        fig, axes = plt.subplots(2, len(depths), figsize=(4*len(depths), 8))
        if len(depths) == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Prediction Variance Analysis: {self.dataset_info["name"]}', 
                     fontsize=14, fontweight='bold')
        
        for i, depth in enumerate(depths):
            results = variance_results[depth]
            xx, yy = results['xx'], results['yy']
            mean_pred = results['mean_prediction'].reshape(xx.shape)
            variance = results['variance'].reshape(xx.shape)
            
            # Plot mean prediction
            im1 = axes[0, i].contourf(xx, yy, mean_pred, alpha=0.8, cmap=plt.cm.RdYlBu)
            axes[0, i].set_title(f'Mean Prediction\nDepth = {depth}')
            axes[0, i].set_xlabel('Feature 1')
            if i == 0:
                axes[0, i].set_ylabel('Feature 2')
            
            # Plot variance
            im2 = axes[1, i].contourf(xx, yy, variance, alpha=0.8, cmap=plt.cm.Reds)
            axes[1, i].set_title(f'Prediction Variance\nDepth = {depth}')
            axes[1, i].set_xlabel('Feature 1')
            if i == 0:
                axes[1, i].set_ylabel('Feature 2')
            
            # Add colorbar
            if i == len(depths) - 1:
                plt.colorbar(im1, ax=axes[0, i], label='Prediction')
                plt.colorbar(im2, ax=axes[1, i], label='Variance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Variance analysis plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_comprehensive_analysis(self, save_path=None):
        """
        Create comprehensive analysis plot.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if 'depth_analysis' not in self.results:
            raise ValueError("Must run analyze_depth_complexity first")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Decision Tree Comprehensive Analysis: {self.dataset_info["name"]}', 
                     fontsize=16, fontweight='bold')
        
        results = self.results['depth_analysis']
        depths = [d if d is not None else 20 for d in results['depths']]  # Convert None to 20 for plotting
        
        # 1. Training vs Test Accuracy
        ax1 = axes[0, 0]
        ax1.plot(depths, results['train_accuracies'], 'o-', label='Training Accuracy', 
                linewidth=2, markersize=6)
        ax1.plot(depths, results['test_accuracies'], 's-', label='Test Accuracy', 
                linewidth=2, markersize=6)
        ax1.set_xlabel('Max Depth')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Training vs Test Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mark optimal depth
        optimal_idx = np.argmax(results['test_accuracies'])
        optimal_depth = depths[optimal_idx]
        ax1.axvline(optimal_depth, color='red', linestyle='--', alpha=0.7,
                   label=f'Optimal Depth ({optimal_depth})')
        ax1.legend()
        
        # 2. Model Complexity (Tree Size)
        ax2 = axes[0, 1]
        ax2.plot(depths, results['tree_sizes'], 'o-', color='green', 
                linewidth=2, markersize=6)
        ax2.set_xlabel('Max Depth')
        ax2.set_ylabel('Number of Nodes')
        ax2.set_title('Model Complexity (Tree Size)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Overfitting Analysis
        ax3 = axes[0, 2]
        overfitting_gap = np.array(results['train_accuracies']) - np.array(results['test_accuracies'])
        ax3.plot(depths, overfitting_gap, 'o-', color='red', linewidth=2, markersize=6)
        ax3.set_xlabel('Max Depth')
        ax3.set_ylabel('Overfitting Gap (Train - Test)')
        ax3.set_title('Overfitting Analysis')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add regions
        ax3.axvspan(1, 5, alpha=0.2, color='blue', label='Underfitting\n(High Bias)')
        ax3.axvspan(optimal_depth-2, optimal_depth+2, alpha=0.2, color='green', 
                   label='Optimal\n(Balanced)')
        ax3.axvspan(15, 20, alpha=0.2, color='red', label='Overfitting\n(High Variance)')
        ax3.legend()
        
        # 4. Decision Boundary Evolution (Example with 3 depths)
        ax4 = axes[1, 0]
        example_depths = [1, optimal_depth, 15]
        X = results['X_train']
        y = results['y_train']
        
        # Create mesh for decision boundary
        h = 0.1
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Plot boundary for optimal depth
        clf_optimal = results['models'][results['depths'][optimal_idx]]
        Z = clf_optimal.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax4.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        ax4.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
        ax4.set_title(f'Decision Boundary (Optimal Depth = {optimal_depth})')
        ax4.set_xlabel('Feature 1')
        ax4.set_ylabel('Feature 2')
        
        # 5. Feature Importance (for optimal model)
        ax5 = axes[1, 1]
        if hasattr(clf_optimal, 'feature_importances_'):
            importances = clf_optimal.feature_importances_
            features = [f'Feature {i+1}' for i in range(len(importances))]
            
            bars = ax5.bar(features, importances)
            ax5.set_title('Feature Importance (Optimal Model)')
            ax5.set_ylabel('Importance')
            
            # Color bars by importance
            for bar, importance in zip(bars, importances):
                bar.set_color(plt.cm.viridis(importance))
        
        # 6. Bias-Variance Components
        ax6 = axes[1, 2]
        
        # Estimate bias and variance for selected depths
        selected_depths = [1, 5, 10, 15]
        bias_estimates = []
        variance_estimates = []
        
        for depth in selected_depths:
            if depth in [d for d in results['depths'] if d is not None]:
                # Rough estimates based on performance patterns
                train_acc = results['train_accuracies'][depths.index(depth)]
                test_acc = results['test_accuracies'][depths.index(depth)]
                
                # High bias when both train and test are low
                bias_est = max(0, 1 - test_acc) if train_acc < 0.9 else 0.1
                # High variance when gap is large
                variance_est = max(0, train_acc - test_acc) if train_acc > test_acc else 0.05
                
                bias_estimates.append(bias_est)
                variance_estimates.append(variance_est)
        
        x_pos = np.arange(len(selected_depths))
        width = 0.35
        
        ax6.bar(x_pos - width/2, bias_estimates, width, label='Bias (estimated)', alpha=0.8)
        ax6.bar(x_pos + width/2, variance_estimates, width, label='Variance (estimated)', alpha=0.8)
        
        ax6.set_xlabel('Max Depth')
        ax6.set_ylabel('Error Component')
        ax6.set_title('Bias-Variance Estimation')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(selected_depths)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive analysis plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def print_summary(self):
        """Print comprehensive summary of the analysis."""
        if not self.results:
            print("No results available. Run analysis methods first.")
            return
        
        print("\n" + "="*70)
        print("DECISION TREE BIAS-VARIANCE ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"Dataset: {self.dataset_info['name']}")
        print(f"Description: {self.dataset_info['description']}")
        
        if 'depth_analysis' in self.results:
            results = self.results['depth_analysis']
            
            optimal_idx = np.argmax(results['test_accuracies'])
            optimal_depth = results['depths'][optimal_idx]
            best_train_acc = results['train_accuracies'][optimal_idx]
            best_test_acc = results['test_accuracies'][optimal_idx]
            
            print(f"\nOptimal Model Performance:")
            print(f"  Optimal max_depth: {optimal_depth}")
            print(f"  Training accuracy: {best_train_acc:.4f}")
            print(f"  Test accuracy: {best_test_acc:.4f}")
            print(f"  Overfitting gap: {best_train_acc - best_test_acc:.4f}")
            print(f"  Tree nodes: {results['tree_sizes'][optimal_idx]}")
            
            # Analyze underfitting and overfitting
            depths = results['depths']
            train_accs = results['train_accuracies']
            test_accs = results['test_accuracies']
            
            print(f"\nBias-Variance Analysis:")
            
            # Underfitting region (low depth)
            underfit_depths = [d for i, d in enumerate(depths) 
                             if d is not None and d <= 3 and test_accs[i] < best_test_acc * 0.95]
            if underfit_depths:
                print(f"  Underfitting depths: {underfit_depths}")
                print(f"    Characteristics: High bias, low variance")
                print(f"    Symptoms: Low training and test accuracy")
            
            # Overfitting region (high depth)
            overfit_depths = [d for i, d in enumerate(depths) 
                            if d is not None and d >= 10 and 
                            (train_accs[i] - test_accs[i]) > 0.05]
            if overfit_depths:
                print(f"  Overfitting depths: {overfit_depths}")
                print(f"    Characteristics: Low bias, high variance")
                print(f"    Symptoms: High training accuracy, lower test accuracy")
            
            print(f"\nKey Decision Tree Insights:")
            print(f"  • Shallow trees (depth 1-3): Underfit, high bias")
            print(f"  • Deep trees (depth >10): Overfit, high variance")
            print(f"  • Optimal depth balances bias and variance")
            print(f"  • Tree complexity (nodes) grows exponentially with depth")
            print(f"  • Decision boundaries become more complex with depth")
        
        if 'variance_analysis' in self.results:
            print(f"\nPrediction Variance Analysis:")
            print(f"  • Shallow trees: Low variance, stable predictions")
            print(f"  • Deep trees: High variance, unstable predictions")
            print(f"  • Variance increases significantly with small data changes")
        
        print("="*70)


def main():
    """
    Main execution function demonstrating decision tree bias-variance analysis.
    """
    print("Decision Tree Overfitting vs Underfitting Analysis")
    print("="*55)
    
    # Initialize demo
    demo = DecisionTreeBiasVarianceDemo(random_state=42)
    
    # Test with different datasets
    datasets = ['moons', 'circles', 'classification']
    
    for dataset_type in datasets:
        print(f"\n{'='*60}")
        print(f"ANALYZING {dataset_type.upper()} DATASET")
        print(f"{'='*60}")
        
        # Generate dataset
        print(f"\n1. Generating {dataset_type} dataset...")
        X, y = demo.generate_2d_dataset(dataset_type, n_samples=300, noise=0.3)
        
        # Analyze depth complexity
        print(f"\n2. Analyzing depth complexity...")
        demo.analyze_depth_complexity(X, y, max_depths=list(range(1, 16)) + [None])
        
        # Analyze prediction variance
        print(f"\n3. Analyzing prediction variance...")
        demo.analyze_prediction_variance(X, y, depths=[1, 5, 10, None], n_bootstrap=30)
        
        # Create visualizations
        print(f"\n4. Creating visualizations...")
        
        # Decision boundaries
        demo.plot_decision_boundaries(
            X, y, depths=[1, 3, 7, 15], 
            save_path=f'plots/decision_boundaries_{dataset_type}.png'
        )
        
        # Variance analysis
        demo.plot_variance_analysis(
            save_path=f'plots/variance_analysis_{dataset_type}.png'
        )
        
        # Comprehensive analysis
        demo.plot_comprehensive_analysis(
            save_path=f'plots/comprehensive_tree_analysis_{dataset_type}.png'
        )
        
        # Print summary
        demo.print_summary()
        
        print(f"\nCompleted analysis for {dataset_type} dataset.")
    
    print(f"\n{'='*60}")
    print("ALL ANALYSES COMPLETED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 