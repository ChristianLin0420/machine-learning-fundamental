"""
Impact of Feature Scaling on Machine Learning Models
===================================================

This module analyzes how different feature scaling methods affect
the performance and behavior of various machine learning algorithms,
specifically focusing on:

- K-Means Clustering: Convergence speed and cluster quality
- Principal Component Analysis (PCA): Component interpretation and variance
- Support Vector Machines (SVM): Decision boundaries and classification accuracy

Analysis includes:
- Performance metrics comparison
- Convergence analysis
- Decision boundary visualization
- Feature importance changes
- Computational efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from sklearn.datasets import make_blobs, make_classification, load_wine, load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, silhouette_score, adjusted_rand_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scaling_from_scratch import (
    StandardScaler, MinMaxScaler, RobustScaler, L2Normalizer,
    load_sample_datasets
)
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelImpactAnalyzer:
    """
    Comprehensive analyzer for the impact of feature scaling on ML models.
    
    This class evaluates how different scaling methods affect:
    - Clustering algorithms (K-Means)
    - Dimensionality reduction (PCA)
    - Classification algorithms (SVM)
    """
    
    def __init__(self):
        """Initialize the model impact analyzer."""
        self.scalers = {
            'No Scaling': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'L2Normalizer': L2Normalizer()
        }
        self.results = {}
    
    def analyze_kmeans_impact(self, X, y_true=None, n_clusters=3, save_path=None):
        """
        Analyze the impact of scaling on K-Means clustering.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y_true : array-like, optional
            True cluster labels
        n_clusters : int
            Number of clusters
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        dict : Analysis results
        """
        print("Analyzing K-Means impact...")
        
        results = {}
        
        # Test each scaling method
        for scaler_name, scaler in self.scalers.items():
            print(f"  Testing {scaler_name}...")
            
            # Apply scaling
            if scaler is None:
                X_scaled = X
            else:
                X_scaled = scaler.fit_transform(X)
            
            # Track convergence
            inertias = []
            times = []
            
            # Run K-Means with convergence tracking
            start_time = time.time()
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
            
            # Manual iteration tracking for convergence analysis
            kmeans_manual = KMeans(n_clusters=n_clusters, random_state=42, 
                                 n_init=1, max_iter=1, init='k-means++')
            
            # Initialize centroids
            kmeans_manual.fit(X_scaled)
            current_centroids = kmeans_manual.cluster_centers_.copy()
            
            convergence_inertias = []
            convergence_times = []
            
            for iteration in range(50):  # Max 50 iterations for tracking
                iter_start = time.time()
                
                # One iteration
                kmeans_iter = KMeans(n_clusters=n_clusters, random_state=42, 
                                   n_init=1, max_iter=1, init=current_centroids)
                kmeans_iter.fit(X_scaled)
                
                convergence_inertias.append(kmeans_iter.inertia_)
                convergence_times.append(time.time() - iter_start)
                
                # Check convergence
                centroid_shift = np.sum(np.sqrt(np.sum((current_centroids - kmeans_iter.cluster_centers_)**2, axis=1)))
                current_centroids = kmeans_iter.cluster_centers_.copy()
                
                if centroid_shift < 1e-4:
                    break
            
            # Final clustering
            labels = kmeans.fit_predict(X_scaled)
            total_time = time.time() - start_time
            
            # Compute metrics
            inertia = kmeans.inertia_
            
            # Silhouette score
            try:
                silhouette = silhouette_score(X_scaled, labels)
            except:
                silhouette = -1
            
            # ARI if true labels available
            ari = adjusted_rand_score(y_true, labels) if y_true is not None else None
            
            results[scaler_name] = {
                'labels': labels,
                'inertia': inertia,
                'silhouette': silhouette,
                'ari': ari,
                'total_time': total_time,
                'n_iterations': len(convergence_inertias),
                'convergence_inertias': convergence_inertias,
                'convergence_times': convergence_times,
                'centroids': kmeans.cluster_centers_
            }
        
        # Store results
        self.results['kmeans'] = results
        
        # Create visualization
        self._plot_kmeans_analysis(X, results, save_path)
        
        return results
    
    def _plot_kmeans_analysis(self, X, results, save_path=None):
        """Plot K-Means analysis results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Clustering results (first 2 features)
        ax1 = axes[0, 0]
        if X.shape[1] >= 2:
            for i, (scaler_name, result) in enumerate(results.items()):
                if i >= 3:  # Limit to first 3 for readability
                    break
                
                # Apply scaling for visualization
                scaler = self.scalers[scaler_name]
                X_scaled = X if scaler is None else scaler.fit_transform(X)
                
                labels = result['labels']
                scatter = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                    c=labels, alpha=0.6, s=30, 
                                    label=f'{scaler_name} (Sil: {result["silhouette"]:.3f})')
            
            ax1.set_xlabel('Feature 1 (scaled)')
            ax1.set_ylabel('Feature 2 (scaled)')
            ax1.set_title('K-Means Clustering Results')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Need ≥2 features for 2D plot', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Plot 2: Inertia comparison
        ax2 = axes[0, 1]
        scaler_names = list(results.keys())
        inertias = [results[name]['inertia'] for name in scaler_names]
        
        bars = ax2.bar(scaler_names, inertias, alpha=0.8)
        ax2.set_ylabel('Inertia (WCSS)')
        ax2.set_title('Final Inertia by Scaling Method')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, inertia in zip(bars, inertias):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{inertia:.1f}', ha='center', va='bottom')
        
        # Plot 3: Silhouette scores
        ax3 = axes[0, 2]
        silhouettes = [results[name]['silhouette'] for name in scaler_names]
        
        bars = ax3.bar(scaler_names, silhouettes, alpha=0.8, color='orange')
        ax3.set_ylabel('Silhouette Score')
        ax3.set_title('Silhouette Score by Scaling Method')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, sil in zip(bars, silhouettes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{sil:.3f}', ha='center', va='bottom')
        
        # Plot 4: Convergence analysis
        ax4 = axes[1, 0]
        for scaler_name, result in results.items():
            convergence_inertias = result['convergence_inertias']
            if convergence_inertias:
                ax4.plot(convergence_inertias, label=scaler_name, marker='o', markersize=4)
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Inertia')
        ax4.set_title('Convergence Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Iterations to convergence
        ax5 = axes[1, 1]
        iterations = [results[name]['n_iterations'] for name in scaler_names]
        
        bars = ax5.bar(scaler_names, iterations, alpha=0.8, color='green')
        ax5.set_ylabel('Iterations to Convergence')
        ax5.set_title('Convergence Speed by Scaling Method')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, iter_count in zip(bars, iterations):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{iter_count}', ha='center', va='bottom')
        
        # Plot 6: Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary table
        summary_data = []
        for scaler_name in scaler_names:
            result = results[scaler_name]
            summary_data.append([
                scaler_name,
                f"{result['inertia']:.1f}",
                f"{result['silhouette']:.3f}",
                f"{result['n_iterations']}",
                f"{result['total_time']:.3f}s"
            ])
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Method', 'Inertia', 'Silhouette', 'Iterations', 'Time'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax6.set_title('K-Means Performance Summary')
        
        plt.suptitle('K-Means Clustering: Impact of Feature Scaling', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"K-Means analysis saved to {save_path}")
        
        plt.show()
    
    def analyze_pca_impact(self, X, save_path=None):
        """
        Analyze the impact of scaling on PCA.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        dict : Analysis results
        """
        print("Analyzing PCA impact...")
        
        results = {}
        
        # Test each scaling method
        for scaler_name, scaler in self.scalers.items():
            print(f"  Testing {scaler_name}...")
            
            # Apply scaling
            if scaler is None:
                X_scaled = X
            else:
                X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            n_components = min(4, X_scaled.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            # Compute metrics
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            components = pca.components_
            
            results[scaler_name] = {
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance': cumulative_variance,
                'components': components,
                'transformed_data': X_pca,
                'pca_object': pca
            }
        
        # Store results
        self.results['pca'] = results
        
        # Create visualization
        self._plot_pca_analysis(X, results, save_path)
        
        return results
    
    def _plot_pca_analysis(self, X, results, save_path=None):
        """Plot PCA analysis results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Explained variance ratio
        ax1 = axes[0, 0]
        for scaler_name, result in results.items():
            variance_ratio = result['explained_variance_ratio']
            ax1.plot(range(1, len(variance_ratio) + 1), variance_ratio, 
                    marker='o', label=scaler_name, markersize=6)
        
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Explained Variance by Component')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative explained variance
        ax2 = axes[0, 1]
        for scaler_name, result in results.items():
            cumulative_var = result['cumulative_variance']
            ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                    marker='s', label=scaler_name, markersize=6)
        
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Variance Explained')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
        
        # Plot 3: First two principal components
        ax3 = axes[0, 2]
        for i, (scaler_name, result) in enumerate(results.items()):
            if i >= 3:  # Limit for readability
                break
            
            X_pca = result['transformed_data']
            if X_pca.shape[1] >= 2:
                ax3.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30, 
                          label=f'{scaler_name}')
        
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_title('First Two Principal Components')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Component loadings (first component)
        ax4 = axes[1, 0]
        feature_names = [f'F{i}' for i in range(X.shape[1])]
        
        for i, (scaler_name, result) in enumerate(results.items()):
            if i >= 3:  # Limit for readability
                break
            
            loadings = result['components'][0]  # First component
            ax4.bar(np.arange(len(loadings)) + i*0.2, loadings, 
                   width=0.2, alpha=0.8, label=scaler_name)
        
        ax4.set_xlabel('Features')
        ax4.set_ylabel('Loading')
        ax4.set_title('First Principal Component Loadings')
        ax4.set_xticks(range(len(feature_names)))
        ax4.set_xticklabels(feature_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Variance explained comparison
        ax5 = axes[1, 1]
        scaler_names = list(results.keys())
        first_pc_variance = [results[name]['explained_variance_ratio'][0] 
                           for name in scaler_names]
        
        bars = ax5.bar(scaler_names, first_pc_variance, alpha=0.8)
        ax5.set_ylabel('Variance Explained by PC1')
        ax5.set_title('First PC Variance by Scaling Method')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, var in zip(bars, first_pc_variance):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{var:.3f}', ha='center', va='bottom')
        
        # Plot 6: Summary table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary table
        summary_data = []
        for scaler_name in scaler_names:
            result = results[scaler_name]
            var_ratio = result['explained_variance_ratio']
            cum_var = result['cumulative_variance']
            
            summary_data.append([
                scaler_name,
                f"{var_ratio[0]:.3f}",
                f"{cum_var[1] if len(cum_var) > 1 else cum_var[0]:.3f}",
                f"{cum_var[-1]:.3f}"
            ])
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Method', 'PC1 Var', '2PC Cum', 'Total Cum'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax6.set_title('PCA Variance Summary')
        
        plt.suptitle('PCA Analysis: Impact of Feature Scaling', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA analysis saved to {save_path}")
        
        plt.show()
    
    def analyze_svm_impact(self, X, y, save_path=None):
        """
        Analyze the impact of scaling on SVM classification.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            Target labels
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        dict : Analysis results
        """
        print("Analyzing SVM impact...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        results = {}
        
        # Test each scaling method
        for scaler_name, scaler in self.scalers.items():
            print(f"  Testing {scaler_name}...")
            
            # Apply scaling
            if scaler is None:
                X_train_scaled = X_train
                X_test_scaled = X_test
            else:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            
            # Train SVM
            start_time = time.time()
            svm = SVC(kernel='rbf', random_state=42, probability=True)
            svm.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time
            
            # Predictions
            y_pred = svm.predict(X_test_scaled)
            y_proba = svm.predict_proba(X_test_scaled) if hasattr(svm, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            results[scaler_name] = {
                'svm_model': svm,
                'accuracy': accuracy,
                'train_time': train_time,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'n_support_vectors': np.sum(svm.n_support_),
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled
            }
        
        # Store results
        self.results['svm'] = results
        
        # Create visualization
        self._plot_svm_analysis(X_train, X_test, y_train, y_test, results, save_path)
        
        return results
    
    def _plot_svm_analysis(self, X_train, X_test, y_train, y_test, results, save_path=None):
        """Plot SVM analysis results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Decision boundaries (if 2D)
        ax1 = axes[0, 0]
        if X_train.shape[1] >= 2:
            # Use first scaling method for boundary visualization
            first_scaler = list(results.keys())[0]
            result = results[first_scaler]
            
            X_train_scaled = result['X_train_scaled']
            svm_model = result['svm_model']
            
            # Create mesh
            h = 0.02
            x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
            y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            # Predictions on mesh
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            if X_train_scaled.shape[1] > 2:
                # Pad with mean values for additional features
                padding = np.mean(X_train_scaled[:, 2:], axis=0)
                mesh_points = np.hstack([mesh_points, 
                                       np.tile(padding, (mesh_points.shape[0], 1))])
            
            Z = svm_model.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundary
            ax1.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
            scatter = ax1.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], 
                                c=y_train, cmap=plt.cm.RdBu, edgecolors='black')
            ax1.set_xlabel('Feature 1 (scaled)')
            ax1.set_ylabel('Feature 2 (scaled)')
            ax1.set_title(f'SVM Decision Boundary ({first_scaler})')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Need ≥2 features for boundary plot', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        # Plot 2: Accuracy comparison
        ax2 = axes[0, 1]
        scaler_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in scaler_names]
        
        bars = ax2.bar(scaler_names, accuracies, alpha=0.8)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Classification Accuracy by Scaling Method')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([min(accuracies) - 0.05, 1.0])
        
        # Add value annotations
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Plot 3: Number of support vectors
        ax3 = axes[0, 2]
        n_support = [results[name]['n_support_vectors'] for name in scaler_names]
        
        bars = ax3.bar(scaler_names, n_support, alpha=0.8, color='orange')
        ax3.set_ylabel('Number of Support Vectors')
        ax3.set_title('Support Vectors by Scaling Method')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, n_sv in zip(bars, n_support):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{n_sv}', ha='center', va='bottom')
        
        # Plot 4: Training time comparison
        ax4 = axes[1, 0]
        train_times = [results[name]['train_time'] for name in scaler_names]
        
        bars = ax4.bar(scaler_names, train_times, alpha=0.8, color='green')
        ax4.set_ylabel('Training Time (seconds)')
        ax4.set_title('Training Time by Scaling Method')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, time_val in zip(bars, train_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # Plot 5: Confusion matrix for best method
        ax5 = axes[1, 1]
        best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_result = results[best_method]
        
        cm = confusion_matrix(y_test, best_result['y_pred'])
        im = ax5.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax5.set_title(f'Confusion Matrix\n({best_method})')
        
        # Add colorbar
        plt.colorbar(im, ax=ax5)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax5.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        ax5.set_ylabel('True label')
        ax5.set_xlabel('Predicted label')
        
        # Plot 6: Summary table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary table
        summary_data = []
        for scaler_name in scaler_names:
            result = results[scaler_name]
            summary_data.append([
                scaler_name,
                f"{result['accuracy']:.3f}",
                f"{result['n_support_vectors']}",
                f"{result['train_time']:.3f}s"
            ])
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Method', 'Accuracy', 'N Support', 'Time'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax6.set_title('SVM Performance Summary')
        
        plt.suptitle('SVM Classification: Impact of Feature Scaling', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SVM analysis saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_impact_report(self, save_dir='plots'):
        """
        Create comprehensive impact analysis report.
        
        Parameters:
        -----------
        save_dir : str
            Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Creating comprehensive model impact analysis...")
        
        # Load datasets
        datasets = load_sample_datasets()
        
        # Wine dataset for classification
        wine_data = datasets['wine']['data']
        wine_target = datasets['wine']['target']
        
        # Synthetic data for clustering
        synthetic_data = datasets['synthetic']['data']
        
        # Create synthetic clustering labels
        from sklearn.cluster import KMeans
        kmeans_true = KMeans(n_clusters=3, random_state=42)
        synthetic_labels = kmeans_true.fit_predict(synthetic_data)
        
        print("\n1. Analyzing K-Means impact...")
        self.analyze_kmeans_impact(
            synthetic_data, synthetic_labels, n_clusters=3,
            save_path=f'{save_dir}/kmeans_impact.png'
        )
        
        print("\n2. Analyzing PCA impact...")
        self.analyze_pca_impact(
            wine_data,
            save_path=f'{save_dir}/pca_impact.png'
        )
        
        print("\n3. Analyzing SVM impact...")
        self.analyze_svm_impact(
            wine_data, wine_target,
            save_path=f'{save_dir}/svm_impact.png'
        )
        
        print(f"\nComprehensive impact analysis saved to {save_dir}/")


def main():
    """
    Main function to demonstrate model impact analysis.
    """
    print("🎯 FEATURE SCALING IMPACT ON ML MODELS")
    print("=" * 50)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Create analyzer
    analyzer = ModelImpactAnalyzer()
    
    # Run comprehensive analysis
    analyzer.create_comprehensive_impact_report(save_dir='plots/model_impact')
    
    print("\n✅ MODEL IMPACT ANALYSIS COMPLETE!")
    print("📁 Check the 'plots/model_impact' folder for all visualizations.")
    print("🔧 Analysis includes:")
    print("   - K-Means clustering convergence and quality")
    print("   - PCA variance explanation and component interpretation")
    print("   - SVM classification accuracy and decision boundaries")
    print("   - Performance metrics comparison")
    print("   - Computational efficiency analysis")
    
    return analyzer

if __name__ == "__main__":
    main() 