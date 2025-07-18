"""
Feature Scaling & Normalization from Scratch - Advanced Implementation
====================================================================

This module implements comprehensive feature scaling and normalization techniques
from scratch, covering all fundamental scaling methods and their mathematical foundations:

Scaling Methods:
- StandardScaler (Z-score normalization): x' = (x - μ) / σ
- MinMaxScaler (Min-Max normalization): x' = (x - min) / (max - min)
- RobustScaler (Median & IQR based): x' = (x - median) / IQR
- L2Normalizer (Unit vector scaling): x' = x / ||x||_2

Mathematical Foundation:
- Statistical moments and their robust alternatives
- Impact on distance-based algorithms
- Convergence properties in optimization
- Distribution shape preservation

Advanced Features:
- Inverse transformations
- Partial fitting for streaming data
- Feature-wise and sample-wise normalization
- Comprehensive statistical analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_wine, load_diabetes, make_blobs
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
from sklearn.preprocessing import Normalizer as SklearnNormalizer
from typing import Union, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample x is calculated as:
        z = (x - μ) / σ
    
    Where μ is the mean and σ is the standard deviation of the training samples.
    
    This transformation is also known as Z-score normalization and results in
    a distribution with mean=0 and std=1.
    
    Parameters:
    -----------
    with_mean : bool, default=True
        If True, center the data before scaling
    with_std : bool, default=True
        If True, scale the data to unit variance
    """
    
    def __init__(self, with_mean=True, with_std=True):
        """
        Initialize StandardScaler.
        
        Parameters:
        -----------
        with_mean : bool
            Whether to center the data
        with_std : bool
            Whether to scale the data
        """
        self.with_mean = with_mean
        self.with_std = with_std
        
        # Fitted parameters
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        self.n_samples_seen_ = None
        self.n_features_in_ = None
        
    def fit(self, X, y=None):
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self.n_samples_seen_ = n_samples
        self.n_features_in_ = n_features
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(n_features)
        
        if self.with_std:
            # Compute variance
            if self.with_mean:
                self.var_ = np.var(X, axis=0, ddof=0)
            else:
                self.var_ = np.mean(X**2, axis=0)
            
            # Compute scale (standard deviation)
            self.scale_ = np.sqrt(self.var_)
            
            # Handle zero variance features
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.var_ = np.ones(n_features)
            self.scale_ = np.ones(n_features)
        
        return self
    
    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Apply transformation
        X_transformed = X.copy()
        
        if self.with_mean:
            X_transformed = X_transformed - self.mean_
        
        if self.with_std:
            X_transformed = X_transformed / self.scale_
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        """
        Scale back the data to the original representation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to inverse transform
            
        Returns:
        --------
        X_original : array-like, shape (n_samples, n_features)
            Original scaled data
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Apply inverse transformation
        X_original = X.copy()
        
        if self.with_std:
            X_original = X_original * self.scale_
        
        if self.with_mean:
            X_original = X_original + self.mean_
        
        return X_original
    
    def partial_fit(self, X, y=None):
        """
        Online computation of mean and std on X for later scaling.
        
        This is useful for streaming data where all samples cannot fit in memory.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        if self.n_features_in_ is None:
            self.n_features_in_ = n_features
            self.n_samples_seen_ = 0
            self.mean_ = np.zeros(n_features)
            self.var_ = np.zeros(n_features)
        
        # Update statistics using Welford's online algorithm
        if self.with_mean or self.with_std:
            for sample in X:
                self.n_samples_seen_ += 1
                delta = sample - self.mean_
                self.mean_ += delta / self.n_samples_seen_
                
                if self.with_std:
                    delta2 = sample - self.mean_
                    self.var_ += delta * delta2
        
        if self.with_std:
            # Compute scale from variance
            if self.n_samples_seen_ > 1:
                self.var_ = self.var_ / (self.n_samples_seen_ - 1)
            self.scale_ = np.sqrt(self.var_)
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.scale_ = np.ones(n_features)
        
        if not self.with_mean:
            self.mean_ = np.zeros(n_features)
        
        return self


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range.
    
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.
    
    The transformation is given by:
        X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_scaled * (max - min) + min
    
    where min, max = feature_range.
    
    Parameters:
    -----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data
    clip : bool, default=False
        Set to True to clip transformed values to provided feature range
    """
    
    def __init__(self, feature_range=(0, 1), clip=False):
        """
        Initialize MinMaxScaler.
        
        Parameters:
        -----------
        feature_range : tuple
            Target range for scaling
        clip : bool
            Whether to clip values to feature range
        """
        self.feature_range = feature_range
        self.clip = clip
        
        # Fitted parameters
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.n_samples_seen_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y=None):
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self.n_samples_seen_ = n_samples
        self.n_features_in_ = n_features
        
        # Compute data statistics
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Handle constant features (range = 0)
        self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)
        
        # Compute scaling parameters
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_
        
        return self
    
    def transform(self, X):
        """
        Scale features of X according to feature_range.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data
        """
        if self.scale_ is None or self.min_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Apply transformation
        X_transformed = X * self.scale_ + self.min_
        
        # Clip if requested
        if self.clip:
            feature_min, feature_max = self.feature_range
            X_transformed = np.clip(X_transformed, feature_min, feature_max)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        """
        Undo the scaling of X according to feature_range.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to inverse transform
            
        Returns:
        --------
        X_original : array-like, shape (n_samples, n_features)
            Original scaled data
        """
        if self.scale_ is None or self.min_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Apply inverse transformation
        X_original = (X - self.min_) / self.scale_
        
        return X_original


class RobustScaler:
    """
    Scale features using statistics that are robust to outliers.
    
    This Scaler removes the median and scales the data according to
    the quantile range (defaults to IQR: Interquartile Range).
    The IQR is the range between the 1st quartile (25th quantile)
    and the 3rd quartile (75th quantile).
    
    Centering and scaling happen independently on each feature by
    computing the relevant statistics on the samples in the training
    set. Median and interquartile range are then stored to be used on
    later data using the transform method.
    
    The transformation is given by:
        X_scaled = (X - median) / IQR
    
    Parameters:
    -----------
    quantile_range : tuple (q_min, q_max), default=(25.0, 75.0)
        Quantile range used to calculate scale_
    with_centering : bool, default=True
        If True, center the data at the median before scaling
    with_scaling : bool, default=True
        If True, scale the data to interquartile range
    """
    
    def __init__(self, quantile_range=(25.0, 75.0), with_centering=True, with_scaling=True):
        """
        Initialize RobustScaler.
        
        Parameters:
        -----------
        quantile_range : tuple
            Quantile range for robust scaling
        with_centering : bool
            Whether to center the data
        with_scaling : bool
            Whether to scale the data
        """
        self.quantile_range = quantile_range
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        
        # Fitted parameters
        self.center_ = None
        self.scale_ = None
        self.n_features_in_ = None
    
    def fit(self, X, y=None):
        """
        Compute the median and quantiles to be used for scaling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        if self.with_centering:
            self.center_ = np.median(X, axis=0)
        else:
            self.center_ = np.zeros(n_features)
        
        if self.with_scaling:
            q_min, q_max = self.quantile_range
            
            # Compute quantiles
            q1 = np.percentile(X, q_min, axis=0)
            q3 = np.percentile(X, q_max, axis=0)
            
            # Compute IQR
            self.scale_ = q3 - q1
            
            # Handle zero IQR (constant features)
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.scale_ = np.ones(n_features)
        
        return self
    
    def transform(self, X):
        """
        Center and scale the data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data
        """
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Apply transformation
        X_transformed = X.copy()
        
        if self.with_centering:
            X_transformed = X_transformed - self.center_
        
        if self.with_scaling:
            X_transformed = X_transformed / self.scale_
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns:
        --------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X):
        """
        Scale back the data to the original representation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to inverse transform
            
        Returns:
        --------
        X_original : array-like, shape (n_samples, n_features)
            Original scaled data
        """
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted yet. Call 'fit' first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Apply inverse transformation
        X_original = X.copy()
        
        if self.with_scaling:
            X_original = X_original * self.scale_
        
        if self.with_centering:
            X_original = X_original + self.center_
        
        return X_original


class L2Normalizer:
    """
    Normalize samples individually to unit norm.
    
    Each sample (i.e. each row of the data matrix) with at least one
    non zero component is rescaled independently of other samples so
    that its norm (l1, l2 or inf) equals one.
    
    This transformer is able to work both with dense numpy arrays and
    scipy.sparse matrix (use CSR format if you want to avoid the burden
    of a copy / conversion).
    
    Scaling inputs to unit norms is a common operation for text
    classification or clustering for instance.
    
    The transformation is given by:
        X_normalized = X / ||X||_norm
    
    Parameters:
    -----------
    norm : str, default='l2'
        The norm to use to normalize each non zero sample
        ('l1', 'l2', or 'max')
    """
    
    def __init__(self, norm='l2'):
        """
        Initialize L2Normalizer.
        
        Parameters:
        -----------
        norm : str
            Type of norm to use for normalization
        """
        if norm not in ['l1', 'l2', 'max']:
            raise ValueError("norm must be 'l1', 'l2', or 'max'")
        
        self.norm = norm
        self.n_features_in_ = None
    
    def fit(self, X, y=None):
        """
        Do nothing and return the estimator unchanged.
        
        This method is just there to implement the usual API and hence
        work in pipelines.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        return self
    
    def transform(self, X):
        """
        Scale each non zero row of X to unit norm.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data to normalize
            
        Returns:
        --------
        X_normalized : array-like, shape (n_samples, n_features)
            Normalized data
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_normalized = X.copy()
        
        # Compute norms
        if self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=1)
        elif self.norm == 'l2':
            norms = np.sqrt(np.sum(X**2, axis=1))
        elif self.norm == 'max':
            norms = np.max(np.abs(X), axis=1)
        
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        
        # Normalize
        X_normalized = X_normalized / norms.reshape(-1, 1)
        
        return X_normalized
    
    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns:
        --------
        X_normalized : array-like, shape (n_samples, n_features)
            Normalized data
        """
        return self.fit(X, y).transform(X)


class ScalingAnalyzer:
    """
    Comprehensive analyzer for feature scaling effects and comparisons.
    
    This class provides tools for analyzing the impact of different
    scaling methods on data distributions and model performance.
    """
    
    def __init__(self):
        """Initialize the scaling analyzer."""
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'L2Normalizer': L2Normalizer()
        }
        self.sklearn_scalers = {
            'StandardScaler': SklearnStandardScaler(),
            'MinMaxScaler': SklearnMinMaxScaler(),
            'RobustScaler': SklearnRobustScaler(),
            'L2Normalizer': SklearnNormalizer(norm='l2')
        }
        self.scaling_results = {}
    
    def compare_scalers(self, X, feature_names=None):
        """
        Compare all scaling methods on the given data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        feature_names : list, optional
            Names of features
            
        Returns:
        --------
        dict : Scaling results for each method
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        results = {'Original': X}
        
        print("Comparing scaling methods...")
        
        # Apply each scaler
        for name, scaler in self.scalers.items():
            print(f"  Applying {name}...")
            
            try:
                X_scaled = scaler.fit_transform(X)
                results[name] = X_scaled
                
                # Compute statistics
                print(f"    Mean: {np.mean(X_scaled, axis=0)}")
                print(f"    Std:  {np.std(X_scaled, axis=0)}")
                print(f"    Range: [{np.min(X_scaled):.3f}, {np.max(X_scaled):.3f}]")
                
            except Exception as e:
                print(f"    Error: {e}")
                results[name] = None
        
        self.scaling_results = results
        return results
    
    def plot_distributions(self, X, feature_names=None, save_path=None):
        """
        Plot distribution comparisons before and after scaling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        feature_names : list, optional
            Names of features
        save_path : str, optional
            Path to save the plot
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Get scaling results
        if not self.scaling_results:
            self.compare_scalers(X, feature_names)
        
        n_features = min(4, X.shape[1])  # Limit to 4 features for visualization
        n_scalers = len(self.scaling_results)
        
        fig, axes = plt.subplots(n_features, n_scalers, figsize=(4*n_scalers, 3*n_features))
        
        if n_features == 1:
            axes = axes.reshape(1, -1)
        if n_scalers == 1:
            axes = axes.reshape(-1, 1)
        
        for i, feature_idx in enumerate(range(n_features)):
            for j, (scaler_name, X_scaled) in enumerate(self.scaling_results.items()):
                ax = axes[i, j]
                
                if X_scaled is not None:
                    # Plot histogram
                    ax.hist(X_scaled[:, feature_idx], bins=30, alpha=0.7, 
                           density=True, edgecolor='black', linewidth=0.5)
                    
                    # Add statistics
                    mean_val = np.mean(X_scaled[:, feature_idx])
                    std_val = np.std(X_scaled[:, feature_idx])
                    median_val = np.median(X_scaled[:, feature_idx])
                    
                    ax.axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.2f}')
                    ax.axvline(median_val, color='green', linestyle='--', 
                              label=f'Median: {median_val:.2f}')
                    
                    ax.set_title(f'{scaler_name}\n{feature_names[feature_idx]}\n'
                               f'μ={mean_val:.2f}, σ={std_val:.2f}')
                    ax.legend(fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(f'{scaler_name}\n{feature_names[feature_idx]}')
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_boxplots(self, X, feature_names=None, save_path=None):
        """
        Plot boxplot comparisons before and after scaling.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        feature_names : list, optional
            Names of features
        save_path : str, optional
            Path to save the plot
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Get scaling results
        if not self.scaling_results:
            self.compare_scalers(X, feature_names)
        
        n_scalers = len(self.scaling_results)
        fig, axes = plt.subplots(1, n_scalers, figsize=(4*n_scalers, 6))
        
        if n_scalers == 1:
            axes = [axes]
        
        for j, (scaler_name, X_scaled) in enumerate(self.scaling_results.items()):
            ax = axes[j]
            
            if X_scaled is not None:
                # Create boxplot
                box_data = [X_scaled[:, i] for i in range(min(X_scaled.shape[1], 8))]
                labels = feature_names[:len(box_data)]
                
                bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
                
                # Color the boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                ax.set_title(f'{scaler_name}')
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'{scaler_name}')
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Boxplot comparison saved to {save_path}")
        
        plt.show()
    
    def compare_with_sklearn(self, X):
        """
        Compare our implementations with sklearn's implementations.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        dict : Comparison results
        """
        comparison_results = {}
        
        print("Comparing with sklearn implementations...")
        
        for name in self.scalers.keys():
            print(f"\nTesting {name}:")
            
            # Our implementation
            our_scaler = self.scalers[name]
            X_ours = our_scaler.fit_transform(X)
            
            # Sklearn implementation
            sklearn_scaler = self.sklearn_scalers[name]
            X_sklearn = sklearn_scaler.fit_transform(X)
            
            # Compare results
            diff = np.abs(X_ours - X_sklearn)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"  Max difference: {max_diff:.2e}")
            print(f"  Mean difference: {mean_diff:.2e}")
            print(f"  Close to sklearn: {np.allclose(X_ours, X_sklearn, rtol=1e-10)}")
            
            comparison_results[name] = {
                'our_result': X_ours,
                'sklearn_result': X_sklearn,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'close': np.allclose(X_ours, X_sklearn, rtol=1e-10)
            }
        
        return comparison_results


def load_sample_datasets():
    """
    Load sample datasets for scaling analysis.
    
    Returns:
    --------
    dict : Dictionary containing different datasets
    """
    datasets = {}
    
    # Wine dataset
    wine = load_wine()
    datasets['wine'] = {
        'data': wine.data,
        'target': wine.target,
        'feature_names': wine.feature_names,
        'name': 'Wine Dataset'
    }
    
    # Diabetes dataset
    diabetes = load_diabetes()
    datasets['diabetes'] = {
        'data': diabetes.data,
        'target': diabetes.target,
        'feature_names': diabetes.feature_names,
        'name': 'Diabetes Dataset'
    }
    
    # Synthetic dataset with different scales
    np.random.seed(42)
    n_samples = 1000
    
    # Features with vastly different scales
    feature1 = np.random.normal(0, 1, n_samples)  # Small scale
    feature2 = np.random.normal(1000, 500, n_samples)  # Large scale
    feature3 = np.random.exponential(0.1, n_samples)  # Skewed distribution
    feature4 = np.random.uniform(-10, 10, n_samples)  # Uniform distribution
    
    synthetic_data = np.column_stack([feature1, feature2, feature3, feature4])
    
    datasets['synthetic'] = {
        'data': synthetic_data,
        'target': np.random.randint(0, 3, n_samples),
        'feature_names': ['Normal_Small', 'Normal_Large', 'Exponential', 'Uniform'],
        'name': 'Synthetic Multi-Scale Dataset'
    }
    
    return datasets


def main():
    """
    Main function to demonstrate feature scaling implementations.
    """
    print("🎯 FEATURE SCALING & NORMALIZATION FROM SCRATCH")
    print("=" * 60)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load sample datasets
    datasets = load_sample_datasets()
    
    # Test each dataset
    for dataset_name, dataset in datasets.items():
        print(f"\n📊 Analyzing {dataset['name']}")
        print("-" * 50)
        
        X = dataset['data']
        feature_names = dataset['feature_names']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Original data statistics:")
        print(f"  Mean: {np.mean(X, axis=0)[:3]}...")  # Show first 3 features
        print(f"  Std:  {np.std(X, axis=0)[:3]}...")
        print(f"  Range: [{np.min(X):.3f}, {np.max(X):.3f}]")
        
        # Create analyzer
        analyzer = ScalingAnalyzer()
        
        # Compare scaling methods
        scaling_results = analyzer.compare_scalers(X, feature_names)
        
        # Plot distributions
        analyzer.plot_distributions(X, feature_names, 
                                  save_path=f'plots/distributions_{dataset_name}.png')
        
        # Plot boxplots
        analyzer.plot_boxplots(X, feature_names, 
                              save_path=f'plots/boxplots_{dataset_name}.png')
        
        # Compare with sklearn
        comparison = analyzer.compare_with_sklearn(X)
        
        print(f"\n✅ Sklearn comparison passed: {all(r['close'] for r in comparison.values())}")
    
    print(f"\n🔍 Testing Individual Scalers:")
    print("-" * 40)
    
    # Test individual scalers with detailed examples
    X_test = datasets['synthetic']['data'][:100]  # Small subset for detailed analysis
    
    # Test StandardScaler
    print(f"\n📏 StandardScaler Test:")
    std_scaler = StandardScaler()
    X_std = std_scaler.fit_transform(X_test)
    X_inv = std_scaler.inverse_transform(X_std)
    
    print(f"  Original mean: {np.mean(X_test, axis=0)}")
    print(f"  Scaled mean: {np.mean(X_std, axis=0)}")
    print(f"  Scaled std: {np.std(X_std, axis=0)}")
    print(f"  Inverse transform error: {np.max(np.abs(X_test - X_inv)):.2e}")
    
    # Test MinMaxScaler
    print(f"\n📐 MinMaxScaler Test:")
    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    X_mm = mm_scaler.fit_transform(X_test)
    X_inv = mm_scaler.inverse_transform(X_mm)
    
    print(f"  Original range: [{np.min(X_test):.3f}, {np.max(X_test):.3f}]")
    print(f"  Scaled range: [{np.min(X_mm):.3f}, {np.max(X_mm):.3f}]")
    print(f"  Inverse transform error: {np.max(np.abs(X_test - X_inv)):.2e}")
    
    # Test RobustScaler
    print(f"\n🛡️ RobustScaler Test:")
    rob_scaler = RobustScaler()
    X_rob = rob_scaler.fit_transform(X_test)
    X_inv = rob_scaler.inverse_transform(X_rob)
    
    print(f"  Original median: {np.median(X_test, axis=0)}")
    print(f"  Scaled median: {np.median(X_rob, axis=0)}")
    print(f"  Scaled IQR: {np.percentile(X_rob, 75, axis=0) - np.percentile(X_rob, 25, axis=0)}")
    print(f"  Inverse transform error: {np.max(np.abs(X_test - X_inv)):.2e}")
    
    # Test L2Normalizer
    print(f"\n🎯 L2Normalizer Test:")
    l2_scaler = L2Normalizer()
    X_l2 = l2_scaler.fit_transform(X_test)
    
    print(f"  Original norms: {np.linalg.norm(X_test, axis=1)[:5]}")
    print(f"  Normalized norms: {np.linalg.norm(X_l2, axis=1)[:5]}")
    print(f"  All norms ≈ 1: {np.allclose(np.linalg.norm(X_l2, axis=1), 1.0)}")
    
    print("\n✅ FEATURE SCALING IMPLEMENTATION COMPLETE!")
    print("📁 Check the 'plots' folder for scaling visualizations.")
    print("🔧 The implementation covers all key scaling concepts:")
    print("   - StandardScaler (Z-score normalization)")
    print("   - MinMaxScaler (Min-Max scaling)")
    print("   - RobustScaler (Median & IQR based)")
    print("   - L2Normalizer (Unit vector scaling)")
    print("   - Comprehensive statistical analysis")
    print("   - Sklearn compatibility verification")
    
    return datasets, analyzer

if __name__ == "__main__":
    main() 