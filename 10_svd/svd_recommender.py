"""
SVD Recommender System - Advanced Implementation
===============================================

This module implements SVD-based collaborative filtering including:
- Matrix factorization for recommendation systems
- Missing value handling strategies
- User-item matrix reconstruction
- Top-N recommendation generation
- Recommendation accuracy evaluation
- Cold start problem analysis

Mathematical Foundation:
- Rating matrix R ≈ UΣV^T (with missing values)
- Rank-k approximation: R_k = U_k Σ_k V_k^T
- Prediction: r̂_ui = μ + b_u + b_i + q_i^T p_u
- RMSE evaluation on held-out test set
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SVDRecommender:
    """
    SVD-based collaborative filtering recommender system.
    
    Features:
    - Handle sparse rating matrices
    - Multiple missing value strategies
    - Rank-k matrix factorization
    - Top-N recommendation generation
    - Comprehensive evaluation metrics
    """
    
    def __init__(self, n_components=50, random_state=42):
        """
        Initialize SVD recommender.
        
        Parameters:
        -----------
        n_components : int
            Number of latent factors (rank of approximation)
        random_state : int
            Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.user_mean = None
        self.item_mean = None
        self.global_mean = None
        self.U = None
        self.s = None
        self.Vt = None
        self.user_mapping = None
        self.item_mapping = None
        self.filled_matrix = None
        self.original_mask = None
        
    def create_synthetic_dataset(self, n_users=1000, n_items=500, n_ratings=50000, 
                               n_factors=10, noise_level=0.5):
        """
        Create synthetic rating dataset for testing.
        
        Parameters:
        -----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_ratings : int
            Number of ratings to generate
        n_factors : int
            Number of latent factors in true model
        noise_level : float
            Noise level in ratings
            
        Returns:
        --------
        pd.DataFrame : Rating data with columns [user_id, item_id, rating]
        """
        np.random.seed(self.random_state)
        
        # Generate latent factors
        user_factors = np.random.normal(0, 1, (n_users, n_factors))
        item_factors = np.random.normal(0, 1, (n_items, n_factors))
        
        # Generate user and item biases
        user_bias = np.random.normal(0, 0.5, n_users)
        item_bias = np.random.normal(0, 0.5, n_items)
        global_bias = 3.5  # Average rating
        
        # Generate random user-item pairs
        users = np.random.choice(n_users, n_ratings)
        items = np.random.choice(n_items, n_ratings)
        
        # Calculate true ratings
        true_ratings = (
            global_bias + 
            user_bias[users] + 
            item_bias[items] + 
            np.sum(user_factors[users] * item_factors[items], axis=1) +
            noise_level * np.random.normal(0, 1, n_ratings)
        )
        
        # Clip ratings to valid range [1, 5]
        true_ratings = np.clip(true_ratings, 1, 5)
        
        # Create DataFrame
        ratings_df = pd.DataFrame({
            'user_id': users,
            'item_id': items,
            'rating': true_ratings
        })
        
        # Remove duplicates (keep first occurrence)
        ratings_df = ratings_df.drop_duplicates(['user_id', 'item_id'])
        
        print(f"Generated synthetic dataset:")
        print(f"Users: {n_users}, Items: {n_items}")
        print(f"Ratings: {len(ratings_df)}")
        print(f"Sparsity: {1 - len(ratings_df) / (n_users * n_items):.4f}")
        print(f"Rating range: [{ratings_df['rating'].min():.2f}, {ratings_df['rating'].max():.2f}]")
        
        return ratings_df
    
    def load_movielens_subset(self, n_users=500, n_items=300):
        """
        Create a MovieLens-like dataset subset for demonstration.
        
        Parameters:
        -----------
        n_users : int
            Number of users to include
        n_items : int
            Number of items to include
            
        Returns:
        --------
        pd.DataFrame : Rating data
        """
        print("Creating MovieLens-like dataset...")
        
        # Generate realistic rating patterns
        np.random.seed(self.random_state)
        
        ratings_list = []
        
        for user_id in range(n_users):
            # Each user rates between 10 and 100 items
            n_user_ratings = np.random.randint(10, min(101, n_items))
            
            # User preference profile (some users like certain genres)
            user_preference = np.random.normal(0, 1, 5)  # 5 genre preferences
            
            # Select items to rate
            rated_items = np.random.choice(n_items, n_user_ratings, replace=False)
            
            for item_id in rated_items:
                # Item genre profile
                item_genre = np.random.normal(0, 1, 5)
                
                # Base rating from user-item interaction
                base_rating = 3.5 + 0.8 * np.dot(user_preference, item_genre) / 5
                
                # Add noise and user bias
                user_bias = np.random.normal(0, 0.3)
                final_rating = base_rating + user_bias + np.random.normal(0, 0.4)
                
                # Clip to [1, 5] and round to nearest 0.5
                final_rating = np.clip(final_rating, 1, 5)
                final_rating = np.round(final_rating * 2) / 2
                
                ratings_list.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': final_rating
                })
        
        ratings_df = pd.DataFrame(ratings_list)
        
        print(f"MovieLens-like dataset created:")
        print(f"Users: {ratings_df['user_id'].nunique()}")
        print(f"Items: {ratings_df['item_id'].nunique()}")
        print(f"Ratings: {len(ratings_df)}")
        print(f"Average rating: {ratings_df['rating'].mean():.2f}")
        print(f"Rating distribution:")
        print(ratings_df['rating'].value_counts().sort_index())
        
        return ratings_df
    
    def create_rating_matrix(self, ratings_df):
        """
        Convert rating DataFrame to user-item matrix.
        
        Parameters:
        -----------
        ratings_df : pd.DataFrame
            Rating data with columns [user_id, item_id, rating]
            
        Returns:
        --------
        np.ndarray : User-item rating matrix
        """
        # Create user and item mappings
        unique_users = sorted(ratings_df['user_id'].unique())
        unique_items = sorted(ratings_df['item_id'].unique())
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        
        # Create matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        rating_matrix = np.full((n_users, n_items), np.nan)
        
        # Fill matrix with ratings
        for _, row in ratings_df.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            item_idx = self.item_mapping[row['item_id']]
            rating_matrix[user_idx, item_idx] = row['rating']
        
        # Store mask of observed ratings
        self.original_mask = ~np.isnan(rating_matrix)
        
        print(f"Rating matrix shape: {rating_matrix.shape}")
        print(f"Observed ratings: {np.sum(self.original_mask)}")
        print(f"Sparsity: {1 - np.sum(self.original_mask) / rating_matrix.size:.4f}")
        
        return rating_matrix
    
    def fill_missing_values(self, rating_matrix, strategy='user_mean'):
        """
        Fill missing values in rating matrix.
        
        Parameters:
        -----------
        rating_matrix : np.ndarray
            Sparse rating matrix with NaN for missing values
        strategy : str
            Strategy for filling missing values:
            - 'global_mean': Use global average rating
            - 'user_mean': Use user average rating
            - 'item_mean': Use item average rating
            - 'user_item_mean': Use user and item averages
            - 'zero': Fill with zeros
            
        Returns:
        --------
        np.ndarray : Filled rating matrix
        """
        filled_matrix = rating_matrix.copy()
        
        # Calculate means
        self.global_mean = np.nanmean(rating_matrix)
        self.user_mean = np.nanmean(rating_matrix, axis=1)
        self.item_mean = np.nanmean(rating_matrix, axis=0)
        
        print(f"Filling missing values using strategy: {strategy}")
        print(f"Global mean: {self.global_mean:.3f}")
        
        if strategy == 'global_mean':
            filled_matrix = np.nan_to_num(filled_matrix, nan=self.global_mean)
            
        elif strategy == 'user_mean':
            for user_idx in range(rating_matrix.shape[0]):
                user_ratings = rating_matrix[user_idx, :]
                if np.isnan(self.user_mean[user_idx]):
                    # User has no ratings, use global mean
                    fill_value = self.global_mean
                else:
                    fill_value = self.user_mean[user_idx]
                
                filled_matrix[user_idx, np.isnan(user_ratings)] = fill_value
                
        elif strategy == 'item_mean':
            for item_idx in range(rating_matrix.shape[1]):
                item_ratings = rating_matrix[:, item_idx]
                if np.isnan(self.item_mean[item_idx]):
                    # Item has no ratings, use global mean
                    fill_value = self.global_mean
                else:
                    fill_value = self.item_mean[item_idx]
                
                filled_matrix[np.isnan(item_ratings), item_idx] = fill_value
                
        elif strategy == 'user_item_mean':
            # Use user mean + item mean - global mean
            for user_idx in range(rating_matrix.shape[0]):
                for item_idx in range(rating_matrix.shape[1]):
                    if np.isnan(rating_matrix[user_idx, item_idx]):
                        user_avg = self.user_mean[user_idx] if not np.isnan(self.user_mean[user_idx]) else self.global_mean
                        item_avg = self.item_mean[item_idx] if not np.isnan(self.item_mean[item_idx]) else self.global_mean
                        filled_matrix[user_idx, item_idx] = user_avg + item_avg - self.global_mean
                        
        elif strategy == 'zero':
            filled_matrix = np.nan_to_num(filled_matrix, nan=0.0)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.filled_matrix = filled_matrix
        return filled_matrix
    
    def fit(self, rating_matrix, fill_strategy='user_item_mean'):
        """
        Fit SVD model to rating matrix.
        
        Parameters:
        -----------
        rating_matrix : np.ndarray
            User-item rating matrix (may contain NaN)
        fill_strategy : str
            Strategy for handling missing values
        """
        # Fill missing values
        filled_matrix = self.fill_missing_values(rating_matrix, fill_strategy)
        
        # Center the data
        centered_matrix = filled_matrix - self.global_mean
        
        # Perform SVD
        self.U, self.s, self.Vt = np.linalg.svd(centered_matrix, full_matrices=False)
        
        # Keep only top k components
        k = min(self.n_components, len(self.s))
        self.U = self.U[:, :k]
        self.s = self.s[:k]
        self.Vt = self.Vt[:k, :]
        
        print(f"SVD completed:")
        print(f"Components used: {k}")
        print(f"Explained variance ratio: {np.sum(self.s[:k]**2) / np.sum(self.s**2):.4f}")
    
    def predict(self, user_idx=None, item_idx=None):
        """
        Predict ratings using fitted SVD model.
        
        Parameters:
        -----------
        user_idx : int or None
            User index (if None, predict for all users)
        item_idx : int or None
            Item index (if None, predict for all items)
            
        Returns:
        --------
        np.ndarray : Predicted ratings
        """
        if self.U is None:
            raise ValueError("Model not fitted yet")
        
        # Reconstruct rating matrix
        reconstructed = self.U @ np.diag(self.s) @ self.Vt + self.global_mean
        
        if user_idx is not None and item_idx is not None:
            return reconstructed[user_idx, item_idx]
        elif user_idx is not None:
            return reconstructed[user_idx, :]
        elif item_idx is not None:
            return reconstructed[:, item_idx]
        else:
            return reconstructed
    
    def recommend_items(self, user_idx, n_recommendations=10, exclude_seen=True):
        """
        Generate top-N item recommendations for a user.
        
        Parameters:
        -----------
        user_idx : int
            User index
        n_recommendations : int
            Number of recommendations to generate
        exclude_seen : bool
            Whether to exclude items the user has already rated
            
        Returns:
        --------
        list : List of (item_idx, predicted_rating) tuples
        """
        # Get predictions for the user
        user_predictions = self.predict(user_idx=user_idx)
        
        # Create list of (item_idx, prediction) pairs
        item_predictions = list(enumerate(user_predictions))
        
        # Exclude items the user has already seen
        if exclude_seen and self.original_mask is not None:
            item_predictions = [
                (item_idx, pred) for item_idx, pred in item_predictions
                if not self.original_mask[user_idx, item_idx]
            ]
        
        # Sort by predicted rating (descending)
        item_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return item_predictions[:n_recommendations]
    
    def evaluate_recommendations(self, test_ratings, k_list=[5, 10, 20]):
        """
        Evaluate recommendation quality using precision@k and recall@k.
        
        Parameters:
        -----------
        test_ratings : np.ndarray
            Test rating matrix with same shape as training matrix
        k_list : list
            List of k values for evaluation
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        if self.original_mask is None:
            raise ValueError("No training mask available")
        
        # Find test users (users with ratings in test set)
        test_mask = ~np.isnan(test_ratings)
        test_users = np.where(np.any(test_mask, axis=1))[0]
        
        results = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_list}
        
        print(f"Evaluating recommendations for {len(test_users)} test users...")
        
        for user_idx in test_users:
            # Get user's test items and ratings
            user_test_items = np.where(test_mask[user_idx, :])[0]
            user_test_ratings = test_ratings[user_idx, user_test_items]
            
            # Consider items with rating >= 4 as relevant
            relevant_items = set(user_test_items[user_test_ratings >= 4.0])
            
            if len(relevant_items) == 0:
                continue  # Skip users with no relevant items
            
            for k in k_list:
                # Get top-k recommendations
                recommendations = self.recommend_items(user_idx, k, exclude_seen=True)
                recommended_items = set([item_idx for item_idx, _ in recommendations])
                
                # Calculate precision@k and recall@k
                true_positives = len(recommended_items & relevant_items)
                precision = true_positives / k if k > 0 else 0
                recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0
                
                results[k]['precision'].append(precision)
                results[k]['recall'].append(recall)
        
        # Calculate average metrics
        avg_results = {}
        for k in k_list:
            avg_results[k] = {
                'precision': np.mean(results[k]['precision']) if results[k]['precision'] else 0,
                'recall': np.mean(results[k]['recall']) if results[k]['recall'] else 0,
                'f1': 0
            }
            
            # Calculate F1 score
            p, r = avg_results[k]['precision'], avg_results[k]['recall']
            avg_results[k]['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        return avg_results
    
    def plot_analysis(self, rating_matrix, test_matrix=None, save_path=None):
        """
        Plot comprehensive analysis of the recommender system.
        
        Parameters:
        -----------
        rating_matrix : np.ndarray
            Training rating matrix
        test_matrix : np.ndarray, optional
            Test rating matrix for evaluation
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Rating distribution
        ax = axes[0, 0]
        observed_ratings = rating_matrix[self.original_mask]
        ax.hist(observed_ratings, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Rating Distribution')
        ax.grid(True, alpha=0.3)
        
        # 2. Singular values
        ax = axes[0, 1]
        ax.plot(range(1, len(self.s) + 1), self.s, 'bo-', linewidth=2, markersize=4)
        ax.set_xlabel('Component Index')
        ax.set_ylabel('Singular Value')
        ax.set_title('Singular Values')
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative explained variance
        ax = axes[0, 2]
        cumulative_var = np.cumsum(self.s**2) / np.sum(self.s**2)
        ax.plot(range(1, len(self.s) + 1), cumulative_var, 'ro-', linewidth=2, markersize=4)
        ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90%')
        ax.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Explained Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. User activity (number of ratings per user)
        ax = axes[1, 0]
        user_activity = np.sum(self.original_mask, axis=1)
        ax.hist(user_activity, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Ratings')
        ax.set_ylabel('Number of Users')
        ax.set_title('User Activity Distribution')
        ax.grid(True, alpha=0.3)
        
        # 5. Item popularity (number of ratings per item)
        ax = axes[1, 1]
        item_popularity = np.sum(self.original_mask, axis=0)
        ax.hist(item_popularity, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Ratings')
        ax.set_ylabel('Number of Items')
        ax.set_title('Item Popularity Distribution')
        ax.grid(True, alpha=0.3)
        
        # 6. Prediction error analysis
        ax = axes[1, 2]
        if test_matrix is not None:
            test_mask = ~np.isnan(test_matrix)
            if np.any(test_mask):
                predictions = self.predict()
                test_predictions = predictions[test_mask]
                test_actuals = test_matrix[test_mask]
                
                # Scatter plot of predictions vs actual
                ax.scatter(test_actuals, test_predictions, alpha=0.5, s=1)
                
                # Perfect prediction line
                min_val, max_val = min(test_actuals.min(), test_predictions.min()), max(test_actuals.max(), test_predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
                
                ax.set_xlabel('Actual Rating')
                ax.set_ylabel('Predicted Rating')
                ax.set_title('Predictions vs Actual')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
                ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No test data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Predictions vs Actual')
        else:
            ax.text(0.5, 0.5, 'No test data provided', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Predictions vs Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Analysis plot saved to {save_path}")
        
        plt.show()

def demonstrate_recommender_system():
    """
    Demonstrate SVD recommender system with comprehensive analysis.
    """
    print("=" * 70)
    print("SVD RECOMMENDER SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Create recommender
    recommender = SVDRecommender(n_components=50, random_state=42)
    
    # Generate or load dataset
    print("1. GENERATING DATASET...")
    print("-" * 30)
    
    # Try different dataset options
    use_synthetic = True  # Change to False to use MovieLens-like data
    
    if use_synthetic:
        ratings_df = recommender.create_synthetic_dataset(
            n_users=800, n_items=400, n_ratings=40000, n_factors=15
        )
    else:
        ratings_df = recommender.load_movielens_subset(n_users=600, n_items=350)
    
    return recommender, ratings_df

def main():
    """
    Main function to run SVD recommender system analysis.
    """
    print("🎬 SVD RECOMMENDER SYSTEM - ADVANCED IMPLEMENTATION")
    print("=" * 80)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # 1. Create and analyze dataset
    recommender, ratings_df = demonstrate_recommender_system()
    
    # 2. Create rating matrix
    print("\n2. CREATING RATING MATRIX...")
    print("-" * 35)
    rating_matrix = recommender.create_rating_matrix(ratings_df)
    
    # 3. Split data for evaluation
    print("\n3. SPLITTING DATA FOR EVALUATION...")
    print("-" * 40)
    
    # Create train/test split by randomly masking some ratings
    np.random.seed(42)
    test_ratio = 0.2
    
    # Copy original matrix for test set
    test_matrix = rating_matrix.copy()
    train_matrix = rating_matrix.copy()
    
    # Randomly select ratings for test set
    observed_indices = np.where(~np.isnan(rating_matrix))
    n_observed = len(observed_indices[0])
    n_test = int(n_observed * test_ratio)
    
    test_indices = np.random.choice(n_observed, n_test, replace=False)
    test_users = observed_indices[0][test_indices]
    test_items = observed_indices[1][test_indices]
    
    # Remove test ratings from training matrix
    train_matrix[test_users, test_items] = np.nan
    
    # Create mask for training indices (complement of test indices)
    train_indices = np.setdiff1d(np.arange(n_observed), test_indices)
    train_users = observed_indices[0][train_indices]
    train_items = observed_indices[1][train_indices]
    
    # Remove training ratings from test matrix
    test_matrix[train_users, train_items] = np.nan
    
    print(f"Training ratings: {np.sum(~np.isnan(train_matrix))}")
    print(f"Test ratings: {np.sum(~np.isnan(test_matrix))}")
    
    # 4. Train model
    print("\n4. TRAINING SVD MODEL...")
    print("-" * 30)
    recommender.fit(train_matrix, fill_strategy='user_item_mean')
    
    # 5. Evaluate model
    print("\n5. EVALUATING MODEL...")
    print("-" * 25)
    
    # Prediction accuracy
    test_mask = ~np.isnan(test_matrix)
    if np.any(test_mask):
        predictions = recommender.predict()
        test_predictions = predictions[test_mask]
        test_actuals = test_matrix[test_mask]
        
        rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
        mae = mean_absolute_error(test_actuals, test_predictions)
        
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"Baseline RMSE (global mean): {np.sqrt(np.var(test_actuals)):.4f}")
    
    # Recommendation quality
    rec_metrics = recommender.evaluate_recommendations(test_matrix, k_list=[5, 10, 20])
    
    print("\nRecommendation Quality:")
    print("K  | Precision | Recall | F1-Score")
    print("-" * 35)
    for k, metrics in rec_metrics.items():
        print(f"{k:2d} | {metrics['precision']:9.4f} | {metrics['recall']:6.4f} | {metrics['f1']:8.4f}")
    
    # 6. Generate sample recommendations
    print("\n6. SAMPLE RECOMMENDATIONS...")
    print("-" * 32)
    
    # Show recommendations for first few users
    for user_idx in range(min(5, rating_matrix.shape[0])):
        if np.any(recommender.original_mask[user_idx, :]):  # User has ratings
            recommendations = recommender.recommend_items(user_idx, n_recommendations=5)
            
            print(f"\nUser {user_idx} recommendations:")
            for i, (item_idx, pred_rating) in enumerate(recommendations, 1):
                print(f"  {i}. Item {item_idx}: {pred_rating:.2f}")
    
    # 7. Analyze different numbers of components
    print("\n7. COMPONENT ANALYSIS...")
    print("-" * 28)
    
    component_range = [5, 10, 20, 50, 100]
    component_range = [k for k in component_range if k <= min(rating_matrix.shape)]
    
    rmse_results = []
    precision_results = []
    
    for n_comp in component_range:
        # Train model with different number of components
        temp_recommender = SVDRecommender(n_components=n_comp, random_state=42)
        temp_recommender.fit(train_matrix, fill_strategy='user_item_mean')
        
        # Copy the original mask from the main recommender
        temp_recommender.original_mask = recommender.original_mask
        
        # Evaluate
        if np.any(test_mask):
            temp_predictions = temp_recommender.predict()
            temp_test_predictions = temp_predictions[test_mask]
            temp_rmse = np.sqrt(mean_squared_error(test_actuals, temp_test_predictions))
            rmse_results.append(temp_rmse)
        else:
            rmse_results.append(0)
        
        # Recommendation precision
        try:
            temp_metrics = temp_recommender.evaluate_recommendations(test_matrix, k_list=[10])
            precision_results.append(temp_metrics[10]['precision'])
        except:
            # If evaluation fails, use 0 as fallback
            precision_results.append(0.0)
    
    # Plot component analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(component_range, rmse_results, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE vs Number of Components')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(component_range, precision_results, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Precision@10')
    ax2.set_title('Recommendation Precision vs Components')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/svd_recommender_component_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nComponent Analysis Results:")
    print("Components | RMSE   | Precision@10")
    print("-" * 35)
    for i, n_comp in enumerate(component_range):
        print(f"{n_comp:10d} | {rmse_results[i]:6.4f} | {precision_results[i]:12.4f}")
    
    # 8. Generate comprehensive analysis plot
    recommender.plot_analysis(train_matrix, test_matrix, 'plots/svd_recommender_analysis.png')
    
    print("\n✅ SVD RECOMMENDER SYSTEM ANALYSIS COMPLETE!")
    print("📁 Check the 'plots' folder for generated visualizations.")
    print("\n📋 SUMMARY:")
    print(f"• Dataset: {rating_matrix.shape[0]} users, {rating_matrix.shape[1]} items")
    print(f"• Sparsity: {1 - np.sum(recommender.original_mask) / rating_matrix.size:.4f}")
    print(f"• Best RMSE: {min(rmse_results):.4f} (with {component_range[np.argmin(rmse_results)]} components)")
    print(f"• Best Precision@10: {max(precision_results):.4f} (with {component_range[np.argmax(precision_results)]} components)")
    print(f"• Generated 2 comprehensive visualization files")

if __name__ == "__main__":
    main() 