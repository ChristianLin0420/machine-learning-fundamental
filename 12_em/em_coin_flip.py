"""
EM Algorithm for Coin Flipping Problem - Classic Implementation
==============================================================

This module implements the classic coin flipping problem that demonstrates
the core principles of the Expectation-Maximization (EM) algorithm:

Problem Setup:
- Two coins A and B with unknown biases θ_A and θ_B
- Each trial randomly selects one coin (hidden) and flips it multiple times
- We observe only the sequence of heads/tails, not which coin was used
- Goal: Estimate θ_A and θ_B from observed data using EM

Mathematical Foundation:
- E-step: Compute posterior probabilities P(coin=A|data) for each trial
- M-step: Update θ_A and θ_B using weighted maximum likelihood
- Convergence: Guaranteed non-decreasing likelihood at each iteration
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom
import pandas as pd
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CoinFlipEM:
    """
    Expectation-Maximization algorithm for the coin flipping problem.
    
    Features:
    - Classic EM implementation with mathematical rigor
    - Multiple initialization strategies
    - Convergence monitoring and visualization
    - Comprehensive analysis and comparison with ground truth
    - Detailed logging of algorithm progression
    """
    
    def __init__(self, max_iter=100, tol=1e-6, random_state=42):
        """
        Initialize EM algorithm for coin flipping.
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence tolerance for log-likelihood
        random_state : int
            Random seed for reproducibility
        """
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        # Model parameters
        self.theta_A = None
        self.theta_B = None
        
        # Training history
        self.log_likelihood_history = []
        self.theta_A_history = []
        self.theta_B_history = []
        self.n_iter = 0
        self.converged = False
        
        # Data storage
        self.trials = None
        self.n_trials = None
        
    def generate_synthetic_data(self, n_trials=10, flips_per_trial=10, 
                              true_theta_A=0.8, true_theta_B=0.3, 
                              coin_selection_prob=0.5):
        """
        Generate synthetic coin flipping data.
        
        Parameters:
        -----------
        n_trials : int
            Number of experimental trials
        flips_per_trial : int
            Number of coin flips per trial
        true_theta_A : float
            True bias of coin A (probability of heads)
        true_theta_B : float
            True bias of coin B (probability of heads)
        coin_selection_prob : float
            Probability of selecting coin A for each trial
            
        Returns:
        --------
        dict : Generated data with ground truth information
        """
        np.random.seed(self.random_state)
        
        trials = []
        true_coins = []
        
        print(f"Generating synthetic data:")
        print(f"  Number of trials: {n_trials}")
        print(f"  Flips per trial: {flips_per_trial}")
        print(f"  True θ_A (Coin A bias): {true_theta_A}")
        print(f"  True θ_B (Coin B bias): {true_theta_B}")
        print(f"  Coin A selection probability: {coin_selection_prob}")
        
        for trial in range(n_trials):
            # Randomly select which coin to use
            use_coin_A = np.random.random() < coin_selection_prob
            true_coins.append('A' if use_coin_A else 'B')
            
            # Generate flips based on selected coin
            if use_coin_A:
                heads = np.random.binomial(flips_per_trial, true_theta_A)
            else:
                heads = np.random.binomial(flips_per_trial, true_theta_B)
            
            trials.append({
                'trial': trial,
                'heads': heads,
                'tails': flips_per_trial - heads,
                'total_flips': flips_per_trial,
                'true_coin': 'A' if use_coin_A else 'B'
            })
        
        data = {
            'trials': trials,
            'true_theta_A': true_theta_A,
            'true_theta_B': true_theta_B,
            'true_coins': true_coins,
            'n_trials': n_trials,
            'flips_per_trial': flips_per_trial
        }
        
        print(f"  Generated {len(trials)} trials")
        print(f"  Coin A used in {sum(1 for c in true_coins if c == 'A')} trials")
        print(f"  Coin B used in {sum(1 for c in true_coins if c == 'B')} trials")
        
        return data
    
    def initialize_parameters(self, init_method='random'):
        """
        Initialize coin bias parameters.
        
        Parameters:
        -----------
        init_method : str
            Initialization method ('random', 'uniform', 'data_driven')
        """
        np.random.seed(self.random_state)
        
        if init_method == 'random':
            self.theta_A = np.random.uniform(0.1, 0.9)
            self.theta_B = np.random.uniform(0.1, 0.9)
        elif init_method == 'uniform':
            self.theta_A = 0.5
            self.theta_B = 0.5
        elif init_method == 'data_driven':
            # Initialize based on overall data statistics
            total_heads = sum(trial['heads'] for trial in self.trials)
            total_flips = sum(trial['total_flips'] for trial in self.trials)
            overall_rate = total_heads / total_flips
            
            # Add some noise around the overall rate
            self.theta_A = min(0.9, max(0.1, overall_rate + np.random.normal(0, 0.1)))
            self.theta_B = min(0.9, max(0.1, overall_rate + np.random.normal(0, 0.1)))
        
        print(f"Initialized parameters using '{init_method}' method:")
        print(f"  Initial θ_A: {self.theta_A:.4f}")
        print(f"  Initial θ_B: {self.theta_B:.4f}")
        
        # Initialize history
        self.theta_A_history = [self.theta_A]
        self.theta_B_history = [self.theta_B]
    
    def e_step(self):
        """
        Expectation step: Compute posterior probabilities.
        
        For each trial, compute P(coin=A|observed data) using Bayes' theorem:
        P(A|data) = P(data|A) * P(A) / [P(data|A) * P(A) + P(data|B) * P(B)]
        
        Returns:
        --------
        np.ndarray : Posterior probabilities for coin A
        """
        responsibilities = np.zeros(self.n_trials)
        
        for i, trial in enumerate(self.trials):
            heads = trial['heads']
            total_flips = trial['total_flips']
            
            # Compute likelihood of observing this data under each coin
            # P(data|coin) = Binomial(heads; total_flips, theta)
            likelihood_A = binom.pmf(heads, total_flips, self.theta_A)
            likelihood_B = binom.pmf(heads, total_flips, self.theta_B)
            
            # Assume equal prior probabilities P(A) = P(B) = 0.5
            prior_A = 0.5
            prior_B = 0.5
            
            # Compute posterior probabilities using Bayes' theorem
            evidence = likelihood_A * prior_A + likelihood_B * prior_B
            
            if evidence > 0:
                responsibilities[i] = (likelihood_A * prior_A) / evidence
            else:
                responsibilities[i] = 0.5  # Default to equal probability
        
        return responsibilities
    
    def m_step(self, responsibilities):
        """
        Maximization step: Update parameters using weighted MLE.
        
        Parameters:
        -----------
        responsibilities : np.ndarray
            Posterior probabilities from E-step
        """
        # Update θ_A using weighted maximum likelihood
        numerator_A = 0
        denominator_A = 0
        
        # Update θ_B using weighted maximum likelihood  
        numerator_B = 0
        denominator_B = 0
        
        for i, trial in enumerate(self.trials):
            heads = trial['heads']
            total_flips = trial['total_flips']
            
            # Weight for coin A (responsibility)
            weight_A = responsibilities[i]
            weight_B = 1 - responsibilities[i]
            
            # Weighted contributions to θ_A
            numerator_A += weight_A * heads
            denominator_A += weight_A * total_flips
            
            # Weighted contributions to θ_B
            numerator_B += weight_B * heads
            denominator_B += weight_B * total_flips
        
        # Update parameters with regularization to avoid extreme values
        if denominator_A > 0:
            self.theta_A = max(0.001, min(0.999, numerator_A / denominator_A))
        
        if denominator_B > 0:
            self.theta_B = max(0.001, min(0.999, numerator_B / denominator_B))
    
    def compute_log_likelihood(self):
        """
        Compute log-likelihood of observed data given current parameters.
        
        Returns:
        --------
        float : Log-likelihood
        """
        log_likelihood = 0
        
        for trial in self.trials:
            heads = trial['heads']
            total_flips = trial['total_flips']
            
            # Marginal likelihood: P(data) = P(data|A)*P(A) + P(data|B)*P(B)
            likelihood_A = binom.pmf(heads, total_flips, self.theta_A)
            likelihood_B = binom.pmf(heads, total_flips, self.theta_B)
            
            marginal_likelihood = 0.5 * likelihood_A + 0.5 * likelihood_B
            
            if marginal_likelihood > 0:
                log_likelihood += np.log(marginal_likelihood)
            else:
                log_likelihood += -np.inf
        
        return log_likelihood
    
    def fit(self, data, init_method='random'):
        """
        Fit the EM algorithm to coin flipping data.
        
        Parameters:
        -----------
        data : dict
            Data dictionary from generate_synthetic_data()
        init_method : str
            Parameter initialization method
            
        Returns:
        --------
        dict : Results including final parameters and convergence info
        """
        self.trials = data['trials']
        self.n_trials = len(self.trials)
        
        # Initialize parameters
        self.initialize_parameters(init_method)
        
        # Initialize tracking
        self.log_likelihood_history = []
        prev_log_likelihood = -np.inf
        
        print(f"\nStarting EM algorithm...")
        print(f"Convergence tolerance: {self.tol}")
        print(f"Maximum iterations: {self.max_iter}")
        print("-" * 60)
        
        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities
            responsibilities = self.e_step()
            
            # M-step: Update parameters
            self.m_step(responsibilities)
            
            # Compute log-likelihood
            current_log_likelihood = self.compute_log_likelihood()
            self.log_likelihood_history.append(current_log_likelihood)
            
            # Store parameter history
            self.theta_A_history.append(self.theta_A)
            self.theta_B_history.append(self.theta_B)
            
            # Check convergence
            log_likelihood_change = current_log_likelihood - prev_log_likelihood
            
            if iteration % 5 == 0 or iteration < 10:
                print(f"Iter {iteration:3d}: LL = {current_log_likelihood:8.4f}, "
                      f"Change = {log_likelihood_change:8.6f}, "
                      f"θ_A = {self.theta_A:.4f}, θ_B = {self.theta_B:.4f}")
            
            if abs(log_likelihood_change) < self.tol:
                print(f"\nConverged after {iteration + 1} iterations!")
                print(f"Final log-likelihood: {current_log_likelihood:.4f}")
                self.converged = True
                break
                
            prev_log_likelihood = current_log_likelihood
            
        self.n_iter = iteration + 1
        
        if not self.converged:
            print(f"\nMaximum iterations ({self.max_iter}) reached without convergence.")
            print(f"Final log-likelihood: {current_log_likelihood:.4f}")
        
        # Final results
        results = {
            'theta_A_estimated': self.theta_A,
            'theta_B_estimated': self.theta_B,
            'theta_A_true': data['true_theta_A'],
            'theta_B_true': data['true_theta_B'],
            'log_likelihood_final': current_log_likelihood,
            'n_iterations': self.n_iter,
            'converged': self.converged,
            'responsibilities_final': responsibilities
        }
        
        print(f"\nFinal Results:")
        print(f"  Estimated θ_A: {self.theta_A:.4f} (True: {data['true_theta_A']:.4f})")
        print(f"  Estimated θ_B: {self.theta_B:.4f} (True: {data['true_theta_B']:.4f})")
        print(f"  Error θ_A: {abs(self.theta_A - data['true_theta_A']):.4f}")
        print(f"  Error θ_B: {abs(self.theta_B - data['true_theta_B']):.4f}")
        
        return results
    
    def plot_convergence(self, data, save_path=None):
        """
        Plot convergence analysis including log-likelihood and parameter evolution.
        
        Parameters:
        -----------
        data : dict
            Original data with ground truth
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Log-likelihood convergence
        ax1 = axes[0, 0]
        ax1.plot(self.log_likelihood_history, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('EM Iteration')
        ax1.set_ylabel('Log-Likelihood')
        ax1.set_title('EM Convergence: Log-Likelihood')
        ax1.grid(True, alpha=0.3)
        
        if self.converged:
            ax1.axvline(x=len(self.log_likelihood_history)-1, color='r', linestyle='--', 
                       alpha=0.7, label=f'Converged at iteration {self.n_iter}')
            ax1.legend()
        
        # Plot 2: Parameter evolution
        ax2 = axes[0, 1]
        iterations = range(len(self.theta_A_history))
        ax2.plot(iterations, self.theta_A_history, 'b-', linewidth=2, marker='o', 
                markersize=4, label='θ_A (estimated)')
        ax2.plot(iterations, self.theta_B_history, 'r-', linewidth=2, marker='s', 
                markersize=4, label='θ_B (estimated)')
        
        # Add true values as horizontal lines
        ax2.axhline(y=data['true_theta_A'], color='blue', linestyle='--', alpha=0.7, 
                   label=f'θ_A (true) = {data["true_theta_A"]:.3f}')
        ax2.axhline(y=data['true_theta_B'], color='red', linestyle='--', alpha=0.7, 
                   label=f'θ_B (true) = {data["true_theta_B"]:.3f}')
        
        ax2.set_xlabel('EM Iteration')
        ax2.set_ylabel('Parameter Value')
        ax2.set_title('Parameter Evolution During EM')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Trial assignments
        ax3 = axes[1, 0]
        responsibilities = self.e_step()
        trial_numbers = range(len(responsibilities))
        
        # Color code by true coin
        colors = ['blue' if coin == 'A' else 'red' for coin in data['true_coins']]
        scatter = ax3.scatter(trial_numbers, responsibilities, c=colors, alpha=0.7, s=60)
        
        ax3.set_xlabel('Trial Number')
        ax3.set_ylabel('P(Coin A | Data)')
        ax3.set_title('Final Coin Assignments\n(Blue=True A, Red=True B)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Add decision boundary
        ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Decision boundary')
        ax3.legend()
        
        # Plot 4: Accuracy analysis
        ax4 = axes[1, 1]
        
        # Compute assignment accuracy
        predicted_coins = ['A' if r > 0.5 else 'B' for r in responsibilities]
        true_coins = data['true_coins']
        
        correct_assignments = sum(1 for p, t in zip(predicted_coins, true_coins) if p == t)
        accuracy = correct_assignments / len(true_coins)
        
        # Create confusion matrix
        confusion = {'AA': 0, 'AB': 0, 'BA': 0, 'BB': 0}
        for p, t in zip(predicted_coins, true_coins):
            confusion[t + p] += 1
        
        # Plot confusion matrix
        conf_matrix = np.array([[confusion['AA'], confusion['AB']], 
                               [confusion['BA'], confusion['BB']]])
        
        im = ax4.imshow(conf_matrix, cmap='Blues', alpha=0.7)
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(['Pred A', 'Pred B'])
        ax4.set_yticklabels(['True A', 'True B'])
        ax4.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                ax4.text(j, i, conf_matrix[i, j], ha='center', va='center', 
                        fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()
    
    def analyze_sensitivity(self, data, n_runs=20):
        """
        Analyze sensitivity to initialization.
        
        Parameters:
        -----------
        data : dict
            Original data
        n_runs : int
            Number of random initializations to try
            
        Returns:
        --------
        dict : Sensitivity analysis results
        """
        print(f"\nRunning sensitivity analysis with {n_runs} random initializations...")
        
        results = {
            'theta_A_estimates': [],
            'theta_B_estimates': [],
            'log_likelihoods': [],
            'iterations': [],
            'converged_runs': 0
        }
        
        original_random_state = self.random_state
        
        for run in range(n_runs):
            # Use different random seed for each run
            self.random_state = original_random_state + run
            
            # Reset algorithm state
            self.log_likelihood_history = []
            self.theta_A_history = []
            self.theta_B_history = []
            self.converged = False
            
            # Fit with random initialization
            run_results = self.fit(data, init_method='random')
            
            # Store results
            results['theta_A_estimates'].append(run_results['theta_A_estimated'])
            results['theta_B_estimates'].append(run_results['theta_B_estimated'])
            results['log_likelihoods'].append(run_results['log_likelihood_final'])
            results['iterations'].append(run_results['n_iterations'])
            
            if run_results['converged']:
                results['converged_runs'] += 1
        
        # Restore original random state
        self.random_state = original_random_state
        
        # Compute statistics
        results['theta_A_mean'] = np.mean(results['theta_A_estimates'])
        results['theta_A_std'] = np.std(results['theta_A_estimates'])
        results['theta_B_mean'] = np.mean(results['theta_B_estimates'])
        results['theta_B_std'] = np.std(results['theta_B_estimates'])
        results['convergence_rate'] = results['converged_runs'] / n_runs
        
        print(f"Sensitivity Analysis Results:")
        print(f"  θ_A: {results['theta_A_mean']:.4f} ± {results['theta_A_std']:.4f}")
        print(f"  θ_B: {results['theta_B_mean']:.4f} ± {results['theta_B_std']:.4f}")
        print(f"  Convergence rate: {results['convergence_rate']:.1%}")
        print(f"  True θ_A: {data['true_theta_A']:.4f}")
        print(f"  True θ_B: {data['true_theta_B']:.4f}")
        
        return results

def demonstrate_coin_flip_em():
    """
    Comprehensive demonstration of EM algorithm for coin flipping.
    """
    print("🪙 EM ALGORITHM FOR COIN FLIPPING PROBLEM")
    print("=" * 60)
    
    # Create EM instance
    em = CoinFlipEM(max_iter=100, tol=1e-6, random_state=42)
    
    # Generate synthetic data
    data = em.generate_synthetic_data(
        n_trials=20,
        flips_per_trial=10,
        true_theta_A=0.8,
        true_theta_B=0.3,
        coin_selection_prob=0.5
    )
    
    # Fit EM algorithm
    results = em.fit(data, init_method='random')
    
    # Create convergence visualization
    em.plot_convergence(data, 'plots/coin_flip_convergence.png')
    
    # Sensitivity analysis
    sensitivity = em.analyze_sensitivity(data, n_runs=20)
    
    return em, data, results, sensitivity

def compare_different_scenarios():
    """
    Compare EM performance across different problem scenarios.
    """
    print("\n📊 COMPARING DIFFERENT SCENARIOS")
    print("=" * 50)
    
    scenarios = [
        {'name': 'Well-separated coins', 'theta_A': 0.8, 'theta_B': 0.2, 'trials': 15},
        {'name': 'Similar coins', 'theta_A': 0.6, 'theta_B': 0.4, 'trials': 15},
        {'name': 'Extreme bias', 'theta_A': 0.95, 'theta_B': 0.05, 'trials': 15},
        {'name': 'More data', 'theta_A': 0.7, 'theta_B': 0.3, 'trials': 50}
    ]
    
    results_comparison = []
    
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}: {scenario['name']}")
        print("-" * 40)
        
        em = CoinFlipEM(max_iter=100, tol=1e-6, random_state=42)
        
        data = em.generate_synthetic_data(
            n_trials=scenario['trials'],
            flips_per_trial=10,
            true_theta_A=scenario['theta_A'],
            true_theta_B=scenario['theta_B']
        )
        
        results = em.fit(data, init_method='random')
        
        results_comparison.append({
            'scenario': scenario['name'],
            'true_theta_A': scenario['theta_A'],
            'true_theta_B': scenario['theta_B'],
            'estimated_theta_A': results['theta_A_estimated'],
            'estimated_theta_B': results['theta_B_estimated'],
            'error_A': abs(results['theta_A_estimated'] - scenario['theta_A']),
            'error_B': abs(results['theta_B_estimated'] - scenario['theta_B']),
            'iterations': results['n_iterations'],
            'converged': results['converged']
        })
    
    # Create comparison visualization
    plot_scenario_comparison(results_comparison)
    
    return results_comparison

def plot_scenario_comparison(results):
    """
    Plot comparison across different scenarios.
    
    Parameters:
    -----------
    results : list
        List of scenario results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    scenarios = [r['scenario'] for r in results]
    
    # Plot 1: Parameter estimation accuracy
    ax1 = axes[0, 0]
    errors_A = [r['error_A'] for r in results]
    errors_B = [r['error_B'] for r in results]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax1.bar(x - width/2, errors_A, width, label='θ_A Error', alpha=0.7)
    ax1.bar(x + width/2, errors_B, width, label='θ_B Error', alpha=0.7)
    
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Parameter Estimation Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Convergence speed
    ax2 = axes[0, 1]
    iterations = [r['iterations'] for r in results]
    colors = ['green' if r['converged'] else 'red' for r in results]
    
    bars = ax2.bar(scenarios, iterations, color=colors, alpha=0.7)
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Iterations to Convergence')
    ax2.set_title('Convergence Speed')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add convergence status annotations
    for bar, converged in zip(bars, [r['converged'] for r in results]):
        height = bar.get_height()
        status = 'Converged' if converged else 'Max iter'
        ax2.annotate(status, xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # Plot 3: True vs Estimated (θ_A)
    ax3 = axes[1, 0]
    true_A = [r['true_theta_A'] for r in results]
    est_A = [r['estimated_theta_A'] for r in results]
    
    ax3.scatter(true_A, est_A, s=100, alpha=0.7, c='blue')
    ax3.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect estimation')
    ax3.set_xlabel('True θ_A')
    ax3.set_ylabel('Estimated θ_A')
    ax3.set_title('θ_A: True vs Estimated')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Plot 4: True vs Estimated (θ_B)
    ax4 = axes[1, 1]
    true_B = [r['true_theta_B'] for r in results]
    est_B = [r['estimated_theta_B'] for r in results]
    
    ax4.scatter(true_B, est_B, s=100, alpha=0.7, c='red')
    ax4.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect estimation')
    ax4.set_xlabel('True θ_B')
    ax4.set_ylabel('Estimated θ_B')
    ax4.set_title('θ_B: True vs Estimated')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('plots/scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run comprehensive coin flipping EM analysis.
    """
    print("🔬 EXPECTATION-MAXIMIZATION: COIN FLIPPING PROBLEM")
    print("=" * 80)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # 1. Basic demonstration
    em, data, results, sensitivity = demonstrate_coin_flip_em()
    
    # 2. Compare different scenarios
    scenario_results = compare_different_scenarios()
    
    # 3. Summary
    print("\n✅ COIN FLIPPING EM ANALYSIS COMPLETE!")
    print("📁 Check the 'plots' folder for generated visualizations.")
    print("\n📋 SUMMARY:")
    print(f"• Demonstrated classic EM algorithm on coin flipping problem")
    print(f"• Analyzed convergence behavior and parameter estimation accuracy")
    print(f"• Tested sensitivity to initialization across {len(scenario_results)} scenarios")
    print(f"• Generated comprehensive visualizations and analysis")
    
    return em, data, results, scenario_results

if __name__ == "__main__":
    main() 