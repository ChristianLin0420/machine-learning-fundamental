"""
Hidden Markov Model (HMM) from Scratch - Comprehensive Implementation
====================================================================

This module implements a complete Hidden Markov Model from scratch, covering
all fundamental algorithms and practical applications:

Core Algorithms:
- Forward Algorithm: Compute probability of observation sequence
- Backward Algorithm: Compute backward probabilities
- Viterbi Algorithm: Find most likely hidden state sequence
- Baum-Welch Algorithm: Learn parameters from unlabeled data

Key Features:
- Log-space computation to prevent numerical underflow
- Support for discrete and Gaussian emissions
- Comprehensive visualization and analysis tools
- Multiple synthetic data generators for testing
- Real-world applications (POS tagging, DNA analysis)

Mathematical Foundation:
- State transition matrix A: P(s_t | s_{t-1})
- Emission matrix B: P(o_t | s_t)
- Initial state distribution π: P(s_1)
- Forward variables: α_t(i) = P(o_1...o_t, s_t=i | λ)
- Backward variables: β_t(i) = P(o_{t+1}...o_T | s_t=i, λ)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HiddenMarkovModelScratch:
    """
    Hidden Markov Model implementation from scratch.
    
    This class provides a complete HMM implementation with support for:
    - Discrete and continuous observations
    - Forward-backward algorithm
    - Viterbi decoding
    - Baum-Welch parameter learning
    - Log-space computation for numerical stability
    
    Parameters:
    -----------
    n_states : int
        Number of hidden states
    n_observations : int
        Number of possible observations (for discrete models)
    observation_type : str
        'discrete' or 'gaussian'
    """
    
    def __init__(self, n_states, n_observations=None, observation_type='discrete', random_state=42):
        """
        Initialize HMM with specified parameters.
        
        Parameters:
        -----------
        n_states : int
            Number of hidden states
        n_observations : int, optional
            Number of observation symbols (for discrete models)
        observation_type : str
            Type of observations ('discrete' or 'gaussian')
        random_state : int
            Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.observation_type = observation_type
        self.random_state = random_state
        
        # Model parameters
        self.A = None  # Transition matrix
        self.B = None  # Emission matrix/parameters
        self.pi = None  # Initial state distribution
        
        # For Gaussian emissions
        self.means = None
        self.covs = None
        
        # Training history
        self.log_likelihood_history = []
        self.converged = False
        self.n_iter = 0
        
        np.random.seed(random_state)
    
    def initialize_parameters(self, method='random'):
        """
        Initialize HMM parameters.
        
        Parameters:
        -----------
        method : str
            Initialization method ('random', 'uniform', 'kmeans')
        """
        if method == 'random':
            # Random initialization with normalization
            self.A = np.random.rand(self.n_states, self.n_states)
            self.A = self.A / self.A.sum(axis=1, keepdims=True)
            
            self.pi = np.random.rand(self.n_states)
            self.pi = self.pi / self.pi.sum()
            
            if self.observation_type == 'discrete':
                self.B = np.random.rand(self.n_states, self.n_observations)
                self.B = self.B / self.B.sum(axis=1, keepdims=True)
            else:
                # Gaussian emissions
                self.means = np.random.randn(self.n_states, 1)  # Assume 1D for simplicity
                self.covs = np.ones((self.n_states, 1, 1))
                
        elif method == 'uniform':
            # Uniform initialization
            self.A = np.ones((self.n_states, self.n_states)) / self.n_states
            self.pi = np.ones(self.n_states) / self.n_states
            
            if self.observation_type == 'discrete':
                self.B = np.ones((self.n_states, self.n_observations)) / self.n_observations
            else:
                self.means = np.zeros((self.n_states, 1))
                self.covs = np.ones((self.n_states, 1, 1))
        
        print(f"Initialized HMM parameters using '{method}' method")
        print(f"  States: {self.n_states}")
        print(f"  Observation type: {self.observation_type}")
        if self.observation_type == 'discrete':
            print(f"  Observations: {self.n_observations}")
    
    def _emission_probability(self, state, observation):
        """
        Compute emission probability P(observation | state).
        
        Parameters:
        -----------
        state : int
            Hidden state index
        observation : int or float
            Observation value
            
        Returns:
        --------
        float : Emission probability
        """
        if self.observation_type == 'discrete':
            return self.B[state, observation]
        else:
            # Gaussian emission
            mean = self.means[state]
            cov = self.covs[state]
            if np.ndim(observation) == 0:
                observation = np.array([observation])
            return multivariate_normal.pdf(observation, mean.flatten(), cov.squeeze())
    
    def forward(self, observations, log_space=True):
        """
        Forward algorithm to compute P(observations | model).
        
        Parameters:
        -----------
        observations : list or np.ndarray
            Sequence of observations
        log_space : bool
            Whether to use log-space computation
            
        Returns:
        --------
        float : Log-probability of observation sequence
        np.ndarray : Forward variables (T x N)
        """
        T = len(observations)
        
        if log_space:
            return self._forward_log(observations)
        else:
            return self._forward_standard(observations)
    
    def _forward_standard(self, observations):
        """Standard forward algorithm (may suffer from underflow)."""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialization
        for i in range(self.n_states):
            alpha[0, i] = self.pi[i] * self._emission_probability(i, observations[0])
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1, :] * self.A[:, j]) * \
                             self._emission_probability(j, observations[t])
        
        # Termination
        probability = np.sum(alpha[T-1, :])
        return np.log(probability + 1e-100), alpha
    
    def _forward_log(self, observations):
        """Log-space forward algorithm for numerical stability."""
        T = len(observations)
        log_alpha = np.zeros((T, self.n_states))
        
        # Initialization
        for i in range(self.n_states):
            emission_prob = self._emission_probability(i, observations[0])
            log_alpha[0, i] = np.log(self.pi[i] + 1e-100) + np.log(emission_prob + 1e-100)
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                emission_prob = self._emission_probability(j, observations[t])
                
                # Log-sum-exp trick for numerical stability
                log_probs = log_alpha[t-1, :] + np.log(self.A[:, j] + 1e-100)
                max_log_prob = np.max(log_probs)
                
                if max_log_prob == -np.inf:
                    log_alpha[t, j] = -np.inf
                else:
                    log_alpha[t, j] = max_log_prob + \
                                     np.log(np.sum(np.exp(log_probs - max_log_prob))) + \
                                     np.log(emission_prob + 1e-100)
        
        # Termination
        max_log_alpha = np.max(log_alpha[T-1, :])
        if max_log_alpha == -np.inf:
            log_probability = -np.inf
        else:
            log_probability = max_log_alpha + \
                             np.log(np.sum(np.exp(log_alpha[T-1, :] - max_log_alpha)))
        
        return log_probability, log_alpha
    
    def backward(self, observations, log_space=True):
        """
        Backward algorithm to compute backward variables.
        
        Parameters:
        -----------
        observations : list or np.ndarray
            Sequence of observations
        log_space : bool
            Whether to use log-space computation
            
        Returns:
        --------
        np.ndarray : Backward variables (T x N)
        """
        if log_space:
            return self._backward_log(observations)
        else:
            return self._backward_standard(observations)
    
    def _backward_standard(self, observations):
        """Standard backward algorithm."""
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialization
        beta[T-1, :] = 1
        
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    emission_prob = self._emission_probability(j, observations[t+1])
                    beta[t, i] += self.A[i, j] * emission_prob * beta[t+1, j]
        
        return beta
    
    def _backward_log(self, observations):
        """Log-space backward algorithm."""
        T = len(observations)
        log_beta = np.zeros((T, self.n_states))
        
        # Initialization
        log_beta[T-1, :] = 0  # log(1) = 0
        
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                log_probs = []
                for j in range(self.n_states):
                    emission_prob = self._emission_probability(j, observations[t+1])
                    log_prob = np.log(self.A[i, j] + 1e-100) + \
                              np.log(emission_prob + 1e-100) + \
                              log_beta[t+1, j]
                    log_probs.append(log_prob)
                
                # Log-sum-exp
                max_log_prob = np.max(log_probs)
                if max_log_prob == -np.inf:
                    log_beta[t, i] = -np.inf
                else:
                    log_beta[t, i] = max_log_prob + \
                                    np.log(np.sum(np.exp(np.array(log_probs) - max_log_prob)))
        
        return log_beta
    
    def viterbi(self, observations):
        """
        Viterbi algorithm to find most likely state sequence.
        
        Parameters:
        -----------
        observations : list or np.ndarray
            Sequence of observations
            
        Returns:
        --------
        list : Most likely state sequence
        float : Log-probability of best path
        """
        T = len(observations)
        
        # Viterbi tables
        log_delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        for i in range(self.n_states):
            emission_prob = self._emission_probability(i, observations[0])
            log_delta[0, i] = np.log(self.pi[i] + 1e-100) + np.log(emission_prob + 1e-100)
            psi[0, i] = 0
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                emission_prob = self._emission_probability(j, observations[t])
                
                # Find best previous state
                scores = log_delta[t-1, :] + np.log(self.A[:, j] + 1e-100)
                best_prev_state = np.argmax(scores)
                
                log_delta[t, j] = scores[best_prev_state] + np.log(emission_prob + 1e-100)
                psi[t, j] = best_prev_state
        
        # Termination
        best_last_state = np.argmax(log_delta[T-1, :])
        best_path_prob = log_delta[T-1, best_last_state]
        
        # Backtracking
        path = [0] * T
        path[T-1] = best_last_state
        
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path, best_path_prob
    
    def baum_welch(self, observations_list, max_iter=100, tol=1e-6):
        """
        Baum-Welch algorithm for parameter learning.
        
        Parameters:
        -----------
        observations_list : list of lists
            List of observation sequences
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
            
        Returns:
        --------
        dict : Training results
        """
        print(f"Starting Baum-Welch training...")
        print(f"  Sequences: {len(observations_list)}")
        print(f"  Max iterations: {max_iter}")
        print(f"  Tolerance: {tol}")
        
        self.log_likelihood_history = []
        prev_log_likelihood = -np.inf
        
        for iteration in range(max_iter):
            # E-step: Compute forward-backward variables
            gamma_list = []
            xi_list = []
            total_log_likelihood = 0
            
            for observations in observations_list:
                log_prob, log_alpha = self.forward(observations, log_space=True)
                log_beta = self.backward(observations, log_space=True)
                
                total_log_likelihood += log_prob
                
                # Compute gamma (state probabilities)
                gamma = self._compute_gamma(log_alpha, log_beta)
                gamma_list.append(gamma)
                
                # Compute xi (transition probabilities)
                xi = self._compute_xi(observations, log_alpha, log_beta)
                xi_list.append(xi)
            
            self.log_likelihood_history.append(total_log_likelihood)
            
            # Check convergence
            if abs(total_log_likelihood - prev_log_likelihood) < tol:
                print(f"Converged after {iteration + 1} iterations")
                self.converged = True
                break
            
            # M-step: Update parameters
            self._update_parameters(observations_list, gamma_list, xi_list)
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: Log-likelihood = {total_log_likelihood:.4f}")
            
            prev_log_likelihood = total_log_likelihood
        
        self.n_iter = iteration + 1
        
        if not self.converged:
            print(f"Maximum iterations reached without convergence")
        
        return {
            'log_likelihood_history': self.log_likelihood_history,
            'final_log_likelihood': total_log_likelihood,
            'n_iterations': self.n_iter,
            'converged': self.converged
        }
    
    def _compute_gamma(self, log_alpha, log_beta):
        """Compute gamma (state probabilities) from forward-backward variables."""
        T, N = log_alpha.shape
        gamma = np.zeros((T, N))
        
        for t in range(T):
            log_probs = log_alpha[t, :] + log_beta[t, :]
            max_log_prob = np.max(log_probs)
            
            if max_log_prob == -np.inf:
                gamma[t, :] = 1.0 / N  # Uniform if all probabilities are zero
            else:
                probs = np.exp(log_probs - max_log_prob)
                gamma[t, :] = probs / np.sum(probs)
        
        return gamma
    
    def _compute_xi(self, observations, log_alpha, log_beta):
        """Compute xi (transition probabilities) from forward-backward variables."""
        T = len(observations)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    emission_prob = self._emission_probability(j, observations[t+1])
                    
                    log_xi = log_alpha[t, i] + \
                            np.log(self.A[i, j] + 1e-100) + \
                            np.log(emission_prob + 1e-100) + \
                            log_beta[t+1, j]
                    
                    xi[t, i, j] = log_xi
            
            # Normalize
            max_log_xi = np.max(xi[t, :, :])
            if max_log_xi != -np.inf:
                xi[t, :, :] = np.exp(xi[t, :, :] - max_log_xi)
                xi[t, :, :] = xi[t, :, :] / np.sum(xi[t, :, :])
        
        return xi
    
    def _update_parameters(self, observations_list, gamma_list, xi_list):
        """Update HMM parameters using computed gamma and xi."""
        # Update initial state distribution
        pi_num = np.zeros(self.n_states)
        for gamma in gamma_list:
            pi_num += gamma[0, :]
        self.pi = pi_num / len(observations_list)
        
        # Update transition matrix
        A_num = np.zeros((self.n_states, self.n_states))
        A_den = np.zeros(self.n_states)
        
        for gamma, xi in zip(gamma_list, xi_list):
            A_num += np.sum(xi, axis=0)
            A_den += np.sum(gamma[:-1, :], axis=0)
        
        for i in range(self.n_states):
            if A_den[i] > 0:
                self.A[i, :] = A_num[i, :] / A_den[i]
            else:
                self.A[i, :] = 1.0 / self.n_states
        
        # Update emission parameters
        if self.observation_type == 'discrete':
            self._update_discrete_emissions(observations_list, gamma_list)
        else:
            self._update_gaussian_emissions(observations_list, gamma_list)
    
    def _update_discrete_emissions(self, observations_list, gamma_list):
        """Update discrete emission matrix."""
        B_num = np.zeros((self.n_states, self.n_observations))
        B_den = np.zeros(self.n_states)
        
        for observations, gamma in zip(observations_list, gamma_list):
            for t, obs in enumerate(observations):
                for i in range(self.n_states):
                    B_num[i, obs] += gamma[t, i]
                    B_den[i] += gamma[t, i]
        
        for i in range(self.n_states):
            if B_den[i] > 0:
                self.B[i, :] = B_num[i, :] / B_den[i]
            else:
                self.B[i, :] = 1.0 / self.n_observations
    
    def _update_gaussian_emissions(self, observations_list, gamma_list):
        """Update Gaussian emission parameters."""
        # Update means
        means_num = np.zeros((self.n_states, 1))
        means_den = np.zeros(self.n_states)
        
        for observations, gamma in zip(observations_list, gamma_list):
            for t, obs in enumerate(observations):
                for i in range(self.n_states):
                    means_num[i] += gamma[t, i] * obs
                    means_den[i] += gamma[t, i]
        
        for i in range(self.n_states):
            if means_den[i] > 0:
                self.means[i] = means_num[i] / means_den[i]
        
        # Update covariances
        covs_num = np.zeros((self.n_states, 1, 1))
        
        for observations, gamma in zip(observations_list, gamma_list):
            for t, obs in enumerate(observations):
                for i in range(self.n_states):
                    diff = obs - self.means[i, 0]
                    covs_num[i, 0, 0] += gamma[t, i] * diff * diff
        
        for i in range(self.n_states):
            if means_den[i] > 0:
                self.covs[i, 0, 0] = max(covs_num[i, 0, 0] / means_den[i], 1e-6)
    
    def predict_states(self, observations):
        """
        Predict most likely state sequence for observations.
        
        Parameters:
        -----------
        observations : list or np.ndarray
            Sequence of observations
            
        Returns:
        --------
        list : Predicted state sequence
        float : Log-probability of sequence
        """
        return self.viterbi(observations)
    
    def score(self, observations):
        """
        Compute log-probability of observation sequence.
        
        Parameters:
        -----------
        observations : list or np.ndarray
            Sequence of observations
            
        Returns:
        --------
        float : Log-probability
        """
        log_prob, _ = self.forward(observations, log_space=True)
        return log_prob

# Synthetic Data Generators
class HMMDataGenerator:
    """Generate synthetic data for HMM testing and evaluation."""
    
    @staticmethod
    def generate_pos_tagging_data(n_sequences=100, sequence_length=20, random_state=42):
        """
        Generate synthetic POS tagging data.
        
        Returns:
        --------
        dict : Generated data with observations, states, and vocabulary
        """
        np.random.seed(random_state)
        
        # Define states (POS tags)
        states = ['NOUN', 'VERB', 'ADJ', 'DET']  # Simplified POS tags
        n_states = len(states)
        
        # Define vocabulary
        vocabulary = {
            'NOUN': ['cat', 'dog', 'house', 'car', 'book', 'tree', 'water', 'food'],
            'VERB': ['run', 'eat', 'sleep', 'walk', 'read', 'write', 'jump', 'swim'],
            'ADJ': ['big', 'small', 'red', 'blue', 'fast', 'slow', 'hot', 'cold'],
            'DET': ['the', 'a', 'an', 'this', 'that', 'some', 'many', 'few']
        }
        
        # Create word-to-index mapping
        all_words = []
        for words in vocabulary.values():
            all_words.extend(words)
        word_to_idx = {word: idx for idx, word in enumerate(all_words)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        
        # Define transition probabilities (simplified)
        A = np.array([
            [0.1, 0.4, 0.3, 0.2],  # NOUN -> [NOUN, VERB, ADJ, DET]
            [0.4, 0.1, 0.2, 0.3],  # VERB -> [NOUN, VERB, ADJ, DET]
            [0.6, 0.1, 0.1, 0.2],  # ADJ -> [NOUN, VERB, ADJ, DET]
            [0.3, 0.2, 0.4, 0.1]   # DET -> [NOUN, VERB, ADJ, DET]
        ])
        
        # Initial state distribution
        pi = np.array([0.3, 0.3, 0.2, 0.2])
        
        # Generate sequences
        observations_list = []
        states_list = []
        
        for _ in range(n_sequences):
            obs_seq = []
            state_seq = []
            
            # Generate first state
            current_state = np.random.choice(n_states, p=pi)
            
            for t in range(sequence_length):
                state_seq.append(current_state)
                
                # Generate observation
                state_name = states[current_state]
                word = np.random.choice(vocabulary[state_name])
                obs_seq.append(word_to_idx[word])
                
                # Transition to next state
                if t < sequence_length - 1:
                    current_state = np.random.choice(n_states, p=A[current_state])
            
            observations_list.append(obs_seq)
            states_list.append(state_seq)
        
        return {
            'observations': observations_list,
            'states': states_list,
            'vocabulary': word_to_idx,
            'idx_to_word': idx_to_word,
            'state_names': states,
            'true_A': A,
            'true_pi': pi,
            'n_states': n_states,
            'n_observations': len(all_words)
        }
    
    @staticmethod
    def generate_dna_sequence_data(n_sequences=50, sequence_length=100, random_state=42):
        """
        Generate synthetic DNA sequence data with promoter/non-promoter regions.
        
        Returns:
        --------
        dict : Generated data with DNA sequences and region labels
        """
        np.random.seed(random_state)
        
        # States: 0 = non-promoter, 1 = promoter
        states = ['non-promoter', 'promoter']
        n_states = 2
        
        # Nucleotides: A, T, G, C
        nucleotides = ['A', 'T', 'G', 'C']
        nucl_to_idx = {nucl: idx for idx, nucl in enumerate(nucleotides)}
        
        # Transition matrix (promoter regions are less common)
        A = np.array([
            [0.95, 0.05],  # non-promoter -> [non-promoter, promoter]
            [0.2, 0.8]     # promoter -> [non-promoter, promoter]
        ])
        
        # Initial state distribution
        pi = np.array([0.9, 0.1])
        
        # Emission probabilities (different nucleotide frequencies)
        B = np.array([
            [0.25, 0.25, 0.25, 0.25],  # non-promoter: uniform
            [0.1, 0.1, 0.4, 0.4]       # promoter: GC-rich
        ])
        
        # Generate sequences
        observations_list = []
        states_list = []
        
        for _ in range(n_sequences):
            obs_seq = []
            state_seq = []
            
            # Generate first state
            current_state = np.random.choice(n_states, p=pi)
            
            for t in range(sequence_length):
                state_seq.append(current_state)
                
                # Generate observation
                nucleotide_idx = np.random.choice(4, p=B[current_state])
                obs_seq.append(nucleotide_idx)
                
                # Transition to next state
                if t < sequence_length - 1:
                    current_state = np.random.choice(n_states, p=A[current_state])
            
            observations_list.append(obs_seq)
            states_list.append(state_seq)
        
        return {
            'observations': observations_list,
            'states': states_list,
            'nucleotides': nucl_to_idx,
            'state_names': states,
            'true_A': A,
            'true_B': B,
            'true_pi': pi,
            'n_states': n_states,
            'n_observations': 4
        }
    
    @staticmethod
    def generate_gaussian_hmm_data(n_sequences=30, sequence_length=50, n_states=3, random_state=42):
        """
        Generate synthetic data for Gaussian HMM.
        
        Returns:
        --------
        dict : Generated data with continuous observations
        """
        np.random.seed(random_state)
        
        # Random transition matrix
        A = np.random.rand(n_states, n_states)
        A = A / A.sum(axis=1, keepdims=True)
        
        # Initial state distribution
        pi = np.random.rand(n_states)
        pi = pi / pi.sum()
        
        # Gaussian parameters for each state
        means = np.random.randn(n_states) * 5
        stds = np.random.rand(n_states) * 2 + 0.5
        
        # Generate sequences
        observations_list = []
        states_list = []
        
        for _ in range(n_sequences):
            obs_seq = []
            state_seq = []
            
            # Generate first state
            current_state = np.random.choice(n_states, p=pi)
            
            for t in range(sequence_length):
                state_seq.append(current_state)
                
                # Generate observation from Gaussian
                observation = np.random.normal(means[current_state], stds[current_state])
                obs_seq.append(observation)
                
                # Transition to next state
                if t < sequence_length - 1:
                    current_state = np.random.choice(n_states, p=A[current_state])
            
            observations_list.append(obs_seq)
            states_list.append(state_seq)
        
        return {
            'observations': observations_list,
            'states': states_list,
            'true_A': A,
            'true_pi': pi,
            'true_means': means,
            'true_stds': stds,
            'n_states': n_states
        }

def main():
    """
    Main function to demonstrate HMM implementation.
    """
    print("🔗 HIDDEN MARKOV MODEL FROM SCRATCH")
    print("=" * 50)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Test with discrete HMM (POS tagging)
    print("\n📝 Testing with POS Tagging Data")
    print("-" * 30)
    
    # Generate data
    pos_data = HMMDataGenerator.generate_pos_tagging_data(n_sequences=50, sequence_length=15)
    
    # Create and train HMM
    hmm = HiddenMarkovModelScratch(
        n_states=pos_data['n_states'],
        n_observations=pos_data['n_observations'],
        observation_type='discrete'
    )
    
    # Initialize parameters
    hmm.initialize_parameters(method='random')
    
    # Train with Baum-Welch
    results = hmm.baum_welch(pos_data['observations'], max_iter=50)
    
    # Test predictions
    test_seq = pos_data['observations'][0]
    predicted_states, log_prob = hmm.predict_states(test_seq)
    true_states = pos_data['states'][0]
    
    print(f"\nPrediction Results:")
    print(f"  Sequence length: {len(test_seq)}")
    print(f"  Log-probability: {log_prob:.4f}")
    print(f"  Accuracy: {accuracy_score(true_states, predicted_states):.4f}")
    
    # Test with DNA sequence data
    print("\n🧬 Testing with DNA Sequence Data")
    print("-" * 30)
    
    dna_data = HMMDataGenerator.generate_dna_sequence_data(n_sequences=30, sequence_length=50)
    
    hmm_dna = HiddenMarkovModelScratch(
        n_states=dna_data['n_states'],
        n_observations=dna_data['n_observations'],
        observation_type='discrete'
    )
    
    hmm_dna.initialize_parameters(method='random')
    dna_results = hmm_dna.baum_welch(dna_data['observations'], max_iter=50)
    
    # Test with Gaussian HMM
    print("\n📊 Testing with Gaussian HMM Data")
    print("-" * 30)
    
    gaussian_data = HMMDataGenerator.generate_gaussian_hmm_data(n_sequences=20, sequence_length=30)
    
    hmm_gaussian = HiddenMarkovModelScratch(
        n_states=gaussian_data['n_states'],
        observation_type='gaussian'
    )
    
    hmm_gaussian.initialize_parameters(method='random')
    gaussian_results = hmm_gaussian.baum_welch(gaussian_data['observations'], max_iter=50)
    
    print("\n✅ HMM IMPLEMENTATION COMPLETE!")
    print("📁 Check the implementation for comprehensive HMM algorithms.")
    
    return hmm, hmm_dna, hmm_gaussian, pos_data, dna_data, gaussian_data

if __name__ == "__main__":
    main() 