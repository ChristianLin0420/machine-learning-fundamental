#!/usr/bin/env python3
"""
Sequence classification models implemented from scratch using NumPy.

This module provides various sequence encoder architectures for classification:
- Simple RNN classifier
- LSTM classifier  
- GRU classifier
- Bidirectional LSTM
- LSTM with attention mechanism
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import pickle
from abc import ABC, abstractmethod

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 
                   1 / (1 + np.exp(-x)), 
                   np.exp(x) / (1 + np.exp(x)))

def tanh(x):
    """Hyperbolic tangent activation."""
    return np.tanh(x)

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def softmax(x, axis=-1):
    """Numerically stable softmax function."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def xavier_init(shape):
    """Xavier/Glorot initialization."""
    fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else shape[0]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)

def he_init(shape):
    """He initialization for ReLU activations."""
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, shape)


class BaseSequenceClassifier(ABC):
    """Abstract base class for sequence classifiers."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_classes: int, max_length: int):
        """
        Initialize base sequence classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            n_classes: Number of output classes
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.max_length = max_length
        
        # Initialize embeddings
        self.embeddings = xavier_init((vocab_size, embed_dim))
        
        # Initialize classifier layer
        self.W_out = xavier_init((hidden_dim, n_classes))
        self.b_out = np.zeros((n_classes,))
        
        # Training history
        self.history = {'loss': [], 'accuracy': []}
    
    @abstractmethod
    def forward(self, sequences: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def backward(self, sequences: np.ndarray, labels: np.ndarray, 
                outputs: np.ndarray, cache: Dict) -> Dict:
        """Backward pass to compute gradients."""
        pass
    
    def embed_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Convert token indices to embeddings.
        
        Args:
            sequences: (batch_size, seq_length) token indices
            
        Returns:
            embedded: (batch_size, seq_length, embed_dim) embeddings
        """
        return self.embeddings[sequences]
    
    def classify(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Classification layer.
        
        Args:
            hidden_states: (batch_size, hidden_dim) final hidden states
            
        Returns:
            logits: (batch_size, n_classes) class logits
        """
        return np.dot(hidden_states, self.W_out) + self.b_out
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Make predictions on sequences.
        
        Args:
            sequences: (batch_size, seq_length) input sequences
            
        Returns:
            predictions: (batch_size,) predicted class indices
        """
        outputs, _ = self.forward(sequences)
        return np.argmax(outputs, axis=1)
    
    def predict_proba(self, sequences: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            sequences: (batch_size, seq_length) input sequences
            
        Returns:
            probabilities: (batch_size, n_classes) class probabilities
        """
        outputs, _ = self.forward(sequences)
        return softmax(outputs, axis=1)
    
    def compute_loss(self, outputs: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            outputs: (batch_size, n_classes) model outputs
            labels: (batch_size,) true labels
            
        Returns:
            loss: Average cross-entropy loss
        """
        batch_size = outputs.shape[0]
        probs = softmax(outputs, axis=1)
        
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        
        # Cross-entropy loss
        loss = -np.mean(np.log(probs[np.arange(batch_size), labels]))
        return loss
    
    def compute_accuracy(self, outputs: np.ndarray, labels: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = np.argmax(outputs, axis=1)
        return np.mean(predictions == labels)
    
    def save_model(self, filepath: str):
        """Save model parameters to file."""
        model_data = {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'hidden_dim': self.hidden_dim,
            'n_classes': self.n_classes,
            'max_length': self.max_length,
            'embeddings': self.embeddings,
            'W_out': self.W_out,
            'b_out': self.b_out,
            'history': self.history
        }
        
        # Add model-specific parameters
        model_data.update(self._get_model_params())
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load model parameters from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vocab_size = model_data['vocab_size']
        self.embed_dim = model_data['embed_dim']
        self.hidden_dim = model_data['hidden_dim']
        self.n_classes = model_data['n_classes']
        self.max_length = model_data['max_length']
        self.embeddings = model_data['embeddings']
        self.W_out = model_data['W_out']
        self.b_out = model_data['b_out']
        self.history = model_data['history']
        
        # Load model-specific parameters
        self._set_model_params(model_data)
    
    @abstractmethod
    def _get_model_params(self) -> Dict:
        """Get model-specific parameters for saving."""
        pass
    
    @abstractmethod
    def _set_model_params(self, model_data: Dict):
        """Set model-specific parameters from loaded data."""
        pass


class SimpleRNNClassifier(BaseSequenceClassifier):
    """Simple RNN for sequence classification."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_classes: int, max_length: int):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_classes, max_length)
        
        # RNN parameters
        self.W_xh = xavier_init((embed_dim, hidden_dim))
        self.W_hh = xavier_init((hidden_dim, hidden_dim))
        self.b_h = np.zeros((hidden_dim,))
    
    def forward(self, sequences: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through RNN.
        
        Args:
            sequences: (batch_size, seq_length) input sequences
            
        Returns:
            outputs: (batch_size, n_classes) classification logits
            cache: Dictionary containing intermediate values for backprop
        """
        batch_size, seq_length = sequences.shape
        
        # Embed sequences
        embedded = self.embed_sequences(sequences)  # (batch_size, seq_length, embed_dim)
        
        # Initialize hidden state
        h = np.zeros((batch_size, self.hidden_dim))
        
        # Store hidden states for all time steps
        hidden_states = []
        
        for t in range(seq_length):
            x_t = embedded[:, t, :]  # (batch_size, embed_dim)
            
            # RNN cell computation
            h = tanh(np.dot(x_t, self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            hidden_states.append(h.copy())
        
        # Use final hidden state for classification
        final_hidden = hidden_states[-1]
        outputs = self.classify(final_hidden)
        
        cache = {
            'embedded': embedded,
            'hidden_states': hidden_states,
            'final_hidden': final_hidden
        }
        
        return outputs, cache
    
    def backward(self, sequences: np.ndarray, labels: np.ndarray, 
                outputs: np.ndarray, cache: Dict) -> Dict:
        """
        Backward pass through RNN.
        
        Args:
            sequences: (batch_size, seq_length) input sequences
            labels: (batch_size,) true labels
            outputs: (batch_size, n_classes) model outputs
            cache: Forward pass cache
            
        Returns:
            gradients: Dictionary of parameter gradients
        """
        batch_size, seq_length = sequences.shape
        
        # Compute output gradients
        probs = softmax(outputs, axis=1)
        d_outputs = probs.copy()
        d_outputs[np.arange(batch_size), labels] -= 1
        d_outputs /= batch_size
        
        # Gradients for output layer
        dW_out = np.dot(cache['final_hidden'].T, d_outputs)
        db_out = np.sum(d_outputs, axis=0)
        
        # Backpropagate through final hidden state
        d_final_hidden = np.dot(d_outputs, self.W_out.T)
        
        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        d_embeddings = np.zeros_like(cache['embedded'])
        
        # Backpropagate through time
        d_h_next = d_final_hidden
        
        for t in reversed(range(seq_length)):
            # Current hidden state and input
            h_t = cache['hidden_states'][t]
            x_t = cache['embedded'][:, t, :]
            
            # Gradient of tanh
            d_h_raw = d_h_next * (1 - h_t**2)
            
            # Gradients for parameters
            dW_xh += np.dot(x_t.T, d_h_raw)
            dW_hh += np.dot(cache['hidden_states'][t-1].T if t > 0 else 
                           np.zeros((batch_size, self.hidden_dim)).T, d_h_raw)
            db_h += np.sum(d_h_raw, axis=0)
            
            # Gradients for inputs
            d_embeddings[:, t, :] = np.dot(d_h_raw, self.W_xh.T)
            
            # Gradient for previous hidden state
            if t > 0:
                d_h_next = np.dot(d_h_raw, self.W_hh.T)
        
        return {
            'dW_xh': dW_xh,
            'dW_hh': dW_hh,
            'db_h': db_h,
            'dW_out': dW_out,
            'db_out': db_out,
            'd_embeddings': d_embeddings
        }
    
    def _get_model_params(self) -> Dict:
        return {
            'W_xh': self.W_xh,
            'W_hh': self.W_hh,
            'b_h': self.b_h
        }
    
    def _set_model_params(self, model_data: Dict):
        self.W_xh = model_data['W_xh']
        self.W_hh = model_data['W_hh']
        self.b_h = model_data['b_h']


class LSTMClassifier(BaseSequenceClassifier):
    """LSTM for sequence classification."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_classes: int, max_length: int):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_classes, max_length)
        
        # LSTM parameters (input, forget, output, candidate gates)
        input_size = embed_dim + hidden_dim
        
        self.W_f = xavier_init((input_size, hidden_dim))  # Forget gate
        self.b_f = np.ones((hidden_dim,))  # Bias forget gate to 1
        
        self.W_i = xavier_init((input_size, hidden_dim))  # Input gate
        self.b_i = np.zeros((hidden_dim,))
        
        self.W_o = xavier_init((input_size, hidden_dim))  # Output gate
        self.b_o = np.zeros((hidden_dim,))
        
        self.W_c = xavier_init((input_size, hidden_dim))  # Candidate values
        self.b_c = np.zeros((hidden_dim,))
    
    def lstm_cell(self, x_t: np.ndarray, h_prev: np.ndarray, 
                  c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Single LSTM cell forward pass.
        
        Args:
            x_t: (batch_size, embed_dim) input at time t
            h_prev: (batch_size, hidden_dim) previous hidden state
            c_prev: (batch_size, hidden_dim) previous cell state
            
        Returns:
            h_t: (batch_size, hidden_dim) current hidden state
            c_t: (batch_size, hidden_dim) current cell state
            cache: Intermediate values for backprop
        """
        # Concatenate input and previous hidden state
        combined = np.hstack([x_t, h_prev])  # (batch_size, embed_dim + hidden_dim)
        
        # Compute gates
        f_t = sigmoid(np.dot(combined, self.W_f) + self.b_f)  # Forget gate
        i_t = sigmoid(np.dot(combined, self.W_i) + self.b_i)  # Input gate
        o_t = sigmoid(np.dot(combined, self.W_o) + self.b_o)  # Output gate
        c_tilde = tanh(np.dot(combined, self.W_c) + self.b_c)  # Candidate values
        
        # Update cell state
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Update hidden state
        h_t = o_t * tanh(c_t)
        
        cache = {
            'combined': combined,
            'f_t': f_t,
            'i_t': i_t,
            'o_t': o_t,
            'c_tilde': c_tilde,
            'c_prev': c_prev,
            'h_prev': h_prev,
            'c_t': c_t
        }
        
        return h_t, c_t, cache
    
    def forward(self, sequences: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through LSTM.
        
        Args:
            sequences: (batch_size, seq_length) input sequences
            
        Returns:
            outputs: (batch_size, n_classes) classification logits
            cache: Dictionary containing intermediate values for backprop
        """
        batch_size, seq_length = sequences.shape
        
        # Embed sequences
        embedded = self.embed_sequences(sequences)
        
        # Initialize hidden and cell states
        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))
        
        # Store states and caches for all time steps
        hidden_states = []
        cell_states = []
        lstm_caches = []
        
        for t in range(seq_length):
            x_t = embedded[:, t, :]
            h, c, lstm_cache = self.lstm_cell(x_t, h, c)
            
            hidden_states.append(h.copy())
            cell_states.append(c.copy())
            lstm_caches.append(lstm_cache)
        
        # Use final hidden state for classification
        final_hidden = hidden_states[-1]
        outputs = self.classify(final_hidden)
        
        cache = {
            'embedded': embedded,
            'hidden_states': hidden_states,
            'cell_states': cell_states,
            'lstm_caches': lstm_caches,
            'final_hidden': final_hidden
        }
        
        return outputs, cache
    
    def backward(self, sequences: np.ndarray, labels: np.ndarray, 
                outputs: np.ndarray, cache: Dict) -> Dict:
        """
        Backward pass through LSTM.
        
        Args:
            sequences: (batch_size, seq_length) input sequences
            labels: (batch_size,) true labels
            outputs: (batch_size, n_classes) model outputs
            cache: Forward pass cache
            
        Returns:
            gradients: Dictionary of parameter gradients
        """
        batch_size, seq_length = sequences.shape
        
        # Compute output gradients
        probs = softmax(outputs, axis=1)
        d_outputs = probs.copy()
        d_outputs[np.arange(batch_size), labels] -= 1
        d_outputs /= batch_size
        
        # Gradients for output layer
        dW_out = np.dot(cache['final_hidden'].T, d_outputs)
        db_out = np.sum(d_outputs, axis=0)
        
        # Backpropagate through final hidden state
        d_final_hidden = np.dot(d_outputs, self.W_out.T)
        
        # Initialize gradients
        dW_f = np.zeros_like(self.W_f)
        db_f = np.zeros_like(self.b_f)
        dW_i = np.zeros_like(self.W_i)
        db_i = np.zeros_like(self.b_i)
        dW_o = np.zeros_like(self.W_o)
        db_o = np.zeros_like(self.b_o)
        dW_c = np.zeros_like(self.W_c)
        db_c = np.zeros_like(self.b_c)
        d_embeddings = np.zeros_like(cache['embedded'])
        
        # Backpropagate through time
        d_h_next = d_final_hidden
        d_c_next = np.zeros((batch_size, self.hidden_dim))
        
        for t in reversed(range(seq_length)):
            lstm_cache = cache['lstm_caches'][t]
            
            # Gradients from next time step
            d_c_t = d_c_next + d_h_next * lstm_cache['o_t'] * (1 - tanh(lstm_cache['c_t'])**2)
            
            # Gate gradients
            d_o_t = d_h_next * tanh(lstm_cache['c_t'])
            d_f_t = d_c_t * lstm_cache['c_prev']
            d_i_t = d_c_t * lstm_cache['c_tilde']
            d_c_tilde = d_c_t * lstm_cache['i_t']
            
            # Gate activation derivatives
            d_o_raw = d_o_t * lstm_cache['o_t'] * (1 - lstm_cache['o_t'])
            d_f_raw = d_f_t * lstm_cache['f_t'] * (1 - lstm_cache['f_t'])
            d_i_raw = d_i_t * lstm_cache['i_t'] * (1 - lstm_cache['i_t'])
            d_c_raw = d_c_tilde * (1 - lstm_cache['c_tilde']**2)
            
            # Parameter gradients
            dW_f += np.dot(lstm_cache['combined'].T, d_f_raw)
            db_f += np.sum(d_f_raw, axis=0)
            
            dW_i += np.dot(lstm_cache['combined'].T, d_i_raw)
            db_i += np.sum(d_i_raw, axis=0)
            
            dW_o += np.dot(lstm_cache['combined'].T, d_o_raw)
            db_o += np.sum(d_o_raw, axis=0)
            
            dW_c += np.dot(lstm_cache['combined'].T, d_c_raw)
            db_c += np.sum(d_c_raw, axis=0)
            
            # Input gradients
            d_combined = (np.dot(d_f_raw, self.W_f.T) + 
                         np.dot(d_i_raw, self.W_i.T) + 
                         np.dot(d_o_raw, self.W_o.T) + 
                         np.dot(d_c_raw, self.W_c.T))
            
            # Split gradients for input and previous hidden state
            d_x_t = d_combined[:, :self.embed_dim]
            d_h_prev = d_combined[:, self.embed_dim:]
            
            # Store input gradients
            d_embeddings[:, t, :] = d_x_t
            
            # Gradients for next iteration
            d_h_next = d_h_prev
            d_c_next = d_c_t * lstm_cache['f_t']
        
        return {
            'dW_f': dW_f, 'db_f': db_f,
            'dW_i': dW_i, 'db_i': db_i,
            'dW_o': dW_o, 'db_o': db_o,
            'dW_c': dW_c, 'db_c': db_c,
            'dW_out': dW_out, 'db_out': db_out,
            'd_embeddings': d_embeddings
        }
    
    def _get_model_params(self) -> Dict:
        return {
            'W_f': self.W_f, 'b_f': self.b_f,
            'W_i': self.W_i, 'b_i': self.b_i,
            'W_o': self.W_o, 'b_o': self.b_o,
            'W_c': self.W_c, 'b_c': self.b_c
        }
    
    def _set_model_params(self, model_data: Dict):
        self.W_f = model_data['W_f']
        self.b_f = model_data['b_f']
        self.W_i = model_data['W_i']
        self.b_i = model_data['b_i']
        self.W_o = model_data['W_o']
        self.b_o = model_data['b_o']
        self.W_c = model_data['W_c']
        self.b_c = model_data['b_c']


class AttentionLSTMClassifier(LSTMClassifier):
    """LSTM with attention mechanism for sequence classification."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_classes: int, max_length: int, attention_dim: int = 128):
        super().__init__(vocab_size, embed_dim, hidden_dim, n_classes, max_length)
        
        self.attention_dim = attention_dim
        
        # Attention parameters
        self.W_att = xavier_init((hidden_dim, attention_dim))
        self.b_att = np.zeros((attention_dim,))
        self.v_att = xavier_init((attention_dim, 1))
        
        # Update output layer to use attention-weighted hidden state
        self.W_out = xavier_init((hidden_dim, n_classes))
    
    def attention(self, hidden_states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention weights and context vector.
        
        Args:
            hidden_states: List of (batch_size, hidden_dim) hidden states
            
        Returns:
            context: (batch_size, hidden_dim) attention-weighted context
            attention_weights: (batch_size, seq_length) attention weights
        """
        seq_length = len(hidden_states)
        batch_size = hidden_states[0].shape[0]
        
        # Stack hidden states
        H = np.stack(hidden_states, axis=1)  # (batch_size, seq_length, hidden_dim)
        
        # Compute attention scores
        # Reshape for batch computation
        H_flat = H.reshape(-1, self.hidden_dim)  # (batch_size * seq_length, hidden_dim)
        
        # Attention computation
        att_hidden = tanh(np.dot(H_flat, self.W_att) + self.b_att)  # (batch_size * seq_length, attention_dim)
        att_scores = np.dot(att_hidden, self.v_att).reshape(batch_size, seq_length)  # (batch_size, seq_length)
        
        # Apply softmax to get attention weights
        attention_weights = softmax(att_scores, axis=1)  # (batch_size, seq_length)
        
        # Compute context vector
        context = np.sum(H * attention_weights[:, :, np.newaxis], axis=1)  # (batch_size, hidden_dim)
        
        return context, attention_weights
    
    def forward(self, sequences: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass through LSTM with attention.
        
        Args:
            sequences: (batch_size, seq_length) input sequences
            
        Returns:
            outputs: (batch_size, n_classes) classification logits
            cache: Dictionary containing intermediate values for backprop
        """
        # Standard LSTM forward pass
        _, lstm_cache = super().forward(sequences)
        
        # Apply attention to all hidden states
        context, attention_weights = self.attention(lstm_cache['hidden_states'])
        
        # Classification using attention-weighted context
        outputs = self.classify(context)
        
        # Update cache
        lstm_cache.update({
            'context': context,
            'attention_weights': attention_weights
        })
        
        return outputs, lstm_cache
    
    def backward(self, sequences: np.ndarray, labels: np.ndarray, 
                outputs: np.ndarray, cache: Dict) -> Dict:
        """
        Backward pass through LSTM with attention.
        
        Args:
            sequences: (batch_size, seq_length) input sequences
            labels: (batch_size,) true labels
            outputs: (batch_size, n_classes) model outputs
            cache: Forward pass cache
            
        Returns:
            gradients: Dictionary of parameter gradients
        """
        batch_size, seq_length = sequences.shape
        
        # Compute output gradients
        probs = softmax(outputs, axis=1)
        d_outputs = probs.copy()
        d_outputs[np.arange(batch_size), labels] -= 1
        d_outputs /= batch_size
        
        # Gradients for output layer
        dW_out = np.dot(cache['context'].T, d_outputs)
        db_out = np.sum(d_outputs, axis=0)
        
        # Backpropagate through attention
        d_context = np.dot(d_outputs, self.W_out.T)
        
        # Attention gradients
        H = np.stack(cache['hidden_states'], axis=1)  # (batch_size, seq_length, hidden_dim)
        attention_weights = cache['attention_weights']
        
        # Gradient w.r.t. attention weights
        d_att_weights = np.sum(d_context[:, np.newaxis, :] * H, axis=2)  # (batch_size, seq_length)
        
        # Gradient w.r.t. hidden states from attention
        d_H_att = d_context[:, np.newaxis, :] * attention_weights[:, :, np.newaxis]  # (batch_size, seq_length, hidden_dim)
        
        # Softmax gradient
        d_att_scores = d_att_weights * attention_weights - attention_weights * np.sum(d_att_weights * attention_weights, axis=1, keepdims=True)
        
        # Backpropagate through attention network
        H_flat = H.reshape(-1, self.hidden_dim)
        att_hidden = tanh(np.dot(H_flat, self.W_att) + self.b_att)
        
        d_v_att = np.dot(att_hidden.T, d_att_scores.flatten()[:, np.newaxis])
        d_att_hidden = np.dot(d_att_scores.flatten()[:, np.newaxis], self.v_att.T)
        
        # Tanh gradient
        d_att_raw = d_att_hidden * (1 - att_hidden**2)
        
        dW_att = np.dot(H_flat.T, d_att_raw)
        db_att = np.sum(d_att_raw, axis=0)
        
        # Additional gradient for hidden states
        d_H_att_network = np.dot(d_att_raw, self.W_att.T).reshape(batch_size, seq_length, self.hidden_dim)
        d_H_total = d_H_att + d_H_att_network
        
        # Get LSTM gradients (modified to include attention gradients)
        lstm_gradients = super().backward(sequences, labels, outputs, cache)
        
        # Add attention gradients
        lstm_gradients.update({
            'dW_att': dW_att,
            'db_att': db_att,
            'dv_att': d_v_att,
            'dW_out': dW_out,
            'db_out': db_out
        })
        
        return lstm_gradients
    
    def _get_model_params(self) -> Dict:
        base_params = super()._get_model_params()
        base_params.update({
            'attention_dim': self.attention_dim,
            'W_att': self.W_att,
            'b_att': self.b_att,
            'v_att': self.v_att
        })
        return base_params
    
    def _set_model_params(self, model_data: Dict):
        super()._set_model_params(model_data)
        self.attention_dim = model_data['attention_dim']
        self.W_att = model_data['W_att']
        self.b_att = model_data['b_att']
        self.v_att = model_data['v_att']


class SequenceClassifierTrainer:
    """Training utility for sequence classifiers."""
    
    def __init__(self, model: BaseSequenceClassifier, learning_rate: float = 0.001,
                 clip_grad: float = 5.0, l2_reg: float = 0.0):
        """
        Initialize trainer.
        
        Args:
            model: Sequence classifier model
            learning_rate: Learning rate for optimization
            clip_grad: Gradient clipping threshold
            l2_reg: L2 regularization coefficient
        """
        self.model = model
        self.learning_rate = learning_rate
        self.clip_grad = clip_grad
        self.l2_reg = l2_reg
    
    def clip_gradients(self, gradients: Dict) -> Dict:
        """Apply gradient clipping."""
        clipped = {}
        total_norm = 0
        
        # Compute total gradient norm
        for key, grad in gradients.items():
            if isinstance(grad, np.ndarray):
                total_norm += np.sum(grad**2)
        
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > self.clip_grad:
            clip_factor = self.clip_grad / total_norm
            for key, grad in gradients.items():
                if isinstance(grad, np.ndarray):
                    clipped[key] = grad * clip_factor
                else:
                    clipped[key] = grad
        else:
            clipped = gradients
        
        return clipped
    
    def apply_gradients(self, gradients: Dict):
        """Apply gradients to model parameters."""
        # Update embeddings
        if 'd_embeddings' in gradients:
            # Only update embeddings for non-padding tokens
            self.model.embeddings -= self.learning_rate * np.mean(gradients['d_embeddings'], axis=(0, 1))
        
        # Update output layer
        if 'dW_out' in gradients:
            # Add L2 regularization
            reg_term = self.l2_reg * self.model.W_out
            self.model.W_out -= self.learning_rate * (gradients['dW_out'] + reg_term)
        
        if 'db_out' in gradients:
            self.model.b_out -= self.learning_rate * gradients['db_out']
        
        # Update model-specific parameters
        if isinstance(self.model, SimpleRNNClassifier):
            self.model.W_xh -= self.learning_rate * gradients['dW_xh']
            self.model.W_hh -= self.learning_rate * gradients['dW_hh']
            self.model.b_h -= self.learning_rate * gradients['db_h']
        
        elif isinstance(self.model, LSTMClassifier):
            self.model.W_f -= self.learning_rate * gradients['dW_f']
            self.model.b_f -= self.learning_rate * gradients['db_f']
            self.model.W_i -= self.learning_rate * gradients['dW_i']
            self.model.b_i -= self.learning_rate * gradients['db_i']
            self.model.W_o -= self.learning_rate * gradients['dW_o']
            self.model.b_o -= self.learning_rate * gradients['db_o']
            self.model.W_c -= self.learning_rate * gradients['dW_c']
            self.model.b_c -= self.learning_rate * gradients['db_c']
            
            # Attention parameters if applicable
            if isinstance(self.model, AttentionLSTMClassifier):
                if 'dW_att' in gradients:
                    self.model.W_att -= self.learning_rate * gradients['dW_att']
                if 'db_att' in gradients:
                    self.model.b_att -= self.learning_rate * gradients['db_att']
                if 'dv_att' in gradients:
                    self.model.v_att -= self.learning_rate * gradients['dv_att']
    
    def train_epoch(self, train_data, batch_size: int = 32) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_data: Training dataset
            batch_size: Batch size
            
        Returns:
            avg_loss: Average loss for the epoch
            avg_accuracy: Average accuracy for the epoch
        """
        n_samples = len(train_data)
        indices = np.random.permutation(n_samples)
        
        total_loss = 0
        total_accuracy = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_sequences, batch_labels = train_data.get_batch(batch_indices)
            
            # Forward pass
            outputs, cache = self.model.forward(batch_sequences)
            
            # Compute loss and accuracy
            loss = self.model.compute_loss(outputs, batch_labels)
            accuracy = self.model.compute_accuracy(outputs, batch_labels)
            
            # Backward pass
            gradients = self.model.backward(batch_sequences, batch_labels, outputs, cache)
            
            # Clip gradients
            gradients = self.clip_gradients(gradients)
            
            # Apply gradients
            self.apply_gradients(gradients)
            
            total_loss += loss
            total_accuracy += accuracy
            n_batches += 1
        
        return total_loss / n_batches, total_accuracy / n_batches
    
    def evaluate(self, test_data, batch_size: int = 32) -> Tuple[float, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test dataset
            batch_size: Batch size
            
        Returns:
            avg_loss: Average loss
            avg_accuracy: Average accuracy
        """
        n_samples = len(test_data)
        total_loss = 0
        total_accuracy = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = list(range(i, min(i + batch_size, n_samples)))
            batch_sequences, batch_labels = test_data.get_batch(batch_indices)
            
            # Forward pass only
            outputs, _ = self.model.forward(batch_sequences)
            
            # Compute loss and accuracy
            loss = self.model.compute_loss(outputs, batch_labels)
            accuracy = self.model.compute_accuracy(outputs, batch_labels)
            
            total_loss += loss
            total_accuracy += accuracy
            n_batches += 1
        
        return total_loss / n_batches, total_accuracy / n_batches
    
    def train(self, train_data, test_data=None, epochs: int = 50, 
              batch_size: int = 32, verbose: bool = True) -> Dict:
        """
        Train the model.
        
        Args:
            train_data: Training dataset
            test_data: Test dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print progress
            
        Returns:
            history: Training history
        """
        history = {'train_loss': [], 'train_accuracy': [], 
                  'test_loss': [], 'test_accuracy': []}
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_data, batch_size)
            
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            
            # Evaluate on test data if provided
            if test_data is not None:
                test_loss, test_acc = self.evaluate(test_data, batch_size)
                history['test_loss'].append(test_loss)
                history['test_accuracy'].append(test_acc)
            
            # Update model history
            self.model.history['loss'].append(train_loss)
            self.model.history['accuracy'].append(train_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                if test_data is not None:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        return history


if __name__ == "__main__":
    # Test the sequence classification models
    print("Testing Sequence Classification Models")
    print("=" * 50)
    
    # Create dummy data
    vocab_size = 100
    embed_dim = 32
    hidden_dim = 64
    n_classes = 2
    max_length = 20
    batch_size = 16
    
    # Generate random sequences
    np.random.seed(42)
    sequences = np.random.randint(1, vocab_size, (batch_size, max_length))
    labels = np.random.randint(0, n_classes, batch_size)
    
    # Test models
    models = {
        'SimpleRNN': SimpleRNNClassifier(vocab_size, embed_dim, hidden_dim, n_classes, max_length),
        'LSTM': LSTMClassifier(vocab_size, embed_dim, hidden_dim, n_classes, max_length),
        'AttentionLSTM': AttentionLSTMClassifier(vocab_size, embed_dim, hidden_dim, n_classes, max_length)
    }
    
    for name, model in models.items():
        print(f"\nTesting {name}:")
        print("-" * 30)
        
        try:
            # Forward pass
            outputs, cache = model.forward(sequences)
            print(f"Output shape: {outputs.shape}")
            print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            
            # Compute loss and accuracy
            loss = model.compute_loss(outputs, labels)
            accuracy = model.compute_accuracy(outputs, labels)
            print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            
            # Test backward pass
            gradients = model.backward(sequences, labels, outputs, cache)
            print(f"Gradients computed: {list(gradients.keys())}")
            
            # Test predictions
            predictions = model.predict(sequences)
            probs = model.predict_proba(sequences)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Probabilities shape: {probs.shape}")
            print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
            
            print(f"✓ {name} test passed!")
            
        except Exception as e:
            print(f"✗ {name} test failed: {e}")
    
    print("\nSequence classification models test completed!")