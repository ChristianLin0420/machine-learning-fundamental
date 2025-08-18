#!/usr/bin/env python3
"""
PyTorch implementations of sequence classification models.

This module provides modern PyTorch implementations with techniques like:
- Dropout and batch normalization
- Pre-trained embeddings
- Bidirectional RNNs
- Multi-head attention
- Transformer encoders
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import math
from abc import ABC, abstractmethod

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SequenceDatasetPyTorch(Dataset):
    """PyTorch Dataset wrapper for sequence classification."""
    
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        """
        Initialize PyTorch dataset.
        
        Args:
            sequences: (n_samples, max_length) sequence data
            labels: (n_samples,) labels
        """
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class BaseSequenceClassifierPyTorch(nn.Module, ABC):
    """Abstract base class for PyTorch sequence classifiers."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_classes: int, padding_idx: int = 0, dropout: float = 0.1):
        """
        Initialize base PyTorch sequence classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            n_classes: Number of output classes
            padding_idx: Padding token index
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.padding_idx = padding_idx
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, n_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.1, 0.1)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].fill_(0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
                        # Set forget gate bias to 1 for LSTM
                        if isinstance(module, nn.LSTM):
                            n = param.size(0)
                            param.data[n//4:n//2].fill_(1.0)
    
    def create_padding_mask(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for sequences.
        
        Args:
            sequences: (batch_size, seq_length) input sequences
            
        Returns:
            mask: (batch_size, seq_length) boolean mask (True for padding)
        """
        return sequences == self.padding_idx
    
    @abstractmethod
    def encode_sequence(self, embedded: torch.Tensor, 
                       mask: torch.Tensor) -> torch.Tensor:
        """Encode sequence into fixed-size representation."""
        pass
    
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sequences: (batch_size, seq_length) input sequences
            
        Returns:
            logits: (batch_size, n_classes) class logits
        """
        # Create padding mask
        padding_mask = self.create_padding_mask(sequences)
        
        # Embed sequences
        embedded = self.embedding(sequences)  # (batch_size, seq_length, embed_dim)
        embedded = self.dropout_layer(embedded)
        
        # Encode sequence
        encoded = self.encode_sequence(embedded, padding_mask)
        
        # Apply dropout before classification
        encoded = self.dropout_layer(encoded)
        
        # Classification
        logits = self.classifier(encoded)
        
        return logits


class RNNClassifierPyTorch(BaseSequenceClassifierPyTorch):
    """Simple RNN classifier using PyTorch."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_classes: int, num_layers: int = 1, bidirectional: bool = False,
                 padding_idx: int = 0, dropout: float = 0.1, rnn_type: str = 'RNN'):
        """
        Initialize RNN classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            n_classes: Number of output classes
            num_layers: Number of RNN layers
            bidirectional: Whether to use bidirectional RNN
            padding_idx: Padding token index
            dropout: Dropout probability
            rnn_type: Type of RNN ('RNN', 'LSTM', 'GRU')
        """
        # Adjust hidden_dim for bidirectional
        actual_hidden_dim = hidden_dim * (2 if bidirectional else 1)
        
        super().__init__(vocab_size, embed_dim, actual_hidden_dim, n_classes, 
                        padding_idx, dropout)
        
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.rnn_hidden_dim = hidden_dim  # Original hidden dim for RNN
        
        # Create RNN layer
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                embed_dim, hidden_dim, num_layers, 
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                embed_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:  # Simple RNN
            self.rnn = nn.RNN(
                embed_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional, nonlinearity='tanh'
            )
    
    def encode_sequence(self, embedded: torch.Tensor, 
                       mask: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence using RNN.
        
        Args:
            embedded: (batch_size, seq_length, embed_dim) embedded sequences
            mask: (batch_size, seq_length) padding mask
            
        Returns:
            encoded: (batch_size, hidden_dim) encoded representation
        """
        batch_size, seq_length, _ = embedded.shape
        
        # Pack padded sequences for efficiency
        lengths = (~mask).sum(dim=1).cpu()
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        
        # RNN forward pass
        packed_output, hidden = self.rnn(packed_embedded)
        
        # Unpack sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Use last hidden state for classification
        if self.rnn_type.upper() == 'LSTM':
            # For LSTM, hidden is a tuple (h_n, c_n)
            last_hidden = hidden[0]  # Use h_n
        else:
            # For RNN and GRU, hidden is just h_n
            last_hidden = hidden
        
        # last_hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # Take the last layer
        if self.bidirectional:
            # Concatenate forward and backward hidden states from last layer
            forward_hidden = last_hidden[-2, :, :]  # Forward direction of last layer
            backward_hidden = last_hidden[-1, :, :]  # Backward direction of last layer
            encoded = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            encoded = last_hidden[-1, :, :]  # Last layer
        
        return encoded


class AttentionClassifierPyTorch(BaseSequenceClassifierPyTorch):
    """LSTM with attention mechanism using PyTorch."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_classes: int, num_layers: int = 1, bidirectional: bool = True,
                 attention_dim: int = 128, padding_idx: int = 0, dropout: float = 0.1):
        """
        Initialize attention-based classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            n_classes: Number of output classes
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            attention_dim: Attention mechanism dimension
            padding_idx: Padding token index
            dropout: Dropout probability
        """
        # Adjust for bidirectional
        actual_hidden_dim = hidden_dim * (2 if bidirectional else 1)
        
        super().__init__(vocab_size, embed_dim, actual_hidden_dim, n_classes, 
                        padding_idx, dropout)
        
        self.attention_dim = attention_dim
        self.bidirectional = bidirectional
        self.lstm_hidden_dim = hidden_dim
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(actual_hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def encode_sequence(self, embedded: torch.Tensor, 
                       mask: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence using LSTM with attention.
        
        Args:
            embedded: (batch_size, seq_length, embed_dim) embedded sequences
            mask: (batch_size, seq_length) padding mask
            
        Returns:
            encoded: (batch_size, hidden_dim) attention-weighted representation
        """
        batch_size, seq_length, _ = embedded.shape
        
        # LSTM forward pass
        lstm_output, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_dim)
        
        # Compute attention scores
        attention_scores = self.attention(lstm_output).squeeze(-1)  # (batch_size, seq_length)
        
        # Mask padding tokens
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_length)
        
        # Compute weighted sum
        encoded = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        
        # Store attention weights for visualization
        self.last_attention_weights = attention_weights.detach()
        
        return encoded


class TransformerClassifierPyTorch(BaseSequenceClassifierPyTorch):
    """Transformer-based sequence classifier using PyTorch."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_classes: int, num_heads: int = 8, num_layers: int = 3,
                 ff_dim: int = 512, padding_idx: int = 0, dropout: float = 0.1):
        """
        Initialize Transformer classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension (should equal embed_dim for Transformer)
            n_classes: Number of output classes
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            padding_idx: Padding token index
            dropout: Dropout probability
        """
        super().__init__(vocab_size, embed_dim, embed_dim, n_classes, 
                        padding_idx, dropout)
        
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Global pooling for classification
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Update classifier input dimension
        self.classifier = nn.Linear(embed_dim, n_classes)
    
    def encode_sequence(self, embedded: torch.Tensor, 
                       mask: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence using Transformer.
        
        Args:
            embedded: (batch_size, seq_length, embed_dim) embedded sequences
            mask: (batch_size, seq_length) padding mask
            
        Returns:
            encoded: (batch_size, embed_dim) encoded representation
        """
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Transformer expects src_key_padding_mask to be True for padding
        transformer_output = self.transformer(
            embedded, 
            src_key_padding_mask=mask
        )  # (batch_size, seq_length, embed_dim)
        
        # Global average pooling over non-padded tokens
        # Mask out padding tokens
        masked_output = transformer_output.masked_fill(
            mask.unsqueeze(-1), 0
        )
        
        # Compute mean over sequence length, ignoring padding
        lengths = (~mask).sum(dim=1, keepdim=True).float()  # (batch_size, 1)
        encoded = masked_output.sum(dim=1) / lengths  # (batch_size, embed_dim)
        
        return encoded


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_length: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, embed_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        
        Args:
            x: (batch_size, seq_length, embed_dim) input embeddings
            
        Returns:
            x: (batch_size, seq_length, embed_dim) embeddings with positional encoding
        """
        seq_length = x.size(1)
        x = x + self.pe[:seq_length, :].transpose(0, 1)
        return self.dropout(x)


class CNNClassifierPyTorch(BaseSequenceClassifierPyTorch):
    """CNN-based text classifier using PyTorch."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, 
                 n_classes: int, filter_sizes: List[int] = [3, 4, 5],
                 num_filters: int = 100, padding_idx: int = 0, dropout: float = 0.1):
        """
        Initialize CNN classifier.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden layer dimension (not used, kept for compatibility)
            n_classes: Number of output classes
            filter_sizes: List of filter sizes for different conv layers
            num_filters: Number of filters per filter size
            padding_idx: Padding token index
            dropout: Dropout probability
        """
        # CNN output dim is num_filters * len(filter_sizes)
        cnn_output_dim = num_filters * len(filter_sizes)
        
        super().__init__(vocab_size, embed_dim, cnn_output_dim, n_classes, 
                        padding_idx, dropout)
        
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        
        # Create convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=filter_size)
            for filter_size in filter_sizes
        ])
        
        # Update classifier
        self.classifier = nn.Linear(cnn_output_dim, n_classes)
    
    def encode_sequence(self, embedded: torch.Tensor, 
                       mask: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence using CNN.
        
        Args:
            embedded: (batch_size, seq_length, embed_dim) embedded sequences
            mask: (batch_size, seq_length) padding mask
            
        Returns:
            encoded: (batch_size, cnn_output_dim) encoded representation
        """
        # Transpose for conv1d: (batch_size, embed_dim, seq_length)
        embedded = embedded.transpose(1, 2)
        
        conv_outputs = []
        for conv in self.convs:
            # Apply convolution and ReLU
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, conv_length)
            
            # Global max pooling
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            pooled = pooled.squeeze(2)  # (batch_size, num_filters)
            
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        encoded = torch.cat(conv_outputs, dim=1)  # (batch_size, total_filters)
        
        return encoded


class SequenceClassifierTrainerPyTorch:
    """PyTorch trainer for sequence classifiers."""
    
    def __init__(self, model: nn.Module, device: torch.device = device,
                 learning_rate: float = 0.001, weight_decay: float = 1e-5):
        """
        Initialize PyTorch trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use for training
            learning_rate: Learning rate
            weight_decay: L2 regularization coefficient
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    def train_epoch(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            data_loader: Training data loader
            
        Returns:
            avg_loss: Average loss for the epoch
            avg_accuracy: Average accuracy for the epoch
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for sequences, labels in data_loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(sequences)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(data_loader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on data.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            avg_loss: Average loss
            avg_accuracy: Average accuracy
        """
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for sequences, labels in data_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(data_loader)
        avg_accuracy = total_correct / total_samples
        
        return avg_loss, avg_accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None,
              epochs: int = 50, verbose: bool = True) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            verbose: Whether to print progress
            
        Returns:
            history: Training history
        """
        history = {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'best_model.pth')
            
            if verbose and (epoch + 1) % 10 == 0:
                if val_loader is not None:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        return history
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on data.
        
        Args:
            data_loader: Data loader for prediction
            
        Returns:
            predictions: Predicted class indices
            probabilities: Prediction probabilities
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for sequences, _ in data_loader:
                sequences = sequences.to(self.device)
                
                outputs = self.model(sequences)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
        
        predictions = np.concatenate(all_predictions)
        probabilities = np.concatenate(all_probabilities)
        
        return predictions, probabilities


def create_pytorch_model(model_type: str, vocab_size: int, embed_dim: int,
                        hidden_dim: int, n_classes: int, **kwargs) -> nn.Module:
    """
    Factory function to create PyTorch models.
    
    Args:
        model_type: Type of model ('rnn', 'lstm', 'gru', 'attention', 'transformer', 'cnn')
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        hidden_dim: Hidden layer dimension
        n_classes: Number of output classes
        **kwargs: Additional model-specific arguments
        
    Returns:
        model: PyTorch model
    """
    model_type = model_type.lower()
    
    if model_type in ['rnn', 'lstm', 'gru']:
        return RNNClassifierPyTorch(
            vocab_size, embed_dim, hidden_dim, n_classes,
            rnn_type=model_type.upper(), **kwargs
        )
    elif model_type == 'attention':
        return AttentionClassifierPyTorch(
            vocab_size, embed_dim, hidden_dim, n_classes, **kwargs
        )
    elif model_type == 'transformer':
        return TransformerClassifierPyTorch(
            vocab_size, embed_dim, hidden_dim, n_classes, **kwargs
        )
    elif model_type == 'cnn':
        return CNNClassifierPyTorch(
            vocab_size, embed_dim, hidden_dim, n_classes, **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test PyTorch models
    print("Testing PyTorch Sequence Classification Models")
    print("=" * 55)
    
    # Create dummy data
    vocab_size = 100
    embed_dim = 64
    hidden_dim = 128
    n_classes = 3
    batch_size = 16
    seq_length = 20
    
    # Generate random data
    sequences = torch.randint(1, vocab_size, (batch_size, seq_length))
    labels = torch.randint(0, n_classes, (batch_size,))
    
    # Create dataset and data loader
    dataset = SequenceDatasetPyTorch(sequences.numpy(), labels.numpy())
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Test different models
    model_types = ['rnn', 'lstm', 'gru', 'attention', 'transformer', 'cnn']
    
    for model_type in model_types:
        print(f"\nTesting {model_type.upper()}:")
        print("-" * 40)
        
        try:
            # Create model
            if model_type == 'transformer':
                # Transformer requires embed_dim == hidden_dim
                model = create_pytorch_model(
                    model_type, vocab_size, embed_dim, embed_dim, n_classes
                )
            else:
                model = create_pytorch_model(
                    model_type, vocab_size, embed_dim, hidden_dim, n_classes
                )
            
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(sequences)
                print(f"Output shape: {outputs.shape}")
                print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            
            # Test training
            trainer = SequenceClassifierTrainerPyTorch(model)
            
            # One training step
            model.train()
            loss, accuracy = trainer.train_epoch(data_loader)
            print(f"Training loss: {loss:.4f}")
            print(f"Training accuracy: {accuracy:.4f}")
            
            # Test prediction
            predictions, probabilities = trainer.predict(data_loader)
            print(f"Predictions shape: {predictions.shape}")
            print(f"Probabilities shape: {probabilities.shape}")
            
            print(f"✓ {model_type.upper()} test passed!")
            
        except Exception as e:
            print(f"✗ {model_type.upper()} test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nPyTorch models test completed!")