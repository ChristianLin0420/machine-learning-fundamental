"""
Multi-Layer Perceptron (MLP) Implementation from Scratch

This module implements a feedforward neural network with arbitrary architecture
using only NumPy. Includes forward propagation, backpropagation using the
chain rule, and various optimization techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
from activation_functions import (get_activation_function, get_activation_derivative, 
                                sigmoid, softmax)


class FeedforwardNeuralNet:
    """
    Multi-Layer Perceptron (Feedforward Neural Network)
    
    A fully connected neural network with customizable architecture,
    activation functions, and training parameters.
    """
    
    def __init__(self, layers, activations=None, output_activation='sigmoid', 
                 weight_init='xavier', random_state=42):
        """
        Initialize the neural network.
        
        Args:
            layers (list): Number of neurons in each layer [input, hidden1, ..., output]
            activations (list): Activation functions for each hidden layer
            output_activation (str): Activation function for output layer
            weight_init (str): Weight initialization method
            random_state (int): Random seed for reproducibility
        """
        self.layers = layers
        self.n_layers = len(layers)
        self.random_state = random_state
        
        # Set default activations if not provided
        if activations is None:
            activations = ['relu'] * (self.n_layers - 2)  # All hidden layers use ReLU
        
        self.activations = activations + [output_activation]
        
        # Initialize weights and biases
        self._initialize_parameters(weight_init)
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def _initialize_parameters(self, init_method='xavier'):
        """
        Initialize weights and biases for all layers.
        
        Args:
            init_method (str): Initialization method ('xavier', 'he', 'random', 'zeros')
        """
        np.random.seed(self.random_state)
        
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers - 1):
            input_size = self.layers[i]
            output_size = self.layers[i + 1]
            
            # Weight initialization
            if init_method == 'xavier':
                # Xavier/Glorot initialization
                limit = np.sqrt(6 / (input_size + output_size))
                W = np.random.uniform(-limit, limit, (input_size, output_size))
            elif init_method == 'he':
                # He initialization (good for ReLU)
                std = np.sqrt(2 / input_size)
                W = np.random.normal(0, std, (input_size, output_size))
            elif init_method == 'random':
                # Small random values
                W = np.random.normal(0, 0.01, (input_size, output_size))
            elif init_method == 'zeros':
                # Zero initialization (for debugging)
                W = np.zeros((input_size, output_size))
            else:
                raise ValueError(f"Unknown initialization method: {init_method}")
            
            # Bias initialization (usually zeros)
            b = np.zeros((1, output_size))
            
            self.weights.append(W)
            self.biases.append(b)
    
    def _forward_layer(self, X, layer_idx):
        """
        Forward propagation through a single layer.
        
        Args:
            X (np.ndarray): Input to the layer
            layer_idx (int): Index of the layer
            
        Returns:
            tuple: (pre_activation, post_activation)
        """
        # Linear transformation: z = X @ W + b
        z = np.dot(X, self.weights[layer_idx]) + self.biases[layer_idx]
        
        # Apply activation function
        activation_name = self.activations[layer_idx]
        activation_func = get_activation_function(activation_name)
        a = activation_func(z)
        
        return z, a
    
    def forward(self, X):
        """
        Forward propagation through the entire network.
        
        Args:
            X (np.ndarray): Input data (n_samples, n_features)
            
        Returns:
            tuple: (activations, pre_activations)
        """
        activations = [X]  # Store activations for each layer
        pre_activations = []  # Store pre-activations (z values)
        
        current_input = X
        
        # Forward through all layers
        for i in range(self.n_layers - 1):
            z, a = self._forward_layer(current_input, i)
            
            pre_activations.append(z)
            activations.append(a)
            current_input = a
        
        return activations, pre_activations
    
    def _compute_loss(self, y_true, y_pred, loss_type='cross_entropy'):
        """
        Compute loss function.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted values
            loss_type (str): Type of loss function
            
        Returns:
            float: Loss value
        """
        n_samples = y_true.shape[0]
        
        if loss_type == 'cross_entropy':
            # For binary classification with sigmoid output
            if y_pred.shape[1] == 1:
                # Convert {-1, 1} labels to {0, 1}
                y_binary = (y_true + 1) / 2
                
                # Clip predictions to prevent log(0)
                y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
                
                loss = -np.mean(y_binary * np.log(y_pred_clipped) + 
                               (1 - y_binary) * np.log(1 - y_pred_clipped))
            else:
                # Multi-class cross-entropy
                # Convert labels to one-hot if needed
                if y_true.shape[1] == 1:
                    y_one_hot = np.zeros((n_samples, y_pred.shape[1]))
                    y_one_hot[np.arange(n_samples), y_true.flatten().astype(int)] = 1
                    y_true = y_one_hot
                
                # Clip predictions
                y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
                loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        
        elif loss_type == 'mse':
            # Mean squared error for regression
            loss = np.mean((y_true - y_pred) ** 2)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss
    
    def _compute_accuracy(self, y_true, y_pred):
        """
        Compute classification accuracy.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted probabilities
            
        Returns:
            float: Accuracy score
        """
        if y_pred.shape[1] == 1:
            # Binary classification
            predictions = (y_pred > 0.5).astype(int)
            y_binary = ((y_true + 1) / 2).astype(int)
            return np.mean(predictions == y_binary)
        else:
            # Multi-class classification
            predictions = np.argmax(y_pred, axis=1)
            if y_true.shape[1] == 1:
                true_labels = y_true.flatten()
            else:
                true_labels = np.argmax(y_true, axis=1)
            return np.mean(predictions == true_labels)
    
    def _backward_output_layer(self, y_true, y_pred, pre_activation):
        """
        Compute gradients for the output layer.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted values
            pre_activation (np.ndarray): Pre-activation values of output layer
            
        Returns:
            np.ndarray: Gradient with respect to pre-activation (delta)
        """
        activation_name = self.activations[-1]
        
        if activation_name == 'sigmoid' and y_pred.shape[1] == 1:
            # Binary cross-entropy with sigmoid
            # Convert {-1, 1} labels to {0, 1}
            y_binary = (y_true + 1) / 2
            delta = y_pred - y_binary
            
        elif activation_name == 'softmax':
            # Multi-class cross-entropy with softmax
            # Convert labels to one-hot if needed
            if y_true.shape[1] == 1:
                n_samples = y_true.shape[0]
                y_one_hot = np.zeros((n_samples, y_pred.shape[1]))
                y_one_hot[np.arange(n_samples), y_true.flatten().astype(int)] = 1
                y_true = y_one_hot
            
            delta = y_pred - y_true
            
        elif activation_name == 'linear':
            # MSE with linear output (regression)
            activation_deriv = get_activation_derivative(activation_name)
            delta = (y_pred - y_true) * activation_deriv(pre_activation)
            
        else:
            # General case
            activation_deriv = get_activation_derivative(activation_name)
            if y_pred.shape[1] == 1:
                y_binary = (y_true + 1) / 2
                loss_gradient = y_pred - y_binary
            else:
                loss_gradient = y_pred - y_true
            
            delta = loss_gradient * activation_deriv(pre_activation)
        
        return delta
    
    def backward(self, X, y_true, activations, pre_activations):
        """
        Backward propagation to compute gradients.
        
        Args:
            X (np.ndarray): Input data
            y_true (np.ndarray): True labels
            activations (list): Activations from forward pass
            pre_activations (list): Pre-activations from forward pass
            
        Returns:
            tuple: (weight_gradients, bias_gradients)
        """
        n_samples = X.shape[0]
        weight_gradients = []
        bias_gradients = []
        
        # Start with output layer
        y_pred = activations[-1]
        delta = self._backward_output_layer(y_true, y_pred, pre_activations[-1])
        
        # Backpropagate through all layers
        for i in range(self.n_layers - 2, -1, -1):
            # Compute gradients for current layer
            dW = np.dot(activations[i].T, delta) / n_samples
            db = np.mean(delta, axis=0, keepdims=True)
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            # Compute delta for previous layer (if not input layer)
            if i > 0:
                # Backpropagate error
                delta_prev = np.dot(delta, self.weights[i].T)
                
                # Apply derivative of activation function
                activation_name = self.activations[i - 1]
                activation_deriv = get_activation_derivative(activation_name)
                delta = delta_prev * activation_deriv(pre_activations[i - 1])
        
        return weight_gradients, bias_gradients
    
    def update_weights(self, weight_gradients, bias_gradients, learning_rate):
        """
        Update weights and biases using gradient descent.
        
        Args:
            weight_gradients (list): Gradients for weights
            bias_gradients (list): Gradients for biases
            learning_rate (float): Learning rate
        """
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_gradients[i]
            self.biases[i] -= learning_rate * bias_gradients[i]
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        activations, _ = self.forward(X)
        return activations[-1]
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predicted labels
        """
        probabilities = self.predict_proba(X)
        
        if probabilities.shape[1] == 1:
            # Binary classification: convert probabilities to {-1, 1}
            return np.where(probabilities > 0.5, 1, -1)
        else:
            # Multi-class classification: return class indices
            return np.argmax(probabilities, axis=1).reshape(-1, 1)
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01, batch_size=None, 
            validation_data=None, verbose=True, loss_type='cross_entropy'):
        """
        Train the neural network.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            batch_size (int): Batch size for mini-batch gradient descent
            validation_data (tuple): (X_val, y_val) for validation
            verbose (bool): Print training progress
            loss_type (str): Type of loss function
            
        Returns:
            self: Returns self for method chaining
        """
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Set batch size
        if batch_size is None:
            batch_size = X.shape[0]  # Full batch gradient descent
        
        # Reset history
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        if verbose:
            print(f"Training neural network:")
            print(f"  Architecture: {self.layers}")
            print(f"  Activations: {self.activations}")
            print(f"  Epochs: {epochs}, Learning rate: {learning_rate}")
            print(f"  Batch size: {batch_size}")
            print(f"  Loss type: {loss_type}")
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                # Get batch
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                
                # Forward pass
                activations, pre_activations = self.forward(X_batch)
                y_pred = activations[-1]
                
                # Compute loss and accuracy
                batch_loss = self._compute_loss(y_batch, y_pred, loss_type)
                batch_accuracy = self._compute_accuracy(y_batch, y_pred)
                
                # Backward pass
                weight_gradients, bias_gradients = self.backward(
                    X_batch, y_batch, activations, pre_activations)
                
                # Update weights
                self.update_weights(weight_gradients, bias_gradients, learning_rate)
                
                # Accumulate metrics
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                n_batches += 1
            
            # Average metrics over batches
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(avg_accuracy)
            
            # Validation metrics
            if validation_data is not None:
                X_val, y_val = validation_data
                if y_val.ndim == 1:
                    y_val = y_val.reshape(-1, 1)
                
                val_pred = self.predict_proba(X_val)
                val_loss = self._compute_loss(y_val, val_pred, loss_type)
                val_accuracy = self._compute_accuracy(y_val, val_pred)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            if verbose and (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - "
                msg += f"loss: {avg_loss:.4f} - accuracy: {avg_accuracy:.4f}"
                
                if validation_data is not None:
                    msg += f" - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
                
                print(msg)
        
        return self
    
    def evaluate(self, X, y, loss_type='cross_entropy'):
        """
        Evaluate the model on test data.
        
        Args:
            X (np.ndarray): Test features
            y (np.ndarray): Test labels
            loss_type (str): Type of loss function
            
        Returns:
            dict: Evaluation metrics
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        y_pred = self.predict_proba(X)
        loss = self._compute_loss(y, y_pred, loss_type)
        accuracy = self._compute_accuracy(y, y_pred)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': y_pred,
            'predicted_classes': self.predict(X)
        }
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path (str): Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        axes[0].plot(self.history['loss'], 'b-', label='Training Loss', linewidth=2)
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(self.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if self.history['val_accuracy']:
            axes[1].plot(self.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def get_model_summary(self):
        """
        Get a summary of the model architecture and parameters.
        
        Returns:
            dict: Model summary
        """
        total_params = 0
        layer_info = []
        
        for i in range(len(self.weights)):
            layer_params = self.weights[i].size + self.biases[i].size
            total_params += layer_params
            
            layer_info.append({
                'layer': i + 1,
                'input_size': self.weights[i].shape[0],
                'output_size': self.weights[i].shape[1],
                'activation': self.activations[i],
                'parameters': layer_params
            })
        
        return {
            'architecture': self.layers,
            'total_parameters': total_params,
            'layer_details': layer_info
        }


if __name__ == "__main__":
    # Create plots directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    from datasets import make_xor_dataset, create_train_test_split
    
    print("Multi-Layer Perceptron Implementation Test")
    print("=" * 50)
    
    # Test on XOR dataset
    print("\nTesting on XOR Dataset:")
    X, y = make_xor_dataset(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
    
    # Create and train MLP
    mlp = FeedforwardNeuralNet(
        layers=[2, 8, 8, 1],
        activations=['relu', 'relu'],
        output_activation='sigmoid',
        weight_init='xavier',
        random_state=42
    )
    
    # Print model summary
    summary = mlp.get_model_summary()
    print(f"\nModel Architecture:")
    print(f"Layers: {summary['architecture']}")
    print(f"Total parameters: {summary['total_parameters']}")
    
    # Train the model
    mlp.fit(X_train, y_train, epochs=500, learning_rate=0.01, 
            validation_data=(X_test, y_test), verbose=True)
    
    # Evaluate
    train_results = mlp.evaluate(X_train, y_train)
    test_results = mlp.evaluate(X_test, y_test)
    
    print(f"\nResults:")
    print(f"Train accuracy: {train_results['accuracy']:.4f}")
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    
    # Plot training history
    mlp.plot_training_history(save_path="plots/mlp_training_history.png")
    
    print("\nMLP implementation test complete!") 