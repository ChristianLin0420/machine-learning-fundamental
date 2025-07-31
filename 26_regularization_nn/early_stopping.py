"""
Early Stopping Utility for Neural Network Training

This module provides a comprehensive early stopping implementation to prevent
overfitting during neural network training.

Author: ML Fundamentals Course
"""

import numpy as np
from typing import Optional, Union, Callable
import warnings

class EarlyStopping:
    """
    Early stopping utility to halt training when validation performance stops improving.
    
    This implementation provides several modes of early stopping:
    - Simple patience-based stopping
    - Minimum improvement threshold
    - Relative improvement monitoring
    - Best model restoration
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 restore_best_weights: bool = True,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 baseline: Optional[float] = None,
                 verbose: bool = True):
        """
        Initialize Early Stopping
        
        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore model weights from best epoch
            monitor: Metric to monitor ('val_loss', 'val_accuracy', etc.)
            mode: 'min' for metrics to minimize, 'max' for metrics to maximize
            baseline: Baseline value for the monitored metric
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.monitor = monitor
        self.mode = mode.lower()
        self.baseline = baseline
        self.verbose = verbose
        
        # Validation
        if self.mode not in ['min', 'max']:
            raise ValueError("Mode must be 'min' or 'max'")
        
        # Internal state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        
        # Set comparison functions based on mode
        if self.mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
            if self.baseline is not None:
                self.best = self.baseline
        else:
            self.monitor_op = np.greater
            self.best = -np.inf
            if self.baseline is not None:
                self.best = self.baseline
    
    def __call__(self, current_value: float, model_weights: Optional[list] = None) -> bool:
        """
        Check if training should be stopped
        
        Args:
            current_value: Current value of the monitored metric
            model_weights: Current model weights (for restoration)
            
        Returns:
            True if training should be stopped, False otherwise
        """
        return self.check(current_value, model_weights)
    
    def check(self, current_value: float, model_weights: Optional[list] = None) -> bool:
        """
        Check if training should be stopped
        
        Args:
            current_value: Current value of the monitored metric
            model_weights: Current model weights (for restoration)
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if self.monitor_op(current_value - self.min_delta, self.best):
            # Improvement detected
            self.best = current_value
            self.wait = 0
            self.best_epoch = self.get_current_epoch()
            
            # Store best weights if requested
            if self.restore_best_weights and model_weights is not None:
                self.best_weights = [w.copy() for w in model_weights]
            
            if self.verbose:
                print(f"Improvement detected: {self.monitor} = {current_value:.6f}")
        
        else:
            # No improvement
            self.wait += 1
            
            if self.verbose and self.wait > 0:
                print(f"No improvement for {self.wait} epoch(s). "
                      f"Best {self.monitor}: {self.best:.6f} at epoch {self.best_epoch}")
        
        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = self.get_current_epoch()
            if self.verbose:
                print(f"Early stopping triggered at epoch {self.stopped_epoch}")
                print(f"Best {self.monitor}: {self.best:.6f} at epoch {self.best_epoch}")
            return True
        
        return False
    
    def get_current_epoch(self) -> int:
        """Get current epoch number (can be overridden)"""
        # This is a simple counter - in practice, you might pass epoch number
        return getattr(self, '_current_epoch', self.wait + self.best_epoch)
    
    def set_current_epoch(self, epoch: int):
        """Set current epoch number"""
        self._current_epoch = epoch
    
    def get_best_weights(self) -> Optional[list]:
        """Get the best model weights"""
        return self.best_weights
    
    def reset(self):
        """Reset the early stopping state"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        
        if self.mode == 'min':
            self.best = np.inf
            if self.baseline is not None:
                self.best = self.baseline
        else:
            self.best = -np.inf
            if self.baseline is not None:
                self.best = self.baseline
    
    def get_state(self) -> dict:
        """Get the current state of early stopping"""
        return {
            'wait': self.wait,
            'best': self.best,
            'best_epoch': self.best_epoch,
            'stopped_epoch': self.stopped_epoch,
            'patience': self.patience
        }

class AdaptiveEarlyStopping(EarlyStopping):
    """
    Adaptive early stopping that adjusts patience based on training progress
    """
    
    def __init__(self, 
                 initial_patience: int = 10,
                 patience_factor: float = 1.5,
                 max_patience: int = 50,
                 improvement_threshold: float = 0.01,
                 **kwargs):
        """
        Initialize Adaptive Early Stopping
        
        Args:
            initial_patience: Initial patience value
            patience_factor: Factor to multiply patience when improvement is detected
            max_patience: Maximum patience value
            improvement_threshold: Threshold for significant improvement
            **kwargs: Additional arguments for base EarlyStopping
        """
        super().__init__(patience=initial_patience, **kwargs)
        self.initial_patience = initial_patience
        self.patience_factor = patience_factor
        self.max_patience = max_patience
        self.improvement_threshold = improvement_threshold
        self.last_significant_improvement = 0
    
    def check(self, current_value: float, model_weights: Optional[list] = None) -> bool:
        """Check with adaptive patience adjustment"""
        # Calculate improvement
        if self.mode == 'min':
            improvement = self.best - current_value
        else:
            improvement = current_value - self.best
        
        # Check for significant improvement
        if improvement > self.improvement_threshold:
            # Significant improvement - potentially increase patience
            new_patience = min(int(self.patience * self.patience_factor), self.max_patience)
            if new_patience > self.patience and self.verbose:
                print(f"Significant improvement detected. Increasing patience from "
                      f"{self.patience} to {new_patience}")
            self.patience = new_patience
            self.last_significant_improvement = self.get_current_epoch()
        
        return super().check(current_value, model_weights)

class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving
    """
    
    def __init__(self,
                 factor: float = 0.1,
                 patience: int = 10,
                 min_lr: float = 1e-7,
                 mode: str = 'min',
                 min_delta: float = 1e-4,
                 cooldown: int = 0,
                 verbose: bool = True):
        """
        Initialize ReduceLROnPlateau
        
        Args:
            factor: Factor by which learning rate will be reduced
            patience: Number of epochs with no improvement to wait
            min_lr: Minimum learning rate
            mode: 'min' or 'max'
            min_delta: Threshold for measuring improvement
            cooldown: Number of epochs to wait before resuming normal operation
            verbose: Whether to print messages
        """
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.verbose = verbose
        
        # State
        self.wait = 0
        self.cooldown_counter = 0
        self.lr_reductions = 0
        
        if self.mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        else:
            self.monitor_op = np.greater
            self.best = -np.inf
    
    def __call__(self, current_value: float, current_lr: float) -> float:
        """
        Check if learning rate should be reduced
        
        Args:
            current_value: Current value of monitored metric
            current_lr: Current learning rate
            
        Returns:
            New learning rate (same as current if no change)
        """
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return current_lr
        
        if self.monitor_op(current_value - self.min_delta, self.best):
            self.best = current_value
            self.wait = 0
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            new_lr = max(current_lr * self.factor, self.min_lr)
            if new_lr < current_lr:
                if self.verbose:
                    print(f"Reducing learning rate from {current_lr:.6f} to {new_lr:.6f}")
                self.lr_reductions += 1
                self.wait = 0
                self.cooldown_counter = self.cooldown
                return new_lr
        
        return current_lr

def demonstrate_early_stopping():
    """Demonstrate early stopping functionality"""
    print("Early Stopping Demonstration")
    print("=" * 40)
    
    # Simulate a training scenario
    print("\n1. Basic Early Stopping:")
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)
    
    # Simulate validation losses (decreasing then increasing)
    val_losses = [1.0, 0.8, 0.6, 0.5, 0.49, 0.48, 0.481, 0.482, 0.485, 0.49, 0.5, 0.52]
    
    for epoch, loss in enumerate(val_losses):
        early_stopping.set_current_epoch(epoch + 1)
        should_stop = early_stopping.check(loss)
        print(f"Epoch {epoch + 1}: Loss = {loss:.3f}, Should stop: {should_stop}")
        if should_stop:
            break
    
    print(f"\nFinal state: {early_stopping.get_state()}")
    
    # Demonstrate adaptive early stopping
    print("\n2. Adaptive Early Stopping:")
    adaptive_es = AdaptiveEarlyStopping(
        initial_patience=3,
        patience_factor=1.5,
        improvement_threshold=0.05,
        verbose=True
    )
    
    # Reset and test
    adaptive_es.reset()
    for epoch, loss in enumerate(val_losses):
        adaptive_es.set_current_epoch(epoch + 1)
        should_stop = adaptive_es.check(loss)
        print(f"Epoch {epoch + 1}: Loss = {loss:.3f}, Patience = {adaptive_es.patience}, Should stop: {should_stop}")
        if should_stop:
            break
    
    # Demonstrate learning rate reduction
    print("\n3. Learning Rate Reduction:")
    lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=3, verbose=True)
    
    current_lr = 0.01
    for epoch, loss in enumerate(val_losses):
        new_lr = lr_scheduler(loss, current_lr)
        if new_lr != current_lr:
            current_lr = new_lr
        print(f"Epoch {epoch + 1}: Loss = {loss:.3f}, LR = {current_lr:.6f}")

if __name__ == "__main__":
    demonstrate_early_stopping() 