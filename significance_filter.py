"""
Project Mnemosyne - Significance Filter
Real-time analysis of state deltas using isolation forests for anomaly detection.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, List, Tuple, Optional
import logging
from dataclasses import dataclass
import time

@dataclass
class StateDelta:
    """Encapsulates state change data with metadata"""
    timestamp: float
    delta_vector: np.ndarray
    context: Dict[str, Any]
    source: str
    raw_state: Dict[str, Any]
    
    def __post_init__(self):
        """Validate delta vector"""
        if not isinstance(self.delta_vector, np.ndarray):
            raise TypeError("delta_vector must be a numpy array")
        if len(self.delta_vector.shape) != 1:
            raise ValueError("delta_vector must be 1-dimensional")

class SignificanceFilter:
    """
    Filters state changes based on learned significance thresholds.
    Uses incremental learning to adapt to changing state patterns.
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize significance filter
        
        Args:
            contamination: Expected proportion of outliers in the data
            random_state: Random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            warm_start=True  # Enable incremental learning
        )
        
        # Training buffer for incremental learning
        self.training_buffer: List[np.ndarray] = []
        self.buffer_size = 1000
        self.is_fitted = False
        
        # Track state patterns for consistency checking
        self.state_history: List[np.ndarray] = []
        self.history_size = 100
        
        self.logger.info("SignificanceFilter initialized with incremental learning")

    def calculate_delta(self, current_state: Dict[str, Any], 
                       previous_state: Dict[str, Any]) -> np.ndarray:
        """
        Calculate normalized delta vector between states
        
        Args:
            current_state: Current state dictionary
            previous_state: Previous state dictionary
            
        Returns:
            Normalized delta vector as numpy array
            
        Raises:
            ValueError: If states have incompatible structures
        """
        try:
            # Handle empty previous state
            if not previous_state:
                # First state - return zero vector with appropriate dimension
                # Estimate dimension from current state
                sample_vector = self._dict_to_vector(current_state)
                return np.zeros_like(sample_vector)
            
            # Convert dictionaries to comparable vectors
            current_vec = self._dict_to_vector(current_state)
            previous_vec = self._dict_to_vector(previous_state)
            
            # Ensure vectors have same dimension
            if len(current_vec) != len(previous_vec):
                self.logger.warning("State dimension mismatch, attempting alignment")
                # Align vectors by padding with zeros
                max_len = max(len(current_vec), len(previous_vec))
                current_vec = self._pad_vector(current_vec, max_len)
                previous_vec = self._pad_vector(previous_vec, max_len)
            
            # Calculate delta and normalize
            delta = current_vec - previous_vec
            norm_delta = delta / (np.abs(previous_vec) + 1e-8)  # Avoid division by zero
            
            return norm_delta
            
        except Exception as e:
            self.logger.error(f"Failed to calculate delta: {e}")
            # Return zero vector as fallback
            sample_vector = self._dict_to_vector(current_state or previous_state)
            return np.zeros_like(sample_vector) if sample_vector.size > 0 else np.array([0.0])

    def assess_significance(self, delta: np.ndarray) -> Tuple[bool, float, str]:
        """
        Assess significance of state delta using learned model
        
        Args:
            delta: Normalized delta vector
            
        Returns:
            Tuple of (is_significant, significance_score, reason)
        """
        try:
            # Check for minimum delta magnitude
            delta_magnitude = np.linalg.norm(delta)
            if delta_magnitude < 1e-6:  # Essentially no change
                return False, 0.0, "No measurable state change"
            
            # Prepare data for prediction
            X = delta.reshape(1, -1)
            
            # If model is not fitted yet, use simple threshold
            if not self.is_fitted:
                # Initial threshold based on magnitude
                is_sig = delta_magnitude > 0.1  # Initial threshold
                score = delta_magnitude
                reason = "Unfitted model - using magnitude threshold"
                return is_sig, float(score), reason
            
            # Get anomaly score (negative means more anomalous/significant)
            anomaly_score = self.isolation_forest.score_samples(X)[0]
            
            # Convert to significance score (0-1, higher = more significant)
            # Isolation Forest returns negative values for anomalies
            significance_score = max(0.0, -anomaly_score)
            
            # Apply learned threshold
            is_significant = significance_score > 0.7  # Configurable threshold
            
            reason = "Significant state anomaly" if is_significant else "Normal state variation"
            
            # Update training buffer for incremental learning
            self._update_training_buffer(delta)
            
            return is_significant, significance_score, reason
            
        except Exception as e:
            self.logger.error(f"Significance assessment failed: {e}")
            # Conservative fallback: treat as significant if we can't assess
            return True, 1.0, f"Assessment failed: {str(e)}"

    def update_model(self, new_data: List[np.ndarray]):
        """
        Incrementally update the isolation forest model
        
        Args:
            new_data: List of new delta vectors for training
        """
        if not new_data:
            return
            
        try:
            # Add to training buffer
            self.training_buffer.extend(new_data)
            
            # Trim buffer if too large
            if len(self.training_buffer) > self.buffer_size:
                self.training_buffer = self.training_buffer[-self.buffer_size:]
            
            # Convert to numpy array
            X_train = np.array(self.training_buffer)
            
            # Ensure we have enough samples
            if len(X_train) < 10:  # Minimum samples for meaningful training
                return
            
            # Fit or partial fit
            if not self.is_fitted:
                self.isolation_forest.fit(X_train)
                self.is_fitted = True
                self.logger.info(f"Initial model fitted with {len(X_train)} samples")
            else:
                # For IsolationForest with warm_start, we need to refit
                # Note: scikit-learn's IsolationForest doesn't support partial_fit
                # We'll refit periodically instead
                if len(self.training_buffer) % 100 == 0:  # Refit every 100 new samples
                    self.isolation_forest.fit(X_train)
                    self.logger.debug(f"Model refitted with {len(X_train)} samples")
                    
        except Exception as e:
            self.logger.error(f"Model update failed: {e}")

    def _dict_to_vector(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Convert state dictionary to numerical vector"""
        vector_parts = []
        
        def extract_values(obj, prefix=""):
            if isinstance(obj, (int, float)):
                vector_parts.append(float(obj))
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    extract_values(value, f"{prefix}{key}.")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_values(item, f"{prefix}[{i}].")
            elif isinstance(obj, bool):
                vector_parts.append(1.0 if obj else 0.0)
            elif obj is None:
                vector_parts.append(0.0)
            # Skip strings and other non-numeric types
        
        extract_values(state_dict)
        
        if not vector_parts:  # Handle empty state
            return np.array([0.0])
            
        return np.array(vector_parts)

    def _pad_vector(self, vector: np.ndarray, target_length: int) -> np.ndarray:
        """Pad vector to target length with zeros"""
        if len(vector) >= target_length:
            return vector[:target_length]
        
        padded = np.zeros(target_length)
        padded[:len(vector)] = vector
        return padded

    def _update_training_buffer(self, delta: np.ndarray):
        """Update training buffer with new delta"""
        self.training_buffer.append(delta)
        if len(self.training_buffer) > self.buffer_size:
            self.training_buffer.pop(0)