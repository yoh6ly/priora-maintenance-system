"""
Model management and inference for PRIORA System
"""

import joblib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage model loading and inference"""
    
    def __init__(self, model_path: str = 'models'):
        """
        Initialize ModelManager.
        
        Args:
            model_path (str): Path to models directory
        """
        self.model_path = Path(model_path)
        self.models = {}
        self.scalers = {}
        self._load_models()
    
    def _load_models(self):
        """Load all pre-trained models and scalers"""
        try:
            # Load motor type classifier
            self.models['type_classifier'] = joblib.load(
                self.model_path / 'clf_tipo.pkl'
            )
            self.scalers['type_scaler'] = joblib.load(
                self.model_path / 'scaler_tipo.pkl'
            )
            
            # Load AC 3-phase models
            self.models['ac3_detector'] = joblib.load(
                self.model_path / 'modelo_ac3.pkl'
            )
            self.scalers['ac3_scaler'] = joblib.load(
                self.model_path / 'scaler_ac3.pkl'
            )
            
            # Load AC single-phase models
            self.models['ac1_detector'] = joblib.load(
                self.model_path / 'modelo_ac1.pkl'
            )
            self.scalers['ac1_scaler'] = joblib.load(
                self.model_path / 'scaler_ac1.pkl'
            )
            
            # Load DC brushed models
            self.models['dc_detector'] = joblib.load(
                self.model_path / 'modelo_dc.pkl'
            )
            self.scalers['dc_scaler'] = joblib.load(
                self.model_path / 'scaler_dc.pkl'
            )
            
            logger.info("All models loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
    
    def classify_motor_type(self, features: np.ndarray) -> int:
        """
        Classify motor type.
        
        Args:
            features (np.ndarray): Scaled features
            
        Returns:
            int: Motor type (0=AC 3-phase, 1=AC single-phase, 2=DC)
        """
        scaled_features = self.scalers['type_scaler'].transform(features)
        motor_type = self.models['type_classifier'].predict(scaled_features)[0]
        return int(motor_type)
    
    def detect_fault(self, features: np.ndarray, motor_type: int) -> Dict[str, Any]:
        """
        Detect fault probability for specific motor type.
        
        Args:
            features (np.ndarray): Input features
            motor_type (int): Motor type identifier
            
        Returns:
            dict: Prediction results with probability
        """
        motor_models = {
            0: ('ac3_detector', 'ac3_scaler'),
            1: ('ac1_detector', 'ac1_scaler'),
            2: ('dc_detector', 'dc_scaler')
        }
        
        model_key, scaler_key = motor_models[motor_type]
        
        # Scale features
        scaled_features = self.scalers[scaler_key].transform(features)
        
        # Get prediction
        model = self.models[model_key]
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        return {
            'prediction': prediction,
            'probability': probability,
            'fault_probability': probability[1] * 100  # Convert to percentage
        }
    
    def save_model(self, model_name: str, model: Any, overwrite: bool = False):
        """
        Save a model to disk.
        
        Args:
            model_name (str): Model name/identifier
            model (Any): Model object to save
            overwrite (bool): Whether to overwrite existing model
        """
        model_file = self.model_path / f"{model_name}.pkl"
        
        if model_file.exists() and not overwrite:
            logger.warning(f"Model {model_name} already exists. Set overwrite=True to replace")
            return
        
        joblib.dump(model, model_file)
        logger.info(f"Model {model_name} saved successfully")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            dict: Model information
        """
        return {
            'models_loaded': list(self.models.keys()),
            'scalers_loaded': list(self.scalers.keys()),
            'model_path': str(self.model_path)
        }
