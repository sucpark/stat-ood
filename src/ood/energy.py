import torch
import logging
from .base import OODStrategy

log = logging.getLogger(__name__)

class EnergyStrategy(OODStrategy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.temperature = getattr(cfg.model, 'temperature', 1.0)
        
    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        # Energy score doesn't need fitting (non-parametric in feature space, relies on weights)
        # But commonly we might want to calibrate temperature. 
        # For this baseline, we do nothing.
        log.info("Energy Strategy: No fitting required (Hyperparameter: Temperature).")
        pass

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        # Energy score requires LOGITS.
        # We assume 'features' passed here might be the dict from ModelWrapper OR we need to fetch logits differently.
        # But OODCalculator.predict(features) passes 'features' which is model.get_features().
        # ModelWrapper.get_features() currently returns 'pooled_output' by default.
        # We need to change ModelWrapper.get_features() to return a dict or object if we want generic support?
        # OR we just change EnergyStrategy to expect 'logits' in the input tensor?
        # WAIT: predict(features) expects a Tensor.
        # If we use Energy, we must ensure the input to predict() IS logits.
        
        # However, standard interface 'predict(features)' implies feature embedding.
        # Energy is computed on Logits.
        # If we want to support both, we have a disconnect.
        
        # SOLUTION: The input 'features' to predict() is whatever model.get_features() returns.
        # For Mahalanobis: It needs Pooled Output.
        # For Energy: It needs Logits.
        
        # We should make model.get_features() return what is needed based on config, or return a Dict.
        # But predict() signature is usually Tensor.
        
        # Let's assume for this implementation that 'features' passed in IS LOGITS.
        # We will control this in main.py or OODCalculator.
        
        logits = features
        
        # Energy = -T * log(sum(exp(x/T)))
        # Score should be higher for OOD.
        # Energy is usually LOWER for ID (high confidence), HIGHER for OOD.
        # So Energy itself is a good OOD score.
        
        energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        return energy
