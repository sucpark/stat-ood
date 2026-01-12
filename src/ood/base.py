from abc import ABC, abstractmethod
import torch

class OODStrategy(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        
    @abstractmethod
    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Fit the OOD detector using training ID features and labels.
        """
        pass
        
    @abstractmethod
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """
        Calculate OOD scores for input features.
        Higher score should indicate OOD (typically).
        If the metric is a distance, higher is OOD.
        If the metric is confidence (like MSP), lower is OOD. 
        Standardize: Return 'Distance' or 'Uncertainty' (Higher = OOD).
        """
        pass
