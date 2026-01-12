import torch
import numpy as np
from sklearn.covariance import EmpiricalCovariance
import logging
from .base import OODStrategy

log = logging.getLogger(__name__)

class MahalanobisStrategy(OODStrategy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.class_means = None
        self.precision_matrix = None
        self.means_tensor = None
        
    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        log.info("Fitting Mahalanobis Strategy...")
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        
        unique_classes = np.unique(labels)
        self.class_means = {}
        
        # 1. Calculate class means
        for cls in unique_classes:
            self.class_means[cls] = np.mean(features[labels == cls], axis=0)
            
        # 2. Calculate shared covariance
        centered_features = []
        for i, feature in enumerate(features):
            cls = labels[i]
            centered_features.append(feature - self.class_means[cls])
        centered_features = np.array(centered_features)
        
        # Empirical Covariance & Precision
        emp_cov = EmpiricalCovariance(assume_centered=True)
        emp_cov.fit(centered_features)
        self.precision_matrix = torch.from_numpy(emp_cov.precision_).float()
        
        # Convert means to tensor
        num_labels = self.cfg.model.num_labels
        self.means_tensor = torch.zeros((num_labels, features.shape[1]))
        for cls in unique_classes:
             self.means_tensor[cls] = torch.from_numpy(self.class_means[cls]).float()
             
        log.info("Mahalanobis fitting complete.")

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        features = features.cpu() 
        means = self.means_tensor
        precision = self.precision_matrix
        
        # Compute Mahalanobis distance to nearest class mean
        dists = []
        # Optimization: matrix operations for all classes
        # (x - mu)T Sigma^-1 (x - mu)
        # Can differ from `sklearn.distance.mahalanobis` which takes V^-1.
        
        for c in range(means.size(0)):
             # Only if class mean is non-zero (or we assume all classes exist in config)
             # But zero-mean classes (never seen) will yield distance to 0 vector, which might be weird.
             # Ideally we track seen classes.
             
             mean = means[c]
             diff = features - mean
             temp = torch.matmul(diff, precision)
             term = torch.sum(temp * diff, dim=1)
             dists.append(term)
             
        dists = torch.stack(dists, dim=1) # [batch, num_classes]
        min_dists, _ = torch.min(dists, dim=1)
        
        return min_dists # Higher Mahalanobis distance = OOD
