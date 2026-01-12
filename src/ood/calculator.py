import torch
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

class OODCalculator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.class_means = None
        self.shared_cov = None
        self.precision_matrix = None
        
    def fit(self, features, labels):
        log.info("Fitting OOD Calculator (Calculating Means & Precision Matrix)...")
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        
        unique_classes = np.unique(labels)
        self.class_means = {}
        
        # 1. Calculate class means
        for cls in unique_classes:
            self.class_means[cls] = np.mean(features[labels == cls], axis=0)
            
        # 2. Calculate shared covariance (simplified: assuming tied covariance)
        # Centering data
        centered_features = []
        for i, feature in enumerate(features):
            cls = labels[i]
            centered_features.append(feature - self.class_means[cls])
            
        centered_features = np.array(centered_features)
        
        # Calculate precision matrix directly or via EmpiricalCovariance
        emp_cov = EmpiricalCovariance(assume_centered=True)
        emp_cov.fit(centered_features)
        
        self.precision_matrix = torch.from_numpy(emp_cov.precision_).float()
        
        # Convert means to tensor for faster computation
        # Stack means: [num_classes, feature_dim]
        # We need a mapping from label index to mean vector
        num_labels = self.cfg.model.num_labels
        self.means_tensor = torch.zeros((num_labels, features.shape[1]))
        
        for cls in unique_classes:
             self.means_tensor[cls] = torch.from_numpy(self.class_means[cls]).float()

        log.info("OOD Calculator fitting complete.")

    def compute_mahalanobis_score(self, feature, mean, precision):
        # feature: [feature_dim], mean: [feature_dim], precision: [feature_dim, feature_dim]
        # dist = (x - mu)^T * Sigma^-1 * (x - mu)
        diff = feature - mean
        return torch.matmul(torch.matmul(diff, precision), diff)
        
    def predict(self, features):
        """
        Calculate Mahalanobis distance for each sample to the nearest class mean.
        Returns the minimum distance (or negative max score).
        """
        # features: [batch_size, feature_dim]
        features = features.cpu() 
        means = self.means_tensor # [num_classes, feature_dim]
        precision = self.precision_matrix # [feature_dim, feature_dim]
        
        batch_size = features.size(0)
        
        dists = []
        for c in range(means.size(0)):
             # We only compute for classes that have means (simple optimization for zeros?)
             # Actually self.means_tensor might have zeros for classes not in training batch?
             # For now assume mostly full.
             
             mean = means[c] # [feature_dim]
             diff = features - mean # [batch_size, feature_dim]
             temp = torch.matmul(diff, precision)
             term = torch.sum(temp * diff, dim=1) # [batch_size]
             dists.append(term)
             
        dists = torch.stack(dists, dim=1) # [batch_size, num_classes]
        
        # We take the minimum distance (closest class)
        min_dists, _ = torch.min(dists, dim=1)
        
        return min_dists

    def evaluate(self, id_scores, ood_scores):
        """
        Compute AUROC and FPR@TPR95
        id_scores: distances for ID data (lower is better/more confident, but Mahalanobis is distance)
                   Actually, Mahalanobis Distance is HIGHER for OOD.
                   So ID should have LOW distance, OOD should have HIGH distance.
                   We can treat -distance as "confidence score".
                   ID: close to 0 (confidence high). OOD: large (confidence low).
                   Score = -distance.
        """
        id_scores = -id_scores.cpu().numpy()
        ood_scores = -ood_scores.cpu().numpy()
        
        y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)]) # 1: ID, 0: OOD
        y_scores = np.concatenate([id_scores, ood_scores])
        
        # AUROC
        auroc = roc_auc_score(y_true, y_scores)
        
        # FPR @ TPR 95
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        # Find FPR when TPR >= 0.95
        # Since sklearn roc_curve returns increasing FPR, we look for first index where TPR >= 0.95
        idx = np.where(tpr >= 0.95)[0]
        if len(idx) > 0:
            fpr95 = fpr[idx[0]]
        else:
            fpr95 = 1.0 # Should not happen if curve is complete
            
        return auroc, fpr95
