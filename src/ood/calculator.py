import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import logging
from .mahalanobis import MahalanobisStrategy
from .energy import EnergyStrategy

log = logging.getLogger(__name__)

class OODCalculator:
    def __init__(self, cfg):
        self.cfg = cfg
        method = getattr(cfg, 'ood_method', 'mahalanobis')
        
        if method == 'mahalanobis':
            self.strategy = MahalanobisStrategy(cfg)
        elif method == 'energy':
            self.strategy = EnergyStrategy(cfg)
        else:
            raise ValueError(f"Unknown OOD method: {method}")
        
        log.info(f"OOD Calculator Strategy: {method}")
        
    def fit(self, features, labels):
        self.strategy.fit(features, labels)
        
    def predict(self, features):
        return self.strategy.predict(features)

    def evaluate(self, id_scores, ood_scores):
        """
        Compute AUROC and FPR@TPR95.
        Assumes scores are "OOD-ness" (Higher = OOD).
        """
        id_scores = id_scores.cpu().numpy()
        ood_scores = ood_scores.cpu().numpy()
        
        # True labels: 0 for ID, 1 for OOD (Standard for binary classification where Positive Class is OOD)
        # OR: 1 for ID, 0 for OOD?
        # Standard OOD detection:
        # Task: Detect OOD. Positive Class = OOD.
        # ID Scores should be LOW. OOD Scores should be HIGH.
        
        y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
        y_scores = np.concatenate([id_scores, ood_scores])
        
        # AUROC (Probability that OOD sample has higher score than ID sample)
        auroc = roc_auc_score(y_true, y_scores)
        
        # FPR @ TPR 95
        # Recall (TPR) is fraction of OOD detected.
        # FPR is fraction of ID misclassified as OOD.
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Find FPR when TPR >= 0.95 (95% of OOD detected)
        idx = np.where(tpr >= 0.95)[0]
        if len(idx) > 0:
            fpr95 = fpr[idx[0]]
        else:
            fpr95 = 1.0
            
        return auroc, fpr95
