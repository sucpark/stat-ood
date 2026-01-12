import logging

log = logging.getLogger(__name__)

class OODCalculator:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def fit(self, features, labels):
        # TODO: Calculate class means and covariance
        pass
        
    def predict(self, features):
        # TODO: Calculate Mahalanobis distance
        pass
