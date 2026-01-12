import logging
import torch

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def train(self):
        log.info("Starting training...")
        # TODO: Implement manual training loop
        pass
