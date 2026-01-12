import torch.nn as nn
import logging

log = logging.getLogger(__name__)

class ModelWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # TODO: Initialize model
        
    def forward(self, input_ids, attention_mask, labels=None):
        # TODO: Implement forward pass
        pass
