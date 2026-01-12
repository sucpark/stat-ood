import logging

log = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def load(self):
        log.info(f"Loading dataset: {self.cfg.name}")
        # TODO: Implement HuggingFace dataset loading
        pass
