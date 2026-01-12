import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import logging

log = logging.getLogger(__name__)

class ModelWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, cfg.num_labels)
        
        # Hook storage
        self.features = {}
        
    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Encoder forward
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output # [batch_size, hidden_size]
        
        # 2. Classifier forward
        logits = self.classifier(pooled_output)
        
        # 3. Store features (used for OOD calculation later)
        # We store pooled_output which is often used for classification
        self.features['pooled_output'] = pooled_output
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.cfg.num_labels), labels.view(-1))
            
        return loss, logits

    def get_features(self):
        return self.features.get('pooled_output', None)
