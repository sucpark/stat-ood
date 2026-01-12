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
        
        # Pooling strategy
        # Default to 'cls' (original BERT behavior) if not specified
        self.pooling = getattr(cfg, 'pooling', 'cls')
        log.info(f"Initialized ModelWrapper with pooling strategy: {self.pooling}")
        
        # Hook storage
        self.features = {}
        
    def forward(self, input_ids, attention_mask, labels=None):
        # 1. Encoder forward
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. Pooling
        if self.pooling == 'cls':
            # Use pooler_output if available (BERT), else use first token features
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                # Fallback for models without pooler (e.g. DistilBERT sometimes)
                pooled_output = outputs.last_hidden_state[:, 0]
        elif self.pooling == 'mean':
            pooled_output = self._mean_pooling(outputs.last_hidden_state, attention_mask)
        elif self.pooling == 'max':
            pooled_output = self._max_pooling(outputs.last_hidden_state, attention_mask)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # 3. Classifier forward
        logits = self.classifier(pooled_output)
        
        # 4. Store features (used for OOD calculation later)
        self.features['pooled_output'] = pooled_output
        self.features['logits'] = logits
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.cfg.num_labels), labels.view(-1))
            
        return loss, logits

    def _mean_pooling(self, last_hidden_state, attention_mask):
        # Mask [batch, seq_len] -> [batch, seq_len, 1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum of embeddings (observing mask)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        
        # Count of non-padding tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def _max_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        # Set padding tokens to large negative value so they don't affect max
        last_hidden_state[input_mask_expanded == 0] = -1e9
        return torch.max(last_hidden_state, 1)[0]

    def get_features(self, key='pooled_output'):
        return self.features.get(key, None)
