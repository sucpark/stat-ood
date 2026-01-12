import logging
from datasets import load_dataset
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.dataset = None
        
    def load(self):
        log.info(f"Loading dataset: {self.cfg.name}")
        
        # Load dataset from HuggingFace
        if self.cfg.name == "clinc_oos":
            self.dataset = load_dataset("clinc_oos", self.cfg.subset)
        else:
            raise ValueError(f"Dataset {self.cfg.name} not supported.")

        log.info(f"Dataset loaded. Keys: {self.dataset.keys()}")
        
        # Preprocess
        encoded_dataset = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["text"]  # Remove raw text to keep only tensors
        )
        
        # Set format for PyTorch
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'intent']) # 'intent' is the label
        
        # Create DataLoaders
        train_loader = TorchDataLoader(
            encoded_dataset['train'],
            batch_size=self.cfg.loader.batch_size,
            shuffle=True,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=self.cfg.loader.pin_memory
        )
        
        val_loader = TorchDataLoader(
            encoded_dataset['validation'],
            batch_size=self.cfg.loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=self.cfg.loader.pin_memory
        )

        test_loader = TorchDataLoader(
            encoded_dataset['test'],
            batch_size=self.cfg.loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=self.cfg.loader.pin_memory
        )
        
        return train_loader, val_loader, test_loader

    def preprocess_function(self, examples):
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding="max_length",
            max_length=self.cfg.maxlen
        )
