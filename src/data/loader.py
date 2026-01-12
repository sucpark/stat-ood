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
        self.oos_label = 150 # 'oos' label in clinc_oos is 150
        
    def load(self):
        log.info(f"Loading dataset: {self.cfg.name}")
        
        # Load dataset from HuggingFace
        if self.cfg.name == "clinc_oos":
            self.dataset = load_dataset("clinc_oos", self.cfg.subset)
        else:
            raise ValueError(f"Dataset {self.cfg.name} not supported.")

        log.info(f"Dataset loaded. Keys: {self.dataset.keys()}")
        
        # Filter OOD from TRAIN and validation ID
        # We need to filter 'train' split to only have IDs.
        # 'clinc_oos' usually has 'oos' in train/val/test splits depending on subset
        
        # Preprocess
        encoded_dataset = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["text"]
        )
        
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'intent'])

        # Filter function
        def is_id(example):
            return example['intent'] != self.oos_label
            
        def is_ood(example):
            return example['intent'] == self.oos_label

        # TRAIN: ID ONLY
        train_ds = encoded_dataset['train'].filter(is_id)
        
        # VALIDATION: ID ONLY (for model selection)
        val_ds = encoded_dataset['validation'].filter(is_id)
        
        # TEST: Split into ID and OOD for evaluation
        test_ds = encoded_dataset['test']
        test_id_ds = test_ds.filter(is_id)
        test_ood_ds = test_ds.filter(is_ood)
        
        log.info(f"Train (ID) size: {len(train_ds)}")
        log.info(f"Val (ID) size: {len(val_ds)}")
        log.info(f"Test (ID) size: {len(test_id_ds)}")
        log.info(f"Test (OOD) size: {len(test_ood_ds)}")

        # Create DataLoaders
        train_loader = TorchDataLoader(
            train_ds,
            batch_size=self.cfg.loader.batch_size,
            shuffle=True,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=self.cfg.loader.pin_memory
        )
        
        val_loader = TorchDataLoader(
            val_ds,
            batch_size=self.cfg.loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=self.cfg.loader.pin_memory
        )

        test_loader = TorchDataLoader(
            test_id_ds,
            batch_size=self.cfg.loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=self.cfg.loader.pin_memory
        )
        
        # Add OOD loader
        ood_loader = TorchDataLoader(
            test_ood_ds,
            batch_size=self.cfg.loader.batch_size,
            shuffle=False,
            num_workers=self.cfg.loader.num_workers,
            pin_memory=self.cfg.loader.pin_memory
        )
        
        return train_loader, val_loader, test_loader, ood_loader

    def preprocess_function(self, examples):
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding="max_length",
            max_length=self.cfg.maxlen
        )
