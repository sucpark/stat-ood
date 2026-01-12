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
        
        # Load dataset
        if self.cfg.name == "clinc_oos":
            self.dataset = load_dataset("clinc_oos", self.cfg.subset)
            oos_label = 150
            # Define ID/OOD check
            def is_id(example): return example['intent'] != oos_label
            def is_ood(example): return example['intent'] == oos_label
            
        elif self.cfg.name == "massive":
            # qanastek/MASSIVE or AmazonScience/massive
            # We use qanastek/MASSIVE as it's often more accessible formatting
            self.dataset = load_dataset("qanastek/MASSIVE", self.cfg.locale)
            # MASSIVE has 60 intents (0-59). No explicit OOD.
            # Research Mode: Holdout classes for OOD.
            # e.g. Train on 0-49, Test on 0-59. (50-59 are OOD)
            holdout_threshold = self.cfg.get('holdout_threshold', 50)
            
            def is_id(example): return example['intent'] < holdout_threshold
            def is_ood(example): return example['intent'] >= holdout_threshold
            
            log.info(f"MASSIVE mode: Treating labels >= {holdout_threshold} as OOD.")
            
        else:
            raise ValueError(f"Dataset {self.cfg.name} not supported.")

        # Common Preprocessing
        # Filter raw dataset first? No, filter after mapping usually better for consistency or before to save time.
        # Let's filter before to map only needed data? 
        # Actually mapping is fast.
        
        # Preprocess text
        encoded_dataset = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=[col for col in self.dataset['train'].column_names if col != 'intent'] 
            # Remove all text/metadata, keep 'intent' and tensors
        )
        
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'intent'])

        # Split logic
        train_ds = encoded_dataset['train'].filter(is_id)
        val_ds = encoded_dataset['validation'].filter(is_id)
        
        # Test: Split ID and OOD
        test_ds = encoded_dataset['test']
        test_id_ds = test_ds.filter(is_id)
        test_ood_ds = test_ds.filter(is_ood)
        
        log.info(f"Train (ID) size: {len(train_ds)}")
        log.info(f"Val (ID) size: {len(val_ds)}")
        log.info(f"Test (ID) size: {len(test_id_ds)}")
        log.info(f"Test (OOD) size: {len(test_ood_ds)}")

        # Create Loaders
        def create_loader(ds, shuffle=False):
            return TorchDataLoader(
                ds,
                batch_size=self.cfg.loader.batch_size,
                shuffle=shuffle,
                num_workers=self.cfg.loader.num_workers,
                pin_memory=self.cfg.loader.pin_memory
            )
            
        return create_loader(train_ds, True), create_loader(val_ds), create_loader(test_id_ds), create_loader(test_ood_ds)

    def preprocess_function(self, examples):
        # Handle different column names
        text_col = 'utt' if 'utt' in examples else 'text'
        return self.tokenizer(
            examples[text_col],
            truncation=True,
            padding="max_length",
            max_length=self.cfg.maxlen
        )
