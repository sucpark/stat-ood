import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from src.data.loader import DataLoader
from src.models.wrapper import ModelWrapper
from src.engine.trainer import Trainer
from src.ood.calculator import OODCalculator
from src.utils.results import ResultsLogger

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Device Handling
    device_str = cfg.experiment.device
    if device_str == "auto":
        if torch.cuda.is_available():
            device_str = "cuda"
        elif torch.backends.mps.is_available():
            device_str = "mps"
        else:
            device_str = "cpu"

    log.info(f"Using device: {device_str}")

    # Initialize Results Logger
    output_dir = HydraConfig.get().runtime.output_dir
    results = ResultsLogger(output_dir)
    results.log_config(OmegaConf.to_container(cfg, resolve=True))

    # 2. Load Data
    log.info("Initializing Data Loader...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    data_loader = DataLoader(cfg.dataset, tokenizer)
    # UNPACK 4 loaders: train, val, test_id, test_ood
    train_loader, val_loader, test_id_loader, test_ood_loader = data_loader.load()

    # 3. Load Model
    log.info("Initializing Model...")
    model = ModelWrapper(cfg.model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.model.training.learning_rate,
        weight_decay=cfg.model.training.weight_decay
    )

    # 4. Train
    if not cfg.experiment.debug:
        log.info("Initializing Trainer...")
        trainer = Trainer(cfg, model, train_loader, val_loader, optimizer, results)
        trainer.train()

        # Log final validation accuracy from training history
        if results.history:
            final_accuracy = results.history[-1]["val_accuracy"]
            results.log_val_accuracy(final_accuracy)

    # 5. OOD Calculation
    log.info("Starting OOD Fitting...")
    ood_calc = OODCalculator(cfg)

    # Extract features for ID data (Train set) to fit statistics
    log.info("Extracting features from Training set for OOD fitting...")
    model.eval()

    train_features = []
    train_labels = []

    device = torch.device(device_str)
    model.to(device)

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting ID Features"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['intent'].to(device)

            _, _ = model(input_ids, attention_mask)

            # Fetch appropriate features based on OOD method
            if cfg.ood_method == 'energy':
                features = model.get_features('logits')
            else:
                features = model.get_features('pooled_output')

            train_features.append(features.cpu())
            train_labels.append(labels.cpu())

            if cfg.experiment.debug: break

    train_features = torch.cat(train_features)
    train_labels = torch.cat(train_labels)

    ood_calc.fit(train_features, train_labels)

    # 6. Evaluation (ID vs OOD)
    log.info("Evaluating OOD scores on ID and OOD Test sets...")

    # Helper to get scores
    def get_scores(loader):
        scores = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating OOD"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                _, _ = model(input_ids, attention_mask)

                if cfg.ood_method == 'energy':
                    features = model.get_features('logits')
                else:
                    features = model.get_features('pooled_output')

                dists = ood_calc.predict(features)
                scores.append(dists)
                if cfg.experiment.debug: break
        return torch.cat(scores)

    id_dists = get_scores(test_id_loader)
    ood_dists = get_scores(test_ood_loader)

    # Calculate Metrics
    auroc, fpr95 = ood_calc.evaluate(id_dists, ood_dists)

    # Log and save results
    results.log_ood_metrics(auroc, fpr95)
    results.save()

    log.info(f"OOD Results - AUROC: {auroc:.4f}, FPR@95: {fpr95:.4f}")
    log.info("Pipeline finished.")

if __name__ == "__main__":
    main()
