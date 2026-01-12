import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import torch
import numpy as np
from transformers import AutoTokenizer

from src.data.loader import DataLoader
from src.models.wrapper import ModelWrapper
from src.engine.trainer import Trainer
from src.ood.calculator import OODCalculator

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # 1. Init WandB
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.experiment.debug else "online"
    )
    
    log.info(f"Using device: {cfg.experiment.device}")
    
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
        trainer = Trainer(cfg, model, train_loader, val_loader, optimizer)
        trainer.train()
    
    # 5. OOD Calculation
    log.info("Starting OOD Fitting...")
    ood_calc = OODCalculator(cfg)
    
    # Extract features for ID data (Train set) to fit statistics
    log.info("Extracting features from Training set for OOD fitting...")
    model.eval()
    
    train_features = []
    train_labels = []
    
    device = torch.device(cfg.experiment.device)
    model.to(device)
    
    with torch.no_grad():
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['intent'].to(device)
            
            _, _ = model(input_ids, attention_mask)
            features = model.get_features()
            
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
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                _, _ = model(input_ids, attention_mask)
                features = model.get_features()
                
                dists = ood_calc.predict(features)
                scores.append(dists)
                if cfg.experiment.debug: break
        return torch.cat(scores)

    id_dists = get_scores(test_id_loader)
    ood_dists = get_scores(test_ood_loader)
    
    # Calculate Metrics
    auroc, fpr95 = ood_calc.evaluate(id_dists, ood_dists)
    
    log.info(f"OOD Results - AUROC: {auroc:.4f}, FPR@95: {fpr95:.4f}")
    
    # 7. Visualization
    log.info("Creating Visualizations...")
    
    # Histogram
    data = [[s, "ID"] for s in id_dists.numpy()] + [[s, "OOD"] for s in ood_dists.numpy()]
    table = wandb.Table(data=data, columns=["Mahalanobis Distance", "Type"])
    
    wandb.log({
        "ood/auroc": auroc,
        "ood/fpr95": fpr95,
        "ood/dist_histogram": wandb.plot.histogram(table, "Mahalanobis Distance", title="ID vs OOD Distance Distribution")
    })
    
    log.info("Pipeline finished.")
    wandb.finish()

if __name__ == "__main__":
    main()
