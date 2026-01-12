import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import torch
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
    train_loader, val_loader, test_loader = data_loader.load()
    
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
    log.info("Starting OOD Calculation...")
    ood_calc = OODCalculator(cfg)
    
    # Extract features for ID data (Train set) to fit statistics
    # Use subset for debug/speed if needed, but typically use full train set
    log.info("Extracting features from Training set for OOD fitting...")
    model.eval()
    
    train_features = []
    train_labels = []
    
    device = torch.device(cfg.experiment.device)
    model.to(device) # Ensure model is on device
    
    # To avoid OOM and save time, we might want to disable gradients
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
    
    # Verify on Test set (In-distribution)
    log.info("Evaluating OOD scores on Test set (ID)...")
    test_dists = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            _, _ = model(input_ids, attention_mask)
            features = model.get_features()
            
            dists = ood_calc.predict(features)
            test_dists.append(dists)
            
            if cfg.experiment.debug: break

    test_dists = torch.cat(test_dists)
    avg_dist = test_dists.mean().item()
    log.info(f"Average Mahalanobis Distance (ID Test): {avg_dist:.4f}")
    wandb.log({"ood/avg_mahalanobis_id": avg_dist})
    
    log.info("Pipeline finished.")
    wandb.finish()

if __name__ == "__main__":
    main()
