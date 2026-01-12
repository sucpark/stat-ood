import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import torch
import os

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # 1. Init WandB
    # Ensure wandb doesn't error out if not logged in by handling it gracefully or assuming user has setup
    # If key is missing, 'wandb.init' might handle it interactively or error.
    # We will assume environment is set or user will login.
    
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled" if cfg.experiment.debug else "online"
    )
    
    log.info(f"Using device: {cfg.experiment.device}")
    
    # Placeholder for training pipeline
    log.info("Starting pipeline...")
    
    # 2. Load Data (TODO)
    
    # 3. Load Model (TODO)
    
    # 4. Train (TODO)
    
    # 5. OOD Calculation (TODO)
    
    log.info("Pipeline finished.")
    wandb.finish()

if __name__ == "__main__":
    main()
