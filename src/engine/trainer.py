import logging
import torch
import torch.nn as nn
from tqdm import tqdm

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, optimizer):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = torch.device(cfg.experiment.device)
        self.model.to(self.device)

    def train(self):
        log.info("Starting training...")

        for epoch in range(self.cfg.model.training.epochs):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.model.training.epochs}")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['intent'].to(self.device)

                self.optimizer.zero_grad()
                loss, logits = self.model(input_ids, attention_mask, labels=labels)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = train_loss / len(self.train_loader)
            log.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

            # Validation
            self.validate(epoch)

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['intent'].to(self.device)

                loss, logits = self.model(input_ids, attention_mask, labels=labels)
                val_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(self.val_loader)
        accuracy = correct / total

        log.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")
