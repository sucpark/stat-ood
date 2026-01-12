import json
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

log = logging.getLogger(__name__)


class ResultsLogger:
    """Simple experiment results logger that saves to JSON and CSV."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = datetime.now()
        self.history = []
        self.metrics = {}
        self.config = {}

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.config = {
            "model": config.get("model", {}).get("name", "unknown"),
            "pooling": config.get("model", {}).get("pooling", "unknown"),
            "dataset": config.get("dataset", {}).get("name", "unknown"),
            "ood_method": config.get("ood_method", "unknown"),
            "epochs": config.get("model", {}).get("training", {}).get("epochs", 0),
            "learning_rate": config.get("model", {}).get("training", {}).get("learning_rate", 0),
            "batch_size": config.get("dataset", {}).get("loader", {}).get("batch_size", 0),
            "seed": config.get("experiment", {}).get("seed", 42),
        }

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
    ):
        """Log metrics for a single epoch."""
        self.history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 6),
        })

    def log_ood_metrics(self, auroc: float, fpr95: float):
        """Log OOD detection metrics."""
        self.metrics["auroc"] = round(float(auroc), 6)
        self.metrics["fpr95"] = round(float(fpr95), 6)

    def log_val_accuracy(self, accuracy: float):
        """Log final validation accuracy."""
        self.metrics["val_accuracy"] = round(float(accuracy), 6)

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def save(self):
        """Save results to JSON and training history to CSV."""
        end_time = datetime.now()
        runtime = (end_time - self.start_time).total_seconds()

        # Build results dict
        results = {
            "metadata": {
                "timestamp": self.start_time.isoformat(),
                "runtime_seconds": round(runtime, 2),
                "git_commit": self._get_git_commit(),
            },
            "config": self.config,
            "metrics": self.metrics,
        }

        # Save JSON
        json_path = self.output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Results saved to {json_path}")

        # Save training history CSV (if any)
        if self.history:
            csv_path = self.output_dir / "training_history.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
                writer.writeheader()
                writer.writerows(self.history)
            log.info(f"Training history saved to {csv_path}")

        return results
