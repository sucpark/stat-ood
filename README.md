# Stat-OOD: Statistical Out-of-Distribution Detector 📊

> **A Native PyTorch Implementation of Mahalanobis Distance-based OOD Detection using BERT & WandB**

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Native-EE4C2C?logo=pytorch)
![WandB](https://img.shields.io/badge/WandB-Tracking-yellow?logo=weightsandbiases)
![License](https://img.shields.io/badge/License-MIT-green)

**Stat-OOD** is a highly transparent and experimental framework for Out-of-Distribution (OOD) detection in Natural Language Understanding (NLU). It combines **Pre-trained Language Models (BERT)** with **Gaussian Discriminant Analysis (Mahalanobis Distance)** to robustly identify unknown user intents.

Designed for researchers and engineers, this project emphasizes **traceability, flexibility, and statistical rigor** by avoiding high-level abstractions in favor of a native PyTorch training loop and direct feature extraction hooks.

---

## 🌟 Key Features

*   **Statistical OOD Detection**: Implements **Mahalanobis Distance** scores based on class-conditional Gaussian distributions.
*   **Native PyTorch Engine**: Custom `Trainer` loop implementing `loss.backward()` directly for maximum control and debuggability.
*   **Full Observability**: Integrated with **Weights & Biases (WandB)** for real-time loss tracking, OOD metric visualization (AUROC, Histograms), and hyperparameter sweeps.
*   **Modern Toolchain**: Built with `uv` for lightning-fast dependency management and `Hydra` for hierarchical configuration.
*   **HuggingFace Integration**: Seamlessly loads models (BERT, RoBERTa) and datasets (CLINC150) from the HF Hub.

---

## 🚀 Getting Started

### Prerequisites

*   **Python 3.12+**
*   **uv** (Modern Python package manager)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/stat-ood.git
    cd stat-ood
    ```

2.  **Initialize Environment**
    ```bash
    uv sync
    ```
    This will automatically create a virtual environment and install all dependencies (PyTorch, Transformers, WandB, etc.).

---

## 🏃 Usage

### 1. Basic Training & Evaluation
Run the full pipeline: Finetuning -> ID Feature Extraction -> OOD Scoring -> Evaluation.

```bash
uv run python main.py
```

### 2. Debug Mode
Run a fast dry-run (1 epoch, minimal data) to verify the pipeline.

```bash
uv run python main.py experiment.debug=true
```

### 3. Hyperparameter Tuning (WandB Sweep)
Optimize Learning Rate, Batch Size, etc. using Bayesian Optimization.

```bash
# Initialize Sweep
wandb sweep configs/sweep.yaml

# Run Agent (replace SWEEP_ID with the ID generated above)
wandb agent <SWEEP_ID>
```

---

## 📊 Methodology

**Stat-OOD** operates in two phases:

1.  **Representation Learning (ID Training)**
    *   Finetune a PLM (e.g., `bert-base-uncased`) on In-Distribution (ID) intent classification data.
    *   We use a standard Cross-Entropy Loss optimization.

2.  **Statistical Scoring (Post-Hoc)**
    *   Extract hidden states $h(x)$ from the last layer (or pooler) for all ID training samples.
    *   Estimate a class-conditional Gaussian distribution $\mathcal{N}(\mu_c, \Sigma)$ for each class $c$.
        *   $\mu_c$: Mean vector of class $c$.
        *   $\Sigma$: Shared covariance matrix across all classes (tied covariance).
    *   For a test input $x$, compute the **Mahalanobis Distance** to the nearest class centroid:
        $$ M(x) = \min_c \sqrt{(h(x) - \mu_c)^T \Sigma^{-1} (h(x) - \mu_c)} $$
    *   **Decision**: High distance $\rightarrow$ OOD. Low distance $\rightarrow$ ID.

---

## 📁 Project Structure

```bash
stat-ood/
├── configs/               # Hydra Configurations
│   ├── config.yaml        # Main config
│   ├── model/             # Model params (BERT, etc)
│   ├── dataset/           # Dataset params (CLINC150)
│   └── sweep.yaml         # WandB Sweep config
├── src/
│   ├── data/              # Data Loader & Preprocessing
│   ├── models/            # Model Wrapper & Hooks
│   ├── engine/            # Manual Training Loop
│   └── ood/               # Mahalanobis Calculator
├── main.py                # Entry Point
├── pyproject.toml         # Dependencies (uv)
└── README.md              # Documentation
```

---

## 📈 Results

*Dataset: CLINC150 (Plus)*

| Metric | Score (Debug) | Score (Tuned) |
| :--- | :--- | :--- |
| **AUROC** | 0.9999 | TBD |
| **FPR @ TPR95** | 0.0001 | TBD |

*(Note: Tuned results pending full sweep execution)*

---

## 🛠 Tech Stack

*   **Environment**: `uv`
*   **Framework**: `PyTorch`
*   **Models**: `HuggingFace Transformers`
*   **Config**: `Hydra`
*   **Tracking**: `Weights & Biases`

---

## License

This project is licensed under the MIT License.
