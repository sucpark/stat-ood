<div align="center">

# 📊 Stat-OOD
### Statistical Out-of-Distribution Detector

**A Native PyTorch Implementation of Mahalanobis Distance-based OOD Detection<br>using BERT & WandB**

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Native-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd?style=for-the-badge&logo=hydra&logoColor=white)](https://hydra.cc)
[![WandB](https://img.shields.io/badge/WandB-Tracking-yellow?style=for-the-badge&logo=weightsandbiases&logoColor=black)](https://wandb.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

<br>

## 📖 Overview

**Stat-OOD** is a highly transparent and experimental framework for **Out-of-Distribution (OOD)** detection in Natural Language Understanding (NLU). It combines **Pre-trained Language Models (BERT)** with **Gaussian Discriminant Analysis (Mahalanobis Distance)** to robustly identify unknown user intents.

Designed for researchers and engineers, this project emphasizes **traceability, flexibility, and statistical rigor** by avoiding high-level abstractions in favor of a native PyTorch training loop and direct feature extraction hooks.

---

## 🌟 Key Features

| Feature | Description |
| :--- | :--- |
| **🧠 Statistical OOD Engine** | Implements **Mahalanobis Distance** scoring based on class-conditional Gaussian distributions. |
| **⚙️ Native PyTorch** | Custom `Trainer` loop implementing `loss.backward()` directly for maximum control and debuggability. |
| **📉 Full Observability** | Integrated with **WandB** for real-time loss tracking, OOD metric visualization (AUROC, Histograms), and Sweeps. |
| **⚡ Modern Toolchain** | Built with `uv` for lightning-fast dependency management and `Hydra` for hierarchical configuration. |
| **🤗 HuggingFace Ready** | Seamlessly loads models (BERT, RoBERTa) and datasets (CLINC150) from the HF Hub. |

---

## 🚀 Getting Started

### Prerequisites

*   **Python 3.12+**
*   **uv** (Modern Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/sucpark/stat-ood.git
cd stat-ood

# 2. Initialize Environment
uv sync
```

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

# Run Agent (replace SWEEP_ID with the generated ID)
wandb agent <SWEEP_ID>
```

---

## 📊 Methodology

**Stat-OOD** operates in two phases:

### Phase 1: Representation Learning (ID Training)
Finetune a PLM (e.g., `bert-base-uncased`) on **In-Distribution (ID)** intent classification data using standard Cross-Entropy Loss.

### Phase 2: Statistical Scoring (Post-Hoc)
1.  **Feature Extraction**: Extract hidden states $h(x)$ from the last layer (or pooler).
2.  **Gaussian Fitting**: Estimate a class-conditional Gaussian distribution $\mathcal{N}(\mu_c, \Sigma)$ for each class $c$.
    *   $\mu_c$: Mean vector of class $c$.
    *   $\Sigma$: Shared covariance matrix across all classes.
3.  **Scoring**: Compute the **Mahalanobis Distance** to the nearest class centroid for a test input $x$:

<div align="center">

$$ M(x) = \min_c \sqrt{(h(x) - \mu_c)^T \Sigma^{-1} (h(x) - \mu_c)} $$

</div>

*   **Decision**: High distance $\rightarrow$ OOD. Low distance $\rightarrow$ ID.

---

## 📁 Project Structure

```bash
stat-ood/
├── configs/            # ⚙️ Hydra Configurations
│   ├── config.yaml     # Main config
│   ├── model/          # Model params (BERT, etc)
│   ├── dataset/        # Dataset params (CLINC150)
│   └── sweep.yaml      # WandB Sweep config
├── src/
│   ├── data/           # 📦 Data Loader & Preprocessing
│   ├── models/         # 🤖 Model Wrapper & Hooks
│   ├── engine/         # 🚂 Manual Training Loop
│   └── ood/            # 📐 Mahalanobis Calculator
├── main.py             # 🚀 Entry Point
└── README.md           # 📄 Documentation
```

---

## 📈 Performance

*Dataset: CLINC150 based evaluation*

| Metric | Score (Debug) | Score (Tuned) |
| :--- | :--- | :--- |
| **AUROC** | 0.9999 | *TBD* |
| **FPR @ TPR95** | 0.0001 | *TBD* |

---

## 🛠 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![WandB](https://img.shields.io/badge/WandB-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

</div>

---

## License

This project is licensed under the MIT License.
