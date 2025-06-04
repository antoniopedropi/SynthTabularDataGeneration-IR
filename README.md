
# SynthTabularDataGeneration-IR

**SynthTabularDataGeneration-IR** is a research-oriented repository for evaluating synthetic data generation techniques tailored to **imbalanced regression** tasks over tabular datasets. It includes various oversampling strategies, learner models, custom metrics, and statistical analysis tools to benchmark performance across multiple datasets.

---

## 🎯 Motivation

Handling underrepresented regions in regression target distributions is a well-known challenge. This repository provides implementations for:

- Classical resampling (e.g., Random Undersampling/Oversampling)
- SMOTE-based and relevance-aware strategies (e.g., SMOGN, WERCS, WSMOTER)
- Advanced techniques including:
  - **DAVID** (VAE-based)
  - **CART-IR** (CART-based synthetic generation with relevance and density adaptation)

It also supports custom evaluation metrics to better reflect performance on rare target regions.

---

## 📁 Repository Structure

```
SynthTabularDataGeneration-IR/
├── data/                     # Raw and processed datasets
├── results/                  # Output tables, plots, rankings
├── scripts/                  # High-level execution scripts
├── learners/                 # Model-specific evaluation logic
├── oversampling/            # Implementations of resampling strategies
├── metrics/                 # Custom evaluation metrics (RW-RMSE, SERA, etc.)
├── utils/                   # Helper functions (e.g., plotting, relevance)
├── main.py                  # Unified experimental pipeline
├── environment.yml          # Conda environment file
└── README.md
```

---

## ⚙️ Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/antoniopedropi/SynthTabularDataGeneration-IR.git
cd SynthTabularDataGeneration-IR
```

2. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate synth-ir
```

---

## 🚀 How to Run

Run full evaluation with 2×5-fold CV, oversampling, and metrics:

```bash
python main.py --config configs/cart_ir.yaml
```

Or run individual scripts, e.g.,

```bash
python scripts/run_cart_ir.py
python scripts/run_vaey.py
```

Use `--help` on each script for options.

---

## 📊 Output Artifacts

All results are saved to the `results/` directory:

- Metric tables: RMSE, RW-RMSE, SERA
- Runtime benchmarks
- CD diagrams and Wilcoxon test plots
- Summary tables of best-performing strategies

---

## 📏 Supported Metrics

Implemented custom metrics include:

- Relevance-weighted MSE / MAE / R²
- Root RW-RMSE
- SERA (Synthetic Error Reduction Area)

These help assess model performance especially on rare/extreme target values.

---

## 📚 Citation

If you use this codebase, please cite our paper:

```bibtex
@misc{pinheiro2025cartbasedsynthetictabulardata,
  title     = {CART-based Synthetic Tabular Data Generation for Imbalanced Regression},
  author    = {António Pedro Pinheiro and Rita P. Ribeiro},
  year      = {2025},
  eprint    = {2506.02811},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2506.02811}
}
```

---

## 📬 Contact

**António Pedro Pinheiro**  
📧 antoniopedropi [at] gmail [dot] com  
🔗 [https://github.com/antoniopedropi](https://github.com/antoniopedropi)

---
