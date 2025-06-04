
# SynthTabularDataGeneration-IR

**SynthTabularDataGeneration-IR** is a research-oriented repository for evaluating synthetic data generation techniques tailored to **imbalanced regression** tasks over tabular datasets. It includes various oversampling strategies, learner models, custom metrics, and statistical analysis tools to benchmark performance across multiple datasets.

---

## 🎯 Motivation

Handling underrepresented regions in regression target distributions is a well-known challenge. This repository provides implementations for:

- Classical resampling (e.g., Random Undersampling/Oversampling)
- SMOTE-based and relevance-aware strategies (e.g., SMOGN, WERCS, WSMOTER)
- Advanced techniques including:
  - **DAVID** (VAE-based)
  - **GenCART-IR** (CART-based synthetic generation with relevance and density adaptation) - originally proposed

It also supports custom evaluation metrics to better reflect performance on rare target regions.

---

## 📁 Repository Structure

```
SynthTabularDataGeneration-IR/
├── data/                     # Raw and processed datasets
├── results/                  # Output tables, plots, rankings
├── functions/                # Adapted functions 
├── runtime/                  # Runtime statistics
└── README.md
```

---

## ⚙️ Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/antoniopedropi/SynthTabularDataGeneration-IR.git
cd SynthTabularDataGeneration-IR
```

2. Run main script:

```bash
python automated_script_datasets_final.py
```


## 📊 Output Artifacts

All results are saved to the `results/` directory:

- Metric tables: RMSE, RW-RMSE, SERA
- Runtime benchmarks
- Wilcoxon test plots
- Summary tables of best-performing strategies

---

## 📏 Supported Metrics

Implemented custom metrics include:

- Relevance-weighted MSE / RMSE / MAE / R²
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
📧 up201704931 [at] up [dot] pt  
🔗 [https://github.com/antoniopedropi](https://github.com/antoniopedropi)

---
