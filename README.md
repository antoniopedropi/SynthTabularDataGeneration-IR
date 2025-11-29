
# SynthTabularDataGeneration-IR

**SynthTabularDataGeneration-IR** is a research-oriented repository for evaluating synthetic data generation techniques tailored to **imbalanced regression** tasks over tabular datasets. It includes various oversampling strategies, learner models, custom metrics, and statistical analysis tools to benchmark performance across multiple datasets.

---

## Supplementary Materials

- **CodeOcean Capsule**: a capsule with the code also presented in this repository, fully reproducible, in an already set up Puthon environment, available with the DOI: [10.24433/CO.7826905.v2](https://doi.org/10.24433/CO.7826905.v2)
- **Zenodo Repository**: an "Imbalanced Regression Dataset Repository" comprising 62 datasets tailored for imbalanced regression tasks, available with the DOI: [10.5281/zenodo.17429036](https://doi.org/10.5281/zenodo.17429036)
- **GitHub Dataset Repository**: more details on the "Imbalanced Regression Dataset Repository" also published in Zenodo are available at: [Datasets-ImbalancedRegression](https://antoniopedropi.github.io/Datasets-ImbalancedRegression/)

---

## Motivation

Handling underrepresented regions in regression target distributions is a well-known challenge. This repository provides implementations for:

- Classical resampling (Random Undersampling/Oversampling, WERCS)
- Introduction of Noise (Gaussian Noise - GN)
- SMOTE-based strategies (SMOTER, SMOGN, WSMOTER, G-SMOTER)
- Deep Learning techniques (VAE-based: DAVID and TVAE, GAN-based: CTGAN and CopulaGAN, Diffusion-based: TabDiff, Other: KNNOR-REG)
- **CARTGen-IR** (CART-based synthetic generation with relevance and density adaptation) - originally proposed in our study

It also supports custom evaluation metrics to better reflect performance on rare target regions.

---

## Repository Structure

```
SynthTabularDataGeneration-IR/
├── datasets/                 # Processed datasets
├── results/                  # Output tables, plots, rankings, runtime statistics
├── functions/                # Adapted functions
└── README.md
```

---

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/antoniopedropi/SynthTabularDataGeneration-IR.git
cd SynthTabularDataGeneration-IR
```

2. Run main script:

```bash
python automated_script_article.py
```

---

## Supported Metrics

Implemented and used custom metrics include:

- Standard Metrics: RMSE
- Relevance-weighted Metrics: RW-RMSE
- SERA (Squared Error Residuals Area)
- DenseWeight-weighted Metrics: DW-RMSE and DW-SERA (originally proposed)

These help assess model performance especially on rare/extreme target values.

---

## Output Files

All results are saved to the `results/` directory:

- Runtime benchmarks
- Wilcoxon test plots
- Summary tables of best-performing strategies
- Bayesian test plots
- Toy Visualizations
- Hyperparameter Study

---

## Citation

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

## Contact

**António Pedro Pinheiro**  
📧 up201704931@up.pt  
🔗 [https://github.com/antoniopedropi](https://github.com/antoniopedropi)

---
