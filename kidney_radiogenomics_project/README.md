# Kidney Radiogenomics Project

This repository provides a research scaffold for:
- Self-supervised pretraining of a 3D imaging encoder on CT/MRI
- Genomics encoder (autoencoder or GNN)
- Cross-modal transformer fusion with clinical metadata
- Classification (benign/malignant, RCC subtypes) and survival prediction (Cox)

## Quick Start
1. Create and activate a Python 3.10+ environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Adjust `config/config.yaml` for data paths and hyperparameters.
4. Run SSL pretraining: `python training/ssl_pretrain.py`
5. Train fusion model: `python training/train_fusion.py`

> Note: Datasets are placeholders—connect TCIA (imaging) and TCGA (genomics) to the dataloaders.
