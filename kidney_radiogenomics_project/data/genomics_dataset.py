import numpy as np
import torch
from torch.utils.data import Dataset

class GenomicsDataset(Dataset):
    """
    Placeholder dataset for genomics (e.g., RNA-seq, mutations).
    Expects a numpy array or pandas DataFrame per patient ID aligned with imaging dataset.
    """
    def __init__(self, X, gene_names=None):
        self.X = np.asarray(X, dtype=np.float32)
        self.gene_names = gene_names

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])
