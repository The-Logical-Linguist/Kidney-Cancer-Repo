import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

class KidneyImagingDataset(Dataset):
    """
    Placeholder dataset for 3D CT/MRI volumes.
    Expects NIfTI files (.nii or .nii.gz) and a CSV mapping patient_id to file path and labels.
    """
    def __init__(self, df, patch_size=(96,96,96), augment=None):
        self.df = df.reset_index(drop=True)
        self.patch = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _load_volume(self, path):
        vol = nib.load(path).get_fdata().astype(np.float32)
        vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-6)
        return vol

    def _random_crop(self, vol):
        D,H,W = vol.shape
        d,h,w = self.patch
        sd = np.random.randint(0, max(1, D-d))
        sh = np.random.randint(0, max(1, H-h))
        sw = np.random.randint(0, max(1, W-w))
        return vol[sd:sd+d, sh:sh+h, sw:sw+w]

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        vol = self._load_volume(row['image_path'])
        if min(vol.shape) >= min(self.patch):
            vol = self._random_crop(vol)
        vol = np.expand_dims(vol, 0)  # (1, D, H, W)

        y_bin = int(row.get('label_binary', 0))
        y_sub = int(row.get('label_subtype', 0))
        time = float(row.get('survival_time', 0.0))
        event = int(row.get('survival_event', 0))

        x = torch.from_numpy(vol)
        return x, y_bin, y_sub, torch.tensor([time], dtype=torch.float32), torch.tensor([event], dtype=torch.float32)
