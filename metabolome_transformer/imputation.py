import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import clamp_nonnegative


class ArrayMaskDataset(Dataset):
    def __init__(self, x, mask):
        self.x = x.astype(np.float32)
        self.mask = mask.astype(bool)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.mask[idx], dtype=torch.bool),
        )


@torch.no_grad()
def impute_missing_values(pipeline, df: pd.DataFrame, id_column: str | None = None, batch_size: int = 256):
    id_column = id_column or df.columns[0]
    metadata = df[[id_column]].copy()
    raw = df[pipeline.metabolite_columns].copy()

    missing_mask = raw.isna()
    z = pipeline.zscore(raw)
    z_values = z.values
    mask_values = missing_mask.values
    z_filled = np.where(mask_values, 0.0, z_values)

    dataset = ArrayMaskDataset(z_filled, mask_values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    pipeline.model.eval()
    for xb, mb in loader:
        xb = xb.to(pipeline.device)
        mb = mb.to(pipeline.device)
        out = pipeline.model(xb, mb)
        preds.append(out["mu"].detach().cpu().numpy())

    pred_z = np.concatenate(preds, axis=0)
    pred_z_df = pd.DataFrame(pred_z, columns=pipeline.metabolite_columns, index=raw.index)
    pred_raw_df = pipeline.inverse_zscore(pred_z_df)
    pred_raw_df = pred_raw_df.apply(pd.to_numeric, errors="coerce")
    pred_raw_df[:] = clamp_nonnegative(pred_raw_df.values)

    imputed = raw.copy()
    for col in pipeline.metabolite_columns:
        miss = missing_mask[col]
        imputed.loc[miss, col] = pred_raw_df.loc[miss, col]

    return pd.concat([metadata, imputed], axis=1)
