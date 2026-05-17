import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, random_split

from .model import BinaryDiseaseClassifier


class BinaryTableDataset(Dataset):
    def __init__(self, values, labels=None):
        self.values = values.astype(np.float32)
        self.mask = np.isnan(self.values)
        self.values = np.where(self.mask, 0.0, self.values)
        self.labels = None if labels is None else labels.astype(np.float32)

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.values[idx], dtype=torch.float32)
        mask = torch.tensor(self.mask[idx], dtype=torch.bool)
        if self.labels is None:
            return x, mask
        return x, mask, torch.tensor(self.labels[idx], dtype=torch.float32)


def _make_dataset(pipeline, df, label_column=None):
    z = pipeline.zscore(df[pipeline.metabolite_columns])
    labels = None if label_column is None else pd.to_numeric(df[label_column], errors="coerce").values
    if labels is not None and np.isnan(labels).any():
        raise ValueError("Labels must be numeric and non-missing.")
    return BinaryTableDataset(z.values, labels)


def _compute_metrics(y_true, prob):
    out = {}
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["average_precision"] = float(average_precision_score(y_true, prob))
    out["n"] = int(len(y_true))
    return out


def train_binary_classifier(
    pipeline,
    df: pd.DataFrame,
    label_column: str,
    output_dir,
    task: str = "current",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    freeze_backbone: bool = False,
    seed: int = 42,
):
    if task not in {"current", "future"}:
        raise ValueError("task must be 'current' or 'future'.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _make_dataset(pipeline, df, label_column=label_column)
    generator = torch.Generator().manual_seed(seed)
    val_size = int(round(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Training set is empty after validation split.")
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    model = BinaryDiseaseClassifier(pipeline.model)
    model.to(pipeline.device)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    pos = float(np.sum(dataset.labels == 1))
    neg = float(np.sum(dataset.labels == 0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=pipeline.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_size > 0 else None
    metric_rows = []
    best_metric = -np.inf
    best_path = output_dir / "classifier.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, mb, yb in train_loader:
            xb = xb.to(pipeline.device)
            mb = mb.to(pipeline.device)
            yb = yb.to(pipeline.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, mb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_loader is not None:
            y_true, prob = predict_binary_probabilities(model, val_loader, pipeline.device)
            metrics = _compute_metrics(y_true, prob)
            row.update({f"val_{k}": v for k, v in metrics.items()})
            score = metrics.get("roc_auc", -row["train_loss"])
        else:
            score = -row["train_loss"]

        metric_rows.append(row)
        if score > best_metric:
            best_metric = score
            save_binary_classifier(
                model,
                best_path,
                pipeline,
                task=task,
                label_column=label_column,
                freeze_backbone=freeze_backbone,
            )

    pd.DataFrame(metric_rows).to_csv(output_dir / "training_metrics.csv", index=False)
    return best_path


@torch.no_grad()
def predict_binary_probabilities(model, loader, device):
    model.eval()
    labels = []
    probs = []
    for batch in loader:
        if len(batch) == 3:
            xb, mb, yb = batch
            labels.append(yb.numpy())
        else:
            xb, mb = batch
        xb = xb.to(device)
        mb = mb.to(device)
        prob = torch.sigmoid(model(xb, mb)).detach().cpu().numpy()
        probs.append(prob)
    y_true = np.concatenate(labels) if labels else np.array([])
    prob = np.concatenate(probs)
    return y_true, prob


def save_binary_classifier(model, path, pipeline, **metadata):
    payload = {
        "model_state_dict": model.state_dict(),
        "hparams": pipeline.hparams,
        "metabolite_columns": pipeline.metabolite_columns,
        "metadata": metadata,
    }
    torch.save(payload, path)
    with open(Path(path).with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_binary_classifier(pipeline, checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = BinaryDiseaseClassifier(pipeline.model)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(pipeline.device)
    model.eval()
    return model, payload.get("metadata", {})


def predict_binary_classifier(pipeline, df: pd.DataFrame, checkpoint_path, id_column: str | None = None, batch_size: int = 256):
    id_column = id_column or df.columns[0]
    model, _ = load_binary_classifier(pipeline, checkpoint_path)
    dataset = _make_dataset(pipeline, df, label_column=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    _, prob = predict_binary_probabilities(model, loader, pipeline.device)
    out = pd.DataFrame({
        id_column: df[id_column].values,
        "probability": prob,
        "prediction": (prob >= 0.5).astype(int),
    })
    return out
