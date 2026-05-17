import numpy as np
import pandas as pd
import torch

from .constants import (
    DEFAULT_CKPT_PATH,
    DEFAULT_HPARAMS_PATH,
    DEFAULT_METABOLITE_COLUMNS_PATH,
    DEFAULT_METABOLITE_STATS_PATH,
    DEFAULT_SEMANTIC_EMBEDDING_PATH,
)
from .model import MetabolomeTransformer, safe_load_backbone_state_dict
from .utils import choose_device, load_json, read_lines


class MetabolomeTransformerPipeline:
    def __init__(
        self,
        ckpt_path=DEFAULT_CKPT_PATH,
        hparams_path=DEFAULT_HPARAMS_PATH,
        semantic_embedding_path=DEFAULT_SEMANTIC_EMBEDDING_PATH,
        metabolite_columns_path=DEFAULT_METABOLITE_COLUMNS_PATH,
        stats_path=DEFAULT_METABOLITE_STATS_PATH,
        device="auto",
    ):
        self.device = choose_device(device)
        self.ckpt_path = str(ckpt_path)
        self.hparams_path = str(hparams_path)
        self.semantic_embedding_path = str(semantic_embedding_path)
        self.metabolite_columns_path = str(metabolite_columns_path)
        self.stats_path = str(stats_path) if stats_path is not None else None

        self.hparams = load_json(self.hparams_path)
        self.metabolite_columns = read_lines(self.metabolite_columns_path)
        self.metabolite_mean = pd.Series(0.0, index=self.metabolite_columns, dtype=float)
        self.metabolite_std = pd.Series(1.0, index=self.metabolite_columns, dtype=float)
        if self.stats_path:
            self.load_stats(self.stats_path)

        semantic = np.load(self.semantic_embedding_path)
        self.semantic_embeddings = torch.tensor(semantic, dtype=torch.float32)
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()

    def load_stats(self, stats_path):
        stats_df = pd.read_csv(stats_path)
        required = {"metabolite", "mean", "std"}
        if not required.issubset(stats_df.columns):
            raise ValueError("Stats file must contain columns: metabolite, mean, std")
        stats_df = stats_df.set_index("metabolite")
        missing = [c for c in self.metabolite_columns if c not in stats_df.index]
        if missing:
            raise ValueError(f"Stats file is missing {len(missing)} metabolites.")
        self.metabolite_mean = stats_df.loc[self.metabolite_columns, "mean"].astype(float)
        self.metabolite_std = stats_df.loc[self.metabolite_columns, "std"].astype(float).replace(0, 1.0)

    def _build_model(self):
        model = MetabolomeTransformer(
            num_metabolites=len(self.metabolite_columns),
            semantic_embeddings=self.semantic_embeddings,
            hidden_size=self.hparams["hidden_size"],
            num_layers=self.hparams["num_layers"],
            num_heads=self.hparams["num_heads"],
            ff_size=self.hparams["ff_size"],
            dropout=self.hparams["dropout"],
            num_induce_tokens=self.hparams["num_induce"],
            proj_dim=self.hparams["proj_dim"],
        )
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        safe_load_backbone_state_dict(model, state_dict)
        return model

    def align_input_dataframe(self, df: pd.DataFrame, id_column: str | None = None) -> pd.DataFrame:
        df = df.copy()
        id_column = id_column or df.columns[0]
        if id_column not in df.columns:
            raise ValueError(f"Input file must contain ID column '{id_column}'.")
        missing_cols = [c for c in self.metabolite_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Input file is missing {len(missing_cols)} metabolite columns. "
                f"First missing columns: {missing_cols[:10]}"
            )
        out = df[[id_column] + self.metabolite_columns].copy()
        for col in self.metabolite_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def zscore(self, df_values: pd.DataFrame) -> pd.DataFrame:
        return (df_values[self.metabolite_columns] - self.metabolite_mean) / self.metabolite_std

    def inverse_zscore(self, z_df: pd.DataFrame) -> pd.DataFrame:
        return z_df[self.metabolite_columns] * self.metabolite_std + self.metabolite_mean
