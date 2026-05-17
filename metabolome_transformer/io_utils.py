from pathlib import Path

import pandas as pd


def read_table(path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table extension: {suffix}")


def write_table(df: pd.DataFrame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".tsv", ".txt"}:
        df.to_csv(path, sep="\t", index=False)
    else:
        raise ValueError(f"Unsupported table extension: {suffix}")
