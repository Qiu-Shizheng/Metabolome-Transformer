from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent

DEFAULT_PRETRAIN_DIR = REPO_ROOT / "assets" / "pretrained"
DEFAULT_CKPT_PATH = DEFAULT_PRETRAIN_DIR / "best_general_transformer_pretrain.pt"
DEFAULT_HPARAMS_PATH = DEFAULT_PRETRAIN_DIR / "best_hparams.json"
DEFAULT_SEMANTIC_EMBEDDING_PATH = DEFAULT_PRETRAIN_DIR / "metabolite_semantic_embeddings.npy"
DEFAULT_METABOLITE_COLUMNS_PATH = DEFAULT_PRETRAIN_DIR / "metabolite_columns.txt"
DEFAULT_METABOLITE_STATS_PATH = DEFAULT_PRETRAIN_DIR / "metabolite_stats.csv"
