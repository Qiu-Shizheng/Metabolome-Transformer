import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, ff_size=2048, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h2 = self.ln2(x)
        return x + self.ff(h2)


class MetabolomeTransformer(nn.Module):
    def __init__(
        self,
        num_metabolites: int,
        semantic_embeddings: torch.Tensor,
        hidden_size=512,
        num_layers=8,
        num_heads=8,
        ff_size=2048,
        dropout=0.1,
        num_induce_tokens=4,
        proj_dim=128,
    ):
        super().__init__()
        self.num_metabolites = num_metabolites
        self.hidden_size = hidden_size
        self.num_induce = num_induce_tokens

        self.register_buffer("semantic_embeddings", semantic_embeddings.clone())
        sem_dim = semantic_embeddings.shape[1]

        self.mask_embed = nn.Embedding(2, hidden_size)
        self.semantic_proj = nn.Linear(sem_dim, hidden_size, bias=False)
        self.value_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.film_gamma = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.film_beta = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.induce_tokens = (
            nn.Parameter(torch.randn(1, num_induce_tokens, hidden_size) * 0.02)
            if num_induce_tokens > 0
            else None
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ff_size=ff_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(hidden_size)
        self.mu_head = nn.Linear(hidden_size, 1)
        self.logvar_head = nn.Linear(hidden_size, 1)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, proj_dim),
        )

    def forward(self, x: torch.Tensor, mask_bool: torch.Tensor):
        b, n = x.shape
        if n != self.num_metabolites:
            raise ValueError(f"Expected {self.num_metabolites} metabolites, got {n}.")

        ids = torch.arange(n, device=x.device).unsqueeze(0).expand(b, -1)
        sem_vec = self.semantic_proj(self.semantic_embeddings[ids])

        val = self.value_mlp(x.unsqueeze(-1))
        gamma = self.film_gamma(sem_vec)
        beta = self.film_beta(sem_vec)
        mask_vec = self.mask_embed(mask_bool.long())
        token = gamma * val + beta + mask_vec + sem_vec

        cls = self.cls_token.expand(b, 1, self.hidden_size)
        if self.induce_tokens is not None:
            induce = self.induce_tokens.expand(b, self.num_induce, self.hidden_size)
            h = torch.cat([cls, induce, token], dim=1)
        else:
            h = torch.cat([cls, token], dim=1)

        for block in self.blocks:
            h = block(h)

        cls_preln = h[:, 0, :]
        h = self.final_ln(h)
        cls_postln = h[:, 0, :]
        offset = 1 + (self.num_induce if self.induce_tokens is not None else 0)
        h_tokens = h[:, offset:, :]

        mu = self.mu_head(h_tokens).squeeze(-1)
        logvar = torch.clamp(self.logvar_head(h_tokens).squeeze(-1), -5.0, 3.0)
        z = F.normalize(self.proj_head(cls_postln), dim=-1)
        return {"mu": mu, "logvar": logvar, "cls": cls_postln, "cls_preln": cls_preln, "z": z}


class BinaryDiseaseClassifier(nn.Module):
    def __init__(self, backbone: MetabolomeTransformer, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(backbone.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(backbone.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, mask_bool: torch.Tensor):
        features = self.backbone(x, mask_bool)["cls"]
        return self.head(features).squeeze(-1)


def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def safe_load_backbone_state_dict(model, state_dict):
    state_dict = strip_module_prefix(state_dict)
    incompat = model.load_state_dict(state_dict, strict=False)
    missing = set(incompat.missing_keys)
    unexpected = set(incompat.unexpected_keys)
    if missing:
        raise RuntimeError(f"Missing keys when loading model: {sorted(missing)}")
    allowed = {"mu_head.weight", "mu_head.bias", "logvar_head.weight", "logvar_head.bias"}
    not_allowed = unexpected - allowed
    if not_allowed:
        raise RuntimeError(f"Unexpected keys not allowed: {sorted(not_allowed)}")
