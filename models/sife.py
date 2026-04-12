import torch
import torch.nn as nn

from .grl import GRL


# ---------------------------------------------------------
# Self-attention block (Transformer-like, lightweight)
# ---------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        mlp_out = self.mlp(h)
        x = x + self.dropout(mlp_out)

        return x


# ---------------------------------------------------------
# Fi(.) module (EEG-irrelevant / subject-invariant extractor)
# ---------------------------------------------------------
class FiModule(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))

        self.layers = nn.ModuleList([
            SelfAttentionBlock(dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"FiModule expected seq_len={self.seq_len}, got T={x.size(1)}"
            )

        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        e_i_seq = self.proj(x)
        return e_i_seq


# ---------------------------------------------------------
# Token-level MLP head for subject prediction
# ---------------------------------------------------------
class SubjectHead(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        num_classes: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),                  # (B, T, D) -> (B, T*D)
            nn.Linear(seq_len * dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B, T, D), got {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"SubjectHead expected seq_len={self.seq_len}, got T={x.size(1)}"
            )
        return self.net(x)


# ---------------------------------------------------------
# Main SIFE module
# ---------------------------------------------------------
class SIFE(nn.Module):
    """
    Subject-Invariant Feature Extraction module.

    Notes:
    - `num_subjects` here should be the number of ACTIVE training subjects,
      i.e. len(train_subjects), not the total number of subjects in the dataset.
    - The caller is expected to remap subject ids to local class ids in [0, K-1].
    """

    def __init__(
        self,
        dim: int = 128,
        seq_len: int = 77,
        num_subjects: int = 3,
        fi_layers: int = 2,
        num_heads: int = 4,
        grl_lambda: float = 1.0,
        dropout: float = 0.1,
        classifier_hidden_dim: int = None,
    ):
        super().__init__()

        if num_subjects <= 1:
            raise ValueError(f"SIFE requires at least 2 active subject classes, got {num_subjects}.")

        self.dim = dim
        self.seq_len = seq_len
        self.num_subjects = num_subjects

        self.fi = FiModule(
            dim=dim,
            seq_len=seq_len,
            num_layers=fi_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.grl = GRL(lambda_=grl_lambda)

        self.subject_discriminator = SubjectHead(
            dim=dim,
            seq_len=seq_len,
            num_classes=num_subjects,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

        self.subject_classifier = SubjectHead(
            dim=dim,
            seq_len=seq_len,
            num_classes=num_subjects,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
        )

    def forward(self, E_seq: torch.Tensor):
        """
        Args:
            E_seq: (B, T, D)

        Returns:
            dict with:
              - E_i_seq: (B, T, D)
              - E_s_seq: (B, T, D)
              - E_i:     (B, D)   [solo per monitoring]
              - E_s:     (B, D)   [solo per monitoring]
              - pred_subject_i: (B, K) from token-level E_i_seq
              - pred_subject_s: (B, K) from token-level E_s_seq
        """
        if E_seq.ndim != 3:
            raise ValueError(f"Expected E_seq with shape (B, T, D), got {tuple(E_seq.shape)}")

        if E_seq.size(1) != self.seq_len:
            raise ValueError(
                f"SIFE expected seq_len={self.seq_len}, got T={E_seq.size(1)}"
            )

        # Step 1: compute E_i
        E_i_seq = self.fi(E_seq)  # (B, T, D)

        # Step 2: compute E_s
        E_s_seq = E_seq - E_i_seq

        # solo per monitoring/debug
        E_i = E_i_seq.mean(dim=1)  # (B, D)
        E_s = E_s_seq.mean(dim=1)  # (B, D)

        # Step 4: adversarial + classification on token-level sequences
        E_i_seq_grl = self.grl(E_i_seq)
        pred_subject_i = self.subject_discriminator(E_i_seq_grl)  # (B, K)
        pred_subject_s = self.subject_classifier(E_s_seq)         # (B, K)

        return {
            "E_i_seq": E_i_seq,
            "E_s_seq": E_s_seq,
            "E_i": E_i,
            "E_s": E_s,
            "pred_subject_i": pred_subject_i,
            "pred_subject_s": pred_subject_s,
        }