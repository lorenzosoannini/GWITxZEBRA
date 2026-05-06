import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .grl import GRL


# ---------------------------------------------------------
# ZEBRA-like transformer block
# ---------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    """
    Lightweight pre-norm Transformer block.

    Input / output:
        x: (B, T, D)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SelfAttentionBlock expects (B, T, D), got {tuple(x.shape)}")

        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


# ---------------------------------------------------------
# ZEBRA-like invariant extractor Fi(.)
# ---------------------------------------------------------
class FiModule(nn.Module):
    """
    Subject-invariant extractor.

    More faithful to ZEBRA than the earlier lightweight version:
      - stack of residual transformer blocks
      - final LayerNorm
      - no extra projection head at the end

    Input:
        x: (B, T, D)

    Output:
        E_i_seq: (B, T, D)
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        num_layers: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_pos_embed: bool = False,
    ):
        super().__init__()

        self.dim = int(dim)
        self.seq_len = int(seq_len)
        self.use_pos_embed = bool(use_pos_embed)
        self.use_gradient_checkpointing = False

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.dim))
        else:
            self.register_parameter("pos_embed", None)

        self.layers = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=self.dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(self.dim)

    def set_gradient_checkpointing(self, enable: bool = True):
        self.use_gradient_checkpointing = enable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"FiModule expects (B, T, D), got {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"FiModule expected seq_len={self.seq_len}, got T={x.size(1)}"
            )
        if x.size(2) != self.dim:
            raise ValueError(
                f"FiModule expected dim={self.dim}, got D={x.size(2)}"
            )

        if self.pos_embed is not None:
            x = x + self.pos_embed

        for layer in self.layers:
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.norm(x)
        return x


# ---------------------------------------------------------
# Token-level subject heads
# ---------------------------------------------------------
class SubjectHead(nn.Module):
    """
    Token-level subject classifier/discriminator.

    Input:
        x: (B, T, D)

    Output:
        logits: (B, K)
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        num_classes: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = int(hidden_dim or dim)

        self.dim = int(dim)
        self.seq_len = int(seq_len)
        self.num_classes = int(num_classes)

        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),  # (B, T, D) -> (B, T*D)
            nn.Linear(self.seq_len * self.dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SubjectHead expects (B, T, D), got {tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"SubjectHead expected seq_len={self.seq_len}, got T={x.size(1)}"
            )
        if x.size(2) != self.dim:
            raise ValueError(
                f"SubjectHead expected dim={self.dim}, got D={x.size(2)}"
            )

        return self.net(x)


# ---------------------------------------------------------
# Main SIFE
# ---------------------------------------------------------
class SIFE(nn.Module):
    """
    Subject-Invariant Feature Extraction module.

    ZEBRA-aligned logic:
      - E_i_seq = Fi(E_seq)
      - E_s_seq = E_seq - E_i_seq
      - adversarial subject prediction on E_i_seq
      - direct subject classification on E_s_seq

    Notes:
      - `num_subjects` should be the number of ACTIVE training subjects
      - caller should remap original subject ids to [0, K-1]
    """

    def __init__(
        self,
        dim: int = 128,
        seq_len: int = 77,
        num_subjects: int = 3,
        fi_layers: int = 8,
        num_heads: int = 4,
        grl_lambda: float = 1.0,
        dropout: float = 0.1,
        classifier_hidden_dim: int = None,
        fi_mlp_ratio: float = 4.0,
        fi_use_pos_embed: bool = False,
    ):
        super().__init__()

        if num_subjects <= 1:
            raise ValueError(
                f"SIFE requires at least 2 active subject classes, got {num_subjects}."
            )

        self.dim = int(dim)
        self.seq_len = int(seq_len)
        self.num_subjects = int(num_subjects)

        self.fi = FiModule(
            dim=self.dim,
            seq_len=self.seq_len,
            num_layers=int(fi_layers),
            num_heads=int(num_heads),
            mlp_ratio=float(fi_mlp_ratio),
            dropout=float(dropout),
            use_pos_embed=bool(fi_use_pos_embed),
        )

        self.grl = GRL(lambda_=float(grl_lambda))

        self.subject_discriminator = SubjectHead(
            dim=self.dim,
            seq_len=self.seq_len,
            num_classes=self.num_subjects,
            hidden_dim=classifier_hidden_dim,
            dropout=float(dropout),
        )

        self.subject_classifier = SubjectHead(
            dim=self.dim,
            seq_len=self.seq_len,
            num_classes=self.num_subjects,
            hidden_dim=classifier_hidden_dim,
            dropout=float(dropout),
        )

    def set_gradient_checkpointing(self, enable: bool = True):
        self.fi.set_gradient_checkpointing(enable)

    def forward(self, E_seq: torch.Tensor):
        """
        Args:
            E_seq: (B, T, D)

        Returns:
            dict with:
              - E_i_seq: (B, T, D)
              - E_s_seq: (B, T, D)
              - E_i: (B, D) pooled monitor
              - E_s: (B, D) pooled monitor
              - pred_subject_i: (B, K)
              - pred_subject_s: (B, K)
        """
        if E_seq.ndim != 3:
            raise ValueError(f"SIFE expects E_seq with shape (B, T, D), got {tuple(E_seq.shape)}")
        if E_seq.size(1) != self.seq_len:
            raise ValueError(
                f"SIFE expected seq_len={self.seq_len}, got T={E_seq.size(1)}"
            )
        if E_seq.size(2) != self.dim:
            raise ValueError(
                f"SIFE expected dim={self.dim}, got D={E_seq.size(2)}"
            )

        # ZEBRA-like invariant branch
        E_i_seq = self.fi(E_seq)

        # residual specific branch
        E_s_seq = E_seq - E_i_seq

        # pooled monitors only
        E_i = E_i_seq.mean(dim=1)
        E_s = E_s_seq.mean(dim=1)

        # subject adversarial / direct heads
        pred_subject_i = self.subject_discriminator(self.grl(E_i_seq))
        pred_subject_s = self.subject_classifier(E_s_seq)

        return {
            "E_i_seq": E_i_seq,
            "E_s_seq": E_s_seq,
            "E_i": E_i,
            "E_s": E_s,
            "pred_subject_i": pred_subject_i,
            "pred_subject_s": pred_subject_s,
        }