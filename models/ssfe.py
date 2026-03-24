import torch
import torch.nn as nn
import torch.nn.functional as F

from .grl import GRL


class _MLPProjector(nn.Module):
    """
    Generic projector:
        input  -> hidden -> hidden -> output
    with LayerNorm + GELU + Dropout.
    """

    def __init__(self, in_dim=128, hidden_dim=256, out_dim=512, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.normalize(x, dim=-1)
        return x


class SSFEProjector(nn.Module):
    """
    ZEBRA-like Semantic-Specific Feature Extraction module.

    Inputs:
        E_i: subject-invariant / semantic branch
        E_s: subject-specific / nuisance branch
        E  : full anchor feature

    Outputs:
        F_s: semantic CLIP-aligned feature
        F_i: adversarial / non-semantic CLIP branch (via GRL)
        F  : anchor CLIP feature from full representation
    """

    def __init__(
        self,
        in_dim=128,
        hidden_dim=256,
        out_dim=512,
        dropout=0.1,
        grl_lambda=1.0,
    ):
        super().__init__()

        self.grl = GRL(lambda_=grl_lambda)

        self.semantic_projector = _MLPProjector(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

        self.invariant_projector = _MLPProjector(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

        self.anchor_projector = _MLPProjector(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            dropout=dropout,
        )

    def forward(
        self,
        E_i: torch.Tensor,
        E_s: torch.Tensor,
        E: torch.Tensor,
    ):
        """
        Args:
            E_i: (B, D)
            E_s: (B, D)
            E:   (B, D)

        Returns:
            dict with:
                F_s: (B, out_dim)
                F_i: (B, out_dim)
                F:   (B, out_dim)
        """
        # semantic projector: should align with CLIP semantics
        F_s = self.semantic_projector(E_i)

        # invariant/nuisance projector:
        # same alignment loss can be used, but GRL reverses gradients
        E_s_grl = self.grl(E_s)
        F_i = self.invariant_projector(E_s_grl)

        # anchor from full representation
        F_anchor = self.anchor_projector(E)

        return {
            "F_s": F_s,
            "F_i": F_i,
            "F": F_anchor,
        }


# ---------------------------------------------------------
# LOSSES
# ---------------------------------------------------------
def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def cosine_alignment_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean cosine alignment loss:
        1 - cos(pred, target)

    Args:
        pred:   (B, D)
        target: (B, D)
    """
    pred = _normalize(pred)
    target = _normalize(target)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def info_nce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric InfoNCE between pred and target.

    Positive pairs are aligned by batch index.
    All other batch items act as negatives.

    Args:
        pred:   (B, D)
        target: (B, D)
        temperature: softmax temperature

    Returns:
        scalar loss
    """
    pred = _normalize(pred)
    target = _normalize(target)

    logits = pred @ target.t()
    logits = logits / temperature

    labels = torch.arange(pred.size(0), device=pred.device)

    loss_pt = F.cross_entropy(logits, labels)
    loss_tp = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_pt + loss_tp)