import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .grl import GRL


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _flatten_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor (B, T, D), got shape {tuple(x.shape)}")
    return x.reshape(x.shape[0], -1)


def _maybe_resize_tokens(x: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """
    Resize token length only if needed.

    This is NOT part of the original ZEBRA design; it is only a compatibility
    fallback for the user's EEG backbone if token count != target_tokens.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor (B, T, D), got shape {tuple(x.shape)}")

    if x.shape[1] == target_tokens:
        return x

    x = x.transpose(1, 2)                  # (B, D, T_in)
    x = F.adaptive_avg_pool1d(x, target_tokens)
    x = x.transpose(1, 2)                  # (B, target_tokens, D)
    return x


def _build_positive_mask_from_group_ids(group_ids: torch.Tensor) -> torch.Tensor:
    """
    group_ids: (B,)
    returns:
        positive_mask: (B, B) bool
    """
    if group_ids.ndim != 1:
        raise ValueError(f"group_ids must be 1D, got shape {tuple(group_ids.shape)}")

    return group_ids[:, None] == group_ids[None, :]


def _multi_positive_contrastive_directional_loss(
    query: torch.Tensor,
    target: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.07,
    exclude_self: bool = False,
) -> torch.Tensor:
    """
    Directional multi-positive contrastive loss.

    query:         (B, D)
    target:        (B, D)
    positive_mask: (B, B) bool, where positive_mask[i,j]=True means j is a positive for i

    If exclude_self=True, the diagonal is removed from positives.
    Rows with zero positives after masking are ignored.
    """
    if query.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"_multi_positive_contrastive_directional_loss expects 2D tensors. "
            f"Got query={tuple(query.shape)}, target={tuple(target.shape)}"
        )

    if query.shape[0] != target.shape[0]:
        raise ValueError(
            f"query and target must have same batch size, got {query.shape[0]} vs {target.shape[0]}"
        )

    if positive_mask.shape != (query.shape[0], target.shape[0]):
        raise ValueError(
            f"positive_mask must have shape {(query.shape[0], target.shape[0])}, "
            f"got {tuple(positive_mask.shape)}"
        )

    query = _normalize(query)
    target = _normalize(target)

    logits = (query @ target.t()) / temperature
    log_probs = F.log_softmax(logits, dim=-1)

    pos_mask = positive_mask.to(device=query.device, dtype=torch.bool)

    if exclude_self:
        diag = torch.eye(pos_mask.shape[0], device=pos_mask.device, dtype=torch.bool)
        pos_mask = pos_mask & (~diag)

    pos_mask_f = pos_mask.float()
    pos_count = pos_mask_f.sum(dim=-1)  # (B,)

    valid_rows = pos_count > 0
    if not valid_rows.any():
        # fallback sicuro: se non esistono positivi multipli,
        # usiamo la diagonale standard
        labels = torch.arange(query.shape[0], device=query.device)
        return F.cross_entropy(logits, labels)

    # average log-prob over positives for each query row
    loss_per_row = -((log_probs * pos_mask_f).sum(dim=-1) / pos_count.clamp_min(1.0))
    loss = loss_per_row[valid_rows].mean()
    return loss


def multi_positive_info_nce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.07,
    exclude_self: bool = False,
) -> torch.Tensor:
    """
    Symmetric multi-positive InfoNCE.

    Positives are defined by equality of group_ids.
    """
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"multi_positive_info_nce_loss expects 2D tensors. "
            f"Got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )

    if group_ids.ndim != 1:
        raise ValueError(f"group_ids must be 1D, got shape {tuple(group_ids.shape)}")

    if pred.shape[0] != target.shape[0] or pred.shape[0] != group_ids.shape[0]:
        raise ValueError(
            f"Batch size mismatch: pred={pred.shape[0]}, target={target.shape[0]}, group_ids={group_ids.shape[0]}"
        )

    positive_mask = _build_positive_mask_from_group_ids(group_ids.to(pred.device))

    loss_pt = _multi_positive_contrastive_directional_loss(
        query=pred,
        target=target,
        positive_mask=positive_mask,
        temperature=temperature,
        exclude_self=exclude_self,
    )

    loss_tp = _multi_positive_contrastive_directional_loss(
        query=target,
        target=pred,
        positive_mask=positive_mask.t(),
        temperature=temperature,
        exclude_self=exclude_self,
    )

    return 0.5 * (loss_pt + loss_tp)


def multi_positive_sequence_info_nce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.07,
    exclude_self: bool = False,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"multi_positive_sequence_info_nce_loss requires same shape for pred and target, "
            f"got pred={tuple(pred.shape)} vs target={tuple(target.shape)}"
        )

    pred_flat = _flatten_sequence(pred)
    target_flat = _flatten_sequence(target)
    return multi_positive_info_nce_loss(
        pred_flat,
        target_flat,
        group_ids=group_ids,
        temperature=temperature,
        exclude_self=exclude_self,
    )


# ---------------------------------------------------------
# OPTIONAL SIMPLE ADAPTER
# ---------------------------------------------------------
class _SimpleSequenceAdapter(nn.Module):
    """
    Simple fallback adapter:
      1. resize tokens only if needed
      2. token-wise MLP projection
      3. L2 normalization on feature dim

    Kept only as an option. The preferred choice for ZEBRA-style experiments
    is adapter_type="zebra_like".
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 1664,
        target_tokens: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.target_tokens = int(target_tokens)

        self.token_mlp = nn.Sequential(
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
        x = _maybe_resize_tokens(x, self.target_tokens)
        y = self.token_mlp(x)
        return y


# ---------------------------------------------------------
# ZEBRA-LIKE PROJECTOR BLOCKS
# ---------------------------------------------------------
class _ZebraSemanticBottleneck(nn.Module):
    """
    Very close to ZEBRA's SemanticBottleneck.

    Input:
        x: (B, T, D_in)

    Output:
        y: (B, T, D_out)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        seq_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.use_gradient_checkpointing = False

        self.mlp = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_dim),
        )

        self.project = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def set_gradient_checkpointing(self, enable: bool = True):
        self.use_gradient_checkpointing = bool(enable)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)      # (B, D_in, T)
        x = self.mlp(x)            # (B, D_in, T)
        x = x.transpose(1, 2)      # (B, T, D_in)
        x = self.project(x)        # (B, T, D_out)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"_ZebraSemanticBottleneck expects (B, T, D), got {tuple(x.shape)}")

        if self.training and self.use_gradient_checkpointing:
            return checkpoint(self._forward_impl, x, use_reentrant=False)

        return self._forward_impl(x)


class _ZebraChannelWiseAttention(nn.Module):
    """
    Closer replica of ZEBRA's ChannelWiseAttention.

    Important:
    this is intentionally NOT a standard token self-attention implementation.
    It follows the algebra used in the original ZEBRA code as closely as possible.
    """

    def __init__(self, in_dim: int, num_heads: int = 1):
        super().__init__()

        if in_dim % num_heads != 0:
            raise ValueError(f"in_dim={in_dim} must be divisible by num_heads={num_heads}")

        self.use_gradient_checkpointing = False

        self.query_proj = nn.Linear(in_dim, in_dim)
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.value_proj = nn.Linear(in_dim, in_dim)
        self.out_proj = nn.Linear(in_dim, in_dim)

        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads

    def set_gradient_checkpointing(self, enable: bool = True):
        self.use_gradient_checkpointing = bool(enable)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape

        q = self.query_proj(x)   # [B, N, C]
        k = self.key_proj(x)     # [B, N, C]
        v = self.value_proj(x)   # [B, N, C]

        # This mirrors the original ZEBRA code path.
        scores = (q.transpose(-2, -1) @ k) / (self.head_dim ** 0.5)   # [B, C, C]
        attn = torch.softmax(scores, dim=-1)                          # [B, C, C]

        attn_out = attn @ v.transpose(-2, -1)                         # [B, C, N]
        attn_out = attn_out.transpose(-2, -1)                         # [B, N, C]

        out = self.out_proj(attn_out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"_ZebraChannelWiseAttention expects (B, N, C), got {tuple(x.shape)}")

        if self.training and self.use_gradient_checkpointing:
            return checkpoint(self._forward_impl, x, use_reentrant=False)

        return self._forward_impl(x)


class _ZebraLikeSequenceAdapter(nn.Module):
    """
    ZEBRA-like projector:
      1. resize tokens only if needed
      2. token mixing on sequence dimension
      3. channel projection
      4. channel-wise attention refinement
      5. L2 normalization
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 1664,
        target_tokens: int = 256,
        dropout: float = 0.1,   # kept for interface compatibility; unused here
        num_heads: int = 1,
    ):
        super().__init__()
        self.target_tokens = int(target_tokens)

        seq_hidden_dim = max(hidden_dim, self.target_tokens)

        self.bottleneck = _ZebraSemanticBottleneck(
            in_dim=in_dim,
            out_dim=out_dim,
            seq_dim=self.target_tokens,
            hidden_dim=seq_hidden_dim,
        )

        self.broadcaster = _ZebraChannelWiseAttention(
            in_dim=out_dim,
            num_heads=num_heads,
        )

    def set_gradient_checkpointing(self, enable: bool = True):
        self.bottleneck.set_gradient_checkpointing(enable)
        self.broadcaster.set_gradient_checkpointing(enable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _maybe_resize_tokens(x, self.target_tokens)
        x = self.bottleneck(x)
        x = self.broadcaster(x)
        return x


def _build_sequence_adapter(
    adapter_type: str,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    target_tokens: int,
    dropout: float,
):
    if adapter_type == "simple":
        return _SimpleSequenceAdapter(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            target_tokens=target_tokens,
            dropout=dropout,
        )

    if adapter_type == "zebra_like":
        return _ZebraLikeSequenceAdapter(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            target_tokens=target_tokens,
            dropout=dropout,
            num_heads=1,
        )

    raise ValueError(f"Unsupported adapter_type: {adapter_type}")

# ---------------------------------------------------------
# STAGE-1 CLIP PRETRAIN PROJECTOR
# ---------------------------------------------------------
class PretrainCLIPProjector(nn.Module):
    """
    ZEBRA-like pretraining projector.

    Purpose:
        During stage1, align the generic EEG sequence representation E_seq
        directly to CLIP image tokens.

    ZEBRA counterpart:
        clip_embed_all = self.to_clip(brain_embed)

    Input:
        E_seq: (B, T_eeg, D_eeg)

    Output:
        clip_pred: (B, target_tokens, out_dim)
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 1664,
        target_tokens: int = 256,
        adapter_type: str = "zebra_like",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.target_tokens = int(target_tokens)
        self.adapter_type = adapter_type

        self.projector = _build_sequence_adapter(
            adapter_type=adapter_type,
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            target_tokens=self.target_tokens,
            dropout=dropout,
        )

    def set_gradient_checkpointing(self, enable: bool = True):
        if hasattr(self.projector, "set_gradient_checkpointing"):
            self.projector.set_gradient_checkpointing(enable)

    def forward(self, E_seq: torch.Tensor) -> torch.Tensor:
        if E_seq.ndim != 3:
            raise ValueError(
                f"PretrainCLIPProjector expects E_seq with shape (B, T, D), "
                f"got {tuple(E_seq.shape)}"
            )

        return self.projector(E_seq)

# ---------------------------------------------------------
# CLASSIFIERS / PROJECTORS
# ---------------------------------------------------------
class ImageClassifier(nn.Module):
    """
    ZEBRA-like image classifier.

    Input:
        x: (B, T, D)

    Output:
        logits: (B, num_classes)
    """

    def __init__(self, embed_dim: int = 1664, num_classes: int = 40):
        super().__init__()
        self.attn_proj = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ImageClassifier expects (B, T, D), got {tuple(x.shape)}")

        attn_weights = self.attn_proj(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x_weighted = (attn_weights * x).sum(dim=1)
        logits = self.classifier(x_weighted)
        return logits


class ImageDiscriminator(nn.Module):
    """
    ZEBRA-like adversarial image discriminator.
    GRL is applied exactly before the same attention-pool classifier head.
    """

    def __init__(
        self,
        embed_dim: int = 1664,
        num_classes: int = 40,
        grl_lambda: float = 1.0,
    ):
        super().__init__()
        self.grl = GRL(lambda_=grl_lambda)
        self.attn_proj = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ImageDiscriminator expects (B, T, D), got {tuple(x.shape)}")

        x = self.grl(x)
        attn_weights = self.attn_proj(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x_weighted = (attn_weights * x).sum(dim=1)
        logits = self.classifier(x_weighted)
        return logits


class AnchorTextProjector(nn.Module):
    """
    Close counterpart of ZEBRA's CLIPProj:

        x = mean over tokens
        x = linear projection

    Normalization is applied at the end because your losses expect normalized text anchors.
    """

    def __init__(self, in_dim: int = 1664, out_dim: int = 1280):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"AnchorTextProjector expects (B, T, D), got {tuple(x.shape)}")

        x = torch.mean(x, dim=1)   # (B, in_dim)
        x = x @ self.proj          # (B, out_dim)
        return x


# ---------------------------------------------------------
# SSFE
# ---------------------------------------------------------
class SSFEProjector(nn.Module):
    """
    ZEBRA-aligned SSFE module.

    Structure:
        F   = P(E)        general anchor representation
        F_s = P_s(E_i)    semantic-specific branch
        F_i = F - F_s     residual semantic-invariant branch

    Supervision:
        - semantic image cls on F_s
        - adversarial image dis on F_i
        - anchor visual alignment directly on F
        - anchor image cls on F using the SAME image classifier
        - anchor text alignment on F through a ZEBRA-like CLIPProj head
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 1664,
        target_tokens: int = 256,
        adapter_type: str = "zebra_like",
        dropout: float = 0.1,
        grl_lambda: float = 1.0,
        num_image_classes: int = 40,
        text_out_dim: int = 1280,
    ):
        super().__init__()

        self.target_tokens = int(target_tokens)
        self.adapter_type = adapter_type

        # P(E) -> F
        self.general_projector = _build_sequence_adapter(
            adapter_type=adapter_type,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            target_tokens=target_tokens,
            dropout=dropout,
        )

        # P_s(E_i) -> F_s
        self.semantic_projector = _build_sequence_adapter(
            adapter_type=adapter_type,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            target_tokens=target_tokens,
            dropout=dropout,
        )

        # Shared classifier, as in ZEBRA logic
        self.image_classifier = ImageClassifier(
            embed_dim=out_dim,
            num_classes=num_image_classes,
        )

        self.image_discriminator = ImageDiscriminator(
            embed_dim=out_dim,
            num_classes=num_image_classes,
            grl_lambda=grl_lambda,
        )

        self.anchor_text_projector = AnchorTextProjector(
            in_dim=out_dim,
            out_dim=text_out_dim,
        )

    def set_gradient_checkpointing(self, enable: bool = True):
        if hasattr(self.general_projector, "set_gradient_checkpointing"):
            self.general_projector.set_gradient_checkpointing(enable)
        if hasattr(self.semantic_projector, "set_gradient_checkpointing"):
            self.semantic_projector.set_gradient_checkpointing(enable)

    def forward(
        self,
        E_i: torch.Tensor,
        E: torch.Tensor,
    ):
        if E_i.ndim != 3 or E.ndim != 3:
            raise ValueError(
                f"SSFEProjector expects 3D inputs (B, T, D). "
                f"Got E_i={tuple(E_i.shape)}, E={tuple(E.shape)}"
            )

        # F = P(E)
        F_general = self.general_projector(E)

        # F_s = P_s(E_i)
        F_s = self.semantic_projector(E_i)

        # F_i = F - F_s
        F_i = F_general - F_s

        # ZEBRA-like heads
        pred_image_cls = self.image_classifier(F_s)
        pred_image_dis = self.image_discriminator(F_i)

        # anchor visual is directly F
        F_anchor_visual = F_general

        # reuse same image classifier on F
        pred_image_cls_anchor = self.image_classifier(F_general)

        # text anchor from F
        anchor_text_embed = self.anchor_text_projector(F_general)

        return {
            "F_s": F_s,
            "F_i": F_i,
            "F": F_general,
            "F_anchor_visual": F_anchor_visual,
            "pred_image_cls": pred_image_cls,
            "pred_image_dis": pred_image_dis,
            "pred_image_cls_anchor": pred_image_cls_anchor,
            "anchor_text_embed": anchor_text_embed,
        }


# ---------------------------------------------------------
# LOSSES
# ---------------------------------------------------------
def cosine_alignment_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"cosine_alignment_loss expects 2D tensors. "
            f"Got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )

    pred = _normalize(pred)
    target = _normalize(target)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def info_nce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(
            f"info_nce_loss expects 2D tensors. "
            f"Got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )

    pred = _normalize(pred)
    target = _normalize(target)

    logits = pred @ target.t()
    logits = logits / temperature

    labels = torch.arange(pred.size(0), device=pred.device)

    loss_pt = F.cross_entropy(logits, labels)
    loss_tp = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_pt + loss_tp)


def sequence_cosine_alignment_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"sequence_cosine_alignment_loss requires same shape for pred and target, "
            f"got pred={tuple(pred.shape)} vs target={tuple(target.shape)}"
        )

    pred_flat = _flatten_sequence(pred)
    target_flat = _flatten_sequence(target)
    return cosine_alignment_loss(pred_flat, target_flat)


def sequence_info_nce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"sequence_info_nce_loss requires same shape for pred and target, "
            f"got pred={tuple(pred.shape)} vs target={tuple(target.shape)}"
        )

    pred_flat = _flatten_sequence(pred)
    target_flat = _flatten_sequence(target)
    return info_nce_loss(pred_flat, target_flat, temperature=temperature)