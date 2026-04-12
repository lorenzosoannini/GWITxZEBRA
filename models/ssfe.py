import torch
import torch.nn as nn
import torch.nn.functional as F

from .grl import GRL


# ---------------------------------------------------------
# ADAPTERS
# ---------------------------------------------------------
class _SimpleSequenceAdapter(nn.Module):
    """
    Simple adapter:
      1. resize tokens only if needed
      2. token-wise MLP projection
      3. L2 normalization on feature dim
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 768,
        target_tokens: int = 49,
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

    def _maybe_resize_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B, T, D), got shape {tuple(x.shape)}")

        if x.shape[1] == self.target_tokens:
            return x

        x = x.transpose(1, 2)  # (B, D, T_in)
        x = F.adaptive_avg_pool1d(x, self.target_tokens)  # (B, D, target_tokens)
        x = x.transpose(1, 2)  # (B, target_tokens, D)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_resize_tokens(x)
        y = self.token_mlp(x)
        y = F.normalize(y, dim=-1)
        return y


class _ZebraSemanticBottleneck(nn.Module):
    """
    ZEBRA-like token mixing + channel projection.

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

        self.seq_mlp = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_dim),
        )

        self.channel_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)        # (B, D_in, T)
        x = self.seq_mlp(x)          # (B, D_in, T)
        x = x.transpose(1, 2)        # (B, T, D_in)
        x = self.channel_proj(x)     # (B, T, D_out)
        return x


class _ZebraChannelWiseAttention(nn.Module):
    """
    ZEBRA-like attention refinement preserving token length.
    """

    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.num_heads
        dh = self.head_dim

        q = self.query_proj(x).view(b, t, h, dh).transpose(1, 2)
        k = self.key_proj(x).view(b, t, h, dh).transpose(1, 2)
        v = self.value_proj(x).view(b, t, h, dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (dh ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.out_proj(out)
        return out


class _ZebraLikeSequenceAdapter(nn.Module):
    """
    ZEBRA-like adapter.

    Behavior:
      1. resize tokens only if needed
      2. token mixing on sequence dimension
      3. channel projection
      4. attention refinement
      5. L2 normalization
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 768,
        target_tokens: int = 49,
        dropout: float = 0.1,
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

        self.attn = _ZebraChannelWiseAttention(
            dim=out_dim,
            num_heads=num_heads,
        )

    def _maybe_resize_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (B, T, D), got shape {tuple(x.shape)}")

        if x.shape[1] == self.target_tokens:
            return x

        x = x.transpose(1, 2)
        x = F.adaptive_avg_pool1d(x, self.target_tokens)
        x = x.transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._maybe_resize_tokens(x)
        x = self.bottleneck(x)
        x = self.attn(x)
        x = F.normalize(x, dim=-1)
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
# CLASSIFIERS / PROJECTORS
# ---------------------------------------------------------
class ImageClassifier(nn.Module):
    """
    Single-label image-class semantic classifier over image classes.

    Input:
        x: (B, T, D)

    Output:
        logits: (B, num_classes)
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 40):
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
    Single-label adversarial image-class semantic discriminator over image classes.

    GRL is applied here, as in the ZEBRA code.
    """

    def __init__(
        self,
        embed_dim: int = 768,
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
    ZEBRA-like CLIP text projector from F:
      mean pooling over tokens
      linear projection to text embedding space
      L2 normalization
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"AnchorTextProjector expects (B, T, D), got {tuple(x.shape)}")

        x = x.mean(dim=1)
        x = self.proj(x)
        x = F.normalize(x, dim=-1)
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
        - anchor text alignment on F through a simple CLIP-like projector
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        out_dim: int = 768,
        target_tokens: int = 49,
        adapter_type: str = "zebra_like",
        dropout: float = 0.1,
        grl_lambda: float = 1.0,
        num_image_classes: int = 40,
        text_out_dim: int = 512,
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

        # Shared image classifier used both on F_s and F, as in ZEBRA
        self.image_classifier = ImageClassifier(
            embed_dim=out_dim,
            num_classes=num_image_classes,
        )

        self.image_discriminator = ImageDiscriminator(
            embed_dim=out_dim,
            num_classes=num_image_classes,
            grl_lambda=grl_lambda,
        )

        # Simple CLIP-like text head on F
        self.anchor_text_projector = AnchorTextProjector(
            in_dim=out_dim,
            out_dim=text_out_dim,
        )

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

        # Semantic heads
        pred_image_cls = self.image_classifier(F_s)
        pred_image_dis = self.image_discriminator(F_i)

        # Anchor heads on F
        # Visual anchor is applied directly on F, as in ZEBRA
        F_anchor_visual = F_general

        # Reuse same classifier on F
        pred_image_cls_anchor = self.image_classifier(F_general)

        # Simple text projector on F
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
# HELPERS
# ---------------------------------------------------------
def _normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def _flatten_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor (B, T, D), got shape {tuple(x.shape)}")
    return x.reshape(x.shape[0], -1)


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