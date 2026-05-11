import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvResidualBlock(nn.Module):
    """
    Lightweight temporal residual block for EEG.

    Input / output:
        x: (B, C, T)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        dropout: float = 0.1,
        expansion: int = 2,
    ):
        super().__init__()

        padding = kernel_size // 2
        hidden = int(channels * expansion)

        self.norm1 = nn.GroupNorm(1, channels)
        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.pwconv = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ConvResidualBlock expects (B,C,T), got {tuple(x.shape)}")

        h = self.norm1(x)
        h = self.dwconv(h)
        h = self.pwconv(h)
        x = x + h
        x = self.norm2(x)
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embedding for token sequences.

    Input / output:
        x: (B, T, D)
    """

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        self.dim = int(dim)
        self.max_len = int(max_len)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Positional embedding expects (B,T,D), got {tuple(x.shape)}")
        t = x.shape[1]
        if t > self.max_len:
            raise ValueError(f"Sequence length {t} exceeds max_len={self.max_len}")
        return x + self.pe[:, :t, :].to(dtype=x.dtype, device=x.device)


class EEGTokenBackbone(nn.Module):
    """
    EEG token backbone for direct EEG -> CLIP-token alignment.

    Goal:
        Replace the GWIT LSTM classifier-oriented backbone with a token-oriented
        EEG encoder whose output is already close to the shape expected by the
        ZEBRA-like pipeline.

    Input:
        x: usually (B, 128, 440), but also accepts (B, 440, 128)

    Output dict:
        sequence: (B, target_tokens, token_dim)
        pooled:   (B, token_dim)

    Suggested default for current experiments:
        target_tokens=256
        token_dim=512
    """

    def __init__(
        self,
        in_channels: int = 128,
        input_time: int = 440,
        stem_dim: int = 256,
        token_dim: int = 512,
        target_tokens: int = 256,
        conv_blocks: int = 4,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        transformer_mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cls_token: bool = False,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.input_time = int(input_time)
        self.stem_dim = int(stem_dim)
        self.token_dim = int(token_dim)
        self.target_tokens = int(target_tokens)
        self.use_cls_token = bool(use_cls_token)

        self.stem = nn.Sequential(
            nn.Conv1d(self.in_channels, self.stem_dim, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(1, self.stem_dim),
            nn.GELU(),
            nn.Conv1d(self.stem_dim, self.stem_dim, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(1, self.stem_dim),
            nn.GELU(),
        )

        self.conv_blocks = nn.ModuleList(
            [
                ConvResidualBlock(
                    channels=self.stem_dim,
                    kernel_size=15 if i % 2 == 0 else 31,
                    dropout=dropout,
                )
                for i in range(int(conv_blocks))
            ]
        )

        self.to_token_dim = nn.Sequential(
            nn.Conv1d(self.stem_dim, self.token_dim, kernel_size=1),
            nn.GroupNorm(1, self.token_dim),
            nn.GELU(),
        )

        self.pos_embed = SinusoidalPositionalEmbedding(self.token_dim, max_len=max(self.target_tokens + 1, 4096))

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        else:
            self.register_parameter("cls_token", None)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(transformer_heads),
            dim_feedforward=int(self.token_dim * transformer_mlp_ratio),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(transformer_layers),
            norm=nn.LayerNorm(self.token_dim),
        )

        self.final_norm = nn.LayerNorm(self.token_dim)

        self._init_weights()

    def _init_weights(self):
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def _ensure_bct(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"EEGTokenBackbone expects 3D input, got {tuple(x.shape)}")

        # Accept both (B, C, T) and (B, T, C).
        if x.shape[1] == self.in_channels:
            return x
        if x.shape[2] == self.in_channels:
            return x.transpose(1, 2)

        raise ValueError(
            f"Cannot infer EEG layout. Expected one dimension to equal in_channels={self.in_channels}, "
            f"got shape={tuple(x.shape)}"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self._ensure_bct(x).float()  # (B, C, T)

        h = self.stem(x)                 # (B, stem_dim, T)

        for block in self.conv_blocks:
            h = block(h)

        # Explicit tokenization: produce target_tokens along the temporal/token axis.
        h = F.interpolate(
            h,
            size=self.target_tokens,
            mode="linear",
            align_corners=False,
        )                                # (B, stem_dim, target_tokens)

        h = self.to_token_dim(h)         # (B, token_dim, target_tokens)
        h = h.transpose(1, 2)            # (B, target_tokens, token_dim)

        if self.cls_token is not None:
            cls = self.cls_token.expand(h.shape[0], -1, -1)
            h = torch.cat([cls, h], dim=1)

        h = self.pos_embed(h)
        h = self.transformer(h)
        h = self.final_norm(h)

        if self.cls_token is not None:
            pooled = h[:, 0]
            seq = h[:, 1:]
        else:
            seq = h
            pooled = h.mean(dim=1)

        return {
            "sequence": seq,
            "pooled": pooled,
        }


def load_eeg_token_backbone_from_ckpt(backbone: EEGTokenBackbone, ckpt_path: str, strict: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    if not isinstance(sd, dict):
        raise ValueError(
            f"Unsupported checkpoint format in {ckpt_path}. "
            f"Expected a state_dict or a dict containing 'model_state_dict'."
        )

    clean_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("eeg_token_backbone."):
            k = k[len("eeg_token_backbone."):]
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        clean_sd[k] = v

    missing, unexpected = backbone.load_state_dict(clean_sd, strict=strict)

    print("[EEG Token Backbone Load]")
    print(f"Source: {ckpt_path}")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    if strict and (len(missing) > 0 or len(unexpected) > 0):
        raise RuntimeError(
            f"Error loading EEG token backbone from {ckpt_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )

    return missing, unexpected
