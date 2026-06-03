import os
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from data.eeg_cvpr_reve_positions import load_cvpr128_reve_positions


class REVEZebraBackbone(nn.Module):
    """
    REVE-Large EEG backbone for the EEG-ZEBRA pipeline.

    Input:
        eeg: [B, 128, 200]

    Internal:
        REVE-Large returns [B, 128, 1, 1216]

    Output:
        {
            "sequence": [B, 128, 1216],
            "pooled":   [B, 1216],
        }

    This backbone intentionally preserves the native REVE representation:
        tokens = 128 electrode tokens
        dim    = 1216 REVE-Large hidden dimension

    The projection to OpenCLIP space [B, 256, 1664] is left to
    Stage1CLIPProjector / SSFEProjector.
    """

    def __init__(
        self,
        *,
        model_name: str = "brain-bzh/reve-large",
        positions_path: Optional[str] = None,
        freeze_reve: bool = True,
        output_dim: int = 1216,
        output_tokens: int = 128,
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze_reve = freeze_reve
        self.output_dim = int(output_dim)
        self.output_tokens = int(output_tokens)

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
        }

        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.reve = AutoModel.from_pretrained(model_name, **model_kwargs)

        positions = load_cvpr128_reve_positions(
            path=positions_path,
            device="cpu",
            dtype=torch.float32,
        )

        if tuple(positions.shape) != (128, 3):
            raise RuntimeError(
                f"Expected CVPR128 positions shape (128, 3), got {tuple(positions.shape)}"
            )

        self.register_buffer("positions", positions, persistent=False)

        if freeze_reve:
            self.freeze()

    def freeze(self):
        self.reve.eval()
        for p in self.reve.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.reve.parameters():
            p.requires_grad = True
        self.reve.train()
        self.freeze_reve = False

    @staticmethod
    def _extract_reve_tokens(out: Any) -> torch.Tensor:
        """
        REVE-Large may return [B, C, P, D].
        For our pipeline:
            [B, C, P, D] -> [B, C*P, D]

        With the current CVPR setup:
            [B, 128, 1, 1216] -> [B, 128, 1216]
        """

        def normalize_tensor_shape(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 4:
                b, c, p, d = x.shape
                return x.reshape(b, c * p, d)

            if x.ndim == 3:
                return x

            if x.ndim == 2:
                return x.unsqueeze(1)

            raise RuntimeError(f"Unsupported REVE output tensor shape: {tuple(x.shape)}")

        if isinstance(out, torch.Tensor):
            return normalize_tensor_shape(out)

        for attr in ["last_hidden_state", "hidden_states", "embeddings", "tokens"]:
            if hasattr(out, attr):
                val = getattr(out, attr)

                if attr == "hidden_states" and isinstance(val, (tuple, list)):
                    val = val[-1]

                if isinstance(val, torch.Tensor):
                    return normalize_tensor_shape(val)

        if isinstance(out, dict):
            for key in ["last_hidden_state", "hidden_states", "embeddings", "tokens", "x", "output"]:
                if key in out:
                    val = out[key]

                    if key == "hidden_states" and isinstance(val, (tuple, list)):
                        val = val[-1]

                    if isinstance(val, torch.Tensor):
                        return normalize_tensor_shape(val)

            for _, val in out.items():
                if isinstance(val, torch.Tensor) and val.ndim in (2, 3, 4):
                    return normalize_tensor_shape(val)

        if isinstance(out, (tuple, list)):
            for val in out:
                if isinstance(val, torch.Tensor) and val.ndim in (2, 3, 4):
                    return normalize_tensor_shape(val)

        raise RuntimeError("Could not extract REVE tokens from output.")

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            eeg: [B, 128, 200]

        Returns:
            dict with:
                sequence: [B, 128, 1216]
                pooled:   [B, 1216]
        """

        if eeg.ndim != 3:
            raise ValueError(f"Expected EEG shape [B, 128, 200], got {tuple(eeg.shape)}")

        if eeg.shape[1] != 128:
            raise ValueError(f"Expected 128 EEG channels, got {eeg.shape[1]}")

        bsz = eeg.shape[0]

        positions = self.positions.to(device=eeg.device, dtype=eeg.dtype)
        positions = positions.unsqueeze(0).expand(bsz, -1, -1).contiguous()

        if self.freeze_reve:
            self.reve.eval()
            with torch.no_grad():
                out = self.reve(eeg, positions)
        else:
            out = self.reve(eeg, positions)

        sequence = self._extract_reve_tokens(out)

        if sequence.ndim != 3:
            raise RuntimeError(f"Expected extracted REVE tokens [B,N,D], got {tuple(sequence.shape)}")

        if sequence.shape[0] != bsz:
            raise RuntimeError(
                f"Batch mismatch: EEG batch={bsz}, REVE tokens batch={sequence.shape[0]}"
            )

        if sequence.shape[1] != self.output_tokens:
            raise RuntimeError(
                f"Expected {self.output_tokens} REVE tokens, got {sequence.shape[1]}"
            )

        if sequence.shape[2] != self.output_dim:
            raise RuntimeError(
                f"Expected REVE dim {self.output_dim}, got {sequence.shape[2]}"
            )

        pooled = sequence.mean(dim=1)

        return {
            "sequence": sequence,
            "pooled": pooled,
        }


def build_reve_zebra_backbone(
    *,
    model_name: str = "brain-bzh/reve-large",
    positions_path: Optional[str] = None,
    freeze_reve: bool = True,
    torch_dtype: Optional[torch.dtype] = None,
) -> REVEZebraBackbone:
    return REVEZebraBackbone(
        model_name=model_name,
        positions_path=positions_path,
        freeze_reve=freeze_reve,
        torch_dtype=torch_dtype,
    )