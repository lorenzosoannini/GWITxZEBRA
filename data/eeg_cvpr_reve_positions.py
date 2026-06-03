from pathlib import Path

import torch
import numpy as np


def load_cvpr128_reve_positions(
    path: str | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load CVPR EEG 128-channel positions aligned with REVE.

    Shape:
        [128, 3]

    Notes:
        - 126 channels are directly available in brain-bzh/reve-positions.
        - FFT9h and FFT10h are filled from MNE standard_1005.
        - The coordinate systems are identical for the 126 common channels
          according to the affine-fit validation.
    """
    if path is None:
        path = Path(__file__).resolve().parent / "eeg_cvpr128_reve_positions_affine.npy"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing REVE positions file: {path}. "
            "Run scripts/build_cvpr128_positions_with_mne_fit.py first."
        )

    arr = np.load(path).astype("float32")

    if arr.shape != (128, 3):
        raise RuntimeError(f"Expected positions shape (128, 3), got {arr.shape}")

    positions = torch.from_numpy(arr).to(device=device, dtype=dtype)

    if not torch.isfinite(positions).all():
        raise RuntimeError("Positions contain NaN or Inf.")

    return positions