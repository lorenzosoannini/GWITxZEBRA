import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class EEGReconstructionDecoder(nn.Module):
    """
    Input:
        E_seq: (B, T, D)  → (B, 440, 128)

    Output:
        EEG reconstructed: (B, C, T) → (B, 128, 440)
    """

    def __init__(
        self,
        in_dim=128,
        hidden_dim=256,
        out_channels=128,
        num_res_blocks=3,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock1D(hidden_dim) for _ in range(num_res_blocks)]
        )

        self.output_proj = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, E_seq):
        # E_seq: (B, T, D) → (B, D, T)
        x = E_seq.permute(0, 2, 1)

        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.output_proj(x)

        # (B, C, T)
        return x