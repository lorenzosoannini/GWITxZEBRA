import torch.nn as nn
from torch.utils.checkpoint import checkpoint


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
    def __init__(
        self,
        in_dim=128,
        hidden_dim=256,
        out_channels=128,
        num_res_blocks=3,
        use_gradient_checkpointing=False,
    ):
        super().__init__()

        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.input_proj = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock1D(hidden_dim) for _ in range(num_res_blocks)]
        )
        self.output_proj = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, E_seq):
        x = E_seq.permute(0, 2, 1)
        x = self.input_proj(x)

        for block in self.res_blocks:
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.output_proj(x)
        return x