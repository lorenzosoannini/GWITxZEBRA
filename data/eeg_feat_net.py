import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGFeatNet(nn.Module):
    """
    EEGFeatNet compatible with both (B, T, C) and (B, C, T) inputs.
    Internally feeds LSTM with (B, T, C) where C == in_channels.
    """

    def __init__(self, in_channels, n_features, projection_dim, num_layers):
        super().__init__()

        self.hidden_size = n_features
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=in_channels,
            hidden_size=n_features,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(n_features, projection_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
          - (B, T, C)  where C == in_channels  [preferred]
          - (B, C, T)  where C == in_channels  [auto-fixed]
        Returns:
          - (B, projection_dim) L2-normalized embeddings
        """
        if x.ndim != 3:
            raise ValueError(f"EEGFeatNet expects a 3D tensor (B,*,*). Got shape={tuple(x.shape)}")

        in_ch = self.encoder.input_size  # == in_channels

        # Ensure last dim is the feature dim (=in_channels), as required by batch_first LSTM
        if x.shape[-1] != in_ch:
            if x.shape[1] == in_ch:
                # (B, C, T) -> (B, T, C)
                x = x.permute(0, 2, 1).contiguous()
            else:
                raise ValueError(
                    f"Input shape {tuple(x.shape)} is incompatible with in_channels={in_ch}. "
                    f"Expected last dim == {in_ch} (B,T,C) or dim1 == {in_ch} (B,C,T)."
                )

        # initial hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)

        _, (hn, _) = self.encoder(x, (h0, c0))

        feat = hn[-1]  # last LSTM layer output (B, hidden_size)

        proj = self.fc(feat)
        proj = F.normalize(proj, dim=-1)

        return proj