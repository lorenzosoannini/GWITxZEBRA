import torch
import torch.nn as nn


class GWITEEGBackbone(nn.Module):
    def __init__(self, in_channels=128, hidden_size=128, num_layers=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        # accetta sia (B,128,440) che (B,440,128)
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got {x.shape}")

        if x.shape[1] == 128 and x.shape[2] != 128:
            x = x.transpose(1, 2)

        h0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )
        c0 = torch.zeros(
            self.num_layers,
            x.size(0),
            self.hidden_size,
            device=x.device,
            dtype=x.dtype,
        )

        out, (h_n, _) = self.encoder(x, (h0, c0))
        feat = h_n[-1]

        return {
            "sequence": out,   # (B,T,D)
            "pooled": feat,    # (B,D)
        }


def load_eeg_backbone_from_ckpt(backbone, ckpt_path, strict=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Caso 1: checkpoint tipo {"model_state_dict": ...}
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        # Caso 2: file salvato direttamente come state_dict
        sd = ckpt

    if not isinstance(sd, dict):
        raise ValueError(
            f"Unsupported checkpoint format in {ckpt_path}. "
            f"Expected a state_dict or a dict containing 'model_state_dict'."
        )

    new_sd = {}
    for k, v in sd.items():
        # rimuove eventuale prefisso DDP
        if k.startswith("module."):
            k = k[len("module."):]

        # Caso già corretto: encoder.*
        if k.startswith("encoder."):
            new_sd[k] = v
            continue

        # Caso salvato come backbone.encoder.*
        if k.startswith("backbone.encoder."):
            new_sd[k[len("backbone."):]] = v
            continue

        # Caso salvato come eeg_backbone.encoder.*
        if k.startswith("eeg_backbone.encoder."):
            new_sd[k[len("eeg_backbone."):]] = v
            continue

    # Fallback: se non abbiamo trovato chiavi encoder.*, prova a usare lo state_dict così com'è
    # Utile per i file stage1 salvati con backbone.state_dict()
    if len(new_sd) == 0:
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_sd[k] = v

    missing, unexpected = backbone.load_state_dict(new_sd, strict=strict)

    print("[EEG Backbone Load]")
    print(f"Source: {ckpt_path}")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    if strict and (len(missing) > 0 or len(unexpected) > 0):
        raise RuntimeError(
            f"Error loading EEG backbone from {ckpt_path}: "
            f"missing={missing}, unexpected={unexpected}"
        )