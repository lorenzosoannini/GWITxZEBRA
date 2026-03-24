import os
import torch
import pickle
import numpy as np

from .eeg_feat_net import EEGFeatNet
from .name_map_ID import id_to_caption, id_to_caption_TVIZ


class EEGCaptionClassifier:
    """
    Clean version of GWIT EEG → caption system.
    Implements EEGFeatNet + KNN + caption lookup.
    """

    def __init__(self, dataset_name: str, root: str, device="cuda"):
        """
        Parameters
        ----------
        dataset_name : str
            Must contain 'CVPR' or 'TVIZ'
        root : str
            Path to GWIT_clean/data folder
        """

        self.dataset_name = dataset_name
        self.device = device

        is_cvpr = "CVPR" in dataset_name.upper()

        # -----------------------------------------
        # 1) Load correct caption dictionary
        # -----------------------------------------
        self.caption_dict = id_to_caption if is_cvpr else id_to_caption_TVIZ

        # -----------------------------------------
        # 2) Build EEG feature extractor model
        # -----------------------------------------
        if is_cvpr:
            self.model = EEGFeatNet(
                in_channels=128,
                n_features=128,
                projection_dim=128,
                num_layers=4
            )
            ckpt_path = os.path.join(root, "eegfeat_cvpr.pth")
            knn_path = os.path.join(root, "knn_cvpr.pkl")
        else:
            self.model = EEGFeatNet(
                in_channels=14,
                n_features=128,
                projection_dim=128,
                num_layers=4
            )
            ckpt_path = os.path.join(root, "eegfeat_tviz.pth")
            knn_path = os.path.join(root, "knn_tviz.pkl")

        # -----------------------------------------
        # 3) Load model checkpoint (with DataParallel fix)
        # -----------------------------------------
        state = torch.load(ckpt_path, map_location=device)

        # Handle checkpoints that include wrapper
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        # Remove "module." prefix if present (DataParallel)
        clean_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                clean_state[k[len("module."):]] = v
            else:
                clean_state[k] = v

        self.model.load_state_dict(clean_state, strict=True)
        self.model.to(device)
        self.model.eval()

        # -----------------------------------------
        # 4) Load KNN model
        # -----------------------------------------
        with open(knn_path, "rb") as f:
            self.knn = pickle.load(f)

    # -----------------------------------------
    # 5) Main caption inference
    # -----------------------------------------
    @torch.no_grad()
    def predict_captions(self, eeg_batch):
        # stack
        eeg_tensor = torch.stack([torch.as_tensor(eeg, dtype=torch.float32) for eeg in eeg_batch]).to(self.device)

        in_ch = self.model.encoder.input_size  # 128 per CVPR, 14 per TVIZ
        eeg_tensor = eeg_tensor.contiguous()

        # GWIT-style: se input è (B,C,T) con C=in_ch, swap -> (B,T,C) usando view
        if eeg_tensor.shape[1] == in_ch and eeg_tensor.shape[-1] != in_ch:
            eeg_tensor = eeg_tensor.view(-1, eeg_tensor.shape[2], eeg_tensor.shape[1])  # (B,T,C)
        # se è già (B,T,C) (last dim == in_ch) non fare nulla
        elif eeg_tensor.shape[-1] == in_ch:
            pass
        else:
            raise RuntimeError(f"Unexpected EEG shape {tuple(eeg_tensor.shape)} for in_channels={in_ch}")

        proj = self.model(eeg_tensor)
        pred_labels = self.knn.predict(proj.detach().cpu().numpy())
        return ["image of " + self.caption_dict[int(l)] for l in pred_labels]