import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from data.name_map_ID import id_to_caption
except ImportError:
    try:
        from name_map_ID import id_to_caption
    except ImportError:
        id_to_caption = None


class EEGRevePTHDataset(Dataset):
    """
    Dataset based on the original EEG CVPR .pth file.

    It returns EEG already prepared for REVE-Large:

        original EEG: [128, L], int16, 1000 Hz
        standardize: (eeg - means) / stddevs
        crop:        crop_start : crop_start + crop_len
        resample:    1000 Hz -> 200 Hz
        pad/crop:    final length = 200, according to pad_mode

    Output item, GWIT/HF-style:
        {
            "image": Tensor[3, image_size, image_size], optional, RGB in [-1, 1],
            "conditioning_pixel_values": Tensor[128, 200], REVE EEG,
            "conditioning_image": Tensor[128, 440], optional GWIT-style EEG,
            "caption": str,
            "label_folder": str,
            "label": LongTensor scalar,
            "subject": LongTensor scalar,
        }
    """

    def __init__(
        self,
        *,
        pth_path: str,
        subjects: Optional[Sequence[int]] = None,
        image_root: Optional[str] = None,
        return_image: bool = True,
        return_conditioning_image: bool = False,
        image_size: int = 512,
        image_extensions: Sequence[str] = (".JPEG", ".jpg", ".jpeg", ".png"),
        generated_captions_path: Optional[str] = None,
        crop_start: int = 20,
        crop_len: int = 440,
        original_fs: float = 1000.0,
        target_fs: float = 200.0,
        pad_to_len: int = 200,
        pad_mode: str = "center",
        standardize: bool = True,
        max_samples_per_subject: Optional[int] = None,
        build_index_cache_path: Optional[str] = None,
    ):
        super().__init__()

        self.pth_path = Path(pth_path)
        if not self.pth_path.exists():
            raise FileNotFoundError(f"Missing pth file: {self.pth_path}")

        self.image_root = Path(image_root) if image_root is not None else None
        self.return_image = bool(return_image)
        self.return_conditioning_image = bool(return_conditioning_image)
        self.image_extensions = tuple(image_extensions)

        if self.return_image and self.image_root is None:
            print(
                "[EEG REVE DATASET] return_image=True but image_root=None. "
                "The dataset will not return the 'image' field."
            )
            self.return_image = False

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((int(image_size), int(image_size)), antialias=True),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        )

        self.generated_captions = None
        if generated_captions_path is not None:
            generated_captions_path = Path(generated_captions_path)
            if not generated_captions_path.exists():
                raise FileNotFoundError(
                    f"Missing generated captions file: {generated_captions_path}"
                )
            with open(generated_captions_path, "r") as f:
                self.generated_captions = json.load(f)

        self.crop_start = int(crop_start)
        self.crop_len = int(crop_len)
        self.crop_end = self.crop_start + self.crop_len
        self.original_fs = float(original_fs)
        self.target_fs = float(target_fs)
        self.pad_to_len = int(pad_to_len)
        self.pad_mode = str(pad_mode).lower()
        if self.pad_mode not in {"center", "right", "left"}:
            raise ValueError(
                f"Unsupported pad_mode={pad_mode!r}. Expected one of: center, right, left."
            )
        self.standardize = bool(standardize)

        ratio = self.original_fs / self.target_fs
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError(
                f"Expected integer original_fs/target_fs ratio, got {ratio}"
            )
        self.downsample_factor = int(round(ratio))

        print(f"[EEG REVE DATASET] Loading {self.pth_path} ...")
        obj = torch.load(str(self.pth_path), map_location="cpu", weights_only=False)

        required_keys = {"dataset", "labels", "images", "means", "stddevs"}
        missing = required_keys - set(obj.keys())
        if missing:
            raise KeyError(f"Missing keys in pth file: {missing}")

        self.raw_dataset = obj["dataset"]
        self.labels = obj["labels"]
        self.images = obj["images"]
        self.means = obj["means"].float()      # [128, 1]
        self.stddevs = obj["stddevs"].float()  # [128, 1]

        if tuple(self.means.shape) != (128, 1):
            raise RuntimeError(f"Expected means shape (128,1), got {tuple(self.means.shape)}")

        if tuple(self.stddevs.shape) != (128, 1):
            raise RuntimeError(f"Expected stddevs shape (128,1), got {tuple(self.stddevs.shape)}")

        subject_set = None if subjects is None else set(int(s) for s in subjects)

        # Build valid sample index.
        self.indices = []
        per_subject_count = {}

        for pth_idx, sample in enumerate(self.raw_dataset):
            subj = int(sample["subject"])
            eeg = sample["eeg"]

            if subject_set is not None and subj not in subject_set:
                continue

            if eeg.shape[0] != 128:
                continue

            if eeg.shape[1] < self.crop_end:
                continue

            if max_samples_per_subject is not None:
                c = per_subject_count.get(subj, 0)
                if c >= max_samples_per_subject:
                    continue
                per_subject_count[subj] = c + 1

            self.indices.append(pth_idx)

        if len(self.indices) == 0:
            raise RuntimeError("No valid EEG samples found after filtering.")

        print(f"[EEG REVE DATASET] Total raw samples: {len(self.raw_dataset)}")
        print(f"[EEG REVE DATASET] Valid samples:     {len(self.indices)}")
        print(f"[EEG REVE DATASET] Subjects filter:   {subjects}")
        print(f"[EEG REVE DATASET] return_image:      {self.return_image}")
        print(f"[EEG REVE DATASET] return_cond_image: {self.return_conditioning_image}")
        if self.image_root is not None:
            print(f"[EEG REVE DATASET] image_root:        {self.image_root}")
        print(f"[EEG REVE DATASET] crop:              {self.crop_start}:{self.crop_end}")
        print(f"[EEG REVE DATASET] resample:          {self.original_fs:g} -> {self.target_fs:g}")
        print(f"[EEG REVE DATASET] pad_to_len:        {self.pad_to_len}")
        print(f"[EEG REVE DATASET] pad_mode:          {self.pad_mode}")

        if build_index_cache_path is not None:
            self.save_index_json(build_index_cache_path)

    def __len__(self):
        return len(self.indices)

    def _pad_or_crop_time(self, x: np.ndarray) -> np.ndarray:
        """
        Pad/crop temporal axis to `pad_to_len`.

        x: [128, T]

        pad_mode:
            - "center": split padding before/after the real signal.
            - "right": append padding after the real signal.
            - "left": prepend padding before the real signal.
        """
        t = x.shape[-1]

        if t == self.pad_to_len:
            return x

        if t > self.pad_to_len:
            if self.pad_mode == "center":
                start = (t - self.pad_to_len) // 2
                end = start + self.pad_to_len
                return x[..., start:end]
            if self.pad_mode == "left":
                return x[..., -self.pad_to_len :]
            return x[..., : self.pad_to_len]

        pad_total = self.pad_to_len - t

        if self.pad_mode == "center":
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
        elif self.pad_mode == "left":
            pad_left = pad_total
            pad_right = 0
        else:
            pad_left = 0
            pad_right = pad_total

        return np.pad(
            x,
            pad_width=((0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        )

    def _preprocess_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Input:
            eeg: [128, L], int16/float

        Output:
            eeg_reve: [128, pad_to_len], float32
        """
        eeg = eeg.float()

        if self.standardize:
            eeg = (eeg - self.means) / self.stddevs

        eeg = eeg[:, self.crop_start : self.crop_end]  # [128, 440]

        eeg_np = eeg.numpy().astype(np.float32)

        eeg_np = resample_poly(
            eeg_np,
            up=1,
            down=self.downsample_factor,
            axis=-1,
        ).astype(np.float32)

        eeg_np = self._pad_or_crop_time(eeg_np).astype(np.float32)

        if eeg_np.shape != (128, self.pad_to_len):
            raise RuntimeError(
                f"Expected preprocessed EEG shape (128,{self.pad_to_len}), got {eeg_np.shape}"
            )

        return torch.from_numpy(eeg_np)

    def _preprocess_gwit_conditioning_image(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        GWIT/CVPR-style EEG view for compatibility.

        This follows the normalization used in GWIT's EEGDatasetCVPR:
            eeg = eeg[:, 20:460]
            norm = max(eeg) / 2
            eeg = (eeg - norm) / norm

        Output:
            Tensor[128, 440]
        """
        eeg = eeg.float()
        eeg = eeg[:, 20:460]

        norm = torch.max(eeg) / 2.0
        eps = torch.tensor(1e-8, dtype=eeg.dtype, device=eeg.device)
        norm = torch.where(torch.abs(norm) < eps, eps, norm)

        eeg = (eeg - norm) / norm

        if tuple(eeg.shape) != (128, 440):
            raise RuntimeError(
                f"Expected GWIT conditioning_image shape (128,440), got {tuple(eeg.shape)}"
            )

        return eeg.float()

    def _build_caption(self, label_idx: int, label_folder: str) -> str:
        if id_to_caption is not None and int(label_idx) in id_to_caption:
            return "image of a " + str(id_to_caption[int(label_idx)])
        return "image of a " + str(label_folder)

    def _find_image_path(self, image_name: str, label_folder: str) -> Path:
        if self.image_root is None:
            raise RuntimeError("image_root is None.")

        candidates = []
        for ext in self.image_extensions:
            candidates.append(self.image_root / label_folder / f"{image_name}{ext}")
            candidates.append(self.image_root / f"{image_name}{ext}")

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Could not find image for image_name={image_name}, label_folder={label_folder}. "
            f"Tried examples: {candidates[:4]}"
        )

    def _load_image(self, image_name: str, label_folder: str) -> torch.Tensor:
        image_path = self._find_image_path(image_name, label_folder)
        image = Image.open(image_path).convert("RGB")
        return self.image_transform(image)

    def _get_generated_caption(self, image_name: str, label_folder: str):
        if self.generated_captions is None:
            return None

        keys = [
            image_name,
            f"{label_folder}/{image_name}",
            f"{label_folder}/{image_name}.JPEG",
            f"{image_name}.JPEG",
        ]

        for key in keys:
            if key in self.generated_captions:
                return self.generated_captions[key]

        return None

    def __getitem__(self, idx: int):
        pth_idx = int(self.indices[idx])
        sample = self.raw_dataset[pth_idx]

        eeg_reve = self._preprocess_eeg(sample["eeg"])

        subject = int(sample["subject"])
        image_idx = int(sample["image"])
        label_idx = int(sample["label"])

        image_name = str(self.images[image_idx])
        label_folder = str(self.labels[label_idx])
        caption = self._build_caption(label_idx, label_folder)

        item = {
            "conditioning_pixel_values": eeg_reve,  # [128, 200], REVE input
            "caption": caption,
            "label_folder": label_folder,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "subject": torch.tensor(subject, dtype=torch.long),
        }

        if self.return_image:
            item["image"] = self._load_image(image_name, label_folder)

        if self.return_conditioning_image:
            item["conditioning_image"] = self._preprocess_gwit_conditioning_image(
                sample["eeg"]
            )

        generated_caption = self._get_generated_caption(image_name, label_folder)
        if generated_caption is not None:
            item["caption_generated"] = generated_caption

        return item

    def save_index_json(self, path: str):
        """
        Save metadata table useful for debugging and later CLIP precompute.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []

        for local_idx, pth_idx in enumerate(self.indices):
            sample = self.raw_dataset[pth_idx]
            image_idx = int(sample["image"])
            label_idx = int(sample["label"])

            image_name = str(self.images[image_idx])
            label_folder = str(self.labels[label_idx])

            rows.append(
                {
                    "local_idx": int(local_idx),
                    "pth_index": int(pth_idx),
                    "subject": int(sample["subject"]),
                    "image_idx": image_idx,
                    "image_name": image_name,
                    "label": label_idx,
                    "label_folder": label_folder,
                    "caption": self._build_caption(label_idx, label_folder),
                    "orig_len": int(sample["eeg"].shape[1]),
                    "crop_start": self.crop_start,
                    "crop_end": self.crop_end,
                    "target_fs": self.target_fs,
                    "pad_to_len": self.pad_to_len,
                    "pad_mode": self.pad_mode,
                }
            )

        with open(path, "w") as f:
            json.dump(rows, f, indent=2)

        print(f"[EEG REVE DATASET] Saved index json: {path}")