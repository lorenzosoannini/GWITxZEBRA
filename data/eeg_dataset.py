import os
import torch
from torch.utils.data import Dataset
import numpy as np
from datasets import load_dataset, concatenate_datasets
from torchvision import transforms

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ---------------------------------------------------------
# Stable Diffusion normalization for real images
# ---------------------------------------------------------
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512), antialias=True),
    transforms.Lambda(lambda x: x * 2.0 - 1.0),
])


def _normalize_subject_list(subjects):
    if subjects is None:
        return None
    subjects = [int(s) for s in subjects]
    return sorted(set(subjects))


def _safe_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace("/", "_")


def _split_indices_deterministic(n, val_ratio=0.1, seed=42):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    if n <= 1:
        return idx.tolist(), []

    n_val = int(round(n * val_ratio))
    n_val = min(max(n_val, 1), n - 1)

    val_idx = np.sort(idx[:n_val]).tolist()
    train_idx = np.sort(idx[n_val:]).tolist()
    return train_idx, val_idx


def _limit_indices_per_subject(indices, max_samples_per_subject=None):
    """
    Deterministically keeps only the first max_samples_per_subject indices.
    """
    if max_samples_per_subject is None:
        return indices
    return indices[: min(len(indices), int(max_samples_per_subject))]


def _load_full_hf_pool(dataset_name, cache_dir=None):
    ds_train = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    ds_val = load_dataset(dataset_name, split="validation", cache_dir=cache_dir)
    ds_test = load_dataset(dataset_name, split="test", cache_dir=cache_dir)

    ds_train = ds_train.add_column("__hf_split__", ["train"] * len(ds_train))
    ds_val = ds_val.add_column("__hf_split__", ["validation"] * len(ds_val))
    ds_test = ds_test.add_column("__hf_split__", ["test"] * len(ds_test))

    full_ds = concatenate_datasets([ds_train, ds_val, ds_test])
    return full_ds


def _build_subject_index_map(full_data, subjects):
    """
    Build a mapping: subject_id -> list of global indices in full_data.
    Single pass over subject column, avoids repeated dataset.filter calls.
    """
    subject_set = set(int(s) for s in subjects)
    subject_to_indices = {int(s): [] for s in subjects}

    subject_col = full_data["subject"]
    for idx, subj in enumerate(subject_col):
        subj = int(subj)
        if subj in subject_set:
            subject_to_indices[subj].append(idx)

    return subject_to_indices


def _resolve_subject_subdir(root_dir, dataset_name, subj):
    """
    Supports both:
      1) root_dir/subjK/...
      2) root_dir/<safe_dataset_name>/subjK/...
    """
    safe_name = _safe_dataset_name(dataset_name)

    candidate_1 = os.path.join(root_dir, f"subj{subj}")
    candidate_2 = os.path.join(root_dir, safe_name, f"subj{subj}")

    if os.path.isdir(candidate_1):
        return candidate_1
    if os.path.isdir(candidate_2):
        return candidate_2

    raise FileNotFoundError(
        f"Could not find subject directory for subj{subj} under either:\n"
        f"  {candidate_1}\n"
        f"  {candidate_2}"
    )


class EEGImageDataset(Dataset):
    """
    Subject-aware GWIT/ZEBRA dataset loader.

    Pool source:
        - all HF splits combined: train + validation + test

    subset_mode:
        - "train": train subset of seen subjects
        - "val":   validation subset of seen subjects
        - "test":  all samples of unseen/test subjects

    Latents (optional):
        Expected structure:
            latents_dir/
                subj1/
                    posterior_mean.npy
                    posterior_logvar.npy
            or
            latents_dir/<safe_dataset_name>/subj1/...

    CLIP embeds (optional):
        Expected structure:
            clip_embeds_dir/
                subj1/
                    clip_img_embeds.npy
                    clip_text_embeds.npy
            or
            clip_embeds_dir/<safe_dataset_name>/subj1/...

        Supported image shapes:
            pooled:   (N, D)
            sequence: (N, T, D)

        Supported text shapes:
            pooled:   (N, D)
    """

    def __init__(
        self,
        dataset_name,
        subjects,
        subset_mode,
        image_column,
        conditioning_image_column,
        caption_column,
        tokenizer,
        args,
        root,
        accelerator,
        use_precomputed_latents: bool = False,
        latents_dir: str = None,
        use_precomputed_clip_embeds: bool = False,
        clip_embeds_dir: str = None,
        val_ratio: float = 0.1,
        split_seed: int = 42,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.subjects = _normalize_subject_list(subjects)
        self.subset_mode = subset_mode
        self.image_column = image_column
        self.cond_column = conditioning_image_column
        self.caption_column = caption_column
        self.tokenizer = tokenizer
        self.args = args
        self.root = root
        self.accelerator = accelerator

        self.use_precomputed_latents = bool(use_precomputed_latents)
        self.latents_dir = latents_dir

        self.use_precomputed_clip_embeds = bool(use_precomputed_clip_embeds)
        self.clip_embeds_dir = clip_embeds_dir
        self.clip_embed_rank = None  # 2 for pooled, 3 for sequence
        self.clip_text_embed_dim = None

        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)

        if self.subset_mode == "train":
            self.max_samples_per_subject = getattr(args, "max_train_samples_per_subject", None)
        elif self.subset_mode == "val":
            self.max_samples_per_subject = getattr(args, "max_val_samples_per_subject", None)
        else:
            self.max_samples_per_subject = getattr(args, "max_test_samples_per_subject", None)

        assert self.subjects is not None and len(self.subjects) > 0, \
            "subjects must be a non-empty list"
        assert self.subset_mode in {"train", "val", "test"}, \
            f"Invalid subset_mode={self.subset_mode}"

        # ---------------------------------------------------------
        # Captioner
        # ---------------------------------------------------------
        if getattr(args, "caption_from_classifier", False):
            from .caption_classifier import EEGCaptionClassifier

            if accelerator is not None:
                captioner_device = accelerator.device
            else:
                captioner_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.captioner = EEGCaptionClassifier(
                dataset_name=dataset_name,
                root=root,
                device=str(captioner_device),
            )
        else:
            self.captioner = None

        # ---------------------------------------------------------
        # Load full HF dataset pool
        # ---------------------------------------------------------
        full_data = _load_full_hf_pool(
            dataset_name=dataset_name,
            cache_dir=args.cache_dir,
        )
        print(f"[EEG DATASET] Loaded full HF pool: {len(full_data)} samples")

        # ---------------------------------------------------------
        # Filter selected subjects from full pool
        # ---------------------------------------------------------
        subject_to_indices = _build_subject_index_map(full_data, self.subjects)
        n_after_subject_filter = sum(len(v) for v in subject_to_indices.values())
        print(
            f"[EEG DATASET] After subject index map {self.subjects}: {n_after_subject_filter} samples"
        )

        # ---------------------------------------------------------
        # Optional containers
        # ---------------------------------------------------------
        self.latents_by_subject = None
        if self.use_precomputed_latents:
            assert self.latents_dir is not None, \
                "You must set latents_dir if use_precomputed_latents=True"
            self.latents_by_subject = {}

        self.clip_embeds_by_subject = None
        self.clip_text_embeds_by_subject = None
        if self.use_precomputed_clip_embeds:
            assert self.clip_embeds_dir is not None, \
                "You must set clip_embeds_dir if use_precomputed_clip_embeds=True"
            self.clip_embeds_by_subject = {}
            self.clip_text_embeds_by_subject = {}

        # ---------------------------------------------------------
        # Build final dataset subject by subject
        # ---------------------------------------------------------
        per_subject_datasets = []
        self.sample_index_within_subject = []

        for subj in self.subjects:
            subj_global_indices = subject_to_indices.get(subj, [])
            n_subj = len(subj_global_indices)

            if n_subj == 0:
                print(f"[EEG DATASET] Warning: subject {subj} has 0 samples")
                continue

            # split on local indices [0..n_subj-1]
            if self.subset_mode in {"train", "val"}:
                train_idx, val_idx = _split_indices_deterministic(
                    n=n_subj,
                    val_ratio=self.val_ratio,
                    seed=self.split_seed + subj,
                )
                keep_local_idx = train_idx if self.subset_mode == "train" else val_idx
            else:
                keep_local_idx = list(range(n_subj))

            # optional per-subject cap
            keep_local_idx = _limit_indices_per_subject(
                keep_local_idx,
                max_samples_per_subject=self.max_samples_per_subject,
            )

            # map local subject indices -> global full_data indices
            keep_global_idx = [subj_global_indices[i] for i in keep_local_idx]

            ds_subj_final = full_data.select(keep_global_idx)
            local_indices = keep_local_idx

            n_kept = len(ds_subj_final)
            print(
                f"[EEG DATASET] Subject {subj} | mode={self.subset_mode} | "
                f"kept {n_kept}/{n_subj} | cap={self.max_samples_per_subject}"
            )

            if n_kept == 0:
                continue

            # ---------------------------------------------
            # Load subject latents
            # ---------------------------------------------
            if self.use_precomputed_latents:
                subj_dir = _resolve_subject_subdir(self.latents_dir, self.dataset_name, subj)
                mean_path = os.path.join(subj_dir, "posterior_mean.npy")
                logv_path = os.path.join(subj_dir, "posterior_logvar.npy")

                assert os.path.exists(mean_path), f"Missing {mean_path}"
                assert os.path.exists(logv_path), f"Missing {logv_path}"

                means = np.load(mean_path, mmap_mode="r")
                logvars = np.load(logv_path, mmap_mode="r")

                assert len(means) == n_subj, (
                    f"Latents length mismatch for subj{subj}: "
                    f"{len(means)} vs full-pool subject samples {n_subj}"
                )
                assert len(logvars) == n_subj, (
                    f"Latents length mismatch for subj{subj}: "
                    f"{len(logvars)} vs full-pool subject samples {n_subj}"
                )

                self.latents_by_subject[subj] = {
                    "mean": means,
                    "logvar": logvars,
                }

            # ---------------------------------------------
            # Load subject CLIP image/text embeds
            # ---------------------------------------------
            if self.use_precomputed_clip_embeds:
                subj_dir = _resolve_subject_subdir(self.clip_embeds_dir, self.dataset_name, subj)

                clip_img_path = os.path.join(subj_dir, "clip_img_embeds.npy")
                clip_text_path = os.path.join(subj_dir, "clip_text_embeds.npy")

                assert os.path.exists(clip_img_path), f"Missing {clip_img_path}"
                assert os.path.exists(clip_text_path), f"Missing {clip_text_path}"

                clip_embeds = np.load(clip_img_path, mmap_mode="r")
                clip_text_embeds = np.load(clip_text_path, mmap_mode="r")

                assert len(clip_embeds) == n_subj, (
                    f"CLIP image embeds length mismatch for subj{subj}: "
                    f"{len(clip_embeds)} vs full-pool subject samples {n_subj}"
                )

                if clip_embeds.ndim not in (2, 3):
                    raise ValueError(
                        f"Expected CLIP image embeds with shape (N, D) or (N, T, D) for subj{subj}, "
                        f"got shape {clip_embeds.shape}"
                    )

                assert len(clip_text_embeds) == n_subj, (
                    f"CLIP text embeds length mismatch for subj{subj}: "
                    f"{len(clip_text_embeds)} vs full-pool subject samples {n_subj}"
                )

                if clip_text_embeds.ndim != 2:
                    raise ValueError(
                        f"Expected CLIP text embeds with shape (N, D) for subj{subj}, "
                        f"got shape {clip_text_embeds.shape}"
                    )

                if self.clip_embed_rank is None:
                    self.clip_embed_rank = clip_embeds.ndim
                else:
                    assert self.clip_embed_rank == clip_embeds.ndim, (
                        f"Inconsistent CLIP image embed rank across subjects: "
                        f"previous rank={self.clip_embed_rank}, subj{subj} rank={clip_embeds.ndim}"
                    )

                if self.clip_text_embed_dim is None:
                    self.clip_text_embed_dim = int(clip_text_embeds.shape[1])
                else:
                    assert self.clip_text_embed_dim == int(clip_text_embeds.shape[1]), (
                        f"Inconsistent CLIP text embed dim across subjects: "
                        f"previous dim={self.clip_text_embed_dim}, "
                        f"subj{subj} dim={clip_text_embeds.shape[1]}"
                    )

                self.clip_embeds_by_subject[subj] = clip_embeds
                self.clip_text_embeds_by_subject[subj] = clip_text_embeds

                print(
                    f"[EEG DATASET] Loaded CLIP image embeds for subj{subj}: shape={clip_embeds.shape} "
                    f"| mode={'sequence' if clip_embeds.ndim == 3 else 'pooled'}"
                )
                print(
                    f"[EEG DATASET] Loaded CLIP text embeds for subj{subj}: shape={clip_text_embeds.shape}"
                )

            # ---------------------------------------------
            # Save mapping final idx -> (subject, local_idx)
            # ---------------------------------------------
            for li in local_indices:
                self.sample_index_within_subject.append((subj, li))

            per_subject_datasets.append(ds_subj_final)

        assert len(per_subject_datasets) > 0, (
            f"No samples left after filtering. subjects={self.subjects}, mode={self.subset_mode}"
        )

        self.data = (
            per_subject_datasets[0]
            if len(per_subject_datasets) == 1
            else concatenate_datasets(per_subject_datasets)
        )

        assert len(self.data) == len(self.sample_index_within_subject), (
            f"Length mismatch: data={len(self.data)} vs mapping={len(self.sample_index_within_subject)}"
        )

        print(
            f"[EEG DATASET] Final dataset | mode={self.subset_mode} | "
            f"subjects={self.subjects} | total={len(self.data)}"
        )

        if self.use_precomputed_clip_embeds:
            print(
                f"[EEG DATASET] CLIP image embedding mode: "
                f"{'sequence-level' if self.clip_embed_rank == 3 else 'pooled'}"
            )
            print(
                f"[EEG DATASET] CLIP text embedding dim: {self.clip_text_embed_dim}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        subj, local_idx = self.sample_index_within_subject[idx]

        # ---------------------------------------------------------
        # IMAGE or LATENTS
        # ---------------------------------------------------------
        if self.use_precomputed_latents:
            posterior_mean = torch.from_numpy(
                np.asarray(self.latents_by_subject[subj]["mean"][local_idx]).copy()
            ).float()

            posterior_logvar = torch.from_numpy(
                np.asarray(self.latents_by_subject[subj]["logvar"][local_idx]).copy()
            ).float()

            img = None
        else:
            img = to_tensor(np.array(example[self.image_column]))
            posterior_mean = None
            posterior_logvar = None

        # ---------------------------------------------------------
        # Optional CLIP image embedding
        # ---------------------------------------------------------
        clip_img_embeds = None
        if self.use_precomputed_clip_embeds:
            clip_img_embeds = torch.from_numpy(
                np.asarray(self.clip_embeds_by_subject[subj][local_idx]).copy()
            ).float()

            # IMPORTANTE:
            # per il prior e per unCLIP servono i raw CLIP image tokens,
            # quindi qui NON normalizziamo.
            # pooled  -> (D)
            # seq-lvl -> (T, D)

        # ---------------------------------------------------------
        # Optional CLIP text embedding
        # ---------------------------------------------------------
        clip_text_embeds = None
        if self.use_precomputed_clip_embeds:
            clip_text_embeds = torch.from_numpy(
                np.asarray(self.clip_text_embeds_by_subject[subj][local_idx]).copy()
            ).float()

        # ---------------------------------------------------------
        # EEG conditioning map (C,T)
        # ---------------------------------------------------------
        cond_arr = np.array(example[self.cond_column], dtype=np.float32)
        cond = torch.from_numpy(cond_arr).float()

        # ---------------------------------------------------------
        # Subject ID
        # ---------------------------------------------------------
        subject = torch.tensor(int(example["subject"]), dtype=torch.long)

        # ---------------------------------------------------------
        # Image label
        # ---------------------------------------------------------
        image_label = torch.tensor(int(example["label"]), dtype=torch.long)

        # ---------------------------------------------------------
        # Caption logic
        # ---------------------------------------------------------
        if self.captioner is not None:
            if "CVPR" in self.dataset_name.upper():
                eeg_for_caption = cond_arr
            else:
                eeg_for_caption = np.array(example["eeg_no_resample"], dtype=np.float32)
            caption = self.captioner.predict_captions([eeg_for_caption])[0]
        else:
            caption = example[self.caption_column]

        # ---------------------------------------------------------
        # Tokenize
        # ---------------------------------------------------------
        tokens = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        out = {
            "conditioning_pixel_values": cond,
            "input_ids": tokens.input_ids.squeeze(0),
            "eeg_subjects": subject,
            "image_labels": image_label,
            "caption_text": caption,
        }

        if self.use_precomputed_latents:
            out["posterior_mean"] = posterior_mean
            out["posterior_logvar"] = posterior_logvar
        else:
            out["pixel_values"] = img

        if self.use_precomputed_clip_embeds:
            out["clip_img_embeds"] = clip_img_embeds
            out["clip_text_embeds"] = clip_text_embeds

        return out


# ---------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------
def _maybe_subsample_dataset(dataset, max_samples):
    if max_samples is None:
        return dataset

    original_len = len(dataset)
    keep = min(max_samples, original_len)

    dataset.data = dataset.data.select(range(keep))
    if hasattr(dataset, "sample_index_within_subject"):
        dataset.sample_index_within_subject = dataset.sample_index_within_subject[:keep]

    print(f"[EEG DATASET] max_samples={max_samples}: {original_len} → {len(dataset)}")
    return dataset


def make_train_dataset(args, tokenizer, accelerator):
    dataset = EEGImageDataset(
        dataset_name=args.dataset_name,
        subjects=args.train_subjects,
        subset_mode="train",
        image_column=args.image_column,
        conditioning_image_column=args.conditioning_image_column,
        caption_column=args.caption_column,
        tokenizer=tokenizer,
        args=args,
        root=args.data_root,
        accelerator=accelerator,
        use_precomputed_latents=getattr(args, "use_precomputed_latents", False),
        latents_dir=getattr(args, "latents_dir", None),
        use_precomputed_clip_embeds=getattr(args, "use_precomputed_clip_embeds", False),
        clip_embeds_dir=getattr(args, "clip_embeds_dir", None),
        val_ratio=getattr(args, "val_ratio", 0.1),
        split_seed=getattr(args, "split_seed", 42),
    )
    return _maybe_subsample_dataset(dataset, getattr(args, "max_train_samples", None))


def make_val_dataset(args, tokenizer, accelerator):
    dataset = EEGImageDataset(
        dataset_name=args.dataset_name,
        subjects=args.val_subjects,
        subset_mode="val",
        image_column=args.image_column,
        conditioning_image_column=args.conditioning_image_column,
        caption_column=args.caption_column,
        tokenizer=tokenizer,
        args=args,
        root=args.data_root,
        accelerator=accelerator,
        use_precomputed_latents=getattr(args, "use_precomputed_latents", False),
        latents_dir=getattr(args, "latents_dir", None),
        use_precomputed_clip_embeds=getattr(args, "use_precomputed_clip_embeds", False),
        clip_embeds_dir=getattr(args, "clip_embeds_dir", None),
        val_ratio=getattr(args, "val_ratio", 0.1),
        split_seed=getattr(args, "split_seed", 42),
    )
    return dataset


def make_test_dataset(args, tokenizer, accelerator):
    dataset = EEGImageDataset(
        dataset_name=args.dataset_name,
        subjects=args.test_subjects,
        subset_mode="test",
        image_column=args.image_column,
        conditioning_image_column=args.conditioning_image_column,
        caption_column=args.caption_column,
        tokenizer=tokenizer,
        args=args,
        root=args.data_root,
        accelerator=accelerator,
        use_precomputed_latents=getattr(args, "use_precomputed_latents", False),
        latents_dir=getattr(args, "latents_dir", None),
        use_precomputed_clip_embeds=getattr(args, "use_precomputed_clip_embeds", False),
        clip_embeds_dir=getattr(args, "clip_embeds_dir", None),
        val_ratio=getattr(args, "val_ratio", 0.1),
        split_seed=getattr(args, "split_seed", 42),
    )
    return dataset


# ---------------------------------------------------------
# Safe collate_fn
# ---------------------------------------------------------
def make_collate_fn(dataset_name):
    def collate(batch):
        out = {}
        for key in batch[0]:
            values = [b[key] for b in batch]
            if isinstance(values[0], torch.Tensor):
                out[key] = torch.stack(values)
            else:
                out[key] = values
        return out
    return collate