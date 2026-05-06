import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from torchvision import transforms
from tqdm import tqdm

from models.openclip_zebra import FrozenOpenCLIPImageEmbedder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute ZEBRA-like OpenCLIP image token embeddings for EEG dataset."
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_root", type=str, required=True)

    parser.add_argument(
        "--arch",
        type=str,
        default="ViT-bigG-14",
        help="OpenCLIP visual backbone used by ZEBRA.",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s39b_b160k",
        help="OpenCLIP pretrained weights used by ZEBRA.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu. If omitted, auto-detected.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Flush memmap and save progress every N batches.",
    )
    return parser.parse_args()


def _safe_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace("/", "_")


def load_full_hf_pool(dataset_name, cache_dir=None):
    ds_train = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    ds_val = load_dataset(dataset_name, split="validation", cache_dir=cache_dir)
    ds_test = load_dataset(dataset_name, split="test", cache_dir=cache_dir)

    ds_train = ds_train.add_column("__hf_split__", ["train"] * len(ds_train))
    ds_val = ds_val.add_column("__hf_split__", ["validation"] * len(ds_val))
    ds_test = ds_test.add_column("__hf_split__", ["test"] * len(ds_test))

    return concatenate_datasets([ds_train, ds_val, ds_test])


def build_subject_index_map(full_data, subjects):
    subject_set = set(int(s) for s in subjects)
    subject_to_indices = {int(s): [] for s in subjects}

    subject_col = full_data["subject"]
    for idx, subj in enumerate(subject_col):
        subj = int(subj)
        if subj in subject_set:
            subject_to_indices[subj].append(idx)

    return subject_to_indices


def to_zebra_tensor(img):
    """
    Convert image exactly like eeg_dataset.py:
    ToTensor -> Resize(512,512) -> map to [-1, 1]
    """
    if hasattr(img, "convert"):
        pil_img = img.convert("RGB")
    else:
        from PIL import Image
        pil_img = Image.fromarray(np.array(img)).convert("RGB")

    tensor = transforms.ToTensor()(pil_img)   # [0, 1]
    tensor = transforms.Resize((512, 512), antialias=True)(tensor)
    tensor = tensor * 2.0 - 1.0               # [-1, 1]
    return tensor


def save_numpy_safe(path: str, array: np.ndarray):
    path = Path(path)
    tmp_path = path.parent / f"{path.stem}.tmp.npy"
    np.save(tmp_path, array)
    os.replace(tmp_path, path)


def save_json_safe(path: str, payload: dict):
    path = Path(path)
    tmp_path = path.parent / f"{path.stem}.tmp.json"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def open_or_create_memmap(memmap_path: str, shape, mode="r+"):
    return np.memmap(memmap_path, dtype=np.float32, mode=mode, shape=shape)


def main():
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] device = {device}")
    print(f"[INFO] Loading full HF pool for {args.dataset_name} ...")
    full_ds = load_full_hf_pool(args.dataset_name, cache_dir=args.cache_dir)
    print(f"[INFO] Total samples in full HF pool: {len(full_ds)}")

    subject_to_indices = build_subject_index_map(full_ds, args.subjects)

    safe_name = _safe_dataset_name(args.dataset_name)
    base_out_dir = os.path.join(args.output_root, safe_name)
    os.makedirs(base_out_dir, exist_ok=True)

    print("[INFO] Loading FrozenOpenCLIPImageEmbedder...")
    embedder = FrozenOpenCLIPImageEmbedder(
        arch=args.arch,
        version=args.pretrained,
        device=device,
        init_device=device,
        freeze=True,
        output_tokens=True,
        only_tokens=True,
        cache_dir=args.cache_dir,
    )
    embedder = embedder.to(device)
    embedder.eval()

    # ---------------------------------------------------------
    # Sanity check on embedder output shape
    # ---------------------------------------------------------
    with torch.no_grad():
        dummy = torch.empty(2, 3, 512, 512, device=device, dtype=torch.float32).uniform_(-1.0, 1.0)
        dummy_tokens = embedder(dummy)

    if dummy_tokens.ndim != 3:
        raise RuntimeError(
            f"FrozenOpenCLIPImageEmbedder must return sequence-level tokens (B, T, D), "
            f"but got shape {tuple(dummy_tokens.shape)}"
        )

    token_count = int(dummy_tokens.shape[1])
    embed_dim = int(dummy_tokens.shape[2])

    print(f"[INFO] Dummy embedder output shape: {tuple(dummy_tokens.shape)}")
    print(f"[INFO] Dummy token count T = {token_count}")
    print(f"[INFO] Dummy embed dim   D = {embed_dim}")

    for subj in args.subjects:
        subj = int(subj)
        subj_indices = subject_to_indices.get(subj, [])
        n_subj = len(subj_indices)

        if n_subj == 0:
            print(f"[WARN] subj{subj}: empty, skipping")
            continue

        subj_dir = os.path.join(base_out_dir, f"subj{subj}")
        os.makedirs(subj_dir, exist_ok=True)

        out_path = os.path.join(subj_dir, "clip_img_embeds.npy")
        meta_path = os.path.join(subj_dir, "meta.json")
        progress_path = os.path.join(subj_dir, "progress.json")
        memmap_path = os.path.join(subj_dir, "clip_img_embeds.memmap")

        print(f"[INFO] subj{subj}: {n_subj} samples")

        final_shape = (n_subj, token_count, embed_dim)
        start_idx = 0

        # ---------------------------------------------
        # Fast path: already completed final .npy exists
        # ---------------------------------------------
        if os.path.exists(out_path):
            existing = np.load(out_path, mmap_mode="r")
            if tuple(existing.shape) != final_shape:
                raise RuntimeError(
                    f"Existing final file {out_path} has shape {tuple(existing.shape)}, "
                    f"expected {final_shape}"
                )
            print(f"[INFO] subj{subj}: final file already exists, skipping")
            continue

        # ---------------------------------------------
        # Resume from progress if available
        # ---------------------------------------------
        if os.path.exists(progress_path):
            with open(progress_path, "r") as f:
                progress = json.load(f)

            saved_shape = tuple(progress["shape"])
            if saved_shape != final_shape:
                raise RuntimeError(
                    f"Progress shape mismatch for subj{subj}: saved {saved_shape}, expected {final_shape}"
                )

            start_idx = int(progress["written_until"])
            print(
                f"[INFO] subj{subj}: resume from {start_idx}/{n_subj} "
                f"using memmap {memmap_path}"
            )

            if not os.path.exists(memmap_path):
                raise RuntimeError(
                    f"Missing memmap file for resume: {memmap_path}"
                )

            mmap = open_or_create_memmap(memmap_path, final_shape, mode="r+")
        else:
            print(f"[INFO] subj{subj}: creating new memmap with shape {final_shape}")
            mmap = open_or_create_memmap(memmap_path, final_shape, mode="w+")
            mmap.flush()

            save_json_safe(
                progress_path,
                {
                    "dataset_name": args.dataset_name,
                    "subject": subj,
                    "shape": list(final_shape),
                    "written_until": 0,
                    "arch": args.arch,
                    "pretrained": args.pretrained,
                },
            )

        batch_counter = 0

        for i in tqdm(range(start_idx, n_subj, args.batch_size), desc=f"subj{subj}"):
            batch_local_idx = list(range(i, min(i + args.batch_size, n_subj)))
            batch_global_idx = [subj_indices[j] for j in batch_local_idx]

            batch = full_ds.select(batch_global_idx)
            images = batch[args.image_column]

            img_tensors = [to_zebra_tensor(img) for img in images]
            pixel_values = torch.stack(img_tensors, dim=0).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                tokens = embedder(pixel_values)

            if tokens.ndim != 3:
                raise RuntimeError(
                    f"Expected sequence-level CLIP tokens with shape (B, T, D), "
                    f"got {tuple(tokens.shape)} for subj{subj}, batch starting at local idx {i}"
                )

            expected_bsz = len(batch_local_idx)
            if tokens.shape[0] != expected_bsz:
                raise RuntimeError(
                    f"Batch size mismatch for subj{subj}: expected {expected_bsz} outputs, "
                    f"got {tokens.shape[0]}"
                )

            if tokens.shape[1] != token_count or tokens.shape[2] != embed_dim:
                raise RuntimeError(
                    f"Token shape mismatch for subj{subj}: got {tuple(tokens.shape)}, "
                    f"expected (B, {token_count}, {embed_dim})"
                )

            if i == start_idx:
                print(
                    f"[INFO] subj{subj} first batch token shape: {tuple(tokens.shape)} "
                    f"(B={tokens.shape[0]}, T={tokens.shape[1]}, D={tokens.shape[2]})"
                )

            tokens_np = tokens.detach().cpu().float().numpy().astype(np.float32)

            mmap[i : i + expected_bsz] = tokens_np

            batch_counter += 1
            if batch_counter % args.save_every == 0:
                mmap.flush()
                save_json_safe(
                    progress_path,
                    {
                        "dataset_name": args.dataset_name,
                        "subject": subj,
                        "shape": list(final_shape),
                        "written_until": i + expected_bsz,
                        "arch": args.arch,
                        "pretrained": args.pretrained,
                    },
                )
                print(
                    f"[INFO] subj{subj}: flushed memmap at {i + expected_bsz}/{n_subj}"
                )

        mmap.flush()
        del mmap

        # ---------------------------------------------
        # Convert memmap -> final npy once at the end
        # ---------------------------------------------
        print(f"[INFO] subj{subj}: converting memmap to final .npy ...")
        mmap_read = open_or_create_memmap(memmap_path, final_shape, mode="r")
        final_embs = np.asarray(mmap_read, dtype=np.float32)
        save_numpy_safe(out_path, final_embs)
        del mmap_read

        meta = {
            "dataset_name": args.dataset_name,
            "subject": subj,
            "num_samples": int(n_subj),
            "embedding_shape": list(final_shape),
            "arch": args.arch,
            "pretrained": args.pretrained,
            "sequence_level": True,
            "producer": "FrozenOpenCLIPImageEmbedder",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # progress no longer needed once final npy exists
        if os.path.exists(progress_path):
            os.remove(progress_path)

        print(f"[DONE] subj{subj} -> shape={final_shape}")
        print(f"[DONE] saved: {out_path}")

    print("=================================")
    print("✔ OpenCLIP ZEBRA-style precompute DONE")
    print("=================================")


if __name__ == "__main__":
    main()