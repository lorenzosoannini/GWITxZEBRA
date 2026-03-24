import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute CLIP image embeddings for full EEG dataset pool, saved subject-wise."
    )
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", required=True,
                        help="Subjects to process, e.g. 1 2 3 4 5 6")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_root", type=str, required=True)
    return parser.parse_args()


def load_full_hf_pool(dataset_name, cache_dir=None):
    ds_train = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    ds_val = load_dataset(dataset_name, split="validation", cache_dir=cache_dir)
    ds_test = load_dataset(dataset_name, split="test", cache_dir=cache_dir)

    ds_train = ds_train.add_column("__hf_split__", ["train"] * len(ds_train))
    ds_val = ds_val.add_column("__hf_split__", ["validation"] * len(ds_val))
    ds_test = ds_test.add_column("__hf_split__", ["test"] * len(ds_test))

    full_ds = concatenate_datasets([ds_train, ds_val, ds_test])
    return full_ds


def build_subject_index_map(full_data, subjects):
    subject_set = set(int(s) for s in subjects)
    subject_to_indices = {int(s): [] for s in subjects}

    subject_col = full_data["subject"]
    for idx, subj in enumerate(subject_col):
        subj = int(subj)
        if subj in subject_set:
            subject_to_indices[subj].append(idx)

    return subject_to_indices


def main():
    args = parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Loading CLIP processor/model...")
    processor = CLIPImageProcessor.from_pretrained(args.clip_model_name)
    model = CLIPVisionModelWithProjection.from_pretrained(args.clip_model_name).to(device)
    model.eval()

    print("[INFO] Loading full HF pool...")
    full_ds = load_full_hf_pool(args.dataset_name, cache_dir=args.cache_dir)
    print(f"[INFO] Full pool size: {len(full_ds)}")

    subject_to_indices = build_subject_index_map(full_ds, args.subjects)

    safe_name = args.dataset_name.replace("/", "_")
    base_out_dir = os.path.join(args.output_root, safe_name)
    os.makedirs(base_out_dir, exist_ok=True)

    for subj in args.subjects:
        subj = int(subj)
        subj_indices = subject_to_indices.get(subj, [])
        n_subj = len(subj_indices)

        if n_subj == 0:
            print(f"[WARN] Subject {subj} has 0 samples, skipping.")
            continue

        print(f"[INFO] Subject {subj}: {n_subj} samples")

        subj_dir = os.path.join(base_out_dir, f"subj{subj}")
        os.makedirs(subj_dir, exist_ok=True)

        out_path = os.path.join(subj_dir, "clip_img_embeds.npy")

        # Resume support
        all_embs = []
        start_idx = 0

        if os.path.exists(out_path):
            existing = np.load(out_path)
            all_embs = [x for x in existing]
            start_idx = len(existing)
            print(f"[INFO] Resuming subj{subj} from {start_idx}/{n_subj}")

        for i in tqdm(range(start_idx, n_subj, args.batch_size), desc=f"subj{subj}", mininterval=1):
            batch_local_idx = list(range(i, min(i + args.batch_size, n_subj)))
            batch_global_idx = [subj_indices[j] for j in batch_local_idx]

            batch = full_ds.select(batch_global_idx)
            images = batch[args.image_column]

            # HF image feature is usually already PIL-like; enforce RGB
            pil_images = []
            for img in images:
                if hasattr(img, "convert"):
                    pil_images.append(img.convert("RGB"))
                else:
                    # fallback if needed
                    from PIL import Image
                    pil_images.append(Image.fromarray(np.array(img)).convert("RGB"))

            inputs = processor(images=pil_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)

            with torch.no_grad():
                out = model(pixel_values=pixel_values)
                emb = out.image_embeds
                emb = F.normalize(emb, dim=-1)

            emb_np = emb.detach().cpu().float().numpy().astype(np.float32)
            all_embs.extend(list(emb_np))

            # Save progressively
            np.save(out_path, np.array(all_embs, dtype=np.float32))

        final_embs = np.array(all_embs, dtype=np.float32)
        np.save(out_path, final_embs)

        print(f"[DONE] subj{subj} -> {out_path} | shape={final_embs.shape}")

    print("===========================================")
    print("✔ CLIP embedding precomputation completed.")
    print(f"✔ Output root: {base_out_dir}")
    print("===========================================")


if __name__ == "__main__":
    main()