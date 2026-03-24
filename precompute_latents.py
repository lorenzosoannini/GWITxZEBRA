import os
import argparse
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from diffusers import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm


# ---------------------------------------------------------
# Image -> tensor in [-1, 1], shape 3x512x512
# ---------------------------------------------------------
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512), antialias=True),
    transforms.Lambda(lambda x: x * 2.0 - 1.0),
])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute SD VAE latents from the full HF pool, saved per subject"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Manojb/stable-diffusion-2-1-base"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6],
        help="Subjects to precompute. Example: --subjects 1 2 3 4 5 6"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=25,
        help="Periodic save every (batch_size * save_every) samples"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/latents"
    )

    return parser.parse_args()


def load_full_hf_pool(dataset_name, cache_dir=None):
    ds_train = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    ds_val = load_dataset(dataset_name, split="validation", cache_dir=cache_dir)
    ds_test = load_dataset(dataset_name, split="test", cache_dir=cache_dir)

    ds_train = ds_train.add_column("__hf_split__", ["train"] * len(ds_train))
    ds_val = ds_val.add_column("__hf_split__", ["validation"] * len(ds_val))
    ds_test = ds_test.add_column("__hf_split__", ["test"] * len(ds_test))

    return concatenate_datasets([ds_train, ds_val, ds_test])


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    subjects = sorted(set(int(s) for s in args.subjects))
    safe_name = args.dataset_name.replace("/", "_")

    # ---------------------------------------------------------
    # LOAD FULL DATASET POOL
    # ---------------------------------------------------------
    print("[INFO] Loading full HF pool: train + validation + test")
    ds_full = load_full_hf_pool(args.dataset_name, cache_dir=args.cache_dir)
    print(f"[INFO] Full pool size: {len(ds_full)} samples")

    # ---------------------------------------------------------
    # LOAD VAE
    # ---------------------------------------------------------
    print("[INFO] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.model_name,
        subfolder="vae"
    ).to(device, dtype=dtype)
    vae.eval()

    # ---------------------------------------------------------
    # PROCESS SUBJECT BY SUBJECT
    # ---------------------------------------------------------
    for subj in subjects:
        print("===================================================")
        print(f"[INFO] Processing subject {subj}")

        # Filter subject from full pool
        ds_subj = ds_full.filter(lambda x: int(x["subject"]) == subj)
        n = len(ds_subj)

        print(f"[INFO] Subject {subj}: {n} samples in full pool")
        if n == 0:
            print(f"[WARNING] Subject {subj} has 0 samples. Skipping.")
            continue

        # Output directory: output_root/<dataset_name>/subjK
        out_dir = os.path.join(
            args.output_root,
            safe_name,
            f"subj{subj}"
        )
        os.makedirs(out_dir, exist_ok=True)

        mean_path = os.path.join(out_dir, "posterior_mean.npy")
        logv_path = os.path.join(out_dir, "posterior_logvar.npy")

        # ---------------------------------------------------------
        # RESUME IF POSSIBLE
        # ---------------------------------------------------------
        posterior_means = []
        posterior_logvars = []
        start_idx = 0

        if os.path.exists(mean_path) and os.path.exists(logv_path):
            existing_means = np.load(mean_path)
            existing_logvars = np.load(logv_path)

            posterior_means = list(existing_means)
            posterior_logvars = list(existing_logvars)
            start_idx = existing_means.shape[0]

            if start_idx > n:
                raise RuntimeError(
                    f"Existing latents for subj{subj} have more samples than current dataset: "
                    f"{start_idx} > {n}"
                )

            print(f"[INFO] Resuming subj{subj} from index {start_idx}/{n}")

        # ---------------------------------------------------------
        # MAIN LOOP
        # ---------------------------------------------------------
        print(f"[INFO] Starting latent computation for subj{subj} with batch_size={args.batch_size}")

        for i in tqdm(range(start_idx, n, args.batch_size), mininterval=1, desc=f"subj{subj}"):
            batch = ds_subj[i: i + args.batch_size]
            images = batch["image"]

            imgs = [to_tensor(np.array(img)) for img in images]
            imgs = torch.stack(imgs).to(device, dtype=dtype)

            with torch.no_grad():
                posterior = vae.encode(imgs).latent_dist
                means = posterior.mean.cpu().numpy()
                logv = posterior.logvar.cpu().numpy()

            posterior_means.extend(list(means))
            posterior_logvars.extend(list(logv))

            # periodic save
            if len(posterior_means) % (args.batch_size * args.save_every) == 0:
                np.save(mean_path, np.array(posterior_means))
                np.save(logv_path, np.array(posterior_logvars))
                print(f"[CHECKPOINT] subj{subj}: saved {len(posterior_means)} / {n}")

        # ---------------------------------------------------------
        # FINAL SAVE
        # ---------------------------------------------------------
        posterior_means = np.array(posterior_means)
        posterior_logvars = np.array(posterior_logvars)

        assert posterior_means.shape[0] == n, (
            f"Final mean count mismatch for subj{subj}: "
            f"{posterior_means.shape[0]} vs expected {n}"
        )
        assert posterior_logvars.shape[0] == n, (
            f"Final logvar count mismatch for subj{subj}: "
            f"{posterior_logvars.shape[0]} vs expected {n}"
        )

        np.save(mean_path, posterior_means)
        np.save(logv_path, posterior_logvars)

        print(f"[DONE] subj{subj}")
        print(f"       mean:  {mean_path}")
        print(f"       logvar:{logv_path}")
        print(f"       shape mean:   {posterior_means.shape}")
        print(f"       shape logvar: {posterior_logvars.shape}")

    print("===================================================")
    print("✔ Latent precomputation completed for all requested subjects.")


if __name__ == "__main__":
    main()