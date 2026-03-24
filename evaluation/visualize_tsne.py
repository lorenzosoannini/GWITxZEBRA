#!/usr/bin/env python
# coding=utf-8

import os
os.environ.pop("MPLBACKEND", None)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.manifold import TSNE
from collections import Counter


class EEGAlignEncoderMean(nn.Module):
    def __init__(self, c_in=128, d_out=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Linear(hidden, d_out)

        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.proj.bias)

    def forward(self, eeg_ct):  # (B,C,T)
        x = self.net(eeg_ct)     # (B,hidden,T)
        x = x.mean(dim=-1)       # (B,hidden)
        x = self.proj(x)         # (B,d_out)
        x = F.normalize(x, dim=-1)
        return x


class EEGAlignEncoderAttn(nn.Module):
    def __init__(self, c_in=128, d_out=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

        self.attn = nn.Conv1d(hidden, 1, kernel_size=1, bias=True)

        self.proj = nn.Linear(hidden, d_out)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.proj.bias)

        nn.init.zeros_(self.attn.weight)
        nn.init.zeros_(self.attn.bias)

    def forward(self, eeg_ct):  # (B,C,T)
        x = self.net(eeg_ct)            # (B,hidden,T)
        logits = self.attn(x)           # (B,1,T)
        w = F.softmax(logits, dim=-1)   # (B,1,T)
        x = (x * w).sum(dim=-1)         # (B,hidden)
        x = self.proj(x)                # (B,d_out)
        x = F.normalize(x, dim=-1)
        return x


def parse_args():
    p = argparse.ArgumentParser(description="t-SNE visualization for EEGAlign latent space")

    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    p.add_argument("--subject_num", type=int, default=4)

    p.add_argument("--clip_embeds_dir", type=str, required=True,
                   help="Folder containing clip_img_embeds.npy aligned with dataset split/subject.")
    p.add_argument("--eeg_align_ckpt", type=str, required=True)

    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--max_samples", type=int, default=500,
                   help="Max number of samples to visualize after filtering.")
    p.add_argument("--top_k_classes", type=int, default=10,
                   help="Keep only top-K most frequent captions/classes for cleaner plots.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--n_iter", type=int, default=1000)

    p.add_argument("--legacy_mean_pool", action="store_true",
                   help="Use the old EEGAlignEncoder with mean pooling.")

    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def sanitize_label(x: str) -> str:
    if not isinstance(x, str):
        x = str(x)
    x = x.replace("image of ", "").strip()
    return x


def load_filtered_dataset(dataset_name, split, subject_num):
    ds = load_dataset(dataset_name, split=split).with_format("torch")
    if subject_num != 0:
        ds = ds.filter(lambda x: int(x["subject"]) == int(subject_num))
    return ds


def subset_by_top_classes(ds, labels, top_k_classes, max_samples, seed):
    cnt = Counter(labels)
    keep_classes = [c for c, _ in cnt.most_common(top_k_classes)]

    keep_idx = [i for i, lab in enumerate(labels) if lab in keep_classes]

    rng = np.random.default_rng(seed)
    if len(keep_idx) > max_samples:
        keep_idx = rng.choice(keep_idx, size=max_samples, replace=False).tolist()

    keep_idx = sorted(keep_idx)
    return keep_idx, keep_classes


@torch.no_grad()
def extract_eeg_embeddings(ds, eeg_align, indices, device):
    all_z = []
    all_labels = []

    for idx in indices:
        ex = ds[idx]
        eeg = ex["conditioning_image"]
        if not torch.is_tensor(eeg):
            eeg = torch.as_tensor(eeg)
        eeg = eeg.unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,C,T)

        z = eeg_align(eeg).squeeze(0).detach().cpu().numpy()
        all_z.append(z)

        label = sanitize_label(ex["caption"])
        all_labels.append(label)

    return np.stack(all_z, axis=0), all_labels


def load_clip_embeddings(clip_embeds_dir, indices):
    path = os.path.join(clip_embeds_dir, "clip_img_embeds.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"clip_img_embeds.npy not found: {path}")

    z_img = np.load(path).astype(np.float32)
    z_img = z_img[indices]
    z_img = z_img / (np.linalg.norm(z_img, axis=1, keepdims=True) + 1e-8)
    return z_img


def make_color_map(labels):
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(uniq) <= 10 else "tab20")
    color_map = {lab: cmap(i % cmap.N) for i, lab in enumerate(uniq)}
    return color_map


def plot_tsne_eeg_only(z_eeg, labels, out_path, perplexity, n_iter, seed):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        random_state=seed,
        learning_rate="auto",
    )
    xy = tsne.fit_transform(z_eeg)

    color_map = make_color_map(labels)

    plt.figure(figsize=(10, 8))
    for lab in sorted(set(labels)):
        idx = [i for i, x in enumerate(labels) if x == lab]
        plt.scatter(
            xy[idx, 0], xy[idx, 1],
            s=28, alpha=0.8,
            color=color_map[lab],
            label=lab
        )

    plt.title("t-SNE of EEG embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(fontsize=8, markerscale=1.2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_tsne_eeg_and_img(z_eeg, z_img, labels, out_path, perplexity, n_iter, seed):
    X = np.concatenate([z_eeg, z_img], axis=0)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="pca",
        random_state=seed,
        learning_rate="auto",
    )
    xy = tsne.fit_transform(X)

    n = len(z_eeg)
    xy_eeg = xy[:n]
    xy_img = xy[n:]

    color_map = make_color_map(labels)

    plt.figure(figsize=(10, 8))
    for lab in sorted(set(labels)):
        idx = [i for i, x in enumerate(labels) if x == lab]

        plt.scatter(
            xy_img[idx, 0], xy_img[idx, 1],
            s=36, alpha=0.75,
            color=color_map[lab],
            marker="o",
            label=f"{lab} (img)"
        )

        plt.scatter(
            xy_eeg[idx, 0], xy_eeg[idx, 1],
            s=36, alpha=0.9,
            color=color_map[lab],
            marker="x"
        )

    plt.title("t-SNE of EEG and image embeddings")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(fontsize=7, markerscale=1.1, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_filtered_dataset(args.dataset_name, args.split, args.subject_num)

    labels_all = [sanitize_label(ds[i]["caption"]) for i in range(len(ds))]

    keep_idx, keep_classes = subset_by_top_classes(
        ds=ds,
        labels=labels_all,
        top_k_classes=args.top_k_classes,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    print(f"[INFO] Split={args.split}, subject={args.subject_num}")
    print(f"[INFO] Total filtered samples: {len(ds)}")
    print(f"[INFO] Using {len(keep_idx)} samples from top-{args.top_k_classes} classes:")
    print("       ", keep_classes)

    # Load correct encoder architecture
    if args.legacy_mean_pool:
        eeg_align = EEGAlignEncoderMean(c_in=128, d_out=512, hidden=256).to(device)
    else:
        eeg_align = EEGAlignEncoderAttn(c_in=128, d_out=512, hidden=256).to(device)

    sd = torch.load(args.eeg_align_ckpt, map_location="cpu")
    eeg_align.load_state_dict(sd, strict=True)
    eeg_align.eval()

    z_eeg, labels = extract_eeg_embeddings(ds, eeg_align, keep_idx, device)
    z_img = load_clip_embeddings(args.clip_embeds_dir, keep_idx)

    np.save(os.path.join(args.out_dir, f"z_eeg_{args.split}_subj{args.subject_num}.npy"), z_eeg)
    np.save(os.path.join(args.out_dir, f"z_img_{args.split}_subj{args.subject_num}.npy"), z_img)

    with open(os.path.join(args.out_dir, f"labels_{args.split}_subj{args.subject_num}.txt"), "w") as f:
        for lab in labels:
            f.write(lab + "\n")

    arch_name = "meanpool" if args.legacy_mean_pool else "attnpool"

    plot_tsne_eeg_only(
        z_eeg=z_eeg,
        labels=labels,
        out_path=os.path.join(args.out_dir, f"tsne_eeg_only_{arch_name}_{args.split}_subj{args.subject_num}.png"),
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        seed=args.seed,
    )

    plot_tsne_eeg_and_img(
        z_eeg=z_eeg,
        z_img=z_img,
        labels=labels,
        out_path=os.path.join(args.out_dir, f"tsne_eeg_img_{arch_name}_{args.split}_subj{args.subject_num}.png"),
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        seed=args.seed,
    )

    print("[DONE] Saved outputs to:", args.out_dir)


if __name__ == "__main__":
    main()