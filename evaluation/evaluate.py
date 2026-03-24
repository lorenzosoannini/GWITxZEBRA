import os
import re
import argparse
import shutil
import tempfile
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from tqdm import tqdm
from scipy.stats import entropy

import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torchvision.models import ViT_H_14_Weights, vit_h_14

from pytorch_fid import fid_score
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import accuracy

from transformers import CLIPProcessor, CLIPModel


# =========================================================
# GLOBAL SETUP
# =========================================================

EVAL_SIZE = 512

transform = transforms.Compose([
    transforms.Resize((EVAL_SIZE, EVAL_SIZE), antialias=True),
    transforms.ToTensor(),
])

LPIPS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
lpips_metric = LearnedPerceptualImagePatchSimilarity(
    net_type="alex",
    normalize=True,
).to(LPIPS_DEVICE)

# output_{sampleid}_{k}_{label}.png
GEN_RE = re.compile(r"^output_(\d+)_(\d+)_(.+)\.(png|jpg|jpeg)$")
# gt_{sampleid}_00_{label}.png
GT_RE = re.compile(r"^gt_(\d+)_(\d+)_(.+)\.(png|jpg|jpeg)$")


# =========================================================
# UTILS
# =========================================================

def print_results_table(results: dict):
    rows = []
    for k, v in results.items():
        if v is None:
            rows.append((k, "—"))
        else:
            if isinstance(v, (float, np.floating)):
                rows.append((k, f"{float(v):.4f}"))
            else:
                rows.append((k, str(v)))

    if not rows:
        print("No results to display.")
        return

    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows)
    line = "+" + "-" * (col1 + 2) + "+" + "-" * (col2 + 2) + "+"

    print("\nFinal results")
    print(line)
    print(f"| {'Metric'.ljust(col1)} | {'Value'.ljust(col2)} |")
    print(line)
    for m, val in rows:
        print(f"| {m.ljust(col1)} | {val.ljust(col2)} |")
    print(line)


def imread(filename):
    return np.asarray(Image.open(filename).convert("RGB"), dtype=np.uint8)


def pil_to_tensor_01(path):
    img = Image.open(path).convert("RGB")
    return transform(img)


def sort_key_legacy(fname: str):
    m = re.search(r"_(\d+)_(\d+)_", fname)
    if m:
        return (int(m.group(1)), int(m.group(2)), fname)
    return (10**9, 10**9, fname)


def parse_prompt_from_label(label: str) -> str:
    return label.replace("_", " ").strip()


def parse_gen_filename(fname: str):
    m = GEN_RE.match(fname)
    if m is None:
        return None
    sample_id = int(m.group(1))
    k = int(m.group(2))
    label = m.group(3)
    return sample_id, k, label


def parse_gt_filename(fname: str):
    m = GT_RE.match(fname)
    if m is None:
        return None
    sample_id = int(m.group(1))
    k = int(m.group(2))
    label = m.group(3)
    return sample_id, k, label


def list_image_files(folder):
    return sorted(
        [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )


def build_grouped_pairs(gen_folder, gt_folder):
    """
    Returns dict:
      sample_id -> {
        "label": str,
        "gen": [paths sorted by k],
        "gt": one gt path
      }

    Expected:
      generated: output_{sid}_{k}_{label}.png
      gt:        gt_{sid}_00_{label}.png
    """
    gen_files = list_image_files(gen_folder)
    gt_files = list_image_files(gt_folder)

    gt_by_sample = {}
    for f in gt_files:
        parsed = parse_gt_filename(f)
        if parsed is None:
            continue
        sid, k, label = parsed
        path = os.path.join(gt_folder, f)
        if sid not in gt_by_sample:
            gt_by_sample[sid] = {"label": label, "path": path}

    grouped = defaultdict(lambda: {"label": None, "gen": [], "gt": None})

    for f in gen_files:
        parsed = parse_gen_filename(f)
        if parsed is None:
            continue
        sid, k, label = parsed
        grouped[sid]["label"] = label
        grouped[sid]["gen"].append((k, os.path.join(gen_folder, f)))

    final = {}
    for sid, item in grouped.items():
        if sid not in gt_by_sample:
            continue
        item["gen"] = [p for _, p in sorted(item["gen"], key=lambda x: x[0])]
        item["gt"] = gt_by_sample[sid]["path"]
        final[sid] = item

    return dict(sorted(final.items(), key=lambda x: x[0]))


def flatten_grouped_pairs(grouped_pairs):
    """
    Returns:
      gen_paths_flat, gt_paths_flat, prompts_flat

    GT is repeated to match each generated image.
    """
    gen_paths = []
    gt_paths = []
    prompts = []

    for sid in sorted(grouped_pairs.keys()):
        gt_path = grouped_pairs[sid]["gt"]
        prompt = parse_prompt_from_label(grouped_pairs[sid]["label"])
        for g in grouped_pairs[sid]["gen"]:
            gen_paths.append(g)
            gt_paths.append(gt_path)
            prompts.append(prompt)

    return gen_paths, gt_paths, prompts


def load_images_as_tensors_from_paths(paths, batch_size):
    for i in range(0, len(paths), batch_size):
        batch = [pil_to_tensor_01(p) for p in paths[i:i + batch_size]]
        yield torch.stack(batch)


def calc_lpips_batch(img1_batch, img2_batch, device):
    return lpips_metric(img1_batch.to(device), img2_batch.to(device))


# =========================================================
# INCEPTION SCORE
# =========================================================

def inception_score_from_paths(image_paths, device="cuda", batch_size=32, resize=True):
    device = torch.device(device)

    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    up = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False).to(device)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).detach().cpu().numpy()

    N = len(image_paths)
    preds = np.zeros((N, 1000), dtype=np.float32)

    print("Computing predictions using Inception v3")
    for i in tqdm(range(0, N, batch_size), desc="IS"):
        images = np.array([imread(f).astype(np.float32) for f in image_paths[i:i + batch_size]])
        images = images.transpose((0, 3, 1, 2))
        images /= 255.0
        batch = torch.from_numpy(images).float().to(device)
        preds[i:i + len(batch)] = get_pred(batch)

    py = np.mean(preds, axis=0)
    scores = [entropy(preds[i], py) for i in range(preds.shape[0])]
    return float(np.exp(np.mean(scores)))


# =========================================================
# GENERALIZATION ACCURACY (legacy-compatible)
# =========================================================

def n_way_top_k_acc(pred, class_id, n_way, num_trials=50, top_k=1):
    pick_range = [i for i in range(len(pred)) if i != class_id]
    acc_list = []

    for _ in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way - 1, replace=False)
        pred_picked = torch.cat([pred[class_id].unsqueeze(0), pred[idxs_picked]])

        acc = accuracy(
            pred_picked.unsqueeze(0),
            torch.tensor([0], device=pred.device),
            task="multiclass",
            num_classes=50,
            top_k=top_k,
        )
        acc_list.append(acc.item())

    return np.mean(acc_list), np.std(acc_list)


def compute_ga_legacy(gen_paths, gt_paths, limit, device):
    print("Computing Generalization Accuracy (GA)")
    weights = ViT_H_14_Weights.DEFAULT
    model = vit_h_14(weights=weights).to(device)
    preprocess = weights.transforms()
    model.eval()

    n_way = 50
    num_trials = 50
    top_k = 1
    acc_list = []

    assert len(gen_paths) == len(gt_paths), "GA legacy requires aligned gen/gt flat paths."

    for j in tqdm(range(0, len(gt_paths), limit), desc="GA"):
        real_image = Image.open(gt_paths[j]).convert("RGB")
        gt = preprocess(real_image).unsqueeze(0).to(device)
        gt_class_id = model(gt).squeeze(0).softmax(0).argmax().item()

        for i in range(limit):
            if j + i >= len(gen_paths):
                break
            gen_img = Image.open(gen_paths[j + i]).convert("RGB")
            pred = preprocess(gen_img).unsqueeze(0).to(device)
            pred_out = model(pred).squeeze(0).softmax(0).detach()

            acc, _ = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
            acc_list.append(acc)

    return float(np.mean(acc_list)) if len(acc_list) > 0 else None


# =========================================================
# FID
# =========================================================

def compute_fid_from_folders(gt_folder_like, gen_folder_like, device):
    return float(
        fid_score.calculate_fid_given_paths(
            [gt_folder_like, gen_folder_like],
            batch_size=50,
            device=device,
            dims=2048,
        )
    )


def compute_fid_legacy_from_flat_pairs(gen_paths, gt_paths, device):
    temp_gt = tempfile.mkdtemp(prefix="fid_gt_legacy_flat_")
    temp_gen = tempfile.mkdtemp(prefix="fid_gen_legacy_flat_")
    try:
        for i, (g, t) in enumerate(zip(gen_paths, gt_paths)):
            shutil.copy(t, os.path.join(temp_gt, f"{i:06d}.png"))
            shutil.copy(g, os.path.join(temp_gen, f"{i:06d}.png"))
        return compute_fid_from_folders(temp_gt, temp_gen, device)
    finally:
        shutil.rmtree(temp_gt, ignore_errors=True)
        shutil.rmtree(temp_gen, ignore_errors=True)


def compute_fid_grouped_all(grouped_pairs, device):
    temp_gt = tempfile.mkdtemp(prefix="fid_gt_grouped_all_")
    temp_gen = tempfile.mkdtemp(prefix="fid_gen_grouped_all_")
    try:
        counter = 0
        for sid in sorted(grouped_pairs.keys()):
            gt_path = grouped_pairs[sid]["gt"]
            for gen_path in grouped_pairs[sid]["gen"]:
                shutil.copy(gt_path, os.path.join(temp_gt, f"{counter:06d}.png"))
                shutil.copy(gen_path, os.path.join(temp_gen, f"{counter:06d}.png"))
                counter += 1
        return compute_fid_from_folders(temp_gt, temp_gen, device)
    finally:
        shutil.rmtree(temp_gt, ignore_errors=True)
        shutil.rmtree(temp_gen, ignore_errors=True)


def compute_fid_grouped_first(grouped_pairs, device):
    temp_gt = tempfile.mkdtemp(prefix="fid_gt_grouped_first_")
    temp_gen = tempfile.mkdtemp(prefix="fid_gen_grouped_first_")
    try:
        counter = 0
        for sid in sorted(grouped_pairs.keys()):
            gt_path = grouped_pairs[sid]["gt"]
            gen_list = grouped_pairs[sid]["gen"]
            if len(gen_list) == 0:
                continue
            shutil.copy(gt_path, os.path.join(temp_gt, f"{counter:06d}.png"))
            shutil.copy(gen_list[0], os.path.join(temp_gen, f"{counter:06d}.png"))
            counter += 1
        return compute_fid_from_folders(temp_gt, temp_gen, device)
    finally:
        shutil.rmtree(temp_gt, ignore_errors=True)
        shutil.rmtree(temp_gen, ignore_errors=True)


# =========================================================
# LPIPS
# =========================================================

def compute_lpips_legacy(gen_paths, gt_paths, batch_size, device):
    vals = []
    for preds_batch, target_batch in tqdm(
        zip(
            load_images_as_tensors_from_paths(gen_paths, batch_size),
            load_images_as_tensors_from_paths(gt_paths, batch_size),
        ),
        desc="LPIPS legacy",
    ):
        batch_vals = calc_lpips_batch(preds_batch, target_batch, device=device)
        vals.append(batch_vals.detach().cpu())

    if len(vals) == 0:
        return None

    vals = torch.cat(vals, dim=0)
    return float(vals.mean().item())


def compute_lpips_grouped(grouped_pairs, device):
    mean_vals = []
    best_vals = []

    for sid in tqdm(sorted(grouped_pairs.keys()), desc="LPIPS grouped"):
        gt = pil_to_tensor_01(grouped_pairs[sid]["gt"]).unsqueeze(0)
        gen_list = grouped_pairs[sid]["gen"]
        if len(gen_list) == 0:
            continue

        preds = torch.stack([pil_to_tensor_01(p) for p in gen_list], dim=0)
        gt_rep = gt.repeat(preds.size(0), 1, 1, 1)

        vals = calc_lpips_batch(preds, gt_rep, device=device).detach().cpu()
        mean_vals.append(vals.mean().item())
        best_vals.append(vals.min().item())

    out = {}
    out["LPIPS_mean"] = float(np.mean(mean_vals)) if len(mean_vals) > 0 else None
    out["LPIPS_best"] = float(np.mean(best_vals)) if len(best_vals) > 0 else None
    return out


# =========================================================
# CLIP METRICS
# =========================================================

@torch.no_grad()
def compute_clip_metrics_legacy(gen_paths, gt_paths, batch_size, device, prompts, clip_model_name):
    dev = torch.device(device)
    model = CLIPModel.from_pretrained(clip_model_name).to(dev)
    proc = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()

    clip_i2i = []
    clip_i2t = []

    for i in tqdm(range(0, len(gen_paths), batch_size), desc="CLIP legacy"):
        gen_imgs = [Image.open(p).convert("RGB") for p in gen_paths[i:i + batch_size]]
        gt_imgs = [Image.open(p).convert("RGB") for p in gt_paths[i:i + batch_size]]
        txts = prompts[i:i + batch_size]

        gen_in = proc(images=gen_imgs, return_tensors="pt")
        gt_in = proc(images=gt_imgs, return_tensors="pt")
        gen_in = {k: v.to(dev) for k, v in gen_in.items()}
        gt_in = {k: v.to(dev) for k, v in gt_in.items()}

        gen_img_emb = model.get_image_features(**gen_in)
        gt_img_emb = model.get_image_features(**gt_in)
        gen_img_emb = F.normalize(gen_img_emb, dim=-1)
        gt_img_emb = F.normalize(gt_img_emb, dim=-1)

        clip_i2i.append((gen_img_emb * gt_img_emb).sum(dim=-1).detach().cpu().numpy())

        txt_in = proc(text=txts, return_tensors="pt", padding=True, truncation=True)
        txt_in = {k: v.to(dev) for k, v in txt_in.items()}
        txt_emb = model.get_text_features(**txt_in)
        txt_emb = F.normalize(txt_emb, dim=-1)

        clip_i2t.append((gen_img_emb * txt_emb).sum(dim=-1).detach().cpu().numpy())

    out = {}
    out["CLIP I2I Cosine"] = float(np.mean(np.concatenate(clip_i2i, axis=0))) if len(clip_i2i) > 0 else None
    out["CLIP I2T Cosine"] = float(np.mean(np.concatenate(clip_i2t, axis=0))) if len(clip_i2t) > 0 else None
    return out


@torch.no_grad()
def compute_clip_metrics_grouped(grouped_pairs, batch_size, device, clip_model_name):
    dev = torch.device(device)
    model = CLIPModel.from_pretrained(clip_model_name).to(dev)
    proc = CLIPProcessor.from_pretrained(clip_model_name)
    model.eval()

    i2i_mean_vals = []
    i2i_best_vals = []
    i2t_mean_vals = []
    i2t_best_vals = []

    sample_ids = sorted(grouped_pairs.keys())

    for start in tqdm(range(0, len(sample_ids), batch_size), desc="CLIP grouped"):
        batch_sids = sample_ids[start:start + batch_size]

        for sid in batch_sids:
            item = grouped_pairs[sid]
            gen_paths = item["gen"]
            gt_path = item["gt"]
            prompt = parse_prompt_from_label(item["label"])

            if len(gen_paths) == 0:
                continue

            gen_imgs = [Image.open(p).convert("RGB") for p in gen_paths]
            gt_imgs = [Image.open(gt_path).convert("RGB")] * len(gen_paths)
            prompts = [prompt] * len(gen_paths)

            gen_in = proc(images=gen_imgs, return_tensors="pt")
            gt_in = proc(images=gt_imgs, return_tensors="pt")
            gen_in = {k: v.to(dev) for k, v in gen_in.items()}
            gt_in = {k: v.to(dev) for k, v in gt_in.items()}

            gen_img_emb = model.get_image_features(**gen_in)
            gt_img_emb = model.get_image_features(**gt_in)
            gen_img_emb = F.normalize(gen_img_emb, dim=-1)
            gt_img_emb = F.normalize(gt_img_emb, dim=-1)

            txt_in = proc(text=prompts, return_tensors="pt", padding=True, truncation=True)
            txt_in = {k: v.to(dev) for k, v in txt_in.items()}
            txt_emb = model.get_text_features(**txt_in)
            txt_emb = F.normalize(txt_emb, dim=-1)

            i2i = (gen_img_emb * gt_img_emb).sum(dim=-1).detach().cpu().numpy()
            i2t = (gen_img_emb * txt_emb).sum(dim=-1).detach().cpu().numpy()

            i2i_mean_vals.append(float(np.mean(i2i)))
            i2i_best_vals.append(float(np.max(i2i)))
            i2t_mean_vals.append(float(np.mean(i2t)))
            i2t_best_vals.append(float(np.max(i2t)))

    out = {}
    out["CLIP_I2I_mean"] = float(np.mean(i2i_mean_vals)) if len(i2i_mean_vals) > 0 else None
    out["CLIP_I2I_best"] = float(np.mean(i2i_best_vals)) if len(i2i_best_vals) > 0 else None
    out["CLIP_I2T_mean"] = float(np.mean(i2t_mean_vals)) if len(i2t_mean_vals) > 0 else None
    out["CLIP_I2T_best"] = float(np.mean(i2t_best_vals)) if len(i2t_best_vals) > 0 else None
    return out


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--limit",
        type=int,
        default=4,
        help="Used by legacy GA, i.e. number of generated images per sample.",
    )
    parser.add_argument("--GA", action="store_true")
    parser.add_argument("--guess", action="store_true")

    parser.add_argument("--clip_metrics", action="store_true")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32")

    parser.add_argument(
        "--eval_mode",
        type=str,
        default="grouped",
        choices=["legacy", "grouped"],
        help="legacy = GWIT-comparable flat evaluation; grouped = multi-sample-per-EEG evaluation.",
    )

    parser.add_argument(
        "--fid_variant",
        type=str,
        default="all",
        choices=["all", "first"],
        help="Only for grouped mode: all uses all generated images, first uses only first sample per EEG.",
    )

    args = parser.parse_args()

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"

    gen_folder = (
        os.path.join(args.controlnet_path, "guess", "generated")
        if args.guess else
        os.path.join(args.controlnet_path, "generated")
    )
    gt_folder = (
        os.path.join(args.controlnet_path, "guess", "ground_truth")
        if args.guess else
        os.path.join(args.controlnet_path, "ground_truth")
    )

    if not os.path.isdir(gen_folder):
        raise FileNotFoundError(f"Generated folder not found: {gen_folder}")
    if not os.path.isdir(gt_folder):
        raise FileNotFoundError(f"Ground-truth folder not found: {gt_folder}")

    grouped_pairs = build_grouped_pairs(gen_folder, gt_folder)
    if len(grouped_pairs) == 0:
        raise RuntimeError("No valid grouped pairs found. Check filename format.")

    gen_paths_flat, gt_paths_flat, prompts_flat = flatten_grouped_pairs(grouped_pairs)

    results = {}

    if args.eval_mode == "legacy":
        print(f"[EVAL] legacy mode | num_gen={len(gen_paths_flat)} | num_gt_flat={len(gt_paths_flat)}")

        # IS
        results["Inception Score"] = inception_score_from_paths(
            gen_paths_flat,
            device=device,
            batch_size=args.batch_size,
        )
        print(f"Inception Score: {results['Inception Score']:.4f}")

        # GA
        if args.GA:
            results["MEAN GA"] = compute_ga_legacy(
                gen_paths=gen_paths_flat,
                gt_paths=gt_paths_flat,
                limit=args.limit,
                device=device,
            )
            print(f"MEAN GA: {results['MEAN GA']:.4f}")

        # FID
        results["FID"] = compute_fid_legacy_from_flat_pairs(
            gen_paths=gen_paths_flat,
            gt_paths=gt_paths_flat,
            device=device,
        )
        print(f"FID: {results['FID']:.4f}")

        # LPIPS
        results["Mean LPIPS"] = compute_lpips_legacy(
            gen_paths=gen_paths_flat,
            gt_paths=gt_paths_flat,
            batch_size=50,
            device=LPIPS_DEVICE,
        )
        print(f"Mean LPIPS: {results['Mean LPIPS']:.4f}")

        # CLIP
        if args.clip_metrics:
            clip_res = compute_clip_metrics_legacy(
                gen_paths=gen_paths_flat,
                gt_paths=gt_paths_flat,
                batch_size=args.batch_size,
                device=device,
                prompts=prompts_flat,
                clip_model_name=args.clip_model_name,
            )
            results.update(clip_res)
            if results["CLIP I2I Cosine"] is not None:
                print(f"CLIP I2I Cosine: {results['CLIP I2I Cosine']:.4f}")
            if results["CLIP I2T Cosine"] is not None:
                print(f"CLIP I2T Cosine: {results['CLIP I2T Cosine']:.4f}")

    else:
        num_samples = len(grouped_pairs)
        num_gen = len(gen_paths_flat)
        print(f"[EVAL] grouped mode | num_samples={num_samples} | num_generated={num_gen}")

        # IS
        results["Inception Score"] = inception_score_from_paths(
            gen_paths_flat,
            device=device,
            batch_size=args.batch_size,
        )
        print(f"Inception Score: {results['Inception Score']:.4f}")

        # Optional GA
        if args.GA:
            results["MEAN GA"] = compute_ga_legacy(
                gen_paths=gen_paths_flat,
                gt_paths=gt_paths_flat,
                limit=args.limit,
                device=device,
            )
            print(f"MEAN GA: {results['MEAN GA']:.4f}")

        # FID grouped
        if args.fid_variant == "all":
            results["FID_grouped_all"] = compute_fid_grouped_all(grouped_pairs, device=device)
            print(f"FID_grouped_all: {results['FID_grouped_all']:.4f}")
        else:
            results["FID_grouped_first"] = compute_fid_grouped_first(grouped_pairs, device=device)
            print(f"FID_grouped_first: {results['FID_grouped_first']:.4f}")

        # Also keep legacy-like FID on flat pairs
        results["FID_legacy"] = compute_fid_legacy_from_flat_pairs(
            gen_paths=gen_paths_flat,
            gt_paths=gt_paths_flat,
            device=device,
        )
        print(f"FID_legacy: {results['FID_legacy']:.4f}")

        # LPIPS grouped
        lpips_res = compute_lpips_grouped(grouped_pairs, device=LPIPS_DEVICE)
        results.update(lpips_res)
        if results["LPIPS_mean"] is not None:
            print(f"LPIPS_mean: {results['LPIPS_mean']:.4f}")
        if results["LPIPS_best"] is not None:
            print(f"LPIPS_best: {results['LPIPS_best']:.4f}")

        # CLIP grouped
        if args.clip_metrics:
            clip_res = compute_clip_metrics_grouped(
                grouped_pairs=grouped_pairs,
                batch_size=args.batch_size,
                device=device,
                clip_model_name=args.clip_model_name,
            )
            results.update(clip_res)
            for k in ["CLIP_I2I_mean", "CLIP_I2I_best", "CLIP_I2T_mean", "CLIP_I2T_best"]:
                if results.get(k) is not None:
                    print(f"{k}: {results[k]:.4f}")

    print_results_table(results)