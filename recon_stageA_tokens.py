import os
import sys
import json
import time
import shutil
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Project-relative imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = PROJECT_ROOT / "models"
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))

from data.eeg_dataset import make_test_dataset, make_collate_fn
from models.eeg_backbone import GWITEEGBackbone, load_eeg_backbone_from_ckpt
from models.sife import SIFE
from models.ssfe import SSFEProjector
from models.prior import PriorNetwork, BrainDiffusionPrior

torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def seed_everything(seed=42, cudnn_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def atomic_torch_save(obj, final_path: Path):
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, final_path)


def atomic_json_save(obj, final_path: Path):
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, final_path)


def copy_file_atomic(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, dst)


def get_existing_shard_indices(shards_dir: Path, prefix: str):
    existing = set()
    if not shards_dir.exists():
        return existing

    for p in shards_dir.glob(f"{prefix}_*.pt"):
        stem = p.stem
        try:
            idx = int(stem.split("_")[-1])
            existing.add(idx)
        except ValueError:
            pass
    return existing


def print_mem(prefix=""):
    try:
        import psutil
        vm = psutil.virtual_memory()
        used = (vm.total - vm.available) / (1024 ** 3)
        avail = vm.available / (1024 ** 3)
        print(f"[RAM] {prefix} used={used:.2f} GB | avail={avail:.2f} GB")
    except Exception:
        pass


# ---------------------------------------------------------------------
# Arg parser
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Stage A: EEG -> prior token shards")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory finale su Drive")
    parser.add_argument("--tmp_dir", type=str, default="/content/gwit_runtime/stageA_tmp",
                        help="Directory temporanea locale")

    parser.add_argument("--test_subjects", type=int, nargs="+", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--conditioning_image_column", type=str, default="conditioning_image")
    parser.add_argument("--caption_column", type=str, default="caption")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prior_inference_steps", type=int, default=20)
    parser.add_argument("--prior_cond_scale", type=float, default=1.0)

    parser.add_argument("--save_gts", action="store_true")
    parser.add_argument("--max_test_samples", type=int, default=None)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--copy_to_drive_every", type=int, default=1,
                        help="Copia su Drive ogni N shard")
    parser.add_argument("--sleep_after_copy", type=float, default=0.0,
                        help="Piccola pausa dopo ogni copia, se vuoi alleggerire I/O")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
def load_trained_modules(model_dir: Path, device: torch.device):
    eeg_cfg = load_json(model_dir / "eeg_backbone_config.json")
    sife_cfg = load_json(model_dir / "sife_config.json")
    ssfe_cfg = load_json(model_dir / "ssfe_projector_config.json")
    prior_cfg = load_json(model_dir / "diffusion_prior_config.json")

    eeg_backbone = GWITEEGBackbone(
        in_channels=int(eeg_cfg["in_channels"]),
        hidden_size=int(eeg_cfg["hidden_size"]),
        num_layers=int(eeg_cfg["num_layers"]),
    )
    load_eeg_backbone_from_ckpt(eeg_backbone, str(model_dir / "eeg_backbone.pt"))
    eeg_backbone.to(device).eval()

    sife = SIFE(
        dim=int(sife_cfg["dim"]),
        seq_len=int(sife_cfg["seq_len"]),
        num_subjects=int(sife_cfg["num_subjects"]),
        fi_layers=int(sife_cfg["fi_layers"]),
        num_heads=int(sife_cfg["num_heads"]),
        grl_lambda=float(sife_cfg["grl_lambda_sife"]),
    )
    sife.load_state_dict(torch.load(model_dir / "sife.pt", map_location="cpu"))
    sife.to(device).eval()

    ssfe = SSFEProjector(
        in_dim=int(ssfe_cfg["in_dim"]),
        hidden_dim=int(ssfe_cfg["hidden_dim"]),
        out_dim=int(ssfe_cfg["out_dim"]),
        target_tokens=int(ssfe_cfg["target_tokens"]),
        adapter_type=ssfe_cfg["adapter_type"],
        num_image_classes=int(ssfe_cfg["num_image_classes"]),
        grl_lambda=float(ssfe_cfg["grl_lambda_ssfe"]),
        text_out_dim=int(ssfe_cfg["text_out_dim"]),
    )
    ssfe.load_state_dict(torch.load(model_dir / "ssfe_projector.pt", map_location="cpu"))
    ssfe.to(device).eval()

    prior_net = PriorNetwork(
        dim=int(prior_cfg["prior_dim"]),
        num_tokens=int(prior_cfg["prior_num_tokens"]),
        num_timesteps=int(prior_cfg["prior_timesteps"]),
        depth=int(prior_cfg["prior_depth"]),
        heads=int(prior_cfg["prior_heads"]),
        dim_head=52,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        norm_in=False,
        norm_out=True,
        final_proj=True,
        normformer=False,
        causal=False,
        learned_query_mode="pos_emb",
    )

    diffusion_prior = BrainDiffusionPrior(
        net=prior_net,
        image_embed_dim=int(prior_cfg["prior_dim"]),
        timesteps=int(prior_cfg["prior_timesteps"]),
        sample_timesteps=None,
        cond_drop_prob=float(prior_cfg["prior_cond_drop_prob"]),
        text_cond_drop_prob=None,
        image_cond_drop_prob=None,
        loss_type="l2",
        predict_x_start=True,
        predict_v=False,
        beta_schedule="cosine",
        training_clamp_l2norm=False,
        sampling_clamp_l2norm=False,
        sampling_final_clamp_l2norm=False,
        init_image_embed_l2norm=False,
        image_embed_scale=None,
    )
    diffusion_prior.load_state_dict(torch.load(model_dir / "diffusion_prior.pt", map_location="cpu"))
    diffusion_prior.to(device).eval()

    return eeg_backbone, sife, ssfe, diffusion_prior


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
def build_test_dataset(args, tokenizer):
    class InferenceArgs:
        pass

    ds_args = InferenceArgs()
    ds_args.dataset_name = args.dataset_name
    ds_args.data_root = args.data_root
    ds_args.image_column = args.image_column
    ds_args.conditioning_image_column = args.conditioning_image_column
    ds_args.caption_column = args.caption_column
    ds_args.cache_dir = None
    ds_args.caption_from_classifier = False
    ds_args.use_precomputed_latents = False
    ds_args.latents_dir = None
    ds_args.use_precomputed_clip_embeds = False
    ds_args.clip_embeds_dir = None
    ds_args.max_test_samples_per_subject = None
    ds_args.val_ratio = args.val_ratio
    ds_args.split_seed = args.split_seed
    ds_args.test_subjects = args.test_subjects

    dataset = make_test_dataset(ds_args, tokenizer, accelerator=None)

    if args.max_test_samples is not None:
        keep = min(int(args.max_test_samples), len(dataset))
        dataset.data = dataset.data.select(range(keep))
        dataset.sample_index_within_subject = dataset.sample_index_within_subject[:keep]
        print(f"[DATASET] truncated test set to {keep} samples")

    return dataset


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    drive_out = Path(args.output_dir)
    tmp_out = Path(args.tmp_dir)

    drive_out.mkdir(parents=True, exist_ok=True)
    tmp_out.mkdir(parents=True, exist_ok=True)

    drive_tokens_dir = drive_out / "token_shards"
    drive_gts_dir = drive_out / "gt_shards"
    tmp_tokens_dir = tmp_out / "token_shards"
    tmp_gts_dir = tmp_out / "gt_shards"

    for d in [drive_tokens_dir, tmp_tokens_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if args.save_gts:
        for d in [drive_gts_dir, tmp_gts_dir]:
            d.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    test_dataset = build_test_dataset(args, tokenizer)
    collate_fn = make_collate_fn(args.dataset_name)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    eeg_backbone, sife, ssfe, diffusion_prior = load_trained_modules(model_dir, device)

    manifest_path_drive = drive_out / "manifest.json"
    manifest_path_tmp = tmp_out / "manifest.json"

    manifest = {
        "num_shards": 0,
        "num_samples": 0,
        "token_shape_per_sample": None,
        "gt_shape_per_sample": None,
        "completed_shards": [],
    }

    completed = set()
    if args.resume:
        completed_drive = get_existing_shard_indices(drive_tokens_dir, "prior_tokens_shard")
        completed_tmp = get_existing_shard_indices(tmp_tokens_dir, "prior_tokens_shard")
        completed = completed_drive | completed_tmp
        print(f"[RESUME] found {len(completed)} completed shard(s)")

        if manifest_path_drive.exists():
            try:
                with open(manifest_path_drive, "r") as f:
                    manifest = json.load(f)
            except Exception:
                pass

    print_mem("before loop")

    copied_since_last = 0

    with torch.no_grad():
        for shard_idx, batch in enumerate(tqdm(test_loader, desc="Stage A: generating prior tokens")):
            if shard_idx in completed:
                continue

            eeg_cond = batch["conditioning_pixel_values"].to(device, dtype=torch.float32, non_blocking=True)
            gt_images = batch["pixel_values"].to(device, dtype=torch.float32, non_blocking=True)

            gt_images_01 = ((gt_images + 1.0) / 2.0).clamp(0, 1)

            eeg_feats = eeg_backbone(eeg_cond)
            E_seq = eeg_feats["sequence"]

            sife_out = sife(E_seq)
            E_i_seq = sife_out["E_i_seq"]

            ssfe_out = ssfe(
                E_i=E_i_seq.float(),
                E=E_seq.float(),
            )
            F_s = ssfe_out["F_s"]

            prior_tokens = diffusion_prior.sample(
                text_embed=F_s.float(),
                cond_scale=float(args.prior_cond_scale),
                timesteps=int(args.prior_inference_steps),
            )

            # move to cpu before save
            prior_tokens_cpu = prior_tokens.detach().cpu()

            local_token_path = tmp_tokens_dir / f"prior_tokens_shard_{shard_idx:05d}.pt"
            drive_token_path = drive_tokens_dir / f"prior_tokens_shard_{shard_idx:05d}.pt"

            atomic_torch_save(prior_tokens_cpu, local_token_path)

            if args.save_gts:
                gts_cpu = gt_images_01.detach().cpu()
                local_gt_path = tmp_gts_dir / f"gts_shard_{shard_idx:05d}.pt"
                drive_gt_path = drive_gts_dir / f"gts_shard_{shard_idx:05d}.pt"
                atomic_torch_save(gts_cpu, local_gt_path)
            else:
                gts_cpu = None
                local_gt_path = None
                drive_gt_path = None

            if manifest["token_shape_per_sample"] is None:
                manifest["token_shape_per_sample"] = list(prior_tokens_cpu.shape[1:])
            if args.save_gts and manifest["gt_shape_per_sample"] is None and gts_cpu is not None:
                manifest["gt_shape_per_sample"] = list(gts_cpu.shape[1:])

            manifest["num_shards"] = max(manifest["num_shards"], shard_idx + 1)
            manifest["num_samples"] += int(prior_tokens_cpu.shape[0])
            if shard_idx not in manifest["completed_shards"]:
                manifest["completed_shards"].append(shard_idx)
                manifest["completed_shards"].sort()

            atomic_json_save(manifest, manifest_path_tmp)

            copied_since_last += 1
            if copied_since_last >= int(args.copy_to_drive_every):
                copy_file_atomic(local_token_path, drive_token_path)
                if args.save_gts and local_gt_path is not None:
                    copy_file_atomic(local_gt_path, drive_gt_path)
                atomic_json_save(manifest, manifest_path_drive)
                copied_since_last = 0
                if args.sleep_after_copy > 0:
                    time.sleep(args.sleep_after_copy)

            # cleanup hard
            del eeg_cond, gt_images, gt_images_01
            del eeg_feats, E_seq, sife_out, E_i_seq, ssfe_out, F_s
            del prior_tokens, prior_tokens_cpu
            if gts_cpu is not None:
                del gts_cpu

            if device.type == "cuda":
                torch.cuda.empty_cache()

            print_mem(f"after shard {shard_idx:05d}")

    # final sync of anything not yet copied
    all_local_token_shards = sorted(tmp_tokens_dir.glob("prior_tokens_shard_*.pt"))
    for p in all_local_token_shards:
        dst = drive_tokens_dir / p.name
        if not dst.exists():
            copy_file_atomic(p, dst)

    if args.save_gts:
        all_local_gt_shards = sorted(tmp_gts_dir.glob("gts_shard_*.pt"))
        for p in all_local_gt_shards:
            dst = drive_gts_dir / p.name
            if not dst.exists():
                copy_file_atomic(p, dst)

    atomic_json_save(manifest, manifest_path_drive)
    atomic_json_save(manifest, manifest_path_tmp)

    print(f"[DONE] wrote {len(manifest['completed_shards'])} shard(s)")
    print(f"[DONE] total samples counted = {manifest['num_samples']}")
    print(f"[SAVED] {manifest_path_drive}")


if __name__ == "__main__":
    main()