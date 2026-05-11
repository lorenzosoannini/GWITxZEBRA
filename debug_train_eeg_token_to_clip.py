import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from data.eeg_dataset_clean import (
    make_train_dataset,
    make_val_dataset,
    make_collate_fn,
)
from models.eeg_token_backbone import EEGTokenBackbone
from models.ssfe import (
    PretrainCLIPProjector,
    multi_positive_sequence_info_nce_loss,
)

logger = get_logger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def sanitize_config_for_trackers(config):
    clean = {}
    for k, v in config.items():
        if v is None:
            clean[k] = "None"
        elif isinstance(v, (int, float, str, bool)):
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            clean[k] = ",".join(str(x) for x in v)
        else:
            clean[k] = str(v)
    return clean


def silence_external_loggers():
    noisy_loggers = [
        "httpx",
        "httpcore",
        "urllib3",
        "requests",
        "huggingface_hub",
        "datasets",
        "fsspec",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_group_ids_from_batch(batch, key: str, device: torch.device) -> torch.Tensor:
    if key not in batch:
        raise KeyError(
            f"Missing required batch key '{key}'. "
            f"Make sure data/eeg_dataset_clean.py returns it."
        )

    group_ids = batch[key]
    if not isinstance(group_ids, torch.Tensor):
        group_ids = torch.tensor(group_ids, device=device, dtype=torch.long)
    else:
        group_ids = group_ids.to(device=device, dtype=torch.long)

    if group_ids.ndim != 1:
        raise ValueError(f"Expected {key} to be 1D, got shape {tuple(group_ids.shape)}")

    return group_ids


def flatten_norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float().reshape(x.shape[0], -1), dim=-1)


def paired_cosine(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    pred = flatten_norm(pred_tokens)
    target = flatten_norm(target_tokens)
    return (pred * target).sum(dim=-1).mean()


def paired_mse(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_tokens.float(), target_tokens.float())


def paired_l1(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_tokens.float(), target_tokens.float())


def topk_group_retrieval(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    group_ids: torch.Tensor,
    k: int = 1,
) -> torch.Tensor:
    pred = flatten_norm(pred_tokens)
    target = flatten_norm(target_tokens)

    sim = pred @ target.t()
    k = min(int(k), sim.shape[1])

    topk_idx = sim.topk(k=k, dim=1).indices
    topk_group_ids = group_ids[topk_idx]

    ok = topk_group_ids.eq(group_ids[:, None]).any(dim=1).float().mean()
    return ok


def retrieval_margin(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    pred = flatten_norm(pred_tokens)
    target = flatten_norm(target_tokens)
    sim = pred @ target.t()

    if sim.shape[0] <= 1:
        return torch.tensor(float("nan"), device=sim.device)

    diag = sim.diag()
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    offdiag_max = sim.masked_fill(eye, -float("inf")).max(dim=1).values
    return (diag - offdiag_max).mean()


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
@torch.no_grad()
def run_validation(
    val_dataloader,
    eeg_token_backbone,
    clip_projector,
    args,
    accelerator,
):
    backbone_was_training = eeg_token_backbone.training
    projector_was_training = clip_projector.training

    eeg_token_backbone.eval()
    clip_projector.eval()

    device = accelerator.device

    names = [
        "total",
        "loss",
        "cos",
        "mse",
        "l1",
        "top1",
        "top5",
        "margin",
        "pred_norm",
        "target_norm",
        "seq_norm",
    ]
    acc = {name: torch.tensor(0.0, device=device) for name in names}

    for batch in val_dataloader:
        eeg_cond = batch["conditioning_pixel_values"].to(device, dtype=torch.float32)
        clip_target = batch["clip_img_embeds"].to(device, dtype=torch.float32)

        if clip_target.ndim != 3:
            raise RuntimeError(
                f"Expected clip_img_embeds shape (B,T,D), got {tuple(clip_target.shape)}"
            )

        visual_group_ids = get_group_ids_from_batch(batch, "visual_group_ids", device)

        bsz = eeg_cond.shape[0]
        bsz_t = torch.tensor(float(bsz), device=device)

        eeg_feats = eeg_token_backbone(eeg_cond.float())
        E_seq = eeg_feats["sequence"]
        pred_tokens = clip_projector(E_seq.float())

        loss = multi_positive_sequence_info_nce_loss(
            pred_tokens.float(),
            clip_target.float(),
            group_ids=visual_group_ids,
            temperature=args.temperature,
            exclude_self=False,
        )

        cos = paired_cosine(pred_tokens, clip_target)
        mse = paired_mse(pred_tokens, clip_target)
        l1 = paired_l1(pred_tokens, clip_target)
        top1 = topk_group_retrieval(pred_tokens, clip_target, visual_group_ids, k=1)
        top5 = topk_group_retrieval(pred_tokens, clip_target, visual_group_ids, k=5)
        margin = retrieval_margin(pred_tokens, clip_target)

        pred_norm = pred_tokens.float().reshape(pred_tokens.shape[0], -1).norm(dim=-1).mean()
        target_norm = clip_target.float().reshape(clip_target.shape[0], -1).norm(dim=-1).mean()
        seq_norm = E_seq.float().reshape(E_seq.shape[0], -1).norm(dim=-1).mean()

        acc["total"] += bsz_t
        acc["loss"] += loss * bsz_t
        acc["cos"] += cos * bsz_t
        acc["mse"] += mse * bsz_t
        acc["l1"] += l1 * bsz_t
        acc["top1"] += top1 * bsz_t
        acc["top5"] += top5 * bsz_t
        if not torch.isnan(margin):
            acc["margin"] += margin * bsz_t
        acc["pred_norm"] += pred_norm * bsz_t
        acc["target_norm"] += target_norm * bsz_t
        acc["seq_norm"] += seq_norm * bsz_t

    gathered = accelerator.gather_for_metrics(
        torch.stack([acc[name] for name in names]).unsqueeze(0)
    )
    summed = gathered.sum(dim=0).view(-1)

    total = summed[0]
    if total.item() <= 0:
        raise RuntimeError("Validation dataloader is empty.")

    metrics = {
        "val/loss": (summed[1] / total).item(),
        "val/cos": (summed[2] / total).item(),
        "val/mse": (summed[3] / total).item(),
        "val/l1": (summed[4] / total).item(),
        "val/top1": (summed[5] / total).item(),
        "val/top5": (summed[6] / total).item(),
        "val/margin": (summed[7] / total).item(),
        "val/pred_norm": (summed[8] / total).item(),
        "val/target_norm": (summed[9] / total).item(),
        "val/seq_norm": (summed[10] / total).item(),
    }

    eeg_token_backbone.train(backbone_was_training)
    clip_projector.train(projector_was_training)

    return metrics


# ---------------------------------------------------------
# Args
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Debug EEGTokenBackbone direct EEG-to-CLIP training. "
            "No SIFE, no SSFE, no prior, no GRL."
        )
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--conditioning_image_column", type=str, default="conditioning_image")

    parser.add_argument("--train_subjects", type=int, nargs="+", required=True)
    parser.add_argument("--val_subjects", type=int, nargs="+", required=True)
    parser.add_argument("--test_subjects", type=int, nargs="+", required=True)

    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_train_samples_per_subject", type=int, default=None)
    parser.add_argument("--max_val_samples_per_subject", type=int, default=None)
    parser.add_argument("--max_test_samples_per_subject", type=int, default=None)

    parser.add_argument("--use_image_hash_visual_ids", action="store_true")
    parser.add_argument("--visual_ids_root", type=str, default=None)

    parser.add_argument("--use_precomputed_clip_embeds", action="store_true")
    parser.add_argument("--clip_embeds_dir", type=str, required=True)

    # EEG token backbone
    parser.add_argument("--eeg_in_channels", type=int, default=128)
    parser.add_argument("--eeg_input_time", type=int, default=440)
    parser.add_argument("--eeg_stem_dim", type=int, default=256)
    parser.add_argument("--eeg_token_dim", type=int, default=512)
    parser.add_argument("--eeg_target_tokens", type=int, default=256)
    parser.add_argument("--eeg_conv_blocks", type=int, default=4)
    parser.add_argument("--eeg_transformer_layers", type=int, default=4)
    parser.add_argument("--eeg_transformer_heads", type=int, default=8)
    parser.add_argument("--eeg_transformer_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--eeg_dropout", type=float, default=0.1)
    parser.add_argument("--eeg_use_cls_token", action="store_true")

    # CLIP projector head
    parser.add_argument("--projector_hidden_dim", type=int, default=512)
    parser.add_argument("--projector_out_dim", type=int, default=1664)
    parser.add_argument("--projector_target_tokens", type=int, default=256)
    parser.add_argument(
        "--projector_adapter_type",
        type=str,
        default="simple",
        choices=["simple", "zebra_like"],
        help=(
            "For this new backbone, 'simple' is recommended first because the backbone "
            "already produces target_tokens. 'zebra_like' is available for ablation."
        ),
    )

    # Optimization
    parser.add_argument("--train_batch_size", type=int, default=40)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=None)
    parser.add_argument("--projector_lr", type=float, default=None)

    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    parser.add_argument("--max_train_steps", type=int, default=4000)
    parser.add_argument("--num_train_epochs", type=int, default=100)

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--validation_steps", type=int, default=200)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--console_log_every", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)

    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, default="debug_eeg_token_to_clip")
    parser.add_argument("--logging_dir", type=str, default="logs")

    parser.add_argument("--allow_tf32", action="store_true")

    args = parser.parse_args()

    if isinstance(args.report_to, str) and args.report_to.lower() in {"none", "null", "no"}:
        args.report_to = None

    # This script always requires precomputed CLIP tokens.
    args.use_precomputed_clip_embeds = True

    return args


# ---------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------
def get_sorted_checkpoints(output_dir, prefix="checkpoint"):
    checkpoints = []
    if not os.path.isdir(output_dir):
        return checkpoints

    for path in os.listdir(output_dir):
        full = os.path.join(output_dir, path)
        if os.path.isdir(full) and path.startswith(f"{prefix}-"):
            try:
                checkpoints.append((int(path.split("-")[-1]), full))
            except ValueError:
                continue

    return sorted(checkpoints, key=lambda x: x[0])


def rotate_checkpoints(output_dir, max_checkpoints, prefix="checkpoint"):
    if max_checkpoints is None or max_checkpoints <= 0:
        return

    checkpoints = get_sorted_checkpoints(output_dir, prefix=prefix)
    if len(checkpoints) <= max_checkpoints:
        return

    import shutil

    for step, path in checkpoints[: len(checkpoints) - max_checkpoints]:
        logger.info(f"Removing old checkpoint {path} (step {step})")
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=logging_dir,
        ),
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    silence_external_loggers()
    logger.info(accelerator.state)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    train_dataset = make_train_dataset(args, accelerator)
    val_dataset = make_val_dataset(args, accelerator)
    collate_fn = make_collate_fn(args.dataset_name)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.val_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    accelerator.print(
        f"[DATA] train={len(train_dataset)} | val={len(val_dataset)} | "
        f"train_subjects={args.train_subjects} | val_subjects={args.val_subjects} | "
        f"test_subjects={args.test_subjects}"
    )

    # ---------------------------------------------------------
    # Infer CLIP dimensions
    # ---------------------------------------------------------
    first_ex = train_dataset[0]
    if "clip_img_embeds" not in first_ex:
        raise RuntimeError("Dataset must return clip_img_embeds.")

    clip_ex = first_ex["clip_img_embeds"]
    if clip_ex.ndim != 2:
        raise RuntimeError(
            f"Expected sequence-level clip_img_embeds shape (T,D), got {tuple(clip_ex.shape)}"
        )

    inferred_clip_tokens, inferred_clip_dim = clip_ex.shape

    if int(args.projector_target_tokens) != int(inferred_clip_tokens):
        raise ValueError(
            f"projector_target_tokens={args.projector_target_tokens}, "
            f"but dataset has {inferred_clip_tokens} CLIP tokens."
        )

    if int(args.projector_out_dim) != int(inferred_clip_dim):
        raise ValueError(
            f"projector_out_dim={args.projector_out_dim}, "
            f"but dataset has CLIP dim={inferred_clip_dim}."
        )

    if int(args.eeg_target_tokens) != int(inferred_clip_tokens):
        raise ValueError(
            f"eeg_target_tokens={args.eeg_target_tokens}, "
            f"but dataset has {inferred_clip_tokens} CLIP tokens. "
            f"For this first direct test, keep EEG tokens equal to CLIP tokens."
        )

    accelerator.print(
        f"[CLIP] tokens={inferred_clip_tokens} | dim={inferred_clip_dim}"
    )

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------
    eeg_token_backbone = EEGTokenBackbone(
        in_channels=args.eeg_in_channels,
        input_time=args.eeg_input_time,
        stem_dim=args.eeg_stem_dim,
        token_dim=args.eeg_token_dim,
        target_tokens=args.eeg_target_tokens,
        conv_blocks=args.eeg_conv_blocks,
        transformer_layers=args.eeg_transformer_layers,
        transformer_heads=args.eeg_transformer_heads,
        transformer_mlp_ratio=args.eeg_transformer_mlp_ratio,
        dropout=args.eeg_dropout,
        use_cls_token=args.eeg_use_cls_token,
    )

    clip_projector = PretrainCLIPProjector(
        in_dim=args.eeg_token_dim,
        hidden_dim=args.projector_hidden_dim,
        out_dim=args.projector_out_dim,
        target_tokens=args.projector_target_tokens,
        adapter_type=args.projector_adapter_type,
        dropout=args.eeg_dropout,
    )

    eeg_token_backbone.requires_grad_(True)
    clip_projector.requires_grad_(True)

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    lr_main = float(args.learning_rate)
    backbone_lr = float(args.backbone_lr) if args.backbone_lr is not None else lr_main
    projector_lr = float(args.projector_lr) if args.projector_lr is not None else lr_main

    param_groups = [
        {
            "params": [p for p in eeg_token_backbone.parameters() if p.requires_grad],
            "lr": backbone_lr,
            "name": "eeg_token_backbone",
        },
        {
            "params": [p for p in clip_projector.parameters() if p.requires_grad],
            "lr": projector_lr,
            "name": "clip_projector",
        },
    ]

    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    if args.max_train_steps is not None:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    else:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    (
        eeg_token_backbone,
        clip_projector,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        eeg_token_backbone,
        clip_projector,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

    # Keep EEG modules in fp32; Accelerate handles mixed precision contexts.
    eeg_token_backbone.to(accelerator.device, dtype=torch.float32)
    clip_projector.to(accelerator.device, dtype=torch.float32)

    params_to_clip = []
    seen = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad and id(p) not in seen:
                params_to_clip.append(p)
                seen.add(id(p))

    if accelerator.is_main_process and args.report_to is not None:
        accelerator.init_trackers(
            args.tracker_project_name,
            config=sanitize_config_for_trackers(vars(args).copy()),
        )

    accelerator.print(
        f"[MODE] EEGTokenBackbone direct EEG→CLIP | "
        f"backbone_dim={args.eeg_token_dim} | tokens={args.eeg_target_tokens} | "
        f"projector={args.projector_adapter_type} | max_steps={args.max_train_steps}"
    )

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    global_step = 0

    progress_bar = tqdm(
        total=args.max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="Training EEGTokenBackbone EEG→CLIP",
    )

    debug_printed_once = False

    train_accum_modules = [eeg_token_backbone, clip_projector]

    for epoch in range(args.num_train_epochs):
        eeg_token_backbone.train()
        clip_projector.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(*train_accum_modules):
                eeg_cond = batch["conditioning_pixel_values"].to(
                    accelerator.device,
                    dtype=torch.float32,
                )
                clip_target = batch["clip_img_embeds"].to(
                    accelerator.device,
                    dtype=torch.float32,
                )

                if clip_target.ndim != 3:
                    raise RuntimeError(
                        f"Expected clip_img_embeds shape (B,T,D), got {tuple(clip_target.shape)}"
                    )

                visual_group_ids = get_group_ids_from_batch(
                    batch,
                    "visual_group_ids",
                    accelerator.device,
                )

                eeg_feats = eeg_token_backbone(eeg_cond.float())
                E_seq = eeg_feats["sequence"]
                pred_tokens = clip_projector(E_seq.float())

                if pred_tokens.shape != clip_target.shape:
                    raise RuntimeError(
                        f"pred_tokens shape {tuple(pred_tokens.shape)} != "
                        f"clip_target shape {tuple(clip_target.shape)}"
                    )

                loss = multi_positive_sequence_info_nce_loss(
                    pred_tokens.float(),
                    clip_target.float(),
                    group_ids=visual_group_ids,
                    temperature=args.temperature,
                    exclude_self=False,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients and params_to_clip:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                with torch.no_grad():
                    cos = paired_cosine(pred_tokens, clip_target)
                    mse = paired_mse(pred_tokens, clip_target)
                    l1 = paired_l1(pred_tokens, clip_target)
                    top1 = topk_group_retrieval(pred_tokens, clip_target, visual_group_ids, k=1)
                    top5 = topk_group_retrieval(pred_tokens, clip_target, visual_group_ids, k=5)
                    margin = retrieval_margin(pred_tokens, clip_target)

                    pred_norm = pred_tokens.float().reshape(pred_tokens.shape[0], -1).norm(dim=-1).mean()
                    target_norm = clip_target.float().reshape(clip_target.shape[0], -1).norm(dim=-1).mean()
                    seq_norm = E_seq.float().reshape(E_seq.shape[0], -1).norm(dim=-1).mean()

                values = {
                    "train/loss": loss.item(),
                    "train/cos": cos.item(),
                    "train/mse": mse.item(),
                    "train/l1": l1.item(),
                    "train/top1": top1.item(),
                    "train/top5": top5.item(),
                    "train/margin": margin.item() if not torch.isnan(margin) else 0.0,
                    "train/pred_norm": pred_norm.item(),
                    "train/target_norm": target_norm.item(),
                    "train/seq_norm": seq_norm.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": global_step / num_update_steps_per_epoch,
                }

                if not debug_printed_once and accelerator.is_main_process:
                    accelerator.print(f"[DEBUG] eeg_cond: {tuple(eeg_cond.shape)}")
                    accelerator.print(f"[DEBUG] E_seq: {tuple(E_seq.shape)}")
                    accelerator.print(f"[DEBUG] pred_tokens: {tuple(pred_tokens.shape)}")
                    accelerator.print(f"[DEBUG] clip_target: {tuple(clip_target.shape)}")
                    accelerator.print(f"[DEBUG] visual_group_ids: {tuple(visual_group_ids.shape)}")
                    debug_printed_once = True

                accelerator.log(values, step=global_step)

                if accelerator.is_main_process and (
                    global_step == 1 or global_step % args.console_log_every == 0
                ):
                    logger.info(
                        f"[step {global_step}] "
                        f"loss={values['train/loss']:.4f} | "
                        f"cos={values['train/cos']:.4f} | "
                        f"top1={values['train/top1']:.4f} | "
                        f"top5={values['train/top5']:.4f} | "
                        f"margin={values['train/margin']:.4f} | "
                        f"seq_norm={values['train/seq_norm']:.2f} | "
                        f"pred_norm={values['train/pred_norm']:.2f} | "
                        f"target_norm={values['train/target_norm']:.2f}"
                    )

                if global_step % args.validation_steps == 0:
                    val_metrics = run_validation(
                        val_dataloader=val_dataloader,
                        eeg_token_backbone=eeg_token_backbone,
                        clip_projector=clip_projector,
                        args=args,
                        accelerator=accelerator,
                    )
                    accelerator.log(val_metrics, step=global_step)

                    if accelerator.is_main_process:
                        logger.info(
                            f"[val step {global_step}] "
                            f"loss={val_metrics['val/loss']:.4f} | "
                            f"cos={val_metrics['val/cos']:.4f} | "
                            f"top1={val_metrics['val/top1']:.4f} | "
                            f"top5={val_metrics['val/top5']:.4f} | "
                            f"margin={val_metrics['val/margin']:.4f} | "
                            f"seq_norm={val_metrics['val/seq_norm']:.2f} | "
                            f"pred_norm={val_metrics['val/pred_norm']:.2f} | "
                            f"target_norm={val_metrics['val/target_norm']:.2f}"
                        )

                if global_step % args.checkpointing_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(checkpoint_dir)
                    if accelerator.is_main_process:
                        logger.info(f"[SAVE] checkpoint saved to {checkpoint_dir}")
                    rotate_checkpoints(
                        args.output_dir,
                        args.checkpoints_total_limit,
                        prefix="checkpoint",
                    )

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    # ---------------------------------------------------------
    # Final save
    # ---------------------------------------------------------
    if accelerator.is_main_process:
        unwrapped_backbone = accelerator.unwrap_model(eeg_token_backbone)
        unwrapped_projector = accelerator.unwrap_model(clip_projector)

        torch.save(
            unwrapped_backbone.state_dict(),
            os.path.join(args.output_dir, "eeg_token_backbone.pt"),
        )
        torch.save(
            unwrapped_projector.state_dict(),
            os.path.join(args.output_dir, "clip_projector.pt"),
        )

        save_json(
            os.path.join(args.output_dir, "eeg_token_backbone_config.json"),
            {
                "in_channels": int(args.eeg_in_channels),
                "input_time": int(args.eeg_input_time),
                "stem_dim": int(args.eeg_stem_dim),
                "token_dim": int(args.eeg_token_dim),
                "target_tokens": int(args.eeg_target_tokens),
                "conv_blocks": int(args.eeg_conv_blocks),
                "transformer_layers": int(args.eeg_transformer_layers),
                "transformer_heads": int(args.eeg_transformer_heads),
                "transformer_mlp_ratio": float(args.eeg_transformer_mlp_ratio),
                "dropout": float(args.eeg_dropout),
                "use_cls_token": bool(args.eeg_use_cls_token),
            },
        )

        save_json(
            os.path.join(args.output_dir, "clip_projector_config.json"),
            {
                "in_dim": int(args.eeg_token_dim),
                "hidden_dim": int(args.projector_hidden_dim),
                "out_dim": int(args.projector_out_dim),
                "target_tokens": int(args.projector_target_tokens),
                "adapter_type": args.projector_adapter_type,
                "temperature": float(args.temperature),
                "train_subjects": [int(s) for s in args.train_subjects],
                "val_subjects": [int(s) for s in args.val_subjects],
                "test_subjects": [int(s) for s in args.test_subjects],
                "use_image_hash_visual_ids": bool(args.use_image_hash_visual_ids),
                "visual_ids_root": args.visual_ids_root,
                "clip_embeds_dir": args.clip_embeds_dir,
            },
        )

        logger.info(f"[DONE] saved final model to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
