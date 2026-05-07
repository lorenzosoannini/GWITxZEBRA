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
from diffusers.utils import check_min_version, is_wandb_available
from tqdm.auto import tqdm

from data.eeg_dataset_clean import (
    make_train_dataset,
    make_val_dataset,
    make_collate_fn,
)
from models.eeg_backbone import GWITEEGBackbone, load_eeg_backbone_from_ckpt
from models.sife import SIFE
from models.eeg_reconstruction import EEGReconstructionDecoder
from models.ssfe import (
    SSFEProjector,
    PretrainCLIPProjector,
    multi_positive_info_nce_loss,
    multi_positive_sequence_info_nce_loss,
)
from models.prior import PriorNetwork, BrainDiffusionPrior

if is_wandb_available():
    import wandb  # noqa: F401

check_min_version("0.31.0.dev0")
logger = get_logger(__name__)


# ---------------------------------------------------------
# SUBJECT / BATCH UTILS
# ---------------------------------------------------------
def build_subject_remap(subjects):
    subjects = sorted(set(int(s) for s in subjects))
    return {subj: i for i, subj in enumerate(subjects)}


def remap_subject_targets(subjects_tensor: torch.Tensor, subject_to_local: dict) -> torch.Tensor:
    device = subjects_tensor.device
    remapped = [subject_to_local[int(s)] for s in subjects_tensor.detach().cpu().tolist()]
    return torch.tensor(remapped, device=device, dtype=torch.long)


def get_group_ids_from_batch(batch, key: str, device: torch.device) -> torch.Tensor:
    if key not in batch:
        raise KeyError(
            f"Missing required batch key '{key}'. "
            f"Make sure data/eeg_dataset_clean.py returns it in __getitem__."
        )

    group_ids = batch[key]
    if not isinstance(group_ids, torch.Tensor):
        group_ids = torch.tensor(group_ids, device=device, dtype=torch.long)
    else:
        group_ids = group_ids.to(device=device, dtype=torch.long)

    if group_ids.ndim != 1:
        raise ValueError(f"Expected {key} to be 1D after collate, got shape {tuple(group_ids.shape)}")

    return group_ids


def set_trainable(module, trainable: bool):
    if module is None:
        return
    module.requires_grad_(trainable)
    module.train() if trainable else module.eval()


# ---------------------------------------------------------
# VALIDATION METRICS
# ---------------------------------------------------------
def run_validation_metrics(
    val_dataloader,
    eeg_backbone,
    sife,
    recon_decoder,
    stage1_clip_projector,
    ssfe_projector,
    diffusion_prior,
    subject_to_local,
    args,
    accelerator,
):
    previous_modes = {
        "eeg_backbone": eeg_backbone.training,
        "sife": sife.training if sife is not None else None,
        "recon_decoder": recon_decoder.training if recon_decoder is not None else None,
        "stage1_clip_projector": stage1_clip_projector.training if stage1_clip_projector is not None else None,
        "ssfe_projector": ssfe_projector.training if ssfe_projector is not None else None,
        "diffusion_prior": diffusion_prior.training if diffusion_prior is not None else None,
    }

    eeg_backbone.eval()
    if sife is not None:
        sife.eval()
    if recon_decoder is not None:
        recon_decoder.eval()
    if stage1_clip_projector is not None:
        stage1_clip_projector.eval()
    if ssfe_projector is not None:
        ssfe_projector.eval()
    if diffusion_prior is not None:
        diffusion_prior.eval()

    device = accelerator.device
    names = [
        "total",
        "loss_subject_inv",
        "loss_subject_spec",
        "acc_subject_inv",
        "acc_subject_spec",
        "E_i_norm",
        "E_s_norm",
        "Ei_Es_norm_ratio",
        "Ei_Es_cos",
        "loss_recon",
        "stage1_clip_loss",
        "stage1_clip_top1",
        "loss_image_cls",
        "loss_image_dis",
        "acc_image_cls",
        "acc_image_dis",
        "anchor_cls_loss",
        "anchor_visual_loss",
        "anchor_visual_s_loss",
        "loss_anchor_text",
        "anchor_top1",
        "anchor_s_top1",
        "prior_loss",
    ]
    acc = {name: torch.tensor(0.0, device=device) for name in names}

    with torch.no_grad():
        for batch in val_dataloader:
            eeg_cond = batch["conditioning_pixel_values"].to(device, dtype=torch.float32)
            image_labels = batch["image_labels"].to(device, dtype=torch.long)

            bsz = eeg_cond.shape[0]
            bsz_t = torch.tensor(float(bsz), device=device)
            acc["total"] += bsz_t

            eeg_feats = eeg_backbone(eeg_cond.float())
            E_seq = eeg_feats["sequence"]

            E_i_seq = None
            F_s = None

            if sife is not None:
                sife_out = sife(E_seq)
                E_i_seq = sife_out["E_i_seq"]
                E_i = sife_out["E_i"]
                E_s = sife_out["E_s"]
                pred_subject_i = sife_out["pred_subject_i"]
                pred_subject_s = sife_out["pred_subject_s"]

                subject_targets = remap_subject_targets(
                    batch["eeg_subjects"].to(device),
                    subject_to_local,
                )

                loss_subject_inv = F.cross_entropy(pred_subject_i, subject_targets)
                loss_subject_spec = F.cross_entropy(pred_subject_s, subject_targets)
                acc_subject_inv = (pred_subject_i.argmax(dim=-1) == subject_targets).float().mean()
                acc_subject_spec = (pred_subject_s.argmax(dim=-1) == subject_targets).float().mean()

                E_i_norm = E_i.norm(dim=-1).mean()
                E_s_norm = E_s.norm(dim=-1).mean()
                Ei_Es_ratio = E_i_norm / (E_s_norm + 1e-8)
                Ei_Es_cos = F.cosine_similarity(E_i, E_s, dim=-1).mean()

                acc["loss_subject_inv"] += loss_subject_inv * bsz_t
                acc["loss_subject_spec"] += loss_subject_spec * bsz_t
                acc["acc_subject_inv"] += acc_subject_inv * bsz_t
                acc["acc_subject_spec"] += acc_subject_spec * bsz_t
                acc["E_i_norm"] += E_i_norm * bsz_t
                acc["E_s_norm"] += E_s_norm * bsz_t
                acc["Ei_Es_norm_ratio"] += Ei_Es_ratio * bsz_t
                acc["Ei_Es_cos"] += Ei_Es_cos * bsz_t

            if recon_decoder is not None:
                eeg_recon = recon_decoder(E_seq.float())
                if args.recon_loss_type == "mse":
                    loss_recon = F.mse_loss(eeg_recon, eeg_cond.float())
                elif args.recon_loss_type == "smooth_l1":
                    loss_recon = F.smooth_l1_loss(eeg_recon, eeg_cond.float())
                elif args.recon_loss_type == "l1":
                    loss_recon = F.l1_loss(eeg_recon, eeg_cond.float())
                else:
                    raise ValueError(f"Unsupported recon_loss_type: {args.recon_loss_type}")

                acc["loss_recon"] += loss_recon * bsz_t

            if getattr(args, "use_stage1_clip_pretrain", False):
                if stage1_clip_projector is None:
                    raise RuntimeError("Validation requested stage1 clip pretrain but projector is None.")

                clip_target = batch["clip_img_embeds"].to(device, dtype=torch.float32)
                visual_group_ids = get_group_ids_from_batch(batch, "visual_group_ids", device)

                stage1_clip_pred = stage1_clip_projector(E_seq.float())

                loss_stage1_clip = multi_positive_sequence_info_nce_loss(
                    stage1_clip_pred.float(),
                    clip_target.float(),
                    group_ids=visual_group_ids,
                    temperature=args.stage1_clip_temperature,
                    exclude_self=False,
                )

                acc["stage1_clip_loss"] += loss_stage1_clip * bsz_t

                pred_flat = F.normalize(
                    stage1_clip_pred.float().reshape(stage1_clip_pred.shape[0], -1),
                    dim=-1,
                )
                tgt_flat = F.normalize(
                    clip_target.float().reshape(clip_target.shape[0], -1),
                    dim=-1,
                )
                sim = pred_flat @ tgt_flat.t()
                top1_group = visual_group_ids[sim.argmax(dim=1)]
                top1 = (top1_group == visual_group_ids).float().mean()
                acc["stage1_clip_top1"] += top1 * bsz_t

            if ssfe_projector is not None:
                if E_i_seq is None:
                    raise RuntimeError("Validation SSFE requires E_i_seq from SIFE.")

                ssfe_out = ssfe_projector(E_i=E_i_seq.float(), E=E_seq.float())
                F_s = ssfe_out["F_s"]
                F_anchor_visual = ssfe_out["F_anchor_visual"]

                pred_image_cls = ssfe_out["pred_image_cls"]
                pred_image_dis = ssfe_out["pred_image_dis"]
                pred_image_cls_anchor = ssfe_out["pred_image_cls_anchor"]

                loss_image_cls = F.cross_entropy(pred_image_cls.float(), image_labels)
                loss_image_dis = F.cross_entropy(pred_image_dis.float(), image_labels)
                loss_anchor_cls = F.cross_entropy(pred_image_cls_anchor.float(), image_labels)

                acc_image_cls = (pred_image_cls.argmax(dim=-1) == image_labels).float().mean()
                acc_image_dis = (pred_image_dis.argmax(dim=-1) == image_labels).float().mean()

                acc["loss_image_cls"] += loss_image_cls * bsz_t
                acc["loss_image_dis"] += loss_image_dis * bsz_t
                acc["anchor_cls_loss"] += loss_anchor_cls * bsz_t
                acc["acc_image_cls"] += acc_image_cls * bsz_t
                acc["acc_image_dis"] += acc_image_dis * bsz_t

                if args.lambda_anchor_visual > 0.0 or args.lambda_anchor_visual_s > 0.0:
                    visual_group_ids = get_group_ids_from_batch(batch, "visual_group_ids", device)
                    clip_target = batch["clip_img_embeds"].to(device, dtype=torch.float32)

                    if args.lambda_anchor_visual > 0.0:
                        loss_anchor_visual = multi_positive_sequence_info_nce_loss(
                            F_anchor_visual.float(),
                            clip_target.float(),
                            group_ids=visual_group_ids,
                            temperature=args.anchor_visual_temperature,
                            exclude_self=False,
                        )
                        acc["anchor_visual_loss"] += loss_anchor_visual * bsz_t

                        pred_flat = F.normalize(
                            F_anchor_visual.float().reshape(F_anchor_visual.shape[0], -1),
                            dim=-1,
                        )
                        tgt_flat = F.normalize(
                            clip_target.float().reshape(clip_target.shape[0], -1),
                            dim=-1,
                        )
                        sim = pred_flat @ tgt_flat.t()
                        top1_group = visual_group_ids[sim.argmax(dim=1)]
                        top1 = (top1_group == visual_group_ids).float().mean()
                        acc["anchor_top1"] += top1 * bsz_t

                    if args.lambda_anchor_visual_s > 0.0:
                        loss_anchor_visual_s = multi_positive_sequence_info_nce_loss(
                            F_s.float(),
                            clip_target.float(),
                            group_ids=visual_group_ids,
                            temperature=args.anchor_visual_temperature,
                            exclude_self=False,
                        )
                        acc["anchor_visual_s_loss"] += loss_anchor_visual_s * bsz_t

                        pred_s_flat = F.normalize(
                            F_s.float().reshape(F_s.shape[0], -1),
                            dim=-1,
                        )
                        tgt_flat_s = F.normalize(
                            clip_target.float().reshape(clip_target.shape[0], -1),
                            dim=-1,
                        )
                        sim_s = pred_s_flat @ tgt_flat_s.t()
                        top1_s_group = visual_group_ids[sim_s.argmax(dim=1)]
                        top1_s = (top1_s_group == visual_group_ids).float().mean()
                        acc["anchor_s_top1"] += top1_s * bsz_t

                if args.lambda_anchor_text > 0.0:
                    text_group_ids = get_group_ids_from_batch(batch, "text_group_ids", device)
                    text_target = batch["clip_text_embeds"].to(device, dtype=torch.float32)

                    loss_anchor_text = multi_positive_info_nce_loss(
                        ssfe_out["anchor_text_embed"].float(),
                        text_target.float(),
                        group_ids=text_group_ids,
                        temperature=args.anchor_text_temperature,
                        exclude_self=False,
                    )
                    acc["loss_anchor_text"] += loss_anchor_text * bsz_t

                if diffusion_prior is not None:
                    clip_target = batch["clip_img_embeds"].to(device, dtype=torch.float32)
                    prior_loss, _ = diffusion_prior(
                        text_embed=F_s.float(),
                        image_embed=clip_target.float(),
                    )
                    acc["prior_loss"] += prior_loss * bsz_t

    gathered = accelerator.gather_for_metrics(
        torch.stack([acc[name] for name in names]).unsqueeze(0)
    )
    summed = gathered.sum(dim=0).view(-1)

    total_global = summed[0]
    if total_global.item() == 0:
        raise RuntimeError("Validation dataloader is empty: cannot compute validation metrics.")

    metrics = {
        "val/loss_subject_inv": (summed[1] / total_global).item(),
        "val/loss_subject_spec": (summed[2] / total_global).item(),
        "val/acc_subject_inv": (summed[3] / total_global).item(),
        "val/acc_subject_spec": (summed[4] / total_global).item(),
        "val/E_i_norm": (summed[5] / total_global).item(),
        "val/E_s_norm": (summed[6] / total_global).item(),
        "val/Ei_Es_norm_ratio": (summed[7] / total_global).item(),
        "val/Ei_Es_cos": (summed[8] / total_global).item(),
        "val/loss_recon": (summed[9] / total_global).item(),

        "val/stage1_clip_loss": (summed[10] / total_global).item(),
        "val/stage1_clip_top1": (summed[11] / total_global).item(),

        "val/loss_image_cls": (summed[12] / total_global).item(),
        "val/loss_image_dis": (summed[13] / total_global).item(),
        "val/acc_image_cls": (summed[14] / total_global).item(),
        "val/acc_image_dis": (summed[15] / total_global).item(),

        "val/anchor_cls_loss": (summed[16] / total_global).item(),
        "val/anchor_visual_loss": (summed[17] / total_global).item(),
        "val/anchor_visual_s_loss": (summed[18] / total_global).item(),
        "val/loss_anchor_text": (summed[19] / total_global).item(),
        "val/anchor_top1": (summed[20] / total_global).item(),
        "val/anchor_s_top1": (summed[21] / total_global).item(),
    }

    if diffusion_prior is not None:
        metrics["val/prior_loss"] = (summed[22] / total_global).item()

    eeg_backbone.train(previous_modes["eeg_backbone"])
    if sife is not None:
        sife.train(previous_modes["sife"])
    if recon_decoder is not None:
        recon_decoder.train(previous_modes["recon_decoder"])
    if stage1_clip_projector is not None:
        stage1_clip_projector.train(previous_modes["stage1_clip_projector"])
    if ssfe_projector is not None:
        ssfe_projector.train(previous_modes["ssfe_projector"])
    if diffusion_prior is not None:
        diffusion_prior.train(previous_modes["diffusion_prior"])

    return metrics


# ---------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description=(
            "EEG-only training: EEG backbone + optional SIFE/reconstruction/SSFE/prior. "
            "Uses data/eeg_dataset_clean.py. No GWIT diffusion branch."
        )
    )

    parser.add_argument("--output_dir", type=str, default="eeg-only-model")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--eeg_backbone_lr", type=float, default=None)
    parser.add_argument("--sife_lr", type=float, default=None)
    parser.add_argument("--recon_lr", type=float, default=None)
    parser.add_argument("--ssfe_lr", type=float, default=None)
    parser.add_argument("--prior_lr", type=float, default=None)

    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--console_log_every", type=int, default=10)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--tracker_project_name", type=str, default="eeg_only")

    # DATA
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--conditioning_image_column", type=str, default="conditioning_image")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_train_samples_per_subject", type=int, default=None)
    parser.add_argument("--max_val_samples_per_subject", type=int, default=None)
    parser.add_argument("--max_test_samples_per_subject", type=int, default=None)

    parser.add_argument("--train_subjects", type=int, nargs="+", required=True)
    parser.add_argument("--val_subjects", type=int, nargs="+", required=True)
    parser.add_argument("--test_subjects", type=int, nargs="+", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--validation_steps", type=int, default=100)

    # Precomputed CLIP embeds
    parser.add_argument("--use_precomputed_clip_embeds", action="store_true")
    parser.add_argument("--clip_embeds_dir", type=str, default=None)

    # EEG backbone
    parser.add_argument("--eeg_backbone_ckpt", type=str, default=None)
    parser.add_argument("--eeg_backbone_hidden_size", type=int, default=128)
    parser.add_argument("--eeg_backbone_num_layers", type=int, default=4)

    # SIFE
    parser.add_argument("--use_sife", action="store_true")
    parser.add_argument("--sife_num_layers", type=int, default=8)
    parser.add_argument("--sife_num_heads", type=int, default=4)
    parser.add_argument("--grl_lambda_sife", type=float, default=1.0)
    parser.add_argument("--lambda_subject_inv", type=float, default=1.0)
    parser.add_argument("--lambda_subject_spec", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing_sife", action="store_true")

    # EEG reconstruction
    parser.add_argument("--use_eeg_reconstruction", action="store_true")
    parser.add_argument("--recon_hidden_dim", type=int, default=256)
    parser.add_argument("--recon_num_blocks", type=int, default=3)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--recon_loss_type", type=str, default="l1", choices=["mse", "smooth_l1", "l1"])

    # Stage-1 CLIP visual pretraining
    parser.add_argument("--use_stage1_clip_pretrain", action="store_true")
    parser.add_argument("--stage1_clip_hidden_dim", type=int, default=256)
    parser.add_argument("--stage1_clip_out_dim", type=int, default=1664)
    parser.add_argument("--stage1_clip_target_tokens", type=int, default=256)
    parser.add_argument(
        "--stage1_clip_adapter_type",
        type=str,
        default="zebra_like",
        choices=["simple", "zebra_like"],
    )
    parser.add_argument("--stage1_clip_lr", type=float, default=None)
    parser.add_argument("--lambda_stage1_clip", type=float, default=1.0)
    parser.add_argument("--stage1_clip_temperature", type=float, default=0.07)
    parser.add_argument("--gradient_checkpointing_stage1_clip", action="store_true")
    parser.add_argument("--load_stage1_clip_projector_path", type=str, default=None)

    # SSFE
    parser.add_argument("--use_ssfe", action="store_true")
    parser.add_argument("--ssfe_hidden_dim", type=int, default=256)
    parser.add_argument("--ssfe_out_dim", type=int, default=1664)
    parser.add_argument("--ssfe_target_tokens", type=int, default=256)
    parser.add_argument("--ssfe_adapter_type", type=str, default="zebra_like", choices=["simple", "zebra_like"])
    parser.add_argument("--num_image_classes", type=int, default=40)
    parser.add_argument("--grl_lambda_ssfe", type=float, default=1.0)
    parser.add_argument("--lambda_ssfe", type=float, default=1.0)
    parser.add_argument("--lambda_image_cls", type=float, default=1.0)
    parser.add_argument("--lambda_image_dis", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing_ssfe", action="store_true")

    # Anchor losses
    parser.add_argument("--lambda_anchor_cls", type=float, default=0.0)
    parser.add_argument("--lambda_anchor_visual", type=float, default=0.0)
    parser.add_argument("--lambda_anchor_visual_s", type=float, default=0.0)
    parser.add_argument("--lambda_anchor_text", type=float, default=0.0)
    parser.add_argument("--anchor_visual_temperature", type=float, default=0.07)
    parser.add_argument("--anchor_text_temperature", type=float, default=0.07)

    # Prior
    parser.add_argument("--use_prior", action="store_true")
    parser.add_argument("--lambda_prior", type=float, default=1.0)
    parser.add_argument("--prior_num_tokens", type=int, default=None)
    parser.add_argument("--prior_dim", type=int, default=None)
    parser.add_argument("--prior_depth", type=int, default=6)
    parser.add_argument("--prior_heads", type=int, default=32)
    parser.add_argument("--prior_timesteps", type=int, default=100)
    parser.add_argument("--prior_cond_drop_prob", type=float, default=0.2)
    parser.add_argument("--gradient_checkpointing_prior", action="store_true")

    # Staged training
    parser.add_argument(
        "--training_stage",
        type=str,
        default="full",
        choices=["full", "stage1", "stage2", "stage2_joint", "stage3"],
    )
    parser.add_argument("--load_sife_path", type=str, default=None)
    parser.add_argument("--load_recon_path", type=str, default=None)
    parser.add_argument("--load_ssfe_path", type=str, default=None)
    parser.add_argument("--load_prior_path", type=str, default=None)

    args = parser.parse_args(input_args)

    if isinstance(args.report_to, str) and args.report_to.lower() in {"none", "null", "no"}:
        args.report_to = None

    if args.use_ssfe and not args.use_sife:
        raise ValueError("--use_ssfe requires --use_sife")
    if args.use_prior and not args.use_ssfe:
        raise ValueError("--use_prior requires --use_ssfe")
    if args.use_prior and not args.use_precomputed_clip_embeds:
        raise ValueError("--use_prior requires --use_precomputed_clip_embeds")
    if args.use_stage1_clip_pretrain and not args.use_precomputed_clip_embeds:
        raise ValueError("--use_stage1_clip_pretrain requires --use_precomputed_clip_embeds")

    if args.use_stage1_clip_pretrain and args.training_stage not in {"stage1", "full"}:
        raise ValueError("--use_stage1_clip_pretrain is intended for stage1 or full training")
    if (
        args.lambda_anchor_cls > 0.0
        or args.lambda_anchor_visual > 0.0
        or args.lambda_anchor_visual_s > 0.0
        or args.lambda_anchor_text > 0.0
    ) and not args.use_ssfe:
        raise ValueError("Anchor losses require --use_ssfe")
    if (
        args.lambda_anchor_visual > 0.0
        or args.lambda_anchor_visual_s > 0.0
    ) and not args.use_precomputed_clip_embeds:
        raise ValueError("--lambda_anchor_visual / --lambda_anchor_visual_s require --use_precomputed_clip_embeds")
    if args.lambda_anchor_text > 0.0 and not args.use_precomputed_clip_embeds:
        raise ValueError("--lambda_anchor_text requires --use_precomputed_clip_embeds")

    if args.training_stage == "stage2":
        if not args.use_sife or not args.use_ssfe:
            raise ValueError("stage2 requires --use_sife and --use_ssfe")
        if args.load_sife_path is None:
            raise ValueError("stage2 requires --load_sife_path")
        if args.eeg_backbone_ckpt is None:
            raise ValueError("stage2 requires --eeg_backbone_ckpt")
        
    if args.training_stage == "stage2_joint":
        if not args.use_sife:
            raise ValueError("stage2_joint requires --use_sife")
        if not args.use_ssfe:
            raise ValueError("stage2_joint requires --use_ssfe")
        if not args.use_prior:
            raise ValueError("stage2_joint requires --use_prior")
        if not args.use_precomputed_clip_embeds:
            raise ValueError("stage2_joint requires --use_precomputed_clip_embeds")
        if args.load_sife_path is None:
            raise ValueError("stage2_joint requires --load_sife_path")
        if args.eeg_backbone_ckpt is None:
            raise ValueError("stage2_joint requires --eeg_backbone_ckpt")

    if args.training_stage == "stage3":
        if not args.use_sife or not args.use_ssfe or not args.use_prior:
            raise ValueError("stage3 requires --use_sife, --use_ssfe and --use_prior")
        if args.load_sife_path is None or args.load_ssfe_path is None:
            raise ValueError("stage3 requires --load_sife_path and --load_ssfe_path")
        if args.eeg_backbone_ckpt is None:
            raise ValueError("stage3 requires --eeg_backbone_ckpt")

    return args


# ---------------------------------------------------------
# CHECKPOINT UTILS
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
        try:
            shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Error while removing checkpoint {path}: {e}")


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


def infer_clip_dims_from_dataset(train_dataset, args, accelerator):
    """
    Infer CLIP token count/dim from the first training example.

    Required by SSFE/prior because both expect sequence-level CLIP image tokens:
        clip_img_embeds: (T, D)
    """
    anchor_text_dim = 1280
    inferred_tokens = None
    inferred_clip_dim = None

    if args.use_ssfe or args.use_prior or args.use_stage1_clip_pretrain:
        if not args.use_precomputed_clip_embeds:
            raise ValueError("--use_ssfe/--use_prior requires --use_precomputed_clip_embeds")

        first_ex = train_dataset[0]

        if "clip_img_embeds" not in first_ex:
            raise RuntimeError("SSFE/prior requires clip_img_embeds in dataset.")

        clip_ex = first_ex["clip_img_embeds"]
        if clip_ex.ndim != 2:
            raise RuntimeError(
                f"Expected clip_img_embeds shape (T, D), got {tuple(clip_ex.shape)}. "
                f"Use sequence-level CLIP image embeddings, not pooled embeddings."
            )

        inferred_tokens, inferred_clip_dim = clip_ex.shape

        if args.use_ssfe or args.use_prior:
            if args.ssfe_target_tokens != inferred_tokens:
                raise ValueError(
                    f"ssfe_target_tokens={args.ssfe_target_tokens}, "
                    f"but dataset tokens={inferred_tokens}"
                )
            if args.ssfe_out_dim != inferred_clip_dim:
                raise ValueError(
                    f"ssfe_out_dim={args.ssfe_out_dim}, "
                    f"but dataset dim={inferred_clip_dim}"
                )

        if args.use_stage1_clip_pretrain:
            if args.stage1_clip_target_tokens != inferred_tokens:
                raise ValueError(
                    f"stage1_clip_target_tokens={args.stage1_clip_target_tokens}, "
                    f"but dataset tokens={inferred_tokens}"
                )
            if args.stage1_clip_out_dim != inferred_clip_dim:
                raise ValueError(
                    f"stage1_clip_out_dim={args.stage1_clip_out_dim}, "
                    f"but dataset dim={inferred_clip_dim}"
                )

        if "clip_text_embeds" in first_ex:
            anchor_text_dim = int(first_ex["clip_text_embeds"].shape[0])
        elif args.lambda_anchor_text > 0.0:
            raise RuntimeError("--lambda_anchor_text > 0 requires clip_text_embeds in dataset.")

        accelerator.print(
            f"[SSFE/CLIP] tokens={inferred_tokens} | "
            f"dim={inferred_clip_dim} | text_dim={anchor_text_dim}"
        )

    return anchor_text_dim, inferred_tokens, inferred_clip_dim


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    silence_external_loggers()
    logger.info(accelerator.state)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    active_train_subjects = sorted(set(int(s) for s in args.train_subjects))
    subject_to_local = build_subject_remap(active_train_subjects)

    train_dataset = make_train_dataset(args, accelerator)
    val_dataset = make_val_dataset(args, accelerator)
    collate_fn = make_collate_fn(args.dataset_name)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.val_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    accelerator.print(
        f"[DATA] train={len(train_dataset)} | val={len(val_dataset)} | "
        f"train_subjects={args.train_subjects} | val_subjects={args.val_subjects} | "
        f"test_subjects={args.test_subjects} | active_sife_classes={len(active_train_subjects)}"
    )

    eeg_backbone = GWITEEGBackbone(
        in_channels=128,
        hidden_size=args.eeg_backbone_hidden_size,
        num_layers=args.eeg_backbone_num_layers,
    )

    if args.eeg_backbone_ckpt is not None:
        load_eeg_backbone_from_ckpt(eeg_backbone, args.eeg_backbone_ckpt)
        accelerator.print(f"[LOAD] Loaded EEG backbone from {args.eeg_backbone_ckpt}")

    sample_cond = train_dataset[0]["conditioning_pixel_values"]
    if sample_cond.ndim == 2:
        dummy_eeg = sample_cond.unsqueeze(0).float()
    elif sample_cond.ndim == 3:
        dummy_eeg = sample_cond[:1].float()
    else:
        raise ValueError(f"Unexpected conditioning shape: {tuple(sample_cond.shape)}")

    eeg_backbone.eval()
    with torch.no_grad():
        inferred_seq_len = int(eeg_backbone(dummy_eeg)["sequence"].shape[1])
    eeg_backbone.train()

    accelerator.print(f"[AUTO] inferred seq_len = {inferred_seq_len}")

    sife = None
    if args.use_sife:
        sife = SIFE(
            dim=args.eeg_backbone_hidden_size,
            seq_len=inferred_seq_len,
            num_subjects=len(active_train_subjects),
            fi_layers=args.sife_num_layers,
            num_heads=args.sife_num_heads,
            grl_lambda=args.grl_lambda_sife,
        )

        if args.gradient_checkpointing_sife:
            sife.set_gradient_checkpointing(True)
            accelerator.print("[SIFE] gradient checkpointing enabled")

        if args.load_sife_path is not None:
            sife.load_state_dict(torch.load(args.load_sife_path, map_location="cpu"))
            accelerator.print(f"[LOAD] Loaded SIFE from {args.load_sife_path}")

    recon_decoder = None
    if args.use_eeg_reconstruction:
        recon_decoder = EEGReconstructionDecoder(
            in_dim=args.eeg_backbone_hidden_size,
            hidden_dim=args.recon_hidden_dim,
            out_channels=128,
            num_res_blocks=args.recon_num_blocks,
        )

        if args.load_recon_path is not None:
            recon_decoder.load_state_dict(torch.load(args.load_recon_path, map_location="cpu"))
            accelerator.print(f"[LOAD] Loaded recon decoder from {args.load_recon_path}")

    anchor_text_dim, inferred_clip_tokens, inferred_clip_dim = infer_clip_dims_from_dataset(
        train_dataset=train_dataset,
        args=args,
        accelerator=accelerator,
    )

    stage1_clip_projector = None
    if args.use_stage1_clip_pretrain:
        stage1_clip_projector = PretrainCLIPProjector(
            in_dim=args.eeg_backbone_hidden_size,
            hidden_dim=args.stage1_clip_hidden_dim,
            out_dim=args.stage1_clip_out_dim,
            target_tokens=args.stage1_clip_target_tokens,
            adapter_type=args.stage1_clip_adapter_type,
        )

        if args.gradient_checkpointing_stage1_clip:
            stage1_clip_projector.set_gradient_checkpointing(True)
            accelerator.print("[STAGE1 CLIP] gradient checkpointing enabled")

        if args.load_stage1_clip_projector_path is not None:
            stage1_clip_projector.load_state_dict(
                torch.load(args.load_stage1_clip_projector_path, map_location="cpu")
            )
            accelerator.print(
                f"[LOAD] Loaded stage1 CLIP projector from {args.load_stage1_clip_projector_path}"
            )

    ssfe_projector = None
    if args.use_ssfe:
        ssfe_projector = SSFEProjector(
            in_dim=args.eeg_backbone_hidden_size,
            hidden_dim=args.ssfe_hidden_dim,
            out_dim=args.ssfe_out_dim,
            target_tokens=args.ssfe_target_tokens,
            adapter_type=args.ssfe_adapter_type,
            num_image_classes=args.num_image_classes,
            grl_lambda=args.grl_lambda_ssfe,
            text_out_dim=anchor_text_dim,
        )

        if args.gradient_checkpointing_ssfe:
            ssfe_projector.set_gradient_checkpointing(True)
            accelerator.print("[SSFE] gradient checkpointing enabled")

        if args.load_ssfe_path is not None:
            ssfe_projector.load_state_dict(torch.load(args.load_ssfe_path, map_location="cpu"))
            accelerator.print(f"[LOAD] Loaded SSFE from {args.load_ssfe_path}")

    diffusion_prior = None
    prior_dim = None
    prior_num_tokens = None

    if args.use_prior:
        prior_num_tokens = (
            int(args.prior_num_tokens)
            if args.prior_num_tokens is not None
            else int(inferred_clip_tokens)
        )
        prior_dim = (
            int(args.prior_dim)
            if args.prior_dim is not None
            else int(args.ssfe_out_dim)
        )

        if prior_num_tokens != int(inferred_clip_tokens):
            raise ValueError(
                f"prior_num_tokens={prior_num_tokens}, "
                f"but dataset tokens={inferred_clip_tokens}"
            )
        if prior_dim != int(inferred_clip_dim):
            raise ValueError(
                f"prior_dim={prior_dim}, but dataset dim={inferred_clip_dim}"
            )

        prior_network = PriorNetwork(
            dim=prior_dim,
            num_tokens=prior_num_tokens,
            num_timesteps=int(args.prior_timesteps),
            depth=int(args.prior_depth),
            heads=int(args.prior_heads),
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

        if args.gradient_checkpointing_prior and hasattr(prior_network, "set_gradient_checkpointing"):
            prior_network.set_gradient_checkpointing(True)
            accelerator.print("[PRIOR] gradient checkpointing enabled")

        diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=prior_dim,
            timesteps=int(args.prior_timesteps),
            cond_drop_prob=float(args.prior_cond_drop_prob),
            predict_x_start=True,
            training_clamp_l2norm=False,
            sampling_clamp_l2norm=False,
            use_image_embed_scale=False,
        )

        if args.load_prior_path is not None:
            diffusion_prior.load_state_dict(torch.load(args.load_prior_path, map_location="cpu"))
            accelerator.print(f"[LOAD] Loaded prior from {args.load_prior_path}")

    is_stage1 = args.training_stage == "stage1"
    is_stage2 = args.training_stage == "stage2"
    is_stage2_joint = args.training_stage == "stage2_joint"
    is_stage3 = args.training_stage == "stage3"
    is_full = args.training_stage == "full"

    if is_stage1:
        set_trainable(eeg_backbone, True)
        set_trainable(sife, True)
        set_trainable(recon_decoder, True)
        set_trainable(stage1_clip_projector, True)
        set_trainable(ssfe_projector, False)
        set_trainable(diffusion_prior, False)
    elif is_stage2:
        set_trainable(eeg_backbone, False)
        set_trainable(sife, False)
        set_trainable(recon_decoder, False)
        set_trainable(stage1_clip_projector, False)
        set_trainable(ssfe_projector, True)
        set_trainable(diffusion_prior, False)
    elif is_stage2_joint:
        set_trainable(eeg_backbone, False)
        set_trainable(sife, False)
        set_trainable(recon_decoder, False)
        set_trainable(stage1_clip_projector, False)
        set_trainable(ssfe_projector, True)
        set_trainable(diffusion_prior, True)
    elif is_stage3:
        set_trainable(eeg_backbone, False)
        set_trainable(sife, False)
        set_trainable(recon_decoder, False)
        set_trainable(stage1_clip_projector, False)
        set_trainable(ssfe_projector, False)
        set_trainable(diffusion_prior, True)
    else:
        set_trainable(eeg_backbone, True)
        set_trainable(sife, True)
        set_trainable(recon_decoder, True)
        set_trainable(stage1_clip_projector, True)
        set_trainable(ssfe_projector, True)
        set_trainable(diffusion_prior, True)

    lr_main = float(args.learning_rate)
    lr_eeg = float(args.eeg_backbone_lr) if args.eeg_backbone_lr is not None else lr_main
    lr_sife = float(args.sife_lr) if args.sife_lr is not None else lr_main
    lr_recon = float(args.recon_lr) if args.recon_lr is not None else lr_main
    lr_stage1_clip = float(args.stage1_clip_lr) if args.stage1_clip_lr is not None else lr_main
    lr_ssfe = float(args.ssfe_lr) if args.ssfe_lr is not None else lr_main
    lr_prior = float(args.prior_lr) if args.prior_lr is not None else lr_main

    module_param_specs = [
        ("eeg", eeg_backbone, lr_eeg),
        ("sife", sife, lr_sife),
        ("recon", recon_decoder, lr_recon),
        ("stage1_clip", stage1_clip_projector, lr_stage1_clip),
        ("ssfe", ssfe_projector, lr_ssfe),
        ("prior", diffusion_prior, lr_prior),
    ]

    param_groups = []
    for name, module, lr in module_param_specs:
        if module is None:
            continue
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": lr, "name": name})

    if not param_groups:
        raise ValueError("No trainable modules found. Check --training_stage and enabled modules.")

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
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

    prepare_items = [eeg_backbone]
    if sife is not None:
        prepare_items.append(sife)
    if recon_decoder is not None:
        prepare_items.append(recon_decoder)
    if stage1_clip_projector is not None:
        prepare_items.append(stage1_clip_projector)
    if ssfe_projector is not None:
        prepare_items.append(ssfe_projector)
    if diffusion_prior is not None:
        prepare_items.append(diffusion_prior)
    prepare_items += [optimizer, train_dataloader, val_dataloader, lr_scheduler]

    prepared = accelerator.prepare(*prepare_items)

    idx = 0
    eeg_backbone = prepared[idx]
    idx += 1
    if sife is not None:
        sife = prepared[idx]
        idx += 1
    if recon_decoder is not None:
        recon_decoder = prepared[idx]
        idx += 1
    if stage1_clip_projector is not None:
        stage1_clip_projector = prepared[idx]
        idx += 1
    if ssfe_projector is not None:
        ssfe_projector = prepared[idx]
        idx += 1
    if diffusion_prior is not None:
        diffusion_prior = prepared[idx]
        idx += 1
    optimizer = prepared[idx]
    idx += 1
    train_dataloader = prepared[idx]
    idx += 1
    val_dataloader = prepared[idx]
    idx += 1
    lr_scheduler = prepared[idx]

    accum_models = []
    for module in [eeg_backbone, sife, recon_decoder, stage1_clip_projector, ssfe_projector, diffusion_prior]:
        if module is not None and any(p.requires_grad for p in module.parameters()):
            accum_models.append(module)

    params_to_clip = []
    seen = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad and id(p) not in seen:
                params_to_clip.append(p)
                seen.add(id(p))

    global_step = 0
    first_epoch = 0
    resume_step_in_epoch = 0

    if args.resume_from_checkpoint:
        if os.path.isdir(args.resume_from_checkpoint):
            resume_path = args.resume_from_checkpoint
        else:
            resume_path = os.path.join(args.output_dir, args.resume_from_checkpoint)
            if not os.path.isdir(resume_path):
                raise ValueError(f"Checkpoint folder {args.resume_from_checkpoint} not found.")

        accelerator.print(f"Resuming from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)

        basename = os.path.basename(resume_path)
        try:
            global_step = int(basename.split("-")[-1])
        except ValueError:
            global_step = 0

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step_in_epoch = global_step % num_update_steps_per_epoch

        accelerator.print(
            f"Parsed global_step={global_step}, first_epoch={first_epoch}, "
            f"resume_step_in_epoch={resume_step_in_epoch}"
        )

    # Keep trainable EEG modules in fp32. Mixed precision is still handled by Accelerator.
    for module in [eeg_backbone, sife, recon_decoder, stage1_clip_projector, ssfe_projector, diffusion_prior]:
        if module is not None:
            module.to(accelerator.device, dtype=torch.float32)

    if accelerator.is_main_process:
        if args.report_to is not None:
            accelerator.init_trackers(
                args.tracker_project_name,
                config=sanitize_config_for_trackers(vars(args).copy()),
            )

        accelerator.print(
            f"[MODE] EEG-only clean dataset | stage={args.training_stage} | "
            f"use_sife={args.use_sife} | use_recon={args.use_eeg_reconstruction} | "
            f"use_stage1_clip={args.use_stage1_clip_pretrain} | "
            f"use_ssfe={args.use_ssfe} | use_prior={args.use_prior}"
        )

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Training steps",
    )

    debug_printed_once = False

    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_sums = {}
        epoch_logged_steps = 0

        def add_epoch(name, value):
            epoch_sums[name] = epoch_sums.get(name, 0.0) + float(value)

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step_in_epoch:
                continue

            with accelerator.accumulate(*accum_models):
                eeg_cond = batch["conditioning_pixel_values"].to(accelerator.device, dtype=torch.float32)
                image_labels = batch["image_labels"].to(accelerator.device, dtype=torch.long)

                if is_stage2 or is_stage2_joint or is_stage3:
                    with torch.no_grad():
                        eeg_feats = eeg_backbone(eeg_cond.float())
                        E_seq = eeg_feats["sequence"]
                        E_pooled = eeg_feats["pooled"]

                        if sife is not None:
                            sife_out = sife(E_seq)
                            E_i_seq = sife_out["E_i_seq"]
                            E_s_seq = sife_out["E_s_seq"]
                            E_i = sife_out["E_i"]
                            E_s = sife_out["E_s"]
                            pred_subject_i = sife_out["pred_subject_i"]
                            pred_subject_s = sife_out["pred_subject_s"]
                        else:
                            E_i_seq = E_s_seq = E_i = E_s = pred_subject_i = pred_subject_s = None
                else:
                    eeg_feats = eeg_backbone(eeg_cond.float())
                    E_seq = eeg_feats["sequence"]
                    E_pooled = eeg_feats["pooled"]

                    if sife is not None:
                        sife_out = sife(E_seq)
                        E_i_seq = sife_out["E_i_seq"]
                        E_s_seq = sife_out["E_s_seq"]
                        E_i = sife_out["E_i"]
                        E_s = sife_out["E_s"]
                        pred_subject_i = sife_out["pred_subject_i"]
                        pred_subject_s = sife_out["pred_subject_s"]
                    else:
                        E_i_seq = E_s_seq = E_i = E_s = pred_subject_i = pred_subject_s = None

                zero = torch.tensor(0.0, device=accelerator.device)

                sife_loss_inv = zero
                sife_loss_spec = zero
                recon_loss = zero
                stage1_clip_loss = zero
                stage1_clip_top1 = zero
                image_cls_loss = zero
                image_dis_loss = zero
                anchor_cls_loss = zero
                anchor_visual_loss = zero
                anchor_visual_s_loss = zero
                anchor_text_loss = zero
                ssfe_loss = zero
                prior_loss = zero

                inv_acc = zero
                spec_acc = zero
                image_cls_acc = zero
                image_dis_acc = zero
                anchor_cls_acc = zero
                E_i_norm = zero
                E_s_norm = zero
                Ei_Es_norm_ratio = zero
                Ei_Es_cos = zero
                inv_entropy = zero
                spec_entropy = zero
                inv_entropy_norm = zero
                spec_entropy_norm = zero
                random_subject_acc = zero
                inv_acc_gap_vs_random = zero
                spec_acc_gap_vs_random = zero

                F_s = None

                if sife is not None and (is_stage1 or is_full):
                    subject_targets = remap_subject_targets(
                        batch["eeg_subjects"].to(accelerator.device),
                        subject_to_local,
                    )

                    sife_loss_inv = F.cross_entropy(pred_subject_i, subject_targets)
                    sife_loss_spec = F.cross_entropy(pred_subject_s, subject_targets)

                    inv_acc = (pred_subject_i.argmax(dim=-1) == subject_targets).float().mean()
                    spec_acc = (pred_subject_s.argmax(dim=-1) == subject_targets).float().mean()

                    E_i_norm = E_i.norm(dim=-1).mean()
                    E_s_norm = E_s.norm(dim=-1).mean()
                    Ei_Es_norm_ratio = E_i_norm / (E_s_norm + 1e-8)
                    Ei_Es_cos = F.cosine_similarity(E_i, E_s, dim=-1).mean()

                    p_i = torch.softmax(pred_subject_i, dim=-1)
                    p_s = torch.softmax(pred_subject_s, dim=-1)

                    inv_entropy = -(p_i * torch.log(p_i + 1e-8)).sum(dim=-1).mean()
                    spec_entropy = -(p_s * torch.log(p_s + 1e-8)).sum(dim=-1).mean()

                    max_entropy = torch.log(
                        torch.tensor(float(len(active_train_subjects)), device=accelerator.device)
                    )
                    inv_entropy_norm = inv_entropy / (max_entropy + 1e-8)
                    spec_entropy_norm = spec_entropy / (max_entropy + 1e-8)

                    random_subject_acc = torch.tensor(
                        1.0 / float(len(active_train_subjects)),
                        device=accelerator.device,
                    )
                    inv_acc_gap_vs_random = inv_acc - random_subject_acc
                    spec_acc_gap_vs_random = spec_acc - random_subject_acc

                if recon_decoder is not None and (is_stage1 or is_full):
                    eeg_recon = recon_decoder(E_seq.float())

                    if args.recon_loss_type == "mse":
                        recon_loss = F.mse_loss(eeg_recon, eeg_cond.float())
                    elif args.recon_loss_type == "smooth_l1":
                        recon_loss = F.smooth_l1_loss(eeg_recon, eeg_cond.float())
                    elif args.recon_loss_type == "l1":
                        recon_loss = F.l1_loss(eeg_recon, eeg_cond.float())
                    else:
                        raise ValueError(f"Unsupported recon_loss_type: {args.recon_loss_type}")
                    
                if stage1_clip_projector is not None and (is_stage1 or is_full):
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

                    stage1_clip_pred = stage1_clip_projector(E_seq.float())

                    stage1_clip_loss = multi_positive_sequence_info_nce_loss(
                        stage1_clip_pred.float(),
                        clip_target.float(),
                        group_ids=visual_group_ids,
                        temperature=args.stage1_clip_temperature,
                        exclude_self=False,
                    )

                    pred_flat = F.normalize(
                        stage1_clip_pred.float().reshape(stage1_clip_pred.shape[0], -1),
                        dim=-1,
                    )
                    tgt_flat = F.normalize(
                        clip_target.float().reshape(clip_target.shape[0], -1),
                        dim=-1,
                    )
                    sim = pred_flat @ tgt_flat.t()
                    top1_group = visual_group_ids[sim.argmax(dim=1)]
                    stage1_clip_top1 = (top1_group == visual_group_ids).float().mean()

                if ssfe_projector is not None and (is_stage2 or is_stage2_joint or is_full):
                    if E_i_seq is None:
                        raise RuntimeError("SSFE requires E_i_seq from SIFE.")

                    ssfe_out = ssfe_projector(E_i=E_i_seq.float(), E=E_seq.float())

                    F_s = ssfe_out["F_s"]
                    F_anchor_visual = ssfe_out["F_anchor_visual"]
                    anchor_text_embed = ssfe_out["anchor_text_embed"]

                    pred_image_cls = ssfe_out["pred_image_cls"]
                    pred_image_dis = ssfe_out["pred_image_dis"]
                    pred_image_cls_anchor = ssfe_out["pred_image_cls_anchor"]

                    image_cls_loss = F.cross_entropy(pred_image_cls.float(), image_labels)
                    image_dis_loss = F.cross_entropy(pred_image_dis.float(), image_labels)
                    anchor_cls_loss = F.cross_entropy(pred_image_cls_anchor.float(), image_labels)

                    image_cls_acc = (pred_image_cls.argmax(dim=-1) == image_labels).float().mean()
                    image_dis_acc = (pred_image_dis.argmax(dim=-1) == image_labels).float().mean()
                    anchor_cls_acc = (pred_image_cls_anchor.argmax(dim=-1) == image_labels).float().mean()

                    if args.lambda_anchor_visual > 0.0 or args.lambda_anchor_visual_s > 0.0:
                        visual_group_ids = get_group_ids_from_batch(
                            batch,
                            "visual_group_ids",
                            accelerator.device,
                        )
                        clip_target = batch["clip_img_embeds"].to(
                            accelerator.device,
                            dtype=torch.float32,
                        )

                        if clip_target.ndim != 3:
                            raise RuntimeError(
                                f"Expected clip_img_embeds shape (B,T,D), got {tuple(clip_target.shape)}"
                            )

                        if args.lambda_anchor_visual > 0.0:
                            anchor_visual_loss = multi_positive_sequence_info_nce_loss(
                                F_anchor_visual.float(),
                                clip_target.float(),
                                group_ids=visual_group_ids,
                                temperature=args.anchor_visual_temperature,
                                exclude_self=False,
                            )

                        if args.lambda_anchor_visual_s > 0.0:
                            anchor_visual_s_loss = multi_positive_sequence_info_nce_loss(
                                F_s.float(),
                                clip_target.float(),
                                group_ids=visual_group_ids,
                                temperature=args.anchor_visual_temperature,
                                exclude_self=False,
                            )

                    if args.lambda_anchor_text > 0.0:
                        text_group_ids = get_group_ids_from_batch(
                            batch,
                            "text_group_ids",
                            accelerator.device,
                        )
                        text_target = batch["clip_text_embeds"].to(
                            accelerator.device,
                            dtype=torch.float32,
                        )

                        anchor_text_loss = multi_positive_info_nce_loss(
                            anchor_text_embed.float(),
                            text_target.float(),
                            group_ids=text_group_ids,
                            temperature=args.anchor_text_temperature,
                            exclude_self=False,
                        )

                    ssfe_loss = (
                        float(args.lambda_image_cls) * image_cls_loss
                        + float(args.lambda_image_dis) * image_dis_loss
                        + float(args.lambda_anchor_cls) * anchor_cls_loss
                        + float(args.lambda_anchor_visual) * anchor_visual_loss
                        + float(args.lambda_anchor_visual_s) * anchor_visual_s_loss
                        + float(args.lambda_anchor_text) * anchor_text_loss
                    )

                elif ssfe_projector is not None and is_stage3:
                    if E_i_seq is None:
                        raise RuntimeError("Stage3 prior requires E_i_seq from SIFE.")

                    with torch.no_grad():
                        F_s = ssfe_projector(E_i=E_i_seq.float(), E=E_seq.float())["F_s"]

                if diffusion_prior is not None and (is_stage3 or is_stage2_joint or is_full):
                    if F_s is None:
                        raise RuntimeError("Prior requires F_s from SSFE.")

                    clip_target = batch["clip_img_embeds"].to(
                        accelerator.device,
                        dtype=torch.float32,
                    )

                    if clip_target.ndim != 3:
                        raise RuntimeError(
                            f"Expected clip_img_embeds shape (B,T,D), got {tuple(clip_target.shape)}"
                        )

                    prior_loss, prior_pred = diffusion_prior(
                        text_embed=F_s.float(),
                        image_embed=clip_target.float(),
                    )

                if not debug_printed_once and accelerator.is_main_process:
                    accelerator.print(f"[DEBUG] E_seq: {tuple(E_seq.shape)} | E_pooled: {tuple(E_pooled.shape)}")
                    if E_i_seq is not None:
                        accelerator.print(f"[DEBUG] E_i_seq: {tuple(E_i_seq.shape)} | E_s_seq: {tuple(E_s_seq.shape)}")
                    if ssfe_projector is not None and F_s is not None:
                        accelerator.print(f"[DEBUG] F_s: {tuple(F_s.shape)}")
                    debug_printed_once = True

                loss_total = zero

                if sife is not None and (is_stage1 or is_full):
                    loss_total = loss_total + float(args.lambda_subject_inv) * sife_loss_inv
                    loss_total = loss_total + float(args.lambda_subject_spec) * sife_loss_spec

                if recon_decoder is not None and (is_stage1 or is_full):
                    loss_total = loss_total + float(args.lambda_recon) * recon_loss

                if stage1_clip_projector is not None and (is_stage1 or is_full):
                    loss_total = loss_total + float(args.lambda_stage1_clip) * stage1_clip_loss

                if ssfe_projector is not None and (is_stage2 or is_stage2_joint or is_full):
                    loss_total = loss_total + float(args.lambda_ssfe) * ssfe_loss

                if diffusion_prior is not None and (is_stage3 or is_stage2_joint or is_full):
                    loss_total = loss_total + float(args.lambda_prior) * prior_loss

                accelerator.backward(loss_total)

                if accelerator.sync_gradients and params_to_clip:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                current_epoch = global_step / num_update_steps_per_epoch
                epoch_logged_steps += 1

                values = {
                    "loss": loss_total.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "E_norm": E_pooled.norm(dim=-1).mean().item(),
                    "loss_subject_inv": sife_loss_inv.item(),
                    "loss_subject_spec": sife_loss_spec.item(),
                    "acc_subject_inv": inv_acc.item(),
                    "acc_subject_spec": spec_acc.item(),
                    "E_i_norm": E_i_norm.item(),
                    "E_s_norm": E_s_norm.item(),
                    "Ei_Es_norm_ratio": Ei_Es_norm_ratio.item(),
                    "Ei_Es_cos": Ei_Es_cos.item(),
                    "entropy_subject_inv": inv_entropy.item(),
                    "entropy_subject_spec": spec_entropy.item(),
                    "entropy_subject_inv_norm": inv_entropy_norm.item(),
                    "entropy_subject_spec_norm": spec_entropy_norm.item(),
                    "random_subject_acc": random_subject_acc.item(),
                    "acc_subject_inv_gap_vs_random": inv_acc_gap_vs_random.item(),
                    "acc_subject_spec_gap_vs_random": spec_acc_gap_vs_random.item(),
                    "loss_recon": recon_loss.item(),
                    "loss_stage1_clip": stage1_clip_loss.item(),
                    "stage1_clip_top1": stage1_clip_top1.item(),
                    "loss_ssfe": ssfe_loss.item(),
                    "loss_image_cls": image_cls_loss.item(),
                    "loss_image_dis": image_dis_loss.item(),
                    "acc_image_cls": image_cls_acc.item(),
                    "acc_image_dis": image_dis_acc.item(),
                    "loss_anchor_cls": anchor_cls_loss.item(),
                    "acc_anchor_cls": anchor_cls_acc.item(),
                    "loss_anchor_visual": anchor_visual_loss.item(),
                    "loss_anchor_visual_s": anchor_visual_s_loss.item(),
                    "loss_anchor_text": anchor_text_loss.item(),
                    "loss_prior": prior_loss.item(),
                    "epoch": current_epoch,
                }

                for k, v in values.items():
                    if k != "epoch":
                        add_epoch(k, v)

                if accelerator.is_main_process and (
                    global_step % args.console_log_every == 0 or global_step == 1
                ):
                    logger.info(
                        f"[step {global_step}] "
                        f"loss={values['loss']:.4f} | "
                        f"inv={values['loss_subject_inv']:.4f} | "
                        f"spec={values['loss_subject_spec']:.4f} | "
                        f"inv_acc={values['acc_subject_inv']:.4f} | "
                        f"spec_acc={values['acc_subject_spec']:.4f} | "
                        f"recon={values['loss_recon']:.4f} | "
                        f"stage1_clip={values['loss_stage1_clip']:.4f} | "
                        f"stage1_top1={values['stage1_clip_top1']:.4f} | "
                        f"ssfe={values['loss_ssfe']:.4f} | "
                        f"img_cls={values['loss_image_cls']:.4f} | "
                        f"img_dis={values['loss_image_dis']:.4f} | "
                        f"anchor_v={values['loss_anchor_visual']:.4f} | "
                        f"anchor_vs={values['loss_anchor_visual_s']:.4f} | "
                        f"anchor_t={values['loss_anchor_text']:.4f} | "
                        f"prior={values['loss_prior']:.4f}"
                    )

                accelerator.log(values, step=global_step)

                if global_step % args.validation_steps == 0:
                    val_metrics = run_validation_metrics(
                        val_dataloader,
                        eeg_backbone,
                        sife,
                        recon_decoder,
                        stage1_clip_projector,
                        ssfe_projector,
                        diffusion_prior,
                        subject_to_local,
                        args,
                        accelerator,
                    )

                    if accelerator.is_main_process:
                        val_metrics["epoch"] = current_epoch
                        accelerator.log(val_metrics, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    final_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    tmp_path = final_path + "_tmp"

                    if os.path.exists(tmp_path):
                        import shutil
                        shutil.rmtree(tmp_path)

                    accelerator.save_state(tmp_path, safe_serialization=False)
                    os.replace(tmp_path, final_path)

                    logger.info(f"[SAFE SAVE] Saved checkpoint to {final_path}")
                    rotate_checkpoints(
                        args.output_dir,
                        args.checkpoints_total_limit,
                        prefix="checkpoint",
                    )

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process and epoch_logged_steps > 0:
            epoch_payload = {"epoch_only/epoch": float(epoch + 1)}
            for k, v in epoch_sums.items():
                epoch_payload[f"epoch_only/{k}"] = v / epoch_logged_steps
            accelerator.log(epoch_payload, step=global_step)

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        eeg_unwrapped = accelerator.unwrap_model(eeg_backbone)
        torch.save(eeg_unwrapped.state_dict(), os.path.join(args.output_dir, "eeg_backbone.pt"))
        save_json(
            os.path.join(args.output_dir, "eeg_backbone_config.json"),
            {
                "in_channels": 128,
                "hidden_size": int(args.eeg_backbone_hidden_size),
                "num_layers": int(args.eeg_backbone_num_layers),
            },
        )

        if sife is not None:
            sife_unwrapped = accelerator.unwrap_model(sife)
            torch.save(sife_unwrapped.state_dict(), os.path.join(args.output_dir, "sife.pt"))
            save_json(
                os.path.join(args.output_dir, "sife_config.json"),
                {
                    "dim": int(args.eeg_backbone_hidden_size),
                    "seq_len": int(inferred_seq_len),
                    "num_subjects": int(len(active_train_subjects)),
                    "train_subjects": active_train_subjects,
                    "fi_layers": int(args.sife_num_layers),
                    "num_heads": int(args.sife_num_heads),
                    "grl_lambda_sife": float(args.grl_lambda_sife),
                },
            )

        if recon_decoder is not None:
            recon_unwrapped = accelerator.unwrap_model(recon_decoder)
            torch.save(
                recon_unwrapped.state_dict(),
                os.path.join(args.output_dir, "eeg_reconstruction_decoder.pt"),
            )
            save_json(
                os.path.join(args.output_dir, "eeg_reconstruction_decoder_config.json"),
                {
                    "in_dim": int(args.eeg_backbone_hidden_size),
                    "hidden_dim": int(args.recon_hidden_dim),
                    "out_channels": 128,
                    "num_res_blocks": int(args.recon_num_blocks),
                    "loss_type": args.recon_loss_type,
                    "lambda_recon": float(args.lambda_recon),
                },
            )

        if stage1_clip_projector is not None:
            stage1_clip_unwrapped = accelerator.unwrap_model(stage1_clip_projector)
            torch.save(
                stage1_clip_unwrapped.state_dict(),
                os.path.join(args.output_dir, "stage1_clip_projector.pt"),
            )
            save_json(
                os.path.join(args.output_dir, "stage1_clip_projector_config.json"),
                {
                    "in_dim": int(args.eeg_backbone_hidden_size),
                    "hidden_dim": int(args.stage1_clip_hidden_dim),
                    "out_dim": int(args.stage1_clip_out_dim),
                    "target_tokens": int(args.stage1_clip_target_tokens),
                    "adapter_type": args.stage1_clip_adapter_type,
                    "lambda_stage1_clip": float(args.lambda_stage1_clip),
                    "stage1_clip_temperature": float(args.stage1_clip_temperature),
                },
            )

        if ssfe_projector is not None:
            ssfe_unwrapped = accelerator.unwrap_model(ssfe_projector)
            torch.save(ssfe_unwrapped.state_dict(), os.path.join(args.output_dir, "ssfe_projector.pt"))
            save_json(
                os.path.join(args.output_dir, "ssfe_projector_config.json"),
                {
                    "in_dim": int(args.eeg_backbone_hidden_size),
                    "hidden_dim": int(args.ssfe_hidden_dim),
                    "out_dim": int(args.ssfe_out_dim),
                    "target_tokens": int(args.ssfe_target_tokens),
                    "adapter_type": args.ssfe_adapter_type,
                    "num_image_classes": int(args.num_image_classes),
                    "grl_lambda_ssfe": float(args.grl_lambda_ssfe),
                    "text_out_dim": int(anchor_text_dim),
                    "lambda_ssfe": float(args.lambda_ssfe),
                    "lambda_image_cls": float(args.lambda_image_cls),
                    "lambda_image_dis": float(args.lambda_image_dis),
                    "lambda_anchor_cls": float(args.lambda_anchor_cls),
                    "lambda_anchor_visual": float(args.lambda_anchor_visual),
                    "lambda_anchor_visual_s": float(args.lambda_anchor_visual_s),
                    "lambda_anchor_text": float(args.lambda_anchor_text),
                    "anchor_visual_temperature": float(args.anchor_visual_temperature),
                    "anchor_text_temperature": float(args.anchor_text_temperature),
                },
            )

        if diffusion_prior is not None:
            prior_unwrapped = accelerator.unwrap_model(diffusion_prior)
            torch.save(prior_unwrapped.state_dict(), os.path.join(args.output_dir, "diffusion_prior.pt"))
            save_json(
                os.path.join(args.output_dir, "diffusion_prior_config.json"),
                {
                    "use_prior": True,
                    "lambda_prior": float(args.lambda_prior),
                    "prior_num_tokens": int(prior_num_tokens),
                    "prior_dim": int(prior_dim),
                    "prior_depth": int(args.prior_depth),
                    "prior_heads": int(args.prior_heads),
                    "prior_timesteps": int(args.prior_timesteps),
                    "prior_cond_drop_prob": float(args.prior_cond_drop_prob),
                    "use_image_embed_scale": False,
                },
            )

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
