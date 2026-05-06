import argparse
import contextlib
import gc
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    PretrainedConfig,
)

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from data.eeg_dataset import (
    make_train_dataset,
    make_val_dataset,
    make_test_dataset,
    make_collate_fn,
)
from models.eeg_backbone import GWITEEGBackbone, load_eeg_backbone_from_ckpt
from models.sife import SIFE
from models.eeg_reconstruction import EEGReconstructionDecoder
from models.ssfe import SSFEProjector, info_nce_loss, sequence_info_nce_loss
from models.prior import PriorNetwork, BrainDiffusionPrior

# W&B
if is_wandb_available():
    import wandb

check_min_version("0.31.0.dev0")
logger = get_logger(__name__)

VAE_SCALE_FACTOR = 0.18215


# ---------------------------------------------------------
# TEXT ENCODER IMPORT
# ---------------------------------------------------------
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


# ---------------------------------------------------------
# SUBJECT LABEL UTILS
# ---------------------------------------------------------
def build_subject_remap(subjects):
    subjects = sorted(set(int(s) for s in subjects))
    return {subj: i for i, subj in enumerate(subjects)}


def remap_subject_targets(subjects_tensor: torch.Tensor, subject_to_local: dict) -> torch.Tensor:
    device = subjects_tensor.device
    remapped = [subject_to_local[int(s)] for s in subjects_tensor.detach().cpu().tolist()]
    return torch.tensor(remapped, device=device, dtype=torch.long)


# ---------------------------------------------------------
# VALIDATION / TEST GENERATION
# ---------------------------------------------------------
def log_validation(
    vae,
    text_encoder,
    tokenizer,
    unet,
    controlnet,
    args,
    accelerator,
    weight_dtype,
    step,
    is_final_validation=False,
):
    logger.info("Running validation...")

    gc.collect()
    torch.cuda.empty_cache()

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
    else:
        controlnet = ControlNetModel.from_pretrained(
            args.output_dir,
            torch_dtype=weight_dtype,
        )

    pipeline_kwargs = dict(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    if vae is not None:
        pipeline_kwargs["vae"] = vae

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        **pipeline_kwargs,
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if is_final_validation:
        dataset_eval = make_test_dataset(args, tokenizer, accelerator)
        tracker_key = "test"
    else:
        dataset_eval = make_val_dataset(args, tokenizer, accelerator)
        tracker_key = "validation"

    n_examples = min(5, len(dataset_eval))
    image_logs = []

    if accelerator.mixed_precision != "no":
        inference_ctx = torch.autocast(
            accelerator.device.type,
            dtype=weight_dtype,
        )
    else:
        inference_ctx = contextlib.nullcontext()

    for i in range(n_examples):
        ex = dataset_eval[i]

        conditioning = ex["conditioning_pixel_values"].unsqueeze(0).to(
            accelerator.device,
            dtype=weight_dtype,
        )

        validation_prompt = tokenizer.decode(
            ex["input_ids"],
            skip_special_tokens=True,
        ).strip()

        real_image = None
        if "pixel_values" in ex:
            real_image = ex["pixel_values"].unsqueeze(0)

        images = []
        for _ in range(args.num_validation_images):
            with inference_ctx:
                out = pipeline(
                    prompt=validation_prompt,
                    image=conditioning,
                    added_cond_kwargs={},
                    num_inference_steps=20,
                    generator=generator,
                )
            images.append(out.images[0])

        image_logs.append(
            {
                "validation_image": real_image,
                "images": images,
                "validation_prompt": validation_prompt,
                "subject": int(ex["eeg_subjects"].item()),
            }
        )

        gc.collect()
        torch.cuda.empty_cache()

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb_images = []
            for log in image_logs:
                validation_image = log["validation_image"]
                validation_prompt = log["validation_prompt"]
                subject = log["subject"]

                if validation_image is not None:
                    v = validation_image[0].detach().cpu().permute(1, 2, 0).numpy()
                    wandb_images.append(
                        wandb.Image(v, caption=f"GT | subj={subject}")
                    )

                for gen in log["images"]:
                    wandb_images.append(
                        wandb.Image(gen, caption=f"subj={subject} | {validation_prompt}")
                    )

            tracker.log({tracker_key: wandb_images}, step=step)

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs

def run_validation_metrics(
    val_dataloader,
    eeg_backbone,
    sife,
    recon_decoder,
    ssfe_projector,
    diffusion_prior,
    subject_to_local,
    args,
    accelerator,
):
    prev_eeg_training = eeg_backbone.training
    prev_sife_training = sife.training if sife is not None else None
    prev_recon_training = recon_decoder.training if recon_decoder is not None else None
    prev_ssfe_training = ssfe_projector.training if ssfe_projector is not None else None
    prev_prior_training = diffusion_prior.training if diffusion_prior is not None else None

    eeg_backbone.eval()
    if sife is not None:
        sife.eval()
    if recon_decoder is not None:
        recon_decoder.eval()
    if ssfe_projector is not None:
        ssfe_projector.eval()
    if diffusion_prior is not None:
        diffusion_prior.eval()

    device = accelerator.device

    total = torch.tensor(0.0, device=device)

    # --- accumulators as tensors ---
    loss_subject_inv_tot = torch.tensor(0.0, device=device)
    loss_subject_spec_tot = torch.tensor(0.0, device=device)
    acc_subject_inv_tot = torch.tensor(0.0, device=device)
    acc_subject_spec_tot = torch.tensor(0.0, device=device)
    Ei_norm_tot = torch.tensor(0.0, device=device)
    Es_norm_tot = torch.tensor(0.0, device=device)
    Ei_Es_ratio_tot = torch.tensor(0.0, device=device)
    Ei_Es_cos_tot = torch.tensor(0.0, device=device)

    loss_recon_tot = torch.tensor(0.0, device=device)

    loss_image_cls_tot = torch.tensor(0.0, device=device)
    loss_image_dis_tot = torch.tensor(0.0, device=device)
    acc_image_cls_tot = torch.tensor(0.0, device=device)
    acc_image_dis_tot = torch.tensor(0.0, device=device)

    loss_anchor_cls_tot = torch.tensor(0.0, device=device)
    loss_anchor_visual_tot = torch.tensor(0.0, device=device)
    loss_anchor_text_tot = torch.tensor(0.0, device=device)
    top1_anchor_tot = torch.tensor(0.0, device=device)

    loss_prior_tot = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in val_dataloader:
            eeg_cond = batch["conditioning_pixel_values"].to(
                accelerator.device,
                dtype=torch.float32,
            )
            image_labels = batch["image_labels"].to(
                accelerator.device,
                dtype=torch.long,
            )

            bsz = eeg_cond.shape[0]
            bsz_t = total.new_tensor(float(bsz))
            total += bsz_t

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
                    batch["eeg_subjects"].to(accelerator.device),
                    subject_to_local,
                )

                loss_subject_inv = F.cross_entropy(pred_subject_i, subject_targets)
                loss_subject_spec = F.cross_entropy(pred_subject_s, subject_targets)

                acc_subject_inv = (pred_subject_i.argmax(dim=-1) == subject_targets).float().mean()
                acc_subject_spec = (pred_subject_s.argmax(dim=-1) == subject_targets).float().mean()

                Ei_norm = E_i.norm(dim=-1).mean()
                Es_norm = E_s.norm(dim=-1).mean()
                Ei_Es_ratio = Ei_norm / (Es_norm + 1e-8)
                Ei_Es_cos = F.cosine_similarity(E_i, E_s, dim=-1).mean()

                loss_subject_inv_tot += loss_subject_inv * bsz_t
                loss_subject_spec_tot += loss_subject_spec * bsz_t
                acc_subject_inv_tot += acc_subject_inv * bsz_t
                acc_subject_spec_tot += acc_subject_spec * bsz_t
                Ei_norm_tot += Ei_norm * bsz_t
                Es_norm_tot += Es_norm * bsz_t
                Ei_Es_ratio_tot += Ei_Es_ratio * bsz_t
                Ei_Es_cos_tot += Ei_Es_cos * bsz_t

            if recon_decoder is not None:
                eeg_recon = recon_decoder(E_seq.float())
                eeg_target = eeg_cond.float()

                if args.recon_loss_type == "mse":
                    loss_recon = F.mse_loss(eeg_recon, eeg_target)
                elif args.recon_loss_type == "smooth_l1":
                    loss_recon = F.smooth_l1_loss(eeg_recon, eeg_target)
                elif args.recon_loss_type == "l1":
                    loss_recon = F.l1_loss(eeg_recon, eeg_target)
                else:
                    raise ValueError(f"Unsupported recon_loss_type: {args.recon_loss_type}")

                loss_recon_tot += loss_recon * bsz_t

            if ssfe_projector is not None:
                if E_i_seq is None:
                    raise RuntimeError("Validation SSFE requires E_i_seq from SIFE.")

                ssfe_param_dtype = next(ssfe_projector.parameters()).dtype
                prior_param_dtype = (
                    next(diffusion_prior.parameters()).dtype
                    if diffusion_prior is not None
                    else ssfe_param_dtype
                )

                ssfe_out = ssfe_projector(
                    E_i=E_i_seq.to(dtype=ssfe_param_dtype),
                    E=E_seq.to(dtype=ssfe_param_dtype),
                )

                F_s = ssfe_out["F_s"]
                F_anchor_visual = ssfe_out["F_anchor_visual"]

                pred_image_cls = ssfe_out["pred_image_cls"]
                pred_image_dis = ssfe_out["pred_image_dis"]
                pred_image_cls_anchor = ssfe_out["pred_image_cls_anchor"]

                loss_image_cls = F.cross_entropy(pred_image_cls.float(), image_labels)
                loss_image_dis = F.cross_entropy(pred_image_dis.float(), image_labels)

                acc_image_cls = (pred_image_cls.argmax(dim=-1) == image_labels).float().mean()
                acc_image_dis = (pred_image_dis.argmax(dim=-1) == image_labels).float().mean()

                loss_image_cls_tot += loss_image_cls * bsz_t
                loss_image_dis_tot += loss_image_dis * bsz_t
                acc_image_cls_tot += acc_image_cls * bsz_t
                acc_image_dis_tot += acc_image_dis * bsz_t

                loss_anchor_cls = F.cross_entropy(pred_image_cls_anchor.float(), image_labels)
                loss_anchor_cls_tot += loss_anchor_cls * bsz_t

                if args.lambda_anchor_visual > 0.0:
                    clip_target_ssfe = batch["clip_img_embeds"].to(
                        accelerator.device,
                        dtype=ssfe_param_dtype,
                    )

                    loss_anchor_visual = sequence_info_nce_loss(
                        F_anchor_visual.float(),
                        clip_target_ssfe.float(),
                        temperature=args.anchor_visual_temperature,
                    )

                    loss_anchor_visual_tot += loss_anchor_visual * bsz_t

                    pred_flat = F.normalize(
                        F_anchor_visual.float().reshape(F_anchor_visual.shape[0], -1),
                        dim=-1,
                    )
                    tgt_flat = F.normalize(
                        clip_target_ssfe.float().reshape(clip_target_ssfe.shape[0], -1),
                        dim=-1,
                    )

                    sim = pred_flat @ tgt_flat.t()
                    top1 = (
                        sim.argmax(dim=1)
                        == torch.arange(sim.shape[0], device=sim.device)
                    ).float().mean()

                    top1_anchor_tot += top1 * bsz_t

                if args.lambda_anchor_text > 0.0:
                    text_target = batch["clip_text_embeds"].to(
                        accelerator.device,
                        dtype=ssfe_param_dtype,
                    )

                    loss_anchor_text = info_nce_loss(
                        ssfe_out["anchor_text_embed"].float(),
                        text_target.float(),
                        temperature=args.anchor_text_temperature,
                    )

                    loss_anchor_text_tot += loss_anchor_text * bsz_t

                if diffusion_prior is not None:
                    clip_target_prior = batch["clip_img_embeds"].to(
                        accelerator.device,
                        dtype=prior_param_dtype,
                    )

                    prior_loss, _ = diffusion_prior(
                        text_embed=F_s.to(dtype=prior_param_dtype),
                        image_embed=clip_target_prior,
                    )
                    loss_prior_tot += prior_loss * bsz_t

    gathered = accelerator.gather_for_metrics(
        torch.stack([
            total,
            loss_subject_inv_tot,
            loss_subject_spec_tot,
            acc_subject_inv_tot,
            acc_subject_spec_tot,
            Ei_norm_tot,
            Es_norm_tot,
            Ei_Es_ratio_tot,
            Ei_Es_cos_tot,
            loss_recon_tot,
            loss_image_cls_tot,
            loss_image_dis_tot,
            acc_image_cls_tot,
            acc_image_dis_tot,
            loss_anchor_cls_tot,
            loss_anchor_visual_tot,
            loss_anchor_text_tot,
            top1_anchor_tot,
            loss_prior_tot,
        ]).unsqueeze(0)
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
        "val/loss_image_cls": (summed[10] / total_global).item(),
        "val/loss_image_dis": (summed[11] / total_global).item(),
        "val/acc_image_cls": (summed[12] / total_global).item(),
        "val/acc_image_dis": (summed[13] / total_global).item(),
        "val/anchor_cls_loss": (summed[14] / total_global).item(),
        "val/anchor_visual_loss": (summed[15] / total_global).item(),
        "val/loss_anchor_text": (summed[16] / total_global).item(),
        "val/anchor_top1": (summed[17] / total_global).item(),
    }

    if diffusion_prior is not None:
        metrics["val/prior_loss"] = (summed[18] / total_global).item()

    if prev_eeg_training:
        eeg_backbone.train()
    else:
        eeg_backbone.eval()

    if sife is not None:
        if prev_sife_training:
            sife.train()
        else:
            sife.eval()

    if recon_decoder is not None:
        if prev_recon_training:
            recon_decoder.train()
        else:
            recon_decoder.eval()

    if ssfe_projector is not None:
        if prev_ssfe_training:
            ssfe_projector.train()
        else:
            ssfe_projector.eval()

    if diffusion_prior is not None:
        if prev_prior_training:
            diffusion_prior.train()
        else:
            diffusion_prior.eval()

    return metrics


# ---------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="GWIT training with pretrained GWIT EEG backbone + SIFE + EEG reconstruction + SSFE + prior + subject-wise splits"
    )

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="controlnet-model")
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=None)

    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

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
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--console_log_every",
        type=int,
        default=10,
        help="Print training losses to console every N optimizer steps.",
    )
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard", help='["tensorboard","wandb","all"]')
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    # DATA
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--conditioning_image_column", type=str, default="conditioning_image")
    parser.add_argument("--caption_column", type=str, default="caption")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_train_samples_per_subject", type=int, default=None)
    parser.add_argument("--max_val_samples_per_subject", type=int, default=None)
    parser.add_argument("--max_test_samples_per_subject", type=int, default=None)

    # subject-wise split
    parser.add_argument("--train_subjects", type=int, nargs="+", required=True)
    parser.add_argument("--val_subjects", type=int, nargs="+", required=True)
    parser.add_argument("--test_subjects", type=int, nargs="+", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_steps", type=int, default=100)
    parser.add_argument("--tracker_project_name", type=str, default="clean_gwit")
    parser.add_argument(
        "--log_validation_images",
        action="store_true",
        help="If set, generate and log validation/test images to W&B.",
    )

    # EEG / caption
    parser.add_argument("--caption_from_classifier", action="store_true")
    parser.add_argument("--captioner_root", type=str, default=None)
    parser.add_argument("--data_root", type=str, required=True)

    # Precomputed latents
    parser.add_argument("--use_precomputed_latents", action="store_true")
    parser.add_argument("--latents_dir", type=str, default=None)

    # Precomputed CLIP embeds
    parser.add_argument("--use_precomputed_clip_embeds", action="store_true")
    parser.add_argument("--clip_embeds_dir", type=str, default=None)

    # Drop coarse text control
    parser.add_argument("--drop_coarse_control_prob", type=float, default=0.0)
    parser.add_argument("--log_drop_coarse_control", action="store_true")

    # GWIT EEG backbone
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
    parser.add_argument(
        "--gradient_checkpointing_sife",
        action="store_true",
        help="Enable gradient checkpointing inside SIFE/FiModule to reduce VRAM usage.",
    )

    # EEG reconstruction
    parser.add_argument("--use_eeg_reconstruction", action="store_true")
    parser.add_argument("--recon_hidden_dim", type=int, default=256)
    parser.add_argument("--recon_num_blocks", type=int, default=3)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--recon_loss_type", type=str, default="l1", choices=["mse", "smooth_l1", "l1"])

    # SSFE
    parser.add_argument("--use_ssfe", action="store_true")
    parser.add_argument("--ssfe_hidden_dim", type=int, default=256)
    parser.add_argument("--ssfe_out_dim", type=int, default=1664)
    parser.add_argument("--ssfe_target_tokens", type=int, default=256)
    parser.add_argument(
        "--ssfe_adapter_type",
        type=str,
        default="zebra_like",
        choices=["simple", "zebra_like"],
    )
    parser.add_argument("--num_image_classes", type=int, default=40)
    parser.add_argument("--grl_lambda_ssfe", type=float, default=1.0)
    parser.add_argument("--lambda_ssfe", type=float, default=1.0)
    parser.add_argument("--lambda_image_cls", type=float, default=1.0)
    parser.add_argument("--lambda_image_dis", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing_ssfe", action="store_true")

    # ANCHOR F
    parser.add_argument("--lambda_anchor_cls", type=float, default=0.0)
    parser.add_argument("--lambda_anchor_visual", type=float, default=0.0)
    parser.add_argument("--lambda_anchor_text", type=float, default=0.0)

    parser.add_argument("--anchor_visual_temperature", type=float, default=0.07)
    parser.add_argument("--anchor_text_temperature", type=float, default=0.07)

    # PRIOR
    parser.add_argument("--use_prior", action="store_true")
    parser.add_argument("--lambda_prior", type=float, default=1.0)
    parser.add_argument("--prior_num_tokens", type=int, default=None,
                        help="If None, inferred from the first training batch clip_img_embeds.shape[1].")
    parser.add_argument("--prior_dim", type=int, default=None,
                        help="If None, inferred from --ssfe_out_dim.")
    parser.add_argument("--prior_depth", type=int, default=6)
    parser.add_argument("--prior_heads", type=int, default=32)
    parser.add_argument("--prior_timesteps", type=int, default=100)
    parser.add_argument("--prior_cond_drop_prob", type=float, default=0.2)
    parser.add_argument(
        "--gradient_checkpointing_prior",
        action="store_true",
        help="Enable gradient checkpointing inside PriorNetwork transformer blocks.",
    )

    # Training policy
    parser.add_argument("--freeze_controlnet", action="store_true")
    parser.add_argument(
        "--train_eeg_only",
        action="store_true",
        help="Train only EEG backbone + optional SIFE/SSFE/recon, skipping GWIT diffusion branch entirely.",
    )
    parser.add_argument(
        "--training_stage",
        type=str,
        default="full",
        choices=["full", "stage1", "stage2", "stage3"],
        help="Training stage for staged training.",
    )

    parser.add_argument("--load_sife_path", type=str, default=None)
    parser.add_argument("--load_recon_path", type=str, default=None)
    parser.add_argument("--load_ssfe_path", type=str, default=None)
    parser.add_argument("--load_prior_path", type=str, default=None)

    args = parser.parse_args(input_args)

    if args.use_ssfe and not args.use_sife:
        raise ValueError("--use_ssfe requires --use_sife")
    if args.use_prior and not args.use_ssfe:
        raise ValueError("--use_prior currently requires --use_ssfe")
    if args.use_prior and not args.use_precomputed_clip_embeds:
        raise ValueError("--use_prior requires --use_precomputed_clip_embeds")
    
    if (
        args.lambda_anchor_cls > 0.0
        or args.lambda_anchor_visual > 0.0
        or args.lambda_anchor_text > 0.0
    ) and not args.use_ssfe:
        raise ValueError("Anchor losses require --use_ssfe")

    if args.lambda_anchor_visual > 0.0 and not args.use_precomputed_clip_embeds:
        raise ValueError("--lambda_anchor_visual requires --use_precomputed_clip_embeds")

    if args.training_stage == "stage2":
        if not args.use_sife:
            raise ValueError("stage2 requires --use_sife")
        if not args.use_ssfe:
            raise ValueError("stage2 requires --use_ssfe")
        if args.load_sife_path is None:
            raise ValueError("stage2 requires --load_sife_path")
        if args.eeg_backbone_ckpt is None:
            raise ValueError("stage2 requires --eeg_backbone_ckpt")

    if args.training_stage == "stage3":
        if not args.use_sife:
            raise ValueError("stage3 requires --use_sife")
        if not args.use_ssfe:
            raise ValueError("stage3 requires --use_ssfe")
        if not args.use_prior:
            raise ValueError("stage3 requires --use_prior")
        if args.load_sife_path is None:
            raise ValueError("stage3 requires --load_sife_path")
        if args.load_ssfe_path is None:
            raise ValueError("stage3 requires --load_ssfe_path")
        if args.eeg_backbone_ckpt is None:
            raise ValueError("stage3 requires --eeg_backbone_ckpt")
        
    if args.training_stage in {"stage1", "stage2", "stage3"} and not args.train_eeg_only:
        raise ValueError("Staged training requires --train_eeg_only")
    
    if args.training_stage == "full" and args.freeze_controlnet:
        raise ValueError("full training with --freeze_controlnet is not supported because it only wastes diffusion compute.")

    return args


# ---------------------------------------------------------
# UTILS: CHECKPOINT MANAGEMENT
# ---------------------------------------------------------
def get_sorted_checkpoints(output_dir, prefix="checkpoint"):
    checkpoints = []
    if not os.path.isdir(output_dir):
        return checkpoints

    for path in os.listdir(output_dir):
        full = os.path.join(output_dir, path)
        if os.path.isdir(full) and path.startswith(f"{prefix}-"):
            try:
                step = int(path.split("-")[-1])
                checkpoints.append((step, full))
            except ValueError:
                continue
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def rotate_checkpoints(output_dir, max_checkpoints, prefix="checkpoint"):
    if max_checkpoints is None or max_checkpoints <= 0:
        return
    checkpoints = get_sorted_checkpoints(output_dir, prefix=prefix)
    if len(checkpoints) <= max_checkpoints:
        return

    num_to_remove = len(checkpoints) - max_checkpoints
    for i in range(num_to_remove):
        step, path = checkpoints[i]
        logger.info(f"Removing old checkpoint {path} (step {step})")
        try:
            import shutil
            shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Error while removing checkpoint {path}: {e}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.pretrained_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    _empty_tok = tokenizer(
        "",
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    empty_input_ids = _empty_tok.input_ids

    need_diffusion_branch = not args.train_eeg_only

    if need_diffusion_branch:
        text_encoder_cls = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path,
            args.revision,
        )
    else:
        text_encoder_cls = None

    if need_diffusion_branch:
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="scheduler",
        )
    else:
        noise_scheduler = None

    if need_diffusion_branch:
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant,
        )

        text_dim = int(text_encoder.config.hidden_size)
        accelerator.print(f"[AUTO] text_dim = {text_dim}")
    else:
        text_encoder = None
        text_dim = None

    anchor_text_dim = None

    if need_diffusion_branch:
        if args.use_precomputed_latents:
            vae = None
        else:
            vae = AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="vae",
                revision=args.revision,
                variant=args.variant,
            )
    else:
        vae = None

    if need_diffusion_branch:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
            variant=args.variant,
        )

        logger.info("Initializing ControlNet from UNet")
        n_subjects = 6 if "CVPR" in args.dataset_name.upper() else 24
        controlnet = ControlNetModel.from_unet(unet, n_subjects=n_subjects)
    else:
        unet = None
        controlnet = None

    # SIFE: classi locali solo sui soggetti di train
    active_train_subjects = sorted(set(int(s) for s in args.train_subjects))
    subject_to_local = build_subject_remap(active_train_subjects)

    # ---------------------------------------------------------
    # Datasets / Dataloaders
    # ---------------------------------------------------------
    train_dataset = make_train_dataset(args, tokenizer, accelerator)
    val_dataset = make_val_dataset(args, tokenizer, accelerator)
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

    # ---------------------------------------------------------
    # GWIT EEG backbone
    # ---------------------------------------------------------
    eeg_backbone = GWITEEGBackbone(
        in_channels=128,
        hidden_size=args.eeg_backbone_hidden_size,
        num_layers=args.eeg_backbone_num_layers,
    )
    eeg_backbone.train()

    if args.eeg_backbone_ckpt is not None:
        load_eeg_backbone_from_ckpt(eeg_backbone, args.eeg_backbone_ckpt)
        accelerator.print(f"[LOAD] Loaded EEG backbone encoder from {args.eeg_backbone_ckpt}")

    # ---------------------------------------------------------
    # Infer seq_len from EEG backbone
    # ---------------------------------------------------------
    sample_ex = train_dataset[0]
    sample_cond = sample_ex["conditioning_pixel_values"]

    if sample_cond.ndim == 2:
        dummy_eeg = sample_cond.unsqueeze(0).float()   # (1, C, T)
    elif sample_cond.ndim == 3:
        dummy_eeg = sample_cond[:1].float()            # già batchato, raro
    else:
        raise ValueError(f"Unexpected conditioning shape: {tuple(sample_cond.shape)}")

    was_training = eeg_backbone.training
    eeg_backbone.eval()
    with torch.no_grad():
        dummy_out = eeg_backbone(dummy_eeg)
    if was_training:
        eeg_backbone.train()
        
    inferred_seq_len = int(dummy_out["sequence"].shape[1])
    accelerator.print(f"[AUTO] inferred SIFE seq_len = {inferred_seq_len}")

    # ---------------------------------------------------------
    # SIFE
    # ---------------------------------------------------------
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

        sife.train()

        if args.load_sife_path is not None:
            sife.load_state_dict(torch.load(args.load_sife_path, map_location="cpu"))
            accelerator.print(f"[LOAD] Loaded SIFE from {args.load_sife_path}")

    # ---------------------------------------------------------
    # EEG reconstruction decoder
    # ---------------------------------------------------------
    recon_decoder = None
    if args.use_eeg_reconstruction:
        recon_decoder = EEGReconstructionDecoder(
            in_dim=args.eeg_backbone_hidden_size,
            hidden_dim=args.recon_hidden_dim,
            out_channels=128,
            num_res_blocks=args.recon_num_blocks,
        )
        recon_decoder.train()

        if args.load_recon_path is not None:
            recon_decoder.load_state_dict(torch.load(args.load_recon_path, map_location="cpu"))
            accelerator.print(f"[LOAD] Loaded recon decoder from {args.load_recon_path}")

    # ---------------------------------------------------------
    # SSFE projector
    # ---------------------------------------------------------
    if args.use_ssfe and args.use_precomputed_clip_embeds:
        first_ex = train_dataset[0]

        if "clip_img_embeds" not in first_ex:
            raise RuntimeError("SSFE with OpenCLIP supervision requires clip_img_embeds in dataset.")
        if "clip_text_embeds" not in first_ex:
            raise RuntimeError("SSFE with anchor text supervision requires clip_text_embeds in dataset.")

        clip_ex = first_ex["clip_img_embeds"]
        clip_text_ex = first_ex["clip_text_embeds"]

        if clip_ex.ndim != 2:
            raise RuntimeError(
                f"SSFE expects sequence-level clip_img_embeds with shape (T, D), "
                f"but got {tuple(clip_ex.shape)}"
            )

        if clip_text_ex.ndim != 1:
            raise RuntimeError(
                f"SSFE expects pooled clip_text_embeds with shape (D,), "
                f"but got {tuple(clip_text_ex.shape)}"
            )

        inferred_num_tokens, inferred_clip_dim = clip_ex.shape
        inferred_text_dim = int(clip_text_ex.shape[0])

        if int(args.ssfe_target_tokens) != inferred_num_tokens:
            raise ValueError(
                f"Mismatch: ssfe_target_tokens={args.ssfe_target_tokens} but dataset CLIP token count={inferred_num_tokens}. "
                f"For the current setup, keep --ssfe_target_tokens aligned with the CLIP target."
            )

        if int(args.ssfe_out_dim) != inferred_clip_dim:
            raise ValueError(
                f"Mismatch: ssfe_out_dim={args.ssfe_out_dim} but dataset CLIP embed dim={inferred_clip_dim}. "
                f"Set --ssfe_out_dim accordingly."
            )

        anchor_text_dim = inferred_text_dim

        accelerator.print(
            f"[SSFE/CLIP] dataset_tokens={inferred_num_tokens} | "
            f"dataset_dim={inferred_clip_dim} | "
            f"text_dim={anchor_text_dim} | "
            f"ssfe_target_tokens={args.ssfe_target_tokens} | "
            f"ssfe_out_dim={args.ssfe_out_dim}"
        )
    else:
        if args.lambda_anchor_text > 0.0:
            raise ValueError("--lambda_anchor_text > 0 requires --use_precomputed_clip_embeds with clip_text_embeds available.")
        anchor_text_dim = 1280

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
            
        ssfe_projector.train()

        accelerator.print(
            f"[SSFE] adapter_type={args.ssfe_adapter_type} | "
            f"target_tokens={args.ssfe_target_tokens} | "
            f"out_dim={args.ssfe_out_dim}"
        )

        if args.load_ssfe_path is not None:
            ssfe_projector.load_state_dict(torch.load(args.load_ssfe_path, map_location="cpu"))
            accelerator.print(f"[LOAD] Loaded SSFE projector from {args.load_ssfe_path}")

    # ---------------------------------------------------------
    # PRIOR
    # ---------------------------------------------------------
    diffusion_prior = None
    prior_dim = None
    prior_num_tokens = None

    if args.use_prior:
        if len(train_dataset) == 0:
            raise RuntimeError("Empty train_dataset, cannot infer prior dimensions.")

        first_ex = train_dataset[0]
        if "clip_img_embeds" not in first_ex:
            raise RuntimeError("Prior requires clip_img_embeds in dataset.")

        clip_ex = first_ex["clip_img_embeds"]
        if clip_ex.ndim != 2:
            raise RuntimeError(
                f"Prior expects sequence-level clip_img_embeds with shape (T, D), "
                f"but got {tuple(clip_ex.shape)}"
            )

        inferred_num_tokens, inferred_clip_dim = clip_ex.shape
        prior_num_tokens = int(args.prior_num_tokens) if args.prior_num_tokens is not None else int(inferred_num_tokens)
        prior_dim = int(args.prior_dim) if args.prior_dim is not None else int(args.ssfe_out_dim)

        if prior_num_tokens != inferred_num_tokens:
            raise ValueError(
                f"Mismatch: prior_num_tokens={prior_num_tokens} but dataset clip tokens={inferred_num_tokens}"
            )

        if prior_dim != inferred_clip_dim:
            raise ValueError(
                f"Mismatch: prior_dim={prior_dim} but dataset CLIP embed dim={inferred_clip_dim}. "
                f"Set --ssfe_out_dim / --prior_dim accordingly."
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

        if args.gradient_checkpointing_prior:
            if hasattr(prior_network, "set_gradient_checkpointing"):
                prior_network.set_gradient_checkpointing(True)
                accelerator.print("[PRIOR] gradient checkpointing enabled")
            else:
                accelerator.print("[PRIOR] gradient checkpointing requested, but not implemented in current PriorNetwork")

        diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=prior_dim,
            timesteps=int(args.prior_timesteps),
            cond_drop_prob=float(args.prior_cond_drop_prob),
            predict_x_start=True,
            training_clamp_l2norm=False,
            sampling_clamp_l2norm=False,
        )

        diffusion_prior.train()

        accelerator.print(
            f"[PRIOR] enabled | num_tokens={prior_num_tokens} | dim={prior_dim} | "
            f"depth={args.prior_depth} | heads={args.prior_heads} | timesteps={args.prior_timesteps}"
        )

        if args.load_prior_path is not None:
            diffusion_prior.load_state_dict(torch.load(args.load_prior_path, map_location="cpu"))
            accelerator.print(f"[LOAD] Loaded diffusion prior from {args.load_prior_path}")

    # ---------------------------------------------------------
    # Freeze base models
    # ---------------------------------------------------------
    if vae is not None:
        vae.requires_grad_(False)

    if unet is not None:
        unet.requires_grad_(False)

    if text_encoder is not None:
        text_encoder.requires_grad_(False)

    if controlnet is not None:
        if args.freeze_controlnet:
            controlnet.requires_grad_(False)
            controlnet.eval()
        else:
            controlnet.train()

    if args.enable_xformers_memory_efficient_attention and need_diffusion_branch:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers not available")
        
    is_stage1 = args.training_stage == "stage1"
    is_stage2 = args.training_stage == "stage2"
    is_stage3 = args.training_stage == "stage3"
    is_full = args.training_stage == "full"

    if is_stage1:
        eeg_backbone.requires_grad_(True)
        eeg_backbone.train()

        if sife is not None:
            sife.requires_grad_(True)
            sife.train()

        if recon_decoder is not None:
            recon_decoder.requires_grad_(True)
            recon_decoder.train()

        if ssfe_projector is not None:
            ssfe_projector.requires_grad_(False)
            ssfe_projector.eval()

        if diffusion_prior is not None:
            diffusion_prior.requires_grad_(False)
            diffusion_prior.eval()

    elif is_stage2:
        eeg_backbone.requires_grad_(False)
        eeg_backbone.eval()

        if sife is not None:
            sife.requires_grad_(False)
            sife.eval()

        if recon_decoder is not None:
            recon_decoder.requires_grad_(False)
            recon_decoder.eval()

        if ssfe_projector is not None:
            ssfe_projector.requires_grad_(True)
            ssfe_projector.train()

        if diffusion_prior is not None:
            diffusion_prior.requires_grad_(False)
            diffusion_prior.eval()

    elif is_stage3:
        eeg_backbone.requires_grad_(False)
        eeg_backbone.eval()

        if sife is not None:
            sife.requires_grad_(False)
            sife.eval()

        if recon_decoder is not None:
            recon_decoder.requires_grad_(False)
            recon_decoder.eval()

        if ssfe_projector is not None:
            ssfe_projector.requires_grad_(False)
            ssfe_projector.eval()

        if diffusion_prior is not None:
            diffusion_prior.requires_grad_(True)
            diffusion_prior.train()

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    lr_main = float(args.learning_rate)
    lr_eeg = float(args.eeg_backbone_lr) if args.eeg_backbone_lr is not None else lr_main
    lr_sife = float(args.sife_lr) if args.sife_lr is not None else lr_main
    lr_recon = float(args.recon_lr) if args.recon_lr is not None else lr_main
    lr_ssfe = float(args.ssfe_lr) if args.ssfe_lr is not None else lr_main
    lr_prior = float(args.prior_lr) if args.prior_lr is not None else lr_main

    params_main = []
    if controlnet is not None and (not args.freeze_controlnet):
        params_main += [p for p in controlnet.parameters() if p.requires_grad]

    params_eeg = [p for p in eeg_backbone.parameters() if p.requires_grad]
    params_sife = [p for p in sife.parameters() if p.requires_grad] if sife is not None else []
    params_recon = [p for p in recon_decoder.parameters() if p.requires_grad] if recon_decoder is not None else []
    params_ssfe = [p for p in ssfe_projector.parameters() if p.requires_grad] if ssfe_projector is not None else []
    params_prior = [p for p in diffusion_prior.parameters() if p.requires_grad] if diffusion_prior is not None else []

    param_groups = []
    if len(params_main) > 0:
        param_groups.append({"params": params_main, "lr": lr_main})
    if len(params_eeg) > 0:
        param_groups.append({"params": params_eeg, "lr": lr_eeg})
    if len(params_sife) > 0:
        param_groups.append({"params": params_sife, "lr": lr_sife})
    if len(params_recon) > 0:
        param_groups.append({"params": params_recon, "lr": lr_recon})
    if len(params_ssfe) > 0:
        param_groups.append({"params": params_ssfe, "lr": lr_ssfe})
    if len(params_prior) > 0:
        param_groups.append({"params": params_prior, "lr": lr_prior})

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ---------------------------------------------------------
    # Scheduler setup
    # ---------------------------------------------------------
    import math
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

    # ---------------------------------------------------------
    # Prepare
    # ---------------------------------------------------------
    prepare_items = []
    if controlnet is not None:
        prepare_items.append(controlnet)
    prepare_items.append(eeg_backbone)

    if sife is not None:
        prepare_items.append(sife)
    if recon_decoder is not None:
        prepare_items.append(recon_decoder)
    if ssfe_projector is not None:
        prepare_items.append(ssfe_projector)
    if diffusion_prior is not None:
        prepare_items.append(diffusion_prior)
    prepare_items += [optimizer, train_dataloader, val_dataloader, lr_scheduler]

    prepared = accelerator.prepare(*prepare_items)

    idx = 0
    if controlnet is not None:
        controlnet = prepared[idx]
        idx += 1

    eeg_backbone = prepared[idx]
    idx += 1

    if sife is not None:
        sife = prepared[idx]; idx += 1
    if recon_decoder is not None:
        recon_decoder = prepared[idx]; idx += 1
    if ssfe_projector is not None:
        ssfe_projector = prepared[idx]; idx += 1
    if diffusion_prior is not None:
        diffusion_prior = prepared[idx]; idx += 1
    optimizer = prepared[idx]; idx += 1
    train_dataloader = prepared[idx]; idx += 1
    val_dataloader = prepared[idx]; idx += 1
    lr_scheduler = prepared[idx]; idx += 1

    accum_models = []

    if len(params_main) > 0 and controlnet is not None:
        accum_models.append(controlnet)
    if len(params_eeg) > 0:
        accum_models.append(eeg_backbone)
    if len(params_sife) > 0 and sife is not None:
        accum_models.append(sife)
    if len(params_recon) > 0 and recon_decoder is not None:
        accum_models.append(recon_decoder)
    if len(params_ssfe) > 0 and ssfe_projector is not None:
        accum_models.append(ssfe_projector)
    if len(params_prior) > 0 and diffusion_prior is not None:
        accum_models.append(diffusion_prior)

    if len(accum_models) == 0:
        raise ValueError("No trainable modules found for gradient accumulation.")
    
    params_to_clip = []
    seen = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad and id(p) not in seen:
                params_to_clip.append(p)
                seen.add(id(p))

    # ---------------------------------------------------------
    # Resume
    # ---------------------------------------------------------
    global_step = 0
    first_epoch = 0
    resume_step_in_epoch = 0

    if args.resume_from_checkpoint:
        if os.path.isdir(args.resume_from_checkpoint):
            resume_path = args.resume_from_checkpoint
        else:
            candidate = os.path.join(args.output_dir, args.resume_from_checkpoint)
            if os.path.isdir(candidate):
                resume_path = candidate
            else:
                raise ValueError(f"Checkpoint folder {args.resume_from_checkpoint} not found.")

        accelerator.print(f"Resuming from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)

        basename = os.path.basename(resume_path)
        try:
            global_step = int(basename.split("-")[-1])
        except ValueError:
            accelerator.print(
                f"Warning: could not parse global_step from checkpoint name {basename}. Assuming 0."
            )
            global_step = 0

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step_in_epoch = global_step % num_update_steps_per_epoch

        accelerator.print(
            f" → Parsed global_step={global_step}, first_epoch={first_epoch}, "
            f"resume_step_in_epoch={resume_step_in_epoch}"
        )

    # ---------------------------------------------------------
    # Mixed precision dtype
    # ---------------------------------------------------------
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    if vae is not None:
        vae.to(accelerator.device, dtype=torch.float32)
    if unet is not None:
        unet.to(accelerator.device, dtype=weight_dtype)
    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # trainable modules: keep in fp32
    eeg_backbone.to(accelerator.device, dtype=torch.float32)
    if sife is not None:
        sife.to(accelerator.device, dtype=torch.float32)
    if recon_decoder is not None:
        recon_decoder.to(accelerator.device, dtype=torch.float32)
    if ssfe_projector is not None:
        ssfe_projector.to(accelerator.device, dtype=torch.float32)
    if diffusion_prior is not None:
        diffusion_prior.to(accelerator.device, dtype=torch.float32)

    ssfe_dtype = torch.float32
    prior_dtype = torch.float32

    if accelerator.is_main_process:
        config = vars(args).copy()
        accelerator.init_trackers(args.tracker_project_name, config=config)

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Training steps",
    )

    if accelerator.is_main_process:
        accelerator.print(
            f"[MODE] train_eeg_only={args.train_eeg_only} | "
            f"use_sife={args.use_sife} | use_ssfe={args.use_ssfe} | "
            f"use_prior={args.use_prior} | use_eeg_reconstruction={args.use_eeg_reconstruction}"
        )

    # ---------------------------------------------------------
    # TRAIN LOOP
    # ---------------------------------------------------------
    debug_printed_once = False

    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_loss_sum = 0.0
        epoch_diffusion_loss_sum = 0.0
        epoch_recon_loss_sum = 0.0
        epoch_ssfe_loss_sum = 0.0
        epoch_prior_loss_sum = 0.0

        epoch_sife_inv_sum = 0.0
        epoch_sife_spec_sum = 0.0
        epoch_inv_acc_sum = 0.0
        epoch_spec_acc_sum = 0.0

        epoch_img_cls_loss_sum = 0.0
        epoch_img_dis_loss_sum = 0.0
        epoch_img_cls_acc_sum = 0.0
        epoch_img_dis_acc_sum = 0.0

        epoch_anchor_cls_loss_sum = 0.0
        epoch_anchor_visual_loss_sum = 0.0
        epoch_anchor_text_loss_sum = 0.0
        epoch_anchor_cls_acc_sum = 0.0

        epoch_e_norm_sum = 0.0
        epoch_ei_norm_sum = 0.0
        epoch_es_norm_sum = 0.0
        epoch_ei_es_ratio_sum = 0.0
        epoch_ei_es_cos_sum = 0.0

        epoch_inv_entropy_sum = 0.0
        epoch_spec_entropy_sum = 0.0
        epoch_inv_entropy_norm_sum = 0.0
        epoch_spec_entropy_norm_sum = 0.0

        epoch_random_subject_acc_sum = 0.0
        epoch_inv_gap_sum = 0.0
        epoch_spec_gap_sum = 0.0

        epoch_lr_sum = 0.0
        epoch_logged_steps = 0

        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step_in_epoch:
                continue

            with accelerator.accumulate(*accum_models):
                stage1_mode = is_stage1
                stage2_mode = is_stage2
                stage3_mode = is_stage3
                full_mode = is_full

                eeg_cond = batch["conditioning_pixel_values"].to(
                    accelerator.device,
                    dtype=weight_dtype,
                )
                image_labels = batch["image_labels"].to(
                    accelerator.device,
                    dtype=torch.long,
                )

                # ---------------------------------------------------------
                # GWIT EEG backbone (+ optional SIFE)
                # ---------------------------------------------------------
                if stage2_mode or stage3_mode:
                    with torch.no_grad():
                        eeg_feats = eeg_backbone(eeg_cond.float())
                        E_seq = eeg_feats["sequence"]   # (B, T, D)
                        E_pooled = eeg_feats["pooled"]  # (B, D)

                        if sife is not None:
                            sife_out = sife(E_seq)
                            E_i_seq = sife_out["E_i_seq"]
                            E_s_seq = sife_out["E_s_seq"]
                            E_i = sife_out["E_i"]
                            E_s = sife_out["E_s"]
                            pred_subject_i = sife_out["pred_subject_i"]
                            pred_subject_s = sife_out["pred_subject_s"]
                        else:
                            E_i_seq = None
                            E_s_seq = None
                            E_i = None
                            E_s = None
                            pred_subject_i = None
                            pred_subject_s = None
                else:
                    eeg_feats = eeg_backbone(eeg_cond.float())
                    E_seq = eeg_feats["sequence"]   # (B, T, D)
                    E_pooled = eeg_feats["pooled"]  # (B, D)

                    if sife is not None:
                        sife_out = sife(E_seq)
                        E_i_seq = sife_out["E_i_seq"]
                        E_s_seq = sife_out["E_s_seq"]
                        E_i = sife_out["E_i"]
                        E_s = sife_out["E_s"]
                        pred_subject_i = sife_out["pred_subject_i"]
                        pred_subject_s = sife_out["pred_subject_s"]
                    else:
                        E_i_seq = None
                        E_s_seq = None
                        E_i = None
                        E_s = None
                        pred_subject_i = None
                        pred_subject_s = None

                sife_loss_inv = torch.tensor(0.0, device=accelerator.device)
                sife_loss_spec = torch.tensor(0.0, device=accelerator.device)
                recon_loss = torch.tensor(0.0, device=accelerator.device)

                ssfe_loss = torch.tensor(0.0, device=accelerator.device)
                image_cls_loss = torch.tensor(0.0, device=accelerator.device)
                image_dis_loss = torch.tensor(0.0, device=accelerator.device)

                anchor_cls_loss = torch.tensor(0.0, device=accelerator.device)
                anchor_visual_loss = torch.tensor(0.0, device=accelerator.device)
                anchor_text_loss = torch.tensor(0.0, device=accelerator.device)

                prior_loss = torch.tensor(0.0, device=accelerator.device)
                diffusion_loss = torch.tensor(0.0, device=accelerator.device)

                inv_acc = torch.tensor(0.0, device=accelerator.device)
                spec_acc = torch.tensor(0.0, device=accelerator.device)
                image_cls_acc = torch.tensor(0.0, device=accelerator.device)
                image_dis_acc = torch.tensor(0.0, device=accelerator.device)
                anchor_cls_acc = torch.tensor(0.0, device=accelerator.device)

                F_s = None
                F_i = None
                F_anchor = None
                F_anchor_visual = None
                anchor_text_embed = None

                E_i_norm = torch.tensor(0.0, device=accelerator.device)
                E_s_norm = torch.tensor(0.0, device=accelerator.device)
                Ei_Es_norm_ratio = torch.tensor(0.0, device=accelerator.device)

                inv_entropy = torch.tensor(0.0, device=accelerator.device)
                spec_entropy = torch.tensor(0.0, device=accelerator.device)

                random_subject_acc = torch.tensor(0.0, device=accelerator.device)
                inv_acc_gap_vs_random = torch.tensor(0.0, device=accelerator.device)
                spec_acc_gap_vs_random = torch.tensor(0.0, device=accelerator.device)

                inv_entropy_norm = torch.tensor(0.0, device=accelerator.device)
                spec_entropy_norm = torch.tensor(0.0, device=accelerator.device)
                Ei_Es_cos = torch.tensor(0.0, device=accelerator.device)

                if sife is not None and (stage1_mode or full_mode):
                    subject_targets = remap_subject_targets(
                        batch["eeg_subjects"].to(accelerator.device),
                        subject_to_local,
                    )

                    sife_loss_inv = F.cross_entropy(pred_subject_i, subject_targets)
                    sife_loss_spec = F.cross_entropy(pred_subject_s, subject_targets)

                    pred_i_cls = pred_subject_i.argmax(dim=-1)
                    pred_s_cls = pred_subject_s.argmax(dim=-1)
                    inv_acc = (pred_i_cls == subject_targets).float().mean()
                    spec_acc = (pred_s_cls == subject_targets).float().mean()

                    E_i_norm = E_i.norm(dim=-1).mean()
                    E_s_norm = E_s.norm(dim=-1).mean()
                    Ei_Es_norm_ratio = E_i_norm / (E_s_norm + 1e-8)

                    p_i = torch.softmax(pred_subject_i, dim=-1)
                    p_s = torch.softmax(pred_subject_s, dim=-1)

                    inv_entropy = -(p_i * torch.log(p_i + 1e-8)).sum(dim=-1).mean()
                    spec_entropy = -(p_s * torch.log(p_s + 1e-8)).sum(dim=-1).mean()

                    max_entropy = torch.log(
                        torch.tensor(float(len(active_train_subjects)), device=accelerator.device)
                    )
                    inv_entropy_norm = inv_entropy / (max_entropy + 1e-8)
                    spec_entropy_norm = spec_entropy / (max_entropy + 1e-8)

                    Ei_Es_cos = F.cosine_similarity(E_i, E_s, dim=-1).mean()

                    random_subject_acc = torch.tensor(
                        1.0 / float(len(active_train_subjects)),
                        device=accelerator.device,
                    )
                    inv_acc_gap_vs_random = inv_acc - random_subject_acc
                    spec_acc_gap_vs_random = spec_acc - random_subject_acc

                if (not debug_printed_once) and accelerator.is_main_process:
                    accelerator.print(f"[DEBUG] E_seq shape: {tuple(E_seq.shape)}")
                    accelerator.print(f"[DEBUG] E_pooled shape: {tuple(E_pooled.shape)}")
                    if E_i_seq is not None:
                        accelerator.print(f"[DEBUG] E_i_seq shape: {tuple(E_i_seq.shape)}")
                        accelerator.print(f"[DEBUG] E_s_seq shape: {tuple(E_s_seq.shape)}")
                        accelerator.print(f"[DEBUG] E_i shape: {tuple(E_i.shape)}")
                        accelerator.print(f"[DEBUG] E_s shape: {tuple(E_s.shape)}")
                    if pred_subject_i is not None:
                        accelerator.print(f"[DEBUG] pred_subject_i shape: {tuple(pred_subject_i.shape)}")
                        accelerator.print(f"[DEBUG] pred_subject_s shape: {tuple(pred_subject_s.shape)}")
                    accelerator.print(f"[DEBUG] active_train_subjects: {active_train_subjects}")
                    accelerator.print(f"[DEBUG] subject_to_local: {subject_to_local}")

                # ---------------------------------------------------------
                # EEG reconstruction loss from E_seq
                # ---------------------------------------------------------
                if recon_decoder is not None and (stage1_mode or full_mode):
                    eeg_recon = recon_decoder(E_seq.float())
                    eeg_target = eeg_cond.float()

                    if args.recon_loss_type == "mse":
                        recon_loss = F.mse_loss(eeg_recon, eeg_target)
                    elif args.recon_loss_type == "smooth_l1":
                        recon_loss = F.smooth_l1_loss(eeg_recon, eeg_target)
                    elif args.recon_loss_type == "l1":
                        recon_loss = F.l1_loss(eeg_recon, eeg_target)
                    else:
                        raise ValueError(f"Unsupported recon_loss_type: {args.recon_loss_type}")

                    if (not debug_printed_once) and accelerator.is_main_process:
                        accelerator.print(f"[DEBUG] eeg_recon shape: {tuple(eeg_recon.shape)}")
                        accelerator.print(f"[DEBUG] eeg_target shape: {tuple(eeg_target.shape)}")

                # ---------------------------------------------------------
                # SSFE + anchor losses
                # ---------------------------------------------------------
                clip_target_ssfe = None
                clip_target_prior = None

                if ssfe_projector is not None and (stage2_mode or full_mode):
                    if E_i_seq is None:
                        raise RuntimeError("SSFE requires E_i_seq from SIFE.")

                    ssfe_out = ssfe_projector(
                        E_i=E_i_seq.to(dtype=ssfe_dtype),
                        E=E_seq.to(dtype=ssfe_dtype),
                    )

                    F_s = ssfe_out["F_s"]
                    F_anchor = ssfe_out["F"]
                    F_anchor_visual = ssfe_out["F_anchor_visual"]
                    F_i = ssfe_out["F_i"]
                    anchor_text_embed = ssfe_out["anchor_text_embed"]

                    pred_image_cls = ssfe_out["pred_image_cls"]
                    pred_image_dis = ssfe_out["pred_image_dis"]
                    pred_image_cls_anchor = ssfe_out["pred_image_cls_anchor"]

                    image_cls_loss = F.cross_entropy(pred_image_cls.float(), image_labels)
                    image_dis_loss = F.cross_entropy(pred_image_dis.float(), image_labels)
                    anchor_cls_loss = F.cross_entropy(pred_image_cls_anchor.float(), image_labels)

                    image_cls_acc = (pred_image_cls.argmax(dim=-1) == image_labels).float().mean()
                    image_dis_acc = (pred_image_dis.argmax(dim=-1) == image_labels).float().mean()
                    anchor_cls_acc = (
                        pred_image_cls_anchor.argmax(dim=-1) == image_labels
                    ).float().mean()

                    if args.lambda_anchor_visual > 0.0:
                        clip_target_ssfe = batch["clip_img_embeds"].to(
                            accelerator.device,
                            dtype=ssfe_dtype,
                        )

                        if clip_target_ssfe.ndim != 3:
                            raise RuntimeError(
                                f"Anchor visual loss expects sequence-level clip_img_embeds "
                                f"with shape (B, T, D), got {tuple(clip_target_ssfe.shape)}"
                            )

                        anchor_visual_loss = sequence_info_nce_loss(
                            F_anchor_visual.float(),
                            clip_target_ssfe.float(),
                            temperature=args.anchor_visual_temperature,
                        )

                    if args.lambda_anchor_text > 0.0:
                        text_target = batch["clip_text_embeds"].to(
                            accelerator.device,
                            dtype=ssfe_dtype,
                        )

                        anchor_text_loss = info_nce_loss(
                            anchor_text_embed.float(),
                            text_target.float(),
                            temperature=args.anchor_text_temperature,
                        )

                    ssfe_loss = (
                        float(args.lambda_image_cls) * image_cls_loss
                        + float(args.lambda_image_dis) * image_dis_loss
                        + float(args.lambda_anchor_cls) * anchor_cls_loss
                        + float(args.lambda_anchor_visual) * anchor_visual_loss
                        + float(args.lambda_anchor_text) * anchor_text_loss
                    )

                    if (not debug_printed_once) and accelerator.is_main_process:
                        accelerator.print(f"[DEBUG] F_s shape: {tuple(F_s.shape)}")
                        accelerator.print(f"[DEBUG] F_i shape: {tuple(F_i.shape)}")
                        accelerator.print(f"[DEBUG] F shape: {tuple(F_anchor.shape)}")
                        accelerator.print(f"[DEBUG] F_anchor_visual shape: {tuple(F_anchor_visual.shape)}")
                        accelerator.print(
                            f"[DEBUG] F_anchor_visual_is_F: {tuple(F_anchor_visual.shape) == tuple(F_anchor.shape)}"
                        )
                        accelerator.print(f"[DEBUG] anchor_text_embed shape: {tuple(anchor_text_embed.shape)}")
                        accelerator.print(f"[DEBUG] pred_image_cls shape: {tuple(pred_image_cls.shape)}")
                        accelerator.print(f"[DEBUG] pred_image_dis shape: {tuple(pred_image_dis.shape)}")
                        accelerator.print(f"[DEBUG] image_labels shape: {tuple(image_labels.shape)}")
                        if clip_target_ssfe is not None:
                            accelerator.print(f"[DEBUG] clip_target_ssfe shape: {tuple(clip_target_ssfe.shape)}")

                elif ssfe_projector is not None and stage3_mode:
                    if E_i_seq is None:
                        raise RuntimeError("Stage3 prior requires E_i_seq from SIFE.")

                    with torch.no_grad():
                        ssfe_out = ssfe_projector(
                            E_i=E_i_seq.to(dtype=ssfe_dtype),
                            E=E_seq.to(dtype=ssfe_dtype),
                        )
                        F_s = ssfe_out["F_s"]

                    if (not debug_printed_once) and accelerator.is_main_process:
                        accelerator.print(f"[DEBUG] stage3 F_s shape: {tuple(F_s.shape)}")

                # ---------------------------------------------------------
                # PRIOR loss
                #   text_embed  = F_s   (semantic branch only, unchanged)
                #   image_embed = CLIP image tokens
                # ---------------------------------------------------------
                if diffusion_prior is not None and (stage3_mode or full_mode):
                    if F_s is None:
                        raise RuntimeError("Prior requires F_s from SSFE.")

                    if clip_target_prior is None:
                        clip_target_prior = batch["clip_img_embeds"].to(
                            accelerator.device,
                            dtype=prior_dtype,
                        )

                    if clip_target_prior.ndim != 3:
                        raise RuntimeError(
                            f"Prior expects sequence-level clip_img_embeds with shape (B, T, D), "
                            f"but got {tuple(clip_target_prior.shape)}"
                        )

                    prior_loss, prior_pred = diffusion_prior(
                        text_embed=F_s.to(dtype=prior_dtype),
                        image_embed=clip_target_prior,
                    )

                    if (not debug_printed_once) and accelerator.is_main_process:
                        accelerator.print(f"[DEBUG] prior text_embed/F_s shape: {tuple(F_s.shape)}")
                        accelerator.print(f"[DEBUG] prior image_embed/clip_target_prior shape: {tuple(clip_target_prior.shape)}")
                        accelerator.print(f"[DEBUG] prior_pred shape: {tuple(prior_pred.shape)}")

                if (not debug_printed_once) and accelerator.is_main_process:
                    debug_printed_once = True

                if not args.train_eeg_only:
                    # ---------------------------------------------------------
                    # LATENTS
                    # ---------------------------------------------------------
                    if args.use_precomputed_latents:
                        posterior_mean = batch["posterior_mean"].to(
                            accelerator.device,
                            dtype=weight_dtype,
                        )
                        posterior_logvar = batch["posterior_logvar"].to(
                            accelerator.device,
                            dtype=weight_dtype,
                        )
                        noise = torch.randn_like(posterior_mean)
                        latents = posterior_mean + torch.exp(0.5 * posterior_logvar) * noise
                        latents = latents * VAE_SCALE_FACTOR
                    else:
                        pixel_values = batch["pixel_values"].to(
                            accelerator.device,
                            dtype=torch.float32,
                        )
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents.to(weight_dtype) * vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=latents.device,
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # ---------------------------------------------------------
                    # TEXT + drop coarse
                    # ---------------------------------------------------------
                    input_ids = batch["input_ids"].to(noisy_latents.device)

                    drop_prob = float(args.drop_coarse_control_prob)
                    if drop_prob > 0.0 and controlnet.training:
                        bsz = input_ids.shape[0]
                        drop_mask = (torch.rand(bsz, device=input_ids.device) < drop_prob)
                        if drop_mask.any():
                            empty_ids = empty_input_ids.to(input_ids.device).expand(bsz, -1)
                            input_ids = torch.where(drop_mask[:, None], empty_ids, input_ids)

                    encoder_hidden_states = text_encoder(input_ids, return_dict=False)[0]

                    # ---------------------------------------------------------
                    # ControlNet + UNet
                    # ---------------------------------------------------------
                    down_res, mid_res = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=eeg_cond,
                        added_cond_kwargs={},
                        return_dict=False,
                    )

                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        down_block_additional_residuals=[x.to(weight_dtype) for x in down_res],
                        mid_block_additional_residual=mid_res.to(weight_dtype),
                        return_dict=False,
                    )[0]

                    diffusion_loss = ((model_pred.float() - noise.float()) ** 2).mean()

                loss_total = torch.tensor(0.0, device=accelerator.device)

                if not args.train_eeg_only and full_mode:
                    loss_total = loss_total + diffusion_loss

                if sife is not None and (stage1_mode or full_mode):
                    loss_total = (
                        loss_total
                        + float(args.lambda_subject_inv) * sife_loss_inv
                        + float(args.lambda_subject_spec) * sife_loss_spec
                    )

                if recon_decoder is not None and (stage1_mode or full_mode):
                    loss_total = loss_total + float(args.lambda_recon) * recon_loss

                if ssfe_projector is not None and (stage2_mode or full_mode):
                    loss_total = loss_total + float(args.lambda_ssfe) * ssfe_loss

                if diffusion_prior is not None and (stage3_mode or full_mode):
                    loss_total = loss_total + float(args.lambda_prior) * prior_loss

                accelerator.backward(loss_total)

                if accelerator.sync_gradients and len(params_to_clip) > 0:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                current_epoch = global_step / num_update_steps_per_epoch

                epoch_logged_steps += 1
                epoch_loss_sum += loss_total.item()
                epoch_diffusion_loss_sum += diffusion_loss.item()
                epoch_recon_loss_sum += recon_loss.item()
                epoch_ssfe_loss_sum += ssfe_loss.item()
                epoch_prior_loss_sum += prior_loss.item()

                epoch_sife_inv_sum += sife_loss_inv.item()
                epoch_sife_spec_sum += sife_loss_spec.item()
                epoch_inv_acc_sum += inv_acc.item()
                epoch_spec_acc_sum += spec_acc.item()

                epoch_img_cls_loss_sum += image_cls_loss.item()
                epoch_img_dis_loss_sum += image_dis_loss.item()
                epoch_img_cls_acc_sum += image_cls_acc.item()
                epoch_img_dis_acc_sum += image_dis_acc.item()

                epoch_anchor_cls_loss_sum += anchor_cls_loss.item()
                epoch_anchor_visual_loss_sum += anchor_visual_loss.item()
                epoch_anchor_text_loss_sum += anchor_text_loss.item()
                epoch_anchor_cls_acc_sum += anchor_cls_acc.item()

                epoch_e_norm_sum += E_pooled.norm(dim=-1).mean().item()
                epoch_ei_norm_sum += E_i_norm.item()
                epoch_es_norm_sum += E_s_norm.item()
                epoch_ei_es_ratio_sum += Ei_Es_norm_ratio.item()
                epoch_ei_es_cos_sum += Ei_Es_cos.item()

                epoch_inv_entropy_sum += inv_entropy.item()
                epoch_spec_entropy_sum += spec_entropy.item()
                epoch_inv_entropy_norm_sum += inv_entropy_norm.item()
                epoch_spec_entropy_norm_sum += spec_entropy_norm.item()

                epoch_random_subject_acc_sum += random_subject_acc.item()
                epoch_inv_gap_sum += inv_acc_gap_vs_random.item()
                epoch_spec_gap_sum += spec_acc_gap_vs_random.item()

                epoch_lr_sum += lr_scheduler.get_last_lr()[0]

                if accelerator.is_main_process and (global_step % args.console_log_every == 0 or global_step == 1):
                    logger.info(
                        f"[step {global_step}] "
                        f"loss={loss_total.item():.4f} | "
                        f"diff={diffusion_loss.item():.4f} | "
                        f"inv={sife_loss_inv.item():.4f} | "
                        f"spec={sife_loss_spec.item():.4f} | "
                        f"inv_acc={inv_acc.item():.4f} | "
                        f"spec_acc={spec_acc.item():.4f} | "
                        f"rand_acc={random_subject_acc.item():.4f} | "
                        f"inv_gap={inv_acc_gap_vs_random.item():.4f} | "
                        f"spec_gap={spec_acc_gap_vs_random.item():.4f} | "
                        f"E_i_norm={E_i_norm.item():.4f} | "
                        f"E_s_norm={E_s_norm.item():.4f} | "
                        f"Ei_Es_ratio={Ei_Es_norm_ratio.item():.4f} | "
                        f"inv_ent={inv_entropy.item():.4f} | "
                        f"spec_ent={spec_entropy.item():.4f} | "
                        f"inv_ent_norm={inv_entropy_norm.item():.4f} | "
                        f"spec_ent_norm={spec_entropy_norm.item():.4f} | "
                        f"Ei_Es_cos={Ei_Es_cos.item():.4f} | "
                        f"recon={recon_loss.item():.4f} | "
                        f"ssfe={ssfe_loss.item():.4f} | "
                        f"img_cls={image_cls_loss.item():.4f} | "
                        f"img_dis={image_dis_loss.item():.4f} | "
                        f"img_cls_acc={image_cls_acc.item():.4f} | "
                        f"img_dis_acc={image_dis_acc.item():.4f} | "
                        f"anchor_cls={anchor_cls_loss.item():.4f} | "
                        f"anchor_cls_acc={anchor_cls_acc.item():.4f} | "
                        f"anchor_visual={anchor_visual_loss.item():.4f} | "
                        f"anchor_text={anchor_text_loss.item():.4f} | "
                        f"prior={prior_loss.item():.4f}"
                    )

                log_payload = {
                    "loss": loss_total.item(),
                    "diffusion_loss": diffusion_loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "E_norm": E_pooled.norm(dim=-1).mean().item(),
                    "epoch": current_epoch,
                }

                if sife is not None:
                    log_payload["loss_subject_inv"] = sife_loss_inv.item()
                    log_payload["loss_subject_spec"] = sife_loss_spec.item()
                    log_payload["acc_subject_inv"] = inv_acc.item()
                    log_payload["acc_subject_spec"] = spec_acc.item()

                    log_payload["E_i_norm"] = E_i_norm.item()
                    log_payload["E_s_norm"] = E_s_norm.item()
                    log_payload["Ei_Es_norm_ratio"] = Ei_Es_norm_ratio.item()

                    log_payload["entropy_subject_inv"] = inv_entropy.item()
                    log_payload["entropy_subject_spec"] = spec_entropy.item()
                    log_payload["entropy_subject_inv_norm"] = inv_entropy_norm.item()
                    log_payload["entropy_subject_spec_norm"] = spec_entropy_norm.item()
                    log_payload["Ei_Es_cos"] = Ei_Es_cos.item()

                    log_payload["random_subject_acc"] = random_subject_acc.item()
                    log_payload["acc_subject_inv_gap_vs_random"] = inv_acc_gap_vs_random.item()
                    log_payload["acc_subject_spec_gap_vs_random"] = spec_acc_gap_vs_random.item()

                if recon_decoder is not None:
                    log_payload["loss_recon"] = recon_loss.item()

                if ssfe_projector is not None:
                    log_payload["loss_ssfe"] = ssfe_loss.item()
                    log_payload["loss_image_cls"] = image_cls_loss.item()
                    log_payload["loss_image_dis"] = image_dis_loss.item()
                    log_payload["acc_image_cls"] = image_cls_acc.item()
                    log_payload["acc_image_dis"] = image_dis_acc.item()

                    log_payload["loss_anchor_cls"] = anchor_cls_loss.item()
                    log_payload["acc_anchor_cls"] = anchor_cls_acc.item()
                    log_payload["loss_anchor_visual"] = anchor_visual_loss.item()
                    log_payload["loss_anchor_text"] = anchor_text_loss.item()

                if diffusion_prior is not None:
                    log_payload["loss_prior"] = prior_loss.item()

                accelerator.log(log_payload, step=global_step)

                if global_step % args.validation_steps == 0:

                    val_metrics = run_validation_metrics(
                        val_dataloader,
                        eeg_backbone,
                        sife,
                        recon_decoder,
                        ssfe_projector,
                        diffusion_prior,
                        subject_to_local,
                        args,
                        accelerator,
                    )

                    if accelerator.is_main_process:
                        val_metrics["epoch"] = current_epoch
                        accelerator.log(val_metrics, step=global_step)

                        if args.log_validation_images and not args.train_eeg_only:
                            log_validation(
                                vae if not args.use_precomputed_latents else None,
                                text_encoder,
                                tokenizer,
                                unet,
                                controlnet,
                                args,
                                accelerator,
                                weight_dtype,
                                global_step,
                                is_final_validation=False,
                            )

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
            epoch_payload = {
                "epoch_only/epoch": float(epoch + 1),
                "epoch_only/loss": epoch_loss_sum / epoch_logged_steps,
                "epoch_only/diffusion_loss": epoch_diffusion_loss_sum / epoch_logged_steps,
                "epoch_only/lr": epoch_lr_sum / epoch_logged_steps,
                "epoch_only/E_norm": epoch_e_norm_sum / epoch_logged_steps,
            }

            if sife is not None:
                epoch_payload.update({
                    "epoch_only/loss_subject_inv": epoch_sife_inv_sum / epoch_logged_steps,
                    "epoch_only/loss_subject_spec": epoch_sife_spec_sum / epoch_logged_steps,
                    "epoch_only/acc_subject_inv": epoch_inv_acc_sum / epoch_logged_steps,
                    "epoch_only/acc_subject_spec": epoch_spec_acc_sum / epoch_logged_steps,
                    "epoch_only/E_i_norm": epoch_ei_norm_sum / epoch_logged_steps,
                    "epoch_only/E_s_norm": epoch_es_norm_sum / epoch_logged_steps,
                    "epoch_only/Ei_Es_norm_ratio": epoch_ei_es_ratio_sum / epoch_logged_steps,
                    "epoch_only/Ei_Es_cos": epoch_ei_es_cos_sum / epoch_logged_steps,
                    "epoch_only/entropy_subject_inv": epoch_inv_entropy_sum / epoch_logged_steps,
                    "epoch_only/entropy_subject_spec": epoch_spec_entropy_sum / epoch_logged_steps,
                    "epoch_only/entropy_subject_inv_norm": epoch_inv_entropy_norm_sum / epoch_logged_steps,
                    "epoch_only/entropy_subject_spec_norm": epoch_spec_entropy_norm_sum / epoch_logged_steps,
                    "epoch_only/random_subject_acc": epoch_random_subject_acc_sum / epoch_logged_steps,
                    "epoch_only/acc_subject_inv_gap_vs_random": epoch_inv_gap_sum / epoch_logged_steps,
                    "epoch_only/acc_subject_spec_gap_vs_random": epoch_spec_gap_sum / epoch_logged_steps,
                })

            if recon_decoder is not None:
                epoch_payload["epoch_only/loss_recon"] = epoch_recon_loss_sum / epoch_logged_steps

            if ssfe_projector is not None:
                epoch_payload.update({
                    "epoch_only/loss_ssfe": epoch_ssfe_loss_sum / epoch_logged_steps,
                    "epoch_only/loss_image_cls": epoch_img_cls_loss_sum / epoch_logged_steps,
                    "epoch_only/loss_image_dis": epoch_img_dis_loss_sum / epoch_logged_steps,
                    "epoch_only/acc_image_cls": epoch_img_cls_acc_sum / epoch_logged_steps,
                    "epoch_only/acc_image_dis": epoch_img_dis_acc_sum / epoch_logged_steps,
                    "epoch_only/loss_anchor_cls": epoch_anchor_cls_loss_sum / epoch_logged_steps,
                    "epoch_only/loss_anchor_visual": epoch_anchor_visual_loss_sum / epoch_logged_steps,
                    "epoch_only/loss_anchor_text": epoch_anchor_text_loss_sum / epoch_logged_steps,
                    "epoch_only/acc_anchor_cls": epoch_anchor_cls_acc_sum / epoch_logged_steps,
                })

            if diffusion_prior is not None:
                epoch_payload["epoch_only/loss_prior"] = epoch_prior_loss_sum / epoch_logged_steps

            accelerator.log(epoch_payload, step=global_step)

        if global_step >= args.max_train_steps:
            break

    # ---------------------------------------------------------
    # FINAL SAVE
    # ---------------------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet_unwrapped = None
        if controlnet is not None:
            controlnet_unwrapped = accelerator.unwrap_model(controlnet)
            if not args.train_eeg_only:
                controlnet_unwrapped.save_pretrained(args.output_dir)

        eeg_backbone_unwrapped = accelerator.unwrap_model(eeg_backbone)
        torch.save(
            eeg_backbone_unwrapped.state_dict(),
            os.path.join(args.output_dir, "eeg_backbone.pt"),
        )

        with open(os.path.join(args.output_dir, "eeg_backbone_config.json"), "w") as f:
            json.dump(
                {
                    "in_channels": 128,
                    "hidden_size": int(args.eeg_backbone_hidden_size),
                    "num_layers": int(args.eeg_backbone_num_layers),
                },
                f,
                indent=2,
            )

        if sife is not None:
            sife_unwrapped = accelerator.unwrap_model(sife)
            torch.save(
                sife_unwrapped.state_dict(),
                os.path.join(args.output_dir, "sife.pt"),
            )
            with open(os.path.join(args.output_dir, "sife_config.json"), "w") as f:
                json.dump(
                    {
                        "dim": int(args.eeg_backbone_hidden_size),
                        "seq_len": int(inferred_seq_len),
                        "num_subjects": int(len(active_train_subjects)),
                        "train_subjects": active_train_subjects,
                        "fi_layers": int(args.sife_num_layers),
                        "num_heads": int(args.sife_num_heads),
                        "grl_lambda_sife": float(args.grl_lambda_sife),
                    },
                    f,
                    indent=2,
                )

        if recon_decoder is not None:
            recon_unwrapped = accelerator.unwrap_model(recon_decoder)
            torch.save(
                recon_unwrapped.state_dict(),
                os.path.join(args.output_dir, "eeg_reconstruction_decoder.pt"),
            )
            with open(os.path.join(args.output_dir, "eeg_reconstruction_decoder_config.json"), "w") as f:
                json.dump(
                    {
                        "in_dim": int(args.eeg_backbone_hidden_size),
                        "hidden_dim": int(args.recon_hidden_dim),
                        "out_channels": 128,
                        "num_res_blocks": int(args.recon_num_blocks),
                        "loss_type": args.recon_loss_type,
                        "lambda_recon": float(args.lambda_recon),
                    },
                    f,
                    indent=2,
                )

        if ssfe_projector is not None:
            ssfe_unwrapped = accelerator.unwrap_model(ssfe_projector)
            torch.save(
                ssfe_unwrapped.state_dict(),
                os.path.join(args.output_dir, "ssfe_projector.pt"),
            )
            with open(os.path.join(args.output_dir, "ssfe_projector_config.json"), "w") as f:
                json.dump(
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
                        "lambda_anchor_text": float(args.lambda_anchor_text),
                        "anchor_visual_temperature": float(args.anchor_visual_temperature),
                        "anchor_text_temperature": float(args.anchor_text_temperature),
                    },
                    f,
                    indent=2,
                )

        if diffusion_prior is not None:
            prior_unwrapped = accelerator.unwrap_model(diffusion_prior)
            torch.save(
                prior_unwrapped.state_dict(),
                os.path.join(args.output_dir, "diffusion_prior.pt"),
            )
            with open(os.path.join(args.output_dir, "diffusion_prior_config.json"), "w") as f:
                json.dump(
                    {
                        "use_prior": True,
                        "lambda_prior": float(args.lambda_prior),
                        "prior_num_tokens": int(prior_num_tokens),
                        "prior_dim": int(prior_dim),
                        "prior_depth": int(args.prior_depth),
                        "prior_heads": int(args.prior_heads),
                        "prior_timesteps": int(args.prior_timesteps),
                        "prior_cond_drop_prob": float(args.prior_cond_drop_prob),
                    },
                    f,
                    indent=2,
                )

        if args.log_validation_images and not args.train_eeg_only:
            log_validation(
                vae if not args.use_precomputed_latents else None,
                text_encoder,
                tokenizer,
                unet,
                controlnet_unwrapped,
                args,
                accelerator,
                weight_dtype,
                global_step,
                is_final_validation=True,
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)