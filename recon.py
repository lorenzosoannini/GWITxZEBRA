import os
import sys
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Project-relative imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = PROJECT_ROOT / "models"
GEN_MODELS_ROOT = MODELS_ROOT / "generative_models"
for p in (MODELS_ROOT, GEN_MODELS_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from data.eeg_dataset import make_test_dataset, make_collate_fn
from models.eeg_backbone import GWITEEGBackbone, load_eeg_backbone_from_ckpt
from models.sife import SIFE
from models.ssfe import SSFEProjector
from models.prior import PriorNetwork, BrainDiffusionPrior

from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims

# tf32 faster on Ampere/T4
torch.backends.cuda.matmul.allow_tf32 = True


# ---------------------------------------------------------------------
# Small local utils
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


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def tensor_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    """
    x: (3,H,W), assumed in [0,1]
    returns uint8 RGB (H,W,3)
    """
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()
    return x


def save_side_by_side(pred: torch.Tensor, gt: torch.Tensor, out_path: Path, resize_to=224):
    pred_np = tensor_to_uint8_rgb(pred)
    gt_np = tensor_to_uint8_rgb(gt)

    pred_np = cv2.resize(pred_np, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
    gt_np = cv2.resize(gt_np, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)

    pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)

    vis = np.concatenate([pred_bgr, gt_bgr], axis=0)
    cv2.imwrite(str(out_path), vis)


def unclip_recon(
    clip_tokens: torch.Tensor,
    diffusion_engine,
    vector_suffix: torch.Tensor,
    num_samples: int = 1,
    offset_noise_level: float = 0.04,
    device: str = "cuda",
):
    """
    clip_tokens: (1, T, D) or (T, D)
    returns: (num_samples, 3, H, W) in [0,1]
    """
    if clip_tokens.ndim == 2:
        clip_tokens = clip_tokens.unsqueeze(0)

    assert clip_tokens.ndim == 3, f"Expected (B,T,D), got {tuple(clip_tokens.shape)}"
    assert clip_tokens.shape[0] == 1, "This helper currently expects a single sample"

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), diffusion_engine.ema_scope():
        z = torch.randn(num_samples, 4, 96, 96, device=device)

        cond_tokens = clip_tokens.repeat(num_samples, 1, 1).to(device)
        c = {
            "crossattn": cond_tokens,
            "vector": vector_suffix.repeat(num_samples, 1).to(device),
        }

        uc_tokens = torch.randn_like(clip_tokens).repeat(num_samples, 1, 1).to(device)
        uc = {
            "crossattn": uc_tokens,
            "vector": vector_suffix.repeat(num_samples, 1).to(device),
        }

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )

        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(1.0 + sigmas[0] ** 2.0)

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)

        samples = torch.clamp(samples_x * 0.8 + 0.2, min=0.0, max=1.0)
        return samples


# ---------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="EEG -> prior -> unCLIP reconstruction")

    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing eeg_backbone.pt / sife.pt / ssfe_projector.pt / diffusion_prior.pt and configs")

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--clip_embeds_dir", type=str, default=None)
    parser.add_argument("--use_precomputed_clip_embeds", action="store_true")

    parser.add_argument("--test_subjects", type=int, nargs="+", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--conditioning_image_column", type=str, default="conditioning_image")
    parser.add_argument("--caption_column", type=str, default="caption")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for EEG -> prior token generation")
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prior_inference_steps", type=int, default=20,
                        help="Sampling steps of the diffusion prior, ZEBRA-style")
    parser.add_argument("--prior_cond_scale", type=float, default=1.0)

    parser.add_argument("--num_samples_per_image", type=int, default=1,
                        help="Usually keep 1 for fair evaluation")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_pt", action="store_true")

    parser.add_argument("--unclip_ckpt", type=str, required=True,
                        help="Path to unclip6_epoch0_step110000.ckpt")
    parser.add_argument("--unclip_config", type=str, default=None,
                        help="Optional explicit path to unclip6.yaml. If omitted, uses models/generative_models/configs/unclip6.yaml")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Load trained modules
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
        mlp_ratio=4.0,
        dropout=0.0,
        learned_query_mode="pos_emb",
    )

    diffusion_prior = BrainDiffusionPrior(
        net=prior_net,
        image_embed_dim=int(prior_cfg["prior_dim"]),
        timesteps=int(prior_cfg["prior_timesteps"]),
        cond_drop_prob=float(prior_cfg["prior_cond_drop_prob"]),
        predict_x_start=True,
        training_clamp_l2norm=False,
        sampling_clamp_l2norm=False,
    )
    diffusion_prior.load_state_dict(torch.load(model_dir / "diffusion_prior.pt", map_location="cpu"))
    diffusion_prior.to(device).eval()

    return eeg_backbone, sife, ssfe, diffusion_prior, sife_cfg, ssfe_cfg, prior_cfg


# ---------------------------------------------------------------------
# unCLIP loader
# ---------------------------------------------------------------------
def prepare_unclip(args, device):
    if args.unclip_config is not None:
        config_path = Path(args.unclip_config)
    else:
        config_path = PROJECT_ROOT / "models" / "generative_models" / "configs" / "unclip6.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"unclip config non trovato: {config_path}")
    if not os.path.exists(args.unclip_ckpt):
        raise FileNotFoundError(f"unclip checkpoint non trovato: {args.unclip_ckpt}")

    config = OmegaConf.load(str(config_path))
    config = OmegaConf.to_container(config, resolve=True)

    unclip_params = config["model"]["params"]
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    conditioner_config = unclip_params["conditioner_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]

    first_stage_config["target"] = "sgm.models.autoencoder.AutoencoderKL"
    sampler_config["params"]["num_steps"] = 38

    diffusion_engine = DiffusionEngine(
        network_config=network_config,
        denoiser_config=denoiser_config,
        first_stage_config=first_stage_config,
        conditioner_config=conditioner_config,
        sampler_config=sampler_config,
        scale_factor=scale_factor,
        disable_first_stage_autocast=disable_first_stage_autocast,
    )

    ckpt = torch.load(args.unclip_ckpt, map_location="cpu")
    diffusion_engine.load_state_dict(ckpt["state_dict"], strict=False)
    del ckpt

    diffusion_engine.eval().requires_grad_(False)
    diffusion_engine.to(device)

    return diffusion_engine


def prepare_vector_suffix(diffusion_engine, device):
    batch = {
        "jpg": torch.randn(1, 3, 1, 1, device=device),
        "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2, device=device),
    }
    out = diffusion_engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    return vector_suffix


# ---------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames_generated"
    vis_dir = output_dir / "vis_img"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

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
    ds_args.use_precomputed_clip_embeds = args.use_precomputed_clip_embeds
    ds_args.clip_embeds_dir = args.clip_embeds_dir
    ds_args.max_test_samples_per_subject = None
    ds_args.val_ratio = args.val_ratio
    ds_args.split_seed = args.split_seed
    ds_args.test_subjects = args.test_subjects

    test_dataset = make_test_dataset(ds_args, tokenizer, accelerator=None)
    collate_fn = make_collate_fn(args.dataset_name)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    eeg_backbone, sife, ssfe, diffusion_prior, sife_cfg, ssfe_cfg, prior_cfg = load_trained_modules(
        model_dir=model_dir,
        device=device,
    )

    diffusion_engine = prepare_unclip(args, device)
    vector_suffix = prepare_vector_suffix(diffusion_engine, device)

    all_recons = []
    all_gts = []

    sample_index = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating reconstructions"):
            eeg_cond = batch["conditioning_pixel_values"].to(device, dtype=torch.float32)
            gt_images = batch["pixel_values"].to(device, dtype=torch.float32)

            # Dataset images are in [-1,1], convert to [0,1] for saving/metrics later
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

            for i in range(prior_tokens.shape[0]):
                sample_index += 1

                pred_samples = unclip_recon(
                    clip_tokens=prior_tokens[i:i+1],
                    diffusion_engine=diffusion_engine,
                    vector_suffix=vector_suffix,
                    num_samples=args.num_samples_per_image,
                    device=str(device),
                )

                pred_img = pred_samples[0].detach().cpu()
                gt_img = gt_images_01[i].detach().cpu()

                all_recons.append(pred_img)
                all_gts.append(gt_img)

                if args.save_vis:
                    save_side_by_side(
                        pred=pred_img,
                        gt=gt_img,
                        out_path=vis_dir / f"frame_{sample_index:05d}.jpg",
                        resize_to=224,
                    )

    all_recons = torch.stack(all_recons, dim=0)
    all_gts = torch.stack(all_gts, dim=0)

    # ZEBRA salva a 256x256
    resize_256 = transforms.Resize((256, 256))
    all_recons_256 = resize_256(all_recons).float()
    all_gts_256 = resize_256(all_gts).float()

    print(f"[DONE] all_recons: {tuple(all_recons_256.shape)}")
    print(f"[DONE] all_gts:    {tuple(all_gts_256.shape)}")

    if args.save_pt:
        torch.save(all_recons_256, frames_dir / "all_recons.pt")
        torch.save(all_gts_256, frames_dir / "all_gts.pt")
        print(f"[SAVED] {frames_dir / 'all_recons.pt'}")
        print(f"[SAVED] {frames_dir / 'all_gts.pt'}")


if __name__ == "__main__":
    main()