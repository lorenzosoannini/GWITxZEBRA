import os
import sys
import gc
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from safetensors.torch import load_file as load_safetensors

# ---------------------------------------------------------------------
# Project-relative imports
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = PROJECT_ROOT / "models"
GEN_MODELS_ROOT = MODELS_ROOT / "generative_models"

for p in [PROJECT_ROOT, MODELS_ROOT, GEN_MODELS_ROOT]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# alias richiesto da molti import interni SGM
import generative_models.sgm as gm_sgm
sys.modules["sgm"] = gm_sgm

from generative_models.sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    append_dims,
)
from generative_models.sgm.modules.encoders.modules import ConcatTimestepEmbedderND
from generative_models.sgm.modules.autoencoding.temporal_ae import VideoDecoder

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


def log_ram(tag: str):
    try:
        import psutil
        vm = psutil.virtual_memory()
        used = (vm.total - vm.available) / (1024 ** 3)
        avail = vm.available / (1024 ** 3)
        print(f"[RAM] {tag} | used={used:.2f} GB | avail={avail:.2f} GB")
    except Exception as e:
        print(f"[RAM] {tag} | psutil unavailable: {e}")


def log_vram(tag: str, device: torch.device):
    if device.type != "cuda":
        return
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(f"[VRAM] {tag} | allocated={allocated:.2f} GB | reserved={reserved:.2f} GB")


def maybe_empty_cuda_cache(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def hard_cleanup(device: torch.device):
    gc.collect()
    maybe_empty_cuda_cache(device)


def tensor_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    x = x.detach().to(dtype=torch.float32).cpu().clamp(0, 1)
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


def save_pred_only(pred: torch.Tensor, out_path: Path, resize_to=224):
    pred_np = tensor_to_uint8_rgb(pred)
    pred_np = cv2.resize(pred_np, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
    pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), pred_bgr)


def resolve_model_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.lower()
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported --model_dtype={dtype_name}. Use fp16 or fp32.")


# ---------------------------------------------------------------------
# Lightweight unCLIP engine
# ---------------------------------------------------------------------
class MinimalUnclipEngine(torch.nn.Module):
    def __init__(
        self,
        network_config,
        denoiser_config,
        first_stage_config,
        sampler_config,
        scale_factor: float = 1.0,
        disable_first_stage_autocast: bool = False,
        network_wrapper=None,
        en_and_decode_n_samples_a_time=None,
    ):
        super().__init__()

        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(
            default(
                network_wrapper,
                "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper",
            )
        )(model, compile_model=False)

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = instantiate_from_config(sampler_config)
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        self._init_first_stage(first_stage_config)

    def _init_first_stage(self, config):
        model = instantiate_from_config(config).eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False
        self.first_stage_model = model

    @property
    def device(self):
        return next(self.parameters()).device

    def ema_scope(self, context=None):
        class _NullCtx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                return False

        return _NullCtx()

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])

        import math
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []

        for n in range(n_rounds):
            z_chunk = z[n * n_samples: (n + 1) * n_samples].to(torch.float32)
            if isinstance(self.first_stage_model.decoder, VideoDecoder):
                kwargs = {"timesteps": len(z_chunk)}
            else:
                kwargs = {}

            # decode sempre in fp32
            with torch.autocast("cuda", enabled=False):
                out = self.first_stage_model.decode(z_chunk, **kwargs)

            all_out.append(out)

        out = torch.cat(all_out, dim=0)
        return out


# ---------------------------------------------------------------------
# Build vector suffix manually
# ---------------------------------------------------------------------
@torch.no_grad()
def build_vector_suffix(device: torch.device, dtype: torch.dtype):
    size_embedder = ConcatTimestepEmbedderND(outdim=256).to(device=device, dtype=dtype).eval()
    crop_embedder = ConcatTimestepEmbedderND(outdim=256).to(device=device, dtype=dtype).eval()

    original_size = torch.tensor([[768.0, 768.0]], device=device, dtype=dtype)
    crop_coords = torch.tensor([[0.0, 0.0]], device=device, dtype=dtype)

    size_vec = size_embedder(original_size)
    crop_vec = crop_embedder(crop_coords)

    vector_suffix = torch.cat([size_vec, crop_vec], dim=1)

    del size_embedder, crop_embedder, original_size, crop_coords, size_vec, crop_vec
    return vector_suffix


# ---------------------------------------------------------------------
# unCLIP helper
# ---------------------------------------------------------------------
@torch.no_grad()
def unclip_recon_batch(
    clip_tokens: torch.Tensor,
    engine: MinimalUnclipEngine,
    vector_suffix: torch.Tensor,
    num_samples_per_image: int = 1,
    offset_noise_level: float = 0.04,
    device: str = "cuda",
    debug_stats: bool = False,

):
    assert clip_tokens.ndim == 3, f"Expected (B,T,D), got {tuple(clip_tokens.shape)}"

    bsz = clip_tokens.shape[0]
    total = bsz * num_samples_per_image
    target_dtype = next(engine.parameters()).dtype

    use_amp = ("cuda" in device) and (target_dtype == torch.float16)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16), engine.ema_scope():
        z = torch.randn(total, 4, 96, 96, device=device, dtype=target_dtype)

        cond_tokens = clip_tokens.repeat_interleave(num_samples_per_image, dim=0).to(
            device=device, dtype=target_dtype, non_blocking=True
        )
        vector = vector_suffix.repeat(total, 1).to(
            device=device, dtype=target_dtype, non_blocking=True
        )

        if debug_stats:
            print("[DEBUG unCLIP] cond_tokens shape:", tuple(cond_tokens.shape))
            print(
                "[DEBUG unCLIP] cond_tokens min/max/mean/std:",
                cond_tokens.min().item(),
                cond_tokens.max().item(),
                cond_tokens.mean().item(),
                cond_tokens.std().item(),
            )
            print(
                "[DEBUG unCLIP] vector min/max/mean/std:",
                vector.min().item(),
                vector.max().item(),
                vector.mean().item(),
                vector.std().item(),
            )

        c = {
            "crossattn": cond_tokens,
            "vector": vector,
        }

        uc_tokens = torch.randn_like(cond_tokens)
        uc = {
            "crossattn": uc_tokens,
            "vector": vector,
        }

        noise = torch.randn_like(z)
        sigmas = engine.sampler.discretization(engine.sampler.num_steps)
        sigma = sigmas[0].to(device=z.device, dtype=target_dtype)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device, dtype=target_dtype), z.ndim
            )

        sigma0 = sigmas[0].to(device=z.device, dtype=target_dtype)
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            torch.tensor(1.0, device=z.device, dtype=target_dtype) + sigma0 ** 2.0
        )

        def denoiser(x, sigma_in, c_in):
            return engine.denoiser(engine.model, x, sigma_in, c_in)

        samples_z = engine.sampler(denoiser, noised_z, cond=c, uc=uc)

        if debug_stats:
            print(
                "[DEBUG unCLIP] samples_z min/max/mean/std:",
                samples_z.min().item(),
                samples_z.max().item(),
                samples_z.mean().item(),
                samples_z.std().item(),
            )
            print("[DEBUG unCLIP] nan in samples_z:", torch.isnan(samples_z).any().item())

        samples_x = engine.decode_first_stage(samples_z)

        if debug_stats:
            print(
                "[DEBUG unCLIP] samples_x min/max/mean/std:",
                samples_x.min().item(),
                samples_x.max().item(),
                samples_x.mean().item(),
                samples_x.std().item(),
            )
            print("[DEBUG unCLIP] nan in samples_x:", torch.isnan(samples_x).any().item())

        samples = torch.clamp(samples_x * 0.8 + 0.2, min=0.0, max=1.0)

        if debug_stats:
            print(
                "[DEBUG unCLIP] samples min/max/mean/std:",
                samples.min().item(),
                samples.max().item(),
                samples.mean().item(),
                samples.std().item(),
            )
            print("[DEBUG unCLIP] nan in samples:", torch.isnan(samples).any().item())

        del z, cond_tokens, vector, c, uc_tokens, uc
        del noise, sigmas, sigma, sigma0, noised_z, samples_z, samples_x

        return samples


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Stage B: prior tokens -> unCLIP reconstructions")

    parser.add_argument("--stageA_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--unclip_ckpt", type=str, required=True)
    parser.add_argument("--unclip_config", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples_per_image", type=int, default=1)
    parser.add_argument("--decode_batch_size", type=int, default=1)
    parser.add_argument("--model_dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_minibatch", action="store_true",
                        help="Riprende anche dentro uno shard saltando minibatch già salvati")
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--save_pred_only", action="store_true")
    parser.add_argument("--save_final_manifest", action="store_true")
    parser.add_argument("--log_ram", action="store_true")

    parser.add_argument("--debug_unclip_stats", action="store_true",
                        help="Stampa statistiche numeriche interne di unCLIP per il primo minibatch")

    return parser.parse_args()


# ---------------------------------------------------------------------
# prepare minimal unCLIP
# ---------------------------------------------------------------------
def prepare_unclip(args, device: torch.device):
    if args.unclip_config is not None:
        config_path = Path(args.unclip_config)
    else:
        config_path = PROJECT_ROOT / "models" / "generative_models" / "configs" / "unclip6.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"unclip config non trovato: {config_path}")
    if not os.path.exists(args.unclip_ckpt):
        raise FileNotFoundError(f"unclip checkpoint non trovato: {args.unclip_ckpt}")

    model_dtype = resolve_model_dtype(args.model_dtype)

    if args.log_ram:
        log_ram("before OmegaConf.load")

    config = OmegaConf.load(str(config_path))
    config = OmegaConf.to_container(config, resolve=True)

    unclip_params = config["model"]["params"]
    network_config = unclip_params["network_config"]
    denoiser_config = unclip_params["denoiser_config"]
    first_stage_config = unclip_params["first_stage_config"]
    sampler_config = unclip_params["sampler_config"]
    scale_factor = unclip_params["scale_factor"]
    disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]

    first_stage_config["target"] = "sgm.models.autoencoder.AutoencoderKL"
    sampler_config["params"]["num_steps"] = 38

    if args.log_ram:
        log_ram("before MinimalUnclipEngine init")

    engine = MinimalUnclipEngine(
        network_config=network_config,
        denoiser_config=denoiser_config,
        first_stage_config=first_stage_config,
        sampler_config=sampler_config,
        scale_factor=scale_factor,
        disable_first_stage_autocast=disable_first_stage_autocast,
        network_wrapper=None,
    )

    if args.log_ram:
        log_ram("after MinimalUnclipEngine init")

    engine.eval().requires_grad_(False)

    print(f"[UNCLIP] target model dtype: {model_dtype}")
    print("[UNCLIP] before engine.to(device)")
    engine.to(device=device)
    print("[UNCLIP] after engine.to(device)")
    # cast solo UNet / denoiser path, NON il first-stage VAE

    if model_dtype == torch.float16:
        engine.model.to(dtype=torch.float16)
        engine.denoiser.to(dtype=torch.float16)

    # first-stage sempre in fp32 per stabilità numerica
    engine.first_stage_model.to(dtype=torch.float32)
    log_vram("after engine.to(device)", device)

    # liberiamo subito il più possibile prima del checkpoint load
    del config, unclip_params, network_config, denoiser_config, first_stage_config, sampler_config
    hard_cleanup(device)

    if args.log_ram:
        log_ram("after moving empty engine to GPU")

    print("[UNCLIP] before checkpoint load")
    if args.unclip_ckpt.endswith(".safetensors"):
        state_dict = load_safetensors(args.unclip_ckpt, device="cpu")
        print("[UNCLIP] loaded safetensors checkpoint")
    else:
        ckpt = torch.load(args.unclip_ckpt, map_location="cpu")
        print("[UNCLIP] loaded torch checkpoint")
        state_dict = ckpt["state_dict"]
        del ckpt

    print(f"[UNCLIP] loaded state_dict keys: {len(state_dict)}")

    is_already_filtered = all(
        k.startswith(("model.", "denoiser.", "first_stage_model."))
        for k in state_dict.keys()
    )

    if is_already_filtered:
        load_state = state_dict
        print("[UNCLIP] checkpoint already filtered")
    else:
        print("[UNCLIP] checkpoint not filtered, filtering now")
        load_state = {
            k: v for k, v in state_dict.items()
            if k.startswith(("model.", "denoiser.", "first_stage_model."))
        }
        print(f"[UNCLIP] filtered keys: {len(load_state)}")

    unet_dtype = torch.float16 if args.model_dtype == "fp16" else torch.float32

    for k, v in list(load_state.items()):
        if torch.is_tensor(v) and v.is_floating_point():
            if k.startswith("first_stage_model."):
                load_state[k] = v.to(dtype=torch.float32)
            else:
                load_state[k] = v.to(dtype=unet_dtype)

    print("[UNCLIP] before load_state_dict")
    missing, unexpected = engine.load_state_dict(load_state, strict=False)
    print("[UNCLIP] after load_state_dict")
    print(f"[UNCLIP] missing keys: {len(missing)}")
    print(f"[UNCLIP] unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("[UNCLIP] first missing keys:", missing[:20])
    if len(unexpected) > 0:
        print("[UNCLIP] first unexpected keys:", unexpected[:20])

    del state_dict, load_state
    hard_cleanup(device)

    if args.log_ram:
        log_ram("after unclip ckpt load")
    log_vram("after unclip ckpt load", device)

    return engine


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_model_dtype(args.model_dtype)

    stageA_dir = Path(args.stageA_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_shards_dir = stageA_dir / "token_shards"
    gt_shards_dir = stageA_dir / "gt_shards"

    if not token_shards_dir.exists():
        raise FileNotFoundError(f"Token shards dir non trovata: {token_shards_dir}")

    recon_shards_dir = output_dir / "recon_shards"
    recon_shards_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = output_dir / "vis_img"
    pred_dir = output_dir / "pred_img"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_pred_only:
        pred_dir.mkdir(parents=True, exist_ok=True)

    token_shards = sorted(token_shards_dir.glob("prior_tokens_shard_*.pt"))
    if len(token_shards) == 0:
        raise RuntimeError(f"Nessuno shard trovato in {token_shards_dir}")

    if args.log_ram:
        log_ram("before prepare_unclip")

    engine = prepare_unclip(args, device)
    vector_suffix = build_vector_suffix(device, next(engine.parameters()).dtype)

    if args.log_ram:
        log_ram("after prepare_unclip")
    log_vram("after prepare_unclip", device)

    resize_256 = transforms.Resize((256, 256))

    global_index = 0
    decoded_shards = 0

    shard_pbar = tqdm(token_shards, desc="Stage B: decoding shards", dynamic_ncols=True)

    for shard_idx, token_path in enumerate(shard_pbar):
        out_recon_path = recon_shards_dir / f"recons_shard_{shard_idx:05d}.pt"

        if args.resume and out_recon_path.exists():
            shard_tokens = torch.load(token_path, map_location="cpu")
            n_shard = int(shard_tokens.shape[0])
            del shard_tokens
            global_index += n_shard
            decoded_shards += 1
            shard_pbar.set_postfix({
                "status": "resume-shard",
                "decoded_shards": decoded_shards,
                "decoded_samples": global_index,
            })
            hard_cleanup(device)
            continue

        if args.log_ram:
            log_ram(f"before load shard {shard_idx}")

        prior_tokens_cpu = torch.load(token_path, map_location="cpu")
        if prior_tokens_cpu.ndim != 3:
            raise RuntimeError(f"Shard {token_path.name}: expected (B,T,D), got {tuple(prior_tokens_cpu.shape)}")

        gt_images_cpu = None
        gt_path = gt_shards_dir / f"gts_shard_{shard_idx:05d}.pt"
        if gt_path.exists():
            gt_images_cpu = torch.load(gt_path, map_location="cpu")

        if args.log_ram:
            log_ram(f"after load shard {shard_idx}")

        shard_recons = []
        bsz = prior_tokens_cpu.shape[0]

        mb_starts = list(range(0, bsz, args.decode_batch_size))
        mb_pbar = tqdm(
            mb_starts,
            desc=f"Shard {shard_idx:02d}",
            leave=False,
            dynamic_ncols=True,
        )

        for mb_idx, start in enumerate(mb_pbar):
            end = min(start + args.decode_batch_size, bsz)
            mb_out_path = recon_shards_dir / f"recons_shard_{shard_idx:05d}_mb_{mb_idx:05d}.pt"

            if args.resume_minibatch and mb_out_path.exists():
                recon_mb_256 = torch.load(mb_out_path, map_location="cpu")
                shard_recons.append(recon_mb_256)

                resumed_count = min(args.decode_batch_size, bsz - start)
                if not (args.save_pred_only or args.save_vis):
                    global_index += resumed_count

                mb_pbar.set_postfix({
                    "samples": f"{end}/{bsz}",
                    "status": "resumed",
                })
                continue

            tokens_mb = prior_tokens_cpu[start:end].to(
                device=device,
                dtype=model_dtype,
                non_blocking=True,
            )

            debug_this_mb = bool(args.debug_unclip_stats and shard_idx == 0 and mb_idx == 0)

            recon_mb = unclip_recon_batch(
                clip_tokens=tokens_mb,
                engine=engine,
                vector_suffix=vector_suffix,
                num_samples_per_image=args.num_samples_per_image,
                device=str(device),
                debug_stats=debug_this_mb,
            ).detach().cpu()

            recon_mb_256 = resize_256(recon_mb.to(torch.float32)).float()
            torch.save(recon_mb_256, mb_out_path)
            shard_recons.append(recon_mb_256)

            if args.save_pred_only or args.save_vis:
                for local_i in range(end - start):
                    pred_img = recon_mb[local_i * args.num_samples_per_image].detach().cpu()

                    if args.save_pred_only:
                        save_pred_only(
                            pred=pred_img,
                            out_path=pred_dir / f"pred_{global_index + 1:05d}.jpg",
                            resize_to=224,
                        )

                    if args.save_vis and gt_images_cpu is not None:
                        gt_img = gt_images_cpu[start + local_i].detach().cpu()
                        save_side_by_side(
                            pred=pred_img,
                            gt=gt_img,
                            out_path=vis_dir / f"frame_{global_index + 1:05d}.jpg",
                            resize_to=224,
                        )
                        del gt_img

                    global_index += 1
                    del pred_img
            else:
                global_index += (end - start)

            del tokens_mb
            if not (args.save_pred_only or args.save_vis):
                del recon_mb
            del recon_mb_256
            hard_cleanup(device)

            if device.type == "cuda":
                vram_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
                mb_pbar.set_postfix({
                    "samples": f"{end}/{bsz}",
                    "vram_gb": f"{vram_alloc:.2f}",
                })
            else:
                mb_pbar.set_postfix({
                    "samples": f"{end}/{bsz}",
                })

            if args.log_ram:
                log_ram(f"after minibatch {start}:{end} of shard {shard_idx}")
            log_vram(f"after minibatch {start}:{end} of shard {shard_idx}", device)

        shard_recons_tensor = torch.cat(shard_recons, dim=0)
        torch.save(shard_recons_tensor, out_recon_path)

        if args.resume_minibatch:
            for mb_idx in range(len(mb_starts)):
                mb_out_path = recon_shards_dir / f"recons_shard_{shard_idx:05d}_mb_{mb_idx:05d}.pt"
                if mb_out_path.exists():
                    mb_out_path.unlink()

        del shard_recons_tensor
        del shard_recons
        del prior_tokens_cpu
        if gt_images_cpu is not None:
            del gt_images_cpu

        decoded_shards += 1
        hard_cleanup(device)

        shard_pbar.set_postfix({
            "status": "done",
            "decoded_shards": decoded_shards,
            "decoded_samples": global_index,
        })

        if args.log_ram:
            log_ram(f"after save shard {shard_idx}")
        log_vram(f"after save shard {shard_idx}", device)

    if args.save_final_manifest:
        manifest = {
            "decoded_shards": decoded_shards,
            "decoded_samples": global_index,
            "num_samples_per_image": args.num_samples_per_image,
            "decode_batch_size": args.decode_batch_size,
            "model_dtype": args.model_dtype,
            "resume": args.resume,
            "resume_minibatch": args.resume_minibatch,
            "stageA_dir": str(stageA_dir),
            "output_dir": str(output_dir),
            "unclip_ckpt": args.unclip_ckpt,
        }
        with open(output_dir / "decode_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    print(f"[DONE] decoded shards: {decoded_shards}")
    print(f"[DONE] decoded samples: {global_index}")


if __name__ == "__main__":
    main()