import argparse
import contextlib
import os
import re

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
    UNet2DConditionModel,
)

from data.eeg_dataset import make_test_dataset


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = text.replace("image of ", "")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_\-]", "", text)
    if len(text) == 0:
        text = "sample"
    return text[:max_len]


def tensor_chw_to_pil_01(x: torch.Tensor) -> Image.Image:
    """
    x: (3,H,W) in [0,1]
    """
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def tensor_chw_to_pil_neg1_1(x: torch.Tensor) -> Image.Image:
    """
    x: (3,H,W) in [-1,1]
    """
    x = ((x.detach().cpu().clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)
    return tensor_chw_to_pil_01(x)


def get_weight_dtype(dtype_str: str):
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    return torch.float32


def chunk_list(xs, batch_size):
    for i in range(0, len(xs), batch_size):
        yield xs[i:i + batch_size]


# ---------------------------------------------------------
# Args
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with trained GWIT ControlNet on subject-wise test split."
    )

    # Model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_path", type=str, required=True)

    # Dataset
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)

    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--conditioning_image_column", type=str, default="conditioning_image")
    parser.add_argument("--caption_column", type=str, default="caption")

    parser.add_argument("--test_subjects", type=int, nargs="+", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--max_test_samples_per_subject", type=int, default=None)

    # Caption logic
    parser.add_argument("--caption_from_classifier", action="store_true")
    parser.add_argument("--captioner_root", type=str, default=None)

    # Generation
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_images_per_sample", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)

    # Runtime
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
    )
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = get_weight_dtype(args.mixed_precision)

    os.makedirs(args.output_dir, exist_ok=True)
    gen_dir = os.path.join(args.output_dir, "generated")
    gt_dir = os.path.join(args.output_dir, "ground_truth")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Minimal args namespace for dataset factory
    # ---------------------------------------------------------
    class DatasetArgs:
        pass

    ds_args = DatasetArgs()
    ds_args.dataset_name = args.dataset_name
    ds_args.data_root = args.data_root
    ds_args.cache_dir = args.cache_dir
    ds_args.image_column = args.image_column
    ds_args.conditioning_image_column = args.conditioning_image_column
    ds_args.caption_column = args.caption_column
    ds_args.test_subjects = args.test_subjects
    ds_args.val_ratio = args.val_ratio
    ds_args.split_seed = args.split_seed
    ds_args.max_test_samples_per_subject = args.max_test_samples_per_subject
    ds_args.max_train_samples_per_subject = None
    ds_args.max_val_samples_per_subject = None
    ds_args.max_train_samples = None
    ds_args.caption_from_classifier = args.caption_from_classifier
    ds_args.captioner_root = args.captioner_root
    ds_args.use_precomputed_latents = False
    ds_args.latents_dir = None
    ds_args.use_precomputed_clip_embeds = False
    ds_args.clip_embeds_dir = None

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    test_dataset = make_test_dataset(ds_args, tokenizer, accelerator=None)
    print(f"[GEN] Test dataset size: {len(test_dataset)} | test_subjects={args.test_subjects}")

    # ---------------------------------------------------------
    # Models / pipeline
    # ---------------------------------------------------------
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path,
        torch_dtype=weight_dtype,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        torch_dtype=weight_dtype,
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        unet=unet,
        vae=vae,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    autocast_ctx = (
        torch.autocast(device.type, dtype=weight_dtype)
        if args.mixed_precision in {"fp16", "bf16"} and device.type == "cuda"
        else contextlib.nullcontext()
    )

    # ---------------------------------------------------------
    # Generation
    # ---------------------------------------------------------
    all_indices = list(range(len(test_dataset)))
    total_batches = (len(all_indices) + args.batch_size - 1) // args.batch_size

    for batch_indices in tqdm(chunk_list(all_indices, args.batch_size), total=total_batches, desc="Generating"):
        batch_examples = [test_dataset[idx] for idx in batch_indices]

        prompts = []
        safe_prompts = []
        conds = []

        for idx, ex in zip(batch_indices, batch_examples):
            prompt = tokenizer.decode(
                ex["input_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            if len(prompt) == 0:
                prompt = "image"

            safe_prompt = sanitize_filename(prompt)

            prompts.append(prompt)
            safe_prompts.append(safe_prompt)
            conds.append(ex["conditioning_pixel_values"])

            gt_path = os.path.join(gt_dir, f"gt_{idx:05d}_00_{safe_prompt}.png")
            if not os.path.exists(gt_path):
                tensor_chw_to_pil_neg1_1(ex["pixel_values"]).save(gt_path)

        cond_batch = torch.stack(conds, dim=0).to(device=device, dtype=weight_dtype)

        for j in range(args.num_images_per_sample):
            prompts_run = []
            conds_run = []
            meta_run = []
            generators = []

            for local_b, idx in enumerate(batch_indices):
                safe_prompt = safe_prompts[local_b]
                out_path = os.path.join(
                    gen_dir,
                    f"output_{idx:05d}_{j:02d}_{safe_prompt}.png"
                )

                if os.path.exists(out_path):
                    continue

                prompts_run.append(prompts[local_b])
                conds_run.append(cond_batch[local_b])
                meta_run.append((idx, safe_prompt))

                gen = torch.Generator(device=device)
                gen.manual_seed(args.seed + idx * 1000 + j)
                generators.append(gen)

            if len(prompts_run) == 0:
                continue

            conds_run = torch.stack(conds_run, dim=0)

            with torch.no_grad():
                with autocast_ctx:
                    result = pipe(
                        prompt=prompts_run,
                        image=conds_run,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=generators,
                    )

            for img, (idx, safe_prompt) in zip(result.images, meta_run):
                out_path = os.path.join(
                    gen_dir,
                    f"output_{idx:05d}_{j:02d}_{safe_prompt}.png"
                )
                img.save(out_path)

    print(f"[GEN] Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()