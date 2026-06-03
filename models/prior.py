import random
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffusers.models.autoencoders.vae import Decoder
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import (
    l2norm,
    default,
    exists,
    RotaryEmbedding,
    SinusoidalPosEmb,
    MLP,
    Rearrange,
    repeat,
    rearrange,
    prob_mask_like,
    LayerNorm,
    RelPosBias,
    Attention,
    FeedForward,
)


class FlaggedCausalTransformer(nn.Module):
    """
    ZEBRA-style transformer wrapper using dalle2-pytorch primitives.

    Notes:
    - ZEBRA's PriorNetwork class has causal=True as constructor default,
      but train_zebra.py instantiates it with causal=False.
    - For alignment with ZEBRA training, use causal=False in the PriorNetwork call.
    """

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        norm_in: bool = False,
        norm_out: bool = True,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        final_proj: bool = True,
        normformer: bool = False,
        rotary_emb: bool = True,
        causal: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()
        self.rel_pos_bias = RelPosBias(heads=heads)
        rotary_emb = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            causal=causal,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_emb=rotary_emb,
                        ),
                        FeedForward(
                            dim=dim,
                            mult=ff_mult,
                            dropout=ff_dropout,
                            post_activation_norm=normformer,
                        ),
                    ]
                )
            )

        self.norm = LayerNorm(dim, stable=True) if norm_out else nn.Identity()
        self.project_out = nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()

    def set_gradient_checkpointing(self, enabled: bool = True):
        self.use_gradient_checkpointing = bool(enabled)

    def _forward_layer(self, x: torch.Tensor, attn: nn.Module, ff: nn.Module, attn_bias: torch.Tensor) -> torch.Tensor:
        x = attn(x, attn_bias=attn_bias) + x
        x = ff(x) + x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, device = x.shape[1], x.device
        x = self.init_norm(x)
        attn_bias = self.rel_pos_bias(n, n + 1, device=device)

        for attn, ff in self.layers:
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint(
                    self._forward_layer,
                    x,
                    attn,
                    ff,
                    attn_bias,
                    use_reentrant=False,
                )
            else:
                x = self._forward_layer(x, attn, ff, attn_bias)

        out = self.norm(x)
        return self.project_out(out)


class PriorNetwork(nn.Module):
    """
    Faithful ZEBRA-style token prior network using dalle2-pytorch primitives.

    Expected shapes:
        image_embed: (B, num_tokens, dim)
        brain_embed/text_embed: (B, num_tokens, dim)

    ZEBRA train_zebra.py uses:
        dim=1664
        depth=6
        dim_head=52
        heads=1664 // 52 = 32
        causal=False
        num_tokens=256
        learned_query_mode='pos_emb'
    """

    def __init__(
        self,
        dim: int,
        num_timesteps: Optional[int] = None,
        num_time_embeds: int = 1,
        num_tokens: int = 256,
        causal: bool = False,
        learned_query_mode: str = "pos_emb",
        depth: int = 6,
        heads: int = 32,
        dim_head: int = 52,
        ff_mult: int = 4,
        norm_in: bool = False,
        norm_out: bool = True,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        final_proj: bool = True,
        normformer: bool = False,
        rotary_emb: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds)
            if exists(num_timesteps)
            else nn.Sequential(
                SinusoidalPosEmb(dim),
                MLP(dim, dim * num_time_embeds),
            ),
            Rearrange("b (n d) -> b n d", n=num_time_embeds),
        )

        if self.learned_query_mode == "token":
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        elif self.learned_query_mode == "pos_emb":
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        elif self.learned_query_mode == "all_pos_emb":
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens * 2 + 1, dim) * scale)
        elif self.learned_query_mode == "none":
            self.learned_query = None
        else:
            raise ValueError(f"Unsupported learned_query_mode={learned_query_mode}")

        self.causal_transformer = FlaggedCausalTransformer(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            norm_in=norm_in,
            norm_out=norm_out,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            final_proj=final_proj,
            normformer=normformer,
            rotary_emb=rotary_emb,
            causal=causal,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    def set_gradient_checkpointing(self, enabled: bool = True):
        self.causal_transformer.set_gradient_checkpointing(enabled)

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        self.num_tokens = num_tokens
        self.self_cond = False

    def forward_with_cond_scale(self, *args, cond_scale: float = 1.0, **kwargs):
        logits = self.forward(*args, **kwargs)
        if cond_scale == 1:
            return logits

        null_logits = self.forward(
            *args,
            brain_cond_drop_prob=1.0,
            image_cond_drop_prob=1.0,
            **kwargs,
        )
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed: torch.Tensor,
        diffusion_timesteps: torch.Tensor,
        *,
        self_cond=None,
        brain_embed: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
        brain_cond_drop_prob: float = 0.0,
        text_cond_drop_prob: Optional[float] = None,
        image_cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob

        if image_embed.ndim != 3:
            raise ValueError(f"image_embed must be (B,T,D), got {tuple(image_embed.shape)}")

        batch, num_tokens, dim = image_embed.shape
        device, dtype = image_embed.device, image_embed.dtype

        if dim != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {dim}")
        if num_tokens != self.num_tokens:
            raise ValueError(f"Expected num_tokens={self.num_tokens}, got {num_tokens}")

        if brain_embed is None:
            raise ValueError("brain_embed/text_embed must be supplied")
        if brain_embed.shape != image_embed.shape:
            raise ValueError(
                f"brain_embed/text_embed shape {tuple(brain_embed.shape)} must match image_embed shape {tuple(image_embed.shape)}"
            )

        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device=device)
        brain_keep_mask = rearrange(brain_keep_mask, "b -> b 1 1")

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device=device)
        image_keep_mask = rearrange(image_keep_mask, "b -> b 1 1")

        null_brain_embeds = self.null_brain_embeds.to(device=device, dtype=dtype)
        brain_embed = torch.where(brain_keep_mask, brain_embed, null_brain_embeds[None])

        null_image_embed = self.null_image_embed.to(device=device, dtype=dtype)
        image_embed = torch.where(image_keep_mask, image_embed, null_image_embed[None])

        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        if self.learned_query_mode == "token":
            learned_queries = repeat(
                self.learned_query.to(device=device, dtype=dtype),
                "n d -> b n d",
                b=batch,
            )
        elif self.learned_query_mode == "pos_emb":
            pos_embs = repeat(
                self.learned_query.to(device=device, dtype=dtype),
                "n d -> b n d",
                b=batch,
            )
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device=device, dtype=dtype)
        elif self.learned_query_mode == "all_pos_emb":
            pos_embs = repeat(
                self.learned_query.to(device=device, dtype=dtype),
                "n d -> b n d",
                b=batch,
            )
            learned_queries = torch.empty((batch, 0, dim), device=device, dtype=dtype)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=device, dtype=dtype)

        tokens = torch.cat((brain_embed, time_embed, image_embed, learned_queries), dim=-2)
        if self.learned_query_mode == "all_pos_emb":
            tokens = tokens + pos_embs

        tokens = self.causal_transformer(tokens)
        pred_image_embed = tokens[..., -self.num_tokens :, :]
        return pred_image_embed


class BrainDiffusionPrior(DiffusionPrior):
    def set_gradient_checkpointing(self, enabled: bool = True):
        if hasattr(self.net, "set_gradient_checkpointing"):
            self.net.set_gradient_checkpointing(enabled)
    """
    ZEBRA-style BrainDiffusionPrior subclassing dalle2_pytorch.DiffusionPrior.

    Differences from the stock DiffusionPrior:
    - accepts text_embed / image_embed sequences directly;
    - returns (loss, pred) from forward;
    - supports deterministic generator in DDPM sampling;
    - keeps ZEBRA behavior of not applying image_embed_scale unless explicitly enabled by dalle2-pytorch config.
    """

    def __init__(self, *args, voxel2clip=None, **kwargs):
        # ZEBRA train_zebra.py passes condition_on_text_encodings=False and image_embed_scale=None.
        # Those are handled by the parent DiffusionPrior.
        super().__init__(*args, **kwargs)
        self.voxel2clip = voxel2clip

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        text_cond=None,
        self_cond=None,
        clip_denoised=True,
        cond_scale=1.0,
        generator=None,
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=t,
            text_cond=text_cond,
            self_cond=self_cond,
            clip_denoised=clip_denoised,
            cond_scale=cond_scale,
        )
        if generator is None:
            noise = torch.randn_like(x)
        else:
            noise = torch.randn(x.shape, device=device, dtype=x.dtype, generator=generator)

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps=None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_image_embed = self.p_sample_loop_ddim(*args, timesteps=timesteps, **kwargs)

        # Match ZEBRA: do not undo/apply image_embed_scale here.
        return normalized_image_embed

    @torch.no_grad()
    def p_sample_loop_ddim(
        self,
        shape,
        text_cond,
        *,
        timesteps,
        eta=1.0,
        cond_scale=1.0,
        generator=None,
    ):
        batch = shape[0]
        device = self.device
        alphas = self.noise_scheduler.alphas_cumprod_prev
        total_timesteps = self.noise_scheduler.num_timesteps

        times = torch.linspace(-1.0, total_timesteps, steps=timesteps + 1, device=device)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if generator is None:
            image_embed = torch.randn(shape, device=device)
        else:
            image_embed = torch.randn(shape, device=device, generator=generator)
        x_start = None

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for time, time_next in time_pairs:
            alpha = alphas[time]
            alpha_next = alphas[time_next]
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.net.self_cond else None

            pred = self.net.forward_with_cond_scale(
                image_embed,
                time_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
                **text_cond,
            )

            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(image_embed, t=time_cond, v=pred)
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(image_embed, t=time_cond, noise=pred)

            if not self.predict_x_start:
                x_start.clamp_(-1.0, 1.0)

            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)

            pred_noise = self.noise_scheduler.predict_noise_from_start(image_embed, t=time_cond, x0=x_start)

            if time_next < 0:
                image_embed = x_start
                continue

            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            if time_next > 0:
                if generator is None:
                    noise = torch.randn_like(image_embed)
                else:
                    noise = torch.randn(image_embed.shape, device=device, dtype=image_embed.dtype, generator=generator)
            else:
                noise = 0.0
            image_embed = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise

        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale=1.0, generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            image_embed = torch.randn(shape, device=device)
        else:
            image_embed = torch.randn(shape, device=device, generator=generator)

        x_start = None
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in range(self.noise_scheduler.num_timesteps - 1, -1, -1):
            times = torch.full((batch,), i, device=device, dtype=torch.long)
            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(
                image_embed,
                times,
                text_cond=text_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
                generator=generator,
            )

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(image_embed))
        image_embed_noisy = self.noise_scheduler.q_sample(x_start=image_embed, t=times, noise=noise)

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()

        pred = self.net(
            image_embed_noisy,
            times,
            self_cond=self_cond,
            text_cond_drop_prob=self.text_cond_drop_prob,
            image_cond_drop_prob=self.image_cond_drop_prob,
            **text_cond,
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise

        loss = nn.functional.mse_loss(pred, target)
        return loss, pred

    def forward(
        self,
        *args,
        text_embed: Optional[torch.Tensor] = None,
        image_embed: Optional[torch.Tensor] = None,
        voxel: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        text=None,
        text_encodings=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exists(voxel):
            assert exists(self.voxel2clip), "voxel2clip must be supplied to pass voxel/brain inputs"
            assert not exists(text_embed), "cannot pass both text_embed and voxel"
            clip_voxels = self.voxel2clip(voxel)
            if isinstance(clip_voxels, tuple):
                text_embed = clip_voxels[0]
            else:
                text_embed = clip_voxels

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        if exists(text):
            text_embed, text_encodings = self.clip.embed_text(text)

        assert exists(text_embed), "text_embed or voxel/text must be supplied"
        assert exists(image_embed), "image_embed or image must be supplied"

        if text_embed.ndim != 3:
            raise ValueError(f"text_embed must be (B,T,D), got {tuple(text_embed.shape)}")
        if image_embed.ndim != 3:
            raise ValueError(f"image_embed must be (B,T,D), got {tuple(image_embed.shape)}")
        if text_embed.shape != image_embed.shape:
            raise ValueError(
                f"text_embed shape {tuple(text_embed.shape)} must match image_embed shape {tuple(image_embed.shape)}"
            )

        text_cond = dict(text_embed=text_embed)
        if self.condition_on_text_encodings:
            assert exists(text_encodings), "text encodings must be present if condition_on_text_encodings=True"
            text_cond = {**text_cond, "text_encodings": text_encodings}

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)
        loss, pred = self.p_losses(image_embed, times, text_cond=text_cond, *args, **kwargs)
        return loss, pred

    @torch.no_grad()
    def sample(
        self,
        *,
        text_embed: torch.Tensor,
        cond_scale: float = 1.0,
        timesteps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if text_embed.ndim != 3:
            raise ValueError(f"text_embed must be (B,T,D), got {tuple(text_embed.shape)}")
        return self.p_sample_loop(
            shape=text_embed.shape,
            text_cond={"text_embed": text_embed},
            timesteps=timesteps,
            cond_scale=cond_scale,
            generator=generator,
        )
    

class BlurryReconDecoder(nn.Module):
    """
    ZEBRA-style blurry reconstruction decoder.

    Input:
        clip_vision_embed: (B, N, C), usually prior output with:
            N = 256 tokens
            C = 1664 CLIP ViT-bigG feature dim

    Output:
        b_up:
            predicted image-variation-autoencoder latent, usually compared
            against AutoencoderKL.encode(image).latent_dist.mode() * 0.18215

        b_aux:
            auxiliary 49-token feature map, kept for ZEBRA compatibility.
            It is not necessarily used in the current training loss.
    """

    def __init__(self, vision_dim: int = 1664):
        super().__init__()

        self.maps_projector = nn.Sequential(
            nn.Conv2d(vision_dim, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 128, 1, bias=False),
            nn.GroupNorm(1, 128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 1, bias=True),
        )

        self.bdropout = nn.Dropout(0.3)
        self.bnorm = nn.GroupNorm(1, 64)

        self.bupsampler = Decoder(
            in_channels=64,
            out_channels=4,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[32, 64, 128],
            layers_per_block=1,
        )

        self.b_maps_projector = nn.Sequential(
            nn.Conv2d(64, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 1, bias=True),
        )

    def forward(self, clip_vision_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if clip_vision_embed.ndim != 3:
            raise ValueError(
                f"clip_vision_embed must be (B, N, C), got {tuple(clip_vision_embed.shape)}"
            )

        bsz, num_tokens, channels = clip_vision_embed.shape

        side = int(math.sqrt(num_tokens))
        if side * side != num_tokens:
            raise ValueError(
                f"BlurryReconDecoder expects a square token grid, got N={num_tokens}. "
                f"For ZEBRA/unCLIP ViT-bigG this should usually be 256 = 16x16."
            )

        x = rearrange(
            clip_vision_embed,
            "b (h w) c -> b c h w",
            h=side,
            w=side,
        )

        # Match ZEBRA: token grid -> 7x7 feature map before latent decoder.
        x = F.interpolate(x, size=(7, 7), mode="bilinear", align_corners=False)

        b = self.maps_projector(x)
        b = self.bdropout(b)
        b = self.bnorm(b)

        b_aux = self.b_maps_projector(b).flatten(2).permute(0, 2, 1)
        b_aux = b_aux.view(bsz, 49, 512)

        b_up = self.bupsampler(b)

        return b_up, b_aux


def build_zebra_prior(
    *,
    clip_seq_dim: int = 256,
    clip_emb_dim: int = 1664,
    timesteps: int = 100,
    cond_drop_prob: float = 0.2,
    depth: int = 6,
    use_gradient_checkpointing: bool = False,
) -> BrainDiffusionPrior:
    """Convenience constructor matching train_zebra.py defaults."""
    dim_head = 52
    heads = clip_emb_dim // dim_head
    prior_network = PriorNetwork(
        dim=clip_emb_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=clip_seq_dim,
        learned_query_mode="pos_emb",
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    return BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=clip_emb_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=cond_drop_prob,
        image_embed_scale=None,
    )
