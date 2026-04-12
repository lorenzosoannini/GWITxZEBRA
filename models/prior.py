import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# helpers
# ---------------------------------------------------------
def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else d() if callable(d) else d


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return F.normalize(x, dim=dim, eps=eps)


def prob_mask_like(shape, prob, device):
    if prob <= 0:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if prob >= 1:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    return torch.rand(shape, device=device) < prob


# ---------------------------------------------------------
# sinusoidal timestep embedding
# ---------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,) int or float
        returns: (B, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant", value=0)

        return emb


# ---------------------------------------------------------
# transformer block
# ---------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.norm2(x)
        x = x + self.ff(h)
        return x


# ---------------------------------------------------------
# prior network (ZEBRA-like but simplified)
# ---------------------------------------------------------
class PriorNetwork(nn.Module):
    """
    Predicts denoised image-token embeddings from:
        - noisy image tokens
        - timestep embedding
        - brain/EEG semantic tokens

    Shapes:
        image_embed: (B, T, D)
        brain_embed: (B, T, D)
        output:      (B, T, D)
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        num_timesteps: int = 1000,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        learned_query_mode: str = "pos_emb",
    ):
        super().__init__()

        self.dim = dim
        self.num_tokens = num_tokens
        self.num_timesteps = num_timesteps
        self.learned_query_mode = learned_query_mode
        self.self_cond = False  # kept for interface similarity

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        if learned_query_mode in {"pos_emb", "all_pos_emb", "token"}:
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        else:
            self.learned_query = None

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward_with_cond_scale(
        self,
        image_embed: torch.Tensor,
        diffusion_timesteps: torch.Tensor,
        *,
        brain_embed: Optional[torch.Tensor] = None,
        cond_scale: float = 1.0,
        brain_cond_drop_prob: float = 0.0,
        image_cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        logits = self.forward(
            image_embed=image_embed,
            diffusion_timesteps=diffusion_timesteps,
            brain_embed=brain_embed,
            brain_cond_drop_prob=brain_cond_drop_prob,
            image_cond_drop_prob=image_cond_drop_prob,
        )

        if cond_scale == 1.0:
            return logits

        null_logits = self.forward(
            image_embed=image_embed,
            diffusion_timesteps=diffusion_timesteps,
            brain_embed=brain_embed,
            brain_cond_drop_prob=1.0,
            image_cond_drop_prob=1.0,
        )
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed: torch.Tensor,
        diffusion_timesteps: torch.Tensor,
        *,
        brain_embed: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,   # unused, kept for compatibility
        brain_cond_drop_prob: float = 0.0,
        image_cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        image_embed: (B, T, D) noisy target image tokens
        brain_embed: (B, T, D) conditioning EEG/semantic tokens
        diffusion_timesteps: (B,)
        """
        if image_embed.ndim != 3:
            raise ValueError(f"image_embed must be (B, T, D), got {tuple(image_embed.shape)}")

        b, t, d = image_embed.shape
        if d != self.dim:
            raise ValueError(f"Expected image_embed dim={self.dim}, got {d}")
        if t != self.num_tokens:
            raise ValueError(f"Expected num_tokens={self.num_tokens}, got {t}")

        if brain_embed is None:
            brain_embed = torch.zeros_like(image_embed)

        if brain_embed.ndim != 3:
            raise ValueError(f"brain_embed must be (B, T, D), got {tuple(brain_embed.shape)}")
        if brain_embed.shape != image_embed.shape:
            raise ValueError(
                f"brain_embed shape {tuple(brain_embed.shape)} must match image_embed shape {tuple(image_embed.shape)}"
            )

        device = image_embed.device
        dtype = image_embed.dtype

        # classifier-free style masking
        brain_keep_mask = prob_mask_like((b,), 1.0 - brain_cond_drop_prob, device=device).view(b, 1, 1)
        image_keep_mask = prob_mask_like((b,), 1.0 - image_cond_drop_prob, device=device).view(b, 1, 1)

        null_brain = self.null_brain_embeds.to(device=device, dtype=dtype).unsqueeze(0)
        null_image = self.null_image_embed.to(device=device, dtype=dtype).unsqueeze(0)

        brain_embed = torch.where(brain_keep_mask, brain_embed, null_brain)
        image_embed = torch.where(image_keep_mask, image_embed, null_image)

        # time token
        time_embed = self.time_mlp(diffusion_timesteps.float()).to(dtype=dtype).unsqueeze(1)  # (B,1,D)

        # optional learned positional/query injection
        if self.learned_query_mode == "pos_emb" and self.learned_query is not None:
            pos = self.learned_query.to(device=device, dtype=dtype).unsqueeze(0)  # (1,T,D)
            image_embed = image_embed + pos
        elif self.learned_query_mode == "all_pos_emb" and self.learned_query is not None:
            pos = self.learned_query.to(device=device, dtype=dtype).unsqueeze(0)
            brain_embed = brain_embed + pos
            image_embed = image_embed + pos

        # concatenate [brain tokens | time token | image tokens]
        x = torch.cat([brain_embed, time_embed, image_embed], dim=1)  # (B, 2T+1, D)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # predict only image-token part
        pred_image_embed = x[:, -self.num_tokens :, :]
        pred_image_embed = self.out_proj(pred_image_embed)
        return pred_image_embed


# ---------------------------------------------------------
# simple diffusion schedule
# ---------------------------------------------------------
class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()

        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]],
            dim=0,
        )

        self.timesteps = timesteps

        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev, persistent=False)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=False,
        )

    def sample_random_times(self, batch_size: int, device) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=device).long()

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1).to(dtype=x_start.dtype)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1).to(dtype=x_start.dtype)

        return sqrt_alpha * x_start + sqrt_one_minus * noise


# ---------------------------------------------------------
# diffusion prior
# ---------------------------------------------------------
class BrainDiffusionPrior(nn.Module):
    """
    Simplified ZEBRA-like diffusion prior.

    Training:
        condition with semantic EEG tokens (brain_embed / F_s)
        target is CLIP image token sequence

    Loss:
        default predict_x_start=True => MSE(pred_x0, target_tokens)
    """

    def __init__(
        self,
        net: PriorNetwork,
        image_embed_dim: int,
        timesteps: int = 100,
        cond_drop_prob: float = 0.2,
        predict_x_start: bool = True,
        training_clamp_l2norm: bool = False,
        sampling_clamp_l2norm: bool = False,
    ):
        super().__init__()

        self.net = net
        self.image_embed_dim = image_embed_dim
        self.cond_drop_prob = cond_drop_prob
        self.predict_x_start = predict_x_start
        self.training_clamp_l2norm = training_clamp_l2norm
        self.sampling_clamp_l2norm = sampling_clamp_l2norm

        self.noise_scheduler = DiffusionSchedule(timesteps=timesteps)

    @property
    def device(self):
        return next(self.parameters()).device

    def l2norm_clamp_embed(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, dim=-1)

    def p_losses(
        self,
        image_embed: torch.Tensor,
        times: torch.Tensor,
        *,
        text_cond: dict,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(
            x_start=image_embed,
            t=times,
            noise=noise,
        )

        pred = self.net(
            image_embed=image_embed_noisy,
            diffusion_timesteps=times,
            brain_embed=text_cond["text_embed"],
            brain_cond_drop_prob=self.cond_drop_prob,
            image_cond_drop_prob=0.0,
        )

        if self.predict_x_start:
            target = image_embed
            if self.training_clamp_l2norm:
                pred = self.l2norm_clamp_embed(pred)
        else:
            target = noise

        loss = F.mse_loss(pred, target)
        return loss, pred

    def forward(
        self,
        *,
        text_embed: torch.Tensor,
        image_embed: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        text_embed:  (B, T, D) semantic conditioning tokens, e.g. F_s
        image_embed: (B, T, D) CLIP target image tokens
        """
        if text_embed.ndim != 3:
            raise ValueError(f"text_embed must be (B,T,D), got {tuple(text_embed.shape)}")
        if image_embed.ndim != 3:
            raise ValueError(f"image_embed must be (B,T,D), got {tuple(image_embed.shape)}")
        if text_embed.shape != image_embed.shape:
            raise ValueError(
                f"text_embed shape {tuple(text_embed.shape)} must match image_embed shape {tuple(image_embed.shape)}"
            )

        batch = image_embed.shape[0]
        device = image_embed.device
        times = self.noise_scheduler.sample_random_times(batch, device=device)

        text_cond = {"text_embed": text_embed}
        loss, pred = self.p_losses(
            image_embed=image_embed,
            times=times,
            text_cond=text_cond,
            noise=noise,
        )
        return loss, pred

    @torch.no_grad()
    def sample(
        self,
        *,
        text_embed: torch.Tensor,
        cond_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Simple ancestral sampling for debugging/inference.

        text_embed: (B, T, D)
        returns:    (B, T, D)
        """
        if text_embed.ndim != 3:
            raise ValueError(f"text_embed must be (B,T,D), got {tuple(text_embed.shape)}")

        b, t, d = text_embed.shape
        device = text_embed.device

        if generator is None:
            x = torch.randn((b, t, d), device=device, dtype=text_embed.dtype)
        else:
            x = torch.randn((b, t, d), device=device, dtype=text_embed.dtype, generator=generator)

        for i in reversed(range(self.noise_scheduler.timesteps)):
            times = torch.full((b,), i, device=device, dtype=torch.long)

            pred_x0 = self.net.forward_with_cond_scale(
                image_embed=x,
                diffusion_timesteps=times,
                brain_embed=text_embed,
                cond_scale=cond_scale,
                brain_cond_drop_prob=0.0,
                image_cond_drop_prob=0.0,
            )

            if self.sampling_clamp_l2norm:
                pred_x0 = self.l2norm_clamp_embed(pred_x0)

            if i == 0:
                x = pred_x0
                break

            alpha_t = self.noise_scheduler.alphas[i].to(device=x.device, dtype=x.dtype)
            alpha_bar_t = self.noise_scheduler.alphas_cumprod[i].to(device=x.device, dtype=x.dtype)
            beta_t = self.noise_scheduler.betas[i].to(device=x.device, dtype=x.dtype)

            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)

            # derive epsilon from x_t and x0
            eps = (x - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)

            mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t + 1e-8)) * eps)
            var = beta_t
            x = mean + torch.sqrt(var) * noise

        return x