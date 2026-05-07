import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum
from torch.utils.checkpoint import checkpoint


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
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# ---------------------------------------------------------
# timestep embedding
# ---------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant", value=0)

        return emb


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        expansion_factor: float = 2.0,
        depth: int = 2,
        norm: bool = False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = lambda: nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [
            nn.Sequential(
                nn.Linear(dim_in, hidden_dim),
                nn.SiLU(),
                norm_fn(),
            )
        ]

        for _ in range(depth - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    norm_fn(),
                )
            )

        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


# ---------------------------------------------------------
# custom layernorm + transformer pieces from dalle2 / ZEBRA style
# ---------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, fp16_eps=1e-3, stable=False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach().clamp(min=1e-8)

        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class RelPosBias(nn.Module):
    def __init__(
        self,
        heads: int = 8,
        num_buckets: int = 32,
        max_distance: int = 128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact + 1e-8)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()

        val_if_large = torch.min(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1),
        )
        return torch.where(is_small, n, val_if_large)

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def FeedForward(
    dim,
    mult=4,
    dropout=0.0,
    post_activation_norm=False,
):
    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        dropout=0.0,
        causal=False,
        cosine_sim=True,
        cosine_sim_scale=16,
    ):
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor, mask=None, attn_bias=None) -> torch.Tensor:
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        q = q * self.scale

        nk, nv = map(
            lambda t: repeat(t, "d -> b 1 d", b=b),
            self.null_kv.unbind(dim=-2),
        )
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        if exists(attn_bias):
            sim = sim + attn_bias

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32).type(sim.dtype)
        attn = self.dropout(attn)

        out = einsum("b h i j, b j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FlaggedCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_in=False,
        norm_out=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        final_proj=True,
        normformer=False,
        causal=True,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()
        self.rel_pos_bias = RelPosBias(heads=heads)

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
        self.gradient_checkpointing = bool(enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, device = x.shape[1], x.device

        x = self.init_norm(x)
        attn_bias = self.rel_pos_bias(n, n + 1, device=device)

        for attn, ff in self.layers:
            if self.gradient_checkpointing and self.training:
                def attn_forward(x, attn=attn, attn_bias=attn_bias):
                    return attn(x, attn_bias=attn_bias)

                def ff_forward(x, ff=ff):
                    return ff(x)

                x = checkpoint(attn_forward, x, use_reentrant=False) + x
                x = checkpoint(ff_forward, x, use_reentrant=False) + x
            else:
                x = attn(x, attn_bias=attn_bias) + x
                x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)


# ---------------------------------------------------------
# noise scheduler: faithful to dalle2 style
# ---------------------------------------------------------
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class NoiseScheduler(nn.Module):
    def __init__(
        self,
        *,
        beta_schedule="cosine",
        timesteps=100,
        loss_type="l2",
        p2_loss_weight_gamma=0.0,
        p2_loss_weight_k=1,
    ):
        super().__init__()

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise NotImplementedError(f"Unsupported beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        timesteps_, = betas.shape
        self.num_timesteps = int(timesteps_)

        if loss_type == "l2":
            loss_fn = F.mse_loss
        elif loss_type == "l1":
            loss_fn = F.l1_loss
        elif loss_type == "huber":
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError(f"Unsupported loss_type: {loss_type}")

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent=False)

        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer("posterior_variance", posterior_variance)
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.0
        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma,
        )

    def sample_random_times(self, batch_size: int, device=None) -> torch.Tensor:
        device = default(device, self.betas.device)
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_sample_from_to(self, x_from, from_t, to_t, noise=None):
        shape = x_from.shape
        noise = default(noise, lambda: torch.randn_like(x_from))

        alpha = extract(self.sqrt_alphas_cumprod, from_t, shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, from_t, shape)
        alpha_next = extract(self.sqrt_alphas_cumprod, to_t, shape)
        sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, to_t, shape)

        return x_from * (alpha_next / alpha) + noise * (sigma_next * alpha - sigma * alpha_next) / alpha

    def calculate_v(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0)
            / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )


# ---------------------------------------------------------
# prior network: faithful to ZEBRA token prior
# ---------------------------------------------------------
class PriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps=None,
        num_time_embeds=1,
        num_tokens=257,
        causal=False,
        learned_query_mode="none",
        depth=6,
        heads=32,
        dim_head=52,
        ff_mult=4,
        norm_in=False,
        norm_out=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        final_proj=True,
        normformer=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds)
            if exists(num_timesteps)
            else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)),
            rearrange_module("b (n d) -> b n d", n=num_time_embeds),
        )

        if self.learned_query_mode == "token":
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        elif self.learned_query_mode == "pos_emb":
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        elif self.learned_query_mode == "all_pos_emb":
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens * 2 + 1, dim) * scale)
        else:
            self.learned_query = None

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
            causal=causal,
        )

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        self.num_tokens = num_tokens
        self.self_cond = False

    def set_gradient_checkpointing(self, enabled: bool = True):
        self.causal_transformer.set_gradient_checkpointing(enabled)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=1.0,
        **kwargs,
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, brain_cond_drop_prob=1.0, image_cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        self_cond=None,
        brain_embed=None,
        text_embed=None,
        brain_cond_drop_prob=0.0,
        text_cond_drop_prob=None,
        image_cond_drop_prob=0.0,
    ):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob

        if image_embed.ndim != 3:
            raise ValueError(f"image_embed must be (B, T, D), got {tuple(image_embed.shape)}")

        batch, num_tokens, dim = image_embed.shape
        device, dtype = image_embed.device, image_embed.dtype

        if dim != self.dim:
            raise ValueError(f"Expected embed dim {self.dim}, got {dim}")
        if num_tokens != self.num_tokens:
            raise ValueError(f"Expected num_tokens {self.num_tokens}, got {num_tokens}")

        if brain_embed is None:
            brain_embed = torch.zeros_like(image_embed)

        if brain_embed.ndim != 3:
            raise ValueError(f"brain_embed must be (B, T, D), got {tuple(brain_embed.shape)}")
        if brain_embed.shape != image_embed.shape:
            raise ValueError(
                f"brain_embed shape {tuple(brain_embed.shape)} must match image_embed shape {tuple(image_embed.shape)}"
            )

        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device=device)
        brain_keep_mask = rearrange(brain_keep_mask, "b -> b 1 1")

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device=device)
        image_keep_mask = rearrange(image_keep_mask, "b -> b 1 1")

        null_brain_embeds = self.null_brain_embeds.to(device=device, dtype=dtype)
        brain_embed = torch.where(
            brain_keep_mask,
            brain_embed,
            null_brain_embeds[None],
        )

        null_image_embed = self.null_image_embed.to(device=device, dtype=dtype)
        image_embed = torch.where(
            image_keep_mask,
            image_embed,
            null_image_embed[None],
        )

        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)

        time_embed = self.to_time_embeds(diffusion_timesteps)

        if self.learned_query_mode == "token":
            learned_queries = repeat(self.learned_query.to(device=device, dtype=dtype), "n d -> b n d", b=batch)
        elif self.learned_query_mode == "pos_emb":
            pos_embs = repeat(self.learned_query.to(device=device, dtype=dtype), "n d -> b n d", b=batch)
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device=device, dtype=dtype)
        elif self.learned_query_mode == "all_pos_emb":
            pos_embs = repeat(self.learned_query.to(device=device, dtype=dtype), "n d -> b n d", b=batch)
            learned_queries = torch.empty((batch, 0, dim), device=device, dtype=dtype)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=device, dtype=dtype)

        tokens = torch.cat(
            (
                brain_embed,
                time_embed,
                image_embed,
                learned_queries,
            ),
            dim=-2,
        )

        if self.learned_query_mode == "all_pos_emb":
            tokens = tokens + pos_embs

        tokens = self.causal_transformer(tokens)
        pred_image_embed = tokens[..., -self.num_tokens:, :]
        return pred_image_embed


class rearrange_module(nn.Module):
    def __init__(self, pattern, **kwargs):
        super().__init__()
        self.pattern = pattern
        self.kwargs = kwargs

    def forward(self, x):
        return rearrange(x, self.pattern, **self.kwargs)


# ---------------------------------------------------------
# diffusion prior: faithful to ZEBRA logic, sequence version
# ---------------------------------------------------------
class BrainDiffusionPrior(nn.Module):
    def __init__(
        self,
        net,
        *,
        image_embed_dim,
        timesteps=100,
        sample_timesteps=None,
        cond_drop_prob=0.0,
        text_cond_drop_prob=None,
        image_cond_drop_prob=None,
        loss_type="l2",
        predict_x_start=True,
        predict_v=False,
        beta_schedule="cosine",
        sampling_clamp_l2norm=False,
        sampling_final_clamp_l2norm=False,
        training_clamp_l2norm=False,
        init_image_embed_l2norm=False,
        image_embed_scale=None,
    ):
        super().__init__()

        self.sample_timesteps = sample_timesteps

        self.noise_scheduler = NoiseScheduler(
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            loss_type=loss_type,
        )

        self.net = net
        self.image_embed_dim = image_embed_dim

        assert net.dim == self.image_embed_dim, (
            f"PriorNetwork dim={net.dim} but image_embed_dim={self.image_embed_dim}"
        )

        self.text_cond_drop_prob = default(text_cond_drop_prob, cond_drop_prob)
        self.image_cond_drop_prob = default(image_cond_drop_prob, cond_drop_prob)

        self.can_classifier_guidance = self.text_cond_drop_prob > 0.0 or self.image_cond_drop_prob > 0.0

        self.predict_x_start = predict_x_start
        self.predict_v = predict_v

        self.image_embed_scale = default(image_embed_scale, self.image_embed_dim ** 0.5)

        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.sampling_final_clamp_l2norm = sampling_final_clamp_l2norm
        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm

        self.register_buffer("_dummy", torch.tensor([True]), persistent=False)

    @property
    def device(self):
        return self._dummy.device

    def l2norm_clamp_embed(self, image_embed):
        return l2norm(image_embed) * self.image_embed_scale

    def p_mean_variance(
        self,
        x,
        t,
        text_cond,
        self_cond=None,
        clip_denoised=False,
        cond_scale=1.0,
    ):
        if cond_scale != 1.0 and not self.can_classifier_guidance:
            raise ValueError(
                "cond_scale != 1 requires conditional dropout during training."
            )

        pred = self.net.forward_with_cond_scale(
            x,
            t,
            cond_scale=cond_scale,
            self_cond=self_cond,
            **text_cond,
        )

        if self.predict_v:
            x_start = self.noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        elif self.predict_x_start:
            x_start = pred
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        if clip_denoised and not self.predict_x_start:
            x_start.clamp_(-1.0, 1.0)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = self.l2norm_clamp_embed(x_start)

        model_mean, posterior_variance, posterior_log_variance = self.noise_scheduler.q_posterior(
            x_start=x_start,
            x_t=x,
            t=t,
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        *,
        text_cond,
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
    def p_sample_loop_ddpm(
        self,
        shape,
        text_cond,
        *,
        cond_scale=1.0,
        generator=None,
    ):
        batch, device = shape[0], self.device

        if generator is None:
            image_embed = torch.randn(shape, device=device, dtype=text_cond["text_embed"].dtype)
        else:
            image_embed = torch.randn(shape, device=device, dtype=text_cond["text_embed"].dtype, generator=generator)

        x_start = None

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in reversed(range(0, self.noise_scheduler.num_timesteps)):
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

    @torch.no_grad()
    def p_sample_loop_ddim(
        self,
        shape,
        text_cond,
        *,
        timesteps,
        eta=1.0,
        cond_scale=1.0,
    ):
        batch = shape[0]
        device = self.device
        alphas = self.noise_scheduler.alphas_cumprod_prev
        total_timesteps = self.noise_scheduler.num_timesteps
        dtype = text_cond["text_embed"].dtype

        times = torch.linspace(-1.0, total_timesteps, steps=timesteps + 1, device=device)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        image_embed = torch.randn(shape, device=device, dtype=dtype)
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

            pred_noise = self.noise_scheduler.predict_noise_from_start(
                image_embed,
                t=time_cond,
                x0=x_start,
            )

            if time_next < 0:
                image_embed = x_start
                continue

            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(image_embed) if time_next > 0 else 0.0

            image_embed = (
                x_start * alpha_next.sqrt()
                + c1 * noise
                + c2 * pred_noise
            )

        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        text_cond,
        *,
        timesteps=None,
        cond_scale=1.0,
        generator=None,
    ):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            normalized_image_embed = self.p_sample_loop_ddpm(
                shape,
                text_cond=text_cond,
                cond_scale=cond_scale,
                generator=generator,
            )
        else:
            normalized_image_embed = self.p_sample_loop_ddim(
                shape,
                text_cond=text_cond,
                timesteps=timesteps,
                cond_scale=cond_scale,
            )

        image_embed = normalized_image_embed / self.image_embed_scale
        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_scaled = image_embed * self.image_embed_scale
        image_embed_noisy = self.noise_scheduler.q_sample(
            x_start=image_embed_scaled,
            t=times,
            noise=noise,
        )

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(
                    image_embed_noisy,
                    times,
                    **text_cond,
                ).detach()

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
            target = self.noise_scheduler.calculate_v(image_embed_scaled, times, noise)
        elif self.predict_x_start:
            target = image_embed_scaled
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss, pred

    def forward(
        self,
        *,
        text_embed: torch.Tensor,
        image_embed: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if text_embed.ndim != 3:
            raise ValueError(f"text_embed must be (B,T,D), got {tuple(text_embed.shape)}")
        if image_embed.ndim != 3:
            raise ValueError(f"image_embed must be (B,T,D), got {tuple(image_embed.shape)}")
        if text_embed.shape != image_embed.shape:
            raise ValueError(
                f"text_embed shape {tuple(text_embed.shape)} must match image_embed shape {tuple(image_embed.shape)}"
            )

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch, device=device)

        text_cond = dict(text_embed=text_embed)
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
        timesteps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if text_embed.ndim != 3:
            raise ValueError(f"text_embed must be (B,T,D), got {tuple(text_embed.shape)}")

        timesteps = default(timesteps, self.sample_timesteps)
        return self.p_sample_loop(
            shape=text_embed.shape,
            text_cond={"text_embed": text_embed},
            timesteps=timesteps,
            cond_scale=cond_scale,
            generator=generator,
        )