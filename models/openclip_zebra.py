from typing import Optional

import kornia
import open_clip
import torch
import torch.nn as nn
from einops import rearrange, repeat


def expand_dims_like(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Expands x with trailing singleton dims until it matches y.ndim.
    """
    while x.ndim < y.ndim:
        x = x.unsqueeze(-1)
    return x


def autocast(fn):
    """
    Lightweight replacement for ZEBRA's internal @autocast decorator.
    Uses CUDA autocast only when running on CUDA.
    """
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    @property
    def is_trainable(self):
        return self._is_trainable

    @property
    def ucg_rate(self):
        return self._ucg_rate

    @property
    def input_key(self):
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value):
        self._input_key = value


class FrozenOpenCLIPImageEmbedder(AbstractEmbModel):
    """
    Standalone version of ZEBRA's FrozenOpenCLIPImageEmbedder.

    Key behavior preserved:
    - OpenCLIP vision backbone loading
    - CLIP preprocessing exactly in tensor space
    - optional token output
    - optional token-only output
    - optional token L2 normalization used by ZEBRA
    """

    def __init__(
        self,
        arch: str = "ViT-bigG-14",
        version: str = "laion2b_s39b_b160k",
        device: str = "cuda",
        init_device: str = "cpu",
        max_length: int = 77,
        freeze: bool = True,
        antialias: bool = True,
        ucg_rate: float = 0.0,
        unsqueeze_dim: bool = False,
        repeat_to_max_len: bool = False,
        num_image_crops: int = 0,
        output_tokens: bool = False,
        l2_norm_tokens: bool = False,
        only_tokens: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=torch.device(init_device),
            pretrained=version,
            cache_dir=cache_dir,
        )

        # Match ZEBRA: keep only vision side
        if hasattr(model, "transformer"):
            del model.transformer

        self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)

        self.device_name = device
        self.max_length = max_length
        self.antialias = antialias
        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None

        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens

        if only_tokens:
            assert output_tokens, "only_tokens=True requires output_tokens=True"
        self.only_tokens = only_tokens

        self.l2_norm_tokens = l2_norm_tokens
        if l2_norm_tokens:
            assert output_tokens, "l2_norm_tokens=True requires output_tokens=True"

        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]),
            persistent=False,
        )

        if freeze:
            self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects x in [-1, 1], as in ZEBRA.
        """
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=self.antialias,
        )
        x = (x + 1.0) / 2.0
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def encode_with_vision_transformer(self, img: torch.Tensor):
        if img.dim() == 5:
            assert self.max_crops == img.shape[1]
            img = rearrange(img, "b n c h w -> (b n) c h w")

        img = self.preprocess(img)

        if not self.output_tokens:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        else:
            assert self.model.visual.output_tokens
            x, tokens = self.model.visual(img)

            if self.l2_norm_tokens:
                token_shape = tokens.shape
                tokens = tokens.flatten(1)
                tokens = torch.nn.functional.normalize(tokens, dim=-1)
                tokens = (tokens - 0.0002) / 0.0015
                tokens = tokens.view(token_shape)
                tokens = (tokens * 1.0957) + 0.1598

        if self.max_crops > 0:
            x = rearrange(x, "(b n) d -> b n d", n=self.max_crops)

            x = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate)
                    * torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
                )
                * x
            )

            if tokens is not None:
                tokens = rearrange(tokens, "(b n) t d -> b t (n d)", n=self.max_crops)

        if self.output_tokens:
            return x, tokens

        return x

    @autocast
    def forward(self, image: torch.Tensor, no_dropout: bool = False):
        z = self.encode_with_vision_transformer(image)
        tokens = None

        if self.output_tokens:
            z, tokens = z[0], z[1]

        z = z.to(image.dtype)

        if self.ucg_rate > 0.0 and (not no_dropout) and not (self.max_crops > 0):
            z = (
                torch.bernoulli(
                    (1.0 - self.ucg_rate) * torch.ones(z.shape[0], device=z.device)
                )[:, None]
                * z
            )

            if tokens is not None:
                tokens = (
                    expand_dims_like(
                        torch.bernoulli(
                            (1.0 - self.ucg_rate)
                            * torch.ones(tokens.shape[0], device=tokens.device)
                        ),
                        tokens,
                    )
                    * tokens
                )

        if self.unsqueeze_dim:
            z = z[:, None, :]

        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len

            if self.only_tokens:
                return tokens

            return tokens, z

        if self.repeat_to_max_len:
            if z.dim() == 2:
                z_ = z[:, None, :]
            else:
                z_ = z
            return repeat(z_, "b 1 d -> b n d", n=self.max_length), z

        if self.pad_to_max_len:
            assert z.dim() == 3
            z_pad = torch.cat(
                (
                    z,
                    torch.zeros(
                        z.shape[0],
                        self.max_length - z.shape[1],
                        z.shape[2],
                        device=z.device,
                        dtype=z.dtype,
                    ),
                ),
                dim=1,
            )
            return z_pad, z_pad[:, 0, ...]

        return z

    def encode(self, image: torch.Tensor):
        return self(image)
    
class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Standalone version of ZEBRA's FrozenOpenCLIPEmbedder2.
    Used for OpenCLIP text embeddings, with optional pooled output.
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch: str = "ViT-bigG-14",
        version: str = "laion2b_s39b_b160k",
        device: str = "cuda",
        max_length: int = 77,
        freeze: bool = True,
        layer: str = "last",
        always_return_pooled: bool = False,
        legacy: bool = True,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        assert layer in self.LAYERS

        load_device = torch.device(device if torch.cuda.is_available() else "cpu")

        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            device=load_device,
            pretrained=version,
            cache_dir=cache_dir,
        )

        if hasattr(model, "visual"):
            del model.visual

        self.model = model
        self.device = load_device
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        self.layer = layer
        self.legacy = legacy

        if freeze:
            self.freeze()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                outputs["penultimate"] = x  # already [B, T, D]
            x = r(x, attn_mask=attn_mask)
        outputs["last"] = x  # already [B, T, D]
        return outputs


    def pool(self, x: torch.Tensor, text: torch.Tensor):
        x = x[torch.arange(x.shape[0], device=x.device), text.argmax(dim=-1)]
        x = x @ self.model.text_projection
        return x


    def encode_with_transformer(self, text: torch.Tensor):
        x = self.model.token_embedding(text)          # [B, T, D]
        x = x + self.model.positional_embedding       # [B, T, D]

        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)

        if self.legacy:
            out = x[self.layer]
            out = self.model.ln_final(out)
            return out
        else:
            out_last = x["last"]
            out_last = self.model.ln_final(out_last)
            pooled = self.pool(out_last, text)
            x["last"] = out_last
            x["pooled"] = pooled
            return x

    @autocast
    def forward(self, text):
        tokens = open_clip.tokenize(text).to(self.device)
        z = self.encode_with_transformer(tokens)

        if not self.return_pooled and self.legacy:
            return z

        if self.return_pooled:
            assert not self.legacy
            return z[self.layer], z["pooled"]

        return z[self.layer]

    def encode(self, text):
        return self(text)