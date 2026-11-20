"""Channel-conditioned physiological signal encoder."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up = nn.Linear(256, 2048)
        self.down = nn.Linear(1024, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


def rotary_coefficients(positions, head_dim, dtype):
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32)
        / head_dim
    )
    angle = positions.float()[..., None] * frequency
    return angle.cos()[:, None].to(dtype), angle.sin()[:, None].to(dtype)


def apply_rotary(x, rotary):
    cosine, sine = rotary
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), -1
    ).flatten(-2)


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 8
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.qkv = nn.Linear(256, 768)
        self.out = nn.Linear(256, 256)
        self.mlp = SwiGLU()
        self.attention_gate = nn.Linear(256, 256)

    def forward(self, h, present, modulation, rotary):
        gamma_attn, beta_attn, gamma_ffn, beta_ffn = modulation.chunk(4, -1)
        batch, length, dim = h.shape
        attention_input = self.norm1(h) * (1 + gamma_attn) + beta_attn
        q, k, v = (
            self.qkv(attention_input)
            .reshape(batch, length, 3, self.heads, dim // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        attended = F.scaled_dot_product_attention(
            apply_rotary(q, rotary),
            apply_rotary(k, rotary),
            v,
            attn_mask=present[:, None, None, :],
            dropout_p=0.0,
        )
        gate = (
            self.attention_gate(attention_input)
            .sigmoid()
            .reshape(batch, length, self.heads, dim // self.heads)
            .transpose(1, 2)
        )
        attended = attended * gate
        h = h + self.out(attended.transpose(1, 2).reshape(batch, length, dim))
        return h + self.mlp(self.norm2(h) * (1 + gamma_ffn) + beta_ffn)


class LingSantePulseEncoder(nn.Module):
    """Encode normalized signal windows using fixed channel conditions.

    Inputs: normalized 128-Hz ``x``, observed and present masks [N, T],
    and channel vectors [N, 384]. T must be divisible by 32. A row is one
    physical channel in one window. Output: all final-LayerNorm tokens,
    ordered CLS, four registers, patches. No pooling or L2 normalization.
    """

    sample_rate = 128
    patch_size = 32
    dim = 256
    num_prefix_tokens = 5

    def __init__(self):
        super().__init__()
        self.patch = nn.Linear(32, 256)
        self.validity = nn.Linear(64, 256, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, 256))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 256))
        self.register_tokens = nn.Parameter(torch.zeros(1, 4, 256))
        self.blocks = nn.ModuleList([EncoderBlock() for _ in range(12)])
        self.text_adapter = nn.Sequential(
            nn.Linear(384, 256, bias=False), nn.LayerNorm(256)
        )
        self.film_heads = nn.ModuleList([nn.Linear(256, 1024) for _ in self.blocks])
        self.norm = nn.LayerNorm(256)

    def forward(self, x, observed, sample_present, channel_vectors):
        if x.ndim != 2 or x.shape[-1] % 32 or x.shape[-1] == 0:
            raise ValueError(
                "x must have shape [N, T], with positive T divisible by 32"
            )
        if observed.shape != x.shape or sample_present.shape != x.shape:
            raise ValueError("Masks must have the same shape as x")
        if channel_vectors.shape != (len(x), 384):
            raise ValueError("channel_vectors must have shape [N, 384]")
        observed = observed.bool() & sample_present.bool()
        if not bool(observed.any(-1).all()):
            raise ValueError("Each channel window must contain observed samples")
        x = torch.where(observed, x, 0.0)
        if not bool(torch.isfinite(x).all() & torch.isfinite(channel_vectors).all()):
            raise ValueError("Observed samples and channel vectors must be finite")
        batch, samples = x.shape
        patches = x.reshape(batch, -1, 32)
        om = observed.reshape_as(patches)
        pm = sample_present.bool().reshape_as(patches)
        h = self.patch(patches) + self.validity(torch.cat((om, pm), -1).to(x.dtype))
        h = torch.where(om.any(-1, keepdim=True), h, self.mask_token)
        h = torch.cat(
            (
                self.cls.expand(batch, -1, -1),
                self.register_tokens.expand(batch, -1, -1),
                h,
            ),
            1,
        )
        positions = torch.cat(
            (
                torch.arange(-5, 0, device=x.device, dtype=torch.float32),
                torch.arange(samples // 32, device=x.device).float() * 2,
            )
        )
        positions = positions.expand(batch, -1)
        present = torch.cat(
            (torch.ones(batch, 5, device=x.device, dtype=torch.bool), pm.any(-1)), 1
        )
        condition = self.text_adapter(channel_vectors[:, None])
        # Prefix and patch tokens use the same physical channel condition.
        bank = torch.cat((condition, condition), 1)
        ids = torch.cat(
            (
                torch.zeros(batch, 5, device=x.device, dtype=torch.long),
                torch.ones(batch, samples // 32, device=x.device, dtype=torch.long),
            ),
            1,
        )
        rotary = rotary_coefficients(positions, 32, h.dtype)
        for block, head in zip(self.blocks, self.film_heads):
            modulation = head(bank).gather(1, ids[..., None].expand(-1, -1, 1024))
            h = block(h, present, modulation, rotary)
        return self.norm(h)
