"""Audio-JEPA: Joint-Embedding Predictive Architecture for Source Separation & Prior Extraction.

Implements LeCun's Joint-Embedding Predictive Architecture (I-JEPA) adapted for audio waveforms:
  1. AudioPatchEmbed: 1D convolutional front-end converting raw audio into patch tokens.
  2. JEPAEncoderBlock: Multi-head self-attention encoder blocks with residual connections.
  3. AudioJEPA: Context encoder and predictor trained to predict target encoder representations of masked audio patches.
  4. JEPAGuidedSeparationHead: FiLM-conditioned waveform separation decoder utilizing invariant JEPA priors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# 1. 1D Waveform Patch Embedding
# ---------------------------------------------------------------------------


class AudioPatchEmbed(nn.Module):
    """Converts 1D raw waveform [B, 1, T] into patch tokens [B, N, D]."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 128,
        patch_size: int = 256,
        stride: int = 128,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Conv1d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        h = self.proj(x)  # [B, embed_dim, N]
        h = h.transpose(1, 2)  # [B, N, embed_dim]
        return self.norm(h)


# ---------------------------------------------------------------------------
# 2. Transformer / Residual Encoder Block
# ---------------------------------------------------------------------------


class JEPAEncoderBlock(nn.Module):
    """Self-attention block for JEPA representations."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 3. Audio-JEPA Architecture
# ---------------------------------------------------------------------------


class AudioJEPA(nn.Module):
    """Joint-Embedding Predictive Architecture for Audio Signals."""

    def __init__(
        self,
        embed_dim: int = 128,
        patch_size: int = 256,
        stride: int = 128,
        depth: int = 4,
        pred_depth: int = 2,
        mask_ratio: float = 0.5,
        ema_decay: float = 0.996,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay

        self.patch_embed = AudioPatchEmbed(embed_dim=embed_dim, patch_size=patch_size, stride=stride)

        # Context Encoder (f_theta)
        self.context_blocks = nn.ModuleList([JEPAEncoderBlock(embed_dim=embed_dim) for _ in range(depth)])
        self.context_norm = nn.LayerNorm(embed_dim)

        # Target Encoder (g_xi) - initialized as exact copy of Context Encoder
        self.target_blocks = nn.ModuleList([JEPAEncoderBlock(embed_dim=embed_dim) for _ in range(depth)])
        self.target_norm = nn.LayerNorm(embed_dim)
        self._init_target_encoder()

        # Predictor Network (p_psi)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pred_blocks = nn.ModuleList([JEPAEncoderBlock(embed_dim=embed_dim) for _ in range(pred_depth)])
        self.pred_proj = nn.Linear(embed_dim, embed_dim)

    def _init_target_encoder(self) -> None:
        """Initializes target encoder with identical weights to context encoder and freezes grads."""
        for t_p, c_p in zip(self.target_blocks.parameters(), self.context_blocks.parameters()):
            t_p.data.copy_(c_p.data)
            t_p.requires_grad = False
        for t_p, c_p in zip(self.target_norm.parameters(), self.context_norm.parameters()):
            t_p.data.copy_(c_p.data)
            t_p.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        """Updates target encoder weights via Exponential Moving Average (EMA)."""
        m = momentum if momentum is not None else self.ema_decay
        for t_p, c_p in zip(self.target_blocks.parameters(), self.context_blocks.parameters()):
            t_p.data.mul_(m).add_(c_p.data, alpha=1.0 - m)
        for t_p, c_p in zip(self.target_norm.parameters(), self.context_norm.parameters()):
            t_p.data.mul_(m).add_(c_p.data, alpha=1.0 - m)

    def extract_prior(self, x: Tensor) -> Tensor:
        """Extracts global invariant acoustic prior vector z_JEPA in R^{B, embed_dim}."""
        with torch.no_grad():
            patches = self.patch_embed(x)
            h = patches
            for block in self.target_blocks:
                h = block(h)
            h = self.target_norm(h)
            z_jepa = 0.5 * (h.mean(dim=1) + h.amax(dim=1))
            return z_jepa

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Computes masked Audio-JEPA self-supervised predictive loss."""
        patches = self.patch_embed(x)  # [B, N, D]
        B, N, D = patches.shape

        num_masked = int(N * self.mask_ratio)
        perm = torch.randperm(N, device=x.device)
        context_idx = perm[num_masked:]
        target_idx = perm[:num_masked]

        context_tokens = patches[:, context_idx, :]

        h_context = context_tokens
        for block in self.context_blocks:
            h_context = block(h_context)
        h_context = self.context_norm(h_context)

        with torch.no_grad():
            h_target_full = patches
            for block in self.target_blocks:
                h_target_full = block(h_target_full)
            h_target_full = self.target_norm(h_target_full)
            target_reps = h_target_full[:, target_idx, :]

        pred_input = self.mask_token.repeat(B, num_masked, 1) + h_context.mean(dim=1, keepdim=True)
        h_pred = pred_input
        for block in self.pred_blocks:
            h_pred = block(h_pred)
        pred_reps = self.pred_proj(h_pred)

        loss = F.l1_loss(pred_reps, target_reps)
        cos_sim = F.cosine_similarity(pred_reps, target_reps, dim=-1).mean()

        return {
            "loss": loss,
            "cos_sim": cos_sim,
            "pred_reps": pred_reps,
            "target_reps": target_reps,
        }


# ---------------------------------------------------------------------------
# 4. JEPA-Guided Single-Source Separation Head
# ---------------------------------------------------------------------------


class JEPAGuidedSeparationHead(nn.Module):
    """Waveform Single-Source Separation Decoder conditioned on frozen Audio-JEPA prior."""

    def __init__(
        self,
        jepa: AudioJEPA,
        embed_dim: int = 128,
        base_channels: int = 32,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.jepa = jepa
        self.jepa.eval()
        for p in self.jepa.parameters():
            p.requires_grad = False

        self.embed_dim = embed_dim
        self.cond_proj = nn.Linear(embed_dim, base_channels * 2)

        self.in_conv = nn.Conv1d(1, base_channels, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2**i
            pad = 2 * dilation
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(base_channels, base_channels, kernel_size=5, padding=pad, dilation=dilation),
                    nn.SiLU(),
                    nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.SiLU(),
                )
            )
        self.out_conv = nn.Conv1d(base_channels, 1, kernel_size=7, padding=3)

    def forward(
        self,
        mix: Tensor,
        clean_source_ref: Tensor | None = None,
        prior_z: Tensor | None = None,
    ) -> Tensor:
        """Separates target stem from mixture guided by JEPA acoustic prior."""
        if mix.ndim == 2:
            mix = mix.unsqueeze(1)

        if prior_z is None:
            assert clean_source_ref is not None, "Either prior_z or clean_source_ref must be provided"
            prior_z = self.jepa.extract_prior(clean_source_ref)

        cond = self.cond_proj(prior_z)
        gamma, beta = cond.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        h = F.silu(self.in_conv(mix))
        h = h * (1.0 + gamma) + beta

        for block in self.blocks:
            h = h + block(h)

        out_stem = torch.tanh(self.out_conv(h))
        return out_stem


__all__ = [
    "AudioJEPA",
    "AudioPatchEmbed",
    "JEPAEncoderBlock",
    "JEPAGuidedSeparationHead",
]
