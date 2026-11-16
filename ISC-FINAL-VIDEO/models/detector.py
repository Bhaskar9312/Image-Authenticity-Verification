"""
State-of-the-Art Multi-Modal AI Video Forgery Detection Architecture (v4).
Engineered with:
  1. ConvNeXt-Tiny Spatial & High-Level Artifact Branch (Pre-trained on ImageNet-12k).
  2. Fixed 8x8 Orthonormal Block DCT Spectral Frequency Branch.
  3. Differential Motion BiGRU Temporal Coherence Branch with Temporal Self-Attention.
  4. Dynamically Calibrated Multi-Branch Logit Fusion with learnable temperature scaling.
"""

import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvNeXtSpatialBranch(nn.Module):
    """
    Spatial & Artifact Branch: Pre-trained ConvNeXt-Tiny backbone with L2-normalized feature representation.
    ConvNeXt inverted-bottleneck 7x7 depthwise convolutions naturally capture high-frequency
    facial blending seams, skin texture anomalies, and generative diffusion artifacts.
    """
    def __init__(self, pretrained=True, dropout=0.30):
        super().__init__()
        try:
            self.backbone = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=0)
        except Exception as exc:
            warnings.warn(f"Failed to load pretrained convnext_tiny ({exc}); initializing random.")
            self.backbone = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)

        # Freeze stem and first 2 stages by default to preserve low-level visual priors
        for name, param in self.backbone.named_parameters():
            if "stages.0" in name or "stages.1" in name or "stem" in name:
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1)
        )

    def unfreeze_stages(self, from_stage=2):
        """Unfreezes stages from_stage onwards for fine-tuning."""
        for name, param in self.backbone.named_parameters():
            if any(f"stages.{s}" in name for s in range(from_stage, 4)):
                param.requires_grad = True

    def extract_features(self, x):
        """Extracts L2-normalized 768-dim feature representation."""
        raw = self.backbone(x)
        return F.normalize(raw, p=2, dim=-1)

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.classifier(feats).squeeze(-1)
        return logits, feats


class FixedBlockDCT(nn.Module):
    """
    Orthonormal 8x8 block Discrete Cosine Transform (DCT) across RGB channels.
    Exposes high-frequency grid artifacts left by neural upsamplers and generative decoders.
    """
    def __init__(self):
        super().__init__()
        basis = torch.zeros(8, 8, 8, 8)
        for u in range(8):
            au = math.sqrt(1 / 8) if u == 0 else math.sqrt(2 / 8)
            for v in range(8):
                av = math.sqrt(1 / 8) if v == 0 else math.sqrt(2 / 8)
                for x in range(8):
                    for y in range(8):
                        basis[u, v, x, y] = (
                            au * av *
                            math.cos(math.pi * (2 * x + 1) * u / 16) *
                            math.cos(math.pi * (2 * y + 1) * v / 16)
                        )
        kernels = basis.reshape(64, 1, 8, 8).repeat(3, 1, 1, 1)
        self.register_buffer("weight", kernels)

    def forward(self, x):
        coeff = F.conv2d(x, self.weight, stride=8, groups=3)
        return torch.sign(coeff) * torch.log1p(torch.abs(coeff) * 3.0)


class FrequencySpectralBranch(nn.Module):
    """
    Frequency Branch: Analyzes DCT spectral artifacts across 192 coefficient planes.
    Equipped with GroupNorm and GELU activations.
    """
    def __init__(self, dropout=0.30):
        super().__init__()
        self.dct = FixedBlockDCT()
        self.backbone = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 192, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 192),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def extract_features(self, x):
        return self.pool(self.backbone(self.dct(x))).flatten(1)

    def forward(self, x):
        feats = self.extract_features(x)
        logits = self.classifier(feats).squeeze(-1)
        return logits, feats


class TemporalBiGRUBranch(nn.Module):
    """
    Temporal Branch: Motion-normalized differential BiGRU with temporal self-attention.
    Captures facial landmark jitter, lip-sync drift, and boundary flickering over time.
    """
    def __init__(self, input_size=768 * 2 + 1, hidden_size=128, dropout=0.30):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attn_proj = nn.Linear(hidden_size * 2, 64)
        self.attn_v = nn.Linear(64, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        z = self.project(x)
        out, _ = self.gru(z) # (Batch, Seq_Len, hidden_size * 2)
        attn = torch.tanh(self.attn_proj(out))
        scores = self.attn_v(attn).squeeze(-1)
        attn_weights = F.softmax(scores, dim=-1)
        context = (out * attn_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(context).squeeze(-1)
        return logits, attn_weights


class VideoForgeryDetector(nn.Module):
    """
    Complete Multi-Modal AI Video Forgery Detector with Calibrated Multi-Branch Fusion.
    - Spatial: ConvNeXt-Tiny L2-normalized visual & seam inspection (768-dim)
    - Frequency: 8x8 Orthonormal DCT spectral anomaly detection (256-dim)
    - Temporal: Differential BiGRU with temporal attention (1537-dim input)
    - Calibrated learned fusion with per-branch temperature scaling
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.spatial = ConvNeXtSpatialBranch(pretrained=pretrained)
        self.frequency = FrequencySpectralBranch()
        self.temporal = TemporalBiGRUBranch(input_size=768 * 2 + 1)

        # Initial fusion logits: balanced across all 3 forensic modalities
        self.weight_logits = nn.Parameter(torch.tensor([1.2, 0.8, 0.8]))
        # Learnable per-branch calibration temperatures
        self.log_temperatures = nn.Parameter(torch.zeros(3))

    @property
    def fusion_weights(self):
        return F.softmax(self.weight_logits, dim=0)

    def extract_temporal_input(self, frame_features):
        """
        frame_features: (Batch, Seq_len, 768) L2-normalized embeddings.
        Computes:
          - feats: (Batch, Seq_len, 768)
          - diffs: (Batch, Seq_len, 768) normalized frame differentials
          - cos_dist: (Batch, Seq_len, 1) cosine distance from previous frame
        """
        b, t, d = frame_features.shape
        diffs = torch.zeros_like(frame_features)
        cos_dist = torch.zeros(b, t, 1, device=frame_features.device, dtype=frame_features.dtype)

        if t > 1:
            diffs[:, 1:] = frame_features[:, 1:] - frame_features[:, :-1]
            sims = (frame_features[:, 1:] * frame_features[:, :-1]).sum(dim=-1, keepdim=True)
            cos_dist[:, 1:] = (1.0 - sims).clamp(0.0, 2.0)

        return torch.cat([frame_features, diffs, cos_dist], dim=-1) # (Batch, Seq_len, 1537)

    def forward(self, frames, precomputed_features=None):
        """
        frames: (Batch, Seq_Len, 3, H, W)
        precomputed_features: Optional (Batch, Seq_Len, 768) L2-normalized features
        """
        b, t, c, h, w = frames.shape
        flat_frames = frames.reshape(b * t, c, h, w)

        # 1. Spatial Branch
        if precomputed_features is not None:
            spatial_frame_logits = self.spatial.classifier(precomputed_features.reshape(b * t, -1)).squeeze(-1)
            frame_feats = precomputed_features
        else:
            spatial_frame_logits, frame_feats = self.spatial(flat_frames)
            frame_feats = frame_feats.reshape(b, t, 768)

        spatial_frame_logits = spatial_frame_logits.reshape(b, t)
        spatial_logits = spatial_frame_logits.mean(dim=1)

        # 2. Frequency Branch
        freq_frame_logits, _ = self.frequency(flat_frames)
        freq_frame_logits = freq_frame_logits.reshape(b, t)
        frequency_logits = freq_frame_logits.mean(dim=1)

        # 3. Temporal Branch
        temporal_input = self.extract_temporal_input(frame_feats)
        temporal_logits, temporal_attn = self.temporal(temporal_input)

        # 4. Calibration & Fusion
        temps = torch.exp(self.log_temperatures).clamp(0.5, 3.0)
        branch_logits = torch.stack([
            spatial_logits / temps[0],
            frequency_logits / temps[1],
            temporal_logits / temps[2]
        ], dim=1) # (Batch, 3)

        fused_score = (branch_logits * self.fusion_weights.unsqueeze(0)).sum(dim=1)

        # Frame-level combined scores for temporal timeline analysis
        frame_ai_scores = torch.sigmoid(0.60 * spatial_frame_logits + 0.40 * freq_frame_logits)

        return {
            "score": fused_score,
            "probability": torch.sigmoid(fused_score),
            "spatial_score": spatial_logits,
            "frequency_score": frequency_logits,
            "temporal_score": temporal_logits,
            "branch_probabilities": {
                "spatial": torch.sigmoid(spatial_logits),
                "frequency": torch.sigmoid(frequency_logits),
                "temporal": torch.sigmoid(temporal_logits)
            },
            "fusion_weights": self.fusion_weights,
            "frame_ai_scores": frame_ai_scores,
            "temporal_attention": temporal_attn
        }
