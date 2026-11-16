import os
import math
import joblib
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn.functional as F
from transformers import CLIPModel

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 200_000_000

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE_PATH = BASE_DIR / "models" / "detector_v3.joblib"

MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

class ImageForgeryDetector:
    _instance = None

    def __init__(self, bundle_path=None):
        self.bundle_path = Path(bundle_path or DEFAULT_BUNDLE_PATH)
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Image detector bundle not found: {self.bundle_path}")

        print(f"[ImageDetector] Loading bundle from: {self.bundle_path}")
        self.bundle = joblib.load(self.bundle_path)
        self.scaler = self.bundle["scaler"]
        self.head = self.bundle["head"]
        self.temp = float(self.bundle.get("temperature", 1.0))
        self.backbone_name = self.bundle.get("backbone", "openai/clip-vit-large-patch14")
        self.crop = int(self.bundle.get("crop", 224))
        self.band_fake = float(self.bundle.get("band_fake", 0.80))
        self.band_real = float(self.bundle.get("band_real", 0.20))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ImageDetector] Initializing CLIP backbone ({self.backbone_name}) on: {self.device}")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.clip = CLIPModel.from_pretrained(self.backbone_name, torch_dtype=dtype).to(self.device).eval()
        for p in self.clip.parameters():
            p.requires_grad_(False)
        print("[ImageDetector] Ready for inference!")

    @classmethod
    def get_instance(cls, bundle_path=None):
        """Singleton instance to avoid reloading multi-gigabyte models into VRAM."""
        if cls._instance is None:
            cls._instance = ImageForgeryDetector(bundle_path)
        return cls._instance

    def _ensure_min_size(self, img):
        w, h = img.size
        if min(w, h) < self.crop:
            s = self.crop / min(w, h)
            img = img.resize((max(self.crop, math.ceil(w * s)), max(self.crop, math.ceil(h * s))), Image.Resampling.BICUBIC)
        return img

    def _eval_views(self, img):
        """Extracts 10 deterministic views: center + 4 corners/offsets, with & without horizontal flip."""
        img = self._ensure_min_size(img.convert("RGB"))
        w, h = img.size
        xs = [(w - self.crop) // 2, 0, w - self.crop, 0, w - self.crop]
        ys = [(h - self.crop) // 2, 0, 0, h - self.crop, h - self.crop]
        out = []
        for x0, y0 in zip(xs, ys):
            a = np.asarray(img.crop((x0, y0, x0 + self.crop, y0 + self.crop)))
            out.append(a)
            out.append(a[:, ::-1])
        return np.ascontiguousarray(np.stack(out))

    @torch.no_grad()
    def _embed(self, batch_u8):
        x = torch.from_numpy(batch_u8).permute(0, 3, 1, 2).float().div(255.0)
        mean = MEAN.to(self.device)
        std = STD.to(self.device)
        x = ((x.to(self.device) - mean) / std).to(dtype=self.clip.dtype)
        pooled = self.clip.vision_model(pixel_values=x).pooler_output
        f = self.clip.visual_projection(pooled).float()
        return F.normalize(f, dim=-1).cpu().numpy()

    def analyze(self, image_path):
        """
        Analyzes an image file for AI forgery / synthetic generation.
        Returns complete forensic dictionary.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGB")
        width, height = img.size

        views = self._eval_views(img) # (10, 224, 224, 3)
        embeddings = self._embed(views) # (10, 768)

        feats = self.scaler.transform(embeddings)
        z = self.head.decision_function(feats) / self.temp
        pv = 1.0 / (1.0 + np.exp(-z))
        p_ai = float(pv.mean())
        spread = float(pv.std())

        if p_ai >= self.band_fake:
            verdict = "AI GENERATED / MANIPULATED"
            verdict_code = "FAKE"
            confidence = "High" if p_ai >= 0.85 else "Moderate"
        elif p_ai <= self.band_real:
            verdict = "AUTHENTIC (REAL PHOTOGRAPH)"
            verdict_code = "REAL"
            confidence = "High" if p_ai <= 0.15 else "Moderate"
        else:
            verdict = "SUSPICIOUS / UNCERTAIN"
            verdict_code = "UNCERTAIN"
            confidence = "Borderline"

        if verdict != "SUSPICIOUS / UNCERTAIN" and spread > 0.25:
            verdict = "SUSPICIOUS (HIGH ARTIFACT VARIANCE)"
            verdict_code = "UNCERTAIN"
            confidence = "Borderline"

        ai_pct = round(p_ai * 100.0, 1)
        real_pct = round((1.0 - p_ai) * 100.0, 1)

        return {
            "media_type": "image",
            "file_name": path.name,
            "image_dimensions": f"{width}x{height}",
            "verdict": {
                "overall_decision": verdict,
                "code": verdict_code,
                "confidence_level": confidence,
                "overall_ai_percentage": ai_pct,
                "overall_real_percentage": real_pct
            },
            "content_composition": {
                "ai_content_percentage": ai_pct if verdict_code == "FAKE" else (0.0 if verdict_code == "REAL" else ai_pct),
                "real_content_percentage": 100.0 - (ai_pct if verdict_code == "FAKE" else (0.0 if verdict_code == "REAL" else ai_pct)),
                "summary": (
                    f"Analyzed {len(views)} visual perspective regions across {width}x{height} resolution. "
                    f"Estimated AI probability: {ai_pct}%, view variance spread: {spread:.3f}."
                )
            },
            "forensic_details": {
                "clip_backbone": self.backbone_name,
                "calibrated_temperature": round(self.temp, 3),
                "view_variance_spread": round(spread, 4),
                "views_analyzed": len(views)
            }
        }
