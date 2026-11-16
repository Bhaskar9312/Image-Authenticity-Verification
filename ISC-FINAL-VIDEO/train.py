"""
End-to-End Robust Training Pipeline for AI Video Forgery Detection (v4).
Trains ConvNeXt-Tiny Spatial, 8x8 Orthonormal DCT Frequency, and Differential BiGRU Temporal modalities
across 34,000+ multi-source real & deepfake video frames and generative AI faces with compression degradation.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

from models.detector import VideoForgeryDetector
from dataset import VideoFaceFramesDataset, DEFAULT_FRAME_SIZE

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class SmoothBCEWithLogitsLoss(nn.Module):
    """BCE with label smoothing and optional class balance weighting."""
    def __init__(self, smoothing=0.04, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        smoothed_targets = targets * (1.0 - 2.0 * self.smoothing) + self.smoothing
        return self.bce(logits, smoothed_targets)


def evaluate_split(model, loader, device):
    """Evaluates the multi-branch ensemble on held-out splits."""
    model.eval()
    all_scores, all_labels = [], []
    branch_scores = {"spatial": [], "frequency": [], "temporal": []}
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for frames, labels, *rest in loader:
            frames = frames.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(frames)
                loss = criterion(out["score"], labels)

            total_loss += loss.item() * len(labels)
            probs = out["probability"].cpu().numpy()
            all_scores.extend(probs)
            all_labels.extend(labels.cpu().numpy())

            for b_name in ["spatial", "frequency", "temporal"]:
                b_prob = out["branch_probabilities"][b_name].cpu().numpy()
                branch_scores[b_name].extend(b_prob)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    binary_preds = (all_scores >= 0.5).astype(int)

    acc = accuracy_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    prec = precision_score(all_labels, binary_preds, zero_division=0)
    rec = recall_score(all_labels, binary_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auc = 0.5

    branch_aucs = {}
    for b_name, b_probs in branch_scores.items():
        try:
            branch_aucs[b_name] = float(roc_auc_score(all_labels, np.array(b_probs)))
        except Exception:
            branch_aucs[b_name] = 0.5

    return {
        "loss": total_loss / max(1, len(all_labels)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "branch_aucs": branch_aucs
    }


def train_robust_ensemble(epochs=6, batch_size=16, num_frames=4, lr=2e-4, device="cuda", resume=False):
    print("\n" + "=" * 70)
    print(">>> TRAINING ROBUST MULTI-MODAL FORGERY DETECTOR (v4 - INDUSTRIAL SOTA)")
    print("=" * 70)

    train_ds = VideoFaceFramesDataset(
        root_dir="data/frames",
        split="train",
        num_frames=num_frames,
        is_training=True,
        degrade_prob=0.70,
        include_hf_images=True
    )
    val_ds = VideoFaceFramesDataset(
        root_dir="data/frames",
        split="val",
        num_frames=num_frames,
        is_training=False,
        degrade_prob=0.0,
        include_hf_images=False
    )
    test_ds = VideoFaceFramesDataset(
        root_dir="data/frames",
        split="test",
        num_frames=num_frames,
        is_training=False,
        degrade_prob=0.0,
        include_hf_images=False
    )

    print(f"Dataset clips: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    print(f"Sampled frames per clip: {num_frames} (Resolution: {DEFAULT_FRAME_SIZE}x{DEFAULT_FRAME_SIZE})")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    best_model_path = CHECKPOINT_DIR / "best_forgery_detector.pt"
    model = VideoForgeryDetector(pretrained=True).to(device)

    if resume and best_model_path.exists():
        print(f"[Train] Resuming weights from: {best_model_path}")
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)

    # Unfreeze ConvNeXt stages 1, 2, and 3 for comprehensive boundary & generative seam learning
    model.spatial.unfreeze_stages(from_stage=1)

    backbone_params = [p for n, p in model.spatial.backbone.named_parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not any(p is bp for bp in backbone_params) and p.requires_grad]

    print(f"Trainable ConvNeXt Backbone params (stages 1-3): {sum(p.numel() for p in backbone_params):,}")
    print(f"Trainable Heads / Spectral / Temporal params:     {sum(p.numel() for p in head_params):,}")

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1.5e-5, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr, "weight_decay": 1e-4}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    criterion = SmoothBCEWithLogitsLoss(smoothing=0.04)

    best_val_auc = 0.0
    best_model_path = CHECKPOINT_DIR / "best_forgery_detector.pt"
    history = []

    print("\nStarting Training with Multi-Task Auxiliary Guidance & Multi-Source Datasets...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        all_train_preds, all_train_labels = [], []

        for batch_idx, (frames, labels, _, is_static) in enumerate(train_loader):
            frames = frames.to(device)
            labels = labels.to(device)
            is_video = (~is_static).to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(frames)
                l_fused = criterion(out["score"], labels)
                l_spatial = criterion(out["spatial_score"], labels)
                l_freq = criterion(out["frequency_score"], labels)
                if is_video.any():
                    l_temp = criterion(out["temporal_score"][is_video], labels[is_video])
                else:
                    l_temp = torch.tensor(0.0, device=device)

                # Multi-task auxiliary loss with strong spatial boundary & temporal guidance
                loss = l_fused + 0.60 * l_spatial + 0.20 * l_freq + 0.40 * l_temp

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labels)
            all_train_preds.extend(out["probability"].detach().cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = accuracy_score(all_train_labels, (np.array(all_train_preds) >= 0.5).astype(int))
                sys.stdout.write(f"\rEpoch [{epoch:2d}/{epochs}] Step [{batch_idx + 1:3d}/{len(train_loader)}] "
                                 f"Loss: {total_loss / len(all_train_labels):.4f} | Acc: {batch_acc * 100:.1f}%\n")
                sys.stdout.flush()

        scheduler.step()
        epoch_time = time.time() - start_time
        train_acc = accuracy_score(all_train_labels, (np.array(all_train_preds) >= 0.5).astype(int))

        # Evaluate on validation split
        val_metrics = evaluate_split(model, val_loader, device)

        print(f"\n--> Epoch {epoch:2d} finished in {epoch_time:.1f}s | Train Acc: {train_acc * 100:.1f}% | "
              f"Val Acc: {val_metrics['accuracy'] * 100:.1f}% | Val AUC: {val_metrics['auc']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")
        print(f"    Branch AUCs: Spatial={val_metrics['branch_aucs']['spatial']:.3f}, "
              f"Freq={val_metrics['branch_aucs']['frequency']:.3f}, "
              f"Temporal={val_metrics['branch_aucs']['temporal']:.3f}")
        print(f"    Fusion Weights: {model.fusion_weights.detach().cpu().numpy().round(3)}")

        epoch_stat = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_ds),
            "train_acc": float(train_acc),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_f1": val_metrics["f1"],
            "val_branch_aucs": val_metrics["branch_aucs"],
            "fusion_weights": model.fusion_weights.detach().cpu().numpy().tolist()
        }
        history.append(epoch_stat)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            print(f"    *** NEW BEST VAL AUC: {best_val_auc:.4f}! Saving checkpoint...")
            torch.save({
                "model_state_dict": model.state_dict(),
                "fusion_weights": model.fusion_weights.detach().cpu().numpy().tolist(),
                "val_metrics": val_metrics,
                "epoch": epoch
            }, best_model_path)

    # Save training history
    with open(CHECKPOINT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final evaluation on held-out test set
    print("\n" + "=" * 70)
    print(">>> EVALUATION ON HELD-OUT TEST SET (best_forgery_detector.pt)")
    print("=" * 70)
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_metrics = evaluate_split(model, test_loader, device)

    print(f"Test Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Test Precision: {test_metrics['precision'] * 100:.2f}%")
    print(f"Test Recall:    {test_metrics['recall'] * 100:.2f}%")
    print(f"Test F1-Score:  {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC:   {test_metrics['auc']:.4f}")
    print(f"Test Branch AUCs: {test_metrics['branch_aucs']}")

    with open(CHECKPOINT_DIR / "test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nRobust retraining complete! Checkpoint saved to: {best_model_path.resolve()}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Robust AI Video Forgery Detector")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_frames", type=int, default=4, help="Frames per video clip")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from existing best_forgery_detector.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    train_robust_ensemble(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        lr=args.lr,
        device=device,
        resume=args.resume
    )
