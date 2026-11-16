"""
Comprehensive Model Evaluation Script for AI Video Forgery Detection.
Evaluates the trained model on held-out test dataset, generates classification metrics,
branch ablation comparisons, confusion matrices, and ROC/PR curves.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg") # Headless plotting backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)

from models.detector import VideoForgeryDetector
from dataset import VideoFaceFramesDataset

OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(checkpoint_path="checkpoints/best_forgery_detector.pt",
                   batch_size=4, num_frames=4, device="cuda"):
    print("\n" + "=" * 70)
    print("       AI VIDEO FORGERY DETECTOR - TEST SET BENCHMARK EVALUATION")
    print("=" * 70)

    test_ds = VideoFaceFramesDataset(
        root_dir="data/frames",
        split="test",
        num_frames=num_frames,
        is_training=False
    )
    print(f"Loaded Test Dataset: {len(test_ds)} video clips (Sampled {num_frames} frames/clip)")

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VideoForgeryDetector(pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    all_labels = []
    all_fused_probs = []
    branch_probs = {"spatial": [], "frequency": [], "temporal": []}
    video_ids = []

    print("Running inference across test clips...")
    with torch.no_grad():
        for frames, labels, vids, *rest in test_loader:
            frames = frames.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(frames)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_fused_probs.extend(out["probability"].cpu().numpy().tolist())
            video_ids.extend(vids)

            for b in ["spatial", "frequency", "temporal"]:
                branch_probs[b].extend(out["branch_probabilities"][b].cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred_prob = np.array(all_fused_probs)
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)

    # Core Metrics
    acc = accuracy_score(y_true, y_pred_bin)
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec = recall_score(y_true, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()

    # Per-branch ablation metrics
    branch_metrics = {}
    for b, probs in branch_probs.items():
        b_p = np.array(probs)
        b_bin = (b_p >= 0.5).astype(int)
        try:
            b_auc = roc_auc_score(y_true, b_p)
        except Exception:
            b_auc = 0.5
        branch_metrics[b] = {
            "accuracy": float(accuracy_score(y_true, b_bin)),
            "roc_auc": float(b_auc),
            "f1": float(f1_score(y_true, b_bin, zero_division=0))
        }

    # Print results
    print("\n" + "-" * 70)
    print("                     OVERALL CLASSIFICATION RESULTS")
    print("-" * 70)
    print(f"  Test Accuracy  : {acc * 100:.2f}%")
    print(f"  ROC-AUC        : {auc:.4f}")
    print(f"  Precision (AI) : {prec * 100:.2f}%")
    print(f"  Recall (AI)    : {rec * 100:.2f}%")
    print(f"  F1-Score       : {f1:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("-" * 70)
    print("                 MULTI-BRANCH ABLATION COMPARISON")
    print("-" * 70)
    print(f"  Spatial Branch     : Acc={branch_metrics['spatial']['accuracy'] * 100:.1f}%, AUC={branch_metrics['spatial']['roc_auc']:.4f}")
    print(f"  Frequency Branch   : Acc={branch_metrics['frequency']['accuracy'] * 100:.1f}%, AUC={branch_metrics['frequency']['roc_auc']:.4f}")
    print(f"  Temporal Branch    : Acc={branch_metrics['temporal']['accuracy'] * 100:.1f}%, AUC={branch_metrics['temporal']['roc_auc']:.4f}")
    print(f"  * FUSED ENSEMBLE * : Acc={acc * 100:.2f}%, AUC={auc:.4f}")
    print("=" * 70)

    # 1. Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["REAL", "AI (FAKE)"]
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="Test Set Confusion Matrix",
           ylabel="True Label", xlabel="Predicted Label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black", fontsize=14)
    fig.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # 2. Plot ROC Curves (Ensemble + Branches)
    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"Ensemble (AUC = {auc:.3f})")

    colors = {"spatial": "#10B981", "frequency": "#F59E0B", "temporal": "#8B5CF6"}
    for b, probs in branch_probs.items():
        b_fpr, b_tpr, _ = roc_curve(y_true, np.array(probs))
        b_auc = branch_metrics[b]["roc_auc"]
        plt.plot(b_fpr, b_tpr, color=colors.get(b, "#6B7280"), lw=1.5, linestyle="--",
                 label=f"{b.capitalize()} Branch (AUC = {b_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("ROC Curves - Video Forgery Detection", fontsize=13)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = OUTPUT_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # Save metrics JSON
    metrics_report = {
        "overall": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
        },
        "branch_ablation": branch_metrics,
        "plots": {
            "confusion_matrix": str(cm_path.resolve()),
            "roc_curve": str(roc_path.resolve())
        }
    }
    with open(OUTPUT_DIR / "evaluation_report.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    print(f"\nEvaluation plots and JSON report saved to: {OUTPUT_DIR.resolve()}")
    return metrics_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI Video Forgery Detector")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_forgery_detector.pt",
                        help="Path to trained checkpoint")
    parser.add_argument("--num_frames", type=int, default=4, help="Frames per video clip")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_model(checkpoint_path=args.checkpoint, num_frames=args.num_frames, device=device)
