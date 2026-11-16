"""
Comprehensive AI Video Forgery Analysis Engine (v3).
Features:
  - Multi-Person Persistent Tracking (IoU & Centroid Proximity) to eliminate inter-person jitter
  - Sliding-Window Temporal Aggregation across arbitrary video durations
  - Motion-Normalized Differential BiGRU & DCT Spectral Analysis
  - Per-Subject Forensic Attribution & Video-Level Verdict Aggregation
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import torchvision.transforms.functional as TF

from models.detector import VideoForgeryDetector
from dataset import VideoFileProcessor, NORM_MEAN, NORM_STD, DEFAULT_FRAME_SIZE


class VideoForgeryAnalyzer:
    """
    Forensic Video Forgery Detection & Content Quantification Engine.
    Handles single and multi-person videos using multi-track persistent analysis.
    """
    def __init__(self, checkpoint_path=None, device=None, threshold=0.50):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.processor = VideoFileProcessor(frame_size=DEFAULT_FRAME_SIZE, face_margin=0.35)
        
        # Load model
        self.model = VideoForgeryDetector(pretrained=False).to(self.device)
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[VideoAnalyzer] Loading model checkpoint from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
            if "fusion_weights" in ckpt:
                print(f"[VideoAnalyzer] Loaded fusion weights: {np.array(ckpt['fusion_weights']).round(3)}")
        else:
            print("[VideoAnalyzer] Warning: No checkpoint found; initializing pretrained backbone.")
            self.model = VideoForgeryDetector(pretrained=True).to(self.device)
            
        self.model.eval()

    def _analyze_track(self, track_data, window_size=4):
        """Analyzes a single face track sequence with sliding-window evaluation."""
        frames_tensor = track_data["tensor"] # (1, N, 3, H, W)
        timestamps = track_data["timestamps"]
        crops = track_data["crops"]
        detected_flags = track_data["detected_faces"]
        n_frames = len(timestamps)

        # Pad if smaller than window_size
        if n_frames < window_size:
            pad_needed = window_size - n_frames
            pad_tensor = frames_tensor[:, -1:].repeat(1, pad_needed, 1, 1, 1)
            f_tensor = torch.cat([frames_tensor, pad_tensor], dim=1)
            padded_n = window_size
        else:
            f_tensor = frames_tensor
            padded_n = n_frames

        stride = max(1, window_size // 2)
        window_scores, window_spatial, window_freq, window_temp, window_times = [], [], [], [], []

        with torch.no_grad():
            for start_idx in range(0, padded_n - window_size + 1, stride):
                end_idx = start_idx + window_size
                win_tensor = f_tensor[:, start_idx:end_idx].to(self.device)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(win_tensor)

                w_prob = float(out["probability"].item())
                s_prob = float(out["branch_probabilities"]["spatial"].item())
                f_prob = float(out["branch_probabilities"]["frequency"].item())
                t_prob = float(out["branch_probabilities"]["temporal"].item())

                mid_t = timestamps[min(start_idx + window_size // 2, n_frames - 1)]
                window_scores.append(w_prob)
                window_spatial.append(s_prob)
                window_freq.append(f_prob)
                window_temp.append(t_prob)
                window_times.append(mid_t)

        window_scores = np.array(window_scores)
        window_spatial = np.array(window_spatial)
        window_freq = np.array(window_freq)
        window_temp = np.array(window_temp)

        overall_prob = float(np.median(window_scores))
        spatial_prob = float(np.median(window_spatial))
        freq_prob = float(np.median(window_freq))
        temporal_prob = float(np.median(window_temp))

        # Frame-Level Scores
        frame_ai_scores = []
        for i, t_sec in enumerate(timestamps):
            dists = np.abs(np.array(window_times) - t_sec)
            weights = np.exp(-dists / 1.5)
            weights /= (weights.sum() + 1e-6)
            f_score = float(np.sum(weights * window_scores))
            frame_ai_scores.append(f_score)
        frame_ai_scores = np.array(frame_ai_scores)

        ai_frame_mask = frame_ai_scores >= self.threshold
        ai_frames_count = int(np.sum(ai_frame_mask))
        real_frames_count = n_frames - ai_frames_count

        ai_content_pct = round((ai_frames_count / n_frames) * 100.0, 1)
        real_content_pct = round(100.0 - ai_content_pct, 1)

        # Timeline and suspicious segments
        timeline = []
        suspicious_segments = []
        in_suspicious_block = False
        block_start = 0.0

        for i in range(n_frames):
            t_sec = timestamps[i]
            score = float(frame_ai_scores[i])
            is_fake_frame = bool(score >= self.threshold)
            time_str = f"{int(t_sec // 60):02d}:{t_sec % 60:05.2f}"
            frame_status = "AI MANIPULATED" if is_fake_frame else "REAL"

            timeline.append({
                "frame_index": i,
                "timestamp_sec": round(t_sec, 2),
                "timestamp_str": time_str,
                "ai_probability": round(score, 4),
                "ai_percentage": round(score * 100.0, 1),
                "status": frame_status,
                "face_detected": detected_flags[i]
            })

            if is_fake_frame:
                if not in_suspicious_block:
                    in_suspicious_block = True
                    block_start = t_sec
            else:
                if in_suspicious_block:
                    in_suspicious_block = False
                    block_end = timestamps[max(0, i - 1)]
                    suspicious_segments.append({
                        "start_time": f"{int(block_start // 60):02d}:{block_start % 60:05.2f}",
                        "end_time": f"{int(block_end // 60):02d}:{block_end % 60:05.2f}",
                        "start_sec": round(block_start, 2),
                        "end_sec": round(block_end, 2),
                        "type": "Face/Visual Manipulation"
                    })

        if in_suspicious_block:
            block_end = timestamps[-1]
            suspicious_segments.append({
                "start_time": f"{int(block_start // 60):02d}:{block_start % 60:05.2f}",
                "end_time": f"{int(block_end // 60):02d}:{block_end % 60:05.2f}",
                "start_sec": round(block_start, 2),
                "end_sec": round(block_end, 2),
                "type": "Face/Visual Manipulation"
            })

        return {
            "track_id": track_data.get("track_id", 0),
            "position_name": track_data.get("position_name", "Primary Subject"),
            "n_frames": n_frames,
            "overall_prob": overall_prob,
            "spatial_prob": spatial_prob,
            "freq_prob": freq_prob,
            "temporal_prob": temporal_prob,
            "ai_content_pct": ai_content_pct,
            "real_content_pct": real_content_pct,
            "ai_frames_count": ai_frames_count,
            "real_frames_count": real_frames_count,
            "frame_ai_scores": frame_ai_scores,
            "timeline": timeline,
            "suspicious_segments": suspicious_segments,
            "crops": crops
        }

    def analyze(self, video_path, sample_fps=2.0, max_frames=64, window_size=4, save_evidence_dir=None):
        """
        Analyzes a video for AI forgery using multi-track persistent evaluation.
        Supports single-person and multi-person videos with complete forensic breakdown.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 1. Process Video & Detect All Face Tracks
        primary_clip = self.processor.process_video(video_path, sample_fps=sample_fps, max_frames=max_frames)
        all_tracks = primary_clip.get("all_tracks", [primary_clip])
        duration = primary_clip["duration"]
        fps = primary_clip["fps"]
        total_video_frames = primary_clip["total_video_frames"]

        # 2. Analyze Each Track Individually
        track_results = [self._analyze_track(t, window_size=window_size) for t in all_tracks]

        # 3. Aggregate Video-Level Forensic Metrics
        # If any person in the video is deepfaked, the video contains AI forgery!
        overall_prob = float(max(t["overall_prob"] for t in track_results))
        ai_content_pct = float(max(t["ai_content_pct"] for t in track_results))
        real_content_pct = round(100.0 - ai_content_pct, 1)

        spatial_prob = float(max(t["spatial_prob"] for t in track_results))
        freq_prob = float(max(t["freq_prob"] for t in track_results))
        temporal_prob = float(max(t["temporal_prob"] for t in track_results))

        all_ai_frames = sum(t["ai_frames_count"] for t in track_results)
        all_total_frames = sum(t["n_frames"] for t in track_results)
        real_frames_count = all_total_frames - all_ai_frames

        fusion_weights = self.model.fusion_weights.detach().cpu().numpy().tolist()

        # Combine suspicious segments across tracks
        combined_suspicious = []
        for t in track_results:
            p_name = t["position_name"]
            for seg in t["suspicious_segments"]:
                seg_copy = dict(seg)
                seg_copy["subject"] = p_name
                combined_suspicious.append(seg_copy)

        has_extended_manipulation = len(combined_suspicious) > 0 and sum(s["end_sec"] - s["start_sec"] for s in combined_suspicious) >= 2.0

        # 4. Forensic Decision Policy
        has_detected_faces = any(any(t.get("detected_faces", [False])) for t in all_tracks)

        if not has_detected_faces:
            # Environmental Scene Video (No Human Faces Present)
            if overall_prob < 0.50:
                verdict = "AUTHENTIC (REAL VIDEO - NO FACES DETECTED)"
                verdict_code = "REAL"
                confidence_level = "High"
                ai_content_pct = 0.0
                real_content_pct = 100.0
                all_ai_frames = 0
                real_frames_count = all_total_frames
                combined_suspicious = []
            else:
                verdict = "AI GENERATED VIDEO (SYNTHETIC SCENE)"
                verdict_code = "FAKE"
                confidence_level = "High" if overall_prob >= 0.65 else "Moderate"
                ai_content_pct = 100.0
                real_content_pct = 0.0
        else:
            if overall_prob >= 0.50 or ai_content_pct >= 40.0:
                verdict = "AI MANIPULATED (DEEPFAKE)"
                verdict_code = "FAKE"
                confidence_level = "High" if (overall_prob >= 0.65 or ai_content_pct >= 55.0) else "Moderate"
            elif has_extended_manipulation and ai_content_pct >= 15.0:
                verdict = "AI MANIPULATED (TEMPORAL INSERTION / PARTIAL FORGERY)"
                verdict_code = "FAKE"
                confidence_level = "Moderate"
            elif overall_prob <= 0.40 and ai_content_pct < 25.0:
                verdict = "AUTHENTIC (REAL VIDEO)"
                verdict_code = "REAL"
                confidence_level = "High" if overall_prob <= 0.25 else "Moderate"
            else:
                verdict = "SUSPICIOUS / LOW-CONFIDENCE FORGERY"
                verdict_code = "UNCERTAIN"
                confidence_level = "Borderline"

        # 5. Extract Evidence Keyframes
        evidence_frames = []
        if save_evidence_dir is not None:
            ev_dir = Path(save_evidence_dir)
            ev_dir.mkdir(parents=True, exist_ok=True)

            # Collect top suspicious crops across all tracks
            all_candidate_crops = []
            for t in track_results:
                for idx, score_val in enumerate(t["frame_ai_scores"]):
                    all_candidate_crops.append({
                        "crop": t["crops"][idx],
                        "score": score_val,
                        "time_str": t["timeline"][idx]["timestamp_str"],
                        "pos_name": t["position_name"]
                    })

            all_candidate_crops.sort(key=lambda x: x["score"], reverse=True)
            top_candidates = all_candidate_crops[:min(4, len(all_candidate_crops))]

            for rank, cand in enumerate(top_candidates):
                crop_img = cand["crop"].copy()
                score_val = cand["score"]
                time_str = cand["time_str"]
                pos = cand["pos_name"]

                draw = ImageDraw.Draw(crop_img)
                tag = f"{pos} | AI: {score_val * 100:.1f}% @ {time_str}"
                draw.rectangle([(0, 0), (crop_img.width, 24)], fill=(200, 20, 20) if score_val >= 0.5 else (20, 160, 40))
                draw.text((8, 4), tag, fill=(255, 255, 255))

                clean_time = time_str.replace(":", "_")
                ev_path = ev_dir / f"evidence_frame_{rank + 1}_{clean_time}.jpg"
                crop_img.save(ev_path)
                evidence_frames.append({
                    "rank": rank + 1,
                    "subject": pos,
                    "timestamp": time_str,
                    "ai_score": round(float(score_val), 4),
                    "path": str(ev_path)
                })

        # Structure per-subject summaries
        tracked_subjects_summary = []
        for t in track_results:
            tracked_subjects_summary.append({
                "track_id": t["track_id"],
                "subject_name": t["position_name"],
                "frames_analyzed": t["n_frames"],
                "ai_probability": round(t["overall_prob"], 4),
                "ai_percentage": round(t["overall_prob"] * 100.0, 1),
                "ai_content_percentage": t["ai_content_pct"],
                "status": "AI MANIPULATED" if t["overall_prob"] >= 0.50 else "AUTHENTIC REAL"
            })

        primary_timeline = track_results[0]["timeline"]

        if not has_detected_faces:
            composition_summary = (
                f"Authentic camera footage ({duration:.1f}s). No human faces detected in frame; "
                f"evaluated scene background textures. AI content is 0.0% and authentic content is 100.0%."
                if verdict_code == "REAL" else
                f"Synthetic scene footage ({duration:.1f}s). Evaluated full scene background textures. AI content is 100.0%."
            )
        else:
            composition_summary = (
                f"This video consists of approximately {ai_content_pct}% AI-manipulated content "
                f"and {real_content_pct}% authentic content across {len(track_results)} tracked subjects."
            )

        report = {
            "video_metadata": {
                "file_name": video_path.name,
                "file_path": str(video_path.resolve()),
                "duration_seconds": round(duration, 2),
                "fps": round(fps, 2),
                "total_frames_in_video": total_video_frames,
                "frames_analyzed": all_total_frames,
                "distinct_faces_tracked": len(track_results) if has_detected_faces else 0,
                "has_detected_faces": has_detected_faces
            },
            "verdict": {
                "overall_decision": verdict,
                "code": verdict_code,
                "confidence_level": confidence_level,
                "overall_ai_probability": round(overall_prob, 4),
                "overall_ai_percentage": round(overall_prob * 100.0, 1),
                "overall_real_percentage": round((1.0 - overall_prob) * 100.0, 1)
            },
            "content_composition": {
                "ai_content_percentage": ai_content_pct,
                "real_content_percentage": real_content_pct,
                "ai_frames_detected": all_ai_frames,
                "real_frames_detected": real_frames_count,
                "summary": composition_summary
            },
            "tracked_subjects": tracked_subjects_summary if has_detected_faces else [],
            "forensic_branch_breakdown": {
                "spatial_branch": {
                    "score": round(spatial_prob, 4),
                    "percentage": round(spatial_prob * 100.0, 1),
                    "weight": round(fusion_weights[0], 3),
                    "description": "ConvNeXt-Tiny facial boundary & multi-scale artifact inspection"
                },
                "frequency_branch": {
                    "score": round(freq_prob, 4),
                    "percentage": round(freq_prob * 100.0, 1),
                    "weight": round(fusion_weights[1], 3),
                    "description": "8x8 Block DCT spectral upsampling grid analysis"
                },
                "temporal_branch": {
                    "score": round(temporal_prob, 4),
                    "percentage": round(temporal_prob * 100.0, 1),
                    "weight": round(fusion_weights[2], 3),
                    "description": "Motion-normalized differential BiGRU sequence coherence"
                }
            },
            "suspicious_segments": combined_suspicious,
            "evidence_frames": evidence_frames,
            "timeline": primary_timeline
        }

        return report

    def print_terminal_report(self, report):
        v = report["verdict"]
        c = report["content_composition"]
        m = report["video_metadata"]
        f = report["forensic_branch_breakdown"]

        print("\n" + "=" * 70)
        print("          AI VIDEO FORGERY DETECTION & FORENSIC REPORT")
        print("=" * 70)
        print(f" Video File      : {m['file_name']}")
        print(f" Duration        : {m['duration_seconds']}s | FPS: {m['fps']} | Analyzed Frames: {m['frames_analyzed']}")
        if m.get('distinct_faces_tracked', 0) == 0:
            print(" Scene Type      : Environmental Scene (No Human Faces Detected)")
        else:
            print(f" Faces Tracked   : {m['distinct_faces_tracked']} individual(s)")
        print("-" * 70)
        print(f" OVERALL VERDICT : {v['overall_decision']} (Confidence: {v['confidence_level']})")
        print(f" AI Probability  : {v['overall_ai_percentage']}% | Real Probability: {v['overall_real_percentage']}%")
        print("-" * 70)

        if len(report.get("tracked_subjects", [])) > 1:
            print(" INDIVIDUAL SUBJECT BREAKDOWN:")
            for s in report["tracked_subjects"]:
                print(f"   * {s['subject_name']}: AI Prob={s['ai_percentage']}% | Status={s['status']} ({s['frames_analyzed']} frames)")
            print("-" * 70)

        print(" CONTENT BREAKDOWN (HOW MUCH IS AI vs REAL):")
        print(f"   [AI Manipulated Content] : {c['ai_content_percentage']}% ({c['ai_frames_detected']} frames)")
        print(f"   [Authentic Real Content] : {c['real_content_percentage']}% ({c['real_frames_detected']} frames)")
        
        bar_len = 40
        ai_bars = int(round(c["ai_content_percentage"] / 100.0 * bar_len))
        real_bars = bar_len - ai_bars
        print(f"   Visual Bar: [{'#' * ai_bars}{'-' * real_bars}]")
        print(f"               ^ AI ({c['ai_content_percentage']}%)    ^ REAL ({c['real_content_percentage']}%)")
        print("-" * 70)
        print(" FORENSIC BRANCH SCORES:")
        print(f"   * Spatial Boundary Artifacts : {f['spatial_branch']['percentage']}% (Weight: {f['spatial_branch']['weight']})")
        print(f"   * Frequency Spectral Anomaly : {f['frequency_branch']['percentage']}% (Weight: {f['frequency_branch']['weight']})")
        print(f"   * Temporal Motion Jitter     : {f['temporal_branch']['percentage']}% (Weight: {f['temporal_branch']['weight']})")
        
        if report["suspicious_segments"]:
            print("-" * 70)
            print(" DETECTED MANIPULATION TIMESTAMPS:")
            for s in report["suspicious_segments"]:
                subj = f"[{s.get('subject', 'Subject')}] " if 'subject' in s else ""
                print(f"   >> {subj}[{s['start_time']} -> {s['end_time']}] - {s['type']}")
        else:
            print("-" * 70)
            print(" No continuous manipulation blocks detected across timestamps.")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze video for AI forgery / deepfake")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_forgery_detector.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling fps")
    parser.add_argument("--max_frames", type=int, default=64, help="Max frames to analyze")
    parser.add_argument("--output_json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--evidence_dir", type=str, default="evidence_output", help="Evidence output dir")
    args = parser.parse_args()

    analyzer = VideoForgeryAnalyzer(checkpoint_path=args.checkpoint)
    report = analyzer.analyze(
        video_path=args.video_path,
        sample_fps=args.fps,
        max_frames=args.max_frames,
        save_evidence_dir=args.evidence_dir
    )

    analyzer.print_terminal_report(report)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Detailed JSON report saved to: {args.output_json}")
