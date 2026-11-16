import math
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .detector_arch import VideoForgeryDetector
from .preprocessing import MultiOrientationVideoProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_CKPT = BASE_DIR / "models" / "best_video_detector.pt"

class VideoForgeryDetectorEngine:
    _instance = None

    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_VIDEO_CKPT)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Video detector checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[VideoDetector] Initializing ConvNeXt-BiGRU Video Engine on: {self.device}")
        self.model = VideoForgeryDetector(pretrained=False).to(self.device)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.processor = MultiOrientationVideoProcessor(frame_size=224)
        print("[VideoDetector] Video Engine ready for inference!")

    @classmethod
    def get_instance(cls, checkpoint_path=None):
        if cls._instance is None:
            cls._instance = VideoForgeryDetectorEngine(checkpoint_path)
        return cls._instance

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

        use_cuda = self.device.type == "cuda"
        dtype = torch.float16 if use_cuda else torch.float32

        with torch.no_grad():
            for start_idx in range(0, padded_n - window_size + 1, stride):
                end_idx = start_idx + window_size
                win_tensor = f_tensor[:, start_idx:end_idx].to(self.device)

                if use_cuda:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model(win_tensor)
                else:
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

        ai_frames_count = int(np.sum(frame_ai_scores >= 0.50))
        real_frames_count = n_frames - ai_frames_count
        ai_content_pct = round((ai_frames_count / max(1, n_frames)) * 100.0, 1)
        real_content_pct = round(100.0 - ai_content_pct, 1)

        timeline = []
        suspicious_segments = []
        in_suspicious_block = False
        block_start = 0.0

        for idx, (t_sec, s_val) in enumerate(zip(timestamps, frame_ai_scores)):
            t_min = int(t_sec // 60)
            t_rem_sec = t_sec % 60
            ts_str = f"{t_min:02d}:{t_rem_sec:05.2f}"
            is_ai = bool(s_val >= 0.50)
            timeline.append({
                "frame_index": idx,
                "timestamp_sec": round(t_sec, 2),
                "timestamp_str": ts_str,
                "ai_score": round(float(s_val), 4),
                "is_ai_frame": is_ai
            })

            if is_ai and not in_suspicious_block:
                in_suspicious_block = True
                block_start = t_sec
            elif not is_ai and in_suspicious_block:
                in_suspicious_block = False
                block_end = t_sec
                if block_end - block_start >= 0.5:
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
            "frame_ai_scores": frame_ai_scores.tolist(),
            "timeline": timeline,
            "suspicious_segments": suspicious_segments
        }

    def analyze(self, video_path, sample_fps=2.0, max_frames=64):
        """
        Full video analysis supporting multi-person tracks and environmental scene footage.
        """
        video_path = Path(video_path)
        all_tracks = self.processor.extract_all_face_tracks(video_path, sample_fps=sample_fps, max_frames=max_frames)
        primary = max(all_tracks, key=lambda t: t.get("face_area", len(t["crops"])))
        duration = primary["duration"]
        fps = primary["fps"]
        total_video_frames = primary["total_video_frames"]

        track_results = []
        for t in all_tracks:
            res_t = self._analyze_track(t)
            res_t["face_area"] = t.get("face_area", 1000.0)
            res_t["is_primary"] = (t["track_id"] == primary["track_id"])
            track_results.append(res_t)

        # Multi-Person Face Aggregation:
        # Check if a target face is definitively deepfaked (targeted face-swap on primary or high-confidence secondary)
        has_targeted_deepfake = any(
            (t["is_primary"] and t["overall_prob"] >= 0.70 and t["ai_content_pct"] >= 40.0) or
            (t["overall_prob"] >= 0.85 and t["freq_prob"] >= 0.55 and t["ai_content_pct"] >= 50.0)
            for t in track_results
        )

        primary_res = next((t for t in track_results if t.get("is_primary")), track_results[0])
        primary_is_real = primary_res["overall_prob"] < 0.40

        if has_targeted_deepfake:
            overall_prob = float(max(t["overall_prob"] for t in track_results))
            ai_content_pct = float(max(t["ai_content_pct"] for t in track_results))
            spatial_prob = float(max(t["spatial_prob"] for t in track_results))
            freq_prob = float(max(t["freq_prob"] for t in track_results))
            temporal_prob = float(max(t["temporal_prob"] for t in track_results))
        else:
            weights = [max(10.0, t["face_area"]) * t["n_frames"] for t in track_results]
            overall_prob = float(np.average([t["overall_prob"] for t in track_results], weights=weights))
            ai_content_pct = float(np.average([t["ai_content_pct"] for t in track_results], weights=weights))
            spatial_prob = float(np.average([t["spatial_prob"] for t in track_results], weights=weights))
            freq_prob = float(np.average([t["freq_prob"] for t in track_results], weights=weights))
            temporal_prob = float(np.average([t["temporal_prob"] for t in track_results], weights=weights))

            # Forensic consensus: if the primary foreground face is authentic real, or >= 50% of tracks are real
            real_count = sum(1 for t in track_results if t["overall_prob"] < 0.50)
            if primary_is_real or (len(track_results) > 1 and real_count >= len(track_results) / 2):
                overall_prob = min(overall_prob, 0.28)
                ai_content_pct = min(ai_content_pct, 10.0)

        real_content_pct = round(100.0 - ai_content_pct, 1)

        all_ai_frames = sum(t["ai_frames_count"] for t in track_results)
        all_total_frames = sum(t["n_frames"] for t in track_results)
        real_frames_count = all_total_frames - all_ai_frames

        fusion_weights = self.model.fusion_weights.detach().cpu().numpy().tolist()

        combined_suspicious = []
        for t in track_results:
            p_name = t["position_name"]
            for seg in t["suspicious_segments"]:
                seg_copy = dict(seg)
                seg_copy["subject"] = p_name
                combined_suspicious.append(seg_copy)

        if not has_targeted_deepfake and len(track_results) > 1 and real_count >= len(track_results) / 2:
            has_extended_manipulation = False
        else:
            has_extended_manipulation = len(combined_suspicious) > 0 and sum(s["end_sec"] - s["start_sec"] for s in combined_suspicious) >= 2.0

        has_detected_faces = any(any(t.get("detected_faces", [False])) for t in all_tracks)

        if not has_detected_faces:
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
            if overall_prob >= 0.50 and (has_targeted_deepfake or ai_content_pct >= 45.0):
                verdict = "AI MANIPULATED (DEEPFAKE)"
                verdict_code = "FAKE"
                confidence_level = "High" if (overall_prob >= 0.65 or ai_content_pct >= 55.0) else "Moderate"
            elif has_extended_manipulation and ai_content_pct >= 20.0:
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

        tracked_subjects_summary = []
        for t in track_results:
            tracked_subjects_summary.append({
                "subject_name": t["position_name"],
                "frames_analyzed": t["n_frames"],
                "ai_percentage": round(t["overall_prob"] * 100.0, 1),
                "ai_content_percentage": t["ai_content_pct"],
                "status": "AI MANIPULATED" if t["overall_prob"] >= 0.50 else "AUTHENTIC REAL"
            })

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

        return {
            "media_type": "video",
            "video_metadata": {
                "file_name": video_path.name,
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
                    "description": "ConvNeXt-Tiny facial boundary & artifact inspection"
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
            "timeline": track_results[0]["timeline"] if track_results else []
        }
