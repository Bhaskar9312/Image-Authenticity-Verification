"""
Robust Video & Face Dataset Loading Pipeline with Dynamic Self-Blended Image (SBI) Augmentation.
Incorporates:
  1. Multi-source video clips: Celeb-DF v2, SAFL (Avatarify + DeepFaceLab), DFDC, and User In-the-Wild footage.
  2. Diverse synthetic face images: SAFL (DeepFaceLab, FaceApp, Dall-E), FFHQ, and StyleGAN.
  3. Dynamic Self-Blended Image (SBI) synthesis on authentic human faces to learn universal blending seams.
  4. Real-world social media compression & sensor degradation with consistent clip-level parameters.
  5. Distance & resolution scaling augmentation on authentic real faces to eliminate resolution bias.
  6. Multi-Person Persistent Face Tracker with IoU & Centroid Proximity matching.
"""

import os
import sys
import glob
import random
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import mediapipe as mp

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
DEFAULT_FRAME_SIZE = 224


def sample_degradation_params(prob=0.70):
    """Samples consistent degradation parameters for an entire video clip."""
    if random.random() > prob:
        return None
    return {
        "scale": random.uniform(0.40, 0.85) if random.random() < 0.60 else 1.0,
        "blur_sigma": random.uniform(0.5, 1.5) if random.random() < 0.40 else 0.0,
        "noise_std": random.uniform(2.0, 6.0) if random.random() < 0.40 else 0.0,
        "gamma": random.uniform(0.80, 1.25) if random.random() < 0.50 else 1.0,
        "jpeg_q": random.randint(18, 75) if random.random() < 0.70 else None
    }


def apply_social_media_degradation(img_np, params=None, prob=0.70):
    """
    Applies real-world compression degradation (WhatsApp / social media / mobile sensor).
    If params is passed, applies that consistent set of transformations.
    """
    if params is None:
        params = sample_degradation_params(prob=prob)
    if params is None:
        return img_np

    h, w = img_np.shape[:2]
    out = img_np.copy()

    # 1. Downscaling & Upscaling (sensor distance / bitrate limitation)
    if params.get("scale", 1.0) < 0.99:
        scale = params["scale"]
        sw, sh = max(32, int(w * scale)), max(32, int(h * scale))
        small = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Gaussian Blur (camera focus / motion blur)
    if params.get("blur_sigma", 0.0) > 0.0:
        out = cv2.GaussianBlur(out, (0, 0), params["blur_sigma"])

    # 3. Additive Sensor Noise
    if params.get("noise_std", 0.0) > 0.0:
        noise = np.random.normal(0, params["noise_std"], out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Gamma Correction
    if abs(params.get("gamma", 1.0) - 1.0) > 0.02:
        gamma = params["gamma"]
        lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        out = cv2.LUT(out, lut)

    # 5. JPEG Compression Blockiness
    if params.get("jpeg_q") is not None:
        q = params["jpeg_q"]
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return out


def generate_sbi_bgr(bgr, transform_params=None):
    """
    Generates a Self-Blended Image (SBI) by blending an altered inner-face region
    onto the authentic outer-face using a Gaussian-feathered mask.
    Forces the network to learn universal blending boundaries, color discrepancies,
    and texture mismatches without acting as a mere resolution shortcut.
    """
    h, w = bgr.shape[:2]
    if transform_params is None:
        cx_r = random.uniform(0.48, 0.52)
        cy_r = random.uniform(0.50, 0.55)
        ax_r = random.uniform(0.24, 0.32)
        ay_r = random.uniform(0.30, 0.40)
        ang = random.uniform(-8, 8)
        ksize = random.choice([11, 15, 21])
        alpha = random.uniform(0.92, 1.08)
        beta = random.uniform(-10, 10)
        scale = random.uniform(0.80, 0.98) if random.random() < 0.5 else 1.0
        dx = random.uniform(-3, 3) if random.random() < 0.5 else 0.0
        dy = random.uniform(-3, 3) if random.random() < 0.5 else 0.0
    else:
        cx_r, cy_r, ax_r, ay_r, ang, ksize, alpha, beta, scale, dx, dy = transform_params

    # 1. Elliptical Face Mask
    mask = np.zeros((h, w), dtype=np.float32)
    center = (int(w * cx_r), int(h * cy_r))
    axes = (max(8, int(w * ax_r)), max(8, int(h * ay_r)))
    cv2.ellipse(mask, center, axes, ang, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask = np.expand_dims(mask, -1) # (H, W, 1)

    # 2. Inner face synthesis perturbation
    inner = bgr.astype(np.float32).copy()
    inner = np.clip(inner * alpha + beta, 0, 255)

    if scale < 0.99:
        sw, sh = max(8, int(w * scale)), max(8, int(h * scale))
        small = cv2.resize(inner, (sw, sh), interpolation=cv2.INTER_LINEAR)
        inner = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    if dx != 0.0 or dy != 0.0:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        inner = cv2.warpAffine(inner, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 3. Alpha Blending
    blended = (bgr.astype(np.float32) * (1.0 - mask) + inner * mask).astype(np.uint8)
    return blended


def calculate_box_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) of two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


class VideoFaceFramesDataset(Dataset):
    """
    PyTorch Dataset yielding (frames_tensor, label, video_id).
    Supports multi-source video clips, SAFL images, HF faces, and on-the-fly SBI.
    """
    def __init__(self, root_dir="data/frames", split="train", num_frames=4,
                 is_training=True, degrade_prob=0.70, frame_size=DEFAULT_FRAME_SIZE,
                 include_hf_images=True, sbi_prob=0.15):
        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.is_training = is_training
        self.degrade_prob = degrade_prob if is_training else 0.0
        self.frame_size = frame_size
        self.sbi_prob = sbi_prob if is_training else 0.0
        self.samples = []

        split_dir = self.root_dir / split
        if split_dir.exists():
            for cls_name, label in [("real", 0.0), ("fake", 1.0)]:
                cls_dir = split_dir / cls_name
                if cls_dir.exists():
                    for clip_folder in cls_dir.iterdir():
                        if clip_folder.is_dir():
                            img_files = sorted(list(clip_folder.glob("*.jpg")) + list(clip_folder.glob("*.png")))
                            if len(img_files) > 0:
                                self.samples.append({
                                    "is_static_image": False,
                                    "folder": clip_folder,
                                    "files": img_files,
                                    "label": label,
                                    "cls_name": cls_name,
                                    "video_id": clip_folder.name
                                })

        # Include diverse photographic and AI synthetic face images in training
        if is_training and include_hf_images:
            # 1. SAFL Diverse Face Images (DeepFaceLab, FaceApp, CelebA)
            safl_real_imgs = sorted(glob.glob("C:/Users/Tankala Bhaskar teja/ISC-VIDEO-KOMAL/dataset/SAFL/images/real/*.jpg"))
            safl_fake_imgs = sorted(glob.glob("C:/Users/Tankala Bhaskar teja/ISC-VIDEO-KOMAL/dataset/SAFL/images/fake/*.jpg")) + \
                             sorted(glob.glob("C:/Users/Tankala Bhaskar teja/ISC-VIDEO-KOMAL/dataset/SAFL/images/fake/*.png"))
            
            random.seed(42)
            sel_safl_r = random.sample(safl_real_imgs, min(1000, len(safl_real_imgs))) if safl_real_imgs else []
            sel_safl_f = random.sample(safl_fake_imgs, min(1000, len(safl_fake_imgs))) if safl_fake_imgs else []

            for p in sel_safl_r:
                self.samples.append({
                    "is_static_image": True,
                    "path": p,
                    "label": 0.0,
                    "cls_name": "real",
                    "video_id": f"safl_{Path(p).stem}"
                })
            for p in sel_safl_f:
                self.samples.append({
                    "is_static_image": True,
                    "path": p,
                    "label": 1.0,
                    "cls_name": "fake",
                    "video_id": f"safl_{Path(p).stem}"
                })

            # 2. Hugging Face Diverse Images (FFHQ & StyleGAN from hf_extracted)
            hf_real_imgs = sorted(glob.glob("C:/Users/Tankala Bhaskar teja/ISC-FINAL-VIDEO/data/hf_extracted/real/*.jpg"))
            hf_fake_imgs = sorted(glob.glob("C:/Users/Tankala Bhaskar teja/ISC-FINAL-VIDEO/data/hf_extracted/fake/*.jpg"))
            sel_hf_r = random.sample(hf_real_imgs, min(1500, len(hf_real_imgs))) if hf_real_imgs else []
            sel_hf_f = random.sample(hf_fake_imgs, min(1500, len(hf_fake_imgs))) if hf_fake_imgs else []

            for p in sel_hf_r:
                self.samples.append({
                    "is_static_image": True,
                    "path": p,
                    "label": 0.0,
                    "cls_name": "real",
                    "video_id": f"hf_{Path(p).stem}"
                })
            for p in sel_hf_f:
                self.samples.append({
                    "is_static_image": True,
                    "path": p,
                    "label": 1.0,
                    "cls_name": "fake",
                    "video_id": f"hf_{Path(p).stem}"
                })

        random.seed()

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, total_frames):
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])
            return indices

        if self.is_training:
            stride = random.choice([1, 2, 3])
            max_start = max(0, total_frames - 1 - (self.num_frames - 1) * stride)
            start = random.randint(0, max_start)
            indices = [min(total_frames - 1, start + i * stride) for i in range(self.num_frames)]
            return sorted(indices)
        else:
            return np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target_label = sample["label"]
        is_static = sample.get("is_static_image", False)

        # Check for Self-Blended Image (SBI) dynamic augmentation on real samples
        apply_sbi = self.is_training and (target_label == 0.0) and (random.random() < self.sbi_prob)
        if apply_sbi:
            target_label = 1.0 # Pseudo-fake with real blending seam

        if is_static:
            frames = self._load_static_image_clip(sample["path"], apply_sbi=apply_sbi)
        else:
            frames = self._load_frames_folder_clip(sample["files"], apply_sbi=apply_sbi)

        label_tensor = torch.tensor(target_label, dtype=torch.float32)
        return frames, label_tensor, sample["video_id"], is_static

    def _load_static_image_clip(self, img_path, apply_sbi=False):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            bgr = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)

        if apply_sbi:
            bgr = generate_sbi_bgr(bgr)

        # Multi-scale distance augmentation applied equally across classes
        if self.is_training and random.random() < 0.60:
            scale_dist = random.uniform(0.30, 1.0)
            h_orig, w_orig = bgr.shape[:2]
            dw, dh = max(32, int(w_orig * scale_dist)), max(32, int(h_orig * scale_dist))
            bgr = cv2.resize(cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA), (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        h, w = bgr.shape[:2]
        # Simulate camera drift and perspective shift over frames
        dx = np.linspace(0, random.uniform(-5, 5), self.num_frames)
        dy = np.linspace(0, random.uniform(-5, 5), self.num_frames)
        dscale = np.linspace(1.0, random.uniform(0.97, 1.03), self.num_frames)

        # Sample consistent degradation params across the entire clip
        clip_deg_params = sample_degradation_params(prob=self.degrade_prob) if self.is_training else None

        frames = []
        apply_hflip = self.is_training and (random.random() < 0.5)

        for i in range(self.num_frames):
            M = np.float32([
                [dscale[i], 0, dx[i] + (1 - dscale[i]) * w / 2],
                [0, dscale[i], dy[i] + (1 - dscale[i]) * h / 2]
            ])
            warped = cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            if clip_deg_params:
                warped = apply_social_media_degradation(warped, params=clip_deg_params)

            warped = cv2.resize(warped, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            if apply_hflip:
                pil_img = TF.hflip(pil_img)

            t_frame = TF.normalize(TF.to_tensor(pil_img), mean=NORM_MEAN, std=NORM_STD)
            frames.append(t_frame)

        return torch.stack(frames, dim=0)

    def _load_frames_folder_clip(self, files, apply_sbi=False):
        indices = self._sample_indices(len(files))
        frames = []
        apply_hflip = self.is_training and (random.random() < 0.5)

        # Consistent SBI params across the 4 frames of this clip
        sbi_params = None
        if apply_sbi:
            sbi_params = (
                random.uniform(0.48, 0.52), random.uniform(0.50, 0.55),
                random.uniform(0.24, 0.32), random.uniform(0.30, 0.40),
                random.uniform(-8, 8), random.choice([11, 15, 21]),
                random.uniform(0.92, 1.08), random.uniform(-10, 10),
                random.uniform(0.80, 0.98) if random.random() < 0.5 else 1.0,
                random.uniform(-3, 3) if random.random() < 0.5 else 0.0,
                random.uniform(-3, 3) if random.random() < 0.5 else 0.0
            )

        # Consistent degradation params and distance scale across the entire clip (equal across classes)
        clip_deg_params = sample_degradation_params(prob=self.degrade_prob) if self.is_training else None
        apply_dist_scale = self.is_training and (random.random() < 0.60)
        dist_scale = random.uniform(0.30, 1.0) if apply_dist_scale else 1.0

        for f_idx in indices:
            img_path = files[f_idx]
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                bgr = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)

            if apply_sbi:
                bgr = generate_sbi_bgr(bgr, transform_params=sbi_params)

            if apply_dist_scale and dist_scale < 0.99:
                h_orig, w_orig = bgr.shape[:2]
                dw, dh = max(32, int(w_orig * dist_scale)), max(32, int(h_orig * dist_scale))
                bgr = cv2.resize(cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA), (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

            if clip_deg_params:
                bgr = apply_social_media_degradation(bgr, params=clip_deg_params)

            bgr = cv2.resize(bgr, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            if apply_hflip:
                pil_img = TF.hflip(pil_img)

            t_frame = TF.normalize(TF.to_tensor(pil_img), mean=NORM_MEAN, std=NORM_STD)
            frames.append(t_frame)

        return torch.stack(frames, dim=0)


class VideoFileProcessor:
    """
    Extracts face-aligned frame sequences from any raw video file (.mp4, .avi, etc.).
    Maintains persistent multi-person face tracking across frames using IoU & centroid proximity.
    Eliminates identity hopping and inter-person coordinate averaging.
    """
    def __init__(self, frame_size=DEFAULT_FRAME_SIZE, face_margin=0.35):
        self.frame_size = frame_size
        self.face_margin = face_margin
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1, # 0 for short-range (<2m), 1 for full-range (<5m)
            min_detection_confidence=0.35
        )

    def detect_faces_multi_orientation(self, rgb_frame):
        """
        Detects faces across multiple orientations (0, 90 CW, 270 CCW, 180).
        Supports subjects lying down on bed/sofa, leaning sideways, or vertical/horizontal phone tilt.
        Returns: (detections, active_rgb, rotation_angle)
        """
        # 1. Standard upright orientation (0 deg)
        res = self.detector.process(rgb_frame)
        if res.detections:
            return res.detections, rgb_frame, 0

        # For rotated orientations (90, 270, 180), require higher confidence (>= 0.55)
        # to prevent inanimate background clutter/textures from triggering phantom detections
        rotations = [
            (90, cv2.ROTATE_90_CLOCKWISE),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
            (180, cv2.ROTATE_180)
        ]
        for angle, rot_code in rotations:
            r_img = cv2.rotate(rgb_frame, rot_code)
            res_r = self.detector.process(r_img)
            if res_r.detections:
                high_conf_dets = [d for d in res_r.detections if d.score[0] >= 0.55]
                if high_conf_dets:
                    return high_conf_dets, r_img, angle

        return [], rgb_frame, 0

    def extract_all_face_tracks(self, video_path, sample_fps=2.0, max_frames=64, min_track_len=2):
        """
        Scans video and tracks all distinct faces into coherent temporal trajectories.
        Features multi-orientation face detection and gap imputation.
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        duration = total_frames / max(1.0, video_fps)
        step = max(1, int(round(video_fps / sample_fps)))

        tracks = {} # track_id: {'crops': [], 'timestamps': [], 'boxes': [], 'last_box': box, 'gap': 0}
        next_track_id = 0
        frame_idx = 0
        sampled_count = 0

        while cap.isOpened() and sampled_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                dets, active_rgb, rot = self.detect_faces_multi_orientation(raw_rgb)
                h, w = active_rgb.shape[:2]

                curr_detections = []
                for d in dets:
                    bb = d.location_data.relative_bounding_box
                    bx = max(0, int(bb.xmin * w))
                    by = max(0, int(bb.ymin * h))
                    bw = int(bb.width * w)
                    bh = int(bb.height * h)

                    mx = int(bw * self.face_margin)
                    my = int(bh * self.face_margin)
                    x1 = max(0, bx - mx)
                    y1 = max(0, by - my)
                    x2 = min(w, bx + bw + mx)
                    y2 = min(h, by + bh + my)
                    curr_detections.append([x1, y1, x2, y2])

                t_sec = frame_idx / video_fps
                matched_tracks = set()
                matched_dets = set()

                # Match detections to active tracks
                for t_id, t_info in tracks.items():
                    last_box = t_info["last_box"]
                    best_iou, best_d = 0.0, -1
                    for di, d_box in enumerate(curr_detections):
                        if di in matched_dets:
                            continue
                        iou = calculate_box_iou(last_box, d_box)
                        cd = abs((last_box[0] + last_box[2]) / 2.0 - (d_box[0] + d_box[2]) / 2.0) / float(w)
                        if iou > 0.15 or (cd < 0.20 and iou > 0.03):
                            if iou > best_iou:
                                best_iou = iou
                                best_d = di

                    if best_d >= 0:
                        matched_tracks.add(t_id)
                        matched_dets.add(best_d)
                        box = curr_detections[best_d]
                        smoothed = [int(0.80 * box[i] + 0.20 * last_box[i]) for i in range(4)]
                        t_info["last_box"] = smoothed
                        t_info["boxes"].append(smoothed)
                        t_info["timestamps"].append(t_sec)
                        t_info["gap"] = 0
                        t_info["actual_detections"] = t_info.get("actual_detections", 0) + 1

                        crop = active_rgb[smoothed[1]:smoothed[3], smoothed[0]:smoothed[2]]
                        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                            crop = cv2.resize(active_rgb, (self.frame_size, self.frame_size))
                        else:
                            crop = cv2.resize(crop, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
                        t_info["crops"].append(Image.fromarray(crop))
                    else:
                        # Carry forward previous bounding box if temporary occlusion (up to 2 missed frames)
                        # ONLY for established tracks with at least 2 actual face detections
                        if t_info.get("actual_detections", 0) >= 2 and t_info.get("gap", 0) < 2 and len(t_info["crops"]) > 0:
                            t_info["gap"] = t_info.get("gap", 0) + 1
                            t_info["timestamps"].append(t_sec)
                            t_info["crops"].append(t_info["crops"][-1])

                # New tracks for unmatched detections
                for di, d_box in enumerate(curr_detections):
                    if di not in matched_dets and next_track_id < 4:
                        crop = active_rgb[d_box[1]:d_box[3], d_box[0]:d_box[2]]
                        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                            crop = cv2.resize(active_rgb, (self.frame_size, self.frame_size))
                        else:
                            crop = cv2.resize(crop, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)

                        tracks[next_track_id] = {
                            "track_id": next_track_id,
                            "crops": [Image.fromarray(crop)],
                            "timestamps": [t_sec],
                            "boxes": [d_box],
                            "last_box": d_box,
                            "gap": 0,
                            "actual_detections": 1
                        }
                        next_track_id += 1

                sampled_count += 1

            frame_idx += 1

        cap.release()

        # Build output structure for valid tracks
        # A valid tracked face must have at least 3 genuine detections across time (or 2 for very short clips),
        # ensuring single-frame background clutter or inanimate objects don't trigger false tracks.
        required_detections = 2 if sampled_count <= 4 else 3
        valid_tracks = []
        for t_id, t_info in tracks.items():
            if t_info.get("actual_detections", 0) >= required_detections and len(t_info["crops"]) >= required_detections:
                avg_cx = np.mean([(b[0] + b[2]) / 2.0 for b in t_info["boxes"]])
                pos_label = "Left" if avg_cx < 0.40 * w else ("Right" if avg_cx > 0.60 * w else "Center")

                t_tensors = [TF.normalize(TF.to_tensor(c), mean=NORM_MEAN, std=NORM_STD) for c in t_info["crops"]]
                tensor = torch.stack(t_tensors, dim=0).unsqueeze(0) # (1, N, 3, H, W)

                valid_tracks.append({
                    "track_id": t_id,
                    "position_name": f"Person ({pos_label})",
                    "tensor": tensor,
                    "crops": t_info["crops"],
                    "timestamps": t_info["timestamps"],
                    "detected_faces": [True] * len(t_info["crops"]),
                    "fps": video_fps,
                    "duration": duration,
                    "total_video_frames": total_frames
                })

        # Fallback if no faces detected throughout entire video: extract center crops from actual video frames
        if not valid_tracks:
            cap_fb = cv2.VideoCapture(video_path)
            center_crops = []
            c_times = []
            f_i = 0
            while cap_fb.isOpened() and len(center_crops) < min(16, max_frames):
                ret, frame = cap_fb.read()
                if not ret:
                    break
                if f_i % step == 0:
                    h_fb, w_fb = frame.shape[:2]
                    min_dim = min(h_fb, w_fb)
                    cx_fb, cy_fb = w_fb // 2, h_fb // 2
                    half_fb = min_dim // 2
                    c_bgr = frame[max(0, cy_fb - half_fb):min(h_fb, cy_fb + half_fb),
                                  max(0, cx_fb - half_fb):min(w_fb, cx_fb + half_fb)]
                    c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
                    c_resized = cv2.resize(c_rgb, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
                    center_crops.append(Image.fromarray(c_resized))
                    c_times.append(f_i / video_fps)
                f_i += 1
            cap_fb.release()

            if not center_crops:
                blank = Image.new("RGB", (self.frame_size, self.frame_size), (128, 128, 128))
                center_crops = [blank]
                c_times = [0.0]

            t_tensors = [TF.normalize(TF.to_tensor(c), mean=NORM_MEAN, std=NORM_STD) for c in center_crops]
            tensor = torch.stack(t_tensors, dim=0).unsqueeze(0)
            valid_tracks.append({
                "track_id": 0,
                "position_name": "Environmental Scene (No Face Detected)",
                "tensor": tensor,
                "crops": center_crops,
                "timestamps": c_times,
                "detected_faces": [False] * len(center_crops),
                "fps": video_fps,
                "duration": duration,
                "total_video_frames": total_frames
            })

        return valid_tracks

    def process_video(self, video_path, sample_fps=2.0, max_frames=64):
        """
        Processes video and returns the primary track while attaching all tracked individuals.
        Guarantees 100% backward-compatibility.
        """
        tracks = self.extract_all_face_tracks(video_path, sample_fps=sample_fps, max_frames=max_frames)
        # Select longest track as primary
        primary = max(tracks, key=lambda t: len(t["crops"]))
        primary["all_tracks"] = tracks
        return primary
