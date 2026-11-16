import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import mediapipe as mp
from pathlib import Path

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def calculate_box_iou(box1, box2):
    """Computes IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

class MultiOrientationVideoProcessor:
    def __init__(self, frame_size=224, face_margin=0.25):
        self.frame_size = frame_size
        self.face_margin = face_margin
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.35
        )

    def detect_faces_multi_orientation(self, rgb_frame):
        """
        Detects faces across multiple orientations (0, 90 CW, 270 CCW, 180).
        Automatically rectifies rotated faces to upright geometry.
        """
        # 1. Standard upright orientation (0 deg)
        res = self.detector.process(rgb_frame)
        if res.detections:
            return res.detections, rgb_frame, 0

        # For rotated orientations, require confidence >= 0.55 to avoid background clutter
        rotations = [
            (90, cv2.ROTATE_90_CLOCKWISE),
            (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
            (180, cv2.ROTATE_180)
        ]
        for angle, rot_code in rotations:
            r_img = cv2.rotate(rgb_frame, rot_code)
            res_r = self.detector.process(r_img)
            if res_r.detections:
                high_conf = [d for d in res_r.detections if d.score[0] >= 0.55]
                if high_conf:
                    return high_conf, r_img, angle

        return [], rgb_frame, 0

    def extract_all_face_tracks(self, video_path, sample_fps=2.0, max_frames=64):
        """
        Extracts temporal face tracks with multi-orientation detection and background clutter rejection.
        If no faces exist, falls back to environmental scene center crops.
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        duration = total_frames / max(1.0, video_fps)
        step = max(1, int(round(video_fps / sample_fps)))

        tracks = {}
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
                        if t_info.get("actual_detections", 0) >= 2 and t_info.get("gap", 0) < 2 and len(t_info["crops"]) > 0:
                            t_info["gap"] = t_info.get("gap", 0) + 1
                            t_info["timestamps"].append(t_sec)
                            t_info["crops"].append(t_info["crops"][-1])

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

        required_detections = 2 if sampled_count <= 4 else 3
        valid_tracks = []
        for t_id, t_info in tracks.items():
            if t_info.get("actual_detections", 0) >= required_detections and len(t_info["crops"]) >= required_detections:
                avg_cx = np.mean([(b[0] + b[2]) / 2.0 for b in t_info["boxes"]])
                avg_w = float(np.mean([b[2] - b[0] for b in t_info["boxes"]]))
                avg_h = float(np.mean([b[3] - b[1] for b in t_info["boxes"]]))
                avg_area = max(100.0, avg_w * avg_h)
                pos_label = "Left" if avg_cx < 0.40 * w else ("Right" if avg_cx > 0.60 * w else "Center")
                t_tensors = [TF.normalize(TF.to_tensor(c), mean=NORM_MEAN, std=NORM_STD) for c in t_info["crops"]]
                tensor = torch.stack(t_tensors, dim=0).unsqueeze(0)

                valid_tracks.append({
                    "track_id": t_id,
                    "position_name": f"Person ({pos_label})",
                    "tensor": tensor,
                    "crops": t_info["crops"],
                    "timestamps": t_info["timestamps"],
                    "detected_faces": [True] * len(t_info["crops"]),
                    "face_area": avg_area,
                    "face_width": avg_w,
                    "face_height": avg_h,
                    "fps": video_fps,
                    "duration": duration,
                    "total_video_frames": total_frames
                })

        # Fallback if no faces detected throughout entire video: scene center crops
        if not valid_tracks:
            cap_fb = cv2.VideoCapture(video_path)
            center_crops = []
            c_times = []
            f_i = 0
            while cap_fb.isOpened() and len(center_crops) < min(16, max_frames):
                ret_fb, f_fb = cap_fb.read()
                if not ret_fb:
                    break
                if f_i % step == 0:
                    h_fb, w_fb = f_fb.shape[:2]
                    min_dim = min(h_fb, w_fb)
                    cx_fb, cy_fb = w_fb // 2, h_fb // 2
                    half_fb = min_dim // 2
                    c_bgr = f_fb[max(0, cy_fb - half_fb):min(h_fb, cy_fb + half_fb),
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
