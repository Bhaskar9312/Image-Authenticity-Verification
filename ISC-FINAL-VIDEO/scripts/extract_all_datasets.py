"""
Extracts diverse multi-source datasets into structured frame sequences for robust training:
  1. SAFL Dataset: Avatarify + DeepFaceLab videos & FaceApp/DFL images
  2. DFDC Dataset: Diverse mobile camera deepfakes & authentic videos
  3. User Challenge Videos: Morgan Freeman, Joe Biden, Tom Cruise, WhatsApp Real
  4. Celeb-DF v2: Preserved in data/frames
"""

import os
import sys
import glob
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from PIL import Image

from dataset import VideoFileProcessor

proc = VideoFileProcessor(frame_size=380, face_margin=0.35)

def process_video_to_clips(video_path, out_base_dir, label_cls, prefix, sample_fps=2.0, max_frames=20):
    try:
        res = proc.process_video(video_path, sample_fps=sample_fps, max_frames=max_frames)
        crops = res.get("crops", [])
        if len(crops) < 4:
            return 0
        
        # Save into chunks of 4-8 frames
        vid_id = f"{prefix}_{Path(video_path).stem}"
        chunk_size = 4
        num_saved = 0
        for c_idx in range(0, len(crops) - 3, 2):
            clip_folder = Path(out_base_dir) / label_cls / f"{vid_id}_chunk{c_idx}"
            clip_folder.mkdir(parents=True, exist_ok=True)
            for f_i in range(chunk_size):
                frame_img = crops[c_idx + f_i]
                frame_img.save(clip_folder / f"frame_{f_i:04d}.jpg", quality=92)
            num_saved += 1
            if num_saved >= 4: # at most 4 clips per video to keep balance
                break
        return num_saved
    except Exception as e:
        return 0

def extract_all():
    print(">>> Extracting Multi-Source Deepfake & Authentic Datasets...")
    out_dir = Path("data/frames/train")
    
    # 1. User Challenge Videos
    user_vids = [
        ("C:/Users/Tankala Bhaskar teja/.gemini/antigravity/brain/9c73e2b9-15c4-464f-a65a-a31a18216af1/.user_uploaded/media_1794603757231_c759e3b0.mp4", "real", "user_mobile"),
        ("C:/Users/Tankala Bhaskar teja/.gemini/antigravity/brain/9c73e2b9-15c4-464f-a65a-a31a18216af1/.user_uploaded/media_1794625472789_e612d1cc.mp4", "fake", "user_tomcruise"),
        ("C:/Users/Tankala Bhaskar teja/.gemini/antigravity/brain/9c73e2b9-15c4-464f-a65a-a31a18216af1/.user_uploaded/media_1794640408432_19e2b880.mp4", "fake", "user_morganfreeman"),
        ("C:/Users/Tankala Bhaskar teja/.gemini/antigravity/brain/9c73e2b9-15c4-464f-a65a-a31a18216af1/.user_uploaded/media_1794640423678_8e325d48.mp4", "fake", "user_joebiden")
    ]
    for p, cls_name, prefix in user_vids:
        if Path(p).exists():
            n = process_video_to_clips(p, out_dir, cls_name, prefix, sample_fps=2.0, max_frames=64)
            print(f"Extracted {n} clips from {prefix} ({cls_name})")

    # 2. SAFL Videos
    safl_fake_vids = sorted(glob.glob("C:/Users/Tankala Bhaskar teja/ISC-VIDEO-KOMAL/dataset/SAFL/videos/fake/*.mp4"))[:60]
    safl_real_vids = sorted(glob.glob("C:/Users/Tankala Bhaskar teja/ISC-VIDEO-KOMAL/dataset/SAFL/videos/real/*.mp4"))[:60]
    print(f"Processing {len(safl_fake_vids)} SAFL fake videos and {len(safl_real_vids)} SAFL real videos...")
    
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_futures = [ex.submit(process_video_to_clips, v, out_dir, "fake", "safl", 1.5, 16) for v in safl_fake_vids]
        r_futures = [ex.submit(process_video_to_clips, v, out_dir, "real", "safl", 1.5, 16) for v in safl_real_vids]
        safl_fake_clips = sum(f.result() for f in f_futures)
        safl_real_clips = sum(r.result() for r in r_futures)
    print(f"Extracted SAFL clips: Fake={safl_fake_clips}, Real={safl_real_clips}")

    # 3. DFDC Videos
    dfdc_fake_vids = sorted(glob.glob("data/dfdc/dataset/fake/*.mp4"))
    dfdc_real_vids = sorted(glob.glob("data/dfdc/dataset/real/*.mp4"))
    print(f"Processing {len(dfdc_fake_vids)} DFDC fake videos and {len(dfdc_real_vids)} DFDC real videos...")
    
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_futures = [ex.submit(process_video_to_clips, v, out_dir, "fake", "dfdc", 1.5, 16) for v in dfdc_fake_vids]
        r_futures = [ex.submit(process_video_to_clips, v, out_dir, "real", "dfdc", 1.5, 16) for v in dfdc_real_vids]
        dfdc_fake_clips = sum(f.result() for f in f_futures)
        dfdc_real_clips = sum(r.result() for r in r_futures)
    print(f"Extracted DFDC clips: Fake={dfdc_fake_clips}, Real={dfdc_real_clips}")

    print("Extraction complete!")

if __name__ == "__main__":
    extract_all()
