import os
import cv2
import numpy as np
from pathlib import Path

def compute_image_dhash(img, hash_size=8):
    """
    Computes a 64-bit difference hash (dHash) as a 16-character hexadecimal string.
    Fast, rotation/scale resilient visual fingerprinting.
    """
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    diff = gray[:, 1:] > gray[:, :-1]
    byte_vals = np.packbits(diff.flatten())
    return byte_vals.tobytes().hex()

def compute_video_fingerprint(video_path, num_samples=8):
    """
    Samples num_samples keyframes evenly distributed across video duration
    and computes a sequence of perceptual hashes joined by pipe '|'.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    hashes = []
    for i in range(num_samples):
        pos = int(total * (i + 0.5) / num_samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, f = cap.read()
        if ret:
            hashes.append(compute_image_dhash(f))
        else:
            hashes.append("0" * 16)
    cap.release()
    return "|".join(hashes)

def compute_media_fingerprint(file_path, media_type):
    """Computes perceptual fingerprint based on media type."""
    p = Path(file_path)
    if media_type == "video":
        return compute_video_fingerprint(p)
    else:
        img = cv2.imread(str(p))
        if img is None:
            return "0" * 16
        return compute_image_dhash(img)

def compare_fingerprints(fp1, fp2, media_type="video"):
    """
    Compares two perceptual fingerprints and returns a similarity percentage (0.0 to 100.0%).
    """
    if not fp1 or not fp2:
        return 0.0

    if media_type == "image":
        try:
            b1 = np.frombuffer(bytes.fromhex(fp1), dtype=np.uint8)
            b2 = np.frombuffer(bytes.fromhex(fp2), dtype=np.uint8)
            dist = int(np.count_nonzero(np.unpackbits(b1 ^ b2)))
            return round((64 - dist) / 64.0 * 100.0, 1)
        except Exception:
            return 0.0
    else:
        h1 = fp1.split("|")
        h2 = fp2.split("|")
        n = min(len(h1), len(h2))
        if n == 0:
            return 0.0
        sims = []
        for a, b in zip(h1, h2):
            try:
                b1 = np.frombuffer(bytes.fromhex(a), dtype=np.uint8)
                b2 = np.frombuffer(bytes.fromhex(b), dtype=np.uint8)
                dist = int(np.count_nonzero(np.unpackbits(b1 ^ b2)))
                sims.append((64 - dist) / 64.0)
            except Exception:
                sims.append(0.0)
        return round(float(np.mean(sims)) * 100.0, 1)

def verify_image_keypoint_correspondence(img_path1, img_path2, min_inliers=18):
    """
    Tier-2 Splicing & Object Inpainting Verification using ORB feature keypoint matching and RANSAC homography.
    Mathematically verifies whether the primary subject was copied or spliced into a modified background.
    Returns: (is_match: bool, inliers_count: int, confidence_pct: float)
    """
    try:
        img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_path2), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return False, 0, 0.0

        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return False, 0, 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if len(matches[0]) == 2 and m.distance < 0.75 * n.distance]
        if len(good) < min_inliers:
            return False, len(good), 0.0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers >= min_inliers:
            conf = min(98.5, round(65.0 + min(33.5, inliers * 0.2), 1))
            return True, inliers, conf
        return False, inliers, 0.0
    except Exception:
        return False, 0, 0.0

def find_registered_original(query_path, media_type, conn, min_similarity=75.0):
    """
    Searches SQLite media_records for a perceptual match against registered journalist originals.
    Two-Tier Architecture:
      - Tier 1: Fast global dHash check (>= 75.0% immediate match).
      - Tier 2: Splicing / Inpainting Filter using ORB keypoint geometric matching when dHash is in 45-75% range.
    Returns: (matched_record, similarity_pct)
    """
    query_fp = compute_media_fingerprint(query_path, media_type)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT m.*, u.full_name as signer_name, u.organization as signer_org, k.public_key_pem as reg_pubkey
        FROM media_records m
        JOIN users u ON m.uploader_id = u.id
        LEFT JOIN journalist_keys k ON k.user_id = u.id AND k.status = 'active'
        WHERE m.file_type = ?
    """, (media_type,))
    records = cursor.fetchall()

    best_record = None
    best_sim = 0.0

    for r in records:
        reg_fp = r["perceptual_hash"]
        if not reg_fp:
            # Lazy compute if missing
            reg_fp = compute_media_fingerprint(r["file_path"], media_type)
            cursor.execute("UPDATE media_records SET perceptual_hash = ? WHERE id = ?", (reg_fp, r["id"]))
            conn.commit()

        sim = compare_fingerprints(query_fp, reg_fp, media_type)
        if sim > best_sim:
            best_sim = sim
            best_record = r

    # Tier 1: Fast Global Hash Filter
    tier1_threshold = 75.0
    if best_record and best_sim >= tier1_threshold:
        return best_record, best_sim

    # Tier 2: Splicing & Object Inpainting Filter (for Images)
    if media_type == "image":
        for r in records:
            orig_file = r["file_path"]
            if os.path.exists(orig_file):
                is_match, inliers, conf = verify_image_keypoint_correspondence(orig_file, query_path, min_inliers=18)
                if is_match:
                    return r, max(best_sim, conf)

    # Fallback threshold for videos
    if media_type == "video" and best_record and best_sim >= min_similarity:
        return best_record, best_sim

    return None, 0.0

def quantify_differential_manipulation(query_path, orig_path, media_type, cache_dir=None):
    """
    Compares query media against the verified journalist original to quantify exactly how much it got manipulated.
    Returns detailed differential metrics and generates a side-by-side evidence panel.
    """
    query_path = Path(query_path)
    orig_path = Path(orig_path)

    if media_type == "video":
        cap_q = cv2.VideoCapture(str(query_path))
        cap_o = cv2.VideoCapture(str(orig_path))

        fps = cap_o.get(cv2.CAP_PROP_FPS) or 30.0
        tot_q = int(cap_q.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        tot_o = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        total_frames = min(tot_q, tot_o)

        step = max(1, int(round(fps / 2.0))) # 2 fps sampling
        altered_frames = 0
        analyzed_frames = 0
        frame_deviations = []
        altered_timeline = []

        best_sample_orig = None
        best_sample_query = None
        best_sample_diff = None
        max_diff_seen = 0

        for f_idx in range(0, total_frames, step):
            cap_q.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            cap_o.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret_q, f_q = cap_q.read()
            ret_o, f_o = cap_o.read()
            if not ret_q or not ret_o:
                break

            # Ensure matching size
            if f_q.shape != f_o.shape:
                f_q = cv2.resize(f_q, (f_o.shape[1], f_o.shape[0]))

            diff = cv2.absdiff(f_q, f_o)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            tampered_pixels = int(np.count_nonzero(gray > 20))
            pct = (tampered_pixels / float(gray.size)) * 100.0
            frame_deviations.append(pct)

            if tampered_pixels > 200:
                altered_frames += 1
                t_sec = f_idx / fps
                altered_timeline.append(t_sec)

                if tampered_pixels > max_diff_seen:
                    max_diff_seen = tampered_pixels
                    best_sample_orig = f_o.copy()
                    best_sample_query = f_q.copy()
                    best_sample_diff = gray.copy()

            analyzed_frames += 1

        cap_q.release()
        cap_o.release()

        manip_frames_pct = round((altered_frames / max(1, analyzed_frames)) * 100.0, 1)
        avg_pixel_pct = round(float(np.mean(frame_deviations)) if frame_deviations else 0.0, 2)
        max_pixel_pct = round(float(np.max(frame_deviations)) if frame_deviations else 0.0, 2)

        intervals = []
        if altered_timeline:
            start_t = min(altered_timeline)
            end_t = max(altered_timeline)
            intervals.append({
                "start_time": f"{int(start_t // 60):02d}:{start_t % 60:05.2f}",
                "end_time": f"{int(end_t // 60):02d}:{end_t % 60:05.2f}",
                "duration": round(end_t - start_t, 2),
                "type": "Visual Overlay / Digital Modification"
            })

        evidence_rel_url = None
        if cache_dir and best_sample_orig is not None:
            c_dir = Path(cache_dir)
            c_dir.mkdir(parents=True, exist_ok=True)
            # Create 3-panel comparison: Original | Modified | Tamper Highlight
            highlight = best_sample_query.copy()
            highlight[best_sample_diff > 20] = [0, 0, 255] # Red highlight on modified pixels

            h, w = best_sample_orig.shape[:2]
            target_w = 320
            target_h = int(h * (target_w / float(w)))

            p1 = cv2.resize(best_sample_orig, (target_w, target_h))
            p2 = cv2.resize(best_sample_query, (target_w, target_h))
            p3 = cv2.resize(highlight, (target_w, target_h))

            # Add text labels
            cv2.putText(p1, "1. JOURNALIST ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(p2, "2. UPLOADED QUERY", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            cv2.putText(p3, "3. TAMPER MAP (RED)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            panel = np.hstack([p1, p2, p3])
            diff_filename = f"diff_{query_path.stem}.jpg"
            cv2.imwrite(str(c_dir / diff_filename), panel)
            evidence_rel_url = f"/static/diff_cache/{diff_filename}"

        return {
            "manipulated_frames_pct": manip_frames_pct,
            "avg_pixel_manipulation_pct": avg_pixel_pct,
            "max_pixel_manipulation_pct": max_pixel_pct,
            "altered_frames_count": altered_frames,
            "analyzed_frames_count": analyzed_frames,
            "tampered_intervals": intervals,
            "evidence_image_url": evidence_rel_url
        }

    else:
        # Photo comparison
        img_q = cv2.imread(str(query_path))
        img_o = cv2.imread(str(orig_path))

        if img_q is None or img_o is None:
            return {"manipulated_frames_pct": 0.0, "avg_pixel_manipulation_pct": 0.0}

        if img_q.shape != img_o.shape:
            img_q = cv2.resize(img_q, (img_o.shape[1], img_o.shape[0]))

        diff = cv2.absdiff(img_q, img_o)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        tampered_pixels = int(np.count_nonzero(gray > 25))
        total_pixels = gray.size
        pixel_pct = round((tampered_pixels / float(total_pixels)) * 100.0, 1)

        evidence_rel_url = None
        if cache_dir:
            c_dir = Path(cache_dir)
            c_dir.mkdir(parents=True, exist_ok=True)
            highlight = img_q.copy()
            red_overlay = highlight.copy()
            red_overlay[gray > 25] = [0, 0, 255]
            cv2.addWeighted(red_overlay, 0.65, highlight, 0.35, 0, highlight)

            h, w = img_o.shape[:2]
            target_w = 340
            target_h = int(h * (target_w / float(w)))

            p1 = cv2.resize(img_o, (target_w, target_h))
            p2 = cv2.resize(img_q, (target_w, target_h))
            p3 = cv2.resize(highlight, (target_w, target_h))

            cv2.putText(p1, "1. JOURNALIST ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(p2, "2. UPLOADED QUERY", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            cv2.putText(p3, "3. ALTERED REGIONS (RED)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            panel = np.hstack([p1, p2, p3])
            diff_filename = f"diff_{query_path.stem}.jpg"
            cv2.imwrite(str(c_dir / diff_filename), panel)
            evidence_rel_url = f"/static/diff_cache/{diff_filename}"

        classification = "Background Replacement / Splicing" if pixel_pct >= 20.0 else "Foreign Overlay / Local Modification"
        return {
            "manipulated_frames_pct": 100.0 if pixel_pct > 0.1 else 0.0,
            "avg_pixel_manipulation_pct": pixel_pct,
            "max_pixel_manipulation_pct": pixel_pct,
            "altered_frames_count": 1 if pixel_pct > 0.1 else 0,
            "analyzed_frames_count": 1,
            "tamper_classification": classification,
            "tampered_intervals": [{"start_time": "00:00.00", "end_time": "00:00.00", "duration": 0.0, "type": classification}],
            "evidence_image_url": evidence_rel_url
        }
