import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyze_video import VideoForgeryAnalyzer

analyzer = VideoForgeryAnalyzer(checkpoint_path="checkpoints/best_forgery_detector.pt")

videos = [
    ("1. Real WhatsApp Single Person", "C:/Users/Tankala Bhaskar teja/.gemini/antigravity/brain/9c73e2b9-15c4-464f-a65a-a31a18216af1/.user_uploaded/media_1794603757231_c759e3b0.mp4"),
    ("2. Real Multi-Person Reel (Preetham Reel)", "C:/Users/Tankala Bhaskar teja/.gemini/antigravity/brain/9c73e2b9-15c4-464f-a65a-a31a18216af1/.user_uploaded/media_1794645078858_6560e1bd.mp4"),
    ("3. Real Sideways Lying on Bed (Preetham Bed)", "C:/Users/Tankala Bhaskar teja/Videos/preetham.mp4"),
    ("4. Real Room Walk (Environmental Scene)", "C:/Users/Tankala Bhaskar teja/.gemini/antigravity/brain/9c73e2b9-15c4-464f-a65a-a31a18216af1/.user_uploaded/media_1794650250828_6b5a0710.mp4"),
    ("5. AI Deepfake Tom Cruise", "C:/Users/Tankala Bhaskar teja/.gemini/antigravity/brain/9c73e2b9-15c4-464f-a65a-a31a18216af1/.user_uploaded/media_1794625472789_e612d1cc.mp4"),
    ("6. AI Deepfake Morgan Freeman", "C:/Users/Tankala Bhaskar teja/.gemini/antigravity/brain/9c73e2b9-15c4-464f-a65a-a31a18216af1/.user_uploaded/media_1794640408432_19e2b880.mp4"),
    ("7. AI Deepfake Joe Biden", "C:/Users/Tankala Bhaskar teja/.gemini/antigravity/brain/9c73e2b9-15c4-464f-a65a-a31a18216af1/.user_uploaded/media_1794640423678_8e325d48.mp4")
]

results = []
for name, path in videos:
    print("=" * 65)
    print(f"ANALYZING: {name}")
    rep = analyzer.analyze(path, sample_fps=2.0, max_frames=64)
    v = rep["verdict"]
    c = rep["content_composition"]
    m = rep["video_metadata"]
    
    print(f"Decision   : {v['overall_decision']} (Confidence: {v['confidence_level']})")
    print(f"Probability: AI = {v['overall_ai_percentage']}%, Real = {v['overall_real_percentage']}%")
    print(f"Composition: AI Content = {c['ai_content_percentage']}%, Real Content = {c['real_content_percentage']}%")
    print(f"Subjects   : {m.get('distinct_faces_tracked', 1)} tracked")
    
    if rep.get("tracked_subjects"):
        for s in rep["tracked_subjects"]:
            print(f"   -> {s['subject_name']}: {s['status']} (AI Prob: {s['ai_percentage']}%, AI Content: {s['ai_content_percentage']}%)")
            
    results.append({
        "name": name,
        "verdict": v["overall_decision"],
        "confidence": v["confidence_level"],
        "ai_prob": v["overall_ai_percentage"],
        "ai_content": c["ai_content_percentage"],
        "real_content": c["real_content_percentage"],
        "subjects": rep.get("tracked_subjects", [])
    })

print("\n" + "=" * 65)
print("FINAL AUDIT SUMMARY ACROSS ALL 5 CHALLENGE VIDEOS:")
print("=" * 65)
for r in results:
    status_icon = "[PASS]" if (("Real" in r["name"] and "AUTHENTIC" in r["verdict"]) or ("AI" in r["name"] and "AI MANIPULATED" in r["verdict"])) else "[FAIL]"
    print(f"{status_icon} {r['name']}:")
    print(f"       Verdict: {r['verdict']} | AI Prob: {r['ai_prob']}% | AI Content: {r['ai_content']}%")

with open("evaluation_results/challenge_videos_verification.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved full verification results to evaluation_results/challenge_videos_verification.json")
