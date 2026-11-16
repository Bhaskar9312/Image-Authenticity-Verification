"""
Quick Demonstration Script for AI Video Forgery Detection.
Runs the analysis engine on sample videos in data/uploads/ and prints full reports.
"""

import sys
from pathlib import Path
from analyze_video import VideoForgeryAnalyzer

def main():
    uploads_dir = Path("data/uploads")
    if not uploads_dir.exists():
        print(f"Error: {uploads_dir} does not exist.")
        return

    sample_videos = list(uploads_dir.glob("*.mp4"))
    if not sample_videos:
        print("No .mp4 videos found in data/uploads.")
        return

    # Choose specific video if supplied in args, else pick first sample
    if len(sys.argv) > 1:
        target_video = Path(sys.argv[1])
    else:
        target_video = sample_videos[0]

    print(f"\n==================================================")
    print(f" Running AI Video Forgery Detection Demo")
    print(f" Video: {target_video.name}")
    print(f"==================================================")

    # Check for trained checkpoint
    checkpoint = "checkpoints/best_forgery_detector.pt"
    if not Path(checkpoint).exists():
        fallback = "C:/Users/Tankala Bhaskar teja/ISC-VIDEO-KOMAL/isc_ensemble_improved_output/checkpoints/full_ensemble.pt"
        if Path(fallback).exists():
            checkpoint = fallback
        else:
            checkpoint = None

    analyzer = VideoForgeryAnalyzer(checkpoint_path=checkpoint)
    report = analyzer.analyze(
        video_path=target_video,
        sample_fps=2.0,
        max_frames=64,
        window_size=4,
        save_evidence_dir="evidence_output"
    )

    analyzer.print_terminal_report(report)

if __name__ == "__main__":
    main()
