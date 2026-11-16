# AI-Based Video Forgery Detection & Forensic Content Analyzer (VeriVid AI)

An end-to-end deep learning framework designed to detect AI-generated video forgeries, deepfakes, and facial manipulations, with fine-grained temporal content quantification (**percentage of AI vs Real content**) and second-by-second timeline reconstruction.

---

## 🎯 Problem Statement & Core Capabilities

Traditional deepfake classifiers only output a single global label (`REAL` vs `FAKE`), failing when:
1. **Videos are partially manipulated**: A video may be 90% authentic footage with only a 5-second deepfake face insertion.
2. **Compressions mask visual artifacts**: Artifacts in RGB pixel space get smoothed out by video encoders (H.264 / H.265).
3. **Temporal inconsistency**: Single-frame detectors miss unnatural inter-frame flickering, blinking anomalies, and facial boundary jitters.

### What This System Delivers:
- **Binary Verdict & Confidence Score**: Determines whether a video is an AI-manipulated forgery or authentic real video.
- **Content Composition Breakdown**: Directly calculates **how much content is AI (%) and how much is Real (%)**.
- **Second-by-Second Forgery Timeline**: Pinpoints exact timestamps where AI manipulation occurs.
- **Three-Branch Forensic Multi-Modal Ensemble**:
  1. **Spatial Branch (`EfficientNet-B4`)**: Detects high-resolution visual manipulation artifacts, blending boundaries, skin texture distortions, and resolution mismatches.
  2. **Spectral Frequency Branch (`8x8 Block DCT CNN`)**: Applies an orthonormal 2D Discrete Cosine Transform across RGB channels to identify checkerboard upsampling artifacts and frequency anomalies characteristic of generative AI (GANs, Diffusion, FaceSwap).
  3. **Temporal Branch (`BiGRU + Temporal Attention`)**: Models inter-frame motion coherence and feature differentials (`feats[t] - feats[t-1]`) to catch temporal jitter and face warp flicker across consecutive frames.
  4. **Learned Dynamic Fusion**: Combines branch predictions with learnable weights and temperature scaling.
- **Interactive Web Interface & CLI Tools**: Full web application with interactive timeline charts, evidence keyframe galleries, and split meters.

---

## 🏗️ Model Architecture

```
                  Input Video (.mp4 / .avi / .mov)
                                |
                 [MediaPipe Face Detection / Crop]
                                |
                   Sequence of Face Frames (T, 3, 380, 380)
                                |
        +-----------------------+-----------------------+
        |                       |                       |
        v                       v                       v
 [Spatial Branch]      [Frequency Branch]      [Temporal Branch]
 EfficientNet-B4        8x8 Block Orthonormal    Inter-frame Delta
 Feature Extractor             2D DCT           Concat (1792*2)
        |                       |                       |
   1792-dim Feats        192-dim Coeff CNN          Bi-GRU (2 layers)
        |                       |                       |
   Spatial Score           Spectral Score       Temporal Attention
        |                       |                       |
        +-----------------------+-----------------------+
                                |
                    [Calibrated Logit Fusion]
                     Learned Dynamic Weights
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
     [Video Verdict]                     [Content Composition]
  AI Manipulated vs Real                   % AI vs % Real Content
  Confidence Level (High/Mod)            Second-by-Second Timeline
```

---

## 📁 Repository Structure

```
ISC-FINAL-VIDEO/
├── models/
│   ├── __init__.py
│   └── detector.py             # Spatial, Frequency, Temporal, and Ensemble architectures
├── dataset.py                  # PyTorch dataset loaders & MediaPipe video processor
├── train.py                    # Multi-stage training script (AMP fp16 + Cosine LR)
├── evaluate.py                 # Benchmark evaluation on test set (ROC, PR, Confusion Matrix)
├── analyze_video.py            # Video analysis engine (quantifies % AI vs % Real)
├── app.py                      # Flask Web Dashboard with interactive visualization
├── run_demo.py                 # Quick demonstration script on test videos
├── templates/
│   └── index.html              # Modern, responsive UI template
├── checkpoints/                # Saved model weights & training history
├── data/
│   ├── frames/                 # 1,865 video clips (37,300 cropped face frames)
│   ├── features/               # Precomputed 1792-dim feature tensors (.pt)
│   └── uploads/                # 60 test .mp4 videos from Celeb-DF benchmark
└── requirements.txt            # Python dependencies
```

---

## 🚀 How to Run

### 1. Launch the Interactive Web Dashboard
```bash
python app.py
```
Open `http://localhost:5000` in your browser.
- Upload any video file or pick from 60 benchmark sample videos in `data/uploads/`.
- Click **"Analyze Video for AI Forgery"** to view the Real vs AI composition meter, forensic scores, second-by-second timeline chart, and suspicious keyframe evidence.

### 2. Analyze Any Video via Command Line
```bash
python analyze_video.py "data/uploads/sample_video.mp4" --output_json report.json
```
Prints an executive terminal report:
```
======================================================================
          AI VIDEO FORGERY DETECTION & ANALYSIS REPORT
======================================================================
 Video File      : sample_video.mp4
 Duration        : 8.4s | FPS: 25.0 | Analyzed Frames: 16
----------------------------------------------------------------------
 OVERALL VERDICT : AI MANIPULATED (DEEPFAKE) (Confidence: High)
 AI Probability  : 94.2% | Real Probability: 5.8%
----------------------------------------------------------------------
 CONTENT BREAKDOWN (HOW MUCH IS AI vs REAL):
   [AI Manipulated Content] : 87.5% (14 frames)
   [Authentic Real Content] : 12.5% (2 frames)
   Visual Bar: [###################################-----]
               ^ AI (87.5%)                     ^ REAL (12.5%)
----------------------------------------------------------------------
 FORENSIC BRANCH SCORES:
   * Spatial Boundary Artifacts : 92.1% (Weight: 0.412)
   * Frequency Spectral Anomaly : 84.6% (Weight: 0.201)
   * Temporal Motion Jitter     : 95.8% (Weight: 0.387)
----------------------------------------------------------------------
 DETECTED MANIPULATION TIMESTAMPS:
   >> [00:01.00 -> 00:07.50] - Face/Visual Manipulation
======================================================================
```

### 3. Re-train the Model
```bash
# Stage 1: Pre-train temporal BiGRU on pre-extracted features
python train.py --mode temporal --epochs_stage1 25

# Stage 2: Fine-tune the full 3-Branch Ensemble on video face frames
python train.py --mode frames --epochs_stage2 8 --batch_size 4 --num_frames 4
```

### 4. Evaluate Benchmark Performance
```bash
python evaluate.py --checkpoint checkpoints/best_forgery_detector.pt
```
Generates confusion matrices, ROC curves, and detailed classification reports in `evaluation_results/`.
