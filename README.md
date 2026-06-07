# Deepfake Detection

Detect AI-generated and deepfake images and videos using a **3-model Hugging Face ensemble** with test-time augmentation (TTA). Includes a Gradio web UI, optional EfficientNet training pipeline, and utilities for local dataset setup.

**Repository:** [github.com/Ganeshc-web/deepfake-detection](https://github.com/Ganeshc-web/deepfake-detection)

## Features

- **3-model ensemble** (no training required to run the app):
  - `prithivMLmods/deepfake-detector-model-v1` — SigLIP deepfake detector
  - `capcheck/ai-image-detection` — ViT AI vs real
  - `prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0` — 3-class AI / deepfake / real
- **Test-time augmentation** — original, flip, and center crop averaged per model
- **Gradio web UI** — dark theme, image + video tabs, per-model score breakdown
- **Optional training** — EfficientNet-B0 fine-tuning on the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset
- **Free & local** — models download from Hugging Face on first run; no API key needed

## Project Structure

```
deepfake-detection/
├── app.py                 # Gradio web application
├── requirements.txt
├── README.md
├── data/
│   ├── train/{real,fake}/ # Training images (not in repo — download separately)
│   └── valid/{real,fake}/
├── scripts/
│   ├── setup_data.py      # Extract Kaggle zip into data/
│   ├── test_hf_detector.py
│   └── benchmark_hf_models.py
└── src/
    ├── detector.py        # 3-model ensemble + TTA
    ├── predict.py         # Image and video inference
    ├── dataset.py         # DataLoaders and transforms
    ├── model.py           # EfficientNet-B0 builder
    ├── train.py           # Full training (10 epochs)
    ├── train_fast.py      # Quick training subset
    └── gradcam.py         # Grad-CAM for local EfficientNet model
```

## Quick Start (Run the App)

### 1. Clone and set up environment

```bash
git clone https://github.com/Ganeshc-web/deepfake-detection.git
cd deepfake-detection

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

> **GPU (optional):** For faster inference, install a CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) before the other packages.

### 2. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:7860** in your browser.

On first run, three Hugging Face models (~350 MB each) download automatically. This may take a few minutes.

### 3. Analyze media

**Image tab**
- Upload an image and click **Analyze**
- View **REAL** or **FAKE** with confidence
- See ensemble and per-model fake scores

**Video tab**
- Upload a video and click **Analyze**
- Every 15th frame is scored; majority vote determines the result

> **Note:** Inference on CPU is slower (~15–30 s per image) because the ensemble runs 3 models × 3 TTA views.

## How the Ensemble Works

| Model | Weight | Strength |
|-------|--------|------------|
| deepfake-detector-model-v1 | 45% | Classic deepfakes / synthetic faces |
| capcheck/ai-image-detection | 35% | Diffusion / AI-generated vs real |
| AI-vs-Deepfake-vs-Real-v2.0 | 20% | Explicit AI class (ChatGPT-style images) |

Decision combines weighted scores, multi-model agreement, and guards against false positives when the face model strongly indicates “real.”

## Optional: Train Your Own Model

Training is **not required** to use the app. To fine-tune EfficientNet-B0 on face data:

### Download dataset

Use the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) Kaggle dataset.

Place `140k-real-and-fake-faces.zip` in the project root, then:

```bash
python scripts/setup_data.py
```

Or manually organize:

```
data/train/{real,fake}/
data/valid/{real,fake}/
```

### Train

```bash
# Full training (10 epochs, saves deepfake_model.pth)
python src/train.py

# Fast subset (5 epochs, 2000 samples/class)
python src/train_fast.py
```

The Gradio app uses the Hugging Face ensemble by default, not the local `.pth` checkpoint.

## Requirements

- Python 3.10+ (tested on 3.13)
- ~2 GB disk for Hugging Face models
- ~4 GB RAM recommended for ensemble inference

## Troubleshooting

| Issue | Solution |
|-------|----------|
| App slow on first analyze | Normal — 3 models × TTA on CPU; use GPU if available |
| `Dataset directories not found` | Only needed for training — run `scripts/setup_data.py` |
| Port 7860 in use | Stop other Gradio apps or set `GRADIO_SERVER_PORT` |
| Model download fails | Check internet; optional: `HF_TOKEN` for Hugging Face rate limits |

## License

Educational and research use. Third-party models and datasets have their own licenses:

- [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (Kaggle)
- Hugging Face model cards for ensemble weights

## Author

[Ganeshc-web](https://github.com/Ganeshc-web)
