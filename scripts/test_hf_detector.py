"""Quick smoke test for Hugging Face deepfake detector."""

from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.detector import load_detector


def main() -> None:
    detector = load_detector()
    print(f"Model: {detector.display_name}\n")

    for cls in ("fake", "real"):
        folder = PROJECT_ROOT / "data" / "valid" / cls
        if not folder.exists():
            print(f"Skip {cls}: folder missing")
            continue

        images = list(folder.glob("*.jpg"))
        if not images:
            print(f"Skip {cls}: no images")
            continue

        sample = random.sample(images, min(5, len(images)))
        correct = 0
        for path in sample:
            img = Image.open(path).convert("RGB")
            label, conf = detector.predict_image(img)
            predicted_fake = "FAKE" in label
            expected_fake = cls == "fake"
            ok = predicted_fake == expected_fake
            correct += int(ok)
            status = "OK" if ok else "WRONG"
            print(f"  {cls} | {path.name} -> {label} ({conf:.1f}%) [{status}]")

        print(f"  {cls}: {correct}/{len(sample)} correct\n")


if __name__ == "__main__":
    main()
