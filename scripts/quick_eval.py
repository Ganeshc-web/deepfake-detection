"""Quick ensemble eval on validation samples."""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.detector import load_detector


def main() -> None:
    random.seed(0)
    detector = load_detector()

    for cls in ("real", "fake"):
        folder = PROJECT_ROOT / "data" / "valid" / cls
        images = random.sample(list(folder.glob("*.jpg")), 8)
        wrong = 0
        print(f"\n=== {cls.upper()} ===")
        for path in images:
            result = detector.predict_with_details(Image.open(path).convert("RGB"))
            pred_fake = "FAKE" in result.label
            expected_fake = cls == "fake"
            ok = pred_fake == expected_fake
            wrong += int(not ok)
            scores = {s.name: round(s.fake_prob, 2) for s in result.model_scores}
            status = "WRONG" if not ok else "ok"
            print(f"  {status} {path.name}: {result.label} w={result.fake_prob:.2f} {scores}")
        print(f"  errors: {wrong}/8")


if __name__ == "__main__":
    main()
