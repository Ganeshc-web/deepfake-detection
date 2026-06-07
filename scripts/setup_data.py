"""
Extract the Kaggle 140k Real and Fake Faces archive into the project data layout.

Looks for 140k-real-and-fake-faces.zip in the project root, extracts train/ and
valid/ splits, and links them into data/train/ and data/valid/.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "140k-real-and-fake-faces.zip"
EXTRACT_ROOT = PROJECT_ROOT / "data_source" / "real-vs-fake"
DATA_TRAIN = PROJECT_ROOT / "data" / "train"
DATA_VALID = PROJECT_ROOT / "data" / "valid"

# Paths inside the Kaggle zip archive
ZIP_PREFIX = "real_vs_fake/real-vs-fake/"
SPLITS = ("train", "valid")


def _count_images(folder: Path) -> int:
    """Count image files under a directory."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def extract_train_valid() -> None:
    """Extract only train/ and valid/ images from the Kaggle zip."""
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Dataset zip not found: {ZIP_PATH}\n"
            "Download from: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces"
        )

    prefixes = tuple(f"{ZIP_PREFIX}{split}/" for split in SPLITS)
    print(f"Extracting train/ and valid/ from {ZIP_PATH.name} ...")
    print("This may take several minutes (~93k images).\n")

    extracted = 0
    skipped = 0

    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = [
            name
            for name in zf.namelist()
            if name.startswith(prefixes) and not name.endswith("/")
        ]
        total = len(members)

        for index, name in enumerate(members, start=1):
            relative = name[len(ZIP_PREFIX) :]
            target = EXTRACT_ROOT / relative

            if target.exists():
                skipped += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1

            if index % 2000 == 0 or index == total:
                print(f"  Progress: {index}/{total} files processed ...")

    print(f"\nExtraction complete: {extracted} new, {skipped} already present.")


def _remove_dir(path: Path) -> None:
    """Remove a directory tree if it exists."""
    if path.exists():
        shutil.rmtree(path)


def _link_data_split(source: Path, target: Path) -> None:
    """
    Point data/train or data/valid at the extracted split folder.

    Uses a directory junction on Windows and a symlink elsewhere.
    """
    _remove_dir(target)

    if sys.platform == "win32":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        target.symlink_to(source, target_is_directory=True)


def link_data_folders() -> None:
    """Create data/train and data/valid links to extracted splits."""
    for split in SPLITS:
        source = EXTRACT_ROOT / split
        target = PROJECT_ROOT / "data" / split

        if not source.exists():
            raise FileNotFoundError(f"Extracted split not found: {source}")

        print(f"Linking {target.name} -> {source}")
        _link_data_split(source, target)


def verify_layout() -> None:
    """Print image counts for each class folder."""
    print("\nDataset ready:")
    for split in SPLITS:
        for label in ("real", "fake"):
            folder = PROJECT_ROOT / "data" / split / label
            count = _count_images(folder)
            print(f"  data/{split}/{label}: {count} images")


def main() -> None:
    """Extract zip and wire up data/train and data/valid."""
    if _count_images(DATA_TRAIN) > 0 and _count_images(DATA_VALID) > 0:
        print("Dataset already set up.")
        verify_layout()
        return

    extract_train_valid()
    link_data_folders()
    verify_layout()
    print("\nRun training with: python src/train.py")


if __name__ == "__main__":
    main()
