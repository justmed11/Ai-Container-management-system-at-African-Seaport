"""Reveal the class names stored inside your trained YOLO weights.

You forgot the class names you used during training -- this prints them so you
can see exactly what each model detects and in what order. Whatever this shows
is the SOURCE OF TRUTH; the app now uses these names (not a hardcoded list).

Usage:
    python scripts/show_classes.py
    python scripts/show_classes.py weights/container_iso.pt

If a name is non-descriptive (e.g. '0', 'class0', 'obj'), map it to the app's
canonical label with AICM_LABEL_MAP, e.g.:
    (bash)        AICM_LABEL_MAP="0=container_id,1=iso_type" python app.py
    (PowerShell)  $env:AICM_LABEL_MAP="0=container_id;1=iso_type"; python app.py
"""
from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def show(path: str) -> None:
    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        print(f"Cannot import ultralytics: {e}")
        return
    if not os.path.exists(path):
        print(f"  (missing) {path}")
        return
    try:
        model = YOLO(path)
        names = getattr(model, "names", None)
        print(f"  {path}")
        print(f"      classes = {names}")
        if isinstance(names, dict):
            for idx, nm in sorted(names.items()):
                print(f"        index {idx} -> '{nm}'")
    except Exception as e:
        print(f"  {path}: failed to load ({e})")


def main() -> None:
    args = sys.argv[1:]
    if args:
        paths = args
    else:
        # default: every .pt in the weights/ folder
        wdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
        paths = sorted(glob.glob(os.path.join(wdir, "*.pt"))) or [
            "weights/container_iso.pt", "weights/plate.pt", "weights/damage.pt",
        ]
    print("Trained model class names (this is the ground truth):\n")
    for p in paths:
        show(p)
    print(
        "\nThe app's canonical labels are: container_id, iso_type, plate, "
        "rust, dent, hole.\nIf your names differ and aren't auto-mapped, set "
        "AICM_LABEL_MAP to map them (see this file's header)."
    )


if __name__ == "__main__":
    main()
