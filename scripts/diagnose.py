"""Diagnose the detector -> crop -> OCR pipeline on a single image.

This isolates EXACTLY where reads go wrong, so you can stop guessing:

  * backend per detector  -> tells you if real weights loaded or MOCK is used
  * model.names           -> tells you if class order/names match the app
  * every detection       -> label, confidence, bbox
  * saved crop images     -> open them to see what the OCR actually receives
  * OCR output per crop    -> confirms whether the crop is readable

Usage:
    python scripts/diagnose.py path/to/image.jpg [out_dir]

Tips:
    AICM_IMGSZ=1920 python scripts/diagnose.py img.jpg      # try higher res
    AICM_CROP_PAD=0 python scripts/diagnose.py img.jpg      # exact box crop
"""
from __future__ import annotations

import os
import sys

# Make sibling modules importable when run as `python scripts/diagnose.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from model_layer import crop_detection, get_models  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python scripts/diagnose.py <image> [out_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "diag_out"
    os.makedirs(out_dir, exist_ok=True)

    image = Image.open(img_path).convert("RGB")
    print(f"\nImage: {img_path}  size={image.size}")

    models = get_models()
    ocr = models["ocr"]
    print(f"OCR backend: {ocr.backend}")

    for key in ("container_iso", "plate", "damage"):
        det = models[key]
        print("\n" + "=" * 64)
        print(f"Detector '{key}': backend={det.backend}  weights={det.weights}")
        model = getattr(det, "model", None)
        if model is not None:
            print(f"  model.names = {getattr(model, 'names', None)}")
        else:
            print("  (MOCK) no real weights loaded -> crops are fixed rectangles")

        res = det.predict(image)
        if not res.detections:
            print("  NO DETECTIONS on this image.")
            continue

        for i, d in enumerate(res.detections):
            crop = crop_detection(image, d.bbox)
            fn = os.path.join(out_dir, f"{key}_{i}_{d.label}.png")
            crop.save(fn)
            text = "" if key == "damage" else ocr.read(crop)
            print(f"  #{i} label={d.label:<12} conf={d.confidence:.3f} "
                  f"bbox={d.bbox} crop={crop.size} -> OCR='{text}'")
            print(f"       saved crop: {fn}")

    print(
        "\nNow OPEN the saved crops in " + out_dir + "/ and check:\n"
        "  1. Is the backend 'yolo' (not 'mock')? If mock -> weights not found.\n"
        "  2. Do model.names match the app's expected labels?\n"
        "  3. Is each crop a TIGHT, upright, single-line text region?\n"
        "     - If crops look wrong -> detector/imgsz issue (try AICM_IMGSZ).\n"
        "     - If crops look correct but OCR is wrong -> OCR preprocessing.\n"
    )


if __name__ == "__main__":
    main()
