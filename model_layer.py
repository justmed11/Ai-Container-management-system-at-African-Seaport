"""Model layer for the AI Container Management system.

This module defines thin wrappers around the four trained models plus the
shared OCR head. Each wrapper tries to load the real model (Ultralytics YOLO
for detectors, HuggingFace TrOCR for OCR). If weights / libraries are not
available in the current environment, it transparently falls back to a MOCK
implementation so the full Gradio pipeline still runs end to end for demos.

Replace the MODEL_PATHS with your trained checkpoints and the mock branches
will be skipped automatically.

Models
------
1. ContainerISODetector   -> detects the container-ID marking + ISO-type marking
2. DamageDetector         -> detects rust / dent / hole regions
3. PlateDetector          -> detects the truck license-plate region
4. TrOCRReader            -> microsoft/trocr-base-printed fine-tuned head that
                             reads cropped container-ID, ISO-type and plate text

A single `crop_detection()` helper is used by every detector before OCR.
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration: point these at your trained weights.
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    "container_iso": os.getenv("CONTAINER_ISO_WEIGHTS", "weights/container_iso.pt"),
    "damage":        os.getenv("DAMAGE_WEIGHTS", "weights/damage.pt"),
    "plate":         os.getenv("PLATE_WEIGHTS", "weights/plate.pt"),
    "trocr":         os.getenv("TROCR_DIR", "weights/trocr-base-printed-finetuned"),
}

# Detector inference tuning.
# A detector trained at high resolution but run at the default imgsz=640 often
# produces loose/inaccurate boxes -> the crop includes background or clips
# characters -> OCR quality drops even though OCR is perfect on clean crops.
# Raise imgsz (or set AICM_IMGSZ) so small text regions are localized tightly.
_PREDICT_IMGSZ = int(os.getenv("AICM_IMGSZ", "1280"))
_PREDICT_CONF = float(os.getenv("AICM_CONF", "0.25"))
_OCR_MAX_NEW_TOKENS = int(os.getenv("AICM_OCR_MAX_TOKENS", "20"))


def _torch_device(prefer_index: bool = False):
    """Choose CUDA when available, with an environment-variable override.

    Ultralytics accepts GPU index ``0`` while Hugging Face expects
    ``cuda:0``.  Keeping the choice in one helper prevents YOLO and TrOCR from
    accidentally using different devices.
    """
    requested = os.getenv("AICM_DEVICE", "auto").strip().lower()
    if requested in {"cpu", "-1"}:
        return "cpu"
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False
    if not cuda_ok:
        return "cpu"
    if requested not in {"", "auto", "cuda", "cuda:0", "0"}:
        return int(requested) if prefer_index and requested.isdigit() else requested
    return 0 if prefer_index else "cuda:0"


def _resolve_weights(path: str, keywords: List[str]) -> str:
    """Return `path` if it exists, else try to find a .pt in the weights dir
    whose filename matches one of `keywords`. This handles the common case of
    trained weights being named e.g. best.pt instead of container_iso.pt."""
    if os.path.exists(path):
        return path
    wdir = os.path.dirname(path) or "weights"
    if os.path.isdir(wdir):
        pts = sorted(f for f in os.listdir(wdir) if f.lower().endswith(".pt"))
        for f in pts:
            low = f.lower()
            if any(k in low for k in keywords):
                cand = os.path.join(wdir, f)
                print(f"[model_layer] '{path}' not found; using '{cand}' (keyword match).")
                return cand
    return path


@dataclass
class Detection:
    """A single detected box."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    text: Optional[str] = None       # filled in after OCR


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    backend: str = "mock"


# ---------------------------------------------------------------------------
# Shared cropping helper (used before OCR on every detector output)
# ---------------------------------------------------------------------------
def crop_detection(image: Image.Image, bbox: Tuple[int, int, int, int], pad: Optional[int] = None) -> Image.Image:
    """Crop the detected region (with small padding) so OCR sees only the text.

    Padding defaults to AICM_CROP_PAD (or 4). Set AICM_CROP_PAD=0 to crop the
    exact detector box, matching a manual crop that uses raw box coordinates.
    """
    if pad is None:
        pad = int(os.getenv("AICM_CROP_PAD", "4"))
    w, h = image.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Class-name resolution
# ---------------------------------------------------------------------------
# The app's business logic expects canonical labels: container_id, iso_type,
# plate, rust, dent, hole. A trained YOLO model may use different class names
# or a different class ORDER. We must map the model's OWN names (model.names)
# onto these canonical labels instead of trusting a positional list.
_LABEL_ALIASES = {
    # container id
    "container_id": "container_id", "containerid": "container_id",
    "container": "container_id", "container_number": "container_id",
    "container_no": "container_id", "containerno": "container_id",
    "cntr": "container_id", "bic": "container_id", "number": "container_id",
    "id": "container_id", "code": "container_id", "serial": "container_id",
    # iso type
    "iso_type": "iso_type", "isotype": "iso_type", "iso": "iso_type",
    "iso_code": "iso_type", "type": "iso_type", "size_type": "iso_type",
    "sizetype": "iso_type", "size": "iso_type",
    # plate
    "plate": "plate", "license_plate": "plate", "licenseplate": "plate",
    "number_plate": "plate", "numberplate": "plate", "plate_number": "plate",
    "platenumber": "plate", "license": "plate", "lp": "plate", "reg": "plate",
    # damage
    "rust": "rust", "corrosion": "rust",
    "dent": "dent", "dented": "dent",
    "hole": "hole", "puncture": "hole", "perforation": "hole",
    "damage": "damage", "scratch": "scratch",
}


def _parse_label_overrides() -> dict:
    """Read an explicit class-name -> canonical-label map from AICM_LABEL_MAP.

    Use this when you forgot / used non-descriptive class names in training.
    The keys are your model's OWN class names (or indices); values are the
    app's canonical labels (container_id, iso_type, plate, rust, dent, hole).

    Example (PowerShell):
        $env:AICM_LABEL_MAP="0=container_id;1=iso_type;plate=plate"
    Example (bash):
        AICM_LABEL_MAP="cnum=container_id,ctype=iso_type" python app.py
    """
    raw = os.getenv("AICM_LABEL_MAP", "").strip()
    out = {}
    if raw:
        for pair in re.split(r"[;,]", raw):
            if "=" in pair:
                k, v = pair.split("=", 1)
                key = re.sub(r"[^a-z0-9]+", "_", k.strip().lower()).strip("_")
                out[key] = v.strip()
    return out


_LABEL_OVERRIDES = _parse_label_overrides()
if _LABEL_OVERRIDES:
    print(f"[model_layer] AICM_LABEL_MAP override active: {_LABEL_OVERRIDES}")


def normalize_label(raw: str) -> str:
    """Map a raw model class name onto a canonical app label.

    Priority: explicit AICM_LABEL_MAP override -> built-in aliases -> raw name.
    """
    key = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")
    if key in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[key]
    return _LABEL_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Detector base class
# ---------------------------------------------------------------------------
class _YoloDetector:
    """Loads an Ultralytics YOLO model if possible, else mocks detections."""

    def __init__(self, weights: str, class_names: List[str], keywords: Optional[List[str]] = None):
        self.weights = weights
        self.class_names = class_names
        self.keywords = keywords or []
        self.model = None
        self.backend = "mock"
        self.device = _torch_device(prefer_index=True)
        self._try_load()

    def _try_load(self):
        try:
            from ultralytics import YOLO  # type: ignore
            self.weights = _resolve_weights(self.weights, self.keywords)
            if os.path.exists(self.weights):
                self.model = YOLO(self.weights)
                self.backend = f"yolo-{'cuda' if self.device != 'cpu' else 'cpu'}"
                # Print the model's OWN class names so mismatches are visible.
                names = getattr(self.model, "names", None)
                print(f"[model_layer] Loaded {self.weights} | model classes: {names} "
                      f"| app expects: {self.class_names} | device={self.device} "
                      f"| imgsz={_PREDICT_IMGSZ} conf={_PREDICT_CONF}")
            else:
                print(f"[model_layer] Weights not found: {self.weights} -> using MOCK boxes "
                      f"(this alone will cause wrong crops / bad OCR).")
        except Exception as e:
            print(f"[model_layer] Could not load {self.weights} ({e}); using MOCK.")
            self.model = None
            self.backend = "mock"

    def predict(self, image: Image.Image) -> DetectionResult:
        if self.model is not None:
            # Pass the PIL image directly. Ultralytics treats a NumPy array as
            # BGR, but np.array(pil_rgb) is RGB -> that silently swaps the red/
            # blue channels and degrades detection. A PIL image is read as RGB.
            try:
                res = self.model.predict(
                    image, verbose=False, imgsz=_PREDICT_IMGSZ,
                    conf=_PREDICT_CONF, device=self.device,
                )[0]
            except RuntimeError as exc:
                # A 4 GB GPU can occasionally run out of memory when several
                # models are resident. Retry on CPU instead of breaking the app.
                if self.device == "cpu" or "memory" not in str(exc).lower():
                    raise
                print(f"[model_layer] CUDA inference failed ({exc}); retrying YOLO on CPU.")
                self.device = "cpu"
                self.backend = "yolo-cpu"
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                res = self.model.predict(
                    image, verbose=False, imgsz=_PREDICT_IMGSZ,
                    conf=_PREDICT_CONF, device="cpu",
                )[0]
            # Use the model's OWN class names (res.names / model.names), then
            # normalize onto the app's canonical labels. This fixes wrong crops
            # caused by a class order/name mismatch with a hardcoded list.
            model_names = getattr(res, "names", None) or getattr(self.model, "names", {}) or {}
            dets = []
            for b in res.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                xyxy = tuple(int(v) for v in b.xyxy[0].tolist())
                raw = model_names.get(cls, self.class_names[cls] if cls < len(self.class_names) else str(cls))
                label = normalize_label(raw)
                dets.append(Detection(label=label, confidence=conf, bbox=xyxy))
            if os.getenv("AICM_DEBUG"):
                print(f"[model_layer] {os.path.basename(self.weights)} detections: "
                      + ", ".join(f"{d.label}({d.confidence:.2f})@{d.bbox}" for d in dets))
            return DetectionResult(detections=dets, backend="yolo")
        # No real model loaded -> MOCK. This is almost always the cause of
        # "OCR is perfect on manual crops but wrong in the app": the crop comes
        # from a FIXED demo rectangle, not from your detector. Make it loud.
        if not getattr(self, "_warned_mock", False):
            print(f"[model_layer] *** WARNING: '{os.path.basename(self.weights)}' is running in "
                  f"MOCK mode (real weights NOT loaded). Boxes are fixed demo rectangles, so OCR "
                  f"reads the WRONG region. Fix the weights path/filename to stop this. ***")
            self._warned_mock = True
        if os.getenv("AICM_STRICT"):
            # Strict mode: refuse to fabricate boxes so failures are obvious.
            return DetectionResult(detections=[], backend="mock")
        return self._mock_predict(image)

    def _mock_predict(self, image: Image.Image) -> DetectionResult:  # overridden
        return DetectionResult(detections=[], backend="mock")


# ---------------------------------------------------------------------------
# 1. Container ID + ISO type detector
# ---------------------------------------------------------------------------
class ContainerISODetector(_YoloDetector):
    def __init__(self):
        super().__init__(MODEL_PATHS["container_iso"], ["container_id", "iso_type"],
                         keywords=["container", "iso", "cont", "cntr"])

    def _mock_predict(self, image: Image.Image) -> DetectionResult:
        w, h = image.size
        dets = [
            Detection("container_id", 0.95, (int(w * 0.20), int(h * 0.30), int(w * 0.80), int(h * 0.45))),
            Detection("iso_type", 0.92, (int(w * 0.55), int(h * 0.48), int(w * 0.80), int(h * 0.58))),
        ]
        return DetectionResult(detections=dets, backend="mock")


# ---------------------------------------------------------------------------
# 2. Damage detector (rust / dent / hole)
# ---------------------------------------------------------------------------
class DamageDetector(_YoloDetector):
    def __init__(self):
        super().__init__(MODEL_PATHS["damage"], ["rust", "dent", "hole"],
                         keywords=["damage", "rust", "dent", "hole", "defect"])

    def _mock_predict(self, image: Image.Image) -> DetectionResult:
        # Deterministic pseudo-random based on image size so demos are stable.
        w, h = image.size
        rng = random.Random(w * 7919 + h)
        dets = []
        for label in ["rust", "dent", "hole"]:
            if rng.random() > 0.55:
                x1 = rng.randint(0, max(1, w - 60))
                y1 = rng.randint(0, max(1, h - 60))
                dets.append(Detection(label, round(rng.uniform(0.55, 0.95), 2), (x1, y1, x1 + 50, y1 + 50)))
        return DetectionResult(detections=dets, backend="mock")


# ---------------------------------------------------------------------------
# 3. Truck license-plate detector
# ---------------------------------------------------------------------------
class PlateDetector(_YoloDetector):
    def __init__(self):
        super().__init__(MODEL_PATHS["plate"], ["plate"],
                         keywords=["plate", "license", "lp"])

    def _mock_predict(self, image: Image.Image) -> DetectionResult:
        w, h = image.size
        det = Detection("plate", 0.93, (int(w * 0.35), int(h * 0.70), int(w * 0.65), int(h * 0.82)))
        return DetectionResult(detections=[det], backend="mock")


# ---------------------------------------------------------------------------
# ISO 6346 patterns (used to split a single OCR string into id + iso type)
# ---------------------------------------------------------------------------
# Container number = 4 letters (owner code + category) + 7 digits
# (6-digit serial + 1 check digit), e.g. MSKU1234567.
_CONTAINER_ID_RE = re.compile(r"[A-Z]{4}[0-9]{7}")
# Loose fallback if OCR drops the check digit (4 letters + 6/7 digits).
_CONTAINER_ID_LOOSE_RE = re.compile(r"[A-Z]{4}[0-9]{6,7}")
# ISO size-type code = digit, digit/letter, letter, digit, e.g. 45G1, 22G1,
# 45R1, L5G1 (first char is usually a digit; we allow a letter too, then a
# strict letter in position 3 and a trailing digit).
_ISO_TYPE_RE = re.compile(r"[0-9A-Z][0-9A-Z][A-Z][0-9]")
# Current fleet/dataset format: T 123 ABC -> canonical T123ABC. Keep the
# detector confidence unchanged, but never expose OCR text as a plate unless
# it conforms to this Tanzanian registration format.
_TZ_PLATE_RE = re.compile(r"^T[0-9]{3}[A-Z]{3}$")


# ---------------------------------------------------------------------------
# 4. TrOCR reader (microsoft/trocr-base-printed fine-tuned)
# ---------------------------------------------------------------------------
class TrOCRReader:
    def __init__(self):
        self.processor = None
        self.model = None
        self.backend = "mock"
        self.device = _torch_device()
        self._try_load()

    def _try_load(self):
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
            src = MODEL_PATHS["trocr"]
            if not os.path.isdir(src):
                src = "microsoft/trocr-base-printed"
            # Fast preprocessing is noticeably cheaper on large photos. Older
            # saved processors may not support it, so keep a compatibility
            # fallback rather than preventing the app from starting.
            try:
                self.processor = TrOCRProcessor.from_pretrained(src, use_fast=True)
            except (TypeError, ValueError):
                self.processor = TrOCRProcessor.from_pretrained(src)
            self.model = VisionEncoderDecoderModel.from_pretrained(src)
            self.model.eval()
            if self.device != "cpu":
                try:
                    self.model.to(self.device)
                    torch.backends.cudnn.benchmark = True
                except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                    print(f"[model_layer] Could not place TrOCR on CUDA ({exc}); using CPU.")
                    self.device = "cpu"
                    self.model.to("cpu")
                    torch.cuda.empty_cache()
            self.backend = f"trocr-{'cuda' if self.device != 'cpu' else 'cpu'}"
            print(f"[model_layer] Loaded TrOCR on {self.device} (max tokens={_OCR_MAX_NEW_TOKENS}).")
        except Exception as exc:
            print(f"[model_layer] Could not load TrOCR ({exc}); using MOCK OCR.")
            self.processor = None
            self.model = None
            self.backend = "mock"

    def read(self, crop: Image.Image) -> str:
        if self.model is not None and self.processor is not None:
            import torch
            pixel_values = self.processor(
                images=crop.convert("RGB"), return_tensors="pt"
            ).pixel_values.to(self.device, non_blocking=self.device != "cpu")
            try:
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_new_tokens=_OCR_MAX_NEW_TOKENS,
                        num_beams=1,
                        do_sample=False,
                        use_cache=True,
                    )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                if self.device == "cpu" or "memory" not in str(exc).lower():
                    raise
                # Preserve functionality on low-VRAM cards: switch this OCR
                # model to CPU once, then continue normally for later reads.
                print(f"[model_layer] TrOCR CUDA memory limit reached ({exc}); switching to CPU.")
                self.device = "cpu"
                self.backend = "trocr-cpu"
                self.model.to("cpu")
                torch.cuda.empty_cache()
                pixel_values = pixel_values.to("cpu")
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_new_tokens=_OCR_MAX_NEW_TOKENS,
                        num_beams=1,
                        do_sample=False,
                        use_cache=True,
                    )
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return text.strip()
        return self._mock_read(crop)

    @staticmethod
    def _mock_read(crop: Image.Image) -> str:
        # Mock reader returns an empty string; the orchestration layer fills in
        # demo values from the dataset so the full flow can be exercised.
        return ""

    # --- post-processing / normalisation helpers ---------------------------
    @staticmethod
    def normalize_container_id(raw: str) -> str:
        s = re.sub(r"[^A-Z0-9]", "", raw.upper())
        return s

    @staticmethod
    def normalize_iso(raw: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", raw.upper())

    @staticmethod
    def normalize_plate(raw: str) -> str:
        # Plates are stored in the CSVs as e.g. ``T 941 BLN`` while OCR often
        # returns ``T941BLN``.  Use one compact canonical representation for
        # all comparisons; display formatting remains available from the CSV.
        candidate = re.sub(r"[^A-Z0-9]", "", str(raw).upper())
        return candidate if _TZ_PLATE_RE.fullmatch(candidate) else ""

    @staticmethod
    def split_container_iso(raw: str):
        """Split one OCR string into (container_id, iso_type).

        Your container detector has a single class ('container-number'), so its
        crop often contains BOTH the 11-char container number and the 4-char
        ISO size-type code (same line or stacked). This extracts each using
        the ISO 6346 formats, so:
            'MSKU1234567 45G1'  -> ('MSKU1234567', '45G1')
            'MSKU123456745G1'   -> ('MSKU1234567', '45G1')
            'MSKU1234567'       -> ('MSKU1234567', '')
        """
        s = re.sub(r"[^A-Z0-9]", "", str(raw).upper())
        container_id = ""
        iso_type = ""
        m = _CONTAINER_ID_RE.search(s) or _CONTAINER_ID_LOOSE_RE.search(s)
        if m:
            container_id = m.group(0)
            rest = s[: m.start()] + s[m.end():]
        else:
            rest = s
        im = _ISO_TYPE_RE.search(rest)
        if im:
            iso_type = im.group(0)
        if not container_id:
            # No valid container pattern -> return the cleaned string as-is so
            # nothing is silently lost, and leave iso_type empty.
            container_id = s
            iso_type = ""
        return container_id, iso_type


# ---------------------------------------------------------------------------
# Singleton accessor so models load only once.
# ---------------------------------------------------------------------------
_REGISTRY = {}


def get_models():
    if not _REGISTRY:
        _REGISTRY["container_iso"] = ContainerISODetector()
        _REGISTRY["damage"] = DamageDetector()
        _REGISTRY["plate"] = PlateDetector()
        _REGISTRY["ocr"] = TrOCRReader()
    return _REGISTRY
