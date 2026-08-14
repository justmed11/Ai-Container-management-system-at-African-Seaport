"""Decision / orchestration layer for the AI Container Management system.

This ties the four models + OCR to the Dar es Salaam port reference datasets
and implements the business logic for the four functional areas:

  1. Gate Inspection & Verification  (gate-in)
  2. Inspection Phase                (CV + LLM reports, deep inspection)
  3. Gate-Out Inspection             (verify pickup, issue gate-out ticket)
  4. Yard / Container Location Tracking

The media layer accepts BOTH images and videos. Videos are sampled into
frames; the best-scoring detection across frames is used.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import re
import uuid
from typing import Dict, List, Optional, Tuple

from PIL import Image

from model_layer import crop_detection, get_models

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------
def _read_csv(name: str) -> List[Dict[str, str]]:
    path = os.path.join(DATA_DIR, name)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _append_csv(name: str, row: List):
    path = os.path.join(DATA_DIR, name)
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def normalize_plate(value: str) -> str:
    """Return a canonical Tanzanian fleet plate, or empty when invalid.

    The current dataset uses ``T 123 ABC``. OCR commonly removes spaces, so
    the canonical value is ``T123ABC``. Text outside that format is never used
    as a plate, regardless of detector confidence.
    """
    candidate = "".join(ch for ch in str(value).upper() if ch.isalnum())
    return candidate if re.fullmatch(r"T[0-9]{3}[A-Z]{3}", candidate) else ""


class PortDB:
    """In-memory view of the reference datasets with simple lookups."""

    def __init__(self):
        self.reload()

    def reload(self):
        self.iso_types = {r["iso_code"]: r for r in _read_csv("iso_types.csv")}
        self.importers = {r["tin"]: r for r in _read_csv("importers.csv")}
        self.trucks = {normalize_plate(r["plate"]): r for r in _read_csv("trucks.csv")}
        self.containers = {r["container_id"]: r for r in _read_csv("containers.csv")}
        self.release_orders = _read_csv("release_orders.csv")

    def ro_for_container(self, cid: str) -> Optional[Dict[str, str]]:
        for r in self.release_orders:
            if r["container_id"] == cid:
                return r
        return None

    def ro_for_plate(self, plate: str) -> List[Dict[str, str]]:
        key = normalize_plate(plate)
        return [r for r in self.release_orders
                if normalize_plate(r["declared_plate"]) == key]

    def ro_by_id(self, release_order_id: str) -> Optional[Dict[str, str]]:
        return next((r for r in self.release_orders
                     if r["release_order_id"] == release_order_id), None)

    def gate_ticket(self, ticket_id: str) -> Optional[Dict[str, str]]:
        """Load an exact ticket from the persisted ticket log."""
        ticket_id = (ticket_id or "").strip()
        return next((r for r in _read_csv("gate_tickets.csv")
                     if r["ticket_id"] == ticket_id), None)

    def ticket_is_used(self, ticket_id: str) -> bool:
        marker = f"used ticket {ticket_id}"
        return any(r["ticket_type"] == "TICKET_USED"
                   and r.get("decision") == "USED"
                   and r.get("notes") == marker
                   for r in _read_csv("gate_tickets.csv"))

    def gate_in_has_gate_out(self, ticket_id: str) -> bool:
        marker = f"from gate-in {ticket_id}"
        return any(r["ticket_type"] == "GATE_OUT" and r.get("notes") == marker
                   for r in _read_csv("gate_tickets.csv"))

    def mark_returned(self, release_order: Dict[str, str]) -> None:
        """Persist a matched return movement as completed/returned."""
        release_order["ro_status"] = "returned"
        path = os.path.join(DATA_DIR, "release_orders.csv")
        fields = ["release_order_id", "container_id", "iso_type", "importer_tin",
                  "importer_name", "declared_plate", "ro_status", "customs_cleared",
                  "payment_status", "movement_type"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.release_orders)


DB = PortDB()


def mark_ticket_used(ticket_id: str, expected_type: str) -> Dict:
    """Append a ticket-use audit event after an operator verifies a ticket."""
    ticket_id = (ticket_id or "").strip()
    expected_type = (expected_type or "").strip().upper()
    ticket = DB.gate_ticket(ticket_id)
    if not ticket:
        return {"ok": False, "message": "Ticket not found in the ticket log."}
    if ticket["ticket_type"] != expected_type:
        return {"ok": False,
                "message": f"Expected {expected_type} ticket, found {ticket['ticket_type']}."}
    if ticket["decision"] != "ALLOW":
        return {"ok": False, "message": "Only an approved ticket can be marked used."}
    if DB.ticket_is_used(ticket_id):
        return {"ok": False, "message": f"Ticket {ticket_id} is already marked used."}

    event_id = f"USE-{dt.datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
    _append_csv("gate_tickets.csv", [
        event_id, "TICKET_USED", dt.datetime.now().isoformat(timespec="seconds"),
        ticket["plate"], ticket["container_id"], ticket["iso_type"],
        ticket["release_order_id"], "USED", f"used ticket {ticket_id}",
    ])
    return {"ok": True, "message": f"Ticket {ticket_id} verified, used and logged."}


# ---------------------------------------------------------------------------
# Media handling: load an image, or sample frames from a video
# ---------------------------------------------------------------------------
def load_media_frames(media_path: str, max_frames: Optional[int] = None) -> List[Image.Image]:
    """Return a list of PIL frames. Single image -> one frame. Video -> sampled frames."""
    if max_frames is None:
        # Four representative frames halves detector work versus the previous
        # default while preserving coverage across the whole clip. Override
        # with AICM_VIDEO_FRAMES when a particular camera needs more samples.
        max_frames = max(1, int(os.getenv("AICM_VIDEO_FRAMES", "4")))
    if media_path is None:
        return []
    ext = os.path.splitext(media_path)[1].lower()
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    if ext in video_exts:
        try:
            import cv2  # opencv-python-headless is preinstalled
            cap = cv2.VideoCapture(media_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            step = max(1, total // max_frames)
            frames = []
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
                idx += 1
                if len(frames) >= max_frames:
                    break
            cap.release()
            return frames or [Image.new("RGB", (640, 480))]
        except Exception:
            return [Image.new("RGB", (640, 480))]
    # treat as image
    try:
        return [Image.open(media_path).convert("RGB")]
    except Exception:
        return [Image.new("RGB", (640, 480))]


def _best_frame_detections(frames, detector):
    """Run a detector over frames and keep the frame with the highest summed conf."""
    best = None
    best_score = -1.0
    for fr in frames:
        res = detector.predict(fr)
        score = sum(d.confidence for d in res.detections)
        if score > best_score:
            best_score = score
            best = (fr, res)
    return best if best else (frames[0], detector.predict(frames[0]))


# ---------------------------------------------------------------------------
# Perception: read container id + iso + plate from media
# ---------------------------------------------------------------------------
def read_container_and_iso(frames, demo_hint: Optional[str] = None) -> Dict:
    models = get_models()
    frame, res = _best_frame_detections(frames, models["container_iso"])
    ocr = models["ocr"]
    out = {"container_id": "", "iso_type": "", "backend": res.backend, "detections": []}
    print(f"\n[perception] CONTAINER/ISO | detector backend={res.backend} | "
          f"ocr backend={ocr.backend} | frame size={frame.size} | "
          f"raw detections={len(res.detections)}")
    # Keep only the single highest-confidence box per label so we don't OCR
    # (and overwrite with) a weaker duplicate detection.
    best_by_label = {}
    for d in res.detections:
        if d.label not in best_by_label or d.confidence > best_by_label[d.label].confidence:
            best_by_label[d.label] = d
    for label, d in best_by_label.items():
        crop = crop_detection(frame, d.bbox)
        text = ocr.read(crop)
        if d.label == "container_id":
            # The container detector has ONE class ('container-number'), so its
            # crop may contain both the container number AND the ISO size-type
            # code. Split them apart via the ISO 6346 formats.
            cid, iso = ocr.split_container_iso(text)
            out["container_id"] = cid
            if iso and not out["iso_type"]:
                out["iso_type"] = iso
            print(f"[perception]   {d.label:<12} conf={d.confidence:.3f} bbox={d.bbox} "
                  f"crop={crop.size} -> OCR raw='{text}' -> id='{cid}' iso='{iso}'")
        elif d.label == "iso_type":
            iso = ocr.normalize_iso(text)
            if iso:
                out["iso_type"] = iso
            print(f"[perception]   {d.label:<12} conf={d.confidence:.3f} bbox={d.bbox} "
                  f"crop={crop.size} -> OCR raw='{text}' -> iso='{iso}'")
        else:
            print(f"[perception]   {d.label:<12} conf={d.confidence:.3f} bbox={d.bbox} "
                  f"crop={crop.size} -> OCR raw='{text}'")
        out["detections"].append({"label": d.label, "conf": d.confidence, "text": text})
    print(f"[perception] => container_id='{out['container_id']}' iso_type='{out['iso_type']}'")
    # demo fallback: if OCR is mocked (empty), use the hinted/known record
    if not out["container_id"] and demo_hint:
        rec = DB.containers.get(demo_hint)
        if rec:
            out["container_id"] = rec["container_id"]
            out["iso_type"] = rec["iso_type"]
    return out


def read_plate(frames, demo_hint: Optional[str] = None) -> Dict:
    models = get_models()
    frame, res = _best_frame_detections(frames, models["plate"])
    ocr = models["ocr"]
    out = {"plate": "", "backend": res.backend}
    print(f"\n[perception] PLATE | detector backend={res.backend} | "
          f"ocr backend={ocr.backend} | frame size={frame.size} | "
          f"raw detections={len(res.detections)}")
    # Use only the highest-confidence plate box.
    plate_dets = [d for d in res.detections if d.label == "plate"] or res.detections
    if plate_dets:
        d = max(plate_dets, key=lambda x: x.confidence)
        crop = crop_detection(frame, d.bbox)
        raw_text = ocr.read(crop)
        out["plate"] = ocr.normalize_plate(raw_text)
        print(f"[perception]   {d.label:<12} conf={d.confidence:.3f} bbox={d.bbox} "
              f"crop={crop.size} -> OCR raw='{raw_text}'")
    print(f"[perception] => plate='{out['plate']}'")
    if not out["plate"] and demo_hint:
        out["plate"] = normalize_plate(demo_hint)
    return out


def detect_damage(frames) -> Dict:
    models = get_models()
    frame, res = _best_frame_detections(frames, models["damage"])
    findings = [{"type": d.label, "conf": round(d.confidence, 2), "bbox": d.bbox} for d in res.detections]
    print(f"\n[perception] DAMAGE | detector backend={res.backend} | "
          f"frame size={frame.size} | findings={len(findings)}")
    for f in findings:
        print(f"[perception]   {f['type']:<12} conf={f['conf']} bbox={f['bbox']}")
    return {"findings": findings, "backend": res.backend}


# ---------------------------------------------------------------------------
# 1) GATE INSPECTION & VERIFICATION (gate-in)
# ---------------------------------------------------------------------------
def gate_inspection(plate: str, container_id: str, iso_type: str,
                    arrives_with_container: bool) -> Dict:
    """Gate-in logic for empty arrivals and return-plus-pickup movements.

    Empty trucks are resolved primarily by plate: all pickup orders assigned to
    that plate are searched and the first eligible one is selected.  On a
    return leg, plate + container ID must match the same return order; that
    order is persisted as ``returned`` and the same plate is also searched for
    a different, eligible pickup container.
    """
    plate = normalize_plate(plate)
    container_id = (container_id or "").strip().upper()
    report = {"checks": [], "decision": "DENY", "ticket": None, "notes": [],
              "matched_container": "", "returned_container": "",
              "pickup_candidates": []}

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        return ok

    ros = DB.ro_for_plate(plate)
    plate_known = plate in DB.trucks
    truck_display = DB.trucks.get(plate, {}).get("plate", plate)
    check("Truck plate recognised in fleet", plate_known, truck_display)

    return_ro = None
    if arrives_with_container:
        if not container_id:
            check("Return container ID was read", False, "Plate read but no container ID")
            return _finalize_gate(report, plate, container_id, iso_type, None)
        return_ro = next((r for r in ros
                          if r["movement_type"] == "return"
                          and r["container_id"].upper() == container_id), None)
        check("Plate and container match a return order", return_ro is not None,
              f"plate={truck_display}; container={container_id}")
        if not return_ro:
            return _finalize_gate(report, plate, container_id, iso_type, None)
        already_returned = return_ro["ro_status"] == "returned"
        check("Return has not already been recorded", not already_returned,
              return_ro["release_order_id"])
        if already_returned:
            return _finalize_gate(report, plate, container_id, iso_type, return_ro)
        DB.mark_returned(return_ro)
        report["returned_container"] = container_id
        report["notes"].append(
            f"Marked {container_id} as returned; searching this plate for another pickup.")

    pickup_ros = [r for r in ros
                  if r["movement_type"] == "pickup"
                  and (not arrives_with_container
                       or r["container_id"].upper() != container_id)]
    report["pickup_candidates"] = [r["container_id"] for r in pickup_ros]

    def pickup_ready(candidate):
        cont = DB.containers.get(candidate["container_id"], {})
        return (candidate["ro_status"] == "processed"
                and candidate["customs_cleared"] == "True"
                and candidate["payment_status"] == "paid"
                and cont.get("light_status") == "green")

    if not arrives_with_container and container_id:
        # A supplied ID is a selection hint, never a reason to ignore the plate.
        ro = next((r for r in pickup_ros if r["container_id"].upper() == container_id), None)
    else:
        ro = next((r for r in pickup_ros if pickup_ready(r)), None)

    if ro is None and not arrives_with_container:
        detail = (f"No pickup order for plate {truck_display}"
                  if not pickup_ros else
                  f"No eligible pickup; candidates: {', '.join(report['pickup_candidates'])}")
        check("Matching pickup order found for plate", False, detail)
        return _finalize_gate(report, plate, container_id, iso_type, None)

    # A valid return may enter even if there is no ready onward pickup.
    if ro is None:
        if pickup_ros:
            report["notes"].append(
                "Other pickup orders match the plate, but none is currently eligible.")
        else:
            report["notes"].append("No additional pickup order matches this plate.")
        return _finalize_gate(report, plate, container_id, iso_type, return_ro)

    report["matched_container"] = ro["container_id"]
    check("Matching pickup order found for plate", True,
          f"{ro['release_order_id']} -> {ro['container_id']}")
    check("Plate matches importer-declared plate",
          normalize_plate(ro["declared_plate"]) == plate,
          f"declared={ro['declared_plate']} read={truck_display}")
    check("Release order processed", ro["ro_status"] == "processed", ro["ro_status"])
    check("Customs cleared", ro["customs_cleared"] == "True")
    check("Payment settled", ro["payment_status"] == "paid", ro["payment_status"])

    # Container light condition
    cont = DB.containers.get(ro["container_id"], {})
    light = cont.get("light_status", "unknown")
    check("Container is GREEN light", light == "green", f"light={light}")

    # Nearby-county / transit routing via importer TIN
    imp = DB.importers.get(ro["importer_tin"], {})
    nearby = imp.get("nearby_county") == "True"
    is_transit = imp.get("is_transit") == "True"
    routing = ("NEARBY delivery (" + imp.get("region", "?") + ")") if nearby else (
        "TRANSIT to " + imp.get("delivery_country", "?") if is_transit else
        "UPCOUNTRY (" + imp.get("region", "?") + ")")
    report["notes"].append(f"TIN {ro.get('importer_tin')} -> {imp.get('importer_name','?')}: {routing}")
    report["routing"] = routing

    return _finalize_gate(report, plate, ro["container_id"],
                          iso_type or ro.get("iso_type", ""), ro)


def _finalize_gate(report, plate, container_id, iso_type, ro) -> Dict:
    all_pass = all(c["pass"] for c in report["checks"]) and len(report["checks"]) > 0
    report["decision"] = "ALLOW" if all_pass else "DENY"
    if all_pass:
        ticket_id = f"GIN-{dt.datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
        report["ticket"] = ticket_id
        _append_csv("gate_tickets.csv", [
            ticket_id, "GATE_IN", dt.datetime.now().isoformat(timespec="seconds"),
            DB.trucks.get(normalize_plate(plate), {}).get("plate", plate), container_id, iso_type,
            ro["release_order_id"] if ro else "", "ALLOW",
            report.get("routing", ""),
        ])
    return report


# ---------------------------------------------------------------------------
# 2) INSPECTION PHASE (CV + LLM report, deep inspection)
# ---------------------------------------------------------------------------
def llm_report(container_id: str, iso_type: str, light: str, damage: Dict) -> str:
    """Generate a short, simple inspection report.

    Tries an OpenAI-compatible LLM if OPENAI_API_KEY is set; otherwise uses a
    clean rule-based template so the pipeline is fully offline-capable.
    """
    findings = damage.get("findings", [])
    finding_str = ", ".join(f"{f['type']} ({f['conf']})" for f in findings) or "no visible defects"
    prompt = (
        f"Write a 2-3 sentence container inspection note. Container {container_id}, "
        f"ISO {iso_type}, current light {light}. CV findings: {finding_str}. "
        f"Recommend whether the yellow light can be cleared to green or needs deep inspection."
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=160,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass
    # rule-based fallback
    severe = any(f["type"] == "hole" for f in findings)
    moderate = any(f["type"] == "dent" for f in findings)
    if severe:
        verdict = "Hole detected — escalate to DEEP INSPECTION; do not clear."
    elif moderate:
        verdict = "Dent present — recommend deep inspection before clearing."
    elif findings:
        verdict = "Only surface rust — minor; light can be cleared to GREEN after wash/repair note."
    else:
        verdict = "No defects found — yellow light can be cleared to GREEN."
    return (f"Container {container_id} (ISO {iso_type}) was inspected. "
            f"CV findings: {finding_str}. {verdict}")


def inspection_phase(frames, container_id: str, iso_type: str) -> Dict:
    cont = DB.containers.get(container_id, {})
    light = cont.get("light_status", "yellow")
    damage = detect_damage(frames)
    report_text = llm_report(container_id, iso_type, light, damage)
    deep_needed = any(f["type"] in ("hole", "dent") for f in damage.get("findings", []))
    return {
        "container_id": container_id,
        "iso_type": iso_type,
        "light_before": light,
        "damage": damage,
        "report": report_text,
        "deep_inspection_suggested": deep_needed,
        "cleared_to": "green" if (light == "yellow" and not deep_needed) else light,
    }


# ---------------------------------------------------------------------------
# 3) GATE-OUT INSPECTION
# ---------------------------------------------------------------------------
def gate_out_inspection(plate: str, container_id: str, iso_type: str,
                        gate_in_ticket: str) -> Dict:
    report = {"checks": [], "decision": "DENY", "ticket": None, "notes": []}
    plate = normalize_plate(plate)
    container_id = (container_id or "").strip().upper()
    gate_in_ticket = (gate_in_ticket or "").strip()

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
        return ok

    # The printed gate-in ticket is the source of truth for gate-out. It must
    # be an actual persisted ALLOW ticket, not merely a string starting GIN-.
    ticket = DB.gate_ticket(gate_in_ticket)
    check("Gate-in ticket exists in ticket log", ticket is not None, gate_in_ticket)
    valid_gate_in = bool(ticket and ticket["ticket_type"] == "GATE_IN"
                         and ticket["decision"] == "ALLOW")
    check("Ticket is an approved gate-in ticket", valid_gate_in,
          ticket["ticket_type"] if ticket else "not found")
    check("Gate-in ticket was verified and used at gate-in",
          bool(ticket) and DB.ticket_is_used(gate_in_ticket), gate_in_ticket)
    check("Gate-in ticket has no previous gate-out",
          bool(ticket) and not DB.gate_in_has_gate_out(gate_in_ticket), gate_in_ticket)

    ticket_plate_match = bool(ticket and normalize_plate(ticket["plate"]) == plate)
    check("Observed plate matches gate-in ticket", ticket_plate_match,
          f"ticket={ticket['plate'] if ticket else '?'} read={plate or 'invalid'}")
    ticket_container_match = bool(ticket and ticket["container_id"].upper() == container_id)
    check("Picked container matches gate-in ticket", ticket_container_match,
          f"ticket={ticket['container_id'] if ticket else '?'} read={container_id or 'missing'}")

    ro = DB.ro_by_id(ticket["release_order_id"]) if ticket else None
    check("Ticket links to the pickup release order",
          bool(ro and ro["movement_type"] == "pickup"),
          ticket["release_order_id"] if ticket else "")
    check("Release order is for the picked container",
          bool(ro and ro["container_id"].upper() == container_id), container_id)
    check("Truck picked the RIGHT container (plate matches RO)",
          bool(ro and normalize_plate(ro["declared_plate"]) == plate),
          f"declared={ro['declared_plate'] if ro else '?'}")

    # Release-order, customs, payment and green-light conditions were already
    # frozen into the approved gate-in ticket. At gate-out we only need to
    # prove that the same truck collected the same assigned container using
    # that verified ticket. Rechecking mutable fields here could incorrectly
    # deny an otherwise matching trip after gate-in had already approved it.
    if ticket_plate_match and ticket_container_match:
        report["notes"].append(
            "Gate-out plate and container match the verified gate-in record.")
    all_pass = bool(report["checks"]) and all(c["pass"] for c in report["checks"])
    report["decision"] = "ALLOW" if all_pass else "DENY"
    if all_pass:
        ticket_id = f"GOUT-{dt.datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
        report["ticket"] = ticket_id
        _append_csv("gate_tickets.csv", [
            ticket_id, "GATE_OUT", dt.datetime.now().isoformat(timespec="seconds"),
            plate, container_id, iso_type,
            ro["release_order_id"] if ro else "", "ALLOW",
            f"from gate-in {gate_in_ticket}",
        ])
    return report


# ---------------------------------------------------------------------------
# 4) YARD / CONTAINER LOCATION TRACKING
# ---------------------------------------------------------------------------
def locate_container(container_id: str) -> Dict:
    cont = DB.containers.get(container_id)
    if not cont:
        return {"found": False}
    return {
        "found": True,
        "container_id": container_id,
        "iso_type": cont["iso_type"],
        "terminal": cont["terminal"],
        "plot": cont["plot"],
        "light": cont["light_status"],
    }


def move_container(container_id: str, to_terminal: str, to_plot: str, moved_by: str = "system") -> Dict:
    cont = DB.containers.get(container_id)
    if not cont:
        return {"ok": False, "error": "container not found"}
    from_terminal, from_plot = cont["terminal"], cont["plot"]
    _append_csv("movement_log.csv", [
        dt.datetime.now().isoformat(timespec="seconds"),
        container_id, from_terminal, from_plot, to_terminal, to_plot, moved_by,
    ])
    # update in-memory + persist containers.csv
    cont["terminal"] = to_terminal
    cont["plot"] = to_plot
    _rewrite_containers()
    return {"ok": True, "from": f"{from_terminal} / {from_plot}", "to": f"{to_terminal} / {to_plot}"}


def _rewrite_containers():
    path = os.path.join(DATA_DIR, "containers.csv")
    fields = ["container_id", "iso_type", "light_status", "damage_status", "terminal", "plot", "in_yard"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in DB.containers.values():
            w.writerow({k: r.get(k, "") for k in fields})
