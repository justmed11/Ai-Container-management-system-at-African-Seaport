"""Gradio UI for the AI Container Management system (Port of Dar es Salaam).

Four tabs map to the four functional areas:
  1. Gate Inspection & Verification
  2. Inspection Phase (CV + LLM)
  3. Gate-Out Inspection
  4. Yard / Container Location Tracking

Media input: on each media-driven tab, click "➕ Add a picture/video" to reveal
the options -- browse a file, or use the camera to take a picture OR a video
(one unified camera feature, switched with a radio). Captured media is previewed
first (retake if needed); pressing the tab's Run button sends it.

The Yard tab can additionally READ the container ID from an image/video via the
detector + OCR pipeline -- including reusing the last image processed at a gate
terminal -- while still allowing the container ID to be typed manually.
Run:  python app.py
"""
from __future__ import annotations

import json
import os
import warnings

# --- Silence harmless third-party log noise -------------------------------
# These are library-internal deprecation/user warnings (Gradio<->Starlette
# version drift and the transformers max_length notice). They do not affect
# functionality, so we hide them to keep the console readable.
warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")
warnings.filterwarnings("ignore", message=r".*model-agnostic default `max_length`.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gradio as gr

import pipeline as P

# Cache of the most recent media processed at a gate (Tab 1 / Tab 3) plus the
# detector/OCR read from it. The Yard tab can reuse this "last gate-terminal
# image" to resolve a container ID without re-uploading anything.
_LAST_GATE_MEDIA = {"path": None, "container_id": "", "iso_type": ""}


def _fmt_checks(report: dict) -> str:
    lines = []
    for c in report.get("checks", []):
        mark = "✅" if c["pass"] else "❌"
        detail = f" — {c['detail']}" if c.get("detail") else ""
        lines.append(f"{mark} {c['check']}{detail}")
    for n in report.get("notes", []):
        lines.append(f"ℹ️ {n}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared media input helpers
# ---------------------------------------------------------------------------
def _pick_media(uploaded, cam_photo, cam_video):
    """Return the first media source the user actually provided.

    Priority: a freshly captured photo, then a recorded video, then an uploaded
    file. All three yield a filesystem path that ``load_media_frames`` handles.
    """
    for candidate in (cam_photo, cam_video, uploaded):
        if candidate:
            return candidate
    return None


def _toggle_media(is_open):
    """Show/hide the picture/video options when the add button is clicked."""
    now_open = not is_open
    return (
        now_open,
        gr.update(visible=now_open),
        gr.update(value="➖ Hide picture/video options"
                  if now_open else "➕ Add a picture/video"),
    )


def _switch_capture_mode(mode):
    """Show only the chosen camera widget and clear the other's stale capture."""
    is_photo = mode == "Take a picture"
    return (
        gr.update(visible=is_photo, value=None),      # cam_photo
        gr.update(visible=not is_photo, value=None),  # cam_video
    )


def media_input_block():
    """Render an "Add a picture/video" button that reveals the media options.

    Clicking the button toggles a panel offering:
      * Browse a picture/video from disk (gr.File).
      * A single camera feature with a "Take a picture" / "Take a video" radio
        that swaps between the webcam photo widget (gr.Image) and the webcam
        video widget (gr.Video). Only the selected one is shown, so picture and
        video capture feel like one unified feature.

    The webcam widgets show the captured photo/recorded clip as a preview with a
    clear/redo control, so nothing is sent until the tab's Run button is
    pressed. Returns (upload, cam_photo, cam_video) component handles.
    """
    add_btn = gr.Button("➕ Add a picture/video", variant="secondary")
    with gr.Column(visible=False) as options:
        gr.Markdown("**Browse** a file, or use your **camera** to take a picture "
                    "or a video (switch with the toggle). Captured media is "
                    "previewed here first — retake if needed, then press **Run** "
                    "to send it.")
        upload = gr.File(label="Browse a picture/video",
                         file_types=["image", "video"])
        capture_mode = gr.Radio(
            ["Take a picture", "Take a video"],
            value="Take a picture",
            label="📸 Camera capture",
        )
        cam_photo = gr.Image(sources=["webcam"], type="filepath", visible=True,
                             label="📷 Take a picture (preview before sending)")
        cam_video = gr.Video(sources=["webcam"], visible=False,
                             label="🎥 Take a video (preview before sending)")
        capture_mode.change(_switch_capture_mode, [capture_mode],
                            [cam_photo, cam_video])
    is_open = gr.State(False)
    add_btn.click(_toggle_media, [is_open], [is_open, options, add_btn])
    return upload, cam_photo, cam_video


# ---------------------------------------------------------------------------
# Tab 1: Gate Inspection & Verification
# ---------------------------------------------------------------------------
def run_gate(uploaded, cam_photo, cam_video, arrives_with_container, demo_container, demo_plate):
    media = _pick_media(uploaded, cam_photo, cam_video)
    frames = P.load_media_frames(media)
    if not frames:
        return "No media provided.", "", "", "", ""
    plate_info = P.read_plate(frames, demo_hint=demo_plate or None)
    # An empty truck is identified by plate first.  Container OCR is only
    # required when the operator explicitly marks a return leg.
    if arrives_with_container:
        perc = P.read_container_and_iso(frames, demo_hint=demo_container or None)
    else:
        perc = {"container_id": (demo_container or "").strip().upper(),
                "iso_type": "", "backend": plate_info["backend"]}
    cid, iso = perc["container_id"], perc["iso_type"]
    plate = plate_info["plate"]
    if media:
        _LAST_GATE_MEDIA.update({"path": media, "container_id": cid, "iso_type": iso})
    report = P.gate_inspection(plate, cid, iso, arrives_with_container)

    perception = (f"Detector backend: {perc['backend']} | OCR demo-hint used: "
                  f"{'yes' if (demo_container or demo_plate) else 'no'}\n"
                  f"Plate: {plate}\nContainer ID: {cid}\nISO type: {iso}")
    decision = f"DECISION: {report['decision']}"
    if report.get("ticket"):
        decision += f"\n🎫 Gate-in ticket: {report['ticket']}"
    if report.get("routing"):
        decision += f"\nRouting: {report['routing']}"
    if report.get("returned_container"):
        decision += f"\nReturned: {report['returned_container']}"
    if report.get("matched_container"):
        decision += f"\nPickup matched: {report['matched_container']}"
    elif report.get("pickup_candidates"):
        decision += f"\nOther plate matches: {', '.join(report['pickup_candidates'])}"
    return perception, _fmt_checks(report), decision, report.get("ticket") or "", ""


def mark_gate_in_ticket_used(ticket_id):
    result = P.mark_ticket_used(ticket_id, "GATE_IN")
    status = ("✅ " if result["ok"] else "❌ ") + result["message"]
    # A verified gate-in ticket is copied directly to the gate-out form.
    return status, ticket_id if result["ok"] else ""


# ---------------------------------------------------------------------------
# Tab 2: Inspection Phase
# ---------------------------------------------------------------------------
def run_inspection(uploaded, cam_photo, cam_video, demo_container):
    media = _pick_media(uploaded, cam_photo, cam_video)
    frames = P.load_media_frames(media)
    if not frames:
        return "No media provided.", "", ""
    perc = P.read_container_and_iso(frames, demo_hint=demo_container or None)
    cid = perc["container_id"] or demo_container
    iso = perc["iso_type"]
    result = P.inspection_phase(frames, cid, iso)
    damage_str = "\n".join(
        f"• {f['type']} (conf {f['conf']})" for f in result["damage"]["findings"]
    ) or "No defects detected."
    summary = (f"Container {result['container_id']} | ISO {result['iso_type']}\n"
               f"Light before: {result['light_before']} -> after: {result['cleared_to']}\n"
               f"Deep inspection suggested: {result['deep_inspection_suggested']}")
    return damage_str, result["report"], summary


# ---------------------------------------------------------------------------
# Tab 3: Gate-Out Inspection
# ---------------------------------------------------------------------------
def run_gate_out(uploaded, cam_photo, cam_video, gate_in_ticket, demo_container, demo_plate):
    media = _pick_media(uploaded, cam_photo, cam_video)
    frames = P.load_media_frames(media)
    if not frames:
        return "No media provided.", "", "", "", ""
    perc = P.read_container_and_iso(frames, demo_hint=demo_container or None)
    plate_info = P.read_plate(frames, demo_hint=demo_plate or None)
    cid, iso, plate = perc["container_id"], perc["iso_type"], plate_info["plate"]
    if media:
        _LAST_GATE_MEDIA.update({"path": media, "container_id": cid, "iso_type": iso})
    report = P.gate_out_inspection(plate, cid, iso, gate_in_ticket)
    perception = f"Plate: {plate}\nContainer ID: {cid}\nISO type: {iso}"
    decision = f"DECISION: {report['decision']}"
    if report.get("ticket"):
        decision += f"\n🎫 Gate-out ticket: {report['ticket']}"
    return perception, _fmt_checks(report), decision, report.get("ticket") or "", ""


def mark_gate_out_ticket_used(ticket_id):
    result = P.mark_ticket_used(ticket_id, "GATE_OUT")
    return ("✅ " if result["ok"] else "❌ ") + result["message"]


# ---------------------------------------------------------------------------
# Tab 4: Yard / Location Tracking
# ---------------------------------------------------------------------------
def run_locate(container_id):
    info = P.locate_container(container_id.strip())
    if not info["found"]:
        return "Container not found in yard records."
    return (f"Container {info['container_id']} (ISO {info['iso_type']})\n"
            f"Terminal: {info['terminal']}\nPlot/Block: {info['plot']}\n"
            f"Light: {info['light']}")


def run_move(container_id, to_terminal, to_plot):
    res = P.move_container(container_id.strip(), to_terminal.strip(), to_plot.strip())
    if not res.get("ok"):
        return f"Error: {res.get('error')}"
    return f"Moved {container_id}\nFrom: {res['from']}\nTo:   {res['to']}\n(movement_log.csv updated)"


def read_container_id_from_media(uploaded, cam_photo, cam_video, use_last_gate):
    """Resolve a container ID for the Yard tab via detector + OCR.

    Sources, in order of the user's choice:
      * "Use last image from a gate terminal" -> reuse the cached gate read (or
        re-run OCR on that cached image if only the path is available).
      * Otherwise the image/video the user browsed or captured on this tab.
    Returns (container_id, status_message). The container ID feeds the existing
    Container ID textbox, so the user can still edit or type it manually.
    """
    if use_last_gate:
        if _LAST_GATE_MEDIA.get("container_id"):
            return _LAST_GATE_MEDIA["container_id"], (
                "Using last gate-terminal read:\n"
                f"Container ID: {_LAST_GATE_MEDIA['container_id']}\n"
                f"ISO type: {_LAST_GATE_MEDIA.get('iso_type') or '—'}")
        if _LAST_GATE_MEDIA.get("path"):
            media, src = _LAST_GATE_MEDIA["path"], "last gate-terminal image"
        else:
            return "", ("No gate-terminal image yet. Run a gate or gate-out inspection "
                        "first, or provide an image/video here (or type the ID).")
    else:
        media = _pick_media(uploaded, cam_photo, cam_video)
        src = "provided image/video"
        if not media:
            return "", "Provide an image/video, tick 'use last gate image', or type the ID."
    frames = P.load_media_frames(media)
    if not frames:
        return "", "Could not read the media. Try another file or type the container ID."
    perc = P.read_container_and_iso(frames)
    cid = perc["container_id"]
    if not cid:
        return "", (f"Could not read a container ID from the {src}. "
                    "Type it manually, or add model weights for real OCR.")
    return cid, (f"Read from {src} (detector: {perc['backend']}):\n"
                 f"Container ID: {cid}\nISO type: {perc['iso_type'] or '—'}")


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------
TERMINALS = [
    "TICTS Container Terminal",
    "TPA Berth 1-7 (General Cargo)",
    "TPA ICD Kurasini",
    "Ubungo ICD",
]

with gr.Blocks(title="AI Container Management — Port of Dar es Salaam") as demo:
    gr.Markdown("#  AI Container Management — Port of Dar es Salaam\n"
                "End-to-end pipeline: container-ID/ISO detection, damage detection, "
                "plate detection, TrOCR reading, gate logic, inspection and yard tracking. "
                "On each tab, click **➕ Add a picture/video** to browse a file or take a "
                "picture/video with your camera. Captured media is **previewed first** "
                "(retake if needed); press **Run** to send it.")

    with gr.Tab("1️⃣ Gate Inspection & Verification"):
        gr.Markdown("For an empty arrival, the system reads the plate and finds its pickup "
                    "container in the CSV data. Tick **return leg** when a truck arrives with "
                    "a container: the system reads plate + container ID, records the return, "
                    "then searches the same plate for a different pickup container.")
        with gr.Row():
            with gr.Column():
                g_media, g_cam_photo, g_cam_video = media_input_block()
            with gr.Column():
                g_with = gr.Checkbox(label="Truck arrives WITH a container (return leg)", value=False)
                g_demo_c = gr.Textbox(label="Demo container ID (used if OCR weights absent)")
                g_demo_p = gr.Textbox(label="Demo plate (used if OCR weights absent)")
                g_btn = gr.Button("Run gate inspection", variant="primary")
        g_perc = gr.Textbox(label="Perception (plate / container / ISO)", lines=4)
        g_checks = gr.Textbox(label="Verification checks", lines=10)
        g_decision = gr.Textbox(label="Decision & ticket", lines=4)
        with gr.Row():
            g_issued_ticket = gr.Textbox(label="Issued gate-in ticket", interactive=False)
            g_use_btn = gr.Button("Verify and use gate-in ticket", variant="secondary")
        g_use_status = gr.Textbox(label="Ticket log status", interactive=False)
        g_btn.click(run_gate, [g_media, g_cam_photo, g_cam_video, g_with, g_demo_c, g_demo_p],
                    [g_perc, g_checks, g_decision, g_issued_ticket, g_use_status])

    with gr.Tab("2️⃣ Inspection Phase"):
        gr.Markdown("General CV inspection (rust / dent / hole) + short LLM report per "
                    "container ID + ISO type. Helps clear **yellow-light** containers; "
                    "flags deep inspection when needed.")
        with gr.Row():
            with gr.Column():
                i_media, i_cam_photo, i_cam_video = media_input_block()
            with gr.Column():
                i_demo_c = gr.Textbox(label="Demo container ID (if OCR weights absent)")
                i_btn = gr.Button("Run inspection", variant="primary")
        i_damage = gr.Textbox(label="CV damage findings", lines=6)
        i_report = gr.Textbox(label="LLM inspection report", lines=6)
        i_summary = gr.Textbox(label="Light status / deep inspection", lines=4)
        i_btn.click(run_inspection, [i_media, i_cam_photo, i_cam_video, i_demo_c],
                    [i_damage, i_report, i_summary])

    with gr.Tab("3️⃣ Gate-Out Inspection"):
        gr.Markdown("Verify the truck picked the RIGHT container, all release orders present, "
                    "green-light condition, and a valid gate-in ticket — then issue the gate-out ticket.")
        with gr.Row():
            with gr.Column():
                o_media, o_cam_photo, o_cam_video = media_input_block()
            with gr.Column():
                o_ticket = gr.Textbox(label="Gate-in ticket ID (GIN-...)")
                o_demo_c = gr.Textbox(label="Demo container ID (if OCR weights absent)")
                o_demo_p = gr.Textbox(label="Demo plate (if OCR weights absent)")
                o_btn = gr.Button("Run gate-out inspection", variant="primary")
        o_perc = gr.Textbox(label="Perception", lines=3)
        o_checks = gr.Textbox(label="Verification checks", lines=8)
        o_decision = gr.Textbox(label="Decision & ticket", lines=3)
        with gr.Row():
            o_issued_ticket = gr.Textbox(label="Issued gate-out ticket", interactive=False)
            o_use_btn = gr.Button("Verify and use gate-out ticket", variant="secondary")
        o_use_status = gr.Textbox(label="Ticket log status", interactive=False)
        o_btn.click(run_gate_out, [o_media, o_cam_photo, o_cam_video, o_ticket, o_demo_c, o_demo_p],
                    [o_perc, o_checks, o_decision, o_issued_ticket, o_use_status])
        o_use_btn.click(mark_gate_out_ticket_used, [o_issued_ticket], [o_use_status])

        # Verifying the gate-in ticket automatically transfers it to gate-out.
        # The gate-out inspection then issues GOUT automatically when the
        # detected plate and container match that verified gate-in record.
        g_use_btn.click(mark_gate_in_ticket_used, [g_issued_ticket],
                        [g_use_status, o_ticket])

    with gr.Tab("4️⃣ Yard / Location Tracking"):
        gr.Markdown("Track where a container is. Yard is divided into lettered blocks + "
                    "numbered rows/slots/tiers (e.g. `B07-3-2`). Record moves between terminals/blocks.\n\n"
                    "You can **type the container ID**, or **read it from an image/video** "
                    "(detector + OCR) — including the **last image processed at a gate terminal**.")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**Locate**")
                l_cid = gr.Textbox(label="Container ID (type it, or read it from media below)")
                l_use_last = gr.Checkbox(label="Use last image from a gate terminal", value=False)
                l_media, l_cam_photo, l_cam_video = media_input_block()
                l_read = gr.Button("🔎 Read container ID from image")
                l_read_status = gr.Textbox(label="OCR read", lines=3)
                l_btn = gr.Button("Locate", variant="primary")
                l_out = gr.Textbox(label="Current location", lines=5)
                l_read.click(read_container_id_from_media,
                             [l_media, l_cam_photo, l_cam_video, l_use_last],
                             [l_cid, l_read_status])
                l_btn.click(run_locate, [l_cid], [l_out])
            with gr.Column():
                gr.Markdown("**Move / re-stack**")
                m_cid = gr.Textbox(label="Container ID (type it, or read it from media below)")
                m_use_last = gr.Checkbox(label="Use last image from a gate terminal", value=False)
                m_media, m_cam_photo, m_cam_video = media_input_block()
                m_read = gr.Button("🔎 Read container ID from image")
                m_read_status = gr.Textbox(label="OCR read", lines=3)
                m_term = gr.Dropdown(TERMINALS, label="To terminal")
                m_plot = gr.Textbox(label="To plot/block (e.g. C12-4-1)")
                m_btn = gr.Button("Record move", variant="primary")
                m_out = gr.Textbox(label="Result", lines=5)
                m_read.click(read_container_id_from_media,
                             [m_media, m_cam_photo, m_cam_video, m_use_last],
                             [m_cid, m_read_status])
                m_btn.click(run_move, [m_cid, m_term, m_plot], [m_out])


def _find_free_port(preferred: int, host: str = "0.0.0.0", tries: int = 50) -> int:
    """Return the first bindable port at/after `preferred`.

    Avoids the '[Errno 10048] only one usage of each socket address' crash that
    happens when 7860 is still held by a previous run or another program.
    """
    import socket

    for candidate in range(preferred, preferred + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue
    # Fall back to letting the OS pick any free port.
    return 0


if __name__ == "__main__":
    # Default to 127.0.0.1 so the printed link is actually browsable.
    # NOTE: do NOT browse to http://0.0.0.0:PORT -- 0.0.0.0 is a "bind to all
    # interfaces" address, not a reachable one, and Windows browsers show
    # "site can't be reached". Use localhost / 127.0.0.1 instead.
    # Set GRADIO_SERVER_NAME=0.0.0.0 only if you need LAN/other-device access.
    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    port = _find_free_port(preferred_port, host)
    if port != preferred_port:
        print(f"[app] Port {preferred_port} busy; using free port {port or 'auto'} instead.")
    open_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print(f"[app] Open this in your browser: {{http://{open_host}}}:{port or 7860}")
    # inbrowser=True auto-opens the correct URL; server_port=None lets the OS
    # pick a port if nothing in the scanned range was bindable.
    demo.launch(server_name=host, server_port=port or None, inbrowser=True)
