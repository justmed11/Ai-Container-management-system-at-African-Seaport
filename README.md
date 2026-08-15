# Ai-Container-management-system-at-African-Seaport
The project is about a system that uses computer vision model to catch records from containers id picking up and returning truck license plates, normal inspecting containers inspection using computer vision models. All of these initially were done manually that are error prone on busy days at ports and time taking that the system aim to fix.

# AI Container Management — Port of Dar es Salaam

Research project: **Design of Artificial Intelligence on Containers Management**.

An end-to-end computer-vision + OCR + LLM pipeline that automates gate
operations, container inspection and yard tracking at a container terminal,
modelled on the Port of Dar es Salaam (TPA / TICTS).

## Models

| # | Model | Task | Suggested backbone |
|---|-------|------|--------------------|
| 1 | **Container-ID + ISO detector** | locate the container number + ISO-type markings | YOLOv8 |
| 2 | **Damage detector** | detect `rust`, `dent`, `hole` | YOLOv8-seg |
| 3 | **Plate detector** | locate the truck license plate | YOLOv8 |
| 4 | **TrOCR reader** | read container ID, ISO type and plate text | `microsoft/trocr-base-printed` (fine-tuned) |

A shared **cropping step** (`crop_detection`) feeds each detector box into TrOCR.

## Pipeline stages (Gradio tabs)

1. **Gate Inspection & Verification** — plate vs importer-declared plate, release
   order processed, customs cleared, payment, green-light, return/pickup logic,
   and TIN-based nearby-county / transit routing → auto gate-in ticket.

   Plate matching is space-insensitive (`T 123 ABC` in CSV matches OCR
   `T123ABC`). Empty arrivals are resolved by plate to an eligible pickup
   container. For a checked return leg, the plate and container ID must match
   the same return order; the order is marked `returned`, then the plate is
   searched for a different eligible pickup container.

   OCR plate text is accepted only when it matches the fleet's Tanzanian plate
   format (`T 123 ABC`, canonicalized as `T123ABC`). Gate-out requires the
   exact approved gate-in ticket from `gate_tickets.csv`; its plate, assigned
   pickup container, and release order must all match the observed gate-out.
   After gate-in passes, **Verify and use gate-in ticket** logs the use event
   and automatically copies that ticket ID into the gate-out form. When the
   gate-out camera reads the same plate and assigned container, the system
   automatically allows the movement and issues the `GOUT-...` ticket. Checks
   already approved at gate-in are not redundantly re-evaluated at gate-out.
   The gate-out ticket can then be verified and logged with its own use button.
2. **Inspection Phase** — CV damage inspection + short LLM report per container,
   clears yellow lights, suggests deep inspection.
3. **Gate-Out Inspection** — verifies the right container was picked, orders
   present, green light, valid gate-in ticket → gate-out ticket.
4. **Yard / Location Tracking** — lettered blocks + numbered rows/slots/tiers
   (e.g. `B07-3-2`); records every move between terminals/blocks.

All media inputs accept **images and videos** (videos are frame-sampled).

## Run

```bash
pip install -r requirements.txt
python app.py        # opens http://localhost:7860
```

### Performance settings

The app automatically runs YOLO and TrOCR on CUDA when PyTorch detects an
NVIDIA GPU. If GPU memory is exhausted, the affected model safely retries on
CPU instead of stopping the application. Useful optional settings:

```bash
# Windows PowerShell examples
$env:AICM_DEVICE="auto"       # auto, cpu, or a CUDA device such as 0
$env:AICM_IMGSZ="960"         # faster than 1280; use 1280 for smaller text
$env:AICM_VIDEO_FRAMES="4"    # sampled frames per video
$env:AICM_OCR_MAX_TOKENS="20"
python app.py
```

The console reports `yolo-cuda` and `trocr-cuda` when GPU acceleration is
active. The first request is slower because the models and CUDA runtime are
being initialized; later requests reuse the loaded models.

Without trained weights the app runs in **MOCK mode** (detectors return demo
boxes, TrOCR is bypassed) so you can exercise the whole flow. Use the
"Demo container ID / plate" fields to drive decisions from the dataset.
Drop your weights in `weights/` (see `model_layer.MODEL_PATHS`) and real
inference activates automatically.

## Datasets (`data/`)

Synthetic but realistic reference data the system uses for decisions:

- `iso_types.csv` — ISO 6346 size/type reference
- `importers.csv` — importers with TIN, region, nearby/transit flags
- `trucks.csv` — fleet plates + transporters
- `containers.csv` — container state: ISO, light, damage, terminal, plot
- `release_orders.csv` — gate verification source of truth
- `yard_plots.csv` — block/row/slot/tier map per terminal
- `gate_tickets.csv` — runtime ticket log (gate-in / gate-out)
- `movement_log.csv` — runtime yard movement history
- `scenarios.csv` — 15 documented edge cases (plate spoofing, unpaid/on-hold
  orders, red/yellow lights, transit cargo, returns, unknown truck, not-in-yard,
  reefer, OCR look-alikes) mapped to the exact record + expected decision

Scale: ~600 containers, ~310 release orders, 150 trucks, 40 importers.

Regenerate with: `python scripts/synth_data.py`
Validate edge cases with: `python scripts/validate_scenarios.py` (15/15 pass)
