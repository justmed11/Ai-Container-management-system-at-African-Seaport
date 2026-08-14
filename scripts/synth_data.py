"""Synthesize a base reference dataset for the Dar es Salaam Port (TPA / TICTS).

The AI Container Management system reads these CSVs to make gate, inspection,
and yard-tracking decisions. All data is fictional but structured to be
realistic for the Port of Dar es Salaam, Tanzania.
"""
import csv
import os
import random

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference vocabularies (Port of Dar es Salaam context)
# ---------------------------------------------------------------------------
# ISO 6346 owner prefixes (shipping lines that call at Dar es Salaam)
OWNER_PREFIXES = ["MSCU", "MAEU", "CMAU", "HLCU", "PCIU", "ONEU", "TGHU", "OOLU", "SEGU", "TCLU"]

# ISO 6346 size/type codes -> human readable
ISO_TYPES = {
    "22G1": "20ft Standard (General)",
    "42G1": "40ft Standard (General)",
    "45G1": "40ft High-Cube (General)",
    "22R1": "20ft Reefer",
    "45R1": "40ft High-Cube Reefer",
    "22T1": "20ft Tank",
    "22U1": "20ft Open-Top",
    "42U1": "40ft Open-Top",
    "22P1": "20ft Flat-Rack",
    "45G0": "40ft High-Cube (General, vented)",
}

# Tanzanian regions and whether they are "nearby" the Dar es Salaam port hinterland.
# Nearby = same/adjacent region cluster (short-haul); else upcountry / transit.
REGIONS = [
    ("Dar es Salaam", True),
    ("Pwani (Coast)", True),
    ("Morogoro", True),
    ("Tanga", True),
    ("Dodoma", False),
    ("Arusha", False),
    ("Mwanza", False),
    ("Mbeya", False),
    ("Kigoma", False),
    ("Mtwara", False),
]

# Transit / landlocked destinations served via Dar es Salaam (Central Corridor)
TRANSIT_COUNTRIES = ["Tanzania", "Tanzania", "Tanzania", "DR Congo", "Zambia", "Burundi", "Rwanda", "Uganda", "Malawi"]

TERMINALS = [
    "TICTS Container Terminal",
    "TPA Berth 1-7 (General Cargo)",
    "TPA ICD Kurasini",
    "Ubungo ICD",
]

IMPORTER_NAMES = [
    "Bakhresa Group Ltd", "MeTL Trading Co", "Azania Logistics Ltd",
    "Kilimanjaro Imports Ltd", "Serengeti Distributors", "Tanga Cement Traders",
    "Coastal Hardware Co", "Uhuru Motors Ltd", "Zanzibar Spice Importers",
    "Mwanza Agro Supplies", "Mbeya Highland Traders", "Dodoma Steel Ltd",
    "Kariakoo General Merchants", "Nyerere Electronics Ltd", "Tanzanite Freight Ltd",
    "Indian Ocean Trading", "Selous Commodities", "Ruvuma Farm Inputs",
    "Kigamboni Builders Ltd", "Great Lakes Transit Co",
]

TRANSPORTERS = [
    "Super Doll Trailer Mfg", "Nyamburi Transport", "Express Cargo Movers",
    "Mwakatobe Haulage", "Coastal Truckers Ltd", "Central Corridor Logistics",
    "Kibo Freight Lines", "Simba Movers Ltd",
]

LIGHTS = ["green", "yellow", "red"]
DAMAGE_TYPES = ["none", "rust", "dent", "hole"]


def iso6346_check_digit(owner_serial: str) -> int:
    """Compute ISO 6346 check digit for the 10-char (4 letters + 6 digits) body."""
    values = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # ISO 6346 letter values skip multiples of 11 (10 included? standard mapping)
    val = 10
    for ch in letters:
        if val % 11 == 0:
            val += 1
        values[ch] = val
        val += 1
    total = 0
    for i, ch in enumerate(owner_serial):
        if ch.isalpha():
            v = values[ch]
        else:
            v = int(ch)
        total += v * (2 ** i)
    cd = total % 11
    return 0 if cd == 10 else cd


def make_container_id():
    prefix = random.choice(OWNER_PREFIXES)
    serial = "".join(random.choice("0123456789") for _ in range(6))
    body = prefix + serial
    cd = iso6346_check_digit(body)
    return f"{body}{cd}"


def make_plate():
    return f"T {random.randint(100,999)} {''.join(random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(3))}"


def make_tin():
    return f"{random.randint(100,199)}-{random.randint(100,999)}-{random.randint(100,999)}"


def make_plot(terminal_idx):
    # Yard divided into lettered blocks A-H and numbered rows/slots
    block = random.choice("ABCDEFGH")
    row = random.randint(1, 20)
    slot = random.randint(1, 6)
    tier = random.randint(1, 4)
    return f"{block}{row:02d}-{slot}-{tier}"


# ---------------------------------------------------------------------------
# 1. ISO type reference
# ---------------------------------------------------------------------------
with open(os.path.join(DATA_DIR, "iso_types.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["iso_code", "description", "length_ft", "is_reefer"])
    for code, desc in ISO_TYPES.items():
        length = 20 if code.startswith("2") else 40
        is_reefer = "R" in code
        w.writerow([code, desc, length, str(is_reefer)])

# ---------------------------------------------------------------------------
# 2. Importers (with TIN, region, nearby flag, transit destination)
# ---------------------------------------------------------------------------
importers = []
for name in IMPORTER_NAMES:
    tin = make_tin()
    region, nearby = random.choice(REGIONS)
    dest_country = random.choice(TRANSIT_COUNTRIES)
    is_transit = dest_country != "Tanzania"
    importers.append({
        "tin": tin,
        "importer_name": name,
        "region": region,
        "delivery_country": dest_country,
        "nearby_county": str(nearby and not is_transit),
        "is_transit": str(is_transit),
    })

with open(os.path.join(DATA_DIR, "importers.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tin", "importer_name", "region", "delivery_country", "nearby_county", "is_transit"])
    w.writeheader()
    w.writerows(importers)

# ---------------------------------------------------------------------------
# 3. Transporters & trucks
# ---------------------------------------------------------------------------
trucks = []
for i in range(40):
    plate = make_plate()
    trucks.append({
        "plate": plate,
        "transporter": random.choice(TRANSPORTERS),
        "trailer_type": random.choice(["20ft skeletal", "40ft skeletal", "flatbed"]),
        "verified": str(random.random() > 0.1),
    })
with open(os.path.join(DATA_DIR, "trucks.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["plate", "transporter", "trailer_type", "verified"])
    w.writeheader()
    w.writerows(trucks)

# ---------------------------------------------------------------------------
# 4. Containers (current state: ISO type, light, damage, yard location)
# ---------------------------------------------------------------------------
containers = []
used_ids = set()
for _ in range(120):
    cid = make_container_id()
    while cid in used_ids:
        cid = make_container_id()
    used_ids.add(cid)
    iso = random.choice(list(ISO_TYPES.keys()))
    light = random.choices(LIGHTS, weights=[0.6, 0.3, 0.1])[0]
    damage = "none" if light == "green" else random.choices(DAMAGE_TYPES, weights=[0.3, 0.3, 0.25, 0.15])[0]
    t_idx = random.randrange(len(TERMINALS))
    containers.append({
        "container_id": cid,
        "iso_type": iso,
        "light_status": light,
        "damage_status": damage,
        "terminal": TERMINALS[t_idx],
        "plot": make_plot(t_idx),
        "in_yard": str(random.random() > 0.2),
    })
with open(os.path.join(DATA_DIR, "containers.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["container_id", "iso_type", "light_status", "damage_status", "terminal", "plot", "in_yard"])
    w.writeheader()
    w.writerows(containers)

# ---------------------------------------------------------------------------
# 5. Release orders (the gate verifies against these)
# ---------------------------------------------------------------------------
release_orders = []
in_yard_containers = [c for c in containers if c["in_yard"] == "True"]
for i, c in enumerate(random.sample(in_yard_containers, min(60, len(in_yard_containers)))):
    imp = random.choice(importers)
    truck = random.choice(trucks)
    ro_status = random.choices(["processed", "pending", "on-hold"], weights=[0.7, 0.2, 0.1])[0]
    release_orders.append({
        "release_order_id": f"RO-2026-{1000+i}",
        "container_id": c["container_id"],
        "iso_type": c["iso_type"],
        "importer_tin": imp["tin"],
        "importer_name": imp["importer_name"],
        "declared_plate": truck["plate"],
        "ro_status": ro_status,
        "customs_cleared": str(ro_status == "processed" and random.random() > 0.1),
        "payment_status": random.choice(["paid", "paid", "unpaid"]),
        "movement_type": random.choice(["pickup", "pickup", "return"]),
    })
with open(os.path.join(DATA_DIR, "release_orders.csv"), "w", newline="") as f:
    fn = ["release_order_id", "container_id", "iso_type", "importer_tin", "importer_name",
          "declared_plate", "ro_status", "customs_cleared", "payment_status", "movement_type"]
    w = csv.DictWriter(f, fieldnames=fn)
    w.writeheader()
    w.writerows(release_orders)

# ---------------------------------------------------------------------------
# 6. Yard plots / blocks map (letter+number naming scheme)
# ---------------------------------------------------------------------------
with open(os.path.join(DATA_DIR, "yard_plots.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["terminal", "block", "rows", "slots_per_row", "max_tier", "reefer_block"])
    for t in TERMINALS:
        for block in "ABCDEFGH":
            w.writerow([t, block, 20, 6, 4, str(block in ("G", "H"))])

# ---------------------------------------------------------------------------
# 7. Gate ticket log (seeded empty-ish; system appends at runtime)
# ---------------------------------------------------------------------------
with open(os.path.join(DATA_DIR, "gate_tickets.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ticket_id", "ticket_type", "timestamp", "plate", "container_id",
                "iso_type", "release_order_id", "decision", "notes"])

# ---------------------------------------------------------------------------
# 8. Container movement log (yard tracking history)
# ---------------------------------------------------------------------------
with open(os.path.join(DATA_DIR, "movement_log.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "container_id", "from_terminal", "from_plot", "to_terminal", "to_plot", "moved_by"])

print("Datasets written to", DATA_DIR)
for fn in sorted(os.listdir(DATA_DIR)):
    path = os.path.join(DATA_DIR, fn)
    with open(path) as fh:
        n = sum(1 for _ in fh)
    print(f"  {fn}: {n} rows")
