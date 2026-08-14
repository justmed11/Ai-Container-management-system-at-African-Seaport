import csv
import os
import tempfile
import unittest
from unittest.mock import patch

import pipeline


def write_csv(folder, name, fields, rows):
    with open(os.path.join(folder, name), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class GateLogicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        write_csv(d, "iso_types.csv", ["iso_code"], [{"iso_code": "22G1"}])
        write_csv(d, "importers.csv",
                  ["tin", "importer_name", "region", "delivery_country",
                   "nearby_county", "is_transit"],
                  [{"tin": "TIN1", "importer_name": "Importer", "region": "Dar es Salaam",
                    "delivery_country": "Tanzania", "nearby_county": "True",
                    "is_transit": "False"}])
        write_csv(d, "trucks.csv", ["plate", "transporter", "trailer_type", "verified"],
                  [{"plate": "T 123 ABC", "transporter": "Carrier",
                    "trailer_type": "skeletal", "verified": "True"}])
        write_csv(d, "containers.csv",
                  ["container_id", "iso_type", "light_status", "damage_status",
                   "terminal", "plot", "in_yard"],
                  [{"container_id": "MSCU1234566", "iso_type": "22G1",
                    "light_status": "green", "damage_status": "none",
                    "terminal": "TICTS", "plot": "A01-1-1", "in_yard": "True"},
                   {"container_id": "TCLU7654321", "iso_type": "22G1",
                    "light_status": "green", "damage_status": "none",
                    "terminal": "TICTS", "plot": "A01-1-2", "in_yard": "False"}])
        fields = ["release_order_id", "container_id", "iso_type", "importer_tin",
                  "importer_name", "declared_plate", "ro_status", "customs_cleared",
                  "payment_status", "movement_type"]
        write_csv(d, "release_orders.csv", fields,
                  [{"release_order_id": "RO-PICKUP", "container_id": "MSCU1234566",
                    "iso_type": "22G1", "importer_tin": "TIN1", "importer_name": "Importer",
                    "declared_plate": "T 123 ABC", "ro_status": "processed",
                    "customs_cleared": "True", "payment_status": "paid",
                    "movement_type": "pickup"},
                   {"release_order_id": "RO-RETURN", "container_id": "TCLU7654321",
                    "iso_type": "22G1", "importer_tin": "TIN1", "importer_name": "Importer",
                    "declared_plate": "T 123 ABC", "ro_status": "processed",
                    "customs_cleared": "True", "payment_status": "paid",
                    "movement_type": "return"}])
        write_csv(d, "gate_tickets.csv",
                  ["ticket_id", "ticket_type", "timestamp", "plate", "container_id",
                   "iso_type", "release_order_id", "decision", "notes"], [])
        self.data_patch = patch.object(pipeline, "DATA_DIR", d)
        self.data_patch.start()
        pipeline.DB = pipeline.PortDB()

    def tearDown(self):
        self.data_patch.stop()
        self.tmp.cleanup()

    def test_compact_ocr_plate_matches_spaced_csv_plate(self):
        report = pipeline.gate_inspection("T123ABC", "", "", False)
        self.assertEqual(report["decision"], "ALLOW")
        self.assertEqual(report["matched_container"], "MSCU1234566")

    def test_return_is_recorded_and_other_pickup_is_found(self):
        report = pipeline.gate_inspection("T123ABC", "TCLU7654321", "22G1", True)
        self.assertEqual(report["decision"], "ALLOW")
        self.assertEqual(report["returned_container"], "TCLU7654321")
        self.assertEqual(report["matched_container"], "MSCU1234566")
        pipeline.DB.reload()
        returned = pipeline.DB.ro_for_container("TCLU7654321")
        self.assertEqual(returned["ro_status"], "returned")

    def test_wrong_return_container_does_not_match_another_return(self):
        report = pipeline.gate_inspection("T123ABC", "WRONG0000000", "22G1", True)
        self.assertEqual(report["decision"], "DENY")

    def test_non_tanzanian_plate_shape_is_rejected(self):
        self.assertEqual(pipeline.normalize_plate("RMDU7040101"), "")
        report = pipeline.gate_inspection("RMDU7040101", "", "", False)
        self.assertEqual(report["decision"], "DENY")

    def test_gate_out_requires_exact_gate_in_ticket_plate_and_container(self):
        gate_in = pipeline.gate_inspection("T123ABC", "", "", False)
        self.assertEqual(gate_in["decision"], "ALLOW")
        self.assertTrue(pipeline.mark_ticket_used(gate_in["ticket"], "GATE_IN")["ok"])

        gate_out = pipeline.gate_out_inspection(
            "T123ABC", "MSCU1234566", "22G1", gate_in["ticket"]
        )
        self.assertEqual(gate_out["decision"], "ALLOW")

    def test_gate_out_denies_fake_or_mismatched_ticket(self):
        fake = pipeline.gate_out_inspection(
            "T123ABC", "MSCU1234566", "22G1", "GIN-20260802-FAKE00"
        )
        self.assertEqual(fake["decision"], "DENY")

        gate_in = pipeline.gate_inspection("T123ABC", "", "", False)
        pipeline.mark_ticket_used(gate_in["ticket"], "GATE_IN")
        mismatch = pipeline.gate_out_inspection(
            "T123ABC", "TCLU7654321", "22G1", gate_in["ticket"]
        )
        self.assertEqual(mismatch["decision"], "DENY")

    def test_gate_in_ticket_cannot_be_used_twice(self):
        gate_in = pipeline.gate_inspection("T123ABC", "", "", False)
        pipeline.mark_ticket_used(gate_in["ticket"], "GATE_IN")
        first = pipeline.gate_out_inspection(
            "T123ABC", "MSCU1234566", "22G1", gate_in["ticket"]
        )
        second = pipeline.gate_out_inspection(
            "T123ABC", "MSCU1234566", "22G1", gate_in["ticket"]
        )
        self.assertEqual(first["decision"], "ALLOW")
        self.assertEqual(second["decision"], "DENY")

    def test_unverified_gate_in_ticket_is_denied_at_gate_out(self):
        gate_in = pipeline.gate_inspection("T123ABC", "", "", False)
        gate_out = pipeline.gate_out_inspection(
            "T123ABC", "MSCU1234566", "22G1", gate_in["ticket"]
        )
        self.assertEqual(gate_out["decision"], "DENY")

    def test_matching_verified_gate_in_automatically_issues_gate_out(self):
        gate_in = pipeline.gate_inspection("T123ABC", "", "", False)
        used = pipeline.mark_ticket_used(gate_in["ticket"], "GATE_IN")
        self.assertTrue(used["ok"])

        # These mutable conditions were already approved at gate-in. An exact
        # ticket/plate/container match must remain sufficient at gate-out.
        pipeline.DB.ro_by_id("RO-PICKUP")["customs_cleared"] = "False"
        pipeline.DB.containers["MSCU1234566"]["light_status"] = "red"
        gate_out = pipeline.gate_out_inspection(
            "T123ABC", "MSCU1234566", "22G1", gate_in["ticket"]
        )
        self.assertEqual(gate_out["decision"], "ALLOW")
        self.assertTrue(gate_out["ticket"].startswith("GOUT-"))

    def test_gate_in_and_gate_out_ticket_use_events_are_logged(self):
        gate_in = pipeline.gate_inspection("T123ABC", "", "", False)
        self.assertTrue(pipeline.mark_ticket_used(gate_in["ticket"], "GATE_IN")["ok"])
        self.assertTrue(pipeline.DB.ticket_is_used(gate_in["ticket"]))

        gate_out = pipeline.gate_out_inspection(
            "T123ABC", "MSCU1234566", "22G1", gate_in["ticket"]
        )
        self.assertTrue(pipeline.mark_ticket_used(gate_out["ticket"], "GATE_OUT")["ok"])
        self.assertTrue(pipeline.DB.ticket_is_used(gate_out["ticket"]))

        with open(os.path.join(self.tmp.name, "gate_tickets.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(sum(r["ticket_type"] == "TICKET_USED" for r in rows), 2)


if __name__ == "__main__":
    unittest.main()
