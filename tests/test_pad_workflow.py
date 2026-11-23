"""End-to-end checks for the trial-pad selection and pad-aware lift readings."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from foamnest import demo, pads, web  # noqa: E402
from foamnest.__main__ import application  # noqa: E402
from foamnest.store import Library  # noqa: E402
from image_lab.validation import Invalid  # noqa: E402


class PadWorkflowCase(unittest.TestCase):
    def setUp(self):
        self.library = Library(Path(tempfile.mkdtemp()) / "samples")
        self.run = demo.create(self.library)
        self.face = next(photo for photo in self.run["photos"] if photo["role"] == "face")
        self.frame = web.sample_frame(self.library, self.run, self.face)

    def marks(self, purpose):
        return [mark for mark in self.run["marks"]
                if mark["photo"] == self.face["id"] and mark["purpose"] == purpose]

    def trial_marks(self):
        lane = next(mark for mark in self.marks("lane") if mark["title"].startswith("Entry"))
        return {"line": lane, "bearing_zone": self.marks("bearing-zone")[0],
                "keepout": self.marks("keepout")[0],
                "pad_a": next(mark for mark in self.marks("trial-pad") if " A " in mark["title"]),
                "pad_b": next(mark for mark in self.marks("trial-pad") if " B " in mark["title"])}

    def test_pad_trial_recommends_and_explains_tradeoff(self):
        summary, rows, overlay = pads.pad_trial(
            self.frame, self.trial_marks(),
            {"minimum_bearing_mm2": 200, "minimum_approach_mm": 10})
        self.assertEqual(summary["eligible_pads"], 2)
        self.assertEqual(summary["adopted_pad"], "Trial pad B / deeper reach")
        by_title = {row["pad_title"]: row for row in rows}
        pad_a = by_title["Trial pad A / wide bearing"]
        pad_b = by_title["Trial pad B / deeper reach"]
        # A carries more area; B reaches deeper.
        self.assertGreater(pad_a["bearing_coverage_mm2"], pad_b["bearing_coverage_mm2"])
        self.assertGreater(pad_b["approach_to_pad_mm"], pad_a["approach_to_pad_mm"])
        self.assertFalse(pad_a["touches_thin_lip"])
        self.assertIn("bears", summary["recommendation_rationale"])
        self.assertIn("deeper", summary["recommendation_rationale"])
        self.assertEqual(overlay.ndim, 3)

    def test_minimum_bearing_eliminates_both_pads(self):
        summary, rows, _ = pads.pad_trial(
            self.frame, self.trial_marks(),
            {"minimum_bearing_mm2": 5000, "minimum_approach_mm": 10})
        self.assertEqual(summary["adopted_pad"], "none")
        self.assertEqual(summary["eligible_pads"], 0)
        self.assertTrue(all(not row["eligible"] for row in rows))

    def test_minimum_approach_eliminates_shallow_pad_only(self):
        summary, rows, _ = pads.pad_trial(
            self.frame, self.trial_marks(),
            {"minimum_bearing_mm2": 200, "minimum_approach_mm": 50})
        self.assertEqual(summary["adopted_pad"], "Trial pad B / deeper reach")
        self.assertEqual({row["pad_title"]: row["eligible"] for row in rows},
                         {"Trial pad A / wide bearing": False,
                          "Trial pad B / deeper reach": True})

    def test_thin_lip_contact_disqualifies_pad(self):
        import copy
        marks = self.trial_marks()
        bad = copy.deepcopy(marks["pad_a"])
        bad["title"] = "Pad over the thin lip"
        bad["points"] = marks["keepout"]["points"]
        summary, rows, _ = pads.pad_trial(
            self.frame, {**marks, "pad_a": bad},
            {"minimum_bearing_mm2": 200, "minimum_approach_mm": 10})
        self.assertEqual(summary["adopted_pad"], "Trial pad B / deeper reach")
        offender = next(row for row in rows if row["pad_title"] == bad["title"])
        self.assertTrue(offender["touches_thin_lip"])
        self.assertGreater(offender["thin_lip_contact_mm2"], 0)
        self.assertFalse(offender["eligible"])

    def test_duplicate_pad_rejected(self):
        marks = self.trial_marks()
        with self.assertRaises(Invalid):
            pads.pad_trial(self.frame, {**marks, "pad_b": marks["pad_a"]}, {})

    def test_lift_access_distinguishes_three_first_stops(self):
        marks = self.trial_marks()
        pad_a, pad_b = marks["pad_a"], marks["pad_b"]
        on_pad, segments_a, _ = pads.lift_access_pad(
            self.frame, {"line": marks["line"], "pad": pad_a}, {"obstacle_luma": .31})
        self.assertEqual(on_pad["first_stop_state"], "pad")
        self.assertEqual(on_pad["first_stop"], "trial pad")
        self.assertAlmostEqual(on_pad["continuous_approach_mm"],
                               next(row for row in self._trial_rows(marks, pad_a))["approach_to_pad_mm"],
                               places=1)

        on_block, segments_b, _ = pads.lift_access_pad(
            self.frame, {"line": marks["line"], "pad": pad_b}, {"obstacle_luma": .31})
        self.assertEqual(on_block["first_stop_state"], "blocked")
        self.assertEqual(on_block["first_stop"], "original-photo block")
        self.assertEqual(on_block["baseline_first_stop"], "original-photo block")
        self.assertIn("pad", {segment["state"] for segment in segments_b})

        missing_lane = next(mark for mark in self.marks("lane") if "missing" in mark["title"])
        unseen, segments_u, _ = pads.lift_access_pad(
            self.frame, {"line": missing_lane, "pad": pad_b}, {"obstacle_luma": .31})
        self.assertEqual(unseen["first_stop_state"], "unseen")
        self.assertEqual(unseen["first_stop"], "missing evidence")
        self.assertEqual(segments_u[0]["state"], "clear")
        self.assertEqual(segments_u[1]["state"], "unseen")

    def _trial_rows(self, marks, pad):
        _, rows, _ = pads.pad_trial(self.frame, marks,
                                    {"minimum_bearing_mm2": 200, "minimum_approach_mm": 10})
        return [row for row in rows if row["pad_id"] == pad["id"]]


class PadRoutesCase(unittest.TestCase):
    def setUp(self):
        self.client = application(Path(tempfile.mkdtemp())).test_client()
        self.run = self.client.post("/nest/demo").get_json()
        self.key = self.run["id"]
        self.face = next(photo for photo in self.run["photos"] if photo["role"] == "face")

    def marks(self, purpose):
        return [mark for mark in self.run["marks"]
                if mark["photo"] == self.face["id"] and mark["purpose"] == purpose]

    def test_config_advertises_new_workflow(self):
        config = self.client.get("/nest/config").get_json()
        self.assertIn("pad-trial", config["readings"])
        self.assertIn("lift-access-pad", config["readings"])
        for purpose in ("bearing-zone", "keepout", "trial-pad"):
            self.assertIn(purpose, config["marks"])

    def test_readings_save_with_photo_and_pad_identity(self):
        lane = next(mark["id"] for mark in self.marks("lane") if mark["title"].startswith("Entry"))
        zone = self.marks("bearing-zone")[0]["id"]
        keep = self.marks("keepout")[0]["id"]
        pad_a, pad_b = (mark["id"] for mark in self.marks("trial-pad"))
        response = self.client.post(f"/nest/runs/{self.key}/readings", json={
            "revision": self.run["revision"], "mode": "pad-trial", "photo": self.face["id"],
            "marks": {"line": lane, "bearing_zone": zone, "keepout": keep,
                      "pad_a": pad_a, "pad_b": pad_b},
            "options": {"minimum_bearing_mm2": 200, "minimum_approach_mm": 10}})
        self.assertEqual(response.status_code, 201)
        trial = response.get_json()["reading"]
        self.assertEqual(trial["summary"]["adopted_pad"], "Trial pad B / deeper reach")
        self.assertEqual(len(trial["rows"]), 2)

        pad_b_mark = next(mark for mark in self.marks("trial-pad") if " B " in mark["title"])
        response = self.client.post(f"/nest/runs/{self.key}/readings", json={
            "revision": response.get_json()["run"]["revision"], "mode": "lift-access-pad",
            "photo": self.face["id"], "marks": {"line": lane, "pad": pad_b_mark["id"]},
            "options": {"obstacle_luma": .31}})
        self.assertEqual(response.status_code, 201)
        reading = response.get_json()["reading"]
        self.assertEqual(reading["summary"]["source_photo"], self.face["title"])
        self.assertEqual(reading["summary"]["trial_pad"], pad_b_mark["title"])
        self.assertIn(reading["summary"]["first_stop_state"], {"pad", "blocked", "unseen"})
        overlay = self.client.get(f"/nest/runs/{self.key}/readings/{reading['id']}/overlay")
        self.assertEqual(overlay.status_code, 200)
        self.assertEqual(overlay.mimetype, "image/png")
        packet = self.client.get(f"/nest/runs/{self.key}/readings/{reading['id']}/packet")
        self.assertEqual(packet.status_code, 200)

    def test_wrong_mark_purpose_rejected(self):
        lane = next(mark["id"] for mark in self.marks("lane") if mark["title"].startswith("Entry"))
        zone = self.marks("bearing-zone")[0]["id"]
        keep = self.marks("keepout")[0]["id"]
        pad_b = self.marks("trial-pad")[1]["id"]
        response = self.client.post(f"/nest/runs/{self.key}/readings", json={
            "revision": self.run["revision"], "mode": "pad-trial", "photo": self.face["id"],
            "marks": {"line": lane, "bearing_zone": zone, "keepout": keep,
                      "pad_a": lane, "pad_b": pad_b},
            "options": {"minimum_bearing_mm2": 200, "minimum_approach_mm": 10}})
        self.assertEqual(response.status_code, 400)

    def test_option_range_enforced(self):
        lane = next(mark["id"] for mark in self.marks("lane") if mark["title"].startswith("Entry"))
        pad = self.marks("trial-pad")[0]
        response = self.client.post(f"/nest/runs/{self.key}/readings", json={
            "revision": self.run["revision"], "mode": "lift-access-pad",
            "photo": self.face["id"], "marks": {"line": lane, "pad": pad["id"]},
            "options": {"obstacle_luma": 9}})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
