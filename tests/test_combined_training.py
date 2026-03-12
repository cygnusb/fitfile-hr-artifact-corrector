from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hf_corrector.model import _split_group_indices
from hf_corrector.training import _align_paired_records, _discover_paired_tours, prepare_combined_training_groups
from hf_corrector.types import FitRecord


def _make_records(
    *,
    count: int,
    start: datetime,
    hr_base: float,
    time_offset_seconds: float = 0.0,
) -> list[FitRecord]:
    records: list[FitRecord] = []
    for idx in range(count):
        ts = start + timedelta(seconds=idx + time_offset_seconds)
        records.append(
            FitRecord(
                timestamp=ts,
                heart_rate=hr_base + (idx % 7),
                power=180.0 + (idx % 5) * 10.0,
                cadence=85.0 + (idx % 3),
                speed=8.0 + idx * 0.01,
                altitude=400.0 + idx * 0.2,
                grade=0.5,
                raw={},
            )
        )
    return records


class AlignPairedRecordsTest(unittest.TestCase):
    def test_aligns_shifted_streams_with_monotonic_matches(self) -> None:
        base = datetime(2026, 3, 12, 8, 0, tzinfo=UTC)
        chest = _make_records(count=8, start=base, hr_base=120.0)
        optical = _make_records(count=6, start=base, hr_base=118.0, time_offset_seconds=1.0)

        aligned_optical, aligned_chest = _align_paired_records(
            optical,
            chest,
            max_gap_seconds=1.5,
        )

        self.assertEqual(len(aligned_optical), 6)
        self.assertEqual(len(aligned_chest), 6)
        matched_times = [record.timestamp for record in aligned_chest]
        self.assertEqual(matched_times, sorted(matched_times))
        gaps = [
            abs((optical_record.timestamp - chest_record.timestamp).total_seconds())
            for optical_record, chest_record in zip(aligned_optical, aligned_chest)
        ]
        self.assertTrue(all(gap <= 1.5 for gap in gaps))


class DiscoverPairedToursTest(unittest.TestCase):
    def test_discovers_expected_pair_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tour_dir = root / "tour1"
            tour_dir.mkdir()
            (tour_dir / "demo_chest.fit.gz").write_text("", encoding="utf-8")
            (tour_dir / "demo_optical.fit.gz").write_text("", encoding="utf-8")

            discovered = _discover_paired_tours(root)

        self.assertEqual(len(discovered), 1)
        found_tour, optical_file, chest_file = discovered[0]
        self.assertEqual(found_tour.name, "tour1")
        self.assertTrue(optical_file.name.endswith("_optical.fit.gz"))
        self.assertTrue(chest_file.name.endswith("_chest.fit.gz"))


class PrepareCombinedTrainingGroupsTest(unittest.TestCase):
    def test_combines_chest_and_weighted_paired_groups(self) -> None:
        base = datetime(2026, 3, 12, 8, 0, tzinfo=UTC)
        chest_only = _make_records(count=20, start=base, hr_base=120.0)
        pair_chest = _make_records(count=18, start=base, hr_base=125.0)
        pair_optical = _make_records(count=18, start=base, hr_base=119.0, time_offset_seconds=1.0)

        fake_paths = {
            "hr-chest/a.fit.gz": chest_only,
            "touren/tour1/demo_chest.fit.gz": pair_chest,
            "touren/tour1/demo_optical.fit.gz": pair_optical,
        }

        def fake_load_fit_records(path: str | Path) -> list[FitRecord]:
            return fake_paths[str(path)]

        with patch("hf_corrector.training._iter_fit_files", return_value=[Path("hr-chest/a.fit.gz")]), patch(
            "hf_corrector.training._discover_paired_tours",
            return_value=[
                (
                    Path("touren/tour1"),
                    Path("touren/tour1/demo_optical.fit.gz"),
                    Path("touren/tour1/demo_chest.fit.gz"),
                )
            ],
        ), patch("hf_corrector.training.load_fit_records", side_effect=fake_load_fit_records):
            groups, report = prepare_combined_training_groups(
                chest_dir="hr-chest",
                tours_dir="touren",
                pair_match_max_seconds=1.5,
                paired_weight=3,
                min_paired_points=10,
            )

        self.assertEqual(len(groups), 4)
        self.assertEqual(sum(group.source == "chest_only" for group in groups), 1)
        self.assertEqual(sum(group.source == "paired_optical_to_chest" for group in groups), 3)
        self.assertEqual(report["dataset"]["paired_tours_accepted"], 1)
        self.assertEqual(report["dataset"]["paired_rows"], 17)
        self.assertEqual(report["paired_tours"][0]["matched_records"], 17)
        paired_targets = [group.y for group in groups if group.source == "paired_optical_to_chest"]
        self.assertEqual(len(paired_targets), 3)
        self.assertTrue(all(np.isfinite(targets).all() for targets in paired_targets))


class GroupSplitTest(unittest.TestCase):
    def test_repeated_group_ids_stay_on_same_side_of_split(self) -> None:
        train_idx, val_idx = _split_group_indices(
            ["a", "paired:tour1", "paired:tour1", "b"],
            val_fraction=0.25,
        )

        self.assertTrue(train_idx)
        self.assertTrue(val_idx)
        paired_positions = {1, 2}
        self.assertTrue(paired_positions.issubset(set(train_idx)) or paired_positions.issubset(set(val_idx)))


if __name__ == "__main__":
    unittest.main()
