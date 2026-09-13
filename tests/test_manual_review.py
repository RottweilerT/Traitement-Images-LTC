from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from image_pipeline.config import DEFAULT_CONFIG
from image_pipeline.manual_review import (
    MANIFEST_FILENAME,
    MANUAL_DIRNAME,
    REPORT_FILENAME,
    record_error,
    record_output,
    record_source_start,
    run_manual_review,
)


class ManualReviewTests(unittest.TestCase):
    def _config(self, root: Path):
        return replace(
            DEFAULT_CONFIG,
            input_dir=root / "input",
            output_dir=root / "output",
        )

    def test_deleted_outputs_create_one_original_copy_per_missing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            source = config.input_dir / "Lot_001" / "001.tif"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"TIFF-ORIGINAL-STRICT")
            lot_output = config.output_dir / "Lot_001-ps"
            lot_config = replace(config, output_dir=lot_output)

            record_source_start(lot_config, source)
            for index in range(1, 5):
                record_output(
                    lot_config,
                    source,
                    lot_output / f"001-{index}.tif",
                    frame_index=1,
                    subject_index=index,
                    qc_passed=True,
                    qc_problems=[],
                )
            lot_output.mkdir(parents=True, exist_ok=True)
            (lot_output / "001-1.tif").write_bytes(b"processed-1")
            (lot_output / "001-3.tif").write_bytes(b"processed-3")

            summary = run_manual_review(config)

            manual_dir = lot_output / MANUAL_DIRNAME
            self.assertEqual((manual_dir / "001-2.tif").read_bytes(), source.read_bytes())
            self.assertEqual((manual_dir / "001-4.tif").read_bytes(), source.read_bytes())
            self.assertEqual(summary["missing_outputs"], 2)
            self.assertEqual(summary["copies_created"], 2)
            report = (lot_output / REPORT_FILENAME).read_text(encoding="utf-8")
            self.assertIn("001-2.tif", report)
            self.assertIn("001-4.tif", report)

    def test_automatic_failure_without_output_copies_original_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            source = config.input_dir / "Lot_002" / "007.tif"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"ORIGINAL-007")
            lot_output = config.output_dir / "Lot_002-ps"
            lot_config = replace(config, output_dir=lot_output)

            record_source_start(lot_config, source)
            record_error(
                lot_config,
                source,
                RuntimeError("Contour trop incertain"),
            )

            summary = run_manual_review(config)

            copied = lot_output / MANUAL_DIRNAME / "007.tif"
            self.assertEqual(copied.read_bytes(), source.read_bytes())
            self.assertEqual(summary["automatic_failures"], 1)
            report = (lot_output / REPORT_FILENAME).read_text(encoding="utf-8")
            self.assertIn("Contour trop incertain", report)

    def test_partial_automatic_failures_create_multiple_source_copies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            source = config.input_dir / "Lot_003" / "010.tif"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"ORIGINAL-010")
            lot_output = config.output_dir / "Lot_003-ps"
            lot_config = replace(config, output_dir=lot_output)

            record_source_start(lot_config, source)
            record_output(
                lot_config,
                source,
                lot_output / "010-1.tif",
                frame_index=1,
                subject_index=1,
                qc_passed=True,
                qc_problems=[],
            )
            lot_output.mkdir(parents=True, exist_ok=True)
            (lot_output / "010-1.tif").write_bytes(b"processed")
            record_error(
                lot_config,
                source,
                RuntimeError("Sujet 2 refusé"),
                frame_index=1,
                subject_index=2,
            )
            record_error(
                lot_config,
                source,
                RuntimeError("Sujet 3 refusé"),
                frame_index=1,
                subject_index=3,
            )

            summary = run_manual_review(config)

            manual_dir = lot_output / MANUAL_DIRNAME
            self.assertEqual(
                (manual_dir / "010-echec-f1-s2.tif").read_bytes(),
                source.read_bytes(),
            )
            self.assertEqual(
                (manual_dir / "010-echec-f1-s3.tif").read_bytes(),
                source.read_bytes(),
            )
            self.assertEqual(summary["automatic_failures"], 2)

    def test_existing_manual_copy_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            source = config.input_dir / "Lot_004" / "020.tif"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"ORIGINAL")
            lot_output = config.output_dir / "Lot_004-ps"
            lot_config = replace(config, output_dir=lot_output)

            record_source_start(lot_config, source)
            record_output(
                lot_config,
                source,
                lot_output / "020-2.tif",
                frame_index=1,
                subject_index=2,
                qc_passed=True,
                qc_problems=[],
            )
            manual_dir = lot_output / MANUAL_DIRNAME
            manual_dir.mkdir(parents=True, exist_ok=True)
            manual_file = manual_dir / "020-2.tif"
            manual_file.write_bytes(b"TRAVAIL-MANUEL-A-PRESERVER")

            summary = run_manual_review(config)

            self.assertEqual(manual_file.read_bytes(), b"TRAVAIL-MANUEL-A-PRESERVER")
            self.assertEqual(summary["copies_existing"], 1)

    def test_present_qc_failure_is_reported_without_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            source = config.input_dir / "Lot_005" / "030.tif"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"ORIGINAL")
            lot_output = config.output_dir / "Lot_005-ps"
            lot_config = replace(config, output_dir=lot_output)

            record_source_start(lot_config, source)
            record_output(
                lot_config,
                source,
                lot_output / "030.tif",
                frame_index=1,
                subject_index=1,
                qc_passed=False,
                qc_problems=["angle résiduel trop important"],
            )
            lot_output.mkdir(parents=True, exist_ok=True)
            (lot_output / "030.tif").write_bytes(b"processed")

            summary = run_manual_review(config)

            self.assertEqual(summary["qc_to_review"], 1)
            self.assertFalse((lot_output / MANUAL_DIRNAME / "030.tif").exists())
            report = (lot_output / REPORT_FILENAME).read_text(encoding="utf-8")
            self.assertIn("angle résiduel trop important", report)
            self.assertTrue((lot_output / MANIFEST_FILENAME).exists())


if __name__ == "__main__":
    unittest.main()
