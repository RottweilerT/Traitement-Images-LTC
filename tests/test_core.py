from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from image_pipeline.color import cmyk_percent_to_u8, replace_background
from image_pipeline.config import DEFAULT_CONFIG
from image_pipeline.io_utils import NameAllocator
from image_pipeline.models import OrientationEstimate, ProtectedSubject, RotationResult
from image_pipeline.pipeline import _output_dir_for_source

try:
    import cv2
except ImportError:  # Les tests géométriques seront actifs après installation.
    cv2 = None


class ColorAndNamingTests(unittest.TestCase):
    def test_requested_cmyk_bytes_are_exact(self) -> None:
        self.assertEqual(cmyk_percent_to_u8((84, 82, 73, 95)), (214, 209, 186, 242))

    def test_background_pixels_are_exact_in_memory(self) -> None:
        rgb = np.full((9, 9, 3), (180, 80, 40), dtype=np.uint8)
        alpha = np.zeros((9, 9), dtype=np.uint8)
        alpha[3:6, 3:6] = 255
        rotated = RotationResult(
            rgb=rgb,
            expected_original_rgb=rgb.copy(),
            alpha=alpha,
            protected_core=(alpha == 255).astype(np.uint8),
            affine_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            orientation_before=OrientationEstimate(determinable=True),
            alpha_mass_before=9.0,
            perimeter_before=8.0,
            aspect_ratio_before=1.0,
            old_background_rgb=(255, 255, 255),
        )
        result = replace_background(rotated, DEFAULT_CONFIG)
        output = np.asarray(result.image)
        # Depuis la 1.5.0, le sujet reste en RVB et le fond utilise le noir
        # de référence RVB 5, 0, 2 (#050002).
        expected = np.asarray(DEFAULT_CONFIG.rgb_background, dtype=np.uint8)
        self.assertEqual(tuple(expected), (5, 0, 2))
        self.assertEqual(result.image.mode, "RGB")
        self.assertTrue(np.all(output[alpha == 0] == expected))
        # Les pixels du sujet ne doivent pas être modifiés.
        self.assertTrue(np.all(output[alpha == 255] == rgb[alpha == 255]))

    def test_single_output_keeps_source_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocator = NameAllocator(root, ".tif")

            path, index = allocator.reserve(
                "document_A",
                numbered=False,
            )

            self.assertIsNone(index)
            self.assertEqual(path.name, "document_A.tif")
            NameAllocator.release(path)

    def test_numbered_outputs_use_hyphen_and_continue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "document_A-1.tif").write_bytes(b"existing")
            (root / "document_A-3.jpg").write_bytes(b"existing")
            allocator = NameAllocator(root, ".tif")

            path, index = allocator.reserve(
                "document_A",
                numbered=True,
            )

            self.assertEqual(index, 4)
            self.assertEqual(path.name, "document_A-4.tif")
            NameAllocator.release(path)

    def test_single_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "document_A.tif"
            existing.write_bytes(b"original")
            allocator = NameAllocator(root, ".tif")

            with self.assertRaises(FileExistsError):
                allocator.reserve(
                    "document_A",
                    numbered=False,
                )

            self.assertEqual(existing.read_bytes(), b"original")

@unittest.skipIf(cv2 is None, "OpenCV n'est pas installé")
class GeometryTests(unittest.TestCase):
    def test_rotation_is_rigid_and_reduces_skew(self) -> None:
        from image_pipeline.geometry import (
            detect_subject_inclination,
            straighten_subject_without_clipping,
        )

        config = replace(
            DEFAULT_CONFIG,
            final_padding_px=12,
            min_rotation_deg=0.0,
        )
        alpha = np.zeros((180, 180), dtype=np.uint8)
        rectangle = ((90.0, 90.0), (34.0, 112.0), 13.0)
        box = cv2.boxPoints(rectangle).astype(np.int32)
        cv2.fillConvexPoly(alpha, box, 255)
        rgb = np.full((180, 180, 3), (90, 130, 170), dtype=np.uint8)
        subject = ProtectedSubject(
            original_rgb=rgb.copy(),
            protected_rgb=rgb.copy(),
            alpha=alpha,
            protected_core=(alpha == 255).astype(np.uint8),
            old_background_rgb=(240, 240, 240),
        )
        before = detect_subject_inclination(alpha, config)
        self.assertTrue(before.determinable)
        result = straighten_subject_without_clipping(subject, before, config)
        after = detect_subject_inclination(result.alpha, config)
        self.assertTrue(after.determinable)
        self.assertLess(abs(after.correction_deg), 1.0)
        singular_values = np.linalg.svd(result.affine_matrix[:, :2], compute_uv=False)
        self.assertTrue(np.allclose(singular_values, (1.0, 1.0), atol=1e-8))
        self.assertGreaterEqual(np.min(np.nonzero(result.alpha)[0]), 4)

class OutputRoutingTests(unittest.TestCase):
    def test_top_level_folder_gets_ps_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            source = input_dir / "Lot_001" / "001.tif"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"test")

            config = replace(
                DEFAULT_CONFIG,
                input_dir=input_dir,
                output_dir=output_dir,
            )

            result = _output_dir_for_source(source, config)

            self.assertEqual(result, output_dir / "Lot_001-ps")

    def test_file_directly_in_input_keeps_root_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            source = input_dir / "001.tif"
            input_dir.mkdir(parents=True)
            source.write_bytes(b"test")

            config = replace(
                DEFAULT_CONFIG,
                input_dir=input_dir,
                output_dir=output_dir,
            )

            result = _output_dir_for_source(source, config)

            self.assertEqual(result, output_dir)

@unittest.skipIf(cv2 is None, "OpenCV n'est pas installé")
class SegmentationStructureTests(unittest.TestCase):
    def test_existing_alpha_is_trusted_without_calling_ai(self) -> None:
        from image_pipeline.models import FrameData
        from image_pipeline.segmentation import generate_precise_cutout

        class EngineThatMustNotRun:
            def remove_background(self, *_args, **_kwargs):
                raise AssertionError("L'IA ne doit pas remplacer un alpha existant")

        rgb = np.full((40, 50, 3), (120, 70, 30), dtype=np.uint8)
        alpha = np.zeros((40, 50), dtype=np.uint8)
        alpha[8:33, 11:39] = 255
        frame = FrameData(Path("alpha.png"), 1, rgb, alpha)
        result = generate_precise_cutout(frame, EngineThatMustNotRun(), DEFAULT_CONFIG)
        self.assertTrue(np.array_equal(result.alpha, alpha))
        self.assertTrue(np.array_equal(result.edge_rgb, rgb))

    def test_two_disjoint_subjects_are_extracted_in_reading_order(self) -> None:
        from image_pipeline.segmentation import detect_subject_regions

        alpha = np.zeros((160, 240), dtype=np.uint8)
        alpha[20:90, 25:75] = 255
        alpha[55:140, 150:220] = 255
        config = replace(
            DEFAULT_CONFIG,
            split_subjects=True,
            component_grouping_gap_px=4,
            min_subject_area_ratio=0.001,
            crop_padding_px=5,
        )
        regions = detect_subject_regions(alpha, config)
        self.assertEqual(len(regions), 2)
        self.assertLess(regions[0].bbox_xyxy[0], regions[1].bbox_xyxy[0])


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(cv2 is None, "OpenCV n'est pas installé")
class MarginTests(unittest.TestCase):
    def _run_on_synthetic_scan(self, dpi: float, angle: float) -> list[dict]:
        import json

        from PIL import Image

        from image_pipeline.pipeline import process_batch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            scan = np.full((400, 500, 3), (128, 128, 128), dtype=np.uint8)
            rectangle = ((250.0, 200.0), (220.0, 150.0), angle)
            box = cv2.boxPoints(rectangle).astype(np.int32)
            cv2.fillConvexPoly(scan, box, (200, 40, 60))
            Image.fromarray(scan).save(input_dir / "timbre.tif", dpi=(dpi, dpi))

            config = replace(
                DEFAULT_CONFIG,
                input_dir=input_dir,
                output_dir=output_dir,
            )
            process_batch(config)
            reports = [
                json.loads(line)
                for path in output_dir.rglob("controle_qualite.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            outputs = list(output_dir.rglob("timbre*.tif"))
            self.assertEqual(len(outputs), 1)
            return reports

    def _margin_check(self, reports: list[dict]) -> dict:
        checks = [
            check
            for report in reports
            for check in report.get("checks", [])
            if check.get("name") == "marge_exacte"
        ]
        self.assertEqual(len(checks), 1)
        return checks[0]

    def test_margin_is_exactly_1mm_once_at_300_dpi(self) -> None:
        check = self._margin_check(self._run_on_synthetic_scan(300.0, 4.0))
        self.assertTrue(check["passed"], check)
        self.assertEqual(check["expected"]["gauche_droite_px"], 12)
        self.assertEqual(
            set(check["measured"].values()),
            {12},
        )

    def test_margin_follows_resolution_at_600_dpi(self) -> None:
        check = self._margin_check(self._run_on_synthetic_scan(600.0, 0.0))
        self.assertTrue(check["passed"], check)
        self.assertEqual(set(check["measured"].values()), {24})


@unittest.skipIf(cv2 is None, "OpenCV n'est pas installé")
class SmoothRotationTests(unittest.TestCase):
    """La rotation ne doit créer ni ligne de décalage, ni halo, ni escalier."""

    ANGLE = 2.3

    def _tilted_scan(self) -> tuple[np.ndarray, np.ndarray]:
        """Scan simulé : timbre incliné sur fond gris, intégré ×4 comme un capteur."""

        k = 4
        height, width = 160 * k, 240 * k
        stamp = np.full((height, width, 3), (235, 225, 200), dtype=np.uint8)
        for y in range(10 * k, height - 10 * k, 9 * k):
            stamp[y : y + k] = (70, 50, 130)  # traits fins horizontaux (1 px)
        canvas_h, canvas_w = 320 * k, 400 * k
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -self.ANGLE, 1.0)
        matrix[:, 2] += (canvas_w / 2 - width / 2, canvas_h / 2 - height / 2)
        rotated = cv2.warpAffine(stamp, matrix, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
        coverage = cv2.warpAffine(
            np.ones((height, width), np.float32), matrix, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
        )
        scan = rotated * coverage[..., None] + 128.0 * (1.0 - coverage[..., None])
        scan = cv2.resize(scan, (400, 320), interpolation=cv2.INTER_AREA)
        return np.clip(np.rint(scan), 0, 255).astype(np.uint8), stamp

    def _process(self) -> tuple[np.ndarray, np.ndarray]:
        from image_pipeline.geometry import (
            detect_subject_inclination,
            straighten_subject_without_clipping,
        )
        from image_pipeline.models import FrameData
        from image_pipeline.segmentation import (
            detect_subject_regions,
            detect_subjects_on_uniform_gray_background,
            protect_subject_interior,
        )

        scan, _ = self._tilted_scan()
        frame = FrameData(
            source_path=Path("timbre.tif"),
            frame_index=0,
            rgb=scan,
            source_alpha=np.full(scan.shape[:2], 255, dtype=np.uint8),
            dpi=(300.0, 300.0),
        )
        segmentation = detect_subjects_on_uniform_gray_background(frame, DEFAULT_CONFIG)
        self.assertIsNotNone(segmentation)
        region = detect_subject_regions(segmentation.alpha, DEFAULT_CONFIG)[0]
        subject = protect_subject_interior(segmentation, region, DEFAULT_CONFIG)
        orientation = detect_subject_inclination(subject.alpha, DEFAULT_CONFIG)
        result = straighten_subject_without_clipping(
            subject, orientation, DEFAULT_CONFIG, padding_xy=(10, 10)
        )
        return result.rgb, result.alpha

    def test_thin_lines_have_no_offset_steps(self) -> None:
        rgb, alpha = self._process()
        opaque_cols = np.nonzero(np.all(alpha[alpha.shape[0] // 2 - 20 :][:40] == 255, axis=0))[0]
        cols = opaque_cols[10:-10]
        darkness = 255.0 - rgb[..., 1].astype(np.float64)
        # Suivi d'un trait fin : position verticale (centre de gravité) colonne
        # par colonne. Au plus proche voisin, elle saute d'un pixel entier.
        centre_row = alpha.shape[0] // 2
        window = slice(centre_row - 4, centre_row + 5)
        best = np.argmax(darkness[window, cols].mean(axis=1))
        rows = np.arange(centre_row - 4, centre_row + 5)[max(0, best - 2) : best + 3]
        weights = darkness[rows][:, cols] - darkness[rows][:, cols].min(axis=0)
        centroid = (weights * rows[:, None]).sum(axis=0) / np.maximum(weights.sum(axis=0), 1e-9)
        jumps = np.abs(np.diff(centroid))
        self.assertLess(float(jumps.max()), 0.35, "ligne de décalage détectée")

    def test_no_halo_brighter_than_paper(self) -> None:
        scan, _ = self._tilted_scan()
        rgb, alpha = self._process()
        # Intérieur du sujet (à plus de 2 px du contour, dont la couleur est
        # décontaminée du fond gris et peut varier d'un niveau d'arrondi).
        opaque = cv2.erode((alpha == 255).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        paper_max = scan.reshape(-1, 3).max(axis=0)
        self.assertTrue(np.all(rgb[opaque] <= paper_max), "halo plus clair que l'original")

    def test_edges_are_smooth_not_staircase(self) -> None:
        _, alpha = self._process()
        cols = np.nonzero(alpha.max(axis=0) == 255)[0][15:-15]
        # Hauteur du bord supérieur, au sous-pixel, colonne par colonne.
        top = alpha[:40, cols].astype(np.float64).sum(axis=0) / 255.0
        self.assertLess(float(np.ptp(top)), 0.8, "bord en escalier")

    def test_zero_angle_copies_pixels_unchanged(self) -> None:
        from image_pipeline.geometry import straighten_subject_without_clipping

        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
        alpha = np.full((40, 60), 255, dtype=np.uint8)
        subject = ProtectedSubject(
            original_rgb=rgb.copy(),
            protected_rgb=rgb.copy(),
            alpha=alpha,
            protected_core=np.ones((40, 60), dtype=np.uint8),
            old_background_rgb=(128, 128, 128),
        )
        result = straighten_subject_without_clipping(
            subject, OrientationEstimate(determinable=True), DEFAULT_CONFIG, padding_xy=(0, 0)
        )
        self.assertTrue(np.array_equal(result.rgb, rgb))


@unittest.skipIf(cv2 is None, "OpenCV n'est pas installé")
class DustTests(unittest.TestCase):
    def test_dust_specks_do_not_enlarge_the_subject(self) -> None:
        """Des poussières sur le fond ne doivent ni agrandir le cadrage ni
        empêcher la mesure de l'inclinaison (régression 1.7.0)."""

        from image_pipeline.geometry import detect_subject_inclination
        from image_pipeline.models import FrameData
        from image_pipeline.segmentation import (
            detect_subject_regions,
            detect_subjects_on_uniform_gray_background,
        )

        # Bords anticrénelés comme sur un vrai scan : rendu ×4 puis réduction.
        big = np.full((2400, 2000, 3), (42, 45, 43), dtype=np.uint8)
        box = cv2.boxPoints(((1000.0, 1200.0), (800.0, 1600.0), 1.8)).astype(np.int32)
        cv2.fillConvexPoly(big, box, (235, 140, 60))
        scan = cv2.resize(big, (500, 600), interpolation=cv2.INTER_AREA)
        for x, y in ((12, 15), (480, 20), (30, 570), (470, 585), (250, 30)):
            cv2.circle(scan, (x, y), 1, (220, 220, 220), -1)  # poussières

        frame = FrameData(
            source_path=Path("scan.tif"),
            frame_index=0,
            rgb=scan,
            source_alpha=np.full(scan.shape[:2], 255, dtype=np.uint8),
            dpi=(300.0, 300.0),
        )
        segmentation = detect_subjects_on_uniform_gray_background(frame, DEFAULT_CONFIG)
        self.assertIsNotNone(segmentation)
        regions = detect_subject_regions(segmentation.alpha, DEFAULT_CONFIG)
        self.assertEqual(len(regions), 1)
        ys, xs = np.nonzero(regions[0].alpha > 0)
        width = xs.max() - xs.min() + 1
        height = ys.max() - ys.min() + 1
        self.assertLess(width, 230, "une poussière a été rattachée au sujet")
        self.assertLess(height, 420, "une poussière a été rattachée au sujet")
        orientation = detect_subject_inclination(regions[0].alpha, DEFAULT_CONFIG)
        self.assertTrue(orientation.determinable)
        self.assertAlmostEqual(abs(orientation.correction_deg), 1.8, delta=0.3)


@unittest.skipIf(cv2 is None, "OpenCV n'est pas installé")
class CloseSubjectsTests(unittest.TestCase):
    def _scan(self, gap_px: int):
        from image_pipeline.models import FrameData

        k = 4
        big = np.full((600 * k, 700 * k, 3), (42, 45, 43), dtype=np.uint8)
        # Deux blocs posés presque bord à bord, séparés par un couloir sombre
        # (ombre des bords du papier) et se touchant sur une petite longueur.
        left = ((60 * k, 60 * k), ((340 - gap_px) * k, 540 * k))
        right = ((340 * k, 50 * k), (640 * k, 530 * k))
        cv2.rectangle(big, left[0], left[1], (235, 230, 215), -1)
        cv2.rectangle(big, right[0], right[1], (235, 230, 215), -1)
        cv2.rectangle(big, ((340 - gap_px) * k, 60 * k), (340 * k, 530 * k), (18, 18, 18), -1)
        cv2.rectangle(big, ((340 - gap_px) * k, 300 * k), (340 * k, 302 * k), (235, 230, 215), -1)
        for x in (150, 500):
            cv2.rectangle(big, ((x - 40) * k, 200 * k), ((x + 40) * k, 400 * k), (40, 40, 60), -1)
        scan = cv2.resize(big, (700, 600), interpolation=cv2.INTER_AREA)
        return FrameData(
            source_path=Path("scan.tif"),
            frame_index=0,
            rgb=scan,
            source_alpha=np.full(scan.shape[:2], 255, dtype=np.uint8),
            dpi=(300.0, 300.0),
        )

    def _regions(self, frame, config=DEFAULT_CONFIG):
        from image_pipeline.segmentation import (
            detect_subject_regions,
            detect_subjects_on_uniform_gray_background,
        )

        segmentation = detect_subjects_on_uniform_gray_background(frame, config)
        self.assertIsNotNone(segmentation)
        return detect_subject_regions(
            segmentation.alpha, config, separation_mask=segmentation.separation_mask
        )

    def test_blocks_separated_by_thin_dark_channel_are_split(self) -> None:
        regions = self._regions(self._scan(gap_px=3))
        self.assertEqual(len(regions), 2)
        # Ordre de lecture : le bloc de gauche d'abord, même posé plus bas.
        self.assertLess(regions[0].bbox_xyxy[0], regions[1].bbox_xyxy[0])
        # Chaque bloc est complet (aucune partie perdue par la séparation).
        for region in regions:
            ys, xs = np.nonzero(region.alpha > 0)
            self.assertGreater(ys.max() - ys.min(), 470)

    def test_separation_can_be_disabled(self) -> None:
        config = replace(DEFAULT_CONFIG, separate_close_subjects=False)
        self.assertEqual(len(self._regions(self._scan(gap_px=3), config)), 1)

    def test_dark_ink_inside_a_stamp_does_not_split_it(self) -> None:
        # Un seul élément avec une large zone d'encre sombre à l'intérieur.
        regions = self._regions(self._scan(gap_px=0))
        self.assertEqual(len(regions), 1)


@unittest.skipIf(cv2 is None, "OpenCV n'est pas installé")
class ThickDarkAreaTests(unittest.TestCase):
    def test_thick_dark_band_touching_the_edge_never_splits_an_item(self) -> None:
        """Une large bande d'encre sombre traversant un timbre de bord à bord
        ressemble au fond, mais elle est épaisse : ce n'est pas un couloir."""

        from image_pipeline.models import FrameData
        from image_pipeline.segmentation import (
            detect_subject_regions,
            detect_subjects_on_uniform_gray_background,
        )

        k = 4
        big = np.full((600 * k, 600 * k, 3), (42, 45, 43), dtype=np.uint8)
        cv2.rectangle(big, (100 * k, 80 * k), (500 * k, 520 * k), (235, 230, 215), -1)
        # Bande sombre presque neutre, de 40 px, d'un bord à l'autre.
        cv2.rectangle(big, (280 * k, 80 * k), (320 * k, 520 * k), (30, 32, 40), -1)
        scan = cv2.resize(big, (600, 600), interpolation=cv2.INTER_AREA)
        frame = FrameData(
            source_path=Path("scan.tif"),
            frame_index=0,
            rgb=scan,
            source_alpha=np.full(scan.shape[:2], 255, dtype=np.uint8),
            dpi=(300.0, 300.0),
        )
        segmentation = detect_subjects_on_uniform_gray_background(frame, DEFAULT_CONFIG)
        regions = detect_subject_regions(
            segmentation.alpha, DEFAULT_CONFIG, separation_mask=segmentation.separation_mask
        )
        self.assertEqual(len(regions), 1)


class ScannerEdgeStripTests(unittest.TestCase):
    def test_white_strip_along_bottom_edge_is_ignored(self) -> None:
        from image_pipeline.models import FrameData
        from image_pipeline.segmentation import detect_subjects_on_uniform_gray_background

        scan = np.full((500, 400, 3), (42, 45, 43), dtype=np.uint8)
        scan[-12:] = (235, 235, 235)  # carton de fond plus court que la vitre
        scan[100:300, 100:300] = (200, 120, 60)
        frame = FrameData(
            source_path=Path("scan.tif"),
            frame_index=0,
            rgb=scan,
            source_alpha=np.full(scan.shape[:2], 255, dtype=np.uint8),
            dpi=(300.0, 300.0),
        )
        segmentation = detect_subjects_on_uniform_gray_background(frame, DEFAULT_CONFIG)
        self.assertIsNotNone(segmentation, "la bande blanche fait échouer le scan")
        self.assertTrue(np.all(segmentation.alpha[-12:] == 0))
        ys, xs = np.nonzero(segmentation.alpha)
        self.assertEqual((ys.min(), ys.max(), xs.min(), xs.max()), (100, 299, 100, 299))
