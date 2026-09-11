"""Objets de données échangés entre les étapes du pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


UInt8Array = np.ndarray


@dataclass(slots=True)
class FrameData:
    source_path: Path
    frame_index: int
    rgb: UInt8Array
    source_alpha: UInt8Array
    dpi: tuple[float, float] | None = None
    source_icc_profile: bytes | None = None


@dataclass(slots=True)
class SegmentationResult:
    original_rgb: UInt8Array
    edge_rgb: UInt8Array
    alpha: UInt8Array
    old_background_rgb: tuple[int, int, int]


@dataclass(slots=True)
class SubjectRegion:
    detection_order: int
    bbox_xyxy: tuple[int, int, int, int]
    alpha: UInt8Array


@dataclass(slots=True)
class ProtectedSubject:
    original_rgb: UInt8Array
    protected_rgb: UInt8Array
    alpha: UInt8Array
    protected_core: UInt8Array
    old_background_rgb: tuple[int, int, int]


@dataclass(slots=True)
class OrientationEstimate:
    axis_angle_deg: float = 0.0
    correction_deg: float = 0.0
    target_axis_deg: float = 0.0
    confidence: float = 0.0
    method: str = "indéterminé"
    determinable: bool = False
    reason: str = ""


@dataclass(slots=True)
class RotationResult:
    rgb: UInt8Array
    expected_original_rgb: UInt8Array
    alpha: UInt8Array
    protected_core: UInt8Array
    affine_matrix: np.ndarray
    orientation_before: OrientationEstimate
    alpha_mass_before: float
    perimeter_before: float
    aspect_ratio_before: float | None
    old_background_rgb: tuple[int, int, int]


@dataclass(slots=True)
class CompositeResult:
    image: Image.Image
    background_values: tuple[int, ...]
    target_mode: str
    cmyk_certifiable: bool
    exact_subject_color_preservation: bool = False
    output_icc_profile: bytes | None = None


@dataclass(slots=True)
class QCCheck:
    name: str
    passed: bool
    message: str
    measured: Any = None
    expected: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "measured": self.measured,
            "expected": self.expected,
        }


@dataclass(slots=True)
class QCReport:
    source: str
    frame_index: int
    subject_index: int
    output: str
    checks: list[QCCheck] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def problems(self) -> list[str]:
        return [check.message for check in self.checks if not check.passed]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "frame_index": self.frame_index,
            "subject_index": self.subject_index,
            "output": self.output,
            "status": "OK" if self.passed else "AVERTISSEMENT",
            "checks": [check.as_dict() for check in self.checks],
            "metrics": self.metrics,
        }


@dataclass(slots=True)
class ProcessSummary:
    sources_found: int = 0
    sources_processed: int = 0
    frames_processed: int = 0
    outputs_written: int = 0
    qc_failures: int = 0
    processing_errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "sources_found": self.sources_found,
            "sources_processed": self.sources_processed,
            "frames_processed": self.frames_processed,
            "outputs_written": self.outputs_written,
            "qc_failures": self.qc_failures,
            "processing_errors": self.processing_errors,
        }
