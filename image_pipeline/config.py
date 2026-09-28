"""Configuration centralisée du pipeline.

Les paramètres les plus fréquemment modifiés sont regroupés dans
``DEFAULT_CONFIG`` à la fin de ce fichier.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Paramètres immuables d'une exécution du pipeline."""

    # Entrées / sorties
    input_dir: Path = Path("input")
    output_dir: Path = Path("output")
    recursive: bool = True
    supported_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    )
    max_megapixels: float = 120.0

    # Un alpha source validé est toujours prioritaire.
    existing_alpha_background_threshold: int = 10
    protected_core_alpha: int = 250
    trust_existing_alpha: bool = True

    # Méthode prioritaire pour les scans sur fond gris uniforme. Elle retire
    # uniquement le fond connecté au cadre et ne demande aucune prédiction IA.
    prefer_uniform_gray_background: bool = True
    # Couverture partielle (anticrénelage) des pixels du contour, calculée à
    # partir du mélange sujet/fond gris : évite l'escalier du bord après
    # redressement. L'intérieur du sujet n'est pas concerné.
    subpixel_edges: bool = True
    # Bande étroite d'une autre couleur le long d'un bord du scan (carton de
    # fond plus petit que la vitre, marge blanche du scanner) : exclue avant
    # la mesure du fond. Au-delà de 5 % de la dimension, elle n'est pas
    # considérée comme parasite et le scan est refusé comme avant.
    ignore_scanner_edge_strips: bool = True
    scanner_edge_max_fraction: float = 0.05
    scanner_edge_safety_px: int = 3
    gray_background_neutral_tolerance: int = 32
    gray_background_max_spread: float = 25.0
    gray_background_min_distance: float = 6.0
    gray_background_max_distance: float = 22.0
    gray_background_percentile: float = 99.0
    gray_background_slack: float = 3.0
    gray_background_max_subject_fraction: float = 0.92
    remove_shallow_border_artifacts: bool = True
    gray_border_artifact_max_depth_ratio: float = 0.05
    gray_border_artifact_max_area_ratio: float = 0.02

    # Détection / extraction. Par défaut, tous les éléments du masque restent
    # sur un même canevas afin de ne pas séparer accidentellement un détail.
    split_subjects: bool = True
    component_seed_alpha: int = 32
    component_grouping_gap_px: int = 2
    min_subject_area_ratio: float = 0.001
    # Séparation d'éléments posés presque bord à bord : le couloir de fond
    # (même plus sombre que le fond, ombre du papier) relié au fond extérieur
    # est retiré, puis une érosion coupe les points de contact. N'agit que sur
    # le découpage en sujets, jamais sur le détourage final.
    separate_close_subjects: bool = True
    separation_background_distance: float = 60.0
    separation_erosion_px: int = 3
    # Largeur maximale (px) d'un couloir de séparation : une zone sombre plus
    # large appartient au dessin d'un timbre et ne coupe jamais un élément.
    separation_max_channel_px: int = 8
    retain_all_uncertain_pixels: bool = True
    # Distance maximale (px) entre un pixel semi-transparent et le sujet
    # auquel il est rattaché.
    uncertain_pixel_max_distance_px: int = 16
    crop_padding_px: int = 48

    # Redressement géométrique
    straighten: bool = True
    orientation_alpha_threshold: int = 64
    min_orientation_anisotropy: float = 0.08
    min_rectangularity: float = 0.72
    min_rotation_deg: float = 0.25
    max_rotation_deg: float = 45.0
    final_padding_px: int = 32

    # Marge physique ajoutée autour de chaque sujet extrait. La résolution du
    # TIFF source est conservée ; 300 dpi sont utilisés si elle est absente.
    output_margin_mm: float = 1.0
    fallback_dpi: float = 300.0

    # Fond demandé : C=84 %, M=82 %, J=73 %, N=95 %.
    cmyk_background_percent: tuple[float, float, float, float] = (
        84.0,
        82.0,
        73.0,
        95.0,
    )
    # Les scans fournis sont des TIFF RVB sans profil ICC. Une conversion de
    # tout le document en CMJN modifierait donc leur apparence selon le logiciel
    # d'affichage. Par défaut, le sujet reste en RVB sRGB et seul le fond reçoit
    # l'équivalent visuel demandé. #00000c est la couleur de fond de référence.
    preserve_subject_rgb: bool = True
    rgb_background: tuple[int, int, int] = (5, 0, 2)
    output_format: str = "TIFF"  # TIFF recommandé | JPEG | PNG (aperçu RVB)
    output_cmyk_icc_profile: Path | None = None
    tiff_compression: str = "tiff_lzw"
    jpeg_quality: int = 100

    # Contrôle qualité
    qa_alpha_area_tolerance: float = 0.02
    qa_perimeter_tolerance: float = 0.08
    qa_aspect_ratio_tolerance: float = 0.02
    qa_residual_skew_deg: float = 0.75
    qa_min_border_margin_px: int = 4
    qa_halo_color_distance: float = 18.0
    qa_halo_advantage: float = 5.0
    qa_max_suspect_halo_fraction: float = 0.15
    nonzero_exit_on_qc_failure: bool = True

    # Journalisation
    log_filename: str = "traitements.log"
    journal_filename: str = "controle_qualite.jsonl"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_dir", Path(self.input_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.output_cmyk_icc_profile is not None:
            object.__setattr__(
                self,
                "output_cmyk_icc_profile",
                Path(self.output_cmyk_icc_profile),
            )

        if self.output_format.upper() not in {"TIFF", "JPEG", "PNG"}:
            raise ValueError(f"format de sortie non pris en charge : {self.output_format}")
        if len(self.cmyk_background_percent) != 4 or any(
            not 0.0 <= value <= 100.0 for value in self.cmyk_background_percent
        ):
            raise ValueError("La couleur CMJN doit contenir quatre valeurs entre 0 et 100.")
        if len(self.rgb_background) != 3 or any(
            not 0 <= value <= 255 for value in self.rgb_background
        ):
            raise ValueError("La couleur RVB doit contenir trois valeurs entre 0 et 255.")
        if not 0 <= self.existing_alpha_background_threshold < 255:
            raise ValueError("Seuil alpha de fond invalide.")
        if not 0 < self.protected_core_alpha <= 255:
            raise ValueError("protected_core_alpha doit être compris entre 1 et 255.")
        if not 0 <= self.gray_background_neutral_tolerance <= 255:
            raise ValueError("La tolérance de neutralité du fond gris est invalide.")
        if not 0 <= self.gray_background_percentile <= 100:
            raise ValueError("Le percentile du fond gris doit être compris entre 0 et 100.")
        if not 0 < self.gray_background_max_subject_fraction < 1:
            raise ValueError("La fraction maximale de sujet sur fond gris est invalide.")
        if not 0 <= self.gray_border_artifact_max_depth_ratio <= 0.25:
            raise ValueError("La profondeur maximale des résidus de bord est invalide.")
        if not 0 <= self.gray_border_artifact_max_area_ratio <= 0.25:
            raise ValueError("L'aire maximale des résidus de bord est invalide.")
        if (
            self.gray_background_min_distance < 0
            or self.gray_background_max_distance < self.gray_background_min_distance
        ):
            raise ValueError("Les distances du fond gris sont invalides.")
        if self.crop_padding_px < 0 or self.final_padding_px < 0:
            raise ValueError("Les marges ne peuvent pas être négatives.")
        if self.output_margin_mm < 0:
            raise ValueError("La marge physique ne peut pas être négative.")
        if self.fallback_dpi <= 0:
            raise ValueError("La résolution de repli doit être positive.")
        if self.max_megapixels <= 0:
            raise ValueError("max_megapixels doit être strictement positif.")

    @property
    def output_extension(self) -> str:
        return {"TIFF": ".tif", "JPEG": ".jpg", "PNG": ".png"}[
            self.output_format.upper()
        ]

    def serializable(self) -> dict[str, Any]:
        """Retourne une version JSON-compatible de la configuration."""

        data = asdict(self)
        for key, value in tuple(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data

    def effective_dpi(
        self,
        source_dpi: tuple[float, float] | None,
    ) -> tuple[float, float]:
        """Retourne une résolution exploitable pour la marge et le TIFF."""

        if source_dpi is None:
            return self.fallback_dpi, self.fallback_dpi
        x_dpi, y_dpi = source_dpi
        if x_dpi <= 0 or y_dpi <= 0:
            return self.fallback_dpi, self.fallback_dpi
        return float(x_dpi), float(y_dpi)

    def output_margin_pixels(
        self,
        source_dpi: tuple[float, float] | None,
    ) -> tuple[int, int]:
        """Convertit la marge millimétrique en pixels X/Y, sans être en dessous."""

        return self.millimeters_to_pixels(self.output_margin_mm, source_dpi)

    def millimeters_to_pixels(
        self,
        millimeters: float,
        source_dpi: tuple[float, float] | None,
    ) -> tuple[int, int]:
        """Convertit une longueur physique en pixels X/Y par excès."""

        x_dpi, y_dpi = self.effective_dpi(source_dpi)
        x_pixels = int(math.ceil(millimeters * x_dpi / 25.4))
        y_pixels = int(math.ceil(millimeters * y_dpi / 25.4))
        return max(0, x_pixels), max(0, y_pixels)


# ---------------------------------------------------------------------------
# PARAMÈTRES PRINCIPAUX À MODIFIER SI LE SCRIPT EST UTILISÉ SANS ARGUMENTS CLI
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = PipelineConfig(
    input_dir=Path("input"),
    output_dir=Path("output"),
    split_subjects=True,
    straighten=True,
    cmyk_background_percent=(84.0, 82.0, 73.0, 95.0),
    output_format="TIFF",
)
