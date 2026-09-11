"""Segmentation, séparation optionnelle des sujets et protection des pixels."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy import ndimage

from .config import PipelineConfig
from .models import (
    FrameData,
    ProtectedSubject,
    SegmentationResult,
    SubjectRegion,
)


class SegmentationError(RuntimeError):
    """La segmentation n'a pas produit un masque exploitable."""


def estimate_old_background_color(
    rgb: np.ndarray,
    alpha: np.ndarray,
    background_threshold: int,
) -> tuple[int, int, int]:
    """Estime la couleur de l'ancien fond pour le détecteur de halo."""

    background_pixels = rgb[alpha <= background_threshold]
    if background_pixels.size == 0:
        corners = np.concatenate(
            (
                rgb[0, :, :],
                rgb[-1, :, :],
                rgb[:, 0, :],
                rgb[:, -1, :],
            ),
            axis=0,
        )
        background_pixels = corners
    median = np.median(background_pixels, axis=0)
    return tuple(int(round(value)) for value in median)  # type: ignore[return-value]


def _outer_border_selector(shape: tuple[int, int]) -> np.ndarray:
    """Sélectionne une bande extérieure utilisée comme échantillon du fond."""

    height, width = shape
    thickness = max(2, min(16, int(round(min(height, width) * 0.01))))
    selector = np.zeros((height, width), dtype=bool)
    selector[:thickness, :] = True
    selector[-thickness:, :] = True
    selector[:, :thickness] = True
    selector[:, -thickness:] = True
    return selector


def _remove_shallow_border_artifacts(
    subject: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """Retire les petites bandes du scanner collées au cadre extérieur.

    Une composante n'est retirée que si elle touche le bord, reste peu profonde
    depuis ce bord et occupe une faible partie de l'image. Un élément important
    ou profondément engagé dans l'image est donc conservé.
    """

    if not config.remove_shallow_border_artifacts:
        return subject

    height, width = subject.shape
    image_area = height * width
    max_area = image_area * config.gray_border_artifact_max_area_ratio
    max_vertical_depth = height * config.gray_border_artifact_max_depth_ratio
    max_horizontal_depth = width * config.gray_border_artifact_max_depth_ratio
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        subject.astype(np.uint8),
        connectivity=8,
    )
    cleaned = subject.copy()
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_top = y == 0
        touches_bottom = y + component_height == height
        touches_left = x == 0
        touches_right = x + component_width == width
        shallow = (
            ((touches_top or touches_bottom) and component_height <= max_vertical_depth)
            or ((touches_left or touches_right) and component_width <= max_horizontal_depth)
        )
        if shallow and area <= max_area:
            cleaned[labels == label] = False
    return cleaned


def detect_subjects_on_uniform_gray_background(
    frame: FrameData,
    config: PipelineConfig,
) -> SegmentationResult | None:
    """Détoure sans IA les éléments posés sur un fond gris uniforme.

    Seuls les pixels ressemblant au gris mesuré sur le cadre *et reliés au
    cadre* deviennent transparents. Une couleur identique située à l'intérieur
    d'un timbre reste donc intacte. Le masque n'est ni érodé, ni fermé, ni
    lissé : dentelures et bordures sont conservées à la résolution du scan.
    ``None`` indique que le fond ne peut pas être identifié avec assez de
    sécurité par cette méthode déterministe.
    """

    if not config.prefer_uniform_gray_background:
        return None

    border = _outer_border_selector(frame.rgb.shape[:2])
    border_rgb = frame.rgb[border].astype(np.float32)
    median_rgb = np.median(border_rgb, axis=0)
    if float(np.ptp(median_rgb)) > config.gray_background_neutral_tolerance:
        return None

    lab = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border_lab = lab[border]
    median_lab = np.median(border_lab, axis=0)
    border_distances = np.linalg.norm(border_lab - median_lab, axis=1)
    # Le décile le plus éloigné est volontairement ignoré : il peut contenir
    # une languette du scanner ou une poussière sans représenter le vrai fond.
    spread = float(np.percentile(border_distances, 90.0))
    if spread > config.gray_background_max_spread:
        return None

    adaptive_distance = float(
        np.percentile(border_distances, config.gray_background_percentile)
        + config.gray_background_slack
    )
    threshold = float(
        np.clip(
            adaptive_distance,
            config.gray_background_min_distance,
            config.gray_background_max_distance,
        )
    )
    distance_to_gray = np.linalg.norm(lab - median_lab, axis=2)
    gray_like = (distance_to_gray <= threshold).astype(np.uint8)

    count, labels = cv2.connectedComponents(gray_like, connectivity=8)
    if count <= 1:
        return None
    outside_labels = np.unique(labels[border & (labels > 0)])
    if outside_labels.size == 0:
        return None
    exterior_gray = np.isin(labels, outside_labels)
    subject = _remove_shallow_border_artifacts(~exterior_gray, config)
    subject_fraction = float(np.mean(subject))
    if (
        subject_fraction <= config.min_subject_area_ratio
        or subject_fraction >= config.gray_background_max_subject_fraction
    ):
        return None

    alpha = np.where(subject, 255, 0).astype(np.uint8)
    background_rgb = tuple(int(round(value)) for value in median_rgb)
    return SegmentationResult(
        original_rgb=frame.rgb,
        edge_rgb=frame.rgb.copy(),
        alpha=alpha,
        old_background_rgb=background_rgb,  # type: ignore[arg-type]
    )


def generate_precise_cutout(
    frame: FrameData,
    logger: logging.Logger,
    config: PipelineConfig,
) -> SegmentationResult:
    """Détoure par alpha existant ou par détection du fond gris extérieur."""

    if config.trust_existing_alpha and np.any(frame.source_alpha < 255):
        # Un alpha déjà présent constitue généralement une information plus
        # précise que toute nouvelle prédiction. Le réutiliser évite de rogner
        # un détourage validé en amont.
        alpha = frame.source_alpha.copy()
        if not np.any(alpha > 0):
            raise SegmentationError("Le canal alpha source est entièrement vide.")
        background_rgb = estimate_old_background_color(
            frame.rgb,
            alpha,
            config.existing_alpha_background_threshold,
        )
        return SegmentationResult(
            original_rgb=frame.rgb,
            edge_rgb=frame.rgb.copy(),
            alpha=alpha,
            old_background_rgb=background_rgb,
        )

    gray_result = detect_subjects_on_uniform_gray_background(frame, config)
    if gray_result is not None:
        logger.info(
            "  Fond gris uniforme détecté : détourage conservateur pleine résolution"
        )
        return gray_result

    raise SegmentationError(
        "Fond extérieur non exploitable par la méthode sans IA : "
        "le fond gris doit être suffisamment uniforme, neutre et visible "
        "tout autour des éléments."
    )


def _expanded_bbox(
    selector: np.ndarray,
    padding: int,
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(selector)
    if xs.size == 0:
        raise SegmentationError("Région de sujet vide.")
    height, width = selector.shape
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(width, int(xs.max()) + 1 + padding)
    y1 = min(height, int(ys.max()) + 1 + padding)
    return x0, y0, x1, y1


def detect_subject_regions(
    alpha: np.ndarray,
    config: PipelineConfig,
) -> list[SubjectRegion]:
    """Détecte un canevas global ou plusieurs sujets indépendants.

    En mode multi-sujets, les composantes sont regroupées après dilatation.
    Avec un alpha source semi-transparent, les pixels incertains peuvent être
    affectés à l'ancre la plus proche afin de conserver les petits détails.
    """

    present = alpha > 0
    if not np.any(present):
        raise SegmentationError("Le masque est vide.")

    if not config.split_subjects:
        bbox = _expanded_bbox(present, config.crop_padding_px)
        x0, y0, x1, y1 = bbox
        return [
            SubjectRegion(
                detection_order=1,
                bbox_xyxy=bbox,
                alpha=alpha[y0:y1, x0:x1].copy(),
            )
        ]

    seed = (alpha >= config.component_seed_alpha).astype(np.uint8)
    if not np.any(seed):
        seed = present.astype(np.uint8)
    gap = max(0, config.component_grouping_gap_px)
    if gap:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (gap * 2 + 1, gap * 2 + 1),
        )
        grouped = cv2.dilate(seed, kernel, iterations=1)
    else:
        grouped = seed

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        grouped,
        connectivity=8,
    )
    image_area = alpha.shape[0] * alpha.shape[1]
    minimum_area = max(1, int(image_area * config.min_subject_area_ratio))
    valid_original_labels: list[int] = []
    for label in range(1, count):
        # Mesure l'ancre réelle plutôt que sa version dilatée.
        real_area = int(np.count_nonzero(seed & (labels == label)))
        if real_area >= minimum_area:
            valid_original_labels.append(label)

    if not valid_original_labels:
        # Préservation prioritaire : ne pas jeter le seul élément détecté parce
        # qu'il est petit.
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        valid_original_labels = [largest]

    anchors = np.zeros_like(labels, dtype=np.int32)
    for new_label, old_label in enumerate(valid_original_labels, start=1):
        anchors[labels == old_label] = new_label

    has_soft_alpha = bool(np.any((alpha > 0) & (alpha < 255)))
    if config.retain_all_uncertain_pixels and has_soft_alpha:
        _, nearest_indices = ndimage.distance_transform_edt(
            anchors == 0,
            return_indices=True,
        )
        assigned_labels = anchors[tuple(nearest_indices)]
    else:
        assigned_labels = anchors

    region_payloads: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for label in range(1, len(valid_original_labels) + 1):
        selector = present & (assigned_labels == label)
        if not np.any(selector):
            continue
        bbox = _expanded_bbox(selector, config.crop_padding_px)
        x0, y0, x1, y1 = bbox
        region_alpha = np.where(selector, alpha, 0)[y0:y1, x0:x1].copy()
        region_payloads.append((bbox, region_alpha))

    # Ordre explicite : du haut vers le bas, puis de gauche à droite.
    region_payloads.sort(key=lambda item: (item[0][1], item[0][0]))
    return [
        SubjectRegion(index, bbox, region_alpha)
        for index, (bbox, region_alpha) in enumerate(region_payloads, start=1)
    ]


def protect_subject_interior(
    segmentation: SegmentationResult,
    region: SubjectRegion,
    config: PipelineConfig,
) -> ProtectedSubject:
    """Restaure exactement les pixels originaux dans le cœur opaque du sujet.

    Les éventuels pixels semi-transparents d'un alpha source sont conservés ;
    le cœur opaque reprend toujours exactement les couleurs de l'original.
    """

    x0, y0, x1, y1 = region.bbox_xyxy
    original = segmentation.original_rgb[y0:y1, x0:x1].copy()
    edge_rgb = segmentation.edge_rgb[y0:y1, x0:x1].copy()
    alpha = region.alpha
    if original.shape[:2] != alpha.shape:
        raise SegmentationError("Dimensions incohérentes lors de l'extraction du sujet.")

    protected_core = alpha >= config.protected_core_alpha
    protected_rgb = edge_rgb
    protected_rgb[protected_core] = original[protected_core]

    if not np.array_equal(protected_rgb[protected_core], original[protected_core]):
        raise SegmentationError("La protection des pixels intérieurs a échoué.")

    return ProtectedSubject(
        original_rgb=original,
        protected_rgb=protected_rgb,
        alpha=alpha.copy(),
        protected_core=protected_core.astype(np.uint8),
        old_background_rgb=segmentation.old_background_rgb,
    )
