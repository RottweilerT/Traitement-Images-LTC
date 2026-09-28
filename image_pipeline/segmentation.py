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
    d'un timbre reste donc intacte. Le masque n'est ni érodé ni fermé :
    dentelures et bordures sont conservées à la résolution du scan ; seule la
    rangée de pixels du contour reçoit une transparence partielle (voir
    ``_subpixel_edge_alpha``). Une bande parasite étroite le long d'un bord du
    scan est exclue au préalable (voir ``_scanner_edge_strips``).
    ``None`` indique que le fond ne peut pas être identifié avec assez de
    sécurité par cette méthode déterministe.
    """

    if not config.prefer_uniform_gray_background:
        return None

    rgb = frame.rgb
    top, bottom, left, right = _scanner_edge_strips(rgb, config)
    if top or bottom or left or right:
        # Bandes étroites le long des bords du scan (bord du carton noir,
        # marge blanche du scanner) : exclues avant la mesure du fond, puis
        # rendues transparentes dans le résultat.
        height, width = rgb.shape[:2]
        inner = rgb[top : height - bottom, left : width - right]
        result = _detect_on_gray_area(inner, config)
        if result is None:
            return None
        window = (slice(top, height - bottom), slice(left, width - right))
        alpha = np.zeros((height, width), dtype=np.uint8)
        alpha[window] = result.alpha
        edge_rgb = rgb.copy()
        edge_rgb[window] = result.edge_rgb
        separation_mask = None
        if result.separation_mask is not None:
            separation_mask = np.zeros((height, width), dtype=np.uint8)
            separation_mask[window] = result.separation_mask
        return SegmentationResult(
            original_rgb=rgb,
            edge_rgb=edge_rgb,
            alpha=alpha,
            old_background_rgb=result.old_background_rgb,
            separation_mask=separation_mask,
        )
    return _detect_on_gray_area(rgb, config)


def _scanner_edge_strips(
    rgb: np.ndarray,
    config: PipelineConfig,
) -> tuple[int, int, int, int]:
    """Largeur (px) des bandes parasites le long des 4 bords du scan.

    Le fond de référence est la médiane des 4 côtés. Un côté dont la couleur
    s'en écarte nettement est parcouru vers l'intérieur, ligne par ligne,
    tant que la ligne entière (sa médiane) reste différente du fond. Si la
    bande dépasse ``scanner_edge_max_fraction`` de la dimension, ce n'est pas
    une bande parasite : rien n'est exclu et la détection échoue normalement.
    """

    if not config.ignore_scanner_edge_strips:
        return 0, 0, 0, 0
    height, width = rgb.shape[:2]
    band = max(4, int(round(min(height, width) * 0.01)))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    side_medians = [
        np.median(lab[:band].reshape(-1, 3), axis=0),
        np.median(lab[-band:].reshape(-1, 3), axis=0),
        np.median(lab[:, :band].reshape(-1, 3), axis=0),
        np.median(lab[:, -band:].reshape(-1, 3), axis=0),
    ]
    reference = np.median(np.stack(side_medians), axis=0)
    limit = config.gray_background_max_spread

    def walk(lines: np.ndarray, maximum: int) -> int:
        # lines : médiane Lab de chaque ligne, du bord vers l'intérieur.
        count = 0
        for value in lines[: maximum + 1]:
            if float(np.linalg.norm(value - reference)) > limit:
                count += 1
            else:
                break
        if count == 0 or count > maximum:
            return 0
        return count + config.scanner_edge_safety_px

    max_rows = int(height * config.scanner_edge_max_fraction)
    max_cols = int(width * config.scanner_edge_max_fraction)
    depth_rows = min(height, max_rows + 1)
    depth_cols = min(width, max_cols + 1)
    top_lines = np.median(lab[:depth_rows], axis=1)
    bottom_lines = np.median(lab[::-1][:depth_rows], axis=1)
    left_lines = np.median(lab[:, :depth_cols], axis=0)
    right_lines = np.median(lab[:, ::-1][:, :depth_cols], axis=0)
    strips = [
        walk(top_lines, max_rows) if np.linalg.norm(side_medians[0] - reference) > limit else 0,
        walk(bottom_lines, max_rows) if np.linalg.norm(side_medians[1] - reference) > limit else 0,
        walk(left_lines, max_cols) if np.linalg.norm(side_medians[2] - reference) > limit else 0,
        walk(right_lines, max_cols) if np.linalg.norm(side_medians[3] - reference) > limit else 0,
    ]
    return strips[0], strips[1], strips[2], strips[3]


def _detect_on_gray_area(
    rgb: np.ndarray,
    config: PipelineConfig,
) -> SegmentationResult | None:
    """Détection du fond gris sur une zone de scan (voir fonction publique)."""

    border = _outer_border_selector(rgb.shape[:2])
    border_rgb = rgb[border].astype(np.float32)
    median_rgb = np.median(border_rgb, axis=0)
    if float(np.ptp(median_rgb)) > config.gray_background_neutral_tolerance:
        return None

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
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

    background_rgb = tuple(int(round(value)) for value in median_rgb)
    separation_mask = _separation_mask(
        subject,
        exterior_gray,
        distance_to_gray,
        threshold,
        config,
    )
    if config.subpixel_edges:
        alpha, edge_rgb = _subpixel_edge_alpha(rgb, subject, median_rgb)
    else:
        alpha = np.where(subject, 255, 0).astype(np.uint8)
        edge_rgb = rgb.copy()
    return SegmentationResult(
        original_rgb=rgb,
        edge_rgb=edge_rgb,
        alpha=alpha,
        old_background_rgb=background_rgb,  # type: ignore[arg-type]
        separation_mask=separation_mask,
    )


def _separation_mask(
    subject: np.ndarray,
    exterior_gray: np.ndarray,
    distance_to_gray: np.ndarray,
    threshold: float,
    config: PipelineConfig,
) -> np.ndarray | None:
    """Masque des pixels sûrs du sujet, utilisé seulement pour séparer.

    Entre deux éléments posés presque bord à bord, le fond visible n'est qu'un
    couloir de quelques pixels, souvent plus sombre que le fond (ombre des
    bords du papier) : il n'est donc pas reconnu comme fond et relie les deux
    éléments. Ici, les pixels proches de la couleur du fond (tolérance élargie)
    *et reliés au fond extérieur* sont retirés, puis une légère érosion coupe
    les points de contact de 1 à 2 pixels. Les zones sombres enfermées à
    l'intérieur d'un timbre ne sont pas concernées, car non reliées au fond.
    """

    if not config.separate_close_subjects:
        return None
    wide = max(float(config.separation_background_distance), threshold)
    near_background = (distance_to_gray <= wide).astype(np.uint8)
    count, labels = cv2.connectedComponents(near_background, connectivity=8)
    if count <= 1:
        return None
    touching = np.unique(labels[exterior_gray & (labels > 0)])
    channel = np.isin(labels, touching[touching > 0]) & subject
    # Seul un couloir *étroit* sépare deux éléments. Une zone sombre épaisse
    # (encre foncée, photo sombre, cadre coloré) touchant le bord d'un timbre
    # n'est pas un couloir : l'ouverture morphologique la retrouve et elle est
    # conservée dans le sujet.
    half_width = max(1, int(config.separation_max_channel_px) // 2)
    disk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * half_width + 1, 2 * half_width + 1),
    )
    thick = cv2.morphologyEx(channel.astype(np.uint8), cv2.MORPH_OPEN, disk) > 0
    channel = channel & ~cv2.dilate(thick.astype(np.uint8), disk).astype(bool)
    safe = subject & ~channel
    radius = max(0, int(config.separation_erosion_px))
    if radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )
        safe = cv2.erode(safe.astype(np.uint8), kernel) > 0
    return safe.astype(np.uint8)


# Distance minimale (RVB) entre le sujet et le fond pour estimer une
# couverture partielle fiable ; en dessous, le contour reste binaire.
_EDGE_MIN_CONTRAST = 24.0
# Couverture en dessous de laquelle un pixel extérieur reste transparent.
_EDGE_MIN_ALPHA = 8
# Distance maximale (px) jusqu'au pixel plein servant de couleur de référence.
_EDGE_MAX_REFERENCE_DISTANCE = 4.0


def _subpixel_edge_alpha(
    rgb: np.ndarray,
    subject: np.ndarray,
    background_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estime la couverture partielle des seuls pixels du contour.

    Au bord d'un sujet incliné, un pixel du scan est en partie sujet et en
    partie fond gris : ``pixel = a × sujet + (1 - a) × fond``. Un masque
    binaire transforme ce bord en escalier, qui devient une ligne pointillée
    une fois le sujet redressé. Ici, sur la seule bande d'un pixel de part et
    d'autre du contour, la proportion ``a`` est calculée à partir de la couleur
    du sujet voisin (prise deux pixels à l'intérieur) et du fond mesuré. La
    couleur de ces pixels est « décontaminée » (le gris est retiré) dans
    ``edge_rgb``. L'intérieur du sujet n'est jamais modifié.
    """

    subject_u8 = subject.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    inner_ring = subject & ~(cv2.erode(subject_u8, kernel) > 0)
    outer_ring = (cv2.dilate(subject_u8, kernel) > 0) & ~subject
    band = inner_ring | outer_ring

    alpha = np.where(subject, 255, 0).astype(np.uint8)
    edge_rgb = rgb.copy()
    if not np.any(band):
        return alpha, edge_rgb

    # Couleur de référence du sujet : pixel plein le plus proche, à au moins
    # deux pixels du contour (non mélangé au fond).
    solid = cv2.erode(subject_u8, kernel, iterations=2) > 0
    if not np.any(solid):
        solid = subject
    distances, indices = ndimage.distance_transform_edt(~solid, return_indices=True)
    band_y, band_x = np.nonzero(band)
    ref_y = indices[0][band_y, band_x]
    ref_x = indices[1][band_y, band_x]
    # Détail fin ou isolé (poussière, pointe) : aucun pixel plein à proximité,
    # la couleur de référence serait celle d'un autre objet. On garde alors
    # la décision binaire d'origine.
    near_solid = distances[band_y, band_x] <= _EDGE_MAX_REFERENCE_DISTANCE

    pixel = rgb[band_y, band_x].astype(np.float64)
    foreground = rgb[ref_y, ref_x].astype(np.float64)
    background = np.asarray(background_rgb, dtype=np.float64)[None, :]
    direction = foreground - background
    contrast_sq = np.sum(direction * direction, axis=1)
    reliable = (contrast_sq >= _EDGE_MIN_CONTRAST**2) & near_solid
    coverage = np.sum((pixel - background) * direction, axis=1) / np.maximum(
        contrast_sq, 1e-9
    )
    coverage = np.clip(coverage, 0.0, 1.0)

    band_alpha = np.rint(coverage * 255.0).astype(np.int32)
    band_alpha[band_alpha < _EDGE_MIN_ALPHA] = 0
    # Contraste insuffisant : on garde la décision binaire d'origine.
    was_subject = subject[band_y, band_x]
    band_alpha = np.where(reliable, band_alpha, np.where(was_subject, 255, 0))
    alpha[band_y, band_x] = band_alpha.astype(np.uint8)

    # Couleur décontaminée : on retire la part de fond gris du pixel mélangé.
    a = np.maximum(band_alpha / 255.0, 1e-6)[:, None]
    decontaminated = background + (pixel - background) / a
    decontaminated = np.where(
        (reliable & (band_alpha > 0) & (band_alpha < 255))[:, None],
        np.clip(np.rint(decontaminated), 0, 255),
        pixel,
    )
    edge_rgb[band_y, band_x] = decontaminated.astype(np.uint8)
    return alpha, edge_rgb


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


def _reading_order(
    payloads: list[tuple[tuple[int, int, int, int], np.ndarray]],
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Ordre de lecture : rangées de haut en bas, puis de gauche à droite.

    Deux éléments côte à côte dont les hauteurs se chevauchent largement sont
    dans la même rangée, même si l'un est posé quelques pixels plus haut.
    """

    remaining = sorted(payloads, key=lambda item: item[0][1])
    ordered: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    while remaining:
        first = remaining.pop(0)
        row = [first]
        row_top, row_bottom = first[0][1], first[0][3]
        rest = []
        for item in remaining:
            _, top, _, bottom = item[0]
            overlap = min(bottom, row_bottom) - max(top, row_top)
            shortest = min(bottom - top, row_bottom - row_top)
            if shortest > 0 and overlap >= 0.5 * shortest:
                row.append(item)
            else:
                rest.append(item)
        row.sort(key=lambda item: item[0][0])
        ordered.extend(row)
        remaining = rest
    return ordered


def detect_subject_regions(
    alpha: np.ndarray,
    config: PipelineConfig,
    separation_mask: np.ndarray | None = None,
) -> list[SubjectRegion]:
    """Détecte un canevas global ou plusieurs sujets indépendants.

    En mode multi-sujets, les composantes sont regroupées après dilatation.
    Avec ``separation_mask``, les ancres sont prises dans ce masque, où les
    couloirs étroits entre éléments proches sont retirés. Les pixels opaques
    d'un élément ancré et les pixels semi-transparents proches sont ensuite
    rattachés à l'ancre la plus proche ; une poussière isolée est écartée.
    L'ordre de sortie suit la lecture : rangées de haut en bas, puis de gauche
    à droite.
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
    if separation_mask is not None:
        # Graines issues du masque de séparation : les couloirs étroits entre
        # deux éléments proches n'y relient plus les éléments.
        separated = (seed > 0) & (separation_mask > 0)
        if np.any(separated):
            seed = separated.astype(np.uint8)
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
    needs_assignment = separation_mask is not None or (
        config.retain_all_uncertain_pixels and has_soft_alpha
    )
    if needs_assignment:
        distances, nearest_indices = ndimage.distance_transform_edt(
            anchors == 0,
            return_indices=True,
        )
        assigned_labels = anchors[tuple(nearest_indices)]
        # 1) Pixels opaques d'un élément dont une partie est une ancre (bords,
        #    dentelures retirés par l'érosion de séparation) : rattachés à
        #    l'ancre la plus proche.
        blob_count, blobs = cv2.connectedComponents(
            (alpha >= config.component_seed_alpha).astype(np.uint8),
            connectivity=8,
        )
        anchored_blobs = np.unique(blobs[(anchors > 0) & (blobs > 0)])
        in_anchored_blob = np.isin(blobs, anchored_blobs) & (blobs > 0)
        # 2) Pixels réellement incertains (faible alpha) proches d'un sujet.
        # Une poussière isolée (élément sans ancre) reste écartée : sinon elle
        # agrandit le cadrage jusqu'aux bords du scan et fausse l'inclinaison.
        uncertain_nearby = (
            config.retain_all_uncertain_pixels
            & (anchors == 0)
            & (alpha < config.component_seed_alpha)
            & (distances <= config.uncertain_pixel_max_distance_px)
        )
        assigned_labels = np.where(
            (anchors > 0) | in_anchored_blob | uncertain_nearby,
            assigned_labels,
            0,
        )
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

    region_payloads = _reading_order(region_payloads)
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
