"""Mesure d'inclinaison et rotation rigide sans coupe ni redimensionnement."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .config import PipelineConfig
from .models import OrientationEstimate, ProtectedSubject, RotationResult


def _binary_mask(alpha: np.ndarray, threshold: int) -> np.ndarray:
    return (alpha >= threshold).astype(np.uint8)


def calculate_mask_perimeter(alpha: np.ndarray, threshold: int = 128) -> float:
    """Somme les périmètres externes du masque à un seuil donné."""

    mask = _binary_mask(alpha, threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def oriented_aspect_ratio(alpha: np.ndarray, threshold: int = 64) -> float | None:
    """Calcule le ratio grand/petit côté du rectangle orienté minimal."""

    ys, xs = np.nonzero(alpha >= threshold)
    if xs.size < 5:
        return None
    points = np.column_stack((xs, ys)).astype(np.float32)
    (_, _), (width, height), _ = cv2.minAreaRect(points)
    shorter = min(float(width), float(height))
    longer = max(float(width), float(height))
    if shorter <= 1e-9:
        return None
    return longer / shorter


def _correction_to_nearest_cardinal(angle_deg: float) -> tuple[float, float]:
    """Retourne la rotation OpenCV et l'axe cible (0° ou 90°)."""

    normalized = angle_deg % 180.0
    correction = ((normalized + 45.0) % 90.0) - 45.0
    target = (normalized - correction) % 180.0
    if math.isclose(target, 180.0, abs_tol=1e-7):
        target = 0.0
    return correction, target


def _rectangle_orientation(
    points_xy: np.ndarray,
    foreground_area: int,
) -> tuple[float, float]:
    """Angle du plus long côté et rectangularité du masque."""

    rect = cv2.minAreaRect(points_xy.astype(np.float32))
    box = cv2.boxPoints(rect)
    candidates: list[tuple[float, float]] = []
    for index in range(4):
        vector = box[(index + 1) % 4] - box[index]
        length = float(np.linalg.norm(vector))
        angle = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 180.0
        candidates.append((length, angle))
    longest = max(length for length, _ in candidates)
    near_longest = [
        (length, angle)
        for length, angle in candidates
        if length >= longest * 0.98
    ]
    # Pour un objet presque carré, l'axe nécessitant le plus petit redressement
    # est le choix le moins intrusif.
    _, chosen_angle = min(
        near_longest,
        key=lambda item: abs(_correction_to_nearest_cardinal(item[1])[0]),
    )
    width, height = rect[1]
    rect_area = max(float(width) * float(height), 1.0)
    rectangularity = min(1.0, foreground_area / rect_area)
    return chosen_angle, rectangularity


def detect_subject_inclination(
    alpha: np.ndarray,
    config: PipelineConfig,
) -> OrientationEstimate:
    """Aligne le sujet sur le côté le plus long de son rectangle orienté.

    Cette mesure suit les bords extérieurs du sujet. Elle évite l'axe PCA,
    lequel peut suivre une masse intérieure diagonale et produire un angle
    visuellement faux pour un objet pourtant bordé droit.
    """

    mask = alpha >= config.orientation_alpha_threshold
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        return OrientationEstimate(reason="Trop peu de pixels pour mesurer l'axe.")

    points = np.column_stack((xs, ys)).astype(np.float32)
    rect_angle, rectangularity = _rectangle_orientation(
        points,
        int(xs.size),
    )
    angle = rect_angle
    confidence = rectangularity
    method = "plus_long_bord"

    correction, target = _correction_to_nearest_cardinal(angle)
    if abs(correction) > config.max_rotation_deg + 1e-7:
        return OrientationEstimate(
            axis_angle_deg=angle,
            correction_deg=correction,
            target_axis_deg=target,
            confidence=confidence,
            method=method,
            determinable=False,
            reason=(
                f"Correction calculée ({correction:.2f}°) supérieure à la "
                f"limite ({config.max_rotation_deg:.2f}°)."
            ),
        )
    return OrientationEstimate(
        axis_angle_deg=angle,
        correction_deg=correction,
        target_axis_deg=target,
        confidence=confidence,
        method=method,
        determinable=True,
    )


def _expanded_rotation_matrix(
    width: int,
    height: int,
    angle_deg: float,
    padding_xy: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crée une matrice de rotation avec canevas englobant tous les coins."""

    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    corners = np.array(
        [[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]],
        dtype=np.float64,
    )
    transformed = corners @ matrix.T
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    padding_x, padding_y = padding_xy
    output_width = int(math.ceil(maximum[0] - minimum[0] + 1)) + 2 * padding_x
    output_height = int(math.ceil(maximum[1] - minimum[1] + 1)) + 2 * padding_y
    matrix[0, 2] += padding_x - minimum[0]
    matrix[1, 2] += padding_y - minimum[1]
    return matrix, (output_width, output_height)


def _warp(
    array: np.ndarray,
    matrix: np.ndarray,
    output_size: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    """Applique la matrice affine avec un fond extérieur à zéro."""

    return cv2.warpAffine(
        array,
        matrix,
        output_size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# En dessous de cette couverture Lanczos, la couleur n'est pas assez stable
# pour être « dé-prémultipliée » : on utilise alors la couleur bilinéaire.
_LANCZOS_MIN_COVERAGE = 0.02


def _neighbourhood_range(
    rgb: np.ndarray,
    matrix: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum et maximum des 4 pixels source entourant chaque point tourné.

    Le min/max de chaque bloc 2×2 est calculé sur la source, puis lu au plus
    proche voisin avec un décalage d'un demi-pixel, ce qui sélectionne
    exactement le bloc qui entoure la position source de chaque pixel.
    """

    block = np.ones((2, 2), dtype=np.uint8)
    shifted = matrix.copy()
    shifted[:, 2] += matrix[:, :2] @ np.array([0.5, 0.5])
    low = _warp(cv2.erode(rgb, block, anchor=(0, 0)), shifted, output_size, cv2.INTER_NEAREST)
    high = _warp(cv2.dilate(rgb, block, anchor=(0, 0)), shifted, output_size, cv2.INTER_NEAREST)
    return low.astype(np.float32), high.astype(np.float32)


def _rotate_colors_lanczos(
    rgb: np.ndarray,
    alpha_float_source: np.ndarray,
    alpha_float: np.ndarray,
    matrix: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Tourne les couleurs avec un filtre Lanczos, en alpha prémultiplié.

    Le calcul prémultiplié empêche la couleur de l'ancien fond de fuir dans le
    contour. Dans le sujet opaque, le résultat est exactement le Lanczos des
    couleurs. Tout au bord, là où la couverture Lanczos devient trop faible
    (oscillations du filtre), la couleur bilinéaire prend le relais.
    """

    premultiplied = rgb.astype(np.float32) * alpha_float_source[:, :, None]
    lanczos_premultiplied = _warp(premultiplied, matrix, output_size, cv2.INTER_LANCZOS4)
    lanczos_alpha = _warp(alpha_float_source, matrix, output_size, cv2.INTER_LANCZOS4)
    linear_premultiplied = _warp(premultiplied, matrix, output_size, cv2.INTER_LINEAR)

    lanczos_color = lanczos_premultiplied / np.maximum(lanczos_alpha, 1e-6)[:, :, None]
    # Anti-halo : le filtre Lanczos « déborde » légèrement sur les traits fins
    # (liserés clairs ou sombres parallèles aux lignes). Chaque valeur est
    # bornée par le minimum et le maximum des 4 pixels source qui l'entourent :
    # aucune couleur plus claire ou plus foncée que l'original n'apparaît.
    low, high = _neighbourhood_range(rgb, matrix, output_size)
    lanczos_color = np.clip(lanczos_color, low, high)
    linear_color = linear_premultiplied / np.maximum(alpha_float, 1e-6)[:, :, None]
    stable = (lanczos_alpha >= _LANCZOS_MIN_COVERAGE)[:, :, None]
    color = np.where(stable, lanczos_color, linear_color)
    return np.clip(np.rint(color), 0, 255).astype(np.uint8)


def _crop_rotated_subject(
    arrays: tuple[np.ndarray, ...],
    alpha_index: int,
    matrix: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Cadre exactement le sujet après sa rotation, sans ajouter de marge."""

    alpha = arrays[alpha_index]
    ys, xs = np.nonzero(alpha > 0)
    if xs.size == 0:
        raise ValueError("Le sujet a disparu après la rotation.")
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    cropped = tuple(array[y0:y1, x0:x1].copy() for array in arrays)
    adjusted = matrix.copy()
    adjusted[0, 2] -= x0
    adjusted[1, 2] -= y0
    return cropped, adjusted


def _add_post_rotation_border(
    arrays: tuple[np.ndarray, ...],
    padding_xy: tuple[int, int],
    matrix: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Ajoute la bordure extérieure seulement après le cadrage du sujet."""

    padding_x, padding_y = padding_xy
    framed: list[np.ndarray] = []
    for array in arrays:
        pad_width = [(padding_y, padding_y), (padding_x, padding_x)]
        pad_width.extend((0, 0) for _ in range(array.ndim - 2))
        framed.append(np.pad(array, pad_width, mode="constant", constant_values=0))
    adjusted = matrix.copy()
    adjusted[0, 2] += padding_x
    adjusted[1, 2] += padding_y
    return tuple(framed), adjusted


def straighten_subject_without_clipping(
    subject: ProtectedSubject,
    orientation: OrientationEstimate,
    config: PipelineConfig,
    padding_xy: tuple[int, int] | None = None,
) -> RotationResult:
    """Applique uniquement une rotation rigide, sur un canevas agrandi.

    Les couleurs sont rééchantillonnées avec un filtre Lanczos : contrairement
    au plus proche voisin, il ne duplique ni ne saute de lignes de pixels, ce
    qui supprime les lignes de décalage (effet d'escalier) à l'intérieur du
    sujet. L'alpha est interpolé linéairement pour un contour lisse. Sans
    rotation, les pixels sont copiés sans aucun rééchantillonnage.
    """

    requested_angle = 0.0
    if (
        config.straighten
        and orientation.determinable
        and abs(orientation.correction_deg) >= config.min_rotation_deg
    ):
        requested_angle = orientation.correction_deg

    if padding_xy is None:
        padding_xy = (config.final_padding_px, config.final_padding_px)

    height, width = subject.alpha.shape
    matrix, output_size = _expanded_rotation_matrix(
        width,
        height,
        requested_angle,
        (0, 0),
    )

    if requested_angle == 0.0:
        # Aucune rotation : aucun rééchantillonnage, les pixels sont copiés
        # tels quels (matrice identité, canevas identique).
        rgb = subject.protected_rgb.copy()
        expected = subject.original_rgb.copy()
        alpha = subject.alpha.copy()
        core = subject.protected_core.copy()
    else:
        alpha_float_source = subject.alpha.astype(np.float32) / 255.0
        # Couverture (alpha) en bilinéaire : contour lisse, sans oscillation
        # ni halo fantôme autour du sujet.
        alpha_float = np.clip(
            _warp(alpha_float_source, matrix, output_size, cv2.INTER_LINEAR),
            0.0,
            1.0,
        )
        alpha = np.clip(np.rint(alpha_float * 255.0), 0, 255).astype(np.uint8)
        # Couleurs en Lanczos sur l'image entière : aucune ligne de décalage
        # (pas de pixels dupliqués ou sautés) et netteté conservée.
        rgb = _rotate_colors_lanczos(
            subject.protected_rgb,
            alpha_float_source,
            alpha_float,
            matrix,
            output_size,
        )
        expected = _rotate_colors_lanczos(
            subject.original_rgb,
            alpha_float_source,
            alpha_float,
            matrix,
            output_size,
        )
        core = _warp(
            subject.protected_core,
            matrix,
            output_size,
            cv2.INTER_NEAREST,
        )

    # Ordre obligatoire : rotation, cadrage exact, puis bordure physique.
    (rgb, expected, alpha, core), matrix = _crop_rotated_subject(
        (rgb, expected, alpha, core),
        alpha_index=2,
        matrix=matrix,
    )
    (rgb, expected, alpha, core), matrix = _add_post_rotation_border(
        (rgb, expected, alpha, core),
        padding_xy=padding_xy,
        matrix=matrix,
    )
    return RotationResult(
        rgb=rgb,
        expected_original_rgb=expected,
        alpha=alpha,
        protected_core=core,
        affine_matrix=matrix,
        orientation_before=orientation,
        alpha_mass_before=float(subject.alpha.astype(np.float64).sum() / 255.0),
        perimeter_before=calculate_mask_perimeter(subject.alpha),
        aspect_ratio_before=oriented_aspect_ratio(
            subject.alpha,
            config.orientation_alpha_threshold,
        ),
        old_background_rgb=subject.old_background_rgb,
    )
