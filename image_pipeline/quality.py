"""Contrôles automatiques mesurables exécutés avant la livraison du fichier."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from .config import PipelineConfig
from .geometry import (
    calculate_mask_perimeter,
    detect_subject_inclination,
    oriented_aspect_ratio,
)
from .models import CompositeResult, QCCheck, QCReport, RotationResult


def _relative_difference(value: float, reference: float) -> float:
    if abs(reference) <= 1e-12:
        return float("inf") if abs(value) > 1e-12 else 0.0
    return abs(value - reference) / abs(reference)


def _subject_margins(alpha: np.ndarray) -> tuple[int, int, int, int]:
    """Retourne les marges gauche, haute, droite et basse en pixels."""

    ys, xs = np.nonzero(alpha > 0)
    if xs.size == 0:
        return -1, -1, -1, -1
    height, width = alpha.shape
    return (
        int(xs.min()),
        int(ys.min()),
        int(width - 1 - xs.max()),
        int(height - 1 - ys.max()),
    )


def _halo_suspicion_fraction(
    rotated: RotationResult,
    config: PipelineConfig,
) -> float | None:
    """Heuristique : bord plus proche de l'ancien fond que du cœur voisin."""

    edge = (rotated.alpha > 0) & (rotated.alpha < 255)
    core = rotated.protected_core > 0
    if not np.any(edge):
        return 0.0
    if not np.any(core):
        return None

    _, nearest_indices = ndimage.distance_transform_edt(
        ~core,
        return_indices=True,
    )
    nearest_core_rgb = rotated.rgb[tuple(nearest_indices)]
    edge_rgb = rotated.rgb[edge].astype(np.float64)
    neighboring_core = nearest_core_rgb[edge].astype(np.float64)
    old_background = np.asarray(rotated.old_background_rgb, dtype=np.float64)
    distance_to_background = np.linalg.norm(edge_rgb - old_background, axis=1)
    distance_to_core = np.linalg.norm(edge_rgb - neighboring_core, axis=1)
    suspect = (
        (distance_to_background < config.qa_halo_color_distance)
        & (
            distance_to_background + config.qa_halo_advantage
            < distance_to_core
        )
    )
    return float(np.mean(suspect))


def run_quality_control(
    saved_path: Path,
    source_name: str,
    frame_index: int,
    subject_index: int,
    rotated: RotationResult,
    composite: CompositeResult,
    config: PipelineConfig,
    required_margin_px_xy: tuple[int, int] | None = None,
) -> QCReport:
    """Vérifie couleur, uniformité, coupe, contenu, angle, ratio et halo."""

    report = QCReport(
        source=source_name,
        frame_index=frame_index,
        subject_index=subject_index,
        output=saved_path.name,
    )

    with Image.open(saved_path) as saved_image:
        encoded_mode = saved_image.mode
        decoded = np.asarray(
            saved_image.convert(composite.target_mode),
            dtype=np.uint8,
        ).copy()

    same_shape = decoded.shape[:2] == rotated.alpha.shape
    report.checks.append(
        QCCheck(
            name="dimensions_sortie",
            passed=same_shape,
            message=(
                "Dimensions du fichier conformes."
                if same_shape
                else "Les dimensions du fichier sauvegardé diffèrent du masque."
            ),
            measured=list(decoded.shape[:2]),
            expected=list(rotated.alpha.shape),
        )
    )
    if not same_shape:
        return report

    mode_ok = encoded_mode == composite.target_mode
    report.checks.append(
        QCCheck(
            name="mode_colorimetrique",
            passed=mode_ok,
            message=(
                f"Mode {encoded_mode} conforme."
                if mode_ok
                else f"Mode encodé {encoded_mode}, mode attendu {composite.target_mode}."
            ),
            measured=encoded_mode,
            expected=composite.target_mode,
        )
    )

    color_encoding_ok = (
        composite.exact_subject_color_preservation or composite.cmyk_certifiable
    )
    report.checks.append(
        QCCheck(
            name="encodage_colorimetrique",
            passed=color_encoding_ok,
            message=(
                "TIFF RVB sRGB sans perte : couleurs du sujet préservées."
                if composite.exact_subject_color_preservation
                else (
                    "Le TIFF CMJN sans perte permet le contrôle exact des canaux."
                    if composite.cmyk_certifiable
                    else "Le format choisi ne permet pas de certifier les couleurs."
                )
            ),
            measured={"format": config.output_format.upper(), "mode": composite.target_mode},
            expected="TIFF sans perte avec gestion colorimétrique explicite",
        )
    )

    # Compare le fichier réellement relu aux pixels RVB attendus. Ce contrôle
    # détecte une conversion, une correction automatique ou une dérive de
    # couleur introduite au moment de la sauvegarde.
    # La comparaison exacte porte sur le cœur pleinement opaque. Les pixels
    # semi-transparents de l'unique bande anticrénelée du contour sont, par
    # définition, mélangés au nouveau fond et ne doivent pas être confondus
    # avec une dérive colorimétrique du sujet.
    core_for_color = (rotated.protected_core > 0) & (rotated.alpha == 255)
    color_core_count = int(np.count_nonzero(core_for_color))
    if composite.target_mode == "RGB" and color_core_count:
        changed_saved = int(
            np.count_nonzero(
                np.any(decoded[core_for_color] != rotated.rgb[core_for_color], axis=1)
            )
        )
        max_saved_delta = int(
            np.max(
                np.abs(
                    decoded[core_for_color].astype(np.int16)
                    - rotated.rgb[core_for_color].astype(np.int16)
                )
            )
        )
        saved_color_ok = changed_saved == 0
    elif composite.target_mode == "CMYK":
        changed_saved = 0
        max_saved_delta = 0
        saved_color_ok = composite.cmyk_certifiable
    else:
        changed_saved = color_core_count
        max_saved_delta = 255
        saved_color_ok = False
    report.checks.append(
        QCCheck(
            name="fidelite_couleur_sujet_sauvegarde",
            passed=saved_color_ok,
            message=(
                f"Les {color_core_count} pixels intérieurs relus sont strictement identiques."
                if saved_color_ok and composite.target_mode == "RGB"
                else (
                    "Sortie CMJN certifiée par valeurs de canaux."
                    if saved_color_ok
                    else f"Dérive détectée sur {changed_saved} pixel(s) intérieurs."
                )
            ),
            measured={
                "pixels_controles": color_core_count,
                "pixels_modifies": changed_saved,
                "ecart_max_canal": max_saved_delta,
            },
            expected={"pixels_modifies": 0, "ecart_max_canal": 0},
        )
    )

    background_mask = rotated.alpha == 0
    background_count = int(np.count_nonzero(background_mask))
    if background_count:
        background_pixels = decoded[background_mask]
        expected_background = np.asarray(
            composite.background_values,
            dtype=np.uint8,
        )
        exact_fraction = float(
            np.mean(np.all(background_pixels == expected_background, axis=1))
        )
        channel_min = background_pixels.min(axis=0)
        channel_max = background_pixels.max(axis=0)
        uniform = bool(np.array_equal(channel_min, channel_max))
        exact = exact_fraction == 1.0
    else:
        exact_fraction = 0.0
        uniform = False
        exact = False
        channel_min = np.array([], dtype=np.uint8)
        channel_max = np.array([], dtype=np.uint8)

    report.checks.extend(
        [
            QCCheck(
                name="couleur_fond",
                passed=exact,
                message=(
                    "Tous les pixels de fond ont la valeur demandée."
                    if exact
                    else (
                        f"Seulement {exact_fraction:.2%} des pixels de fond ont "
                        "la valeur attendue."
                    )
                ),
                measured={
                    "fraction_exacte": exact_fraction,
                    "minimum": channel_min.tolist(),
                    "maximum": channel_max.tolist(),
                },
                expected=list(composite.background_values),
            ),
            QCCheck(
                name="uniformite_fond",
                passed=uniform,
                message=(
                    "Le fond extérieur au sujet est uniforme."
                    if uniform
                    else "Le fond extérieur au sujet n'est pas uniforme."
                ),
                measured={"pixels_controles": background_count},
                expected="une seule valeur par canal",
            ),
        ]
    )

    margins = _subject_margins(rotated.alpha)
    exact_margin_expected = required_margin_px_xy is not None
    if required_margin_px_xy is None:
        required_margin_px_xy = (
            config.qa_min_border_margin_px,
            config.qa_min_border_margin_px,
        )
    required_x, required_y = required_margin_px_xy
    left, top, right, bottom = margins
    not_cut = (
        left >= required_x
        and right >= required_x
        and top >= required_y
        and bottom >= required_y
    )
    report.checks.append(
        QCCheck(
            name="marge_min_et_sujet_non_coupe",
            passed=not_cut,
            message=(
                f"Marge d'au moins {config.output_margin_mm:g} mm présente sur les quatre côtés."
                if not_cut
                else (
                    "Marge insuffisante ou sujet coupé : "
                    f"gauche={left}px, haut={top}px, droite={right}px, bas={bottom}px."
                )
            ),
            measured={
                "gauche_px": left,
                "haut_px": top,
                "droite_px": right,
                "bas_px": bottom,
            },
            expected={
                "gauche_droite_min_px": required_x,
                "haut_bas_min_px": required_y,
                "marge_mm": config.output_margin_mm,
            },
        )
    )

    if exact_margin_expected:
        # La bordure est ajoutée une seule fois, après un cadrage au ras du
        # sujet : chaque côté doit donc mesurer exactement la marge calculée.
        # Une marge plus large signale une bordure ajoutée deux fois ou un
        # cadrage qui laisse de l'espace vide autour du sujet.
        exact = (
            left == required_x
            and right == required_x
            and top == required_y
            and bottom == required_y
        )
        report.checks.append(
            QCCheck(
                name="marge_exacte",
                passed=exact,
                message=(
                    f"Marge exacte de {config.output_margin_mm:g} mm "
                    f"({required_x} px × {required_y} px) sur les quatre côtés."
                    if exact
                    else (
                        "Marge différente de la valeur attendue : "
                        f"gauche={left}px, haut={top}px, droite={right}px, "
                        f"bas={bottom}px (attendu {required_x}px à gauche/droite, "
                        f"{required_y}px en haut/bas)."
                    )
                ),
                measured={
                    "gauche_px": left,
                    "haut_px": top,
                    "droite_px": right,
                    "bas_px": bottom,
                },
                expected={
                    "gauche_droite_px": required_x,
                    "haut_bas_px": required_y,
                    "marge_mm": config.output_margin_mm,
                },
            )
        )

    alpha_mass_after = float(rotated.alpha.astype(np.float64).sum() / 255.0)
    alpha_area_delta = _relative_difference(
        alpha_mass_after,
        rotated.alpha_mass_before,
    )
    perimeter_after = calculate_mask_perimeter(rotated.alpha)
    perimeter_delta = _relative_difference(
        perimeter_after,
        rotated.perimeter_before,
    )
    contour_ok = (
        alpha_area_delta <= config.qa_alpha_area_tolerance
        and perimeter_delta <= config.qa_perimeter_tolerance
    )
    report.checks.append(
        QCCheck(
            name="conservation_contours",
            passed=contour_ok,
            message=(
                "Masse alpha et périmètre compatibles avec une rotation rigide."
                if contour_ok
                else (
                    "Variation anormale du masque : "
                    f"aire {alpha_area_delta:.2%}, périmètre {perimeter_delta:.2%}."
                )
            ),
            measured={
                "variation_masse_alpha": alpha_area_delta,
                "variation_perimetre": perimeter_delta,
            },
            expected={
                "masse_alpha_max": config.qa_alpha_area_tolerance,
                "perimetre_max": config.qa_perimeter_tolerance,
            },
        )
    )

    # Le filtre Lanczos lit 4 pixels autour de chaque point : près du bord du
    # cœur, il mélange légitimement la bande de contour détourée. Le contrôle
    # porte donc sur le cœur situé à plus de 4 pixels de ce bord.
    # Sans rotation, aucun rééchantillonnage : tout le cœur est contrôlé.
    core = rotated.protected_core > 0
    rotation_applied = not np.allclose(
        rotated.affine_matrix[:, :2],
        np.eye(2),
        atol=1e-12,
    )
    if rotation_applied:
        core = cv2.erode(core.astype(np.uint8), np.ones((9, 9), dtype=np.uint8)) > 0
    core_count = int(np.count_nonzero(core))
    if core_count:
        changed_core = int(
            np.count_nonzero(
                np.any(
                    rotated.rgb[core] != rotated.expected_original_rgb[core],
                    axis=1,
                )
            )
        )
        interior_ok = changed_core == 0
    else:
        changed_core = 0
        interior_ok = False
    report.checks.append(
        QCCheck(
            name="interieur_inchange",
            passed=interior_ok,
            message=(
                (
                    f"Les {core_count} pixels intérieurs n'ont subi que la rotation."
                    if rotation_applied
                    else f"Les {core_count} pixels protégés gardent leurs valeurs RVB."
                )
                if interior_ok
                else (
                    "Impossible de garantir l'intérieur : "
                    f"{changed_core} pixel(s) modifié(s), cœur={core_count}."
                )
            ),
            measured={"pixels_coeur": core_count, "pixels_modifies": changed_core},
            expected={"pixels_modifies": 0},
        )
    )

    if not config.straighten:
        straighten_ok = True
        residual = None
        straighten_message = "Redressement désactivé explicitement."
    elif not rotated.orientation_before.determinable:
        straighten_ok = False
        residual = None
        straighten_message = (
            "Inclinaison non vérifiable : " + rotated.orientation_before.reason
        )
    else:
        after_orientation = detect_subject_inclination(rotated.alpha, config)
        if after_orientation.determinable:
            residual = abs(after_orientation.correction_deg)
            straighten_ok = residual <= config.qa_residual_skew_deg
            straighten_message = (
                f"Inclinaison résiduelle {residual:.3f}°."
                if straighten_ok
                else f"Inclinaison résiduelle excessive : {residual:.3f}°."
            )
        else:
            residual = None
            straighten_ok = False
            straighten_message = (
                "L'orientation n'est plus mesurable après redressement : "
                + after_orientation.reason
            )
    report.checks.append(
        QCCheck(
            name="redressement",
            passed=straighten_ok,
            message=straighten_message,
            measured=residual,
            expected=f"<= {config.qa_residual_skew_deg}°",
        )
    )

    linear = rotated.affine_matrix[:, :2]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    rigid_error = float(np.max(np.abs(singular_values - 1.0)))
    determinant = float(np.linalg.det(linear))
    rigid = rigid_error <= 1e-7 and abs(abs(determinant) - 1.0) <= 1e-7
    aspect_after = oriented_aspect_ratio(
        rotated.alpha,
        config.orientation_alpha_threshold,
    )
    if rotated.aspect_ratio_before is None or aspect_after is None:
        aspect_delta = None
        aspect_ok = rigid
    else:
        aspect_delta = _relative_difference(
            aspect_after,
            rotated.aspect_ratio_before,
        )
        aspect_ok = rigid and aspect_delta <= config.qa_aspect_ratio_tolerance
    report.checks.append(
        QCCheck(
            name="proportions",
            passed=aspect_ok,
            message=(
                "Transformation rigide : aucune échelle non proportionnelle."
                if aspect_ok
                else "Une variation géométrique supérieure à la tolérance est détectée."
            ),
            measured={
                "valeurs_singulieres": singular_values.tolist(),
                "determinant": determinant,
                "variation_ratio_oriente": aspect_delta,
            },
            expected={
                "echelles": [1.0, 1.0],
                "variation_ratio_max": config.qa_aspect_ratio_tolerance,
            },
        )
    )

    halo_fraction = _halo_suspicion_fraction(rotated, config)
    halo_ok = (
        halo_fraction is not None
        and halo_fraction <= config.qa_max_suspect_halo_fraction
    )
    report.checks.append(
        QCCheck(
            name="halo_ancien_fond",
            passed=halo_ok,
            message=(
                f"Fraction de contour suspecte : {halo_fraction:.2%}."
                if halo_fraction is not None and halo_ok
                else (
                    "Présence possible d'un halo de l'ancien fond : "
                    f"{halo_fraction:.2%}."
                    if halo_fraction is not None
                    else "Halo non mesurable faute de cœur opaque."
                )
            ),
            measured=halo_fraction,
            expected=f"<= {config.qa_max_suspect_halo_fraction:.2%}",
        )
    )

    report.metrics.update(
        {
            "background_pixels_checked": background_count,
            "background_exact_fraction": exact_fraction,
            "subject_margins_px": {
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            },
            "required_margin_px": {"x": required_x, "y": required_y},
            "alpha_mass_before": rotated.alpha_mass_before,
            "alpha_mass_after": alpha_mass_after,
            "alpha_area_relative_delta": alpha_area_delta,
            "perimeter_before": rotated.perimeter_before,
            "perimeter_after": perimeter_after,
            "perimeter_relative_delta": perimeter_delta,
            "protected_core_pixels": core_count,
            "changed_core_pixels": changed_core,
            "saved_color_changed_core_pixels": changed_saved,
            "saved_color_max_channel_delta": max_saved_delta,
            "residual_skew_deg": residual,
            "halo_suspicion_fraction": halo_fraction,
        }
    )
    return report
