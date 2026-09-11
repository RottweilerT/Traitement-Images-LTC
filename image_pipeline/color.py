"""Conversion colorimétrique et création d'un fond parfaitement uniforme."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from .config import PipelineConfig
from .models import CompositeResult, RotationResult


class ColorManagementError(RuntimeError):
    """Erreur de profil ou de conversion colorimétrique."""


def cmyk_percent_to_u8(
    percentages: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    """Convertit quatre pourcentages CMJN en valeurs exactes sur 8 bits."""

    return tuple(int(round(value * 255.0 / 100.0)) for value in percentages)  # type: ignore[return-value]


def cmyk_percent_to_preview_rgb(
    percentages: tuple[float, float, float, float],
) -> tuple[int, int, int]:
    """Approximation RVB sans profil, réservée à la sortie PNG d'aperçu."""

    cyan, magenta, yellow, black = (value / 100.0 for value in percentages)
    return (
        int(round(255.0 * (1.0 - cyan) * (1.0 - black))),
        int(round(255.0 * (1.0 - magenta) * (1.0 - black))),
        int(round(255.0 * (1.0 - yellow) * (1.0 - black))),
    )


def _read_profile(profile_path: Path) -> tuple[ImageCms.ImageCmsProfile, bytes]:
    if not profile_path.exists() or not profile_path.is_file():
        raise ColorManagementError(f"Profil ICC introuvable : {profile_path}")
    try:
        profile_bytes = profile_path.read_bytes()
        profile = ImageCms.getOpenProfile(str(profile_path))
    except (OSError, ImageCms.PyCMSError) as exc:
        raise ColorManagementError(
            f"Impossible d'ouvrir le profil ICC {profile_path}: {exc}"
        ) from exc
    return profile, profile_bytes


def convert_subject_rgb_to_cmyk(
    rgb: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, bytes | None]:
    """Convertit le sujet RVB en CMJN, avec profil d'impression si fourni."""

    image = Image.fromarray(rgb, mode="RGB")
    profile_path = config.output_cmyk_icc_profile
    if profile_path is None:
        return np.asarray(image.convert("CMYK"), dtype=np.uint8).copy(), None

    destination_profile, profile_bytes = _read_profile(profile_path)
    try:
        source_profile = ImageCms.createProfile("sRGB")
        transform = ImageCms.buildTransformFromOpenProfiles(
            source_profile,
            destination_profile,
            "RGB",
            "CMYK",
            renderingIntent=1,  # colorimétrie relative
        )
        converted = ImageCms.applyTransform(image, transform)
    except (OSError, ValueError, ImageCms.PyCMSError) as exc:
        raise ColorManagementError(
            "Le profil ICC fourni ne permet pas une conversion RVB → CMJN : "
            f"{profile_path} ({exc})"
        ) from exc
    return np.asarray(converted, dtype=np.uint8).copy(), profile_bytes


def _alpha_composite_integer(
    foreground: np.ndarray,
    alpha: np.ndarray,
    background: tuple[int, ...],
) -> np.ndarray:
    """Composition entière déterministe, sans résidu ni transparence."""

    foreground_u32 = foreground.astype(np.uint32)
    alpha_u32 = alpha.astype(np.uint32)[:, :, None]
    background_u32 = np.asarray(background, dtype=np.uint32).reshape(1, 1, -1)
    output = (
        foreground_u32 * alpha_u32
        + background_u32 * (255 - alpha_u32)
        + 127
    ) // 255
    result = np.clip(output, 0, 255).astype(np.uint8)
    # Rend explicite la garantie sur chaque pixel totalement extérieur.
    result[alpha == 0] = np.asarray(background, dtype=np.uint8)
    return result


def replace_background(
    rotated: RotationResult,
    config: PipelineConfig,
) -> CompositeResult:
    """Compose le sujet sur le fond demandé, en dernier dans le pipeline.

    En mode de préservation (par défaut), le sujet reste en RVB sRGB et le fond
    devient #00000c. Le mode CMJN intégral reste disponible uniquement si cette
    protection est désactivée explicitement avec un profil de destination.
    """

    output_format = config.output_format.upper()
    if config.preserve_subject_rgb:
        # Garantie principale : aucune conversion RVB -> CMJN n'est appliquée
        # au timbre. Les pixels opaques du sujet restent donc numériquement
        # identiques après la composition et l'enregistrement TIFF sans perte.
        background_rgb = tuple(int(value) for value in config.rgb_background)
        composited_rgb = _alpha_composite_integer(
            rotated.rgb,
            rotated.alpha,
            background_rgb,
        )
        return CompositeResult(
            image=Image.fromarray(composited_rgb, mode="RGB"),
            background_values=background_rgb,
            target_mode="RGB",
            cmyk_certifiable=False,
            exact_subject_color_preservation=output_format == "TIFF",
            output_icc_profile=ImageCms.ImageCmsProfile(
                ImageCms.createProfile("sRGB")
            ).tobytes(),
        )

    if output_format in {"TIFF", "JPEG"}:
        background = cmyk_percent_to_u8(config.cmyk_background_percent)
        foreground, profile_bytes = convert_subject_rgb_to_cmyk(rotated.rgb, config)
        composited = _alpha_composite_integer(foreground, rotated.alpha, background)
        return CompositeResult(
            image=Image.fromarray(composited, mode="CMYK"),
            background_values=background,
            target_mode="CMYK",
            cmyk_certifiable=output_format == "TIFF",
            exact_subject_color_preservation=False,
            output_icc_profile=profile_bytes,
        )

    background_rgb = cmyk_percent_to_preview_rgb(config.cmyk_background_percent)
    composited_rgb = _alpha_composite_integer(
        rotated.rgb,
        rotated.alpha,
        background_rgb,
    )
    return CompositeResult(
        image=Image.fromarray(composited_rgb, mode="RGB"),
        background_values=background_rgb,
        target_mode="RGB",
        cmyk_certifiable=False,
        exact_subject_color_preservation=False,
    )
