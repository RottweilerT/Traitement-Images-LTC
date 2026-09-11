"""Découverte, extraction, nommage, sauvegarde atomique et journaux."""

from __future__ import annotations

import io
import json
import logging
import logging.handlers
import os
import re
import tempfile
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from .config import PipelineConfig
from .models import CompositeResult, FrameData


LOGGER_NAME = "traitement_images_cmjn"


class InputImageError(RuntimeError):
    """Erreur lisible liée à une image source."""


def setup_logging(config: PipelineConfig) -> logging.Logger:
    """Configure la console et un journal rotatif dans le dossier de sortie."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.handlers.RotatingFileHandler(
        config.output_dir / config.log_filename,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def discover_source_files(config: PipelineConfig) -> list[Path]:
    """Retourne les fichiers image acceptés dans un ordre déterministe."""

    input_dir = config.input_dir.resolve()
    output_dir = config.output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Dossier d'entrée introuvable : {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"L'entrée n'est pas un dossier : {input_dir}")

    iterator = input_dir.rglob("*") if config.recursive else input_dir.glob("*")
    extensions = {extension.lower() for extension in config.supported_extensions}
    files: list[Path] = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        resolved = path.resolve()
        # Empêche de retraiter les sorties si le dossier de sortie est inclus
        # dans le dossier d'entrée.
        if resolved == output_dir or output_dir in resolved.parents:
            continue
        files.append(path)
    return sorted(files, key=lambda item: str(item).casefold())


def _convert_frame_to_srgb(
    frame: Image.Image,
    icc_profile: bytes | None,
) -> Image.Image:
    """Produit un rendu RVB colorimétrique, avec repli sûr sans profil ICC."""

    if icc_profile and frame.mode in {"RGB", "RGBA", "CMYK", "L", "LA"}:
        try:
            alpha = frame.getchannel("A") if "A" in frame.getbands() else None
            base_mode = "CMYK" if frame.mode == "CMYK" else "RGB"
            base = frame.convert(base_mode)
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
            srgb_profile = ImageCms.createProfile("sRGB")
            converted = ImageCms.profileToProfile(
                base,
                source_profile,
                srgb_profile,
                outputMode="RGB",
            )
            if alpha is not None:
                converted.putalpha(alpha)
            return converted
        except (OSError, ValueError, ImageCms.PyCMSError):
            # Le profil peut être tronqué ou incompatible. La conversion Pillow
            # standard demeure préférable à l'arrêt de tout le lot.
            pass
    return frame.convert("RGBA" if "A" in frame.getbands() else "RGB")


def extract_images_from_source(
    source_path: Path,
    config: PipelineConfig,
) -> Iterator[FrameData]:
    """Extrait chaque page/trame raster d'un fichier dans l'ordre natif.

    JPEG/PNG/BMP/WebP produisent généralement une trame. Les TIFF et WebP
    multipages/animés peuvent en produire plusieurs. L'orientation EXIF est
    appliquée avant toute mesure géométrique.
    """

    try:
        with Image.open(source_path) as image:
            frame_count = int(getattr(image, "n_frames", 1))
            for zero_based_index in range(frame_count):
                image.seek(zero_based_index)
                raw = ImageOps.exif_transpose(image.copy())
                pixels = raw.width * raw.height
                if pixels > config.max_megapixels * 1_000_000:
                    raise InputImageError(
                        f"{source_path.name}, trame {zero_based_index + 1} : "
                        f"{pixels / 1_000_000:.1f} Mpx dépasse la limite de "
                        f"{config.max_megapixels:.1f} Mpx."
                    )

                info = dict(image.info)
                icc_profile = info.get("icc_profile")
                dpi_value = info.get("dpi")
                dpi: tuple[float, float] | None = None
                if isinstance(dpi_value, tuple) and len(dpi_value) >= 2:
                    dpi = (float(dpi_value[0]), float(dpi_value[1]))

                if "A" in raw.getbands():
                    source_alpha = np.asarray(raw.getchannel("A"), dtype=np.uint8)
                elif raw.mode == "P" and "transparency" in raw.info:
                    source_alpha = np.asarray(raw.convert("RGBA").getchannel("A"))
                else:
                    source_alpha = np.full((raw.height, raw.width), 255, np.uint8)

                rendered = _convert_frame_to_srgb(raw, icc_profile)
                rgb = np.asarray(rendered.convert("RGB"), dtype=np.uint8).copy()
                if source_alpha.shape != rgb.shape[:2]:
                    source_alpha = np.asarray(
                        raw.convert("RGBA").getchannel("A"), dtype=np.uint8
                    )

                yield FrameData(
                    source_path=source_path,
                    frame_index=zero_based_index + 1,
                    rgb=rgb,
                    source_alpha=source_alpha.copy(),
                    dpi=dpi,
                    source_icc_profile=icc_profile,
                )
    except (UnidentifiedImageError, OSError) as exc:
        raise InputImageError(f"Impossible de lire {source_path}: {exc}") from exc


class NameAllocator:
    """Alloue sans écrasement des noms ``base_N.ext``.

    Un fichier vide est créé en mode exclusif pour réserver chaque nom. Il est
    ensuite remplacé par le fichier temporaire validé. Cette stratégie évite
    les collisions même si deux processus écrivent dans le même dossier.
    """

    def __init__(self, output_dir: Path, extension: str) -> None:
        self.output_dir = output_dir
        self.extension = extension
        self._next_by_stem: dict[str, int] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

    def _initial_index(self, source_stem: str) -> int:
        pattern = re.compile(rf"^{re.escape(source_stem)}_(\d+)(?:\.[^.]+)?$")
        greatest = 0
        for path in self.output_dir.iterdir():
            if not path.is_file():
                continue
            match = pattern.match(path.name)
            if match:
                greatest = max(greatest, int(match.group(1)))
        return greatest + 1

    def reserve(self, source_stem: str) -> tuple[Path, int]:
        if source_stem not in self._next_by_stem:
            self._next_by_stem[source_stem] = self._initial_index(source_stem)
        index = self._next_by_stem[source_stem]
        while True:
            candidate = self.output_dir / f"{source_stem}_{index}{self.extension}"
            try:
                with candidate.open("xb"):
                    pass
                self._next_by_stem[source_stem] = index + 1
                return candidate, index
            except FileExistsError:
                index += 1

    @staticmethod
    def release(path: Path) -> None:
        """Libère uniquement une réservation vide créée par l'allocateur."""

        try:
            if path.exists() and path.stat().st_size == 0:
                path.unlink()
        except OSError:
            pass


def save_to_temporary_file(
    composite: CompositeResult,
    reserved_target: Path,
    frame: FrameData,
    config: PipelineConfig,
) -> Path:
    """Écrit l'image dans un fichier temporaire situé près de la cible."""

    handle = tempfile.NamedTemporaryFile(
        prefix=".traitement-",
        suffix=config.output_extension,
        dir=config.output_dir,
        delete=False,
    )
    temporary_path = Path(handle.name)
    handle.close()

    save_options: dict[str, object] = {}
    # La résolution est toujours inscrite dans la sortie : elle donne à la
    # marge calculée en pixels sa dimension physique garantie de 2 mm.
    save_options["dpi"] = config.effective_dpi(frame.dpi)
    if composite.output_icc_profile is not None:
        save_options["icc_profile"] = composite.output_icc_profile

    output_format = config.output_format.upper()
    if output_format == "TIFF":
        save_options["compression"] = config.tiff_compression
    elif output_format == "JPEG":
        save_options.update(
            quality=config.jpeg_quality,
            subsampling=0,
            optimize=False,
        )

    try:
        composite.image.save(temporary_path, format=output_format, **save_options)
        # Pillow ferme le fichier après save(). Sous Windows, appeler fsync()
        # sur un descripteur rouvert en lecture seule déclenche Errno 9.
        # Une vérification immédiate évite néanmoins de valider un fichier vide.
        if temporary_path.stat().st_size == 0:
            raise OSError(f"Le fichier temporaire est vide : {temporary_path}")
        return temporary_path
    except Exception:
        temporary_path.unlink(missing_ok=True)
        NameAllocator.release(reserved_target)
        raise


def commit_temporary_file(temporary_path: Path, reserved_target: Path) -> None:
    """Remplace atomiquement la réservation par le résultat contrôlé."""

    os.replace(temporary_path, reserved_target)


def append_journal(config: PipelineConfig, payload: dict[str, object]) -> None:
    """Ajoute une ligne JSON horodatée au journal qualité."""

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    journal_path = config.output_dir / config.journal_filename
    with journal_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
