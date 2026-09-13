"""Orchestration de toutes les étapes, fichier par fichier et sujet par sujet."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from .color import replace_background
from .config import PipelineConfig
from .geometry import (
    detect_subject_inclination,
    straighten_subject_without_clipping,
)
from .io_utils import (
    NameAllocator,
    append_journal,
    commit_temporary_file,
    discover_source_files,
    extract_images_from_source,
    save_to_temporary_file,
    setup_logging,
)
from .manual_review import record_error, record_output, record_source_start
from .models import FrameData, ProcessSummary, SegmentationResult, SubjectRegion
from .quality import run_quality_control
from .segmentation import (
    detect_subject_regions,
    generate_precise_cutout,
    protect_subject_interior,
)


def _journal_error(
    config: PipelineConfig,
    source: Path,
    error: Exception,
    *,
    frame_index: int | None = None,
    subject_index: int | None = None,
) -> None:
    try:
        append_journal(
            config,
            {
                "event": "processing_error",
                "source": str(source),
                "frame_index": frame_index,
                "subject_index": subject_index,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
    except OSError:
        # Une panne du journal ne doit pas masquer l'erreur de traitement.
        pass


def process_one_subject(
    frame: FrameData,
    region: SubjectRegion,
    segmentation: SegmentationResult,
    allocator: NameAllocator,
    config: PipelineConfig,
    logger: logging.Logger,
    numbered: bool,
) -> tuple[Path, bool]:

    """Traite, contrôle et sauvegarde un sujet ; renvoie (chemin, QC_OK)."""

    protected = protect_subject_interior(segmentation, region, config)
    orientation = detect_subject_inclination(protected.alpha, config)
    if orientation.determinable:
        logger.info(
            "  Sujet %d : inclinaison %.3f° (%s, confiance %.2f)",
            region.detection_order,
            orientation.correction_deg,
            orientation.method,
            orientation.confidence,
        )
    else:
        logger.warning(
            "  Sujet %d : orientation ambiguë — %s",
            region.detection_order,
            orientation.reason,
        )

    margin_px_xy = config.output_margin_pixels(frame.dpi)
    logger.info(
        "  Sujet %d : marge foncée %.2f mm (%d px × %d px à %.1f × %.1f dpi)",
        region.detection_order,
        config.output_margin_mm,
        margin_px_xy[0],
        margin_px_xy[1],
        *config.effective_dpi(frame.dpi),
    )
    rotated = straighten_subject_without_clipping(
        protected,
        orientation,
        config,
        padding_xy=margin_px_xy,
    )
    composite = replace_background(rotated, config)

    target, sequence_number = allocator.reserve(
        frame.source_path.stem,
        numbered=numbered,
    )
    temporary: Path | None = None
    try:
        temporary = save_to_temporary_file(composite, target, frame, config)
        report = run_quality_control(
            saved_path=temporary,
            source_name=frame.source_path.name,
            frame_index=frame.frame_index,
            subject_index=region.detection_order,
            rotated=rotated,
            composite=composite,
            config=config,
            required_margin_px_xy=margin_px_xy,
        )
        report.output = target.name
        commit_temporary_file(temporary, target)
        temporary = None

        try:
            record_output(
                config,
                frame.source_path,
                target,
                frame_index=frame.frame_index,
                subject_index=region.detection_order,
                qc_passed=report.passed,
                qc_problems=list(report.problems),
            )
        except (OSError, ValueError) as exc:
            logger.error("Impossible d'écrire le manifeste manuel pour %s : %s", target, exc)

        try:
            append_journal(
                config,
                {
                    "event": "quality_control",
                    "sequence_number": sequence_number,
                    **report.as_dict(),
                },
            )
        except OSError as exc:
            logger.error("Impossible d'écrire le journal QC pour %s : %s", target, exc)
        if report.passed:
            logger.info("  ✓ %s — tous les contrôles sont conformes", target.name)
        else:
            for problem in report.problems:
                logger.warning("  ⚠ %s — %s", target.name, problem)
        return target, report.passed
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        NameAllocator.release(target)
        raise


def process_one_frame(
    frame: FrameData,
    allocator: NameAllocator,
    config: PipelineConfig,
    logger: logging.Logger,
    numbered: bool,
) -> tuple[list[tuple[Path, bool]], int]:
    """Segmente une trame puis traite ses sujets dans l'ordre de détection."""

    logger.info("  Trame %d : détection déterministe du fond…", frame.frame_index)
    segmentation = generate_precise_cutout(frame, logger, config)
    regions = detect_subject_regions(segmentation.alpha, config)
    numbered = numbered or len(regions) > 1
    logger.info("  Trame %d : %d sujet(s) à produire", frame.frame_index, len(regions))

    results: list[tuple[Path, bool]] = []
    errors = 0
    for region in regions:
        try:
            results.append(
                process_one_subject(
                    frame,
                    region,
                    segmentation,
                    allocator,
                    config,
                    logger,
                    numbered,
                )
            )
        except Exception as exc:
            errors += 1
            logger.exception(
                "Échec de %s, trame %d, sujet %d : %s",
                frame.source_path.name,
                frame.frame_index,
                region.detection_order,
                exc,
            )
            _journal_error(
                config,
                frame.source_path,
                exc,
                frame_index=frame.frame_index,
                subject_index=region.detection_order,
            )
            try:
                record_error(
                    config,
                    frame.source_path,
                    exc,
                    frame_index=frame.frame_index,
                    subject_index=region.detection_order,
                )
            except (OSError, ValueError) as manifest_exc:
                logger.error(
                    "Impossible d'écrire le manifeste manuel pour %s : %s",
                    frame.source_path,
                    manifest_exc,
                )
    return results, errors


def process_source_file(
    source: Path,
    allocator: NameAllocator,
    config: PipelineConfig,
    logger: logging.Logger,
    summary: ProcessSummary,
) -> None:
    """Extrait et traite toutes les trames d'un fichier source."""

    try:
        record_source_start(config, source)
    except (OSError, ValueError) as exc:
        logger.error("Impossible d'initialiser le manifeste manuel pour %s : %s", source, exc)

    frames = iter(extract_images_from_source(source, config))

    try:
        first_frame = next(frames)
    except StopIteration:
        raise RuntimeError("Le fichier ne contient aucune trame exploitable.")

    try:
        second_frame = next(frames)
    except StopIteration:
        second_frame = None

    # Dès qu'une deuxième trame existe, toutes les sorties du fichier
    # doivent être numérotées : base-1.ext, base-2.ext, etc.
    numbered = second_frame is not None

    def process_frame(frame: FrameData) -> None:
        try:
            results, subject_errors = process_one_frame(
                frame,
                allocator,
                config,
                logger,
                numbered,
            )
            summary.frames_processed += 1
            summary.processing_errors += subject_errors
            for _, qc_ok in results:
                summary.outputs_written += 1
                if not qc_ok:
                    summary.qc_failures += 1
        except Exception as exc:
            summary.processing_errors += 1
            logger.exception(
                "Échec de %s, trame %d : %s",
                source.name,
                frame.frame_index,
                exc,
            )
            _journal_error(
                config,
                source,
                exc,
                frame_index=frame.frame_index,
            )
            try:
                record_error(
                    config,
                    source,
                    exc,
                    frame_index=frame.frame_index,
                )
            except (OSError, ValueError) as manifest_exc:
                logger.error(
                    "Impossible d'écrire le manifeste manuel pour %s : %s",
                    source,
                    manifest_exc,
                )

    process_frame(first_frame)

    if second_frame is not None:
        process_frame(second_frame)

    for frame in frames:
        process_frame(frame)


def _output_dir_for_source(source: Path, config: PipelineConfig) -> Path:
    """Détermine le dossier de sortie correspondant au dossier source."""

    input_dir = config.input_dir.resolve()
    source = source.resolve()
    relative = source.relative_to(input_dir)

    # Fichier placé directement dans input : comportement historique.
    if len(relative.parts) <= 1:
        return config.output_dir

    # Le premier sous-dossier de input représente le lot.
    lot_name = relative.parts[0]
    return config.output_dir / f"{lot_name}-ps"


def process_batch(config: PipelineConfig) -> ProcessSummary:
    """Fonction principale : découvre, traite, contrôle et journalise le lot."""

    logger = setup_logging(config)
    sources = discover_source_files(config)
    summary = ProcessSummary(sources_found=len(sources))
    if not sources:
        logger.warning("Aucun fichier image compatible dans %s", config.input_dir)
        return summary

    append_journal(
        config,
        {
            "event": "batch_start",
            "source_count": len(sources),
            "configuration": config.serializable(),
        },
    )
    allocators: dict[Path, NameAllocator] = {}
    logger.info(
        "Démarrage sans IA : %d fichier(s), sortie %s, couleurs du sujet %s, fond %s",
        len(sources),
        config.output_format.upper(),
        "RVB sRGB strictement préservées" if config.preserve_subject_rgb else "converties en CMJN",
        config.rgb_background if config.preserve_subject_rgb else config.cmyk_background_percent,
    )

    for source_index, source in enumerate(sources, start=1):
        logger.info("[%d/%d] %s", source_index, len(sources), source)
        source_config = config
        try:
            source_output_dir = _output_dir_for_source(source, config)
            source_config = replace(config, output_dir=source_output_dir)

            allocator = allocators.get(source_output_dir)
            if allocator is None:
                allocator = NameAllocator(
                    source_output_dir,
                    source_config.output_extension,
                )
                allocators[source_output_dir] = allocator

            process_source_file(
                source,
                allocator,
                source_config,
                logger,
                summary,
            )
            summary.sources_processed += 1
        except Exception as exc:
            summary.processing_errors += 1
            logger.exception("Échec du fichier %s : %s", source, exc)
            _journal_error(source_config, source, exc)
            try:
                record_error(source_config, source, exc)
            except (OSError, ValueError) as manifest_exc:
                logger.error(
                    "Impossible d'écrire le manifeste manuel pour %s : %s",
                    source,
                    manifest_exc,
                )

    append_journal(
        config,
        {
            "event": "batch_end",
            "summary": summary.as_dict(),
        },
    )
    logger.info(
        "Terminé : %d sortie(s), %d avertissement(s) QC, %d erreur(s).",
        summary.outputs_written,
        summary.qc_failures,
        summary.processing_errors,
    )
    return summary
