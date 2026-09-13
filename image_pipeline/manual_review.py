"""Suivi des sorties et préparation des reprises manuelles.

Ce module ne transforme jamais les images. Il mémorise la relation entre les
sources et les sorties produites, puis prépare des copies binaires des originaux
lorsqu'une sortie a été supprimée après contrôle ou qu'un traitement automatique
n'a pas abouti.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .config import PipelineConfig

MANIFEST_FILENAME = "controle_manuel_manifest.json"
MANUAL_DIRNAME = "A_TRAITER_MANUELLEMENT"
REPORT_FILENAME = "CONTROLE_MANUEL.txt"
MANIFEST_VERSION = 1


def _source_identity(source: Path, config: PipelineConfig) -> tuple[str, str]:
    resolved = source.resolve()
    try:
        relative = resolved.relative_to(config.input_dir.resolve()).as_posix()
    except ValueError:
        relative = source.name
    return relative, str(resolved)


def _empty_manifest(config: PipelineConfig) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "input_dir": str(config.input_dir.resolve()),
        "output_dir": str(config.output_dir.resolve()),
        "sources": {},
    }


def _manifest_path(config: PipelineConfig) -> Path:
    return config.output_dir / MANIFEST_FILENAME


def _load_manifest_path(path: Path, config: PipelineConfig | None = None) -> dict[str, Any]:
    if not path.exists():
        if config is None:
            raise FileNotFoundError(path)
        return _empty_manifest(config)
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Manifeste de contrôle manuel incompatible : {path}")
    data.setdefault("sources", {})
    return data


def _write_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=".controle-manuel-",
        suffix=".json",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _source_entry(
    manifest: dict[str, Any],
    source: Path,
    config: PipelineConfig,
) -> tuple[str, dict[str, Any]]:
    relative, absolute = _source_identity(source, config)
    sources = manifest.setdefault("sources", {})
    entry = sources.setdefault(
        relative,
        {
            "source": relative,
            "source_absolute": absolute,
            "outputs": [],
            "errors": [],
        },
    )
    entry["source_absolute"] = absolute
    entry.setdefault("outputs", [])
    entry.setdefault("errors", [])
    return relative, entry


def record_source_start(config: PipelineConfig, source: Path) -> None:
    """Crée l'entrée manifeste d'une source avant son traitement."""

    path = _manifest_path(config)
    manifest = _load_manifest_path(path, config)
    _source_entry(manifest, source, config)
    _write_manifest(path, manifest)


def record_output(
    config: PipelineConfig,
    source: Path,
    output: Path,
    *,
    frame_index: int,
    subject_index: int,
    qc_passed: bool,
    qc_problems: list[str],
) -> None:
    """Enregistre une sortie effectivement écrite et son état QC."""

    path = _manifest_path(config)
    manifest = _load_manifest_path(path, config)
    _, entry = _source_entry(manifest, source, config)
    record = {
        "name": output.name,
        "frame_index": int(frame_index),
        "subject_index": int(subject_index),
        "qc_passed": bool(qc_passed),
        "qc_problems": list(qc_problems),
    }
    outputs = entry["outputs"]
    for index, existing in enumerate(outputs):
        if existing.get("name") == output.name:
            outputs[index] = record
            break
    else:
        outputs.append(record)
    _write_manifest(path, manifest)


def record_error(
    config: PipelineConfig,
    source: Path,
    error: Exception,
    *,
    frame_index: int | None = None,
    subject_index: int | None = None,
) -> None:
    """Enregistre un refus ou une erreur qui nécessite potentiellement une reprise."""

    path = _manifest_path(config)
    manifest = _load_manifest_path(path, config)
    _, entry = _source_entry(manifest, source, config)
    record = {
        "frame_index": frame_index,
        "subject_index": subject_index,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if record not in entry["errors"]:
        entry["errors"].append(record)
    _write_manifest(path, manifest)


def _resolve_source(
    entry: dict[str, Any],
    manifest: dict[str, Any],
    config: PipelineConfig,
) -> Path:
    relative = Path(str(entry.get("source", "")))
    candidates = [config.input_dir / relative]
    manifest_input = manifest.get("input_dir")
    if manifest_input:
        candidates.append(Path(str(manifest_input)) / relative)
    absolute = entry.get("source_absolute")
    if absolute:
        candidates.append(Path(str(absolute)))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return candidates[0]


def _copy_without_overwrite(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    shutil.copy2(source, target)
    return "created"


def _manual_name_for_missing_output(output_name: str, source: Path) -> str:
    return f"{Path(output_name).stem}{source.suffix}"


def _manual_name_for_error(
    source: Path,
    error: dict[str, Any],
    *,
    single_plain_name: bool,
    ordinal: int,
) -> str:
    if single_plain_name:
        return source.name
    frame = error.get("frame_index")
    subject = error.get("subject_index")
    parts = [source.stem, "echec"]
    if frame is not None:
        parts.append(f"f{frame}")
    if subject is not None:
        parts.append(f"s{subject}")
    if frame is None and subject is None:
        parts.append(str(ordinal))
    return "-".join(parts) + source.suffix


def run_manual_review(config: PipelineConfig) -> dict[str, int]:
    """Analyse les manifestes et prépare les fichiers à reprendre manuellement.

    Les originaux sont uniquement lus puis copiés avec ``shutil.copy2``. Ils ne
    sont jamais déplacés, renommés, ouverts en écriture ou supprimés.
    """

    output_root = config.output_dir
    if not output_root.exists():
        return {
            "manifests_found": 0,
            "missing_outputs": 0,
            "automatic_failures": 0,
            "qc_to_review": 0,
            "copies_created": 0,
            "copies_existing": 0,
            "sources_missing": 0,
        }

    manifest_paths = sorted(output_root.rglob(MANIFEST_FILENAME))
    totals = {
        "manifests_found": len(manifest_paths),
        "missing_outputs": 0,
        "automatic_failures": 0,
        "qc_to_review": 0,
        "copies_created": 0,
        "copies_existing": 0,
        "sources_missing": 0,
    }

    for manifest_path in manifest_paths:
        manifest = _load_manifest_path(manifest_path)
        lot_dir = manifest_path.parent
        manual_dir = lot_dir / MANUAL_DIRNAME
        deleted_lines: list[str] = []
        failure_lines: list[str] = []
        qc_lines: list[str] = []

        sources = manifest.get("sources", {})
        for source_key in sorted(sources, key=str.casefold):
            entry = sources[source_key]
            source = _resolve_source(entry, manifest, config)
            source_exists = source.exists() and source.is_file()
            outputs = list(entry.get("outputs", []))
            errors = list(entry.get("errors", []))

            missing = [
                record
                for record in outputs
                if not (lot_dir / str(record.get("name", ""))).is_file()
            ]
            for record in missing:
                totals["missing_outputs"] += 1
                output_name = str(record.get("name", "sortie_inconnue"))
                if source_exists:
                    manual_name = _manual_name_for_missing_output(output_name, source)
                    target = manual_dir / manual_name
                    state = _copy_without_overwrite(source, target)
                    totals[f"copies_{state}"] += 1
                    deleted_lines.append(
                        f"- {output_name} | source: {source_key} | copie: "
                        f"{MANUAL_DIRNAME}/{manual_name}"
                    )
                else:
                    totals["sources_missing"] += 1
                    deleted_lines.append(
                        f"- {output_name} | source introuvable: {source_key} | COPIE IMPOSSIBLE"
                    )

            present_qc_failures = [
                record
                for record in outputs
                if not bool(record.get("qc_passed", True))
                and (lot_dir / str(record.get("name", ""))).is_file()
            ]
            for record in present_qc_failures:
                totals["qc_to_review"] += 1
                problems = record.get("qc_problems") or []
                detail = "; ".join(str(item) for item in problems) or "contrôle qualité non conforme"
                qc_lines.append(f"- {record.get('name')} | source: {source_key} | {detail}")

            effective_errors = errors[:]
            if not outputs and not errors:
                effective_errors.append(
                    {
                        "frame_index": None,
                        "subject_index": None,
                        "error_type": "NoOutput",
                        "message": "Aucune sortie produite par le traitement automatique.",
                    }
                )

            for ordinal, error in enumerate(effective_errors, start=1):
                totals["automatic_failures"] += 1
                message = str(error.get("message", "Erreur automatique sans détail."))
                if source_exists:
                    plain_name = len(effective_errors) == 1 and not outputs
                    manual_name = _manual_name_for_error(
                        source,
                        error,
                        single_plain_name=plain_name,
                        ordinal=ordinal,
                    )
                    target = manual_dir / manual_name
                    state = _copy_without_overwrite(source, target)
                    totals[f"copies_{state}"] += 1
                    failure_lines.append(
                        f"- source: {source_key} | raison: {message} | copie: "
                        f"{MANUAL_DIRNAME}/{manual_name}"
                    )
                else:
                    totals["sources_missing"] += 1
                    failure_lines.append(
                        f"- source introuvable: {source_key} | raison: {message} | COPIE IMPOSSIBLE"
                    )

        report_lines = [
            "CONTROLE MANUEL - TRAITEMENT IMAGES LTC",
            "=======================================",
            f"Dossier: {lot_dir.name}",
            "",
            "SORTIES SUPPRIMEES APRES CONTROLE",
            "----------------------------------",
        ]
        report_lines.extend(deleted_lines or ["Aucune."])
        report_lines.extend(
            [
                "",
                "ECHECS / REFUS AUTOMATIQUES",
                "----------------------------",
            ]
        )
        report_lines.extend(failure_lines or ["Aucun."])
        report_lines.extend(
            [
                "",
                "SORTIES QC A VERIFIER (ENCORE PRESENTES)",
                "-----------------------------------------",
            ]
        )
        report_lines.extend(qc_lines or ["Aucune."])
        report_lines.extend(
            [
                "",
                "RAPPEL",
                "------",
                "Les fichiers placés dans A_TRAITER_MANUELLEMENT sont des copies",
                "binaires des originaux. Les fichiers du dossier input ne sont jamais modifiés.",
                "",
            ]
        )
        (lot_dir / REPORT_FILENAME).write_text(
            "\n".join(report_lines),
            encoding="utf-8",
            newline="\n",
        )

    return totals
