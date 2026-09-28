"""Interface en ligne de commande du pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import DEFAULT_CONFIG, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "Détoure, redresse et place chaque sujet sur un fond uniforme sans modifier ses couleurs."
        ),
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG.input_dir,
        help="dossier des images source (défaut : input)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_CONFIG.output_dir,
        help="dossier de sortie (défaut : output)",
    )
    parser.add_argument(
        "--format",
        choices=("tiff", "jpeg", "png"),
        default=DEFAULT_CONFIG.output_format.lower(),
        help="TIFF préserve exactement les couleurs RVB du sujet",
    )
    parser.add_argument(
        "--icc-profile",
        type=Path,
        default=DEFAULT_CONFIG.output_cmyk_icc_profile,
        help="profil ICC CMJN de l'imprimeur",
    )
    parser.add_argument(
        "--split-subjects",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.split_subjects,
        help="séparer les sujets disjoints (activé par défaut)",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=DEFAULT_CONFIG.output_margin_mm,
        help="marge foncée autour de chaque sujet, en millimètres (défaut : 1)",
    )
    parser.add_argument(
        "--straighten",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.straighten,
        help="activer/désactiver le redressement automatique",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CONFIG.recursive,
        help="parcourir les sous-dossiers",
    )
    parser.add_argument(
        "--allow-qc-warnings",
        action="store_true",
        help="renvoyer le code 0 même si un contrôle qualité avertit",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="afficher la configuration effective puis quitter",
    )
    parser.add_argument(
        "--bilan-manuel",
        action="store_true",
        help=(
            "détecter les sorties supprimées/refusées et préparer les copies "
            "des originaux dans A_TRAITER_MANUELLEMENT"
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return replace(
        DEFAULT_CONFIG,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_format=args.format.upper(),
        output_cmyk_icc_profile=args.icc_profile,
        split_subjects=args.split_subjects,
        output_margin_mm=args.margin_mm,
        straighten=args.straighten,
        recursive=args.recursive,
        nonzero_exit_on_qc_failure=not args.allow_qc_warnings,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        if args.print_config:
            print(json.dumps(config.serializable(), indent=2, ensure_ascii=False))
            return 0
        if args.bilan_manuel:
            from .manual_review import run_manual_review

            review = run_manual_review(config)
            print(
                "Bilan manuel : "
                f"{review['manifests_found']} manifeste(s), "
                f"{review['missing_outputs']} sortie(s) supprimée(s), "
                f"{review['automatic_failures']} échec(s)/refus automatique(s), "
                f"{review['qc_to_review']} sortie(s) QC à vérifier, "
                f"{review['copies_created']} copie(s) préparée(s)."
            )
            return 0 if review["manifests_found"] else 3
        # Import tardif : --help, --print-config et --bilan-manuel restent disponibles
        # sans lancer le moteur de traitement d'image.
        from .pipeline import process_batch

        summary = process_batch(config)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERREUR : {exc}", file=sys.stderr)
        return 1

    if summary.processing_errors:
        return 1
    if not summary.sources_found:
        return 3
    if summary.qc_failures and config.nonzero_exit_on_qc_failure:
        return 2
    return 0
