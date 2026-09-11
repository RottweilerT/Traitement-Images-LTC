#!/usr/bin/env python3
"""Point d'entrée exécutable.

Exemple :
    python run_pipeline.py ./input --output-dir ./output
"""

from image_pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
