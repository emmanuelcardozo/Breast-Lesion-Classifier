"""Punto de entrada para entrenar EfficientNet-B0 con BUS-BRA."""

from __future__ import annotations

import argparse

from breast_lesion_classifier.training import train_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/busbra_efficientnet_b0.yaml",
        help="Ruta a la configuración YAML",
    )
    args = parser.parse_args()
    checkpoint = train_from_config(args.config)
    print(f"Mejor checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

