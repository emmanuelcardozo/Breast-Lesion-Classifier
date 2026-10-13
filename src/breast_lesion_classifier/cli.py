import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breast-lesion-classifier",
        description="Herramientas reproducibles para el proyecto de clasificación mamaria.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("audit", help="Auditar un dataset y construir su manifiesto")
    subparsers.add_parser("split", help="Crear particiones agrupadas y estratificadas")
    train_parser = subparsers.add_parser("train", help="Entrenar un modelo configurado")
    train_parser.add_argument(
        "--config",
        default="configs/busbra_efficientnet_b0.yaml",
        help="Ruta al archivo YAML del experimento",
    )
    subparsers.add_parser("evaluate", help="Evaluar un modelo congelado")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    elif args.command == "train":
        from .training import train_from_config

        checkpoint = train_from_config(args.config)
        print(f"Mejor checkpoint: {checkpoint}")
    else:
        parser.error(f"El comando '{args.command}' todavía no está implementado")
    return 0
