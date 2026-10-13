from breast_lesion_classifier.cli import build_parser


def test_cli_exposes_expected_commands() -> None:
    help_text = build_parser().format_help()
    for command in ("audit", "split", "train", "evaluate"):
        assert command in help_text

