from pathlib import Path

import pandas as pd
import pytest

from breast_lesion_classifier.data.busbra import load_and_validate_manifest, load_official_fold_csv


def test_manifest_rejects_patient_leakage(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"image_path": "a.png", "label": 0, "patient_id": "p1", "split": "train"},
            {"image_path": "b.png", "label": 1, "patient_id": "p1", "split": "val"},
            {"image_path": "c.png", "label": 1, "patient_id": "p2", "split": "test"},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Fuga de pacientes"):
        load_and_validate_manifest(path)


def test_official_fold_csv_is_converted_to_splits(tmp_path: Path) -> None:
    path = tmp_path / "5-fold-cv.csv"
    pd.DataFrame(
        [
            {"ID": "bus_0001-l", "Pathology": "malignant", "kFold": 1},
            {"ID": "bus_0001-r", "Pathology": "malignant", "kFold": 1},
            {"ID": "bus_0002-l", "Pathology": "benign", "kFold": 2},
            {"ID": "bus_0003-l", "Pathology": "benign", "kFold": 3},
        ]
    ).to_csv(path, index=False)

    frame = load_official_fold_csv(path, test_fold=1, validation_fold=2)

    assert frame.set_index("ID").loc["bus_0001-l", "split"] == "test"
    assert frame.set_index("ID").loc["bus_0002-l", "split"] == "val"
    assert frame.set_index("ID").loc["bus_0003-l", "split"] == "train"
    assert frame.set_index("ID").loc["bus_0001-r", "patient_id"] == "bus_0001"
