"""Dataset y controles de integridad para manifiestos de BUS-BRA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REQUIRED_SPLITS = {"train", "val", "test"}


def load_official_fold_csv(
    path: str | Path,
    *,
    test_fold: int,
    validation_fold: int,
    image_extension: str = ".png",
) -> pd.DataFrame:
    """Convierte el CSV oficial de BUS-BRA a un manifiesto estandarizado.

    El archivo oficial contiene ``ID``, ``Pathology`` y ``kFold``. Se usa un
    fold para test, otro para validación y los restantes para entrenamiento.
    El paciente se deriva del prefijo anterior al guion, por ejemplo
    ``bus_0001-l`` y ``bus_0001-r`` pertenecen a ``bus_0001``.
    """
    frame = pd.read_csv(path)
    required = {"ID", "Pathology", "kFold"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas del CSV oficial: {missing}")
    if test_fold == validation_fold:
        raise ValueError("test_fold y validation_fold deben ser diferentes")

    frame = frame.copy()
    frame["kFold"] = pd.to_numeric(frame["kFold"], errors="raise").astype(int)
    available = set(frame["kFold"].unique())
    for fold in (test_fold, validation_fold):
        if fold not in available:
            raise ValueError(f"El fold {fold} no existe; disponibles: {sorted(available)}")

    frame["image_path"] = frame["ID"].astype(str) + image_extension
    frame["label"] = frame["Pathology"].astype(str).str.lower().str.strip()
    frame["patient_id"] = frame["ID"].astype(str).str.rsplit("-", n=1).str[0]
    frame["split"] = "train"
    frame.loc[frame["kFold"] == validation_fold, "split"] = "val"
    frame.loc[frame["kFold"] == test_fold, "split"] = "test"

    patient_folds = frame.groupby("patient_id")["kFold"].nunique()
    leaked = patient_folds[patient_folds > 1]
    if not leaked.empty:
        examples = leaked.index.astype(str).tolist()[:10]
        raise ValueError(f"El CSV asigna un paciente a múltiples folds: {examples}")
    return frame


def load_and_validate_manifest(
    path: str | Path,
    *,
    image_column: str = "image_path",
    label_column: str = "label",
    patient_column: str = "patient_id",
    split_column: str = "split",
) -> pd.DataFrame:
    """Carga un manifiesto y bloquea fuga de pacientes entre particiones."""
    frame = pd.read_csv(path)
    required = {image_column, label_column, patient_column, split_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el manifiesto: {missing}")
    if frame.empty:
        raise ValueError("El manifiesto está vacío")
    if frame[list(required)].isnull().any().any():
        raise ValueError("El manifiesto contiene valores nulos en columnas obligatorias")

    frame = frame.copy()
    frame[split_column] = frame[split_column].astype(str).str.lower().str.strip()
    unexpected = sorted(set(frame[split_column]) - REQUIRED_SPLITS)
    if unexpected:
        raise ValueError(f"Particiones no reconocidas: {unexpected}")
    absent = sorted(REQUIRED_SPLITS - set(frame[split_column]))
    if absent:
        raise ValueError(f"Faltan particiones requeridas: {absent}")

    patient_splits = frame.groupby(patient_column)[split_column].nunique()
    leaked = patient_splits[patient_splits > 1]
    if not leaked.empty:
        examples = leaked.index.astype(str).tolist()[:10]
        raise ValueError(f"Fuga de pacientes entre particiones: {examples}")
    return frame


class SquarePad:
    """Agrega relleno simétrico para conservar la relación de aspecto."""

    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, image: Any) -> Any:
        from torchvision.transforms import functional as F

        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return F.pad(image, [left, top, right, bottom], fill=self.fill)


def build_transforms(image_size: int, *, train: bool) -> Any:
    """Transformaciones moderadas para ecografía en escala de grises."""
    from torchvision import transforms

    operations: list[Any] = [
        transforms.Grayscale(num_output_channels=3),
        SquarePad(fill=0),
        transforms.Resize((image_size, image_size), antialias=True),
    ]
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8, fill=0),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


class BusBraDataset:
    """Dataset liviano basado en un manifiesto auditado."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_root: str | Path,
        image_column: str,
        label_column: str,
        positive_labels: Iterable[Any],
        transform: Any,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.image_column = image_column
        self.label_column = label_column
        self.positive_labels = {str(value).lower() for value in positive_labels}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        from PIL import Image
        import torch

        row = self.frame.iloc[index]
        image_path = Path(str(row[self.image_column]))
        if not image_path.is_absolute():
            image_path = self.image_root / image_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as source:
            image = source.convert("L")
        image = self.transform(image)
        label = float(str(row[self.label_column]).lower() in self.positive_labels)
        return image, torch.tensor(label, dtype=torch.float32)
