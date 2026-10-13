"""Entrenamiento reproducible de EfficientNet-B0 para BUS-BRA."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from .data.busbra import (
    BusBraDataset,
    build_transforms,
    load_and_validate_manifest,
    load_official_fold_csv,
)
from .models.efficientnet_b0 import build_efficientnet_b0, set_backbone_trainable


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_auc: float
    val_ap: float


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(dataset: Any, *, batch_size: int, workers: int, shuffle: bool, device: Any):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _positive_weight(frame: pd.DataFrame, label_column: str, positive_labels: list[Any]) -> float:
    positive = {str(value).lower() for value in positive_labels}
    is_positive = frame[label_column].map(lambda value: str(value).lower() in positive)
    positives = int(is_positive.sum())
    negatives = int((~is_positive).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("La partición de entrenamiento debe contener ambas clases")
    return negatives / positives


def _write_runtime_manifest(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Guarda el manifiesto resuelto para trazabilidad y validación común."""
    path = output_dir / "resolved_manifest.csv"
    frame.to_csv(path, index=False)
    return path


def _run_epoch(model, loader, criterion, device, *, optimizer=None, scaler=None):
    import torch

    training = optimizer is not None
    model.train(training)
    losses, labels, scores = [], [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images).squeeze(1)
                loss = criterion(logits, target)
            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
        losses.append(float(loss.detach().cpu()))
        labels.extend(target.detach().cpu().numpy().tolist())
        scores.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    return float(np.mean(losses)), np.asarray(labels), np.asarray(scores)


def train_from_config(config_path: str | Path) -> Path:
    """Entrena, guarda el mejor checkpoint y devuelve su ruta."""
    import torch

    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["project"]["seed"])
    seed_everything(seed)

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if "fold_csv" in data_cfg:
        frame = load_official_fold_csv(
            data_cfg["fold_csv"],
            test_fold=int(data_cfg["test_fold"]),
            validation_fold=int(data_cfg["validation_fold"]),
            image_extension=str(data_cfg.get("image_extension", ".png")),
        )
    else:
        frame = load_and_validate_manifest(
            data_cfg["manifest"],
            image_column=data_cfg["image_column"],
            label_column=data_cfg["label_column"],
            patient_column=data_cfg["patient_column"],
            split_column=data_cfg["split_column"],
        )
    frame = load_and_validate_manifest(
        _write_runtime_manifest(frame, output_dir),
        image_column=data_cfg["image_column"],
        label_column=data_cfg["label_column"],
        patient_column=data_cfg["patient_column"],
        split_column=data_cfg["split_column"],
    )
    split_col = data_cfg["split_column"]
    subsets = {name: frame[frame[split_col] == name].copy() for name in ("train", "val", "test")}
    datasets = {
        name: BusBraDataset(
            subset,
            image_root=data_cfg["image_root"],
            image_column=data_cfg["image_column"],
            label_column=data_cfg["label_column"],
            positive_labels=data_cfg["positive_labels"],
            transform=build_transforms(int(data_cfg["image_size"]), train=name == "train"),
        )
        for name, subset in subsets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        name: _loader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            workers=int(data_cfg["num_workers"]),
            shuffle=name == "train",
            device=device,
        )
        for name, dataset in datasets.items()
    }

    model, weights = build_efficientnet_b0(
        pretrained=model_cfg["weights"] == "imagenet",
        dropout=float(model_cfg["dropout"]),
    )
    model.to(device)
    freeze_epochs = int(model_cfg["freeze_backbone_epochs"])
    set_backbone_trainable(model, trainable=freeze_epochs == 0)

    pos_weight = _positive_weight(
        subsets["train"], data_cfg["label_column"], data_cfg["positive_labels"]
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    amp_enabled = bool(train_cfg["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=True) if amp_enabled else None

    best_auc = float("-inf")
    epochs_without_improvement = 0
    history: list[EpochMetrics] = []
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            set_backbone_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(train_cfg["learning_rate"]) / 10,
                weight_decay=float(train_cfg["weight_decay"]),
            )
        train_loss, _, _ = _run_epoch(
            model, loaders["train"], criterion, device, optimizer=optimizer, scaler=scaler
        )
        with torch.inference_mode():
            val_loss, labels, scores = _run_epoch(model, loaders["val"], criterion, device)
        val_auc = float(roc_auc_score(labels, scores))
        val_ap = float(average_precision_score(labels, scores))
        metrics = EpochMetrics(epoch, train_loss, val_loss, val_auc, val_ap)
        history.append(metrics)
        print(json.dumps(asdict(metrics), ensure_ascii=False))

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "weights": str(weights) if weights is not None else None,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(train_cfg["early_stopping_patience"]):
                break

    pd.DataFrame([asdict(item) for item in history]).to_csv(
        output_dir / "history.csv", index=False
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return checkpoint_path
