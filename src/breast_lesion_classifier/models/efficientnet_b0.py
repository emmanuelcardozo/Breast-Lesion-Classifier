"""EfficientNet-B0 binaria basada en la API multi-weight de torchvision."""

from __future__ import annotations

from typing import Any


def build_efficientnet_b0(
    *,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> tuple[Any, Any | None]:
    """Construye EfficientNet-B0 con una única salida logit.

    Devuelve ``(model, weights)``. El segundo elemento permite documentar qué
    pesos y normalización corresponden a la inicialización seleccionada.
    """
    try:
        from torch import nn
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "No se pudo importar PyTorch/torchvision. Instalá el extra "
            "'deep-learning' con versiones compatibles."
        ) from exc

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    return model, weights


def set_backbone_trainable(model: Any, trainable: bool) -> None:
    """Congela o habilita el extractor, manteniendo entrenable la cabeza."""
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

