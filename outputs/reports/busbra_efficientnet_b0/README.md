# EfficientNet B0 entrenada con BUS BRA

## Resumen de la corrida

Esta carpeta documenta el experimento de clasificación binaria de lesiones mamarias entrenado con BUS-BRA mediante EfficientNet-B0. La clase positiva es malignidad y la evaluación se realizó sobre el fold de prueba reservado.

Este experimento es distinto del protocolo principal del Trabajo Final, que propone entrenar con BUSI y utilizar BUS-BRA como validación externa. Sus resultados no deben presentarse como validación externa de un modelo entrenado con BUSI.

## Resultados

El conjunto de prueba contiene 376 imágenes: 254 benignas y 122 malignas.

| Métrica | Resultado |
|---|---:|
| AUC ROC | 0.9352 |
| Average precision | 0.8960 |
| Accuracy con umbral 0.5 | 0.8697 |
| Sensibilidad con umbral 0.5 | 0.7951 |
| Especificidad con umbral 0.5 | 0.9055 |
| Precisión para malignidad | 0.8017 |
| F1 para malignidad | 0.7984 |

### Matriz de confusión con umbral 0.5

|  | Predicción benigna | Predicción maligna |
|---|---:|---:|
| Referencia benigna | 230 | 24 |
| Referencia maligna | 25 | 97 |

El mejor AUC de validación registrado fue 0.8518 en la época 20. La pérdida de entrenamiento continuó descendiendo mientras la pérdida de validación mostró oscilaciones, por lo que los resultados deben interpretarse junto con las curvas y sin asumir generalización clínica.

## Archivos de la corrida

### Modelo

- `models/busbra_efficientnet_b0/best_model.pt`: checkpoint seleccionado mediante validación.

### Métricas

- `outputs/metrics/busbra_efficientnet_b0/history.csv`: historial por época.
- `outputs/metrics/busbra_efficientnet_b0/test_predictions.csv`: puntuación y predicción por imagen.
- `outputs/metrics/busbra_efficientnet_b0/resolved_manifest.csv`: particiones resueltas a partir del CSV oficial de folds.

### Figuras

- `outputs/figures/resultados_efficientnet_b0/training_curves.png`: pérdida y métricas de validación.
- `outputs/figures/resultados_efficientnet_b0/test_curves.png`: curvas ROC y precisión recall del test.

## Reproducción

La configuración versionada se encuentra en `configs/busbra_efficientnet_b0.yaml`. El entrenamiento puede iniciarse desde el repositorio con:

```powershell
python -m breast_lesion_classifier train --config configs/busbra_efficientnet_b0.yaml
```

El notebook utilizado en Google Colab se encuentra en:

```text
notebooks/Entrenamiento_BUS_BRA_EfficientNet_B0_Colab.ipynb
```

## Alcance y limitaciones

- Las métricas corresponden a imágenes y no prueban utilidad clínica.
- El umbral 0.5 es un punto de referencia y no un umbral clínico validado.
- Los intervalos de confianza por paciente todavía no fueron calculados.
- La calibración, los umbrales de alta sensibilidad y el análisis sistemático de errores permanecen pendientes.
- El modelo no detecta lesiones durante un examen completo. Clasifica una imagen que ya contiene una lesión localizada.
- Antes de comunicar resultados definitivos debe verificarse la independencia por paciente del fold utilizado y documentarse la versión exacta de los datos y del entorno de entrenamiento.

