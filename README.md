# Clasificador de lesiones mamarias

Proyecto académico de la Maestría en Inteligencia Artificial de la Facultad de Ingeniería de la Universidad de Buenos Aires (FIUBA), desarrollado para la asignatura Fundamentos de Medicina y Aplicaciones de IA.

El objetivo es construir y evaluar un clasificador binario de lesiones mamarias en ecografías. BUSI se utilizará para desarrollo y evaluación interna; BUS-BRA quedará reservado para validación externa sin reajuste. La clase positiva será malignidad.

> Estado actual: repositorio inicial y protocolo versión 2. Todavía no se descargaron datasets, no se entrenaron modelos y no existen resultados experimentales.

## Principios del proyecto

- Evitar fuga de información mediante particiones por paciente o por grupos de imágenes relacionadas.
- Auditar duplicados, etiquetas, máscaras y marcas gráficas antes de entrenar.
- Elegir modelo, época y umbral únicamente con entrenamiento y validación.
- Mantener congelados el test interno y las cohortes externas hasta cerrar el protocolo.
- Reportar sensibilidad, especificidad, falsos negativos, discriminación, calibración e incertidumbre.
- Tratar Grad-CAM como herramienta de auditoría, no como explicación causal ni validación clínica.
- No distribuir imágenes clínicas ni credenciales en Git.

## Estructura

```text
configs/              Configuraciones versionadas de experimentos
data/
  raw/                Datos originales, inmutables y fuera de Git
  interim/            Archivos intermedios de auditoría
  processed/          Datos preparados para entrenamiento
docs/                 Protocolo, decisiones y ficha del modelo
models/               Pesos locales, fuera de Git
notebooks/            Exploración; el pipeline reproducible vive en scripts
outputs/
  figures/            Figuras exportadas
  metrics/            Métricas y tablas
  reports/            Informes derivados
scripts/              Puntos de entrada ejecutables
src/breast_lesion_classifier/
  data/                Inventario, auditoría, particiones y transformaciones
  evaluation/          Métricas, bootstrap y calibración
  explainability/      Grad-CAM y análisis de errores
  models/              Baselines y modelos PyTorch
tests/                 Pruebas automatizadas
```

## Preparación del entorno

Se recomienda Python 3.11.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

La instalación base no descarga los datasets. Las dependencias de aprendizaje profundo se incorporarán cuando se verifique el hardware disponible y se congelen sus versiones.

## Primer flujo de trabajo

1. Registrar fuente, versión, licencia, fecha y SHA-256 de cada descarga.
2. Ejecutar el inventario de BUSI.
3. Identificar imágenes, máscaras, duplicados exactos y candidatos perceptuales.
4. Revisar conflictos de etiquetas y marcas de medición.
5. Publicar el manifiesto de auditoría sin imágenes.
6. Definir particiones 60/20/20 estratificadas y agrupadas.
7. Implementar baselines antes de ajustar EfficientNet-B0.

Comando inicial disponible:

```powershell
python -m breast_lesion_classifier --help
```

## Entrenamiento con BUS-BRA

El módulo `breast_lesion_classifier.training` entrena EfficientNet-B0 directamente
desde los archivos oficiales `5-fold-cv.csv` o `10-fold-cv.csv`. Usa `ID`,
`Pathology` y `kFold` para localizar las imágenes y construir entrenamiento,
validación y prueba. Antes de iniciar, controla que ningún paciente aparezca en
folds diferentes. `bus_data.csv` se conserva como fuente de metadatos y no se usa
para sustituir la etiqueta de patología por BI-RADS.

```powershell
python -m pip install -e ".[dev,deep-learning]"
$env:PYTHONPATH="src"
python scripts/train_busbra.py --config configs/busbra_efficientnet_b0.yaml
```

La configuración congela inicialmente el backbone, usa pérdida binaria ponderada,
AdamW, precisión mixta en GPU y detención temprana según AUC de validación. Guarda
`best_model.pt`, `history.csv` y la configuración resuelta bajo `outputs/`.

El protocolo V2 original usa BUSI para entrenamiento y reserva BUS-BRA para
validación externa. Entrenar con BUS-BRA constituye un experimento separado y
debe identificarse de ese modo en resultados e informes.

## Datos y privacidad

`data/`, `models/` y `outputs/` ignoran archivos generados o descargados, salvo archivos `.gitkeep`. No deben versionarse imágenes, archivos DICOM, identificadores de pacientes, tokens, claves ni pesos del modelo. Los manifiestos publicables usarán identificadores opacos.

## Reproducibilidad

Cada corrida deberá registrar configuración, semilla, versiones, hardware, particiones, huellas de los datos y commit de Git. El archivo `configs/baseline.yaml` es el punto de partida y será ampliado cuando se implemente el pipeline.

## Uso previsto

El proyecto es exclusivamente académico y retrospectivo. No es un producto sanitario, no detecta lesiones en un examen completo y no debe utilizarse para diagnóstico, indicación de biopsia ni seguimiento de pacientes.

## Autor

Esp. Ing. Emmanuel Cardozo
