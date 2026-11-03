# cp-sagittal-gait-classification

Clasificación binaria de niños control vs parálisis cerebral a partir de curvas cinemáticas sagitales (cadera, rodilla y tobillo, ambos lados) normalizadas al ciclo de marcha.

## Datos

`derivatives/` (no versionado) con un `gait_cycles_normalized_sagittal.parquet` por trial y `time_series_prediction/subject_split.csv` para el grupo de cada sujeto. Solo se usan los sujetos con ciclos normalizados bilaterales completos; el loader avisa de los que quedan fuera.

## Estructura

```
├── src/sagittal.py                         # Carga, CV agrupada por sujeto, modelos y métricas a nivel paciente
├── notebooks/sagittal_control_vs_case.ipynb  # EDA, evaluación (AUC por paciente) y conclusiones
└── docs/                                   # Informe (formato paper) y decisiones metodológicas, en LaTeX y PDF
```

## Evaluación

Validación cruzada repetida (4 folds × 5 repeticiones), estratificada y **agrupada por sujeto**. La métrica principal es el AUC a nivel paciente con IC95 % por bootstrap de sujetos; sensibilidad, especificidad y exactitud balanceada se reportan aparte a umbral fijo 0.5, sin calibrar.

## Documentos

- [Informe](docs/informe_clasificacion_sagital.pdf): métodos, resultados y limitaciones.
- [Decisiones metodológicas](docs/decisiones_metodologicas.pdf): qué se decidió y por qué (validación cruzada, ponderación, AUC, bootstrap, permutación).

Compilar con `xelatex` (dos pasadas) desde `docs/`.
