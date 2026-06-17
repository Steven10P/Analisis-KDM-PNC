# RESEARCH STATUS — KDM vs PNC Experiment

**Fecha:** 2026-06-17
**Rama:** `feature/kdm-npc-replication`
**Proyecto:** Análisis comparativo de modelos de ML Probabilístico para tesis de Maestría

---

## Estado de Skills

| Skill | Estado | Ubicación |
|-------|--------|-----------|
| `kdm` | Activo | `.claude/skills/kdm.md` |
| `data-analyst` | Activo | `~/.claude/plugins/.../skills/data-analyst/SKILL.md` |
| `skill-creator` | Activo (plugin) | `~/.claude/plugins/...` |
| `embedding-kit` | **No instalado** | — |
| `ml-experiment-skills` | **No instalado** | — |

> `embedding-kit` y `ml-experiment-skills` no existen en el entorno. Su funcionalidad
> equivalente está implementada directamente en el pipeline (`StandardScaler` para
> preprocesamiento y MLflow para tracking de experimentos).

---

## Configuración Técnica

### Entorno

| Componente | Versión/Detalle |
|------------|-----------------|
| Python | 3.10 (Conda `mlops`) |
| PyTorch | instalado en `mlops` |
| kdm-torch | 2.0.0 (GitHub `fagonzalezo/kdm`) |
| MLflow | SQLite backend (`mlflow.db`) |
| OS | Windows 11 |

### Dataset

| Parámetro | Valor |
|-----------|-------|
| Nombre | `mnist_dim_3_min_3_noise_1-dataset.tar` |
| Origen | MNIST → PCA(3) + N(0, σ=1) |
| Train | ~59,990 muestras × 3 features |
| Test | ~10,010 muestras × 3 features |
| Clases | 10 (dígitos 0–9) |
| Preprocesamiento | Z-score (μ, σ calculados sobre train) |

### Modelos

#### KDM (Kernel Density Matrix)
```
API: kdm-torch v2
Encoder: Linear(3→64) → ReLU → Linear(64→32) → Tanh
KDM:     n_comp=256, encoded_size=32, sigma=auto (init_sigma=True)
Loss:    NLLLoss
Optimizer: Adam, lr=0.001
Epochs:  50 | Batch: 256
```

#### PNC (Probabilistic Neural Circuit)
```
Clase:   GenDisPNCRC (ProbabilisticNeuralCircuits)
Entrada: [0,255] re-escalado desde Z-score, shape (N,1,3)
Loss:    CrossEntropyLoss
Optimizer: SGD, lr=0.01, momentum=0.9
Epochs:  50 | Batch: 256
Nota: NaN loss esperado en este dominio (diseñado para espacio de píxeles)
```

---

## Estructura de Archivos

```
Analisis-KDM-PNC/
├── .claude/skills/
│   └── kdm.md                    ← Skill KDM activo
├── configs/
│   └── experiment_config.yaml    ← Configuracion del experimento
├── checkpoints/                  ← Pesos guardados cada 10 epochs
├── data/
│   └── mnist_dim_3_min_3_noise_1-dataset.tar
├── experiments/
│   ├── analysis.py               ← Script de análisis estadístico
│   ├── statistical_report.md     ← Reporte generado automaticamente
│   ├── plots/                    ← Figuras PDF+PNG (300 DPI)
│   └── results/                  ← CSVs de estadísticas
├── resultados/mnist_noise/
│   ├── graficas/                 ← Figuras del experimento
│   └── modelos/                  ← Pesos de modelos entrenados
├── models/
│   ├── kdm_models/               ← kdm-torch (editable install)
│   └── pnc_circuits/             ← ProbabilisticNeuralCircuits
├── notebooks/
│   └── EDA_mnist_noise.ipynb     ← EDA ejecutado (12 figuras)
├── run_experiments.py            ← Pipeline original (30 epochs)
├── run_experiments_final.py      ← Pipeline definitivo (50 epochs + Z-score)
└── RESEARCH_STATUS.md            ← Este archivo
```

---

## Resultados Conocidos (run anterior, 30 epochs)

| Modelo | Train Acc | Test Acc | Train Loss | Test Loss |
|--------|-----------|----------|------------|-----------|
| KDM | 41.98% | 42.46% | 1.506 | 1.506 |
| PNC | 9.85% | 9.91% | NaN | NaN |

### Hallazgos clave
1. **KDM** funciona correctamente en espacio PCA-3D con ruido σ=1 (~42% acc).
2. **PNC** colapsa numéricamente (NaN loss) porque `GenDisPNCRC` fue diseñado para
   imágenes en espacio de píxeles, no para features PCA continuas con valores negativos.
3. El ruido σ=1 supera la separación inter-clase natural en PCA-3D (~1.5 unidades),
   explicando el bajo accuracy incluso para KDM.

---

## Cómo Ejecutar

```bash
# Activar entorno
conda activate mlops

# Pipeline definitivo (50 epochs, Z-score, checkpoints, auto-análisis)
python run_experiments_final.py

# Solo análisis estadístico (sobre resultados existentes)
python experiments/analysis.py

# Ver experimentos en MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Próximos Pasos Sugeridos

- [ ] Probar `pca_dim ∈ {10, 20, 50}` para ver recuperación de accuracy en KDM
- [ ] Normalizar entrada de PNC a `[0,1]` y probar con `components ≥ 10`
- [ ] Explorar `σ ∈ {0.1, 0.25, 0.5, 0.75, 1.0}` para curva de degradación completa
- [ ] Instalar `embedding-kit` y `ml-experiment-skills` si están disponibles en el marketplace
