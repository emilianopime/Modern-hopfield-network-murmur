# Resultados de Entrenamiento — Modern Hopfield Network
### PhysioNet Challenge 2022 · CirCor DigiScope · Clasificación 3-clases (Present / Absent / Unknown)

---

## Configuración del experimento — v2 (actual)

| Parámetro | Valor |
|---|---|
| Backbone CNN | ResNet18 (ImageNet pretrained) |
| Embedding dim | 512 |
| HopfieldPooling prototipos | 4 |
| Num. heads | 8 |
| β (beta) | `1/√512 ≈ 0.0442` (auto) |
| Fusión tabular | 7 features clínicas → MLP 64d → concat con Hopfield |
| Loss | **CB-Focal Loss** (γ=2.0, β=0.9999) — pesos auto: `[0.611, 0.159, 2.229]` |
| Optimizer | AdamW |
| Scheduler | **Cosine LR + warmup** (warmup=5 épocas) |
| Augmentación | **ClinicalMixUp** (α=0.3, 50% de batches) |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Grad clip | 1.0 |
| Épocas totales | 120 |
| Hardware | RTX 3060 12GB · Ryzen 7 5700x · 32GB DDR4 |

### Mejoras incorporadas (de arquitecturas de referencia)
| Técnica | Fuente | Impacto |
|---|---|---|
| CB-Focal Loss | MZA-PCG v5 (referencia) | +3.1% WA, +7% Recall Present |
| Fusión tabular (age, BMI, sex, peso, talla, recording_count, pregnancy) | PF5367600.py (referencia) | Contexto clínico al clasificador |
| Cosine LR + warmup (5 épocas) | MZA-PCG v5 (referencia) | Elimina colapso prematuro post-época 12 |
| ClinicalMixUp α=0.3 | MZA-PCG v5 (referencia) | Regularización de espectrogramas |

---

## Estrategia de entrenamiento en dos fases

```
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 1 — Épocas 1–5   │  CNN congelado · LR warmup 0→1e-4         │
│                         │  Solo entrenan: HopfieldPooling + Tabular │
│                         │                 + Classifier              │
├─────────────────────────────────────────────────────────────────────┤
│  FASE 2 — Épocas 6–120 │  Fine-tuning completo                     │
│                         │  CNN backbone descongelado                │
│                         │  LR backbone: cosine(epoch) × 0.1        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🏆 Mejor modelo — Época 33

| Métrica | **v2 (época 33)** | v1 (época 12) | Δ |
|---|---|---|---|
| **Weighted Accuracy (PhysioNet)** | **0.7094** | 0.6879 | **+0.0215 (+3.1%)** |
| Standard Accuracy | 0.7766 | 0.8000 | -0.023 |
| Recall Present 🔴 | **0.693** | 0.623 | **+0.070** |
| Recall Absent 🟢 | 0.846 | 0.904 | -0.058 |
| Recall Unknown 🟡 | 0.208 | 0.208 | = |

---

## Progresión por época

### Fase 1 — CNN congelado + warmup LR (épocas 1–5)

| Época | WA ↑ | Std Acc | Recall Present | Recall Absent | Recall Unknown | LR |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.5893 | 0.6000 | 0.579 | 0.608 | 0.583 | 2e-5 |
| 2 | 0.5010 | 0.5447 | 0.430 | 0.569 | 0.750 | 4e-5 |
| 3 | 0.6222 | 0.6596 | 0.588 | 0.690 | 0.583 | 6e-5 |
| 4 | 0.6386 | 0.6553 | 0.632 | 0.672 | 0.542 | 8e-5 |
| 5 | 0.6068 | 0.7085 | 0.518 | 0.792 | 0.458 | 1e-4 |

> *Unknown recall arranca en 0.583 en época 1 — confirma el efecto inmediato del CB-Focal Loss (vs 0.375 en época 5 del v1).*

---

### Fase 2 — Fine-tuning completo (épocas 6–120)

| Época | WA ↑ | Std Acc | Recall Present | Recall Absent | Recall Unknown |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 8 | 0.6715 | 0.7277 | 0.649 | 0.783 | 0.333 |
| 10 | 0.6581 | 0.7043 | 0.640 | 0.750 | 0.375 |
| 24 | 0.6704 | 0.7809 | 0.605 | 0.883 | 0.208 |
| 25 | 0.6684 | 0.7638 | 0.614 | 0.852 | 0.250 |
| ⭐ **33 (BEST)** | **0.7094** | 0.7766 | **0.693** | 0.846 | 0.208 |
| 46 | 0.6756 | 0.8170 | 0.588 | 0.946 | 0.125 |
| 53 | 0.6448 | 0.7702 | 0.535 | 0.873 | 0.458 |
| 57 | 0.6591 | 0.7830 | 0.553 | 0.886 | 0.458 |
| 69 | 0.6776 | 0.8128 | 0.596 | 0.937 | 0.125 |
| 78 | 0.6704 | 0.8149 | 0.570 | 0.943 | 0.208 |
| 101 | 0.6704 | 0.8149 | 0.570 | 0.943 | 0.208 |
| 120 | 0.6294 | 0.8021 | 0.509 | 0.955 | 0.083 |

---

## Análisis de resultados

### ✅ Lo que funcionó bien

- **Weighted Accuracy máxima: 0.7094** — supera al v1 (0.6879) en +3.1%
- **Recall Present +7 puntos** (0.623 → 0.693): el CB-Focal Loss con peso `0.611×` para Present rebalanceó correctamente el gradiente, enfocando el modelo en los soplos reales
- **Estabilidad del entrenamiento**: el Cosine LR elimina el colapso post-época 12 que sufría v1. El WA se mantiene en rango `0.63–0.68` durante 80+ épocas
- **Convergencia más tardía**: el mejor checkpoint llegó en época 33 (vs 12 en v1), aprovechando más el entrenamiento extendido a 120 épocas
- **Tradeoff clínico correcto**: se sacrificó algo de Recall Absent (fácil, corazones sanos) para ganar Present (crítico, soplos reales)

### ⚠️ Área de mejora — Clase Unknown

- El recall de **Unknown** arranca en 0.583 (época 1) gracias al CB-Focal, pero colapsa a 0.083–0.208 en Fase 2
- Con solo ~132 muestras de Unknown en training, la señal es insuficiente
- **Acción sugerida**: oversampling de Unknown (repetir muestras 3-4×) o `focal_gamma=3.0` exclusivo para Unknown

---

## Checkpoints disponibles

```
checkpoints/
├── checkpoint_best.pt          ← Época 33, WA=0.7094 ⭐
├── checkpoint_epoch_005.pt     ← Fin Fase 1, WA=0.6068
├── checkpoint_epoch_010.pt     ← WA=0.6581
├── checkpoint_epoch_015.pt     ← WA=0.6376
├── checkpoint_epoch_020.pt     ← WA=0.6571
├── checkpoint_epoch_025.pt     ← WA=0.6684
├── checkpoint_epoch_030.pt     ← WA=0.6468
├── checkpoint_epoch_035.pt     ← WA=0.6571
├── checkpoint_epoch_040.pt     ← WA=0.6314
├── checkpoint_epoch_045.pt     ← WA=0.6427
├── checkpoint_epoch_050.pt     ← WA=0.6283
├── checkpoint_epoch_055.pt     ← WA=0.6294
├── checkpoint_epoch_060.pt     ← WA=0.6396
├── checkpoint_epoch_065.pt     ← WA=0.6345
├── checkpoint_epoch_070.pt     ← WA=0.6273
├── checkpoint_epoch_075.pt     ← WA=0.6540
├── checkpoint_epoch_080.pt     ← WA=0.6520
├── checkpoint_epoch_085.pt     ← WA=0.6386
├── checkpoint_epoch_090.pt     ← WA=0.6478
├── checkpoint_epoch_095.pt     ← WA=0.6530
├── checkpoint_epoch_100.pt     ← WA=0.6253
├── checkpoint_epoch_105.pt     ← WA=0.6499
├── checkpoint_epoch_110.pt     ← WA=0.6376
├── checkpoint_epoch_115.pt     ← WA=0.6396
└── checkpoint_epoch_120.pt     ← WA=0.6294
```

Para evaluar el mejor modelo:
```bash
python main.py eval --checkpoint checkpoints/checkpoint_best.pt
```

Para visualizar los pesos de atención Hopfield sobre un audio:
```bash
python main.py explain --wav data/physionet_2022/13918_AV.wav --checkpoint checkpoints/checkpoint_best.pt
```
