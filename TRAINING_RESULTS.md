# Resultados de Entrenamiento — Modern Hopfield Network
### PhysioNet Challenge 2022 · CirCor DigiScope · Clasificación 3-clases (Present / Absent / Unknown)

---

## 🏆 Comparativa de versiones

| Métrica | v1 (época 12) | v2 (época 33) | **v3 (época 117)** | Δ v2→v3 |
|---|:---:|:---:|:---:|:---:|
| **Weighted Accuracy** | 0.6879 | 0.7094 | **0.7197** | **+0.0103 (+1.5%)** |
| Standard Accuracy | 0.8000 | 0.7766 | 0.7511 | -0.026 |
| Recall Present 🔴 | 0.623 | 0.693 | **0.711** | **+0.018** |
| Recall Absent 🟢 | 0.904 | 0.846 | 0.783 | -0.063 |
| Recall Unknown 🟡 | 0.208 | 0.208 | **0.500** | **+0.292 🚀** |
| Mejor checkpoint | época 12 | época 33 | **época 117** | desplazado +84 épocas |

> **Unknown recall +29.2 pp** (0.208→0.500): el WeightedRandomSampler (Unknown×10) atacó directamente el colapso de clase minoritaria.

---

## Configuración del experimento — v3 (actual)

| Parámetro | v2 | **v3** | Cambio |
|---|---|---|---|
| Backbone CNN | ResNet18 pretrained | ResNet18 pretrained | = |
| Freeze backbone | 5 épocas | **12 épocas** | +7 ← consolida Hopfield antes de perturbar CNN |
| Dropout classifier | 0.30 | **0.45** | +0.15 ← frena memorización |
| Learning rate | 1e-4 | **5e-5** | ×0.5 ← convergencia más lenta |
| Weight decay | 1e-4 | **5e-4** | ×5 ← más regularización L2 |
| LR Scheduler | Cosine + warmup | **SGDR (T₀=20)** | Warm Restarts cada 20 épocas |
| Label smoothing | — | **ε=0.1** | Reduce sobreconfianza en Absent |
| WeightedSampler | — | **[5.0, 1.0, 10.0]** | Unknown×10, Present×5 |
| SpecAugment | — | **p=0.5** | Enmascara 15% freq + 15% tiempo |
| Augmentación | ClinicalMixUp α=0.3 | ClinicalMixUp α=0.3 | = |
| Épocas | 120 | 120 | = |
| Hardware | RTX 3060 12GB · Ryzen 7 5700x | RTX 3060 12GB · Ryzen 7 5700x | = |

---

## Estrategia de entrenamiento en dos fases — v3

```
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 1 — Épocas 1–12  │  CNN congelado · LR warmup 0→5e-5         │
│                         │  Solo entrenan: HopfieldPooling + Tabular │
│                         │                 + Classifier              │
│                         │  WeightedSampler: Unknown×10, Present×5  │
├─────────────────────────────────────────────────────────────────────┤
│  FASE 2 — Épocas 13–120│  Fine-tuning completo (CNN descongelado)  │
│                         │  SGDR: restarts en épocas 26,46,66,86,106│
│                         │  LR backbone: sgdr_lr(epoch) × 0.1       │
│                         │  SpecAugment + ClinicalMixUp activos      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🏆 Mejor modelo v3 — Época 117

| Métrica | Valor |
|---|---|
| **Weighted Accuracy (PhysioNet)** | **0.7197** |
| Standard Accuracy | 0.7511 |
| Recall Present 🔴 | **0.711** |
| Recall Absent 🟢 | 0.783 |
| Recall Unknown 🟡 | **0.500** ← era 0.208 en v2 |

---

## Progresión por época — v3

### Fase 1 — CNN congelado + warmup LR (épocas 1–12)

| Época | WA ↑ | Recall Present | Recall Absent | Recall Unknown | LR |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.3758 | 0.500 | 0.063 | **0.833** | 1e-5 |
| 2 | 0.4004 | 0.518 | 0.096 | **0.875** | 2e-5 |
| 3 | 0.4825 | 0.667 | 0.099 | **0.792** | 3e-5 |
| 5 | 0.5133 | 0.649 | 0.202 | **0.875** | 5e-5 |
| 12 | 0.4322 | 0.579 | 0.084 | **0.875** | 4.0e-5 |

> *Unknown recall >0.79 durante toda la Fase 1 — WeightedRandomSampler funciona desde el arranque.*

---

### Fase 2 — Fine-tuning + SGDR (épocas 13–120)

| Época | WA ↑ | Recall Present | Recall Absent | Recall Unknown | Evento |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 14 | 0.5164 | 0.693 | 0.136 | 0.875 | CNN descongelado (época 13) |
| 25 | 0.5205 | 0.675 | 0.178 | 0.875 | Fin ciclo 1 SGDR |
| 26 | — | — | — | — | 🔄 Restart 1 |
| 35 | 0.5862 | 0.711 | 0.319 | 0.833 | |
| 45 | 0.5986 | 0.728 | 0.343 | 0.750 | Fin ciclo 2 SGDR |
| 46 | — | — | — | — | 🔄 Restart 2 |
| 60 | 0.6273 | 0.693 | 0.515 | 0.625 | |
| 65 | 0.5698 | 0.640 | 0.428 | 0.667 | Fin ciclo 3 SGDR |
| 66 | — | — | — | — | 🔄 Restart 3 |
| 75 | 0.6704 | 0.719 | 0.605 | 0.583 | |
| 80 | 0.6786 | 0.711 | 0.645 | 0.583 | |
| 85 | 0.6520 | 0.693 | 0.605 | 0.542 | Fin ciclo 4 SGDR |
| 86 | — | — | — | — | 🔄 Restart 4 |
| 87 | 0.7074 | 0.763 | 0.666 | 0.458 | ✓ Nuevo récord post-restart |
| 92 | 0.6940 | 0.702 | 0.741 | 0.417 | |
| 98 | 0.7166 | 0.746 | 0.723 | 0.458 | ⭐ |
| 105 | 0.7023 | 0.719 | 0.726 | 0.458 | Fin ciclo 5 SGDR |
| 106 | 0.7074 | 0.711 | 0.747 | 0.500 | 🔄 Restart 5 |
| 110 | 0.7136 | 0.763 | 0.675 | 0.500 | |
| ⭐ **117 (BEST)** | **0.7197** | 0.711 | **0.783** | **0.500** | |
| 120 | 0.6674 | 0.702 | 0.599 | 0.708 | Fin |

---

## Análisis de resultados v3

### ✅ Lo que funcionó bien

- **Weighted Accuracy: 0.7197** — nuevo máximo histórico del proyecto (+1.5% sobre v2)
- **Unknown recall: 0.500** ← era 0.208 en v1 y v2 (+29.2 pp). El WeightedRandomSampler (Unknown×10) resolvió el colapso de clase minoritaria que persistía desde v1
- **Checkpoint tardío: época 117** — el objetivo era llegar a época 60+, el modelo encontró su mejor punto en época 117 gracias al SGDR
- **SGDR efectivo**: cada restart (épocas 26, 46, 66, 86, 106) produjo un salto de WA. El mayor salto en restart 4 (época 86→87: 0.6520→0.7074)
- **Present recall: 0.711** — sube respecto a v2 (0.693), el CB-Focal Loss + WeightedSampler mantiene el foco en soplos

### ⚠️ Tradeoffs observados

- **Absent recall: 0.783** — baja respecto a v2 (0.846). Esperado: el WeightedSampler reduce la frecuencia de Absent en training, sacrificando algo de esa clase para ganar Unknown y Present
- **Standard Accuracy: 0.7511** — menor que v2 (0.7766) por el mismo motivo. La métrica del challenge (Weighted Accuracy) pesa más: Present×5, Unknown×3, Absent×1
- **WA volátil en fase media**: épocas 50-85 muestran oscilaciones 0.57-0.68 — el SGDR causa subidas y bajadas. Usar siempre el checkpoint_best.pt, no el del checkpoint más reciente

### 📊 Resumen clínico del tradeoff

| Escenario | Clase | Peso Challenge | Impacto |
|---|---|:---:|---|
| Soplo real → clasificado como presente | Present | **5** | ✅ Alta recompensa |
| Normal → clasificado como ausente | Absent | **1** | Baja recompensa |
| Dudoso → clasificado como unknown | Unknown | **3** | ✅ Recuperado en v3 |

La v3 optimiza correctamente: maximiza la captura de casos clínicamente importantes (Present + Unknown) a costa del caso más fácil (Absent, corazones sanos).

---

## Checkpoints disponibles (v3)

```
checkpoints/
├── checkpoint_best.pt          ← Época 117, WA=0.7197 ⭐ (v3)
├── checkpoint_epoch_005.pt
├── checkpoint_epoch_010.pt
├── checkpoint_epoch_015.pt
...
└── checkpoint_epoch_120.pt
```

Para evaluar el mejor modelo:
```bash
python main.py eval --checkpoint checkpoints/checkpoint_best.pt
```

Para visualizar los pesos de atención Hopfield:
```bash
python main.py explain --wav data/physionet_2022/13918_AV.wav --checkpoint checkpoints/checkpoint_best.pt
```

---

## Historial de versiones

| Versión | Fecha | Mejor WA | Mejor época | Mejora clave |
|---|---|:---:|:---:|---|
| v1 | 2026-05-24 | 0.6879 | 12 | Línea base MHN |
| v2 | 2026-05-24 | 0.7094 | 33 | CB-Focal Loss + TabularEncoder + ClinicalMixUp |
| **v3** | **2026-05-26** | **0.7197** | **117** | SGDR + WeightedSampler + SpecAugment + Label Smoothing |
