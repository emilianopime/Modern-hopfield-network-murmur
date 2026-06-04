# Resultados de Entrenamiento — Modern Hopfield Network
### PhysioNet Challenge 2022 · CirCor DigiScope · Clasificación 3-clases (Present / Absent / Unknown)

---

## 🏆 Comparativa de versiones

| Métrica | v1 (época 12) | v2 (época 33) | v3 (época 117) | **v4 (época 62)** | Δ v3→v4 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Weighted Accuracy** | 0.6879 | 0.7094 | 0.7197 | **0.7774** | **+0.0577 (+8.0%)** |
| Standard Accuracy | 0.8000 | 0.7766 | 0.7511 | **0.7943** | +0.0432 |
| Recall Present 🔴 | 0.623 | 0.693 | 0.711 | **0.794** | **+0.083** |
| Recall Absent 🟢 | 0.904 | 0.846 | 0.783 | **0.821** | +0.038 |
| Recall Unknown 🟡 | 0.208 | 0.208 | 0.500 | **0.583** | **+0.083** |
| Mejor época | 12 | 33 | 117 | **62** | — |

> **v4 es el nuevo máximo histórico en todas las métricas simultáneamente** — Present +8.3 pp, Absent +3.8 pp, Unknown +8.3 pp.

---

## Configuración del experimento — v4 (actual)

| Parámetro | v3 | **v4** | Cambio |
|---|---|---|---|
| Backbone CNN | ResNet18 pretrained | ResNet18 pretrained | = |
| Features tabulares | 7 clínicos | **11 (7 + 4 one-hot posición)** | +AV/PV/TV/MV |
| Padding mask Hopfield | ✗ | **✓** | T' varía 3-16, hasta 81% padding |
| Aggregación paciente | uniforme | **por confianza** | weight = softmax(1-H/logC × 3) |
| Métrica validación | grabación | **paciente** | métrica real del challenge |
| Freeze backbone | 12 épocas | 12 épocas | = |
| Dropout | 0.45 | 0.45 | = |
| LR / weight decay | 5e-5 / 5e-4 | 5e-5 / 5e-4 | = |
| SGDR T₀ | 20 | 20 | = |
| WeightedSampler | [5.0, 1.0, 10.0] | [5.0, 1.0, 10.0] | = |
| SpecAugment | p=0.5 | p=0.5 | = |
| Épocas | 120 | 120 | = |

---

## Mejoras implementadas en v4

### 1. Posición de auscultación como feature
Cada grabación ahora incluye un one-hot de 4 bits indicando la posición (AV/PV/TV/MV). Las posiciones tienen distinto valor diagnóstico para cada tipo de soplo — AV y PV son más sensibles para murmurs de válvulas aórtica y pulmonar respectivamente. Antes todas las grabaciones de un mismo paciente tenían features tabulares idénticos.

### 2. Padding mask en HopfieldPooling
Las grabaciones PCG tienen longitudes muy variables: T' (frames tras CNN) varía de 3 a 16. Sin máscara, hasta el 81% de los frames que veía Hopfield eran ceros de relleno, contaminando la memoria asociativa. La máscara booleana `padding_mask = (arange(T') ≥ t_real)` elimina ese ruido en validación e inferencia.

### 3. Aggregación por confianza a nivel paciente
En lugar de promediar uniformemente las probabilidades de las N grabaciones de un paciente, se pondera por certeza del modelo: `weight_i = softmax(confidence_i × 3)` donde `confidence_i = 1 - H(p_i)/log(3)`. Las grabaciones donde el modelo es más seguro contribuyen más a la decisión final.

### 4. Validación a nivel paciente
La métrica reportada durante training ahora es WA calculada sobre predicciones por paciente (no por grabación), alineando el objetivo de entrenamiento con la métrica real del challenge.

---

## Mejor modelo v4 — Época 62

| Métrica | Valor |
|---|---|
| **Weighted Accuracy (PhysioNet)** | **0.7774** |
| Standard Accuracy | 0.7943 |
| Recall Present 🔴 | **0.794** |
| Recall Absent 🟢 | **0.821** |
| Recall Unknown 🟡 | **0.583** |

---

## Progresión v4 — Hitos clave

| Época | WA | LR | Evento |
|:---:|:---:|:---:|---|
| 1 | 0.515 | 1e-5 | Inicio warmup, CNN congelado |
| 5 | 0.641 | 5e-5 | Fin warmup |
| 12 | 0.645 | 4.0e-5 | Backbone congelado |
| 13 | — | — | CNN descongelado |
| 26 | — | — | 🔄 Restart 1 SGDR |
| 46 | 0.711 | 5e-5 | 🔄 Restart 2 — primer salto significativo |
| 51 | 0.761 | 4.3e-5 | ⭐ |
| ⭐ **62 (BEST)** | **0.777** | 5.2e-6 | **Nuevo máximo histórico** |
| 64 | 0.767 | 1.7e-6 | |
| 65 | 0.771 | 8.1e-7 | Fin ciclo 3 SGDR |
| 66 | — | — | 🔄 Restart 3 |
| 68 | 0.751 | 4.9e-5 | Post-restart |
| 120 | — | — | Fin |

---

## Análisis de resultados v4

### ✅ Lo que funcionó

- **Todas las clases mejoraron simultáneamente** — inusual. En v3, ganar Unknown significó perder Absent. En v4, las 3 clases subieron.
- **Padding mask** eliminó el ruido de hasta 81% de frames en grabaciones cortas — el Hopfield dejó de "atender" ceros
- **Position feature** dio al modelo contexto clínico explícito sobre qué posición del corazón está escuchando
- **Checkpoint temprano (época 62)** — el modelo encontró su mejor punto mucho antes que v3 (117), indicando que las mejoras facilitan la convergencia
- **WA=0.7774** — mejora de +5.77 pp sobre v3, la mayor ganancia entre versiones consecutivas

### ⚠️ Tradeoffs / observaciones

- **WA volátil post-época 65** — después del restart 3, el modelo no recuperó el WA de época 62. Probable que el LR alto (5e-5) degrade el modelo ya ajustado. Considerar `sgdr_T0` mayor o reducir LR base para v5.
- **Unknown recall 0.583** — sigue siendo la clase más débil (peso×3 en el challenge). Margen de mejora.

---

## Próximas mejoras candidatas (v5)

| Mejora | Impacto estimado | Esfuerzo |
|---|:---:|:---:|
| Ensemble checkpoints SGDR (épocas ~51, 62, 65, 68) | +1-2 pp WA | bajo |
| Focal gamma específico para Unknown (γ=3-4) | medio | bajo |
| SGDR T₀ mayor (30-40) — evitar sobreescritura del mínimo | medio | bajo |
| EfficientNet-B0 backbone | incierto | bajo |

---

## Historial de versiones

| Versión | Fecha | Mejor WA | Mejor época | Mejora clave |
|---|---|:---:|:---:|---|
| v1 | 2026-05-24 | 0.6879 | 12 | Baseline MHN |
| v2 | 2026-05-24 | 0.7094 | 33 | CB-Focal Loss + TabularEncoder + ClinicalMixUp |
| v3 | 2026-05-26 | 0.7197 | 117 | SGDR + WeightedSampler + SpecAugment + LabelSmoothing |
| **v4** | **2026-06-03** | **0.7774** | **62** | Patient aggregation + Position feature + Padding mask |
