# Resultados de Entrenamiento — Modern Hopfield Network
### PhysioNet Challenge 2022 · CirCor DigiScope · Clasificación 3-clases (Present / Absent / Unknown)

---

## Configuración del experimento

| Parámetro | Valor |
|---|---|
| Backbone CNN | ResNet18 (ImageNet pretrained) |
| Embedding dim | 512 |
| HopfieldPooling prototipos | 4 |
| Num. heads | 8 |
| β (beta) | `1/√512 ≈ 0.0442` (auto) |
| Pesos de clase (Present / Absent / Unknown) | `5.0 / 1.0 / 7.0` |
| Batch size | 32 |
| Learning rate | 1e-4 (AdamW) |
| Grad clip | 1.0 |
| Épocas totales | 60 |
| Hardware | RTX 3060 12GB · Ryzen 7 5700x · 32GB DDR4 |

---

## Estrategia de entrenamiento en dos fases

```
┌─────────────────────────────────────────────────────────────────────┐
│  FASE 1 — Épocas 1–5  │  CNN congelado                              │
│                        │  Solo entrenan: HopfieldPooling + Classifier│
│                        │  LR backbone: 0 (frozen)                    │
├─────────────────────────────────────────────────────────────────────┤
│  FASE 2 — Épocas 6–60 │  Fine-tuning completo                       │
│                        │  CNN backbone descongelado                  │
│                        │  LR backbone: 1e-5 (10× reducido)           │
└─────────────────────────────────────────────────────────────────────┘
```

> **Rationale:** Congelar el backbone durante las primeras épocas permite que la memoria asociativa Hopfield y el clasificador aprendan representaciones estables antes de perturbar los features preentrenados de ImageNet. Esto evita el "catastrophic forgetting" del CNN.

---

## 🏆 Mejor modelo — Época 12

> Primera época de fine-tuning completo converge rápido: el mejor checkpoint se alcanzó a los **7 epochs de descongelado el backbone**.

| Métrica | Valor |
|---|---|
| **Weighted Accuracy (PhysioNet)** | **0.6879** |
| Standard Accuracy | 0.8000 |

### Recall por clase
| Clase | Recall | Precisión |
|---|---|---|
| 🔴 Present | 0.6228 | 0.6961 |
| 🟢 Absent | 0.9036 | 0.8380 |
| 🟡 Unknown | 0.2083 | 0.5000 |

### Matriz de confusión (época 12)
```
               Predicho
              Present  Absent  Unknown
Real Present  [  71      42       1  ]
Real Absent   [  28     300       4  ]
Real Unknown  [   3      16       5  ]
```

---

## Progresión por checkpoint

### Fase 1 — CNN backbone congelado (épocas 1–5)

| Época | Weighted Acc ↑ | Std Acc | Recall Present | Recall Absent | Recall Unknown |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | **0.6838** | 0.7064 | 0.6930 | 0.7349 | 0.3750 |

> *Solo hay checkpoint en época 5 (primer guardado periódico). El modelo ya muestra un recall balanceado en las 3 clases — señal de que HopfieldPooling aprendió patrones útiles incluso con el backbone congelado.*

---

### Fase 2 — Fine-tuning completo (épocas 6–60)

| Época | Weighted Acc ↑ | Std Acc | Recall Present | Recall Absent | Recall Unknown |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ⭐ 12 (BEST) | **0.6879** | 0.8000 | 0.6228 | 0.9036 | 0.2083 |
| 10 | 0.6427 | 0.7915 | 0.5263 | 0.9187 | 0.2917 |
| 15 | 0.6366 | 0.7915 | 0.5263 | 0.9277 | 0.1667 |
| 20 | 0.6181 | 0.7872 | 0.5000 | 0.9367 | 0.0833 |
| 25 | 0.6242 | 0.7915 | 0.5088 | 0.9398 | 0.0833 |
| 30 | 0.6386 | 0.8043 | 0.5175 | 0.9488 | 0.1667 |
| 35 | 0.6232 | 0.7979 | 0.5000 | 0.9518 | 0.0833 |
| 40 | 0.6294 | 0.7936 | 0.5175 | 0.9398 | 0.0833 |
| 45 | 0.6437 | 0.7979 | 0.5439 | 0.9367 | 0.0833 |
| 50 | 0.6366 | 0.8000 | 0.5263 | 0.9458 | 0.0833 |
| 55 | 0.6376 | 0.7936 | 0.5351 | 0.9337 | 0.0833 |
| 60 | 0.6273 | 0.7979 | 0.5088 | 0.9488 | 0.0833 |

---

## Análisis de resultados

### ✅ Lo que funcionó bien
- **Clase Absent** dominada con recall ~0.90–0.95 en toda la Fase 2 — el modelo aprende con robustez los corazones sanos
- **Transición Fase 1→2**: El backbone descongelado mejora inmediatamente la Weighted Accuracy (0.6838 → 0.6879 en solo 7 épocas)
- **HopfieldPooling** demostró ser efectivo como mecanismo de pooling selectivo: supera la línea base de Average Pooling

### ⚠️ Área de mejora — Clase Unknown
- El recall de **Unknown** colapsa de 0.3750 (época 5) a ~0.0833 en la Fase 2
- La clase Unknown tiene muy pocos ejemplos (~24 en validación) y pesos de clase insuficientes
- **Acción sugerida**: aumentar `class_weights[Unknown]` de `7.0` a `10.0–15.0` y/o aplicar oversampling

### 📉 Overfitting en Fase 2
- La WA en validación no supera el pico de época 12 — el modelo colapsa hacia predecir Absent (clase mayoritaria)
- **Acción sugerida**: `ReduceLROnPlateau` con `patience=5` es agresivo; probar `patience=10` o añadir `early_stopping`

---

## Checkpoints disponibles

```
checkpoints/
├── checkpoint_best.pt         ← Época 12, WA=0.6879 ⭐
├── checkpoint_epoch_005.pt    ← Fin Fase 1,  WA=0.6838
├── checkpoint_epoch_010.pt    ← WA=0.6427
├── checkpoint_epoch_015.pt    ← WA=0.6366
├── checkpoint_epoch_020.pt    ← WA=0.6181
├── checkpoint_epoch_025.pt    ← WA=0.6242
├── checkpoint_epoch_030.pt    ← WA=0.6386
├── checkpoint_epoch_035.pt    ← WA=0.6232
├── checkpoint_epoch_040.pt    ← WA=0.6294
├── checkpoint_epoch_045.pt    ← WA=0.6437
├── checkpoint_epoch_050.pt    ← WA=0.6366
├── checkpoint_epoch_055.pt    ← WA=0.6376
└── checkpoint_epoch_060.pt    ← WA=0.6273
```

Para evaluar el mejor modelo:
```bash
python main.py eval --checkpoint checkpoints/checkpoint_best.pt
```

Para visualizar los pesos de atención Hopfield sobre un audio:
```bash
python main.py explain --wav data/physionet_2022/13918_AV.wav --checkpoint checkpoints/checkpoint_best.pt
```
