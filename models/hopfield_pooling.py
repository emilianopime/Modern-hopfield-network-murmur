"""
models/hopfield_pooling.py — FASE 3: Modern Hopfield Network (HopfieldPooling)

Recibe la secuencia temporal de embeddings de la CNN y la colapsa en un
vector representativo mediante memoria asociativa continua.

Input:  (B, T', embedding_dim)   — secuencia de frames de la CNN
Output: (B, quantity * output_size) — vector plano listo para el clasificador

Principio de funcionamiento:
    HopfieldPooling mantiene `quantity` patrones aprendibles (queries estáticos)
    que actúan como "prototipos clínicos" (ej: soplo sistólico, S1 normal).
    La regla de actualización:
        M(Q, K, V) = softmax(β · Q · Kᵀ) · V
    compara cada prototipo contra TODOS los frames del audio.
    Solo los frames con alta correlación exponencial con un prototipo almacenado
    contribuyen significativamente a la salida (ruido → correlación casi cero).

Por qué β = 1/√d:
    Con d = embedding_dim, esta inicialización es equivalente a la
    escala estándar de atención de Transformer (Vaswani et al., 2017).
    Evita saturación del softmax en dimensiones altas.
    Ajustar β durante entrenamiento es el hiperparámetro más crítico.
"""

import math

import torch
import torch.nn as nn

from config import HopfieldConfig

try:
    from hflayers import HopfieldPooling as _HopfieldPooling
    HOPFIELD_AVAILABLE = True
except ImportError:
    HOPFIELD_AVAILABLE = False


class HopfieldPoolingLayer(nn.Module):
    """
    Wrapper de HopfieldPooling con inicialización de β automática y
    fallback a atención multi-cabeza estándar si hopfield-layers no está instalado.

    Args:
        embedding_dim: Dimensión de los embeddings de entrada (salida de la CNN).
        config:        Instancia de HopfieldConfig.
    """

    def __init__(self, embedding_dim: int, config: HopfieldConfig):
        super().__init__()
        self.cfg = config
        self.embedding_dim = embedding_dim

        # β = 1/√d si no se especifica manualmente
        beta = config.beta if config.beta is not None else 1.0 / math.sqrt(embedding_dim)

        if HOPFIELD_AVAILABLE:
            self.pool = _HopfieldPooling(
                input_size=embedding_dim,
                hidden_size=config.hidden_size,
                output_size=config.output_size,
                num_heads=config.num_heads,
                quantity=config.quantity,
                scaling=beta,
                dropout=config.dropout,
                # Los patrones K y V son aprendibles (memoria asociativa dinámica)
                pattern_projection_as_static=False,
                normalize_stored_pattern=True,
                normalize_state_pattern=True,
            )
            self.output_dim = config.quantity * config.output_size
        else:
            # ── Fallback: Multi-head attention con queries aprendibles ────────
            # Comportamiento matemáticamente equivalente al MHN continuo.
            # Usar mientras se instala hopfield-layers.
            import warnings
            warnings.warn(
                "hopfield-layers no encontrado. Usando MultiheadAttention como fallback. "
                "Instalar con: pip install hopfield-layers"
            )
            self.pool = None
            self.learned_queries = nn.Parameter(
                torch.randn(config.quantity, embedding_dim) / math.sqrt(embedding_dim)
            )
            self.mha = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            # Escala β como temperatura (equivale a multiplicar Q por β antes de softmax)
            self.beta = nn.Parameter(torch.tensor(beta))
            self.output_dim = config.quantity * embedding_dim

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T', embedding_dim) — secuencia de embeddings de la CNN.

        Returns:
            (B, output_dim) — representación plana para el clasificador.
        """
        if HOPFIELD_AVAILABLE:
            # HopfieldPooling espera (B, T', input_size)
            # Devuelve (B, quantity, output_size)
            out = self.pool(x)                          # (B, quantity, output_size)
            # Extraer pesos de atención para interpretabilidad clínica
            # Shape: (B, num_heads, quantity, T')
            self.last_attn_weights = self.pool.get_association_matrix(x).detach()
        else:
            # Fallback: queries aprendibles × eje temporal via MHA
            B = x.shape[0]
            queries = self.learned_queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)
            # Escalar queries por β (temperatura de la memoria asociativa)
            queries = queries * self.beta
            out, self.last_attn_weights = self.mha(
                query=queries,
                key=x,
                value=x,
                need_weights=True,
                average_attn_weights=False,
            )                                           # (B, quantity, embedding_dim)

        # Aplanar (B, quantity, out_size) → (B, quantity * out_size)
        return out.reshape(out.shape[0], -1)

    @property
    def last_attention_weights(self) -> torch.Tensor | None:
        """
        Pesos de atención del último forward pass.
        Indica qué frames temporales activaron cada prototipo almacenado.
        Shape: (B, num_heads, quantity, T') — usar para interpretabilidad clínica.
        """
        if hasattr(self, "last_attn_weights"):
            return self.last_attn_weights
        return None
