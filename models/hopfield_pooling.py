import math
import warnings

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
    Memoria asociativa continua sobre la secuencia temporal de embeddings CNN.
    Entrada:  (B, T', embedding_dim)
    Salida:   (B, quantity * output_size)

    Usa hflayers.HopfieldPooling si está instalado; cae a MultiheadAttention en caso contrario.
    Los pesos de atención quedan disponibles en `last_attention_weights`.
    """

    def __init__(self, embedding_dim: int, config: HopfieldConfig):
        super().__init__()
        self.cfg = config
        self.embedding_dim = embedding_dim
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
                pattern_projection_as_static=False,
                normalize_stored_pattern=True,
                normalize_state_pattern=True,
            )
            self.output_dim = config.quantity * config.output_size
        else:
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
            self.beta = nn.Parameter(torch.tensor(beta))
            self.output_dim = config.quantity * embedding_dim

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HOPFIELD_AVAILABLE:
            out = self.pool(x)
            self.last_attn_weights = self.pool.get_association_matrix(x).detach()
        else:
            B = x.shape[0]
            queries = self.learned_queries.unsqueeze(0).expand(B, -1, -1) * self.beta
            out, self.last_attn_weights = self.mha(
                query=queries, key=x, value=x,
                need_weights=True, average_attn_weights=False,
            )

        return out.reshape(out.shape[0], -1)

    @property
    def last_attention_weights(self) -> torch.Tensor | None:
        return getattr(self, "last_attn_weights", None)
