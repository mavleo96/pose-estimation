from .dift import DIFT
from .embedding_model import Dinov2EmbeddingModel, Dinov3EmbeddingModel
from .pca import PCA

MODEL_MAP = {
    "dinov2": Dinov2EmbeddingModel,
    "dinov3": Dinov3EmbeddingModel,
}

__all__ = [
    "PCA",
    "Dinov2EmbeddingModel",
    "Dinov3EmbeddingModel",
    "MODEL_MAP",
    "DIFT",
]
