from .dift import DIFTModel
from .embedding_model import Dinov2EmbeddingModel, Dinov3EmbeddingModel
from .pca import PCA

MODEL_MAP = {
    "dinov2": Dinov2EmbeddingModel,
    "dinov3": Dinov3EmbeddingModel,
}

__all__ = [
    "DIFTModel",
    "Dinov2EmbeddingModel",
    "Dinov3EmbeddingModel",
    "PCA",
    "MODEL_MAP",
]
