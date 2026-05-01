from .dift import DIFTModel, DIFTModelV2
from .embedding_model import Dinov2EmbeddingModel, Dinov3EmbeddingModel
from .pca import PCA

MODEL_MAP = {
    "dinov2": Dinov2EmbeddingModel,
    "dinov3": Dinov3EmbeddingModel,
}

__all__ = [
    "DIFTModel",
    "DIFTModelV2",
    "Dinov2EmbeddingModel",
    "Dinov3EmbeddingModel",
    "PCA",
    "MODEL_MAP",
]
