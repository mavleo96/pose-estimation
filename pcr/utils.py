import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import Dinov2WithRegistersModel, DINOv3ViTModel

# Global model cache to avoid reloading models
_model_cache = {}


def get_dino_model(model_name="dinov2", device="cuda"):
    """Get DINO model with caching to avoid reloading in worker processes."""
    cache_key = f"{model_name}_{device}"

    if cache_key not in _model_cache:
        # Check if we're in a worker process (spawn method)
        if os.getpid() != os.getppid():
            # In worker process, load model fresh
            model = {
                "dinov2": Dinov2WithRegistersModel.from_pretrained(
                    "facebook/dinov2-with-registers-large"
                ),
                "dinov3": DINOv3ViTModel.from_pretrained(
                    "facebook/dinov3-vitl16-pretrain-lvd1689m"
                ),
            }[model_name]
            model.eval()
            model.to(device)
            _model_cache[cache_key] = model
        else:
            # In main process, load and cache
            model = {
                "dinov2": Dinov2WithRegistersModel.from_pretrained(
                    "facebook/dinov2-with-registers-large"
                ),
                "dinov3": DINOv3ViTModel.from_pretrained(
                    "facebook/dinov3-vitl16-pretrain-lvd1689m"
                ),
            }[model_name]
            model.eval()
            model.to(device)
            _model_cache[cache_key] = model

    return _model_cache[cache_key]


def dino_embeddings(img, batch_size=20, device="cuda", model_name="dinov2"):
    assert img.ndim == 4
    assert img.shape[1:] == (3, 224, 224)

    model = get_dino_model(model_name, device)

    loader = DataLoader(TensorDataset(img), batch_size=batch_size, shuffle=False)

    embeddings_list = []
    with torch.no_grad():
        for (batch_img,) in tqdm(loader, desc=f"computing {model_name} embeddings"):
            batch_img = batch_img.to(device)
            batch_emb = model(batch_img)
            embeddings_list.append(batch_emb.last_hidden_state.cpu())

    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings


def embedding_interpolation(
    last_hidden_state, pixel_coords, image_size=(224, 224), model_name="dinov2"
):
    patch_size = {"dinov2": 14, "dinov3": 16}[model_name]
    b, p, d = last_hidden_state.shape
    h_img, w_img = image_size
    h_feat, w_feat = h_img // patch_size, w_img // patch_size
    n = pixel_coords.shape[0]

    assert b == 1
    assert p == h_feat * w_feat + 5

    device = pixel_coords.device

    embeddings = (
        last_hidden_state[:, 5:, :].reshape(b, h_feat, w_feat, d).permute(0, 3, 1, 2)
    )

    norm = torch.tensor([[w_img - 1, h_img - 1]], dtype=torch.float32, device=device)
    norm_coords = pixel_coords / norm * 2 - 1
    grid = norm_coords.view(1, n, 1, 2)

    sampled = F.grid_sample(
        embeddings, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    return sampled.squeeze(-1).permute(0, 2, 1)


def transform_from_Rt(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    single = False
    if R.ndim == 2:
        assert R.shape == (3, 3)
        assert t.shape == (3,)
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
        single = True
    else:
        assert R.ndim == 3 and R.shape[-2:] == (3, 3)
        assert t.ndim == 2 and t.shape[-1] == 3
        assert R.shape[0] == t.shape[0]

    N = R.shape[0]
    device, dtype = R.device, R.dtype

    T = torch.zeros((N, 4, 4), device=device, dtype=dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0

    return T[0] if single else T