import torch.nn as nn
from transformers import Dinov2WithRegistersModel, DINOv3ViTModel


class Dinov2EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Dinov2WithRegistersModel.from_pretrained(
            "facebook/dinov2-with-registers-large"
        )
        self.dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_registers = 4

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    def forward(self, x):
        _, _, H, W = x.shape
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0

        output = self.model(x)

        # Note: first 5 tokens are for the CLS + registers hence we skip them
        # return {
        #     "pooler_output": output.pooler_output,
        #     "patch_embeddings": output.last_hidden_state[:, 1 + self.num_registers :, :],
        # }
        return output.last_hidden_state[:, 1 + self.num_registers :, :]


class Dinov3EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DINOv3ViTModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
        self.dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.num_registers = self.model.config.num_register_tokens

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    def forward(self, x):
        _, _, H, W = x.shape
        assert H % self.patch_size == 0
        assert W % self.patch_size == 0

        output = self.model(x)

        # Note: first 5 tokens are for the CLS + registers hence we skip them
        # return {
        #     "pooler_output": output.pooler_output,
        #     "patch_embeddings": output.last_hidden_state[:, 1 + self.num_registers :, :],
        # }
        return output.last_hidden_state[:, 1 + self.num_registers :, :]
