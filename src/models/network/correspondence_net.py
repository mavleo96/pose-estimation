import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import logging
from src.utils.logging import get_logger
from torch_geometric.nn import GCNConv
from einops import rearrange
from src.utils.edges import compute_edges_batched

logger = get_logger(__name__)

class CorrespondenceNet(pl.LightningModule):
    def __init__(
            self, 
            model_name, 
            graph_encoder,
            descriptor_size=1024 + 6,
            **kwargs):
        super().__init__()
        self.model_name = model_name
        # TODO: make this configurable
        self.num_patches = 16
        self.descriptor_size = descriptor_size
        self.graph_encoder = graph_encoder

        logger.info("Initialize CorrespondenceNet done!")


    # def process_inputs(self, src_feats, tar_feat, src_masks, tar_mask):
    #     feat_size = (self.num_patches, self.num_patches)

    #     tar_mask = F.interpolate(tar_mask.unsqueeze(1), size=feat_size)
    #     tar_mask = rearrange(tar_mask, "b 1 h w -> b 1 (h w)")
    #     tar_feat = F.normalize(tar_feat, dim=1)
    #     tar_feat = rearrange(tar_feat, "b c h w -> b 1 c (h w)")

    #     src_masks = F.interpolate(src_masks, size=feat_size)
    #     src_masks = rearrange(src_masks, "b n h w -> b n (h w)")
    #     src_feats = F.normalize(src_feats, dim=2)
    #     src_feats = rearrange(src_feats, "b n c h w -> b n c (h w)")

    #     return src_feats, tar_feat, src_masks, tar_mask

    # def forward_by_chunk(self, src_feats, tar_feat, src_masks, tar_mask):
    #     src_feats = self.graph_encoder(src_feats, src_masks)
    #     tar_feat = self.graph_encoder(tar_feat, tar_mask)
    #     return src_feats, tar_feat


    # def forward(self, src_feats, tar_feat, src_masks, tar_mask):
    #     src_feats, tar_feat, src_masks, tar_mask = self.process_inputs(src_feats, tar_feat, src_masks, tar_mask)
    #     src_feats, tar_feat = self.forward_by_chunk(src_feats, tar_feat, src_masks, tar_mask)
    #     return src_feats, tar_feat

    def process_inputs(self, feat, mask, expand):
        feat_size = (self.num_patches, self.num_patches)

        if expand: # i.e mask.ndim == 3
            mask = F.interpolate(mask.unsqueeze(1), size=feat_size)
            mask = rearrange(mask, "b 1 h w -> b 1 (h w)")
        else: # i.e mask.ndim == 4
            mask = F.interpolate(mask, size=feat_size)
            mask = rearrange(mask, "b n h w -> b n (h w)")

        if expand: # i.e feat.ndim == 4
            feat = F.normalize(feat, dim=1)
            feat = rearrange(feat, "b c h w -> b 1 c (h w)")
        else: # feat.ndim == 5
            feat = F.normalize(feat, dim=2)
            feat = rearrange(feat, "b n c h w -> b n c (h w)")

        return feat, mask
        
    def forward_by_chunk(self, feat, mask):
        feat = self.graph_encoder(feat, mask)
        return feat

    def forward(self, tar_feat, tar_mask, src_encodings):
        expand = True if tar_feat.ndim == 4 else False
        tar_feat, tar_mask = self.process_inputs(tar_feat, tar_mask, expand)
        tar_feat = self.forward_by_chunk(tar_feat, tar_mask)
        tar_feat = rearrange(tar_feat, "b n c (h w) -> b n c h w", h=self.num_patches, w=self.num_patches)
        if expand:
            tar_feat = tar_feat.squeeze(1)
        return tar_feat

class DepthFeatureExtractor(nn.Module):
    def __init__(self, model_name, out_channels=6):
        super().__init__()
        self.model_name = model_name
        self.encoder = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, padding=1),  # B, 6, 224, 224
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)  # B, 6, 224, 224
        x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)  # → B, 6, 16, 16
        return x


class GraphEncoder(nn.Module):
    # TODO: graph encoder is currently only accepting 4D inputs (B, G, C, N)
    def __init__(self, in_channels=1030, hidden_channels=1024, out_channels=1024):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # TODO: make this configurable
        self.num_patches = 16

        logger.info("Initialize GraphEncoder done!")

    def prepare_batch(self, x, edge_list):
        B, G, C, N = x.shape # N: H * W
        total_edges = sum([edge.shape[1] for edge in edge_list])

        x_flat = rearrange(x, "b g c n -> (b g n) c")
        batch = torch.arange(B * G * N, device=x.device)

        edge_index_list = []
        for i in range(B):
            for j in range(G):
                offset = (i * G + j) * N
                edge_index = edge_list[i * G + j]
                edge_index_list.append(edge_index + offset)
        edge_index_flat = torch.cat(edge_index_list, dim=1).to(x.device)

        return x_flat, edge_index_flat, batch, B, G, N, C

    def forward(self, x, mask):
        edge_list = compute_edges_batched(mask, self.num_patches, self.num_patches)
        x, edge_index, batch, B, G, N, C = self.prepare_batch(x, edge_list)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        x = rearrange(x, "(b g n) c -> b g c n", b=B, g=G, n=N)
        return x


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import numpy as np
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
    from PIL import Image
    from hydra.experimental import compose, initialize

    with initialize(config_path="../../../configs/"):
        cfg = compose(config_name="train.yaml")


    dff = instantiate(cfg.model.depth_feature_extractor).to('cuda')
    # dff = DepthFeatureExtractor().to('cuda')
    dff_input = torch.randn(4, 1, 224, 224).to('cuda')
    dff_output = dff(dff_input)
    print(f"input shapes: {dff_input.shape}")
    print(f"output shapes: {dff_output.shape}")

    # fuse with DINOv2 features
    dino_feats = torch.randn(4, 1024, 16, 16).to('cuda')
    fused_feats = torch.cat([dff_output, dino_feats], dim=1)
    print(f"input shapes: {fused_feats.shape}")
    print(f"output shapes: {fused_feats.shape}")

    # pass through the correspondence net
    correspondence_net = instantiate(cfg.model.correspondence_net).to('cuda')
    masks = torch.tensor(np.random.randint(0, 2, (4, 224, 224)), dtype=torch.float).to('cuda')
    encodings = correspondence_net(fused_feats, masks, None)
    print(f"input shapes: {fused_feats.shape}")
    print(f"output shapes: {encodings.shape}")
