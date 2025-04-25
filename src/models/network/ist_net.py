import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.utils.batch import BatchedData, gather
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ISTNet(pl.LightningModule):
    def __init__(
        self,
        model_name,
        backbone,
        regressor,
        depth_backbone,
        depth_fuser,
        max_batch_size,
        patch_size=14,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.patch_size = patch_size
        self.backbone = backbone
        self.regressor = regressor
        self.depth_backbone = depth_backbone
        self.depth_fuser = depth_fuser
        self.max_batch_size = max_batch_size
        self._init_weights()
        logger.info("Init for ISTNet done!")

    def get_toUpdate_parameters(self):
        return list(self.backbone.parameters()) + list(self.regressor.parameters()) + list(self.depth_backbone.parameters()) + list(self.depth_fuser.parameters())

    def _init_weights(self):
        """Init weights for the MLP"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        logger.info("Init weights for ISTNet done!")

    # @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs, processed_depths):
        batch_rgbs = BatchedData(batch_size=self.max_batch_size, data=processed_rgbs)
        batch_depths = BatchedData(batch_size=self.max_batch_size, data=processed_depths)
        patch_features = BatchedData(batch_size=self.max_batch_size)
        for idx_batch in range(len(batch_rgbs)):
            rgb_feats = self.backbone(batch_rgbs[idx_batch])
            depth_feats = self.depth_backbone(batch_depths[idx_batch])
            fused_feats = self.depth_fuser(depth_feats, rgb_feats)
            patch_features.cat(fused_feats)
        return patch_features.data

    def forward(self, src_img, tar_img, src_depth, tar_depth, src_pts, tar_pts):
        src_feat = self.forward_by_chunk(src_img, src_depth)
        tar_feat = self.forward_by_chunk(tar_img, tar_depth)

        src_feat_ = gather(src_feat, src_pts.clone())
        tar_feat_ = gather(tar_feat, tar_pts.clone())
        feats = torch.cat([tar_feat_, src_feat_], dim=1)

        scale = self.regressor.scale_predictor(feats)
        cos_sin_inplane = self.regressor.inplane_predictor(feats)
        if self.regressor.normalize_output:
            cos_sin_inplane = F.normalize(cos_sin_inplane, dim=1)
        return {
            "scale": scale.squeeze(1),
            "inplane": cos_sin_inplane,
        }

    def inference_by_chunk(
        self,
        src_feat,
        tar_feat,
        src_pts,
        tar_pts,
        max_batch_size,
    ):
        batch_src_feat = BatchedData(batch_size=max_batch_size, data=src_feat)
        batch_tar_feat = BatchedData(batch_size=max_batch_size, data=tar_feat)
        batch_src_pts = BatchedData(batch_size=max_batch_size, data=src_pts)
        batch_tar_pts = BatchedData(batch_size=max_batch_size, data=tar_pts)

        pred_scales = BatchedData(batch_size=max_batch_size)
        pred_cosSin_inplanes = BatchedData(batch_size=max_batch_size)

        for idx_sample in range(len(batch_src_feat)):
            pred_scales_, pred_cosSin_inplanes_ = self.inference(
                batch_src_feat[idx_sample],
                batch_tar_feat[idx_sample],
                batch_src_pts[idx_sample],
                batch_tar_pts[idx_sample],
            )
            pred_scales.cat(pred_scales_)
            pred_cosSin_inplanes.cat(pred_cosSin_inplanes_)
        return pred_scales.data, pred_cosSin_inplanes.data

    def inference(self, src_feat, tar_feat, src_pts, tar_pts):
        src_feat_ = gather(src_feat, src_pts.clone())
        tar_feat_ = gather(tar_feat, tar_pts.clone())
        feats = torch.cat([tar_feat_, src_feat_], dim=1)

        scale = self.regressor.scale_predictor(feats)
        cos_sin_inplane = self.regressor.inplane_predictor(feats)

        # same as forward but keep its structure and output angle
        B, N = src_pts.shape[:2]
        device = src_pts.device

        pred_scales = torch.full((B, N), -1000, dtype=src_feat.dtype, device=device)
        pred_cosSin_inplanes = torch.full(
            (B, N, 2), -1000, dtype=src_feat.dtype, device=device
        )

        src_mask = torch.logical_and(src_pts[:, :, 0] != -1, src_pts[:, :, 1] != -1)
        tar_mask = torch.logical_and(tar_pts[:, :, 0] != -1, tar_pts[:, :, 1] != -1)
        assert torch.sum(src_mask) == torch.sum(tar_mask)

        pred_scales[src_mask] = scale.squeeze(1)
        pred_cosSin_inplanes[src_mask] = cos_sin_inplane
        return pred_scales, pred_cosSin_inplanes


class DepthFeatureExtractor(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # B, 32, 224, 224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # B, 64, 224, 224
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # B, 128, 224, 224
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),  # B, 256, 224, 224
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder(x)  # B, 256, 224, 224
        x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=False)  # → B, 256, 16, 16
        return x

class DepthFuser(nn.Module):
    # concat depth and rgb features and fuse them from 512 channels to 256 channels
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

    def forward(self, depth_feat, rgb_feat):
        x = torch.cat([depth_feat, rgb_feat], dim=1)
        x = self.conv1(x)         # 16x16x512 -> 16x16x256
        return x

class Regressor(nn.Module):
    """
    A simple MLP to regress scale and rotation from DINOv2 features
    """

    def __init__(
        self,
        descriptor_size,
        hidden_dim,
        use_tanh_act,
        normalize_output,
    ):
        super(Regressor, self).__init__()
        self.descriptor_size = descriptor_size
        self.normalize_output = normalize_output
        self.use_tanh_act = use_tanh_act

        self.scale_predictor = nn.Sequential(
            nn.Linear(descriptor_size * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.inplane_predictor = nn.Sequential(
            nn.Linear(descriptor_size * 2, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Tanh() if self.use_tanh_act else nn.Identity(),
        )
        self._reset_parameters()
        logger.info(f"Init for Regressor with done!")

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
    from hydra.experimental import compose, initialize

    with initialize(config_path="../../../configs/"):
        cfg = compose(config_name="train.yaml")

    model = instantiate(cfg.model.ist_net)
    model = model.to("cuda")

    images = torch.rand(2, 3, 224, 224).to(device="cuda")
    depths = torch.rand(2, 1, 224, 224).to(device="cuda")
    keypoints = torch.rand(2, 256, 2).to(device="cuda")
    keypoints[keypoints < 0.5] = -1
    keypoints = keypoints.long()
    pred = model(images, images, depths, depths, keypoints, keypoints)
    print(pred)
