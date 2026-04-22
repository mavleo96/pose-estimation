from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel

_MODEL_DIR = Path(__file__).parent


class ControlNetDIFT(nn.Module):
    """
    Depth-conditioned Image-to-Feature Transformer (DIFT).
    """

    controlnet_id = "lllyasviel/sd-controlnet-depth"
    unet_id = "runwayml/stable-diffusion-v1-5"

    # stride = 8
    feat_size = (32, 32)

    def __init__(self, up_block_indices: tuple[int, ...], timestep: int):
        super().__init__()

        self.up_block_indices = up_block_indices

        self.scheduler = DDPMScheduler.from_pretrained(self.unet_id, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(self.unet_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.unet_id, subfolder="unet")
        self.controlnet = ControlNetModel.from_pretrained(self.controlnet_id)

        self.register_buffer("null_emb", torch.load(_MODEL_DIR / "null_emb.pt", weights_only=True))

        # DIFT default: t=261 for semantic tasks
        self.register_buffer("timestep", torch.tensor(timestep))

        self._feature_cache = {}
        self._register_hooks()

    @property
    def device(self):
        return self.vae.device

    @property
    def dtype(self):
        return self.vae.dtype

    @property
    def dim(self):
        num_dims = 0
        if 0 in self.up_block_indices:
            num_dims += 1280
        if 1 in self.up_block_indices:
            num_dims += 1280
        if 2 in self.up_block_indices:
            num_dims += 640
        if 3 in self.up_block_indices:
            num_dims += 320
        return num_dims

    def enable_xformers(self):
        self.unet.enable_xformers_memory_efficient_attention()
        self.controlnet.enable_xformers_memory_efficient_attention()

    def _register_hooks(self):
        for block_idx in self.up_block_indices:
            tag = f"up_block_{block_idx}"

            def make_hook(key):
                def hook(module, input, output):
                    # Note: ResnetBlock2D returns a tensor, CrossAttnUpBlock2D returns a tuple
                    out = output[0] if isinstance(output, tuple) else output
                    self._feature_cache[key] = out.detach().clone()

                return hook

            self.unet.up_blocks[block_idx].resnets[-1].register_forward_hook(make_hook(tag))

    def _to_noisy_latent(self, rgb, seed):
        # Note: we use the mean of the latent distribution for deterministic output
        # and avoid stochastic sampling across ensemble seeds
        latent = self.vae.encode(rgb).latent_dist.mean * self.vae.config.scaling_factor
        # latent = self.vae.encode(rgb).latent_dist.sample()

        # Note: we use a fixed seed per ensemble member for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device=self.device, dtype=self.dtype)
        return self.scheduler.add_noise(latent, noise, self.timestep)

    def forward(self, rgb, depth_cond, ensemble_size=1, skip_controlnet=False):
        assert rgb.ndim == 4 and rgb.shape[1] == 3
        assert rgb.shape == depth_cond.shape
        B, _, H, W = rgb.shape

        null_emb = self.null_emb.repeat(B, 1, 1)
        feature_agg = None

        for seed in range(ensemble_size):
            self._feature_cache.clear()
            noisy_latent = self._to_noisy_latent(rgb, seed=seed)

            # ControlNet produces residuals that steer the U-Net toward the depth geometry
            if not skip_controlnet:
                down_block_res, mid_block_res = self.controlnet(
                    noisy_latent,
                    self.timestep,
                    encoder_hidden_states=null_emb,
                    controlnet_cond=depth_cond,
                    conditioning_scale=1.0,
                    return_dict=False,
                )
            else:
                down_block_res = None
                mid_block_res = None

            # U-Net forward; hooks populate _feature_cache as side effect
            self.unet(
                noisy_latent,
                self.timestep,
                encoder_hidden_states=null_emb,
                down_block_additional_residuals=down_block_res,
                mid_block_additional_residual=mid_block_res,
            )

            # Note: We upsample each block's features to common resolution and concatenate
            # Block 0 -> 1280 channels, 8x8    (for 512x512 input)
            # Block 1 -> 1280 channels, 16x16
            # Block 2 -> 640 channels,  32x32
            # Block 3 -> 320 channels,  64x64
            block_features = [
                F.interpolate(
                    self._feature_cache[f"up_block_{block_idx}"],
                    size=self.feat_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for block_idx in self.up_block_indices
            ]
            features = torch.cat(block_features, dim=1)
            feature_agg = features if feature_agg is None else feature_agg + features

            # explicitly free everything
            del noisy_latent, block_features, features
            if not skip_controlnet:
                del down_block_res, mid_block_res
            self._feature_cache.clear()
            torch.cuda.empty_cache()

        return feature_agg / ensemble_size
