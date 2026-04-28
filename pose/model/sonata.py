# import torch
# import torch.nn as nn

# import sonata


# class SonataEncoderModel(nn.Module):
#     """
#     Sonata Encoder Model
#     """

#     model_id = "facebook/sonata"

#     def __init__(self, enc_patch_size):
#         super().__init__()

#         self.enc_patch_size = enc_patch_size

#         custom_config = dict(
#             enc_patch_size=[enc_patch_size for _ in range(5)],
#             enable_flash=True,
#             freeze_encoder=True,
#             shuffle_orders=False,
#         )
#         self.model = sonata.model.load(
#             "sonata", repo_id=self.model_id, custom_config=custom_config
#         )
#         for module in self.model.modules():
#             if hasattr(module, "shuffle_orders"):
#                 module.shuffle_orders = False

#     @property
#     def device(self):
#         return next(self.model.parameters()).device

#     @property
#     def dtype(self):
#         return next(self.model.parameters()).dtype

#     @property
#     def dim(self):
#         return 1088

#     def forward(self, point):
#         B = point["offset"].shape[0]
#         assert B == 1

#         out = self.model(point)

#         for _ in range(2):
#             parent = out.pop("pooling_parent")
#             inverse = out.pop("pooling_inverse")
#             parent.feat = torch.cat([parent.feat, out.feat[inverse]], dim=-1)
#             out = parent
#         while "pooling_parent" in out.keys():
#             parent = out.pop("pooling_parent")
#             inverse = out.pop("pooling_inverse")
#             parent.feat = out.feat[inverse]
#             out = parent

#         return out.feat[out.inverse].reshape(1, -1, self.dim)

#         # # We map the local inverse to the global inverse using the offset
#         # offset = out.offset
#         # vox_starts = torch.tensor([0, *offset.tolist()[:-1]], device=offset.device)
#         # global_inv = (out.inverse.reshape(B, -1) + vox_starts[:, None]).reshape(-1)

#         # return out.feat[global_inv].reshape(B, -1, self.dim)
