import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

from ..utils.image import make_pixel_grid


class PatchCorrespondenceFinder(nn.Module):
    """
    Compute valid patch correspondences between query and template feature embeddings.

    Correspondences computed by cosine similarity between feature embeddings.
    Filtered using foreground masks, similarity threshold and cycle consistency mask with patch
    threshold.

    Args:
        sim_threshold (float): Similarity threshold for valid correspondences.
        patch_threshold (float | None): Patch threshold for valid correspondences.
            If None, no cycle consistency is applied.
            Default: None.
    """

    def __init__(self, sim_threshold=0.5, patch_threshold=None):
        super().__init__()
        assert 0.0 <= sim_threshold <= 1.0, "sim_threshold must be in [0.0, 1.0]"
        assert patch_threshold is None or patch_threshold >= 0, (
            "patch_threshold must be None or >= 0"
        )

        self.sim_threshold = sim_threshold
        self.patch_threshold = patch_threshold

    def __repr__(self):
        return (
            f"PatchCorrespondenceFinder(sim_threshold={self.sim_threshold},"
            f" patch_threshold={self.patch_threshold})"
        )

    def forward(self, tar_feat, src_feats, tar_mask, src_mask, feat_size):
        """
        Compute valid patch correspondences between query and template feature embeddings.

        Args:
            tar_feat  : (B, S1, D)    — query feature embeddings
            src_feats : (B, N, S2, D) — template feature embeddings, N views per item.
                        Pass (B, 1, S2, D) for single-view (val) mode.
            tar_mask  : (B, S1)       — query foreground mask
            src_mask  : (B, N, S2)    — template foreground mask
            feat_size : (H, W)        — feature map size

        Returns:
            dict:
                idx_tar2src : (B, N, S1) — forward NN: query patch -> template patch
                score_tar2src : (B, N, S1) — forward NN: query patch -> template patch
                mask : (B, N, S1) bool
        """
        # Compute cosine similarity
        tar_norm = F.normalize(tar_feat, dim=-1)  # (B, S1, D)
        src_norm = F.normalize(src_feats, dim=-1)  # (B, N, S2, D)
        sim = einsum(tar_norm, src_norm, "b s1 d, b n s2 d -> b n s1 s2")

        # Apply foreground masks and sim threshold
        sim = sim * tar_mask[:, None, :, None]
        sim = sim * src_mask[:, :, None, :]
        sim = sim.masked_fill(sim < self.sim_threshold, 0.0)

        # Find the best template/query patch correspondence
        score_tar2src, idx_tar2src = sim.max(dim=-1)  # (B, N, S1)
        _, idx_src2tar = sim.max(dim=-2)  # (B, N, S2)

        # Validity masks
        mask_sim = score_tar2src >= self.sim_threshold
        mask_cycle = self._cycle_mask(idx_tar2src, idx_src2tar, feat_size)

        # Combine masks and update scores and indices
        mask = mask_sim & mask_cycle
        score_tar2src = score_tar2src * mask.float()
        idx_tar2src = idx_tar2src * mask.long()

        return {
            "idx_tar2src": idx_tar2src,
            "score_tar2src": score_tar2src,
            "mask": mask,
        }

    def _cycle_mask(self, idx_tar2src, idx_src2tar, feat_size):
        """
        Checks if tar -> src -> tar is within patch_threshold.

        Args:
            idx_tar2src : (B, N, S1) — forward NN: query patch -> template patch
            idx_src2tar : (B, N, S2) — backward NN: template patch -> query patch
            feat_size : (H, W) — feature map size

        Returns:
            mask : (B, N, S1) bool
        """
        if self.patch_threshold is None:
            return torch.ones_like(idx_tar2src, dtype=torch.bool)

        B, N, _ = idx_tar2src.shape

        # patch grid locations — (S1, 2) in (x, y) patch coordinates
        patch_locs = make_pixel_grid(*feat_size, device=idx_tar2src.device).reshape(-1, 2)

        # forward: query patch q -> template patch s = idx_tar2src[q]
        # backward: s -> q' = idx_src2tar[s]
        # gather q' for each q
        q_back = idx_src2tar.gather(dim=2, index=idx_tar2src)  # (B, N, S1)

        # 2D locations of q and q'
        loc_q = patch_locs[None, None, :, :].expand(B, N, -1, -1)  # (B, N, S1, 2)
        loc_q_back = patch_locs[q_back]  # (B, N, S1, 2)

        dist = torch.norm((loc_q_back - loc_q).float(), dim=-1)  # (B, N, S1)
        return dist <= self.patch_threshold


__all__ = ["PatchCorrespondenceFinder"]

if __name__ == "__main__":
    finder = PatchCorrespondenceFinder(sim_threshold=0.5, patch_threshold=1.0)
    print(finder)

    feat = torch.randn(2, 7, 7, 128)

    tar_feat = feat[:, :5, :5, :].reshape(2, 25, 128)
    tar_feat[0, [13, 18], :] = tar_feat[0, [18, 13], :]
    tar_feat[1, [14, 22], :] = tar_feat[1, [22, 14], :]
    tar_mask = torch.ones(2, 25).bool()

    src_feat = feat[:, 2:, 2:, :].reshape(2, 25, 128)
    src_feat = torch.stack(
        [
            torch.stack([torch.randn(25, 128), torch.randn(25, 128), src_feat[0]], dim=0),
            torch.stack([torch.randn(25, 128), src_feat[1], torch.randn(25, 128)], dim=0),
        ]
    )
    src_mask = torch.ones(2, 3, 25).bool()

    feat_size = (5, 5)

    matches = finder(tar_feat, src_feat, tar_mask, src_mask, feat_size)

    # top template indices in each batch item
    print("best template indices:")
    best_template_indices = matches["score_tar2src"].sum(dim=2).argmax(dim=1)
    print(best_template_indices)

    # top match indices in best template for each batch item
    print("\nbest match indices:")
    best_match_indices = matches["idx_tar2src"][
        torch.arange(2)[:, None], best_template_indices[:, None]
    ]
    print("best match for batch item 0:")
    print(best_match_indices[0, 0].reshape(5, 5))
    print("best match for batch item 1:")
    print(best_match_indices[1, 0].reshape(5, 5))
