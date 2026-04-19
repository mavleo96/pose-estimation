import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


class PointCorrespondenceFinder(nn.Module):
    """
    Compute valid correspondences between two sets of point embeddings via cosine similarity.

    Correspondences computed by cosine similarity between feature embeddings.

    Args:
        sim_threshold (float): Minimum cosine similarity for a valid correspondence.
            Default: 0.0
    """

    def __init__(self, sim_threshold=0.0):
        super().__init__()
        assert 0.0 <= sim_threshold <= 1.0, "sim_threshold must be in [0.0, 1.0]"
        self.sim_threshold = sim_threshold

    def __repr__(self):
        return f"PointCorrespondenceFinder(sim_threshold={self.sim_threshold})"

    def forward(self, tar_feat, src_feat, tar_mask=None, src_mask=None):
        """
        Compute valid correspondences between two sets of point embeddings via cosine similarity.

        Args:
            tar_feat : (B, T, D) — query point embeddings
            src_feat : (B, S, D) — source point embeddings
            tar_mask : (B, T)    — valid query points, or None
            src_mask : (B, S)    — valid source points, or None

        Returns:
            dict:
                idx_tar2src   : (B, T)    — forward NN: query point -> source point
                score_tar2src : (B, T)    — cosine similarity of that match
        """
        # Compute cosine similarity
        tar_norm = F.normalize(tar_feat, dim=-1)  # (B, T, D)
        src_norm = F.normalize(src_feat, dim=-1)  # (B, S, D)
        sim = einsum(tar_norm, src_norm, "b t d, b s d -> b t s")

        # Apply foreground masks
        if tar_mask is not None:
            sim = sim * tar_mask[:, :, None].float()
        if src_mask is not None:
            sim = sim * src_mask[:, None, :].float()

        # Find the best source/query point correspondence
        score_tar2src, idx_tar2src = sim.max(dim=-1)  # (B, T)

        # Validity masks
        mask = score_tar2src >= self.sim_threshold
        mask = mask * tar_mask

        return {
            "idx_tar2src": idx_tar2src,
            "score_tar2src": score_tar2src,
            "mask": mask,
        }


__all__ = ["PointCorrespondenceFinder"]

if __name__ == "__main__":
    import torch

    finder = PointCorrespondenceFinder(sim_threshold=0.0)
    print(finder)

    tar_feat = torch.randn(1, 10, 128)
    src_feat = torch.randn(1, 10, 128)

    valid_matches = torch.randperm(5)
    tar_feat[0, :5] = src_feat[0, valid_matches]

    tar_mask = torch.ones(1, 10).bool()
    tar_mask[0, 4:7] = False
    # src_mask = torch.randint(0, 2, (1, 10)).bool()
    matches = finder(tar_feat, src_feat, tar_mask)

    # valid matches
    print("valid matches:")
    print(valid_matches)

    # masks
    print("\ntar mask:")
    print(tar_mask)
    # print("src mask:")
    # print(src_mask)

    # top match indices
    print("\nbest match indices:")
    print(matches["idx_tar2src"])

    print("\nbest match scores:")
    print(matches["score_tar2src"])

    print("\nbest match mask:")
    print(matches["mask"])
