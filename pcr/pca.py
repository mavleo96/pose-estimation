import torch
import torch.nn as nn


class PCA(nn.Module):
    def __init__(self, d, n_components):
        super().__init__()
        self.d = d
        self.n_components = n_components

        self.register_buffer("e", torch.zeros(d))
        self.register_buffer("v", torch.zeros(d, d))
        self.register_buffer("mean", torch.zeros(d))

    def fit(self, x):
        assert x.ndim > 1
        assert x.shape[-1] == self.d

        cov = x.reshape(-1, self.d).T.cov()
        e, v = torch.linalg.eigh(cov)

        self.v.data = v
        self.e.data = e
        self.mean.data = x.reshape(-1, self.d).mean(0)

    def forward(self, x):
        assert x.ndim > 1
        assert x.shape[-1] == self.d

        return (x - self.mean) @ self.v[:, -self.n_components :]

    def dict(self):
        return {
            "n_components": self.n_components,
            "mean": self.mean.data.cpu().numpy(),
            "e": self.e.data.cpu().numpy(),
            "v": self.v.data.cpu().numpy(),
        }

    def from_dict(self, data):
        self.n_components = data["n_components"]
        self.mean.data = torch.from_numpy(data["mean"]).float()
        self.e.data = torch.from_numpy(data["e"]).float()
        self.v.data = torch.from_numpy(data["v"]).float()


if __name__ == "__main__":
    x = torch.randn(3, 100, 1024)
    q = torch.randn(7, 1024)
    pca = PCA(1024, 5)
    pca.fit(x)

    print(f"pca.mean.shape {pca.mean.shape}")
    print(f"pca.e.shape {pca.e.shape}")
    print(f"pca.v.shape {pca.v.shape}")
    print(f"pca(x).shape {pca(x).shape}")
    print(f"pca(q).shape {pca(q).shape}")
