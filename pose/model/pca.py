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

    @property
    def device(self):
        return self.e.device

    @torch.no_grad()
    def fit(self, x):
        assert x.ndim > 1
        assert x.shape[-1] == self.d

        x_cpu = x.reshape(-1, self.d).cpu()

        cov = x_cpu.T.cov()
        e, v = torch.linalg.eigh(cov)

        self.v.data = v.to(self.device)
        self.e.data = e.to(self.device)
        self.mean.data = x_cpu.mean(0).to(self.device)

    def forward(self, x):
        assert x.ndim > 1
        assert x.shape[-1] == self.d

        return (x - self.mean) @ self.v[:, -self.n_components :]


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
