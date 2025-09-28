import torch
import torch.nn as nn


class PCA(nn.Module):
    def __init__(self, d, n_components):
        super().__init__()
        self.d = d
        self.n_components = n_components

        self.e = torch.zeros(d)
        self.v = torch.zeros(d, d)

    def fit(self, x):
        assert x.ndim > 1
        assert x.shape[-1] == self.d

        cov = x.reshape(-1, self.d).T.cov()
        e, v = torch.linalg.eigh(cov)

        self.v.data = v
        self.e.data = e

    def forward(self, x):
        assert x.ndim > 1
        assert x.shape[-1] == self.d

        return x @ self.v[:, -self.n_components :]


if __name__ == "__main__":
    x = torch.randn(3, 100, 1024)
    q = torch.randn(7, 1024)
    pca = PCA(1024, 5)
    pca.fit(x)

    print(pca.e.shape)
    print(pca.v.shape)
    print(pca(x).shape)
    print(pca(q).shape)
