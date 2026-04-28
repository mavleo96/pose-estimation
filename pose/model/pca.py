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

    def __repr__(self):
        return f"PCA(d={self.d}, n_components={self.n_components})"

    @property
    def device(self):
        return self.e.device

    @property
    def dtype(self):
        return self.e.dtype

    @torch.no_grad()
    def fit(self, x, subsample=False, num_samples=500000):
        assert x.ndim == 2
        assert x.shape[-1] == self.d
        assert not subsample or num_samples > 0

        if subsample and x.shape[0] > num_samples:
            indices = torch.randperm(x.shape[0])[:num_samples]
            x = x[indices]

        # Note: we fit on CPU to avoid out of memory errors and in float32 to avoid precision issues
        x_cpu = x.reshape(-1, self.d).cpu().float()
        cov = x_cpu.T.cov()
        e, v = torch.linalg.eigh(cov)

        self.v.data = v.to(self.device, dtype=self.v.dtype)
        self.e.data = e.to(self.device, dtype=self.e.dtype)
        self.mean.data = x_cpu.mean(0).to(self.device, dtype=self.mean.dtype)

    def forward(self, x):
        assert x.ndim > 1
        assert x.shape[-1] == self.d

        return (x - self.mean) @ self.v[:, -self.n_components :]


if __name__ == "__main__":
    x = torch.randn(3, 100, 1024, dtype=torch.float16, device="cuda")
    q = torch.randn(7, 1024, dtype=torch.float16, device="cuda")
    pca = PCA(1024, 5)
    pca.fit(x)
    pca.to(device="cuda", dtype=torch.float16)

    print(f"pca.mean.shape {pca.mean.shape} device {pca.mean.device} dtype {pca.mean.dtype}")
    print(f"pca.e.shape {pca.e.shape} device {pca.e.device} dtype {pca.e.dtype}")
    print(f"pca.v.shape {pca.v.shape} device {pca.v.device} dtype {pca.v.dtype}")
    print(f"pca(x).shape {pca(x).shape} device {pca(x).device} dtype {pca(x).dtype}")
    print(f"pca(q).shape {pca(q).shape} device {pca(q).device} dtype {pca(q).dtype}")
