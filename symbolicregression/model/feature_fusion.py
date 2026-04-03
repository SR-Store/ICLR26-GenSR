import torch
import torch.nn as nn


def sample_gaussian_multi(mu: torch.Tensor, logvar: torch.Tensor, n_samples: int) -> torch.Tensor:
    B, D = mu.shape
    std = torch.exp(0.5 * logvar)
    mu_exp = mu.unsqueeze(1).expand(B, n_samples, D)
    std_exp = std.unsqueeze(1).expand(B, n_samples, D)
    eps = torch.randn(B, n_samples, D, device=mu.device, dtype=mu.dtype)
    return mu_exp + eps * std_exp


class FeatureFusion(nn.Module):

    def __init__(self, latent_dim=512, dec_emb_dim=512, seq_len=200, n_samples=200):
        super().__init__()
        self.latent_dim = latent_dim
        self.dec_emb_dim = dec_emb_dim
        self.seq_len = seq_len
        self.n_samples = n_samples

        self.expand_proj = nn.Linear(latent_dim, seq_len * dec_emb_dim)
        self.relu = nn.ReLU()
        self.refine_proj = nn.Linear(dec_emb_dim, dec_emb_dim)

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        samples = sample_gaussian_multi(mu, logvar, self.n_samples)
        z = samples.mean(dim=1)
        x = self.expand_proj(z)
        x = self.relu(x)
        x = x.view(-1, self.seq_len, self.dec_emb_dim)
        return self.refine_proj(x)

