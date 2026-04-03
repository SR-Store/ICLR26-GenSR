import torch

class DiagonalCMAES:
    def __init__(self, feature_dim, mu, sigma, device, c1_min=-1, manual_c1=None):
        self.feature_dim = feature_dim
        self.mu = mu
        self.device = device
        self.sigma = sigma
        self.cov_diag = torch.ones(feature_dim, device=device)
        self.ps = torch.zeros(feature_dim, device=device)
        self.p_c = torch.zeros(feature_dim, device=device)
        self.weights = torch.log(torch.tensor(mu + 0.5, device=device)) - torch.log(torch.arange(1, mu + 1, device=device, dtype=torch.float32))
        self.weights = self.weights / self.weights.sum()
        self.mu_eff = 1 / (self.weights ** 2).sum()
        self.chiN = (feature_dim ** 0.5) * (1 - 1 / (4 * feature_dim) + 1 / (21 * feature_dim ** 2))
        self.manual_c1 = manual_c1
        self.c1_min = c1_min
        self._update_params()

    def _update_params(self):
        self.cs = (self.mu_eff + 2) / (self.feature_dim + self.mu_eff + 3)
        self.ds = 1 + 2 * max(0, torch.sqrt(self.mu_eff / self.feature_dim) - 1) + self.cs
        self.cc = (4 + self.mu_eff / self.feature_dim) / (self.feature_dim + 4 + 2 * self.mu_eff / self.feature_dim)
        if self.manual_c1 is not None:
            self.c1 = self.manual_c1
        elif self.c1_min != -1:
            c1 = 2 / ((self.feature_dim + 1.3) ** 2 + self.mu_eff)
            self.c1 = max(c1, self.c1_min)
        else:
            self.c1 = 2 / ((self.feature_dim + 1.3) ** 2 + self.mu_eff)

    def ask(self, mean, pop_size, lb=-5, ub=5):
        pop = mean.unsqueeze(0).repeat(pop_size, 1)
        if pop_size > 1:
            diversity = torch.randn(pop_size-1, self.feature_dim, device=self.device) * self.sigma * torch.sqrt(self.cov_diag).unsqueeze(0)
            pop[1:] += diversity
        return pop

    def tell(self, pop, fitness_values, candidates):
        sorted_indices = torch.argsort(fitness_values, descending=True)
        selected_indices = sorted_indices[:self.mu]
        past_mean = pop.mean(dim=0)
        new_mean = torch.zeros_like(past_mean)
        for i, idx in enumerate(selected_indices):
            new_mean += self.weights[i] * candidates[idx]
        y = (new_mean - past_mean) / self.sigma
        cs_factor = torch.sqrt(self.cs * (2 - self.cs) * self.mu_eff)
        cc_factor = torch.sqrt(self.cc * (2 - self.cc) * self.mu_eff)
        self.ps = (1 - self.cs) * self.ps + cs_factor * y
        self.p_c = (1 - self.cc) * self.p_c + cc_factor * y
        self.cov_diag = (1 - self.c1) * self.cov_diag + self.c1 * (self.p_c ** 2)
        self.sigma = self.sigma * torch.exp((self.cs / self.ds) * (torch.norm(self.ps) / self.chiN - 1))
        self._update_params()
        return new_mean

    def get_state(self):
        return {
            'sigma': self.sigma,
            'ps_norm': torch.norm(self.ps).item(),
            'cs': self.cs,
            'ds': self.ds,
            'cc': self.cc,
            'c1': self.c1,
            'chiN': self.chiN,
            'cov_diag_mean': self.cov_diag.mean().item(),
            'cov_diag_std': self.cov_diag.std().item(),
            'p_c_mean': self.p_c.mean().item(),
            'p_c_std': self.p_c.std().item(),
        }


class DiagonalCMAES_compre:
    def __init__(self, feature_dim, mu, sigma, device, c1_min=-1, manual_c1=None):
        self.feature_dim = feature_dim
        self.mu = mu
        self.device = device
        self.sigma = sigma
        self.cov_diag = torch.ones(feature_dim, device=device)
        self.ps = torch.zeros(feature_dim, device=device)
        self.p_c = torch.zeros(feature_dim, device=device)
        self.weights = torch.log(torch.tensor(mu + 0.5, device=device)) - torch.log(torch.arange(1, mu + 1, device=device, dtype=torch.float32))
        self.weights = self.weights / self.weights.sum()
        self.mu_eff = 1 / (self.weights ** 2).sum()
        self.chiN = (feature_dim ** 0.5) * (1 - 1 / (4 * feature_dim) + 1 / (21 * feature_dim ** 2))
        self.manual_c1 = manual_c1
        self.c1_min = c1_min
        self._update_params()

    def _update_params(self):
        self.cs = (self.mu_eff + 2) / (self.feature_dim + self.mu_eff + 3)
        self.ds = 1 + 2 * max(0, torch.sqrt(self.mu_eff / self.feature_dim) - 1) + self.cs
        self.cc = (4 + self.mu_eff / self.feature_dim) / (self.feature_dim + 4 + 2 * self.mu_eff / self.feature_dim)
        if self.manual_c1 is not None:
            self.c1 = self.manual_c1
        elif self.c1_min != -1:
            c1 = 2 / ((self.feature_dim + 1.3) ** 2 + self.mu_eff)
            self.c1 = max(c1, self.c1_min)
        else:
            self.c1 = 2 / ((self.feature_dim + 1.3) ** 2 + self.mu_eff)

    def ask(self, mean, pop_size, lb=-5, ub=5):
        pop = mean.unsqueeze(0).repeat(pop_size, 1)
        if pop_size > 1:
            diversity = torch.randn(pop_size-1, self.feature_dim, device=self.device) * self.sigma * torch.sqrt(self.cov_diag).unsqueeze(0)
            pop[1:] += diversity
        return pop

    def tell(self, pop, fitness_values, candidates):
        sorted_indices = torch.argsort(fitness_values, descending=True)
        selected_indices = sorted_indices[:self.mu]
        past_mean = pop.mean(dim=0)
        new_mean = torch.zeros_like(past_mean)
        for i, idx in enumerate(selected_indices):
            new_mean += self.weights[i] * candidates[idx]
        y = (new_mean - past_mean) / self.sigma
        cs_factor = torch.sqrt(self.cs * (2 - self.cs) * self.mu_eff)
        cc_factor = torch.sqrt(self.cc * (2 - self.cc) * self.mu_eff)
        self.ps = (1 - self.cs) * self.ps + cs_factor * y
        self.p_c = (1 - self.cc) * self.p_c + cc_factor * y
        self.cov_diag = (1 - self.c1) * self.cov_diag + self.c1 * (self.p_c ** 2)
        self.cov_diag = self.cov_diag * self.mask
        self.sigma = self.sigma * torch.exp((self.cs / self.ds) * (torch.norm(self.ps) / self.chiN - 1))
        self._update_params()
        return new_mean

    def get_state(self):
        return {
            'sigma': self.sigma,
            'ps_norm': torch.norm(self.ps).item(),
            'cs': self.cs,
            'ds': self.ds,
            'cc': self.cc,
            'c1': self.c1,
            'chiN': self.chiN,
            'cov_diag_mean': self.cov_diag.mean().item(),
            'cov_diag_std': self.cov_diag.std().item(),
            'p_c_mean': self.p_c.mean().item(),
            'p_c_std': self.p_c.std().item(),
        }
