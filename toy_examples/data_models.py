import torch

def pricing(d: int = 5, 
            sigma: torch.Tensor = torch.eye(5), 
            mu_0: torch.Tensor = torch.Tensor([6.55,6.72,6.60,6.54,6.42]), 
            eps: float = 1.5
    ):

    mu = lambda theta: mu_0 + eps * theta

    def D_theta(theta: torch.Tensor, n: int) -> torch.Tensor:
        return torch.distributions.MultivariateNormal(mu(theta), sigma).sample((n,)).T
    
    def loss(theta: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        theta = theta.unsqueeze(-1)  # (d,) -> (d,1)
        return torch.mean(theta.T @ z)
    
    info = {"theta_optimal": mu_0 / (2 * eps), "theta_stable": mu_0 / eps}

    return D_theta, loss, mu, info

if __name__ == "__main__":
    D_theta, loss, mu, info = pricing()
    theta = torch.zeros(5)
    z = D_theta(theta, 10)
    print("Sampled data points:", z.shape)
    print("Loss:", loss(theta, z))
    print("Info:", info)