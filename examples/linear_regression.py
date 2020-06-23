import arviz as az
import torch
import torch.distributions.constraints

import smallx


def linear_regression_model(D: int) -> smallx.Model:
    m = smallx.Model()
    m.param("beta", (D,), None)
    m.param("sigma", (1,), torch.distributions.constraints.positive)

    @m.model_block
    def f(d, p, logp):
        # prior
        logp += -(0.5 / d.prior_scale2) * (p.beta ** 2).sum()
        # implicit improper uniform prior on sigma

        # likelihood
        N = d.y.size(0)
        sigma2 = p.sigma[0] ** 2
        y_hat = d.x @ p.beta
        logp += -0.5 * N * sigma2.log() - (0.5 / sigma2) * ((d.y - y_hat) ** 2).sum()

        return logp

    return m


if __name__ == "__main__":
    torch.random.manual_seed(1234)

    # Simulate data
    N = 1000
    D = 3
    x = torch.randn((N, D))
    beta = torch.randn(D)
    y = x @ beta + torch.randn(N)

    # Get model
    m = linear_regression_model(D)

    data = {"y": y, "x": x, "prior_scale2": 1.0}

    # Run SVGD
    svgd_config = dict(n_particles=10, step_size=0.01, n_iter=2000, verbosity=100,)
    svgd_inference = smallx.SteinVariationalGradientDescent(m, data, svgd_config).run()
    svgd_posterior = svgd_inference.get_posterior_samples_as_inferencedata()

    # Run Metropolis sampling
    met_config = dict(q_sd=0.05, n_samples=10000, verbosity=1000,)
    met_inference = smallx.Metropolis(m, data, met_config).run()
    met_posterior = met_inference.get_posterior_samples_as_inferencedata()

    print("\nSVGD summary:")
    print(az.summary(svgd_posterior))
    print("\nMetropolis summary:")
    print(az.summary(met_posterior))
    print("\nSimulated beta:")
    print(beta)
