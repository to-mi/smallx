import arviz as az
import torch
import torch.nn.functional as F

import smallx


def logistic_regression_model(D: int) -> smallx.Model:
    m = smallx.Model()
    m.param("beta", (D,), None)

    @m.model_block
    def f(d, p, logp):
        # prior
        logp += -(0.5 / d.prior_scale2) * (p.beta ** 2).sum()

        # likelihood
        logits = d.x @ p.beta
        logp += -F.binary_cross_entropy_with_logits(logits, d.y, reduction="sum")

        return logp

    return m


if __name__ == "__main__":
    torch.random.manual_seed(1234)

    # Simulate data.
    N = 1000
    D = 3
    x = torch.randn((N, D))
    beta = torch.randn(D)
    y = torch.sigmoid(x @ beta).bernoulli()

    # Get model
    m = logistic_regression_model(D)

    data = {"y": y, "x": x, "prior_scale2": 1.0}

    # Run SVGD
    svgd_config = dict(n_particles=10, step_size=0.01, n_iter=2000, verbosity=100,)
    svgd_inference = smallx.SteinVariationalGradientDescent(m, data, svgd_config).run()
    svgd_posterior = svgd_inference.get_posterior_samples_as_inferencedata()

    # Run Metropolis sampling
    met_config = dict(q_sd=0.1, n_samples=10000, verbosity=1000,)
    met_inference = smallx.Metropolis(m, data, met_config).run()
    met_posterior = met_inference.get_posterior_samples_as_inferencedata()

    print("\nSVGD summary:")
    print(az.summary(svgd_posterior))
    print("\nMetropolis summary:")
    print(az.summary(met_posterior))
    print("\nSimulated beta:")
    print(beta)
