import arviz as az
import torch

import smallx


def multivariate_normal_model(D: int) -> smallx.Model:
    # Create model.
    m = smallx.Model()
    # Add parameter named mu, size (D,), and no constraint on range of values.
    m.param("mu", (D,), None)

    # Add code to compute the unnormalized log posterior probability:
    # The function decorated with @m.model_block should take three arguments:
    # * d refers to data and should be dict, which will be automatically wrapped in
    #   SimpleNamespace (that is, one refers to the elements with d.name).
    # * p refers to parameters (dict, automatically wrapped in SimpleNamespace).
    # * logp to the log posterior probability accumulator (one should add to this!).
    # The function can return anything (return value is not used).
    # One can add multiple model blocks and their logp's are summed.
    @m.model_block
    def f(d, p, logp):
        # prior
        logp += torch.distributions.MultivariateNormal(
            torch.zeros(D), torch.diag(torch.ones(D))
        ).log_prob(p.mu)

        # likelihood
        logp += torch.distributions.MultivariateNormal(
            p.mu, torch.diag(torch.ones(D))
        ).log_prob(d.y)
        return logp

    return m


if __name__ == "__main__":
    # Simulate data
    D = 3
    y = torch.ones(D)

    # Get model
    m = multivariate_normal_model(D)

    data = {"y": y}

    # Run SVGD
    svgd_config = dict(n_particles=10, step_size=0.01, n_iter=1000, verbosity=100,)
    svgd_inference = smallx.SteinVariationalGradientDescent(m, data, svgd_config).run()
    svgd_posterior = svgd_inference.get_posterior_samples_as_inferencedata()

    # Run Metropolis sampling
    met_config = dict(q_sd=1.0, n_samples=10000, verbosity=1000,)
    met_inference = smallx.Metropolis(m, data, met_config).run()
    met_posterior = met_inference.get_posterior_samples_as_inferencedata()

    print("\nSVGD summary:")
    print(az.summary(svgd_posterior))
    print("\nMetropolis summary:")
    print(az.summary(met_posterior))
    print("\nSimulated data:")
    print(y)
