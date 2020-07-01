# Small Expectations (`smallx`) — a small probabilistic programming library

**Note: this is more of a learning exercise and not so much intended for serious use.**

Small Expectations (`smallx`) is a small probabilistic programming library for inference in Bayesian
models, built on top of PyTorch to make use of its automatic differentiation engine.

A model is defined in Small Expectations by declaring the parameters to be inferred and
coding up the log unnormalized probability of the model (similar to Stan).

## Features

 * Relatively small and simple design: `Model` class for representing the probabilistic model and `Inference` class for interfacing to computational inference algorithms and holding the posterior.
 * Algorithms: Stein Variational Gradient Descent and Metropolis Markov chain Monte Carlo sampling with Gaussian proposal distribution.
 * [ArviZ](https://arviz-devs.github.io/arviz/) integration (getting the posterior representation as `InferenceData` object).

## Getting started

Create conda environment:
```
conda env create -f environment/environment.yml
```

Activate the environment:
```
conda activate smallx
```

Run an example: 
```
python -m examples.multivariate_normal
```


## Example: multivariate normal distribution

This shows sampling from a multivariate normal distribution as a pedagogical example. There are more examples in the `examples` folder.


```{python}
import arviz as az
import torch

import smallx


# Simulate data.
D = 3
y = torch.ones(D)
data = {"y": y}

# Create model.
m = smallx.Model()
# Add parameter named mu, size (D,), and no constraint on range of values.
m.param('mu', (D,), None)

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
```

## TODO

 * Make code run on GPU.
 * More examples. Plot results and compute some predictions.
 * Allow setting initial points for inference.
 * Add tests. Currently `pytest` just checks black formatting and mypy types.
 * Consider allowing log prob functions to take a batch of parameter candidates and accumulate the corresponding log probabilities to a vector. This would allow computing the log probs for SVGD without looping over the particles and running multiple MCMC chains "in parallel". The user would have the responsibility to write the log prob function appropriately. The `model_block` decorator could take a parameter which would indicate whether the block handles batched parameters or not (and automatic looping over the batch would be done in the latter case).
 * Interface to [LittleMCMC](https://github.com/eigenfoo/littlemcmc) for NUTS/HMC sampling?

## Contact

Tomi Peltola, tomi.peltola@tmpl.fi, http://www.tmpl.fi

## License

MIT License. See `LICENSE`.
