from typing import Callable

import torch
from tqdm import tqdm


def gaussian_random_walk_metropolis(
    x: torch.Tensor,
    logp: Callable[[torch.Tensor], torch.Tensor],
    q_sd: float,
    n_samples: int,
    verbosity: int = 1,
) -> torch.Tensor:
    """Runs Metropolis Markov chain Monte Carlo sampling.

    Args:
        x: Initial point (vector of length n_parameters).
        logp: Function that returns the unnormalized log probability for x.
        q_sd: Standard deviation of the Gaussian proposal distribution.
        n_samples: Number of samples to collect (=number of iterations to run).
        verbosity: Interval between updating the progress bar. Set to 0 to
            disable progress bar.

    Returns:
        The collected samples (n_iter x n_parameters matrix).
    """
    n_parameters = x.size(0)
    samples = torch.zeros_like(x).unsqueeze(0).repeat(n_samples, 1)
    logp_cur = logp(x)
    n_accepted = 0

    with tqdm(desc="Metropolis", total=n_samples, disable=verbosity <= 0) as pbar:
        last_pb_update = 0

        for i in range(n_samples):
            x_prop = x + q_sd * torch.randn(
                n_parameters, dtype=x.dtype, device=x.device, requires_grad=False
            )
            logp_prop = logp(x_prop)

            # acc prob: min(1, p_prop q(x_cur | x_prop) / p_cur q(x_prop | x_cur))
            log_ratio = (logp_prop - logp_cur).item()

            accept = (
                log_ratio >= 0.0
                or torch.log(torch.rand(1, dtype=x.dtype)).item() < log_ratio
            )

            if accept:
                x = x_prop
                logp_cur = logp_prop
                n_accepted += 1

            samples[i, :] = x

            iter1 = i + 1
            if verbosity > 0 and (iter1 % verbosity == 0 or iter1 == n_samples):
                pbar.set_postfix_str(
                    f"log prob={logp_cur.item():.2f}, acceptance rate={n_accepted / iter1:.2f}"
                )
                pbar.update(iter1 - last_pb_update)
                last_pb_update = iter1

    # TODO: return logp trace
    return samples
