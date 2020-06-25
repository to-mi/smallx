from typing import Callable, Optional, Union

import torch
import torch.optim
from tqdm import tqdm


def svgd(
    x: torch.Tensor,
    logp: Callable[[torch.Tensor], torch.Tensor],
    optimizer: Optional[Union[float, torch.optim.Optimizer]],  # type: ignore
    n_iter: int,
    verbosity: int = 1,
) -> torch.Tensor:
    """Runs Stein Variational Gradient Descent.

    Args:
        x: Particles (n_particles x n_parameters matrix).
        logp: Function that returns the unnormalized log probability for each
            particle.
        optimizer: Torch optimizer instance (bound to x) or float interpreted
            as learning rate for Adam optimizer.
        n_iter: Number of iterations to run.
        verbosity: Interval between updating the progress bar. Set to 0 to
            disable progress bar.

    Returns:
        The particles x (the same object as input; x is modified in place).
    """
    # TODO: update the optimizer interface to something more sensible?
    if optimizer is None:
        optimizer = torch.optim.Adam((x,))
    elif isinstance(optimizer, float):
        optimizer = torch.optim.Adam((x,), lr=optimizer)

    n_particles = x.size(0)
    log_n_p = torch.log(torch.tensor(n_particles, dtype=x.dtype, device=x.device))

    with tqdm(desc="SVGD", total=n_iter, disable=verbosity <= 0) as pbar:
        last_pb_update = 0

        for i in range(n_iter):
            optimizer.zero_grad()
            particle_log_probs = logp(x)
            particle_log_probs.backward(x.new_ones((n_particles,)))

            with torch.no_grad():
                # d2 will be squared Euclidean distance.
                G = x @ x.t()
                dG = torch.diag(G)
                d2 = dG + dG.unsqueeze(0).t() - 2.0 * G
                # h2 is length scale, using median of distances per
                # log(num_particles) following the SVGD paper.
                # Adding 1e-10 to avoid possible division by zero.
                h2 = d2.median() / log_n_p + 1e-10
                k = torch.exp((-1.0 / h2) * d2)
                dk = (k.sum(0).diag() - k) @ x / h2
                phi = -(k @ x.grad + dk) / n_particles

                x.grad = phi
                optimizer.step()

            iter1 = i + 1
            if verbosity > 0 and (iter1 % verbosity == 0 or iter1 == n_iter):
                pbar.set_postfix_str(
                    f"mean log prob={particle_log_probs.mean().item():.2f}"
                )
                pbar.update(iter1 - last_pb_update)
                last_pb_update = iter1

            # TODO: check for convergence?

    return x
