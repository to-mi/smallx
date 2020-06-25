from __future__ import annotations

import abc
from types import SimpleNamespace
from typing import Dict, List, Optional

import arviz as az
import numpy as np
import torch

from .inference_algorithms.gaussian_random_walk_metropolis import (
    gaussian_random_walk_metropolis,
)
from .inference_algorithms.svgd import svgd
from .model import DataType, Model, ParametersNumPyType, ParametersType


class Inference(abc.ABC):
    """Interface to computational inference algorithms and the posterior.

    Attributes:
        model: Probabilistic model.
        data: Data for the model.
        config: Configuration of the inference algorithm.
    """

    _default_config: Dict = {}

    def __init__(self, model: Model, data: DataType, config: Optional[Dict] = None):
        """Initialize the Inference object.

        Args:
            model: Probabilistic model.
            data: Data for the model.
            config: Configuration of the inference algorithm.
        """
        self.model = model
        self.data = data
        self.config = self.default_config
        if config is not None:
            self.config.update(config)

    @abc.abstractmethod
    def run(self, override_config: Optional[Dict] = None) -> Inference:
        """Runs the inference algorithm.

        This should be implemented by the concrete classes bound to an inference
        algorithm.

        Args:
            override_config: Overrides for the configuration.

        Returns:
            self
        """
        raise NotImplementedError()

    @property
    def packed_samples(self):
        raise NotImplementedError()

    def get_posterior_samples(self) -> List[ParametersType]:
        """Get the posterior samples as list of Dicts of torch tensors.

        Posterior inference should be run before calling this."""
        return self._get_posterior_samples_from_matrix(self.packed_samples)

    def get_posterior_samples_as_numpy(self) -> List[ParametersNumPyType]:
        """Get the posterior samples as list of Dict of numpy arrays.

        Posterior inference should be run before calling this."""
        return self._get_posterior_samples_from_matrix_as_numpy(self.packed_samples)

    def get_posterior_samples_as_inferencedata(self) -> az.InferenceData:
        """Get the posterior samples as ArviZ InferenceData-object.

        Posterior inference should be run before calling this."""
        return self._get_posterior_samples_from_matrix_as_inferencedata(
            self.packed_samples
        )

    @property
    def default_config(self) -> Dict:
        return self._default_config

    def _get_run_config(self, override_config: Optional[Dict]) -> SimpleNamespace:
        run_config_dict = self.config
        if override_config is not None:
            run_config_dict = {**run_config_dict, **override_config}
        run_config = SimpleNamespace(**run_config_dict)

        return run_config

    def _get_posterior_samples_from_matrix(
        self, matrix: torch.Tensor
    ) -> List[ParametersType]:
        samples = []
        for sample in matrix:
            samples.append(self.model.unpack(sample))
        return samples

    def _get_posterior_samples_from_matrix_as_numpy(
        self, matrix: torch.Tensor
    ) -> List[ParametersNumPyType]:
        samples = [
            {key: value.detach().cpu().numpy() for key, value in sample.items()}
            for sample in self._get_posterior_samples_from_matrix(matrix)
        ]
        return samples

    def _get_posterior_samples_from_matrix_as_inferencedata(
        self, matrix: torch.Tensor
    ) -> az.InferenceData:
        np_samples_list = self._get_posterior_samples_from_matrix_as_numpy(matrix)
        dictdata = {}

        for param_name in self.model.parameters.keys():
            dictdata[param_name] = np.stack(
                [sample[param_name] for sample in np_samples_list]
            )[
                np.newaxis, ...
            ]  # first dim should be "chain" for conversion to Inference Data

        infdata = az.convert_to_inference_data(dictdata)
        infdata.posterior.attrs["inference_library"] = "smallx"
        infdata.posterior.attrs["inference_library_version"] = "0.0"

        return infdata


class SteinVariationalGradientDescent(Inference):
    """Interface to Stein Variational Gradient Descent and its posterior.

    Attributes:
        model: Probabilistic model.
        data: Data for the model.
        config: Configuration of the inference algorithm.
    """

    _default_config: Dict = dict(
        n_particles=10, step_size=0.01, n_iter=1000, verbosity=10,
    )

    def __init__(self, model: Model, data: DataType, config: Optional[Dict] = None):
        super().__init__(model, data, config)
        self._particles: torch.Tensor = torch.tensor(())

    def run(
        self,
        override_config: Optional[Dict] = None,
        force_reinit_particles: bool = False,
    ) -> SteinVariationalGradientDescent:
        """Runs Stein Variational Gradient Descent.

        The algorithm optimizes a set of particles for representing the posterior.

        Args:
            override_config: Overrides for the configuration.
            force_reinit_particles: Reinitialize particles (or continue from old).

        Returns:
            self
        """
        run_config = self._get_run_config(override_config)

        self._initialize_particles(run_config, force_reinit_particles)

        # run inference
        svgd(
            self._particles,
            self._log_prob_wrapper,
            run_config.step_size,
            run_config.n_iter,
            run_config.verbosity,
        )

        return self

    @property
    def packed_samples(self):
        if self._particles.size(0) == 0:
            raise RuntimeError("SVGD has not been run yet.")
        return self._particles

    def _log_prob_wrapper(self, particles):
        return torch.stack(
            [
                self.model.log_prob_packed(particles[i, :], self.data)
                for i in range(particles.size(0))
            ]
        )

    def _initialize_particles(
        self, run_config: SimpleNamespace, force_reinit_particles: bool
    ):
        if self._particles.size(0) == 0 or force_reinit_particles:
            self._particles = torch.randn(
                run_config.n_particles,
                self.model.number_of_parameters,
                dtype=self.model.dtype,
                device=self.model.device,
                requires_grad=True,
            )
        else:
            if self._particles.size(0) != run_config.n_particles:
                raise ValueError(
                    "Cannot change the number of particles if they are already initialized"
                )


class Metropolis(Inference):
    """Interface to random walk Metropolis sampling and its posterior.

    Attributes:
        model: Probabilistic model.
        data: Data for the model.
        config: Configuration of the inference algorithm.
    """

    _default_config: Dict = dict(
        n_samples=10000, q_sd=1.0, verbosity=100,
    )

    def __init__(self, model: Model, data: DataType, config: Optional[Dict] = None):
        super().__init__(model, data, config)
        self._packed_samples = torch.tensor(())

    def run(self, override_config: Optional[Dict] = None) -> Metropolis:
        """Runs Metropolis sampling.

        Runs Metropolis Markov chain Monte Carlo sampling using symmetric Gaussian
        proposals.

        Args:
            override_config: Overrides for the configuration.

        Returns:
            self
        """
        run_config = self._get_run_config(override_config)

        if self._packed_samples.size(0) == 0:
            x = run_config.q_sd * torch.randn(
                self.model.number_of_parameters,
                dtype=self.model.dtype,
                device=self.model.device,
                requires_grad=False,
            )
        else:
            x = self._packed_samples[-1, :].detach().clone()

        new_samples = gaussian_random_walk_metropolis(
            x,
            self._log_prob_wrapper,
            run_config.q_sd,
            run_config.n_samples,
            run_config.verbosity,
        )

        if self._packed_samples.size(0) == 0:
            self._packed_samples = new_samples
        else:
            self._packed_samples = torch.cat((self._packed_samples, new_samples), dim=0)

        return self

    @property
    def packed_samples(self):
        if self._packed_samples.size(0) == 0:
            raise RuntimeError("Sampling has not been done yet.")
        return self._packed_samples

    def _log_prob_wrapper(self, x):
        return self.model.log_prob_packed(x, self.data)
