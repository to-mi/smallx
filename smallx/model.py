from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import constraint_registry, constraints

ModelBlockFunctionType = Callable[[SimpleNamespace, SimpleNamespace, torch.Tensor], Any]
InternalParametersType = Dict[
    str, Tuple[Tuple[int], torch.distributions.Transform, slice]
]
DataType = Dict[str, Any]
ParametersType = Dict[str, torch.Tensor]
ParametersNumPyType = Dict[str, np.ndarray]


class Model:
    """Holds the probabilistic model.

    This class holds knowledge about the parameters of the model and
    how to calculate the unnormalized log probability of the model given
    some data.

    Attributes:
        parameters: Parameter names (keys) and internal information
            (sizes, transformations, and slice in packed parameters tensor).
        model_block_f: List of function that define the unnormalized log
            probability.
        number_of_parameters: Number of parameters.
        dtype: Dtype (for torch) for the parameters.
        device: Device (for torch) for holding the parameters.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """Initialize the model object.

        Args:
            dtype: Dtype (for torch) for the parameters.
            device: Device (for torch) for holding the parameters.
        """
        self.parameters: InternalParametersType = {}
        self.model_block_f: List[ModelBlockFunctionType] = []
        self.number_of_parameters: int = 0
        self.dtype: torch.dtype = dtype
        self.device: torch.device = device

    def log_prob(self, parameters: ParametersType, data: DataType) -> torch.Tensor:
        """Calculates the unnormalized log probability in the constrained space.

        This function does not apply any transformations. The parameters are assumed
        to have appropriate values within their constraints.

        Args:
            parameters: Parameters of the model.
            data: Data for the model.

        Returns:
            The unnormalized log probability (without accounting for any transformations).
        """
        wrapped_parameters = SimpleNamespace(**parameters)
        wrapped_data = SimpleNamespace(**data)
        logp = torch.zeros((), dtype=self.dtype, device=self.device)
        for i in range(len(self.model_block_f)):
            self.model_block_f[i](wrapped_data, wrapped_parameters, logp)
        return logp

    def log_prob_packed(
        self, packed_parameters: torch.Tensor, data: DataType
    ) -> torch.Tensor:
        """Calculates the unnormalized log probability in the unconstrained space.

        Accounts for the Jacobian of the transformations from the unconstrained
        space to the constrained values.

        Args:
            packed_parameters: Parameters of the model in unconstrained space.
            data: Data for the model.

        Returns:
            The unnormalized log probability (accounting for transformations).
        """
        parameters, log_j = self.unpack_and_get_log_j(packed_parameters)
        return log_j + self.log_prob(parameters, data)

    def model_block(
        self, block: ModelBlockFunctionType
    ) -> Callable[[DataType, ParametersType, torch.Tensor], Any]:
        """Registers a model block to the model.

        Each registered block will be called in the calculation of the unnormalized
        log probability of the model. The data and parameters arguments of the block
        are assumed to be Dicts and are automatically wrapped in SimpleNamespaces, so
        they should be referred to in the model block function using dot-notation
        (e.g., `data.y`, not `data["y"]`).

        Args:
            block: Function to register as model block. The function should take
                three arguments - data, parameters, logp - and increment logp
                according to the unnormalized log probability of this block. The
                model block function can return anything - the return value is not
                used by the library.

        Returns:
            The model block with automatic wrapping of data Dict and parameters Dict in
            SimpleNamespaces.
        """
        self.model_block_f.append(block)

        # Wrap the original function d and p args in SimpleNamespaces
        # (note: the type of the wrapped function has Dicts instead SimpleNamespaces).
        def wrapped_block(d: DataType, p: ParametersType, logp: torch.Tensor):
            return block(SimpleNamespace(**d), SimpleNamespace(**p), logp)

        return wrapped_block

    def param(
        self,
        name: str,
        size: Tuple[int],
        constraint: Optional[torch.distributions.constraints.Constraint],
    ):
        """Registers a parameter for the model.

        Args:
            name: Name of the parameter.
            size: Size of the parameter.
            constraint: Constraint for the parameter (e.g., positivity).
        """
        ne = int(np.prod(list(size)))
        pack_indices = slice(self.number_of_parameters, self.number_of_parameters + ne)
        self.number_of_parameters += ne
        if constraint is None:
            constraint = constraints.real
        transformation = constraint_registry.biject_to(constraint)
        self.parameters[name] = (size, transformation, pack_indices)

    def unpack(
        self, packed_parameters: torch.Tensor, skip_tf: bool = False
    ) -> ParametersType:
        """Unpacks parameters from unconstrained vector to constrained values.

        Args:
            packed_parameters: Vector containing the flattened unconstrained
                parameters.
            skip_tf: Skip transformation to constrained space.

        Returns:
            Parameter dictionary, with values in the constrained space (if
            skip_tf is False, default) or unconstrained space (if skip_tf is
            True).
        """
        parameters, _ = self.unpack_and_get_log_j(
            packed_parameters, skip_tf, calculate_log_jacobian=False
        )
        return parameters

    def unpack_and_get_log_j(
        self,
        packed_parameters: torch.Tensor,
        skip_tf: bool = False,
        calculate_log_jacobian: bool = True,
    ) -> Tuple[ParametersType, torch.Tensor]:
        """Unpacks parameters, returning also log absolute determinant of the Jacobian.

        Unpacks parameters from unconstrained vector to constrained values and computes the
        log absolute determinant of the Jacobian of the transformations.

        Args:
            packed_parameters: Vector containing the flattened unconstrained
                parameters.
            skip_tf: Skip transformation to constrained space.
            calculate_log_jacobian: Whether to calculate the Jacobian term.

        Returns:
            Tuple containing the parameters Dict and the log absolute determinant of the
            Jacobian of the transformations (or zero if calculate_log_jacobian is False).
        """
        if skip_tf and calculate_log_jacobian:
            raise ValueError(
                "Skipping transformation and returning log Jacobian is not supported"
            )

        parameters = {}
        log_j = torch.zeros((), dtype=self.dtype, device=self.device)

        for key, (size, tf, pack_indices) in self.parameters.items():
            untf_parameter = packed_parameters[pack_indices].view(*size)
            if skip_tf:
                parameters[key] = untf_parameter
            else:
                parameters[key] = tf(untf_parameter)
            if calculate_log_jacobian:
                log_j += tf.log_abs_det_jacobian(untf_parameter, parameters[key]).sum()
        return parameters, log_j

    def pack(self, parameters: ParametersType) -> torch.Tensor:
        """Packs parameters and transforms to unconstrained space.

        Args:
            parameters: Dict of constrained parameters, assumed to be in the same
                order as the parameters registered in the model.

        Returns:
            Vector containing transformed and flattened (packed) parameters.
        """
        packed_parameters = torch.stack(
            [
                tf.inv(val).view(-1)
                for (val, (_, tf, _)) in zip(
                    parameters.values(), self.parameters.values()
                )
            ]
        )
        return packed_parameters
