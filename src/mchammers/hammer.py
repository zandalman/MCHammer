from __future__ import annotations

import abc
import warnings
from collections.abc import Callable
from typing import Any, Dict, Sequence, Union

import numpy as np
from numpy.typing import NDArray

Numeric = Union[int, float]

__all__ = ["Sampler", "SamplerBasic", "SamplerMPI", "SamplerBasicMPI"]


class Sampler(abc.ABC):
    """
    The Sampler object.

    Parameters:
    ------------
        num_step:
            The number of steps.
        num_walker:
            The number of walkers.
        num_dim:
            The number of dimensions.
        prior_bounds:
            The prior bounds on the model parameters.
        state_init:
            The initial state.
        log_prob_func:
            The log probability function.
        args:
            The arguments of the log probability function.
        kwargs:
            The keyword arguments of the log probability function.
        frac_burn:
            The burn fraction.
        seed:
            The random number generator seed.
        flatten:
            Whether to flatten the final sample array.
    """

    def __init__(
        self,
        num_step: int,
        num_walker: int,
        num_dim: int,
        prior_bounds: Sequence[tuple[Numeric, Numeric]],
        state_init: NDArray[Numeric],
        log_prob_func: Callable[..., NDArray[Numeric]],
        args: Sequence[Any] | None = None,
        kwargs: Dict[Any, Any] | None = None,
        frac_burn: Numeric = 0.2,
        seed: int | None = None,
        flatten: bool = True,
    ):
        assert (
            np.abs(frac_burn - 0.5) <= 0.5
        ), f"Burn fraction {frac_burn:.3g} outside the allowed range [0, 1]."

        assert (
            len(prior_bounds) == num_dim
        ), f"Number of prior bounds {len(prior_bounds):d} does not match number of dimensions {num_dim:d}."

        for i, (low, high) in enumerate(prior_bounds):
            assert (
                low <= high
            ), f"Lower bound {low:.3g} is larger than upper bound {high:.3g} for parameter f{i:d}."

        self.num_step = num_step
        self.num_walker = num_walker
        self.num_dim = num_dim
        self.prior_bounds = prior_bounds
        self.state_init = state_init
        self.log_prob_func = log_prob_func
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self.frac_burn = frac_burn
        self.seed = seed
        self.flatten = flatten

        self.idx_step = 0
        self.rate_accept = 0.0
        self.state_curr = np.zeros((self.num_walker, self.num_dim))
        self.rng = np.random.default_rng(self.seed)
        self.num_step_burn = int(self.frac_burn * self.num_step)
        self.samples = np.zeros(
            (self.num_step - self.num_step_burn, self.num_walker, self.num_dim)
        )

        self.groups = [np.full(self.num_walker, True)]
        self.size_groups = [self.num_walker]

    @abc.abstractmethod
    def sample_prop(self, idx_group: int) -> NDArray[Numeric]:
        """
        Sample the proposal distribution.

        Parameters:
        ------------
            idx_group:
                The index of the current group.

        Returns:
        ------------
            An array of samples of the proposal distribution.
        """
        return np.array([0.0])  # pragma: no cover

    @abc.abstractmethod
    def prob_accept(
        self, log_prob_curr: NDArray[Numeric], log_prob_prop: NDArray[Numeric]
    ) -> NDArray[Numeric]:
        """
        Calculate the acceptance probability.

        Parameters:
        ------------
            log_prob_curr:
                The log probability of the current state.
            log_prob_prop:
                The log probability of the proposed state.

        Returns:
        ------------
            The acceptance probability.
        """
        return np.array([0.0])  # pragma: no cover

    def calc_rate_accept(self) -> None:
        """Calculate the acceptance rate."""
        # calculate the acceptance rate
        self.rate_accept /= (self.num_step - self.num_step_burn) * self.num_walker
        if self.rate_accept <= 0.1:
            warnings.warn(
                f"Acceptance rate {self.rate_accept:.3g} is very low.",
                UserWarning,
                stacklevel=2,
            )
        elif self.rate_accept >= 0.9:
            warnings.warn(
                f"Acceptance rate {self.rate_accept:.3g} is very high.",
                UserWarning,
                stacklevel=2,
            )

    def post_process(self) -> None:
        """Post-process the samples."""
        self.calc_rate_accept()
        if self.flatten:
            self.samples = self.samples.reshape(-1, self.num_dim)

    def step(self) -> None:
        """Step the sampler."""
        for idx_group in range(len(self.groups)):
            # get group
            group = self.groups[idx_group]
            size_group = self.size_groups[idx_group]
            state_curr = self.state_curr[group]

            # calculate the proposal state
            state_prop = self.sample_prop(idx_group)

            # calculate the acceptance probability
            log_prob_curr = self.log_prob_func(state_curr, *self.args, **self.kwargs)
            log_prob_prop = self.log_prob_func(state_prop, *self.args, **self.kwargs)
            prob_accept = self.prob_accept(log_prob_curr, log_prob_prop)

            # move accepted proposals into the current state
            cond_accept = self.rng.random(size_group) < prob_accept
            state_curr[cond_accept] = state_prop[cond_accept]
            self.state_curr[group] = state_curr

        # if burn-in stage is over, add current state to samples
        if self.idx_step >= self.num_step_burn:
            self.rate_accept += np.sum(cond_accept)
            self.samples[self.idx_step - self.num_step_burn] = self.state_curr

        # increment the step index
        self.idx_step += 1

    def run(self) -> None:
        """
        Run the sampler.

        Parameters:
        ------------
            state_init:
                The initial state.
        """
        # reset samples
        self.samples = np.zeros(
            (self.num_step - self.num_step_burn, self.num_walker, self.num_dim)
        )

        self.state_curr = self.state_init
        for _i in range(self.num_step - 1):
            self.step()

        self.post_process()


class SamplerMPI(Sampler):
    """
    The SamplerMPI object.

    Parameters:
    ------------
        comm:
            The MPI communicator.
        mpi_sum:
            The MPI sum operation.
        size:
            The number of MPI ranks.
        rank:
            The MPI rank.
    """

    def __init__(
        self,
        num_step: int,
        num_walker: int,
        num_dim: int,
        prior_bounds: Sequence[tuple[Numeric, Numeric]],
        state_init: NDArray[Numeric],
        log_prob_func: Callable[..., NDArray[Numeric]],
        comm: object,
        mpi_sum: object,
        size: int,
        rank: int,
        args: Sequence[Any] | None = None,
        kwargs: Dict[Any, Any] | None = None,
        frac_burn: Numeric = 0.2,
        seed: int | None = None,
        flatten: bool = True,
    ):
        super().__init__(
            num_step,
            num_walker,
            num_dim,
            prior_bounds,
            state_init,
            log_prob_func,
            args,
            kwargs,
            frac_burn,
            seed,
            flatten,
        )
        self.comm = comm
        self.mpi_sim = mpi_sum
        self.size = size
        self.rank = rank

        num_walker_per_rank = int(np.ceil(self.num_walker / size))
        idx_walker_min = min(self.rank * num_walker_per_rank, self.num_walker)
        idx_walker_max = min((self.rank + 1) * num_walker_per_rank, self.num_walker)
        self.slice = slice(idx_walker_min, idx_walker_max)

        group = np.full(self.num_walker, False)
        group[self.slice] = True
        self.groups = [group]
        self.size_groups = [np.sum(group)]

    def post_process(self) -> None:
        rate_accept_buff = np.array([0.0])
        samples_buff = np.zeros_like(self.samples)
        self.comm.Reduce(  # type: ignore[attr-defined]
            np.array([self.rate_accept]), rate_accept_buff, op=self.mpi_sim, root=0
        )
        self.comm.Reduce(self.samples, samples_buff, op=self.mpi_sim, root=0)  # type: ignore[attr-defined]
        if self.rank == 0:
            self.rate_accept = rate_accept_buff[0]
            self.samples = samples_buff
            super().post_process()


class SamplerBasic(Sampler):
    """
    The SamplerBasic object.

    Parameters:
    ------------
        std:
            The standard deviation of the proposal distribution relative to the prior bounds.
    """

    def __init__(
        self,
        num_step: int,
        num_walker: int,
        num_dim: int,
        prior_bounds: Sequence[tuple[Numeric, Numeric]],
        state_init: NDArray[Numeric],
        log_prob_func: Callable[..., NDArray[Numeric]],
        std_rel_prop: Numeric,
        args: Sequence[Any] | None = None,
        kwargs: Dict[Any, Any] | None = None,
        frac_burn: Numeric = 0.2,
        seed: int | None = None,
        flatten: bool = True,
    ):
        super().__init__(
            num_step,
            num_walker,
            num_dim,
            prior_bounds,
            state_init,
            log_prob_func,
            args,
            kwargs,
            frac_burn,
            seed,
            flatten,
        )
        assert (
            np.abs(std_rel_prop - 0.5) <= 0.5
        ), f"Relative standard deviation of the proposal distribution {std_rel_prop:.3g} outside the allowed range [0, 1]."

        self.std_prop = np.zeros((self.num_walker, self.num_dim))
        for i, (low, high) in enumerate(self.prior_bounds):
            self.std_prop[:, i] = std_rel_prop * (high - low)

    def sample_prop(self, idx_group: int) -> NDArray[Numeric]:
        assert idx_group == 0
        return self.rng.normal(
            self.state_curr, self.std_prop, size=(self.num_walker, self.num_dim)
        )

    def prob_accept(
        self, log_prob_curr: NDArray[Numeric], log_prob_prop: NDArray[Numeric]
    ) -> NDArray[Numeric]:
        return np.exp(log_prob_prop - log_prob_curr)


class SamplerBasicMPI(SamplerMPI):
    """
    The SamplerBasic object.

    Parameters:
    ------------
        cov:
            The covariance of the proposal distribution in each dimension.
    """

    def __init__(
        self,
        num_step: int,
        num_walker: int,
        num_dim: int,
        prior_bounds: Sequence[tuple[Numeric, Numeric]],
        state_init: NDArray[Numeric],
        log_prob_func: Callable[..., NDArray[Numeric]],
        comm: object,
        mpi_sum: object,
        size: int,
        rank: int,
        std_rel_prop: Numeric,
        args: Sequence[Any] | None = None,
        kwargs: Dict[Any, Any] | None = None,
        frac_burn: Numeric = 0.2,
        seed: int | None = None,
        flatten: bool = True,
    ):
        super().__init__(
            num_step,
            num_walker,
            num_dim,
            prior_bounds,
            state_init,
            log_prob_func,
            comm,
            mpi_sum,
            size,
            rank,
            args,
            kwargs,
            frac_burn,
            seed,
            flatten,
        )
        assert (
            np.abs(std_rel_prop - 0.5) <= 0.5
        ), f"Relative standard deviation of the proposal distribution {std_rel_prop:.3g} outside the allowed range [0, 1]."

        self.std_prop = np.zeros((self.num_walker, self.num_dim))
        for i, (low, high) in enumerate(self.prior_bounds):
            self.std_prop[:, i] = std_rel_prop * (high - low)

    def sample_prop(self, idx_group: int) -> NDArray[Numeric]:
        group = self.groups[idx_group]
        size_group = self.size_groups[idx_group]
        return self.rng.normal(
            self.state_curr[group],
            self.std_prop[group],
            size=(size_group, self.num_dim),
        )

    def prob_accept(
        self, log_prob_curr: NDArray[Numeric], log_prob_prop: NDArray[Numeric]
    ) -> NDArray[Numeric]:
        return np.exp(log_prob_prop - log_prob_curr)


class SamplerStretch(Sampler):
    """
    The SamplerStretch object.
    Implements the affine-invariant ensemble sampler using the parallel stretch move.

    Parameters:
    ------------
        a:
            The stretch move parameter, a > 1.
    """

    def __init__(
        self,
        num_step: int,
        num_walker: int,
        num_dim: int,
        prior_bounds: Sequence[tuple[Numeric, Numeric]],
        state_init: NDArray[Numeric],
        log_prob_func: Callable[..., NDArray[Numeric]],
        args: Sequence[Any] | None = None,
        kwargs: Dict[Any, Any] | None = None,
        a: Numeric = 2.0,
        frac_burn: Numeric = 0.2,
        seed: int | None = None,
        flatten: bool = True,
    ):
        super().__init__(
            num_step,
            num_walker,
            num_dim,
            prior_bounds,
            state_init,
            log_prob_func,
            args,
            kwargs,
            frac_burn,
            seed,
            flatten,
        )
        assert (
            a > 1.0
        ), f"The stretch move parameter 'a' must be greater than 1. Got {a}."
        self.a = a

        # Split walkers into two groups for parallel updates
        half = num_walker // 2
        self.groups = [np.arange(half), np.arange(half, num_walker)]
        self.size_groups = [half, half]

    def sample_Z(self, size: int) -> NDArray[Numeric]:
        """
        Sample Z from g(Z) ∝ 1/sqrt(Z) within [1/a, a] using the inverse transform sampling
        """
        lower, upper = 1 / self.a, self.a
        u = self.rng.uniform(0, 1, size=size)  # Uniform random variable
        return (u * (np.sqrt(upper) - np.sqrt(lower)) + np.sqrt(lower)) ** 2

    def sample_prop(self, idx_group: int) -> NDArray[Numeric]:
        """
        Sample the proposal distribution using the stretch move.

        Parameters:
        ------------
            group:
                Indices of the walkers in the current group.
            comp_group:
                Indices of the walkers in the complementary group.

        Returns:
        ------------
            An array of proposed samples.
        """
        group = self.groups[idx_group]
        comp_group = self.groups[1 - idx_group]
        size_group = len(group)
        state_curr = self.state_curr[group]

        # Draw complementary walkers for each walker in the group
        walker_comp = self.rng.choice(comp_group, size=size_group, replace=True)
        X_j = self.state_curr[walker_comp]

        # Sample Z for the stretch move
        Z = self.sample_Z(size_group)

        # Calculate the proposed position Y
        Y = X_j + Z[:, None] * (state_curr - X_j)
        self.Z = Z
        return Y

    def prob_accept(
        self, log_prob_curr: NDArray[Numeric], log_prob_prop: NDArray[Numeric]
    ) -> NDArray[Numeric]:
        q = (self.Z ** (self.num_dim - 1)) * np.exp(log_prob_prop - log_prob_curr)
        return np.minimum(1, q)
