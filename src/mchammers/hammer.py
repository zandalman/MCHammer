from __future__ import annotations

import abc
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["Sampler", "SamplerBasic"]


def __dir__():
    return __all__


class Sampler(abc.ABC):
    """
    The Hammer object.

    Parameters:
    ------------
        num_step:
            The number of steps.
        num_walker:
            The number of walkers.
        num_dim:
            The number of dimensions.
        log_prob_func:
            The log probability function.
        frac_burn:
            The burn fraction.
        seed:
            The random number generator seed.
    """

    def __init__(
        self,
        num_step: int,
        num_walker: int,
        num_dim: int,
        log_prob_func: Callable[[NDArray[float]], NDArray[float]],
        frac_burn: float = 0.2,
        seed: int | None = None,
    ):
        assert (
            np.abs(frac_burn - 0.5) <= 0.5
        ), f"Burn fraction {frac_burn:.3g} outside the allowed range [0, 1]."

        self.num_step = num_step
        self.num_walker = num_walker
        self.num_dim = num_dim
        self.log_prob_func = log_prob_func
        self.frac_burn = frac_burn
        self.seed = seed

        self.idx_step = 0
        self.rate_accept = 0.0
        self.state_curr = np.zeros((self.num_walker, self.num_dim))
        self.rng = np.random.default_rng(self.seed)
        self.num_step_burn = int(self.frac_burn * self.num_step)
        self.samples = np.zeros(
            (self.num_step - self.num_step_burn, self.num_walker, self.num_dim)
        )

        self.groups = self.init_groups()
        self.size_groups = [np.sum(group) for group in self.groups]

    @abc.abstractmethod
    def init_groups(self) -> list[NDArray[bool]]:
        return []

    @abc.abstractmethod
    def sample_prop(self, state: NDArray[float], idx_group: int) -> NDArray[float]:
        return np.array([0.0])

    @abc.abstractmethod
    def prob_accept(
        self, log_prob_curr: NDArray[float], log_prob_prop: NDArray[float]
    ) -> NDArray[float]:
        return np.array([0.0])

    def step(self) -> None:
        """
        Step the sampler.
        """
        for idx_group in range(len(self.groups)):
            # get the group
            group = self.groups[idx_group]
            size = self.size_groups[idx_group]

            # calculate the proposal state
            state_prop = self.sample_prop(self.state_curr, idx_group)

            # calculate the acceptance probability
            log_prob_curr = self.log_prob_func(self.state_curr[group])
            log_prob_prop = self.log_prob_func(state_prop)
            prob_accept = self.prob_accept(log_prob_curr, log_prob_prop)

            # move accepted proposals into the current state
            cond_accept = self.rng.random(size) < prob_accept
            self.state_curr[cond_accept] = state_prop[cond_accept]

        # if burn-in stage is over, add current state to samples
        if self.idx_step >= self.num_step_burn:
            self.rate_accept += np.sum(cond_accept)
            self.samples[self.idx_step - self.num_step_burn] = self.state_curr

        # increment the step index
        self.idx_step += 1

    def run(self, state_init: NDArray) -> None:
        """
        Run the sampler.

        Args
            state_init (np.ndarray):
                The initial state.
        """
        assert state_init.shape == (
            self.num_walker,
            self.num_dim,
        ), f"Initial state shape ({state_init.shape[0]:%d}, {state_init.shape[1]:%d}) \
                does not match number of walkers and number of dimensions ({self.num_walker:%d}, {self.num_dim:%d})"

        self.state_curr = state_init
        for _i in range(self.num_step - 1):
            self.step()

        # calculate the acceptance rate
        self.rate_accept /= (self.num_step - self.num_step_burn) * self.num_walker
        if self.rate_accept <= 0.1:
            warnings.warn(
                "Acceptance rate {self.rate_accept:%.3g} is very low.",
                UserWarning,
                stacklevel=2,
            )
        elif self.rate_accept >= 0.9:
            warnings.warn(
                "Acceptance rate {self.rate_accept:%.3g} is very high.",
                UserWarning,
                stacklevel=2,
            )


class SamplerBasic(Sampler):
    def __init__(
        self,
        num_step: int,
        num_walker: int,
        num_dim: int,
        log_prob_func: Callable[[NDArray[float]], NDArray[float]],
        cov: NDArray[float],
        frac_burn: float = 0.2,
        seed: int | None = None,
    ):
        super().__init__(num_step, num_walker, num_dim, log_prob_func, frac_burn, seed)
        self.cov = cov

    def init_groups(self):
        return [np.full(self.num_walker, True)]

    def sample_prop(self, state: NDArray[float], idx_group: int) -> NDArray[float]:
        assert idx_group == 0
        return self.rng.normal(
            state, self.cov[None, :], size=(self.num_walker, self.num_dim)
        )

    def prob_accept(self, log_prob_curr, log_prob_prop):
        return np.exp(log_prob_prop - log_prob_curr)


# class SamplerStretch(Sampler):
#     def __init__(
#         self,
#         num_step: int,
#         num_walker: int,
#         num_dim: int,
#         log_prob_func: Callable[[NDArray[float]], NDArray[float]],
#         frac_burn: float = 0.2,
#         seed: int | None = None,
#     ):
#         super().__init__(num_step, num_walker, num_dim, log_prob_func, frac_burn, seed)

#     def init_groups(self):
#         size_group1 = self.num_walker // 2
#         group1 = np.full(self.num_walker, True)
#         group2 = np.full(self.num_walker, True)
#         group1[size_group1:] = False
#         group2[:size_group1] = False
#         return [group1, group2]

#     def sample_prop(self, state: NDArray[float], idx_group: int):
#         group = self.groups[idx_group]
#         size1 = self.size_groups[idx_group]
#         size2 = self.num_walker - size1
#         other = self.rng.integers(low=0, high=size2, size=size1, dtype=int)
#         xi = self.rng.random(self.num_walker)
#         wparam = (xi + 1.0)**2 / 2.0
#         return state[other] + wparam[:, None] * (state[group] - state[other])

#     def prob_accept(self, log_prob_curr, log_prob_prop):

#         return np.exp(log_prob_prop - log_prob_curr)
