from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

import mchammers


def test_sampler_basic():
    """
    Test that the sampler recovers a multi-variate Gaussian
    distribution with approximately the correct standard deviation.
    """

    def log_prob_func(x, mu=0, sig=1):
        return -0.5 * np.sum(((x - mu) / sig) ** 2, axis=-1)

    rng = np.random.default_rng()
    state_init = rng.normal(0, 1, (32, 3))

    prior_bounds = [(-1, 1), (-1, 1), (-1, 1)]
    sampler = mchammers.SamplerBasic(
        2**12, 32, 3, prior_bounds, state_init, log_prob_func, 0.75
    )
    sampler.run()
    std = np.std(sampler.samples, axis=0)
    assert np.all(std) < 1.2


def test_sampler_stretch():
    def log_prob_func(x, mu=0, sig=1):
        return -0.5 * np.sum(((x - mu) / sig) ** 2, axis=-1)

    rng = np.random.default_rng()
    state_init = rng.normal(0, 1, (32, 3))

    prior_bounds = [(-1, 1), (-1, 1), (-1, 1)]
    sampler = mchammers.SamplerStretch(
        num_step=2**12,
        num_walker=32,
        num_dim=3,
        prior_bounds=prior_bounds,
        state_init=state_init,
        log_prob_func=log_prob_func,
        a=2,  # a=2 is implemented in GW10
        frac_burn=0.2,
    )
    sampler.run()

    std = np.std(sampler.samples, axis=0)
    assert np.all(std < 1.2), f"Standard deviation out of bounds: {std}"


@pytest.mark.mpi
def test_sampler_basic_mpi():
    # load MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    def log_prob_func(x, mu=0, sig=1):
        return -0.5 * np.sum(((x - mu) / sig) ** 2, axis=-1)

    rng = np.random.default_rng()
    state_init = rng.normal(0, 1, (32, 3))

    prior_bounds = [(-1, 1), (-1, 1), (-1, 1)]
    sampler = mchammers.SamplerBasicMPI(
        2**12,
        32,
        3,
        prior_bounds,
        state_init,
        log_prob_func,
        comm,
        MPI.SUM,
        size,
        rank,
        0.75,
    )
    sampler.run()
    if rank == 0:
        std = np.std(sampler.samples, axis=0)
        assert np.all(std) < 1.2
