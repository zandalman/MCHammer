from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

import mchammers as hammer


@pytest.mark.mpi
def test_sampler_basic_mpi():
    # load MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    def log_prob_func(x, mu=0, sig=1):
        return -0.5 * np.sum(((x - mu) / sig) ** 2, axis=-1)

    sampler = hammer.SamplerBasicMPI(
        2**12,
        32,
        3,
        log_prob_func,
        comm,
        MPI.SUM,
        size,
        rank,
        np.array([1.65, 1.65, 1.65]),
    )
    state_init = 2.0 * sampler.rng.random((sampler.num_walker, sampler.num_dim)) - 1.0
    sampler.run(state_init)
    if rank == 0:
        std = np.std(sampler.samples, axis=0)
        assert np.all(std) < 1.2
