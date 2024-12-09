from __future__ import annotations

import numpy as np

import mchammers


def test_sampler_basic():
    """
    Test that the sampler recovers a multi-variate Gaussian
    distribution with approximately the correct standard deviation.
    """

    def log_prob_func(x, mu=0, sig=1):
        return -0.5 * np.sum(((x - mu) / sig) ** 2, axis=-1)

    rng = np.random.default_rng()
    sampler = mchammers.SamplerBasic(2**12, 32, 3, log_prob_func, np.array([1.65]))
    state_init = 2.0 * rng.random((sampler.num_walker, sampler.num_dim)) - 1.0
    sampler.run(state_init)
    std = np.std(sampler.samples, axis=(0, 1))
    assert np.all(std) < 1.2
