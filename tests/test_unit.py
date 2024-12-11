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

    prior_bounds = [(-1, 1), (-1, 1), (-1, 1)]
    sampler = mchammers.SamplerBasic(2**12, 32, 3, prior_bounds, log_prob_func, 0.75)
    sampler.run()
    std = np.std(sampler.samples, axis=0)
    assert np.all(std) < 1.2
