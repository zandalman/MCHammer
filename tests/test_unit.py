from __future__ import annotations

import numpy as np

import mchammers


def test_gauss_hammer():
    """
    Test that the sampler recovers a multi-variate Gaussian
    distribution with approximately the correct standard deviation.
    """

    class GaussHammer(mchammers.hammer.Hammer):
        def log_prob(self, x, mu, sig):
            return -0.5 * np.sum(((x - mu) / sig) ** 2, axis=-1)

    std_prob = 1.65
    num_step = 2**12
    num_walk = 32
    num_dim = 3

    state_init = np.zeros((num_walk, num_dim))
    state_init[:] = np.linspace(-1, 1, num_walk)[:, None]

    gh = GaussHammer(std_prob, num_step, num_walk, num_dim, mu=0, sig=1)
    gh.run(state_init)
    std = np.std(gh.samples, axis=(0, 1))
    assert np.all(std) < 1.2
