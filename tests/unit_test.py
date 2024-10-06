from __future__ import annotations

import numpy as np

import MCHammer


def test_sampler():
    num_step = 2**10
    num_walk = 32
    num_dim = 3

    def log_prob_func(x, mu, sig):
        return -0.5 * np.sum(((x - mu) / sig) ** 2, axis=-1)

    initial = np.zeros((num_walk, num_dim))
    initial[:] = np.linspace(-1, 1, num_walk)[:, None]

    hammer = MCHammer.Hammer(
        "placeholder.h5",
        num_step,
        num_walk,
        num_dim,
        log_prob_func,
        (np.zeros(num_dim), np.ones(num_dim)),
        1.65,
        initial,
        frac_burn=0.2,
    )

    hammer.run()
    std = np.std(hammer.samples.reshape((-1, num_dim)))

    assert np.all(std) < 2.0
