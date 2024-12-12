from __future__ import annotations

import contextlib

import numpy as np

import mchammers


def test_sampler_base_class():
    def log_prob_func(x, mu=0, sig=1):
        return -0.5 * np.sum(((x - mu) / sig) ** 2, axis=-1)

    rng = np.random.default_rng()
    state_init = rng.normal(0, 1, (32, 3))

    prior_bounds = [(-1, 1), (-1, 1), (-1, 1)]
    sampler = mchammers.SamplerBasic(
        2**12, 32, 3, prior_bounds, state_init, log_prob_func, 0.75
    )

    assert type(sampler.num_dim) is int
    assert type(sampler.num_step) is int
    assert type(sampler.num_walker) is int
    assert sampler.state_init.shape == (32, 3)

    sampler.rate_accept = 1e6
    with contextlib.suppress(UserWarning):
        assert sampler.calc_rate_accept() is None  # type: ignore[func-returns-value]

    sampler.rate_accept = 0
    with contextlib.suppress(UserWarning):
        assert sampler.calc_rate_accept() is None  # type: ignore[func-returns-value]
