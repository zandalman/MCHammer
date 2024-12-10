from __future__ import annotations

import numpy as np

import mchammers


def test_sampler_stretch():
    def log_prob_func(x, mu=0, sig=1):
        return -0.5 * np.sum(((x - mu) / sig) ** 2, axis=-1)

    sampler = mchammers.SamplerStretch(
        num_step=2**12,
        num_walker=32,
        num_dim=3,
        log_prob_func=log_prob_func,
        a=2,  # a=2 is implemented in GW10
        frac_burn=0.2,
    )
    state_init = sampler.rng.normal(
        loc=0, scale=1, size=(sampler.num_walker, sampler.num_dim)
    )
    sampler.run(state_init)

    std = np.std(sampler.samples, axis=0)
    print(f"Standard deviation of samples: {std}")
    assert np.all(std < 1.2), f"Standard deviation out of bounds: {std}"
    return sampler.samples