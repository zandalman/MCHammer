from __future__ import annotations

import numpy as np


class Hammer:
    def __init__(
        self,
        outfile_name,
        num_step,
        num_walk,
        num_param,
        log_prob_func,
        log_prob_args,
        std_proposal,
        initial,
        frac_burn=0.2,
    ):
        """
        Initialize the Hammer object.

        Args
            outfile_name (string):    The name of the output HDF5 file.
            num_step (int):           The number of steps.
            num_walk (int):           The number of walkers.
            num_param (int):          The number of parameters.
            log_prob_func (function): The log probability function of the target distribution.
            log_prob_args (tuple):    The arguments for the log probability function of the target distribution.
            std_proposal (float):     The STD of the proposal distribution.
            initial (np.ndarray):     The array of initial walker positions.
            frac_burn (float):        The burn fraction.

        Attrs
            samples (np.ndarray): The array of samples.
            step_current (int):   The current step.
            rate_accept (float):  The acceptance rate.
        """
        self.outfile_name = outfile_name
        self.num_step = num_step
        self.num_walk = num_walk
        self.num_param = num_param
        self.log_prob_func = log_prob_func
        self.log_prob_args = log_prob_args
        self.std_proposal = std_proposal
        self.frac_burn = frac_burn
        self.initial = initial

        self.rng = np.random.default_rng()
        self.samples = np.zeros((self.num_step, self.num_walk, self.num_param))
        self.samples[0] = self.initial
        self.step_current = 0
        self.rate_accept = 0.0

    def step(self):
        """
        Step the Metropolis-Hastings algorithm.
        """
        current = self.samples[self.step_current]
        proposal = current + self.rng.normal(
            0, self.std_proposal, size=(self.num_walk, self.num_param)
        )

        log_prob_current = self.log_prob_func(current, *self.log_prob_args)
        log_prob_proposal = self.log_prob_func(proposal, *self.log_prob_args)
        prob_accept = np.exp(log_prob_proposal - log_prob_current)

        cond_accept = self.rng.random(self.num_walk) < prob_accept
        if self.step_current > self.frac_burn * self.num_step:
            self.rate_accept += np.sum(cond_accept)

        self.samples[self.step_current + 1] = current
        self.samples[self.step_current + 1, cond_accept] = proposal[cond_accept]

        self.step_current += 1

    def run(self):
        """
        Run the Metropolis-Hastings algorithm.
        """
        for _i in range(self.num_step - 1):
            self.step()

        self.samples = self.samples[int(self.frac_burn * self.num_step) :]
        self.rate_accept /= self.num_step * self.num_walk * (1 - self.frac_burn)
