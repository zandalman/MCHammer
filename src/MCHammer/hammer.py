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
        initial,
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
            initial (np.ndarray):     The array of initial walker positions.
            std_proposal (float):     The STD of the proposal distribution.
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
        self.initial = initial

        self.rng = np.random.default_rng()
        self.samples = np.zeros((self.num_step, self.num_walk, self.num_param))
        self.samples[0] = self.initial
        self.step_current = 0
