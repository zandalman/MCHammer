from __future__ import annotations
import numpy as np
import abc

__all__ = ["Hammer"]


def __dir__():
    return __all__


class Hammer(abc.ABC):
    """
    The Hammer object.

    Args
        state_init (np.ndarray): The initial state.
        std_prop (float):        The standard deviation of the proposal distirbution.
        num_step (int):          The number of steps.
        num_walk (int):          The number of walkers.
        num_dim (int):           The number of dimensions.
        path_outfile (str):      The path to the output HDF5 file.
        frac_burn (float):       The burn fraction.
        seed (int):              The random number generator seed.
        log_prob_args (dict):    The arguments for the log probability function.

    Attr
        idx_step (int):                       The index of the current step.
        rate_accept(float):                   The acceptance rate.
        state_curr (np.ndarray):              The current state.
        rng (np.random._generator.Generator): The random number generator.
        num_sample (int):                     The number of samples.
        samples (np.ndarray):                 The sample array.
    """
    def __init__(
            self, 
            state_init, 
            std_prop, 
            num_step, 
            num_walk, 
            num_dim, 
            path_outfile=None, 
            frac_burn=0.2, 
            seed=None, 
            *args,
            **kwargs
        ):
        
        assert (frac_burn >= 0) and (frac_burn <= 1), \
            "Burn fraction %.3g outside the allowed range [0, 1]." % frac_burn
        
        assert state_init.shape == (num_walk, num_dim), \
            "Initial state shape (%d, %d) does not match number of walkers and number of dimensions (%d, %d)" % \
            (state_init.shape[0], state_init.shape[1], num_walk, num_dim)
        
        self.state_init = state_init
        self.std_prop = std_prop
        self.num_step = num_step
        self.num_walk = num_walk
        self.num_dim = num_dim
        self.path_outfile = path_outfile
        self.frac_burn = frac_burn
        self.seed = seed
        self.log_prob_args = args
        self.log_prob_kwargs = kwargs
        
        self.idx_step = 0
        self.rate_accept = 0.0
        self.state_curr = self.state_init
        self.rng = np.random.default_rng(self.seed)
        self.num_step_burn = int(self.frac_burn * self.num_step)
        self.samples = np.zeros((self.num_step - self.num_step_burn, self.num_walk, self.num_dim))

    @abc.abstractmethod
    def log_prob(self, x, **kwargs):
        return 0.0
    
    def step(self):
        """ Step the Metropolis-Hastings algorithm. """
        # calculated the proposal state
        state_prop = self.state_curr + self.rng.normal(
            0, self.std_prop, size=(self.num_walk, self.num_dim)
        )

        # calculate the acceptance probability
        log_prob_curr = self.log_prob(self.state_curr, *self.log_prob_args, **self.log_prob_kwargs)
        log_prob_prop = self.log_prob(state_prop, *self.log_prob_args, **self.log_prob_kwargs)
        prob_accept = np.exp(log_prob_prop - log_prob_curr)
        
        # move accepted proposals into the current state
        cond_accept = self.rng.random(self.num_walk) < prob_accept
        self.state_curr[cond_accept] = state_prop[cond_accept]
        
        # if burn-in stage is over, add current state to samples
        if self.idx_step >= self.num_step_burn:
            self.rate_accept += np.sum(cond_accept)
            self.samples[self.idx_step - self.num_step_burn] = self.state_curr

        # increment the step index
        self.idx_step += 1

    def run(self):
        """
        Run the Metropolis-Hastings algorithm.
        """
        for _i in range(self.num_step - 1):
            self.step()

        # calculate the acceptance rate
        self.rate_accept /= (self.num_step - self.num_step_burn) * self.num_walk
