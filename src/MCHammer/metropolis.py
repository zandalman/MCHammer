from .hammer import Hammer
import numpy as np

class Metropolis(Hammer):
    def __init__(self,
        outfile_name,
        num_step,
        num_walk,
        num_param,
        log_prob_func,
        log_prob_args,
        initial,):
        
        Hammer.__init__(self,
        outfile_name,
        num_step,
        num_walk,
        num_param,
        log_prob_func,
        log_prob_args,
        initial,)

        self.std_proposal = 1.65
        self.frac_burn = 0.2
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
