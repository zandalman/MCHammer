from .hammer import Hammer
import numpy as np

class HMC(Hammer):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Hamiltonian Monte Carlo object.
        """
        self.name = "HMC"

    def step(self):
        """
        Step the Hamiltonian Monte Carlo algorithm.
        """
        return ...

    def run(self):
        """
        Run the Hamiltonian Monte Carlo algorithm.
        """
        return ...