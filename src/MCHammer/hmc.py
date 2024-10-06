from __future__ import annotations

from .hammer import Hammer


class HMC(Hammer):
    def __init__(self):
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
