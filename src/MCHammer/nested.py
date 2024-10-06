from .hammer import Hammer
import numpy as np

class Nested(Hammer):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Nested Sampling Monte Carlo object.
        """
        self.name = "Nested"

    def step(self):
        """
        Step the Nested Sampling Monte Carlo algorithm.
        """
        return ...

    def run(self):
        """
        Run the Nested Sampling Monte Carlo algorithm.
        """
        return ...