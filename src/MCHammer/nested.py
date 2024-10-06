from __future__ import annotations

from .hammer import Hammer


class Nested(Hammer):
    def __init__(self):
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
