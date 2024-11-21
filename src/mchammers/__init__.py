"""
Copyright (c) 2024 The Princeton Hammers. All rights reserved.

mchammers: A toy package for sampling posteriors. Our final project for APC 524 at Princeton University.
"""

from __future__ import annotations

from ._version import version as __version__
from .hammer import Hammer

__all__ = ["__version__", "Hammer"]
