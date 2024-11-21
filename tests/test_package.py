from __future__ import annotations

import importlib.metadata

import mchammers as m


def test_version():
    assert importlib.metadata.version("mchammers") == m.__version__
