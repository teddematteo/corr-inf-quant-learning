"""Compatibility helpers for Python 3.9."""

import sys
from dataclasses import dataclass as _dataclass
from itertools import zip_longest


def dataclass(*args, **kwargs):
    if sys.version_info < (3, 10):
        kwargs.pop("slots", None)
    return _dataclass(*args, **kwargs)


_count = getattr(int, "bit_count", lambda value: bin(value).count("1"))


def bit_count(value: int) -> int:
    return _count(int(value))


def strict_zip(*iterables):
    """Keep unequal-length checks on Python versions without zip(strict=True)."""
    sentinel = object()
    for values in zip_longest(*iterables, fillvalue=sentinel):
        if any(value is sentinel for value in values):
            raise ValueError("strict_zip arguments have different lengths")
        yield values
