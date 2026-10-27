"""Workspace composites package.

The ``sego_crossparadigm`` generator lives in :mod:`.sego`; importing this
package registers it (via that module's ``@composite_generator`` decorator) and
re-exports it for convenience.
"""
from .sego import sego_crossparadigm

__all__ = ["sego_crossparadigm"]
