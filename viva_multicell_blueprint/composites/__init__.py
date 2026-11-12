"""Workspace composites package.

The ``sego_crossparadigm`` generator lives in :mod:`.sego`; importing this
package registers it (via that module's ``@composite_generator`` decorator) and
re-exports it for convenience. :mod:`.grammar_demo` adds the stage-2
``grammar_rules_demo`` generator (viva-grammar's rule engine swapped into the
rules slot).
"""
from .sego import sego_crossparadigm, sego_crossparadigm_large
from .grammar_demo import grammar_rules_demo

__all__ = ["sego_crossparadigm", "sego_crossparadigm_large", "grammar_rules_demo"]
