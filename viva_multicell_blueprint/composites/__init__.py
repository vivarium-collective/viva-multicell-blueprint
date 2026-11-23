"""Workspace composites package.

The ``sego_crossparadigm`` generator lives in :mod:`.sego`; importing this
package registers it (via that module's ``@composite_generator`` decorator) and
re-exports it for convenience. :mod:`.grammar_demo` adds the stage-2
``grammar_rules_demo`` generator (viva-grammar's rule engine swapped into the
rules slot).
"""
from .sego import sego_crossparadigm, sego_crossparadigm_large
from .pipeline import blueprint_pipeline
from .grammar_demo import grammar_rules_demo
from .bounded_particles import bounded_particles
from .scene_model import scene_model
from .coupled_subcellular import (coupled_fate_oscillator, coupled_particle_catalysis,
                                  coupled_triple_cell)
from .fateful_tissue import fateful_tissue
from .tri_coupled import tri_coupled

__all__ = ["sego_crossparadigm", "sego_crossparadigm_large", "grammar_rules_demo",
           "blueprint_pipeline", "bounded_particles", "scene_model",
           "coupled_fate_oscillator", "coupled_particle_catalysis", "coupled_triple_cell",
           "fateful_tissue", "tri_coupled"]
