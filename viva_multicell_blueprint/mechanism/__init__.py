"""Shared, backend-neutral mechanism components (spec sections 5, 6).

The Process classes are re-exported here so the workspace core's registration walk
(``core.build_core`` -> ``register_workspace_processes``) discovers them and
registers ``local:FieldProcess`` / ``local:RulesProcess`` / ``local:SystemicODEProcess``
/ ``local:ImmuneRecruitmentProcess``. Compiled composites address the shared
mechanism by that ``local:<name>`` form.
"""
from __future__ import annotations

from viva_multicell_blueprint.mechanism.actuation import CellActuationMergeStep
from viva_multicell_blueprint.mechanism.bounded_particles import BoundedParticlesProcess
from viva_multicell_blueprint.mechanism.counts import CellCountStep
from viva_multicell_blueprint.mechanism.fateful_tissue import FatefulTissueProcess
from viva_multicell_blueprint.mechanism.tri_coupled import TriCoupledProcess
from viva_multicell_blueprint.mechanism.fateful_consensus import FatefulConsensusProcess
from viva_multicell_blueprint.mechanism.particle_consensus import ParticleConsensusProcess
from viva_multicell_blueprint.mechanism.heterotypic_regions import HeterotypicRegionsProcess
from viva_multicell_blueprint.mechanism.fields import FieldProcess
from viva_multicell_blueprint.mechanism.immune import ImmuneRecruitmentProcess
from viva_multicell_blueprint.mechanism.rules import RulesProcess
from viva_multicell_blueprint.mechanism.subcellular import SubcellularProcess
from viva_multicell_blueprint.mechanism.systemic_ode import SystemicODEProcess

__all__ = [
    "CellActuationMergeStep",
    "BoundedParticlesProcess",
    "FatefulTissueProcess",
    "TriCoupledProcess",
    "FatefulConsensusProcess",
    "ParticleConsensusProcess",
    "HeterotypicRegionsProcess",
    "CellCountStep",
    "FieldProcess",
    "RulesProcess",
    "SystemicODEProcess",
    "ImmuneRecruitmentProcess",
    "SubcellularProcess",
]
