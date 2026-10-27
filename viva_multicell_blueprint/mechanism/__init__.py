"""Shared, backend-neutral mechanism components (spec sections 5, 6).

The Process classes are re-exported here so the workspace core's registration walk
(``core.build_core`` -> ``register_workspace_processes``) discovers them and
registers ``local:FieldProcess`` / ``local:RulesProcess`` / ``local:SystemicODEProcess``
/ ``local:ImmuneRecruitmentProcess``. Compiled composites address the shared
mechanism by that ``local:<name>`` form.
"""
from __future__ import annotations

from viva_multicell_blueprint.mechanism.fields import FieldProcess
from viva_multicell_blueprint.mechanism.immune import ImmuneRecruitmentProcess
from viva_multicell_blueprint.mechanism.rules import RulesProcess
from viva_multicell_blueprint.mechanism.systemic_ode import SystemicODEProcess

__all__ = [
    "FieldProcess",
    "RulesProcess",
    "SystemicODEProcess",
    "ImmuneRecruitmentProcess",
]
