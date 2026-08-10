"""lamellyx -- scriptable lipid bilayer boxes for GROMACS.

A Python stand-in for CHARMM-GUI's Membrane Builder. One call gives a packed,
solvated, salted bilayer with topol.top, index.ndx and the CHARMM36
equilibration and production mdp series:

    from lamellyx import MembraneConfig, build_membrane
    build_membrane(MembraneConfig(output_dir="popc_box", x=8.0, y=8.0))

It does not invent force-field parameters. CHARMM36 parameters for POPC,
water and ions are bundled; anything else comes from a toppar directory the
caller points at.

Submodules are imported lazily so that a problem in one stage does not stop
the others being usable.
"""

__version__ = "1.0.0"

from .membrane import (MembraneConfig, MembraneResult, build_membrane,  # noqa: F401
                       check_settings)

__all__ = ["MembraneConfig", "MembraneResult", "build_membrane",
           "check_settings"]
