"""Loading the bundled lipid and water libraries.

The libraries are coordinates only: conformers of a lipid taken from an
equilibrated bilayer, and a cube of equilibrated water. They carry no force
field. Parameters come from a toppar directory the caller supplies, which is
the one thing this tool cannot invent and will not guess at.

To add a lipid the library does not have, run `make_data` against any
equilibrated system containing it.
"""

from __future__ import annotations

import os

import numpy as np

from .bilayer import LipidLibrary
from .solvate import WaterTemplate

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def available_lipids(data_dir=None):
    d = data_dir or DATA_DIR
    if not os.path.isdir(d):
        return []
    return sorted(f.split("_conformers.npz")[0] for f in os.listdir(d)
                  if f.endswith("_conformers.npz"))


def load_lipid_library(resname, midplane=0.0, data_dir=None):
    """Conformers for one lipid, positioned about a bilayer midplane at `z`."""
    d = data_dir or DATA_DIR
    path = os.path.join(d, "%s_conformers.npz" % resname.upper())
    if not os.path.exists(path):
        have = ", ".join(available_lipids(d)) or "none"
        raise FileNotFoundError(
            "no conformer library for %s (have: %s). Build one with "
            "`python -m lamellyx.make_data <gromacs_dir> --lipid %s`"
            % (resname, have, resname))
    z = np.load(path, allow_pickle=False)
    return LipidLibrary(
        resname=resname.upper(),
        atomnames=[str(a) for a in z["atomnames"]],
        head=int(z["head"]),
        upper=z["upper"].astype(np.float64),
        lower=z["lower"].astype(np.float64),
        head_z_upper=float(midplane + z["head_offset"]),
        head_z_lower=float(midplane - z["head_offset_lower"]),
        head_z_sd=float(z["head_z_sd"]),
        midplane=float(midplane),
    )


def load_water_template(resname="TIP3", data_dir=None):
    d = data_dir or DATA_DIR
    path = os.path.join(d, "%s_cube.npz" % resname.upper())
    if not os.path.exists(path):
        raise FileNotFoundError(
            "no water library for %s. Build one with "
            "`python -m lamellyx.make_data <gromacs_dir> --water %s`"
            % (resname, resname))
    z = np.load(path, allow_pickle=False)
    return WaterTemplate(
        xyz=z["xyz"].astype(np.float64),
        cell=float(z["cell"]),
        atomnames=[str(a) for a in z["atomnames"]],
        resname=resname.upper(),
    )


def lipid_extent(lib):
    """(below midplane, above midplane) reach of the lipids, in Angstrom."""
    up = lib.upper if len(lib.upper) else lib.lower
    lo = lib.lower if len(lib.lower) else lib.upper
    top = float((up[:, :, 2] + lib.head_z_upper).max())
    bot = float((lo[:, :, 2] + lib.head_z_lower).min())
    return bot, top
