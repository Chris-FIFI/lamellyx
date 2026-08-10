"""Extract the lipid and water libraries the builder ships with.

Run once against an equilibrated reference system. The output is coordinates
only -- conformers of a molecule a force field has already accepted -- which
is what lets the builder start a new bilayer from realistic shapes instead of
idealised ones.

    python -m lamellyx.make_data <gromacs_dir> [--lipid POPC]

Force-field parameters are NOT extracted. Those stay in whatever toppar
directory the user points the builder at.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

from . import bilayer, fileio, solvate, topology

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def lipid_path(resname):
    return os.path.join(DATA_DIR, "%s_conformers.npz" % resname.upper())


def water_path(resname):
    return os.path.join(DATA_DIR, "%s_cube.npz" % resname.upper())


def extract(gromacs_dir, lipid="POPC", water="TIP3", head="P", cube=20.0,
            seed=0, verbose=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    tops = topology.load_toppar(os.path.join(gromacs_dir, "toppar"))
    atoms, box = fileio.read_gro(os.path.join(gromacs_dir, "step5_input.gro"))

    lib = bilayer.extract_library(atoms, box, lipid, tops[lipid].natoms, head)
    # Store conformers relative to the bilayer midplane rather than to the
    # source box's absolute z, so a new bilayer can be built anywhere.
    out = lipid_path(lipid)
    np.savez_compressed(
        out,
        atomnames=np.array(lib.atomnames),
        head=np.int64(lib.head),
        upper=lib.upper.astype(np.float32),
        lower=lib.lower.astype(np.float32),
        head_offset=np.float64(lib.head_z_upper - lib.midplane),
        head_offset_lower=np.float64(lib.midplane - lib.head_z_lower),
        head_z_sd=np.float64(lib.head_z_sd),
        source=np.array(os.path.abspath(gromacs_dir)),
    )
    if verbose:
        print("%s: %d upper + %d lower conformers, head %.2f A from the "
              "midplane -> %s (%.0f kB)"
              % (lipid, len(lib.upper), len(lib.lower),
                 lib.head_z_upper - lib.midplane, out,
                 os.path.getsize(out) / 1024))

    tpl = solvate.extract_water_template(
        atoms, box, water, tops[water].natoms, cube=cube, rng=rng)
    outw = water_path(water)
    np.savez_compressed(
        outw,
        atomnames=np.array(tpl.atomnames),
        xyz=tpl.xyz.astype(np.float32),
        cell=np.float64(tpl.cell),
        source=np.array(os.path.abspath(gromacs_dir)),
    )
    if verbose:
        print("%s: %d molecules in a %.0f A cube -> %s (%.0f kB)"
              % (water, len(tpl.xyz), tpl.cell, outw,
                 os.path.getsize(outw) / 1024))
    return out, outw


def install(gromacs_dir, lipid="POPC", water="TIP3", head="P",
            toppar=True, verbose=True):
    """Populate `data/` from a reference system you already have locally.

    The published repository ships no `data/` directory. The conformers come
    out of an equilibrated CHARMM-GUI bilayer and the .itp files are CHARMM36
    parameters converted by CHARMM-GUI; neither is this package's work, and
    neither carries a licence that clearly permits redistributing it. Keeping
    them out of the repository makes the licensing question go away, at the
    cost of one command after install.

    Run it against any CHARMM-GUI GROMACS directory:

        python -m lamellyx setup <gromacs_dir>
    """
    out, outw = extract(gromacs_dir, lipid=lipid, water=water, head=head,
                        verbose=verbose)
    copied = []
    if toppar:
        src = os.path.join(gromacs_dir, "toppar")
        if not os.path.isdir(src):
            raise FileNotFoundError(
                "no toppar/ in %s -- point this at a CHARMM-GUI GROMACS "
                "directory, the one holding step5_input.gro" % gromacs_dir)
        dst = os.path.join(DATA_DIR, "toppar")
        os.makedirs(dst, exist_ok=True)
        want = {"%s.itp" % lipid.upper(), "%s.itp" % water.upper(),
                "POT.itp", "CLA.itp", "forcefield.itp"}
        for fn in sorted(os.listdir(src)):
            if fn in want:
                shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))
                copied.append(fn)
        missing = sorted(want - set(copied))
        if missing:
            raise FileNotFoundError(
                "%s has no %s -- this reference does not cover everything the "
                "bundled set needs" % (src, ", ".join(missing)))
        if verbose:
            print("toppar: copied %s -> %s" % (", ".join(copied), dst))
    return {"conformers": out, "water": outw, "toppar": copied,
            "data_dir": DATA_DIR, "source": os.path.abspath(gromacs_dir)}


def is_installed(data_dir=None):
    d = data_dir or DATA_DIR
    return (os.path.isdir(d)
            and any(f.endswith("_conformers.npz") for f in os.listdir(d))
            and os.path.isdir(os.path.join(d, "toppar")))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gromacs_dir", help="directory with toppar/ and step5_input.gro")
    p.add_argument("--lipid", default="POPC")
    p.add_argument("--water", default="TIP3")
    p.add_argument("--head", default="P", help="lipid reference atom")
    p.add_argument("--no-toppar", dest="toppar", action="store_false",
                   help="conformers only, leave the .itp files alone")
    install(**vars(p.parse_args(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
