"""Prepare a ligand for CGenFF: PDB -> mol2 with pH-aware protonation.

CGenFF -- the ParamChem web server or the licensed binary -- reads a mol2 or
sdf, not a bare PDB, and it needs the molecule protonated the way it will be
simulated. A crystallographic or hand-drawn PDB is usually missing hydrogens and
says nothing about protonation state. This module drives Open Babel to add the
hydrogens appropriate for a pH (7.0 by default) and write the mol2 the web
server (or the binary) then parameterises.

Like pdb2gmx.py and the cgenff-binary path, this DRIVES a tool; it does not
reimplement its chemistry. Open Babel's pH treatment is a set of pKa rules, not
a calculation -- the same kind of model CHARMM-GUI's Ligand Reader leans on --
and its assignment for an unusual acid or base deserves a look. So the report
says which pH was used and how many hydrogens were added: the choice is made
visible, not buried.

The licence-free path this enables, end to end:

    lamellyx mol2 lig.pdb lig.mol2      # here: protonate at pH 7, write mol2
    # upload lig.mol2 to https://cgenff.silcsbio.com  ->  download lig.str
    lamellyx ligand lig.str out_dir     # our converter, no binary, no licence

> There is no Open Babel on this machine, so the invocation is tested against a
> stub, exactly as the cgenff binary is. Confirm the real `obabel` output on a
> known molecule before trusting a live run.
"""

from __future__ import annotations

import os
import shutil
import subprocess

OBABEL_CANDIDATES = ("obabel", "obabel.exe")


class OpenBabelNotFound(RuntimeError):
    pass


def find_obabel(obabel=None):
    """Locate the Open Babel executable. Raises OpenBabelNotFound naming what it
    tried and how to get it, rather than failing obscurely downstream."""
    tried = []
    names = [obabel] if obabel else []
    env = os.environ.get("OBABEL")
    if env:
        names.append(env)
    names += list(OBABEL_CANDIDATES)
    for name in names:
        if not name:
            continue
        tried.append(name)
        path = name if os.path.sep in name else shutil.which(name)
        if path and os.path.exists(path):
            return path
    raise OpenBabelNotFound(
        "no Open Babel executable found. Tried: %s. Install it (conda install "
        "-c conda-forge openbabel, or pip install openbabel-wheel), set OBABEL "
        "to its full path, or pass obabel=... . If you already have a "
        "protonated mol2/sdf, skip this step and take it straight to ParamChem."
        % ", ".join(tried))


def _read_pdb_atoms(path):
    """(n_atoms, n_hydrogens) from a PDB's ATOM/HETATM records. The element is
    read from columns 77-78 when present, else guessed from the atom name."""
    n, nh = 0, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                n += 1
                elem = line[76:78].strip() if len(line) >= 78 else ""
                if not elem:
                    name = line[12:16].strip()
                    elem = name[0] if name else ""
                if elem.upper() == "H":
                    nh += 1
    return n, nh


def _read_mol2_atoms(path):
    """(n_atoms, n_hydrogens) from a mol2's @<TRIPOS>ATOM block. The element is
    the part of the SYBYL atom type before the dot (C.3 -> C, H -> H)."""
    n, nh, in_atoms = 0, 0, False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("@<TRIPOS>"):
                in_atoms = s.upper() == "@<TRIPOS>ATOM"
                continue
            if in_atoms and s:
                parts = s.split()
                if len(parts) >= 6:
                    n += 1
                    if parts[5].split(".")[0].upper() == "H":
                        nh += 1
    return n, nh


def pdb_to_mol2(pdb, out_mol2, ph=7.0, obabel=None, add_hydrogens=True,
                extra_args=()):
    """Convert a ligand PDB to a mol2, protonated for `ph` (default 7.0).

    Drives `obabel <pdb> -O <out_mol2> -p <ph>`: the `-p` flag adds the
    hydrogens Open Babel judges appropriate at that pH. With add_hydrogens
    False the structure is converted as-is (no `-p`), for a PDB that is already
    correctly protonated. Returns a report naming the pH and the hydrogen count
    change, so the protonation decision is visible.
    """
    exe = find_obabel(obabel)
    pdb = os.path.abspath(pdb)
    if not os.path.exists(pdb):
        raise FileNotFoundError(pdb)
    out_mol2 = os.path.abspath(out_mol2)
    os.makedirs(os.path.dirname(out_mol2) or ".", exist_ok=True)

    cmd = [exe, pdb, "-O", out_mol2]
    if add_hydrogens:
        cmd += ["-p", "%g" % ph]
    cmd += list(extra_args)

    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("obabel did not finish within 600 s (a hang?). "
                           "command: %s" % " ".join(cmd))
    if run.returncode != 0:
        tail = (run.stderr or run.stdout or "").strip().splitlines()[-25:]
        raise RuntimeError(
            "obabel failed (exit %d).\ncommand: %s\n\n%s"
            % (run.returncode, " ".join(cmd), "\n".join(tail)))

    # Open Babel can exit 0 having converted nothing (unreadable input); a mol2
    # that is missing or has no atom block is that failure, not an empty result.
    if not os.path.exists(out_mol2):
        raise RuntimeError(
            "obabel exited 0 but wrote no %s.\ncommand: %s\n\n%s"
            % (out_mol2, " ".join(cmd), (run.stderr or "").strip()))
    n_mol2, nh_mol2 = _read_mol2_atoms(out_mol2)
    if n_mol2 == 0:
        raise RuntimeError(
            "obabel wrote a mol2 with no atoms -- it likely could not read the "
            "PDB. command: %s\n\n%s" % (" ".join(cmd), (run.stderr or "").strip()))

    n_pdb, nh_pdb = _read_pdb_atoms(pdb)
    return {
        "ok": True,
        "mol2": out_mol2,
        "obabel": exe,
        "ph": ph if add_hydrogens else None,
        "pdb_atoms": n_pdb,
        "mol2_atoms": n_mol2,
        "hydrogens_added": nh_mol2 - nh_pdb,
        "note": (
            (("Hydrogens added by Open Babel for pH %.1f. Its pH treatment is a "
              "set of pKa rules, not a calculation -- check the state of any "
              "acid, base or unusual group before parameterising. " % ph)
             if add_hydrogens else
             "Converted as-is; no hydrogens added for a pH. ")
            + "Next: upload this mol2 to the ParamChem server "
              "(cgenff.silcsbio.com, no licence) for a .str, then run "
              "`lamellyx ligand`."),
    }


def generate_ligand_topology_from_pdb(pdb, out_dir, ph=7.0, resname=None,
                                      cgenff_ff=None, penalty_flag=None,
                                      obabel=None, cgenff=None, output_flag=None,
                                      extra_args=(), keep_mol2=True):
    """PDB -> mol2 (Open Babel, pH-protonated) -> cgenff binary -> .str ->
    GROMACS topology, in one call.

    The fully-automatic path, which needs BOTH Open Babel and a CGenFF licence.
    Without a licence, run `pdb_to_mol2` alone and take the mol2 to the ParamChem
    web server. Returns the topology report, extended with the mol2 and the
    protonation summary.
    """
    from . import cgenff as _cgenff       # module; `cgenff` is the binary path

    # Validate resname before it is joined onto a path below -- otherwise a
    # "../x" would escape out_dir through the mol2 name, before the converter
    # (which also checks) ever sees it.
    if resname:
        _cgenff._check_resname(resname)
    os.makedirs(out_dir, exist_ok=True)
    mol2 = os.path.join(out_dir, (resname or "ligand") + ".mol2")
    prep_report = pdb_to_mol2(pdb, mol2, ph=ph, obabel=obabel)
    report = _cgenff.generate_ligand_topology_from_molecule(
        mol2, out_dir, resname=resname, cgenff_ff=cgenff_ff,
        penalty_flag=penalty_flag, cgenff=cgenff, output_flag=output_flag,
        extra_args=extra_args)
    if not keep_mol2:
        os.remove(mol2)
    else:
        report["mol2"] = mol2
    report["ph"] = prep_report["ph"]
    report["hydrogens_added"] = prep_report["hydrogens_added"]
    report["obabel"] = prep_report["obabel"]
    return report
