"""Translating PDB atom and residue names into CHARMM's.

Only a handful of names actually differ, but each one that is missed becomes a
silently absent atom, so the translation is explicit and every unmatched name
is reported rather than dropped.
"""

from __future__ import annotations

# Atom names that differ between the PDB chemical component dictionary and
# the CHARMM36 topology, keyed by residue.
ATOM_MAP = {
    "ILE": {"CD1": "CD"},
    # PDB writes the second carboxyl-terminal oxygen as OXT; CHARMM's charged
    # C-terminus is OT1/OT2. With an amide cap neither exists and OXT is
    # dropped -- handled by DROP_IF_CAPPED below.
    "*": {"OXT": "OT2"},
}

# Residue names that differ. Histidine's CHARMM name encodes its protonation:
#   HSD  neutral, proton on ND1   (CHARMM-GUI's default at pH 7)
#   HSE  neutral, proton on NE2
#   HSP  doubly protonated, +1
RESIDUE_MAP = {"HIS": "HSD", "HID": "HSD", "HIE": "HSE", "HIP": "HSP"}

# Reverse, for reading a CHARMM structure back as PDB-named.
RESIDUE_MAP_INVERSE = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS"}

# Atoms in the input that have no counterpart once a terminal cap is applied.
DROP_IF_CAPPED = {"OXT", "OT1", "OT2"}


def map_residue(resname, his_variant="HSD"):
    resname = resname.strip().upper()
    if resname in ("HIS", "HID", "HIE", "HIP"):
        return RESIDUE_MAP.get(resname) if resname != "HIS" else his_variant
    return RESIDUE_MAP.get(resname, resname)


def map_atom(resname, atomname, capped=True):
    """Return the CHARMM atom name, or None if the atom should be dropped."""
    resname = resname.strip().upper()
    atomname = atomname.strip().upper()
    if capped and atomname in DROP_IF_CAPPED:
        return None
    per_res = ATOM_MAP.get(resname, {})
    if atomname in per_res:
        return per_res[atomname]
    if atomname in ATOM_MAP["*"]:
        return ATOM_MAP["*"][atomname]
    return atomname


def is_hydrogen(atomname):
    n = atomname.strip().upper()
    if not n:
        return False
    if n[0].isdigit():
        n = n[1:]
    return bool(n) and n[0] == "H"
