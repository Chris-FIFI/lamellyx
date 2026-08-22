"""GROMACS topology handling: reading .itp, writing topol.top and index.ndx.

The builder does not generate force-field parameters. It reads the per-molecule
.itp files that already exist -- from CHARMM-GUI, from pdb2gmx, or from a lipid
library -- and uses them for three things: the exact atom order each molecule
must appear in, the bond graph needed to rebuild hydrogens, and the charges
needed to work out how many counter-ions to add.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field

import numpy as np

_SECTION = re.compile(r"^\s*\[\s*([A-Za-z_0-9]+)\s*\]")


@dataclass
class MoleculeTopology:
    """One `[ moleculetype ]` block."""

    name: str
    resid: list = field(default_factory=list)      # residue number, as written
    resname: list = field(default_factory=list)
    atomname: list = field(default_factory=list)
    charge: list = field(default_factory=list)
    mass: list = field(default_factory=list)
    bonds: list = field(default_factory=list)      # (i, j), zero-based

    @property
    def natoms(self):
        return len(self.atomname)

    @property
    def total_charge(self):
        return float(np.sum(self.charge))

    def keys(self):
        """(resid, atomname) in topology order -- the builder's atom identity."""
        return list(zip(self.resid, self.atomname))

    def adjacency(self):
        """Bond graph as a list of neighbour lists."""
        adj = [[] for _ in range(self.natoms)]
        for i, j in self.bonds:
            adj[i].append(j)
            adj[j].append(i)
        return adj


def parse_itp(path):
    """Return {molecule name: MoleculeTopology} for one .itp file.

    Preprocessor lines are honoured only far enough to skip #ifdef'd blocks
    that would otherwise be read as topology -- position restraints in
    particular, which sit in their own section and are not needed here.
    """
    mols, cur, section = {}, None, None
    with open(path) as fh:
        for raw in fh:
            line = raw.split(";")[0].rstrip()
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            m = _SECTION.match(line)
            if m:
                section = m.group(1).lower()
                if section == "moleculetype":
                    cur = None
                continue
            parts = line.split()
            # A short or non-numeric data line is corrupt input; turn the bare
            # IndexError/ValueError it would raise into one clear error naming
            # the file and line, so a bad .itp fails legibly rather than deep in
            # a build or a placement.
            try:
                if section == "moleculetype":
                    cur = MoleculeTopology(name=parts[0])
                    mols[cur.name] = cur
                elif section == "atoms" and cur is not None:
                    # nr type resnr residue atom cgnr charge mass
                    cur.resid.append(int(parts[2]))
                    cur.resname.append(parts[3])
                    cur.atomname.append(parts[4])
                    cur.charge.append(float(parts[6]) if len(parts) > 6 else 0.0)
                    cur.mass.append(float(parts[7]) if len(parts) > 7 else 0.0)
                elif section == "bonds" and cur is not None:
                    cur.bonds.append((int(parts[0]) - 1, int(parts[1]) - 1))
                elif section == "constraints" and cur is not None:
                    cur.bonds.append((int(parts[0]) - 1, int(parts[1]) - 1))
                elif section == "settles" and cur is not None:
                    # Rigid water declares no bonds at all, only a settles line
                    # naming its oxygen. Without this the two O-H distances look
                    # like non-bonded contacts at 0.96 A, and a contact report on
                    # a solvated box is then dominated by tens of thousands of
                    # water molecules being flagged as clashing with themselves.
                    o = int(parts[0]) - 1
                    cur.bonds += [(o, o + 1), (o, o + 2), (o + 1, o + 2)]
            except (IndexError, ValueError) as exc:
                raise ValueError("malformed [ %s ] line in %s: %r (%s)"
                                 % (section, path, line, exc))
    return mols


def load_toppar(directory):
    """Parse every .itp in a toppar directory. Returns {molname: topology}."""
    out = {}
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".itp") or fn == "forcefield.itp":
            continue
        try:
            out.update(parse_itp(os.path.join(directory, fn)))
        except Exception as exc:                     # noqa: BLE001
            raise RuntimeError(f"could not parse {fn}: {exc}") from exc
    return out


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

TOPOL_HEADER = """\
; Generated by lamellyx
;
; Molecule parameters are unchanged from the toppar directory they came from
; -- this file only records how many of each molecule the .gro contains.
"""


def write_topol(path, itp_includes, molecules, system_name="membrane system",
                prefix="toppar/"):
    """Write topol.top.

    `itp_includes` is the ordered list of .itp file names to #include, and
    `molecules` the ordered [(name, count)] -- which must match the order the
    molecules appear in the .gro, because GROMACS assumes it does.

    `prefix` is where the .itp files live relative to this file. Pointing it
    at a shared directory instead of a copied one saves the parameters being
    duplicated into every build.
    """
    lines = [TOPOL_HEADER, '#include "%sforcefield.itp"' % prefix]
    for inc in itp_includes:
        lines.append('#include "%s%s"' % (prefix, inc))
    lines.append("")
    lines.append("[ system ]")
    lines.append("; Name")
    lines.append(system_name)
    lines.append("")
    lines.append("[ molecules ]")
    lines.append("; Compound\t#mols")
    for name, count in molecules:
        if count:
            lines.append("%-8s%12d" % (name, count))
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def write_index(path, groups):
    """Write an index.ndx. `groups` is an ordered [(name, 0-based indices)]."""
    with open(path, "w", newline="\n") as fh:
        for name, idx in groups:
            fh.write("[ %s ]\n" % name)
            idx = np.asarray(idx, dtype=np.int64) + 1
            for s in range(0, len(idx), 15):
                fh.write(" ".join("%d" % v for v in idx[s:s + 15]) + "\n")
            fh.write("\n" if len(idx) == 0 else "")


def standard_groups(natoms, protein_slice, membrane_slice, solvent_slice):
    """The five groups CHARMM-GUI's mdp files reference.

    tc_grps is `SOLU MEMB SOLV` and comm_grps `SOLU_MEMB SOLV`, so all five
    must exist and together cover every atom exactly once.
    """
    a = np.arange(natoms)
    solu = a[protein_slice]
    memb = a[membrane_slice]
    solv = a[solvent_slice]
    covered = len(solu) + len(memb) + len(solv)
    if covered != natoms:
        raise ValueError(
            f"index groups cover {covered} atoms but the system has {natoms}")
    return [
        ("SOLU", solu),
        ("MEMB", memb),
        ("SOLV", solv),
        ("SOLU_MEMB", np.concatenate([solu, memb])),
        ("SYSTEM", a),
    ]


def membrane_groups(natoms, membrane_slice, solvent_slice):
    """Index groups for a bilayer with no protein.

    There is no SOLU group, because grompp rejects an empty one -- which is
    also why the mdp series for a pure membrane couples MEMB and SOLV only.
    """
    a = np.arange(natoms)
    memb, solv = a[membrane_slice], a[solvent_slice]
    if len(memb) + len(solv) != natoms:
        raise ValueError("index groups cover %d atoms but the system has %d"
                         % (len(memb) + len(solv), natoms))
    return [("MEMB", memb), ("SOLV", solv), ("SYSTEM", a)]


def copy_toppar(src, dst, only=None):
    """Copy the parameter files a build actually uses.

    `only` is the list of molecule names in the system. Without it the whole
    directory is copied, which for a CHARMM-GUI toppar means dragging four
    megabytes of protein topology into a build that contains no protein.
    Parameters are never regenerated, only copied.
    """
    if os.path.abspath(src) == os.path.abspath(dst):
        return []
    os.makedirs(dst, exist_ok=True)
    wanted = None if only is None else {"%s.itp" % m for m in only}
    copied = []
    for fn in sorted(os.listdir(src)):
        keep = fn.endswith((".itp", ".prm", ".str"))
        if keep and wanted is not None and fn.endswith(".itp"):
            keep = fn in wanted or fn == "forcefield.itp"
        if keep:
            shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
            copied.append(fn)
    return copied


def identical_molecules(tops, names):
    """Group molecule names whose topologies are the same molecule.

    A homotetramer built as PROA/PROB/PROC/PROD is four copies of one
    topology. Declaring it once and asking for four of it is both smaller on
    disk and closer to what the system actually is.
    """
    groups, seen = [], {}
    for n in names:
        m = tops[n]
        key = (m.natoms, tuple(m.atomname), tuple(m.resname),
               tuple(sorted(m.bonds)), round(m.total_charge, 6))
        if key in seen:
            groups[seen[key]].append(n)
        else:
            seen[key] = len(groups)
            groups.append([n])
    return groups


def directory_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total
