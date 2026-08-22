"""Structure file reading and writing.

Everything inside the builder is in ANGSTROM. GROMACS .gro files are the only
place nanometres appear, and the conversion happens in this module and nowhere
else. Getting that wrong is silent and catastrophic, so it lives in one place.
"""

from __future__ import annotations

import numpy as np

# Atom-name prefixes that do not mean what the first letter says.
_TWO_LETTER = {"CL", "NA", "MG", "ZN", "FE", "CA", "MN", "CU", "SE", "BR"}
# Residue names that are a single ion, mapped to their element.
_ION_ELEMENT = {
    "POT": "K", "SOD": "NA", "CLA": "CL", "CAL": "CA", "MG": "MG",
    "ZN": "ZN", "CES": "CS", "RUB": "RB", "LIT": "LI",
}


def guess_element(name: str, resname: str = "") -> str:
    """Element from a CHARMM/PDB atom name.

    Ion residues are looked up by residue name -- an atom called ``CAL`` in
    residue ``CAL`` is calcium, but ``CA`` in residue ``LEU`` is a carbon.
    """
    resname = resname.strip().upper()
    if resname in _ION_ELEMENT:
        return _ION_ELEMENT[resname]
    n = name.strip().upper()
    if not n:
        return "X"
    # PDB puts a digit first for some hydrogens (1HB, 2HB ...)
    if n[0].isdigit():
        n = n[1:]
    if not n:
        return "X"
    if len(n) >= 2 and n[:2] in _TWO_LETTER and not resname:
        return n[:2]
    return n[0]


class Atoms:
    """A flat array-of-structures atom container.

    Deliberately not a general-purpose topology object. It holds exactly the
    fields the builder needs, all as parallel numpy arrays of the same length.
    """

    _FIELDS = ("name", "resname", "resid", "chain", "segid", "element")

    def __init__(self, name, resname, resid, xyz, chain=None, segid=None,
                 element=None):
        self.name = np.asarray(name, dtype="<U6")
        self.resname = np.asarray(resname, dtype="<U6")
        self.resid = np.asarray(resid, dtype=np.int64)
        self.xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        n = len(self.name)
        self.chain = (np.full(n, " ", dtype="<U1") if chain is None
                      else np.asarray(chain, dtype="<U1"))
        self.segid = (np.full(n, "", dtype="<U6") if segid is None
                      else np.asarray(segid, dtype="<U6"))
        if element is None:
            element = [guess_element(a, r) for a, r in zip(self.name, self.resname)]
        self.element = np.asarray(element, dtype="<U2")
        for f in self._FIELDS:
            if len(getattr(self, f)) != n:
                raise ValueError(f"field {f!r} has wrong length")
        if len(self.xyz) != n:
            raise ValueError("xyz has wrong length")

    def __len__(self):
        return len(self.name)

    def __getitem__(self, sel):
        # A bare integer would give numpy scalars and a zero-dimensional xyz,
        # so it is turned into a one-element selection rather than failing
        # later with "len() of unsized object".
        if isinstance(sel, (int, np.integer)):
            sel = [int(sel)]
        return Atoms(self.name[sel], self.resname[sel], self.resid[sel],
                     self.xyz[sel], self.chain[sel], self.segid[sel],
                     self.element[sel])

    def copy(self):
        return self[np.arange(len(self))]

    @staticmethod
    def concat(parts):
        parts = [p for p in parts if len(p) > 0]
        if not parts:
            return Atoms([], [], [], np.zeros((0, 3)))
        return Atoms(
            np.concatenate([p.name for p in parts]),
            np.concatenate([p.resname for p in parts]),
            np.concatenate([p.resid for p in parts]),
            np.concatenate([p.xyz for p in parts]),
            np.concatenate([p.chain for p in parts]),
            np.concatenate([p.segid for p in parts]),
            np.concatenate([p.element for p in parts]),
        )

    def residue_starts(self):
        """Indices where a new residue begins, assuming atoms are grouped."""
        if len(self) == 0:
            return np.zeros(0, dtype=int)
        changed = np.ones(len(self), dtype=bool)
        changed[1:] = ((self.resid[1:] != self.resid[:-1]) |
                       (self.resname[1:] != self.resname[:-1]) |
                       (self.chain[1:] != self.chain[:-1]) |
                       (self.segid[1:] != self.segid[:-1]))
        return np.flatnonzero(changed)


# --------------------------------------------------------------------------
# PDB
# --------------------------------------------------------------------------

def read_pdb(path, keep_hetatm=True, keep_altloc_first=True):
    names, resns, resids, xyz, chains, segids = [], [], [], [], [], []
    seen = set()
    with open(path) as fh:
        for line in fh:
            rec = line[:6]
            if rec == "ATOM  " or (keep_hetatm and rec == "HETATM"):
                alt = line[16]
                name = line[12:16].strip()
                resid = int(line[22:26])
                chain = line[21]
                if keep_altloc_first and alt not in (" ", "A"):
                    continue
                key = (chain, resid, line[26], name)
                if keep_altloc_first and key in seen:
                    continue
                seen.add(key)
                names.append(name)
                resns.append(line[17:21].strip())
                resids.append(resid)
                chains.append(chain)
                segids.append(line[72:76].strip())
                xyz.append((float(line[30:38]), float(line[38:46]),
                            float(line[46:54])))
    return Atoms(names, resns, resids, np.array(xyz).reshape(-1, 3),
                 chains, segids)


def write_pdb(path, atoms, box=None, title="built by lamellyx"):
    """Write a PDB. Atom and residue numbers wrap, as the format requires."""
    with open(path, "w", newline="\n") as fh:
        fh.write(f"REMARK    {title}\n")
        if box is not None:
            fh.write("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n"
                     % (box[0], box[1], box[2], 90.0, 90.0, 90.0))
        last_seg = None
        serial = 0
        for i in range(len(atoms)):
            seg = atoms.segid[i]
            if last_seg is not None and seg != last_seg:
                serial += 1
                fh.write("TER   %5d\n" % (serial % 100000))
            last_seg = seg
            serial += 1
            nm = atoms.name[i]
            # PDB centres names of <4 chars in columns 13-16
            nm4 = nm.ljust(4) if len(nm) >= 4 else (" " + nm).ljust(4)
            fh.write("ATOM  %5d %4s %-4s%1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f"
                     "      %-4s%2s\n" % (
                         serial % 100000, nm4, atoms.resname[i][:4],
                         atoms.chain[i] or " ", atoms.resid[i] % 10000,
                         atoms.xyz[i, 0], atoms.xyz[i, 1], atoms.xyz[i, 2],
                         1.0, 0.0, atoms.segid[i][:4],
                         atoms.element[i].rjust(2)))
        fh.write("END\n")


# --------------------------------------------------------------------------
# GRO  (the only place nanometres exist)
# --------------------------------------------------------------------------

def read_gro(path):
    """Return (Atoms in Angstrom, box in Angstrom)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    n = int(lines[1])
    rows = lines[2:2 + n]
    resid = np.fromiter((int(r[0:5]) for r in rows), dtype=np.int64, count=n)
    resname = np.array([r[5:10].strip() for r in rows], dtype="<U6")
    name = np.array([r[10:15].strip() for r in rows], dtype="<U6")
    xyz = np.array([[float(r[20:28]), float(r[28:36]), float(r[36:44])]
                    for r in rows]) * 10.0
    box = np.array([float(x) for x in lines[2 + n].split()[:3]]) * 10.0
    return Atoms(name, resname, resid, xyz), box


def write_gro(path, atoms, box, title="built by lamellyx"):
    """Write a .gro. `atoms` and `box` are in Angstrom; output is nanometres."""
    n = len(atoms)
    xyz = atoms.xyz / 10.0
    resid = atoms.resid % 100000
    out = [title, "%5d" % n]
    fmt = "%5d%-5s%5s%5d%8.3f%8.3f%8.3f".__mod__
    append = out.append
    for i in range(n):
        append(fmt((resid[i], atoms.resname[i][:5], atoms.name[i][:5],
                    (i + 1) % 100000, xyz[i, 0], xyz[i, 1], xyz[i, 2])))
    append("%10.5f%10.5f%10.5f" % (box[0] / 10.0, box[1] / 10.0, box[2] / 10.0))
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(out) + "\n")


def renumber_by_residue(atoms, start=1):
    """Give every residue a sequential number, restarting nowhere.

    GROMACS does not care what the numbers are, but duplicated numbers inside
    one molecule type make a .gro unreadable by eye, and by VMD.
    """
    starts = atoms.residue_starts()
    newid = np.zeros(len(atoms), dtype=np.int64)
    ends = np.append(starts[1:], len(atoms))
    for k, (s, e) in enumerate(zip(starts, ends)):
        newid[s:e] = start + k
    out = atoms.copy()
    out.resid = newid
    return out
