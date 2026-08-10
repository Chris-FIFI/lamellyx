"""Quality checks on a built system.

The point of this module is rule 4 of the project's working notes: assert on
the quantity that would be wrong if the method failed. A build that produced
the right number of atoms in the right order can still be unusable because two
of them are 0.4 A apart, and only a contact check will say so.

Bonded pairs are excluded throughout. A 0.96 A O-H bond is not a clash, and a
contact metric that counts one tells you nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geom


def exclusion_pairs(mol, depth=2):
    """Encoded (i, j) pairs within `depth` bonds of each other, i < j.

    depth=2 covers 1-2 and 1-3 neighbours -- bonds and bond angles.
    """
    adj = mol.adjacency()
    n = mol.natoms
    out = []
    for i in range(n):
        frontier, seen = {i}, {i}
        for _ in range(depth):
            nxt = set()
            for a in frontier:
                nxt.update(adj[a])
            nxt -= seen
            seen |= nxt
            frontier = nxt
        for j in seen:
            if j > i:
                out.append((i, j))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def system_exclusions(blocks, natoms, depth=2):
    """Encoded exclusions for a whole system.

    `blocks` is the ordered [(MoleculeTopology, count)] the .gro was built
    from. Encoding is i*natoms + j so the result is one sorted int64 array.
    """
    chunks = []
    offset = 0
    for mol, count in blocks:
        pairs = exclusion_pairs(mol, depth)
        for _ in range(count):
            if len(pairs):
                chunks.append((pairs + offset))
            offset += mol.natoms
    if not chunks:
        return np.zeros(0, dtype=np.int64)
    allp = np.vstack(chunks)
    return np.sort(allp[:, 0] * natoms + allp[:, 1])


@dataclass
class ContactReport:
    min_distance: float = np.inf
    min_pair: tuple = ()
    n_below: dict = field(default_factory=dict)
    by_group: dict = field(default_factory=dict)
    heavy_min: float = np.inf
    heavy_below: int = 0
    worst_heavy: list = field(default_factory=list)   # (distance, label, label)

    def format(self, thresholds=(1.5, 2.0, 2.4), heavy_cut=2.4, show=6):
        lines = ["   closest non-bonded pair : %.3f A  %s"
                 % (self.min_distance, " -- ".join(self.min_pair))]
        lines.append("   closest heavy-heavy    : %.3f A" % self.heavy_min)
        for t in thresholds:
            lines.append("   pairs below %.1f A       : %d"
                         % (t, self.n_below.get(t, 0)))
        for k in sorted(self.by_group):
            lines.append("   %-22s : %.3f A" % (k, self.by_group[k]))
        if self.worst_heavy:
            lines.append("   heavy-atom pairs below %.1f A : %d"
                         % (heavy_cut, self.heavy_below))
            for d, a, b in self.worst_heavy[:show]:
                lines.append("      %.2f A  %s -- %s" % (d, a, b))
        return "\n".join(lines)


def contacts(atoms, box, exclusions=None, cutoff=2.6, groups=None,
             thresholds=(1.5, 2.0, 2.4)):
    """Closest non-bonded approaches in the system.

    `groups` is an optional {label: boolean mask} used to break the answer
    down by species pair, which is how you tell "the water is fine but a lipid
    is inside the protein" from "everything is fine".
    """
    xyz = atoms.xyz
    n = len(atoms)
    grid = geom.CellGrid(xyz, box, cutoff)
    qi, pj, d = grid.pairs(xyz, cutoff)
    keep = qi < pj
    qi, pj, d = qi[keep], pj[keep], d[keep]
    if exclusions is not None and len(exclusions):
        code = qi * n + pj
        qi, pj, d = (arr[~_isin_sorted(code, exclusions)] for arr in (qi, pj, d))

    rep = ContactReport()
    if len(d) == 0:
        rep.min_distance = float(cutoff)
        rep.min_pair = ("none below %.2f A" % cutoff, "")
        rep.heavy_min = float(cutoff)
        rep.n_below = {t: 0 for t in thresholds}
        return rep

    k = int(np.argmin(d))
    rep.min_distance = float(d[k])
    rep.min_pair = (_label(atoms, qi[k]), _label(atoms, pj[k]))
    rep.n_below = {t: int((d < t).sum()) for t in thresholds}

    heavy = atoms.element != "H"
    hm = heavy[qi] & heavy[pj]
    rep.heavy_min = float(d[hm].min()) if hm.any() else float(cutoff)
    # Heavy-atom contacts are the only ones worth naming. A 1.7 A O...H is a
    # hydrogen bond; a 1.7 A C...C is a mistake.
    hcut = hm & (d < 2.4)
    rep.heavy_below = int(hcut.sum())
    if hcut.any():
        hi, hj, hd = qi[hcut], pj[hcut], d[hcut]
        for k in np.argsort(hd)[:12]:
            rep.worst_heavy.append(
                (float(hd[k]), _label(atoms, hi[k]), _label(atoms, hj[k])))

    if groups:
        labels = list(groups)
        for a in range(len(labels)):
            for b in range(a, len(labels)):
                ma, mb = groups[labels[a]], groups[labels[b]]
                sel = (ma[qi] & mb[pj]) | (mb[qi] & ma[pj])
                if sel.any():
                    rep.by_group["%s-%s" % (labels[a], labels[b])] = \
                        float(d[sel].min())
    return rep


def _isin_sorted(values, sorted_ref):
    """Membership test against a pre-sorted array.

    The empty case is not hypothetical: a molecule with no bonds -- water in
    every CHARMM topology, and every monatomic ion -- produces no exclusions
    at all, and clamping an index into a zero-length array reads element -1.
    """
    values = np.asarray(values)
    if len(sorted_ref) == 0:
        return np.zeros(values.shape, dtype=bool)
    idx = np.searchsorted(sorted_ref, values)
    np.clip(idx, 0, len(sorted_ref) - 1, out=idx)
    return sorted_ref[idx] == values


def _label(atoms, i):
    return "%s%d:%s" % (atoms.resname[i], atoms.resid[i], atoms.name[i])


def bond_lengths(mol, xyz):
    """Every bond length in one molecule, for a sanity check on rebuilt atoms."""
    b = np.array(mol.bonds, dtype=np.int64)
    if len(b) == 0:
        return np.zeros(0)
    return np.linalg.norm(xyz[b[:, 0]] - xyz[b[:, 1]], axis=1)


def check_bond_lengths(mol, xyz, lo=0.85, hi=2.2):
    """Raise if any bond is impossible. Catches a mis-assigned atom instantly."""
    bl = bond_lengths(mol, xyz)
    bad = np.flatnonzero((bl < lo) | (bl > hi))
    if len(bad):
        b = np.array(mol.bonds)
        detail = ", ".join(
            "%s%s-%s%s %.2fA" % (mol.resid[b[k, 0]], mol.atomname[b[k, 0]],
                                 mol.resid[b[k, 1]], mol.atomname[b[k, 1]], bl[k])
            for k in bad[:8])
        raise ValueError("%s: %d impossible bond lengths (%s)"
                         % (mol.name, len(bad), detail))
    return bl


def density(atoms, box):
    """System density in g/cm3, as a blunt check that solvation worked."""
    mass = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974,
            "S": 32.06, "K": 39.098, "CL": 35.45, "NA": 22.99}
    m = np.array([mass.get(e.upper(), 12.0) for e in atoms.element]).sum()
    vol = float(np.prod(box)) * 1e-24          # A^3 -> cm3
    return m / 6.02214076e23 / vol


def core_water_report(atoms, box, protein_mask, lipid_mask, core_z,
                      water_resname="TIP3", oxygen="OH2", pore_radius=None):
    """Water sitting between the phosphate planes, and what it is near.

    This is reported because it is the one defect that passed every other
    check. A build can have perfect charge, density and contacts and still
    have hundreds of waters wedged into grooves on the lipid-facing surface
    of the protein, which minimisation then expels violently. Nothing in the
    counts shows it. This does.
    """
    from .solvate import enclosure

    lo, hi = core_z
    is_o = (atoms.resname == water_resname) & (atoms.name == oxygen)
    ow = atoms.xyz[is_o]
    inc = (ow[:, 2] > lo) & (ow[:, 2] < hi)
    out = {"count": int(inc.sum()), "slab_A": [round(float(lo), 2),
                                               round(float(hi), 2)]}
    if not inc.any():
        return out
    oc = ow[inc]
    lip = atoms.xyz[lipid_mask]
    pro = atoms.xyz[protein_mask]
    if len(lip):
        d = geom.CellGrid(lip, box, 12.0).min_distance(oc)
        out["min_lipid_distance_A"] = round(float(d.min()), 2)
        out["in_lipid_within_6A"] = int((d < 6.0).sum())
    if len(pro):
        e = enclosure(oc, pro, box)
        out["min_enclosure"] = round(float(e.min()), 2)
        out["poorly_enclosed_below_0.9"] = int((e < 0.9).sum())
        # Distance from the pore axis. The lipid-distance and enclosure tests
        # above both compare locally, so a groove on the outside of the
        # protein satisfies them; only a radius says "down the middle". On
        # HCN4 the offending waters were 12-42 A off-axis while every local
        # test called them buried pore water.
        d = oc[:, :2] - pro[:, :2].mean(0)
        for k in (0, 1):
            w = np.abs(d[:, k])
            d[:, k] = np.minimum(w, box[k] - w)
        r = np.hypot(d[:, 0], d[:, 1])
        out["max_radius_A"] = round(float(r.max()), 2)
        out["median_radius_A"] = round(float(np.median(r)), 2)
        if pore_radius:
            out["beyond_pore_radius"] = int((r > float(pore_radius)).sum())
    return out
