"""Relieving side-chain clashes without moving the backbone.

Homology models routinely arrive with side chains overlapping, most often
across a symmetry interface where two copies of the same rotamer meet. In the
HCN4 tetramer, ILE517 CD1 sits 1.80 A from HIS512 CE1 of the neighbouring
subunit -- an impossible contact that the model carries in all four corners.

CHARMM-GUI removes these quietly by minimising, which is why its output is
about 0.25 A RMSD from the model it was given. This module does the same job
in a way that can be checked: it rotates side-chain torsions only. Bond
lengths and bond angles are untouched by construction, the backbone does not
move at all, and every rotation is reported.

A torsion is rotatable if turning it moves only side-chain atoms of one
residue -- so chi1 onwards, never the backbone, and never a ring bond.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geom, hbuild, validate

BACKBONE = {"N", "HN", "H", "CA", "HA", "HA1", "HA2", "C", "O",
            "CAY", "HY1", "HY2", "HY3", "CY", "OY",
            "NT", "HNT", "CAT", "HT1", "HT2", "HT3", "OT1", "OT2", "OXT"}

# Atoms belonging to a terminal patch rather than to the residue proper.
CAP = {"CAY", "HY1", "HY2", "HY3", "CY", "OY", "HN",
       "NT", "HNT", "CAT", "HT1", "HT2", "HT3"}


@dataclass
class RefineReport:
    n_clashes_before: int = 0
    n_clashes_after: int = 0
    worst_before: float = np.inf
    worst_after: float = np.inf
    rotations: list = field(default_factory=list)   # (resid, resname, degrees)
    rmsd: float = 0.0

    def summary(self):
        s = ("   clashes under %.1f A: %d -> %d ; closest %.2f -> %.2f A ; "
             "%d side chains rotated, RMSD %.3f A")
        return s % (0.0, self.n_clashes_before, self.n_clashes_after,
                    self.worst_before, self.worst_after, len(self.rotations),
                    self.rmsd)


def sidechain_torsions(mol, max_moving=40):
    """Rotatable side-chain bonds as (axis_from, axis_to, moving atoms).

    Rotating `moving` about the axis leaves every bond length and bond angle
    in the molecule unchanged, so the result is always chemically valid.
    """
    adj = mol.adjacency()
    heavy = [not n.upper().startswith("H") for n in mol.atomname]
    resid = list(mol.resid)
    out = []
    for i, j in mol.bonds:
        if not (heavy[i] and heavy[j]):
            continue
        if resid[i] != resid[j]:
            continue
        for a, b in ((i, j), (j, i)):
            mv = hbuild.rotating_set(adj, b, a, limit=max_moving)
            if not mv:
                continue
            if any(resid[m] != resid[a] for m in mv):
                continue
            names_mv = {mol.atomname[m] for m in mv}
            # An N-terminal cap hangs off the N-CA bond, so turning that bond
            # moves the cap and nothing else -- it is the terminus's phi, and
            # it is free. Copying it from a reference conformation is exactly
            # what puts the cap carbonyl on top of the residue's own, so it
            # has to stay adjustable even though its atoms are named like
            # backbone. The C-terminal cap is not included: the bond there is
            # the amide, and rotating it would break planarity.
            is_cap = (names_mv <= CAP
                      and {mol.atomname[a], mol.atomname[b]} == {"CA", "N"})
            if not is_cap and any(n in BACKBONE for n in names_mv):
                continue
            if not any(heavy[m] for m in mv):
                continue          # pure hydrogen rotor, hbuild handled it
            out.append((a, b, np.array(sorted(mv), dtype=np.int64)))
            break
    return out


def _residue_atoms(mol):
    """{resid: index array} for one molecule."""
    out = {}
    for i, r in enumerate(mol.resid):
        out.setdefault(int(r), []).append(i)
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}


def relieve_clashes(xyz, mols, box, pbc=(True, True, True), clash=2.45,
                    reach=2.90, sweeps=3, step_deg=10.0, bias=0.02,
                    verbose=True):
    """Rotate clashing side chains apart, leaving everything else alone.

    Two distances, and the difference between them is the whole design:

    `clash` is what counts as broken. Below about 2.45 A there is no
    legitimate heavy-atom contact in a protein, so a pair closer than that
    marks a residue for repair. Above it there are thousands of real ones --
    a hydrogen bond is 2.8 to 3.0 A -- and a repair pass that treats those as
    errors will dismantle the fold. It is worth stating the failure mode
    plainly, because it is not obvious and it is silent: at a 3.2 A threshold
    this function rotates two and a half thousand side chains, moves the
    structure 1.6 A, and makes the worst contact worse.

    `reach` is how far the score can see. The penalty is quartic in the
    overlap, so a 1.8 A clash counts about fourteen thousand times a 2.8 A
    hydrogen bond, and the search fixes the first without noticing the second.
    """
    xyz = np.array(xyz, dtype=np.float64, copy=True)
    xyz0 = xyz.copy()
    n = len(xyz)

    offsets, off = [], 0
    for m in mols:
        offsets.append(off)
        off += m.natoms
    if off != n:
        raise ValueError("coordinates do not match the molecules given")

    blocks = [(m, 1) for m in mols]
    excl = validate.system_exclusions(blocks, n)
    heavy = np.concatenate(
        [[not a.upper().startswith("H") for a in m.atomname] for m in mols])
    resid_of = np.concatenate([m.resid for m in mols])
    molid_of = np.concatenate([[k] * m.natoms for k, m in enumerate(mols)])
    resname_of = np.concatenate([m.resname for m in mols])

    tors = {k: sidechain_torsions(m) for k, m in enumerate(mols)}
    res_atoms = {k: _residue_atoms(m) for k, m in enumerate(mols)}
    rep = RefineReport()

    def find_clashes(x):
        grid = geom.CellGrid(x, box, clash, pbc)
        qi, pj, d = grid.pairs(x, clash)
        k = qi < pj
        qi, pj, d = qi[k], pj[k], d[k]
        k = heavy[qi] & heavy[pj]
        qi, pj, d = qi[k], pj[k], d[k]
        k = ~validate._isin_sorted(qi * n + pj, excl)
        return qi[k], pj[k], d[k]

    def severity(x):
        """One number for how broken the structure is: count and depth."""
        _, _, d = find_clashes(x)
        if len(d) == 0:
            return 0.0, 0, np.inf
        return (float((np.maximum(0.0, clash - d) ** 2).sum()), int(len(d)),
                float(d.min()))

    # 1-2 and 1-3 partners per atom, so a rotation is not scored against the
    # atoms it is rigidly attached to
    near = {}
    for k, m in enumerate(mols):
        adj = m.adjacency()
        per = []
        for i in range(m.natoms):
            s = {i}
            for a in adj[i]:
                s.add(a)
                s.update(adj[a])
            per.append(s)
        near[k] = per

    sev0, n0, w0 = severity(xyz)
    rep.n_clashes_before, rep.worst_before = n0, w0
    if verbose:
        print("   %d heavy-atom pairs closer than %.2f A, worst %.2f A"
              % (n0, clash, w0 if n0 else clash))
    if n0 == 0:
        rep.n_clashes_after, rep.worst_after = 0, np.inf
        return xyz, rep

    best_xyz, best_sev, best_n, best_w = xyz.copy(), sev0, n0, w0
    angles = np.deg2rad(np.arange(step_deg, 360.0, step_deg))

    for sweep in range(sweeps):
        qi, pj, d = find_clashes(xyz)
        if len(d) == 0:
            break
        involved = {}
        for a, b, dd in zip(qi, pj, d):
            for t in (a, b):
                key = (int(molid_of[t]), int(resid_of[t]))
                involved[key] = min(involved.get(key, np.inf), float(dd))
        order = sorted(involved, key=lambda k: involved[k])

        moved = 0
        for mi, rid in order:
            cands = [(a, b, mv) for a, b, mv in tors[mi]
                     if len(mv) and int(m_resid(mols, mi, mv[0])) == rid]
            if not cands:
                continue
            for a, b, mv in reversed(cands):        # outermost torsion first
                mv_g = mv + offsets[mi]
                hv = mv_g[heavy[mv_g]]
                if not len(hv):
                    continue
                origin = xyz[b + offsets[mi]]
                axis = origin - xyz[a + offsets[mi]]
                if np.linalg.norm(axis) < 1e-6:
                    continue

                own = set(int(x) for x in mv_g)
                for t in mv:
                    own.update(int(q) + offsets[mi] for q in near[mi][t])
                own = np.array(sorted(own), dtype=np.int64)

                # a fresh grid each time: scoring against where the last
                # rotation used to be is how a repair pass makes things worse
                grid = geom.CellGrid(xyz, box, reach, pbc)

                def score(pos, _own=own, _g=grid):
                    _, pjj, dd = _g.pairs(pos, reach)
                    if len(dd) == 0:
                        return 0.0
                    keep = ~np.isin(pjj, _own)
                    if not keep.any():
                        return 0.0
                    return float((np.maximum(0.0, reach - dd[keep]) ** 4).sum())

                base = score(xyz[hv])
                if base <= 1e-12:
                    continue
                best_ang, best_s = 0.0, base
                for ang in angles:
                    trial = hbuild._rotate_about(xyz[hv], origin, axis, ang)
                    s = score(trial) + bias * (1.0 - np.cos(ang))
                    if s < best_s - 1e-9:
                        best_s, best_ang = s, float(ang)
                if best_ang:
                    trial_all = xyz.copy()
                    trial_all[mv_g] = hbuild._rotate_about(
                        xyz[mv_g], origin, axis, best_ang)
                    sev, nn, ww = severity(trial_all)
                    if sev < best_sev - 1e-12:      # only if globally better
                        xyz = trial_all
                        best_sev, best_n, best_w = sev, nn, ww
                        best_xyz = xyz.copy()
                        rep.rotations.append(
                            (rid, str(resname_of[b + offsets[mi]]),
                             round(np.rad2deg(best_ang), 1)))
                        moved += 1
        if verbose:
            print("   sweep %d: %d side chains rotated, %d pairs left, "
                  "worst %.2f A" % (sweep, moved, best_n,
                                    best_w if best_n else clash))
        if moved == 0:
            break

    xyz = best_xyz
    rep.n_clashes_after, rep.worst_after = best_n, best_w
    rep.rmsd = geom.rmsd(xyz, xyz0)
    return xyz, rep


def m_resid(mols, mi, local_index):
    return mols[mi].resid[local_index]
