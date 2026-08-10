"""Rebuilding hydrogens and terminal caps onto a heavy-atom model.

A homology model or a trimmed crystal structure has heavy atoms only, and no
ACE/CT3 caps. The topology wants all of them, in a fixed order. This module
fills in every atom the topology asks for and the model does not have.

The method is the one CHARMM's HBUILD uses: an atom's position is defined by a
bond length, a bond angle and a dihedral relative to three atoms that are
already placed. Bond lengths and angles are rigid, so they transfer exactly
from a reference structure; dihedrals do not, so anchors are chosen to avoid
depending on one wherever that is possible.

For an atom `a` bonded to `b`:

  * if `b` has two other placed neighbours, the frame (b; c, d) fixes `a`
    completely -- no torsion is involved and the transfer is exact. This is
    what happens for every hydrogen on a carbon that carries two or more
    heavy substituents, which is most of them.

  * otherwise the frame has to reach out to a 1-4 atom and the dihedral is
    copied from the reference. That happens for methyl, hydroxyl, thiol and
    ammonium hydrogens, and for the caps -- all rotations that are free or
    very nearly free. `optimise_rotatable` afterwards spins each of these
    groups to the setting with the fewest close contacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geom, names as naming

# A frame built from three nearly-collinear atoms is numerically useless.
MIN_SIN = 0.15
# Anchors further away than this are not a real bonded neighbourhood.
MAX_ANCHOR = 4.0


@dataclass
class RebuildReport:
    molecule: str = ""
    n_atoms: int = 0
    n_known: int = 0
    n_rigid: int = 0
    n_torsional: int = 0
    failed: list = field(default_factory=list)          # topology indices
    unmatched_input: list = field(default_factory=list)  # (resid, name)
    built_heavy: list = field(default_factory=list)      # (resid, name)
    max_anchor_dist: float = 0.0

    def summary(self):
        return ("%-6s %5d atoms: %5d from the model, %4d placed rigidly, "
                "%4d by copied torsion, %d failed"
                % (self.molecule, self.n_atoms, self.n_known, self.n_rigid,
                   self.n_torsional, len(self.failed)))


def match_to_topology(mol, atoms, his_variant="HSD", capped=True):
    """Map a heavy-atom model onto topology atom order.

    Returns (known index array, coordinates for those indices, report bits).
    Matching is by (residue number, CHARMM atom name); anything the topology
    does not contain is reported rather than dropped silently.
    """
    want = {}
    for i, (rid, nm) in enumerate(zip(mol.resid, mol.atomname)):
        want[(int(rid), nm)] = i

    known_idx, known_xyz, unmatched = [], [], []
    seen = set()
    for i in range(len(atoms)):
        rid = int(atoms.resid[i])
        cnm = naming.map_atom(atoms.resname[i], atoms.name[i], capped=capped)
        if cnm is None:
            continue
        key = (rid, cnm)
        j = want.get(key)
        if j is None or j in seen:
            unmatched.append((rid, atoms.name[i]))
            continue
        seen.add(j)
        known_idx.append(j)
        known_xyz.append(atoms.xyz[i])
    return (np.array(known_idx, dtype=np.int64),
            np.array(known_xyz, dtype=np.float64).reshape(-1, 3),
            unmatched)


def _choose_anchors(a, adj, placed, ref):
    """Pick (p1, p2, p3, is_rigid) for atom `a`, or None if not yet possible."""
    parents = [b for b in adj[a] if placed[b]]
    if not parents:
        return None
    # Prefer the heaviest-connected parent: it will have the most siblings.
    parents.sort(key=lambda b: -len(adj[b]))
    for b in parents:
        sibs = [c for c in adj[b] if placed[c] and c != a]
        # rigid: two other substituents of the same parent
        if len(sibs) >= 2:
            best, best_sin = None, MIN_SIN
            for ii in range(len(sibs)):
                for jj in range(ii + 1, len(sibs)):
                    s = float(geom.collinearity(ref[b], ref[sibs[ii]],
                                                ref[sibs[jj]])[0])
                    if s > best_sin:
                        best, best_sin = (sibs[ii], sibs[jj]), s
            if best is not None:
                return b, best[0], best[1], True
        # torsional: reach to a 1-4 atom
        for c in sibs:
            for d in adj[c]:
                if d != a and d != b and placed[d]:
                    if float(geom.collinearity(ref[b], ref[c], ref[d])[0]) > MIN_SIN:
                        return b, c, d, False
    return None


def rebuild(mol, ref_xyz, known_idx, known_xyz):
    """Fill in every topology atom absent from the model.

    `ref_xyz` is a complete reference conformation of the same molecule, in
    topology order. Returns (xyz in topology order, RebuildReport).
    """
    n = mol.natoms
    ref = np.asarray(ref_xyz, dtype=np.float64).reshape(-1, 3)
    if len(ref) != n:
        raise ValueError(f"reference has {len(ref)} atoms, topology wants {n}")

    xyz = np.full((n, 3), np.nan)
    placed = np.zeros(n, dtype=bool)
    xyz[known_idx] = known_xyz
    placed[known_idx] = True

    rep = RebuildReport(molecule=mol.name, n_atoms=n, n_known=int(placed.sum()))
    for j in np.flatnonzero(~placed):
        if not naming.is_hydrogen(mol.atomname[j]):
            rep.built_heavy.append((int(mol.resid[j]), mol.atomname[j]))

    adj = mol.adjacency()
    torsional = set()       # (b, c) bonds whose dihedral had to be copied
    todo = list(np.flatnonzero(~placed))

    while todo:
        progressed = False
        deferred = []
        for a in todo:
            found = _choose_anchors(a, adj, placed, ref)
            if found is None:
                deferred.append(a)
                continue
            p1, p2, p3, rigid = found
            o_r, R_r = geom.frames(ref[p1], ref[p2], ref[p3])
            local = geom.to_local(ref[a], o_r, R_r)
            o_t, R_t = geom.frames(xyz[p1], xyz[p2], xyz[p3])
            xyz[a] = geom.to_world(local, o_t, R_t)[0]
            placed[a] = True
            progressed = True
            rep.max_anchor_dist = max(
                rep.max_anchor_dist,
                float(np.linalg.norm(ref[a] - ref[p3])))
            if rigid:
                rep.n_rigid += 1
            else:
                rep.n_torsional += 1
                torsional.add((p1, p2))
        if not progressed:
            rep.failed = [int(x) for x in deferred]
            break
        todo = deferred

    if np.isnan(xyz).any() and not rep.failed:
        rep.failed = [int(i) for i in np.flatnonzero(np.isnan(xyz).any(1))]
    return xyz, rep, torsional


# --------------------------------------------------------------------------
# torsion cleanup
# --------------------------------------------------------------------------

def _rotate_about(points, origin, axis, angle):
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    v = points - origin
    return (origin + v * c + np.cross(axis, v) * s
            + axis * (v @ axis)[:, None] * (1.0 - c))


def rotating_set(adj, b, c, limit=8):
    """Atoms that move when the b-c bond is turned: b's side, minus b itself.

    Returns None if the bond is in a ring, or if more than `limit` atoms would
    move -- turning a whole side chain is not a free rotation and must not
    happen behind the caller's back.
    """
    seen = {c, b}
    stack = [n for n in adj[b] if n != c]
    out = []
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        if x == c:
            return None                      # ring: the bond is not rotatable
        seen.add(x)
        out.append(x)
        if len(out) > limit:
            return None
        stack.extend(n for n in adj[x] if n not in seen)
    return out


def collect_rotatable_groups(mol, torsional_axes, offset=0):
    """Turn copied-torsion bonds into (axis atom, axis atom, moving atoms).

    Only groups made up entirely of hydrogens are kept -- methyl, hydroxyl,
    thiol and ammonium rotations, which are free, and which are exactly the
    ones whose dihedral had to be copied rather than derived. Amide and cap
    torsions are left where the reference put them: they are not free, and
    minimisation is the right place to relax them.
    """
    adj = mol.adjacency()
    is_h = [naming.is_hydrogen(n) for n in mol.atomname]
    groups = []
    for b, c in sorted(torsional_axes):
        mv = rotating_set(adj, b, c)
        if not mv or not all(is_h[m] for m in mv):
            continue
        groups.append((b + offset, c + offset,
                       np.array(sorted(mv), dtype=np.int64) + offset))
    return groups


def optimise_rotatable(xyz, groups, box=None, step_deg=15.0,
                       contact=2.6, cutoff=4.5):
    """Spin each free hydrogen group to its least-clashing setting.

    `groups` comes from `collect_rotatable_groups`, and may span several
    molecules, so that contacts across a chain interface are seen.
    """
    xyz = np.array(xyz, dtype=np.float64, copy=True)
    if not groups:
        return xyz, 0

    if box is None:
        box = xyz.max(0) - xyz.min(0) + 50.0
        pbc = (False, False, False)
    else:
        pbc = (True, True, True)
    grid = geom.CellGrid(xyz, box, cutoff, pbc)
    angles = np.deg2rad(np.arange(step_deg, 360.0, step_deg))
    moved = 0

    for b, c, group in groups:
        origin, axis = xyz[b], xyz[b] - xyz[c]
        if np.linalg.norm(axis) < 1e-6:
            continue
        exclude = np.unique(np.concatenate([group, [b, c]]))

        def score(pos, _ex=exclude):
            _, pj, d = grid.pairs(pos, cutoff)
            if len(pj) == 0:
                return 0.0
            keep = ~np.isin(pj, _ex)
            if not keep.any():
                return 0.0
            return float((np.maximum(0.0, contact - d[keep]) ** 2).sum())

        best_pos, best = xyz[group], score(xyz[group])
        if best <= 1e-12:
            continue
        for ang in angles:
            trial = _rotate_about(xyz[group], origin, axis, ang)
            s = score(trial)
            if s < best - 1e-9:
                best, best_pos = s, trial
        if best_pos is not xyz[group]:
            xyz[group] = best_pos
            moved += 1
    return xyz, moved


# --------------------------------------------------------------------------
# whole-protein driver
# --------------------------------------------------------------------------

def rebuild_protein(mols, ref_xyz_per_mol, model, chain_for_mol,
                    his_variant="HSD", capped=True, optimise=True,
                    verbose=True):
    """Rebuild every protein chain and return one array in topology order.

    `mols` is the ordered list of MoleculeTopology (PROA, PROB, ...),
    `ref_xyz_per_mol` the matching complete reference coordinates, and
    `chain_for_mol` the model chain id each one corresponds to.
    """
    out, reports, groups, offset = [], [], [], 0
    for mol, ref, chain in zip(mols, ref_xyz_per_mol, chain_for_mol):
        sub = model[model.chain == chain]
        if len(sub) == 0:
            have = sorted(set(model.chain.tolist()))
            hint = ""
            if have in ([" "], [""], []):
                hint = (" The model has no chain identifiers at all -- column "
                        "22 is blank. A PDB written out of a .gro looks like "
                        "this, because .gro has no chain column.")
            raise ValueError(
                "%s expects model chain %r, which is not in the file. Chains "
                "present: %s.%s"
                % (mol.name, chain, ", ".join(repr(c) for c in have) or "none",
                   hint))
        kidx, kxyz, unmatched = match_to_topology(
            mol, sub, his_variant=his_variant, capped=capped)

        # A name mismatch used to surface much later as "could not place 4255
        # atoms", which reads like the packer failed rather than the matching.
        # Say what actually happened, with the numbers that prove it.
        if len(kidx) < 0.5 * len(sub):
            t_lo, t_hi = int(min(mol.resid)), int(max(mol.resid))
            m_lo, m_hi = int(sub.resid.min()), int(sub.resid.max())
            extra = ""
            if (t_lo, t_hi) != (m_lo, m_hi):
                extra = (" The topology numbers its residues %d-%d and model "
                         "chain %s is numbered %d-%d, so nothing lines up. A "
                         "protein taken out of a .gro is renumbered from 1 "
                         "and has to be mapped back."
                         % (t_lo, t_hi, chain, m_lo, m_hi))
            raise ValueError(
                "%s: matched only %d of %d atoms in model chain %s against "
                "the topology.%s"
                % (mol.name, len(kidx), len(sub), chain, extra))

        xyz, rep, tors = rebuild(mol, ref, kidx, kxyz)
        rep.unmatched_input = unmatched
        if rep.failed:
            bad = ", ".join("%s%s" % (mol.resid[i], mol.atomname[i])
                            for i in rep.failed[:10])
            raise RuntimeError(
                f"{mol.name}: could not place {len(rep.failed)} atoms ({bad})")
        groups.extend(collect_rotatable_groups(mol, tors, offset))
        out.append(xyz)
        reports.append(rep)
        offset += mol.natoms
        if verbose:
            print("   " + rep.summary())
            if unmatched:
                print("     input atoms with no topology counterpart: %d %s"
                      % (len(unmatched), unmatched[:6]))

    xyz = np.vstack(out)
    if optimise:
        xyz, moved = optimise_rotatable(xyz, groups)
        if verbose:
            print("   torsion cleanup: %d of %d free hydrogen groups rotated"
                  % (moved, len(groups)))
    return xyz, reports
