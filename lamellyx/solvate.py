"""Water and counter-ions.

Water is tiled from a cube cut out of an equilibrated reference system, the
same way `gmx solvate` tiles spc216.gro, then thinned wherever it overlaps
something. Ions replace water molecules rather than being inserted, so the
number of atoms is predictable and no new holes appear.

The ion count follows CHARMM-GUI's rule exactly:

    n_salt   = round(concentration * n_water / 55.5)
    n_cation = n_salt + max(0, -q)
    n_anion  = n_salt + max(0, +q)

for system charge q. On the reference system that gives 62 Cl- and 70 K+ for
0.15 M and q = -8, which is what CHARMM-GUI produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import bilayer as _bilayer
from . import geom

WATER_MOLARITY = 55.5


@dataclass
class WaterTemplate:
    """An equilibrated cube of water, ready to tile."""

    xyz: np.ndarray                  # (K, natoms, 3), origin at the cube corner
    cell: float
    atomnames: list = field(default_factory=list)
    resname: str = "TIP3"

    @property
    def natoms(self):
        return self.xyz.shape[1]


def extract_water_template(atoms, box, resname="TIP3", natoms=3, cube=20.0,
                           z_range=None, rng=None):
    """Cut a cube of bulk water out of a reference system.

    `z_range` should be a slab of pure bulk -- above the protein, below the
    box edge. Taking the cube from anywhere near a solute would bake that
    solute's hydration shell into every tile.
    """
    sel = atoms.resname == resname
    if not sel.any():
        raise ValueError(f"reference contains no {resname}")
    w = _bilayer.make_whole(atoms.xyz[sel], box, natoms)
    anames = list(atoms.name[sel][:natoms])
    o = w[:, 0, :]

    if z_range is None:
        z_range = (o[:, 2].max() - cube - 1.0, o[:, 2].max() - 1.0)
    lo, hi = z_range
    band = (o[:, 2] >= lo) & (o[:, 2] <= hi)
    if band.sum() < 50:
        raise ValueError(f"only {band.sum()} waters in z {lo:.1f}-{hi:.1f}")

    rng = np.random.default_rng() if rng is None else rng
    # slide a cube around inside the band and keep the densest placement
    best, best_n = None, -1
    for _ in range(40):
        x0 = rng.uniform(0, max(box[0] - cube, 1e-6))
        y0 = rng.uniform(0, max(box[1] - cube, 1e-6))
        z0 = rng.uniform(lo, max(hi - cube, lo + 1e-6))
        inside = (band
                  & (o[:, 0] >= x0) & (o[:, 0] < x0 + cube)
                  & (o[:, 1] >= y0) & (o[:, 1] < y0 + cube)
                  & (o[:, 2] >= z0) & (o[:, 2] < z0 + cube))
        if inside.sum() > best_n:
            best_n, best = int(inside.sum()), (inside, np.array([x0, y0, z0]))
    inside, corner = best
    if best_n < 20:
        raise ValueError("could not find a dense cube of bulk water")
    return WaterTemplate(xyz=w[inside] - corner, cell=float(cube),
                         atomnames=anames, resname=resname)


def tile_water(template, box, z_lo, z_hi, rng, jitter=True, offset=None):
    """Fill a z slab of the box with copies of the template cube.

    The cube is cut out of a larger system, so its faces do not match its own
    periodic images and the tiles overlap slightly where they meet. About a
    sixth of the water is lost closing those seams, which is why `solvate`
    runs several passes at different offsets rather than one.
    """
    c = template.cell
    off = np.zeros(3) if offset is None else np.asarray(offset, dtype=float)
    nx = int(np.ceil(box[0] / c)) + 1
    ny = int(np.ceil(box[1] / c)) + 1
    nz = int(np.ceil((z_hi - z_lo) / c)) + 1
    tiles = []
    for i in range(-1, nx):
        for j in range(-1, ny):
            for k in range(-1, nz):
                shift = np.array([i * c, j * c, z_lo + k * c]) + off
                w = template.xyz + shift
                if jitter:
                    # a random whole-cube rotation about z breaks the seams up
                    R = geom.rotation_z(rng.integers(4) * np.pi / 2)
                    mid = shift + c / 2.0
                    w = (w - mid) @ R.T + mid
                tiles.append(w)
    w = np.concatenate(tiles, axis=0)
    o = w[:, 0, :]
    keep = ((o[:, 0] >= 0) & (o[:, 0] < box[0]) &
            (o[:, 1] >= 0) & (o[:, 1] < box[1]) &
            (o[:, 2] >= z_lo) & (o[:, 2] < z_hi))
    return w[keep]


def _prune_self_overlap(w, box, min_oo=2.45, min_any=1.30):
    """Drop waters that sit on top of each other at the tile seams.

    Both tests are needed. Two waters can have their oxygens a comfortable
    2.5 A apart and still have a hydrogen buried in the other molecule, which
    an oxygen-only check will not see.
    """
    natoms = w.shape[1]
    reach = max(min_oo, min_any)
    while True:
        nmol = len(w)
        flat = geom.wrap(w.reshape(-1, 3), box)
        mol = np.repeat(np.arange(nmol), natoms)
        is_o = (np.arange(len(flat)) % natoms) == 0
        grid = geom.CellGrid(flat, box, reach)
        qi, pj, d = grid.pairs(flat, reach)
        m = mol[qi] < mol[pj]                 # different molecules, once each
        qi, pj, d = qi[m], pj[m], d[m]
        bad = ((is_o[qi] & is_o[pj] & (d < min_oo)) | (d < min_any))
        qi, pj = mol[qi[bad]], mol[pj[bad]]
        if len(qi) == 0:
            return w
        deg = np.bincount(np.concatenate([qi, pj]), minlength=nmol)
        drop = set()
        for a, b in zip(qi, pj):
            if a in drop or b in drop:
                continue
            drop.add(int(a) if deg[a] >= deg[b] else int(b))
        keep = np.ones(nmol, dtype=bool)
        keep[list(drop)] = False
        w = w[keep]


def _free_of(batch, existing, box, min_oo, min_any):
    """Which molecules of `batch` do not overlap `existing`."""
    n_at = batch.shape[1]
    ge_o = geom.CellGrid(geom.wrap(existing[:, 0, :], box), box, min_oo)
    ok = ge_o.min_distance(batch[:, 0, :]) >= min_oo
    ge_all = geom.CellGrid(geom.wrap(existing.reshape(-1, 3), box), box, min_any)
    close = (ge_all.min_distance(geom.wrap(batch.reshape(-1, 3), box)) < min_any)
    return ok & ~close.reshape(len(batch), n_at).any(1)


def solvate(solute_xyz, solute_is_heavy, box, template, rng,
            z_lo=0.0, z_hi=None, min_o_heavy=2.50, min_any=1.30,
            core_z=None, keep_pore_water=True, pore_neighbours=40,
            lipid_xyz=None, protein_xyz=None, pore_min_lipid=10.0,
            pore_enclosure=0.90, pore_radius=None, pore_axis_xy=None,
            min_oo=2.45, fill_passes=4, verbose=True):
    """Fill the box with water, minus anything that overlaps.

    The two distances are not interchangeable and using one for both is a
    quiet way to lose a tenth of the water in the box. `min_o_heavy` applies
    to the water *oxygen* against solute *heavy* atoms -- 2.5 A, which is a
    close contact. `min_any` applies to any atom pair, and has to be small,
    because a water hydrogen donating to a backbone carbonyl sits about 1.8 A
    from it and that water belongs there. Applying the heavy-atom distance to
    hydrogens deletes every hydrogen-bonded water at the interface.

    `core_z` is (lo, hi) of the hydrophobic slab. Water there is removed
    unless it is a pore water, and what identifies a pore water is its
    distance from the *lipids*, not its distance from the protein. Counting
    protein neighbours alone keeps every water sitting in a groove on the
    outside of the protein, wedged against the acyl chains -- in the
    reference system that test keeps three thousand waters where CHARMM-GUI
    kept five, all five of them more than 17 A from any lipid, down the
    channel axis where they belong.
    """
    z_hi = float(box[2]) if z_hi is None else z_hi
    solute_xyz = np.asarray(solute_xyz).reshape(-1, 3)
    heavy = solute_xyz[np.asarray(solute_is_heavy, dtype=bool)]
    g_heavy = geom.CellGrid(heavy, box, min_o_heavy)
    g_all = geom.CellGrid(geom.wrap(solute_xyz, box), box, min_any)

    def drop_on_solute(batch):
        n_at = batch.shape[1]
        flat = geom.wrap(batch.reshape(-1, 3), box)
        bad = g_heavy.min_distance(batch[:, 0, :]) < min_o_heavy   # oxygen only
        bad |= (g_all.min_distance(flat) < min_any
                ).reshape(len(batch), n_at).any(1)
        return batch[~bad]

    n0 = n1 = 0
    w = np.zeros((0, template.natoms, 3))
    for p in range(fill_passes):
        off = np.zeros(3) if p == 0 else rng.uniform(0, template.cell, 3)
        batch = tile_water(template, box, z_lo, z_hi, rng, offset=off)
        n0 += len(batch)
        batch = _prune_self_overlap(batch, box, min_oo, min_any)
        n1 += len(batch)
        batch = drop_on_solute(batch)
        if len(w) and len(batch):
            batch = batch[_free_of(batch, w, box, min_oo, min_any)]
        if len(batch) == 0:
            break
        w = np.concatenate([w, batch]) if len(w) else batch
        if verbose:
            print("   water pass %d: +%d molecules (total %d)"
                  % (p, len(batch), len(w)))
    n2 = len(w)

    n_pore = 0
    if core_z is not None:
        lo, hi = core_z
        o = w[:, 0, :]
        in_core = (o[:, 2] > lo) & (o[:, 2] < hi)
        if in_core.any():
            if keep_pore_water and lipid_xyz is not None and len(lipid_xyz):
                oc = o[in_core]
                gp = geom.CellGrid(heavy, box, 10.0)
                qi, _, _ = gp.pairs(oc, 10.0)
                cnt = np.bincount(qi, minlength=len(oc))
                gl = geom.CellGrid(np.asarray(lipid_xyz).reshape(-1, 3), box,
                                   pore_min_lipid)
                far_from_lipid = gl.min_distance(oc) >= pore_min_lipid
                # Neighbour count alone cannot tell a pore from a dent: a
                # water pressed into a groove on the lipid-facing surface has
                # as many protein atoms within 8 A as one in the lumen, and
                # is 10 A from any lipid, so it passed both earlier tests.
                # Asking which directions are occupied separates them.
                # Protein only. Counting lipids would call a water walled in
                # by acyl chains "enclosed", which is the opposite of the
                # thing being tested for.
                enc = enclosure(oc, heavy if protein_xyz is None
                                else protein_xyz, box)
                buried = ((cnt >= pore_neighbours) & far_from_lipid
                          & (enc >= pore_enclosure))
                # `far_from_lipid` assumes the lipids are packed against the
                # protein, so that anything 10 A from a lipid must be inside
                # it. When packing leaves an annular gap that assumption
                # inverts: water in the gap is far from lipid *because the
                # lipid is missing*, and a groove on the outside of the
                # protein is enclosed enough to pass the other two tests as
                # well. On HCN4 that kept 452 waters against CHARMM-GUI's 5,
                # the furthest 41.7 A off-axis.
                #
                # A pore is a hole down the middle, so say so. This is the
                # same thing CHARMM-GUI asks for as a cylinder radius.
                if pore_radius:
                    axis = (np.asarray(pore_axis_xy, dtype=float)
                            if pore_axis_xy is not None
                            else (heavy if protein_xyz is None
                                  else protein_xyz)[:, :2].mean(0))
                    d = oc[:, :2] - axis
                    for k in (0, 1):
                        dk = np.abs(d[:, k])       # not `w` -- that is the
                        d[:, k] = np.minimum(dk, box[k] - dk)   # water array
                    buried &= np.hypot(d[:, 0], d[:, 1]) <= float(pore_radius)
                n_pore = int(buried.sum())
                drop = np.flatnonzero(in_core)[~buried]
            else:
                drop = np.flatnonzero(in_core)
            keep = np.ones(len(w), dtype=bool)
            keep[drop] = False
            w = w[keep]
    if verbose:
        print("   water: %d tiled, %d after seam pruning, %d after solute "
              "overlap, %d final (%d kept inside the protein)"
              % (n0, n1, n2, len(w), n_pore))
    return w


# --------------------------------------------------------------------------
# ions
# --------------------------------------------------------------------------

# 26 directions on a cube. A water in a pore or an interior cavity has
# protein on nearly every side; one lying in a surface groove has protein on
# one side and open space on the other.
_DIRECTIONS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
     if (i, j, k) != (0, 0, 0)], dtype=np.float64)
_DIRECTIONS /= np.linalg.norm(_DIRECTIONS, axis=1)[:, None]


def enclosure(points, protein_xyz, box, radius=9.0, cone=0.85):
    """How enclosed each point is: the fraction of directions with protein.

    Counting neighbours cannot tell a pore from a dent -- a water pressed
    into a groove on the outside of a protein has as many atoms within 8 A as
    one in the lumen. Asking which *directions* are occupied can: in the
    reference system every pore water scores 1.00, while grooves on the
    lipid-facing surface score 0.4 to 0.8.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    prot = np.asarray(protein_xyz).reshape(-1, 3)
    if not len(points) or not len(prot):
        return np.zeros(len(points))
    grid = geom.CellGrid(prot, box, radius)
    pts = geom.wrap(points, box)
    qi, pj, _ = grid.pairs(pts, radius)
    if not len(qi):
        return np.zeros(len(points))
    v = geom.min_image(grid.points[pj] - pts[qi], box, (True, True, True))
    n = np.linalg.norm(v, axis=1)
    n[n == 0] = 1.0
    hit = ((v / n[:, None]) @ _DIRECTIONS.T) > cone
    acc = np.zeros((len(points), len(_DIRECTIONS)), dtype=np.int32)
    np.add.at(acc, qi, hit.astype(np.int32))
    return (acc > 0).mean(axis=1)


def ion_counts(n_water, system_charge, concentration=0.15):
    """CHARMM-GUI's rule. Returns (n_cation, n_anion).

    A negative concentration used to sail through and return negative counts,
    which reach topol.top as `POT -9` and fail somewhere much less obvious.
    """
    if concentration < 0:
        raise ValueError("salt concentration cannot be negative (got %r)"
                         % concentration)
    if n_water < 0:
        raise ValueError("water count cannot be negative (got %r)" % n_water)
    n_salt = max(int(round(concentration * n_water / WATER_MOLARITY)), 0)
    q = int(round(system_charge))
    return n_salt + max(0, -q), n_salt + max(0, q)


def place_ions(water, box, solute_xyz, n_cation, n_anion, rng,
               min_solute=5.0, min_ion=5.0, exclude_z=None, verbose=True):
    """Replace water molecules with ions.

    Returns (remaining water, cation positions, anion positions). Ions are
    kept away from the solute and from each other so that none starts inside
    a first hydration shell it would have to fight its way out of.
    """
    n_need = n_cation + n_anion
    o = geom.wrap(water[:, 0, :], box)
    ok = np.ones(len(water), dtype=bool)

    if len(solute_xyz):
        g = geom.CellGrid(geom.wrap(np.asarray(solute_xyz).reshape(-1, 3), box),
                          box, min_solute)
        ok &= g.min_distance(o) >= min_solute
    if exclude_z is not None:
        ok &= (o[:, 2] < exclude_z[0]) | (o[:, 2] > exclude_z[1])

    cand = np.flatnonzero(ok)
    if len(cand) < n_need:
        raise RuntimeError(
            f"only {len(cand)} water sites are far enough from the solute for "
            f"{n_need} ions; lower min_solute")

    rng.shuffle(cand)
    chosen = []
    picked = np.zeros((0, 3))
    for idx in cand:
        if len(chosen) == n_need:
            break
        p = o[idx]
        if len(picked):
            d = geom.min_image(picked - p, box, (True, True, True))
            if np.sqrt((d ** 2).sum(1)).min() < min_ion:
                continue
        chosen.append(int(idx))
        picked = np.vstack([picked, p])
    if len(chosen) < n_need:
        raise RuntimeError(
            f"could only place {len(chosen)} of {n_need} ions at min_ion="
            f"{min_ion} A; lower it")

    # dtype matters: an empty list becomes a float array, and indexing with
    # one raises. Asking for no salt at all is the case that gets there.
    chosen = np.array(chosen, dtype=np.int64)
    cat = o[chosen[:n_cation]]
    ani = o[chosen[n_cation:]]
    keep = np.ones(len(water), dtype=bool)
    keep[chosen] = False
    if verbose:
        print("   ions: %d cations, %d anions, replacing water; %d water left"
              % (n_cation, n_anion, int(keep.sum())))
    return water[keep], cat, ani


def size_box_z(solute_xyz, thickness):
    """Box height and the shift that centres the solute in it.

    `thickness` is the water layer wanted between the outermost solute atom
    and the box edge, which is what CHARMM-GUI means by water thickness.
    """
    z = np.asarray(solute_xyz)[:, 2]
    height = float(z.max() - z.min() + 2.0 * thickness)
    shift = float(thickness - z.min())
    return height, shift
