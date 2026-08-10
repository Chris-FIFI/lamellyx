"""Packing a lipid bilayer around a protein.

Lipid conformers are taken from an equilibrated reference bilayer rather than
generated, so every lipid starts from a shape a force field has already
accepted. Placement is: carve the protein's footprint out of a dense candidate
lattice, pick exactly as many sites as there are lipids and as far apart as
possible, drop a random conformer on each, then relax.

Two things about the relaxation are worth knowing.

The target separation is 3.0 A between heavy atoms of different molecules,
because that is what a real bilayer looks like -- in the reference system a
quarter of all lipid heavy atoms have a neighbour within 3.5 A, and the
closest approach anywhere is 2.34 A. A packer that insists on more space than
that is fighting the liquid it is trying to build, and will never converge.

Lipids start shrunk and are grown back to full size as the relaxation runs.
That is the trick `gmx membed` uses, and it is why a lipid whose tail begins
inside the protein ends up beside it rather than deleted.

Chirality note: a lower-leaflet lipid is turned into an upper-leaflet one by
rotating 180 degrees about x, never by negating z. Negating z is a reflection
and would quietly produce the wrong stereoisomer at the glycerol C2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geom

# Separation the relaxation pushes towards, between heavy atoms of different
# molecules. Measured from an equilibrated CHARMM36m POPC bilayer.
TARGET_CONTACT = 3.00
# Anything closer than this after packing is a genuine clash worth repairing.
CLASH = 2.40
# Per-atom radii for the final all-atom pass. A pair's target separation is
# the sum of the two, so heavy-heavy keeps the 3.0 A above while H...H drops
# to 2.0 -- hydrogens genuinely do sit closer. Using one target for every pair
# instead pulls carbons together while it pushes hydrogens apart.
RADIUS_HEAVY = 1.50
RADIUS_H = 1.00
# A pair counts as clashing below this fraction of its target separation:
# 2.4 A between carbons, 1.6 A between hydrogens.
CLASH_FRACTION = 0.80
# How far lipids are shrunk before being grown back to full size.
SHRINK = 0.55
FLIP_X = np.diag([1.0, -1.0, -1.0])       # proper rotation, preserves chirality


def make_whole(xyz, box, natoms_per_mol):
    """Undo periodic wrapping inside each molecule."""
    x = np.asarray(xyz, dtype=np.float64).reshape(-1, natoms_per_mol, 3)
    d = x - x[:, :1, :]
    for a in range(3):
        d[..., a] -= box[a] * np.round(d[..., a] / box[a])
    return x[:, :1, :] + d


# Kept here for callers that already import it from this module.
wrap_molecules = geom.wrap_molecules


@dataclass
class LipidLibrary:
    """Conformers of one lipid species, split by leaflet."""

    resname: str
    atomnames: list
    head: int                     # index of the reference atom (P)
    upper: np.ndarray             # (K, natoms, 3), head atom at the origin
    lower: np.ndarray
    head_z_upper: float = 0.0
    head_z_lower: float = 0.0
    head_z_sd: float = 1.2
    midplane: float = 0.0

    @property
    def natoms(self):
        return len(self.atomnames)

    @property
    def heavy_mask(self):
        return np.array([not n.upper().startswith("H")
                         for n in self.atomnames], dtype=bool)

    def pool(self, leaflet):
        want = self.upper if leaflet > 0 else self.lower
        if len(want):
            return want
        other = self.lower if leaflet > 0 else self.upper
        if not len(other):
            raise ValueError(f"no {self.resname} conformers available")
        return other @ FLIP_X.T


def extract_library(atoms, box, resname, natoms, head_atom="P"):
    """Build a conformer library from an equilibrated reference system."""
    sel = atoms.resname == resname
    n = int(sel.sum())
    if n == 0:
        raise ValueError(f"reference contains no {resname}")
    if n % natoms:
        raise ValueError(f"{n} {resname} atoms is not a multiple of {natoms}")
    nmol = n // natoms
    xyz = make_whole(atoms.xyz[sel], box, natoms)
    anames = list(atoms.name[sel][:natoms])
    if head_atom not in anames:
        raise ValueError(f"{resname} has no atom {head_atom!r}")
    if not (np.array(anames * nmol) == atoms.name[sel]).all():
        raise ValueError(f"{resname} atom order is not identical across molecules")

    head = anames.index(head_atom)
    hz = xyz[:, head, 2]
    mid = 0.5 * (hz.max() + hz.min())
    is_up = hz > mid
    centred = xyz - xyz[:, head:head + 1, :]
    return LipidLibrary(
        resname=resname, atomnames=anames, head=head,
        upper=centred[is_up], lower=centred[~is_up],
        head_z_upper=float(hz[is_up].mean()) if is_up.any() else mid,
        head_z_lower=float(hz[~is_up].mean()) if (~is_up).any() else mid,
        head_z_sd=float(hz[is_up].std()) if is_up.sum() > 1 else 1.2,
        midplane=float(mid),
    )


# --------------------------------------------------------------------------
# site selection
# --------------------------------------------------------------------------

def candidate_sites(box_xy, spacing, rng):
    nx = max(int(np.floor(box_xy[0] / spacing)), 1)
    ny = max(int(np.floor(box_xy[1] / spacing)), 1)
    dx, dy = box_xy[0] / nx, box_xy[1] / ny
    gx, gy = np.meshgrid((np.arange(nx) + 0.5) * dx,
                         (np.arange(ny) + 0.5) * dy, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    pts += rng.uniform(-0.3, 0.3, pts.shape) * np.array([dx, dy])
    return np.mod(pts, box_xy)


def _xy_grid(points_xy, box_xy, cutoff):
    box3 = np.array([box_xy[0], box_xy[1], max(cutoff, 1.0) * 4.0])
    pad = np.column_stack([points_xy, np.zeros(len(points_xy))])
    return geom.CellGrid(pad, box3, cutoff, pbc=(True, True, False)), box3


def choose_sites(n_want, protein_xy, box_xy, exclude_radius, rng,
                 spacing_hint, max_tries=10):
    """Exactly `n_want` well-spread sites that avoid the protein footprint."""
    spacing = spacing_hint
    cand = np.zeros((0, 2))
    for _ in range(max_tries):
        cand = candidate_sites(box_xy, spacing, rng)
        if len(protein_xy):
            grid, _ = _xy_grid(protein_xy, box_xy, max(exclude_radius, 1.0))
            d = grid.min_distance(np.column_stack([cand, np.zeros(len(cand))]))
            cand = cand[d > exclude_radius]
        if len(cand) >= n_want:
            keep = geom.farthest_point_sample(
                cand, n_want, box=np.array([box_xy[0], box_xy[1], 1.0]),
                pbc=(True, True, False), rng=rng)
            return cand[keep]
        spacing *= 0.85
    raise RuntimeError(
        f"only {len(cand)} sites available for {n_want} lipids -- the box is "
        f"too small, the protein too wide, or exclude_radius too large")


def protein_footprint(protein_xyz, z_lo, z_hi):
    m = (protein_xyz[:, 2] >= z_lo) & (protein_xyz[:, 2] <= z_hi)
    return protein_xyz[m, :2]


def slab_extent(protein_xyz, z_lo, z_hi):
    """XY bounding box of the protein *inside the membrane*.

    The margin a membrane needs is measured from the transmembrane
    cross-section, not from the whole protein. A channel with a large
    extracellular domain is much wider above the bilayer than in it, and
    sizing the box from the total extent wastes a great deal of lipid.
    """
    foot = protein_footprint(protein_xyz, z_lo, z_hi)
    if not len(foot):
        raise ValueError("no protein atoms lie in the membrane slab "
                         "(z %.1f to %.1f) -- is the protein oriented?"
                         % (z_lo, z_hi))
    return foot.min(axis=0), foot.max(axis=0)


def footprint_area(points_xy, box_xy, probe=2.4, spacing=1.0):
    """Area the protein occupies in one leaflet, by rasterising its shape.

    Counting grid cells within `probe` of an atom is how CHARMM-GUI measures
    protein area too. A bounding box would be far too generous for anything
    that is not a cylinder.
    """
    if not len(points_xy):
        return 0.0
    nx = max(int(np.ceil(box_xy[0] / spacing)), 1)
    ny = max(int(np.ceil(box_xy[1] / spacing)), 1)
    gx = (np.arange(nx) + 0.5) * (box_xy[0] / nx)
    gy = (np.arange(ny) + 0.5) * (box_xy[1] / ny)
    cx, cy = np.meshgrid(gx, gy, indexing="ij")
    cells = np.column_stack([cx.ravel(), cy.ravel(), np.zeros(cx.size)])
    grid, _ = _xy_grid(points_xy, box_xy, max(probe, 1.0))
    occupied = int((grid.min_distance(cells) <= probe).sum())
    return occupied * (box_xy[0] / nx) * (box_xy[1] / ny)


def lipids_for_free_area(box_xy, protein_xyz, slab, apl, probe=2.4):
    """How many lipids fit in one leaflet once the protein has taken its area."""
    foot = protein_footprint(protein_xyz, *slab)
    taken = footprint_area(foot, box_xy, probe=probe)
    free = float(box_xy[0] * box_xy[1]) - taken
    return max(int(round(free / apl)), 0), taken


def place_leaflet(library, leaflet, sites, rng, z_sd=None):
    """Drop a random conformer, randomly spun about z, on each site.

    Head-group z is jittered less than the reference bilayer's own spread,
    because the relaxation adds roughness of its own and the two compound.
    """
    pool = library.pool(leaflet)
    z_sd = 0.6 * library.head_z_sd if z_sd is None else z_sd
    z0 = library.head_z_upper if leaflet > 0 else library.head_z_lower
    pick = rng.integers(len(pool), size=len(sites))
    zs = z0 + rng.normal(0.0, z_sd, len(sites))
    out = np.empty((len(sites), library.natoms, 3))
    for k, (p, s, z) in enumerate(zip(pick, sites, zs)):
        R = geom.rotation_z(rng.uniform(0.0, 2.0 * np.pi))
        out[k] = pool[p] @ R.T + np.array([s[0], s[1], z])
    return out


# --------------------------------------------------------------------------
# rigid-body relaxation
# --------------------------------------------------------------------------

class RigidLipidRelaxer:
    """Soft repulsion between rigid lipids and a fixed environment."""

    def __init__(self, lipids, heavy_mask, fixed_xyz, box,
                 target=TARGET_CONTACT, skin=1.5, atom_radii=None,
                 fixed_radius=RADIUS_HEAVY):
        self.x = np.array(lipids, dtype=np.float64, copy=True)
        self.nmol, self.natoms = self.x.shape[:2]
        self.heavy = np.asarray(heavy_mask, dtype=bool)
        self.nheavy = int(self.heavy.sum())
        self.fixed = np.asarray(fixed_xyz, dtype=np.float64).reshape(-1, 3)
        self.box = np.asarray(box, dtype=np.float64)
        self.target = float(target)
        self.skin = float(skin)
        self.mol_of = np.repeat(np.arange(self.nmol), self.nheavy)
        self._pairs = None
        self._built_at = None

        # Optional per-atom radii: a pair's target becomes the sum of the two,
        # so one pass can hold carbons 3 A apart and hydrogens 2 A apart at
        # the same time.
        if atom_radii is None:
            self.rad_mob = self.rad_env = None
            self.reach = self.target
        else:
            r = np.asarray(atom_radii, dtype=np.float64)[self.heavy]
            self.rad_mob = np.tile(r, self.nmol)
            self.rad_env = np.concatenate(
                [self.rad_mob, np.full(len(self.fixed), float(fixed_radius))])
            self.reach = float(self.rad_env.max() * 2.0)

    def heavy_coords(self, x=None):
        x = self.x if x is None else x
        return x[:, self.heavy, :].reshape(-1, 3)

    def _rebuild(self):
        h = geom.wrap(self.heavy_coords(), self.box)
        env = np.vstack([h, self.fixed])
        cutoff = self.reach + self.skin
        grid = geom.CellGrid(env, self.box, cutoff)
        qi, pj, _ = grid.pairs(h, cutoff)
        intra = np.zeros(len(qi), dtype=bool)
        own = pj < len(h)
        intra[own] = self.mol_of[qi[own]] == self.mol_of[pj[own]]
        self._pairs = (qi[~intra], pj[~intra])
        self._built_at = h.copy()

    def _displaced(self):
        h = geom.wrap(self.heavy_coords(), self.box)
        d = geom.min_image(h - self._built_at, self.box, (True, True, True))
        return float(np.sqrt((d ** 2).sum(1)).max())

    def worst_contact(self):
        """Closest approach between heavy atoms of different molecules."""
        self._rebuild()
        qi, pj = self._pairs
        if len(qi) == 0:
            return np.inf, 0
        h = geom.wrap(self.heavy_coords(), self.box)
        env = np.vstack([h, self.fixed])
        d = geom.min_image(env[pj] - h[qi], self.box, (True, True, True))
        r = np.sqrt((d ** 2).sum(1))
        return float(r.min()), int((r < CLASH).sum() // 1)

    def run(self, steps, scale_from=1.0, scale_to=1.0, eta=0.35, eta_z=0.05,
            eta_rot=0.004, z_limit=2.0, verbose=False, label=""):
        centroid = self.x.mean(axis=1, keepdims=True)
        z_home = centroid[:, 0, 2].copy()
        base = self.x - centroid

        for step in range(steps):
            frac = step / max(steps - 1, 1)
            scale = scale_from + (scale_to - scale_from) * frac
            self.x = centroid + base * scale

            if self._pairs is None or self._displaced() > 0.5 * self.skin:
                self._rebuild()
            qi, pj = self._pairs
            if len(qi) == 0:
                continue

            h = geom.wrap(self.heavy_coords(), self.box)
            env = np.vstack([h, self.fixed])
            d = geom.min_image(env[pj] - h[qi], self.box, (True, True, True))
            r = np.sqrt((d ** 2).sum(1))
            r = np.where(r < 1e-6, 1e-6, r)
            tgt = (self.target if self.rad_mob is None
                   else self.rad_mob[qi] + self.rad_env[pj])
            over = tgt - r
            hit = over > 0
            if not hit.any():
                continue

            # Severity weighting matters more than it looks. With a force
            # linear in the overlap, a lipid surrounded on all sides gets a
            # mean force of nearly zero and never moves, however bad one of
            # its contacts is. Weighting by the square of the overlap lets the
            # one contact that is actually wrong dominate the others.
            w = over[hit] ** 2 / self.target
            f = -(w / r[hit])[:, None] * d[hit]
            qh, mol = qi[hit], self.mol_of[qi[hit]]

            net = np.empty((self.nmol, 3))
            for a in range(3):
                net[:, a] = np.bincount(mol, weights=f[:, a],
                                        minlength=self.nmol)

            rel = h[qh] - centroid[mol, 0, :]
            tau = rel[:, 0] * f[:, 1] - rel[:, 1] * f[:, 0]
            netT = np.bincount(mol, weights=tau, minlength=self.nmol)
            cnt = np.bincount(mol, minlength=self.nmol).astype(float)
            cnt[cnt == 0] = 1.0
            rad2 = np.bincount(mol, weights=(rel[:, :2] ** 2).sum(1),
                               minlength=self.nmol) / cnt
            rad2[rad2 < 1.0] = 1.0

            shift = net * np.array([eta, eta, eta_z])
            np.clip(shift, -1.2, 1.2, out=shift)
            dtheta = np.clip(eta_rot * netT / np.sqrt(rad2), -0.12, 0.12)

            c, s = np.cos(dtheta), np.sin(dtheta)
            rel_all = self.x - centroid
            rot = np.empty_like(rel_all)
            rot[..., 0] = c[:, None] * rel_all[..., 0] - s[:, None] * rel_all[..., 1]
            rot[..., 1] = s[:, None] * rel_all[..., 0] + c[:, None] * rel_all[..., 1]
            rot[..., 2] = rel_all[..., 2]
            base = rot / max(scale, 1e-9)

            centroid = centroid + shift[:, None, :]
            centroid[:, 0, 0] = np.mod(centroid[:, 0, 0], self.box[0])
            centroid[:, 0, 1] = np.mod(centroid[:, 0, 1], self.box[1])
            # a lipid must not wander out of its leaflet
            centroid[:, 0, 2] = np.clip(centroid[:, 0, 2],
                                        z_home - z_limit, z_home + z_limit)
            self.x = centroid + base * scale

            if verbose and (step % 40 == 0 or step == steps - 1):
                print("      %-8s step %3d scale %.2f: %6d contacts under "
                      "%.1f A, closest %.2f A"
                      % (label, step, scale, int(hit.sum()), self.target,
                         float(r[hit].min())))

        self.x = centroid + base
        return self.x


# --------------------------------------------------------------------------
# Monte-Carlo repair
# --------------------------------------------------------------------------

def mc_polish(lipids, library, leaflet_of, fixed_xyz, box, rng,
              sweeps=4, trials=32, clash=CLASH, reach=2.70, atom_radii=None,
              verbose=True):
    """Re-seed the few lipids the rigid relaxation could not free.

    A lipid whose tail is threaded through a neighbour cannot slide out of it,
    however long the relaxation runs. Replacing that lipid with a different
    conformer at a different rotation can, and usually does, fix it in one go.
    """
    x = np.array(lipids, dtype=np.float64, copy=True)
    # Scoring every atom, not just the heavy ones, is what lets this fix a
    # hydrogen contact without opening a carbon one: a proposal is only kept
    # if it improves the total, and the total sees both.
    heavy = (library.heavy_mask if atom_radii is None
             else np.ones(library.natoms, dtype=bool))
    nheavy = int(heavy.sum())
    nmol = len(x)
    mol_of = np.repeat(np.arange(nmol), nheavy)
    fixed = np.asarray(fixed_xyz).reshape(-1, 3)
    if atom_radii is None:
        rad_atom = rad_mob = rad_env = None
    else:
        rad_atom = np.asarray(atom_radii, dtype=float)   # per lipid atom
        rad_mob = np.tile(rad_atom, nmol)                # per mobile atom
        rad_env = np.concatenate([rad_mob,
                                  np.full(len(fixed), RADIUS_HEAVY)])
        reach = float(rad_env.max() * 2.0)

    for sweep in range(sweeps):
        h = geom.wrap(x[:, heavy, :].reshape(-1, 3), box)
        env = np.vstack([h, fixed])
        grid = geom.CellGrid(env, box, reach + 0.6)
        qi, pj, d = grid.pairs(h, reach)
        own = pj < len(h)
        intra = np.zeros(len(qi), dtype=bool)
        intra[own] = mol_of[qi[own]] == mol_of[pj[own]]
        qi, pj, d = qi[~intra], pj[~intra], d[~intra]
        # What counts as a clash scales with the pair: 2.4 A between carbons,
        # 1.6 A between hydrogens. A single threshold cannot describe both.
        if rad_mob is None:
            limit = np.full(len(d), clash)
        else:
            limit = CLASH_FRACTION * (rad_mob[qi] + rad_env[pj])
        hit = d < limit
        qi, d = qi[hit], d[hit]
        if len(qi) == 0:
            if verbose:
                print("      polish sweep %d: nothing left to repair" % sweep)
            break
        bad = np.unique(mol_of[qi])
        if verbose:
            print("      polish sweep %d: %d lipids too close to a neighbour "
                  "(closest %.2f A)" % (sweep, len(bad), d.min()))

        def score(pos, m):
            """Overlap of one candidate placement with everything else.

            Only separations below the pair's own target count. Scoring normal
            packing too would let a proposal that eases fifty ordinary 3 A
            contacts win over one that fixes the single 1.5 A clash.
            """
            qii, pjj, dd = grid.pairs(pos, reach)
            if len(dd) == 0:
                return 0.0
            own_atom = pjj < len(h)
            keep = ~(own_atom & (mol_of[np.minimum(pjj, len(h) - 1)] == m))
            if not keep.any():
                return 0.0
            qii, pjj, dd = qii[keep], pjj[keep], dd[keep]
            tgt = reach if rad_atom is None else rad_atom[qii] + rad_env[pjj]
            return float((np.maximum(0.0, tgt - dd) ** 2).sum())

        for m in bad:
            leaf = leaflet_of[m]
            pool = library.pool(leaf)
            cur = x[m]
            best_pos, best = cur, score(cur[heavy], m)
            anchor = cur[library.head].copy()
            for _ in range(trials):
                conf = pool[rng.integers(len(pool))]
                R = geom.rotation_z(rng.uniform(0.0, 2.0 * np.pi))
                jitter = np.array([rng.normal(0, 1.2), rng.normal(0, 1.2),
                                   rng.normal(0, 0.5)])
                trial = conf @ R.T + anchor + jitter
                s = score(trial[heavy], m)
                if s < best:
                    best, best_pos = s, trial
            x[m] = best_pos
    return x


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

@dataclass
class BilayerResult:
    xyz: np.ndarray                     # (M, natoms, 3)
    leaflet: np.ndarray                 # +1 / -1 per lipid
    stats: dict = field(default_factory=dict)


def build_bilayer(library, n_upper, n_lower, box, protein_heavy_xyz, rng,
                  apl=None, exclude_radius=4.0, relax_steps=(80, 35, 200),
                  polish_sweeps=6, protein_all_xyz=None, verbose=True):
    """Place and relax a bilayer around a protein.

    `relax_steps` is (steps while shrunk, steps per growth stage, steps at
    full size). `protein_heavy_xyz` must be heavy atoms only -- hydrogens
    contribute nothing to packing and double the cost. `protein_all_xyz`, if
    given, is used for the final all-atom pass.
    """
    box_xy = box[:2]
    area = float(box_xy[0] * box_xy[1])
    if apl is None:
        apl = area / max((n_upper + n_lower) / 2.0, 1.0)
    spacing = float(np.sqrt(apl))

    slabs = {
        +1: (library.midplane - 3.0, library.head_z_upper + 6.0),
        -1: (library.head_z_lower - 6.0, library.midplane + 3.0),
    }

    placed, leaflet_of = [], []
    for leaflet, n_want in ((+1, n_upper), (-1, n_lower)):
        if n_want == 0:
            continue
        foot = protein_footprint(protein_heavy_xyz, *slabs[leaflet])
        sites = choose_sites(n_want, foot, box_xy, exclude_radius, rng, spacing)
        placed.append(place_leaflet(library, leaflet, sites, rng))
        leaflet_of.append(np.full(n_want, leaflet))
        if verbose:
            print("   %s leaflet: %d sites placed, avoiding %d protein atoms"
                  % ("upper" if leaflet > 0 else "lower", n_want, len(foot)))

    lipids = np.concatenate(placed, axis=0)
    leaflet_of = np.concatenate(leaflet_of)
    heavy = library.heavy_mask

    if verbose:
        print("   relaxing %d lipids against %d protein heavy atoms"
              % (len(lipids), len(protein_heavy_xyz)))
    rel = RigidLipidRelaxer(lipids, heavy, protein_heavy_xyz, box)
    n_shrunk, n_per_stage, n_full = relax_steps

    rel.run(n_shrunk, SHRINK, SHRINK, verbose=verbose, label="shrunk")
    # Grow in stages, relaxing between each. Growing straight to full size
    # makes overlaps faster than a rigid-body relaxation can clear them.
    stages = np.linspace(SHRINK, 1.0, 9)[1:]
    for k, s in enumerate(stages):
        rel.run(n_per_stage, s, s, verbose=verbose and k % 4 == 0,
                label="grow %.2f" % s)
    rel.run(n_full, 1.0, 1.0, eta=0.25, verbose=verbose, label="full")
    worst, nclash = rel.worst_contact()
    if verbose:
        print("   after relaxation: closest inter-molecular heavy pair %.2f A"
              % worst)

    x = rel.x
    for cycle in range(polish_sweeps):
        x = mc_polish(x, library, leaflet_of, protein_heavy_xyz, box, rng,
                      sweeps=1, verbose=verbose)
        rel2 = RigidLipidRelaxer(x, heavy, protein_heavy_xyz, box)
        rel2.run(45, 1.0, 1.0, eta=0.22, verbose=False, label="settle")
        x = rel2.x
        w, nc = rel2.worst_contact()
        if verbose:
            print("      repair cycle %d: closest %.2f A, %d under %.1f A"
                  % (cycle, w, nc, CLASH))
        if nc == 0:
            break

    # Everything above works on heavy atoms, because hydrogens contribute
    # nothing to how lipids pack and double the cost. The catch is that two
    # lipids can sit at a perfectly good 2.4 A heavy-atom separation with a
    # pair of hydrogens almost touching between them.
    #
    # Packing is done on heavy atoms only, and that is deliberate.
    #
    # Two lipids can sit at a perfectly good 2.4 A carbon-carbon separation
    # with a pair of hydrogens nearly touching between them -- a C-H bond is
    # 1.09 A, so two pointing at each other across that contact leave the
    # hydrogens 0.2 A apart. It is tempting to add an all-atom pass. It does
    # not work: with a single target tight enough for hydrogens the carbons
    # are dragged together, and even with per-pair targets the sum of a
    # hundred mild H...H pushes moves a lipid further than the one carbon
    # contact that matters. Measured both ways, the closest heavy pair goes
    # from 2.42 A to 1.13 A while the closest hydrogen pair barely improves.
    #
    # Rigid conformers cannot resolve a sub-Angstrom hydrogen contact at all,
    # because the only way out is to change a torsion. That is what
    # step6.0_minimization is for, and it is why running it is not optional.
    rel2 = RigidLipidRelaxer(x, heavy, protein_heavy_xyz, box)
    worst2, nclash2 = rel2.worst_contact()

    x = wrap_molecules(rel2.x, box)
    stats = {
        "n_lipids": int(len(x)),
        "n_upper": int((leaflet_of > 0).sum()),
        "n_lower": int((leaflet_of < 0).sum()),
        "area_per_lipid_naive": area / max((n_upper + n_lower) / 2.0, 1.0),
        "closest_after_relax": worst,
        "closest_final": worst2,
        "n_clashes_final": nclash2,
    }
    if verbose:
        print("   final: closest inter-molecular heavy pair %.2f A (%d under %.1f A)"
              % (worst2, nclash2, CLASH))
    return BilayerResult(x, leaflet_of, stats)


def from_reference(atoms, box, resname, natoms):
    """Take the lipid coordinates of a reference system as they stand.

    For a matched pair of boxes -- the same protein in two conformations --
    reusing the first box's bilayer keeps the lipid environment identical, so
    a difference between the two is attributable to the protein rather than to
    two independent packings.
    """
    sel = atoms.resname == resname
    return make_whole(atoms.xyz[sel], box, natoms)
