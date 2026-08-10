"""Geometry: superposition, periodic neighbour search, local frames.

The neighbour search is the hot path -- solvating a 125,000-atom box means
asking "is anything within 2.8 A of this point" tens of thousands of times.
It is a uniform cell grid, fully vectorised, numpy only.
"""

from __future__ import annotations

import itertools

import numpy as np

# 27 cell offsets, ordered so the home cell comes first.
_OFFSETS = np.array(
    sorted(itertools.product((0, -1, 1), repeat=3), key=lambda o: sum(map(abs, o))),
    dtype=np.int64,
)


def normalise(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.where(n == 0.0, 1.0, n)


def min_image(d, box, pbc):
    """Apply the minimum-image convention in place-safe fashion.

    `d` is a displacement array (..., 3); `box` the orthorhombic box lengths.
    """
    d = np.asarray(d, dtype=np.float64).copy()
    for a in range(3):
        if pbc[a]:
            d[..., a] -= box[a] * np.round(d[..., a] / box[a])
    return d


def wrap(xyz, box, pbc=(True, True, True)):
    """Wrap atom by atom. This splits molecules -- see `wrap_molecules`."""
    out = np.asarray(xyz, dtype=np.float64).copy()
    for a in range(3):
        if pbc[a]:
            out[:, a] = np.mod(out[:, a], box[a])
    return out


def wrap_molecules(x, box, pbc=(True, True, False)):
    """Bring each molecule's centroid into the box, moving it as one piece.

    `x` is (nmol, natoms, 3). Wrapping atom by atom instead cuts molecules in
    half at the boundary. GROMACS copes with that, but nothing else does: every
    bond length, every picture and every geometric check on the output becomes
    nonsense, and the file looks fine while it happens.
    """
    x = np.asarray(x, dtype=np.float64)
    c = x.mean(axis=1, keepdims=True)
    shift = np.zeros_like(c)
    for a in range(3):
        if pbc[a]:
            shift[..., a] = -box[a] * np.floor(c[..., a] / box[a])
    return x + shift


def wrap_by_blocks(xyz, box, blocks, pbc=(True, True, False)):
    """Whole-molecule wrapping for a system laid out as [(topology, count)]."""
    out = np.array(xyz, dtype=np.float64, copy=True)
    off = 0
    for mol, count in blocks:
        n = mol.natoms * count
        if count:
            seg = out[off:off + n].reshape(count, mol.natoms, 3)
            out[off:off + n] = wrap_molecules(seg, box, pbc).reshape(-1, 3)
        off += n
    if off != len(out):
        raise ValueError("blocks describe %d atoms, got %d" % (off, len(out)))
    return out


# --------------------------------------------------------------------------
# superposition
# --------------------------------------------------------------------------

def kabsch(mobile, target):
    """Rigid transform taking `mobile` onto `target`.

    Returns (R, t) such that ``mobile @ R.T + t`` best matches `target`.
    Reflections are excluded, so R is always a proper rotation.
    """
    P = np.asarray(mobile, dtype=np.float64)
    Q = np.asarray(target, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("kabsch needs two matching (N,3) arrays")
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def rmsd(a, b):
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).sum(1).mean()))


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# local frames  (used to rebuild hydrogens and caps)
# --------------------------------------------------------------------------

def frames(p1, p2, p3):
    """Orthonormal frames from atom triples.

    Each frame has its origin at p1, x along p1->p2, and p3 in the xy plane.
    Returns (origin, R) with R's columns the basis vectors, so that
    ``world = origin + R @ local``.
    """
    p1 = np.atleast_2d(p1); p2 = np.atleast_2d(p2); p3 = np.atleast_2d(p3)
    e1 = normalise(p2 - p1)
    t = p3 - p1
    e2 = normalise(t - (t * e1).sum(1, keepdims=True) * e1)
    e3 = np.cross(e1, e2)
    R = np.stack([e1, e2, e3], axis=-1)
    return p1, R


def to_local(x, origin, R):
    return np.einsum("nji,nj->ni", R, np.atleast_2d(x) - origin)


def to_world(local, origin, R):
    return origin + np.einsum("nij,nj->ni", R, np.atleast_2d(local))


def collinearity(p1, p2, p3):
    """|sin| of the angle at p1. Near zero means the frame is ill-defined."""
    a = normalise(np.atleast_2d(p2) - np.atleast_2d(p1))
    b = normalise(np.atleast_2d(p3) - np.atleast_2d(p1))
    return np.linalg.norm(np.cross(a, b), axis=-1)


# --------------------------------------------------------------------------
# cell grid
# --------------------------------------------------------------------------

class CellGrid:
    """Uniform grid over periodic space for radius queries.

    Points must lie inside the box on any axis marked non-periodic; on
    periodic axes they are wrapped automatically.
    """

    MAX_TABLE = 200_000_000

    def __init__(self, points, box, cutoff, pbc=(True, True, True)):
        self.box = np.asarray(box, dtype=np.float64)
        self.pbc = tuple(bool(p) for p in pbc)
        self.cutoff = float(cutoff)
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive")
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.points = wrap(pts, self.box, self.pbc)

        self.nc = np.maximum(np.floor(self.box / self.cutoff).astype(np.int64), 1)
        self.cs = self.box / self.nc
        ncells = int(np.prod(self.nc))

        ci = self._cell_index(self.points)
        lin = self._linear(ci)
        counts = np.bincount(lin, minlength=ncells)
        maxper = int(counts.max()) if len(self.points) else 1
        if ncells * maxper > self.MAX_TABLE:
            # Almost always a box given in the wrong unit rather than a
            # genuinely huge system: a 12 nm box passed as 120 gives a
            # thousand times the cells for the same atoms.
            raise MemoryError(
                "cell table would be %dx%d for a %.1f x %.1f x %.1f A box "
                "holding %d points at a %.1f A cutoff -- %.0f A^3 per point, "
                "so the box is far larger than its contents. Check the units: "
                "Angstrom here, nanometres in the public API."
                % (ncells, maxper, self.box[0], self.box[1], self.box[2],
                   len(self.points), self.cutoff,
                   float(np.prod(self.box)) / max(len(self.points), 1)))
        order = np.argsort(lin, kind="stable")
        lin_sorted = lin[order]
        starts = np.zeros(ncells + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        rank = np.arange(len(lin), dtype=np.int64) - starts[lin_sorted]
        self.table = np.full((ncells, maxper), -1, dtype=np.int64)
        if len(lin):
            self.table[lin_sorted, rank] = order

    # -- internals ---------------------------------------------------------

    def _cell_index(self, x):
        ci = np.floor(x / self.cs).astype(np.int64)
        for a in range(3):
            if self.pbc[a]:
                ci[:, a] = np.mod(ci[:, a], self.nc[a])
            else:
                np.clip(ci[:, a], 0, self.nc[a] - 1, out=ci[:, a])
        return ci

    def _linear(self, ci):
        return (ci[:, 0] * self.nc[1] + ci[:, 1]) * self.nc[2] + ci[:, 2]

    def _scan(self, q):
        """Yield (candidate index array, valid mask, displacement) per offset."""
        qi = np.floor(q / self.cs).astype(np.int64)
        for off in _OFFSETS:
            ci = qi + off
            ok = np.ones(len(q), dtype=bool)
            for a in range(3):
                if self.pbc[a]:
                    ci[:, a] = np.mod(ci[:, a], self.nc[a])
                else:
                    bad = (ci[:, a] < 0) | (ci[:, a] >= self.nc[a])
                    ok &= ~bad
                    np.clip(ci[:, a], 0, self.nc[a] - 1, out=ci[:, a])
            lin = self._linear(ci)
            cand = self.table[lin]                       # (m, maxper)
            valid = (cand >= 0) & ok[:, None]
            if not valid.any():
                continue
            d = self.points[cand] - q[:, None, :]
            d = min_image(d, self.box, self.pbc)
            yield cand, valid, d

    # -- queries -----------------------------------------------------------

    def min_distance(self, q, chunk=20000):
        """Distance from every query point to the nearest stored point."""
        q = wrap(np.asarray(q, dtype=np.float64).reshape(-1, 3), self.box, self.pbc)
        out = np.full(len(q), np.inf)
        for s in range(0, len(q), chunk):
            sl = slice(s, min(s + chunk, len(q)))
            best = np.full(sl.stop - sl.start, np.inf)
            for cand, valid, d in self._scan(q[sl]):
                r2 = (d ** 2).sum(-1)
                r2[~valid] = np.inf
                np.minimum(best, r2.min(1), out=best)
            out[sl] = best
        return np.sqrt(out)

    def within(self, q, cutoff=None, chunk=20000):
        """Boolean: does each query point have a stored point within cutoff?"""
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        if cutoff > self.cutoff:
            raise ValueError("cutoff exceeds the grid spacing it was built for")
        return self.min_distance(q, chunk=chunk) < cutoff

    def pairs(self, q, cutoff=None, chunk=20000):
        """All (query index, point index, distance) pairs within cutoff."""
        cutoff = self.cutoff if cutoff is None else float(cutoff)
        if cutoff > self.cutoff:
            raise ValueError("cutoff exceeds the grid spacing it was built for")
        q = wrap(np.asarray(q, dtype=np.float64).reshape(-1, 3), self.box, self.pbc)
        qi_all, pj_all, d_all = [], [], []
        for s in range(0, len(q), chunk):
            sl = slice(s, min(s + chunk, len(q)))
            for cand, valid, d in self._scan(q[sl]):
                r = np.sqrt((d ** 2).sum(-1))
                hit = valid & (r < cutoff)
                if not hit.any():
                    continue
                rows, cols = np.nonzero(hit)
                qi_all.append(rows + s)
                pj_all.append(cand[rows, cols])
                d_all.append(r[rows, cols])
        if not qi_all:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                    np.zeros(0))
        qi_all = np.concatenate(qi_all)
        pj_all = np.concatenate(pj_all)
        d_all = np.concatenate(d_all)
        if (self.nc < 3).any():
            # In a box only one or two cells wide, the 27 offsets wrap onto
            # the same cell more than once and every pair is found several
            # times. Harmless for a minimum, but it would multiply forces and
            # inflate contact counts, so collapse to the minimum image.
            key = qi_all * len(self.points) + pj_all
            order = np.lexsort((d_all, key))
            qi_all, pj_all, d_all = qi_all[order], pj_all[order], d_all[order]
            first = np.ones(len(key), dtype=bool)
            first[1:] = key[order][1:] != key[order][:-1]
            qi_all, pj_all, d_all = qi_all[first], pj_all[first], d_all[first]
        return qi_all, pj_all, d_all


def self_min_distance(points, box, cutoff, pbc=(True, True, True)):
    """Nearest-neighbour distance within one set, ignoring self-pairs."""
    grid = CellGrid(points, box, cutoff, pbc)
    q = wrap(np.asarray(points, dtype=np.float64), box, pbc)
    out = np.full(len(q), np.inf)
    for s in range(0, len(q), 20000):
        sl = slice(s, min(s + 20000, len(q)))
        best = np.full(sl.stop - sl.start, np.inf)
        for cand, valid, d in grid._scan(q[sl]):
            r2 = (d ** 2).sum(-1)
            selfpair = cand == (np.arange(sl.start, sl.stop)[:, None])
            r2[~valid | selfpair] = np.inf
            np.minimum(best, r2.min(1), out=best)
        out[sl] = best
    return np.sqrt(out)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def farthest_point_sample(pts, n, box=None, pbc=(True, True, False), rng=None):
    """Pick `n` of `pts` that are as spread out as possible.

    Used to thin a dense lattice of candidate lipid sites down to the exact
    number of lipids wanted, without leaving a hole on one side of the box.
    """
    pts = np.asarray(pts, dtype=np.float64)
    m = len(pts)
    if n >= m:
        return np.arange(m)
    rng = np.random.default_rng() if rng is None else rng
    chosen = np.empty(n, dtype=np.int64)
    chosen[0] = rng.integers(m)
    d = np.full(m, np.inf)
    for k in range(1, n):
        diff = pts - pts[chosen[k - 1]]
        if box is not None:
            diff = min_image(diff, box, pbc)
        np.minimum(d, (diff ** 2).sum(1), out=d)
        d[chosen[:k]] = -1.0
        chosen[k] = int(np.argmax(d))
    return chosen
