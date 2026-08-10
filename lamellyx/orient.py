"""Work out how a protein sits in a membrane, from the protein alone.

Until now the only way to place a protein was `orient_from_pdb`: superimpose it
on a structure that was *already* oriented, usually by CHARMM-GUI. With nothing
to superimpose on, the model was used exactly as deposited — and a PDB whose
membrane normal is not z gets a bilayer built straight through it, sideways,
with no complaint. That silent failure is what this module removes.

Two independent estimates of the membrane normal:

**Hydrophobic slab.** The physical definition. A membrane protein has a belt of
exposed non-polar surface where the lipid tails touch it, and keeps its charged
groups out of that belt. Score a candidate slab by the transfer free energy of
the surface it encloses, using per-atom solvation parameters, and take the slab
that minimises it. This is the same idea as OPM/PPM, simplified: a rigid slab,
searched rather than gradient-minimised. It needs no symmetry and no reference.

**Symmetry axis.** For a homo-oligomer — every ion channel, so the whole HCN
project — the rotational symmetry axis *is* the membrane normal, and it can be
had exactly by superimposing one chain on the next and reading the rotation
axis. Cheap and independent of any energy model.

They are computed separately and compared. Agreement within a few degrees is
the check; disagreement means look at the structure before building anything.

Everything here is in Angstrom, like the rest of the internals.
"""

import numpy as np

from . import fileio, geom

# van der Waals radii, Angstrom (Bondi).
VDW = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
       "SE": 1.90, "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98}
VDW_DEFAULT = 1.70

# Atomic solvation parameters, kcal/(mol*A^2), after Eisenberg & McLachlan
# (1986), sign-flipped so that NEGATIVE means "burying this atom in the
# hydrocarbon core lowers the energy".
SIGMA = {"C": -0.016, "S": -0.021, "N": +0.006, "O": +0.006, "P": +0.006}
SIGMA_DEFAULT = +0.006
SIGMA_ANIONIC = +0.024
SIGMA_CATIONIC = +0.050

ANIONIC = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
CATIONIC = {("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2")}
TERMINAL_O = {"OXT", "OT1", "OT2"}

# The positive-inside rule: loops on the cytoplasmic side of a membrane
# protein are enriched in arginine and lysine. It is what decides which way up
# the protein goes once the normal is known -- the slab search and the
# symmetry axis both leave that free, and getting it wrong puts the
# extracellular face in the cytoplasm.
BASIC = {"ARG", "LYS"}
ACIDIC = {"ASP", "GLU"}

# Hydrophobic mismatch. NATURAL_THICKNESS is POPC's hydrocarbon core; a
# protein whose belt differs from it makes the bilayer stretch or compress,
# and MISMATCH_K is what that costs, kcal/(mol*A^2) of deviation squared.
# Calibrated against the buried-surface gradient measured on the HCN4
# tetramer (about 6 kcal/mol per A of thickness near 30 A).
NATURAL_THICKNESS = 30.0
MISMATCH_K = 1.0


# --------------------------------------------------------------------------
# solvent accessible surface
# --------------------------------------------------------------------------

def _sphere_points(n):
    """Roughly uniform points on the unit sphere (Fibonacci spiral)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


def _neighbour_cells(xyz, cutoff):
    """Bucket points into a dict of cells. No periodicity: this is one
    isolated protein, not a periodic box, so `geom.CellGrid` does not fit."""
    keys = np.floor(xyz / cutoff).astype(np.int64)
    cells = {}
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    return cells, keys


def sasa(xyz, radii, probe=1.4, n_points=96):
    """Solvent accessible surface area per atom, Shrake-Rupley.

    Returns an array of A^2, one per atom.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = len(xyz)
    if n == 0:
        return np.zeros(0)

    ext = radii + probe
    cutoff = float(2.0 * ext.max())
    cells, keys = _neighbour_cells(xyz, cutoff)
    unit = _sphere_points(n_points)

    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1)
               for c in (-1, 0, 1)]
    out = np.zeros(n)
    for i in range(n):
        k = tuple(keys[i])
        near = []
        for off in offsets:
            got = cells.get((k[0] + off[0], k[1] + off[1], k[2] + off[2]))
            if got:
                near.extend(got)
        near = np.array([j for j in near if j != i], dtype=np.int64)
        pts = xyz[i] + unit * ext[i]
        if len(near):
            d2 = ((pts[:, None, :] - xyz[near][None, :, :]) ** 2).sum(axis=2)
            free = ~(d2 < (ext[near] ** 2)[None, :]).any(axis=1)
        else:
            free = np.ones(len(pts), dtype=bool)
        out[i] = 4.0 * np.pi * ext[i] ** 2 * free.mean()
    return out


def atom_sigma(resname, name, element):
    """Solvation parameter for one atom, charged groups picked out by name."""
    if (resname, name) in ANIONIC or name in TERMINAL_O:
        return SIGMA_ANIONIC
    if (resname, name) in CATIONIC:
        return SIGMA_CATIONIC
    return SIGMA.get(element.upper(), SIGMA_DEFAULT)


# --------------------------------------------------------------------------
# the hydrophobic slab search
# --------------------------------------------------------------------------

def _directions(n):
    """Quasi-uniform directions covering a hemisphere (n and -n are the same
    membrane normal, so half the sphere is enough)."""
    pts = _sphere_points(2 * n)
    return pts[pts[:, 2] >= 0.0]


def _best_slab(proj, weight, half_lo, half_hi, step,
               mismatch_k=MISMATCH_K, natural=NATURAL_THICKNESS):
    """Best (centre, half-thickness, score) for one projection axis.

    Sorting once and taking prefix sums turns "sum the weights inside the
    slab" into two lookups, so the whole (centre x thickness) grid costs
    almost nothing per direction.

    The buried-surface term alone falls monotonically as the slab thickens --
    a thicker slab swallows more hydrophobic surface -- so on its own it runs
    to whatever upper bound it is given and then buys a little more by tilting
    off-axis. Measured on the HCN4 tetramer the curve is flat to within
    12 kcal/mol between 32 and 44 A, and that flatness cost up to 6 degrees of
    tilt. `mismatch_k * (thickness - natural)^2` is the hydrophobic-mismatch
    cost of deforming the bilayer, and it puts a real minimum where the lipid
    wants one.

    So the thickness this returns is a FITTED quantity pulled toward the
    lipid's natural value, not an independent measurement of the protein.
    """
    order = np.argsort(proj)
    p = proj[order]
    cum = np.concatenate([[0.0], np.cumsum(weight[order])])

    halves = np.arange(half_lo, half_hi + 1e-9, step)
    centres = np.arange(p[0] + half_lo, p[-1] - half_lo + 1e-9, step)
    if len(centres) == 0:
        centres = np.array([0.5 * (p[0] + p[-1])])

    lo = centres[:, None] - halves[None, :]
    hi = centres[:, None] + halves[None, :]
    i0 = np.searchsorted(p, lo.ravel(), side="left")
    i1 = np.searchsorted(p, hi.ravel(), side="right")
    score = (cum[i1] - cum[i0]).reshape(lo.shape)
    score = score + mismatch_k * (2.0 * halves[None, :] - natural) ** 2

    j = int(np.argmin(score))
    r, c = np.unravel_index(j, score.shape)
    return float(centres[r]), float(halves[c]), float(score[r, c])


def hydrophobic_slab(xyz, weight, n_directions=800, thickness=(20.0, 44.0),
                     step=0.5, refine=True, mismatch_k=MISMATCH_K,
                     natural=NATURAL_THICKNESS):
    """Find the slab whose enclosed surface has the lowest transfer energy.

    `weight` is sigma * SASA per atom, so the score is in kcal/mol.
    Returns (normal, centre_along_normal, half_thickness, score).
    """
    half_lo, half_hi = 0.5 * thickness[0], 0.5 * thickness[1]
    dirs = _directions(n_directions)

    best = None
    for nvec in dirs:
        c, h, s = _best_slab(xyz @ nvec, weight, half_lo, half_hi, step,
                             mismatch_k, natural)
        if best is None or s < best[3]:
            best = (nvec, c, h, s)

    if refine:
        # A cone around the winner, twice, so the answer is not limited by the
        # coarse grid spacing.
        for spread, count, fine in ((0.20, 400, 0.25), (0.05, 400, 0.25),
                                    (0.015, 300, 0.10)):
            axis = best[0]
            cand = axis + spread * _sphere_points(count)
            cand = cand / np.linalg.norm(cand, axis=1)[:, None]
            for nvec in cand:
                c, h, s = _best_slab(xyz @ nvec, weight, half_lo, half_hi,
                                     fine, mismatch_k, natural)
                if s < best[3]:
                    best = (nvec, c, h, s)
    return best


# --------------------------------------------------------------------------
# symmetry axis
# --------------------------------------------------------------------------

def symmetry_axis(atoms):
    """Rotational symmetry axis of a homo-oligomer, exactly.

    Superimposing one chain on the next gives a rotation; its axis is the
    oligomer's axis. Returns (axis, fold, angles_deg, rmsd) or None when the
    chains are not equivalent.
    """
    chains = [c for c in dict.fromkeys(atoms.chain.tolist()) if c.strip()]
    if len(chains) < 2:
        return None

    ca = {}
    for c in chains:
        sel = (atoms.chain == c) & (atoms.name == "CA")
        ca[c] = atoms.xyz[sel]
    n = len(ca[chains[0]])
    if n < 20 or any(len(ca[c]) != n for c in chains):
        return None

    # A chain whose CA atoms are collinear pins down no rotation about its own
    # line, so kabsch returns an arbitrary one and the "symmetry axis" is
    # noise that looks like an answer. Real chains are never collinear; this
    # guard is what stops a degenerate case being reported as exact.
    P = ca[chains[0]] - ca[chains[0]].mean(axis=0)
    sv = np.linalg.svd(P, compute_uv=False)
    if sv[0] <= 0 or sv[1] < 1e-3 * sv[0]:
        return None

    axes, angles, rmsds = [], [], []
    for a, b in zip(chains, chains[1:] + chains[:1]):
        R, t = geom.kabsch(ca[a], ca[b])
        moved = ca[a] @ R.T + t
        rmsds.append(float(np.sqrt(((moved - ca[b]) ** 2).sum(1).mean())))

        # Rotation axis: the eigenvector of R with eigenvalue +1.
        w, v = np.linalg.eig(R)
        k = int(np.argmin(np.abs(w - 1.0)))
        ax = np.real(v[:, k])
        ax = ax / np.linalg.norm(ax)
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0,
                                           -1.0, 1.0)))
        # n and -n describe the same axis; make them agree before averaging.
        if axes and float(ax @ axes[0]) < 0:
            ax = -ax
        axes.append(ax)
        angles.append(float(ang))

    if max(rmsds) > 2.0:
        return None                      # not actually the same chain
    axis = np.mean(axes, axis=0)
    axis = axis / np.linalg.norm(axis)
    mean_angle = float(np.mean(angles))
    fold = int(round(360.0 / mean_angle)) if mean_angle > 1.0 else 0
    return axis, fold, angles, float(max(rmsds))


# --------------------------------------------------------------------------
# the public entry point
# --------------------------------------------------------------------------

def rotation_to_z(nvec):
    """Rotation matrix taking `nvec` onto +z."""
    n = np.asarray(nvec, dtype=np.float64)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    c = float(n @ z)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def positive_inside(atoms, half, margin=3.0):
    """Net charge of the loops on each side of the slab.

    Returns (charge_below, charge_above). The more positive side is the
    cytoplasmic one, which by the usual convention is placed at negative z.
    """
    ca = atoms.name == "CA"
    z = atoms.xyz[ca, 2]
    resn = atoms.resname[ca]
    q = np.zeros(len(resn))
    q[np.isin(resn, list(BASIC))] = +1.0
    q[np.isin(resn, list(ACIDIC))] = -1.0
    below = z < -(half + margin)
    above = z > +(half + margin)
    return float(q[below].sum()), float(q[above].sum())


def _tightest_azimuth(xyz, step_deg=0.25):
    """Rotation about z giving the smallest xy bounding square."""
    th = np.radians(np.arange(0.0, 180.0, step_deg))
    x, y = xyz[:, 0], xyz[:, 1]
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    # Must match geom.rotation_z, which is [[c,-s],[s,c]]. Scanning one
    # direction and applying the other makes the box bigger, not smaller.
    xr = c * x[None, :] - s * y[None, :]
    yr = s * x[None, :] + c * y[None, :]
    side = np.maximum(xr.max(1) - xr.min(1), yr.max(1) - yr.min(1))
    k = int(np.argmin(side))
    return float(th[k]), float(side[k])


def orient(atoms, use_symmetry="auto", n_directions=800,
           thickness=(20.0, 44.0), n_points=96, probe=1.4,
           extracellular_resid=None):
    """Put a protein in the membrane frame: normal on z, core centred on z=0.

    `use_symmetry`:
        "auto"   use the symmetry axis when the protein is a homo-oligomer
                 and it agrees with the hydrophobic slab, else the slab
        "yes"    require a symmetry axis
        "no"     hydrophobic slab only

    `extracellular_resid`
        (first, last) of a residue range known to be outside the cell -- for
        HCN4, the turret (443, 460). Without it the up/down choice is an
        arbitrary tie-break, because no geometric method can tell the two
        faces of a slab apart.

    Returns (rotated_atoms, report).
    """
    heavy = atoms[atoms.element != "H"]
    radii = np.array([VDW.get(e.upper(), VDW_DEFAULT) for e in heavy.element])
    area = sasa(heavy.xyz, radii, probe=probe, n_points=n_points)
    sig = np.array([atom_sigma(r, n, e) for r, n, e
                    in zip(heavy.resname, heavy.name, heavy.element)])
    weight = sig * area

    nvec, centre, half, score = hydrophobic_slab(
        heavy.xyz, weight, n_directions=n_directions, thickness=thickness)

    sym = symmetry_axis(atoms) if use_symmetry in ("auto", "yes") else None
    if use_symmetry == "yes" and sym is None:
        raise ValueError(
            "use_symmetry='yes' but the chains are not equivalent -- this "
            "does not look like a homo-oligomer. Use 'auto' or 'no'.")

    report = {
        "method": "hydrophobic_slab",
        "hydrophobic_thickness_A": round(2.0 * half, 2),
        "transfer_energy_kcal_mol": round(score, 1),
        "n_directions": int(n_directions),
    }

    if sym is not None:
        axis, fold, angles, srms = sym
        if float(axis @ nvec) < 0:
            axis = -axis
        dev = float(np.degrees(np.arccos(np.clip(axis @ nvec, -1.0, 1.0))))
        report.update({
            "symmetry_fold": fold,
            "symmetry_axis_rmsd_A": round(srms, 3),
            "symmetry_rotation_deg": [round(a, 2) for a in angles],
            "slab_vs_symmetry_deg": round(dev, 2),
        })
        if use_symmetry == "yes" or (use_symmetry == "auto" and dev <= 15.0):
            # The symmetry axis is exact where the slab search is a numerical
            # optimum, so prefer it -- but only once the two agree, otherwise
            # the disagreement is the finding and neither should be trusted.
            nvec = axis
            report["method"] = "symmetry_axis"
            centre, half, score = _best_slab(
                heavy.xyz @ nvec, weight, 0.5 * thickness[0],
                0.5 * thickness[1], 0.10)
            report["hydrophobic_thickness_A"] = round(2.0 * half, 2)
            report["transfer_energy_kcal_mol"] = round(score, 1)
        elif use_symmetry == "auto":
            report["warning"] = (
                "the symmetry axis and the hydrophobic slab disagree by "
                "%.1f degrees -- look at the structure before building"
                % dev)

    R = rotation_to_z(nvec)
    out = atoms.copy()
    out.xyz = out.xyz @ R.T

    # Spinning about the normal is free -- the membrane cannot tell -- but it
    # is not free in cost: the box has to contain the xy bounding square, and
    # on a C4 channel the worst azimuth is ~14 A wider than the best, which is
    # a few hundred lipids and their water. Pick the tightest, which also
    # makes the output deterministic instead of depending on the input frame.
    ang, side = _tightest_azimuth(out.xyz)
    Rz = geom.rotation_z(ang)
    out.xyz = out.xyz @ Rz.T
    R = Rz @ R
    report["azimuth_deg"] = round(float(np.degrees(ang)), 2)
    report["xy_bounding_square_A"] = round(float(side), 2)

    # Slab centre to z = 0, and the membrane-spanning part centred on x,y.
    # R takes nvec onto z, so a point's new z is exactly its old projection
    # on nvec -- which is what `centre` already is.
    out.xyz[:, 2] -= float(centre)

    # --- which way up ----------------------------------------------------
    # Neither the slab nor the symmetry axis distinguishes n from -n, so the
    # protein is as likely to come out upside down as not. Nothing below is
    # cosmetic: it decides which leaflet the extracellular face meets.
    qlo, qhi = positive_inside(out, half)
    flipped = False
    if extracellular_resid:
        lo_r, hi_r = (extracellular_resid if len(extracellular_resid) == 2
                      else (min(extracellular_resid), max(extracellular_resid)))
        sel = ((out.resid >= lo_r) & (out.resid <= hi_r) & (out.name == "CA"))
        if not sel.any():
            raise ValueError("no CA atoms found in residues %d-%d -- "
                             "extracellular_resid does not match this model"
                             % (lo_r, hi_r))
        mean_z = float(out.xyz[sel, 2].mean())
        flipped = mean_z < 0.0
        report["updown"] = "from extracellular_resid %d-%d" % (lo_r, hi_r)
        report["extracellular_mean_z_A"] = round(
            -mean_z if flipped else mean_z, 2)
    else:
        # No biology given, so this is only a tie-break: it makes the output
        # deterministic instead of inheriting the input frame. It is NOT a
        # rule. The bulkier extramembrane side goes to +z, which happens to
        # be right for a construct truncated below the membrane and wrong for
        # a full-length channel carrying its cytoplasmic domain.
        heavy_out = out[out.element != "H"]
        z = heavy_out.xyz[:, 2]
        flipped = int((z < -half).sum()) > int((z > half).sum())
        report["updown"] = "arbitrary tie-break -- pass extracellular_resid"
        report["warning_updown"] = (
            "which face is extracellular was NOT determined. The bulkier "
            "side was put at +z to make the result deterministic. Pass "
            "extracellular_resid=(first, last) for a residue range known to "
            "be outside the cell.")

    if flipped:
        Rf = np.diag([1.0, -1.0, -1.0])          # 180 degrees about x
        out.xyz = out.xyz @ Rf.T
        R = Rf @ R
        qlo, qhi = qhi, qlo
    report["flipped_in_z"] = bool(flipped)
    report["positive_inside_charge_below_above"] = [qlo, qhi]
    if qhi > qlo and report["updown"].startswith("from"):
        # Worth saying out loud rather than silently disagreeing.
        report["note_positive_inside"] = (
            "the positive-inside rule would have put the cytoplasmic side at "
            "+z, disagreeing with extracellular_resid. That rule is "
            "unreliable on a construct truncated below the membrane, where "
            "the basic cytoplasmic loops are simply absent.")

    band = np.abs(out.xyz[:, 2]) <= half
    if band.sum() >= 10:
        out.xyz[:, 0] -= out.xyz[band, 0].mean()
        out.xyz[:, 1] -= out.xyz[band, 1].mean()
    else:
        out.xyz[:, 0] -= out.xyz[:, 0].mean()
        out.xyz[:, 1] -= out.xyz[:, 1].mean()

    report["normal_before"] = [round(float(v), 6) for v in nvec]
    report["rotation"] = [[round(float(v), 8) for v in row] for row in R]
    report["extent_z_A"] = [round(float(out.xyz[:, 2].min()), 2),
                            round(float(out.xyz[:, 2].max()), 2)]
    report["extent_xy_A"] = [
        round(float(out.xyz[:, 0].max() - out.xyz[:, 0].min()), 2),
        round(float(out.xyz[:, 1].max() - out.xyz[:, 1].min()), 2)]
    return out, report


def orient_pdb(pdb_in, pdb_out, **kw):
    """Read a PDB, orient it, write it back out. Returns the report."""
    atoms = fileio.read_pdb(pdb_in)
    out, report = orient(atoms, **kw)
    fileio.write_pdb(pdb_out, out, title="oriented in the membrane by lamellyx")
    report["input"] = pdb_in
    report["output"] = pdb_out
    return report


# --------------------------------------------------------------------------
# is this thing oriented at all?
# --------------------------------------------------------------------------

def orientation_check(atoms, n_points=48, tolerance_deg=20.0):
    """Cheap test that a protein is already in the membrane frame.

    Called by the builder before it packs lipids, so that an unoriented PDB
    fails loudly instead of getting a bilayer built through it sideways.
    """
    heavy = atoms[atoms.element != "H"]
    radii = np.array([VDW.get(e.upper(), VDW_DEFAULT) for e in heavy.element])
    area = sasa(heavy.xyz, radii, n_points=n_points)
    sig = np.array([atom_sigma(r, n, e) for r, n, e
                    in zip(heavy.resname, heavy.name, heavy.element)])
    weight = sig * area

    nvec, centre, half, score = hydrophobic_slab(
        heavy.xyz, weight, n_directions=400, refine=True)
    tilt = float(np.degrees(np.arccos(
        np.clip(abs(float(nvec[2])), -1.0, 1.0))))
    return {
        "tilt_from_z_deg": round(tilt, 1),
        "core_centre_z_A": round(float(centre), 2),
        "hydrophobic_thickness_A": round(2.0 * half, 2),
        "oriented": bool(tilt <= tolerance_deg and abs(centre) <= 5.0),
        "tolerance_deg": tolerance_deg,
    }
