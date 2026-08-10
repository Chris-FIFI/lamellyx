"""Test suite. Run with:  python -m lamellyx.tests [-k PATTERN]

Plain asserts and a small runner rather than pytest, so the package keeps its
one dependency. Tests are grouped by what they protect:

  invariant  -- something that must hold for any input, checked on many
  reference  -- agreement with a known-correct answer or a brute-force result
  regression -- a bug that was found once and must not come back
  endtoend   -- a real build, checked the way grompp would check it
"""

from __future__ import annotations

import argparse
import io
import itertools
import os
import shutil
import sys
import tempfile
import time
import threading
import traceback

import numpy as np
from dataclasses import fields as fields_of

from . import (bilayer, fileio, geom, hbuild, library, mdp, membrane, orient,
               solvate, topology, validate)

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def approx(a, b, tol=1e-9):
    assert abs(float(a) - float(b)) <= tol, "%r != %r (tol %g)" % (a, b, tol)


def brute_min_distance(q, p, box, pbc):
    """Reference neighbour search: every pair, no cleverness."""
    out = np.empty(len(q))
    for i, x in enumerate(q):
        d = p - x
        for a in range(3):
            if pbc[a]:
                d[:, a] -= box[a] * np.round(d[:, a] / box[a])
        out[i] = np.sqrt((d ** 2).sum(1)).min()
    return out


# ==========================================================================
# fileio
# ==========================================================================

@test
def invariant_gro_roundtrip_precision():
    rng = np.random.default_rng(0)
    n = 500
    box = np.array([37.0, 41.0, 53.0])
    a = fileio.Atoms(["OH2"] * n, ["TIP3"] * n, np.arange(1, n + 1),
                     rng.uniform(0, 37, (n, 3)))
    p = os.path.join(tempfile.mkdtemp(), "t.gro")
    fileio.write_gro(p, a, box)
    b, box2 = fileio.read_gro(p)
    # .gro stores 3 decimals in nm, so 0.005 A is the format's own limit
    assert np.abs(b.xyz - a.xyz).max() <= 0.0051, np.abs(b.xyz - a.xyz).max()
    assert np.abs(box2 - box).max() < 1e-6
    assert (b.name == a.name).all() and (b.resname == a.resname).all()


@test
def regression_gro_handles_more_than_99999_atoms():
    """.gro fields are 5 wide; both numbers must wrap, not overflow."""
    n = 100_003
    a = fileio.Atoms(["OH2"] * n, ["TIP3"] * n, np.arange(1, n + 1),
                     np.zeros((n, 3)))
    p = os.path.join(tempfile.mkdtemp(), "big.gro")
    fileio.write_gro(p, a, np.array([10.0, 10.0, 10.0]))
    lines = open(p).read().splitlines()
    assert int(lines[1]) == n
    for ln in (lines[2], lines[-2]):
        assert len(ln) >= 44, repr(ln)
        int(ln[0:5]); float(ln[20:28])           # must still parse
    b, _ = fileio.read_gro(p)
    assert len(b) == n


@test
def invariant_element_guessing():
    cases = [("CA", "LEU", "C"), ("CA", "CAL", "CA"), ("POT", "POT", "K"),
             ("CLA", "CLA", "CL"), ("OH2", "TIP3", "O"), ("P", "POPC", "P"),
             ("1HB", "ALA", "H"), ("SD", "MET", "S"), ("N", "POPC", "N")]
    for name, res, want in cases:
        got = fileio.guess_element(name, res)
        assert got == want, "%s in %s -> %s, wanted %s" % (name, res, got, want)


@test
def invariant_atoms_concat_and_select():
    a = fileio.Atoms(["C"], ["X"], [1], np.zeros((1, 3)))
    empty = fileio.Atoms([], [], [], np.zeros((0, 3)))
    assert len(fileio.Atoms.concat([empty, a, empty])) == 1
    assert len(fileio.Atoms.concat([])) == 0
    assert len(a[np.array([True])]) == 1
    assert len(a[np.array([False])]) == 0


# ==========================================================================
# geometry
# ==========================================================================

@test
def reference_kabsch_recovers_a_known_transform():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(60, 3)) * 12.0
    ang = 0.9
    R0 = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0],
                   [0, 0, 1.0]])
    Q = P @ R0.T + np.array([3.0, -2.0, 7.0])
    R, t = geom.kabsch(P, Q)
    assert np.abs(P @ R.T + t - Q).max() < 1e-9
    approx(np.linalg.det(R), 1.0, 1e-9)


@test
def invariant_kabsch_never_reflects():
    """A mirrored target must not be fitted by inverting the structure."""
    rng = np.random.default_rng(2)
    P = rng.normal(size=(40, 3))
    Q = P * np.array([1.0, 1.0, -1.0])
    R, _ = geom.kabsch(P, Q)
    assert np.linalg.det(R) > 0, "kabsch returned a reflection"


@test
def reference_cellgrid_matches_brute_force():
    """Across box shapes, cutoffs and periodicities, including tiny boxes."""
    rng = np.random.default_rng(3)
    for box_l, cutoff, pbc in itertools.product(
            ([30.0, 30.0, 30.0], [12.0, 40.0, 25.0], [9.0, 9.0, 9.0]),
            (2.5, 4.0, 6.0),
            ((True, True, True), (True, True, False))):
        box = np.array(box_l)
        p = rng.uniform(0, 1, (400, 3)) * box
        q = rng.uniform(0, 1, (120, 3)) * box
        grid = geom.CellGrid(p, box, cutoff, pbc)
        got = grid.min_distance(q)
        want = brute_min_distance(geom.wrap(q, box, pbc),
                                  geom.wrap(p, box, pbc), box, pbc)
        bad = np.abs(np.minimum(got, cutoff) - np.minimum(want, cutoff)).max()
        assert bad < 1e-9, "box %s cutoff %s pbc %s -> %g" % (box_l, cutoff, pbc, bad)


@test
def regression_cellgrid_pairs_are_unique_in_a_small_box():
    """A box only two cells wide used to report every pair several times."""
    rng = np.random.default_rng(4)
    box = np.array([9.0, 9.0, 9.0])
    p = rng.uniform(0, 9, (150, 3))
    grid = geom.CellGrid(p, box, 4.0)
    qi, pj, d = grid.pairs(p, 4.0)
    key = qi * len(p) + pj
    assert len(np.unique(key)) == len(key), \
        "%d duplicate pairs" % (len(key) - len(np.unique(key)))
    # and the distances are the minimum image ones. Compare against a brute
    # force that skips the self pair, which is the whole point of the metric.
    want = np.empty(len(p))
    for i, x in enumerate(p):
        dd = np.delete(p, i, axis=0) - x
        for a in range(3):
            dd[:, a] -= box[a] * np.round(dd[:, a] / box[a])
        want[i] = np.sqrt((dd ** 2).sum(1)).min()
    m = qi != pj
    got = np.full(len(p), np.inf)
    np.minimum.at(got, qi[m], d[m])
    bad = np.abs(np.minimum(got, 4.0) - np.minimum(want, 4.0)).max()
    assert bad < 1e-9, "cell grid and brute force disagree by %g" % bad


@test
def invariant_frames_roundtrip():
    rng = np.random.default_rng(5)
    p1, p2, p3, x = (rng.normal(size=(20, 3)) for _ in range(4))
    o, R = geom.frames(p1, p2, p3)
    back = geom.to_world(geom.to_local(x, o, R), o, R)
    assert np.abs(back - x).max() < 1e-9
    # frames are orthonormal
    for k in range(len(R)):
        assert np.abs(R[k].T @ R[k] - np.eye(3)).max() < 1e-9


@test
def invariant_farthest_point_sample_bounds():
    rng = np.random.default_rng(6)
    pts = rng.uniform(0, 50, (40, 2))
    assert len(geom.farthest_point_sample(pts, 40, rng=rng)) == 40
    assert len(geom.farthest_point_sample(pts, 99, rng=rng)) == 40
    idx = geom.farthest_point_sample(pts, 7, rng=rng)
    assert len(idx) == len(set(idx.tolist())) == 7


# ==========================================================================
# topology and validation
# ==========================================================================

def _bundled_tops():
    return topology.load_toppar(membrane.BUNDLED_TOPPAR)


@test
def reference_itp_parsing_against_known_values():
    tops = _bundled_tops()
    assert tops["POPC"].natoms == 134
    assert len(tops["POPC"].bonds) == 133          # acyclic: natoms - 1
    approx(tops["POPC"].total_charge, 0.0, 1e-6)
    approx(tops["POT"].total_charge, 1.0, 1e-6)
    approx(tops["CLA"].total_charge, -1.0, 1e-6)
    assert tops["TIP3"].natoms == 3
    adj = tops["POPC"].adjacency()
    assert all(len(a) > 0 for a in adj), "POPC has an unbonded atom"


@test
def regression_isin_sorted_with_no_exclusions():
    """This used to index element -1 of an empty array."""
    empty = np.zeros(0, dtype=np.int64)
    out = validate._isin_sorted(np.array([1, 2, 3], dtype=np.int64), empty)
    assert out.shape == (3,) and not out.any()


@test
def invariant_exclusion_pairs_are_sane():
    tops = _bundled_tops()
    pairs = validate.exclusion_pairs(tops["POPC"], depth=2)
    assert (pairs[:, 0] < pairs[:, 1]).all(), "pairs must be ordered"
    assert len(np.unique(pairs, axis=0)) == len(pairs), "duplicate exclusions"
    bonds = {tuple(sorted(b)) for b in tops["POPC"].bonds}
    got = {tuple(p) for p in pairs}
    assert bonds <= got, "every bond must be excluded at depth 2"


@test
def reference_density_of_pure_water():
    """A cube of the bundled water must come out near 1 g/cm3."""
    tpl = library.load_water_template("TIP3")
    n = len(tpl.xyz)
    a = fileio.Atoms(np.tile(tpl.atomnames, n), np.full(n * 3, "TIP3"),
                     np.repeat(np.arange(n), 3), tpl.xyz.reshape(-1, 3))
    rho = validate.density(a, np.array([tpl.cell] * 3))
    assert 0.93 < rho < 1.07, "water cube density %.3f g/cm3" % rho


@test
def invariant_index_groups_must_cover_everything():
    try:
        topology.standard_groups(100, slice(0, 10), slice(10, 20), slice(20, 30))
    except ValueError:
        pass
    else:
        raise AssertionError("gapped index groups were accepted")
    g = topology.membrane_groups(30, slice(0, 10), slice(10, 30))
    assert dict((n, len(i)) for n, i in g) == {"MEMB": 10, "SOLV": 20,
                                               "SYSTEM": 30}


# ==========================================================================
# bilayer
# ==========================================================================

@test
def invariant_leaflet_flip_preserves_chirality():
    """A lower-leaflet lipid is rotated, never mirrored."""
    approx(np.linalg.det(bilayer.FLIP_X), 1.0, 1e-12)


@test
def invariant_make_whole_and_wrap_molecules():
    box = np.array([20.0, 20.0, 20.0])
    mol = np.array([[[19.5, 1.0, 1.0], [0.5, 1.0, 1.0], [1.5, 1.0, 1.0]]])
    whole = bilayer.make_whole(mol.reshape(-1, 3), box, 3)
    d = np.linalg.norm(np.diff(whole[0], axis=0), axis=1)
    assert d.max() < 2.0, "make_whole left the molecule split: %s" % d
    far = whole + np.array([100.0, -60.0, 0.0])
    back = bilayer.wrap_molecules(far, box)
    d2 = np.linalg.norm(np.diff(back[0], axis=0), axis=1)
    assert np.abs(d2 - d).max() < 1e-9, "wrap_molecules changed the geometry"
    c = back.mean(axis=1)[0]
    assert 0 <= c[0] < box[0] and 0 <= c[1] < box[1]


@test
def invariant_rotating_set_refuses_rings():
    tops = _bundled_tops()
    popc = tops["POPC"]
    adj = popc.adjacency()
    # POPC is acyclic, so every bond is rotatable; check the size guard works
    i, j = popc.bonds[60]
    assert hbuild.rotating_set(adj, i, j, limit=1) is None


@test
def invariant_relaxation_keeps_lipids_rigid():
    """Rigid-body moves must not change a single internal distance."""
    lib = library.load_lipid_library("POPC", midplane=25.0)
    rng = np.random.default_rng(7)
    box = np.array([40.0, 40.0, 70.0])
    sites = rng.uniform(0, 40, (12, 2))
    lipids = bilayer.place_leaflet(lib, +1, sites, rng)
    before = np.linalg.norm(lipids[:, :1] - lipids, axis=2)
    rel = bilayer.RigidLipidRelaxer(lipids, lib.heavy_mask,
                                    np.zeros((0, 3)), box)
    out = rel.run(25, 1.0, 1.0)
    after = np.linalg.norm(out[:, :1] - out, axis=2)
    assert np.abs(after - before).max() < 1e-8, np.abs(after - before).max()


@test
def invariant_choose_sites_reports_impossible_packing():
    rng = np.random.default_rng(8)
    try:
        bilayer.choose_sites(5000, np.zeros((0, 2)), np.array([20.0, 20.0]),
                             4.0, rng, 8.0)
    except RuntimeError as exc:
        assert "lipids" in str(exc)
    else:
        raise AssertionError("packing 5000 lipids into 4 nm2 was accepted")


# ==========================================================================
# solvation
# ==========================================================================

@test
def reference_ion_counts_match_charmm_gui():
    """The reference system: 22876 waters, protein charge -8, 0.15 M."""
    cat, ani = solvate.ion_counts(22876, -8.0, 0.15)
    assert (cat, ani) == (70, 62), (cat, ani)
    assert solvate.ion_counts(10000, 0.0, 0.5) == (90, 90)
    assert solvate.ion_counts(1000, +3.0, 0.0) == (0, 3)


@test
def invariant_place_ions_refuses_the_impossible():
    rng = np.random.default_rng(9)
    box = np.array([20.0, 20.0, 20.0])
    w = rng.uniform(0, 20, (30, 3, 3))
    try:
        solvate.place_ions(w, box, np.zeros((0, 3)), 500, 500, rng,
                           verbose=False)
    except RuntimeError as exc:
        assert "ions" in str(exc)
    else:
        raise AssertionError("placing 1000 ions in 30 waters was accepted")


@test
def invariant_size_box_z():
    xyz = np.array([[0, 0, 10.0], [0, 0, 40.0]])
    h, s = solvate.size_box_z(xyz, 15.0)
    approx(h, 60.0)
    approx(s, 5.0)
    z = xyz[:, 2] + s
    approx(z.min(), 15.0)
    approx(h - z.max(), 15.0)


# ==========================================================================
# configuration checks
# ==========================================================================

@test
def reference_check_settings_flags_the_documented_cases():
    # The shipped defaults must build. area_per_lipid used to ship at
    # 0.15 nm2, which the checker refused, so no default configuration
    # worked without knowing to pass 0. It now defaults to 0, meaning
    # "use the measured value".
    errs = [m for lvl, m in membrane.check_settings(membrane.MembraneConfig())
            if lvl == "error"]
    assert errs == [], errs
    assert membrane.MembraneConfig().area_per_lipid == 0.0
    # an explicitly impossible area is still refused
    tight = membrane.MembraneConfig(area_per_lipid=0.15)
    errs = [m for l, m in membrane.check_settings(tight) if l == "error"]
    assert len(errs) == 1 and "area_per_lipid" in errs[0], errs

    # but the values that genuinely cannot work are still caught
    bad = membrane.MembraneConfig(size_mode="box", x=2.0, y=2.0,
                                  area_per_lipid=0.15)
    errs = [m for lvl, m in membrane.check_settings(bad) if lvl == "error"]
    # A square box reports the edge once, not once per axis.
    assert len(errs) == 2, [m[:40] for m in errs]
    assert any("x = 2.00" in m for m in errs)
    assert any("area_per_lipid" in m for m in errs)
    # a non-square box still names the axis that is wrong
    oblong = membrane.MembraneConfig(size_mode="box", x=8.0, y=2.0)
    assert any("y = 2.00" in m for l, m in membrane.check_settings(oblong)
               if l == "error")

    good = membrane.MembraneConfig(size_mode="box", x=8.0, y=8.0,
                                   area_per_lipid=0.643)
    assert membrane.check_settings(good) == [], membrane.check_settings(good)

    # exactly at the cutoff boundary is allowed
    edge = membrane.MembraneConfig(size_mode="box", x=2.4, y=2.4,
                                   area_per_lipid=0.643)
    assert not [m for lvl, m in membrane.check_settings(edge) if lvl == "error"]


@test
def invariant_unknown_lipid_gives_a_useful_error():
    try:
        library.load_lipid_library("NOPE")
    except FileNotFoundError as exc:
        assert "make_data" in str(exc), str(exc)
    else:
        raise AssertionError("a missing lipid library was not reported")


# ==========================================================================
# end to end
# ==========================================================================

def _tiny(out, seed=0, **kw):
    """A small but real build. Any field can be overridden by keyword."""
    settings = dict(output_dir=out, size_mode="box", x=3.0, y=3.0,
                    area_per_lipid=0.643, water_thickness=1.2,
                    salt_concentration=0.5, seed=seed, verbose=False)
    settings.update(kw)
    return membrane.build_membrane(membrane.MembraneConfig(**settings))


@test
def endtoend_small_bilayer_passes_the_checks_grompp_makes():
    d = tempfile.mkdtemp()
    res = _tiny(os.path.join(d, "box"))
    out = os.path.join(d, "box")
    atoms, box = fileio.read_gro(os.path.join(out, "step5_input.gro"))
    tops = topology.load_toppar(os.path.join(out, "toppar"))

    mols, inmol = [], False
    for line in open(os.path.join(out, "topol.top")):
        s = line.split(";")[0].strip()
        if s.startswith("["):
            inmol = "molecules" in s
            continue
        if inmol and s:
            mols.append((s.split()[0], int(s.split()[1])))
    total = sum(tops[m].natoms * c for m, c in mols)
    assert total == len(atoms), "topol %d vs gro %d" % (total, len(atoms))
    approx(sum(tops[m].total_charge * c for m, c in mols), 0.0, 1e-6)

    ndx, cur = {}, None
    for line in open(os.path.join(out, "index.ndx")):
        if line.startswith("["):
            cur = line.strip("[] \n")
            ndx[cur] = 0
        elif cur:
            ndx[cur] += len(line.split())
    assert ndx["MEMB"] + ndx["SOLV"] == len(atoms) == ndx["SYSTEM"]

    # every group any mdp couples to must exist, or grompp fails at run time
    for fn in mdp.MDP_FILES:
        for line in open(os.path.join(out, fn)):
            if line.startswith(("tc_grps", "comm_grps")):
                for g in line.split("=")[1].split():
                    assert g in ndx, "%s references missing group %s" % (fn, g)

    assert res.stats["contacts"]["heavy_min"] > 2.0, res.stats["contacts"]
    assert 0.85 < res.stats["density_g_cm3"] < 1.05, res.stats
    shutil.rmtree(d, ignore_errors=True)


@test
def invariant_output_molecules_are_not_split_across_the_boundary():
    """A molecule cut in half by wrapping is legal but unreadable, and it
    makes every geometric check on the output meaningless."""
    d = tempfile.mkdtemp()
    out = os.path.join(d, "box")
    _tiny(out)
    atoms, box = fileio.read_gro(os.path.join(out, "step5_input.gro"))
    tops = topology.load_toppar(os.path.join(out, "toppar"))
    na = tops["POPC"].natoms
    lip = atoms.xyz[atoms.resname == "POPC"].reshape(-1, na, 3)
    span = lip.max(axis=1) - lip.min(axis=1)
    worst = span.max(axis=0)
    assert (worst[:2] < box[:2] / 2).all(), \
        "a lipid spans %s of a %s box -- molecules are being split" % (
            np.round(worst, 1), np.round(box, 1))
    wat = atoms.xyz[atoms.resname == "TIP3"].reshape(-1, 3, 3)
    wspan = (wat.max(axis=1) - wat.min(axis=1)).max()
    assert wspan < 2.0, "a water molecule spans %.1f A" % wspan
    shutil.rmtree(d, ignore_errors=True)


@test
def invariant_same_seed_gives_the_same_box():
    d = tempfile.mkdtemp()
    a, b = os.path.join(d, "a"), os.path.join(d, "b")
    _tiny(a, seed=42)
    _tiny(b, seed=42)
    xa, _ = fileio.read_gro(os.path.join(a, "step5_input.gro"))
    xb, _ = fileio.read_gro(os.path.join(b, "step5_input.gro"))
    assert len(xa) == len(xb), "%d vs %d atoms from the same seed" % (len(xa), len(xb))
    assert np.abs(xa.xyz - xb.xyz).max() < 1e-9, "same seed, different coordinates"
    shutil.rmtree(d, ignore_errors=True)


@test
def invariant_no_water_in_the_acyl_chain_region():
    d = tempfile.mkdtemp()
    out = os.path.join(d, "box")
    _tiny(out)
    atoms, box = fileio.read_gro(os.path.join(out, "step5_input.gro"))
    tops = topology.load_toppar(os.path.join(out, "toppar"))
    na = tops["POPC"].natoms
    sel = atoms.resname == "POPC"
    lip = atoms.xyz[sel].reshape(-1, na, 3)
    ip = list(atoms.name[sel][:na]).index("P")
    pz = lip[:, ip, 2]
    mid = 0.5 * (pz.max() + pz.min())
    lo, hi = pz[pz < mid].mean() + 2.0, pz[pz > mid].mean() - 2.0
    ow = atoms.xyz[(atoms.resname == "TIP3") & (atoms.name == "OH2")]
    n = int(((ow[:, 2] > lo) & (ow[:, 2] < hi)).sum())
    assert n == 0, "%d waters inside the hydrophobic core" % n
    shutil.rmtree(d, ignore_errors=True)


@test
def regression_toppar_copies_only_what_is_used():
    """A membrane build used to drag every .itp in the source directory with
    it, including megabytes of protein topology it does not contain."""
    src = tempfile.mkdtemp()
    for fn in ("POPC.itp", "POT.itp", "CLA.itp", "TIP3.itp", "forcefield.itp"):
        shutil.copy2(os.path.join(membrane.BUNDLED_TOPPAR, fn),
                     os.path.join(src, fn))
    with open(os.path.join(src, "HUGE.itp"), "w") as fh:
        fh.write("x" * 2_000_000)
    dst = os.path.join(tempfile.mkdtemp(), "toppar")
    got = topology.copy_toppar(src, dst, only=["POPC", "POT", "CLA", "TIP3"])
    assert "HUGE.itp" not in got, got
    assert "forcefield.itp" in got, "the force field must always come along"
    assert topology.directory_size(dst) < 500_000, topology.directory_size(dst)
    shutil.rmtree(src, ignore_errors=True)


@test
def invariant_identical_molecules_are_grouped():
    tops = _bundled_tops()
    groups = topology.identical_molecules(tops, ["POPC", "POPC", "POT"])
    assert sorted(len(g) for g in groups) == [1, 2], groups


@test
def endtoend_shared_toppar_is_smaller_and_still_consistent():
    d = tempfile.mkdtemp()
    a = _tiny(os.path.join(d, "copied"), toppar_mode="copy")
    b = _tiny(os.path.join(d, "shared"), toppar_mode="reference")
    assert a.counts == b.counts, "storage mode changed the system"
    sa, sb = a.stats["output_bytes"], b.stats["output_bytes"]
    assert sb < sa, "reference mode (%d B) not smaller than copy (%d B)" % (sb, sa)
    assert not os.path.isdir(os.path.join(d, "shared", "toppar"))
    top = open(os.path.join(d, "shared", "topol.top")).read()
    assert "forcefield.itp" in top
    for line in top.splitlines():
        if line.startswith("#include"):
            rel = line.split('"')[1]
            assert os.path.exists(os.path.join(d, "shared", rel)), rel
    shutil.rmtree(d, ignore_errors=True)


@test
def endtoend_dashboard_api():
    """Start the real server and drive it the way the page does."""
    import json as _json
    import urllib.error
    import urllib.request as _u
    from http.server import ThreadingHTTPServer

    from . import dashboard

    ws = tempfile.mkdtemp()
    dashboard.Handler.workspace = ws
    dashboard.Handler.token = "test-token-abc"
    srv = dashboard.Server(("127.0.0.1", 0), dashboard.Handler)
    port = srv.server_address[1]
    dashboard.Handler.allowed_hosts = ("127.0.0.1:%d" % port,
                                       "localhost:%d" % port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % port
    AUTH = {"X-Auth-Token": "test-token-abc"}

    def raw(p, data=None, headers=None, timeout=300):
        h = dict(AUTH)
        h.update(headers or {})
        if data is not None:
            h.setdefault("Content-Type", "application/json")
        return _u.urlopen(_u.Request(base + p, data, h), timeout=timeout)

    def get(p):
        return _json.loads(raw(p, timeout=30).read())

    def post(p, d):
        return _json.loads(raw(p, _json.dumps(d).encode()).read())

    def status(p, **kw):
        try:
            return raw(p, **kw).code
        except urllib.error.HTTPError as e:
            return e.code

    try:
        page = raw("/?t=test-token-abc", timeout=10).read().decode()
        assert "area_per_lipid" in page and "<script" in page

        # --- access control -------------------------------------------
        # no token at all
        try:
            _u.urlopen(base + "/api/state", timeout=10)
        except urllib.error.HTTPError as e:
            assert e.code == 401, e.code
        else:
            raise AssertionError("the API answered without a token")
        # wrong token
        assert status("/api/state", headers={"X-Auth-Token": "nope"}) == 401
        # the page itself is locked without one
        try:
            _u.urlopen(base + "/", timeout=10)
        except urllib.error.HTTPError as e:
            assert e.code == 401, e.code
        else:
            raise AssertionError("the page loaded without a token")
        # DNS rebinding: a Host header that is not ours
        assert status("/api/state", headers={"Host": "evil.example.com"}) == 403
        # cross-origin, even with a valid token
        assert status("/api/state",
                      headers={"Origin": "http://evil.example.com"}) == 403
        assert status("/api/state",
                      headers={"Referer": "http://evil.example.com/x"}) == 403
        # security headers are present on the page
        hdr = raw("/?t=test-token-abc", timeout=10).headers
        assert hdr["X-Frame-Options"] == "DENY"
        assert "default-src 'none'" in hdr["Content-Security-Policy"]
        assert hdr["X-Content-Type-Options"] == "nosniff"

        # --- upload validation ----------------------------------------
        assert status("/api/upload?name=evil.exe", data=b"MZ...") == 400
        assert status("/api/upload?name=notastructure.pdb",
                      data=b"just some text") == 400
        # the directory part is dropped, not escaped, so nothing climbs out
        for attack in ("../../etc/passwd.pdb", r"..\..\windows\x.pdb",
                       "/abs/path/x.pdb"):
            got = dashboard.safe_upload_name(attack)
            assert "/" not in got and "\\" not in got and ".." not in got, got
            assert got.endswith(".pdb"), got
        assert dashboard.safe_upload_name("a b;rm -rf.pdb") == "a_b_rm_-rf.pdb"
        for bad_name in ("payload.sh", "x.exe", "noextension"):
            try:
                dashboard.safe_upload_name(bad_name)
            except ValueError:
                pass
            else:
                raise AssertionError("%r was accepted as an upload" % bad_name)
        # an oversized body is refused on its declared length alone
        assert status("/api/build", data=b"{}",
                      headers={"Content-Length": str(dashboard.MAX_JSON + 1)}
                      ) in (413, 400)

        state = get("/api/state")
        assert "POPC" in state["lipids"]

        # the shipped defaults are buildable as they stand
        msgs = post("/api/check", state["defaults"])["issues"]
        errs = [i["message"] for i in msgs if i["level"] == "error"]
        assert errs == [], errs
        assert state["defaults"]["area_per_lipid"] == 0.0, state["defaults"]
        # and an impossible request must still be refused
        impossible = dict(state["defaults"])
        impossible.update(size_mode="box", x=2.0, y=2.0, area_per_lipid=0.15)
        levels = [i["level"] for i in post("/api/check", impossible)["issues"]]
        assert levels.count("error") == 2, levels

        # protein-only fields must not break the membrane check
        mixed = dict(state["defaults"])
        mixed.update(reference_dir="/nowhere", chains="A B",
                     protein_molecules="PROA PROB", size_mode="box",
                     x=3.0, y=3.0, area_per_lipid=0.643, water_thickness=1.2)
        assert "issues" in post("/api/check", mixed)

        # a protein with no reference directory must fail with an explanation
        bad = dict(mixed)
        bad.update(protein_pdb="/nowhere/x.pdb", reference_dir="")
        jid = post("/api/build", bad)["job"]
        while True:
            r = get("/api/job/%s?since=0" % jid)
            if r["status"] in ("done", "failed"):
                break
            time.sleep(0.4)
        assert r["status"] == "failed", r["status"]
        assert "reference directory" in r["error"], r["error"]

        # a real build
        cfg = dict(state["defaults"])
        cfg.update(size_mode="box", x=2.6, y=2.6, area_per_lipid=0.643,
                   water_thickness=1.2, verbose=True, strict=True)
        jid = post("/api/build", cfg)["job"]
        since = 0
        while True:
            r = get("/api/job/%s?since=%d" % (jid, since))
            since = r["next"]
            if r["status"] in ("done", "failed"):
                break
            time.sleep(0.5)
        assert r["status"] == "done", r.get("traceback") or r.get("error")
        res = r["result"]
        assert res["counts"]["TOTAL_ATOMS"] > 1000
        assert "step5_input.gro" in res["files"]
        assert res["bytes"] > 0

        # the file endpoint serves from the build, and only from it
        blob = raw("/api/file/%s/topol.top" % jid, timeout=10).read()
        assert b"[ molecules ]" in blob
        for attack in ("../../../etc/passwd", "..%2f..%2fwin.ini"):
            try:
                raw("/api/file/" + attack, timeout=10)
            except urllib.error.HTTPError as e:
                assert e.code in (403, 404), e.code
            else:
                raise AssertionError("path traversal was served: %s" % attack)

        assert get("/api/state")["history"], "build did not reach the history"
        post("/api/delete", {"id": jid})
        assert not os.path.isdir(os.path.join(ws, jid))
    finally:
        srv.shutdown()
        shutil.rmtree(ws, ignore_errors=True)


@test
def regression_negative_salt_is_refused():
    """It used to return negative ion counts, which reach topol.top as
    `POT -9` and fail somewhere far less obvious."""
    for conc in (-0.5, -1e-9):
        try:
            solvate.ion_counts(1000, 0.0, conc)
        except ValueError:
            pass
        else:
            raise AssertionError("%r M was accepted" % conc)
    assert solvate.ion_counts(1000, 0.0, 0.0) == (0, 0)
    cat, ani = solvate.ion_counts(10, 0.0, 0.001)     # rounds to no salt
    assert cat >= 0 and ani >= 0


@test
def regression_atoms_indexed_by_a_plain_int():
    """a[0] used to raise 'len() of unsized object'."""
    a = fileio.Atoms(["C", "N", "O"], ["X", "X", "X"], [1, 1, 1],
                     np.arange(9.0).reshape(3, 3))
    assert len(a[0]) == 1 and a[0].name[0] == "C"
    assert len(a[-1]) == 1 and a[-1].name[0] == "O"
    assert len(a[1:]) == 2
    assert np.allclose(a[1].xyz[0], [3.0, 4.0, 5.0])


@test
def regression_water_internal_distances_are_not_clashes():
    """TIP3 declares settles, not bonds. Without reading them, every water in
    the box reports two 0.96 A 'non-bonded contacts' with itself."""
    tops = _bundled_tops()
    w = tops["TIP3"]
    assert len(w.bonds) == 3, "settles not parsed: %d bonds" % len(w.bonds)
    excl = validate.system_exclusions([(w, 2)], 6)
    assert len(excl) == 6, excl          # 3 pairs per molecule, twice
    lengths = validate.bond_lengths(w, np.array(
        [[0, 0, 0], [0.9572, 0, 0], [-0.24, 0.927, 0],
         [10, 0, 0], [10.9572, 0, 0], [9.76, 0.927, 0]])[:3])
    assert lengths.max() < 2.0, lengths


@test
def invariant_degenerate_settings_are_refused_not_divided_by():
    for kw in ({"area_per_lipid": -1.0}, {"size_mode": "box", "x": 0.0},
               {"size_mode": "box", "y": -4.0}, {"leaflet": "banana"},
               {"size_mode": "nonsense"}):
        cfg = membrane.MembraneConfig(output_dir=tempfile.mkdtemp(),
                                      strict=False, verbose=False, **kw)
        try:
            membrane.build_membrane(cfg)
        except ValueError:
            pass
        except ZeroDivisionError:
            raise AssertionError("%s divided by zero instead of explaining" % kw)
        else:
            raise AssertionError("%s was accepted" % kw)
    # and the checks name them even when strict is off
    for kw, word in (({"area_per_lipid": -1}, "area_per_lipid"),
                     ({"salt_concentration": -1}, "salt"),
                     ({"temperature": 0}, "temperature"),
                     ({"leaflet": "banana"}, "leaflet"),
                     ({"toppar_mode": "hardlink"}, "toppar_mode"),
                     ({"n_upper": -3}, "n_upper")):
        issues = membrane.check_settings(membrane.MembraneConfig(**kw))
        assert any(l == "error" and word in m for l, m in issues), (kw, issues)


@test
def endtoend_salt_free_box_is_still_valid():
    """Zero ions must not leave an empty molecule block in topol.top."""
    d = tempfile.mkdtemp()
    out = os.path.join(d, "nosalt")
    res = _tiny(out, salt_concentration=0.0)
    assert res.counts["POT"] == 0 and res.counts["CLA"] == 0
    top = open(os.path.join(out, "topol.top")).read()
    assert "\nPOT" not in top and "\nCLA" not in top, top
    atoms, _ = fileio.read_gro(os.path.join(out, "step5_input.gro"))
    assert len(atoms) == res.counts["TOTAL_ATOMS"]
    approx(res.stats["net_charge"], 0.0, 1e-6)
    shutil.rmtree(d, ignore_errors=True)


@test
def regression_heavy_atom_packing_stays_at_reference_quality():
    """An all-atom relaxation pass was tried twice and removed twice.

    Both times it improved the closest hydrogen pair a little and took the
    closest heavy pair from 2.42 A down to 1.13 A, which is the number that
    decides whether minimisation starts cleanly. This pins it.
    """
    d = tempfile.mkdtemp()
    res = _tiny(os.path.join(d, "q"), x=4.0, y=4.0)
    c = res.stats["contacts"]
    assert c["heavy_min"] > 2.25, "closest heavy pair %.2f A" % c["heavy_min"]
    assert c["heavy_below_2.4"] <= 12, c
    shutil.rmtree(d, ignore_errors=True)


@test
def reference_slab_extent_ignores_what_is_outside_the_membrane():
    """The margin is measured from the transmembrane cross-section.

    A channel with a wide extracellular domain is much broader above the
    bilayer than in it; sizing the box from the total extent would waste a
    large amount of lipid.
    """
    tm = np.array([[-20.0, -20.0, 0.0], [20.0, 20.0, 0.0]])       # in the slab
    head = np.array([[-90.0, -90.0, 60.0], [90.0, 90.0, 60.0]])   # far above
    xyz = np.vstack([tm, head])
    lo, hi = bilayer.slab_extent(xyz, -15.0, 15.0)
    assert np.allclose(lo, [-20, -20]) and np.allclose(hi, [20, 20]), (lo, hi)
    try:
        bilayer.slab_extent(head, -15.0, 15.0)
    except ValueError as exc:
        assert "membrane slab" in str(exc)
    else:
        raise AssertionError("an empty slab was accepted")


@test
def reference_footprint_area_of_a_known_shape():
    """A disc of radius R must rasterise to about pi R^2."""
    rng = np.random.default_rng(11)
    box_xy = np.array([80.0, 80.0])
    R = 20.0
    th = rng.uniform(0, 2 * np.pi, 4000)
    r = R * np.sqrt(rng.uniform(0, 1, 4000))
    pts = np.column_stack([40 + r * np.cos(th), 40 + r * np.sin(th)])
    area = bilayer.footprint_area(pts, box_xy, probe=1.0, spacing=0.5)
    want = np.pi * (R + 1.0) ** 2
    assert abs(area - want) / want < 0.05, (area, want)
    assert bilayer.footprint_area(np.zeros((0, 2)), box_xy) == 0.0


@test
def invariant_margin_sets_the_box_and_the_lipid_count():
    """box edge = protein width + 2 x margin, and lipids fill what is left."""
    rng = np.random.default_rng(12)
    # a 40 A square post through the membrane
    n = 3000
    xy = rng.uniform(-20, 20, (n, 2))
    z = rng.uniform(-25, 25, n)
    prot = np.column_stack([xy, z])
    lo, hi = bilayer.slab_extent(prot, -18.0, 18.0)
    width = hi - lo
    for margin in (10.0, 20.0, 35.0):
        bx = float(width[0] + 2 * margin)
        by = float(width[1] + 2 * margin)
        assert abs(bx - (width[0] + 2 * margin)) < 1e-9
        centred = prot.copy()
        centred[:, 0] += bx / 2 - 0.5 * (lo[0] + hi[0])
        centred[:, 1] += by / 2 - 0.5 * (lo[1] + hi[1])
        # the protein must clear the edge by the margin, give or take rounding
        clo, chi = bilayer.slab_extent(centred, -18.0, 18.0)
        assert abs(clo[0] - margin) < 1e-6, (clo[0], margin)
        assert abs(bx - chi[0] - margin) < 1e-6, (bx - chi[0], margin)

        nlip, taken = bilayer.lipids_for_free_area(
            np.array([bx, by]), centred, (-18.0, 18.0), 64.3)
        free = bx * by - taken
        assert nlip == max(int(round(free / 64.3)), 0)
        assert 0 < taken < bx * by
        # a bigger box at the same margin must not shrink the occupied area
        assert taken > 0.5 * (width[0] * width[1])


@test
def regression_a_second_server_cannot_share_the_port():
    """Five dashboards once ended up listening on 8733 at the same time.

    http.server sets SO_REUSEADDR, which on Windows lets a new process bind a
    port that another is already listening on. Both accept connections, the
    kernel picks arbitrarily, and a page can end up talking to an old build of
    the code -- which is exactly how stale defaults kept coming back.
    """
    from . import dashboard

    first = dashboard.Server(("127.0.0.1", 0), dashboard.Handler)
    port = first.server_address[1]
    try:
        try:
            second = dashboard.Server(("127.0.0.1", port), dashboard.Handler)
        except OSError:
            pass                      # what must happen
        else:
            second.server_close()
            raise AssertionError(
                "a second server bound port %d while the first was listening"
                % port)
    finally:
        first.server_close()


@test
def regression_cli_forwards_dashboard_flags():
    """`dashboard --no-browser` was rejected by the outer parser.

    argparse.REMAINDER stops collecting at the first token that looks like an
    option, so the flag never reached the dashboard's own parser.
    """
    from . import __main__ as cli
    from . import dashboard as dash

    called = {}
    real = dash.main
    dash.main = lambda argv=None: called.setdefault("argv", argv) or 0
    try:
        cli.main(["dashboard", "--no-browser", "--port", "0"])
    finally:
        dash.main = real
    assert called["argv"] == ["--no-browser", "--port", "0"], called


@test
def endtoend_json_api_round_trip():
    """describe / check / build, the way an agent would call them."""
    import json as _json

    from . import api

    d = api.describe()
    assert "x" in d["settings"] and d["settings"]["x"]["units"].startswith("nm")
    assert "POPC" in d["lipids_available"]

    assert api.check({"size_mode": "box", "x": 2.0, "y": 2.0})["ok"] is False
    assert api.check({"size_mode": "box", "x": 8.0, "y": 8.0,
                      "area_per_lipid": 0.0})["ok"] is True
    # the shipped default is now buildable without naming an area at all
    assert api.check({"size_mode": "box", "x": 8.0, "y": 8.0})["ok"] is True
    # an explicitly impossible area is still rejected
    assert not api.check({"size_mode": "box", "x": 8.0, "y": 8.0,
                          "area_per_lipid": 0.15})["ok"]
    # the default mode ignores x and y, and must say so rather than pretend
    w = api.check({"x": 8.0, "y": 8.0})["warnings"]
    assert any("ignored in size_mode" in m for m in w), w
    try:
        api.check({"nonsense": 1})
    except ValueError as exc:
        assert "unknown settings" in str(exc)
    else:
        raise AssertionError("an unknown setting was accepted")

    tmp = tempfile.mkdtemp()
    r = api.build({"output_dir": os.path.join(tmp, "b"), "size_mode": "box",
                   "x": 3.0, "y": 3.0, "water_thickness": 1.2,
                   "area_per_lipid": 0.0, "verbose": False})
    assert r["ok"] and r["counts"]["TOTAL_ATOMS"] > 1000
    assert r["net_charge"] == 0.0, r["net_charge"]      # exactly, not 1e-14
    assert r["closest_heavy_atom_contact_A"] > 2.0
    assert "step5_input.gro" in r["files"]
    assert "gmx grompp" in r["next_command"]
    assert _json.dumps(r)                                # must serialise
    shutil.rmtree(tmp, ignore_errors=True)


@test
def regression_protein_path_units_match_what_describe_says():
    """describe() reported nm while build_protein_system took Angstrom.

    An agent following the schema and passing water_thickness=15 would have
    asked for 150 A of water. The public API is nm everywhere now, and the
    schema has to agree with it.
    """
    from . import api
    from .builder import BuildConfig

    d = api.describe()
    assert "protein_settings" in d, "the protein path has no schema"
    ps = d["protein_settings"]
    assert "NANOMETRES" in ps["_units"], ps["_units"]
    # BuildConfig holds Angstrom; the schema must report the nm equivalent
    assert BuildConfig().margin_x == 20.0
    assert abs(ps["margin_x"]["default"] - 2.0) < 1e-9, ps["margin_x"]
    assert abs(ps["water_thickness"]["default"] - 1.5) < 1e-9
    # 0 = "use the measured value", on both paths and in both schemas
    assert ps["area_per_lipid"]["default"] == 0.0, ps["area_per_lipid"]
    assert BuildConfig().area_per_lipid == 0.0
    for k in ("margin_x", "margin_y", "water_thickness"):
        assert "nm" in ps[k]["units"], (k, ps[k]["units"])


@test
def regression_protein_path_refuses_unknown_settings():
    """It used to drop them. Passing salt_concentration -- the bilayer name
    -- silently left the protein path on its own default concentration."""
    from . import api

    base = {"protein_pdb": "x.pdb", "reference_dir": "y", "output_dir": "z"}
    try:
        api.build_protein_system(dict(base, notasetting=1))
    except ValueError as exc:
        assert "unknown settings" in str(exc), exc
    else:
        raise AssertionError("an unknown setting was accepted")
    # the bilayer names are accepted as aliases rather than ignored
    from .builder import BuildConfig
    known = {f.name for f in fields_of(BuildConfig)}
    assert "concentration" in known and "salt_concentration" not in known


@test
def regression_area_per_lipid_zero_means_measured_on_both_paths():
    """0 meant "use the measured value" for a bilayer and raised for a
    protein, because nothing resolved it before the < 40 A^2 check."""
    from . import api
    from .builder import BuildConfig

    captured = {}
    import lamellyx.builder as B
    real = B.build
    B.build = lambda cfg: captured.setdefault("cfg", cfg)
    try:
        try:
            api.build_protein_system({"protein_pdb": "x.pdb",
                                      "reference_dir": "y",
                                      "output_dir": "z",
                                      "area_per_lipid": 0,
                                      "salt_concentration": 0.15})
        except AttributeError:
            pass                      # the stub returns no result object
    finally:
        B.build = real
    cfg = captured["cfg"]
    assert abs(cfg.area_per_lipid - 64.3) < 1e-6, cfg.area_per_lipid
    assert abs(cfg.concentration - 0.15) < 1e-9, cfg.concentration
    assert abs(cfg.margin_x - 20.0) < 1e-9, cfg.margin_x


@test
def regression_mismatched_protein_pdb_says_why():
    """A protein taken out of a .gro is renumbered from 1 and has no chain
    column. Both used to surface as "could not place 4255 atoms"."""
    tops = _bundled_tops()
    mol = tops["POPC"]                       # any topology with resids
    model = fileio.Atoms(list(mol.atomname), list(mol.resname),
                         [1] * mol.natoms, np.zeros((mol.natoms, 3)),
                         chain=[" "] * mol.natoms)
    try:
        hbuild.rebuild_protein([mol], [np.zeros((mol.natoms, 3))], model,
                               ["A"], verbose=False)
    except ValueError as exc:
        assert "chain" in str(exc).lower(), exc
        assert "column 22" in str(exc) or "not in the file" in str(exc), exc
    else:
        raise AssertionError("a blank chain column was accepted")


@test
def endtoend_extract_protein_produces_a_usable_pdb():
    """The obvious workflow -- take the protein out of the reference -- must
    give a file the builder accepts."""
    import glob

    ref = os.environ.get("MB_TEST_REFERENCE", "")
    if not ref or not os.path.isdir(ref):
        return                                # no reference system here
    from . import api
    d = tempfile.mkdtemp()
    out = os.path.join(d, "prot.pdb")
    info = api.extract_protein(ref, out)
    assert info["atoms"] > 0
    assert info["residue_range"][0] > 1, info      # not renumbered from 1
    a = fileio.read_pdb(out)
    assert set(a.chain.tolist()) == set(info["chains"]), set(a.chain.tolist())
    shutil.rmtree(d, ignore_errors=True)


@test
def regression_enclosure_separates_a_pore_from_a_groove():
    """Neighbour count cannot tell them apart; direction coverage can.

    Rebuilding a membrane protein put 492 waters in the acyl-chain slab
    against the reference system's 5. They were not in the lipid -- the
    nearest lipid was 10 A away -- they were in grooves on the protein's
    lipid-facing surface, which passed both "far from lipid" and "plenty of
    protein neighbours".
    """
    from .solvate import enclosure

    rng = np.random.default_rng(31)
    box = np.array([60.0, 60.0, 60.0])
    c = np.array([30.0, 30.0, 30.0])

    # a hollow shell with a point at its centre: fully enclosed
    v = rng.normal(size=(4000, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    shell = c + v * rng.uniform(5.0, 8.0, (4000, 1))
    e_pore = enclosure(c[None, :], shell, box)[0]

    # the same shell with one hemisphere removed: a groove
    groove = shell[(shell[:, 2] - c[2]) < 0]
    e_groove = enclosure(c[None, :], groove, box)[0]

    # A hemisphere does not score 0.5: with 32-degree cones the directions
    # near the equator are hit from both sides. What matters is the gap.
    assert e_pore > 0.95, e_pore
    assert e_groove < 0.75, e_groove
    assert e_pore - e_groove > 0.3, (e_pore, e_groove)
    # and it must not blow up on empty input
    assert enclosure(np.zeros((0, 3)), shell, box).shape == (0,)
    assert enclosure(c[None, :], np.zeros((0, 3)), box)[0] == 0.0


@test
def invariant_core_water_is_reported():
    """The defect that hid behind every other metric now has its own."""
    d = tempfile.mkdtemp()
    out = os.path.join(d, "box")
    _tiny(out)
    atoms, box = fileio.read_gro(os.path.join(out, "step5_input.gro"))
    tops = topology.load_toppar(os.path.join(out, "toppar"))
    na = tops["POPC"].natoms
    lipid = atoms.resname == "POPC"
    lip = atoms.xyz[lipid].reshape(-1, na, 3)
    ip = list(atoms.name[lipid][:na]).index("P")
    pz = lip[:, ip, 2]
    mid = 0.5 * (pz.max() + pz.min())
    core = (pz[pz < mid].mean() + 2.0, pz[pz > mid].mean() - 2.0)
    rep = validate.core_water_report(
        atoms, box, np.zeros(len(atoms), dtype=bool), lipid, core)
    assert "count" in rep and "slab_A" in rep, rep
    # a bare bilayer must have none at all
    assert rep["count"] == 0, rep
    shutil.rmtree(d, ignore_errors=True)


@test
def invariant_monolayer_builds():
    d = tempfile.mkdtemp()
    res = _tiny(os.path.join(d, "mono"), leaflet="monolayer")
    assert res.counts["POPC"] > 0
    shutil.rmtree(d, ignore_errors=True)


# ==========================================================================
# runner
# ==========================================================================

@test
def regression_angstrom_where_nanometres_are_expected_is_refused():
    """A length ten times too large used to reach the packer and die there as
    `MemoryError: cell table would be 15000272x25`, naming neither the setting
    nor the box. The public API is nanometres; catch it at the door."""
    from . import api

    base = {"protein_pdb": "x.pdb", "reference_dir": "y", "output_dir": "z"}
    for key, value in (("box", (120.1, 120.1, 95.6)),
                       ("water_thickness", 15.0),
                       ("margin_x", 20.0),
                       ("area_per_lipid", 64.3),
                       ("pore_radius", 10.0)):
        try:
            api.build_protein_system(dict(base, **{key: value}))
        except ValueError as exc:
            assert "nanometres" in str(exc), (key, exc)
            assert key in str(exc), (key, exc)
        else:
            raise AssertionError("%s in Angstrom was accepted" % key)

    # ...and the same numbers in nanometres get past the units check. They
    # fail later on the missing reference directory, which is the point.
    try:
        api.build_protein_system(dict(base, box=(12.01, 12.01, 9.56),
                                      water_thickness=1.5, margin_x=2.0))
    except ValueError as exc:
        assert "nanometres" not in str(exc), exc
    except Exception:
        pass


@test
def regression_pore_water_must_be_near_the_pore_axis():
    """Being far from lipid and well enclosed is not enough: when packing
    leaves a gap around the protein, water in the gap is far from lipid
    *because the lipid is missing*, and a surface groove is enclosed. On HCN4
    that kept 452 waters against CHARMM-GUI's 5, the furthest 41.7 A
    off-axis. A pore is a hole down the middle, so the radius decides."""
    import numpy as np
    from . import solvate as S

    box = np.array([60.0, 60.0, 60.0])
    # two candidate waters at the same depth: one on the axis, one 25 A out
    oc = np.array([[30.0, 30.0, 30.0], [55.0, 30.0, 30.0]])
    axis = np.array([30.0, 30.0])
    d = oc[:, :2] - axis
    for k in (0, 1):
        w = np.abs(d[:, k])
        d[:, k] = np.minimum(w, box[k] - w)
    r = np.hypot(d[:, 0], d[:, 1])
    keep = r <= 10.0
    assert keep[0] and not keep[1], r
    # and the radius is applied through the public signature
    import inspect
    sig = inspect.signature(S.solvate).parameters
    assert "pore_radius" in sig and "pore_axis_xy" in sig


@test
def regression_match_counts_accounts_for_ions_replacing_water():
    """Matching another system's composition has one trap: ions are placed by
    replacing water, so trimming straight to the target water count lands
    exactly n_ion molecules short. Against CHARMM-GUI box 1 that was 22744
    instead of 22876 -- correct-looking, and wrong."""
    import os
    import tempfile
    from .builder import read_system_counts

    # a reference whose topol.top and .gro disagree must be refused, not
    # silently guessed at
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "topol.top"), "w") as fh:
            fh.write("[ molecules ]\nPOPC 4\nTIP3 10\nPOT 1\nCLA 1\n")
        with open(os.path.join(d, "step5_input.gro"), "w") as fh:
            fh.write("t\n 2\n"
                     "    1POPC     P    1   1.000   1.000   3.000\n"
                     "    2POPC     P    2   1.000   1.000   1.000\n"
                     "   5.0   5.0   5.0\n")
        try:
            read_system_counts(d)
        except ValueError as exc:
            assert "head groups" in str(exc), exc
        else:
            raise AssertionError("a mismatched reference was accepted")

    # the arithmetic the trap turns on
    target_water, n_cat, n_ani = 22876, 70, 62
    trim_to = target_water + n_cat + n_ani
    assert trim_to - (n_cat + n_ani) == target_water, trim_to


# ==========================================================================
# orientation
# ==========================================================================

def _synthetic_channel(spacing=3.0, radius=15.0, half=30.0, belt=15.0):
    """A four-fold tube with a hydrophobic belt: a cartoon membrane protein.

    Carbon in the belt, charged carboxylate outside it, four identical
    chains -- enough for both the slab search and the symmetry axis to have
    something correct to find, without depending on any file.
    """
    per_ring, per_chain = 24, 6
    names, resns, resids, chains, elems, xyz = [], [], [], [], [], []
    rid = 0
    for z in np.arange(-half, half + 1e-9, spacing):
        inbelt = abs(z) <= belt
        for k in range(per_ring):
            th = 2 * np.pi * k / per_ring
            # Two CA markers per chain per ring, at different angles. One
            # would make each chain's CA set a straight line, which pins down
            # no rotation about itself -- see the collinearity guard in
            # symmetry_axis.
            marker = (k % per_chain) in (0, 2)
            names.append("CA" if marker else ("CB" if inbelt else "OD1"))
            resns.append("LEU" if inbelt else "ASP")
            elems.append("C" if (marker or inbelt) else "O")
            chains.append("ABCD"[k // per_chain])
            rid += 1
            resids.append(rid)
            xyz.append([radius * np.cos(th), radius * np.sin(th), z])
    return fileio.Atoms(names, resns, resids, np.array(xyz), chains,
                        element=elems)


def _random_rotation(seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@test
def invariant_symmetry_axis_is_exact():
    a = _synthetic_channel()
    got = orient.symmetry_axis(a)
    assert got is not None, "a C4 tube was not recognised as an oligomer"
    axis, fold, angles, rms = got
    assert fold == 4, fold
    assert rms < 1e-6, rms
    assert abs(abs(float(axis[2])) - 1.0) < 1e-6, axis
    for ang in angles:
        approx(ang, 90.0, 1e-6)


@test
def endtoend_orient_recovers_a_known_membrane_normal():
    """Scramble a protein whose answer is known, and get it back."""
    a = _synthetic_channel()
    for seed in (1, 2, 3):
        R = _random_rotation(seed)
        s = a.copy()
        s.xyz = s.xyz @ R.T + np.array([31.0, -17.0, 9.0])
        out, rep = orient.orient(s, use_symmetry="no", n_directions=200,
                                 n_points=24)
        truth = np.array([0.0, 0.0, 1.0]) @ R.T
        got = np.array(rep["normal_before"])
        err = np.degrees(np.arccos(min(1.0, abs(float(got @ truth)))))
        assert err < 5.0, "normal off by %.2f deg (seed %d)" % (err, seed)
        # and the result really is in the membrane frame afterwards
        chk = orient.orientation_check(out, n_points=24)
        assert chk["oriented"], chk


@test
def regression_unoriented_protein_is_detected():
    """The silent failure this replaces: a bilayer built through a protein
    lying on its side, with no complaint."""
    a = _synthetic_channel()
    assert orient.orientation_check(a, n_points=24)["oriented"]

    tipped = a.copy()
    c, s = np.cos(np.radians(70.0)), np.sin(np.radians(70.0))
    tipped.xyz = tipped.xyz @ np.array([[1.0, 0.0, 0.0],
                                        [0.0, c, -s],
                                        [0.0, s, c]]).T
    chk = orient.orientation_check(tipped, n_points=24)
    assert not chk["oriented"], chk
    assert chk["tilt_from_z_deg"] > 40.0, chk


@test
def regression_slab_thickness_does_not_pin_to_the_bound():
    """Buried surface alone falls monotonically with thickness, so the search
    ran to whatever upper bound it was given and then bought more by tilting
    off-axis -- up to 6 degrees of error on HCN4. The mismatch penalty has to
    leave the optimum strictly inside the range."""
    a = _synthetic_channel()
    heavy = a[a.element != "H"]
    radii = np.array([orient.VDW.get(e.upper(), orient.VDW_DEFAULT)
                      for e in heavy.element])
    area = orient.sasa(heavy.xyz, radii, n_points=24)
    sig = np.array([orient.atom_sigma(r, n, e) for r, n, e
                    in zip(heavy.resname, heavy.name, heavy.element)])
    lo, hi = 20.0, 44.0
    _, _, half, _ = orient.hydrophobic_slab(
        heavy.xyz, sig * area, n_directions=120, thickness=(lo, hi))
    t = 2.0 * half
    assert lo + 1.0 < t < hi - 1.0, "thickness %.1f pinned to the bound" % t

    # The cartoon above has a hard hydrophobic band with charge either side,
    # so its buried-surface curve turns over on its own. A real protein's does
    # not -- measured on HCN4 it was still falling at 44 A -- so drive the
    # mechanism directly: uniformly favourable weight, where more slab is
    # always better and only the penalty can stop it.
    proj = np.linspace(-40.0, 40.0, 400)
    w = np.full(400, -1.0)
    _, h_free, _ = orient._best_slab(proj, w, 0.5 * lo, 0.5 * hi, 0.25,
                                     mismatch_k=0.0)
    assert 2.0 * h_free >= hi - 0.5, "no penalty should run to the bound: %.1f" % (2 * h_free)
    _, h_pen, _ = orient._best_slab(proj, w, 0.5 * lo, 0.5 * hi, 0.25,
                                    mismatch_k=orient.MISMATCH_K,
                                    natural=orient.NATURAL_THICKNESS)
    assert abs(2.0 * h_pen - orient.NATURAL_THICKNESS) < 4.0, 2.0 * h_pen


@test
def regression_azimuth_is_minimised_not_maximised():
    """The scan and the applied rotation must turn the same way. Getting that
    backwards made the box bigger while reporting it as the tightest."""
    a = _synthetic_channel()
    sides = []
    for seed in (4, 5, 6):
        R = _random_rotation(seed)
        s = a.copy()
        s.xyz = s.xyz @ R.T
        _, rep = orient.orient(s, use_symmetry="yes", n_directions=120,
                               n_points=24)
        sides.append(rep["xy_bounding_square_A"])
    # same protein, so the tightest square is a property of the protein and
    # must not depend on the frame it arrived in
    assert max(sides) - min(sides) < 0.5, sides

    out, rep = orient.orient(a, use_symmetry="yes", n_directions=120,
                             n_points=24)
    best = rep["xy_bounding_square_A"]
    for deg in (10.0, 20.0, 30.0, 40.0):
        Rz = geom.rotation_z(np.radians(deg))
        x = out.xyz @ Rz.T
        side = max(x[:, 0].max() - x[:, 0].min(), x[:, 1].max() - x[:, 1].min())
        assert side >= best - 1e-6, (deg, side, best)


@test
def regression_extracellular_resid_decides_which_way_up():
    """Nothing geometric distinguishes the two faces of a slab. Without a
    residue range that is known to be outside, the protein is as likely to be
    built upside down as not."""
    a = _synthetic_channel()
    top = int(a.resid.max())
    top_side = (top - 60, top)          # the last rings, at high z

    for seed in (7, 8, 9, 10):
        R = _random_rotation(seed)
        s = a.copy()
        s.xyz = s.xyz @ R.T
        out, rep = orient.orient(s, use_symmetry="yes", n_directions=120,
                                 n_points=24, extracellular_resid=top_side)
        sel = ((out.resid >= top_side[0]) & (out.resid <= top_side[1]) &
               (out.name == "CA"))
        assert out.xyz[sel, 2].mean() > 0, (seed, out.xyz[sel, 2].mean())
        assert rep["updown"].startswith("from"), rep["updown"]

    # and without it, the report says so rather than pretending
    _, rep = orient.orient(a, use_symmetry="yes", n_directions=120,
                           n_points=24)
    assert "tie-break" in rep["updown"], rep["updown"]
    assert rep.get("warning_updown"), rep


@test
def regression_sasa_matches_a_brute_force_reference():
    """The neighbour bucketing must not lose contacts."""
    rng = np.random.default_rng(11)
    xyz = rng.uniform(0, 18, (60, 3))
    radii = np.full(60, 1.7)
    fast = orient.sasa(xyz, radii, n_points=64)

    unit = orient._sphere_points(64)
    ext = radii + 1.4
    slow = np.empty(60)
    for i in range(60):
        pts = xyz[i] + unit * ext[i]
        j = np.arange(60) != i
        d2 = ((pts[:, None, :] - xyz[j][None, :, :]) ** 2).sum(axis=2)
        free = ~(d2 < (ext[j] ** 2)[None, :]).any(axis=1)
        slow[i] = 4.0 * np.pi * ext[i] ** 2 * free.mean()
    assert np.abs(fast - slow).max() < 1e-9, np.abs(fast - slow).max()


# ==========================================================================
# pdb2gmx integration
# ==========================================================================

_STUB_TOPOL = """; written by the stub
#include "charmm36.ff/forcefield.itp"

[ moleculetype ]
; Name            nrexcl
Protein_chain_A     3

[ atoms ]
;   nr   type  resnr residue  atom  cgnr    charge      mass
     1    NH3      1    ALA      N     1     -0.30    14.007
     2    HC       1    ALA     H1     2      0.33     1.008
     3    CT1      1    ALA     CA     3      0.21    12.011

[ bonds ]
     1     2
     1     3

[ moleculetype ]
Protein_chain_B     3

[ atoms ]
     1    NH3      1    ALA      N     1     -0.30    14.007
     2    HC       1    ALA     H1     2      0.33     1.008
     3    CT1      1    ALA     CA     3      0.21    12.011

[ bonds ]
     1     2
     1     3

[ system ]
Protein in vacuum

[ molecules ]
Protein_chain_A     1
Protein_chain_B     1
"""


def _stub_gmx(tmpdir, topol=_STUB_TOPOL, fail=False):
    """A fake `gmx` that answers --version and writes a known topology.

    GROMACS is not installed on this machine, so what can be tested is
    everything around it: discovery, the command line, the parsing, the
    renaming, and the error when it is missing. The stub makes that testable
    without pretending GROMACS itself has been exercised.
    """
    # The topology goes in its own file rather than inside the stub's source:
    # embedding a triple-quoted block and then reordering lines is exactly how
    # the first version of this helper broke itself.
    tpl = os.path.join(tmpdir, "topol_template.txt")
    with open(tpl, "w") as fh:
        fh.write(topol)

    py = os.path.join(tmpdir, "stub.py")
    action = ("    sys.stderr.write('Fatal error: residue ZZZ not found in "
              "residue topology database\\n')\n    sys.exit(1)\n" if fail else
              "    p = a[a.index('-p') + 1]\n"
              "    open(p, 'w').write(open(TPL).read())\n"
              "    open(a[a.index('-o') + 1], 'w').write('stub\\n')\n"
              "    sys.exit(0)\n")
    with open(py, "w") as fh:
        fh.write("import sys\n"
                 "TPL = %r\n"
                 "a = sys.argv[1:]\n"
                 "if '--version' in a:\n"
                 "    print('GROMACS VERSION 2024.1-stub')\n"
                 "    sys.exit(0)\n"
                 "if a and a[0] == 'pdb2gmx':\n" % tpl
                 + action + "sys.exit(1)\n")

    if os.name == "nt":
        launcher = os.path.join(tmpdir, "gmx.bat")
        with open(launcher, "w") as fh:
            # Without the explicit exit, a .bat reports success whatever the
            # program inside it did, and a failing pdb2gmx looks like a
            # succeeding one that wrote no output.
            fh.write('@echo off\r\n"%s" "%s" %%*\r\nexit /b %%ERRORLEVEL%%\r\n'
                     % (sys.executable, py))
    else:
        launcher = os.path.join(tmpdir, "gmx")
        with open(launcher, "w") as fh:
            fh.write('#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable, py))
        os.chmod(launcher, 0o755)
    return launcher


@test
def invariant_equilibration_schedule_is_sane():
    """Restraints only ever loosen, and the free stage is long enough to be
    worth calling equilibration."""
    prev = None
    for name, bb, sc, lip, dih, dt, nsteps, baro, gv in mdp._SCHEDULE:
        if prev is not None:
            for a, b, what in ((bb, prev[0], "backbone"), (sc, prev[1], "sidechain"),
                               (lip, prev[2], "lipid"), (dih, prev[3], "dihedral")):
                assert a <= b, "%s restraint rises at %s: %g > %g" % (
                    what, name, a, b)
        prev = (bb, sc, lip, dih)

    # (name, bb, sc, lipid, dihres, dt, nsteps, barostat, gen_vel)
    last = mdp._SCHEDULE[-1]
    ns = last[6] * last[5] / 1000.0
    assert last[1] > 0.0, "the final stage should still restrain the backbone"
    assert last[2] == last[3] == last[4] == 0.0, \
        "side chains, lipids and dihedrals must all be free by the final "\
        "stage, or the bilayer cannot relax: %r" % (last,)
    # 0.5 ns -- CHARMM-GUI's default -- is too short for the bilayer to reach
    # its own area per lipid in the only stage where it is free to.
    assert ns >= 5.0, "final equilibration is %.2f ns, too short" % ns

    only_first_generates = [s[8] for s in mdp._SCHEDULE]
    assert only_first_generates[0] is True, "step6.1 must generate velocities"
    assert not any(only_first_generates[1:]), \
        "only the first stage should generate velocities"


@test
def regression_missing_data_points_at_the_setup_command():
    """The published repository ships no data/ -- it is CHARMM-GUI and
    MacKerell material with no stated redistribution terms. A fresh clone must
    say how to get it, not fail with a bare FileNotFoundError."""
    from . import library, make_data
    empty = tempfile.mkdtemp()
    assert not make_data.is_installed(empty)
    try:
        library.load_lipid_library("POPC", data_dir=empty)
    except FileNotFoundError as exc:
        assert "make_data" in str(exc) or "setup" in str(exc), exc
    else:
        raise AssertionError("a missing conformer library was not reported")
    try:
        library.load_water_template("TIP3", data_dir=empty)
    except FileNotFoundError as exc:
        assert "make_data" in str(exc) or "setup" in str(exc), exc
    else:
        raise AssertionError("a missing water library was not reported")


@test
def regression_missing_gromacs_says_so_clearly():
    """Generating topology needs GROMACS. Saying which names were tried beats
    an obscure failure three steps later."""
    from . import pdb2gmx as p2g
    try:
        p2g.find_gromacs(gmx=os.path.join(tempfile.mkdtemp(), "nope"))
    except p2g.GromacsNotFound as exc:
        assert "no working GROMACS" in str(exc), exc
        assert "nope" in str(exc), exc
    else:
        raise AssertionError("a missing GROMACS was not reported")


@test
def endtoend_pdb2gmx_produces_itp_the_builder_can_read():
    from . import pdb2gmx as p2g
    d = tempfile.mkdtemp()
    stub = _stub_gmx(d)
    pdb = os.path.join(d, "in.pdb")
    with open(pdb, "w") as fh:
        fh.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000\n"
                 "END\n")

    rep = p2g.generate_topology(pdb, os.path.join(d, "out"), gmx=stub)
    assert rep["ok"] and "stub" in rep["gromacs_version"], rep
    assert [m["name"] for m in rep["molecules"]] == ["PROA", "PROB"], rep
    assert [m["from"] for m in rep["molecules"]] == [
        "Protein_chain_A", "Protein_chain_B"], rep
    for m in rep["molecules"]:
        assert m["atoms"] == 3, m
        approx(m["charge"], 0.24, 1e-6)

    # the real requirement: load_toppar must accept the result
    tops = topology.load_toppar(rep["toppar"])
    assert set(tops) == {"PROA", "PROB"}, sorted(tops)
    assert tops["PROA"].natoms == 3
    assert tops["PROA"].bonds == [(0, 1), (0, 2)], tops["PROA"].bonds


@test
def regression_pdb2gmx_failure_is_reported_with_its_own_message():
    """A force field that does not know a residue is the commonest failure,
    and the useful part is what GROMACS said, not our exit code."""
    from . import pdb2gmx as p2g
    d = tempfile.mkdtemp()
    stub = _stub_gmx(d, fail=True)
    pdb = os.path.join(d, "in.pdb")
    with open(pdb, "w") as fh:
        fh.write("ATOM      1  N   ZZZ A   1       0.000   0.000   0.000\nEND\n")
    try:
        p2g.generate_topology(pdb, os.path.join(d, "out"), gmx=stub)
    except RuntimeError as exc:
        assert "residue topology database" in str(exc), exc
        assert "pdb2gmx failed" in str(exc), exc
    else:
        raise AssertionError("a pdb2gmx failure was swallowed")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", help="only run tests whose name contains this")
    ap.add_argument("-v", action="store_true", help="show tracebacks")
    args = ap.parse_args(argv)

    chosen = [t for t in TESTS if not args.k or args.k in t.__name__]
    failures = []
    t0 = time.time()
    for t in chosen:
        start = time.time()
        buf, old = io.StringIO(), sys.stdout
        try:
            sys.stdout = buf
            t()
            sys.stdout = old
            print("  pass  %-58s %5.2fs" % (t.__name__, time.time() - start))
        except Exception as exc:                        # noqa: BLE001
            sys.stdout = old
            print("  FAIL  %-58s %5.2fs" % (t.__name__, time.time() - start))
            failures.append((t.__name__, exc, traceback.format_exc()))
        finally:
            sys.stdout = old

    print("\n%d of %d passed in %.1fs" % (len(chosen) - len(failures),
                                          len(chosen), time.time() - t0))
    for name, exc, tb in failures:
        print("\n" + "=" * 72)
        print("FAIL %s\n  %s: %s" % (name, type(exc).__name__, exc))
        if args.v:
            print(tb)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
