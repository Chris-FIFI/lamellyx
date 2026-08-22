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


class SkipTest(Exception):
    """Raise from a test body to skip it rather than fail -- for a check that
    needs an optional local resource, like the non-redistributable base CGenFF
    force field, which is present on a developer's machine but not in CI."""


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
def regression_written_files_use_lf_newlines():
    """Text-mode "w" on Windows turns \\n into CRLF, so lamellyx would emit CRLF
    GROMACS files there -- byte-different from a Linux run and from the LF
    reference converter, and a spurious every-line diff for anyone comparing the
    two. Every writer passes newline="\\n"; this guards it, on the platform where
    it actually manifests (a no-op assertion on Unix, which is LF regardless)."""
    d = tempfile.mkdtemp()
    n = 3
    a = fileio.Atoms(["OH2"] * n, ["TIP3"] * n, np.arange(1, n + 1),
                     np.zeros((n, 3)))
    box = np.array([10.0, 10.0, 10.0])
    gro = os.path.join(d, "t.gro"); fileio.write_gro(gro, a, box)
    pdb = os.path.join(d, "t.pdb"); fileio.write_pdb(pdb, a, box)
    top = os.path.join(d, "topol.top")
    topology.write_topol(top, ["toppar/TIP3.itp"], [("TIP3", n)])
    ndx = os.path.join(d, "index.ndx")
    topology.write_index(ndx, [("System", np.arange(n))])
    for p in (gro, pdb, top, ndx):
        raw = open(p, "rb").read()
        assert b"\r\n" not in raw, "%s has CRLF" % os.path.basename(p)
        assert b"\n" in raw, "%s has no newline at all" % os.path.basename(p)


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

        # --- ligand tab: a CGenFF .str -> topology, no licence ----------
        meoh = "\n".join([
            "* methanol", "*", "read rtf card append", "* top", "*", "36 1",
            "MASS -1 CG331 12.01100", "MASS -1 HGA3 1.00800",
            "MASS -1 OG311 15.99940", "MASS -1 HGP1 1.00800",
            "RESI MEOH 0.000", "GROUP",
            "ATOM C1 CG331 -0.040", "ATOM H1 HGA3 0.090", "ATOM H2 HGA3 0.090",
            "ATOM H3 HGA3 0.090", "ATOM O1 OG311 -0.650", "ATOM HO1 HGP1 0.420",
            "BOND C1 H1", "BOND C1 H2", "BOND C1 H3", "BOND C1 O1", "BOND O1 HO1",
            "END", "read param card flex append", "* par", "*",
            "BONDS", "CG331 HGA3 322.00 1.1110", "CG331 OG311 428.00 1.4200",
            "OG311 HGP1 545.00 0.9600",
            "ANGLES", "HGA3 CG331 HGA3 35.50 108.40",
            "HGA3 CG331 OG311 45.90 108.89", "CG331 OG311 HGP1 50.00 106.00",
            "DIHEDRALS", "HGA3 CG331 OG311 HGP1 0.1400 3 0.00",
            "NONBONDED", "CG331 0.0 -0.0780 2.0500", "HGA3 0.0 -0.0240 1.3400",
            "OG311 0.0 -0.1921 1.7650", "HGP1 0.0 -0.0460 0.2245", "END"])
        rep = post("/api/ligand", {"str_text": meoh})
        assert rep["resname"] == "MEOH" and rep["n_atoms"] == 6, rep
        assert abs(rep["net_charge"]) < 1e-9, rep
        assert rep["files"] == ["MEOH.itp", "MEOH_atomtypes.itp"], rep
        # the .itp text rides along so the page can show it inline
        assert "[ moleculetype ]" in rep.get("itp_text", ""), "no inline .itp"
        itp = raw("/api/file/ligands/MEOH.itp", timeout=10).read()
        assert b"[ moleculetype ]" in itp and b"[ atoms ]" in itp, itp[:80]
        # an empty stream is a clean 400, not a 500
        assert status("/api/ligand", data=b'{"str_text": ""}') == 400
        # a malformed JSON body is a clean 400, not a 500
        assert status("/api/ligand", data=b"{not valid json") == 400
        # a real (non-self-contained) stream with no toppar is refused with 400,
        # the converter's own supplement diagnostic, not a 500
        assert status("/api/ligand",
                      data=_json.dumps({"str_text": _JZ4_STR}).encode()) == 400
        # a body that is valid JSON but not an object, or a wrong-typed field,
        # is a clean 400 -- not a 500 from a .get()/.strip() on the wrong type
        for bad in (b"[]", b"123", b"null", b'{"str_text": 12345}',
                    b'{"resname": ["x"], "str_text": "y"}'):
            assert status("/api/ligand", data=bad) == 400, bad
        assert status("/api/delete", data=b"{}") == 400        # missing 'id'
        assert status("/api/delete", data=b"[]") == 400        # not an object
        assert status("/api/check", data=b"123") == 400        # not an object

        # the ligand tab remembers the last force-field directory used, so the
        # long toppar path need not be re-typed. Build with a (minimal) FF, then
        # /api/state should report it for the page to pre-fill.
        ffdir = os.path.join(ws, "ff")
        os.makedirs(ffdir, exist_ok=True)
        with open(os.path.join(ffdir, "top_all36_cgenff.rtf"), "w") as fh:
            fh.write("MASS -1 CG331 12.011 C\n")
        with open(os.path.join(ffdir, "par_all36_cgenff.prm"), "w") as fh:
            fh.write("BONDS\n")
        assert get("/api/state").get("last_cgenff_ff", "") == "", "ff already set"
        post("/api/ligand", {"str_text": meoh, "cgenff_ff": ffdir})
        assert get("/api/state")["last_cgenff_ff"] == ffdir, "ff not remembered"

        assert get("/api/state")["history"], "build did not reach the history"
        post("/api/delete", {"id": jid})
        assert not os.path.isdir(os.path.join(ws, jid))
    finally:
        srv.shutdown()
        shutil.rmtree(ws, ignore_errors=True)


@test
def regression_dashboard_does_not_shadow_window_history():
    """The page defined `function history()` for the build list, which shadowed
    window.history; the token-scrub call `history.replaceState(...)` then threw
    on load and every button handler bound after it silently failed -- the whole
    UI was dead. Guard the name so the collision cannot come back, and keep the
    ligand tab wired to its endpoint."""
    from . import dashboard
    assert "function history(" not in dashboard.PAGE, \
        "the page shadows window.history again -- rename the history() function"
    assert "history.replaceState" in dashboard.PAGE      # still scrubs the token
    for needle in ('id="tab-ligand"', "/api/ligand", "Make topology",
                   "showLigand"):
        assert needle in dashboard.PAGE, needle


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
    # check_system validates a real bilayer build too (MEMB/SOLV, no SOLU)
    cs = api.check_system({"system_dir": os.path.join(tmp, "b")})
    assert cs["ok"] and cs["index_checked"] == "MEMB+SOLV", cs
    assert cs["atoms"] == r["counts"]["TOTAL_ATOMS"], (cs["atoms"], r["counts"])
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
def regression_check_refuses_a_lipid_it_cannot_build():
    """`check()` exists so a configuration can be rejected before it costs a
    minute. A mixture used to pass it and then raise NotImplementedError deep
    inside build(), which is the same shape as any other parameter the API
    accepts and cannot honour."""
    from .api import check

    r = check({"x": 6.0, "y": 6.0, "composition": {"POPC": 0.7, "POPE": 0.3}})
    assert not r["ok"], r
    assert any("mixed bilayers" in e for e in r["errors"]), r["errors"]

    r = check({"x": 6.0, "y": 6.0, "lipid": "POPE"})
    assert not r["ok"], r
    assert any("conformer library" in e for e in r["errors"]), r["errors"]
    # the message must say how to fix it, not just that it is wrong
    assert any("make_data" in e for e in r["errors"]), r["errors"]

    # and the configuration that does work is still accepted
    assert check({"x": 6.0, "y": 6.0})["ok"]


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


# ==========================================================================
# cgenff -- ligand topology from a CGenFF stream file
# ==========================================================================

# A hand-written ethanol .str, faithful to ParamChem's format: a `read rtf`
# half naming the atoms, charges and bonds, and a `read param` half with the
# bonded and Lennard-Jones parameters, penalties in the comments. The values
# are chosen so the GROMACS numbers can be worked out by hand in the tests.
_ETOH_STR = """\
* Toppar stream file for ethanol -- lamellyx test fixture
*

read rtf card append
* Topologies
*
36 1

MASS -1 CG331   12.01100 ! aliphatic C for CH3
MASS -1 CG321   12.01100 ! aliphatic C for CH2
MASS -1 OG311   15.99940 ! hydroxyl O
MASS -1 HGA3     1.00800 ! aliphatic H, CH3
MASS -1 HGA2     1.00800 ! aliphatic H, CH2
MASS -1 HGP1     1.00800 ! polar H

RESI ETOH   0.000
GROUP
ATOM C1  CG331  -0.270 !   0.000
ATOM H11 HGA3    0.090 !   0.000
ATOM H12 HGA3    0.090 !   0.000
ATOM H13 HGA3    0.090 !   0.000
ATOM C2  CG321   0.050 !   0.000
ATOM H21 HGA2    0.090 !   0.000
ATOM H22 HGA2    0.090 !   0.000
ATOM O1  OG311  -0.650 !   0.000
ATOM HO1 HGP1    0.420 !   0.000
BOND C1 C2
BOND C1 H11
BOND C1 H12
BOND C1 H13
BOND C2 O1
BOND C2 H21
BOND C2 H22
BOND O1 HO1

END

read param card flex append
* Parameters
*

BONDS
CG321 CG331   222.50   1.5280 ! ETOH, penalty= 0.6
CG321 OG311   428.00   1.4200 ! ETOH, penalty= 1.0
CG331 HGA3    322.00   1.1110 ! ETOH, penalty= 0.0
CG321 HGA2    309.00   1.1110 ! ETOH, penalty= 0.0
OG311 HGP1    545.00   0.9600 ! ETOH, penalty= 0.0

ANGLES
CG331 CG321 OG311   75.70   110.10 ! ETOH, penalty= 2.0
CG321 CG331 HGA3    34.60   110.10  22.53  2.1790 ! ETOH, penalty= 0.0
HGA3  CG331 HGA3    35.50   108.40   5.40  1.8020 ! ETOH, penalty= 0.0
CG331 CG321 HGA2    34.60   110.10  22.53  2.1790 ! ETOH, penalty= 0.0
OG311 CG321 HGA2    45.90   108.89 ! ETOH, penalty= 0.0
HGA2  CG321 HGA2    35.50   109.00   5.40  1.8020 ! ETOH, penalty= 0.0
CG321 OG311 HGP1    50.00   106.00 ! ETOH, penalty= 0.0

DIHEDRALS
OG311 CG321 CG331 HGA3   0.1600  3   0.00 ! ETOH, penalty= 0.0
HGA2  CG321 CG331 HGA3   0.1600  3   0.00 ! ETOH, penalty= 0.0
CG331 CG321 OG311 HGP1   0.1400  3   0.00 ! ETOH, penalty= 0.0
HGA2  CG321 OG311 HGP1   0.1400  3   0.00 ! ETOH, penalty= 0.0
X     CG321 CG331 X      0.1600  3   0.00 ! wildcard, penalty= 0.0
X     CG321 OG311 X      0.1400  3   0.00 ! wildcard, penalty= 0.0

IMPROPERS

NONBONDED nbxmod 5 atom cdiel
CG331   0.0  -0.0780   2.0400   0.0  -0.0100   1.9000 ! penalty 0
CG321   0.0  -0.0560   2.0100   0.0  -0.0100   1.9000
OG311   0.0  -0.1921   1.7650
HGA3    0.0  -0.0240   1.3400
HGA2    0.0  -0.0280   1.3400
HGP1    0.0  -0.0460   0.2245

END
RETURN
"""


@test
def reference_cgenff_unit_conversions():
    """The conversions CHARMM->GROMACS, checked against numbers worked out by
    hand. A wrong factor here is the "silently wrong energies" failure the whole
    package is arranged to avoid, so every factor gets a known answer."""
    from . import cgenff as c

    # bond: b0 A -> nm; kb doubles, kcal->kJ, /A^2 -> /nm^2  (x 836.8)
    b0, kb = c.bond_to_gmx(222.50, 1.5280)
    approx(b0, 0.15280, 1e-9)
    approx(kb, 222.50 * 836.8, 1e-4)          # 186188.0

    # angle with Urey-Bradley: force constants double, kcal->kJ; UB length A->nm
    th0, kth, ub0, kub = c.angle_to_gmx(34.60, 110.10, 22.53, 2.1790)
    approx(th0, 110.10, 1e-9)
    approx(kth, 34.60 * 8.368, 1e-4)          # 289.5328
    approx(ub0, 0.21790, 1e-9)
    approx(kub, 22.53 * 836.8, 1e-3)          # 18853.104

    # a plain angle carries no UB term
    _, _, ub0b, kubb = c.angle_to_gmx(75.70, 110.10, 0.0, 0.0)
    approx(ub0b, 0.0, 1e-12)
    approx(kubb, 0.0, 1e-12)

    # proper dihedral: NO factor of two, only kcal->kJ
    phi0, kphi = c.dihedral_to_gmx(0.1600, 0.00)
    approx(phi0, 0.0, 1e-12)
    approx(kphi, 0.1600 * 4.184, 1e-9)        # 0.66944

    # improper: harmonic, so the force constant doubles
    xi0, kxi = c.improper_to_gmx(120.0, 0.0)
    approx(xi0, 0.0, 1e-12)
    approx(kxi, 120.0 * 8.368, 1e-6)          # 1004.16

    # LJ: Rmin/2 (A) -> sigma (nm); |eps| kcal -> kJ
    sigma, eps = c.lj_to_gmx(0.0780, 2.0400)
    approx(sigma, 2.0400 * (0.2 / 2.0 ** (1.0 / 6.0)), 1e-9)
    approx(eps, 0.0780 * 4.184, 1e-9)


@test
def reference_cgenff_parses_charmm_bond_record_forms():
    """CHARMM RTF bonds come in forms the simple fixtures do not have: several
    pairs on one BOND line, and the DOUBLE / TRIPLE keywords. Each declares the
    same kind of bond; IC and other records are ignored, not misread."""
    from . import cgenff as c
    rtf = "\n".join([
        "read rtf card append", "* t", "*", "36 1",
        "RESI X 0.000", "GROUP",
        "ATOM A CG331 0.0", "ATOM B CG331 0.0", "ATOM D CG331 0.0",
        "ATOM E CG331 0.0",
        "BOND A B  B D",                      # two pairs on one line
        "DOUBLE D E",                         # a double bond is still one bond
        "IC A B D E 1.5 110 180 110 1.5",     # ignored
        "PATCHING FIRS NONE LAST NONE",       # ignored
        "END"])
    s = c.parse_stream(rtf)
    assert len(s.atoms) == 4, s.atoms
    assert s.bonds == [("A", "B"), ("B", "D"), ("D", "E")], s.bonds


@test
def invariant_cgenff_type_names_infer_the_right_element():
    """A real ParamChem stream has no MASS records, so an atom's mass -- and the
    atomic number that reaches [ atomtypes ] -- is read off its CGenFF type name.
    A wrong element is a silent mass error, so check the inference across the
    periodic subset CGenFF uses, including the two-letter halogens that must beat
    the one-letter match (CLGA1 is chlorine, not carbon)."""
    from . import cgenff as c
    expect = {
        "CG331": ("C", 12.011), "HGA3": ("H", 1.008), "OG311": ("O", 15.999),
        "NG2R61": ("N", 14.007), "SG311": ("S", 32.06), "PG1": ("P", 30.974),
        "FGA1": ("F", 18.998), "CLGA1": ("CL", 35.45), "BRGA1": ("BR", 79.904),
        "IGR1": ("I", 126.904),
    }
    for t, (elem, mass) in expect.items():
        assert c._element_from_type(t) == elem, (t, c._element_from_type(t))
        approx(c._mass_from_type(t), mass, 0.01)
    # the atomic numbers that reach [ atomtypes ] follow from the mass
    for t, z in (("FGA1", 9), ("CLGA1", 17), ("BRGA1", 35), ("IGR1", 53)):
        assert c._atomic_number(c._mass_from_type(t)) == z, t


@test
def regression_cgenff_malformed_rtf_records_are_refused_cleanly():
    """A truncated MASS/RESI/ATOM record used to crash with a bare IndexError --
    and, pasted into the dashboard, a 500. A fuzz pass over thousands of
    perturbed streams turned it up. Each must be a clean ValueError now."""
    from . import cgenff as c
    for bad in ("ATOM", "ATOM C1", "ATOM C1 CG331",     # missing type / charge
                "MASS", "MASS -1 CG331", "RESI"):        # missing fields
        stream = "read rtf card append\n* t\n*\n36 1\n%s\nEND\n" % bad
        try:
            c.parse_stream(stream)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed record %r was accepted" % bad)


@test
def regression_cgenff_malformed_bond_and_improper_records_are_refused():
    """A BOND with an odd atom count, or an IMPR not a multiple of four, would
    drop an orphan silently -- a missing bond, or a planar centre gone
    non-planar. Both must be refused, not quietly truncated by the grouping."""
    from . import cgenff as c
    odd_bond = _ETOH_STR.replace("BOND C1 C2", "BOND C1 C2 O1")   # three atoms
    try:
        c.parse_stream(odd_bond)
    except ValueError as exc:
        assert "even number" in str(exc), exc
    else:
        raise AssertionError("an odd BOND record was accepted")

    short_impr = _FALD_STR.replace("IMPR C1 O1 H1 H2", "IMPR C1 O1 H1")
    try:
        c.parse_stream(short_impr)
    except ValueError as exc:
        assert "multiple of" in str(exc), exc
    else:
        raise AssertionError("a truncated IMPR record was accepted")


@test
def reference_cgenff_handles_a_degenerate_topology():
    """A ligand can be a single atom -- a bound ion -- with no bonds, angles or
    dihedrals. The graph enumeration and the writer must handle the empty bond
    list and still produce a valid, integer-charge topology, not choke on it."""
    from . import cgenff as c
    ion = ("read rtf card append\n* t\n*\n36 1\nRESI CLA -1.000\nGROUP\n"
           "ATOM CL CLA -1.000\nEND\nread param card flex append\n"
           "BONDS\nANGLES\nDIHEDRALS\nNONBONDED\nCLA 0.0 -0.150 2.27\nEND\n")
    r = c.generate_ligand_topology(ion, tempfile.mkdtemp())
    assert r["n_atoms"] == 1 and r["n_bonds"] == 0 and r["n_angles"] == 0, r
    assert r["n_dihedrals"] == 0 and r["n_lonepairs"] == 0, r
    approx(r["net_charge"], -1.0, 1e-9)
    assert r["net_charge_is_integer"], r
    mol = topology.parse_itp(r["itp"])["CLA"]
    assert mol.natoms == 1 and len(mol.bonds) == 0, (mol.natoms, mol.bonds)


@test
def reference_cgenff_parses_ethanol_stream():
    """Every field the rest of the module relies on comes off the .str intact:
    atoms and charges, the bond list, the masses, the parameter tables, and the
    penalties from the comments."""
    from . import cgenff as c
    s = c.parse_stream(_ETOH_STR)

    assert s.resname == "ETOH", s.resname
    assert len(s.atoms) == 9, len(s.atoms)
    assert [a.name for a in s.atoms[:2]] == ["C1", "H11"]
    assert s.atoms[0].type == "CG331" and s.atoms[0].charge == -0.270
    approx(sum(a.charge for a in s.atoms), 0.0, 1e-9)   # neutral molecule
    assert len(s.bonds) == 8, s.bonds
    assert ("C2", "O1") in s.bonds

    assert s.masses["OG311"] == 15.99940
    assert s.bond_p[("CG321", "CG331")][:2] == (222.50, 1.5280)
    # penalty read out of the comment
    approx(s.bond_p[("CG321", "OG311")][2], 1.0, 1e-9)
    # angle with and without a Urey-Bradley term
    assert s.angle_p[("CG321", "CG331", "HGA3")][2:4] == (22.53, 2.1790)
    assert s.angle_p[("CG331", "CG321", "OG311")][2:4] == (0.0, 0.0)
    # dihedral stored as a list of terms (multiplicity series)
    assert isinstance(s.dihe_p[("OG311", "CG321", "CG331", "HGA3")], list)
    # nonbonded 1-4 columns kept separately from the primary LJ
    eps, rmin2, eps14, rmin2_14, _ = s.nb["CG331"]
    assert (eps, rmin2, eps14, rmin2_14) == (0.0780, 2.0400, 0.0100, 1.9000)


@test
def reference_cgenff_enumerates_angles_and_dihedrals():
    """Angles and dihedrals are not in the .str -- they follow from the bond
    graph, and CHARMM-GUI derives them the same way. Wrong enumeration is wrong
    physics, so the counts are checked against a graph small enough to count."""
    from . import cgenff as c
    s = c.parse_stream(_ETOH_STR)
    names = [a.name for a in s.atoms]

    angles = c.enumerate_angles(names, s.bonds)
    # C1 centre: C(4,2)=6, C2 centre: 6, O1 centre: 1  ->  13
    assert len(angles) == 13, len(angles)

    dihedrals = c.enumerate_dihedrals(names, s.bonds)
    # only the C1-C2 (3x3) and C2-O1 (3x1) bonds carry dihedrals -> 12
    assert len(dihedrals) == 12, len(dihedrals)
    # each dihedral counted once, not once per direction
    canon = {frozenset(((d[0], d[1]), (d[2], d[3]))) for d in dihedrals}
    assert len(canon) == len(dihedrals)

    pairs = c.one_four_pairs(names, s.bonds, dihedrals)
    bonded = {frozenset(b) for b in s.bonds}
    one_three = {frozenset((a, cc)) for a, _, cc in angles}
    for p in pairs:
        assert frozenset(p) not in bonded, p        # never a 1-2 pair
        assert frozenset(p) not in one_three, p      # never a 1-3 pair
    assert pairs, "ethanol has 1-4 pairs"


@test
def endtoend_cgenff_writes_itp_the_topology_reader_accepts():
    """The real requirement: what the converter writes must be something the
    existing topology reader -- the same one the builder uses -- can load, with
    the atoms, bonds and total charge intact."""
    from . import cgenff as c
    d = tempfile.mkdtemp()
    rep = c.generate_ligand_topology(_ETOH_STR, d)

    assert rep["ok"] and rep["resname"] == "ETOH", rep
    assert rep["n_atoms"] == 9 and rep["n_bonds"] == 8, rep
    assert rep["n_angles"] == 13 and rep["n_dihedrals"] == 12, rep
    approx(rep["net_charge"], 0.0, 1e-9)
    assert set(rep["files"]) == {"ETOH.itp", "ETOH_atomtypes.itp"}, rep["files"]
    for f in rep["files"]:
        assert os.path.exists(os.path.join(d, f)), f

    # load it back through the package's own parser
    tops = topology.parse_itp(os.path.join(d, "ETOH.itp"))
    assert set(tops) == {"ETOH"}, sorted(tops)
    mol = tops["ETOH"]
    assert mol.natoms == 9, mol.natoms
    approx(mol.total_charge, 0.0, 1e-6)
    assert len(mol.bonds) == 8, mol.bonds
    # the bond graph survived the round trip: C1 (atom 0) bonds to C2 and 3 H
    adj = mol.adjacency()
    assert len(adj[0]) == 4, adj[0]

    # atom types are declared before they are used, and every one is defined
    with open(os.path.join(d, "ETOH_atomtypes.itp")) as fh:
        atp = fh.read()
    for t in ("CG331", "CG321", "OG311", "HGA3", "HGA2", "HGP1"):
        assert t in atp, t
    assert "[ atomtypes ]" in atp and "[ pairtypes ]" in atp


@test
def regression_cgenff_penalty_is_reported_not_refused():
    """A high CGenFF penalty is REPORTED, not refused: most biological ligands
    lean on analogy, so refusing on penalty would reject the normal case. The
    build succeeds and the worst penalty and offenders come back in the report;
    penalty_flag optionally lists everything above a chosen number."""
    from . import cgenff as c
    d = tempfile.mkdtemp()
    bad = _ETOH_STR.replace("penalty= 1.0", "penalty= 120.5")

    rep = c.generate_ligand_topology(bad, d, penalty_flag=50.0)
    assert rep["ok"] and rep["worst_penalty"] == 120.5, rep
    # the offending bond is named in the reported penalties
    assert any("CG321-OG311" in lbl for lbl, _ in rep["penalties"]), rep["penalties"]
    # and flagged, because it is above the number we asked about
    flag = rep["flagged_above_50.0"]
    assert any(p == 120.5 for _, p in flag), flag


@test
def regression_cgenff_missing_parameter_is_refused_not_silently_dropped():
    """If the .str lacks a parameter the molecule needs, a partial topology
    would run with a wrong (missing) term. That must be refused and the gap
    named, not quietly skipped."""
    from . import cgenff as c
    d = tempfile.mkdtemp()
    # drop the C-O bond parameter the molecule cannot do without
    broken = _ETOH_STR.replace(
        "CG321 OG311   428.00   1.4200 ! ETOH, penalty= 1.0\n", "")
    try:
        c.generate_ligand_topology(broken, d)
    except ValueError as exc:
        assert "missing" in str(exc).lower(), exc
        assert "C2-O1" in str(exc) or "CG321-OG311" in str(exc), exc
    else:
        raise AssertionError("a missing parameter was silently dropped")


@test
def regression_cgenff_dihedral_wildcard_matches():
    """CGenFF gives many dihedrals only as wildcards (X b c X). Dropping the
    exact terms must still resolve through the wildcard, or good molecules fail
    to build."""
    from . import cgenff as c
    d = tempfile.mkdtemp()
    only_wild = (_ETOH_STR
                 .replace("OG311 CG321 CG331 HGA3   0.1600  3   0.00 "
                          "! ETOH, penalty= 0.0\n", "")
                 .replace("HGA2  CG321 CG331 HGA3   0.1600  3   0.00 "
                          "! ETOH, penalty= 0.0\n", ""))
    rep = c.generate_ligand_topology(only_wild, d)
    assert rep["ok"] and rep["n_dihedrals"] == 12, rep


# Formaldehyde: the smallest molecule with an improper (keeping the carbonyl
# planar) and with NO proper dihedrals -- so it exercises the improper path,
# which ethanol cannot, and the empty-dihedral edge case at the same time.
_FALD_STR = """\
* Toppar stream file for formaldehyde -- lamellyx test fixture
*

read rtf card append
* Topologies
*
36 1

MASS -1 CG2O1   12.01100 ! carbonyl C
MASS -1 OG2D1   15.99940 ! carbonyl O
MASS -1 HGR52    1.00800 ! aldehyde H

RESI FALD   0.000
GROUP
ATOM C1  CG2O1   0.420 !   0.000
ATOM O1  OG2D1  -0.510 !   0.000
ATOM H1  HGR52   0.045 !   0.000
ATOM H2  HGR52   0.045 !   0.000
BOND C1 O1
BOND C1 H1
BOND C1 H2
IMPR C1 O1 H1 H2

END

read param card flex append
* Parameters
*

BONDS
CG2O1 OG2D1   620.00   1.2200 ! FALD, penalty= 0.0
CG2O1 HGR52   330.00   1.1000 ! FALD, penalty= 0.0

ANGLES
OG2D1 CG2O1 HGR52   44.00   122.00 ! FALD, penalty= 0.0
HGR52 CG2O1 HGR52   40.00   116.00 ! FALD, penalty= 0.0

DIHEDRALS

IMPROPERS
CG2O1 OG2D1 HGR52 HGR52   14.00   0   0.00 ! FALD, penalty= 3.0

NONBONDED nbxmod 5 atom cdiel
CG2O1   0.0  -0.1050   1.9800
OG2D1   0.0  -0.1200   1.7000
HGR52   0.0  -0.0460   1.1000

END
RETURN
"""


@test
def reference_cgenff_improper_is_converted():
    """The improper path is separate code from the propers and, unlike them,
    has a factor of two. A molecule with an improper and no proper dihedrals
    isolates it. Kpsi 14 -> 14 * 2 * 4.184 = 117.152 kJ/mol/rad^2."""
    from . import cgenff as c
    s = c.parse_stream(_FALD_STR)
    assert s.impropers == [("C1", "O1", "H1", "H2")], s.impropers
    assert ("CG2O1", "OG2D1", "HGR52", "HGR52") in s.impr_p

    xi0, kxi = c.improper_to_gmx(*s.impr_p[
        ("CG2O1", "OG2D1", "HGR52", "HGR52")][:2])
    approx(xi0, 0.0, 1e-12)
    approx(kxi, 14.00 * 8.368, 1e-4)          # 117.152

    d = tempfile.mkdtemp()
    rep = c.generate_ligand_topology(_FALD_STR, d)
    assert rep["n_impropers"] == 1 and rep["n_dihedrals"] == 0, rep
    assert rep["n_angles"] == 3, rep         # O-C-H, O-C-H, H-C-H; no propers

    # the improper is written as a type-2 dihedral with the doubled constant
    with open(os.path.join(d, "FALD.itp")) as fh:
        text = fh.read()
    assert "; impropers (type 2)" in text, text
    # the converted constant appears on a funct-2 line
    hit = [ln for ln in text.splitlines()
           if ln.split()[4:5] == ["2"] and "1.171520e+02" in ln]
    assert hit, "no funct-2 improper line with kxi=117.152:\n" + text

    # and it still loads through the package parser (which ignores impropers)
    mol = topology.parse_itp(os.path.join(d, "FALD.itp"))["FALD"]
    assert mol.natoms == 4 and len(mol.bonds) == 3, (mol.natoms, mol.bonds)


@test
def regression_cgenff_improper_wildcard_matches():
    """CGenFF improper parameters are very often wildcarded on the outer atoms
    (`central X X X`). That must resolve, or a planar centre loses its improper
    and goes non-planar -- a silent geometry error, not a build failure."""
    from . import cgenff as c
    d = tempfile.mkdtemp()
    wild = _FALD_STR.replace(
        "CG2O1 OG2D1 HGR52 HGR52   14.00   0   0.00 ! FALD, penalty= 3.0",
        "CG2O1 X     X     X       14.00   0   0.00 ! FALD, penalty= 3.0")
    rep = c.generate_ligand_topology(wild, d)
    assert rep["ok"] and rep["n_impropers"] == 1, rep
    # the wildcarded improper's penalty still reaches the gate
    assert rep["worst_penalty"] == 3.0, rep


def _section_lines(itp_text, section):
    """The data lines under `[ section ]` in an .itp -- comments and blanks
    stripped, stopping at the next header. Repeated headers (an .itp has two
    [ dihedrals ]) are concatenated, which is what a reader wants."""
    want = "[ %s ]" % section
    out, take = [], False
    for raw in itp_text.splitlines():
        line = raw.split(";")[0].rstrip()
        if not line.strip():
            continue
        if line.strip().startswith("["):
            take = (line.strip() == want)
            continue
        if take:
            out.append(line)
    return out


# Chloromethane with a halogen sigma-hole lone pair. LPH is given no MASS and no
# NONBONDED record on purpose: type-name inference would make it carbon (12.011)
# and the LJ lookup would fail, so the fixture proves the lone-pair path forces
# both to zero. Net charge sums to 0.
_CLM_STR = """\
* Toppar stream file for chloromethane -- lamellyx test fixture (halogen LP)
*

read rtf card append
* Topologies
*
36 1

MASS -1 CG331   12.01100 ! aliphatic C for CH3
MASS -1 HGA3     1.00800 ! aliphatic H, CH3
MASS -1 CLGA1   35.45000 ! aliphatic chlorine

RESI CLM    0.000
GROUP
ATOM C1  CG331  -0.070 !   0.000
ATOM H1  HGA3    0.090 !   0.000
ATOM H2  HGA3    0.090 !   0.000
ATOM H3  HGA3    0.090 !   0.000
ATOM CL1 CLGA1  -0.290 !   0.000
ATOM LP1 LPH     0.090 !   0.000
BOND C1 H1
BOND C1 H2
BOND C1 H3
BOND C1 CL1
LONEPAIR COLINEAR LP1 CL1 C1 DIST 1.5000 SCALE 0.0

END

read param card flex append
* Parameters
*

BONDS
CG331 HGA3    322.00   1.1110 ! CLM, penalty= 0.0
CG331 CLGA1   222.00   1.7760 ! CLM, penalty= 5.0

ANGLES
HGA3 CG331 HGA3    35.50   108.40 ! CLM, penalty= 0.0
HGA3 CG331 CLGA1   34.00   108.00 ! CLM, penalty= 0.0

DIHEDRALS

IMPROPERS

NONBONDED nbxmod 5 atom cdiel
CG331   0.0  -0.0780   2.0500
HGA3    0.0  -0.0240   1.3400
CLGA1   0.0  -0.3430   1.9100

END
RETURN
"""


@test
def reference_cgenff_colinear_lone_pair_becomes_a_virtual_site():
    """A halogen sigma-hole lone pair is a COLINEAR virtual site. It must reach
    the .itp as a [ virtual_sites2 ] funct-2 term with a = -(dist in nm), and
    -- since a virtual site has no bond -- be handed the host atom's own
    1-2/1-3/1-4 exclusions explicitly, or GROMACS makes none for it."""
    from . import cgenff as c
    s = c.parse_stream(_CLM_STR)
    assert len(s.lonepairs) == 1, s.lonepairs
    lp = s.lonepairs[0]
    assert lp.kind.upper().startswith("COLI"), lp.kind
    assert lp.hosts[:2] == ["CL1", "C1"], lp.hosts
    approx(lp.dist, 1.5, 1e-9)

    d = tempfile.mkdtemp()
    rep = c.generate_ligand_topology(_CLM_STR, d)
    assert rep["ok"] and rep["n_lonepairs"] == 1, rep
    approx(rep["net_charge"], 0.0, 1e-9)

    with open(os.path.join(d, "CLM.itp")) as fh:
        itp = fh.read()

    # the virtual site: LP(6) from CL1(5) and C1(1), funct 2, a = -(1.5*0.1) nm
    vlines = _section_lines(itp, "virtual_sites2")
    assert len(vlines) == 1, vlines
    p = vlines[0].split()
    assert p[:4] == ["6", "5", "1", "2"], p
    approx(float(p[4]), -0.15, 1e-9)

    # explicit exclusions: LP(6) against host CL1(5) and all within three bonds
    elines = _section_lines(itp, "exclusions")
    assert len(elines) == 1, elines
    ex = [int(x) for x in elines[0].split()]
    assert ex[0] == 6 and set(ex[1:]) == {1, 2, 3, 4, 5}, ex

    # the LP is massless in [ atoms ], though type inference would give 12.011
    atom_lines = _section_lines(itp, "atoms")
    lp_row = next(ln for ln in atom_lines if ln.split()[4] == "LP1")
    approx(float(lp_row.split()[7]), 0.0, 1e-12)

    # and its atom type carries atomic number 0 and mass 0, LJ absent -> zero
    with open(os.path.join(d, "CLM_atomtypes.itp")) as fh:
        at = fh.read()
    lph = next(ln for ln in _section_lines(at, "atomtypes")
               if ln.split()[0] == "LPH")
    assert lph.split()[1] == "0", lph
    approx(float(lph.split()[2]), 0.0, 1e-12)

    # the whole file still loads through the package parser
    mol = topology.parse_itp(os.path.join(d, "CLM.itp"))["CLM"]
    assert mol.natoms == 6, mol.natoms


@test
def regression_cgenff_noncolinear_lone_pair_is_refused():
    """CGenFF only emits COLINEAR lone pairs; anything else needs a
    virtual_sites3 term this converter does not write. It must refuse loudly,
    not drop the site and return a silently-wrong topology."""
    from . import cgenff as c
    rel = _CLM_STR.replace(
        "LONEPAIR COLINEAR LP1 CL1 C1 DIST 1.5000 SCALE 0.0",
        "LONEPAIR RELATIVE LP1 CL1 C1 H1 DIST 1.5 ANGLE 90.0 DIHE 0.0")
    try:
        c.generate_ligand_topology(rel, tempfile.mkdtemp())
    except ValueError as exc:
        assert "COLINEAR" in str(exc), exc
    else:
        assert False, "a RELATIVE lone pair was not refused"


@test
def regression_cgenff_lone_pair_with_coincident_hosts_is_refused():
    """A COLINEAR lone pair whose two hosts are the same atom would emit a
    virtual_sites2 with a zero direction vector -- a division by zero in
    grompp/mdrun. Refuse it rather than write a broken topology."""
    from . import cgenff as c
    bad = _CLM_STR.replace(
        "LONEPAIR COLINEAR LP1 CL1 C1 DIST 1.5000 SCALE 0.0",
        "LONEPAIR COLINEAR LP1 CL1 CL1 DIST 1.5000 SCALE 0.0")
    try:
        c.generate_ligand_topology(bad, tempfile.mkdtemp())
    except ValueError as exc:
        assert "distinct" in str(exc).lower(), exc
    else:
        assert False, "a lone pair with coincident hosts was accepted"


@test
def regression_cgenff_duplicate_atom_name_is_refused():
    """Two atoms sharing a name silently merge in every name-keyed map -- the
    type table, the bond graph, coordinate matching -- so the parser refuses
    it rather than emit a quietly-wrong topology."""
    from . import cgenff as c
    dup = _ETOH_STR.replace("ATOM H12 HGA3", "ATOM H11 HGA3")
    try:
        c.parse_stream(dup)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower() and "H11" in str(exc), exc
    else:
        assert False, "a duplicate atom name was not refused"


@test
def regression_cgenff_bond_to_undeclared_atom_is_refused():
    """A BOND (or IMPR/LONEPAIR) naming an atom with no ATOM record used to
    surface as a bare KeyError deep in parameter resolution -- the bond loop
    reads the atom's type before the graph's own guard runs. It must be one
    clear error at parse time instead."""
    from . import cgenff as c
    bad = _ETOH_STR.replace("BOND C2 O1", "BOND C2 QQ")
    try:
        c.parse_stream(bad)
    except ValueError as exc:
        assert "QQ" in str(exc) and "ATOM record" in str(exc), exc
    else:
        assert False, "a bond to an undeclared atom was not refused"


@test
def reference_cgenff_flags_a_non_integer_net_charge():
    """CGenFF partial charges sum to the formal (integer) charge; a net that is
    not near an integer means a truncated or hand-edited stream. Reported in the
    build, never refused."""
    from . import cgenff as c
    d = tempfile.mkdtemp()
    ok = c.generate_ligand_topology(_ETOH_STR, os.path.join(d, "ok"))
    assert ok["net_charge_is_integer"] and ok["charge_note"] is None, ok

    unbalanced = _ETOH_STR.replace("ATOM O1  OG311  -0.650",
                                   "ATOM O1  OG311  -0.600")
    bad = c.generate_ligand_topology(unbalanced, os.path.join(d, "bad"))
    assert bad["ok"] and not bad["net_charge_is_integer"], bad
    assert "not close to an integer" in bad["charge_note"], bad
    approx(bad["net_charge"], 0.05, 1e-6)


@test
def reference_cgenff_accepts_a_nonzero_integer_charge():
    """A charged ligand (a carboxylate at -1, say) is normal. The integer-charge
    check must accept it and flag only a *non*-integer net."""
    from . import cgenff as c
    charged = _ETOH_STR.replace("ATOM O1  OG311  -0.650",
                                "ATOM O1  OG311  -1.650")
    r = c.generate_ligand_topology(charged, tempfile.mkdtemp())
    approx(r["net_charge"], -1.0, 1e-6)
    assert r["net_charge_is_integer"] and r["charge_note"] is None, r


@test
def regression_cgenff_resname_cannot_escape_the_output_dir():
    """A residue name is joined onto the output directory to make two filenames,
    so an override of '../x' used to write outside it -- a path traversal that
    matters most through the dashboard, where resname is a user field. It, and
    anything with a separator, dot or space, must be refused."""
    from . import cgenff as c
    d = tempfile.mkdtemp()
    for bad in ("../PWNED", "a/b", "..", "x.itp", "a b", "LIG."):
        try:
            c.generate_ligand_topology(_ETOH_STR, d, resname=bad)
        except ValueError as exc:
            assert "residue name" in str(exc), exc
        else:
            raise AssertionError("resname %r was accepted" % bad)
    # a plain identifier is fine, and lands inside the output directory
    rep = c.generate_ligand_topology(_ETOH_STR, d, resname="DRG1")
    assert os.path.basename(rep["itp"]) == "DRG1.itp", rep
    assert os.path.dirname(os.path.abspath(rep["itp"])) == os.path.abspath(d)


@test
def invariant_ring_dihedrals_and_pairs_are_deduped():
    """A ring is where the graph enumeration earns its keep: a 1-4 pair is
    reachable two ways round the ring (must dedup), one per central bond. Checked
    on a benzene ring directly, so no force-field parameters are needed."""
    from . import cgenff as c
    names = ["C1", "C2", "C3", "C4", "C5", "C6"]
    bonds = [("C1", "C2"), ("C2", "C3"), ("C3", "C4"),
             ("C4", "C5"), ("C5", "C6"), ("C6", "C1")]

    assert len(c.enumerate_angles(names, bonds)) == 6      # one per vertex
    dih = c.enumerate_dihedrals(names, bonds)
    assert len(dih) == 6, dih                              # one per ring bond
    assert len({frozenset(q) for q in dih}) == 6, dih      # no repeated atom set

    pairs = c.one_four_pairs(names, bonds, dih)
    # the three para pairs, each found twice round the ring and deduped to one
    assert {frozenset(p) for p in pairs} == {
        frozenset(("C1", "C4")), frozenset(("C2", "C5")),
        frozenset(("C3", "C6"))}, pairs


@test
def invariant_small_ring_excludes_a_1_4_that_is_also_1_3():
    """In a five-ring a cross-ring pair is three bonds one way and two the other
    -- a 1-4 AND a 1-3. It must NOT be emitted as a 1-4 pair, or it gets a
    spurious scaled interaction on top of its 1-3 exclusion."""
    from . import cgenff as c
    names = ["A", "B", "C", "D", "E"]
    bonds = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")]
    dih = c.enumerate_dihedrals(names, bonds)
    assert c.one_four_pairs(names, bonds, dih) == []       # every pair is 1-2/1-3


def _stub_cgenff(tmpdir, str_text=_ETOH_STR, fail=False, empty=False,
                 output_flag=None):
    """A fake `cgenff` that prints a known .str to stdout, the documented
    behaviour of the real binary. The licensed program is not on this machine,
    so what is testable is the plumbing around it: discovery, the command line,
    capturing stdout, and the errors -- exactly as the pdb2gmx stub does.
    With `output_flag` it instead writes the stream to the file named after that
    flag, the way a build that does not use stdout would."""
    tpl = os.path.join(tmpdir, "str_template.txt")
    with open(tpl, "w") as fh:
        fh.write(str_text)

    if fail:
        action = ("    sys.stderr.write('cgenff: cannot read molecule\\n')\n"
                  "    sys.exit(1)\n")
    elif empty:
        action = "    sys.exit(0)\n"
    elif output_flag:
        action = ("    out = None\n"
                  "    for i, x in enumerate(a):\n"
                  "        if x == %r and i + 1 < len(a):\n"
                  "            out = a[i + 1]\n"
                  "    open(out, 'w').write(open(TPL).read())\n"
                  "    sys.exit(0)\n" % output_flag)
    else:
        action = ("    sys.stdout.write(open(TPL).read())\n"
                  "    sys.exit(0)\n")
    py = os.path.join(tmpdir, "cgenff_stub.py")
    with open(py, "w") as fh:
        fh.write("import sys\nTPL = %r\na = sys.argv[1:]\n"
                 "if True:\n" % tpl + action)

    if os.name == "nt":
        launcher = os.path.join(tmpdir, "cgenff.bat")
        with open(launcher, "w") as fh:
            fh.write('@echo off\r\n"%s" "%s" %%*\r\nexit /b %%ERRORLEVEL%%\r\n'
                     % (sys.executable, py))
    else:
        launcher = os.path.join(tmpdir, "cgenff")
        with open(launcher, "w") as fh:
            fh.write('#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable, py))
        os.chmod(launcher, 0o755)
    return launcher


@test
def regression_cgenff_missing_binary_points_at_paramchem():
    """The binary is licensed and optional. A missing one must say so clearly
    AND name the licence-free alternative, not fail obscurely -- the .str entry
    point needs no binary at all."""
    from . import cgenff as c
    try:
        c.find_cgenff(cgenff=os.path.join(tempfile.mkdtemp(), "nope"))
    except c.CGenFFNotFound as exc:
        assert "nope" in str(exc), exc
        assert "ParamChem" in str(exc), exc
    else:
        raise AssertionError("a missing cgenff binary was not reported")


@test
def endtoend_cgenff_binary_drives_the_converter():
    """mol2 -> (stub) cgenff -> .str -> the same tested converter. The result
    must be identical to feeding the .str directly, and the intermediate .str
    kept alongside the topology."""
    from . import cgenff as c
    d = tempfile.mkdtemp()
    stub = _stub_cgenff(d)
    mol = os.path.join(d, "ligand.mol2")
    with open(mol, "w") as fh:
        fh.write("@<TRIPOS>MOLECULE\nethanol\n")

    rep = c.generate_ligand_topology_from_molecule(
        mol, os.path.join(d, "out"), cgenff=stub)
    assert rep["ok"] and rep["resname"] == "ETOH", rep
    assert rep["n_atoms"] == 9 and rep["n_angles"] == 13, rep
    assert os.path.exists(rep["str"]), rep          # intermediate .str kept
    # the produced topology loads through the package parser
    mol_top = topology.parse_itp(rep["itp"])["ETOH"]
    assert mol_top.natoms == 9, mol_top.natoms


@test
def regression_cgenff_binary_failure_and_silence_are_both_caught():
    """Two ways the binary can betray a pipeline: a non-zero exit, and a zero
    exit with no stream written. Both must raise, not produce an empty
    topology."""
    from . import cgenff as c
    d = tempfile.mkdtemp()
    mol = os.path.join(d, "ligand.mol2")
    with open(mol, "w") as fh:
        fh.write("@<TRIPOS>MOLECULE\nx\n")

    failing = _stub_cgenff(d, fail=True)
    try:
        c.run_cgenff(mol, os.path.join(d, "a.str"), cgenff=failing)
    except RuntimeError as exc:
        assert "cgenff failed" in str(exc), exc
    else:
        raise AssertionError("a failing cgenff was swallowed")

    silent = _stub_cgenff(d, empty=True)
    try:
        c.run_cgenff(mol, os.path.join(d, "b.str"), cgenff=silent)
    except RuntimeError as exc:
        assert "nothing to stdout" in str(exc), exc
    else:
        raise AssertionError("a silent cgenff was swallowed")


@test
def endtoend_cgenff_binary_that_writes_a_file_is_supported():
    """Not every cgenff build writes to stdout; some take a flag naming an
    output file. run_cgenff must drive that form -- `cgenff -f out mol` -- and
    pick the .str up from the file, and still catch a build that writes none."""
    from . import cgenff as c
    d = tempfile.mkdtemp()
    mol = os.path.join(d, "m.mol2")
    with open(mol, "w") as fh:
        fh.write("@<TRIPOS>MOLECULE\nethanol\n")

    stub = _stub_cgenff(d, output_flag="-f")
    out = os.path.join(d, "o.str")
    c.run_cgenff(mol, out, cgenff=stub, output_flag="-f")
    assert os.path.exists(out) and "ETOH" in open(out).read(), out

    # a build that exits 0 but writes no file must still raise
    silent = _stub_cgenff(d, empty=True)
    try:
        c.run_cgenff(mol, os.path.join(d, "none.str"), cgenff=silent,
                     output_flag="-f")
    except RuntimeError as exc:
        assert "wrote no" in str(exc), exc
    else:
        raise AssertionError("a cgenff that wrote no output file was swallowed")


@test
def endtoend_cgenff_api_entry_takes_a_stream_and_validates_input():
    """The flat api.generate_ligand_topology is the drivable surface -- it must
    build from a .str written to disk, and reject the two obvious input
    mistakes (neither source, or both) with a message that says what to pass."""
    from .api import generate_ligand_topology as api_lig
    d = tempfile.mkdtemp()
    strp = os.path.join(d, "etoh.str")
    with open(strp, "w") as fh:
        fh.write(_ETOH_STR)

    rep = api_lig({"str": strp, "output_dir": os.path.join(d, "out")})
    assert rep["ok"] and rep["resname"] == "ETOH", rep

    for bad in ({"output_dir": d},                       # neither source
                {"str": strp, "molecule": "x.mol2",      # both sources
                 "output_dir": d}):
        try:
            api_lig(bad)
        except ValueError as exc:
            assert "exactly one" in str(exc), exc
        else:
            raise AssertionError("ambiguous ligand input was accepted: %r" % bad)

    # output_dir is required
    try:
        api_lig({"str": strp})
    except ValueError as exc:
        assert "output_dir" in str(exc), exc
    else:
        raise AssertionError("missing output_dir was accepted")


# ---- ligand prep: PDB -> mol2 protonation (drives Open Babel) --------------

# A protonated-ethanol mol2 the obabel stub "produces": 3 heavy atoms + 6 H.
_ETOH_MOL2 = """\
@<TRIPOS>MOLECULE
ETOH
 9 8 0 0 0
SMALL
GASTEIGER
@<TRIPOS>ATOM
      1 C1   -0.700  0.000  0.000 C.3    1 LIG    -0.040
      2 C2    0.700  0.000  0.000 C.3    1 LIG     0.140
      3 O1    1.300  1.200  0.000 O.3    1 LIG    -0.400
      4 H1   -1.100 -0.500  0.900 H      1 LIG     0.030
      5 H2   -1.100 -0.500 -0.900 H      1 LIG     0.030
      6 H3   -1.100  1.000  0.000 H      1 LIG     0.030
      7 H4    1.100 -0.500  0.900 H      1 LIG     0.060
      8 H5    1.100 -0.500 -0.900 H      1 LIG     0.060
      9 HO    2.200  1.200  0.000 H      1 LIG     0.420
@<TRIPOS>BOND
     1    1    2 1
     2    1    4 1
     3    1    5 1
     4    1    6 1
     5    2    3 1
     6    2    7 1
     7    2    8 1
     8    3    9 1
"""

# The input: ethanol heavy atoms only, no hydrogens (as a bare PDB often is).
_ETOH_PDB = """\
HETATM    1  C1  LIG     1      -0.700   0.000   0.000  1.00  0.00           C
HETATM    2  C2  LIG     1       0.700   0.000   0.000  1.00  0.00           C
HETATM    3  O1  LIG     1       1.300   1.200   0.000  1.00  0.00           O
END
"""


def _stub_obabel(tmpdir, mol2_text=_ETOH_MOL2, fail=False, empty=False):
    """A fake `obabel` that writes a known mol2 to its -O path, the way the real
    tool does. Open Babel is not on this machine, so what is testable is the
    plumbing: discovery, the -O/-p command line, and the two failure modes."""
    tpl = os.path.join(tmpdir, "mol2_template.txt")
    with open(tpl, "w") as fh:
        fh.write(mol2_text)

    if fail:
        action = ("    sys.stderr.write('0 molecules converted\\n')\n"
                  "    sys.exit(1)\n")
    elif empty:
        # exit 0 but write a molecule with no atom block (unreadable input)
        action = ("    out and open(out, 'w').write('@<TRIPOS>MOLECULE\\nx\\n')\n"
                  "    sys.exit(0)\n")
    else:
        action = ("    out and open(out, 'w').write(open(TPL).read())\n"
                  "    sys.exit(0)\n")
    py = os.path.join(tmpdir, "obabel_stub.py")
    with open(py, "w") as fh:
        fh.write("import sys\nTPL = %r\na = sys.argv[1:]\nout = None\n"
                 "for i, x in enumerate(a):\n"
                 "    if x == '-O' and i + 1 < len(a):\n        out = a[i + 1]\n"
                 "if True:\n" % tpl + action)

    if os.name == "nt":
        launcher = os.path.join(tmpdir, "obabel.bat")
        with open(launcher, "w") as fh:
            fh.write('@echo off\r\n"%s" "%s" %%*\r\nexit /b %%ERRORLEVEL%%\r\n'
                     % (sys.executable, py))
    else:
        launcher = os.path.join(tmpdir, "obabel")
        with open(launcher, "w") as fh:
            fh.write('#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable, py))
        os.chmod(launcher, 0o755)
    return launcher


@test
def endtoend_prep_pdb_to_mol2_protonates_at_ph7():
    """The licence-free front end: a bare-heavy-atom PDB -> a protonated mol2.
    The report must name the pH and count the hydrogens Open Babel added, so the
    protonation choice is visible rather than buried."""
    from . import prep
    d = tempfile.mkdtemp()
    pdb = os.path.join(d, "etoh.pdb")
    with open(pdb, "w") as fh:
        fh.write(_ETOH_PDB)
    stub = _stub_obabel(d)
    out = os.path.join(d, "etoh.mol2")

    rep = prep.pdb_to_mol2(pdb, out, ph=7.0, obabel=stub)
    assert rep["ok"] and os.path.exists(out), rep
    approx(rep["ph"], 7.0, 1e-9)
    assert rep["pdb_atoms"] == 3 and rep["mol2_atoms"] == 9, rep
    assert rep["hydrogens_added"] == 6, rep


@test
def regression_prep_missing_obabel_is_reported():
    """Open Babel is optional; a missing one must say what it tried and how to
    get it, not fail obscurely deeper in the pipeline."""
    from . import prep
    try:
        prep.find_obabel(obabel=os.path.join(tempfile.mkdtemp(), "nope"))
    except prep.OpenBabelNotFound as exc:
        assert "nope" in str(exc) and "openbabel" in str(exc).lower(), exc
    else:
        raise AssertionError("a missing obabel was not reported")


@test
def regression_prep_obabel_failure_and_empty_are_both_caught():
    """Two ways obabel can betray the pipeline: a non-zero exit, and a zero exit
    with an atomless mol2 (unreadable input). Both must raise, not hand back a
    mol2 with no molecule in it."""
    from . import prep
    d = tempfile.mkdtemp()
    pdb = os.path.join(d, "x.pdb")
    with open(pdb, "w") as fh:
        fh.write(_ETOH_PDB)

    failing = _stub_obabel(d, fail=True)
    try:
        prep.pdb_to_mol2(pdb, os.path.join(d, "a.mol2"), obabel=failing)
    except RuntimeError as exc:
        assert "obabel failed" in str(exc), exc
    else:
        raise AssertionError("a failing obabel was swallowed")

    empty = _stub_obabel(d, empty=True)
    try:
        prep.pdb_to_mol2(pdb, os.path.join(d, "b.mol2"), obabel=empty)
    except RuntimeError as exc:
        assert "no atoms" in str(exc), exc
    else:
        raise AssertionError("an atomless mol2 was accepted")


@test
def regression_prep_no_ph_conversion_omits_the_protonation():
    """add_hydrogens=False converts a structure as it stands: the report records
    no pH, and says so, rather than implying a protonation that did not happen."""
    from . import prep
    d = tempfile.mkdtemp()
    pdb = os.path.join(d, "e.pdb")
    with open(pdb, "w") as fh:
        fh.write(_ETOH_PDB)
    rep = prep.pdb_to_mol2(pdb, os.path.join(d, "o.mol2"),
                           add_hydrogens=False, obabel=_stub_obabel(d))
    assert rep["ok"] and rep["ph"] is None, rep
    assert "as-is" in rep["note"], rep["note"]


@test
def endtoend_prep_pdb_all_the_way_to_topology():
    """PDB -> (stub obabel) mol2 -> (stub cgenff) .str -> the tested converter.
    The full-auto path chains both drivers; the result is the same topology the
    .str route gives, with the mol2 kept and the pH recorded."""
    from . import prep
    d = tempfile.mkdtemp()
    pdb = os.path.join(d, "etoh.pdb")
    with open(pdb, "w") as fh:
        fh.write(_ETOH_PDB)
    obabel = _stub_obabel(d)
    cgenff = _stub_cgenff(d)

    rep = prep.generate_ligand_topology_from_pdb(
        pdb, os.path.join(d, "out"), ph=7.0, obabel=obabel, cgenff=cgenff)
    assert rep["ok"] and rep["resname"] == "ETOH", rep
    assert rep["n_atoms"] == 9, rep
    approx(rep["ph"], 7.0, 1e-9)
    assert rep["hydrogens_added"] == 6, rep
    assert os.path.exists(rep["mol2"]) and os.path.exists(rep["itp"]), rep


@test
def endtoend_cli_ligand_routes_str_and_mol2_subcommand_runs():
    """The `ligand` positional routes on its extension (a .str is a stream) and
    the new `mol2` subcommand drives Open Babel. Both are the drivable surface,
    so both are exercised the way a shell would, JSON output and all."""
    import contextlib
    from . import __main__ as cli
    d = tempfile.mkdtemp()

    strp = os.path.join(d, "e.str")
    with open(strp, "w") as fh:
        fh.write(_ETOH_STR)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["ligand", strp, os.path.join(d, "out")])
    assert rc == 0, buf.getvalue()
    assert os.path.exists(os.path.join(d, "out", "ETOH.itp")), buf.getvalue()

    pdb = os.path.join(d, "e.pdb")
    with open(pdb, "w") as fh:
        fh.write(_ETOH_PDB)
    out_mol2 = os.path.join(d, "e.mol2")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["mol2", pdb, out_mol2, "--obabel", _stub_obabel(d)])
    assert rc == 0, buf.getvalue()
    assert os.path.exists(out_mol2), buf.getvalue()


@test
def endtoend_api_ligand_prep_surface():
    """The flat api.prepare_mol2 and the 'pdb' route of generate_ligand_topology
    are the no-licence front-end as an agent drives it. Both must run (tools
    stubbed) and reject the obvious input mistakes."""
    from . import api
    d = tempfile.mkdtemp()
    pdb = os.path.join(d, "e.pdb")
    with open(pdb, "w") as fh:
        fh.write(_ETOH_PDB)
    obabel = _stub_obabel(d)

    rep = api.prepare_mol2({"pdb": pdb, "output": os.path.join(d, "e.mol2"),
                            "ph": 7.0, "obabel": obabel})
    assert rep["ok"] and rep["ph"] == 7.0 and os.path.exists(rep["mol2"]), rep

    for bad, needle in (({"pdb": pdb}, "required"),
                        ({"pdb": pdb, "output": "x", "junk": 1}, "unknown")):
        try:
            api.prepare_mol2(bad)
        except ValueError as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError("prepare_mol2 accepted %r" % bad)

    # the 'pdb' route: PDB -> (stub obabel) mol2 -> (stub cgenff) .str -> topology
    r = api.generate_ligand_topology({
        "pdb": pdb, "output_dir": os.path.join(d, "out"),
        "obabel": obabel, "cgenff": _stub_cgenff(d)})
    assert r["ok"] and r["resname"] == "ETOH", r
    assert r["hydrogens_added"] == 6 and r["ph"] == 7.0, r


@test
def endtoend_ligand_batch_parameterises_a_directory():
    """A whole directory of .str files in one call (one force-field read for the
    batch). A broken stream fails without stopping the rest; the report lists
    every result, each ok ligand in its own subdirectory. Input mistakes refuse."""
    from . import api
    d = tempfile.mkdtemp()
    sdir = os.path.join(d, "streams")
    os.makedirs(sdir)
    for name, s in (("ethanol.str", _ETOH_STR), ("formaldehyde.str", _FALD_STR),
                    ("broken.str", "read rtf card append\nRESI X 0\nGROUP\n"
                                   "ATOM A CG331 0\nBOND A B\nEND\n")):
        with open(os.path.join(sdir, name), "w") as fh:
            fh.write(s)
    r = api.generate_ligand_topologies({"dir": sdir,
                                        "output_dir": os.path.join(d, "out")})
    assert r["n_streams"] == 3 and r["n_ok"] == 2 and r["n_failed"] == 1, r
    assert not r["ok"], r                          # one failed -> batch not ok
    by = {os.path.basename(x["stream"]): x for x in r["results"]}
    assert by["ethanol.str"]["ok"] and by["ethanol.str"]["resname"] == "ETOH"
    assert not by["broken.str"]["ok"] and by["broken.str"]["error"]
    assert os.path.exists(os.path.join(d, "out", "ethanol", "ETOH.itp"))

    for bad, needle in (({"output_dir": "x"}, "exactly one"),
                        ({"dir": sdir}, "output_dir"),
                        ({"dir": sdir, "output_dir": "x", "junk": 1}, "unknown")):
        try:
            api.generate_ligand_topologies(bad)
        except ValueError as exc:
            assert needle in str(exc), (needle, exc)
        else:
            raise AssertionError("batch accepted %r" % bad)


@test
def endtoend_cli_place_routes_and_builds():
    """The `place` subcommand drives placement from the shell and prints the
    report as JSON -- the last of the ligand CLI surface."""
    import contextlib
    import json as _json
    from . import __main__ as cli, cgenff
    d = tempfile.mkdtemp()
    sysd = _synthetic_protein_system(os.path.join(d, "system"))
    ligd = os.path.join(d, "lig")
    cgenff.generate_ligand_topology(_ETOH_STR, ligd)
    pdb = os.path.join(d, "e.pdb")
    _ethanol_pdb(pdb)
    out = os.path.join(d, "placed")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["place", sysd, pdb, os.path.join(ligd, "ETOH.itp"), out])
    assert rc == 0, buf.getvalue()
    r = _json.loads(buf.getvalue())
    assert r["ok"] and r["ligand"] == "ETOH", r
    assert os.path.exists(os.path.join(out, "step5_input.gro"))


# A REAL ParamChem stream (JZ4 / 2-propylphenol, the GROMACS protein-ligand
# tutorial ligand, from Lemkul-Lab/cgenff_charmm2gmx tests/test_data/jz4.str).
# It is here to keep the parser honest against real output, which differs from
# the hand-built fixtures in two ways that matter: it has NO MASS records, and
# its parameter sections carry only the one term CGenFF generated by analogy --
# every standard bond/angle/dihedral comes from the base CGenFF force field,
# which the stream does not include.
_JZ4_STR = """\
* Toppar stream file generated by
* CHARMM General Force Field (CGenFF) program version 2.5
*

read rtf card append
* Topologies generated by
* CHARMM General Force Field (CGenFF) program version 2.5
*
36 1

RESI JZ4            0.000 ! param penalty=   0.900 ; charge penalty=   0.468
GROUP            ! CHARGE   CH_PENALTY
ATOM C4     CG331  -0.275 !    0.319
ATOM C7     CG2R61 -0.107 !    0.000
ATOM C8     CG2R61 -0.114 !    0.000
ATOM C9     CG2R61 -0.105 !    0.000
ATOM C10    CG2R61  0.109 !    0.239
ATOM C11    CG2R61 -0.117 !    0.000
ATOM C12    CG2R61  0.001 !    0.365
ATOM C13    CG321  -0.195 !    0.468
ATOM C14    CG321  -0.176 !    0.342
ATOM OAB    OG311  -0.531 !    0.262
ATOM H1     HGA3    0.090 !    0.000
ATOM H2     HGA3    0.090 !    0.000
ATOM H3     HGA3    0.090 !    0.000
ATOM H4     HGR61   0.115 !    0.000
ATOM H5     HGR61   0.115 !    0.000
ATOM H6     HGR61   0.115 !    0.000
ATOM H7     HGR61   0.115 !    0.000
ATOM H8     HGA2    0.090 !    0.000
ATOM H9     HGA2    0.090 !    0.000
ATOM H10    HGA2    0.090 !    0.000
ATOM H11    HGA2    0.090 !    0.000
ATOM H12    HGP1    0.420 !    0.000

BOND C4   C14
BOND C4   H1
BOND C4   H2
BOND C4   H3
BOND C7   C8
BOND C7   C11
BOND C7   H4
BOND C8   C9
BOND C8   H5
BOND C9   C10
BOND C9   H6
BOND C10  OAB
BOND C10  C12
BOND C11  C12
BOND C11  H7
BOND C12  C13
BOND C13  C14
BOND C13  H8
BOND C13  H9
BOND C14  H10
BOND C14  H11
BOND OAB  H12

END

read param card flex append
* Parameters generated by analogy by
* CHARMM General Force Field (CGenFF) program version 2.5
*

BONDS

ANGLES

DIHEDRALS
CG2R61 CG321  CG321  CG331      0.0400  3     0.00 ! JZ4 , from CG2R61 CG321 CG321 CG321, penalty= 0.9

IMPROPERS

END
RETURN
"""


@test
def reference_cgenff_parses_a_real_paramchem_stream():
    """Parse a real CGenFF stream, not a hand-built one. Two real-world facts
    the fixtures did not have: no MASS records (mass inferred from the type
    name), and near-empty parameter sections (the rest live in the base FF)."""
    from . import cgenff as c
    s = c.parse_stream(_JZ4_STR)
    assert s.resname == "JZ4", s.resname
    assert len(s.atoms) == 22 and len(s.bonds) == 22, (len(s.atoms), len(s.bonds))
    assert s.atoms[0].name == "C4" and s.atoms[0].type == "CG331"
    approx(s.atoms[0].charge, -0.275, 1e-9)
    approx(s.atoms[0].charge_penalty, 0.319, 1e-9)     # read from the comment

    # the stream has no masses; they are inferred from the CGenFF type names
    assert len(s.masses) == 0, s.masses
    approx(s.mass_of("CG331"), 12.011, 1e-6)
    approx(s.mass_of("OG311"), 15.999, 1e-6)
    approx(s.mass_of("HGA3"), 1.008, 1e-6)
    approx(s.mass_of("CG2R61"), 12.011, 1e-6)

    # only the single analogy-generated dihedral is present
    assert len(s.bond_p) == 0 and len(s.angle_p) == 0 and len(s.impr_p) == 0, s
    assert len(s.dihe_p) == 1, s.dihe_p
    ((kchi, mult, delta, pen),) = s.dihe_p[("CG2R61", "CG321", "CG321", "CG331")]
    approx(pen, 0.9, 1e-9)


@test
def regression_cgenff_non_self_contained_stream_is_diagnosed():
    """A real stream cannot be converted yet -- most parameters are in the base
    CGenFF FF, which the merge step (still to come) will read. The failure must
    say exactly that, not imply the stream is broken."""
    from . import cgenff as c
    try:
        c.generate_ligand_topology(_JZ4_STR, tempfile.mkdtemp())
    except ValueError as exc:
        msg = str(exc)
        assert "supplement" in msg.lower(), msg
        assert "top_all36_cgenff" in msg and "par_all36_cgenff" in msg, msg
        # names a real missing term so the user can see what is meant
        assert "CG2R61" in msg, msg
    else:
        raise AssertionError("a non-self-contained stream was converted anyway")


@test
def endtoend_cgenff_merges_missing_params_from_the_base_ff():
    """A real stream carries only its atoms/bonds and the analogy params; the
    rest come from the base CGenFF FF. A stream stripped to its RTF, plus a
    small FF supplying the parameters, must build the same topology as the
    self-contained stream did on its own."""
    from . import cgenff as c
    d = tempfile.mkdtemp()

    # a stream with the parameters removed, like a real ParamChem one
    rtf_half = _ETOH_STR.split("read param")[0]
    stripped = (rtf_half + "read param card flex append\n"
                "BONDS\nANGLES\nDIHEDRALS\nIMPROPERS\nNONBONDED\nEND\n")

    # a tiny base FF: masses in an rtf, the parameters in a prm (reusing the
    # very blocks the self-contained fixture carried inline)
    rtf = os.path.join(d, "top_all36_cgenff.rtf")
    with open(rtf, "w") as fh:
        for t in ("CG331", "CG321"):
            fh.write("MASS -1 %s 12.01100 C\n" % t)
        fh.write("MASS -1 OG311 15.99940 O\n")
        for t in ("HGA3", "HGA2", "HGP1"):
            fh.write("MASS -1 %s 1.00800 H\n" % t)
    prm = os.path.join(d, "par_all36_cgenff.prm")
    with open(prm, "w") as fh:
        fh.write("BONDS" + _ETOH_STR.split("BONDS", 1)[1])   # BONDS..END..RETURN

    merged = c.generate_ligand_topology(stripped, os.path.join(d, "m"),
                                        cgenff_ff=(rtf, prm))
    whole = c.generate_ligand_topology(_ETOH_STR, os.path.join(d, "s"))

    assert merged["base_ff_merged"] is True and whole["base_ff_merged"] is False
    for k in ("n_atoms", "n_bonds", "n_angles", "n_dihedrals", "n_impropers"):
        assert merged[k] == whole[k], (k, merged[k], whole[k])
    approx(merged["net_charge"], 0.0, 1e-9)

    # the two topologies are the same molecule, atom for atom and bond for bond
    mm = topology.parse_itp(merged["itp"])["ETOH"]
    ww = topology.parse_itp(whole["itp"])["ETOH"]
    assert mm.atomname == ww.atomname and sorted(mm.bonds) == sorted(ww.bonds)
    approx(mm.total_charge, 0.0, 1e-6)


@test
def invariant_cgenff_ff_is_cached_by_path_and_mtime():
    """Parsing the base force field is the one real cost of a merge, and a whole
    ligand library reuses one FF, so it is cached -- but keyed by mtime, so an
    edited file is re-read and the cache never goes stale."""
    from . import cgenff as c
    d = tempfile.mkdtemp()
    rtf = os.path.join(d, "top_all36_cgenff.rtf")
    prm = os.path.join(d, "par_all36_cgenff.prm")
    with open(rtf, "w") as fh:
        fh.write("MASS -1 CG331 12.011 C\n")
    with open(prm, "w") as fh:
        fh.write("BONDS\nCG331 CG331 200 1.5\n")
    try:
        a = c.load_cgenff_ff(d)
        assert c.load_cgenff_ff(d) is a             # cache hit: same object
        os.utime(prm, (time.time() + 5, time.time() + 5))   # edit -> newer mtime
        assert c.load_cgenff_ff(d) is not a         # re-read, not stale
    finally:
        c.clear_ff_cache()


@test
def reference_cgenff_jz4_matches_the_gold_standard():
    """The acceptance test. The real JZ4 stream, merged over the base CGenFF FF,
    must reproduce the topology CHARMM-GUI's Ligand Reader / Lemkul's reference
    converter gives -- count for count (validated section-for-section once, and
    locked in here so the converter cannot drift off it). The base FF is
    MacKerell material and not shipped, so point LAMELLYX_CGENFF_FF at a toppar/
    that has top_all36_cgenff.rtf + par_all36_cgenff.prm; without it this skips.
    """
    from . import cgenff as c
    ff = os.environ.get("LAMELLYX_CGENFF_FF")
    if not ff or not os.path.isdir(ff):
        raise SkipTest("set LAMELLYX_CGENFF_FF to a toppar/ with "
                       "top_all36_cgenff.rtf + par_all36_cgenff.prm")
    d = tempfile.mkdtemp()
    rep = c.generate_ligand_topology(_JZ4_STR, d, cgenff_ff=ff)
    assert (rep["n_atoms"], rep["n_bonds"], rep["n_angles"],
            rep["n_dihedrals"], rep["n_impropers"]) == (22, 22, 37, 50, 0), rep
    approx(rep["net_charge"], 0.0, 1e-6)
    assert rep["net_charge_is_integer"], rep
    # 47 one-four pairs, matching the reference converter
    npairs = len(_section_lines(
        open(os.path.join(d, "JZ4.itp")).read(), "pairs"))
    assert npairs == 47, npairs


@test
def endtoend_cgenff_missing_base_ff_file_is_reported():
    """Point the merge at a directory without the CGenFF files and it must say
    which file it wanted and what it is, not fail obscurely."""
    from . import cgenff as c
    try:
        c.load_cgenff_ff(tempfile.mkdtemp())
    except FileNotFoundError as exc:
        assert "top_all36_cgenff.rtf" in str(exc) or "par_all36" in str(exc), exc
        assert "CHARMM-GUI" in str(exc), exc
    else:
        raise AssertionError("a missing base CGenFF file was not reported")


# --- ligand placement ------------------------------------------------------

def _mini_itp(path, name, natoms):
    """A minimal, parseable per-molecule .itp: n atoms in a bonded chain, all
    neutral so the system's net charge is easy to reason about."""
    lines = ["[ moleculetype ]", "%s 3" % name, "", "[ atoms ]"]
    for i in range(1, natoms + 1):
        lines.append("%5d %5s %5d %5s %5s %5d %8.3f %8.3f"
                     % (i, "CT", 1, name, "A%d" % i, i, 0.0, 12.011))
    lines += ["", "[ bonds ]"]
    for i in range(1, natoms):
        lines.append("%5d %5d 1" % (i, i + 1))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_forcefield(path, extra_atomtypes=""):
    with open(path, "w") as fh:
        fh.write("[ defaults ]\n1 2 yes 1.0 1.0\n\n[ atomtypes ]\n"
                 "    CT 6 12.011 0.0 A 3.5e-01 2.7e-01\n" + extra_atomtypes)


def _synthetic_protein_system(d, extra_atomtypes=""):
    """A tiny but structurally real built system: 1 protein (3 atoms), 2 lipids
    (5 each), 4 waters (3 each) = 25 atoms, written with the package's own
    writers so it has the same shape a real build does."""
    os.makedirs(d)
    toppar = os.path.join(d, "toppar")
    os.makedirs(toppar)
    _write_forcefield(os.path.join(toppar, "forcefield.itp"), extra_atomtypes)
    _mini_itp(os.path.join(toppar, "PROA.itp"), "PROA", 3)
    _mini_itp(os.path.join(toppar, "LIP.itp"), "LIP", 5)
    _mini_itp(os.path.join(toppar, "SOL.itp"), "SOL", 3)

    rng = np.random.default_rng(1)
    xyz = rng.uniform(5.0, 20.0, (25, 3))     # Angstrom, all in one corner
    names = np.array(["A%d" % ((i % 5) + 1) for i in range(25)], dtype="<U6")
    resn = np.array(["PROA"] * 3 + ["LIP"] * 10 + ["SOL"] * 12, dtype="<U6")
    resid = np.array([1, 1, 1] + [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
                     + [4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7], dtype=np.int64)
    atoms = fileio.Atoms(names, resn, resid, xyz)
    fileio.write_gro(os.path.join(d, "step5_input.gro"), atoms,
                     np.array([50.0, 50.0, 50.0]))
    topology.write_index(os.path.join(d, "index.ndx"),
                         topology.standard_groups(25, slice(0, 3),
                                                  slice(3, 13), slice(13, 25)))
    topology.write_topol(os.path.join(d, "topol.top"),
                         ["PROA.itp", "LIP.itp", "SOL.itp"],
                         [("PROA", 1), ("LIP", 2), ("SOL", 4)])
    return d


def _ethanol_pdb(path):
    """Ethanol atoms in DELIBERATELY scrambled order (reversed), placed away
    from the synthetic system, so the placer's reordering is actually tested."""
    order = ["C1", "H11", "H12", "H13", "C2", "H21", "H22", "O1", "HO1"]
    scrambled = list(reversed(order))
    xyz = np.array([[28.0 + 1.2 * i, 30.0, 30.0] for i in range(9)])
    atoms = fileio.Atoms(scrambled, ["ETOH"] * 9, [1] * 9, xyz)
    fileio.write_pdb(path, atoms, box=np.array([50.0, 50.0, 50.0]))


@test
def endtoend_ligand_placed_into_a_protein_system():
    """The whole placement path: convert ethanol, place a positioned PDB, and
    check every book stays balanced -- atom order matched to the .itp, counts
    and index groups consistent, the ligand inside SOLU, new atom types added
    exactly once."""
    from . import cgenff, ligand
    d = tempfile.mkdtemp()
    sysd = _synthetic_protein_system(os.path.join(d, "system"))
    ligd = os.path.join(d, "lig")
    cgenff.generate_ligand_topology(_ETOH_STR, ligd)
    pdb = os.path.join(d, "ethanol.pdb")
    _ethanol_pdb(pdb)

    out = os.path.join(d, "system_lig")
    rep = ligand.add_ligand(sysd, pdb, os.path.join(ligd, "ETOH.itp"), out)
    assert rep["ok"] and rep["ligand"] == "ETOH", rep
    assert rep["ligand_atoms"] == 9 and rep["atom_total"] == 34, rep
    approx(rep["net_charge"], 0.0, 1e-6)
    assert set(rep["atomtypes_added"]) == {
        "CG331", "CG321", "OG311", "HGA3", "HGA2", "HGP1"}, rep["atomtypes_added"]

    # [ molecules ] gained ETOH right after the protein, before the lipid
    assert rep["molecules"] == [("PROA", 1), ("ETOH", 1), ("LIP", 2),
                                ("SOL", 4)], rep["molecules"]

    # the .gro: ligand atoms sit at 3..11, in the .itp order, not the PDB order
    new_atoms, _ = fileio.read_gro(os.path.join(out, "step5_input.gro"))
    assert len(new_atoms) == 34
    lig_names = list(new_atoms.name[3:12])
    assert lig_names == ["C1", "H11", "H12", "H13", "C2", "H21", "H22",
                         "O1", "HO1"], lig_names

    # the index groups still cover every atom exactly once, and SOLU grew by 9
    idx = ligand.parse_index(os.path.join(out, "index.ndx"))
    assert len(idx["SOLU"]) == 12, len(idx["SOLU"])          # 3 protein + 9
    assert len(idx["MEMB"]) == 10 and len(idx["SOLV"]) == 12, idx
    covered = len(idx["SOLU"]) + len(idx["MEMB"]) + len(idx["SOLV"])
    assert covered == 34, covered

    # topol.top: the atom types are #included before any moleculetype
    with open(os.path.join(out, "topol.top")) as fh:
        top = fh.read()
    assert "ETOH_atomtypes.itp" in top and "ETOH.itp" in top, top
    assert top.index("ETOH_atomtypes.itp") < top.index("ETOH.itp"), \
        "atom types must be included before the moleculetype that uses them"
    # and the placer copied the ligand files into the new toppar
    assert os.path.exists(os.path.join(out, "toppar", "ETOH.itp"))
    assert rep["closest_ligand_contact_A"] > 0.0, rep


@test
def endtoend_ligand_placed_into_a_multi_protein_system():
    """CJ's channels are tetramers, so the solute is several protein chains. The
    ligand must insert after ALL of them (the end of SOLU), before the membrane,
    and every book still balance."""
    from . import cgenff, ligand
    d = tempfile.mkdtemp()
    sysd = os.path.join(d, "sys")
    os.makedirs(sysd)
    toppar = os.path.join(sysd, "toppar")
    os.makedirs(toppar)
    _write_forcefield(os.path.join(toppar, "forcefield.itp"))
    for nm in ("PROA", "PROB"):
        _mini_itp(os.path.join(toppar, nm + ".itp"), nm, 3)
    _mini_itp(os.path.join(toppar, "LIP.itp"), "LIP", 5)
    _mini_itp(os.path.join(toppar, "SOL.itp"), "SOL", 3)
    rng = np.random.default_rng(2)
    xyz = rng.uniform(5.0, 20.0, (28, 3))
    names = np.array(["A%d" % ((i % 5) + 1) for i in range(28)], dtype="<U6")
    resn = np.array(["PROA"] * 3 + ["PROB"] * 3 + ["LIP"] * 10 + ["SOL"] * 12,
                    dtype="<U6")
    resid = np.array([1, 1, 1, 2, 2, 2] + [3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
                     + [5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8], dtype=np.int64)
    fileio.write_gro(os.path.join(sysd, "step5_input.gro"),
                     fileio.Atoms(names, resn, resid, xyz),
                     np.array([50.0, 50.0, 50.0]))
    topology.write_index(os.path.join(sysd, "index.ndx"),
                         topology.standard_groups(28, slice(0, 6),
                                                  slice(6, 16), slice(16, 28)))
    topology.write_topol(os.path.join(sysd, "topol.top"),
                         ["PROA.itp", "PROB.itp", "LIP.itp", "SOL.itp"],
                         [("PROA", 1), ("PROB", 1), ("LIP", 2), ("SOL", 4)])

    ligd = os.path.join(d, "lig")
    cgenff.generate_ligand_topology(_ETOH_STR, ligd)
    pdb = os.path.join(d, "e.pdb")
    _ethanol_pdb(pdb)
    rep = ligand.add_ligand(sysd, pdb, os.path.join(ligd, "ETOH.itp"),
                            os.path.join(d, "out"))
    assert rep["molecules"] == [("PROA", 1), ("PROB", 1), ("ETOH", 1),
                                ("LIP", 2), ("SOL", 4)], rep["molecules"]
    idx = ligand.parse_index(os.path.join(d, "out", "index.ndx"))
    assert len(idx["SOLU"]) == 15, idx           # 6 protein + 9 ligand
    assert len(idx["SOLU"]) + len(idx["MEMB"]) + len(idx["SOLV"]) == 37
    g, _ = fileio.read_gro(os.path.join(d, "out", "step5_input.gro"))
    assert list(g.name[6:15]) == ["C1", "H11", "H12", "H13", "C2", "H21",
                                  "H22", "O1", "HO1"], list(g.name[6:15])


@test
def regression_topology_parse_itp_malformed_line_is_clean():
    """A short [ atoms ] row used to crash parse_itp with a bare IndexError,
    deep in a build or a placement. It must be a clear ValueError naming the
    section and the file, since parse_itp underpins both v1 and v2."""
    from . import topology
    d = tempfile.mkdtemp()
    p = os.path.join(d, "bad.itp")
    with open(p, "w") as fh:                          # [atoms] row missing cols
        fh.write("[ moleculetype ]\nLIG 3\n[ atoms ]\n1 CT 1\n")
    try:
        topology.parse_itp(p)
    except ValueError as exc:
        assert "atoms" in str(exc) and "bad.itp" in str(exc), exc
    else:
        raise AssertionError("a malformed .itp line was parsed without error")


@test
def endtoend_ligand_with_a_lone_pair_is_placed():
    """A halogenated ligand has a lone pair -- a virtual site in the .itp that a
    docked PDB never contains. Placement must reconstruct its coordinate from
    the host atoms, not demand it, or a halogen drug cannot be placed at all."""
    from . import cgenff, ligand
    d = tempfile.mkdtemp()
    sysd = _synthetic_protein_system(os.path.join(d, "system"))
    ligd = os.path.join(d, "lig")
    cgenff.generate_ligand_topology(_CLM_STR, ligd)     # CLM.itp carries LP1

    # a realistic docked PDB: the five real atoms, no virtual LP1
    real = ["C1", "H1", "H2", "H3", "CL1"]
    xyz = np.array([[28.0, 30, 30], [29, 30.9, 30], [29, 29.1, 30.6],
                    [29, 29.1, 29.4], [26.3, 30, 30]])
    pdb = os.path.join(d, "clm.pdb")
    fileio.write_pdb(pdb, fileio.Atoms(real, ["CLM"] * 5, [1] * 5, xyz),
                     box=np.array([50.0, 50.0, 50.0]))

    out = os.path.join(d, "sys_clm")
    rep = ligand.add_ligand(sysd, pdb, os.path.join(ligd, "CLM.itp"), out)
    assert rep["ok"] and rep["ligand_atoms"] == 6, rep       # 5 real + the LP
    assert rep["lone_pairs_reconstructed"] == ["LP1"], rep
    approx(rep["net_charge"], 0.0, 1e-6)

    # the LP lands at the sigma hole: 1.5 A off Cl, on the far side from C
    g, _ = fileio.read_gro(os.path.join(out, "step5_input.gro"))
    names = list(g.name[3:9])
    assert "LP1" in names, names
    lp, cl = g.xyz[3:9][names.index("LP1")], g.xyz[3:9][names.index("CL1")]
    approx(float(np.linalg.norm(lp - cl)), 1.5, 1e-3)        # |a| = 0.15 nm
    assert lp[0] < cl[0], (lp, cl)                           # away from C1 at x=28

    # bookkeeping still balances with the extra virtual atom in place
    idx = ligand.parse_index(os.path.join(out, "index.ndx"))
    assert len(idx["SOLU"]) == 9, idx                        # 3 protein + 6
    covered = len(idx["SOLU"]) + len(idx["MEMB"]) + len(idx["SOLV"])
    assert covered == len(g) == 31, (covered, len(g))


@test
def regression_lone_pair_with_a_missing_host_is_refused():
    """Reconstructing a lone pair needs its host atoms. A PDB missing one must
    be refused by name, not fail obscurely in the geometry."""
    from . import cgenff, ligand
    d = tempfile.mkdtemp()
    sysd = _synthetic_protein_system(os.path.join(d, "system"))
    ligd = os.path.join(d, "lig")
    cgenff.generate_ligand_topology(_CLM_STR, ligd)
    # a PDB missing CL1 (a host of LP1), and LP1 itself
    real = ["C1", "H1", "H2", "H3"]
    xyz = np.array([[28.0, 30, 30], [29, 30.9, 30], [29, 29.1, 30.6],
                    [29, 29.1, 29.4]])
    pdb = os.path.join(d, "clm.pdb")
    fileio.write_pdb(pdb, fileio.Atoms(real, ["CLM"] * 4, [1] * 4, xyz),
                     box=np.array([50.0, 50.0, 50.0]))
    try:
        ligand.add_ligand(sysd, pdb, os.path.join(ligd, "CLM.itp"),
                          os.path.join(d, "out"))
    except ValueError as exc:
        assert "lone pair" in str(exc).lower() and "CL1" in str(exc), exc
    else:
        assert False, "a lone pair with a missing host was not refused"


@test
def regression_place_malformed_ligand_itp_is_refused_cleanly():
    """A malformed ligand .itp used to reach the placer and crash inside the
    topology parser with a bare IndexError -- uncaught by the CLI, so a
    traceback. It must be one clear ValueError instead."""
    from . import ligand
    d = tempfile.mkdtemp()
    sysd = _synthetic_protein_system(os.path.join(d, "system"))
    itp = os.path.join(d, "bad.itp")
    with open(itp, "w") as fh:                       # [atoms] row missing columns
        fh.write("[ moleculetype ]\nLIG 3\n[ atoms ]\n1\n[ bonds ]\n")
    at = os.path.join(d, "bad_atomtypes.itp")
    with open(at, "w") as fh:
        fh.write("[ atomtypes ]\n")
    pdb = os.path.join(d, "l.pdb")
    fileio.write_pdb(pdb, fileio.Atoms(["C1"], ["LIG"], [1], np.zeros((1, 3))),
                     box=np.array([50.0, 50.0, 50.0]))
    try:
        ligand.add_ligand(sysd, pdb, itp, os.path.join(d, "out"),
                          atomtypes_itp=at)
    except ValueError as exc:
        assert "ligand .itp" in str(exc), exc
    else:
        raise AssertionError("a malformed ligand .itp was accepted")


@test
def regression_place_refuses_a_name_colliding_with_a_system_molecule():
    """A ligand whose moleculetype matches a system molecule (naming it 'LIP'
    into a system that has LIP) would overwrite that molecule's .itp and list
    the name twice in [ molecules ] -- silent corruption. Refuse it."""
    from . import cgenff, ligand
    d = tempfile.mkdtemp()
    sysd = _synthetic_protein_system(os.path.join(d, "system"))    # has LIP
    ligd = os.path.join(d, "lig")
    cgenff.generate_ligand_topology(_ETOH_STR, ligd, resname="LIP")
    pdb = os.path.join(d, "l.pdb")
    _ethanol_pdb(pdb)                    # refused before atom matching anyway
    try:
        ligand.add_ligand(sysd, pdb, os.path.join(ligd, "LIP.itp"),
                          os.path.join(d, "out"))
    except ValueError as exc:
        assert "already names a molecule" in str(exc), exc
    else:
        raise AssertionError("a colliding ligand name was accepted")


@test
def regression_place_resname_labels_the_residue_not_the_moleculetype():
    """--resname overrides the ligand's RESIDUE name (a coordinate label), not
    its moleculetype. [ molecules ] must still name the moleculetype from the
    .itp, or grompp finds no matching [ moleculetype ]."""
    from . import cgenff, ligand
    d = tempfile.mkdtemp()
    sysd = _synthetic_protein_system(os.path.join(d, "system"))
    ligd = os.path.join(d, "lig")
    cgenff.generate_ligand_topology(_ETOH_STR, ligd)               # type ETOH
    pdb = os.path.join(d, "l.pdb")
    _ethanol_pdb(pdb)
    out = os.path.join(d, "out")
    rep = ligand.add_ligand(sysd, pdb, os.path.join(ligd, "ETOH.itp"), out,
                            resname="DRG")
    names = [m[0] for m in rep["molecules"]]
    assert "ETOH" in names and "DRG" not in names, rep["molecules"]
    g, _ = fileio.read_gro(os.path.join(out, "step5_input.gro"))
    assert set(g.resname[3:12]) == {"DRG"}, set(g.resname[3:12])


@test
def endtoend_two_lone_pairs_convert_and_place():
    """A dihalogen carries two lone pairs, so the vsite/exclusion emission and
    the placement reconstruction each run more than once. Both sites must come
    out with distinct hosts, both massless, and both rebuilt at placement."""
    from . import cgenff, ligand
    dbm = "\n".join([
        "* dibromomethane", "*", "read rtf card append", "* t", "*", "36 1",
        "MASS -1 CG321 12.01100", "MASS -1 HGA2 1.00800",
        "MASS -1 BRGA2 79.90400", "RESI DBM 0.000", "GROUP",
        "ATOM C1 CG321 -0.200", "ATOM H1 HGA2 0.100", "ATOM H2 HGA2 0.100",
        "ATOM BR1 BRGA2 -0.050", "ATOM BR2 BRGA2 -0.050",
        "ATOM LP1 LPH 0.050", "ATOM LP2 LPH 0.050",
        "BOND C1 H1", "BOND C1 H2", "BOND C1 BR1", "BOND C1 BR2",
        "LONEPAIR COLINEAR LP1 BR1 C1 DIST 1.6000 SCALE 0.0",
        "LONEPAIR COLINEAR LP2 BR2 C1 DIST 1.6000 SCALE 0.0", "END",
        "read param card flex append", "* p", "*",
        "BONDS", "CG321 HGA2 309.00 1.1110", "CG321 BRGA2 220.00 1.9500",
        "ANGLES", "HGA2 CG321 HGA2 35.50 109.00",
        "HGA2 CG321 BRGA2 32.00 107.00", "BRGA2 CG321 BRGA2 40.00 112.00",
        "DIHEDRALS", "IMPROPERS", "NONBONDED", "CG321 0.0 -0.0560 2.0100",
        "HGA2 0.0 -0.0280 1.3400", "BRGA2 0.0 -0.4800 2.0700", "END"])
    d = tempfile.mkdtemp()
    rep = cgenff.generate_ligand_topology(dbm, os.path.join(d, "lig"))
    assert rep["n_atoms"] == 7 and rep["n_lonepairs"] == 2, rep
    approx(rep["net_charge"], 0.0, 1e-9)

    itp = open(os.path.join(d, "lig", "DBM.itp")).read()
    vs = _section_lines(itp, "virtual_sites2")
    assert len(vs) == 2, vs
    assert {v.split()[1] for v in vs} == {"4", "5"}, vs      # hosts BR1, BR2
    assert len(_section_lines(itp, "exclusions")) == 2, itp
    for a in _section_lines(itp, "atoms"):
        if a.split()[4] in ("LP1", "LP2"):
            approx(float(a.split()[7]), 0.0, 1e-12)          # massless

    sysd = _synthetic_protein_system(os.path.join(d, "sys"))
    real = ["C1", "H1", "H2", "BR1", "BR2"]
    xyz = np.array([[30.0, 30, 30], [31, 31, 30], [31, 29, 30],
                    [27.5, 30, 30], [32.5, 30, 30]])
    pdb = os.path.join(d, "dbm.pdb")
    fileio.write_pdb(pdb, fileio.Atoms(real, ["DBM"] * 5, [1] * 5, xyz),
                     box=np.array([60.0, 60, 60]))
    pr = ligand.add_ligand(sysd, pdb, os.path.join(d, "lig", "DBM.itp"),
                          os.path.join(d, "out"))
    assert pr["ligand_atoms"] == 7, pr
    assert pr["lone_pairs_reconstructed"] == ["LP1", "LP2"], pr


@test
def reference_cgenff_lone_pair_on_a_ring_excludes_through_the_ring():
    """A halogen on an aromatic ring routes its lone pair's exclusions around the
    ring: the LP must be excluded from the ipso carbon (1-2), the two ortho
    carbons (1-3), and -- the subtle part -- the two meta carbons AND the two
    ortho hydrogens (both 1-4, the H reached C-C-H). Getting that ring-borne 1-4
    set right is the point of the test."""
    from . import cgenff as c
    clb = """\
* chlorobenzene with a chlorine lone pair
*
read rtf card append
* top
*
36 1
MASS -1 CG2R61 12.01100
MASS -1 HGR61   1.00800
MASS -1 CLGA1  35.45000
RESI CLB 0.000
GROUP
ATOM C1 CG2R61  0.000
ATOM C2 CG2R61 -0.115
ATOM C3 CG2R61 -0.115
ATOM C4 CG2R61 -0.115
ATOM C5 CG2R61 -0.115
ATOM C6 CG2R61 -0.115
ATOM H2 HGR61   0.115
ATOM H3 HGR61   0.115
ATOM H4 HGR61   0.115
ATOM H5 HGR61   0.115
ATOM H6 HGR61   0.115
ATOM CL1 CLGA1 -0.170
ATOM LP1 LPH    0.170
BOND C1 C2  C2 C3  C3 C4  C4 C5  C5 C6  C6 C1
BOND C2 H2  C3 H3  C4 H4  C5 H5  C6 H6
BOND C1 CL1
LONEPAIR COLINEAR LP1 CL1 C1 DIST 1.6000 SCALE 0.0
END
read param card flex append
* par
*
BONDS
CG2R61 CG2R61 305.00 1.3750
CG2R61 HGR61  340.00 1.0800
CG2R61 CLGA1  240.00 1.7400
ANGLES
CG2R61 CG2R61 CG2R61 40.00 120.00
CG2R61 CG2R61 HGR61  30.00 120.00
CG2R61 CG2R61 CLGA1  45.00 120.00
DIHEDRALS
X CG2R61 CG2R61 X 3.1000 2 180.00
NONBONDED
CG2R61 0.0 -0.0700 1.9924
HGR61  0.0 -0.0300 1.3582
CLGA1  0.0 -0.3430 1.9100
END
"""
    d = tempfile.mkdtemp()
    rep = c.generate_ligand_topology(clb, d)
    assert rep["n_atoms"] == 13 and rep["n_lonepairs"] == 1, rep
    approx(rep["net_charge"], 0.0, 1e-9)
    assert rep["net_charge_is_integer"], rep

    ex = _section_lines(open(os.path.join(d, "CLB.itp")).read(), "exclusions")
    assert len(ex) == 1, ex
    got = set(int(x) for x in ex[0].split()[1:])
    # LP=13; CL1=12 (host), C1=1 (ipso 1-2), C2=2 & C6=6 (ortho 1-3),
    # C3=3 & C5=5 (meta 1-4), H2=7 & H6=11 (ortho hydrogens, 1-4)
    assert got == {1, 2, 3, 5, 6, 7, 11, 12}, sorted(got)


@test
def regression_ligand_pdb_that_disagrees_with_topology_is_refused():
    """A PDB missing an atom the .itp needs, or carrying an extra one, is the
    exact topology/coordinate mismatch the package exists to catch -- it must
    be refused with the offending atoms named, not placed with a hole."""
    from . import cgenff, ligand
    d = tempfile.mkdtemp()
    sysd = _synthetic_protein_system(os.path.join(d, "system"))
    ligd = os.path.join(d, "lig")
    cgenff.generate_ligand_topology(_ETOH_STR, ligd)

    # a PDB with one hydrogen dropped
    order = ["C1", "H11", "H12", "C2", "H21", "H22", "O1", "HO1"]   # no H13
    xyz = np.array([[28.0 + 1.2 * i, 30.0, 30.0] for i in range(len(order))])
    pdb = os.path.join(d, "bad.pdb")
    fileio.write_pdb(pdb, fileio.Atoms(order, ["ETOH"] * len(order),
                                       [1] * len(order), xyz))
    try:
        ligand.add_ligand(sysd, pdb, os.path.join(ligd, "ETOH.itp"),
                          os.path.join(d, "out"))
    except ValueError as exc:
        assert "H13" in str(exc) and "missing" in str(exc).lower(), exc
    else:
        raise AssertionError("a ligand PDB missing an atom was placed anyway")


@test
def regression_ligand_atomtype_collision_is_handled():
    """A ligand type the force field already defines must be dropped if it is
    identical (no duplicate definition, which grompp rejects) and must RAISE if
    it differs (a silent parameter swap otherwise)."""
    from . import cgenff, ligand
    d = tempfile.mkdtemp()
    ligd = os.path.join(d, "lig")
    cgenff.generate_ligand_topology(_ETOH_STR, ligd)
    pdb = os.path.join(d, "ethanol.pdb")
    _ethanol_pdb(pdb)

    # read the exact CG331 line the converter wrote, and seed the force field
    # with it -- an identical prior definition, which must be dropped
    with open(os.path.join(ligd, "ETOH_atomtypes.itp")) as fh:
        cg331 = [ln for ln in fh if ln.split()[:1] == ["CG331"]][0].strip()
    same = _synthetic_protein_system(os.path.join(d, "sys_same"),
                                     extra_atomtypes=cg331 + "\n")
    rep = ligand.add_ligand(same, pdb, os.path.join(ligd, "ETOH.itp"),
                            os.path.join(d, "out_same"))
    assert "CG331" in rep["atomtypes_already_present"], rep
    assert "CG331" not in rep["atomtypes_added"], rep

    # now the same type with a DIFFERENT sigma -- a real conflict, must raise
    diff = _synthetic_protein_system(
        os.path.join(d, "sys_diff"),
        extra_atomtypes="CG331 6 12.011 0.0 A 9.9e-01 9.9e-01\n")
    try:
        ligand.add_ligand(diff, pdb, os.path.join(ligd, "ETOH.itp"),
                          os.path.join(d, "out_diff"))
    except ValueError as exc:
        assert "CG331" in str(exc) and "different" in str(exc).lower(), exc
    else:
        raise AssertionError("a conflicting atom type redefinition was allowed")


@test
def regression_place_drops_pairtypes_the_force_field_already_has():
    """A ligand's _atomtypes.itp carries a [ pairtypes ] entry for every pair of
    its types. Placed into a system whose force field already gives some of
    those pairs (a CGenFF ligand into a CGenFF-aware system), the duplicates
    must be dropped -- writing a pair type twice is a grompp override warning.
    New pairs are kept; an atom-type conflict still refuses."""
    from . import ligand
    d = tempfile.mkdtemp()
    lig_at = os.path.join(d, "L_atomtypes.itp")
    with open(lig_at, "w") as fh:
        fh.write("[ atomtypes ]\n"
                 " CG331 6 12.011 0.0 A 3.6e-01 3.3e-01\n"
                 " OG311 8 15.999 0.0 A 3.1e-01 4.0e-01\n"
                 "[ pairtypes ]\n"
                 " CG331 CG331 1 3.4e-01 4.2e-02\n"       # FF already has this
                 " CG331 OG311 1 3.2e-01 4.5e-02\n")      # new -> keep
    out = os.path.join(d, "out_atomtypes.itp")
    existing_at = {"CG331": "CG331 6 12.011 0.0 A 3.6e-01 3.3e-01"}   # identical
    kept, dropped = ligand._dedup_atomtypes(
        lig_at, existing_at, {("CG331", "CG331")}, out)
    assert dropped == ["CG331"], dropped
    assert [ln.split()[0] for ln in kept] == ["OG311"], kept
    pairs = [tuple(sorted(l.split()[:2]))
             for l in _section_lines(open(out).read(), "pairtypes")]
    assert ("CG331", "CG331") not in pairs, pairs         # dropped (FF has it)
    assert ("CG331", "OG311") in pairs, pairs             # new -> kept


@test
def regression_ligand_atomtype_dedup_tolerates_float_formatting():
    """A system force field can carry the ligand's CGenFF atom types with
    coarser float formatting (0.36349 vs the converter's 3.634867e-01) -- the
    SAME type, not a redefinition. The dedup compares numerically, so it drops
    it rather than refusing a valid placement, while a genuinely different
    parameter still conflicts."""
    from . import ligand
    d = tempfile.mkdtemp()
    lig = os.path.join(d, "L_atomtypes.itp")
    with open(lig, "w") as fh:
        fh.write("[ atomtypes ]\n"
                 " CG331 6 12.011 0.0 A 3.634867e-01 3.263520e-01\n")
    out = os.path.join(d, "out.itp")
    same = {"CG331": "CG331 6 12.011 0.0 A 0.36349 0.32635"}   # coarser format
    kept, dropped = ligand._dedup_atomtypes(lig, same, set(), out)
    assert dropped == ["CG331"] and kept == [], (dropped, kept)

    diff = {"CG331": "CG331 6 12.011 0.0 A 0.99 0.99"}         # real difference
    try:
        ligand._dedup_atomtypes(lig, diff, set(), out)
    except ValueError as exc:
        assert "CG331" in str(exc) and "different" in str(exc).lower(), exc
    else:
        raise AssertionError("a genuinely different atom type was not flagged")


@test
def endtoend_place_ligand_api_entry_builds_and_validates():
    """The flat api.place_ligand is the drivable surface for placement -- it
    must build from a real system + ligand and reject a missing required key
    with a message that names it."""
    from . import cgenff
    from .api import place_ligand
    d = tempfile.mkdtemp()
    sysd = _synthetic_protein_system(os.path.join(d, "system"))
    ligd = os.path.join(d, "lig")
    cgenff.generate_ligand_topology(_ETOH_STR, ligd)
    pdb = os.path.join(d, "ethanol.pdb")
    _ethanol_pdb(pdb)

    rep = place_ligand({"system_dir": sysd, "ligand_pdb": pdb,
                        "ligand_itp": os.path.join(ligd, "ETOH.itp"),
                        "output_dir": os.path.join(d, "out")})
    assert rep["ok"] and rep["atom_total"] == 34, rep

    try:
        place_ligand({"system_dir": sysd, "ligand_pdb": pdb,
                     "output_dir": os.path.join(d, "out2")})   # no ligand_itp
    except ValueError as exc:
        assert "ligand_itp" in str(exc), exc
    else:
        raise AssertionError("a missing required key was accepted")


@test
def endtoend_check_system_catches_topology_coordinate_mismatch():
    """check_system is the pre-grompp sanity pass: a good system validates, and
    a coordinate file that has lost atoms is caught as a count mismatch against
    [ molecules ] and the index -- the exact failure the package exists to
    prevent, reported before a cluster run."""
    from . import cgenff, ligand, api
    d = tempfile.mkdtemp()
    sysd = _synthetic_protein_system(os.path.join(d, "sys"))
    r0 = api.check_system({"system_dir": sysd})
    assert r0["ok"] and r0["atoms"] == 25, r0
    assert r0["index_checked"] == "SOLU+MEMB+SOLV" and r0["next_command"], r0

    ligd = os.path.join(d, "lig")
    cgenff.generate_ligand_topology(_ETOH_STR, ligd)
    pdb = os.path.join(d, "e.pdb")
    _ethanol_pdb(pdb)
    out = os.path.join(d, "out")
    ligand.add_ligand(sysd, pdb, os.path.join(ligd, "ETOH.itp"), out)
    assert api.check_system({"system_dir": out})["ok"], "placed system failed"

    # drop two atoms from the .gro; the mismatch must be caught, not slip through
    gro = os.path.join(out, "step5_input.gro")
    lines = open(gro).read().splitlines()
    lines[1] = str(int(lines[1]) - 2)
    del lines[2:4]
    with open(gro, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    bad = api.check_system({"system_dir": out})
    assert not bad["ok"], bad
    assert any("32" in e and "34" in e for e in bad["errors"]), bad["errors"]

    # a missing directory, and an unknown setting, are clean errors
    try:
        api.check_system({"system_dir": os.path.join(d, "nope")})
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a missing system dir was accepted")
    try:
        api.check_system({"system_dir": sysd, "junk": 1})
    except ValueError as exc:
        assert "unknown" in str(exc), exc
    else:
        raise AssertionError("an unknown setting was accepted")


@test
def endtoend_full_ligand_workflow():
    """The whole v2 journey in one test: a CGenFF stream -> a ligand topology ->
    placed into a protein+membrane system -> validated grompp-ready. If any seam
    between the three pieces drifts, this catches it, and it doubles as an
    executable statement of the workflow."""
    from . import cgenff, ligand, api
    d = tempfile.mkdtemp()

    # 1. topology from a CGenFF stream
    ligd = os.path.join(d, "lig")
    rep = cgenff.generate_ligand_topology(_ETOH_STR, ligd)
    assert rep["ok"] and rep["net_charge_is_integer"], rep

    # 2. place the positioned ligand into a built system
    sysd = _synthetic_protein_system(os.path.join(d, "sys"))
    pdb = os.path.join(d, "e.pdb")
    _ethanol_pdb(pdb)
    out = os.path.join(d, "out")
    place = ligand.add_ligand(sysd, pdb, os.path.join(ligd, "ETOH.itp"), out)
    assert place["ok"], place

    # 3. the result validates as grompp-shaped, and the counts agree end to end
    cs = api.check_system({"system_dir": out})
    assert cs["ok"] and not cs["errors"], cs
    assert cs["atoms"] == place["atom_total"], (cs["atoms"], place["atom_total"])
    assert cs["next_command"] and "grompp" in cs["next_command"], cs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", help="only run tests whose name contains this")
    ap.add_argument("-v", action="store_true", help="show tracebacks")
    args = ap.parse_args(argv)

    chosen = [t for t in TESTS if not args.k or args.k in t.__name__]
    failures, skipped = [], []
    t0 = time.time()
    for t in chosen:
        start = time.time()
        buf, old = io.StringIO(), sys.stdout
        try:
            sys.stdout = buf
            t()
            sys.stdout = old
            print("  pass  %-58s %5.2fs" % (t.__name__, time.time() - start))
        except SkipTest as exc:
            sys.stdout = old
            print("  skip  %-58s (%s)" % (t.__name__, exc))
            skipped.append(t.__name__)
        except Exception as exc:                        # noqa: BLE001
            sys.stdout = old
            print("  FAIL  %-58s %5.2fs" % (t.__name__, time.time() - start))
            failures.append((t.__name__, exc, traceback.format_exc()))
        finally:
            sys.stdout = old

    print("\n%d of %d passed%s in %.1fs"
          % (len(chosen) - len(failures) - len(skipped), len(chosen),
             (", %d skipped" % len(skipped)) if skipped else "",
             time.time() - t0))
    for name, exc, tb in failures:
        print("\n" + "=" * 72)
        print("FAIL %s\n  %s: %s" % (name, type(exc).__name__, exc))
        if args.v:
            print(tb)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
