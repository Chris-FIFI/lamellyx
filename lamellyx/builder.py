"""Putting a system together, end to end.

The order of molecules in the .gro must match `[ molecules ]` in topol.top,
because GROMACS assumes it does and will not tell you if it does not. Every
part of this module exists to make that ordering, and the counts that go with
it, impossible to get wrong by accident.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from . import (bilayer, fileio, geom, hbuild, mdp, orient, refine, solvate,
               topology, validate)


@dataclass
class BuildConfig:
    """Everything the builder needs. Distances in Angstrom."""

    # --- inputs ---
    protein_pdb: str = ""
    reference_dir: str = ""           # a GROMACS dir with toppar/ + step5_input.gro
    output_dir: str = ""

    # --- protein ---
    protein_molecules: tuple = ("PROA", "PROB", "PROC", "PROD")
    chains: tuple = ("A", "B", "C", "D")
    his_variant: str = "HSD"
    capped: bool = True
    # A PDB in the same frame as `protein_pdb` that the reference system was
    # built from. Superimposing it on the reference puts the new model into
    # exactly the reference's membrane frame.
    orient_from_pdb: str = ""
    # With no orientation reference, work the membrane frame out from the
    # protein's own hydrophobic belt and, for a homo-oligomer, its symmetry
    # axis. See orient.py.
    auto_orient: bool = False
    # (first, last) of a residue range known to be outside the cell. Nothing
    # geometric can tell the two faces of a slab apart, so without this the
    # protein may come out upside down.
    extracellular_resid: tuple = ()
    # Refuse to build through a protein that is not in the membrane frame.
    # Turn it off only when you know the model is oriented and you would
    # rather not pay for the check.
    require_oriented: bool = True

    # --- membrane ---
    lipid: str = "POPC"
    lipid_head: str = "P"
    n_upper: int = 0                  # 0 means take it from the reference
    n_lower: int = 0
    reuse_reference_bilayer: bool = False
    # A GROMACS directory whose composition this build must reproduce exactly:
    # same lipids per leaflet, same water, same ions. Two systems that differ
    # only in the conformation of the protein is the whole point of a
    # thermodynamic cycle, and it does not close if the counts drift. Water is
    # trimmed from bulk to hit the target, so the reference must not hold more
    # water than this box can fit.
    match_counts_from: str = ""
    # How close a lipid site may be placed to the protein footprint, in A.
    # 4.0 leaves an annular gap: measured against CHARMM-GUI on HCN4 it put
    # the median protein-atom-to-nearest-lipid at 8.89 A against 7.43, and
    # left 59% of transmembrane protein atoms with no lipid within 8 A
    # against 43%. Water then fills the gap. 2.5 reproduces CHARMM-GUI's
    # packing (7.42 A, 42.5%) at the cost of a slightly tighter worst contact
    # -- 2.19 A rather than 2.22 -- which the first minimisation removes.
    exclude_radius: float = 2.5

    # --- box and solvent ---
    box: tuple = ()                   # explicit (x, y, z); empty means compute
    # Lipid margin beyond the protein in x and y, so the box edge sits this
    # far from the transmembrane cross-section on every side. 20 A each side
    # of a 100 A-wide channel gives a 140 A box. Ignored if `box` is set.
    # Per axis, so X and Y can differ. 20 A = 2.0 nm on each side.
    margin_x: float = 20.0
    margin_y: float = 20.0
    # Area per lipid used to work out how many lipids fit in the free area
    # once the protein has taken its share. Measured POPC is 64.3 A^2.
    # Angstrom^2 here. 0 means "use the measured value", the same as on
    # the bilayer path and in the public API.
    area_per_lipid: float = 0.0
    water_thickness: float = 15.0
    concentration: float = 0.5
    cation: str = "POT"
    anion: str = "CLA"
    water: str = "TIP3"
    keep_pore_water: bool = True
    # How far inside the membrane slab a water must be from every lipid atom
    # to count as a pore water rather than as water in the acyl-chain region.
    pore_min_lipid: float = 10.0
    # A core water must have protein in at least this fraction of directions.
    # Neighbour count alone keeps water in grooves on the lipid-facing face.
    pore_enclosure: float = 0.90
    # ...and it must be within this far of the pore axis. The other three
    # tests all compare against the protein or the lipid locally, so a groove
    # on the outside of the protein can satisfy every one of them; only a
    # radius says "down the middle". CHARMM-GUI asks for the same thing as a
    # cylinder radius. 0 disables it.
    pore_radius: float = 10.0

    # --- refinement ---
    # Homology models arrive with side chains overlapping, usually across a
    # symmetry interface. Rotating them apart before packing is what stops the
    # bilayer being built around a defect.
    relieve_sidechain_clashes: bool = True
    clash_threshold: float = 2.45

    # --- run ---
    temperature: int = 310
    seed: int = 0
    copy_mdp_from_reference: bool = True
    verbose: bool = True


@dataclass
class BuildResult:
    atoms: object = None
    box: np.ndarray = None
    counts: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    files: list = field(default_factory=list)


def extract_protein_pdb(reference_dir, out_pdb,
                        protein_molecules=("PROA", "PROB", "PROC", "PROD"),
                        chains=("A", "B", "C", "D")):
    """Write a protein PDB out of a reference system, ready to rebuild from.

    Taking the protein out of a .gro by hand is the obvious way to get a
    starting structure and it does not work: .gro has no chain column, and it
    renumbers residues from 1 while the topology still expects the original
    numbering. Both failures surface late and look like something else.

    This writes the coordinates with the residue numbers, residue names and
    chain identifiers the topology expects, so the result is a valid input to
    `build`.
    """
    tops = topology.load_toppar(os.path.join(reference_dir, "toppar"))
    atoms, _ = fileio.read_gro(os.path.join(reference_dir, "step5_input.gro"))
    mols = [tops[m] for m in protein_molecules]
    if len(chains) != len(mols):
        raise ValueError("%d molecules but %d chains" % (len(mols), len(chains)))

    parts, off = [], 0
    for mol, ch in zip(mols, chains):
        n = mol.natoms
        if off + n > len(atoms):
            raise ValueError(
                "%s runs past the end of the .gro -- are protein_molecules "
                "listed in the order topol.top declares them?" % mol.name)
        parts.append(fileio.Atoms(
            list(mol.atomname), list(mol.resname), list(mol.resid),
            atoms.xyz[off:off + n],
            chain=[ch] * n, segid=[mol.name] * n))
        off += n
    protein = fileio.Atoms.concat(parts)
    fileio.write_pdb(out_pdb, protein,
                     title="protein from %s, numbered as its topology"
                           % os.path.basename(os.path.abspath(reference_dir)))
    return {"path": os.path.abspath(out_pdb), "atoms": len(protein),
            "chains": list(chains),
            "residue_range": [int(min(protein.resid)),
                              int(max(protein.resid))]}


def read_system_counts(gromacs_dir, lipid="POPC", cation="POT", anion="CLA",
                       water="TIP3", lipid_head="P"):
    """What a built system is made of: lipids per leaflet, water, ions.

    The leaflet split is not in topol.top -- it only has a total -- so it is
    counted from the coordinates, by which side of the bilayer midplane each
    head group sits on.
    """
    import re as _re

    top = open(os.path.join(gromacs_dir, "topol.top"),
               errors="ignore").read()
    counts = dict(_re.findall(r"(?m)^\s*(\w+)\s+(\d+)[ \t]*$", top))
    atoms, _box = fileio.read_gro(os.path.join(gromacs_dir,
                                               "step5_input.gro"))
    head = atoms.xyz[(atoms.resname == lipid) & (atoms.name == lipid_head)]
    if not len(head):
        raise ValueError("no %s %s atoms in %s: is that a %s system?"
                         % (lipid, lipid_head, gromacs_dir, lipid))
    mid = float(head[:, 2].mean())
    n_up = int((head[:, 2] > mid).sum())
    n_lo = int((head[:, 2] < mid).sum())
    out = {
        "n_upper": n_up,
        "n_lower": n_lo,
        "lipid": int(counts.get(lipid, n_up + n_lo)),
        "water": int(counts.get(water, 0)),
        "cation": int(counts.get(cation, 0)),
        "anion": int(counts.get(anion, 0)),
    }
    if out["n_upper"] + out["n_lower"] != out["lipid"]:
        raise ValueError(
            "%s: %d + %d head groups either side of the midplane but "
            "topol.top says %d lipids"
            % (gromacs_dir, n_up, n_lo, out["lipid"]))
    return out


def bilayer_measured_apl(lipid):
    """Measured area per lipid in nm^2, for error messages and defaults."""
    from .membrane import KNOWN_APL
    return KNOWN_APL.get(str(lipid).upper(), 0.643)


def _log(cfg, *a):
    if cfg.verbose:
        print(*a)


# --------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------

def frame_transform(source_pdb, mols, ref_xyz, chains, his_variant, capped):
    """Rigid transform putting `source_pdb` onto the reference protein.

    Returns (R, t, rmsd, n). The rmsd is the assertion that matters: if the
    reference system really was built from this model, it will be a fraction
    of an Angstrom. Anything larger means the two are not the same structure
    and the membrane frame is about to be wrong.
    """
    model = fileio.read_pdb(source_pdb)
    P, Q = [], []
    off = 0
    for mol, ref, ch in zip(mols, ref_xyz, chains):
        sub = model[model.chain == ch]
        idx, xyz, _ = hbuild.match_to_topology(
            mol, sub, his_variant=his_variant, capped=capped)
        P.append(xyz)
        Q.append(ref[idx])
        off += mol.natoms
    P, Q = np.vstack(P), np.vstack(Q)
    R, t = geom.kabsch(P, Q)
    return R, t, geom.rmsd(P @ R.T + t, Q), len(P)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build(cfg):
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed)
    res = BuildResult()

    # ---------------------------------------------------------------- inputs
    _log(cfg, "1. reading the reference system")
    tops = topology.load_toppar(os.path.join(cfg.reference_dir, "toppar"))
    ref_atoms, ref_box = fileio.read_gro(
        os.path.join(cfg.reference_dir, "step5_input.gro"))

    targets = None
    if cfg.match_counts_from:
        targets = read_system_counts(cfg.match_counts_from, cfg.lipid,
                                     cfg.cation, cfg.anion, cfg.water,
                                     cfg.lipid_head)
        cfg = replace(cfg, n_upper=targets["n_upper"],
                      n_lower=targets["n_lower"])
        _log(cfg, "   matching %s: %d/%d lipids, %d water, %d %s, %d %s"
             % (os.path.basename(os.path.abspath(cfg.match_counts_from)),
                targets["n_upper"], targets["n_lower"], targets["water"],
                targets["cation"], cfg.cation, targets["anion"], cfg.anion))

    mols = [tops[m] for m in cfg.protein_molecules]
    lipid_top = tops[cfg.lipid]
    water_top = tops[cfg.water]
    npro = sum(m.natoms for m in mols)
    ref_prot = [ref_atoms.xyz[s:s + m.natoms] for m, s in
                zip(mols, np.cumsum([0] + [m.natoms for m in mols])[:-1])]
    _log(cfg, "   %d protein atoms in %d chains, %s has %d atoms"
         % (npro, len(mols), cfg.lipid, lipid_top.natoms))

    lib = bilayer.extract_library(ref_atoms, ref_box, cfg.lipid,
                                  lipid_top.natoms, cfg.lipid_head)

    # ------------------------------------------------------------ protein
    _log(cfg, "2. placing the protein")
    model = fileio.read_pdb(cfg.protein_pdb)
    _log(cfg, "   %s: %d atoms, chains %s"
         % (os.path.basename(cfg.protein_pdb), len(model),
            "".join(sorted(set(model.chain)))))

    if cfg.orient_from_pdb:
        R, t, rms, n = frame_transform(cfg.orient_from_pdb, mols, ref_prot,
                                       cfg.chains, cfg.his_variant, cfg.capped)
        _log(cfg, "   membrane frame from %s: %d atoms, fit RMSD %.4f A"
             % (os.path.basename(cfg.orient_from_pdb), n, rms))
        if rms > 2.0:
            raise ValueError(
                "the orientation reference does not match the reference "
                "system (RMSD %.2f A) -- these are not the same structure" % rms)
        model.xyz = model.xyz @ R.T + t
        res.stats["orientation_fit_rmsd"] = rms

    elif cfg.auto_orient:
        model, orep = orient.orient(
            model, extracellular_resid=tuple(cfg.extracellular_resid) or None)
        res.stats["orientation"] = orep
        _log(cfg, "   oriented by %s: hydrophobic thickness %.1f A, "
                  "xy square %.1f A%s"
             % (orep["method"], orep["hydrophobic_thickness_A"],
                orep["xy_bounding_square_A"],
                ", flipped" if orep["flipped_in_z"] else ""))
        for key in ("warning", "warning_updown"):
            if orep.get(key):
                _log(cfg, "   WARNING: %s" % orep[key])

    elif cfg.require_oriented:
        # The old silent failure: with no reference and no auto-orientation,
        # the model was used exactly as deposited, and a PDB whose normal is
        # not z got a bilayer built through it sideways without a word.
        chk = orient.orientation_check(model)
        res.stats["orientation_check"] = chk
        if not chk["oriented"]:
            raise ValueError(
                "this protein is not in the membrane frame: its hydrophobic "
                "belt is %.0f degrees off z and centred at z = %.1f A. "
                "Building now would put the bilayer through it sideways. "
                "Set auto_orient=True to orient it here, or orient_from_pdb "
                "to copy the frame from a structure that is already right, "
                "or require_oriented=False if you are certain this is "
                "correct." % (chk["tilt_from_z_deg"], chk["core_centre_z_A"]))
        _log(cfg, "   already in the membrane frame: belt %.1f deg off z"
             % chk["tilt_from_z_deg"])

    _log(cfg, "3. rebuilding hydrogens and terminal caps")
    prot_xyz, reports = hbuild.rebuild_protein(
        mols, ref_prot, model, cfg.chains, his_variant=cfg.his_variant,
        capped=cfg.capped, verbose=cfg.verbose)
    off = 0
    for mol in mols:
        validate.check_bond_lengths(mol, prot_xyz[off:off + mol.natoms])
        off += mol.natoms
    res.stats["protein_atoms"] = int(len(prot_xyz))

    if cfg.relieve_sidechain_clashes:
        _log(cfg, "3b. relieving side-chain clashes")
        # The final box is not known yet, so refine in a snug non-periodic one.
        # The protein does not span a boundary, so no image contacts are lost.
        pad, lo = 10.0, prot_xyz.min(0)
        snug = prot_xyz.max(0) - lo + 2.0 * pad
        fixed, rrep = refine.relieve_clashes(
            prot_xyz - lo + pad, mols, snug, pbc=(False, False, False),
            clash=cfg.clash_threshold, verbose=cfg.verbose)
        prot_xyz = fixed + lo - pad
        off = 0
        for mol in mols:
            validate.check_bond_lengths(mol, prot_xyz[off:off + mol.natoms])
            off += mol.natoms
        _log(cfg, "   %d pairs under %.2f A -> %d ; closest %.2f -> %.2f A ; "
                  "%d side chains rotated, backbone unmoved, RMSD %.3f A"
             % (rrep.n_clashes_before, cfg.clash_threshold, rrep.n_clashes_after,
                rrep.worst_before, rrep.worst_after, len(rrep.rotations),
                rrep.rmsd))
        res.stats["sidechain_refine"] = {
            "clashes_before": rrep.n_clashes_before,
            "clashes_after": rrep.n_clashes_after,
            "closest_before": rrep.worst_before,
            "closest_after": rrep.worst_after,
            "rotations": rrep.rotations,
            "rmsd": rrep.rmsd,
        }

    prot_names = np.concatenate([m.atomname for m in mols])
    prot_resn = np.concatenate([m.resname for m in mols])
    prot_resi = np.concatenate([m.resid for m in mols])
    prot_seg = np.concatenate([[m.name] * m.natoms for m in mols])
    protein = fileio.Atoms(prot_names, prot_resn, prot_resi, prot_xyz,
                           segid=prot_seg)
    prot_heavy = protein.xyz[protein.element != "H"]

    # ---------------------------------------------------------------- box
    # Fail here rather than a minute later inside the packer, where the only
    # symptom is "could only find N sites for M lipids".
    if not cfg.area_per_lipid:
        # 0 means measured. Resolved here as well as in api.py so the
        # low-level BuildConfig behaves the same as the public call rather
        # than falling through to the < 40 A^2 check below.
        cfg = replace(cfg, area_per_lipid=bilayer_measured_apl(cfg.lipid) * 100.0)
    if cfg.area_per_lipid < 40.0:
        raise ValueError(
            "area_per_lipid = %.1f A2 (%.3f nm2) is below the ~40 A2 two "
            "extended acyl chains occupy, so the lipids cannot be placed "
            "side by side. Measured %s is %.1f A2 (%.3f nm2)."
            % (cfg.area_per_lipid, cfg.area_per_lipid / 100.0, cfg.lipid,
               bilayer_measured_apl(cfg.lipid) * 100.0,
               bilayer_measured_apl(cfg.lipid)))

    slab = (lib.head_z_lower - 2.0, lib.head_z_upper + 2.0)
    if cfg.box:
        box = np.array(cfg.box, dtype=float)
        _log(cfg, "4. box fixed by configuration: %.4f x %.4f x %.4f A"
             % tuple(box))
    else:
        # Size x and y from the protein's transmembrane cross-section plus a
        # margin on each side, then centre the protein in it.
        lo, hi = bilayer.slab_extent(prot_heavy, *slab)
        width = hi - lo
        box = ref_box.copy()
        box[0] = float(width[0] + 2.0 * cfg.margin_x)
        box[1] = float(width[1] + 2.0 * cfg.margin_y)
        centre = 0.5 * (lo + hi)
        protein.xyz[:, 0] += box[0] / 2.0 - centre[0]
        protein.xyz[:, 1] += box[1] / 2.0 - centre[1]

        height, shift = solvate.size_box_z(protein.xyz, cfg.water_thickness)
        box[2] = height
        protein.xyz[:, 2] += shift
        lib.head_z_upper += shift
        lib.head_z_lower += shift
        lib.midplane += shift
        slab = (lib.head_z_lower - 2.0, lib.head_z_upper + 2.0)
        prot_heavy = protein.xyz[protein.element != "H"]
        _log(cfg, "4. protein spans %.1f x %.1f A in the membrane; with "
                  "%.1f / %.1f A of lipid each side and %.1f A of water:"
             % (width[0], width[1], cfg.margin_x, cfg.margin_y,
                cfg.water_thickness))
        _log(cfg, "   box %.4f x %.4f x %.4f A (%.2f x %.2f x %.2f nm)"
             % (box[0], box[1], box[2], box[0] / 10, box[1] / 10, box[2] / 10))
    res.box = box

    # ---- how many lipids fit -------------------------------------------
    # With the box sized from the protein there is no reference count to
    # copy: work out the free area in each leaflet and divide by the area
    # per lipid, which is what the reference build did too.
    slab_up = (lib.midplane - 3.0, lib.head_z_upper + 6.0)
    slab_lo = (lib.head_z_lower - 6.0, lib.midplane + 3.0)
    if cfg.box:
        n_upper = cfg.n_upper or len(lib.upper)
        n_lower = cfg.n_lower or len(lib.lower)
        area_up = area_lo = None
    else:
        fit_up, area_up = bilayer.lipids_for_free_area(
            box[:2], prot_heavy, slab_up, cfg.area_per_lipid)
        fit_lo, area_lo = bilayer.lipids_for_free_area(
            box[:2], prot_heavy, slab_lo, cfg.area_per_lipid)
        n_upper = cfg.n_upper or fit_up
        n_lower = cfg.n_lower or fit_lo
        _log(cfg, "   protein occupies %.0f / %.0f A^2 upper and %.0f lower; "
                  "at %.1f A^2 per lipid that leaves room for %d / %d"
             % (area_up, box[0] * box[1], area_lo, cfg.area_per_lipid,
                fit_up, fit_lo))
    res.stats["protein_area_A2"] = area_up
    res.stats["n_upper"] = int(n_upper)
    res.stats["n_lower"] = int(n_lower)

    zmin, zmax = protein.xyz[:, 2].min(), protein.xyz[:, 2].max()
    if zmin < 0 or zmax > box[2]:
        _log(cfg, "   WARNING protein spans z %.1f..%.1f in a box of %.1f"
             % (zmin, zmax, box[2]))
    _log(cfg, "   protein z %.2f .. %.2f, water layer %.2f / %.2f A"
         % (zmin, zmax, zmin, box[2] - zmax))

    # ------------------------------------------------------------ membrane
    if cfg.reuse_reference_bilayer:
        _log(cfg, "5. reusing the reference bilayer as it stands")
        lip_xyz = bilayer.from_reference(ref_atoms, ref_box, cfg.lipid,
                                         lipid_top.natoms)
        leaflet = np.where(lip_xyz[:, lib.head, 2] > lib.midplane, 1, -1)
        bstats = {"n_lipids": len(lip_xyz), "reused": True,
                  "n_upper": int((leaflet > 0).sum()),
                  "n_lower": int((leaflet < 0).sum())}
        rel = bilayer.RigidLipidRelaxer(lip_xyz, lib.heavy_mask, prot_heavy, box)
        w0, _ = rel.worst_contact()
        _log(cfg, "   %d lipids; closest contact to the new protein %.2f A"
             % (len(lip_xyz), w0))
        if w0 < bilayer.CLASH:
            _log(cfg, "   relaxing the reused bilayer against the new protein")
            rel.run(120, 1.0, 1.0, eta=0.25, verbose=cfg.verbose, label="reuse")
            lip_xyz = bilayer.wrap_molecules(rel.x, box)
            w1, nc = rel.worst_contact()
            _log(cfg, "   after relaxation: closest %.2f A (%d under %.1f A)"
                 % (w1, nc, bilayer.CLASH))
            bstats["closest_final"] = w1
            bstats["n_clashes_final"] = nc
        else:
            bstats["closest_final"] = w0
            bstats["n_clashes_final"] = 0
    else:
        _log(cfg, "5. packing a new bilayer: %d upper, %d lower"
             % (n_upper, n_lower))
        bres = bilayer.build_bilayer(
            lib, n_upper, n_lower, box, prot_heavy, rng,
            exclude_radius=cfg.exclude_radius, verbose=cfg.verbose)
        lip_xyz, leaflet, bstats = bres.xyz, bres.leaflet, bres.stats
    res.stats["bilayer"] = bstats

    lipids = fileio.Atoms(
        np.tile(lib.atomnames, len(lip_xyz)),
        np.full(len(lip_xyz) * lipid_top.natoms, cfg.lipid),
        np.repeat(np.arange(1, len(lip_xyz) + 1), lipid_top.natoms),
        lip_xyz.reshape(-1, 3),
        segid=np.full(len(lip_xyz) * lipid_top.natoms, "MEMB"))

    # ------------------------------------------------------------- solvent
    _log(cfg, "6. solvating")
    solute = fileio.Atoms.concat([protein, lipids])
    template = solvate.extract_water_template(
        ref_atoms, ref_box, cfg.water, water_top.natoms, rng=rng)
    _log(cfg, "   water template: %d molecules in a %.0f A cube"
         % (len(template.xyz), template.cell))

    core = (lib.head_z_lower + 2.0, lib.head_z_upper - 2.0)
    wat = solvate.solvate(
        solute.xyz, solute.element != "H", box, template, rng,
        core_z=core, keep_pore_water=cfg.keep_pore_water,
        lipid_xyz=lipids.xyz[lipids.element != "H"],
        protein_xyz=protein.xyz[protein.element != "H"],
        pore_min_lipid=cfg.pore_min_lipid,
        pore_enclosure=cfg.pore_enclosure, pore_radius=cfg.pore_radius,
        pore_axis_xy=prot_heavy[:, :2].mean(0), verbose=cfg.verbose)

    # The bilayer-depth water is reported by validate.core_water_report once
    # the system is assembled, further down.

    if targets is not None:
        # Ions are placed by replacing water, so the count to hit here is the
        # target plus however many ions are about to take a water's place.
        # Trimming straight to the target lands exactly n_ion molecules short.
        want = targets["water"] + targets["cation"] + targets["anion"]
        if len(wat) < want:
            raise ValueError(
                "matching %s needs %d waters but only %d fit in this box. "
                "Water is only ever trimmed, never invented -- the reference "
                "must not be more solvated than this system can hold."
                % (cfg.match_counts_from, want, len(wat)))
        if len(wat) > want:
            # Trim from bulk. Interfacial and pore water is structure; the
            # molecules to give up are the ones furthest from everything.
            o = wat[:, 0, :]
            d = geom.CellGrid(solute.xyz[solute.element != "H"], box,
                              8.0).min_distance(o)
            bulk = np.flatnonzero(d > 6.0)
            if len(bulk) < len(wat) - want:
                raise ValueError(
                    "need to remove %d waters but only %d are in bulk"
                    % (len(wat) - want, len(bulk)))
            drop = rng.permutation(bulk)[:len(wat) - want]
            keep = np.ones(len(wat), dtype=bool)
            keep[drop] = False
            _log(cfg, "   trimmed %d bulk waters to match: %d -> %d "
                      "(%d + %d ions that will replace water)"
                 % (len(drop), len(wat), want, targets["water"],
                    targets["cation"] + targets["anion"]))
            wat = wat[keep]
        assert len(wat) == want, (len(wat), want)

    q_protein = sum(m.total_charge for m in mols)
    q_lipid = lipid_top.total_charge * len(lip_xyz)
    q = q_protein + q_lipid
    if targets is not None:
        n_cat, n_ani = targets["cation"], targets["anion"]
        # q is a float sum over per-molecule charges, so it lands a few times
        # 1e-16 off an integer. Comparing it to exactly zero rejects a
        # perfectly neutral system.
        residual = n_cat - n_ani + q
        if abs(residual) > 1e-6:
            raise ValueError(
                "%s has %d %s and %d %s, leaving a net charge of %+.4f on a "
                "system whose protein and lipid come to %+.4f -- the two "
                "systems do not hold the same molecule"
                % (cfg.match_counts_from, n_cat, cfg.cation, n_ani,
                   cfg.anion, residual, q))
    else:
        n_cat, n_ani = solvate.ion_counts(len(wat), q, cfg.concentration)
    _log(cfg, "   system charge before ions: %+.2f (protein %+.2f, lipid %+.2f)"
         % (q, q_protein, q_lipid))
    _log(cfg, "   %.3f M in %d waters -> %d %s, %d %s"
         % (cfg.concentration, len(wat), n_cat, cfg.cation, n_ani, cfg.anion))

    wat, cat_xyz, ani_xyz = solvate.place_ions(
        wat, box, solute.xyz, n_cat, n_ani, rng, exclude_z=core,
        verbose=cfg.verbose)

    def _ion(name, xyz):
        return fileio.Atoms([name] * len(xyz), [name] * len(xyz),
                            np.arange(1, len(xyz) + 1), xyz,
                            segid=["IONS"] * len(xyz))

    cations, anions = _ion(cfg.cation, cat_xyz), _ion(cfg.anion, ani_xyz)
    water = fileio.Atoms(
        np.tile(template.atomnames, len(wat)),
        np.full(len(wat) * water_top.natoms, cfg.water),
        np.repeat(np.arange(1, len(wat) + 1), water_top.natoms),
        wat.reshape(-1, 3),
        segid=np.full(len(wat) * water_top.natoms, "SOLV"))

    # ------------------------------------------------------------- assemble
    _log(cfg, "7. assembling")
    system = fileio.Atoms.concat([protein, lipids, cations, anions, water])
    blocks = ([(m, 1) for m in mols]
              + [(lipid_top, len(lip_xyz)), (tops[cfg.cation], len(cat_xyz)),
                 (tops[cfg.anion], len(ani_xyz)), (water_top, len(wat))])
    system.xyz = geom.wrap_by_blocks(system.xyz, box, blocks, (True, True, False))
    system = fileio.renumber_by_residue(system)
    res.atoms = system
    expect = sum(m.natoms * c for m, c in blocks)
    if expect != len(system):
        raise AssertionError(
            "topology says %d atoms, coordinates have %d" % (expect, len(system)))

    counts = {"PROTEIN": npro, cfg.lipid: len(lip_xyz), cfg.cation: len(cat_xyz),
              cfg.anion: len(ani_xyz), cfg.water: len(wat),
              "TOTAL_ATOMS": len(system)}
    res.counts = counts
    net = (q + len(cat_xyz) * tops[cfg.cation].total_charge
           + len(ani_xyz) * tops[cfg.anion].total_charge)
    res.stats["net_charge"] = float(net)
    if abs(net) > 1e-6:
        raise AssertionError("system is not neutral: net charge %+.3f" % net)

    res.files = write_output(cfg, res, blocks, mols, lipid_top, water_top,
                             len(cat_xyz), len(ani_xyz), len(wat), len(lip_xyz))

    # -------------------------------------------------------------- checks
    _log(cfg, "8. checking")
    groups = {
        "protein": np.arange(len(system)) < npro,
        "lipid": (np.arange(len(system)) >= npro) &
                 (np.arange(len(system)) < npro + len(lipids)),
    }
    groups["solvent"] = ~(groups["protein"] | groups["lipid"])
    excl = validate.system_exclusions(blocks, len(system))
    rep = validate.contacts(system, box, excl, cutoff=2.6, groups=groups)
    _log(cfg, rep.format())
    _log(cfg, "   density                : %.3f g/cm3"
         % validate.density(system, box))
    res.stats["contacts"] = {
        "min_distance": rep.min_distance, "heavy_min": rep.heavy_min,
        "heavy_below_2.4": rep.heavy_below,
        "n_below": rep.n_below, "by_group": rep.by_group}
    res.stats["density"] = validate.density(system, box)
    cw = validate.core_water_report(system, box, groups["protein"],
                                    groups["lipid"], core,
                                    pore_radius=cfg.pore_radius)
    res.stats["core_water"] = cw
    _log(cfg, "   water between the phosphate planes: %d (nearest lipid "
              "%.1f A, least enclosed %.2f, furthest %.1f A off-axis, "
              "%d beyond the %.0f A pore radius)"
         % (cw["count"], cw.get("min_lipid_distance_A", float("nan")),
            cw.get("min_enclosure", float("nan")),
            cw.get("max_radius_A", float("nan")),
            cw.get("beyond_pore_radius", 0), cfg.pore_radius))
    if cw.get("in_lipid_within_6A"):
        _log(cfg, "   WARNING %d of them are within 6 A of a lipid -- that is "
                  "water in the acyl-chain region" % cw["in_lipid_within_6A"])
    res.stats["seconds"] = time.time() - t_start
    _log(cfg, "\nbuilt %d atoms in %.0f s -> %s"
         % (len(system), res.stats["seconds"], cfg.output_dir))
    return res


def write_output(cfg, res, blocks, mols, lipid_top, water_top,
                 n_cat, n_ani, n_wat, n_lip):
    out = cfg.output_dir
    os.makedirs(out, exist_ok=True)
    files = []

    used = [m.name for m in mols] + [cfg.lipid, cfg.cation, cfg.anion, cfg.water]
    topology.copy_toppar(os.path.join(cfg.reference_dir, "toppar"),
                         os.path.join(out, "toppar"), only=used)
    includes = ["%s.itp" % m.name for m in mols] + [
        "%s.itp" % n for n in (cfg.lipid, cfg.cation, cfg.anion, cfg.water)]
    molecules = ([(m.name, 1) for m in mols]
                 + [(cfg.lipid, n_lip), (cfg.cation, n_cat),
                    (cfg.anion, n_ani), (cfg.water, n_wat)])
    p = os.path.join(out, "topol.top")
    topology.write_topol(p, includes, molecules,
                         system_name=os.path.basename(cfg.output_dir))
    files.append(p)

    p = os.path.join(out, "step5_input.gro")
    fileio.write_gro(p, res.atoms, res.box,
                     title="lamellyx: %s" % os.path.basename(cfg.protein_pdb))
    files.append(p)

    npro = sum(m.natoms for m in mols)
    nlip = n_lip * lipid_top.natoms
    n = len(res.atoms)
    p = os.path.join(out, "index.ndx")
    topology.write_index(p, topology.standard_groups(
        n, slice(0, npro), slice(npro, npro + nlip), slice(npro + nlip, n)))
    files.append(p)

    if cfg.copy_mdp_from_reference:
        copied = mdp.copy_from(cfg.reference_dir, out)
        if len(copied) < len(mdp.MDP_FILES):
            mdp.write_series(out, cfg.temperature)
    else:
        mdp.write_series(out, cfg.temperature)
    files += [os.path.join(out, f) for f in mdp.MDP_FILES]

    p = os.path.join(out, "build_report.json")
    with open(p, "w") as fh:
        json.dump({"config": {k: (list(v) if isinstance(v, tuple) else v)
                              for k, v in asdict(cfg).items()},
                   "counts": res.counts,
                   "box_angstrom": list(map(float, res.box)),
                   "stats": _jsonable(res.stats)}, fh, indent=2)
    files.append(p)
    return files


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
