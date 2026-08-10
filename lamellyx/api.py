"""A flat, machine-readable interface.

Everything here takes and returns plain JSON-compatible values: no objects to
construct, no attributes to discover, no state between calls. That is what
makes it drivable by a script, a language model, or anything else that can
send a dict and read one back -- which is the whole reason this exists rather
than a web form.

    from lamellyx.api import describe, build, check

    describe()                       # every setting, its units and its range
    check({"x": 2.0})                # what is wrong, before spending a minute
    build({"output_dir": "box"})     # counts, box, quality, file paths

The same three are available from the command line as `schema`, `check` and
`build`, all of which speak JSON on stdout.
"""

from __future__ import annotations

import os
from dataclasses import asdict, fields

from . import membrane
from .library import available_lipids
from .membrane import (CHARMM_CUTOFF, KNOWN_APL, MembraneConfig,
                       build_membrane, check_settings)

# Units and meaning for every setting, so a caller never has to guess whether
# a length is nanometres or Angstrom. This is the contract.
UNITS = {
    "x": "nm (full box edge, not a half-width)",
    "y": "nm (full box edge, not a half-width)",
    "area_per_lipid": "nm^2 per lipid; 0 uses the measured value",
    "water_thickness": "nm of water above and below the bilayer",
    "salt_concentration": "mol/L",
    "temperature": "K",
    "n_upper": "lipids in the upper leaflet; the box follows from it in "
               "size_mode='lipid_numbers'",
    "n_lower": "lipids in the lower leaflet",
    "box": "nm (x, y, z); empty means size it from the protein and margin",
    "margin_x": "nm of lipid beyond the protein in X, on each side",
    "margin_y": "nm of lipid beyond the protein in Y, on each side",
}

NOTES = {
    "x": "Used only in size_mode='box'. At least %.1f nm: CHARMM36 uses "
         "%.1f nm cutoffs and a periodic box must be twice its cutoff."
         % (2 * CHARMM_CUTOFF, CHARMM_CUTOFF),
    "y": "Same minimum as x.",
    "size_mode": "How X and Y are decided, mirroring CHARMM-GUI: "
                 "'lipid_numbers' (counts in, box out -- the default), "
                 "'box' (x and y in, counts out), or 'protein_margin'.",
    "area_per_lipid": "0 (the default) uses the measured value -- 0.643 nm2 "
                      "for POPC, which is what CHARMM-GUI does. Anything "
                      "below 0.40 nm2 is refused: two extended acyl chains "
                      "do not fit.",
    "leaflet": "'bilayer' or 'monolayer'.",
    "toppar": "Directory of GROMACS .itp files. Empty uses the bundled "
              "CHARMM36 set, which covers POPC, TIP3P, K+ and Cl- only.",
    "toppar_mode": "'copy' makes each build self-contained; 'reference' "
                   "points at a shared directory and saves space.",
    "strict": "False builds even when the settings cannot run in GROMACS.",
    "auto_orient": "Work the membrane frame out from the protein itself -- "
                   "its hydrophobic belt, plus the symmetry axis when it is a "
                   "homo-oligomer. Use this when there is nothing already "
                   "oriented to copy from. Pass extracellular_resid too, or "
                   "the protein may come out upside down.",
    "extracellular_resid": "[first, last] of a residue range known to be "
                           "OUTSIDE the cell -- for HCN4, the turret "
                           "[443, 460]. Nothing geometric distinguishes the "
                           "two faces of a slab, so without this the up/down "
                           "choice is an arbitrary tie-break.",
    "require_oriented": "True (the default) refuses to build through a "
                        "protein whose hydrophobic belt is not on z. This "
                        "used to build silently and produce a bilayer lying "
                        "across the protein.",
    "orient_from_pdb": "A PDB in the same frame as protein_pdb that the "
                       "reference system was built from. The most exact "
                       "option when you have one: it copies a frame rather "
                       "than deriving one.",
}


def describe():
    """Every setting: default, type, units, and what it means.

    Enough to construct a valid request without reading the source.
    """
    out = {}
    for f in fields(MembraneConfig):
        default = getattr(MembraneConfig(), f.name)
        out[f.name] = {
            "default": list(default) if isinstance(default, tuple) else default,
            "type": type(default).__name__,
            "units": UNITS.get(f.name, ""),
            "note": NOTES.get(f.name, ""),
        }
    return {
        "version": __import__("lamellyx").__version__,
        "settings": out,
        "protein_settings": _describe_protein(),
        "lipids_available": available_lipids(),
        "measured_area_per_lipid_nm2": KNOWN_APL,
        "outputs": [
            "step5_input.gro", "topol.top", "index.ndx",
            "step6.0_minimization.mdp", "step6.1_equilibration.mdp",
            "step6.2_equilibration.mdp", "step6.3_equilibration.mdp",
            "step6.4_equilibration.mdp", "step6.5_equilibration.mdp",
            "step6.6_equilibration.mdp", "step7_production.mdp",
            "toppar/", "build_report.json",
        ],
        "next_command": (
            "gmx grompp -f step6.0_minimization.mdp -c step5_input.gro "
            "-r step5_input.gro -p topol.top -n index.ndx -o min.tpr"),
    }


def _describe_protein():
    """Settings for `build_protein_system`, in the units it actually takes.

    This exists because reporting only the bilayer schema was actively
    misleading: the two configs share field names, and the lower-level
    protein class works in Angstrom. An agent told "nm" that passed
    water_thickness=15 would have asked for 150 A of water.
    """
    from .builder import BuildConfig

    base = BuildConfig()
    out = {}
    for f in fields(BuildConfig):
        default = getattr(base, f.name)
        if f.name in _PROTEIN_SCALED and isinstance(default, (int, float)):
            default = round(float(default) / _PROTEIN_SCALED[f.name], 6)
        out[f.name] = {
            "default": list(default) if isinstance(default, tuple) else default,
            "type": type(default).__name__,
            "units": UNITS.get(f.name, ""),
            "note": NOTES.get(f.name, ""),
        }
    out["margin_x"]["note"] = ("Membrane between the protein and the box edge "
                               "on each side. The box comes out as the "
                               "protein's transmembrane width + 2 x margin.")
    out["margin_y"]["note"] = out["margin_x"]["note"]
    out["_units"] = ("NANOMETRES and mol/L, the same as build(). The "
                     "underlying builder.BuildConfig works in Angstrom; "
                     "build_protein_system converts.")
    out["_required"] = ["protein_pdb", "reference_dir", "output_dir"]
    out["_protein_pdb_must"] = [
        "have residue numbers matching the .itp topology (not renumbered "
        "from 1)",
        "have a chain identifier in column 22",
        "contain the same residues as the topology; hydrogens and terminal "
        "caps are rebuilt for you",
    ]
    return out


def _config(settings):
    known = {f.name for f in fields(MembraneConfig)}
    unknown = sorted(set(settings) - known - {"protein_pdb"})
    if unknown:
        raise ValueError(
            "unknown settings: %s. Call describe() for the full list."
            % ", ".join(unknown))
    return MembraneConfig(**{k: v for k, v in settings.items() if k in known})


def check(settings):
    """Validate without building. Returns {ok, errors, warnings}.

    Cheap, and worth calling first: it catches a box GROMACS would refuse
    before a minute is spent packing one.
    """
    settings = dict(settings or {})
    issues = check_settings(
        _config(settings),
        protein=bool(settings.get("protein_pdb")),
        xy_margin=settings.get("margin_x"))
    errors = [m for lvl, m in issues if lvl == "error"]
    warnings = [m for lvl, m in issues if lvl == "warning"]
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def build(settings):
    """Build a system. Returns a JSON-compatible summary.

    On a refused configuration this raises ValueError with every reason in
    the message, rather than producing something that cannot run.
    """
    cfg = _config(dict(settings or {}))
    res = build_membrane(cfg)
    return {
        "ok": True,
        "output_dir": os.path.abspath(cfg.output_dir),
        "counts": res.counts,
        "box_nm": [round(float(v), 4) for v in res.box_nm],
        "area_per_lipid_nm2": round(res.stats["area_per_lipid_nm2"], 4),
        "bilayer_thickness_nm": round(res.stats["bilayer_thickness_nm"], 3),
        "density_g_cm3": round(res.stats["density_g_cm3"], 4),
        # Rounded so a caller can test it against 0. Summing thousands of
        # partial charges leaves about 1e-14 of noise, and an agent checking
        # `net_charge == 0` would otherwise conclude the system is charged.
        "net_charge": round(float(res.stats["net_charge"]), 6) + 0.0,
        "closest_heavy_atom_contact_A": round(
            res.stats["contacts"]["heavy_min"], 3),
        "heavy_pairs_below_2.4A": res.stats["contacts"]["heavy_below_2.4"],
        "seconds": round(res.stats["seconds"], 1),
        "bytes_on_disk": res.stats.get("output_bytes"),
        "warnings": res.warnings,
        "files": sorted(os.listdir(cfg.output_dir)),
        "next_command": (
            "cd %s && gmx grompp -f step6.0_minimization.mdp "
            "-c step5_input.gro -r step5_input.gro -p topol.top "
            "-n index.ndx -o min.tpr" % os.path.abspath(cfg.output_dir)),
    }


# Settings whose value is a length or an area, and the factor that turns the
# public nanometre value into the Angstrom that BuildConfig works in.
_PROTEIN_SCALED = {"margin_x": 10.0, "margin_y": 10.0, "water_thickness": 10.0,
                   "area_per_lipid": 100.0, "clash_threshold": 1.0,
                   "pore_min_lipid": 1.0, "exclude_radius": 1.0,
                   "pore_radius": 10.0}
# The protein path grew its own names for two settings. The public API uses
# one set of names, and the old ones are still accepted.
_PROTEIN_ALIASES = {"salt_concentration": "concentration",
                    "water_model": "water"}


def _check_lengths_are_nanometres(given, scaled):
    """Catch a nanometre setting given in Angstrom, before it costs a minute.

    Everything public is nanometres, so a value ten times too large is almost
    always a unit mistake. Left alone it does not fail here -- it fails a
    minute later inside the packer as
    `MemoryError: cell table would be 15000272x25`, which names neither the
    setting nor the box.
    """
    LIMITS = {                      # (nm) plausible range for the public value
        "box": (2.4, 50.0),
        "margin_x": (0.0, 10.0), "margin_y": (0.0, 10.0),
        "water_thickness": (0.0, 10.0),
        "area_per_lipid": (0.0, 2.0),
        "pore_radius": (0.0, 5.0),
    }
    for key, (lo, hi) in LIMITS.items():
        if key not in given:
            continue
        v = given[key]
        vals = list(v) if isinstance(v, (list, tuple)) else [v]
        if not vals or not all(isinstance(c, (int, float)) for c in vals):
            continue
        big = [float(c) for c in vals if float(c) > hi]
        if not big:
            continue
        raise ValueError(
            "%s = %s looks like Angstrom, not nanometres. This path takes "
            "nanometres: %s of about %.3g would be %s. If the value really is "
            "meant, pass it in nm (divide by 10)."
            % (key, vals[0] if len(vals) == 1 else tuple(vals), key,
               big[0] / 10.0, "%.3g nm" % (big[0] / 10.0)))
        # a box below the cutoff limit is caught by check(); nothing to do here


def extract_protein(reference_dir, out_pdb, protein_molecules=None,
                    chains=None):
    """Write a topology-matching protein PDB out of a reference system.

    The obvious way to get a starting structure -- pull the protein out of
    the reference .gro -- produces a file the builder cannot use: .gro has no
    chain column and renumbers residues from 1. This does it correctly.
    """
    from .builder import extract_protein_pdb
    kw = {}
    if protein_molecules:
        kw["protein_molecules"] = tuple(protein_molecules)
    if chains:
        kw["chains"] = tuple(chains)
    return extract_protein_pdb(reference_dir, out_pdb, **kw)


def generate_protein_topology(settings):
    """Make the protein's .itp files by running GROMACS' pdb2gmx.

        {"protein_pdb": "hcn4.pdb", "output_dir": "topology_out",
         "forcefield": "charmm36-jul2022"}

    This is the step that used to require CHARMM-GUI. It still does not
    invent force-field parameters -- it drives the tool whose job that is --
    so GROMACS must be installed. It is not bundled, and on a machine without
    it this raises rather than failing later and obscurely.
    """
    from . import pdb2gmx as _p2g

    settings = dict(settings or {})
    for need in ("protein_pdb", "output_dir"):
        if not settings.get(need):
            raise ValueError("%s is required" % need)
    accepted = {"protein_pdb", "output_dir", "forcefield", "water", "gmx",
                "chains", "ignh", "extra_args", "posre"}
    unknown = sorted(set(settings) - accepted)
    if unknown:
        raise ValueError("unknown settings: %s. Accepted: %s"
                         % (", ".join(unknown), ", ".join(sorted(accepted))))
    kw = {k: v for k, v in settings.items()
          if k not in ("protein_pdb", "output_dir")}
    return _p2g.generate_topology(settings["protein_pdb"],
                                  settings["output_dir"], **kw)


def orient_protein(settings):
    """Put a protein into the membrane frame and write it out.

    Useful on its own -- the answer is a PDB you can look at before committing
    to a build -- and it is what `auto_orient` runs internally.

        {"protein_pdb": "in.pdb", "output_pdb": "oriented.pdb",
         "extracellular_resid": [443, 460]}

    Returns the report: the normal it found, the hydrophobic thickness, how
    the two independent estimates compared, and whether it flipped the protein.
    """
    from . import orient as _orient

    settings = dict(settings or {})
    for need in ("protein_pdb", "output_pdb"):
        if not settings.get(need):
            raise ValueError("%s is required" % need)

    accepted = {"protein_pdb", "output_pdb", "extracellular_resid",
                "use_symmetry", "n_directions", "thickness", "n_points"}
    unknown = sorted(set(settings) - accepted)
    if unknown:
        raise ValueError("unknown settings: %s. Accepted: %s"
                         % (", ".join(unknown), ", ".join(sorted(accepted))))

    kw = {k: v for k, v in settings.items()
          if k not in ("protein_pdb", "output_pdb")}
    if kw.get("extracellular_resid"):
        kw["extracellular_resid"] = tuple(kw["extracellular_resid"])
    if kw.get("thickness"):
        # nanometres out here, like every other length in this API
        kw["thickness"] = tuple(float(v) * 10.0 for v in kw["thickness"])

    report = _orient.orient_pdb(settings["protein_pdb"],
                                settings["output_pdb"], **kw)
    report["ok"] = True
    return report


def build_protein_system(settings):
    """Embed a protein in a bilayer. Returns the same shape as `build`.

    Takes the same units and the same names as `build`: NANOMETRES, and
    `salt_concentration` rather than `concentration`. The lower-level
    `builder.BuildConfig` works in Angstrom, and the conversion happens here.

    Needs `protein_pdb` and `reference_dir` -- a directory holding that
    protein's own .itp files. Force-field parameters are never generated
    here, so a sequence has to go through CHARMM-GUI or pdb2gmx once; after
    that any conformation of it can be built from a script.
    """
    from .builder import BuildConfig, build as _build_protein

    settings = dict(settings or {})
    for need in ("protein_pdb", "reference_dir", "output_dir"):
        if not settings.get(need):
            raise ValueError("%s is required to embed a protein" % need)

    known = {f.name for f in fields(BuildConfig)}
    accepted = known | set(_PROTEIN_ALIASES) | {"lipid"}
    unknown = sorted(set(settings) - accepted)
    if unknown:
        # Dropping these silently is how you ask for 0.15 M and get 0.5 M.
        raise ValueError(
            "unknown settings for a protein system: %s. Call describe() for "
            "the full list; note this path uses %s."
            % (", ".join(unknown),
               " and ".join("%s (not %s)" % (v, k)
                            for k, v in _PROTEIN_ALIASES.items())))

    out = {}
    for k, v in settings.items():
        k = _PROTEIN_ALIASES.get(k, k)
        if k not in known:
            continue
        if k == "box" and v:
            # box was the one length still in Angstrom on this path
            out[k] = tuple(float(c) * 10.0 for c in v)
            continue
        if k in _PROTEIN_SCALED and isinstance(v, (int, float)):
            # 0 means "use the measured area", the same as it does on the
            # bilayer path -- it must not fall through to the < 40 A^2 check.
            if k == "area_per_lipid" and not v:
                lipid = settings.get("lipid", "POPC")
                v = KNOWN_APL.get(str(lipid).upper(), 0.643)
            v = float(v) * _PROTEIN_SCALED[k]
        out[k] = v

    _check_lengths_are_nanometres(settings, out)
    cfg = BuildConfig(**out)
    res = _build_protein(cfg)
    return {
        "ok": True,
        "output_dir": os.path.abspath(cfg.output_dir),
        "counts": res.counts,
        "box_nm": [round(float(v) / 10.0, 4) for v in res.box],
        "net_charge": round(float(res.stats["net_charge"]), 6) + 0.0,
        "closest_heavy_atom_contact_A": round(
            res.stats["contacts"]["heavy_min"], 3),
        "heavy_pairs_below_2.4A": res.stats["contacts"].get("heavy_below_2.4"),
        "seconds": round(res.stats["seconds"], 1),
        "files": sorted(os.listdir(cfg.output_dir)),
    }
