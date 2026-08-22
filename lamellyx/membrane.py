"""Build a solvated lipid bilayer, ready for GROMACS.

No protein: just a bilayer, water and salt. Everything the caller is likely to
want to change is a field on `MembraneConfig`, so the whole thing is one call:

    from lamellyx import MembraneConfig, build_membrane
    build_membrane(MembraneConfig(output_dir="popc_box"))

Units in the public configuration are NANOMETRES, matching GROMACS and the way
membrane dimensions are normally quoted. Internally everything is Angstrom.
The conversion happens once, at the top of `build_membrane`, and nowhere else.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import (bilayer, fileio, geom, library, mdp, solvate, topology,
               validate)

BUNDLED_TOPPAR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "toppar")

# Experimental areas per lipid, nm^2, from small-angle scattering.
# POPC: Kucerka, Nieh & Katsaras, BBA 1808:2761 (2011), 0.643 nm^2 at 303 K.
KNOWN_APL = {"POPC": 0.643, "POPE": 0.591, "DPPC": 0.631, "DOPC": 0.674,
             "POPS": 0.550, "DMPC": 0.606}

# The van der Waals and Coulomb cutoffs CHARMM36 is parameterised with. A
# periodic box shorter than twice this cannot satisfy the minimum image
# convention and grompp will refuse it.
CHARMM_CUTOFF = 1.2

KNOWN_FORCEFIELDS = ("CHARMM36", "CHARMM36m")


@dataclass
class MembraneConfig:
    """Everything the builder needs. Lengths in NANOMETRES, salt in molar."""

    output_dir: str = "membrane_box"

    # --- composition -----------------------------------------------------
    lipid: str = "POPC"
    #  {"POPC": 0.8, "POPS": 0.2} builds a mixture; overrides `lipid`.
    composition: dict = field(default_factory=dict)
    leaflet: str = "bilayer"          # "bilayer" or "monolayer"

    # --- geometry (nm) ---------------------------------------------------
    # How the box's X and Y are decided. These mirror CHARMM-GUI's own
    # "Length of XY based on" options:
    #   "lipid_numbers"   give the lipid counts, the box follows   (default)
    #   "box"             give x and y, the lipid counts follow
    #   "protein_margin"  box = protein width + margin_x / margin_y each side
    # A bare bilayer has no protein to take a margin from, so it sizes from
    # the lipid counts. margin_x / margin_y are the protein-path setting.
    size_mode: str = "lipid_numbers"

    n_upper: int = 100                # lipids per leaflet, in lipid_numbers
    n_lower: int = 100
    # Full box edge, not a half-width. Used in "box" mode.
    x: float = 2.0
    y: float = 2.0
    # Lipid between the protein and the box edge, on each side, per axis.
    # 2.0 means the box reaches 2 nm beyond the protein left and right.
    margin_x: float = 2.0
    margin_y: float = 2.0
    # nm^2 per lipid. 0 uses the measured value for the lipid, which is what
    # CHARMM-GUI does -- it has no such field.
    # 0 means "use the measured value" -- 0.643 nm2 for POPC, which is
    # what CHARMM-GUI effectively does. It shipped at 0.15 for a while and
    # the checker refused every build that used the default.
    area_per_lipid: float = 0.0
    water_thickness: float = 1.5      # nm of water above and below

    # --- solvent ---------------------------------------------------------
    salt_concentration: float = 0.5   # molar
    cation: str = "POT"               # CHARMM residue names: POT = K+
    anion: str = "CLA"                # CLA = Cl-
    water_model: str = "TIP3"         # CHARMM-modified TIP3P

    # --- force field and run ---------------------------------------------
    forcefield: str = "CHARMM36"
    toppar: str = ""                  # defaults to the bundled CHARMM36 files
    # "copy" puts the parameters the build uses in its own toppar/;
    # "reference" points topol.top at the shared directory instead, which
    # keeps every build small when you are making many of them.
    toppar_mode: str = "copy"
    temperature: int = 310

    # --- behaviour -------------------------------------------------------
    seed: int = 0
    strict: bool = True               # stop on settings GROMACS cannot run
    verbose: bool = True

    def species(self):
        """Normalised {resname: fraction} composition."""
        if self.composition:
            total = float(sum(self.composition.values()))
            if total <= 0:
                raise ValueError("composition fractions must be positive")
            return {k.upper(): v / total for k, v in self.composition.items()}
        return {self.lipid.upper(): 1.0}

    def effective_apl(self):
        """Area per lipid actually used, nm^2.

        An explicit setting wins; otherwise the measured value for the lipid.
        Deriving it is what CHARMM-GUI does, and why it has no field for it.
        """
        if self.area_per_lipid and self.area_per_lipid > 0:
            return float(self.area_per_lipid)
        return KNOWN_APL.get(next(iter(self.species())), 0.643)

    def resolved_xy(self):
        """(x, y) in nm for the two modes that do not need a protein.

        In lipid_numbers mode the box is whatever holds the lipids asked for
        -- counts in, box out, the same way round as CHARMM-GUI.
        """
        if self.size_mode == "lipid_numbers":
            per = max(int(self.n_upper), int(self.n_lower), 1)
            side = (per * self.effective_apl()) ** 0.5
            return side, side
        return float(self.x), float(self.y)


@dataclass
class MembraneResult:
    atoms: object = None
    box_nm: tuple = ()
    counts: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    files: list = field(default_factory=list)


# --------------------------------------------------------------------------
# sanity checks
# --------------------------------------------------------------------------

def check_settings(cfg, protein=False, xy_margin=None):
    """Settings that are impossible, or merely unwise. Returns [(level, msg)].

    Level "error" means GROMACS or geometry will reject the result outright;
    "warning" means it will run but the physics is questionable.

    With `protein=True` the box edge is not a setting at all -- it comes from
    the protein's width plus margin_x / margin_y on each side -- so the x/y
    checks are replaced by a check on the margin.
    """
    out = []
    minimum = 2.0 * CHARMM_CUTOFF

    # Values that are not numbers of the right sign at all. These come first
    # because everything below divides by or compares against them.
    # 0 is not "no area", it is "use the measured one" -- so only a negative
    # setting is an error, and every check below reads the derived value.
    if cfg.area_per_lipid < 0:
        out.append(("error", "area_per_lipid cannot be negative (%r). Leave "
                             "it at 0 to use the measured value for the lipid."
                    % cfg.area_per_lipid))
    apl = cfg.effective_apl()
    if cfg.water_thickness < 0:
        out.append(("error", "water_thickness cannot be negative (%r)"
                    % cfg.water_thickness))
    if cfg.salt_concentration < 0:
        out.append(("error", "salt_concentration cannot be negative (%r)"
                    % cfg.salt_concentration))
    if cfg.temperature <= 0:
        out.append(("error", "temperature must be above absolute zero (%r K)"
                    % cfg.temperature))
    # A composition naming a lipid with no conformer library builds nothing.
    # It used to pass check() and then raise NotImplementedError a minute into
    # build(), which defeats the point of having check() at all: it exists so a
    # configuration can be rejected before it costs anything.
    wanted = [str(k).upper() for k in cfg.species()]
    if len(wanted) > 1:
        out.append(("error",
                    "mixed bilayers are not supported yet (asked for %s). Each "
                    "species needs its own conformer library and interleaved "
                    "placement -- build one lipid at a time for now."
                    % ", ".join(sorted(wanted))))
    elif wanted:
        from .library import available_lipids
        have = {l.upper() for l in available_lipids()}
        missing = [l for l in wanted if l not in have]
        if missing:
            out.append(("error",
                        "no conformer library for %s (have: %s). Build one with "
                        "`python -m lamellyx.make_data <gromacs_dir> --lipid %s`"
                        % (", ".join(missing),
                           ", ".join(sorted(have)) or "none", missing[0])))

    if cfg.leaflet not in ("bilayer", "monolayer"):
        out.append(("error",
                    "leaflet must be 'bilayer' or 'monolayer', not %r -- an "
                    "unrecognised value used to build a monolayer silently"
                    % cfg.leaflet))
    if cfg.size_mode not in ("lipid_numbers", "box", "protein_margin"):
        out.append(("error",
                    "size_mode must be 'lipid_numbers', 'box' or "
                    "'protein_margin', not %r" % cfg.size_mode))
    if cfg.toppar_mode not in ("copy", "reference"):
        out.append(("error", "toppar_mode must be 'copy' or 'reference', "
                             "not %r" % cfg.toppar_mode))
    for n, v in (("n_upper", cfg.n_upper), ("n_lower", cfg.n_lower)):
        if v < 0:
            out.append(("error", "%s cannot be negative (%r)" % (n, v)))

    if protein:
        m = float(min(cfg.margin_x, cfg.margin_y) if xy_margin is None
                  else xy_margin)
        if m <= 0:
            out.append(("error",
                        "the lipid margin must be positive (%.2f nm). It is "
                        "the width of membrane between the protein and the "
                        "box edge, on each side." % m))
        elif m < 1.0:
            out.append(("warning",
                        "a %.2f nm margin leaves barely one lipid between the "
                        "protein and its own periodic image. 2 nm each side "
                        "is a usual minimum." % m))

    # A setting that is quietly ignored is worse than one that is rejected.
    _d = MembraneConfig()
    if cfg.size_mode != "box" and (cfg.x != _d.x or cfg.y != _d.y):
        out.append(("warning",
                    "x and y are ignored in size_mode=%r -- the box comes "
                    "from %s. Set size_mode='box' to use them."
                    % (cfg.size_mode,
                       "the lipid counts" if cfg.size_mode == "lipid_numbers"
                       else "the protein and its margin")))
    if protein or cfg.size_mode == "protein_margin":
        return out

    # In lipid_numbers mode x and y are outputs, not inputs, so the message
    # has to talk about the count that produced the box rather than a field
    # the user never set.
    from_counts = cfg.size_mode == "lipid_numbers"
    bx, by = cfg.resolved_xy()
    for axis, val in (("x", bx), ("y", by)):
        if val < minimum:
            if from_counts:
                per = max(int(cfg.n_upper), int(cfg.n_lower), 1)
                out.append(("error",
                            "%d lipids per leaflet at %.3f nm2 each make a "
                            "%.2f nm box, below the %.1f nm minimum. %s uses "
                            "%.1f nm cutoffs and a periodic box must be twice "
                            "its cutoff, so grompp would refuse it. Use about "
                            "%d lipids per leaflet or more."
                            % (per, cfg.effective_apl(), val, minimum,
                               cfg.forcefield, CHARMM_CUTOFF,
                               int((minimum ** 2) / cfg.effective_apl()) + 1)))
            else:
                out.append(("error",
                            "%s = %.2f nm is below %.1f nm. This is the full "
                            "box edge, not a half-width. %s uses %.1f nm "
                            "cutoffs, and a periodic box must be at least "
                            "twice the cutoff for the minimum image "
                            "convention, so grompp will refuse this box. Use "
                            "%.1f nm or more."
                            % (axis, val, minimum, cfg.forcefield,
                               CHARMM_CUTOFF, minimum)))
            break
        elif val < 6.0:
            out.append(("warning",
                        "the box comes out %.2f nm in %s, a very small patch. "
                        "Below about 6 nm a bilayer interacts with its own "
                        "periodic image and cannot develop normal "
                        "undulations." % (val, axis)))
            break

    for name in cfg.species():
        ref = KNOWN_APL.get(name)
        if ref and not (0.75 * ref <= apl <= 1.35 * ref):
            out.append(("warning",
                        "area_per_lipid = %.3f nm2 against a measured %.3f "
                        "nm2 for %s. At %.0f%% of the experimental value the "
                        "packing is %s; the bilayer will fight the barostat "
                        "and may not be buildable at all."
                        % (apl, ref, name, 100 * apl / ref,
                           "far too tight" if apl < ref else "far too loose")))
    if apl < 0.40:
        out.append(("error",
                    "area_per_lipid = %.3f nm2 is below the ~0.40 nm2 taken "
                    "by two extended acyl chains. No lipid packs this "
                    "tightly in any phase, and the packer cannot place them."
                    % apl))

    if cfg.water_thickness < 1.0:
        out.append(("warning",
                    "water_thickness = %.2f nm leaves the bilayer close to "
                    "its own periodic image through the solvent."
                    % cfg.water_thickness))
    if cfg.forcefield not in KNOWN_FORCEFIELDS:
        out.append(("warning",
                    "forcefield = %r is only a label here -- parameters come "
                    "from the toppar directory. Known CHARMM lipid force "
                    "fields are %s (there is no CHARMM46)."
                    % (cfg.forcefield, " and ".join(KNOWN_FORCEFIELDS))))
    if cfg.salt_concentration > 1.0:
        out.append(("warning",
                    "salt_concentration = %.2f M is above the solubility "
                    "range usually simulated." % cfg.salt_concentration))
    return out


def _report_checks(cfg, issues):
    errors = [m for lvl, m in issues if lvl == "error"]
    warns = [m for lvl, m in issues if lvl == "warning"]
    if cfg.verbose and (errors or warns):
        print("-" * 72)
        for m in errors:
            print("ERROR   " + _wrap(m))
        for m in warns:
            print("WARNING " + _wrap(m))
        print("-" * 72)
    if errors and cfg.strict:
        raise ValueError(
            "%d setting(s) cannot produce a runnable system; fix them or "
            "pass strict=False to build anyway:\n  - %s"
            % (len(errors), "\n  - ".join(errors)))
    return errors + warns


def _wrap(msg, width=64, indent=8):
    words, lines, cur = msg.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return ("\n" + " " * indent).join(lines)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build_membrane(cfg):
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    res = MembraneResult()

    issues = check_settings(cfg)
    res.warnings = _report_checks(cfg, issues)

    # `strict=False` waives the physics, not the arithmetic. Dividing the box
    # area by a zero area per lipid has no answer to give, so these stay fatal
    # however the checks were configured.
    if cfg.leaflet not in ("bilayer", "monolayer"):
        raise ValueError("leaflet must be 'bilayer' or 'monolayer', not %r"
                         % cfg.leaflet)
    if cfg.size_mode not in ("lipid_numbers", "box"):
        raise ValueError(
            "size_mode %r needs a protein; use build_protein_system, or "
            "'lipid_numbers' / 'box' for a bare bilayer" % cfg.size_mode)
    # 0 means "use the measured value"; a negative number means nothing, and
    # must not fall through to the same branch and be silently ignored.
    if cfg.area_per_lipid < 0 or cfg.effective_apl() <= 0:
        raise ValueError(
            "area_per_lipid must be positive, or 0 to use the measured value "
            "for the lipid -- not %r" % cfg.area_per_lipid)

    nm = 10.0                                   # nm -> Angstrom
    x_nm, y_nm = cfg.resolved_xy()
    for name, val in (("x", x_nm), ("y", y_nm)):
        if val <= 0:
            raise ValueError("%s must be positive, not %r" % (name, val))
    bx, by = x_nm * nm, y_nm * nm
    t_water = cfg.water_thickness * nm
    apl = cfg.effective_apl() * nm * nm         # nm^2 -> A^2
    species = cfg.species()

    toppar = cfg.toppar or BUNDLED_TOPPAR
    if not os.path.isdir(toppar):
        raise FileNotFoundError("toppar directory not found: %s" % toppar)
    tops = topology.load_toppar(toppar)
    # Report the real blocker first. Asking for a mixture used to fail with
    # "POPE.itp is not in ...", which sends you looking for a file when the
    # feature is what is missing.
    if len(species) > 1:
        raise NotImplementedError(
            "mixed bilayers are not supported yet (asked for %s). Each "
            "species needs its own conformer library and interleaved "
            "placement; build one lipid at a time for now."
            % ", ".join(sorted(species)))
    for need in list(species) + [cfg.cation, cfg.anion, cfg.water_model]:
        if need not in tops:
            raise KeyError("%s.itp is not in %s (have: %s)"
                           % (need, toppar, ", ".join(sorted(tops))))

    # ---- how many lipids -------------------------------------------------
    # Giving n_upper alone used to leave n_lower on the area-per-lipid count,
    # so asking for 10 in one leaflet quietly produced 10 against 25.
    per_leaflet = max(int(round(bx * by / apl)), 1)
    if cfg.size_mode == "lipid_numbers":
        n_upper = max(int(cfg.n_upper), 1)
        n_lower = max(int(cfg.n_lower), 0) if cfg.leaflet == "bilayer" else 0
    else:
        # In box mode the counts are outputs, always. Reading n_upper here
        # would pick up its lipid_numbers default of 100 and try to pack a
        # hundred lipids into whatever box was asked for.
        n_upper = per_leaflet
        n_lower = per_leaflet if cfg.leaflet == "bilayer" else 0
    if cfg.verbose:
        how = ("from %d/%d lipids" % (n_upper, n_lower)
               if cfg.size_mode == "lipid_numbers" else "as configured")
        print("1. %s %s, %.2f x %.2f nm %s at %.3f nm2 per lipid"
              % (cfg.forcefield, cfg.leaflet, x_nm, y_nm, how,
                 cfg.effective_apl()))
        print("   %d lipids upper, %d lower (%s)"
              % (n_upper, n_lower,
                 ", ".join("%s %.0f%%" % (k, 100 * v)
                           for k, v in species.items())))

    # ---- bilayer ---------------------------------------------------------
    resname = next(iter(species))
    lipid_top = tops[resname]

    lib = library.load_lipid_library(resname, midplane=0.0)
    bot, top = library.lipid_extent(lib)
    prov_z = (top - bot) + 2.0 * t_water
    shift = t_water - bot
    lib.head_z_upper += shift
    lib.head_z_lower += shift
    lib.midplane += shift
    box = np.array([bx, by, prov_z])
    if cfg.verbose:
        print("2. packing the bilayer (provisional box %.2f x %.2f x %.2f nm)"
              % (bx / nm, by / nm, prov_z / nm))

    empty = np.zeros((0, 3))
    bres = bilayer.build_bilayer(lib, n_upper, n_lower, box, empty, rng,
                                 apl=apl, verbose=cfg.verbose)
    lip_xyz = bres.xyz

    # ---- final box from the packed lipids --------------------------------
    zlo, zhi = lip_xyz[..., 2].min(), lip_xyz[..., 2].max()
    box[2] = (zhi - zlo) + 2.0 * t_water
    lip_xyz[..., 2] += t_water - zlo
    if cfg.verbose:
        print("3. box %.3f x %.3f x %.3f nm (bilayer %.2f nm thick, "
              "%.2f nm water each side)"
              % (box[0] / nm, box[1] / nm, box[2] / nm,
                 (zhi - zlo) / nm, t_water / nm))

    lipids = fileio.Atoms(
        np.tile(lib.atomnames, len(lip_xyz)),
        np.full(len(lip_xyz) * lipid_top.natoms, resname),
        np.repeat(np.arange(1, len(lip_xyz) + 1), lipid_top.natoms),
        lip_xyz.reshape(-1, 3),
        segid=np.full(len(lip_xyz) * lipid_top.natoms, "MEMB"))

    # ---- water and salt --------------------------------------------------
    if cfg.verbose:
        print("4. solvating")
    template = library.load_water_template(cfg.water_model)
    heads_up = lip_xyz[:, lib.head, 2]
    mid = 0.5 * (heads_up.max() + heads_up.min())
    core = (float(heads_up[heads_up < mid].mean()) + 2.0,
            float(heads_up[heads_up > mid].mean()) - 2.0) \
        if n_lower else (float(lip_xyz[..., 2].min()) - 1.0, mid)

    wat = solvate.solvate(
        lipids.xyz, lipids.element != "H", box, template, rng,
        core_z=core, keep_pore_water=False, verbose=cfg.verbose)

    q = lipid_top.total_charge * len(lip_xyz)
    n_cat, n_ani = solvate.ion_counts(len(wat), q, cfg.salt_concentration)
    if cfg.verbose:
        print("   %.3f M in %d waters -> %d %s, %d %s (lipid charge %+.2f)"
              % (cfg.salt_concentration, len(wat), n_cat, cfg.cation,
                 n_ani, cfg.anion, q))
    wat, cat_xyz, ani_xyz = solvate.place_ions(
        wat, box, lipids.xyz, n_cat, n_ani, rng, exclude_z=core,
        verbose=cfg.verbose)

    def _ion(name, xyz):
        return fileio.Atoms([name] * len(xyz), [name] * len(xyz),
                            np.arange(1, len(xyz) + 1), xyz,
                            segid=["SOLV"] * len(xyz))

    water = fileio.Atoms(
        np.tile(template.atomnames, len(wat)),
        np.full(len(wat) * tops[cfg.water_model].natoms, cfg.water_model),
        np.repeat(np.arange(1, len(wat) + 1), tops[cfg.water_model].natoms),
        wat.reshape(-1, 3),
        segid=np.full(len(wat) * tops[cfg.water_model].natoms, "SOLV"))

    # ---- assemble --------------------------------------------------------
    system = fileio.Atoms.concat(
        [lipids, _ion(cfg.cation, cat_xyz), _ion(cfg.anion, ani_xyz), water])
    blocks = [(lipid_top, len(lip_xyz)), (tops[cfg.cation], len(cat_xyz)),
              (tops[cfg.anion], len(ani_xyz)),
              (tops[cfg.water_model], len(wat))]
    system.xyz = geom.wrap_by_blocks(system.xyz, box, blocks, (True, True, False))
    system = fileio.renumber_by_residue(system)
    res.atoms = system
    res.box_nm = tuple(box / nm)
    expect = sum(m.natoms * c for m, c in blocks)
    if expect != len(system):
        raise AssertionError("topology says %d atoms, coordinates have %d"
                             % (expect, len(system)))
    net = (q + len(cat_xyz) * tops[cfg.cation].total_charge
           + len(ani_xyz) * tops[cfg.anion].total_charge)
    if abs(net) > 1e-6:
        raise AssertionError("system is not neutral: %+.3f" % net)

    res.counts = {resname: len(lip_xyz), cfg.cation: len(cat_xyz),
                  cfg.anion: len(ani_xyz), cfg.water_model: len(wat),
                  "TOTAL_ATOMS": len(system)}
    res.stats["net_charge"] = round(float(net), 6)   # 2e-14 is not "nonzero"
    res.stats["area_per_lipid_nm2"] = float(bx * by / max(n_upper, 1)) / 100.0
    res.stats["bilayer_thickness_nm"] = float(zhi - zlo) / nm

    res.files = _write(cfg, res, toppar, blocks, lipid_top, resname,
                       len(lip_xyz), len(cat_xyz), len(ani_xyz), len(wat))

    # ---- check -----------------------------------------------------------
    # The report is written after this, not before, so that what it records
    # is what was actually built rather than what was intended.
    if cfg.verbose:
        print("5. checking")
    nlip = len(lip_xyz) * lipid_top.natoms
    idx = np.arange(len(system))
    groups = {"lipid": idx < nlip, "solvent": idx >= nlip}
    excl = validate.system_exclusions(blocks, len(system))
    rep = validate.contacts(system, box, excl, cutoff=2.6, groups=groups)
    if cfg.verbose:
        print(rep.format())
        print("   density               : %.3f g/cm3"
              % validate.density(system, box))
        print("   area per lipid        : %.3f nm2"
              % res.stats["area_per_lipid_nm2"])
    res.stats["contacts"] = {"min_distance": rep.min_distance,
                             "heavy_min": rep.heavy_min,
                             "heavy_below_2.4": rep.heavy_below}
    if cfg.verbose and (rep.heavy_below or rep.min_distance < 1.0):
        print("   note: run step6.0_minimization first -- the closest pair is "
              "%.2f A, and a contact that needs a torsion to change can only "
              "be opened by minimisation." % rep.min_distance)
    res.stats["density_g_cm3"] = validate.density(system, box)
    res.stats["seconds"] = time.time() - t0
    res.stats["output_bytes"] = topology.directory_size(cfg.output_dir)

    p = os.path.join(cfg.output_dir, "build_report.json")
    with open(p, "w", newline="\n") as fh:
        json.dump({"config": asdict(cfg), "counts": res.counts,
                   "box_nm": list(res.box_nm), "warnings": res.warnings,
                   "stats": _jsonable(res.stats)}, fh, indent=2)
    res.files.append(p)

    if cfg.verbose:
        print("\nbuilt %d atoms in %.0f s -> %s"
              % (len(system), res.stats["seconds"], cfg.output_dir))
    return res


def _write(cfg, res, toppar, blocks, lipid_top, resname, n_lip, n_cat, n_ani,
           n_wat):
    out = cfg.output_dir
    os.makedirs(out, exist_ok=True)
    files = []
    used = [resname, cfg.cation, cfg.anion, cfg.water_model]
    if cfg.toppar_mode == "reference":
        prefix = os.path.relpath(toppar, out).replace("\\", "/") + "/"
    else:
        topology.copy_toppar(toppar, os.path.join(out, "toppar"), only=used)
        prefix = "toppar/"

    includes = ["%s.itp" % n for n in
                (resname, cfg.cation, cfg.anion, cfg.water_model)]
    molecules = [(resname, n_lip), (cfg.cation, n_cat), (cfg.anion, n_ani),
                 (cfg.water_model, n_wat)]
    p = os.path.join(out, "topol.top")
    # Not "POPC bilayer": a system name that begins with a molecule name and
    # a word reads like a [ molecules ] entry to anything parsing loosely.
    topology.write_topol(p, includes, molecules,
                         system_name="%s-%s" % (resname, cfg.leaflet),
                         prefix=prefix)
    files.append(p)

    p = os.path.join(out, "step5_input.gro")
    fileio.write_gro(p, res.atoms, np.array(res.box_nm) * 10.0,
                     title="lamellyx %s bilayer" % resname)
    files.append(p)

    n = len(res.atoms)
    nlip = n_lip * lipid_top.natoms
    p = os.path.join(out, "index.ndx")
    topology.write_index(p, topology.membrane_groups(
        n, slice(0, nlip), slice(nlip, n)))
    files.append(p)

    files += mdp.write_series(out, cfg.temperature, membrane_only=True)
    return files


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o
