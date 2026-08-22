"""Convert a CGenFF stream file (.str) into GROMACS ligand topology.

Like `pdb2gmx.py`, this does not invent force-field science. CGenFF -- the
ParamChem web server or the licensed binary -- does the hard part: it looks at
a drawn molecule and assigns CHARMM atom types, partial charges and bonded
parameters by analogy to the General Force Field. This module only does the
bookkeeping that CHARMM-GUI's Ligand Reader also does, so the result can be
driven from a script instead of a web form:

    read the .str  ->  convert CHARMM units to GROMACS units
                   ->  enumerate the molecule's angles and dihedrals from its
                       bond graph and match each to a parameter
                   ->  write LIG.itp and LIG_atomtypes.itp for the builder

Two things here are genuinely ours, not a re-implementation of CGenFF:

  * The penalty report. CGenFF reports, per parameter and per charge, how far
    it reached by analogy -- the "penalty". A penalty of 200 means it is
    guessing. This surfaces the worst offenders in the build report rather than
    burying them, but it does not refuse: most biological ligands lean on
    analogy, so a hard cutoff would reject the normal case. Judging the number
    is left to the caller (`penalty_flag` lists everything above a chosen one).

  * The unit conversion. Every factor is written at the function that uses it,
    because getting one wrong is the "silently wrong energies" failure this
    whole package is arranged to avoid. Each conversion is a small pure
    function with a hand-checked test.

> Nothing this produces has been through `gmx grompp`. The checks here are
> geometric and bookkeeping only, the same as the rest of lamellyx. The
> acceptance test that still has to be run, when a molecule and a cluster are
> both available, is reproducing a CHARMM-GUI Ligand Reader result
> count-for-count from the same .str.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

# -- unit conversion constants ------------------------------------------------
# CHARMM writes energies in kcal/mol and lengths in Angstrom. GROMACS uses
# kJ/mol and nm. Two subtleties bite here and are the reason each conversion is
# its own function below:
#   * CHARMM harmonic terms are written E = k (x - x0)^2, GROMACS as
#     E = 1/2 k (x - x0)^2, so a factor of two enters bonds, angles, Urey-
#     Bradley and impropers -- but NOT dihedrals, whose functional form matches
#     between the two codes with no factor of two.
#   * a Lennard-Jones minimum is given as Rmin/2 in CHARMM and as sigma in
#     GROMACS; sigma = Rmin / 2^(1/6) = (2 * Rmin/2) / 2^(1/6).
KCAL = 4.184                                   # kJ per kcal
ANG = 0.1                                       # nm per Angstrom
SIGMA_FROM_RMIN2 = 2.0 * ANG / (2.0 ** (1.0 / 6.0))   # (Rmin/2, A) -> sigma, nm

# Advisory only -- ParamChem's own guidance: a penalty under 10 is fine, 10-50
# is worth checking, and over 50 needs real validation. Penalties are reported,
# never refused (see the module docstring), so this is the natural default for
# `penalty_flag` when a caller wants the worst ones listed, not a gate.
PENALTY_ADVISORY = 50.0

_Z_BY_MASS = [
    (1.008, 1), (4.0026, 2), (10.811, 5), (12.011, 6), (14.007, 7),
    (15.999, 8), (18.998, 9), (22.990, 11), (24.305, 12), (26.982, 13),
    (28.085, 14), (30.974, 15), (32.06, 16), (35.45, 17), (39.098, 19),
    (40.078, 20), (55.845, 26), (63.546, 29), (65.38, 30), (79.904, 35),
    (126.90, 53),
]


def _atomic_number(mass):
    """Element from mass. GROMACS wants an atomic number in [ atomtypes ]; the
    .str gives a mass, not an element, so pick the nearest known element."""
    return min(_Z_BY_MASS, key=lambda z: abs(z[0] - mass))[1]


# CGenFF atom type names begin with their element (CG331, HGA3, OG311, CLGA1).
# A real ParamChem .str carries no MASS records -- the masses live in the base
# CGenFF rtf -- so when one is absent the element, and thus the mass, is read
# off the type name. Two-letter elements are matched before one-letter ones so
# CLGA1 is chlorine, not carbon.
_MASS_BY_ELEMENT = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
    "P": 30.974, "S": 32.06, "CL": 35.45, "BR": 79.904, "I": 126.904,
    "NA": 22.990, "MG": 24.305, "K": 39.098, "CA": 40.078, "ZN": 65.38,
    "FE": 55.845,
}
_TWO_LETTER_ELEMENTS = ("CL", "BR", "NA", "MG", "CA", "ZN", "FE")


def _element_from_type(atom_type):
    u = atom_type.upper()
    for p in _TWO_LETTER_ELEMENTS:
        if u.startswith(p):
            return p
    return u[0] if u else "C"


def _mass_from_type(atom_type):
    return _MASS_BY_ELEMENT.get(_element_from_type(atom_type), 12.011)


# -- the parsed stream --------------------------------------------------------


@dataclass
class Atom:
    name: str
    type: str
    charge: float
    charge_penalty: float = 0.0


@dataclass
class LonePair:
    """A CHARMM LONEPAIR: a massless virtual site whose position is constructed
    from real atoms each step. CGenFF uses these for halogen sigma holes.

    `name` is the LP's own atom name (it is also listed as an ATOM). `kind` is
    the construction keyword (CGenFF only emits COLINEAR). `hosts` are the
    constructing atom names in file order; `dist` is in Angstrom, as written.
    """

    name: str
    kind: str
    hosts: list = field(default_factory=list)
    dist: float = 0.0
    angle: float = 0.0
    dihe: float = 0.0
    scale: float = 0.0


@dataclass
class LigandStream:
    """Everything read out of one .str, in CHARMM units still."""

    resname: str = "LIG"
    atoms: list = field(default_factory=list)        # Atom, in file order
    bonds: list = field(default_factory=list)        # (name_i, name_j)
    impropers: list = field(default_factory=list)    # (n1, n2, n3, n4) names
    lonepairs: list = field(default_factory=list)    # LonePair, in file order
    masses: dict = field(default_factory=dict)       # type -> mass
    # parameter tables, keyed by tuples of atom TYPES, CHARMM units:
    bond_p: dict = field(default_factory=dict)       # (t1,t2) -> (kb, b0, pen)
    angle_p: dict = field(default_factory=dict)      # (t1,t2,t3) ->
    #                                    (ktheta, theta0, kub, s0, pen)
    dihe_p: dict = field(default_factory=dict)       # (t1..t4) ->
    #                                    [(kchi, mult, delta, pen), ...]
    impr_p: dict = field(default_factory=dict)       # (t1..t4) -> (kpsi,psi0,pen)
    nb: dict = field(default_factory=dict)           # type ->
    #                          (eps, rmin2, eps14, rmin2_14, pen)

    def type_of(self):
        return {a.name: a.type for a in self.atoms}

    def mass_of(self, atom_type):
        # A real ParamChem stream omits MASS records -- the masses come from the
        # base CGenFF rtf. When one is present, use it; otherwise read the
        # element off the type name, which matches the base rtf for every
        # ordinary organic element.
        if atom_type in self.masses:
            return self.masses[atom_type]
        return _mass_from_type(atom_type)

    def lp_names(self):
        """The atom names that are lone pairs (virtual sites, zero mass)."""
        return {lp.name for lp in self.lonepairs}

    def lp_types(self):
        """The atom types used only by lone pairs -- they carry no real mass or
        Lennard-Jones term, so [ atomtypes ] must write zeros for them even when
        the base force field is absent."""
        names = self.lp_names()
        return {a.type for a in self.atoms if a.name in names}


# A residue name becomes a [ moleculetype ] name and two output filenames, so it
# has to be a plain identifier -- never a path. This blocks the traversal a
# `resname` of "../x" would otherwise allow when the name is joined onto the
# output directory.
_RESNAME_OK = re.compile(r"^[A-Za-z0-9_+-]+$")


def _check_resname(resname):
    if not resname or not _RESNAME_OK.match(resname):
        raise ValueError(
            "invalid residue name %r: use letters, digits, _ + - only. It names "
            "the moleculetype and the output files, so it must not contain a "
            "path separator, '..', a dot or a space." % resname)


# -- parsing ------------------------------------------------------------------

_PENALTY = re.compile(r"penalty\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.I)


def _penalty_from_comment(comment):
    """Pull a `penalty= X` out of a parameter-line comment, else 0."""
    m = _PENALTY.search(comment or "")
    return float(m.group(1)) if m else 0.0


def _split_comment(raw):
    """Return (code, comment) split on the first '!'."""
    i = raw.find("!")
    if i < 0:
        return raw.strip(), ""
    return raw[:i].strip(), raw[i + 1:].strip()


# geometry keywords that end the host-atom list in a LONEPAIR record
_LP_GEOM_KW = {"DIST": "DIST", "DISTANCE": "DIST", "ANGLE": "ANGLE",
               "DIHE": "DIHE", "DIHEDRAL": "DIHE", "SCALE": "SCALE"}


def _parse_lonepair(parts):
    """Parse one CHARMM `LONEPAIR <kind> <lp> <host...> DIST d [ANGLE a ...]`.

    The host-atom count varies by kind (COLINEAR has two, RELATIVE three), so
    the hosts are read up to the first geometry keyword rather than by a fixed
    column. Returns a LonePair, or None if the line is too short to be one.
    """
    if len(parts) < 4:
        return None
    kind = parts[1]
    lp = parts[2]
    hosts, i = [], 3
    while i < len(parts) and parts[i].upper() not in _LP_GEOM_KW:
        hosts.append(parts[i])
        i += 1
    vals = {"DIST": 0.0, "ANGLE": 0.0, "DIHE": 0.0, "SCALE": 0.0}
    while i < len(parts) - 1:
        key = _LP_GEOM_KW.get(parts[i].upper())
        if key:
            try:
                vals[key] = float(parts[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1
    return LonePair(lp, kind, hosts, vals["DIST"], vals["ANGLE"],
                    vals["DIHE"], vals["SCALE"])


# Section headers that appear in a .str's param half and in a full .prm. ATOMS
# and the trailing keyword sections are recognised so their lines are treated
# as (inert) section content, not mistaken for parameters.
_PARAM_HEADERS = {"ATOMS", "BONDS", "ANGLES", "DIHEDRALS", "IMPROPERS",
                  "IMPROPER", "NONBONDED", "NBFIX", "CMAP", "HBOND", "END"}


def _consume_param(section, parts, comment, s):
    """Read one parameter data line into `s`.

    Shared by the .str parser and the base-force-field parser. It is robust to
    the non-parameter lines a real par_all36_cgenff.prm carries -- the
    NONBONDED header's `cutnb 14.0 ctofnb ...` continuation, stray keywords --
    which are skipped rather than misread as an atom type with an enormous
    force constant.
    """
    pen = _penalty_from_comment(comment)
    try:
        if section == "BONDS" and len(parts) >= 4:
            s.bond_p[(parts[0], parts[1])] = (
                float(parts[2]), float(parts[3]), pen)
        elif section == "ANGLES" and len(parts) >= 5:
            kub = float(parts[5]) if len(parts) >= 7 else 0.0
            s0 = float(parts[6]) if len(parts) >= 7 else 0.0
            s.angle_p[(parts[0], parts[1], parts[2])] = (
                float(parts[3]), float(parts[4]), kub, s0, pen)
        elif section == "DIHEDRALS" and len(parts) >= 7:
            quad = (parts[0], parts[1], parts[2], parts[3])
            s.dihe_p.setdefault(quad, []).append(
                (float(parts[4]), int(float(parts[5])), float(parts[6]), pen))
        elif section == "IMPROPERS" and len(parts) >= 7:
            s.impr_p[(parts[0], parts[1], parts[2], parts[3])] = (
                float(parts[4]), float(parts[6]), pen)
        elif section == "NONBONDED" and len(parts) >= 4:
            eps, rmin2 = abs(float(parts[2])), float(parts[3])
            eps14, rmin2_14 = eps, rmin2
            if len(parts) >= 7:
                eps14, rmin2_14 = abs(float(parts[5])), float(parts[6])
            s.nb[parts[0]] = (eps, rmin2, eps14, rmin2_14, pen)
    except (ValueError, IndexError):
        pass    # a header continuation or keyword line, not a parameter


def parse_stream(source):
    """Parse a CGenFF .str. `source` is a path or the file's text.

    The .str has two halves -- `read rtf ... END` describing the molecule
    (atoms, charges, bonds, impropers) and `read para ... END` describing the
    parameters. Both are read here; a molecule with no matching parameter is
    caught later, when the topology is written, not silently dropped.
    """
    if "\n" not in source and os.path.exists(source):
        with open(source) as fh:
            text = fh.read()
    else:
        text = source

    s = LigandStream()
    mode = None          # "rtf" | "param"
    section = None       # within param: bonds/angles/dihedrals/impropers/nb

    for raw in text.splitlines():
        code, comment = _split_comment(raw)
        if not code:
            continue
        low = code.lower()

        # section switches ----------------------------------------------------
        if low.startswith("read") and "rtf" in low:
            mode, section = "rtf", None
            continue
        if low.startswith("read") and ("para" in low or "prm" in low):
            mode, section = "param", None
            continue
        if low in ("end", "return") or low.startswith("end "):
            section = None
            continue

        parts = code.split()

        if mode == "rtf":
            key = parts[0].upper()
            if key == "MASS":
                # MASS -1 CG331 12.01100 -- guarded so a truncated record is a
                # clear error, not a bare IndexError deep in parsing.
                if len(parts) < 4:
                    raise ValueError("malformed MASS record: %r" % code)
                s.masses[parts[2]] = float(parts[3])
            elif key == "RESI":
                if len(parts) < 2:
                    raise ValueError("malformed RESI record: %r" % code)
                s.resname = parts[1]
            elif key == "ATOM":
                # ATOM C1 CG331 -0.270 ! <charge penalty>
                if len(parts) < 4:
                    raise ValueError(
                        "malformed ATOM record %r: expected 'ATOM name type "
                        "charge'" % code)
                pen = 0.0
                mnum = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", comment or "")
                if mnum:
                    pen = float(mnum.group(1))
                s.atoms.append(Atom(parts[1], parts[2], float(parts[3]), pen))
            elif key in ("BOND", "DOUBLE", "TRIPLE"):
                # a BOND line may declare several pairs. An odd atom count means
                # a dropped orphan -- a silently-missing bond -- so refuse it.
                names = parts[1:]
                if len(names) % 2:
                    raise ValueError("malformed %s record (needs an even number "
                                     "of atoms): %r" % (key, code))
                for a, b in zip(names[0::2], names[1::2]):
                    s.bonds.append((a, b))
            elif key in ("IMPR", "IMPROPER"):
                # impropers come in groups of four; a leftover would be dropped,
                # a silently-missing improper (a planar centre goes non-planar).
                names = parts[1:]
                if len(names) % 4:
                    raise ValueError("malformed %s record (needs a multiple of "
                                     "four atoms): %r" % (key, code))
                for q in zip(names[0::4], names[1::4], names[2::4], names[3::4]):
                    s.impropers.append(q)
            elif key.startswith("LONE"):
                # LONEPAIR / LONE -- a constructed virtual site (halogen sigma
                # hole). The LP is also an ATOM record; this line adds only its
                # geometry, kept for the [ virtual_sites2 ] block later.
                lp = _parse_lonepair(parts)
                if lp is not None:
                    s.lonepairs.append(lp)
            continue

        if mode == "param":
            head = parts[0].upper()
            if head in _PARAM_HEADERS:
                section = {"IMPROPER": "IMPROPERS"}.get(head, head)
                continue
            _consume_param(section, parts, comment, s)
            continue

    if not s.atoms:
        raise ValueError("no ATOM records found -- is this a CGenFF .str?")
    names = [a.name for a in s.atoms]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(
            "duplicate atom name(s) in the .str: %s. Atom names must be unique "
            "within a residue -- the bond graph, angle and dihedral enumeration "
            "and coordinate matching all key on the name, so a repeat silently "
            "merges two atoms into one." % ", ".join(dup))

    # Every name a bond, improper or lone pair mentions must be a real ATOM.
    # Caught here so a stray name is one clear error, not a bare KeyError deep
    # in parameter resolution (bonds) or index lookup (impropers, lone pairs).
    known = set(names)
    dangling = set()
    for a, b in s.bonds:
        dangling |= {a, b} - known
    for q in s.impropers:
        dangling |= set(q) - known
    for lp in s.lonepairs:
        dangling |= ({lp.name} | set(lp.hosts)) - known
    if dangling:
        raise ValueError(
            "the .str names atom(s) in a BOND, IMPR or LONEPAIR that have no "
            "ATOM record: %s. Every bonded or constructed atom must be declared."
            % ", ".join(sorted(dangling)))
    return s


# -- the base CGenFF force field, and merging a stream over it -----------------
# A real ParamChem .str carries only the parameters CGenFF generated by analogy.
# The standard bonds, angles, dihedrals, impropers, masses and Lennard-Jones
# terms live in the base CGenFF force field -- top_all36_cgenff.rtf (masses) and
# par_all36_cgenff.prm (everything else). To convert a real ligand the two must
# be merged, with the stream's analogy-generated values overriding the base.
#
# > These files are MacKerell-lab material, the same licensing position as
# > lamellyx/data/, so they are not shipped. The caller points `cgenff_ff` at a
# > directory that has them -- a CHARMM-GUI toppar/ directory does.


def _read_masses(path, s):
    """Read `MASS <n> TYPE mass [element]` lines from an rtf into s.masses."""
    with open(path) as fh:
        for raw in fh:
            code, _ = _split_comment(raw)
            parts = code.split()
            if len(parts) >= 4 and parts[0].upper() == "MASS":
                try:
                    s.masses[parts[2]] = float(parts[3])
                except ValueError:
                    pass


def _parse_prm(path, s):
    """Read a full CHARMM parameter file (par_all36_cgenff.prm) into s."""
    section = None
    with open(path) as fh:
        for raw in fh:
            code, comment = _split_comment(raw)
            if not code.strip() or code.lstrip().startswith("*"):
                continue
            parts = code.split()
            head = parts[0].upper()
            if head in _PARAM_HEADERS:
                section = {"IMPROPER": "IMPROPERS"}.get(head, head)
                continue
            _consume_param(section, parts, comment, s)


# Parsing the ~38k-line par_all36_cgenff.prm is the one real cost of a merge
# (~0.2 s), and parameterising a whole ligand library reuses the same base
# force field every time. Cache it, keyed by path and mtime so an edited file
# is re-read. The base is only ever READ during a merge (apply_base_ff uses
# setdefault and nothing mutates its tables afterwards), so sharing one instance
# across calls is safe.
_FF_CACHE = {}


def clear_ff_cache():
    """Forget any cached base force field. Rarely needed -- the mtime key already
    invalidates an edited file -- but useful in a long-running process."""
    _FF_CACHE.clear()


def load_cgenff_ff(source):
    """Load the base CGenFF force field into a container the merge reads.

    `source` is a directory holding top_all36_cgenff.rtf and
    par_all36_cgenff.prm, or a (rtf_path, prm_path) tuple. The result is cached
    by file path and modification time, so repeated builds against one force
    field parse it once.
    """
    if isinstance(source, (tuple, list)):
        rtf, prm = source
    else:
        rtf = os.path.join(source, "top_all36_cgenff.rtf")
        prm = os.path.join(source, "par_all36_cgenff.prm")
    for p in (rtf, prm):
        if not os.path.exists(p):
            raise FileNotFoundError(
                "base CGenFF file not found: %s. Merging a real .str needs "
                "top_all36_cgenff.rtf and par_all36_cgenff.prm (MacKerell / "
                "CHARMM-GUI material, not shipped with lamellyx). Point "
                "cgenff_ff at a directory that has them -- any CHARMM-GUI "
                "toppar/ does." % p)
    key = (os.path.abspath(rtf), os.path.getmtime(rtf),
           os.path.abspath(prm), os.path.getmtime(prm))
    cached = _FF_CACHE.get(key)
    if cached is not None:
        return cached
    base = LigandStream()
    _read_masses(rtf, base)
    _parse_prm(prm, base)
    _FF_CACHE[key] = base
    return base


def apply_base_ff(stream, base):
    """Fill in from the base force field every parameter the .str did not carry.

    The stream's own values -- the ones CGenFF generated by analogy -- always
    win, so `setdefault` only supplies what is missing.
    """
    for table, btable in (
            (stream.masses, base.masses), (stream.bond_p, base.bond_p),
            (stream.angle_p, base.angle_p), (stream.dihe_p, base.dihe_p),
            (stream.impr_p, base.impr_p), (stream.nb, base.nb)):
        for k, v in btable.items():
            table.setdefault(k, v)


# -- unit conversions (each hand-checked in tests) ----------------------------


def bond_to_gmx(kb, b0):
    """CHARMM bond (Kb kcal/mol/A^2, b0 A) -> GROMACS (b0 nm, kb kJ/mol/nm^2).

    kb doubles (E = k d^2 -> 1/2 k d^2), then kcal->kJ and /A^2 -> /nm^2.
    """
    return b0 * ANG, kb * 2.0 * KCAL / (ANG ** 2)


def angle_to_gmx(kth, th0, kub, s0):
    """CHARMM angle (+ optional Urey-Bradley) -> GROMACS type-5 quintet.

    Returns (theta0 deg, ktheta kJ/mol/rad^2, ub0 nm, kub kJ/mol/nm^2). The
    angle and UB force constants each double for the same reason as a bond.
    """
    return (th0, kth * 2.0 * KCAL,
            s0 * ANG, kub * 2.0 * KCAL / (ANG ** 2))


def dihedral_to_gmx(kchi, delta):
    """CHARMM proper dihedral -> GROMACS type-9 (phi0 deg, kphi kJ/mol).

    No factor of two: E = Kchi(1 + cos(n.phi - delta)) is already the GROMACS
    form. Only kcal -> kJ.
    """
    return delta, kchi * KCAL


def improper_to_gmx(kpsi, psi0):
    """CHARMM improper (Kpsi kcal/mol/rad^2) -> GROMACS type-2 (xi0 deg, kxi
    kJ/mol/rad^2). Harmonic, so the force constant doubles."""
    return psi0, kpsi * 2.0 * KCAL


def lj_to_gmx(eps, rmin2):
    """CHARMM LJ (|eps| kcal/mol, Rmin/2 A) -> GROMACS (sigma nm, eps kJ/mol)."""
    return rmin2 * SIGMA_FROM_RMIN2, eps * KCAL


# -- graph: angles, dihedrals and 1-4 pairs from the bond list ----------------


def _adjacency(atom_names, bonds):
    idx = {n: i for i, n in enumerate(atom_names)}
    adj = {n: [] for n in atom_names}
    for a, b in bonds:
        if a not in idx or b not in idx:
            raise ValueError("bond names a bond graph does not contain: %s-%s"
                             % (a, b))
        adj[a].append(b)
        adj[b].append(a)
    return adj


def enumerate_angles(atom_names, bonds):
    """Every i-j-k with i-j and j-k bonded. One orientation per angle."""
    adj = _adjacency(atom_names, bonds)
    out = []
    for j in atom_names:
        nb = adj[j]
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                out.append((nb[a], j, nb[b]))
    return out


def enumerate_dihedrals(atom_names, bonds):
    """Every i-j-k-l along two adjacent bonds, deduped by its atom set/order."""
    adj = _adjacency(atom_names, bonds)
    seen, out = set(), []
    for j, k in [(a, b) for a, b in bonds]:
        for i in adj[j]:
            if i == k:
                continue
            for l in adj[k]:
                if l == j or l == i:
                    continue
                canon = (i, j, k, l)
                if canon[::-1] in seen or canon in seen:
                    continue
                seen.add(canon)
                out.append(canon)
    return out


def one_four_pairs(atom_names, bonds, dihedrals):
    """1-4 pairs for [ pairs ]: the end atoms of each proper dihedral that are
    not also 1-2 or 1-3 bonded (a small ring can make a 1-4 also 1-3)."""
    bonded = {frozenset(b) for b in bonds}
    one_three = {frozenset((a, c)) for a, _, c in enumerate_angles(atom_names, bonds)}
    pairs, seen = [], set()
    for i, _, _, l in dihedrals:
        key = frozenset((i, l))
        if key in seen or key in bonded or key in one_three or i == l:
            continue
        seen.add(key)
        pairs.append((i, l))
    return pairs


# -- type matching (bonds/angles exact-with-reverse; dihedrals wildcarded) -----


def _match_symmetric(table, key):
    return table.get(key) or table.get(key[::-1])


def _match_wildcard(table, quad):
    """CHARMM dihedral/improper lookup: an exact match wins over a wildcard
    (`X`) match, either atom order. Returns the stored value, or None."""
    best, best_score = None, -1
    for stored, val in table.items():
        for k in (stored, stored[::-1]):
            score, ok = 0, True
            for want, have in zip(quad, k):
                if have == want:
                    score += 1
                elif have == "X":
                    pass
                else:
                    ok = False
                    break
            if ok and score > best_score:
                best, best_score = val, score
    return best


# -- penalty gate -------------------------------------------------------------


def collect_penalties(stream, used_bonds, used_angles, used_dihedrals,
                      used_impropers):
    """Every penalty that actually affects this molecule, with a label.

    Only parameters the molecule USES are counted -- a high penalty on a bond
    type the ligand does not contain is irrelevant and would make the gate
    refuse good molecules.
    """
    out = []
    for a in stream.atoms:
        if a.charge_penalty:
            out.append(("charge %s" % a.name, a.charge_penalty))
    for (t1, t2) in used_bonds:
        p = _match_symmetric(stream.bond_p, (t1, t2))
        if p and p[2]:
            out.append(("bond %s-%s" % (t1, t2), p[2]))
    for tri in used_angles:
        p = _match_symmetric(stream.angle_p, tri)
        if p and p[4]:
            out.append(("angle %s-%s-%s" % tri, p[4]))
    for quad in used_dihedrals:
        terms = (stream.dihe_p.get(quad) or stream.dihe_p.get(quad[::-1])
                 or _match_wildcard(stream.dihe_p, quad))
        if terms:
            worst = max(t[3] for t in terms)
            if worst:
                out.append(("dihedral %s-%s-%s-%s" % quad, worst))
    for quad in used_impropers:
        p = _match_wildcard(stream.impr_p, quad)
        if p and p[2]:
            out.append(("improper %s-%s-%s-%s" % quad, p[2]))
    out.sort(key=lambda kv: kv[1], reverse=True)
    return out


# -- writing ------------------------------------------------------------------


def _resolve_bonded(stream, base_ff_used=False):
    """Enumerate the molecule's angles/dihedrals and pair each term with a
    parameter. Raises if a needed parameter is missing, because a runnable
    topology cannot have a silent gap -- that is a wrong-energy bug, not a
    smaller system. `base_ff_used` only tailors the message."""
    names = [a.name for a in stream.atoms]
    tof = stream.type_of()
    gaps = []

    resolved_bonds = []
    for a, b in stream.bonds:
        p = _match_symmetric(stream.bond_p, (tof[a], tof[b]))
        if p is None:
            gaps.append("bond %s-%s (%s-%s)" % (a, b, tof[a], tof[b]))
        resolved_bonds.append((a, b, p))

    angles = enumerate_angles(names, stream.bonds)
    resolved_angles = []
    for i, j, k in angles:
        p = _match_symmetric(stream.angle_p, (tof[i], tof[j], tof[k]))
        if p is None:
            gaps.append("angle %s-%s-%s (%s-%s-%s)"
                        % (i, j, k, tof[i], tof[j], tof[k]))
        resolved_angles.append((i, j, k, p))

    dihedrals = enumerate_dihedrals(names, stream.bonds)
    resolved_dihedrals = []
    for i, j, k, l in dihedrals:
        typ = (tof[i], tof[j], tof[k], tof[l])
        terms = (stream.dihe_p.get(typ) or stream.dihe_p.get(typ[::-1])
                 or _match_wildcard(stream.dihe_p, typ))
        if not terms:
            gaps.append("dihedral %s-%s-%s-%s (%s-%s-%s-%s)"
                        % (i, j, k, l, *typ))
        resolved_dihedrals.append((i, j, k, l, terms))

    resolved_impropers = []
    for q in stream.impropers:
        typ = tuple(tof[n] for n in q)
        p = _match_wildcard(stream.impr_p, typ)
        if p is None:
            gaps.append("improper %s-%s-%s-%s (%s-%s-%s-%s)" % (q + typ))
        resolved_impropers.append((q, p))

    if gaps:
        if base_ff_used:
            raise ValueError(
                "%d parameter(s) are in neither the .str nor the base CGenFF "
                "force field:\n  %s\nThis is a genuine gap -- the molecule uses "
                "an atom-type combination CGenFF did not provide. Check the "
                ".str, or that the base FF version matches the one CGenFF used."
                % (len(gaps), "\n  ".join(gaps[:12])))
        raise ValueError(
            "%d of this molecule's parameters are not in the .str.\n\n"
            "A CGenFF stream is a SUPPLEMENT to the base force field, not a "
            "complete parameter set: it carries the atoms, charges, bonds and "
            "only the parameters CGenFF had to generate by analogy. The rest "
            "-- standard bonds, angles, dihedrals and Lennard-Jones terms -- "
            "live in top_all36_cgenff.rtf and par_all36_cgenff.prm, which the "
            "stream references but does not include. Pass cgenff_ff pointing at "
            "a directory that has them (any CHARMM-GUI toppar/ does), and they "
            "will be merged in.\n\nFirst few missing:\n  %s"
            % (len(gaps), "\n  ".join(gaps[:12])))

    used = {
        "bonds": {(tof[a], tof[b]) for a, b in stream.bonds},
        "angles": {(tof[i], tof[j], tof[k]) for i, j, k in angles},
        "dihedrals": {(tof[i], tof[j], tof[k], tof[l]) for i, j, k, l in dihedrals},
        "impropers": {tuple(tof[n] for n in q) for q in stream.impropers},
    }
    return resolved_bonds, resolved_angles, resolved_dihedrals, \
        resolved_impropers, angles, dihedrals, used


def _colinear_vsite_a(dist_A):
    """GROMACS [ virtual_sites2 ] funct-2 distance parameter (nm) for a CHARMM
    COLINEAR lone pair. The sign is inverted relative to the CHARMM distance:
    CHARMM measures from the first host away from the second, GROMACS measures
    along host1->host2, so a = -(dist in nm). This reproduces the CHARMM-GUI /
    cgenff_charmm2gmx conversion exactly."""
    return -(dist_A * ANG)


def _lonepair_itp_lines(stream, idx, angles, pairs):
    """[ virtual_sites2 ] and [ exclusions ] for the stream's lone pairs.

    A lone pair has no bond, so GROMACS generates no exclusions for it; it must
    instead be given, explicitly, the same nonbonded exclusions as the host
    atom it sits on -- the host, and the host's 1-2/1-3/1-4 partners. Only
    COLINEAR lone pairs are emitted, the only kind CGenFF produces; any other
    kind is refused rather than dropped, since a missing virtual site is a
    silently-wrong topology.
    """
    if not stream.lonepairs:
        return []

    adj = {a.name: set() for a in stream.atoms}
    for a, b in stream.bonds:
        adj[a].add(b)
        adj[b].add(a)

    vlines = ["", "[ virtual_sites2 ]",
              "; site  host   ref  funct       a(nm)"]
    elines = ["", "[ exclusions ]", ";   ai   aj ..."]
    for lp in stream.lonepairs:
        if not lp.kind.upper().startswith("COLI"):
            raise ValueError(
                "lone pair %s uses %r construction; only COLINEAR lone pairs "
                "are supported (the only kind CGenFF emits). A %r lone pair "
                "needs a virtual_sites3 term this converter does not write, and "
                "dropping it silently would be wrong." % (lp.name, lp.kind,
                                                          lp.kind))
        if len(lp.hosts) < 2:
            raise ValueError("colinear lone pair %s needs two host atoms, got %r"
                             % (lp.name, lp.hosts))
        if lp.hosts[0] == lp.hosts[1]:
            # a virtual_sites2 with i == j has a zero direction vector, which is
            # a division by zero in grompp/mdrun -- refuse it here instead.
            raise ValueError("colinear lone pair %s needs two DISTINCT host "
                             "atoms, but both are %r" % (lp.name, lp.hosts[0]))
        for n in (lp.name, lp.hosts[0], lp.hosts[1]):
            if n not in idx:
                raise ValueError("lone pair %s references unknown atom %r"
                                 % (lp.name, n))
        a_nm = _colinear_vsite_a(lp.dist)
        vlines.append("%5d %5d %5d    2 %11.5f"
                      % (idx[lp.name], idx[lp.hosts[0]], idx[lp.hosts[1]], a_nm))

        host = lp.hosts[0]
        targets = {host} | set(adj[host])                    # host and 1-2
        for i, _j, k in angles:                              # 1-3
            if i == host:
                targets.add(k)
            elif k == host:
                targets.add(i)
        for i, j in pairs:                                   # 1-4
            if i == host:
                targets.add(j)
            elif j == host:
                targets.add(i)
        targets.discard(lp.name)
        ordered = sorted(idx[t] for t in targets)
        elines.append("%5d %s" % (idx[lp.name], " ".join(str(t) for t in ordered)))
    return vlines + elines


def write_atomtypes(path, stream):
    """Write the ligand's new atom types and their 1-4 pair types.

    These must be #included before any [ moleculetype ] that uses them, so they
    live in their own file rather than inside LIG.itp -- GROMACS reads
    [ atomtypes ] only at the top level.
    """
    used_types = sorted({a.type for a in stream.atoms})
    lp_types = stream.lp_types()

    def nb_of(t):
        # A lone-pair type carries no Lennard-Jones term, so a missing NONBONDED
        # entry is normal for it (zeros), an error for any real atom type.
        if t in stream.nb:
            return stream.nb[t]
        if t in lp_types:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        raise ValueError("no NONBONDED (LJ) parameters for atom type %r" % t)

    lines = ["; ligand atom types, generated by lamellyx.cgenff",
             "; charges are per-atom in the .itp; the 0.0 here is required "
             "filler", "[ atomtypes ]",
             ";name  at.num        mass    charge ptype        sigma      epsilon"]
    for t in used_types:
        eps, rmin2, _, _, _ = nb_of(t)
        sigma, eps_kj = lj_to_gmx(eps, rmin2)
        if t in lp_types:
            z, mass = 0, 0.0             # a virtual site has no element or mass
        else:
            mass = stream.mass_of(t)
            z = _atomic_number(mass)
        lines.append("%-6s %6d %11.5f %9.4f     A %12.6e %12.6e"
                     % (t, z, mass, 0.0, sigma, eps_kj))

    # 1-4 pair types for every ordered pair of used types (Lorentz-Berthelot on
    # the 1-4 radii and Berthelot on the 1-4 wells, in CHARMM's convention).
    lines += ["", "[ pairtypes ]",
              ";  i      j func        sigma1-4     epsilon1-4"]
    from math import sqrt
    for a in range(len(used_types)):
        for b in range(a, len(used_types)):
            ta, tb = used_types[a], used_types[b]
            _, _, ea, ra, _ = nb_of(ta)
            _, _, eb, rb, _ = nb_of(tb)
            sigma = (ra + rb) * ANG / (2.0 ** (1.0 / 6.0))
            eps = sqrt(ea * eb) * KCAL
            lines.append("%-6s %-6s   1 %12.6e %12.6e" % (ta, tb, sigma, eps))
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return used_types


def write_itp(path, stream, resolved, resname=None, nrexcl=3):
    """Write LIG.itp: moleculetype, atoms, and every bonded term with its
    parameters written out explicitly, so the file is self-contained given the
    atom types."""
    (rbonds, rangles, rdihedrals, rimpropers,
     angles, dihedrals, _used) = resolved
    resname = resname or stream.resname
    names = [a.name for a in stream.atoms]

    L = ["; ligand topology, generated by lamellyx.cgenff",
         "; parameters converted from CGenFF; NOT yet checked by gmx grompp",
         "", "[ moleculetype ]", "; name        nrexcl",
         "%-13s %d" % (resname, nrexcl), "",
         "[ atoms ]",
         ";  nr   type resnr residue atom cgnr     charge       mass"]
    lp_names = stream.lp_names()
    for i, a in enumerate(stream.atoms, start=1):
        mass = 0.0 if a.name in lp_names else stream.mass_of(a.type)
        L.append("%5d %6s %5d %7s %5s %4d %10.4f %10.4f"
                 % (i, a.type, 1, resname, a.name, i, a.charge, mass))

    idx = {n: i + 1 for i, n in enumerate(names)}

    L += ["", "[ bonds ]", ";  ai    aj func           b0(nm)     kb"]
    for a, b, p in rbonds:
        b0, kb = bond_to_gmx(p[0], p[1])
        L.append("%5d %5d    1 %14.6e %14.6e" % (idx[a], idx[b], b0, kb))

    pairs = one_four_pairs(names, stream.bonds, dihedrals)
    L += ["", "[ pairs ]", ";  ai    aj func"]
    for a, b in pairs:
        L.append("%5d %5d    1" % (idx[a], idx[b]))

    L += ["", "[ angles ]",
          ";  ai    aj    ak func      theta0        ktheta        ub0        kub"]
    for i, j, k, p in rangles:
        th0, kth, ub0, kub = angle_to_gmx(p[0], p[1], p[2], p[3])
        L.append("%5d %5d %5d    5 %11.4f %13.6e %11.6e %13.6e"
                 % (idx[i], idx[j], idx[k], th0, kth, ub0, kub))

    L += ["", "[ dihedrals ]",
          ";  ai    aj    ak    al func       phi0         kphi mult"]
    for i, j, k, l, terms in rdihedrals:
        for kchi, mult, delta, _pen in terms:
            phi0, kphi = dihedral_to_gmx(kchi, delta)
            L.append("%5d %5d %5d %5d    9 %11.4f %13.6e %4d"
                     % (idx[i], idx[j], idx[k], idx[l], phi0, kphi, mult))

    if rimpropers:
        L += ["", "[ dihedrals ]", "; impropers (type 2)",
              ";  ai    aj    ak    al func        xi0          kxi"]
        for q, p in rimpropers:
            xi0, kxi = improper_to_gmx(p[0], p[1])
            a, b, c, d = (idx[n] for n in q)
            L.append("%5d %5d %5d %5d    2 %11.4f %13.6e" % (a, b, c, d, xi0, kxi))

    # lone pairs: constructed virtual sites and their explicit exclusions
    L += _lonepair_itp_lines(stream, idx, angles, pairs)

    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")


# -- top-level entry point ----------------------------------------------------


def generate_ligand_topology(str_source, out_dir, resname=None, cgenff_ff=None,
                             penalty_flag=None):
    """Convert a CGenFF .str into GROMACS ligand topology in `out_dir`.

    A real ParamChem stream is only a supplement to the base CGenFF force field,
    so `cgenff_ff` -- a directory with top_all36_cgenff.rtf and
    par_all36_cgenff.prm, or a (rtf, prm) tuple -- must be given for anything but
    a hand-built, fully self-contained stream. The stream's analogy-generated
    parameters override the base FF.

    The CGenFF penalty is REPORTED, never refused: the report carries the worst
    penalty and the worst offenders, and `penalty_flag` (optional) additionally
    lists every parameter above it. Most biological ligands lean on analogy, so
    refusing on penalty would reject the normal case; judging the number is left
    to the caller.

    Writes `<RESN>.itp` and `<RESN>_atomtypes.itp` and returns a report.
    """
    stream = parse_stream(str_source)
    if resname:
        stream.resname = resname
    resname = stream.resname
    _check_resname(resname)

    if cgenff_ff is not None:
        apply_base_ff(stream, load_cgenff_ff(cgenff_ff))

    resolved = _resolve_bonded(stream, base_ff_used=cgenff_ff is not None)
    used = resolved[6]
    penalties = collect_penalties(
        stream, used["bonds"], used["angles"], used["dihedrals"],
        used["impropers"])
    worst = penalties[0][1] if penalties else 0.0
    flagged = ([(lbl, round(p, 2)) for lbl, p in penalties if p > penalty_flag]
               if penalty_flag is not None else [])

    os.makedirs(out_dir, exist_ok=True)
    itp = os.path.join(out_dir, resname + ".itp")
    atp = os.path.join(out_dir, resname + "_atomtypes.itp")
    used_types = write_atomtypes(atp, stream)
    write_itp(itp, stream, resolved, resname=resname)

    net = sum(a.charge for a in stream.atoms)
    # A CGenFF ligand's partial charges sum to its formal (integer) charge by
    # construction. A net that is not close to an integer means the stream was
    # truncated or its charges hand-edited -- report it, but do not refuse: the
    # caller may have a reason, and grompp will complain anyway.
    charge_note = (None if abs(net - round(net)) < 0.01 else
                   "net charge %.4f is not close to an integer; a CGenFF "
                   "ligand's partial charges should sum to its formal charge, "
                   "so the stream may be truncated or its charges edited" % net)
    return {
        "ok": True,
        "resname": resname,
        "n_atoms": len(stream.atoms),
        "atom_types": used_types,
        "net_charge": round(net, 4) + 0.0,
        "net_charge_is_integer": charge_note is None,
        "charge_note": charge_note,
        "n_bonds": len(stream.bonds),
        "n_angles": len(resolved[4]),
        "n_dihedrals": sum(len(t or []) for *_ , t in resolved[2]),
        "n_impropers": len(stream.impropers),
        "n_lonepairs": len(stream.lonepairs),
        "base_ff_merged": cgenff_ff is not None,
        # The CGenFF penalty is reported, not gated: a high number means CGenFF
        # reached far by analogy and the energies deserve a look, but it is the
        # caller's call, not a build failure.
        "worst_penalty": round(worst, 2),
        "penalties": [(lbl, round(p, 2)) for lbl, p in penalties[:12]],
        "flagged_above_%s"
        % (penalty_flag if penalty_flag is not None else "off"): flagged,
        "files": [os.path.basename(itp), os.path.basename(atp)],
        "itp": itp,
        "atomtypes_itp": atp,
        "note": ("Ligand topology only. To embed it, #include the "
                 "_atomtypes.itp before any [ moleculetype ], #include the "
                 ".itp, place the ligand in the coordinate file and add it to "
                 "[ molecules ]. NOT yet checked by gmx grompp."),
    }


# -- second entry point: drive the licensed CGenFF binary ---------------------
# The .str above is the entry point that needs no licence: a user pastes their
# molecule into the ParamChem web server and downloads the stream. This second
# entry point drives the *licensed* cgenff program on a mol2/sdf directly, so
# the whole thing can run without the web form -- the same reason pdb2gmx.py
# exists. Like pdb2gmx, we drive the tool, we do not reimplement it.
#
# > The exact command line of the licensed binary is NOT verified here -- there
# > is no licensed copy on this machine, and whether the supervisor has one is
# > still an open question. The invocation is therefore configurable, defaults
# > to the documented "molecule in, stream on stdout" form, and everything
# > around it is tested against a stub. Confirm the real CLI before trusting a
# > live run. The .str path needs none of this.

CGENFF_CANDIDATES = ("cgenff", "cgenff.exe")


class CGenFFNotFound(RuntimeError):
    pass


def find_cgenff(cgenff=None):
    """Locate the cgenff binary. Raises CGenFFNotFound naming what it tried.

    Unlike GROMACS, cgenff has no reliable `--version` to probe, so this only
    checks that a named executable resolves -- it does not run it.
    """
    tried = []
    names = [cgenff] if cgenff else []
    env = os.environ.get("CGENFF")
    if env:
        names.append(env)
    names += list(CGENFF_CANDIDATES)
    for name in names:
        if not name:
            continue
        tried.append(name)
        path = name if os.path.sep in name else shutil.which(name)
        if path and os.path.exists(path):
            return path
    raise CGenFFNotFound(
        "no cgenff binary found. Tried: %s. Set CGENFF to the full path or "
        "pass cgenff=... . Driving the binary needs a CGenFF licence; if you "
        "do not have one, obtain a .str from the ParamChem web server and use "
        "generate_ligand_topology() directly -- that needs no binary."
        % ", ".join(tried))


def run_cgenff(molecule, out_str, cgenff=None, output_flag=None, extra_args=()):
    """Run cgenff on a mol2/sdf and write its stream file to `out_str`.

    By default the stream is captured from stdout, which is the documented
    behaviour of the SilcsBio program (`cgenff molecule.mol2 > out.str`). If a
    particular build writes to a file instead, pass `output_flag` (e.g. "-f")
    and the command becomes `cgenff <flag> out_str <extra> molecule`.
    """
    exe = find_cgenff(cgenff)
    molecule = os.path.abspath(molecule)
    if not os.path.exists(molecule):
        raise FileNotFoundError(molecule)
    out_str = os.path.abspath(out_str)

    cmd = [exe]
    if output_flag:
        cmd += [output_flag, out_str]
    cmd += list(extra_args) + [molecule]

    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError("cgenff did not finish within 600 s (a hang?). "
                           "command: %s" % " ".join(cmd))
    if run.returncode != 0:
        tail = (run.stderr or run.stdout or "").strip().splitlines()[-25:]
        raise RuntimeError(
            "cgenff failed (exit %d).\ncommand: %s\n\n%s"
            % (run.returncode, " ".join(cmd), "\n".join(tail)))

    if not output_flag:
        if not (run.stdout or "").strip():
            raise RuntimeError(
                "cgenff exited 0 but wrote nothing to stdout. If this build "
                "writes to a file, pass output_flag=... .\ncommand: %s"
                % " ".join(cmd))
        with open(out_str, "w", newline="\n") as fh:
            fh.write(run.stdout)
    elif not os.path.exists(out_str):
        raise RuntimeError(
            "cgenff exited 0 but wrote no %s.\ncommand: %s" % (out_str,
                                                              " ".join(cmd)))
    return out_str


def generate_ligand_topology_from_molecule(molecule, out_dir, resname=None,
                                          cgenff_ff=None, penalty_flag=None,
                                          cgenff=None, output_flag=None,
                                          extra_args=(), keep_str=True):
    """mol2/sdf -> cgenff binary -> .str -> GROMACS ligand topology.

    The one-call path when a CGenFF licence is available. Everything after the
    binary is the same tested code the .str entry point uses. Returns the same
    report, plus where the intermediate .str went.
    """
    work = tempfile.mkdtemp(prefix="lamellyx_cgenff_")
    tmp_str = os.path.join(work, "ligand.str")
    try:
        run_cgenff(molecule, tmp_str, cgenff=cgenff, output_flag=output_flag,
                   extra_args=extra_args)
        report = generate_ligand_topology(
            tmp_str, out_dir, resname=resname, cgenff_ff=cgenff_ff,
            penalty_flag=penalty_flag)
        if keep_str:
            kept = os.path.join(out_dir, report["resname"] + ".str")
            shutil.copy(tmp_str, kept)
            report["str"] = kept
        report["cgenff"] = find_cgenff(cgenff)
        report["molecule"] = os.path.abspath(molecule)
        return report
    finally:
        shutil.rmtree(work, ignore_errors=True)
