"""Place a ligand into an already-built system.

The converter (`cgenff.py`) makes a ligand's *topology*; this module puts the
ligand's *coordinates* into a system that lamellyx (or CHARMM-GUI) already
built, and extends the bookkeeping around it: the .gro, topol.top, index.ndx.

The contract, chosen deliberately, is the same one the protein path uses: the
caller supplies a ligand PDB **already positioned in the box frame** -- from
docking, or placed by hand. Nothing here guesses where in the protein a drug
should sit; that is a modelling decision, not a packing one. What this does
guarantee is that the atoms end up in the exact order the .itp declares (the
invariant the whole package rests on), that the counts and index groups stay
consistent, and that a new atom type never silently redefines an existing one
-- the one mistake that turns into a hard `gmx grompp` error.

> Like everything else in lamellyx, the result has NOT been through
> `gmx grompp`. The checks here are contact geometry and bookkeeping. The
> ligand's atom types are deduplicated against the target force field so the
> commonest grompp error cannot occur, but grompp is still the real test.

Placement currently targets a **protein** system -- one with a SOLU index group
-- because a drug belongs in the solute temperature-coupling group and that is
where the v2 goal (protein + drug) lives. A bare bilayer has no SOLU group and
is refused with that explanation rather than built into something grompp cannot
couple.
"""

from __future__ import annotations

import os
import re
import shutil

import numpy as np

from . import fileio, topology, validate

_SECTION = re.compile(r"^\s*\[\s*(.+?)\s*\]")
_INCLUDE = re.compile(r'^\s*#include\s+"(.+?)"')


def parse_topol(path):
    """Return (includes, molecules) from a topol.top.

    `includes` is the ordered list of #included basenames with forcefield.itp
    dropped (write_topol re-adds it); `molecules` the ordered [(name, count)].
    """
    includes, molecules, section = [], [], None
    with open(path) as fh:
        for raw in fh:
            line = raw.split(";")[0].rstrip()
            if not line.strip():
                continue
            m = _INCLUDE.match(line)
            if m:
                includes.append(os.path.basename(m.group(1)))
                continue
            sm = _SECTION.match(line)
            if sm:
                section = sm.group(1).lower()
                continue
            if section == "molecules":
                parts = line.split()
                molecules.append((parts[0], int(parts[1])))
    includes = [i for i in includes if i != "forcefield.itp"]
    return includes, molecules


def parse_index(path):
    """Return {group name: [1-based indices]} from an index.ndx."""
    groups, cur = {}, None
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m = _SECTION.match(line)
            if m:
                cur = m.group(1)
                groups[cur] = []
                continue
            if cur is not None:
                groups[cur].extend(int(x) for x in line.split())
    return groups


def existing_atomtypes(toppar_dir):
    """Every atom type the system already defines, name -> the parameter line.

    Read so a ligand's new types can be checked against them: redefining an
    atom type is a hard grompp error, so a collision must be caught here.
    """
    types = {}
    for fn in sorted(os.listdir(toppar_dir)):
        if not fn.endswith(".itp"):
            continue
        section = None
        with open(os.path.join(toppar_dir, fn)) as fh:
            for raw in fh:
                line = raw.split(";")[0].rstrip()
                if not line.strip():
                    continue
                m = _SECTION.match(line)
                if m:
                    section = m.group(1).lower()
                    continue
                if section == "atomtypes":
                    parts = line.split()
                    if parts:
                        types.setdefault(parts[0], line.strip())
    return types


def existing_pairtypes(toppar_dir):
    """The unordered type pairs the system already gives a 1-4 [ pairtypes ]
    entry. A ligand pairtype that merely repeats one of these is dropped rather
    than written a second time -- a duplicate [ pairtypes ] entry is a grompp
    override warning at best."""
    pairs = set()
    for fn in sorted(os.listdir(toppar_dir)):
        if not fn.endswith(".itp"):
            continue
        section = None
        with open(os.path.join(toppar_dir, fn)) as fh:
            for raw in fh:
                line = raw.split(";")[0].rstrip()
                if not line.strip():
                    continue
                m = _SECTION.match(line)
                if m:
                    section = m.group(1).lower()
                    continue
                if section == "pairtypes":
                    parts = line.split()
                    if len(parts) >= 2:
                        pairs.add(tuple(sorted(parts[:2])))
    return pairs


def _cols_match(a, b, tol=1e-4):
    """Two atom-type column lists match if their numbers agree within a relative
    tolerance and their non-numeric columns (the ptype letter) agree exactly.
    Comparing numerically means the same value formatted differently -- our
    converter's `3.634867e-01` versus a CHARMM-GUI file's `0.36349` -- is treated
    as identical, not a spurious redefinition conflict that would refuse a valid
    placement. A genuinely different type differs by far more than the tolerance."""
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            if x != y:
                return False
            continue
        if abs(fx - fy) > tol * max(1.0, abs(fx), abs(fy)):
            return False
    return True


def _dedup_atomtypes(lig_atomtypes_path, existing, existing_pts, out_path):
    """Write the ligand's atomtypes file with atom types and pair types the
    system already defines removed.

    Redefining an atom type with different parameters is a hard grompp error, so
    a genuine conflict there is refused. A pair type the force field already
    gives is simply dropped -- the force field's is authoritative, and this
    avoids emitting a duplicate [ pairtypes ] entry (dropping by the type pair,
    not by comparing values, so different float formatting is never mistaken for
    a conflict)."""
    out_lines, section = [], None
    kept, dropped, conflicts = [], [], []
    with open(lig_atomtypes_path) as fh:
        for raw in fh:
            stripped = raw.rstrip("\n")
            code = raw.split(";")[0].rstrip()
            m = _SECTION.match(code)
            if m:
                section = m.group(1).lower()
                out_lines.append(stripped)
                continue
            if section == "atomtypes" and code.strip():
                name = code.split()[0]
                if name not in existing:
                    kept.append(stripped)
                    out_lines.append(stripped)
                elif _cols_match(code.split()[1:], existing[name].split()[1:]):
                    dropped.append(name)                     # same params -> drop
                else:
                    conflicts.append(name)
            elif section == "pairtypes" and code.strip():
                if tuple(sorted(code.split()[:2])) not in existing_pts:
                    out_lines.append(stripped)               # a new pair -> keep
                # else: the force field already has it -> drop
            else:
                out_lines.append(stripped)
    if conflicts:
        raise ValueError(
            "ligand atom type(s) %s already exist in the target force field "
            "with different parameters. Redefining them would give grompp an "
            "error, and using them silently would be wrong. Rename the ligand "
            "types or build against the force field they came from."
            % ", ".join(conflicts))
    with open(out_path, "w", newline="\n") as fh:
        fh.write("\n".join(out_lines) + "\n")
    return kept, dropped


def order_to_itp(pdb_atoms, itp_mol, resname):
    """Reorder ligand PDB atoms into the .itp's atom order, by atom name.

    The .itp defines the order the .gro must use; a docked or hand-placed PDB
    can list atoms in any order. Matching by name is exact for a ligand, whose
    atom names are unique within its single residue. A missing or extra atom is
    refused -- a topology and a coordinate set that disagree on atom content is
    the exact failure the package exists to prevent.
    """
    want = list(itp_mol.atomname)
    have = list(pdb_atoms.name)
    if len(set(have)) != len(have):
        dup = sorted({n for n in have if have.count(n) > 1})
        raise ValueError("ligand PDB has duplicate atom names: %s" % ", ".join(dup))
    missing = [n for n in want if n not in have]
    extra = [n for n in have if n not in want]
    if missing or extra:
        raise ValueError(
            "ligand PDB atoms do not match the topology.\n  missing (in .itp, "
            "not in PDB): %s\n  extra (in PDB, not in .itp): %s"
            % (", ".join(missing) or "none", ", ".join(extra) or "none"))
    index = {n: i for i, n in enumerate(have)}
    out = pdb_atoms[[index[n] for n in want]]
    n = len(out)
    out.resname = np.full(n, resname[:6], dtype="<U6")
    out.segid = np.full(n, "LIG", dtype="<U6")
    out.chain = np.full(n, "L", dtype="<U1")
    return out


def parse_itp_vsites(path):
    """`[ virtual_sites2 ]` funct-2 rows from a ligand .itp, as
    (site, host1, host2, a_nm) with 1-based atom indices. Empty if the ligand
    has no lone pairs. `topology.parse_itp` ignores this section, so it is read
    here separately, only where placement needs it.
    """
    out, section = [], None
    with open(path) as fh:
        for raw in fh:
            line = raw.split(";")[0].rstrip()
            if not line.strip():
                continue
            m = _SECTION.match(line)
            if m:
                section = m.group(1).lower()
                continue
            if section == "virtual_sites2":
                p = line.split()
                if len(p) >= 5 and p[3] == "2":
                    try:
                        out.append(
                            (int(p[0]), int(p[1]), int(p[2]), float(p[4])))
                    except ValueError:
                        pass    # not a well-formed funct-2 row; skip it
    return out


def reconstruct_vsites(pdb_atoms, lig_mol, ligand_itp, resname):
    """Fill in coordinates for any lone pair the .itp declares but the PDB lacks.

    A lone pair is a constructed particle, not a real atom, so a docked or
    hand-placed ligand PDB never contains it -- yet the topology, and therefore
    the .gro, needs a coordinate for it. Each COLINEAR site is rebuilt from its
    two host atoms with the same funct-2 rule GROMACS uses,
    r = r1 + a*(r2 - r1)/|r2 - r1|, with a in nm (coordinates here are Angstrom,
    so a is scaled by 10). Returns (atoms, [reconstructed names]).
    """
    vsites = parse_itp_vsites(ligand_itp)
    if not vsites:
        return pdb_atoms, []
    names = list(lig_mol.atomname)                     # 1-based: names[i-1]
    have = {n: k for k, n in enumerate(pdb_atoms.name)}
    add_name, add_xyz = [], []
    for site, h1, h2, a_nm in vsites:
        sname = names[site - 1]
        if sname in have:
            continue                                   # PDB already carries it
        n1, n2 = names[h1 - 1], names[h2 - 1]
        missing = [n for n in (n1, n2) if n not in have]
        if missing:
            raise ValueError(
                "cannot reconstruct lone pair %s: its host atom(s) %s are not "
                "in the ligand PDB" % (sname, ", ".join(missing)))
        r1, r2 = pdb_atoms.xyz[have[n1]], pdb_atoms.xyz[have[n2]]
        u = r2 - r1
        d = float(np.linalg.norm(u))
        if d < 1e-6:
            raise ValueError("lone pair %s: its host atoms are coincident" % sname)
        add_name.append(sname)
        add_xyz.append(r1 + (a_nm * 10.0) * (u / d))
    if not add_name:
        return pdb_atoms, []
    extra = fileio.Atoms(add_name, [resname] * len(add_name),
                         [1] * len(add_name), np.array(add_xyz))
    return fileio.Atoms.concat([pdb_atoms, extra]), add_name


def add_ligand(system_dir, ligand_pdb, ligand_itp, out_dir,
               atomtypes_itp=None, resname=None):
    """Insert a positioned ligand into a built protein system.

    `system_dir` holds step5_input.gro, topol.top, index.ndx and toppar/.
    `ligand_pdb` is the ligand positioned in the same frame. `ligand_itp` and
    `atomtypes_itp` (defaulting to <itp stem>_atomtypes.itp) are what
    cgenff.generate_ligand_topology wrote. The result is written to `out_dir`,
    which must not already exist -- the input system is never modified.
    """
    for need, p in (("system_dir", system_dir), ("ligand_pdb", ligand_pdb),
                    ("ligand_itp", ligand_itp)):
        if not os.path.exists(p):
            raise FileNotFoundError("%s: %s" % (need, p))
    if os.path.exists(out_dir):
        raise ValueError("out_dir already exists: %s (refusing to overwrite)"
                         % out_dir)
    if atomtypes_itp is None:
        atomtypes_itp = os.path.splitext(ligand_itp)[0] + "_atomtypes.itp"
    if not os.path.exists(atomtypes_itp):
        raise FileNotFoundError("atomtypes file: %s" % atomtypes_itp)

    try:
        lig_tops = topology.parse_itp(ligand_itp)
    except (ValueError, IndexError) as exc:
        raise ValueError("could not parse the ligand .itp %s -- is it a valid "
                         "GROMACS topology? (%s)" % (ligand_itp, exc))
    if len(lig_tops) != 1:
        raise ValueError("expected one moleculetype in %s, found %s"
                         % (ligand_itp, sorted(lig_tops)))
    lig_name = next(iter(lig_tops))
    lig_mol = lig_tops[lig_name]
    resname = resname or lig_name

    gro = os.path.join(system_dir, "step5_input.gro")
    atoms, box = fileio.read_gro(gro)
    idx = parse_index(os.path.join(system_dir, "index.ndx"))
    if "SOLU" not in idx or not idx["SOLU"]:
        raise ValueError(
            "the system has no (non-empty) SOLU group -- ligand placement "
            "targets protein systems, where a drug joins the solute coupling "
            "group. A bare bilayer has no SOLU group to join.")
    npro, nmemb = len(idx["SOLU"]), len(idx.get("MEMB", []))
    n = len(atoms)
    if npro + nmemb + len(idx.get("SOLV", [])) != n:
        raise ValueError("index groups cover %d atoms but the .gro has %d"
                         % (npro + nmemb + len(idx.get("SOLV", [])), n))

    includes, molecules = parse_topol(os.path.join(system_dir, "topol.top"))
    toppar = os.path.join(system_dir, "toppar")
    sys_tops = topology.load_toppar(toppar)

    # The ligand's moleculetype must not collide with a system molecule: its
    # .itp would overwrite the system's in toppar/, and [ molecules ] would list
    # the name twice against a single, wrong topology -- silent corruption.
    if lig_name in sys_tops:
        raise ValueError(
            "the ligand's moleculetype %r already names a molecule in the "
            "target system, so placing it would overwrite that molecule's "
            "topology. Give the ligand a different name when you generate it "
            "(generate_ligand_topology(resname=...))." % lig_name)

    # find where the protein block ends in [ molecules ], and insert the ligand
    # there so it lands inside SOLU rather than after the water.
    acc, insert_at = 0, None
    for k, (name, count) in enumerate(molecules):
        if name not in sys_tops:
            raise ValueError("topol.top names molecule %r with no .itp in "
                             "toppar/" % name)
        acc += count * sys_tops[name].natoms
        if acc == npro:
            insert_at = k + 1
            break
        if acc > npro:
            break
    if insert_at is None:
        raise ValueError(
            "the SOLU group (%d atoms) does not end on a molecule boundary; "
            "the topology and index.ndx disagree about the protein." % npro)

    # coordinates: match to the .itp order and splice in after the protein.
    # A lone pair is in the .itp but never in a docked PDB, so its coordinate is
    # reconstructed from its hosts before ordering rather than demanded.
    lig_resid = int(atoms.resid[:npro].max()) + 1 if npro else 1
    lig_pdb, lp_rebuilt = reconstruct_vsites(
        fileio.read_pdb(ligand_pdb), lig_mol, ligand_itp, resname)
    lig_atoms = order_to_itp(lig_pdb, lig_mol, resname)
    lig_atoms.resid = np.full(len(lig_atoms), lig_resid, dtype=np.int64)
    if len(lig_atoms) != lig_mol.natoms:
        raise ValueError("ligand PDB has %d atoms, .itp has %d"
                         % (len(lig_atoms), lig_mol.natoms))
    nlig = len(lig_atoms)
    new_atoms = fileio.Atoms.concat([atoms[:npro], lig_atoms, atoms[npro:]])

    # assemble the output directory from a copy, then edit the copy
    shutil.copytree(system_dir, out_dir)
    out_toppar = os.path.join(out_dir, "toppar")
    shutil.copy(ligand_itp, os.path.join(out_toppar, lig_name + ".itp"))
    at_out = os.path.join(out_toppar, lig_name + "_atomtypes.itp")
    kept, dropped = _dedup_atomtypes(
        atomtypes_itp, existing_atomtypes(toppar),
        existing_pairtypes(toppar), at_out)

    fileio.write_gro(os.path.join(out_dir, "step5_input.gro"), new_atoms, box,
                     title="lamellyx: +%s" % lig_name)

    new_includes = ([lig_name + "_atomtypes.itp"] + includes
                    + [lig_name + ".itp"])
    # [ molecules ] names moleculetypes, so it must use the .itp's moleculetype
    # (lig_name) -- not a residue-name override, which only labels the residue in
    # the coordinates and has no matching [ moleculetype ] of its own.
    new_molecules = (molecules[:insert_at] + [(lig_name, 1)]
                     + molecules[insert_at:])
    topology.write_topol(os.path.join(out_dir, "topol.top"), new_includes,
                         new_molecules, system_name=os.path.basename(out_dir))

    topology.write_index(
        os.path.join(out_dir, "index.ndx"),
        topology.standard_groups(len(new_atoms), slice(0, npro + nlig),
                                 slice(npro + nlig, npro + nlig + nmemb),
                                 slice(npro + nlig + nmemb, len(new_atoms))))

    # verify: closest approach of the placed ligand to everything else, so a
    # clash from bad placement is reported now, not discovered by minimisation.
    lig_slice = np.zeros(len(new_atoms), dtype=bool)
    lig_slice[npro:npro + nlig] = True
    rep = validate.contacts(new_atoms, box,
                            groups={"LIG": lig_slice, "rest": ~lig_slice})
    ligand_contact = rep.by_group.get("LIG-rest", float("inf"))

    net = (sum(c * sys_tops[m].total_charge for m, c in molecules)
           + lig_mol.total_charge)
    return {
        "ok": True,
        "output_dir": os.path.abspath(out_dir),
        "ligand": lig_name,
        "ligand_atoms": nlig,
        "atom_total": len(new_atoms),
        "net_charge": round(float(net), 4) + 0.0,
        "atomtypes_added": [ln.split()[0] for ln in kept],
        "atomtypes_already_present": dropped,
        "lone_pairs_reconstructed": lp_rebuilt,
        "closest_ligand_contact_A": round(float(ligand_contact), 3),
        "molecules": new_molecules,
        "note": ("Ligand placed and bookkept. Atom types deduplicated against "
                 "the force field. NOT yet checked by gmx grompp -- run it "
                 "before trusting the energies."),
    }
