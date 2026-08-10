"""Generate protein topology by driving GROMACS' `pdb2gmx`.

The package has never generated force-field parameters, and it still does not:
assigning atom types, charges and bonded terms to a sequence is what pdb2gmx
and CHARMM-GUI exist for, and writing a third one would be a large piece of
work whose failure mode is silently wrong energies. What this module removes
is the *dependency on CHARMM-GUI* for that step, so a new sequence can be taken
end to end from a script on any machine with GROMACS installed.

What comes out is a `toppar/` directory of per-chain `.itp` files named the way
the builder expects (PROA, PROB, ...), which is the half of a `reference_dir`
that describes the protein.

    from lamellyx.pdb2gmx import generate_topology
    generate_topology("hcn4.pdb", "topology_out", forcefield="charmm36-jul2022")

> GROMACS is required and is not bundled. On a machine without it this raises
> with the paths it looked in, rather than failing later and obscurely.

Everything here is tested against a stub `gmx` that emits a known topology --
discovery, the command line, the parsing, the renaming, the error when it is
missing. GROMACS itself is not tested here, because it is not installed here.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

from . import topology

# pdb2gmx names chains Protein_chain_A, Protein_chain_B, ... The builder wants
# PROA, PROB, ... to match what CHARMM-GUI produces, so that a topology from
# either source drops into the same place.
_CHAIN_RE = re.compile(r"^Protein(?:_chain_(\w+))?$")

CANDIDATES = ("gmx", "gmx_mpi", "gmx_d", "gmx_mpi_d")


class GromacsNotFound(RuntimeError):
    pass


def find_gromacs(gmx=None):
    """Locate a usable `gmx`. Raises GromacsNotFound with what it tried."""
    tried = []
    names = [gmx] if gmx else []
    env = os.environ.get("GMX")
    if env:
        names.append(env)
    names += list(CANDIDATES)

    for name in names:
        if not name:
            continue
        tried.append(name)
        path = name if os.path.sep in name else shutil.which(name)
        if not path:
            continue
        try:
            out = subprocess.run([path, "--version"], capture_output=True,
                                 text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0:
            first = ""
            for line in (out.stdout or "").splitlines():
                if "VERSION" in line.upper():
                    first = line.strip()
                    break
            return path, first or "unknown version"

    raise GromacsNotFound(
        "no working GROMACS found. Tried: %s. Set GMX to the full path, or "
        "pass gmx=... . Generating protein topology needs GROMACS; the rest "
        "of this package does not." % ", ".join(tried))


def _split_topology(top_path, work_dir):
    """Pull every [ moleculetype ] out of what pdb2gmx wrote.

    pdb2gmx puts a single chain inline in topol.top and multiple chains in
    sibling topol_Protein_chain_X.itp files, so both shapes have to be
    handled or a monomer and a tetramer behave differently.
    """
    texts = []
    with open(top_path) as fh:
        main = fh.read()
    texts.append(main)
    for line in main.splitlines():
        m = re.match(r'^\s*#include\s+"(.+?)"', line)
        if not m:
            continue
        inc = m.group(1)
        cand = os.path.join(work_dir, os.path.basename(inc))
        if os.path.exists(cand) and os.path.basename(inc).startswith("topol"):
            with open(cand) as fh:
                texts.append(fh.read())

    blocks = []
    for text in texts:
        parts = re.split(r"(?m)^\s*\[\s*moleculetype\s*\]", text)
        for part in parts[1:]:
            # Stop at the next top-level section that cannot belong to a
            # molecule, so a [ system ] block does not get swallowed.
            cut = re.search(r"(?m)^\s*\[\s*(system|molecules)\s*\]", part)
            body = part[:cut.start()] if cut else part
            name = None
            for line in body.splitlines():
                s = line.split(";")[0].strip()
                if s:
                    name = s.split()[0]
                    break
            if name:
                blocks.append((name, "[ moleculetype ]" + body.rstrip() + "\n"))
    return blocks


def generate_topology(protein_pdb, out_dir, forcefield="charmm36-jul2022",
                      water="tip3p", gmx=None, chains=None, ignh=True,
                      extra_args=(), posre=False):
    """Run pdb2gmx and write `out_dir/toppar/PRO*.itp`.

    `chains` renames the molecule types in the order pdb2gmx produced them --
    ("PROA", "PROB", ...) by default, matching CHARMM-GUI.

    Returns a report: the gmx used, the molecule names, atom counts and total
    charges, and where the files went.
    """
    exe, version = find_gromacs(gmx)
    protein_pdb = os.path.abspath(protein_pdb)
    if not os.path.exists(protein_pdb):
        raise FileNotFoundError(protein_pdb)
    out_dir = os.path.abspath(out_dir)
    toppar = os.path.join(out_dir, "toppar")
    os.makedirs(toppar, exist_ok=True)

    work = tempfile.mkdtemp(prefix="lamellyx_pdb2gmx_")
    cmd = [exe, "pdb2gmx",
           "-f", protein_pdb,
           "-o", os.path.join(work, "processed.gro"),
           "-p", os.path.join(work, "topol.top"),
           "-i", os.path.join(work, "posre.itp"),
           "-ff", forcefield,
           "-water", water]
    if ignh:
        # Hydrogens are rebuilt by this package anyway, and a PDB's existing
        # ones are the commonest reason pdb2gmx stops with a naming error.
        cmd.append("-ignh")
    cmd += list(extra_args)

    run = subprocess.run(cmd, capture_output=True, text=True, cwd=work)
    if run.returncode != 0:
        tail = (run.stderr or run.stdout or "").strip().splitlines()[-25:]
        raise RuntimeError(
            "pdb2gmx failed (exit %d).\ncommand: %s\n\n%s"
            % (run.returncode, " ".join(cmd), "\n".join(tail)))

    top_path = os.path.join(work, "topol.top")
    if not os.path.exists(top_path):
        raise RuntimeError("pdb2gmx reported success but wrote no topol.top")

    blocks = _split_topology(top_path, work)
    protein_blocks = [(n, t) for n, t in blocks if _CHAIN_RE.match(n)]
    if not protein_blocks:
        protein_blocks = blocks
    if not protein_blocks:
        raise RuntimeError("no [ moleculetype ] found in what pdb2gmx wrote")

    names = list(chains) if chains else [
        "PRO" + chr(ord("A") + i) for i in range(len(protein_blocks))]
    if len(names) != len(protein_blocks):
        raise ValueError("got %d chains from pdb2gmx but %d names were given"
                         % (len(protein_blocks), len(names)))

    written, report_mols = [], []
    for new, (old, text) in zip(names, protein_blocks):
        # Rename only the molecule's own name line, never anything else that
        # happens to contain the same word.
        lines = text.splitlines()
        for i, line in enumerate(lines):
            s = line.split(";")[0].strip()
            if i and s and not s.startswith("["):
                bits = line.split()
                lines[i] = line.replace(bits[0], new, 1)
                break
        path = os.path.join(toppar, new + ".itp")
        with open(path, "w") as fh:
            fh.write("; renamed from %s by lamellyx.pdb2gmx\n" % old)
            fh.write("\n".join(lines) + "\n")
        written.append(path)

        mol = topology.parse_itp(path)[new]
        report_mols.append({"name": new, "from": old, "atoms": mol.natoms,
                            "charge": round(mol.total_charge, 4)})

    if posre and os.path.exists(os.path.join(work, "posre.itp")):
        shutil.copy(os.path.join(work, "posre.itp"),
                    os.path.join(toppar, "posre.itp"))

    shutil.rmtree(work, ignore_errors=True)
    return {
        "ok": True,
        "gmx": exe,
        "gromacs_version": version,
        "forcefield": forcefield,
        "water": water,
        "output_dir": out_dir,
        "toppar": toppar,
        "molecules": report_mols,
        "files": [os.path.basename(p) for p in written],
        "note": ("These .itp files describe the protein only. A full "
                 "reference_dir also needs lipid, water and ion topologies -- "
                 "the bundled CHARMM36 set covers POPC, TIP3P, K+ and Cl-."),
    }
