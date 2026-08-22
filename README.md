# lamellyx

Lipid bilayers and membrane-protein systems for GROMACS, from Python.

> New here, or prefer not to touch a terminal? Start with
> [QUICKSTART.md](QUICKSTART.md) — double-click a launcher and work in the browser.

CHARMM-GUI is excellent and it is a web form. There is no API, so it cannot be
scripted, batched, put in a pipeline, or driven by a language model. This does
the same job as a function call.

```python
from lamellyx.api import build

build({"output_dir": "popc_box", "x": 8.0, "y": 8.0})
```

```bash
lamellyx build --out popc_box --set x=8 --set y=8
lamellyx dashboard          # or a browser UI, if you prefer one
```

Out comes a directory GROMACS runs as it stands:

```bash
cd popc_box
gmx grompp -f step6.0_minimization.mdp -c step5_input.gro \
           -r step5_input.gro -p topol.top -n index.ndx -o min.tpr
gmx mdrun -deffnm min
```

| File | |
|---|---|
| `step5_input.gro` | coordinates, in the order `topol.top` declares |
| `topol.top` | molecule counts, with the parameters it needs |
| `index.ndx` | the groups the mdp files couple to |
| `step6.0` … `step6.6` | minimisation, then six-stage restrained equilibration |
| `step7_production.mdp` | NPT production, semi-isotropic |
| `build_report.json` | every setting, count and quality check |

## Install

**This is not on PyPI — `pip install lamellyx` will fail.** It is a private
GitHub repository. Clone it and install from the clone:

```bash
git clone https://github.com/Chris-FIFI/lamellyx.git
cd lamellyx
pip install -e .
python -m lamellyx setup <a CHARMM-GUI gromacs directory>
```

`-e` installs it editable, so a `git pull` updates the package with no
reinstall — drop the `-e` for a plain install. If you already have the files
(e.g. from an archive), skip the clone and run `pip install -e .` from inside
the folder.

> If pip warns that `lamellyx.exe` is not on PATH, ignore it and use
> **`python -m lamellyx …`** instead of the bare `lamellyx` command — it always
> works regardless of PATH. Every `lamellyx <cmd>` below can be written
> `python -m lamellyx <cmd>`.

Python 3.9+ and numpy; nothing else. GROMACS is needed to *run* the result, not
to build it — and to generate protein topology, which is the one step that
calls out to `gmx pdb2gmx`.

The second line is required. The lipid conformers, the water template and the
CHARMM36 `.itp` files are CHARMM-GUI and MacKerell-lab material, and neither
states terms permitting redistribution, so they are not in this repository or
the wheel. `setup` copies them out of a CHARMM-GUI directory you already have
and reproduces them bit-for-bit; see [LICENSE](LICENSE).

## Three ways in

**Python** — `lamellyx.api`: `describe()`, `check()`, `build()`,
`build_protein_system()`, `orient_protein()`, `generate_protein_topology()`,
and for ligands `generate_ligand_topology()`, `generate_ligand_topologies()`
(a whole directory), `prepare_mol2()`, `place_ligand()`, `check_system()`.
Plain dicts in and out.

**Command line** — `lamellyx setup | schema | check | build | orient |
topology | mol2 | ligand | ligand-batch | place | check-system | dashboard |
lipids | test`.
Everything prints JSON with `--json`. `check-system <dir>` is a pre-grompp
sanity pass: it confirms `[ molecules ]` and the coordinates agree on the atom
count, every molecule has a topology, the index groups partition every atom
once, and the net charge is integer — the mismatches grompp would catch, caught
before a cluster run.

**Browser** — `lamellyx dashboard` opens a local page with two tabs: a
*Membrane builder* (drop in a PDB, set the parameters, watch the log, download
the files) and a *Ligand topology* tab (paste a CGenFF stream, get the `.itp`).
Localhost only, with a one-time token in the printed link.

For driving it from an agent, read [AGENTS.md](AGENTS.md) — it is the units and
the failure modes in one page.

## Defaults

Lengths are **nanometres**, matching GROMACS.

| Setting | Default |
|---|---|
| `lipid` | `POPC` |
| `leaflet` | `bilayer` |
| `x`, `y` | 4.0, 4.0 nm — the **full box edge** |
| `area_per_lipid` | 0.643 nm² (measured POPC, Kučerka 2011) |
| `water_thickness` | 1.5 nm above and below |
| `salt_concentration` | 0.5 M |
| `cation` / `anion` | `POT` (K⁺) / `CLA` (Cl⁻) |
| `water_model` | `TIP3` (CHARMM-modified TIP3P) |
| `forcefield` | `CHARMM36` |
| `temperature` | 310 K |

The defaults build: 4 × 4 nm, 50 POPC, 12,769 atoms, about seven seconds.

Two limits are enforced rather than warned about, because the result would not
run: **a box below 2.4 nm** (CHARMM36's 1.2 nm cutoffs need twice that for the
minimum image convention) and **an area per lipid below 0.40 nm²** (what two
extended acyl chains occupy in any phase). `strict: false` overrides both.

## Membrane proteins

Sizing is by margin, not by box, which is usually what you actually want:

```python
from lamellyx.api import build_protein_system

build_protein_system({
    "protein_pdb": "channel.pdb",
    "reference_dir": "charmm-gui-1234/gromacs",
    "output_dir": "system",
    "xy_margin": 20.0,        # Angstrom on this path — see AGENTS.md
})
```

The box edge is set to the protein's width **across the membrane** plus the
margin on each side, the protein is centred, and the lipid count follows from
whatever area is left once the protein has taken its share. Measuring the whole
protein instead would inflate the box for anything with a large extracellular
domain.

It also rebuilds hydrogens and terminal caps onto a heavy-atom model, and
rotates apart side chains that clash across a symmetry interface — homology
models arrive with those, and they are otherwise built straight into the
system.

**It cannot invent force-field parameters.** `reference_dir` must hold the
protein's own `.itp` files. `lamellyx topology in.pdb out_dir` will generate
them by driving GROMACS' `pdb2gmx` — so a new sequence no longer needs
CHARMM-GUI — but GROMACS must be installed, and assigning atom types and
charges is still that tool's job, not this one's.

### Orientation

`lamellyx orient in.pdb out.pdb --extracellular 443 460` puts a protein into
the membrane frame from the protein alone: the hydrophobic belt gives the
normal, a homo-oligomer's symmetry axis gives it again independently, and the
two are compared. Building through a protein that is *not* in the membrane
frame is now refused rather than done quietly.

Pass a residue range that is known to be extracellular. Nothing geometric
tells the two faces of a slab apart, so without it the protein is as likely to
come out upside down as not.

## Ligands

A small molecule needs CGenFF parameters, which CHARMM-GUI's Ligand Reader
assigns through the same web form. lamellyx does that step as a function too: a
CGenFF stream in, a GROMACS `.itp` out.

```bash
# no licence: paste your molecule into cgenff.silcsbio.com, download the .str
lamellyx ligand drug.str out_dir --ff charmm-gui-1234/gromacs/toppar
```

```python
from lamellyx.api import generate_ligand_topology

generate_ligand_topology({
    "str": "drug.str",
    "output_dir": "out_dir",
    "cgenff_ff": "charmm-gui-1234/gromacs/toppar",   # merges the base CGenFF FF
})
```

A ParamChem `.str` is only a *supplement* to the base CGenFF force field — it
carries the atoms, charges and the parameters CGenFF generated by analogy, and
references the standard bonds, angles and Lennard-Jones terms without including
them. Point `--ff` / `cgenff_ff` at a directory holding `top_all36_cgenff.rtf`
and `par_all36_cgenff.prm` (any CHARMM-GUI `toppar/` has them) and they are
merged in, the stream's analogy values winning. The output is validated
section-for-section against Lemkul's reference converter on the GROMACS
tutorial ligand (JZ4).

**Starting from a PDB**, still with no licence:

```bash
lamellyx mol2 drug.pdb drug.mol2      # Open Babel, protonated at pH 7
# upload drug.mol2 to cgenff.silcsbio.com, download drug.str
lamellyx ligand drug.str out_dir --ff .../toppar
```

`mol2` reports the pH it used and how many hydrogens it added, so the
protonation choice is visible rather than buried; it needs Open Babel on PATH.
With a CGenFF licence, `lamellyx ligand drug.mol2 out_dir --cgenff /path/to/cgenff`
(or a `.pdb`, which is protonated first) does the whole chain in one call.

The CGenFF **penalty** — how far it reached by analogy — is reported, never
refused: most biological ligands lean on analogy, so a hard cutoff would reject
the normal case. `--penalty-flag 50` additionally lists everything above a
number you choose. **Lone pairs** (halogen sigma holes) become GROMACS
`virtual_sites2` with the host atom's exclusions.

**Placing it:** `lamellyx place system_dir drug.pdb out_dir/DRG.itp built` puts
a positioned ligand into a built protein system — matching the PDB to the `.itp`
atom order, splicing it into the solute group, extending the topology and index,
and deduplicating its atom types against the force field. A lone pair, which no
docked PDB contains, is reconstructed from its host atoms. You supply the
coordinates; nothing here docks. (Building the membrane *around* a bound ligand,
so lipids never pack into it, is designed in [docs/ligand-in-build.md] but not
yet built.)

All of this is on the **dashboard** too, under the *Ligand topology* tab. A
runnable walkthrough that needs no setup is
[`examples/ligand_topology.py`](examples/ligand_topology.py).

For a **library**, `lamellyx ligand-batch streams/ topologies/ --ff .../toppar`
converts a whole directory of streams in one call, reading the base force field
once — much faster than a shell loop, where every `lamellyx ligand` is a fresh
process that re-reads the ~38k-line parameter file.

> As with the rest of lamellyx, a ligand topology has **not** been through
> `gmx grompp`; the checks are geometric and bookkeeping. Reproducing a
> CHARMM-GUI Ligand Reader result count-for-count is the acceptance test that
> still wants a molecule and a cluster.

[docs/ligand-in-build.md]: docs/ligand-in-build.md

## Quality

Against a CHARMM-GUI POPC system built for the same force field:

| | this | CHARMM-GUI |
|---|---|---|
| closest heavy-atom contact | 2.33 Å | 2.34 Å |
| heavy pairs under 2.4 Å | 3 | 3 |
| closest contact including hydrogens | 0.4–0.6 Å | 1.44 Å |
| bulk water density | 102% | 100% |
| phosphate–phosphate thickness | 3.91 nm | 3.7 nm (experiment) |
| water in the hydrophobic core | 0 | 0 |

Heavy-atom packing is at parity. **Hydrogen contacts are not**, and that is a
real limitation. Rigid conformers fix the C–H bond at 1.09 Å, so two pointing
at each other across an otherwise-fine 2.4 Å carbon contact leave the hydrogens
a few tenths of an Ångström apart, and no rigid-body move can open that — only a
torsion can. Minimisation fixes it in a few hundred steps, which is why running
`step6.0` first is not optional. An all-atom relaxation pass was tried twice and
removed twice: both times it improved the hydrogens slightly and took the
closest heavy pair from 2.42 Å to 1.13 Å.

## Tests

```bash
lamellyx test
```

132 tests, about a minute and a half, no pytest. Invariants (rigid moves never change an
internal distance; molecules are never split across the boundary; the same seed
gives the same box), agreement with brute force and with known answers, and
regressions for every bug found so far. One is an acceptance test: the JZ4
ligand converts to the same topology, count for count, that Lemkul's reference
converter gives — it needs the base CGenFF force field, so it skips unless
`LAMELLYX_CGENFF_FF` points at a `toppar/` that has it. Writing them turned up
bugs the way tests do — a rebuilt
system that silently split molecules at the periodic boundary, water's own O–H
bonds counted as clashes, and (found while wiring the ligand UI) a name
collision that shadowed `window.history` and left every dashboard button dead
on load.

## Adding a lipid

The bundled parameters and conformers cover POPC only.

```bash
python -m lamellyx.make_data /path/to/equilibrated_system --lipid POPE
```

That extracts coordinates. You still need `POPE.itp` in a `toppar` directory.

## Licence

MIT for the code. The bundled CHARMM36 parameters are the MacKerell
laboratory's work under their own terms — see [LICENSE](LICENSE) before
redistributing.
