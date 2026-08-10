# lamellyx

Lipid bilayers and membrane-protein systems for GROMACS, from Python.

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

```bash
pip install lamellyx
python -m lamellyx setup <a CHARMM-GUI gromacs directory>
```

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
`build_protein_system()`, `orient_protein()`, `generate_protein_topology()`.
Plain dicts in and out.

**Command line** — `lamellyx setup | schema | check | build | orient |
topology | dashboard | lipids | test`. Everything prints JSON with `--json`.

**Browser** — `lamellyx dashboard` opens a local page: drop in a PDB,
set the parameters, watch the log, download the files. Localhost only, with a
one-time token in the printed link.

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

44 tests, about a minute, no pytest. Invariants (rigid moves never change an
internal distance; molecules are never split across the boundary; the same seed
gives the same box), agreement with brute force and with known answers, and
regressions for every bug found so far. Writing them turned up six, including
one where a rebuilt system silently split molecules at the periodic boundary and
one where water's own O–H bonds were counted as clashes.

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
