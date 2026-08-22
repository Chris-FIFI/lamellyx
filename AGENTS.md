# Using lamellyx from an agent

This package exists because CHARMM-GUI is a web form and cannot be driven by a
language model. Everything here takes and returns plain JSON.

## Install

Not on PyPI — it is a private GitHub repo, so `pip install lamellyx` fails.
Install from a clone:

```bash
git clone https://github.com/Chris-FIFI/lamellyx.git
cd lamellyx
pip install -e .
python -m lamellyx setup <a CHARMM-GUI gromacs directory>
```

Requires Python 3.9+ and numpy. Nothing else. No GROMACS needed to *build* —
only to run the result, and to generate protein topology (below).

**The second line is not optional.** The package ships with no lipid
conformers, no water template and no force-field `.itp` files: they come from
CHARMM-GUI and the MacKerell lab, neither of which states terms permitting
redistribution, so they are not in the repository or the wheel. `setup` copies
them out of any CHARMM-GUI GROMACS directory you already have, and reproduces
them bit-for-bit. Without it, the first build fails with a message saying
exactly this.

## The three calls

```python
from lamellyx.api import describe, check, build

describe()                                  # every setting, units, ranges
check({"x": 8.0, "y": 8.0})                 # {"ok": bool, errors, warnings}
build({"output_dir": "popc_box", "x": 8.0}) # builds; returns a summary
```

Or as commands, all of which print JSON with `--json`:

```bash
lamellyx schema
lamellyx check --set x=8 --set y=8 --json
lamellyx build --out popc_box --set x=8 --set y=8 --json
```

**Call `check` before `build`.** It is instant and catches configurations that
GROMACS would refuse, before a build spends a minute producing them.

## Units — the one thing to get right

| | |
|---|---|
| `x`, `y` | **nanometres, full box edge** — not a half-width, not a margin |
| `area_per_lipid` | **nm² per lipid**, per leaflet. Measured POPC is `0.643` |
| `water_thickness` | **nm** of water above *and* below the bilayer |
| `salt_concentration` | **mol/L** |
| `margin_x`, `margin_y` | **nm** of lipid beyond the protein, on each side |

Three mistakes account for almost every failed build:

1. **Confusing the box with the margin.** `x` is the whole box edge: `x = 2.0`
   is a 2 nm box, not 2 nm either side of anything. "2 nm of membrane around
   the protein" is `margin_x` / `margin_y`, and gives a box of protein width
   + 2 × margin. Those two readings differ by more than a factor of two on a
   real channel.
2. **Naming an `area_per_lipid` you have not measured.** It defaults to `0`,
   meaning "use the measured value" — 0.643 nm² for POPC, which is what
   CHARMM-GUI does. Anything below 0.40 nm² is refused: two extended acyl
   chains do not fit side by side.
3. **A box below 2.4 nm.** CHARMM36 uses 1.2 nm cutoffs and a periodic box
   must be at least twice its cutoff, so `grompp` rejects anything smaller.

## What comes back

```json
{
  "ok": true,
  "output_dir": "/abs/path/popc_box",
  "counts": {"POPC": 200, "POT": 78, "CLA": 78, "TIP3": 8537,
             "TOTAL_ATOMS": 52567},
  "box_nm": [8.0, 8.0, 8.543],
  "area_per_lipid_nm2": 0.64,
  "density_g_cm3": 0.9354,
  "net_charge": 0.0,
  "closest_heavy_atom_contact_A": 2.33,
  "heavy_pairs_below_2.4A": 3,
  "next_command": "cd … && gmx grompp -f step6.0_minimization.mdp …"
}
```

### How to tell a good build from a bad one

- `net_charge` must be exactly `0.0`.
- `closest_heavy_atom_contact_A` should be **above about 2.2**. The reference
  CHARMM-GUI system is 2.34. Below 2.0 means the packing failed.
- `density_g_cm3` should be **0.90–0.98** for a solvated bilayer. Much lower
  means solvation went wrong.
- `heavy_pairs_below_2.4A` in single digits is normal; hundreds is not.

The closest contact *including hydrogens* is deliberately not a pass/fail
number. Packing places rigid conformers, so two hydrogens can end up a few
tenths of an Ångström apart across a perfectly good carbon contact, and only a
torsion change can open that. `step6.0_minimization` fixes it in a few hundred
steps. **Always run the minimisation before anything else.**

## Embedding a protein

```python
from lamellyx.api import extract_protein, build_protein_system

# The protein PDB must match the .itp topology: original residue numbering,
# and a chain identifier in column 22. Pulling it out of the reference .gro
# by hand gives neither -- .gro has no chain column and renumbers from 1.
# This does it correctly:
extract_protein("charmm-gui-1234/gromacs", "protein.pdb")

build_protein_system({
    "protein_pdb": "protein.pdb",
    "reference_dir": "charmm-gui-1234/gromacs",   # holds PROA.itp etc.
    "output_dir": "system",
    "margin_x": 2.0,             # nm, same as everywhere else
    "margin_y": 2.0,
    "water_thickness": 1.5,      # nm
    "salt_concentration": 0.15,  # same name as build()
    "area_per_lipid": 0,         # the default; 0 = measured, same as build()
})
```

`pore_radius` (nm, default 1.0) decides what counts as pore water. Being far
from lipid and well enclosed is not enough: where packing leaves a gap around
the protein, water in the gap is far from lipid *because the lipid is missing*,
and a surface groove is enclosed. Only a radius says "down the middle". Set it
to 0 to disable, or wider for a channel with a large vestibule — and check
`build_report.json`'s `core_water` afterwards, which reports how much water sits
between the phosphate planes and how far off-axis it is.

Same units and same names as `build()`: nanometres and `salt_concentration`.
`describe()["protein_settings"]` reports them, and agrees with this. The
lower-level `builder.BuildConfig` works in Ångström and is converted for you --
only reach for it directly if you know why.

Unknown settings raise rather than being dropped, so a typo or the wrong name
fails loudly instead of leaving a default in place.

## Building a system that matches another one

Two states of the same protein only give a meaningful free-energy difference if
the systems differ in nothing but the protein's conformation. `match_counts_from`
points at a built GROMACS directory and reproduces its composition exactly:

```python
build_protein_system({
    "protein_pdb": "state2.pdb",
    "reference_dir": "charmm-gui-1234/gromacs",
    "output_dir": "state2",
    "match_counts_from": "state1/gromacs",   # same lipids, water and ions
})
```

It takes the lipids per leaflet, the water and the ion counts from that
directory and trims surplus water from bulk — never from the interface or the
pore — to land on the target exactly. Ions are placed by replacing water, so the
trim allows for that; going straight to the target water count lands exactly
n_ion molecules short. Water is only ever removed, never invented, so the
reference must not be more solvated than this box can hold; if it is, the call
says so rather than building something close.

Verified against CHARMM-GUI: given the same protein and the same box, this
reproduces **125,042 atoms, 293 POPC (142 / 151), 22,876 waters, 70 K⁺, 62 Cl⁻**
— every count identical.

> The counts match; the water is not distributed identically. This builder puts
> more of it in the first hydration shell — 4,036 molecules within 4 Å of a
> solute heavy atom against CHARMM-GUI's 2,365 — and correspondingly less in
> bulk. Equilibration redistributes it. Do not expect the two `.gro` files to
> resemble each other molecule for molecule, only to contain the same things.

**This path cannot invent force-field parameters.** It needs a directory
holding the protein's own `.itp` files. If `reference_dir` is missing the call
fails with that explanation rather than a confusing file error.

### Getting those `.itp` files without CHARMM-GUI

```python
from lamellyx.api import generate_protein_topology
generate_protein_topology({"protein_pdb": "hcn4.pdb",
                           "output_dir": "topology_out",
                           "forcefield": "charmm36-jul2022"})
```

This drives GROMACS' own `pdb2gmx` and renames the chains to `PROA`, `PROB`, …
so the result drops straight into `reference_dir`. **GROMACS must be
installed**; it is not bundled, and on a machine without it the call raises
naming what it looked for rather than failing later and obscurely.

Assigning atom types, charges and bonded terms is still not done here — that is
what `pdb2gmx` and CHARMM-GUI are for, and a third implementation's failure
mode is silently wrong energies. What this removes is the *dependency on
CHARMM-GUI* for the step.

## Ligand topology

The same idea for a small molecule: CGenFF assigns the parameters, this converts
its stream to a GROMACS `.itp`.

```python
from lamellyx.api import generate_ligand_topology
generate_ligand_topology({
    "str": "drug.str",              # from cgenff.silcsbio.com (no licence)
    "output_dir": "lig_out",
    "cgenff_ff": "charmm-gui-1234/gromacs/toppar",   # REQUIRED for a real .str
})
```

The one failure mode to know: **a real ParamChem `.str` is a supplement, not a
complete parameter set.** Without `cgenff_ff` pointing at a `toppar/` (with
`top_all36_cgenff.rtf` + `par_all36_cgenff.prm`) the call raises `N of this
molecule's parameters are not in the .str` and tells you to pass it. It is not
a bad stream; the base terms simply live in the force field the stream
references. The report carries `net_charge`, the term counts, `worst_penalty`
and the worst offenders. The CGenFF **penalty is reported, never refused**.

Other entry points, same report shape: `{"molecule": "drug.mol2", ...}` drives
the licensed `cgenff` binary (needs `"cgenff": "/path"`); `{"pdb": "drug.pdb",
"ph": 7.0, ...}` protonates with Open Babel first. `prepare_mol2({"pdb": ...,
"output": "drug.mol2"})` does only the PDB→mol2 step for the no-licence web
route. `place_ligand({"system_dir": ..., "ligand_pdb": ..., "ligand_itp": ...,
"output_dir": ...})` puts a positioned ligand into a built system.
`check_system({"system_dir": ...})` structurally checks any built system before
grompp -- [ molecules ] vs coordinate atom counts, molecule topologies, index
coverage, net charge -- returning {ok, errors, warnings}.

Like everything else here, **not yet through `gmx grompp`** — the checks are
bookkeeping and geometry.

## Orienting a protein in the membrane

A PDB whose membrane normal is not z used to get a bilayer built straight
through it, sideways, without a word. It now refuses:

```python
build_protein_system({..., "auto_orient": True,
                      "extracellular_resid": [443, 460]})
```

or on its own, so you can look at the answer before committing to a build:

```bash
lamellyx orient in.pdb oriented.pdb --extracellular 443 460
```

Two independent estimates of the normal are computed and compared:

| | |
|---|---|
| **hydrophobic slab** | the physical definition — the slab whose enclosed surface has the lowest transfer energy, from per-atom SASA and solvation parameters. Needs no symmetry and no reference |
| **symmetry axis** | for a homo-oligomer, superimposing one chain on the next gives a rotation whose axis *is* the membrane normal. Exact, and independent of any energy model |

They are reported separately as `slab_vs_symmetry_deg`. Agreement within a few
degrees is the check; disagreement is a finding, and neither is then trusted.

Three things worth knowing:

- **`extracellular_resid` matters.** Nothing geometric distinguishes the two
  faces of a slab, so without it the protein is as likely to come out upside
  down as not. The report says which happened. The positive-inside rule is
  *not* used to decide: it is measured and reported, because on a construct
  truncated below the membrane — where the basic cytoplasmic loops are simply
  absent — it points the wrong way.
- **The hydrophobic thickness is fitted, not measured.** Buried surface alone
  falls monotonically with thickness, so a mismatch penalty pulls it toward the
  lipid's natural value (30 Å for POPC). Without it the search ran to whatever
  bound it was given and bought extra surface by tilting — worst case 6° of
  error on a tetramer; with it, 0.1°.
- **The azimuth is minimised too.** Spinning about the normal is free to the
  membrane but not to you: the tightest rotation shrank an HCN4 bounding square
  from 108 Å to 88 Å, which is several hundred lipids and their water.

## Reproducibility

`seed` makes a build deterministic — the same settings and seed give a
byte-identical `.gro`. Every build writes `build_report.json` containing the
full configuration, the counts and the quality checks, so a system can be
traced back to what produced it. Feeding a report's `config` straight back into
`build()` reproduces the system.

> The `box_nm` that `build()` **returns** is rounded to 4 decimal places for
> readability; the `box_nm` written **into `build_report.json`** is full
> precision. Do not compare the two at tight tolerance and conclude the box
> changed — compare report against report, or round both.

## What it will refuse to do

It stops rather than emitting something that cannot run: box below 2.4 nm,
area per lipid below 0.40 nm², negative salt, non-positive temperature, an
unrecognised `leaflet`. Pass `strict: false` to override — the result will
build, and for an impossible area per lipid it is a column of stacked lipids
rather than a bilayer.

Mixed-lipid bilayers are not implemented yet and say so explicitly. The
bundled parameters cover **POPC, TIP3P, K⁺ and Cl⁻ only**; anything else needs
a `toppar` directory and a conformer library built with
`python -m lamellyx.make_data`.
