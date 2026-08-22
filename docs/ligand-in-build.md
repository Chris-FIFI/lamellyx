# Ligand-before-lipid ordering — design note

**Status: designed, not implemented. Needs sign-off + a grompp run before it lands.**

## The ask

> "Do the ligand parameterisation before putting it in the lipid box tho to
> prevent clashes."

Today the flow is out of order. `cgenff.py` builds a ligand's topology, and
`ligand.add_ligand` (in `ligand.py`) inserts a positioned ligand into an
**already-built** system — protein + bilayer + water. Because the bilayer was
packed with no knowledge of the drug, lipids can already sit where the drug
needs to be; the placer reports the closest contact, but the fix is a shove, not
an absence of the clash.

The wanted order: the ligand is part of the **solute the bilayer packs around**,
so lipids are kept off it from the start.

## Why this is deferred, not done in the loop

1. **Not hermetically testable here.** `builder.build()` (the protein+membrane
   path) needs a `reference_dir` — a CHARMM-GUI system with `toppar/` +
   `step5_input.gro`. That is MacKerell/CHARMM-GUI material, not shipped and not
   in the test suite. The suite only exercises the *bilayer-only* path
   (`membrane.build_membrane`, via `_tiny`) and the protein path's *input
   validation*. There is no fixture a ligand-in-build test could run against.
2. **No grompp here.** Like the rest of lamellyx, nothing has been through
   `gmx grompp`; a mispositioned virtual site or an off-by-one in the block
   ordering is exactly the failure grompp catches and geometry checks do not.
3. **It touches the membrane pipeline**, which is explicitly *not* v2 ("v2 is
   the ligand function only"). A change here deserves CJ/Will's eyes.

So the safe autonomous deliverable is this design, not the code.

## The design (additive and guarded)

Everything below runs only when `cfg.ligand_pdb` is set. With no ligand, every
line is unchanged and the existing build is byte-for-byte identical — the new
path cannot regress a ligand-free build.

### 1. Config (`builder.BuildConfig`)

```python
# --- ligand (optional) ---
ligand_pdb: str = ""            # positioned in the same frame as protein_pdb
ligand_itp: str = ""           # from cgenff.generate_ligand_topology
ligand_atomtypes_itp: str = "" # default: <ligand_itp stem>_atomtypes.itp
ligand_resname: str = ""       # default: the moleculetype name in the .itp
```

### 2. Frame-tracking — the one subtle part

The ligand PDB is given in the same frame as `protein_pdb`, so it must undergo
**every rigid transform the protein does**, in the same order:

| build() step | transform on the protein | apply to ligand |
|---|---|---|
| `orient_from_pdb` (L303) | `xyz @ R.T + t` | same `R, t` |
| `auto_orient` (L307) | `orient.orient()` returns the oriented model, **not** the transform | see note |
| box centring x,y (L414-415) | `+= box/2 - centre` | same Δ |
| z sizing shift (L419) | `+= shift` | same Δ |
| sidechain relief (L353) | moves *protein* side chains only | none |

Recommended first cut: support `orient_from_pdb` and the already-oriented case
(`require_oriented`, no transform), and **refuse `auto_orient` + ligand** with a
clear message ("orient the complex first, or pass orient_from_pdb"). Lifting
that later means having `orient.orient` also return `(R, t)` and applying it to
the ligand. The centring/z shifts are plain translations computed from the
protein — capture the Δ and add it to the ligand coordinates too.

Load + order the ligand right after the `protein` Atoms is built (~L379), reusing
`ligand.order_to_itp` so its atoms are in `.itp` order.

### 3. Solute footprint — the actual "prevent clashes"

`prot_heavy` is what the bilayer avoids (`bilayer.lipids_for_free_area` L444-447,
`bilayer.build_bilayer` L493) and what sizes the box (`bilayer.slab_extent`
L408). Replace it with a **solute** footprint:

```python
solute_heavy = np.vstack([prot_heavy, ligand.xyz[ligand.element != "H"]])
```

Use `solute_heavy` at L408, L444-447, L493. With no ligand it equals
`prot_heavy`, so nothing changes. Recompute it after each transform that moves
coordinates (after the box-centring block, as the code already does for
`prot_heavy` at L424).

### 4. Solvation (L508)

```python
solute = fileio.Atoms.concat([protein, ligand, lipids])   # ligand added
```

and pass `protein_xyz=` including the ligand heavy atoms (L519) so pore-water
logic sees it as part of the wall.

### 5. Assembly + bookkeeping

- **Order (L601):** `concat([protein, ligand, lipids, cations, anions, water])`.
  The ligand goes immediately after the protein so it lands inside `SOLU`.
- **Blocks (L602-604):** insert `(ligand_top, 1)` after the protein mols, before
  the lipid block. `ligand_top = topology.parse_itp(cfg.ligand_itp)[name]`.
- **Charge (L559):** `q_protein += ligand_top.total_charge`.
- **Index (`write_output`, L692):** `SOLU = slice(0, npro + nlig)`; `MEMB` and
  `SOLV` shift by `nlig`. `standard_groups` already takes the three slices.
- **Topology (`write_output`, L673-677):**
  - copy `cfg.ligand_itp` and the deduped atomtypes into `out/toppar/`;
  - `includes = [ligand atomtypes] + [PRO*.itp ...] + [ligand.itp] + [lipid,
    ions, water]` — atomtypes **first** (GROMACS reads `[atomtypes]` only at top
    level), the moleculetype `.itp` after the protein;
  - `molecules`: insert `(ligand_resname, 1)` after the protein rows;
  - dedup the ligand atomtypes against the copied `toppar/` with the existing
    `ligand._dedup_atomtypes` / `ligand.existing_atomtypes` — reuse them, do not
    re-implement.

### 6. Virtual sites (lone pairs)

If the ligand `.itp` carries `[ virtual_sites2 ]` (a halogen lone pair), the LP
is one of the `.itp` atoms but is **absent from a docked PDB**. Two options:
either require the LP row in `ligand_pdb` (grompp will rebuild its position
anyway), or construct the LP coordinate from its hosts at assembly time using the
`a` parameter. First cut: require it and say so; `order_to_itp` already refuses a
PDB whose atom set disagrees with the `.itp`, which surfaces the missing LP
clearly rather than silently.

## How to validate before trusting it

1. Build against a real reference (CJ's HCN4 CHARMM-GUI dir) with a docked
   ligand PDB + its cgenff `.itp`.
2. Check the build report: `net_charge == 0`, closest ligand contact > 2 Å,
   `TOTAL_ATOMS` = protein + ligand + lipids + ions + water.
3. **`gmx grompp`** on the result — the real acceptance test. Confirm the
   `[ molecules ]` order matches the `.gro`, atom types resolve, and (if a lone
   pair) the virtual site constructs.
4. Compare a CHARMM-GUI protein+ligand+membrane build of the same inputs
   count-for-count, the same acceptance test used for the plain converter.

## Effort estimate

~40–60 lines in `builder.py` (mostly guarded blocks), the `BuildConfig` fields,
and reuse of `ligand.order_to_itp` / `ligand._dedup_atomtypes`. The risk is all
in §2 (frame-tracking) and §5 (block order) — both grompp-checkable, neither
hermetically testable in the current suite without a shipped reference fixture.
