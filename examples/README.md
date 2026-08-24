# Examples — try lamellyx in five minutes

Two things you can run right now. The first needs **nothing but lamellyx**
(no licence, no GROMACS, no setup); the second needs the one-time `setup` step.

## 1. Ligand topology — zero setup

Turn a CGenFF stream into a GROMACS ligand topology:

```bash
python -m lamellyx ligand examples/chloromethane.str out
```

In `out/` you get two files and a JSON report:

- **`CLM.itp`** — the ligand topology: `[ moleculetype ]`, `[ atoms ]`,
  `[ bonds ]`, `[ angles ]`, and a `[ virtual_sites2 ]` section.
- **`CLM_atomtypes.itp`** — its atom types, to `#include` *before* any
  `[ moleculetype ]`.

Chloromethane (CH₃Cl) carries a chlorine **lone pair** (a sigma hole). Open
`CLM.itp` and find `[ virtual_sites2 ]` — that is the lone pair, rebuilt as a
GROMACS virtual site. The report also flags one parameter with a small CGenFF
**penalty** (the C–Cl bond, `5.0`): a penalty is how far CGenFF reached by
analogy, and lamellyx always reports it rather than hiding it.

### Exercise

1. **Read the input.** Open [`chloromethane.str`](chloromethane.str). Every
   parameter it needs is inline — that is why it converts with no base force
   field (`"base_ff_merged": false` in the report).
2. **See the penalty filter.** Re-run with `--penalty-flag 1` and watch the C–Cl
   bond get listed as above your threshold.
3. **Do your own molecule.** Draw or upload it at
   <https://cgenff.silcsbio.com> (free, no licence) and download the `.str`.
   A real ParamChem stream is only a *supplement* to the base CGenFF force
   field, so add `--ff /path/to/charmm-gui-1234/gromacs/toppar` (the directory
   holding `top_all36_cgenff.rtf` + `par_all36_cgenff.prm`) and lamellyx merges
   the rest in. Starting from a PDB? Run `lamellyx mol2 drug.pdb drug.mol2`
   first to protonate it at pH 7, then upload that.
4. **Prefer Python?** [`ligand_topology.py`](ligand_topology.py) does the same
   through the API and sketches how to place the ligand into a built protein
   system.

## 2. A POPC bilayer — needs `setup` first

The membrane builder needs the CHARMM36 lipid, water and parameter files, which
are not redistributed here (see the top-level `LICENSE`). Copy them out of any
CHARMM-GUI `gromacs` output you already have — **use a real path, do not type
the angle brackets**:

```bash
python -m lamellyx setup /path/to/charmm-gui-1234/gromacs
```

Then build a bilayer:

```bash
python examples/popc_bilayer.py
# or straight from the CLI:
python -m lamellyx build --out popc_box --set x=8 --set y=8
```

Out comes a directory GROMACS runs as it stands (`grompp` → `mdrun`); see the
main [README](../README.md) for the fields and defaults.

---

> Nothing lamellyx writes has been through `gmx grompp` yet — the checks are
> geometric and bookkeeping. Run the `step6.0` minimisation first; that is the
> real test.
