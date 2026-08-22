# Quickstart — no terminal needed

lamellyx builds membrane systems for GROMACS, and turns a small-molecule
"CGenFF stream" into a ligand topology. This page is the gentle version; the
[README](README.md) is the full, technical one.

## 1. Open it

**Double-click `Lamellyx Dashboard.bat`** (on your Desktop, or `run-dashboard.bat`
in the project folder).

A black window opens and, a second later, your web browser opens to the
dashboard. Leave the black window open while you work — closing it stops the
dashboard. Everything below happens in the browser, with two tabs at the top:
**Membrane builder** and **Ligand topology**.

> If the black window shows an error instead of opening the browser: Python
> isn't set up. Install it from python.org (tick "Add Python to PATH"), then
> `pip install -e .` once inside the project folder.

## 2. Make a ligand topology

You need a **CGenFF stream** (a `.str` file) for your molecule. To get one, free
and without a licence:

1. Go to **cgenff.silcsbio.com**, sign in, and upload/draw your molecule.
2. Download the `.str` it gives back.

Then, in the **Ligand topology** tab:

1. Paste the `.str` text into the big box.
2. In **Base force-field directory**, put the path to a CHARMM-GUI `toppar/`
   folder (the one containing `top_all36_cgenff.rtf` and
   `par_all36_cgenff.prm`). A real ParamChem stream needs this; the dashboard
   remembers it after the first time.
3. Press **Make topology**.

You get a report (charge, atom counts, any high "penalties" to check) and two
files to download: `<NAME>.itp` and `<NAME>_atomtypes.itp`. You can also read
the `.itp` inline under *View …*.

> Starting from a PDB instead of a drawing? First turn it into a protonated
> mol2: in a terminal, `python -m lamellyx mol2 drug.pdb drug.mol2` (needs
> Open Babel installed), upload that mol2 to ParamChem, then come back here.

## 3. Build a membrane (optional)

In the **Membrane builder** tab: optionally drop in a protein PDB, choose the
lipid and size, and press **Build**. The log streams as it works, and you get a
folder GROMACS can run.

## 4. What you got, and the one caveat

Everything lamellyx writes is meant to drop straight into GROMACS:

```
gmx grompp -f step6.0_minimization.mdp -c step5_input.gro \
           -r step5_input.gro -p topol.top -n index.ndx -o min.tpr
gmx mdrun -deffnm min
```

**Nothing here has been through `gmx grompp` yet** — the checks are geometric and
bookkeeping. Run the minimisation above first; that is the real test. To sanity-
check a finished system before a cluster run, `python -m lamellyx check-system
<folder>` confirms the topology and coordinates agree.
