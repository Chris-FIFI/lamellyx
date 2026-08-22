"""Turn a CGenFF stream into a GROMACS ligand topology, end to end.

Run it as-is -- it needs nothing but lamellyx and numpy:

    python examples/ligand_topology.py

It converts a small self-contained stream (chloromethane, which has a halogen
lone pair, so you can see the virtual site come out) and prints the report.

For YOUR molecule the only thing that changes is where the stream comes from:

  1. Draw or upload it at https://cgenff.silcsbio.com (no licence) and download
     the .str -- or, from a PDB, run `lamellyx mol2 drug.pdb drug.mol2` first to
     protonate it at pH 7, upload that, and download the stream.
  2. A real ParamChem stream is only a *supplement* to the base CGenFF force
     field, so pass `cgenff_ff=` a CHARMM-GUI `toppar/` directory (the one with
     top_all36_cgenff.rtf + par_all36_cgenff.prm). The toy below is
     self-contained and needs none.
  3. To put the ligand into a built protein system, see `place_it()` at the
     bottom.
"""

import tempfile

from lamellyx.api import generate_ligand_topology

# A complete CGenFF stream for chloromethane, CH3Cl, with a sigma-hole lone pair
# on the chlorine. Everything it needs is inline, so no base force field is
# required -- a real ParamChem stream would instead reference the base FF.
CHLOROMETHANE = """\
* Toppar stream file for chloromethane -- lamellyx example
*

read rtf card append
* Topologies
*
36 1

MASS -1 CG331   12.01100 ! aliphatic C for CH3
MASS -1 HGA3     1.00800 ! aliphatic H, CH3
MASS -1 CLGA1   35.45000 ! aliphatic chlorine

RESI CLM    0.000
GROUP
ATOM C1  CG331  -0.070 !   0.000
ATOM H1  HGA3    0.090 !   0.000
ATOM H2  HGA3    0.090 !   0.000
ATOM H3  HGA3    0.090 !   0.000
ATOM CL1 CLGA1  -0.290 !   0.000
ATOM LP1 LPH     0.090 !   0.000
BOND C1 H1
BOND C1 H2
BOND C1 H3
BOND C1 CL1
LONEPAIR COLINEAR LP1 CL1 C1 DIST 1.5000 SCALE 0.0

END

read param card flex append
* Parameters
*

BONDS
CG331 HGA3    322.00   1.1110 ! penalty= 0.0
CG331 CLGA1   222.00   1.7760 ! penalty= 5.0

ANGLES
HGA3 CG331 HGA3    35.50   108.40 ! penalty= 0.0
HGA3 CG331 CLGA1   34.00   108.00 ! penalty= 0.0

DIHEDRALS

IMPROPERS

NONBONDED nbxmod 5 atom cdiel
CG331   0.0  -0.0780   2.0500
HGA3    0.0  -0.0240   1.3400
CLGA1   0.0  -0.3430   1.9100

END
RETURN
"""


def main():
    out = tempfile.mkdtemp(prefix="lamellyx_ligand_")
    report = generate_ligand_topology({
        "str": CHLOROMETHANE,
        "output_dir": out,
        # for a real ParamChem stream, also pass:
        # "cgenff_ff": "charmm-gui-1234/gromacs/toppar",
        "penalty_flag": 10.0,        # list any parameter whose penalty exceeds 10
    })

    print("residue        :", report["resname"])
    print("atoms          :", report["n_atoms"],
          "(bonds %d, angles %d, dihedrals %d, impropers %d)"
          % (report["n_bonds"], report["n_angles"],
             report["n_dihedrals"], report["n_impropers"]))
    print("lone pairs     :", report["n_lonepairs"], "-> virtual sites in the .itp")
    print("net charge     :", report["net_charge"],
          "(integer: %s)" % report["net_charge_is_integer"])
    if report["charge_note"]:
        print("  ! ", report["charge_note"])
    print("worst penalty  :", report["worst_penalty"])
    if report["penalties"]:
        print("  top penalties:", report["penalties"][:3])
    print("files          :", ", ".join(report["files"]))
    print("written to     :", out)
    print()
    print("Embed it: #include the _atomtypes.itp before any [ moleculetype ],")
    print("then the .itp, place the ligand's coordinates, and add it to")
    print("[ molecules ]. Not yet checked by gmx grompp.")


def place_it():
    """Sketch of the placement step (needs a built protein system + a docked
    PDB, so it is not run here).

        from lamellyx.api import place_ligand
        place_ligand({
            "system_dir": "hcn4_box",         # step5_input.gro, topol.top, ...
            "ligand_pdb": "drug_docked.pdb",  # positioned in the box frame
            "ligand_itp": "out/CLM.itp",      # from generate_ligand_topology
            "output_dir": "hcn4_drug",
        })

    A lone pair, which no docked PDB contains, is reconstructed from its host
    atoms automatically. You supply the coordinates; nothing here docks.
    """


if __name__ == "__main__":
    main()
