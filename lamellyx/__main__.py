"""Command line for lamellyx.

    lamellyx setup <gromacs_dir>       fetch the data the repo does not ship
    lamellyx schema                    what every setting means
    lamellyx check    --set x=2.0      validate without building
    lamellyx build    --out popc_box   build one
    lamellyx orient   in.pdb out.pdb   put a protein in the membrane frame
    lamellyx topology in.pdb out_dir   make its .itp files via gmx pdb2gmx
    lamellyx mol2     lig.pdb lig.mol2 protonate a PDB at pH 7 -> mol2 (no lic.)
    lamellyx ligand   lig.str out_dir  ligand topology from a CGenFF stream
    lamellyx place    sys lig.pdb lig.itp out   put a ligand into a system
    lamellyx check-system  a_built_dir    sanity-check a system before grompp
    lamellyx dashboard                 open the browser UI
    lamellyx lipids                    which lipids are available
    lamellyx test                      run the test suite

Run `setup` once after installing: the lipid conformers, water template and
force-field .itp files are CHARMM-GUI / MacKerell material and are not
redistributed here.

Every subcommand except `dashboard` prints JSON with `--json`, so this is
usable as a tool by anything that can run a command and parse the output.
Lengths are nanometres throughout.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import api


def _parse_set(pairs):
    """--set x=8 --set lipid=POPC  ->  {"x": 8.0, "lipid": "POPC"}"""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit("--set needs name=value, got %r" % p)
        k, v = p.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def _settings(args):
    s = {}
    if getattr(args, "config", None):
        with open(args.config) as fh:
            s.update({k: v for k, v in json.load(fh).items()
                      if not k.startswith("_")})
    if getattr(args, "json_in", None):
        s.update(json.loads(sys.stdin.read() if args.json_in == "-"
                            else args.json_in))
    s.update(_parse_set(getattr(args, "set", None)))
    if getattr(args, "out", None):
        s["output_dir"] = args.out
    return s


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)

    # `dashboard` has its own flags and forwards them verbatim. argparse
    # cannot express that: REMAINDER stops collecting at the first thing that
    # looks like an option, so `dashboard --no-browser` is rejected by the
    # outer parser before the inner one ever sees it. Hand it off first.
    if argv and argv[0] == "dashboard":
        from .dashboard import main as run_dash
        return run_dash(argv[1:])

    p = argparse.ArgumentParser(
        prog="lamellyx", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    def common(q):
        q.add_argument("--config", help="JSON file of settings")
        q.add_argument("--json-in", dest="json_in", metavar="JSON",
                       help="settings as a JSON string, or - to read stdin")
        q.add_argument("--set", action="append", metavar="NAME=VALUE",
                       help="override one setting; repeatable")
        q.add_argument("--json", action="store_true",
                       help="print the result as JSON")
        return q

    sub.add_parser("schema", help="every setting, its units and its range")
    common(sub.add_parser("check", help="validate settings without building"))
    b = common(sub.add_parser("build", help="build a system"))
    b.add_argument("--out", help="output directory")
    b.add_argument("-q", "--quiet", action="store_true")
    e = sub.add_parser("extract-protein",
                       help="write a topology-matching protein PDB out of a "
                            "reference system")
    e.add_argument("reference_dir")
    e.add_argument("out_pdb")
    e.add_argument("--molecules", nargs="+")
    e.add_argument("--chains", nargs="+")
    s = sub.add_parser("setup",
                       help="populate data/ from a CHARMM-GUI directory you "
                            "already have (the repository ships without it)")
    s.add_argument("gromacs_dir",
                   help="a CHARMM-GUI GROMACS directory: toppar/ + "
                        "step5_input.gro")
    s.add_argument("--lipid", default="POPC")
    s.add_argument("--water", default="TIP3")
    s.add_argument("--no-toppar", dest="toppar", action="store_false")

    o = sub.add_parser("orient",
                       help="put a protein in the membrane frame and write it "
                            "out")
    o.add_argument("protein_pdb")
    o.add_argument("out_pdb")
    o.add_argument("--extracellular", nargs=2, type=int, metavar=("FIRST", "LAST"),
                   help="a residue range known to be outside the cell; "
                        "without it, which way up is an arbitrary tie-break")
    o.add_argument("--symmetry", default="auto", choices=("auto", "yes", "no"))

    t = sub.add_parser("topology",
                       help="generate the protein's .itp files with GROMACS' "
                            "pdb2gmx (GROMACS must be installed)")
    t.add_argument("protein_pdb")
    t.add_argument("output_dir")
    t.add_argument("--forcefield", default="charmm36-jul2022")
    t.add_argument("--water", default="tip3p")
    t.add_argument("--gmx", help="path to gmx, if it is not on PATH")

    g = sub.add_parser("ligand",
                       help="make a ligand's GROMACS topology from CGenFF: a "
                            ".str from ParamChem (no licence), a .mol2/.sdf via "
                            "the licensed cgenff binary, or a .pdb (protonated "
                            "to mol2 by Open Babel first)")
    g.add_argument("input", help="a .str stream, a .mol2/.sdf molecule, or a "
                                 ".pdb (Open Babel + cgenff binary)")
    g.add_argument("output_dir")
    g.add_argument("--ff", dest="cgenff_ff",
                   help="directory with top_all36_cgenff.rtf + "
                        "par_all36_cgenff.prm (a CHARMM-GUI toppar/); needed to "
                        "merge a real stream over the base CGenFF force field")
    g.add_argument("--resname", help="override the residue name in the .str")
    g.add_argument("--penalty-flag", dest="penalty_flag", type=float,
                   default=None,
                   help="also list every parameter whose CGenFF penalty exceeds "
                        "this (penalties are always reported, never refused)")
    g.add_argument("--cgenff", help="path to the cgenff binary (mol2/sdf/pdb "
                                    "input)")
    g.add_argument("--ph", type=float, default=7.0,
                   help="protonation pH for a .pdb input (default 7.0)")
    g.add_argument("--obabel", help="path to obabel, if not on PATH (.pdb input)")
    g.add_argument("--output-flag", dest="output_flag",
                   help="flag the cgenff build uses to name its output file, "
                        "if it does not write to stdout")

    m = sub.add_parser("mol2",
                       help="protonate a ligand PDB at a pH (default 7) and "
                            "write a ParamChem-ready mol2 with Open Babel (no "
                            "licence); take it to cgenff.silcsbio.com for a .str")
    m.add_argument("pdb", help="the ligand as a PDB")
    m.add_argument("output", help="the mol2 to write")
    m.add_argument("--ph", type=float, default=7.0,
                   help="protonation pH (default 7.0)")
    m.add_argument("--obabel", help="path to obabel, if it is not on PATH")
    m.add_argument("--no-h", dest="add_hydrogens", action="store_false",
                   help="convert as-is, without adding hydrogens for a pH")

    lb = sub.add_parser("ligand-batch",
                        help="parameterise a whole directory of .str files at "
                             "once; the base force field is read only once, so a "
                             "library is far faster than a shell loop")
    lb.add_argument("dir", help="a directory of .str stream files")
    lb.add_argument("output_dir")
    lb.add_argument("--ff", dest="cgenff_ff",
                    help="toppar/ with top_all36_cgenff.rtf + par_all36_cgenff.prm "
                         "to merge over (needed for real ParamChem streams)")
    lb.add_argument("--penalty-flag", dest="penalty_flag", type=float,
                    default=None,
                    help="also list every parameter whose penalty exceeds this")

    pl = sub.add_parser("place",
                        help="put a positioned ligand into a built protein "
                             "system (extends its .gro, topol.top and index)")
    pl.add_argument("system_dir", help="a built system dir (step5_input.gro, "
                                       "topol.top, index.ndx, toppar/)")
    pl.add_argument("ligand_pdb", help="the ligand, positioned in the box frame")
    pl.add_argument("ligand_itp", help="the ligand .itp from `lamellyx ligand`")
    pl.add_argument("output_dir")
    pl.add_argument("--atomtypes", dest="atomtypes_itp",
                    help="ligand _atomtypes.itp (default: alongside the .itp)")
    pl.add_argument("--resname", help="override the ligand residue name")

    cs = sub.add_parser("check-system",
                        help="structurally check a built system before grompp: "
                             "topology/coordinate atom counts, molecule "
                             "topologies, index coverage, net charge")
    cs.add_argument("system_dir", help="a built system directory")

    sub.add_parser("lipids", help="lipids with a conformer library")
    sub.add_parser("test", help="run the test suite")
    sub.add_parser("dashboard",
                   help="open the browser interface (its own flags are "
                        "passed straight through; try dashboard --help)")

    args = p.parse_args(argv)
    if not args.cmd:
        p.print_help()
        return 0

    if args.cmd == "schema":
        print(json.dumps(api.describe(), indent=2))
        return 0

    if args.cmd == "setup":
        from .make_data import install
        print(json.dumps(install(args.gromacs_dir, lipid=args.lipid,
                                 water=args.water, toppar=args.toppar),
                         indent=2))
        return 0

    if args.cmd == "orient":
        s = {"protein_pdb": args.protein_pdb, "output_pdb": args.out_pdb,
             "use_symmetry": args.symmetry}
        if args.extracellular:
            s["extracellular_resid"] = list(args.extracellular)
        print(json.dumps(api.orient_protein(s), indent=2))
        return 0

    if args.cmd == "topology":
        print(json.dumps(api.generate_protein_topology({
            "protein_pdb": args.protein_pdb,
            "output_dir": args.output_dir,
            "forcefield": args.forcefield,
            "water": args.water,
            **({"gmx": args.gmx} if args.gmx else {}),
        }), indent=2))
        return 0

    if args.cmd == "ligand":
        # Route on the extension so the one positional argument stays: a .str is
        # a ParamChem stream, a .pdb is protonated to mol2 first, anything else
        # is a mol2/sdf molecule for the binary.
        low = args.input.lower()
        key = ("str" if low.endswith(".str")
               else "pdb" if low.endswith(".pdb") else "molecule")
        s = {key: args.input, "output_dir": args.output_dir}
        if key == "pdb":
            s["ph"] = args.ph
            if args.obabel:
                s["obabel"] = args.obabel
        if args.cgenff_ff:
            s["cgenff_ff"] = args.cgenff_ff
        if args.resname:
            s["resname"] = args.resname
        if args.penalty_flag is not None:
            s["penalty_flag"] = args.penalty_flag
        if args.cgenff:
            s["cgenff"] = args.cgenff
        if args.output_flag:
            s["output_flag"] = args.output_flag
        try:
            print(json.dumps(api.generate_ligand_topology(s), indent=2))
        except (ValueError, RuntimeError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 1
        return 0

    if args.cmd == "ligand-batch":
        s = {"dir": args.dir, "output_dir": args.output_dir}
        if args.cgenff_ff:
            s["cgenff_ff"] = args.cgenff_ff
        if args.penalty_flag is not None:
            s["penalty_flag"] = args.penalty_flag
        try:
            r = api.generate_ligand_topologies(s)
        except (ValueError, FileNotFoundError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 1
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1

    if args.cmd == "mol2":
        s = {"pdb": args.pdb, "output": args.output, "ph": args.ph,
             "add_hydrogens": args.add_hydrogens}
        if args.obabel:
            s["obabel"] = args.obabel
        try:
            print(json.dumps(api.prepare_mol2(s), indent=2))
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 1
        return 0

    if args.cmd == "place":
        s = {"system_dir": args.system_dir, "ligand_pdb": args.ligand_pdb,
             "ligand_itp": args.ligand_itp, "output_dir": args.output_dir}
        if args.atomtypes_itp:
            s["atomtypes_itp"] = args.atomtypes_itp
        if args.resname:
            s["resname"] = args.resname
        try:
            print(json.dumps(api.place_ligand(s), indent=2))
        except (ValueError, FileNotFoundError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 1
        return 0

    if args.cmd == "check-system":
        try:
            r = api.check_system({"system_dir": args.system_dir})
        except (ValueError, FileNotFoundError) as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
            return 1
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1

    if args.cmd == "extract-protein":
        r = api.extract_protein(args.reference_dir, args.out_pdb,
                                args.molecules, args.chains)
        print(json.dumps(r, indent=2))
        return 0

    if args.cmd == "lipids":
        d = api.describe()
        print(json.dumps({"available": d["lipids_available"],
                          "measured_area_per_lipid_nm2":
                              d["measured_area_per_lipid_nm2"]}, indent=2))
        return 0

    if args.cmd == "test":
        from .tests import main as run_tests
        return run_tests([])


    if args.cmd == "check":
        r = api.check(_settings(args))
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            for m in r["errors"]:
                print("ERROR   " + m)
            for m in r["warnings"]:
                print("WARNING " + m)
            print("ok" if r["ok"] else "not buildable as configured")
        return 0 if r["ok"] else 1

    if args.cmd == "build":
        s = _settings(args)
        if args.quiet or args.json:
            s.setdefault("verbose", False)
        if not s.get("output_dir"):
            raise SystemExit("build needs --out or output_dir in the settings")
        try:
            r = api.build(s)
        except ValueError as exc:
            if args.json:
                print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
                return 1
            raise SystemExit(str(exc))
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            print("\n%s" % r["output_dir"])
            for k, v in r["counts"].items():
                print("   %-12s %s" % (k, v))
            print("   %-12s %.3f x %.3f x %.3f nm" % ("box", *r["box_nm"]))
            print("\nnext:\n   %s" % r["next_command"])
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
