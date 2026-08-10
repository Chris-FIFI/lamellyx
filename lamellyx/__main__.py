"""Command line for lamellyx.

    lamellyx setup <gromacs_dir>       fetch the data the repo does not ship
    lamellyx schema                    what every setting means
    lamellyx check    --set x=2.0      validate without building
    lamellyx build    --out popc_box   build one
    lamellyx orient   in.pdb out.pdb   put a protein in the membrane frame
    lamellyx topology in.pdb out_dir   make its .itp files via gmx pdb2gmx
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
