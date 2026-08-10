"""Three ways to use lamellyx, shortest first.

    python examples/popc_bilayer.py
"""

import json
import tempfile
import os

from lamellyx.api import build, check, describe

out = os.path.join(tempfile.gettempdir(), "lamellyx_example")


# 1 --- look before you leap ------------------------------------------------
# check() is instant and catches a box GROMACS would refuse.
print("a 2 nm box:", json.dumps(check({"x": 2.0, "y": 2.0})["errors"], indent=2))
print("an 8 nm box:", check({"x": 8.0, "y": 8.0}))


# 2 --- a plain bilayer -----------------------------------------------------
result = build({
    "output_dir": out,
    "x": 6.0,                    # nm, full box edge
    "y": 6.0,
    "area_per_lipid": 0.643,     # nm^2, measured POPC
    "water_thickness": 1.5,      # nm above and below
    "salt_concentration": 0.15,  # physiological
    "seed": 1,                   # same seed, same box
    "verbose": False,
})
print("\nbuilt:", result["counts"])
print("box:", result["box_nm"], "nm")
print("closest heavy-atom contact:",
      result["closest_heavy_atom_contact_A"], "A   (good is > 2.2)")
print("density:", result["density_g_cm3"], "g/cm3  (good is 0.90-0.98)")
print("net charge:", result["net_charge"], "  (must be 0)")
print("\nrun it with:\n  ", result["next_command"])


# 3 --- what else can be set ------------------------------------------------
schema = describe()
print("\nevery setting:")
for name, info in schema["settings"].items():
    units = (" [%s]" % info["units"]) if info["units"] else ""
    print("   %-22s %-14r%s" % (name, info["default"], units))
