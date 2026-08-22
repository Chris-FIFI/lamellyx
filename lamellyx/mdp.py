"""The CHARMM36m equilibration and production mdp series.

These reproduce CHARMM-GUI's GROMACS output for a membrane protein: six
equilibration stages that release position and dihedral restraints in steps,
then production. The restraint force constants are the schedule CHARMM-GUI
uses, and they are what makes it safe to start from a packed rather than an
equilibrated bilayer.

When a reference system is available, `copy_from` is preferred over
`write_series` -- byte-identical run parameters are what make two boxes
comparable, and there is nothing to gain from regenerating them.
"""

from __future__ import annotations

import os
import shutil

_COMMON = """\
cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = 1.2
vdwtype                 = Cut-off
vdw-modifier            = Force-switch
rvdw_switch             = 1.0
rvdw                    = 1.2
coulombtype             = PME
rcoulomb                = 1.2
"""

_TCOUPL = """\
tcoupl                  = v-rescale
tc_grps                 = SOLU MEMB SOLV
tau_t                   = 1.0 1.0 1.0
ref_t                   = {T} {T} {T}
"""

# With no protein there is no SOLU group, and grompp refuses an empty one.
_TCOUPL_MEMB = """\
tcoupl                  = v-rescale
tc_grps                 = MEMB SOLV
tau_t                   = 1.0 1.0
ref_t                   = {T} {T}
"""

_PCOUPL = """\
pcoupl                  = C-rescale
pcoupltype              = semiisotropic
tau_p                   = 5.0
compressibility         = 4.5e-5  4.5e-5
ref_p                   = 1.0     1.0
"""

_COMM = """\
nstcomm                 = 100
comm_mode               = linear
comm_grps               = SOLU_MEMB SOLV
"""

_COMM_MEMB = """\
nstcomm                 = 100
comm_mode               = linear
comm_grps               = MEMB SOLV
"""

# (name, bb, sc, lipid, dihres, dt, nsteps, barostat, gen_vel)
#
# The final stage is 5 ns rather than CHARMM-GUI's 0.5 ns. It is the only stage
# where the lipids and side chains are completely free -- every restraint above
# backbone 50 kJ/mol/nm^2 has been released -- so it is the only one where the
# bilayer can actually relax to its own area per lipid and thickness. Half a
# nanosecond is not long enough to see that happen, which means the state
# production inherits has never been shown to be settled.
_SCHEDULE = [
    ("step6.1_equilibration", 4000.0, 2000.0, 1000.0, 1000.0, 0.001, 125000, False, True),
    ("step6.2_equilibration", 2000.0, 1000.0, 400.0, 400.0, 0.001, 125000, False, False),
    ("step6.3_equilibration", 1000.0, 500.0, 400.0, 200.0, 0.001, 125000, True, False),
    ("step6.4_equilibration", 500.0, 200.0, 200.0, 200.0, 0.002, 250000, True, False),
    ("step6.5_equilibration", 200.0, 50.0, 40.0, 100.0, 0.002, 250000, True, False),
    ("step6.6_equilibration", 50.0, 0.0, 0.0, 0.0, 0.002, 2500000, True, False),
]

MDP_FILES = (["step6.0_minimization.mdp"]
             + ["%s.mdp" % s[0] for s in _SCHEDULE]
             + ["step7_production.mdp"])


def _define(bb, sc, lip, dih):
    return ("define                  = -DPOSRES -DPOSRES_FC_BB=%.1f "
            "-DPOSRES_FC_SC=%.1f -DPOSRES_FC_LIPID=%.1f -DDIHRES "
            "-DDIHRES_FC=%.1f\n" % (bb, sc, lip, dih))


def minimisation(emtol=1000.0, nsteps=5000):
    return (_define(4000.0, 2000.0, 1000.0, 1000.0)
            + "integrator              = steep\n"
            + "emtol                   = %.1f\n" % emtol
            + "nsteps                  = %d\n" % nsteps
            + "nstlist                 = 10\n"
            + _COMMON + ";\n"
            + "constraints             = h-bonds\n"
            + "constraint_algorithm    = LINCS\n")


def _md(bb, sc, lip, dih, dt, nsteps, barostat, gen_vel, T, membrane_only=False):
    out = _define(bb, sc, lip, dih)
    out += ("integrator              = md\n"
            "dt                      = %.3f\n"
            "nsteps                  = %d\n"
            "nstxout-compressed      = 5000\n"
            "nstxout                 = 0\n"
            "nstvout                 = 0\n"
            "nstfout                 = 0\n"
            "nstcalcenergy           = 100\n"
            "nstenergy               = 1000\n"
            "nstlog                  = 1000\n;\n" % (dt, nsteps))
    tc = _TCOUPL_MEMB if membrane_only else _TCOUPL
    out += _COMMON + ";\n" + tc.format(T=T) + ";\n"
    if barostat:
        out += _PCOUPL + ";\n"
    out += ("constraints             = h-bonds\n"
            "constraint_algorithm    = LINCS\n")
    out += "continuation            = yes\n;\n" if not gen_vel else ";\n"
    out += _COMM_MEMB if membrane_only else _COMM
    if gen_vel:
        out += (";\ngen-vel                 = yes\n"
                "gen-temp                = %s\n"
                "gen-seed                = -1\n" % T)
    return out


def production(dt=0.002, nsteps=500000, T=310, membrane_only=False):
    out = ("integrator              = md\n"
           "dt                      = %.3f\n"
           "nsteps                  = %d\n"
           "nstxout-compressed      = 50000\n"
           "nstxout                 = 0\n"
           "nstvout                 = 0\n"
           "nstfout                 = 0\n"
           "nstcalcenergy           = 100\n"
           "nstenergy               = 1000\n"
           "nstlog                  = 1000\n;\n" % (dt, nsteps))
    tc = _TCOUPL_MEMB if membrane_only else _TCOUPL
    out += _COMMON + ";\n" + tc.format(T=T) + ";\n" + _PCOUPL + ";\n"
    out += ("constraints             = h-bonds\n"
            "constraint_algorithm    = LINCS\n"
            "continuation            = yes\n;\n")
    out += _COMM_MEMB if membrane_only else _COMM
    return out


def write_series(directory, temperature=310, membrane_only=False):
    """Write the whole mdp series into `directory`."""
    os.makedirs(directory, exist_ok=True)
    written = []
    p = os.path.join(directory, "step6.0_minimization.mdp")
    with open(p, "w", newline="\n") as fh:
        fh.write(minimisation())
    written.append(p)
    for name, bb, sc, lip, dih, dt, nsteps, baro, gv in _SCHEDULE:
        p = os.path.join(directory, name + ".mdp")
        with open(p, "w", newline="\n") as fh:
            fh.write(_md(bb, sc, lip, dih, dt, nsteps, baro, gv, temperature,
                         membrane_only=membrane_only))
        written.append(p)
    p = os.path.join(directory, "step7_production.mdp")
    with open(p, "w", newline="\n") as fh:
        fh.write(production(T=temperature, membrane_only=membrane_only))
    written.append(p)
    return written


def copy_from(src, dst):
    """Copy an existing mdp series. Returns the files copied."""
    os.makedirs(dst, exist_ok=True)
    out = []
    for fn in MDP_FILES:
        s = os.path.join(src, fn)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(dst, fn))
            out.append(fn)
    return out
