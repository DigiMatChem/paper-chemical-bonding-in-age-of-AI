"""
Launch LOBSTER workflows for EOS relaxations, with PBE (not PBEsol),
for all EOS structures whose interlayer P-P distance is 
smaller than that of the ambient (equilibrium) structure, and the ambient structure.
The EOS subdirs are selected automatically by interlayer P-P distance.
"""

from __future__ import annotations

import glob
import os

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun

from atomate2.lobster.jobs import LobsterMaker
from atomate2.vasp.flows.mp import MPVaspLobsterMaker
from atomate2.vasp.powerups import (
    add_metadata_to_flow,
    update_user_incar_settings,
    update_user_potcar_functional,
    update_vasp_custodian_handlers,
)
from jobflow_remote import set_run_config, submit_flow
from custodian.vasp.handlers import (
    FrozenJobErrorHandler,
    IncorrectSmearingHandler,
    KspacingMetalHandler,
    LargeSigmaHandler,
    MeshSymmetryErrorHandler,
    NonConvergingErrorHandler,
    PositiveEnergyErrorHandler,
    PotimErrorHandler,
    StdErrHandler,
    UnconvergedErrorHandler,
    VaspErrorHandler,
)

# Default custodian handlers, minus the auto_nbands correction.
subset = list(VaspErrorHandler().error_msgs)
subset.remove("auto_nbands")
my_vasp_error_handler = VaspErrorHandler(errors_subset_to_catch=subset)

my_vasp_custodian_handlers = (
    my_vasp_error_handler,
    MeshSymmetryErrorHandler(),
    UnconvergedErrorHandler(),
    NonConvergingErrorHandler(),
    PotimErrorHandler(),
    PositiveEnergyErrorHandler(),
    FrozenJobErrorHandler(),
    StdErrHandler(),
    LargeSigmaHandler(),
    IncorrectSmearingHandler(),
    KspacingMetalHandler(),
)


EOS_ROOT = "eos_data"

# Ambient reference: EOS structures with interlayer P-P distance
# smaller than this one are submitted (the reference itself also).
AMBIENT_SUBDIR = "EOS_GGA_relax_2_EOS_equilibrium_relaxation"

EOS_SKIP = {
    "EOS_GGA_relax_1_EOS_equilibrium_relaxation"
}

PROJECT      = "mag"
LOCAL_WORKER = "local_worker"


def load_relaxed_structure(subdir_abs: str) -> Structure:
    """
    Load the relaxed structure from a VASP calc directory, gz-aware.

    Prefers CONTCAR(.gz); falls back to the final structure in vasprun.xml(.gz).
    """
    for name in ("CONTCAR.gz", "CONTCAR"):
        hits = glob.glob(os.path.join(subdir_abs, "**", name), recursive=True)
        if hits:
            return Structure.from_file(sorted(hits)[0])

    for name in ("vasprun.xml.gz", "vasprun.xml"):
        hits = glob.glob(os.path.join(subdir_abs, "**", name), recursive=True)
        if hits:
            vr = Vasprun(sorted(hits)[0], parse_dos=False, parse_eigen=False,
                         exception_on_bad_xml=False)
            return vr.final_structure

    raise FileNotFoundError(
        f"No CONTCAR(.gz) or vasprun.xml(.gz) found under {subdir_abs}"
    )


def interlayer_pp_distance(structure: Structure, angle_tol_deg: float = 20.0,
                           rmax: float = 8.0) -> float:
    lengths = np.linalg.norm(structure.lattice.matrix, axis=1)
    long_axis = structure.lattice.matrix[int(np.argmax(lengths))]
    axis_hat = long_axis / np.linalg.norm(long_axis)
    cos_tol = np.cos(np.deg2rad(angle_tol_deg))

    dmin = float("inf")
    for site in structure:
        if site.specie.symbol != "P":
            continue
        for nbr in structure.get_neighbors(site, rmax):
            if nbr.specie.symbol != "P":
                continue
            vec = nbr.coords - site.coords
            dist = np.linalg.norm(vec)
            if dist < 1e-6:
                continue
            if abs(np.dot(vec, axis_hat) / dist) >= cos_tol:
                dmin = min(dmin, dist)
    return dmin if np.isfinite(dmin) else float("nan")


def select_subdirs_below_ambient() -> list[str]:
    """
    Return EOS subdirs whose interlayer P-P distance is strictly smaller than
    the ambient reference, logging the decision for each candidate.
    """
    ambient_abs = os.path.join(EOS_ROOT, AMBIENT_SUBDIR)
    ref_d = interlayer_pp_distance(load_relaxed_structure(ambient_abs))
    print(f"Ambient reference ({AMBIENT_SUBDIR}): d(P-P) = {ref_d:.3f} A\n")

    selected = []
    for name in sorted(os.listdir(EOS_ROOT)):
        if name in EOS_SKIP or name == AMBIENT_SUBDIR:
            continue
        path = os.path.join(EOS_ROOT, name)
        if not os.path.isdir(path):
            continue
        try:
            d = interlayer_pp_distance(load_relaxed_structure(path))
        except FileNotFoundError as exc:
            print(f"SKIP {name}: {exc}")
            continue
        if d < ref_d - 1e-6:
            print(f"  select {name}: d(P-P) = {d:.3f} A  (< {ref_d:.3f})")
            selected.append(name)
        else:
            print(f"  skip   {name}: d(P-P) = {d:.3f} A  (>= {ref_d:.3f})")
    return selected


def main():
    task_kwargs = {
        'add_coxxcar_to_task_document': False,
        'save_computational_data_jsons': False,
        'analyze_outputs': False,
        'save_cohp_plots': False,
        'save_cba_jsons': False,
    }

    resources = {
        "nodes": 1,
        "partition": "micro",
        "time": "08:00:00",
        "ntasks": 48,
        "mail_user": "katharina.ueltzen@bam.de",
        "mail_type": "ALL",
        "exclusive": "exclusive",
        "account": "pn73da",
        "qverbatim": "#SBATCH --get-user-env",
    }

    resources_lob = {
        "nodes": 1,
        "partition": "micro",
        "time": "08:00:00",
        "ntasks": 48,
        "mail_user": "katharina.ueltzen@bam.de",
        "mail_type": "ALL",
        "exclusive": "exclusive",
        "account": "pn73da",
        "qverbatim": "#SBATCH --get-user-env",
    }

    eos_subdirs = [AMBIENT_SUBDIR] + select_subdirs_below_ambient()

    for subdir in eos_subdirs:
        subdir_abs = os.path.join(EOS_ROOT, subdir)
        if not os.path.isdir(subdir_abs):
            print(f"SKIP: '{subdir_abs}' is not a directory.")
            continue

        structure = load_relaxed_structure(subdir_abs)
        name_of_eos_calc = subdir

        # Build the LOBSTER flow starting from the provided geometry (no relax).
        lobster = MPVaspLobsterMaker(
            relax_maker=None,
            lobster_maker=LobsterMaker(task_document_kwargs=task_kwargs),
            delete_wavecars=False,
        ).make(structure)

        lobster = update_vasp_custodian_handlers(
            lobster, my_vasp_custodian_handlers
        )
        
        lobster = update_user_incar_settings(
            lobster,
            {"NPAR": 4, "ISMEAR": 1, "SIGMA": 0.05},
        )
        lobster = update_user_potcar_functional(
            lobster, potcar_functional="PBE_54"
        )

        lobster = add_metadata_to_flow(
            flow=lobster,
            additional_fields={"eos_id": f"{name_of_eos_calc}"},
            class_filter=LobsterMaker,
        )

        lobster = set_run_config(
            lobster, name_filter="MP",
            worker="supermuc_slurm_worker", resources=resources,
        )
        lobster = set_run_config(
            lobster, name_filter="lobster_run",
            worker="supermuc_slurm_lobster_worker", resources=resources_lob,
        )

        print(f"Submitting LOBSTER flow for {name_of_eos_calc}")
        print(submit_flow(lobster, worker=LOCAL_WORKER, project=PROJECT))


if __name__ == "__main__":
    main()
