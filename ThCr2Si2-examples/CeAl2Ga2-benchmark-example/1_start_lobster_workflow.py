"""
Launch LOBSTER workflow for CeAl2Ga2.
"""

from __future__ import annotations

from pymatgen.core import Structure

from atomate2.lobster.jobs import LobsterMaker
from atomate2.vasp.flows.mp import MPVaspLobsterMaker
from atomate2.vasp.powerups import (
    update_user_incar_settings,
    update_user_potcar_functional,
    update_vasp_custodian_handlers,
)
from jobflow_remote import set_run_config, submit_flow
import json
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


PROJECT      = "mag"
LOCAL_WORKER = "local_worker"


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


# Load CeAl2Ga2 structure
with open("CeAl2Ga2_606460.json") as f:
    structure_dict = json.load(f)
    structure = Structure.from_dict(structure_dict)


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

lobster = set_run_config(
    lobster, name_filter="MP",
    worker="supermuc_slurm_worker", resources=resources,
)
lobster = set_run_config(
    lobster, name_filter="lobster_run",
    worker="supermuc_slurm_lobster_worker", resources=resources_lob,
)

print(submit_flow(lobster, worker=LOCAL_WORKER, project=PROJECT))
