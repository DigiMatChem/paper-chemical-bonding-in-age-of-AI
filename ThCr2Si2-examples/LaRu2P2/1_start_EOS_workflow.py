from atomate2.vasp.flows.eos import EosMaker
from atomate2.vasp.jobs.eos import EosRelaxMaker
from atomate2.vasp.sets.eos import EosSetGenerator
from atomate2.vasp.powerups import update_user_incar_settings, update_user_potcar_functional

from jobflow_remote import submit_flow, set_run_config
from pymatgen.core.structure import Structure


# Structure from MP
structure = Structure.from_file("LaRu2P2.cif")

eos = EosMaker(
    eos_relax_maker=EosRelaxMaker(
        input_set_generator=EosSetGenerator(
            user_incar_settings={"ISIF": 4},
        )
    ),
    linear_strain=(-0.0274, 0.017),   # ~ f in [0.92, 1.05]
    number_of_frames=10,
).make(structure=structure)

eos = update_user_incar_settings(eos, 
                                        {"NPAR": 4, 
                                        "ISMEAR": 1, 
                                        "SIGMA": 0.05}) 
 
eos = update_user_potcar_functional(eos, potcar_functional="PBE_54")

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

eos = set_run_config(
            eos, name_filter="EOS GGA", worker="supermuc_slurm_worker", resources=resources,
            )

print(submit_flow(eos, worker="local_worker", project="mag"))
