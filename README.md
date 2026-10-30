# paper-chemical-bonding-in-age-of-AI

## About
This repository contains data and code for the publication "In the age of big data and AI: do we still need chemical bonds?" by Mohammed Al-Fahdi et al.

## Project structure
The submodule *MACE_Allegro_LOBSTER_Analysis* compares machine learning interatomic potential (MLIP) structure descriptors with LOBSTER-derived bonding and charge information.

The submodule *ESF-MEGNet-Framework* contains materials property prediction with MEGNet and electronic structure features.

The folder *ThCr2Si2-examples* compares coordination analyses of localenv- and LOBSTER-based algorithms for different compounds of the ThCr2Si2 structure type. It contains scripts to start and postprocess the required calculations and to obtain the plot from the publication. Please note that the raw calculation output is only included in the Zenodo repository.

## Installation
For including the submodules in the download, run `git clone` with the `--recurse-submodules` flag.

`git clone --recurse-submodules git@github.com:DigiMatChem/paper-chemical-bonding-in-age-of-AI.git`

For running the ThCr2Si2-examples scripts, install the packages from [requirements_thcr2si2.txt](ThCr2Si2-examples/requirements_thcr2si2.txt) in an e.g. conda environment with python 3.10.
For running the code in the submodules, please refer to the installation instructions in the submodules.

## License
All code is licensed under a BSD-3 license ([LICENSE_CODE](LICENSE_CODE)), 
while the data of this repository is released under a CC-BY license ([LICENSE_DATA](LICENSE_DATA)).
The latter excludes crystallographic structure data from the MaterialsCoord benchmark and the MaterialsProject (see [note on CeAl2Ga2 structure](ThCr2Si2-examples/CeAl2Ga2-benchmark-example/NOTE.md) and [note on LaRu2P2 structure](ThCr2Si2-examples/LaRu2P2/NOTE.md)).
Licenses of the submodules are included in the linked repositories.

