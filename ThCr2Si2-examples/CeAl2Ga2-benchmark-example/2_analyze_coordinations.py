"""
Compare LOBSTER-ICOHP and geometric coordination environments of CeAl2Ga2
against the MaterialsCoord literature coordination numbers.
Print the results.
"""

from __future__ import annotations

import json
import os
import re

from pymatgen.core import Structure
from pymatgen.analysis.local_env import (
    CrystalNN,
    EconNN,
    JmolNN,
    MinimumDistanceNN,
    MinimumOKeeffeNN,
    MinimumVIRENN,
    VoronoiNN,
)
try:
    from pymatgen.analysis.local_env import BrunnerNN_reciprocal
except ImportError:  
    from pymatgen.analysis.local_env import (
        BrunnerNNReciprocal as BrunnerNN_reciprocal,
    )
from pymatgen.io.lobster.lobsterenv import LobsterNeighbors


STRUCTURE_JSON = "CeAl2Ga2_606460.json"
LOBSTER_DIR = "lobster_data/ceal2ga2"
ICOHP_CUTOFF = 0.1
ELEMENTS = ["Ce", "Al", "Ga"]

GEOMETRIC_METHODS = [
    ("MinimumDistanceNN",          lambda: MinimumDistanceNN()),
    ("MinimumOKeeffeNN",           lambda: MinimumOKeeffeNN()),
    ("MinimumVIRENN",              lambda: MinimumVIRENN()),
    ("JmolNN",                     lambda: JmolNN()),
    ("EconNN(tol=0.5)",            lambda: EconNN(tol=0.5)),
    ("BrunnerNN_reciprocal",       lambda: BrunnerNN_reciprocal()),
    ("VoronoiNN(tol=0.5)",         lambda: VoronoiNN(tol=0.5)),
    ("CrystalNN",                  lambda: CrystalNN()),
    ("CrystalNN(x_diff_weight=0)", lambda: CrystalNN(x_diff_weight=0)),
]


# Helper functions
def _symbol(site):
    try:
        return site.specie.symbol
    except Exception:
        return re.sub(r"[^A-Za-z]", "", site.species_string)


def representative_index(structure, element):
    """
    Get first site index of an element 
    (f. CeAl2Ga2 there is one crystallographically unique site per element)
    """
    for i, site in enumerate(structure):
        if _symbol(site) == element:
            return i
    return None


def load_structure_and_literature(json_path):
    """Load the benchmark structure and the per-element literature coordination."""
    with open(json_path) as f:
        data = json.load(f)
    structure = Structure.from_dict(data)
    literature = {}
    for site in structure:
        el = _symbol(site)
        if el not in literature:
            literature[el] = site.properties["coordination"]
    return structure, literature


def geometric_cn_per_element(nn, structure):
    """
    Element-resolved neighbour counts for localenv-Algorithms.
    """
    bonded = nn.get_bonded_structure(structure)
    out = {}
    for el in ELEMENTS:
        idx = representative_index(structure, el)
        if idx is None:
            continue
        cn = {}
        for cs in bonded.get_connected_sites(idx):
            e = _symbol(cs.site)
            cn[e] = cn.get(e, 0) + 1
        out[el] = cn
    return out


def lobster_cn_per_element(lobster_dir, structure, cutoff):
    """
    Element-resolved neighbour counts from LobsterNeighbors 
    (10 %-strongest-ICOHP cutoff, all bonds).
    """
    icohp = os.path.join(lobster_dir, "ICOHPLIST.lobster.gz")
    charge = os.path.join(lobster_dir, "CHARGE.lobster.gz")
    

    ln = LobsterNeighbors(
        structure=structure,
        filename_icohp=icohp,
        valences_from_charges=True,
        filename_charge=charge,
        perc_strength_icohp=cutoff,
        which_charge="Mulliken",
        additional_condition=0,
    )
    bonded = ln.get_bonded_structure(structure=structure)

    out = {}
    for el in ELEMENTS:
        idx = representative_index(structure, el)
        if idx is None:
            continue
        cn = {}
        for cs in bonded.get_connected_sites(idx):
            e = _symbol(cs.site)
            cn[e] = cn.get(e, 0) + 1
        out[el] = cn
    return out


def compare(pred, literature):
    """
    Return (agrees, list_of_mismatch_strings) for one site.

    For each element appearing in either dict, the predicted count must equal
    the literature value, or be one of the accepted values if a list is given.
    """
    mismatches = []
    for el in sorted(set(pred) | set(literature)):
        p = pred.get(el, 0)
        a = literature.get(el, 0)
        allowed = a if isinstance(a, list) else [a]
        if p not in allowed:
            mismatches.append(f"{el}: predicted {p}, accepted {allowed}")
    return (len(mismatches) == 0, mismatches)


def main():
    structure, literature = load_structure_and_literature(STRUCTURE_JSON)

    # Load structure from JSON for geometric methods
    # Remove oxidation states as in MaterialsCoord benchmark
    geo_structure = structure.copy()
    geo_structure.remove_oxidation_states()

    # Print literature values f. coordination
    print("=" * 70)
    print("Literature coordination (from JSON; lists = several accepted values)")
    print("=" * 70)
    for el in ELEMENTS:
        print(f"  {el}: {literature[el]}")

    predictions = {}

    # Load LOBSTER structure and coordinations
    lob_label = f"LOBSTER (ICOHP cutoff {ICOHP_CUTOFF:g})"
    lob_struct_path = os.path.join(LOBSTER_DIR, "CONTCAR.gz")
   
    try:
        lob_structure = Structure.from_file(lob_struct_path)
        predictions[lob_label] = lobster_cn_per_element(
            LOBSTER_DIR, lob_structure, ICOHP_CUTOFF)
    except Exception as exc:
        predictions[lob_label] = {"__error__": str(exc)}

    # Get coordination of geometric methods
    for label, factory in GEOMETRIC_METHODS:
        try:
            predictions[label] = geometric_cn_per_element(factory(), geo_structure)
        except Exception as exc:
            predictions[label] = {"__error__": str(exc)}

    # Print disagreements per method
    print("\n" + "=" * 70)
    print("Disagreements with literature coordination")
    print("=" * 70)
    for method, per_el in predictions.items():
        print(f"\n{method}")
        if "__error__" in per_el:
            print(f"  method failed: {per_el['__error__']}")
            continue

        any_disagreement = False
        for el in ELEMENTS:
            cn = per_el.get(el)
            if isinstance(cn, dict) and "__error__" in cn:
                any_disagreement = True
                print(f"  {el} site: FAILED ({cn['__error__']})")
                continue
            agrees, mismatches = compare(cn, literature[el])
            if not agrees:
                any_disagreement = True
                print(f"  {el} site: predicted {cn}  vs literature "
                      f"{literature[el]}")
                for m in mismatches:
                    print(f"       -> {m}")
        if not any_disagreement:
            print("  all sites agree with literature")


if __name__ == "__main__":
    main()
