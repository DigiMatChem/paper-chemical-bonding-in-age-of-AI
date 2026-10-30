"""
Postprocessing: is the interlayer P-P contact of LaRu2P2 bonding?

For every LOBSTER subdirectory in LOBSTER_DIR it prints, per calculation:
  - the interlayer P-P distance (shortest P-P along the longest crystal axis),
  - for each LocalEnv (structure-based) algorithm, whether the interlayer P-P
    contact is counted as a bond, and
  - the same for LOBSTER ICOHP coordination at three perc_strength_icohp
    cutoffs (0.05, 0.10, 0.15 of the strongest ICOHP in the compound).
"""

from __future__ import annotations

import os
import warnings

import numpy as np


warnings.filterwarnings("ignore", message=".*No oxidation states specified.*")
warnings.filterwarnings("ignore", message=".*cannot locate an appropriate radius.*")

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


LOBSTER_DIR = "lobster_data"
ICOHP_CUTOFFS = [0.05, 0.10, 0.15]
PP_MATCH_TOL = 0.02

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


def _symbol(site) -> str:
    try:
        return site.specie.symbol
    except Exception:
        return site.species_string


def representative_index(structure: Structure, element: str) -> int | None:
    for i, site in enumerate(structure):
        if _symbol(site) == element:
            return i
    return None


def interlayer_pp_distance(structure: Structure, angle_tol_deg: float = 0.5,
                           rmax: float = 8.0) -> float:
    """Shortest P-P contact lying within angle_tol_deg of the longest lattice
    vector (the interlayer P-P bond in the ThCr2Si2-type structure)."""
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


def interlayer_pp_is_bonding(bonded, structure: Structure, d_pp: float) -> bool:
    """True if the representative P site is connected to a P neighbour at
       the interlayer distance d_pp."""
    idx = representative_index(structure, "P")
    if idx is None or not np.isfinite(d_pp):
        return False
    for cs in bonded.get_connected_sites(idx):
        if _symbol(cs.site) == "P" and abs(cs.dist - d_pp) <= PP_MATCH_TOL:
            return True
    return False


def _verdict(flag) -> str:
    if isinstance(flag, str):
        return f"FAILED: {flag}"
    return "bonding" if flag else "not bonding"


def geometric_bonding(structure: Structure, d_pp: float) -> dict:
    """{method_label: bool or error-message-str}."""
    geo = structure.copy()
    geo.remove_oxidation_states()  # as in MaterialsCoord
    results = {}
    for label, factory in GEOMETRIC_METHODS:
        try:
            bonded = factory().get_bonded_structure(geo)  # as in MaterialsCoord
            results[label] = interlayer_pp_is_bonding(bonded, geo, d_pp)
        except Exception as exc:
            results[label] = str(exc)
    return results


def lobster_bonding(subdir_abs: str, structure: Structure, d_pp: float) -> dict:
    """{cutoff: bool or error-message-str}."""
    icohp = os.path.join(subdir_abs, "ICOHPLIST.lobster.gz")
    charge = os.path.join(subdir_abs, "CHARGE.lobster.gz")
    results = {}
    for cutoff in ICOHP_CUTOFFS:
        try:
            ln = LobsterNeighbors(
                are_coops=False,
                structure=structure,
                filename_icohp=icohp,
                valences_from_charges=True,
                filename_charge=charge,
                perc_strength_icohp=cutoff,
                which_charge="Mulliken",
                additional_condition=0,   # all bonds included
            )
            bonded = ln.get_bonded_structure(structure=structure)
            results[cutoff] = interlayer_pp_is_bonding(bonded, structure, d_pp)
        except Exception as exc:
            results[cutoff] = str(exc)
    return results


def _basis_of(subdir: str) -> str:
    """The local basis id: the trailing run index (0 or 1), else 'unknown'."""
    tail = subdir.rsplit("_", 1)[-1]
    return tail if tail in ("0", "1") else "unknown"


def _max_bonding(distances: list[float]) -> str:
    return f"{max(distances):.3f} A" if distances else "never bonding"


def print_summary(geo_bonding_d: dict, lob_bonding_d: dict) -> None:
    """geo_bonding_d: {method -> [d_pp where bonding]}.
    lob_bonding_d: {(cutoff, basis) -> [d_pp where bonding]}."""
    print("#" * 72)
    print("SUMMARY: maximum interlayer d(P-P) still classified as bonding")
    print("(largest value among the sampled volumes -- resolution-limited)")
    print("#" * 72)

    print("\nLocalEnv (geometry only):")
    for label, _ in GEOMETRIC_METHODS:
        print(f"  {label:<28} {_max_bonding(geo_bonding_d.get(label, []))}")

    print("\nLOBSTER (per local basis, fraction of strongest ICOHP):")
    bases = sorted({b for (_c, b) in lob_bonding_d})
    for basis in bases:
        print(f"  basis {basis}:")
        for cutoff in ICOHP_CUTOFFS:
            dists = lob_bonding_d.get((cutoff, basis), [])
            print(f"    cutoff {cutoff:<4} {_max_bonding(dists)}")


def main():
    subdirs = sorted(
        d for d in os.listdir(LOBSTER_DIR)
        if os.path.isdir(os.path.join(LOBSTER_DIR, d))
    )

    geo_bonding_d: dict[str, list[float]] = {}
    lob_bonding_d: dict[tuple[float, str], list[float]] = {}

    for subdir in subdirs:
        subdir_abs = os.path.join(LOBSTER_DIR, subdir)
        contcar = os.path.join(subdir_abs, "CONTCAR.gz")

        print("=" * 72)
        print(subdir)
        print("=" * 72)

        if not os.path.isfile(contcar):
            print(f"  no CONTCAR.gz in {subdir_abs} -- skipped\n")
            continue

        structure = Structure.from_file(contcar)
        d_pp = interlayer_pp_distance(structure)
        basis = _basis_of(subdir)
        print(f"  interlayer d(P-P) = {d_pp:.3f} A  -- bonding?")

        print("  -- LocalEnv --")
        for label, flag in geometric_bonding(structure, d_pp).items():
            print(f"    {label:<28} {_verdict(flag)}")
            if flag is True:
                geo_bonding_d.setdefault(label, []).append(d_pp)

        print("  -- LOBSTER (fraction of strongest ICOHP) --")
        for cutoff, flag in lobster_bonding(subdir_abs, structure, d_pp).items():
            print(f"    cutoff {cutoff:<4} {_verdict(flag)}")
            if flag is True:
                lob_bonding_d.setdefault((cutoff, basis), []).append(d_pp)
        print()

    print_summary(geo_bonding_d, lob_bonding_d)


if __name__ == "__main__":
    main()
