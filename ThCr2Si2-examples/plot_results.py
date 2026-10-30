"""
Failure-mode figure for geometric vs LOBSTER-ICOHP bond detection in
ThCr2Si2-type compounds.

Subplot 1 (CeAl2Ga2): is the Al-Al metal-metal bond detected? Green = yes,
red = no, per method.

Subplot 2 (LaRu2P2, pressure/volume series): the largest interlayer P-P
distance at which each method still assigns a P-P bond. A dashed line marks
the Hoffmann-Zheng 2.8 A threshold; a grey column background marks methods
that never detect a P-P bond over the sampled range. Markers are coloured by
how well the detection onset matches the threshold: green if a method's
largest bonded distance equals the sampled distance closest to 2.8 A (ideal
onset), red otherwise.
Plotted range: ambient to smallest volume. Only LOBSTER basis 0 is shown, 
basis 1 has same results (see LaRu2P2/3_analyze_P-P-bonding.py).
"""

from __future__ import annotations

import os
from collections import defaultdict

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42

import matplotlib.pyplot as plt

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
except ImportError:  # renamed in newer pymatgen
    from pymatgen.analysis.local_env import (
        BrunnerNNReciprocal as BrunnerNN_reciprocal,
    )
from pymatgen.core import Structure
from pymatgen.io.lobster.lobsterenv import LobsterNeighbors


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CEAL2GA2_LOBSTER_DIR = os.path.join(
    BASE_DIR, "CeAl2Ga2-benchmark-example", "lobster_data", "ceal2ga2")

LARU2P2_ROOT = os.path.join(BASE_DIR, "LaRu2P2", "lobster_data")


def laru2p2_lobster_subdirs() -> list[str]:
    """Basenames of all LaRu2P2 LOBSTER run dirs in LARU2P2_ROOT."""
    subs = [d for d in os.listdir(LARU2P2_ROOT)
            if "__lobster_run_" in d
            and os.path.isdir(os.path.join(LARU2P2_ROOT, d))]
    return sorted(subs)

EOS_EQUILIBRIUM_NAME = "EOS_GGA_relax_2_EOS_equilibrium_relaxation"

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
OUTPUT_PDF = os.path.join(PLOTS_DIR, "thcr2si2_bonding_failure_modes.pdf")
DPI = 300


# coordination detection methods
GEOMETRIC_METHODS = [
    ("MinimumDistanceNN",  lambda: MinimumDistanceNN()),
    ("MinimumOKeeffeNN",   lambda: MinimumOKeeffeNN()),
    ("MinimumVIRENN",      lambda: MinimumVIRENN()),
    ("JmolNN",             lambda: JmolNN()),
    ("EconNN",             lambda: EconNN(tol=0.5)),
    ("BrunnerNN_recip.",   lambda: BrunnerNN_reciprocal()),
    ("VoronoiNN",          lambda: VoronoiNN(tol=0.5)),
    ("CrystalNN",          lambda: CrystalNN()),
    ("CrystalNN\n(x_diff=0)", lambda: CrystalNN(x_diff_weight=0)),
]
LOBSTER_CUTOFFS = [0.10, 0.15]
LOBSTER_LABELS = [f"LOBSTER\nICOHP {c:.2f}" for c in LOBSTER_CUTOFFS]

X_LABELS = [m[0] for m in GEOMETRIC_METHODS] + LOBSTER_LABELS
X_POS = list(range(len(X_LABELS)))
N_GEOM = len(GEOMETRIC_METHODS)

HZ_THRESHOLD = 2.8
C_AXIS_ANGLE_TOL_DEG = 0.5


def load_structure(calc_dir):
    """Read the relaxed structure from CONTCAR.gz directly in calc_dir."""
    return Structure.from_file(os.path.join(calc_dir, "CONTCAR.gz"))


def representative_index(structure, element):
    for i, site in enumerate(structure):
        if site.specie.symbol == element:
            return i
    return None


def geometric_pair_count(nn, structure, center_el, partner_el, along_c=False):
    """Number of `partner_el` neighbours of the representative `center_el` site.
    If along_c, restrict to bonds within C_AXIS_ANGLE_TOL_DEG of the c axis."""
    idx = representative_index(structure, center_el)
    if idx is None:
        return 0
    try:
        info = nn.get_nn_info(structure, idx)
    except Exception:
        return np.nan   # method failed
    c_hat = np.array(structure.lattice.matrix[2], float)
    c_hat /= np.linalg.norm(c_hat)
    cos_tol = np.cos(np.deg2rad(C_AXIS_ANGLE_TOL_DEG))
    p0 = structure[idx].coords
    count = 0
    for n in info:
        if n["site"].specie.symbol != partner_el:
            continue
        if along_c:
            vec = n["site"].coords - p0
            d = np.linalg.norm(vec)
            if d < 1e-6 or abs(np.dot(vec, c_hat) / d) < cos_tol:
                continue
        count += 1
    return count


def lobster_neighbors(lobster_dir, structure, cutoff):
    icohp = os.path.join(lobster_dir, "ICOHPLIST.lobster.gz")
    charge = os.path.join(lobster_dir, "CHARGE.lobster.gz")
    return LobsterNeighbors(
        structure=structure,
        filename_icohp=icohp,
        valences_from_charges=True,
        filename_charge=charge,
        perc_strength_icohp=cutoff,
        which_charge="Mulliken",
        additional_condition=0, 
    )


def lobster_pair_count(lobster_dir, structure, cutoff, center_el, partner_el,
                       along_c=False):
    """
    Count `partner_el` neighbours of the representative `center_el` site in the
    LobsterNeighbors bonded structure. If along_c, restrict to bonds within
    C_AXIS_ANGLE_TOL_DEG of the c axis (interlayer bonds).
    """
    idx = representative_index(structure, center_el)
    if idx is None:
        return 0
    ln = lobster_neighbors(lobster_dir, structure, cutoff)
    bonded = ln.get_bonded_structure(structure=structure)
    c_hat = np.array(structure.lattice.matrix[2], float)
    c_hat /= np.linalg.norm(c_hat)
    cos_tol = np.cos(np.deg2rad(C_AXIS_ANGLE_TOL_DEG))
    p0 = structure[idx].coords
    count = 0
    for cs in bonded.get_connected_sites(idx):
        if cs.site.specie.symbol != partner_el:
            continue
        if along_c:
            vec = cs.site.coords - p0
            d = np.linalg.norm(vec)
            if d < 1e-6 or abs(np.dot(vec, c_hat) / d) < cos_tol:
                continue
        count += 1
    return count


def interlayer_pp_distance(structure, angle_tol_deg=C_AXIS_ANGLE_TOL_DEG,
                           rmax=8.0):
    c_hat = np.array(structure.lattice.matrix[2], float)
    c_hat /= np.linalg.norm(c_hat)
    cos_tol = np.cos(np.deg2rad(angle_tol_deg))
    dmin = float("inf")
    for site in structure:
        if site.specie.symbol != "P":
            continue
        for nbr in structure.get_neighbors(site, rmax):
            if nbr.specie.symbol != "P":
                continue
            vec = nbr.coords - site.coords
            d = np.linalg.norm(vec)
            if d < 1e-6:
                continue
            if abs(np.dot(vec, c_hat) / d) >= cos_tol:
                dmin = min(dmin, d)
    return dmin if np.isfinite(dmin) else np.nan


def collect_alal_detection():
    structure = load_structure(CEAL2GA2_LOBSTER_DIR)

    detected = {}   # x-position -> True / False / None(failed)
    for xi, (_, factory) in enumerate(GEOMETRIC_METHODS):
        try:
            c = geometric_pair_count(factory(), structure, "Al", "Al")
            detected[xi] = None if (isinstance(c, float) and np.isnan(c)) \
                else bool(c >= 1)
        except Exception:
            detected[xi] = None
    for j, cut in enumerate(LOBSTER_CUTOFFS):
        xi = N_GEOM + j
        try:
            c = lobster_pair_count(CEAL2GA2_LOBSTER_DIR, structure, cut,
                                   "Al", "Al", along_c=False)
            detected[xi] = bool(c >= 1)
        except Exception as e:
            print(e)
            detected[xi] = None
    return detected


def collect_pp_onset():
    # Load each LOBSTER structure + its interlayer P-P distance.
    entries = []   # (basis, subdir, structure, d_pp)
    ref_d_pp = None
    for sub in laru2p2_lobster_subdirs():
        d = os.path.join(LARU2P2_ROOT, sub)
        contcar = os.path.join(d, "CONTCAR.gz")
        if not os.path.isfile(contcar):
            print(f"WARNING: no CONTCAR.gz in {sub}")
            continue
        s = Structure.from_file(contcar)
        basis = 1 if sub.endswith("_1") else 0
        d_pp = interlayer_pp_distance(s)
        entries.append((basis, d, s, d_pp))
        # suffix-agnostic match of the ambient (equilibrium) basis-0 dir
        name_part = sub.split("__lobster_run_")[0].rsplit("-", 1)[0]
        if name_part == EOS_EQUILIBRIUM_NAME and basis == 0:
            ref_d_pp = d_pp

    if not entries:
        raise RuntimeError(
            f"No LOBSTER run dirs with a readable structure found under "
            f"{LARU2P2_ROOT}. Check the path and that CONTCAR(.gz) files exist.")

    if ref_d_pp is None:
        # fall back to the largest sampled LOBSTER distance as ambient reference
        ref_d_pp = max(e[3] for e in entries)
        print(f"WARNING: equilibrium LOBSTER dir not found; using max sampled "
              f"d(P-P) = {ref_d_pp:.3f} A as ambient reference")

    geom_structs = [(s, d_pp) for (b, _dir, s, d_pp) in entries if b == 0]

    # onset[xpos][basis] = ("value"/"always", d) or None (never bonded)
    onset = defaultdict(dict)
    all_geom_d = [d for _s, d in geom_structs]
    max_d = max(all_geom_d)

    # geometric methods (basis-independent -> store under basis 0 only)
    for xi, (_, factory) in enumerate(GEOMETRIC_METHODS):
        dists_bonded = []
        for s, d_pp in geom_structs:
            try:
                c = geometric_pair_count(factory(), s, "P", "P", along_c=True)
                if isinstance(c, float) and np.isnan(c):
                    continue
                if c > 1:
                    raise ValueError(
                        f"more than one P-P bond detected (found {c}) at "
                        f"d(P-P) = {d_pp:.3f} A")
                if c == 1:
                    dists_bonded.append(d_pp)
            except Exception:
                continue
        onset[xi][0] = _summarize(dists_bonded, max_d)

    # LOBSTER cutoffs, basis 0 only (basis 1 was verified to match)
    lobster_geom = [(_dir, s, d_pp)
                    for (b, _dir, s, d_pp) in entries if b == 0]
    for j, cut in enumerate(LOBSTER_CUTOFFS):
        xi = N_GEOM + j
        dists_bonded = []
        for _dir, s, d_pp in lobster_geom:
            try:
                c = lobster_pair_count(_dir, s, cut, "P", "P", along_c=True)
                if c >= 1:
                    dists_bonded.append(d_pp)
            except Exception:
                continue
        onset[xi][0] = _summarize(dists_bonded, max_d)

    return onset, max_d, sorted(set(all_geom_d)), ref_d_pp


def _summarize(dists_bonded, max_d):
    """Largest distance still bonded; 'always' if that is the max sampled d."""
    if not dists_bonded:
        return None
    top = max(dists_bonded)
    return ("always", top) if np.isclose(top, max_d) else ("value", top)


def draw_subplot1(ax1, alal):
    """CeAl2Ga2 Al-Al detection (green/red column background)."""
    for xi in X_POS:
        val = alal.get(xi)
        color = ("#4caf50" if val is True else
                 "#e53935" if val is False else "#bdbdbd")
        ax1.axvspan(xi - 0.5, xi + 0.5, color=color, alpha=0.75, lw=0)
        txt = "yes" if val is True else "no" if val is False else "n/a"
        ax1.text(xi, 0.5, txt, ha="center", va="center", fontsize=10,
                 fontweight="bold", color="white")
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xlim(-0.5, len(X_POS) - 0.5)

    ax1.set_xticks(X_POS)
    ax1.tick_params(
        axis="x",
        labelbottom=False,
        bottom=False,
        length=0,
    )

    ax1.set_ylabel("Al–Al bond\ndetected?", fontsize=13)
    ax1.set_title("CeAl$_2$Ga$_2$: metal–metal (Al–Al) bond detection",
                  fontsize=12)
    for xi in range(len(X_POS) + 1):
        ax1.axvline(xi - 0.5, color="white", lw=1.5)
    ax1.axvline(N_GEOM - 0.5, color="black", lw=1.2, ls="-")


def draw_subplot2(ax2, onset, sampled, ref_d_pp, show_xticklabels=True):
    """LaRu2P2 onset of interlayer P-P bond detection."""
    MARKER = "D"
    GREEN = "#4caf50"
    RED = "#e53935"
    y_lo = sampled[0] - 0.09
    d_star = min(sampled, key=lambda d: abs(d - HZ_THRESHOLD))

    any_never = any_green = any_red = False
    for xi in X_POS:
        summary = onset.get(xi, {}).get(0)
        if summary is None:
            ax2.axvspan(xi - 0.5, xi + 0.5, color="#d9d9d9", alpha=0.7, lw=0,
                        zorder=0)
            any_never = True
            continue
        _kind, d = summary
        is_ideal = np.isclose(d, d_star)
        col = GREEN if is_ideal else RED
        any_green |= is_ideal
        any_red |= not is_ideal
        ax2.scatter(xi, d, marker=MARKER, color=col, s=55, zorder=3,
                    edgecolor="black", linewidth=0.5)

    ax2.axhline(HZ_THRESHOLD, ls="--", color="k", lw=1.3)
    ax2.text(-0.35, HZ_THRESHOLD + 0.015,
             "Hoffmann–Zheng 2.8 Å", ha="left", va="bottom",
             fontsize=10)
    ax2.axhline(ref_d_pp, ls=":", color="#333333", lw=1.3)
    ax2.text(-0.35, ref_d_pp + 0.015, "ambient pressure",
             ha="left", va="bottom", fontsize=10, color="#333333")

    ax2.set_ylabel("largest $d$(P–P) with\nP–P bond detected (Å)",
                   fontsize=13)
    ax2.set_title("LaRu$_2$P$_2$: onset of interlayer P–P bond detection",
                  fontsize=12)
    ax2.set_xticks(X_POS)
    if show_xticklabels:
        ax2.set_xticklabels(X_LABELS, rotation=45, ha="right", fontsize=9)
    else:
        ax2.set_xticklabels([])
    ax2.tick_params(axis="y", labelsize=11)
    ax2.set_xlim(-0.5, len(X_POS) - 0.5)

    legend_elems = []
    if any_green:
        legend_elems.append(
            Line2D([0], [0], marker=MARKER, color="w", markerfacecolor=GREEN,
                   markeredgecolor="black", markersize=8,
                   label=f"onset at {d_star:.2f} Å\n(closest to 2.8 Å)"))
    if any_red:
        legend_elems.append(
            Line2D([0], [0], marker=MARKER, color="w", markerfacecolor=RED,
                   markeredgecolor="black", markersize=8,
                   label="onset elsewhere"))
    if any_never:
        legend_elems.append(
            Patch(facecolor="#d9d9d9", alpha=0.7, edgecolor="none",
                  label="no P–P bond detected"))
    y_lo_ax = y_lo - 0.03
    y_hi_ax = 3.099
    hz_frac = (HZ_THRESHOLD - y_lo_ax) / (y_hi_ax - y_lo_ax)
    ax2.legend(handles=legend_elems, loc="upper left",
               bbox_to_anchor=(0.0, hz_frac - 0.02), frameon=True,
               framealpha=1, edgecolor="black", fontsize=9, borderpad=0.3,
               labelspacing=0.4)
    ax2.set_ylim(y_lo_ax, y_hi_ax)
    ax2.axvline(N_GEOM - 0.5, color="black", lw=1.2, ls="-")


def main():
    alal = collect_alal_detection()
    onset, max_d, sampled, ref_d_pp = collect_pp_onset()

    fig = plt.figure(figsize=(7.0, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    draw_subplot1(ax1, alal)
    draw_subplot2(ax2, onset, sampled, ref_d_pp, show_xticklabels=True)

    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PDF, dpi=DPI)
    plt.close(fig)

    print(f"Saved {OUTPUT_PDF}")
    print(
        f"Sampled interlayer d(P-P) range: "
        f"{min(sampled):.3f}-{max(sampled):.3f} A"
    )


if __name__ == "__main__":
    main()
