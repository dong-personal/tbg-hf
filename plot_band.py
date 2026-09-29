import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_LABELS = [r"$\Gamma$", "M", "K", r"$\Gamma$", "K'"]


def read_band_file(filename):
    notes = []
    eigvals = []
    current_eigs = None

    with open(filename, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("%"):
                notes.append(stripped.lstrip("%").strip())
                continue
            if stripped.startswith("#"):
                if current_eigs is not None:
                    eigvals.append(np.array(current_eigs, dtype=float))
                current_eigs = []
                continue
            if current_eigs is None:
                current_eigs = []
            current_eigs.extend(float(num) for num in stripped.split())

    if current_eigs is not None:
        eigvals.append(np.array(current_eigs, dtype=float))

    if not eigvals:
        raise ValueError(f"No eigvals found in {filename}")

    return notes, eigvals


def plot_TBband(
    eigvals, output, kidx, klabels, notes=None, energy_shift=0.81, ylim=(-0.08, 0.08)
):
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, eigs in enumerate(eigvals):
        ax.scatter(
            np.full(eigs.shape, i),
            eigs - energy_shift,
            s=1.5,
            color="red",
            zorder=2,
        )

    ax.set_xticks(kidx, klabels)
    for idx in kidx[1:-1]:
        ax.axvline(x=idx, color="grey", linestyle="--", linewidth=2, zorder=1)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.6, alpha=0.35, zorder=1)
    ax.set_xlim((0, kidx[-1]))
    ax.set_ylim(*ylim)
    ax.set_ylabel(f"Energy - {energy_shift:g} (eV)")
    if notes:
        ax.set_title(notes[0], fontsize=10)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=600)
    plt.close(fig)


def plot_CMband():
    pass


def _k_coordinates(kidx, kdist, npoints):
    kidx = np.asarray(kidx, dtype=float)
    kdist = np.asarray(kdist, dtype=float)
    return np.interp(np.arange(npoints), kidx, kdist)


def _sample_tb_band(eigvals, kidx, kdist, k_step):
    eigvals = [np.asarray(eigs, dtype=float) for eigs in eigvals]
    if not eigvals:
        raise ValueError("TB band data is empty")
    if k_step <= 0:
        raise ValueError("k_step must be positive")

    sample_idx = np.arange(0, len(eigvals), k_step, dtype=int)
    if sample_idx[-1] != len(eigvals) - 1:
        sample_idx = np.append(sample_idx, len(eigvals) - 1)

    tb_kidx = np.rint(
        np.asarray(kidx, dtype=float) / float(kidx[-1]) * (len(eigvals) - 1)
    )
    tb_x = np.interp(sample_idx, tb_kidx, kdist)
    return tb_x, [eigvals[index] for index in sample_idx]


def _find_band_record(twist, uni, db_path, tolerance=1e-12):
    from name import find_rows_by_params

    rows = find_rows_by_params(
        {"trad": twist, "uni": uni},
        db_path=db_path,
        prefix="given",
        tolerance=tolerance,
    )
    if not rows:
        raise ValueError(
            f"No band file found in {db_path} for given_trad={twist}, given_uni={uni}"
        )
    return sorted(rows, key=lambda row: row["timestamp"])[-1]


def main_tb():
    from tb import TBHamSetter

    ham = TBHamSetter(test=False, uni_given=0.000, write_poscar=False)

    pass


def main_cm():
    from cm import BMHamSetter
    from fit import fit

    trad = np.deg2rad(1.05)
    uni = 0.0 * 0.01

    result = fit(trad, uni, np.deg2rad(0))[0]

    trad, uni, theta_s, bi, _, _, _ = result
    print(
        f"theta_t: {np.rad2deg(trad):.4f}, epsilon: {uni:.6f}, theta_s: {np.rad2deg(theta_s):.4f}, biaxial_strain: {bi:.6f}"
    )

    ham = BMHamSetter(
        trad=trad,
        eh=uni,
        phi=theta_s,
        ebi=bi,
        cutoff=5,
        gammaAA=0.11,
        gammaAB=0.11,
        test=False,
    )
    eigs, kidx, _ = ham.calc_band(v=1)
    eigs = np.array(eigs).T
    plt.figure(figsize=(8, 8))
    for i in range(eigs.shape[0] // 2 - 4, eigs.shape[0] // 2 + 4):
        plt.plot(eigs[i], color="black", lw=1)
    plt.xlim(0, 200)
    plt.ylim(-0.02, 0.02)
    plt.xticks(kidx, [r"$\Gamma$", "M", "K", r"$\Gamma$", "K'"])
    plt.ylabel("Energy (eV)")
    plt.title("Twisted Bilayer Graphene Band Structure")
    plt.grid()
    plt.tight_layout()
    plt.show()

    pass


def main_compare(
    twist=1.05,
    uni=0.0,
    k_step=5,
    db_path=Path("data/band_index.csv"),
    output=Path("plot/compare_band.png"),
    shift=0.815,
    show=True,
):
    from cm import BMHamSetter

    db_path = Path(db_path)
    output = Path(output)
    if not db_path.exists():
        from name import build_db

        build_db(db_path=db_path)

    record = _find_band_record(twist, uni, db_path)
    notes, tb_eigs = read_band_file(record["file"])

    cm_ham = BMHamSetter(
        test=False,
        trad=np.deg2rad(float(record["fit_trad"])),
        eh=float(record["fit_uni"]),
        phi=np.deg2rad(float(record["fit_dirc"])),
        ebi=float(record["fit_bi"]),
        hv=0.6,
        cutoff=5,
        gammaAA=0.11,
        gammaAB=0.11,
    )
    eigs, kidx, kdist = cm_ham.calc_band(v=1)
    eigs = np.array(eigs).T
    cm_x = _k_coordinates(kidx, kdist, eigs.shape[1])
    tb_x, tb_sampled = _sample_tb_band(tb_eigs, kidx, kdist, k_step)

    plt.figure(figsize=(8, 8))
    for i in range(eigs.shape[0]):
        plt.plot(cm_x, eigs[i], color="black", lw=1, zorder=1)

    for kx, tb_bands in zip(tb_x, tb_sampled):
        plt.scatter(
            np.full(tb_bands.shape, kx),
            tb_bands - shift,
            s=8,
            color="red",
            alpha=0.75,
            zorder=2,
        )

    for xpos in kdist[1:-1]:
        plt.axvline(x=xpos, color="grey", linestyle="--", linewidth=1, zorder=0)
    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.6, alpha=0.35, zorder=0)

    plt.xlim(kdist[0], kdist[-1])
    plt.ylim(-0.02, 0.02)
    plt.xticks(kdist, [r"$\Gamma$", "M", "K", r"$\Gamma$", "K'"])
    plt.ylabel("Energy (eV)")
    plt.title(
        f"CM vs TB band: twist={twist:g}, uni={uni:g}, stamp={record['timestamp']}"
    )
    plt.grid()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", dpi=600)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Read TB band from {record['file']}")
    print(f"Saved comparison plot to {output}")


if __name__ == "__main__":
    # main_compare(
    #     twist=1.05,
    #     uni=0.0,
    #     k_step=10,
    #     db_path=Path("data/band_index.csv"),
    #     output=Path("plot/compare_band.png"),
    #     shift=0.8125,
    #     show=True,
    # )
    main_cm()
