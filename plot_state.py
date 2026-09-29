import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

COMPONENTS = {
    "total": np.array([0, 1, 2, 3]),
    "top": np.array([0, 1]),
    "bottom": np.array([2, 3]),
    "a": np.array([0, 2]),
    "b": np.array([1, 3]),
    "top-a": np.array([0]),
    "top-b": np.array([1]),
    "bottom-a": np.array([2]),
    "bottom-b": np.array([3]),
}


def _close_poly(poly):
    poly = np.asarray(poly, dtype=float)
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def _normalize_positive(data):
    data = np.asarray(data, dtype=float)
    vmax = np.nanmax(data)
    if vmax > 0:
        data = data / vmax
    return data


def _square_bounds(vertices, padding=0.0):
    if np.allclose(vertices[0], vertices[-1]):
        vertices_for_center = vertices[:-1]
    else:
        vertices_for_center = vertices
    center = np.mean(vertices_for_center, axis=0)
    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)
    half_width = 0.5 * max(xmax - xmin, ymax - ymin) * (1.0 + padding)
    return (
        center[0] - half_width,
        center[0] + half_width,
        center[1] - half_width,
        center[1] + half_width,
    )


def _add_matched_colorbar(fig, ax, mappable, label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    return fig.colorbar(mappable, cax=cax, label=label)


def _nufft_site_density(positions, prob, lat, bounds, grid_size, nufft_sigma=96):
    from solver.utils import nufft_interpolate

    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, grid_size)
    y = np.linspace(ymin, ymax, grid_size)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mesh_point = np.column_stack([xx.ravel(), yy.ravel()])

    density = nufft_interpolate(
        mesh_point,
        prob,
        positions,
        lat,
        nufft_sigma,
    ).reshape(grid_size, grid_size)
    return xx, yy, _normalize_positive(density)


def plot_CMstate(
    wavefunc,
    k,
    Gmesh,
    WScell,
    output,
    lat=None,
    repeat=1,
    padding=0.15,
    component="total",
    grid_size=240,
    title=None,
):
    """
    Plot a continuum-model wavefunction in real space.

    Parameters
    ----------
    wavefunc : ndarray
        Eigenvector in the BM basis. Accepted shapes are (NG*4,) or (NG, 4),
        ordered as [top A, top B, bottom A, bottom B] for each G point.
    k : ndarray
        Two-component crystal momentum in the same reciprocal coordinates as Gmesh.
    Gmesh : ndarray
        Shape (NG, 2), reciprocal lattice vectors used by the wavefunction.
    WScell : ndarray
        Real-space Wigner-Seitz cell vertices, shape (N, 2). It will be drawn
        as the central hexagon boundary.
    output : str or Path
        Output image path.
    lat : ndarray or None
        Kept for compatibility with existing calls. The density is evaluated
        in real space, so neighboring periods appear without explicitly tiling.
    repeat : int
        Kept for compatibility. Values greater than zero enable padding.
    padding : float
        Fractional padding added around the central WScell square bounds when
        repeat is greater than zero.
    component : str
        One of total/top/bottom/a/b/top-a/top-b/bottom-a/bottom-b.
    grid_size : int
        Number of grid points along each direction of the WScell bounding box.
    title : str or None
        Optional figure title.
    """
    if component not in COMPONENTS:
        raise ValueError(
            f"Unknown component {component!r}; choose from {COMPONENTS.keys()}"
        )

    Gmesh = np.asarray(Gmesh, dtype=float)
    WScell = _close_poly(WScell)
    k = np.asarray(k, dtype=float)
    repeat = int(repeat)

    coeff = np.asarray(wavefunc, dtype=complex)
    if coeff.ndim == 1:
        coeff = coeff.reshape(Gmesh.shape[0], 4)
    if coeff.shape != (Gmesh.shape[0], 4):
        raise ValueError(
            f"wavefunc must have shape ({Gmesh.shape[0]}*4,) or "
            f"({Gmesh.shape[0]}, 4), got {coeff.shape}"
        )

    view_padding = padding if repeat > 0 else 0.0
    xmin, xmax, ymin, ymax = _square_bounds(WScell, padding=view_padding)
    x = np.linspace(xmin, xmax, grid_size)
    y = np.linspace(ymin, ymax, grid_size)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])

    phase = np.exp(1j * np.dot(points, (k + Gmesh).T))
    psi = phase @ coeff[:, COMPONENTS[component]]
    density = np.sum(np.abs(psi) ** 2, axis=1).reshape(grid_size, grid_size)
    density = _normalize_positive(density)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    pcm = ax.pcolormesh(
        xx, yy, density, shading="auto", cmap="magma", vmin=0.0, vmax=1.0
    )
    ax.plot(WScell[:, 0], WScell[:, 1], color="cyan", lw=2.0, label="WScell")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    _add_matched_colorbar(fig, ax, pcm, r"$|\psi(r)|^2$ / max")
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output


def plot_TBstate(
    wavefunc,
    positions,
    output,
    WScell=None,
    lat=None,
    repeat_positions=None,
    padding=0.15,
    grid_size=240,
    nufft_sigma=10,
    title=None,
):
    """
    Plot a tight-binding wavefunction on atom/orbital positions.

    Parameters
    ----------
    wavefunc : ndarray
        One TB eigenvector with shape (Norb,).
    positions : ndarray
        Cartesian positions of orbitals, shape (Norb, 2) or (Norb, 3).
    output : str or Path
        Output image path.
    WScell : ndarray or None
        Optional real-space Wigner-Seitz cell vertices to draw.
    lat : ndarray or None
        Real-space lattice vectors, shape (2, 2) or (3, 3). Used to repeat
        site probabilities around the central cell before interpolation.
    repeat_positions : ndarray or None
        Optional extra Cartesian positions for repeated-cell visualization. If
        this is an integer and lat is provided, it is used as the repeat range.
        If this is an array, wavefunc probabilities are tiled to this position count.
    padding : float
        Fractional padding added around the central WScell square bounds.
    grid_size : int
        Number of grid points along each side of the square plot.
    nufft_sigma : int
        Number of Fourier modes used by NUFFT interpolation along each direction.
    title : str or None
        Optional figure title.
    """
    coeff = np.asarray(wavefunc, dtype=complex)
    prob = _normalize_positive(np.abs(coeff) ** 2)

    positions = np.asarray(positions, dtype=float)[:, :2]
    if repeat_positions is not None:
        repeat_positions = np.asarray(repeat_positions, dtype=float)
        if repeat_positions.ndim > 1:
            positions = repeat_positions[:, :2]
            if positions.shape[0] % prob.shape[0] != 0:
                raise ValueError(
                    "repeat_positions must contain an integer number of cells"
                )
            prob = np.tile(prob, positions.shape[0] // prob.shape[0])
        elif lat is not None and int(repeat_positions) > 0:
            lat = np.asarray(lat, dtype=float)[:2, :2]
            repeat = int(repeat_positions)
            shifts = [
                i * lat[0] + j * lat[1]
                for i in range(-repeat, repeat + 1)
                for j in range(-repeat, repeat + 1)
            ]
            positions = (
                positions[np.newaxis, :, :] + np.asarray(shifts)[:, np.newaxis, :]
            ).reshape(-1, 2)
            prob = np.tile(prob, len(shifts))

    if positions.shape[0] != prob.shape[0]:
        raise ValueError(
            f"positions and wavefunc sizes differ: {positions.shape[0]} vs {prob.shape[0]}"
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    if WScell is not None:
        WScell = _close_poly(WScell)
        xmin, xmax, ymin, ymax = _square_bounds(WScell, padding=padding)
    else:
        xmin, xmax, ymin, ymax = _square_bounds(positions, padding=padding)

    if lat is None:
        raise ValueError("lat is required for NUFFT interpolation in plot_TBstate")
    lat = np.asarray(lat, dtype=float)[:2, :2]
    xx, yy, density = _nufft_site_density(
        positions,
        prob,
        lat,
        (xmin, xmax, ymin, ymax),
        grid_size,
        nufft_sigma=nufft_sigma,
    )

    pcm = ax.pcolormesh(
        xx, yy, density, shading="auto", cmap="magma", vmin=0.0, vmax=1.0
    )
    if WScell is not None:
        ax.plot(WScell[:, 0], WScell[:, 1], color="cyan", lw=2.0, label="WScell")
        ax.legend(loc="upper right")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    if title:
        ax.set_title(title)
    _add_matched_colorbar(fig, ax, pcm, r"$|\psi_i|^2$ / max")
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return output


def main_cm():
    from cm import BMHamSetter
    from fit import fit

    trad = np.deg2rad(1.05)
    uni = 0
    k = "K"
    v = 1

    result = fit(trad, uni, np.deg2rad(0))[0]

    trad, uni, theta_s, bi, _, _, _ = result

    ham = BMHamSetter(
        trad=trad,
        eh=uni * 0.001,
        phi=theta_s,
        ebi=bi,
        cutoff=5,
        hv=0.6,
        gammaAA=0.11,
        gammaAB=0.11,
        test=False,
    )

    eigval, eigvec = np.linalg.eigh(ham.ham(ham.kpoints[k], v))
    print(eigvec.shape[1])
    for i in range(-2, 2):
        output = Path(f"plot/{int(uni)}/cm/{k}{i+2}.png")
        plot_CMstate(
            wavefunc=eigvec[:, i + eigvec.shape[1] // 2],
            k=ham.kpoints[k],
            Gmesh=ham.Gmesh,
            WScell=ham.WScell,
            output=output,
            lat=ham.lat,
            repeat=1,
            padding=0.6,
            component="total",
            grid_size=240,
        )
        print(f"saved {output}")


def main_tb():
    from tb import TBHamSetter

    uni = 0
    k = "K"
    ham = TBHamSetter(
        test=False,
        uni_given=uni * 0.001,
        write_poscar=False,
    )
    input_file = Path(f"data/{uni}/{k}.npz")

    result = np.load(input_file)
    eigval = result["eigval"]
    eigvec = result["eigvec"]
    # eigval, eigvec = ham.calc_state(k)
    # np.savez(input_file, eigval=np.array(eigval), eigvec=np.array(eigvec))

    print(f"band energies: {eigval}")
    # exit()
    pos = np.dot(ham.cell.pos, ham.cell.lat)
    for i in range(4):
        output = Path(f"plot/{uni}/tb/{k}{i}.png")
        plot_TBstate(
            wavefunc=eigvec[i + 2, :],
            positions=pos,
            output=output,
            WScell=ham.WScell,
            lat=ham.cell.lat,
            repeat_positions=1,
            padding=0.6,
            grid_size=240,
            nufft_sigma=20,
        )
        print(f"saved {output}")


if __name__ == "__main__":
    main_cm()
