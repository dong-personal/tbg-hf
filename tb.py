# ABA-BA
import numpy as np

import time
import matplotlib.pyplot as plt

from solver.kdtree import cPKDTree
from solver.cell import Cell

from gen_struct import a0, zspace, gen_struct_strain, gen_struct

from dataclasses import dataclass
from solver.mpi import ENV
from datetime import datetime


# 0.3637, 0.3349
def calc_hop(rij: np.ndarray, zij: np.ndarray):
    """
    Calculate hopping parameter according to Slater-Koster relation.
    See ref. [2] for the formulae.

    :param rij: (3,) array, displacement vector between two orbitals in NM
    :return: hopping parameter in eV
    """
    a0 = 0.1418
    a1 = 0.3349
    r_c = 0.6140
    l_c = 0.0265
    gamma0 = 2.8
    gamma1 = 0.44
    decay = 22.18
    q_pi = decay * a0  # 3.145124
    q_sigma = decay * a1  # 7.428082
    n = zij / rij  # 0.921
    v_pp_pi = -gamma0 * np.exp(q_pi * (1 - rij / a0))  # -0.0204
    v_pp_sigma = gamma1 * np.exp(q_sigma * (1 - rij / a1))  # 0.23245
    fc = 1 / (1 + np.exp((rij - r_c) / l_c))  # 1
    hop = (n**2 * v_pp_sigma + (1 - n**2) * v_pp_pi) * fc
    return hop


def make_cell(theta_t, epsilon, theta_s, biaxial_strain=0.0):

    (top, bot), moire_lat_3d = gen_struct_strain(
        theta_t, epsilon, theta_s, biaxial_strain
    )
    pos = np.vstack([top, bot])
    pos = pos.reshape(-1, 3)
    frac = np.dot(pos, np.linalg.inv(moire_lat_3d))

    tree = cPKDTree(pos, moire_lat_3d, isperiodic=np.array([1, 1, 0]), dir="half")
    pairs = tree.query_pairs(0.5)

    hopping = {}
    for rn, (pair, dr) in pairs.items():
        zij = pos[pair[:, 1], 2] - pos[pair[:, 0], 2]

        hop = calc_hop(dr, zij)

        hopping[rn] = (pair, hop)
    cell = Cell(moire_lat_3d, frac, hopping)
    return cell


def make_cell1(i):

    (top, bot), moire_lat_3d = gen_struct(i)
    pos = np.vstack([top, bot])
    pos = pos.reshape(-1, 3)
    frac = np.dot(pos, np.linalg.inv(moire_lat_3d))

    tree = cPKDTree(pos, moire_lat_3d, isperiodic=np.array([1, 1, 0]), dir="half")
    pairs = tree.query_pairs(0.5)

    hopping = {}
    for rn, (pair, dr) in pairs.items():
        zij = pos[pair[:, 1], 2] - pos[pair[:, 0], 2]

        hop = calc_hop(dr, zij)

        hopping[rn] = (pair, hop)
    cell = Cell(moire_lat_3d, frac, hopping)
    return cell


def write_poscar(filename, cell, note=[""]):
    lat = cell.lat * 10.0
    orb_num = cell.num_orb
    orb_pos = cell.pos

    with open(filename, "w") as f:
        f.write(f"moire_cell{note[0]}\n")
        f.write("1.0\n")
        for lat_vec in lat:
            f.write(f"{lat_vec[0]:20.10f}{lat_vec[1]:20.10f}{lat_vec[2]:20.10f}\n")
        f.write("C\n")
        f.write(f"{orb_num}\n")
        f.write("D\n")
        for pos in orb_pos:
            f.write(f"{pos[0]:20.10f}{pos[1]:20.10f}{pos[2]:20.10f}\n")


@dataclass
class TBHamSetter:
    trad_given: float = np.deg2rad(1.05)
    uni_given: float = 0.0
    bi_given: float = 0.0
    dirc_given: float = np.deg2rad(0.0)
    eng_min: float = 0.7
    eng_max: float = 0.9
    predict_num: int = 40
    test: bool = False
    write_poscar: bool = True

    def __post_init__(self):

        from solver.solver import Solver
        from solver.utils import gen_kpath, get_rlat, get_fbz
        from fit import fit

        self.cell = make_cell(
            self.trad_given, self.uni_given, self.dirc_given, self.bi_given
        )

        # para used to make cell upon [trgad, uni, theta_s, bi, ijs.T, mnp.T]
        self.fit = fit(self.trad_given, self.uni_given, self.dirc_given, self.bi_given)[
            0
        ]
        self.note = [
            f"given: trad={np.rad2deg(self.trad_given):.4f}, uni={self.uni_given:.4f}, bi={self.bi_given:.4f}, dirc={np.rad2deg(self.dirc_given):.4f}",
            f"fit: trad={np.rad2deg(self.fit[0]):.8f}, uni={self.fit[1]:.8f}, bi={self.fit[3]:.8f}, dirc={np.rad2deg(self.fit[2]):.8f}",
        ]

        self.datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")

        # self.cell = make_cell1(31)

        print(
            f"Cell has {self.cell.num_orb} orbitals and {self.cell.num_hop} hoppings."
        )

        if self.write_poscar:
            write_poscar(f"data/{self.datetime_str}.vasp", self.cell, self.note)

        self.G = get_rlat(self.cell.lat)

        # rot G[0] to x-axis
        cos_theta = self.G[0, 0] / np.linalg.norm(self.G[0])
        sin_theta = self.G[0, 1] / np.linalg.norm(self.G[0])
        rot = np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])

        self.G[0:2, 0:2] = np.dot(self.G[0:2, 0:2], rot.T)

        self.cell.lat[0:2, 0:2] = np.dot(self.cell.lat[0:2, 0:2], rot.T)

        self.solver = Solver(self.cell)

        self.feast_solver = self.solver.feast_init(
            self.eng_min, self.eng_max, self.predict_num
        )

        self.fbz = get_fbz(self.G[0:2, 0:2])

        self.WScell = get_fbz(self.cell.lat[0:2, 0:2])

        self.knum = 400
        kpoints = np.array(
            [
                [0, 0, 0],
                self.G[0] / 2.0,
                [*self.fbz[0], 0],
                # (2 * self.G[0] + self.G[1]) / 3.0,
                [0, 0, 0],
                # (2 * self.G[1] + self.G[0]) / 3.0,
                [*self.fbz[1], 0],
            ]
        )
        kidx, kpath = gen_kpath(kpoints, self.knum)
        klength = np.linalg.norm(kpoints[1:] - kpoints[:-1], axis=1)
        kpath1d = np.concatenate([[0], np.cumsum(klength)])

        kpath = np.dot(kpath, np.linalg.inv(self.G))

        self.kidx = [kidx]
        self.kpath = [kpath]
        self.kpath1d = [kpath1d]
        self.kpoints = {
            "G": kpoints[0],
            "M": kpoints[1],
            "K": kpoints[2],
            "K'": kpoints[4],
        }
        if self.test:
            kpath = np.dot(kpath, self.G)
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.plot(self.fbz[:, 0], self.fbz[:, 1], "k-")
            plt.plot([0, self.G[0, 0]], [0, self.G[0, 1]], "b--", label="G1")
            plt.plot([0, self.G[1, 0]], [0, self.G[1, 1]], "y--", label="G2")
            plt.plot(kpath[:, 0], kpath[:, 1], "r-")
            plt.legend()
            plt.axis("equal")
            plt.show()
            exit()

    def calc_band(self, kpathidx, op=None):
        if op is not None:

            def calc_valley(eigval, eigvec, k):
                csr_matrix = op(k)
                return eigval, np.real(
                    np.dot(eigvec.conj(), csr_matrix.dot(eigvec.T)).diagonal()
                )

            process = self.solver.op_process(calc_valley)
        else:
            process = self.solver.band_process
        result = self.solver.solve(self.kpath[kpathidx], self.feast_solver, process)

        return result

    def calc_state(self, kpoint):

        ham_setter = self.solver.ham_setter()
        ham = ham_setter(self.kpoints[kpoint].dot(np.linalg.inv(self.G)))

        eigval, eigvec = self.feast_solver(ham)

        return eigval, eigvec


def write_lmps(filename, cell):

    lat = cell.lat * 10.0
    a1 = np.array([1, 0, 0])
    theta = np.arccos(np.dot(lat[0], a1) / np.linalg.norm(lat[0]))
    rot = np.array(
        [
            [np.cos(-theta), -np.sin(-theta), 0],
            [np.sin(-theta), np.cos(-theta), 0],
            [0, 0, 1],
        ]
    )
    lat = np.dot(lat, rot.T)
    print(lat)
    orb_num = cell.num_orb
    orb_pos = np.dot(cell.pos, lat)
    z = orb_pos[:, 2]
    unique_z = np.unique(np.round(z, decimals=6))
    if len(unique_z) != 5:
        raise ValueError("Layer number is not 5!")
    layer_idx = np.zeros(orb_num, dtype=int)
    for i, uz in enumerate(unique_z):
        layer_idx[np.abs(z - uz) < 1e-5] = i + 1

    with open(filename, "w") as f:
        f.write("# structure no relax \n")
        f.write("\n")
        f.write("%d atoms \n" % (len(orb_pos)))
        f.write("5 atom types \n")
        f.write("0.00000000 %.8f xlo xhi \n" % lat[0, 0])
        f.write("0.00000000 %.8f ylo yhi \n" % lat[1, 1])
        f.write("0.00000000 100.00000000 zlo zhi \n")
        f.write(
            "%.8f 0.00000000 0.00000000  xy xz yz\n"
            % (np.dot(lat[0], lat[1]) / np.linalg.norm(lat[0]))
        )
        f.write("\n")
        f.write("Masses \n")
        f.write("\n")
        f.write("  1 12.01100000 # C \n")
        f.write("  2 12.01100000 # C \n")
        f.write("  3 12.01100000 # C \n")
        f.write("  4 12.01100000 # C \n")
        f.write("  5 12.01100000 # C \n")
        f.write("\n")
        f.write("Atoms # atomic \n")
        f.write("\n")
        for i, pos in enumerate(orb_pos):
            idx = layer_idx[i]
            f.write(
                "%6d %6d %15.10f %15.10f %15.10f \n"
                % (i + 1, idx, pos[0], pos[1], pos[2])
            )


def read_lmps(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.strip() == "ITEM: TIMESTEP":
            last_frame_start = i

    lines = lines[last_frame_start:]
    orb_num = int(lines[3].split()[0])
    xlo, xhi, xy = np.array(lines[5].split(), dtype=np.float64)
    ylo, yhi, xz = np.array(lines[6].split(), dtype=np.float64)
    zlo, zhi, yz = np.array(lines[7].split(), dtype=np.float64)

    xlo = xlo - np.min([0, xy, xz, xy + xz])
    xhi = xhi - np.max([0, xy, xz, xy + xz])
    ylo = ylo - np.min([0, yz])
    yhi = yhi - np.max([0, yz])

    lat = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])

    pos_string = lines[9:]
    pos = [[float(x) for x in line.split()[2:5]] for line in pos_string]
    pos = np.array(pos, dtype=np.float64)

    if pos.shape[0] != orb_num:
        raise ValueError("Orbital number does not match!")

    lat = lat / 10.0
    pos = pos / 10.0
    frac = np.dot(pos, np.linalg.inv(lat)) % 1

    pos = np.dot(frac, lat)
    return pos, lat


def write_data(file_name, eigvals, note=None):
    with open(file_name, "w") as f:
        if note is not None:
            for line in note:
                f.write(f"% {line}\n")
        for i, eigs in enumerate(eigvals):
            f.write(f"# kpoints-{i}\n")
            for j, eig in enumerate(eigs):
                f.write(f"{eig:15.8f}\t")
                if (j + 1) % 5 == 0:
                    f.write("\n")
            f.write("\n")


def read_data(filename):
    eigvals = []
    with open(filename, "r") as f:
        lines = f.readlines()

    current_eigs = None
    for line in lines:
        if line.startswith("#"):
            if current_eigs is not None:
                eigvals.append(np.array(current_eigs))
            current_eigs = []
        else:
            nums = line.split()
            for num in nums:
                current_eigs.append(float(num))  # type: ignore
    return eigvals


if __name__ == "__main__":

    import os
    from solver.mpi import ENV
    import matplotlib.pyplot as plt
    import shutil
    from valley import valley_operator

    ENV.redirect_output()
    uni = -0.003
    ham = TBHamSetter(test=False, uni_given=uni)
    eng = ham.calc_band(0)

    datetime_str = ham.datetime_str
    if ENV.rank == 0:
        write_data(f"data/band{datetime_str}.txt", eng, note=ham.note)

        plt.figure(figsize=(8, 8))
        for i, eigs in enumerate(eng):
            plt.scatter(
                np.full(eigs.shape, i), eigs - 0.81, s=1.5, color="red", zorder=2
            )

        kidx = ham.kidx[0]
        plt.xticks(kidx, ["G", "M", "K", "G", "K'"])
        for i in range(1, len(kidx) - 1):
            plt.axvline(x=kidx[i], color="grey", linestyle="--", linewidth=2, zorder=1)
        plt.xlim((0, kidx[-1]))
        plt.ylim(-0.08, 0.08)
        plt.tight_layout()
        plt.savefig(f"plot/band{datetime_str}.png", bbox_inches="tight", dpi=600)

    # ENV.redirect_output()

    # i = 31

    # cell_type = None
    # cell = make_cell(i)
    # prefix = f"data/"

    # print(cell.num_orb, cell.num_hop)

    # prefix = f"data/"
    # if cell_type == "relaxed":
    #     prefix = f"data/"

    # if ENV.rank == 0 and not os.path.exists(prefix):
    #     os.makedirs(prefix)
    # ENV.barrier()
    # write_poscar(prefix + "poscar.vasp", cell)

    # op = valley_operator(cell, i)
    # # print(type(op))
    # # exit()
    # result, kidx = calc_band_op(cell, op=op)
    # eigvals = [res[0] for res in result]  # type: ignore
    # valley_vals = [res[1] for res in result]  # type: ignore
    # np.savetxt(prefix + "kidx.txt", kidx)
    # write_data(prefix + "valley.txt", valley_vals)
    # write_data(prefix + "eigvals.txt", eigvals)

    # # eigvals = read_eigvals(prefix + "eigvals.txt")
    # # kidx = np.loadtxt(prefix + "kidx.txt")

    # if ENV.rank == 0:
    #     plt.figure(figsize=(6, 8))
    #     for i, eigs in enumerate(eigvals):
    #         plt.scatter(np.full(eigs.shape, i), eigs, s=1.5, color="red", zorder=2)
    #     plt.xticks(kidx, ["G", "M", "K", "G", "K'"])
    #     for i in range(1, len(kidx) - 1):
    #         plt.axvline(x=kidx[i], color="grey", linestyle="--", linewidth=2, zorder=1)
    #     plt.xlim((0, kidx[-1]))
    #     plt.ylim(0.7, 0.9)
    #     plt.tight_layout()
    #     plt.savefig(prefix + "band.png", bbox_inches="tight", dpi=600)
