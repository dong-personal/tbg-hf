# single particle ham

import numpy as np
from dataclasses import dataclass
from scipy.sparse import block_diag
import matplotlib.pyplot as plt


def gen_kpath(kpoints, num_kpoints):
    """
    return sum(num_kpoints)+1 points along the kpath
    """
    length = np.linalg.norm(kpoints[1:] - kpoints[:-1], axis=1)
    knum = np.round(length / np.sum(length) * num_kpoints).astype(int)
    kpath = []
    for i in range(len(knum)):
        slope = (kpoints[i + 1] - kpoints[i]) / knum[i]

        tmp = kpoints[i] + slope * np.arange(knum[i])[:, np.newaxis]

        kpath.append(tmp)

    kpath.append(kpoints[-1].reshape(1, -1))  # Ensure the last point is included

    return np.cumsum(np.concatenate([[0], knum]), dtype=int), np.vstack(kpath)


def ws_cell(b1, b2, N=4):
    from scipy.spatial import Voronoi

    # 生成倒格点
    points = []
    ij_list = []
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            p = i * b1 + j * b2
            points.append(p)
            ij_list.append((i, j))
    points = np.array(points)

    # 找到原点对应的点索引
    origin_idx = ij_list.index((0, 0))

    # Voronoi 分解
    vor = Voronoi(points)

    # 原点对应的 region
    region_idx = vor.point_region[origin_idx]
    region = vor.regions[region_idx]

    # 如果 region 中有 -1，说明区域无界，通常说明点取太少了
    if -1 in region or len(region) == 0:
        raise ValueError("Voronoi region is unbounded. Increase N.")

    vertices = vor.vertices[region]

    # 按角度排序，便于画多边形
    center = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    order = np.argsort(angles)
    vertices = vertices[order]
    poly = np.vstack([vertices, vertices[0]])  # 闭合多边形

    return poly


def fold(kpoints, b1, b2, N=4):
    frac = np.dot(kpoints, np.linalg.inv(np.array([b1, b2])))

    frac_int = np.round(frac).astype(int)

    search = np.arange(-N, N + 1)[:, np.newaxis] + frac_int[np.newaxis, :]

    Gmesh = np.dot(search, np.array([b1, b2]))

    dist = np.linalg.norm(Gmesh - kpoints, axis=1)
    idx = np.argmin(dist)
    return kpoints - Gmesh[idx]


@dataclass
class BMHamSetter:
    trad: float = np.deg2rad(1.05)
    dcc: float = 0.142
    hv: float = 2.15 * np.sqrt(3) * dcc * 1
    gammaAA: float = 0.1
    gammaAB: float = 0.1
    a: float = 0.246
    cutoff: int = 4
    mass_term: float = 0.0

    # strain
    Vg: float = 4  # Scalar potential
    bG: float = 3.14  # Grüneisen parameter

    eh: float = 0
    phi: float = 0
    strain_type: str = "uni"

    # exxT: float = 0
    # eyyT: float = 0
    # exyT: float = 0

    # exxB: float = 0
    # eyyB: float = 0
    # exyB: float = 0

    test: bool = True

    def __post_init__(self):
        if self.strain_type == "uni":
            self.set_uni(self.eh, self.phi)
        elif self.strain_type == "shear":
            self.set_shear(self.eh, self.phi)
        else:
            raise ValueError("Invalid strain type. Choose 'uni' or 'shear'.")

        self.E_t = np.array([[self.exxT, self.exyT], [self.exyT, self.eyyT]])
        self.E_b = np.array([[self.exxB, self.exyB], [self.exyB, self.eyyB]])

        self.Rt = np.array(
            [
                [np.cos(self.trad / 2.0), -np.sin(self.trad / 2.0)],
                [np.sin(self.trad / 2.0), np.cos(self.trad / 2.0)],
            ]
        )

        self.Rb = self.Rt.T

        self._build_lattice()
        self._build_Gmesh()
        self.hamp = self._build_tunneling_matrix(v=1)
        self.hamn = self._build_tunneling_matrix(v=-1)

        I = np.eye(2)
        self.preft = self.Rt.T @ (I + self.E_t)
        self.prefb = self.Rb.T @ (I + self.E_b)

        self.ctp, self.cbp = self._build_strain_correction(v=1)
        self.ctn, self.cbn = self._build_strain_correction(v=-1)

    def _build_strain_correction(self, v):
        facA = self.bG / (2 * self.dcc)
        At = facA * np.array([self.exxT - self.eyyT, -2 * self.exyT])  # top layer
        Ab = facA * np.array([self.exxB - self.eyyB, -2 * self.exyB])  # bottom layer

        kt = self.preft @ At
        kb = self.prefb @ Ab

        Vt = self.Vg * (self.exxT + self.eyyT) * np.eye(2)  # top layer
        Vb = self.Vg * (self.exxB + self.eyyB) * np.eye(2)  # bottom layer

        hAt = v * self.h0(kt, v)
        hAb = v * self.h0(kb, v)

        ct = Vt + hAt
        cb = Vb + hAb

        return ct, cb

    def _build_lattice(self):

        b0 = (4 * np.pi / (np.sqrt(3) * self.a)) * np.array(
            [[np.sqrt(3) / 2, -1 / 2], [0, 1]]
        )

        I = np.eye(2)
        T = (I - self.E_b) @ self.Rb - (I - self.E_t) @ self.Rt

        self.G = b0 @ T.T

        self.G = np.array([[-1, -1], [1, 0]]) @ self.G

        K0 = (b0[0] + 2 * b0[1]) / 3.0

        KT = (I - self.E_b) @ self.Rt @ K0
        KB = (I - self.E_t) @ self.Rb @ K0

        self.KT = fold(KT, self.G[0], self.G[1], N=4)
        self.KB = fold(KB, self.G[0], self.G[1], N=4)

        # self.KT = np.dot(fracT, self.G)
        # self.KB = np.dot(fracB, self.G)

        q1 = (self.G[0] - self.G[1]) / 3.0
        q2 = q1 + self.G[1]
        q3 = q1 - self.G[0]

        self.Kt = q2
        self.Kb = -q3

        # self.Kt = self.KT
        # self.Kb = self.KB

        # self.mbz = np.array([q1, -q3, q2, -q1, q3, -q2, q1])
        self.mbz = ws_cell(self.G[0], self.G[1], N=4)

        self.Acr = np.abs(np.linalg.det(self.G))

        self.lat = 2 * np.pi * np.linalg.inv(self.G).T

        # q1R = -(2 * self.lat[0] - self.lat[1]) / 3
        # q2R = q1R + self.lat[0]
        # q3R = q1R + self.lat[0] - self.lat[1]

        self.Ac = np.abs(np.linalg.det(self.lat))

        # self.WScell = np.array([q1R, -q3R, q2R, -q1R, q3R, -q2R, q1R])
        self.WScell = ws_cell(self.lat[0], self.lat[1], N=4)

    def _build_Gmesh(self):

        Gcut = self.cutoff * np.max(np.linalg.norm(self.G, axis=1))
        expand_times = self.cutoff + 2
        X, Y = np.meshgrid(
            np.arange(-expand_times, expand_times),
            np.arange(-expand_times, expand_times),
            indexing="ij",
        )
        X, Y = X.flatten(), Y.flatten()
        Gmesh = np.dot(np.vstack([X, Y]).T, self.G)
        Gmeshnorm = np.linalg.norm(Gmesh, axis=1)
        index = np.where(Gmeshnorm < Gcut)[0]
        sort_index = np.argsort(Gmeshnorm[index])
        self.Gmesh = Gmesh[index][sort_index]

        self.NG = len(self.Gmesh)

        if self.test:
            A_cell = np.array(
                [
                    0 * self.G[0],
                    self.G[0],
                    self.G[0] + self.G[1],
                    self.G[1],
                    0 * self.G[0],
                ]
            )
            print(f"Number of G points: {len(self.Gmesh)}")
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.scatter(self.Gmesh[:, 0], self.Gmesh[:, 1], color="blue", s=10)
            plt.title("G mesh points")
            plt.plot(self.mbz[:, 0], self.mbz[:, 1], color="red", lw=2, label="MBZ")
            plt.plot(A_cell[:, 0], A_cell[:, 1], color="green", lw=2, label="Unit Cell")

            plt.scatter(self.KT[0], self.KT[1], color="orange", s=150, label="K point")
            plt.scatter(self.KB[0], self.KB[1], color="purple", s=150, label="K' point")

            plt.scatter(self.Kt[0], self.Kt[1], color="cyan", s=50, label="q2 point")
            plt.scatter(self.Kb[0], self.Kb[1], color="magenta", s=50, label="q3 point")
            plt.legend()
            ax.set_aspect("equal")
            plt.show()

            # exit()

    def _build_tunneling_matrix(self, v):

        omega = np.exp(2j * np.pi / 3)
        T0 = np.array([[self.gammaAA, self.gammaAB], [self.gammaAB, self.gammaAA]])
        T1 = np.array(
            [
                [self.gammaAA, self.gammaAB * (omega ** (-v))],
                [self.gammaAB * omega ** (v), self.gammaAA],
            ]
        )
        T2 = T1.T
        tn = [T0, T1, T2]
        GN = np.array([[0, 0], -v * self.G[1], v * self.G[0]])

        Gdiff = self.Gmesh[:, np.newaxis, :] - self.Gmesh[np.newaxis, :, :]

        GNGdiff = np.linalg.norm(
            Gdiff[:, :, np.newaxis, :] - GN[np.newaxis, np.newaxis, :, :], axis=-1
        )
        GNindex = np.array(np.where(GNGdiff < 1e-10)).T

        size = len(self.Gmesh) * 4  # 4 for 2 layers and 2 sublattices
        hamv = np.zeros((size, size), dtype=complex)

        for j in range(len(GNindex)):
            col = GNindex[j, 0] * 4
            row = GNindex[j, 1] * 4
            hamv[row : row + 2, col + 2 : col + 4] = tn[GNindex[j, 2]]

        hamv += hamv.conj().T

        self.hdim = size

        return hamv

    def _build_tunnelling_matrix(self, v):
        pass

    def h0(self, k, v):
        H0 = -self.hv * np.array(
            [[0, v * k[0] - 1j * k[1]], [v * k[0] + 1j * k[1], 0]], dtype=np.complex128
        )

        return H0

    def ham(self, k, v):
        if v == -1:
            T = self.hamn
            Ct = self.ctn
            Cb = self.cbn
        else:
            T = self.hamp
            Ct = self.ctp
            Cb = self.cbp
        block = []

        for i in range(len(self.Gmesh)):
            H0 = self.h0(self.preft.dot(k + self.Gmesh[i] - v * self.Kt), v) + Ct
            H1 = self.h0(self.prefb.dot(k + self.Gmesh[i] - v * self.Kb), v) + Cb
            block.append(H0)
            block.append(H1)
        H = block_diag(block).toarray()
        H = H + T
        return H

    def set_uni(self, eh, phi):
        nv = 0.16

        eh = eh / 100
        phi = np.deg2rad(phi)

        exx = eh * (np.cos(phi) ** 2 - nv * np.sin(phi) ** 2)
        eyy = eh * (np.sin(phi) ** 2 - nv * np.cos(phi) ** 2)
        exy = eh * (1 + nv) * np.sin(phi) * np.cos(phi)

        # Symmetric configuration

        self.exyT = exy / 2
        self.exxT = exx / 2
        self.eyyT = eyy / 2

        self.exyB = -exy / 2
        self.exxB = -exx / 2
        self.eyyB = -eyy / 2

    def set_shear(self, eh, phi):
        eh = eh / 100
        phi = np.deg2rad(phi)

        exx = -eh * np.sin(2 * phi)
        eyy = eh * np.sin(2 * phi)
        exy = eh * np.cos(2 * phi)

        # Symmetric configuration

        self.exyT = exy / 2
        self.exxT = exx / 2
        self.eyyT = eyy / 2

        self.exyB = -exy / 2
        self.exxB = -exx / 2
        self.eyyB = -eyy / 2

    def set_strain_tensor(self):
        pass

    def calc_band(self, v, knum=200):

        kpoints = np.array(
            [
                [0, 0],
                self.G[1] / 2.0,
                (2 * self.G[1] + self.G[0]) / 3.0,
                [0, 0],
                (2 * self.G[0] + self.G[1]) / 3.0,
            ]
        )

        kidx, kpath = gen_kpath(kpoints, knum)
        eigs = []

        for k in kpath:
            H = self.ham(k, v)
            eigs.append(np.linalg.eigvalsh(H))

        return eigs, kidx


if __name__ == "__main__":

    ham = BMHamSetter(trad=np.deg2rad(1.05), eh=0, phi=0)
    eigs, kidx = ham.calc_band(v=-1, knum=200)
    eigs = np.array(eigs).T
    plt.figure(figsize=(6, 4))
    for i in range(eigs.shape[0]):
        plt.plot(eigs[i], color="black", lw=1)
    plt.xlim(0, 200)
    plt.ylim(-0.075, 0.075)
    plt.xticks(kidx, [r"$\Gamma$", "M", "K", r"$\Gamma$", "K'"])
    plt.ylabel("Energy (eV)")
    plt.title("Twisted Bilayer Graphene Band Structure")
    plt.grid()
    plt.tight_layout()
    plt.show()
