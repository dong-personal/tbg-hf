import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

from ham import BMHamSetter
from scipy.spatial import KDTree
from opt_einsum import contract

from scipy.stats import unitary_group
import random
from collections import namedtuple


@dataclass
class HFHamSetter(BMHamSetter):

    Nk: int = 12
    dsc: float = 40

    efac: float = 9.0279  # e^2 / 2\epsilon_0

    prefH = 1 / (4 * np.pi**2)
    prefF = prefH / 1

    epsilon: float = 10

    max_iter: int = 40

    mix = 0.2

    def __post_init__(self):
        super().__post_init__()

        mid = self.hdim // 2
        self.active_band: np.ndarray = np.array([mid - 1, mid])
        self.Nb: int = self.active_band.size
        self.ntotal: int = self.Nb * 4  # 4 for spin and valley
        self._build_kmesh()
        self.dk2 = self.Acr / self.Nk**2

        self.solve_spresults()
        self._build_form_factor()
        self._build_kinetic()
        self._build_hf()

        self.dmref = np.zeros(
            (self.Nk * self.Nk, self.ntotal, self.ntotal), dtype=np.complex128
        )  # .reshape(self.Nk * self.Nk, self.Nb, 2, 2, self.Nb, 2, 2)
        for k in range(self.Nk * self.Nk):
            self.dmref[k] = np.eye(self.ntotal, dtype=np.complex128) * 0.5
        #     for s in range(2):
        #         for v in range(2):
        #             self.dmref[k, 0 : self.Nb // 2, s, v, 0 : self.Nb // 2, s, v] = 1
        # self.dmref = self.dmref.reshape(self.Nk * self.Nk, self.ntotal, self.ntotal)

    def _build_kmesh(self):
        Nx, Ny = np.meshgrid(np.arange(self.Nk), np.arange(self.Nk), indexing="ij")
        Nx = Nx.flatten()
        Ny = Ny.flatten()

        self.kmesh = (np.array([Nx, Ny]).T / self.Nk) @ self.G

        GM = (
            np.array([1, 1])
            / 2
            / self.Nk
            * np.linspace(0, self.Nk, self.Nk // 2 + 1)[:, np.newaxis]
        )
        GK = (
            np.array([2, 1])
            / 3
            / self.Nk
            * np.linspace(0, self.Nk, self.Nk // 3 + 1)[:, np.newaxis]
        )

        GKp = (
            np.array([1, 2])
            / 3
            / self.Nk
            * np.linspace(0, self.Nk, self.Nk // 3 + 1)[:, np.newaxis]
        )

        KKp = (
            np.array([-1, 1])
            / 3
            / self.Nk
            * np.linspace(0, self.Nk, self.Nk // 3 + 1)[:, np.newaxis]
        ) + np.array([2, 1]) / 3

        MK = (
            np.array([1, -1])
            / 6
            / self.Nk
            * np.linspace(0, self.Nk, self.Nk // 6 + 1)[:, np.newaxis]
        ) + np.array([1, 1]) / 2

        # G -> M : Nk/2
        # G -> K : Nk/3
        # M -> K : Nk/6
        KPATH = namedtuple("KPATH", ["kidx", "kpathnorm", "matched_indices"])

        def concat(*args):

            kpath = np.concatenate(args, axis=0)
            kpath = kpath @ self.G
            tree = KDTree(self.kmesh)
            _, matched_indices = tree.query(kpath, distance_upper_bound=1e-8)
            kidx = np.array(np.cumsum([0] + [arg.shape[0] for arg in args]))
            kidx[1:] -= 1
            kpathnorm = np.cumsum(
                np.insert(np.linalg.norm(np.diff(kpath, axis=0), axis=1), 0, 0)
            )
            return KPATH(kidx, kpathnorm, matched_indices)

        # G -> K -> M -> G

        kpath1 = concat(GK, KKp[1:], GKp[-2::-1])

        self.kpath = kpath1

        if self.test:
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.scatter(self.kmesh[:, 0], self.kmesh[:, 1], s=10, color="k")
            plt.plot(self.mbz[:, 0], self.mbz[:, 1], color="r", linewidth=2)

            plt.arrow(0, 0, self.G[0, 0], self.G[0, 1], color="b", head_width=0.05)
            plt.arrow(0, 0, self.G[1, 0], self.G[1, 1], color="y", head_width=0.05)

            plt.plot(
                self.kmesh[self.kpath[2], 0], self.kmesh[self.kpath[2], 1], c="g", lw=2
            )
            ax.set_aspect("equal")
            plt.show()

    def solve_spresults(self, v=1):
        eigs, eigvecs = [], []
        eigsp, eigvecsp = [], []
        for k in self.kmesh:
            eig, vec = np.linalg.eigh(self.ham(k, v))
            vec = vec.T
            eigs.append(eig)
            eigvecs.append(vec)
        # [k, n]
        eigs = np.array(eigs)
        # [k, n, Glσ] first index is k, second index is band, third index is G l \sigma
        eigvecs = np.array(eigvecs)[:, self.active_band, :]

        # # [kx, ky, n, G, l, \sigma]
        phiG = eigvecs.reshape(self.Nk, self.Nk, self.Nb, self.NG, 2, 2)

        # bug to be fixed
        phiGC2T = np.conj(
            phiG[:, :, :, :, :, ::-1]
        )  # [kx, ky, n, G, l, \sigma] -> [kx, ky, n, G, l, \sigma] with layer flipped and complex conjugated

        UC2T = np.einsum(
            "xyngls,xymgls->xynm", np.conj(phiG), phiGC2T
        )  # [kx, ky, n, m]

        phase = -np.angle(np.diagonal(UC2T, axis1=2, axis2=3))  # [kx, ky, n]

        phiG = phiG * np.exp(
            -0.5j * phase[:, :, :, np.newaxis, np.newaxis, np.newaxis]
        )  # [kx, ky, n, G, l, \sigma]

        phiG = phiG.reshape(self.Nk * self.Nk, self.Nb, self.NG, 4)  # [k, n, G, lσ]

        print("test:", np.max(np.abs(phiG)))

        # self.phiG = phiG

        # k+G -> -k-G

        # [k, G, 2]
        kplusG = (self.kmesh[:, np.newaxis, :] + self.Gmesh[np.newaxis, :, :]).reshape(
            -1, 2
        )
        tree = KDTree(kplusG)
        _, matched_indices = tree.query(-kplusG, distance_upper_bound=1e-8)

        matched_indices = np.array(matched_indices)
        matchedk = matched_indices // self.NG
        matchedG = matched_indices % self.NG

        mask = matchedk < self.Nk * self.Nk
        matchedk = matchedk[mask]
        matchedG = matchedG[mask]
        kindex, Gindex = np.where(mask.reshape(self.Nk * self.Nk, self.NG))

        phiGp = np.zeros_like(phiG)
        phiGp[kindex, :, Gindex, :] = np.conj(phiG[matchedk, :, matchedG, :])

        if self.test:
            print("kplusG size:", kplusG.shape)

            print("matched_indices", matched_indices)

        print("test:", np.max(np.abs(phiGp)))
        self.phiG = phiG.transpose(1, 0, 3, 2)  # [n, k, lσ, G]
        self.phiGp = phiGp.transpose(1, 0, 3, 2)  # [n, k, lσ, G]

        Nx, Ny = np.meshgrid(np.arange(self.Nk), np.arange(self.Nk), indexing="ij")
        Nx = Nx.flatten()
        Ny = Ny.flatten()
        Nxm, Nym = -Nx % self.Nk, -Ny % self.Nk

        kmeshm = self.kmesh.reshape(self.Nk, self.Nk, 2)[Nxm, Nym]

        diff = (np.abs(kmeshm[:, np.newaxis] - self.kmesh[np.newaxis, :]) < 1e-10).all(
            axis=-1
        )
        midx, idx = np.where(diff)
        eigsp = np.zeros_like(eigs)
        eigsp[midx, :] = eigs[idx, :]

        # [k, n]
        self.eigs = eigs[:, self.active_band]
        self.eigsp = eigsp[:, self.active_band]

        if self.test:
            fig, ax = plt.subplots(figsize=(6, 8))

            for i in range(self.eigs.shape[1]):
                plt.plot(
                    self.kpath.kpathnorm,
                    self.eigs[self.kpath.matched_indices, i],
                    color="r",
                    lw=2,
                )
                plt.plot(
                    self.kpath.kpathnorm,
                    self.eigsp[self.kpath.matched_indices, i],
                    color="b",
                    lw=2,
                )

            plt.xlim(0, self.kpath.kpathnorm[-1])
            plt.xticks(
                [self.kpath.kpathnorm[i] for i in self.kpath.kidx],
                ["G", "K", "K'", "G"],
            )
            for i in self.kpath.kidx:
                ax.axvline(x=self.kpath.kpathnorm[i], color="k", linestyle="--")
            plt.show()

    @staticmethod
    def _random_unitary(n, rng):
        z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        q, r = np.linalg.qr(z)
        phase = np.diag(r)
        phase = np.where(np.abs(phase) > 0, phase / np.abs(phase), 1.0)
        return q * phase.conj()

    def initial_federico(self, filling):

        lenlistP = self.Nk * self.Nk

        Nst = int((filling + 4 * self.Nb // 2) * lenlistP)

        U = np.zeros((8, 8, lenlistP), dtype=np.complex128)

        count = 0

        while count < Nst:
            for a in range(8):  # run over all HF bands

                jr = random.randint(0, lenlistP - 1)

                if np.all(U[:, a, jr]) == 0:
                    U[:, a, jr] = unitary_group.rvs(8)[0]
                    count += 1

        " Order Parameter "

        op = np.einsum("iak, jak -> kij", np.conj(U), U)

        # make zero the off diagonal spin components
        op[:, 4:8, 0:4], op[:, 0:4, 4:8] = 0, 0

        # make zero intervalley terms

        op[:, 2:8, 0:2], op[:, 0:2, 2:8] = 0, 0
        op[:, 4:8, 2:4], op[:, 2:4, 4:8] = 0, 0
        op[:, 6:8, 4:6], op[:, 4:6, 6:8] = 0, 0

        op = op.reshape(self.Nk * self.Nk, 2, 2, 2, 2, 2, 2)

        op = op.transpose(0, 3, 1, 2, 6, 4, 5)

        op = op.reshape(self.Nk * self.Nk, self.ntotal, self.ntotal)
        return op

    def initial_random(self, filling, seed=None):
        """Build a random block-diagonal initial density matrix for HF iteration.

        The flattened single-particle index is interpreted as ``(band, spin, valley)``,
        matching ``hamhf`` and ``getdm`` in ``hf.py``.
        """
        rng = np.random.default_rng(seed)

        nk_total = self.Nk * self.Nk
        n_occ = self.ntotal // 2 + int(filling)
        if not 0 <= n_occ <= self.ntotal:
            raise ValueError(f"filling={filling} gives invalid n_occ={n_occ}")

        dm = np.zeros(
            (nk_total, self.Nb, 2, 2, self.Nb, 2, 2),
            dtype=np.complex128,
        )

        base_occ = np.full((2, 2), n_occ // 4, dtype=int)
        remainder = n_occ - int(base_occ.sum())

        for k in range(nk_total):
            occ = base_occ.copy()
            if remainder > 0:
                choices = rng.choice(4, size=remainder, replace=False)
                for choice in choices:
                    s, v = divmod(int(choice), 2)
                    occ[s, v] += 1

            for s in range(2):
                for v in range(2):
                    m = int(occ[s, v])
                    if m == 0:
                        continue

                    u = self._random_unitary(self.Nb, rng)
                    occupied = u[:, :m].T
                    rho = np.einsum("ai,aj->ij", occupied.conj(), occupied)
                    dm[k, :, s, v, :, s, v] = rho

        return dm.reshape(nk_total, self.ntotal, self.ntotal)

    def _build_form_factor(self):

        # for arbitrary q, k+q = p+Q
        # G + Q
        GplusG = (self.Gmesh[:, np.newaxis] + self.Gmesh[np.newaxis, :]).reshape(-1, 2)
        tree = KDTree(self.Gmesh)
        _, matched_indices = tree.query(GplusG, distance_upper_bound=1e-8)
        matched_indices = np.array(matched_indices)

        mask = matched_indices < self.NG

        # [n, k, lσ, G, Q]
        phiGG = np.zeros(
            (*(self.phiG.shape[0:-1]), self.NG * self.NG), dtype=np.complex128
        )
        phiGG[..., mask] = self.phiG[..., matched_indices[mask]]
        phiGG = phiGG.reshape((*self.phiG.shape, self.NG))

        form_factors = contract("mpigq, nkig->mnkpq", np.conj(phiGG), self.phiG)

        # form_factors = contract("mpig, nkigq->mnkpq", np.conj(self.phiG), phiGG)

        phiGGp = np.zeros(
            (*(self.phiGp.shape[0:-1]), self.NG * self.NG), dtype=np.complex128
        )
        phiGGp[..., mask] = self.phiGp[..., matched_indices[mask]]
        phiGGp = phiGGp.reshape((*self.phiGp.shape, self.NG))

        form_factorsp = contract("mpigq, nkig->mnkpq", np.conj(phiGGp), self.phiGp)

        # [v, m, n, k, p, Q]
        return np.stack((form_factors, form_factorsp), axis=0)

    def _build_kinetic(self):
        # 2 for spin, 2 for valley [k, n, s, v]
        diag = np.zeros((self.Nk * self.Nk, self.Nb, 2, 2), dtype=np.complex128)

        diag[:, :, :, 0] = self.eigs[:, :, np.newaxis]
        diag[:, :, :, 1] = self.eigsp[:, :, np.newaxis]

        diag = diag.reshape(self.Nk * self.Nk, self.ntotal)

        h0 = np.zeros(
            (self.Nk * self.Nk, self.ntotal, self.ntotal), dtype=np.complex128
        )

        for i in range(self.ntotal):
            h0[:, i, i] = diag[:, i]

        self.K = h0

    def _build_hf(self):
        # [v, n, n', k, k', G]
        form_factors = self._build_form_factor()
        conj = np.conj(form_factors)

        # q = -k + p + Q
        kminuskpminusG = (
            -self.kmesh[:, np.newaxis, np.newaxis, :]
            + self.kmesh[np.newaxis, :, np.newaxis, :]
            + self.Gmesh[np.newaxis, np.newaxis, :, :]
        )
        norm = np.linalg.norm(kminuskpminusG, axis=-1)

        zero_mask = norm < 1e-10

        # [k, p, Q]
        VcF = (
            self.efac
            * np.tanh(self.dsc * norm)
            / np.where(norm > 1e-10, norm, 1e-10)
            / self.epsilon
            * self.dk2
            / (4 * np.pi**2)
        )
        VcF[zero_mask] = self.efac * self.dsc / self.epsilon * self.dk2 / (4 * np.pi**2)

        VcG = VcF[0, 0]
        Gnorm = np.linalg.norm(self.Gmesh, axis=-1)
        Gzero_mask = Gnorm < 1e-10
        VcG[Gzero_mask] = 0.0

        # n1 -> m, n2 -> n, n3 -> r, n4 -> s
        self.VH = contract("q, vmrkkq, Vsnppq->kpmnvrsV", VcG, form_factors, conj)

        # D rn C ms
        self.VF = contract("kpq, vmrkpq, Vsnkpq->kpmnvrsV", VcF, form_factors, conj)

        # VH = VH.transpose(0, 2, 3, 4, 1, 5, 6, 7)
        # VF = VF.transpose(0, 2, 3, 4, 7, 1, 5, 6)

        # VH = VH.reshape(self.Nk * self.Nk * 2 * 2 * 2, self.Nk * self.Nk * 2 * 2 * 2)

        # VF = VF.reshape(self.Nk * self.Nk * 2 * 2, 2 * 2, self.Nk * self.Nk * 2 * 2)

    def hamhf(self, dm):
        # dm: [k, nsv, n's'v']
        dm_in = dm - self.dmref
        dm_in = dm_in.reshape(self.Nk * self.Nk, self.Nb, 2, 2, self.Nb, 2, 2)

        H = np.zeros(
            (self.Nk * self.Nk, self.Nb, 2, 2, self.Nb, 2, 2), dtype=np.complex128
        )

        F = np.zeros(
            (self.Nk * self.Nk, self.Nb, 2, 2, self.Nb, 2, 2), dtype=np.complex128
        )

        for s1 in range(2):
            for s2 in range(2):
                H[:, :, s1, 0, :, s1, 0] += contract(
                    "pkmnvrs,pmvrv->kns", self.VH[..., 0], dm_in[:, :, s2, :, :, s2, :]
                )

                H[:, :, s1, 1, :, s1, 1] += contract(
                    "pkmnvrs,pmvrv->kns", self.VH[..., 1], dm_in[:, :, s2, :, :, s2, :]
                )

                F[:, :, s1, :, :, s2, :] = -contract(
                    "kpmnvrsV,pmvsV->knVrv", self.VF, dm_in[:, :, s2, :, :, s1, :]
                )

        H = H.reshape(self.Nk * self.Nk, self.ntotal, self.ntotal)
        F = F.reshape(self.Nk * self.Nk, self.ntotal, self.ntotal)
        ham = self.K + H + F

        if self.test:
            print(
                "Hermitian error:", np.max(np.abs(ham - ham.conj().transpose(0, 2, 1)))
            )

        return ham

    def getdm(self, eigs, states, filling):

        eigs = np.array(eigs).flatten()

        occupied_num = (filling + 4 * self.Nb // 2) * (self.Nk * self.Nk)
        sorted_indices = np.argsort(eigs)
        occupied_indices = sorted_indices[0 : int(occupied_num)]

        fermi_energy = eigs[occupied_indices[-1]]

        occupiedk = occupied_indices // self.ntotal
        occupiedn = occupied_indices % self.ntotal

        dm = np.zeros(
            (self.Nk * self.Nk, self.ntotal, self.ntotal), dtype=np.complex128
        )

        for k in range(self.Nk * self.Nk):
            occupied_states_k = states[k][occupiedn[occupiedk == k]]
            dm[k] = contract("mi,mj->ij", np.conj(occupied_states_k), occupied_states_k)

        dm = dm.reshape(self.Nk * self.Nk, self.Nb, 2, 2, self.Nb, 2, 2)

        dm[:, :, 0, :, :, 1, :] = 0
        dm[:, :, 1, :, :, 0, :] = 0

        dm[:, :, :, 0, :, :, 1] = 0
        dm[:, :, :, 1, :, :, 0] = 0

        dm = dm.reshape(self.Nk * self.Nk, self.ntotal, self.ntotal)

        return dm, fermi_energy

    def scf(self, filling):
        dm = self.initial_federico(filling)
        step = 0
        while True:
            ham = self.hamhf(dm)
            states = []
            eigs = []
            for k in range(self.Nk * self.Nk):
                eig, vec = np.linalg.eigh(ham[k])
                states.append(vec.T)
                eigs.append(eig)
            states = np.array(states)

            # simple mixing
            dm_new, fermi_energy = self.getdm(eigs, states, filling)

            dm_new = dm_new * self.mix + (1 - self.mix) * dm

            error = np.max(np.abs(dm_new - dm))
            step += 1

            print("step:", step, "error:", error)

            if error < 1e-6 or step >= self.max_iter:
                break

            dm = dm_new

        return np.array(eigs), np.array(states), fermi_energy


if __name__ == "__main__":
    hf = HFHamSetter()
    eigs, states, fermi = hf.scf(filling=-1)
    # eigs [k,nsv]
    fig, ax = plt.subplots(figsize=(6, 8))

    for i in range(eigs.shape[1]):
        plt.plot(hf.kpath.kpathnorm, eigs[hf.kpath.matched_indices, i], color="r", lw=2)

    plt.xlim(0, hf.kpath.kpathnorm[-1])
    plt.xticks(
        [hf.kpath.kpathnorm[i] for i in hf.kpath.kidx],
        ["G", "K", "K'", "G"],
    )
    plt.axhline(y=fermi, color="b", lw=2, ls="--", label="Fermi energy")
    for i in hf.kpath.kidx:
        ax.axvline(x=hf.kpath.kpathnorm[i], color="k", linestyle="--")

    plt.ylabel("Energy (eV)")
    plt.show()
