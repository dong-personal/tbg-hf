import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass

from cm import BMHamSetter
from scipy.spatial import KDTree
from opt_einsum import contract

from collections import namedtuple
import numpy as np

from mpi import ENV


def haar_unitary(N, rng):
    """Generate a Haar-random U(N) matrix."""
    z = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))

    q, r = np.linalg.qr(z)

    d = np.diag(r)
    phase = d / np.abs(d)

    return q @ np.diag(np.conj(phase))


def random_density_block(dim, occupation, rng):
    """
    Generate a random Hermitian density matrix with

        Tr(P) = occupation
        0 <= eigenvalues <= 1

    If occupation is integer, P is a projector:
        P^2 = P.

    If occupation is fractional, one eigenvalue is fractional.
    """

    if occupation < -1e-12 or occupation > dim + 1e-12:
        raise ValueError(f"occupation={occupation} outside [0, {dim}]")

    U = haar_unitary(dim, rng)

    occ = np.zeros(dim)

    n_full = int(np.floor(occupation + 1e-12))
    frac = occupation - n_full

    if n_full > 0:
        occ[:n_full] = 1.0

    if frac > 1e-12:
        occ[n_full] = frac

    # U diag(occ) U^\dagger
    return (U * occ) @ U.conj().T


def random_density_matrix(
    Nk,
    Nn,
    filling=0,
    allow_spin_coherence=True,
    allow_valley_coherence=True,
    seed=None,
):
    """
    Generate random density matrix in basis

        |k, n, s, v>

    Array convention:

        P[k, n, s, v, n', s', v']

      = < c^\dagger_{k,n',s',v'}
          c_{k,n,s,v} >

    Parameters
    ----------
    Nk : int
        Number of k points.

    Nn : int
        Number of active bands per spin/valley.

    filling : float
        Filling relative to charge neutrality.

        Total occupation per k:

            Nocc = 2*Nn + filling

        For Nn=2:
            filling = -4 ... +4

    allow_spin_coherence : bool
        True:
            spin coherence/polarization are allowed.

        False:
            spin coherence forbidden:
                P_{up,down} = 0

            AND spin polarization forbidden:
                P_{up,up} = P_{down,down}.

    allow_valley_coherence : bool
        True:
            valley coherence/polarization are allowed.

        False:
            valley coherence forbidden:
                P_{K,K'} = 0

            AND valley polarization forbidden:
                P_{K,K} = P_{K',K'}.

    seed : int or None
        Random seed.

    Returns
    -------
    P : ndarray
        Shape:

        (Nk, Nn, 2, 2, Nn, 2, 2)

        corresponding to

        (k,n,s,v,n',s',v')
    """

    rng = np.random.default_rng(seed)

    Ns = 2
    Nv = 2

    D = Nn * Ns * Nv

    # neutrality = half filling
    Nocc = 2 * Nn + filling

    if Nocc < 0 or Nocc > D:
        raise ValueError(f"filling must satisfy " f"{-2*Nn} <= filling <= {2*Nn}")

    P = np.zeros(
        (Nk, Nn, Ns, Nv, Nn, Ns, Nv),
        dtype=complex,
    )

    # =====================================================
    # Case 1
    # Spin coherence allowed
    # Valley coherence allowed
    #
    # Full random density matrix in (n,s,v)
    # =====================================================
    if allow_spin_coherence and allow_valley_coherence:

        for k in range(Nk):

            Pblock = random_density_block(
                D,
                Nocc,
                rng,
            )

            P[k] = Pblock.reshape(
                Nn,
                Ns,
                Nv,
                Nn,
                Ns,
                Nv,
            )

    # =====================================================
    # Case 2
    # Spin coherence FORBIDDEN
    # Spin polarization FORBIDDEN
    #
    # Valley coherence allowed.
    #
    # Therefore:
    #
    # P_up = P_down
    #
    # Each spin block acts in (n,v).
    # =====================================================
    elif not allow_spin_coherence and allow_valley_coherence:

        block_dim = Nn * Nv

        # Two identical spin blocks
        occupation_per_spin = Nocc / 2

        for k in range(Nk):

            Pblock = random_density_block(
                block_dim,
                occupation_per_spin,
                rng,
            )

            Pblock = Pblock.reshape(
                Nn,
                Nv,
                Nn,
                Nv,
            )

            # SAME block for up and down
            for s in range(Ns):
                P[k, :, s, :, :, s, :] = Pblock

    # =====================================================
    # Case 3
    # Valley coherence FORBIDDEN
    # Valley polarization FORBIDDEN
    #
    # Spin coherence allowed.
    #
    # Therefore:
    #
    # P_K = P_K'
    #
    # Each valley block acts in (n,s).
    # =====================================================
    elif allow_spin_coherence and not allow_valley_coherence:

        block_dim = Nn * Ns

        # Two identical valley blocks
        occupation_per_valley = Nocc / 2

        for k in range(Nk):

            Pblock = random_density_block(
                block_dim,
                occupation_per_valley,
                rng,
            )

            Pblock = Pblock.reshape(
                Nn,
                Ns,
                Nn,
                Ns,
            )

            # SAME block for K and K'
            for v in range(Nv):
                P[k, :, :, v, :, :, v] = Pblock

    # =====================================================
    # Case 4
    # Both spin and valley coherence FORBIDDEN
    # Both spin and valley polarization FORBIDDEN
    #
    # All four (s,v) blocks must be identical.
    #
    # Only band space n remains random.
    # =====================================================
    else:

        block_dim = Nn

        # Four identical spin-valley blocks
        occupation_per_flavor = Nocc / 4

        for k in range(Nk):

            Pblock = random_density_block(
                block_dim,
                occupation_per_flavor,
                rng,
            )

            for s in range(Ns):
                for v in range(Nv):

                    P[k, :, s, v, :, s, v] = Pblock

    P = P.reshape(Nk, Nn * Ns * Nv, Nn * Ns * Nv)

    P = np.transpose(P, (0, 2, 1))

    return P


def init_fmi_valley(
    Nk,
    valley=0,
):
    """
    Valley-polarized FMI at filling = 0. valley = 0 or 1
    """
    Nn = 2
    Ns = 2
    Nv = 2

    rho = np.zeros(
        (Nk, Nn, Ns, Nv, Nn, Ns, Nv),
        dtype=complex,
    )

    # Fill both bands and both spins
    # in one valley.
    for k in range(Nk):
        for n in range(Nn):
            for s in range(Ns):

                rho[k, n, s, valley, n, s, valley] = 1.0

    rho = rho.reshape(Nk, Nn * Ns * Nv, Nn * Ns * Nv)

    return rho


def init_c2ti(
    Nk,
    sign=1,
):
    """
    C2T-breaking initial state at filling = 0.
    sign = +1:
        occupy gamma_x = +1 eigenstate

    sign = -1:
        occupy gamma_x = -1 eigenstate
    """

    Nn = 2
    Ns = 2
    Nv = 2

    rho = np.zeros(
        (Nk, Nn, Ns, Nv, Nn, Ns, Nv),
        dtype=complex,
    )

    gamma0 = np.eye(2, dtype=complex)

    gammax = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=complex,
    )

    # Rank-1 band projector
    Pband = 0.5 * (gamma0 + sign * gammax)

    for k in range(Nk):
        for s in range(Ns):
            for v in range(Nv):

                rho[k, :, s, v, :, s, v] = Pband

    rho = rho.reshape(Nk, Nn * Ns * Nv, Nn * Ns * Nv)
    return rho


def init_ivc(
    Nk,
    sign=1,
    valley_direction="x",
):
    """
    IVC initial density matrix at filling = 0.

    Basis:
        |n, s, v>

    n = band
    s = spin
    v = valley

    Default order parameter:

        gamma_y * s_0 * tau_x

    corresponding to the spin-unpolarized IVC state.

    Parameters
    ----------
    Nk : int
        Total number of k points, i.e. self.Nk * self.Nk.

    sign : +1 or -1
        Choose the two opposite orientations of the IVC order.

    valley_direction : "x" or "y"
        "x":
            gamma_y s_0 tau_x

        "y":
            gamma_y s_0 tau_y

    Returns
    -------
    rho : ndarray
        shape = (Nk, 8, 8)

        Convention is the same as getdm():

            rho_ij = <c_i^\dagger c_j>
    """

    Nn = 2
    Ns = 2
    Nv = 2

    # ----------------------------------------
    # Pauli matrices
    # ----------------------------------------

    gamma_y = np.array(
        [
            [0.0, -1.0j],
            [1.0j, 0.0],
        ],
        dtype=np.complex128,
    )

    s0 = np.eye(
        2,
        dtype=np.complex128,
    )

    tau_x = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.complex128,
    )

    tau_y = np.array(
        [
            [0.0, -1.0j],
            [1.0j, 0.0],
        ],
        dtype=np.complex128,
    )

    if valley_direction == "x":
        tau = tau_x

    elif valley_direction == "y":
        tau = tau_y

    else:
        raise ValueError("valley_direction must be 'x' or 'y'")

    # ----------------------------------------
    # Basis ordering is:
    #
    #     band ⊗ spin ⊗ valley
    #
    # because your array is [n,s,v].
    # ----------------------------------------

    Oivc = np.kron(np.kron(gamma_y, s0), tau)

    # ----------------------------------------
    # Diagonalize the IVC order operator
    #
    # eigenvalues should be:
    #
    #     -1,-1,-1,-1,+1,+1,+1,+1
    #
    # At neutrality occupy four states.
    # ----------------------------------------

    eig, vec = np.linalg.eigh(Oivc)

    if sign > 0:
        occ = np.where(eig > 0)[0]
    else:
        occ = np.where(eig < 0)[0]

    Uocc = vec[:, occ]

    rho0 = Uocc.conj() @ Uocc.T

    rho = np.repeat(
        rho0[np.newaxis, :, :],
        Nk,
        axis=0,
    )

    return rho


@dataclass
class HFHamSetter(BMHamSetter):

    Nk: int = 18
    dsc: float = 40

    efac: float = 9.0279  # e^2 / 2\epsilon_0

    prefH = 1 / (4 * np.pi**2)
    prefF = prefH / 1

    epsilon: float = 7

    max_iter: int = 10

    mix: float = 0.2

    useoda: int = 1

    spin_coherence: bool = False
    valley_coherence: bool = False

    spin_polarization: bool = False
    valley_polarization: bool = False
    coulomb_type: int = 2
    ref_type: int = 2

    def __post_init__(self):
        super().__post_init__()

        mid = self.hdim // 2
        self.active_band: np.ndarray = np.array([mid - 1, mid])
        self.Nb: int = self.active_band.size
        self.ntotal: int = self.Nb * 4  # 4 for spin and valley
        self._build_kmesh()
        self.dk2 = self.Acr / self.Nk**2

        self.solve_spresults()
        # self._build_form_factor()
        self._build_kinetic()
        self._build_hf()

        self.dmref = np.zeros(
            (self.Nk * self.Nk, self.ntotal, self.ntotal), dtype=np.complex128
        )
        if self.ref_type == 1:
            for k in range(self.Nk * self.Nk):
                self.dmref[k] = np.eye(self.ntotal, dtype=np.complex128) * 0.5
        elif self.ref_type == 2:
            self.dmref = self.dmref.reshape(
                self.Nk * self.Nk, self.Nb, 2, 2, self.Nb, 2, 2
            )

            for s in range(2):
                for v in range(2):
                    self.dmref[:, 0 : self.Nb // 2, s, v, 0 : self.Nb // 2, s, v] = 1

            self.dmref = self.dmref.reshape(self.Nk * self.Nk, self.ntotal, self.ntotal)

        elif self.ref_type == 3:

            # [k, n, v]
            eigs = np.array((self.eigs, self.eigsp)).transpose(1, 2, 0)
            sorted_indices = np.argsort(eigs.flatten())
            occupied_num = 2 * self.Nb * self.Nk * self.Nk // 2
            fermi_idx = sorted_indices[occupied_num - 1]
            fermi = eigs.flatten()[fermi_idx]
            deg_tol = 1e-8
            below = np.where(eigs < fermi - deg_tol)
            belownum = below[0].size

            deg = np.where(np.abs(eigs - fermi) < deg_tol)

            degnum = deg[0].size

            self.dmref = self.dmref.reshape(
                self.Nk * self.Nk, self.Nb, 2, 2, self.Nb, 2, 2
            )

            if degnum > 0:
                frac = (occupied_num - belownum) / degnum
                print("frac:", frac)
            for s in range(2):
                self.dmref[below[0], below[1], s, below[2], below[1], s, below[2]] = 1.0

                if degnum > 0:
                    self.dmref[deg[0], deg[1], s, deg[2], deg[1], s, deg[2]] = (
                        occupied_num - belownum
                    ) / degnum
            self.dmref = self.dmref.reshape(self.Nk * self.Nk, self.ntotal, self.ntotal)

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

        MKp = (
            np.array([-1, 1])
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

        # G -> K -> K' -> G
        kpath1 = concat(GK, KKp[1:], GKp[-2::-1])

        # K-> G -> M -> K'
        kpath2 = concat(GK[::-1], GM[1:], MKp[1:])

        self.kpath = kpath2

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

        def solve(v):

            def f(k):
                eig, vec = np.linalg.eigh(self.ham(k, v))
                return eig[self.active_band], vec.T[self.active_band, :]

            results = ENV.map(f, self.kmesh)
            results = ENV.allgather(results)
            eigs, eigvecs = zip(*results)

            eigs = np.array(eigs)
            eigvecs = np.array(eigvecs)

            return eigs, eigvecs

        def C2T(eigvecs):
            # # [kx, ky, n, G, l, \sigma]
            phiG = eigvecs.reshape(self.Nk, self.Nk, self.Nb, self.NG, 2, 2)

            # # # bug to be fixed
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

            phiG = phiG.reshape(self.Nk * self.Nk, self.Nb, self.NG, 4)

            return phiG

        eigs, eigvecs = solve(v)
        eigsp, eigvecsp = solve(-v)

        # [k, n, G, lσ]
        phiG = eigvecs.reshape(self.Nk * self.Nk, self.Nb, self.NG, 4)
        phiGp = eigvecsp.reshape(self.Nk * self.Nk, self.Nb, self.NG, 4)

        def regauge(phiG, phiGp, eigs, eigsp, tol=1e-4):
            Nk2 = self.Nk * self.Nk

            kplusG = (self.kmesh[:, None, :] + self.Gmesh[None, :, :]).reshape(-1, 2)

            tree = KDTree(kplusG)

            dist, matched = tree.query(
                -kplusG,
                distance_upper_bound=1e-8,
            )

            valid = matched < kplusG.shape[0]
            target_flat = np.arange(Nk2 * self.NG)

            target_k = target_flat[valid] // self.NG
            target_G = target_flat[valid] % self.NG

            source_flat = matched[valid]  # type: ignore
            source_k = source_flat // self.NG
            source_G = source_flat % self.NG

            phiG_tr = np.zeros_like(phiGp)

            phiG_tr[target_k, :, target_G, :] = np.conj(phiG[source_k, :, source_G, :])

            U = contract("kngl,kmgl->knm", np.conj(phiGp), phiG_tr)

            # for diagonal sewing matrix
            Uabs = np.abs(U)
            nondiagonal = Uabs[:, 0, 1] > 1e-4
            diagonal = ~nondiagonal

            phase = np.angle(np.diagonal(U, axis1=1, axis2=2))
            phiGp[diagonal] = phiGp[diagonal] * np.exp(
                1j * phase[diagonal, :, np.newaxis, np.newaxis]
            )

            # nondiagonal sewing matrix, U(2) rotation
            for k in np.where(nondiagonal)[0]:
                u, s, Vh = np.linalg.svd(U[k])
                # nearest unitary matrix to B
                Q = u @ Vh
                # transformation acting on row-wise band kets
                R = Q.T
                shape = phiGp[k].shape
                states = phiGp[k].reshape(self.Nb, -1)
                states = R @ states
                phiGp[k] = states.reshape(shape)
            return phiG, phiGp

        def check_valley_sewing(phiG, phiGp):
            Nk2 = self.Nk * self.Nk

            kplusG = (self.kmesh[:, None, :] + self.Gmesh[None, :, :]).reshape(-1, 2)

            tree = KDTree(kplusG)

            dist, matched = tree.query(
                -kplusG,
                distance_upper_bound=1e-8,
            )

            valid = matched < kplusG.shape[0]
            target_flat = np.arange(Nk2 * self.NG)

            target_k = target_flat[valid] // self.NG
            target_G = target_flat[valid] % self.NG

            source_flat = matched[valid]  # type: ignore
            source_k = source_flat // self.NG
            source_G = source_flat % self.NG

            phiG_tr = np.zeros_like(phiGp)

            phiG_tr[target_k, :, target_G, :] = np.conj(phiG[source_k, :, source_G, :])

            U = contract("kngl,kmgl->knm", np.conj(phiGp), phiG_tr)
            phase = np.angle(np.diagonal(U, axis1=1, axis2=2))
            Uabs = np.abs(U)

            nondiagonal = np.where(Uabs[:, 0, 1] > 1e-4)[0]

            print("nondiagonal:", nondiagonal, Uabs[nondiagonal])

            idx = self.kpath.matched_indices
            norm = self.kpath.kpathnorm
            kidx = self.kpath.kidx

            fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
            ax[0].plot(norm, Uabs[idx, 0, 0], color="r", lw=2, label="Uabs[0,0]")
            ax[0].plot(norm, Uabs[idx, 1, 1], color="b", lw=2, label="Uabs[1,1]")
            ax[0].plot(norm, Uabs[idx, 0, 1], color="g", lw=2, label="Uabs[0,1]")
            ax[0].plot(norm, Uabs[idx, 1, 0], color="y", lw=2, label="Uabs[1,0]")
            ax[0].set_xlim(0, norm[-1])
            ax[0].set_xticks([norm[i] for i in kidx], ["K", "G", "M", "K'"])
            for i in kidx:
                ax[0].axvline(x=norm[i], color="k", linestyle="--")
            ax[0].legend()

            ax[1].plot(norm, phase[idx, 0], color="r", lw=2, label="phase[0,0]")
            ax[1].plot(norm, phase[idx, 1], color="b", lw=2, label="phase[1,1]")

            plt.show()

            fig, ax = plt.subplots(figsize=(6, 6))
            plt.scatter(self.kmesh[:, 0], self.kmesh[:, 1], s=10, color="k")

            plt.scatter(
                self.kmesh[nondiagonal, 0], self.kmesh[nondiagonal, 1], s=50, color="b"
            )
            plt.plot(self.mbz[:, 0], self.mbz[:, 1], color="r", linewidth=2)

            plt.arrow(0, 0, self.G[0, 0], self.G[0, 1], color="b", head_width=0.05)
            plt.arrow(0, 0, self.G[1, 0], self.G[1, 1], color="y", head_width=0.05)

            plt.plot(
                self.kmesh[self.kpath[2], 0], self.kmesh[self.kpath[2], 1], c="g", lw=2
            )
            ax.set_aspect("equal")
            plt.show()

        phiG = C2T(eigvecs)
        phiGp = C2T(eigvecsp)
        if self.test:
            check_valley_sewing(phiG, phiGp)
        phiG, phiGp = regauge(phiG, phiGp, eigs, eigsp)

        self.phiG = phiG.transpose(1, 0, 3, 2)  # [n, k, lσ, G]
        self.phiGp = phiGp.transpose(1, 0, 3, 2)  # [n, k, lσ, G]

        # [k, n]
        self.eigs = eigs
        self.eigsp = eigsp
        if self.test:
            check_valley_sewing(phiG, phiGp)

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

    def _build_form_factor(self):

        # for arbitrary q, k+q = p+Q
        # G + Q
        GplusG = (self.Gmesh[:, np.newaxis] + self.Gmesh[np.newaxis, :]).reshape(-1, 2)
        tree = KDTree(self.Gmesh)
        _, matched_indices = tree.query(GplusG, distance_upper_bound=1e-8)
        matched_indices = np.array(matched_indices)

        mask = matched_indices < self.NG

        # [n, p, lσ, G, Q]
        phiGG = np.zeros(
            (*(self.phiG.shape[0:-1]), self.NG * self.NG), dtype=np.complex128
        )
        phiGG[..., mask] = self.phiG[..., matched_indices[mask]]
        phiGG = phiGG.reshape((*self.phiG.shape, self.NG))

        form_factors = contract("mpigq, nkig->mnpkq", np.conj(phiGG), self.phiG)

        # form_factors = contract("mpig, nkigq->mnkpq", np.conj(self.phiG), phiGG)

        phiGGp = np.zeros(
            (*(self.phiGp.shape[0:-1]), self.NG * self.NG), dtype=np.complex128
        )
        phiGGp[..., mask] = self.phiGp[..., matched_indices[mask]]
        phiGGp = phiGGp.reshape((*self.phiGp.shape, self.NG))

        form_factorsp = contract("mpigq, nkig->mnpkq", np.conj(phiGGp), self.phiGp)

        # [v, m, n, p, k, Q]
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

    def screened_coulomb(self, q, type=1):
        norm = np.linalg.norm(q, axis=-1)
        zero_mask = norm < 1e-10
        if type == 1:
            # [k, p, Q]
            VcF = (
                self.efac
                * np.tanh(self.dsc * norm)
                / np.where(norm > 1e-10, norm, 1e-10)
                / self.epsilon
                * self.dk2
                / (4 * np.pi**2)
            )
            VcF[zero_mask] = (
                self.efac * self.dsc / self.epsilon * self.dk2 / (4 * np.pi**2)
            )

        elif type == 2:
            VcF = (
                self.efac
                * (1 - np.exp(-2 * self.dsc * norm))
                / (np.where(norm > 1e-10, norm, 1e-10))
                / self.epsilon
                * self.dk2
                / (4 * np.pi**2)
            )
            # VcF[zero_mask] = (
            #     self.efac * 2 * self.dsc / self.epsilon * self.dk2 / (4 * np.pi**2)
            # )

        VcG = np.copy(VcF[0, 0])
        Gnorm = np.linalg.norm(self.Gmesh, axis=-1)
        Gzero_mask = Gnorm < 1e-10
        VcG[Gzero_mask] = 0.0
        return VcF, VcG

    def _build_hf(self):
        # [v, n, n', k, k', G]
        form_factors = self._build_form_factor()
        conj = np.conj(form_factors)

        # q =  p -k + Q
        kminuskpminusG = (
            self.kmesh[:, np.newaxis, np.newaxis, :]
            - self.kmesh[np.newaxis, :, np.newaxis, :]
            + self.Gmesh[np.newaxis, np.newaxis, :, :]
        )

        # [k, p, Q]
        VcF, VcG = self.screened_coulomb(kminuskpminusG, type=self.coulomb_type)

        # n1 -> m, n2 -> n, n3 -> r, n4 -> s
        self.VH = contract("q, vmrkkq, Vsnppq->kpmnvrsV", VcG, form_factors, conj)

        # D rn C ms
        self.VF = contract("pkq, vmrpkq, Vsnpkq->kpmnvrsV", VcF, form_factors, conj)

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
        ham = self.K + F + H

        # if self.test:
        #     print(
        #         "Hermitian error:", np.max(np.abs(ham - ham.conj().transpose(0, 2, 1)))
        #     )

        return ham

    def getdm(self, eigs, states, filling, deg_tol=1e-8):
        # here we define density matrix as Dij=<ci^dagger cj> = sum_{occupied n} <i|n><n|j>
        # in some paper Dji=<ci^dagger cj>
        # for the defination <A>=Tr(AD^T)

        eigs = np.array(eigs).flatten()

        occupied_num = (filling + 4 * self.Nb // 2) * (self.Nk * self.Nk)
        sorted_indices = np.argsort(eigs)

        occ = np.zeros_like(eigs)

        fermi_energy = eigs[sorted_indices[occupied_num - 1]]

        below = eigs < fermi_energy - deg_tol

        deg = np.abs(eigs - fermi_energy) < deg_tol

        occ[below] = 1.0

        deg_num = np.sum(deg)
        if deg_num > 0:
            occ[deg] = (occupied_num - np.sum(below)) / deg_num

        occ = occ.reshape(self.Nk * self.Nk, self.ntotal)

        # occupied_indices = sorted_indices[0 : int(occupied_num)]

        # occupiedk = occupied_indices // self.ntotal
        # occupiedn = occupied_indices % self.ntotal

        dm = np.zeros(
            (self.Nk * self.Nk, self.ntotal, self.ntotal), dtype=np.complex128
        )

        for k in range(self.Nk * self.Nk):
            # occupied_states_k = states[k][occupiedn[occupiedk == k]]
            dm[k] = contract("m,mi,mj->ij", occ[k], np.conj(states[k]), states[k])

        dm = dm.reshape(self.Nk * self.Nk, self.Nb, 2, 2, self.Nb, 2, 2)

        if not self.spin_coherence:
            dm[:, :, 0, :, :, 1, :] = 0
            dm[:, :, 1, :, :, 0, :] = 0

        if not self.spin_polarization:

            dm_up = dm[:, :, 0, :, :, 0, :]
            dm_down = dm[:, :, 1, :, :, 1, :]

            dm_spin_avg = 0.5 * (dm_up + dm_down)

            dm[:, :, 0, :, :, 0, :] = dm_spin_avg
            dm[:, :, 1, :, :, 1, :] = dm_spin_avg

        if not self.valley_coherence:
            dm[:, :, :, 0, :, :, 1] = 0
            dm[:, :, :, 1, :, :, 0] = 0

        if not self.valley_polarization:

            dm_K = dm[:, :, :, 0, :, :, 0]
            dm_Kp = dm[:, :, :, 1, :, :, 1]

            dm_K_avg = 0.5 * (dm_K + dm_Kp)

            dm[:, :, :, 0, :, :, 0] = dm_K_avg
            dm[:, :, :, 1, :, :, 1] = dm_K_avg

        dm = dm.reshape(self.Nk * self.Nk, self.ntotal, self.ntotal)

        return dm, fermi_energy

    def scf(self, filling):
        # dm = random_density_matrix(
        #     self.Nk * self.Nk,
        #     self.Nb,
        #     filling,
        #     self.spin_coherence,
        #     self.valley_coherence,
        #     seed=20260914,
        # )

        # dm = init_c2ti(self.Nk * self.Nk, sign=1)

        dm = init_fmi_valley(self.Nk * self.Nk, valley=0)

        # dm = init_ivc(self.Nk * self.Nk, valley_direction="x", sign=1)

        step = 0

        while True:

            ham = self.hamhf(dm)

            def f(k):
                eig, vec = np.linalg.eigh(ham[k])
                return eig, vec.T

            results = ENV.map(f, list(range(self.Nk * self.Nk)))
            results = ENV.allgather(results)

            eigs, states = zip(*results)
            # states = []
            # eigs = []
            # for k in range(self.Nk * self.Nk):
            #     eig, vec = np.linalg.eigh(ham[k])
            #     states.append(vec.T)
            #     eigs.append(eig)
            # states = np.array(states)

            dm_new, fermi_energy = self.getdm(eigs, states, filling)

            # oda
            if self.useoda:
                next_ham = self.hamhf(dm_new)
                dm_new = self.oda(ham, next_ham, dm, dm_new)
            # simple mixing
            else:
                dm_new = dm_new * self.mix + (1 - self.mix) * dm

            error = np.max(np.abs(dm_new - dm))
            step += 1

            print("step:", step, "error:", error)
            dm = dm_new
            if error < 1e-6 or step >= self.max_iter:
                break

        return dm, np.array(eigs), np.array(states), fermi_energy

    # optimal damping algorithm (ODA) for mixing
    def oda(self, hamk, nexthamk, dmk, nextdmk):
        dmdiff = nextdmk - dmk
        hamdiff = nexthamk - hamk

        s = 2 * contract("kij,kij->", hamk, dmdiff)
        c = contract("kij,kij->", hamdiff, dmdiff)

        lambda_opt = -s / (2 * c) if c > -s / 2 else 1.0

        return lambda_opt * nextdmk + (1 - lambda_opt) * dmk

    def IVC(self, dm):

        dm = dm.reshape(self.Nk * self.Nk, self.Nb, 2, 2, self.Nb, 2, 2)

        r = np.abs(dm[:, :, :, 0, :, :, 1]) ** 2 + np.abs(dm[:, :, :, 1, :, :, 0]) ** 2

        r = 1 / (self.Nk * self.Nk * 2) * np.sum(r)
        return r

    def total_energy(self, dm):
        ham = self.hamhf(dm)
        energy = contract("kij,kij->", self.K + ham, dm - self.dmref) / 2

        Ne = np.sum(np.diagonal(dm, axis1=1, axis2=2))
        return np.real(energy / Ne)


if __name__ == "__main__":
    ENV.redirect_output()
    hf = HFHamSetter(
        test=False,
        cutoff=5,
        Nk=12,
        spin_coherence=False,
        valley_coherence=True,
        spin_polarization=False,
        valley_polarization=True,
        max_iter=200,
        ref_type=1,
    )
    dm, eigs, states, fermi = hf.scf(filling=0)

    ivc = hf.IVC(dm)
    print("IVC:", ivc)

    total_energy = hf.total_energy(dm)
    print("Total energy:", total_energy)

    # eigs [k,nsv]
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(eigs.shape[1]):
        plt.plot(hf.kpath.kpathnorm, eigs[hf.kpath.matched_indices, i], color="r", lw=2)

    plt.xlim(0, hf.kpath.kpathnorm[-1])
    plt.xticks(
        [hf.kpath.kpathnorm[i] for i in hf.kpath.kidx],
        # ["G", "K", "K'", "G"],
        ["K", "G", "M", "K'"],
    )

    plt.axhline(y=fermi, color="b", lw=2, ls="--", label="Fermi energy")
    for i in hf.kpath.kidx:
        ax.axvline(x=hf.kpath.kpathnorm[i], color="k", linestyle="--")
    plt.yticks((-0.04, -0.02, 0, 0.02, 0.04))
    plt.yticks((-0.03, -0.01, 0.01, 0.03), minor=True)
    ax.tick_params(direction="in", axis="y")
    plt.ylim(-0.05, 0.05)
    plt.ylabel("Energy (eV)")
    plt.show()
    # plt.savefig("eigs.png", dpi=300, bbox_inches="tight")
