import numpy as np

# [1] T. Fukui, Y. Hatsugai, and H. Suzuki, Chern numbers in discretized brillouin zone: Efficient method of computing (spin) hall conductances, J. Phys. Soc. Jpn. 74, 1674 (2005).


def chern_number(eigvecs):
    Nkx, Nky, Nband, Norb = eigvecs.shape

    kxindex = (np.arange(Nkx) + 1) % Nkx

    kyindex = (np.arange(Nky) + 1) % Nky

    # link variables [Nkx, Nky, Nband]
    Ux = np.einsum("xyno,xyno->xyn", np.conj(eigvecs), eigvecs[kxindex, :, :, :])
    Ux /= np.abs(Ux) + 1e-15
    Uy = np.einsum("xyno,xyno->xyn", np.conj(eigvecs), eigvecs[:, kyindex, :, :])
    Uy /= np.abs(Uy) + 1e-15

    # Wilson loop

    W = Ux * Uy[kxindex, :, :] * np.power(Ux[:, kyindex, :], -1) * np.power(Uy, -1)

    F = np.log(W)

    chern_number = np.sum(F, axis=(0, 1)) / (2.0j * np.pi)

    # [norb]
    return np.real(chern_number)


def chern_number_noperiodic(eigvecs):
    Nkx, Nky, Nband, Norb = eigvecs.shape
    Nkx -= 2
    Nky -= 2

    kxindex = np.arange(Nkx + 1)

    kyindex = np.arange(Nky + 1)

    # link variables [Nkx, Nky, Nband]
    Ux = np.einsum(
        "xyno,xyno->xyn",
        np.conj(eigvecs[0 : Nkx + 1, 0 : Nky + 1, :, :]),
        eigvecs[1 : Nkx + 2, 0 : Nky + 1, :, :],
    )
    Ux /= np.abs(Ux) + 1e-15
    Uy = np.einsum(
        "xyno,xyno->xyn",
        np.conj(eigvecs[0 : Nkx + 1, 0 : Nky + 1, :, :]),
        eigvecs[0 : Nkx + 1, 1 : Nky + 2, :, :],
    )
    Uy /= np.abs(Uy) + 1e-15

    # Wilson loop

    W = (
        Ux[0:Nkx, 0:Nky, :]
        * Uy[1 : Nkx + 1, 0:Nky, :]
        * np.power(Ux[0:Nkx, 1 : Nky + 1, :], -1)
        * np.power(Uy[0:Nkx, 0:Nky, :], -1)
    )

    F = np.log(W)

    chern_number = np.sum(F, axis=(0, 1)) / (2.0j * np.pi)
    return np.real(chern_number)


def chern_number_noperiodic_nonabelian(eigvecs):
    Nkx, Nky, Nband, Norb = eigvecs.shape
    Nkx -= 2
    Nky -= 2

    kxindex = np.arange(Nkx + 1)

    kyindex = np.arange(Nky + 1)

    # link variables [Nkx, Nky, Nband]
    Ux = np.einsum(
        "xymo,xyno->xymn",
        np.conj(eigvecs[0 : Nkx + 1, 0 : Nky + 1, :, :]),
        eigvecs[1 : Nkx + 2, 0 : Nky + 1, :, :],
    )
    Uy = np.einsum(
        "xymo,xyno->xymn",
        np.conj(eigvecs[0 : Nkx + 1, 0 : Nky + 1, :, :]),
        eigvecs[0 : Nkx + 1, 1 : Nky + 2, :, :],
    )
    Ux_ky_plus = np.take(Ux, kyindex, axis=1)  # (Nkx, Nky, Nocc, Nocc), k -> k+ey
    Uy_kx_plus = np.take(Uy, kxindex, axis=0)  # (Nkx, Nky, Nocc, Nocc), k -> k+ex

    Ux_ky_inv = np.linalg.inv(Ux_ky_plus)
    Uy_inv = np.linalg.inv(Uy)

    # batched 矩阵乘法（最后两维是矩阵，前两维是 k 网格）
    W = Ux @ Uy_kx_plus @ Ux_ky_inv @ Uy_inv  # (Nkx, Nky, Nocc, Nocc)

    # 6. 取 det W 的 log，得到离散的非阿贝尔 Berry 曲率
    detW = np.linalg.det(W)  # (Nkx, Nky)
    F = np.log(detW)  # log det W

    # 7. 求和得到 Chern 数
    C = np.sum(F) / (2.0j * np.pi)
    C = np.real(C)

    return C


def chern_number_nonabelian(eigvecs, do_unitary_proj=True):
    Nkx, Nky, Nband, Norb = eigvecs.shape

    # 2. 构造 k+ex, k+ey 的索引（周期边界条件）
    kxindex = (np.arange(Nkx) + 1) % Nkx
    kyindex = (np.arange(Nky) + 1) % Nky

    # link variables [Nkx, Nky, Nband]
    Ux = np.einsum("xymo,xyno->xymn", np.conj(eigvecs), eigvecs[kxindex, :, :, :])
    Uy = np.einsum("xymo,xyno->xymn", np.conj(eigvecs), eigvecs[:, kyindex, :, :])

    # 4. 对 Ux, Uy 做 unitary projection（可选，但推荐）
    if do_unitary_proj:
        # SVD 极分解：U = V Σ W^†, 取 V W^† 为最接近的酉矩阵
        Ux_u, _, Ux_vh = np.linalg.svd(Ux)
        Uy_u, _, Uy_vh = np.linalg.svd(Uy)
        Ux = Ux_u @ Ux_vh
        Uy = Uy_u @ Uy_vh

    # 5. 在每个 plaquette 上构造 Wilson loop：
    #    W(k) = Ux(k) Uy(k+ex) Ux(k+ey)^{-1} Uy(k)^{-1}
    Ux_ky_plus = np.take(Ux, kyindex, axis=1)  # (Nkx, Nky, Nocc, Nocc), k -> k+ey
    Uy_kx_plus = np.take(Uy, kxindex, axis=0)  # (Nkx, Nky, Nocc, Nocc), k -> k+ex

    Ux_ky_inv = np.linalg.inv(Ux_ky_plus)
    Uy_inv = np.linalg.inv(Uy)

    # batched 矩阵乘法（最后两维是矩阵，前两维是 k 网格）
    W = Ux @ Uy_kx_plus @ Ux_ky_inv @ Uy_inv  # (Nkx, Nky, Nocc, Nocc)

    # 6. 取 det W 的 log，得到离散的非阿贝尔 Berry 曲率
    detW = np.linalg.det(W)  # (Nkx, Nky)
    F = np.log(detW)  # log det W

    # 7. 求和得到 Chern 数
    C = np.sum(F) / (2.0j * np.pi)
    C = np.real(C)

    return C


if __name__ == "__main__":

    def ham(k):
        from numpy import cos, sin

        kx, ky = k
        t1 = 1.0
        t2 = 1.0
        t3 = 0.5
        m = -1.0
        matrix = np.zeros((2, 2), dtype=complex)
        matrix[0, 1] = 2 * t1 * cos(kx) - 1j * 2 * t1 * cos(ky)
        matrix[1, 0] = 2 * t1 * cos(kx) + 1j * 2 * t1 * cos(ky)
        matrix[0, 0] = m + 2 * t3 * sin(kx) + 2 * t3 * sin(ky) + 2 * t2 * cos(kx + ky)
        matrix[1, 1] = -(
            m + 2 * t3 * sin(kx) + 2 * t3 * sin(ky) + 2 * t2 * cos(kx + ky)
        )
        return matrix

    Nk = 30

    kx = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    ky = np.linspace(-np.pi, np.pi, Nk - 10, endpoint=False)
    kpoints = np.meshgrid(kx, ky, indexing="ij")
    kpoints = np.vstack([k.flatten() for k in kpoints]).T  # type: ignore

    eigvecs = []
    for i in range(len(kpoints)):
        h = ham(kpoints[i])
        eigvals, eigvec = np.linalg.eigh(h)

        eigvecs.append(eigvec.T)

    eigvecs = np.array(eigvecs).reshape(Nk, Nk - 10, 2, 2)
    chern_number = chern_number(eigvecs)
    print("Chern number:", chern_number)
    chern_number_na = chern_number_nonabelian(eigvecs)
    print("Chern number (non-Abelian):", chern_number_na)
