import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes


def gen_kmesh(grid_size):
    kx = [
        np.linspace(0, 1, grid_size[i], endpoint=False) for i in range(len(grid_size))
    ]
    kx = np.meshgrid(*kx, indexing="ij")
    kx = np.vstack([k.flatten() for k in kx]).T  # type: ignore
    return kx


def get_rlat(lattice):
    return 2 * np.pi * np.linalg.inv(lattice).T


def get_fbz(rlat):
    # rlat = get_rlat(lattice)
    dim = rlat.shape[0]

    if dim == 1:
        return np.array([[-rlat[0] / 2], [rlat[0] / 2]])

    from scipy.spatial import Voronoi, voronoi_plot_2d

    mesh = np.meshgrid(*[np.arange(-2, 3, 1) for i in range(dim)])
    mesh = np.vstack([m.flatten() for m in mesh]).T

    points = np.dot(mesh, rlat)

    norm = np.linalg.norm(points, axis=1)
    # origin
    idx = np.where(norm < 1e-12)[0][0]
    vor = Voronoi(points)

    # 原点对应的 region
    region_idx = vor.point_region[idx]
    region = vor.regions[region_idx]

    # 如果 region 中有 -1，说明区域无界，通常说明点取太少了
    if -1 in region or len(region) == 0:
        raise ValueError("Voronoi region is unbounded. Increase N.")

    vertices = vor.vertices[region]

    # 按角度排序，便于画多边形
    center = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    angles = np.mod(angles, 2 * np.pi)  # 将角度限制在 [0, 2π] 范围内
    order = np.argsort(angles)
    vertices = vertices[order]
    poly = np.vstack([vertices, vertices[0]])  # 闭合多边形
    return poly


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


def gaussian(x: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    # [eigs, eng_num]
    part_a = 1.0 / (sigma * np.sqrt(2 * np.pi))
    part_b = np.exp(-((x[None, :] - mu[:, None]) ** 2) / (2 * sigma**2))
    return part_a * part_b


def calc_dos(eigs, eng_min, eng_max, sigma):

    e_step = sigma
    eng_num = int((eng_max - eng_min) / e_step)
    eng = np.linspace(eng_min, eng_max, eng_num, endpoint=True)
    dos = np.zeros(eng.shape[0])

    for i in range(len(eigs)):
        dos += np.sum(gaussian(eng, eigs[i], sigma), axis=0)
    dos = dos / np.linalg.norm(dos)

    return eng, dos


def calc_dos_count(eigs, nbins):

    eigs = np.concatenate(eigs, axis=0)

    dos, eng = np.histogram(eigs, bins=nbins, density=True)

    eng = (eng[1:] + eng[:-1]) / 2  # Get the center of the bins

    return eng, dos


def calc_ldos(eigs, projection, eng_min, eng_max, eng_num, sigma):
    # projection {sites:[knum, eignum]}
    # e_step = sigma
    # eng_num = int((eng_max - eng_min) / e_step)
    eng = np.linspace(eng_min, eng_max, eng_num, endpoint=True)

    dos = np.zeros(eng.shape[0])
    # ldos = {sites: np.zeros(eng.shape[0]) for sites in projection[0].keys()}
    ldos = [np.zeros(eng.shape[0]) for sites in projection[0]]
    total_eigs_num = 0
    print(len(eigs))
    for i in range(len(eigs)):
        tmp = gaussian(eng, eigs[i], sigma)
        dos += np.sum(tmp, axis=0)
        total_eigs_num += eigs[i].shape[0]
        for ip, proj in enumerate(projection[i]):
            # for sites, proj in projection[i].items():
            # [eigs, eng_num]*[eigs]
            ldos[ip] += np.sum(tmp * proj[:, np.newaxis], axis=0)

    # coef = np.linalg.norm(dos)
    # coef = len(eigs) * 4
    coef = 1
    dos = dos / coef
    for ip in range(len(ldos)):
        ldos[ip] = ldos[ip] / coef

    return eng, ldos, dos
    # normalize
    # ldos = ldos / np.linalg.norm(ldos, axis=1)[:, np.newaxis]
    # ax.plot(eng, ldos.T, **kwargs)


def save_structured_array(
    filename, structured_array, fmt="%10.6f", delimiter="\t", title=""
):
    """
    保存结构化数组到文本文件

    Parameters:
    -----------
    filename : str
        输出文件名
    structured_array : numpy.ndarray
        结构化数组
    fmt : str
        数值格式
    delimiter : str
        分隔符
    """
    # 创建表头
    import re

    width = int(re.match(r"%(\d+).+", fmt).group(1))  # type: ignore

    header_list = structured_array.dtype.names

    header_list = [f"{name:<{width}}" for name in header_list]  # 使用指定宽度格式化列名

    header = delimiter.join(header_list)

    header = title + " " + header  # 在表头前添加标题

    # 提取所有列数据
    data_matrix = np.column_stack(
        [structured_array[name] for name in structured_array.dtype.names]
    )

    # 保存到文件
    np.savetxt(filename, data_matrix, header=header, fmt=fmt, delimiter=delimiter)


def nufft_interpolate(
    mesh_point: np.ndarray,
    A: np.ndarray,
    sites: np.ndarray,
    lat: np.ndarray,
    sigma: int,
):

    import finufft

    frac = np.dot(sites, np.linalg.inv(lat))
    xj = frac[:, 0] * 2 * np.pi - np.pi
    yj = frac[:, 1] * 2 * np.pi - np.pi

    Kx = sigma
    Ky = sigma
    # 频率范围：[-Kx,Kx], [-Ky,Ky]
    # ms = (Kx, Ky)

    # step1: 非规则点 -> 傅里叶系数
    # type=1: 非均匀点 (xj,yj) -> 周期频谱 (系数矩阵)
    coeffs = finufft.nufft2d1(xj, yj, A.astype(complex), (Kx, Ky), isign=-1) / A.size

    coeffs = coeffs.reshape(Kx, Ky)  # (ky, kx)

    frac_mesh = (np.dot(mesh_point, np.linalg.inv(lat)) % 1) * 2 * np.pi - np.pi

    # avoid annoying warnning
    X = np.ascontiguousarray(frac_mesh[:, 0])
    Y = np.ascontiguousarray(frac_mesh[:, 1])

    # type=2
    Z = finufft.nufft2d2(X, Y, coeffs, isign=1)
    Z = np.abs(Z)
    return Z
