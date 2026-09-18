import numpy as np

import time
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix

from solver.kdtree import cPKDTree
from solver.cell import Cell

a0 = 0.246  # nm
zspace = 0.3349
# t = -2.8
# lattice = real_lattice


def commensurate_angle(i):
    cos_theta = (3 * i**2 + 3 * i + 0.5) / (3 * i**2 + 3 * i + 1)
    print("theta:", np.rad2deg(np.arccos(cos_theta)))
    return np.arccos(cos_theta)


def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def commensurate_coef(i):
    top = np.array([[i, i + 1], [-(i + 1), 2 * i + 1]])
    bot = np.array([[i + 1, i], [-i, 2 * i + 1]])
    return top, bot


def graphene():

    pos = np.array([[1 / 3.0, 1 / 3.0], [2 / 3.0, 2 / 3.0]])
    lat = a0 * np.array([[1, 0], [0.5, np.sqrt(3) / 2.0]])

    return Cell(lat, pos)


def hetero_layer_range(a0, a1):

    p = a0 + a1

    x_min = np.min([0, a0[0], a1[0], p[0]])
    x_max = np.max([0, a0[0], a1[0], p[0]])
    y_min = np.min([0, a0[1], a1[1], p[1]])
    y_max = np.max([0, a0[1], a1[1], p[1]])
    return np.array(
        [[np.floor(x_min), np.ceil(x_max)], [np.floor(y_min), np.ceil(y_max)]]
    )


def make_hetero_layer(coef, z0):

    xrange, yrange = hetero_layer_range(coef[0], coef[1])
    x = np.arange(xrange[0], xrange[1], dtype=np.int64)
    y = np.arange(yrange[0], yrange[1], dtype=np.int64)
    xgrid, ygrid = np.meshgrid(x, y)
    grid = np.array([xgrid.flatten(), ygrid.flatten()]).T * 3

    Asite = np.array([1, 1], dtype=np.int64)
    Bsite = np.array([2, 2], dtype=np.int64)

    Asite = Asite + grid
    Bsite = Bsite + grid

    [[a, b], [c, d]] = coef
    coef_inv = np.array([[d, -b], [-c, a]])

    A_frac = np.dot(Asite, coef_inv) / ((a * d - b * c) * 3)
    B_frac = np.dot(Bsite, coef_inv) / ((a * d - b * c) * 3)

    Ain = ~((A_frac <= 0) | (A_frac > 1))
    Bin = ~((B_frac <= 0) | (B_frac > 1))

    Ain = np.where(Ain[:, 0] & Ain[:, 1])[0]
    Bin = np.where(Bin[:, 0] & Bin[:, 1])[0]

    A_frac = A_frac[Ain]
    B_frac = B_frac[Bin]

    return A_frac, B_frac


def tbg_pos(moire_lattice, coef_top, coef_bot):

    Asite_top, Bsite_top = make_hetero_layer(coef_top, zspace)
    Asite_bot, Bsite_bot = make_hetero_layer(coef_bot, 0)
    Asite_top_nm = np.dot(Asite_top, moire_lattice)
    Bsite_top_nm = np.dot(Bsite_top, moire_lattice)
    Asite_bot_nm = np.dot(Asite_bot, moire_lattice)
    Bsite_bot_nm = np.dot(Bsite_bot, moire_lattice)
    top_nm = np.vstack((Asite_top_nm, Bsite_top_nm))
    bot_nm = np.vstack((Asite_bot_nm, Bsite_bot_nm))

    return top_nm, bot_nm


def gen_struct(i):

    cell = graphene()

    lat = cell.lat
    theta = commensurate_angle(i)
    top_coef, bot_coef = commensurate_coef(i)
    moire_lat = np.dot(top_coef, lat)
    # print(np.dot(bot_coef, lat.dot(rot(theta).T)) - moire_lat)
    # top_lat = lat
    # bot_lat = lat.dot(rot(theta).T)
    moire_lat_3d = np.zeros((3, 3))
    moire_lat_3d[0:2, 0:2] = moire_lat
    moire_lat_3d[2, 2] = 10

    top_pos, bot_pos = tbg_pos(moire_lat, top_coef, bot_coef)

    top_pos = np.hstack((top_pos, np.ones((top_pos.shape[0], 1)) * zspace + 5))
    bot_pos = np.hstack((bot_pos, np.zeros((bot_pos.shape[0], 1)) + 5))

    pos = (top_pos, bot_pos)

    return pos, moire_lat_3d


def gen_struct_strain(theta_t, epsilon, theta_s, biaxial_strain=0.0):
    from fit import fit, get_moire_lattice

    result = fit(theta_t, epsilon, theta_s, biaxial_strain)[0]
    print(
        f"theta_t: {np.rad2deg(result[0])}, epsilon: {result[1]}, theta_s: {np.rad2deg(result[2])}, biaxial_strain: {result[3]}"
    )

    ijs, mnp = result[4], result[5]

    print(f"ijs: {ijs.T}, mnp: {mnp.T}")

    moire_lat = get_moire_lattice(result[0], result[1], result[2], result[3])
    moire_lat_3d = np.zeros((3, 3))
    moire_lat_3d[0:2, 0:2] = moire_lat
    moire_lat_3d[2, 2] = 10

    top_pos, bot_pos = tbg_pos(moire_lat, ijs.T, mnp.T)

    top_pos = np.hstack((top_pos, np.ones((top_pos.shape[0], 1)) * zspace + 5))
    bot_pos = np.hstack((bot_pos, np.zeros((bot_pos.shape[0], 1)) + 5))

    pos = (top_pos, bot_pos)

    return pos, moire_lat_3d


if __name__ == "__main__":
    i = 31

    pos, lat = gen_struct(i)

    color = ["red", "blue"]
    fig = plt.figure(figsize=(4, 6))
    for i, (p, c) in enumerate(zip(pos, color)):
        plt.scatter(p[:, 0], p[:, 1], c=c, s=0.5)
    plt.axis("equal")
    plt.axis("off")
    plt.show()
    # plt.tight_layout()
    # plt.savefig("plot/tbg_struct.png", dpi=300, bbox_inches="tight")
