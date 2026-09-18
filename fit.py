import numpy as np
import os
import matplotlib.pyplot as plt


def rotation_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def included_angle(vector1, vector2):
    return np.arccos(
        np.dot(vector1, vector2) / np.linalg.norm(vector1) / np.linalg.norm(vector2)
    )


# Uniaxial strain matrix that transform vector in reciprocal space
def uniaxial_strain_matrix_reciprocal(strain, possion_ratio):
    return np.array([[1 / (1 + strain), 0], [0, 1 / (1 - possion_ratio * strain)]])


def uniaxial_strain_matrix_real(strain, possion_ratio):
    return np.array([[1 + strain, 0], [0, 1 - possion_ratio * strain]])


def transform_matrix_biaxial(strain_matrix, theta_s, theta_t, biaxial_strain):
    R_s = rotation_matrix(theta_s)
    R_t = rotation_matrix(theta_t)
    return np.dot(
        R_t,
        (1 + biaxial_strain)
        * np.dot(
            np.dot(rotation_matrix(theta_s), strain_matrix), rotation_matrix(-theta_s)
        ),
    )


def reciprocal(lattice):
    return 2 * np.pi * np.linalg.inv(lattice).T


a0 = 0.246  # nm
k = 4 * np.pi / a0 / np.sqrt(3)
zspace = 0.3349
t = -2.8
real_lattice = np.array([[a0, 0], [a0 / 2, a0 * np.sqrt(3) / 2]])
reciprocal_lattice = reciprocal(real_lattice)

k1 = reciprocal_lattice[0]
k2 = reciprocal_lattice[1]

pson = 1


def S_mat(theta_s, epsilon):
    delta = pson
    E = np.array(
        [
            [
                1 / (1 + epsilon),
                0,
            ],
            [0, 1 / (1 - delta * epsilon)],
        ]
    )
    return np.dot(rotation_matrix(theta_s), np.dot(E, rotation_matrix(-theta_s)))


def real_S_mat(theta_s, epsilon):
    delta = pson
    E = np.array(
        [
            [
                1 + epsilon,
                0,
            ],
            [0, 1 - delta * epsilon],
        ]
    )
    return np.dot(rotation_matrix(theta_s), np.dot(E, rotation_matrix(-theta_s)))


def get_pair(epsilon, theta_s, theta_t, biaxial_strain=0.0):

    strain_matrix = S_mat(theta_s, epsilon) / (1 + biaxial_strain)

    rot = rotation_matrix(theta_t)

    ks = np.dot(reciprocal_lattice, strain_matrix.T)

    kp = np.dot(reciprocal_lattice, rot.T)

    k_moire = ks - kp

    ijs = np.dot(ks, np.linalg.inv(k_moire))
    mnp = np.dot(kp, np.linalg.inv(k_moire))

    ijs = np.int64(np.round(np.abs(ijs) + 1e-5) * np.sign(ijs))
    mnp = np.int64(np.round(np.abs(mnp) + 1e-5) * np.sign(mnp))

    return ijs, mnp


def PMMat(ijs, mnp):
    ((i, k), (j, l)), ((m, q), (n, r)) = ijs, mnp

    mat = np.array([[l * m - j * q, l * n - j * r], [-k * m + i * q, -k * n + i * r]])

    coef = i * l - j * k

    pmmat = mat / (coef * 1.0)
    (a, b), (c, d) = pmmat
    # print(pmmat)

    # phi1 = np.arctan(-b * np.sqrt(3) / (2 * a + b))
    # phi2 = np.arctan(-c * np.sqrt(3) / (-c - 2 * d))
    # theta1, theta2
    if a + d == 0:
        thetap = np.pi / 2.0 * np.sign(-a - 2 * b + 2 * c + d)
    else:
        tan1 = (-a - 2 * b + 2 * c + d) / np.sqrt(3) / (a + d)
        thetap = np.arctan(tan1)

    if a + b - d == 0:
        thetam = np.pi / 2.0 * np.sign(a - b - 2 * c - d)
    else:
        tan2 = (a - b - 2 * c - d) / np.sqrt(3) / (a + b - d)
        thetam = np.arctan(tan2)

    theta1 = (thetap + thetam) / 2
    theta2 = (thetap - thetam) / 2

    theta_s = theta2
    theta_t = theta2 + theta1

    # p1 = np.sqrt(a**2 + b**2 + a * b)
    # p2 = np.sqrt(c**2 + d**2 + c * d)
    # A = -2 * (a * c + b * d) - (a * d + b * c)
    # B = (2 * (p1**2 + p2**2) + A) / 3
    # C = np.sqrt((4 * p1**2 * p2**2 - A**2) / 3)

    # piso = np.sqrt(B - np.sqrt(B**2 - C**2))
    # pan = (B + np.sqrt(B**2 - C**2)) / C
    if (a + d == 0) and ((-a - 2 * b + 2 * c + d) != 0):
        ppplus = (-a - 2 * b + 2 * c + d) / np.sqrt(3) / np.sin(theta1 + theta2)
    else:
        ppplus = (a + d) / np.cos(theta1 + theta2)

    if (a + b - d == 0) and ((a - b - 2 * c - d) != 0):
        ppminus = (a - b - 2 * c - d) / np.sqrt(3) / np.sin(theta1 - theta2)
    else:
        ppminus = (a + b - d) / np.cos(theta1 - theta2)

    x = ppminus / ppplus

    pan = 2 / (1 - x) - 1
    piso = ppplus / (pan + 1)

    euni = (pan - 1) / (1 + pan * pson)
    eiso = piso / (1 - pson * euni) - 1

    from functools import reduce

    # g = reduce(np.gcd, mat.flatten())
    # if g == 1:
    return (theta_t, euni, theta_s, eiso)
    # else:
    #     raise ValueError("The gcd of the matrix is not 1")


def get_moire_lattice(theta_t, epsilon, theta_s, biaxial_strain):

    strain_matrix = S_mat(theta_s, epsilon) / (1 + biaxial_strain)
    rot = rotation_matrix(theta_t)

    ks = np.dot(reciprocal_lattice, strain_matrix.T)
    kp = np.dot(reciprocal_lattice, rot.T)

    k_moire = ks - kp

    return reciprocal(k_moire)


def fit(theta_t, epsilon, theta_s, biaxial_strain=0.0):
    ijs, mnp = get_pair(epsilon, theta_s, theta_t, biaxial_strain)
    search_range = list(range(0, 1))
    from itertools import product

    in_param = (theta_t, epsilon, theta_s, biaxial_strain)
    parameter = list(product(search_range, repeat=8))
    result_list = []
    for p in parameter:
        ijst = ijs + np.array(p[:4]).reshape(2, 2)
        mnpt = mnp + np.array(p[4:]).reshape(2, 2)
        if np.linalg.det(ijs) == 0 or np.linalg.det(mnp) == 0:
            continue
        result = PMMat(ijst, mnpt)

        error = np.linalg.norm(
            np.abs(np.array(in_param) - np.array(result)) / np.array(result)
        )
        # if np.isnan(error):
        #     continue
        result_list.append((*result, ijst, mnpt, error))

    result_list = sorted(result_list, key=lambda x: x[1])
    return result_list


if __name__ == "__main__":

    result = fit(np.deg2rad(1.6), 0.002, np.deg2rad(0))[0]
    print(result)
    moire_lat = get_moire_lattice(result[0], result[1], result[2], result[3])

    ijs, mnp = result[4], result[5]

    theta_t_upper = 0
    strain_upper = result[1]
    theta_s_upper = result[2]
    biaxial_strain_upper = result[3]
    possion_ratio_upper = pson
    uniaxial_strain_upper = uniaxial_strain_matrix_real(
        strain_upper, possion_ratio_upper
    )
    transformer_upper = transform_matrix_biaxial(
        uniaxial_strain_upper, theta_s_upper, theta_t_upper, biaxial_strain_upper
    )

    # ------------------------------------------------
    theta_t_lower = result[0]
    strain_lower = 0
    theta_s_lower = 0
    possion_ratio_lower = pson
    uniaxial_strain_lower = uniaxial_strain_matrix_real(
        strain_lower, possion_ratio_lower
    )
    transformer_lower = transform_matrix_biaxial(
        uniaxial_strain_lower, theta_s_lower, theta_t_lower, 0
    )
    lattice_upper = np.dot(real_lattice, transformer_upper.T)
    lattice_lower = np.dot(real_lattice, transformer_lower.T)
    lat1 = np.dot(ijs.T, lattice_upper)
    lat2 = np.dot(mnp.T, lattice_lower)
    print(lat1)
    error = lat1 - lat2
    print(error)
    print(ijs.T, mnp.T)
    # ks = reciprocal(lattice_upper)
    # kp = reciprocal(lattice_lower)
    # km = ks - kp

    # strain_matrix = S_mat(np.deg2rad(theta_s_upper), strain_upper) / (
    #     1 + biaxial_strain_upper
    # )
    # rot = rotation_matrix(np.deg2rad(theta_t_lower))

    # ks = np.dot(reciprocal_lattice, strain_matrix.T)
    # kp = np.dot(reciprocal_lattice, rot.T)

    # km = ks - kp
    # # print(np.dot(lattice_upper, np.linalg.inv(lattice_lower)))
    # print(np.dot(ks, np.linalg.inv(km)))
    # print(np.dot(kp, np.linalg.inv(km)))
