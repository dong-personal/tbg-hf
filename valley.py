import numpy as np
from solver.cell import Cell
from solver.solver import Solver
from gen_struct import a0
from solver.kdtree import cPKDTree
from gen_struct import commensurate_coef


def valley_operator_per_layer(pos, lat, layer_coef, coef):
    map_dict = {
        (-1, 1): -1,
        (0, 1): 1,
        # (1, 1): 0,
        (-1, 0): 1,
        (1, 0): -1,
        # (-1, -1): 0,
        (0, -1): -1,
        (1, -1): 1,
    }

    tree = cPKDTree(pos, lat, isperiodic=np.array([1, 1]), dir="half")

    # graphene lat
    sublat = np.dot(np.linalg.inv(layer_coef), lat)

    # is A or B
    flag = (np.round(np.dot(pos, np.linalg.inv(sublat)), decimals=10) % 1) * 3

    flag_int = np.around(flag).astype(int)
    error = np.abs(flag - flag_int)
    if np.any(error > 1e-8):
        raise ValueError("Error in AB site")

    if not np.all(flag_int[:, 0] == flag_int[:, 1]):
        raise ValueError("Error in AB site 2")

    site_type = np.unique(flag_int[:, 0])
    if site_type.shape[0] != 2:
        raise ValueError("Error in AB site 3")

    if np.abs(site_type[0] - site_type[1]) == 1:
        site_type = np.sort(site_type)
    else:
        site_type = np.sort(site_type)[::-1]

    print("Site types:", site_type)
    isA = flag_int[:, 0] == site_type[0]
    sigma = np.ones_like(isA, dtype=int)
    # sigma[~isA] = -1

    pairs = tree.query_pairs(a0 * 1.08)

    # NNN pairs
    hopping = {}
    for rn, (pair, dr) in pairs.items():
        NNN = np.where(dr > a0 * 0.9)
        NNN_pairs = pair[NNN]
        diff = pos[NNN_pairs[:, 1]] - pos[NNN_pairs[:, 0]] + np.dot(rn, lat)

        diff = np.dot(diff[:, 0:2], np.linalg.inv(sublat))

        diff_int = np.around(diff).astype(int)

        error = np.abs(diff - diff_int)
        if np.any(error > 1e-8):
            idx = np.any(error > 1e-8, axis=1)
            print("Error in valley operator:", diff[idx], diff_int[idx], error[idx])
            raise ValueError("Error in valley operator")

        eta = np.array([map_dict[tuple(d)] for d in diff_int])
        hopping[rn] = (
            NNN_pairs,
            eta * sigma[NNN_pairs[:, 0]] * coef * 1j / 3 / np.sqrt(3),
        )
    print("Total pairs:", sum([len(p[0]) for p in hopping.values()]))
    return hopping


def valley_operator(cell: Cell, i):
    orb_pos = np.dot(cell.pos, cell.lat)
    coef_top, coef_bot = commensurate_coef(i)

    # from top to bot
    z = np.sort(np.unique(np.round(orb_pos[:, 2], decimals=6)))[::-1]

    zdif = np.abs(orb_pos[:, 2][:, np.newaxis] - z[np.newaxis, :])

    layer_idx = np.where(zdif < 1e-5)[1]

    unique_layers = np.arange(len(z))

    hopping = {}
    for i in unique_layers:
        index = layer_idx == i
        pos_layer = orb_pos[index][:, 0:2]
        lat = cell.lat[0:2, 0:2]

        if len(unique_layers) == 5:
            if i < 2:
                layer_coef = coef_top
            else:
                layer_coef = coef_bot
            coef = [1, 1, 1, 1, 1]
        elif len(unique_layers) == 2:
            if i == 0:
                layer_coef = coef_top
            else:
                layer_coef = coef_bot
            coef = [1, 1]

        elif len(unique_layers) == 1:
            layer_coef = coef_top
            coef = [1]

        hopping_layer = valley_operator_per_layer(pos_layer, lat, layer_coef, coef[i])
        for rn, hop in hopping_layer.items():
            idx = hop[0]
            val = hop[1]
            # global index
            global_idx = np.zeros_like(idx)
            global_idx[:, 0] = np.where(index)[0][idx[:, 0]]
            global_idx[:, 1] = np.where(index)[0][idx[:, 1]]

            new_key = (rn[0], rn[1], 0)
            if new_key in hopping:
                hopping[new_key] = (
                    np.vstack([hopping[new_key][0], global_idx]),
                    np.hstack([hopping[new_key][1], val]),
                )
            else:
                hopping[new_key] = (global_idx, val)

    NNN_cell = Cell(cell.lat, cell.pos, hopping=hopping)
    solver = Solver(NNN_cell)
    ham_setter = solver.ham_setter()
    return ham_setter
