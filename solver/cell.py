import numpy as np


class Cell:
    lat: np.ndarray = None  # type: ignore
    pos: np.ndarray = None  # type: ignore
    hopping: dict = None  # type: ignore
    onsite: np.ndarray = None  # type: ignore

    def __init__(self, lat, pos, hopping=None, onsite=None):
        self.lat = lat
        self.pos = pos
        self.hopping = hopping  # type: ignore
        self.onsite = onsite  # type: ignore

    # @staticmethod
    # def check_hopping(hopping):

    @property
    def num_orb(self):
        if self.pos is None:
            raise ValueError("Position is not set.")
        return self.pos.shape[0]

    @property
    def num_hop(self):
        if self.hopping is None:
            raise ValueError("Hopping is not set.")
        num_hop = 0
        for rn, hop in self.hopping.items():
            num_hop += len(hop[0])
        return num_hop


def expand_cell(cell, expand_coef):

    from time import time

    expand_coef = np.array(expand_coef, dtype=int)
    coef = [np.arange(0, c) for c in expand_coef]
    coef = np.meshgrid(*coef)
    coef = [c.flatten() for c in coef]
    index = np.vstack(coef).T

    new_pos = index[:, np.newaxis, :] + cell.pos[np.newaxis, :, :]

    new_pos = new_pos.reshape(-1, cell.pos.shape[1])
    new_pos = new_pos @ np.diag(1 / expand_coef)

    new_lat = np.diag(expand_coef) @ cell.lat

    if cell.hopping is None and cell.onsite is None:
        return Cell(new_lat, new_pos)
    else:

        if isinstance(cell.onsite, np.ndarray):
            new_onsite = np.tile(cell.onsite, expand_coef.prod())
        else:
            new_onsite = cell.onsite

        cell_offset = np.arange(len(index)) * cell.num_orb
        cell_stride = np.zeros_like(expand_coef, dtype=int)
        cell_stride[0] = 1
        for i in range(1, cell_stride.shape[0], 1):
            cell_stride[i] = cell_stride[i - 1] * expand_coef[i - 1]

        total_cell_num = expand_coef.prod()

        hopping_expand = {}
        for rn, hop in cell.hopping.items():

            new_pairs = np.zeros((total_cell_num, *hop[0].shape), dtype=hop[0].dtype)
            new_pairs += hop[0]
            # [rnz, rny, rnx, p_num, 2]
            new_pairs[:, :, 0] += cell_offset[:, np.newaxis]
            new_hopping = np.zeros((total_cell_num, *hop[1].shape), dtype=hop[1].dtype)
            new_hopping += hop[1]

            if np.all(np.array(rn) == 0):
                new_pairs[:, :, 1] += cell_offset[:, np.newaxis]
                hopping_expand[rn] = {
                    rn: (new_pairs.reshape(-1, 2), new_hopping.reshape(-1))
                }
            else:

                new_rn = (np.array(rn, dtype=int) + index) // expand_coef
                offset = (
                    np.dot((np.array(rn) + index) % expand_coef, cell_stride)
                ) * cell.num_orb

                new_pairs[:, :, 1] += offset[:, np.newaxis]

                # unique_rows = np.unique(new_rn, axis=0)
                mask = np.any(
                    new_rn != 0,
                    axis=1,
                )
                unique_rows = np.array(list(set(map(tuple, new_rn[mask]))))
                unique_rows = np.vstack(
                    [np.zeros((1, new_rn.shape[-1]), dtype=int), unique_rows]
                )

                hopping_expand[rn] = {}

                for i in range(len(unique_rows)):

                    mask = np.all(new_rn == unique_rows[i][np.newaxis, :], axis=1)

                    hopping_expand[rn][tuple(unique_rows[i])] = (
                        new_pairs[mask].reshape(-1, 2),
                        new_hopping[mask].reshape(-1),
                    )

        keys = [value.keys() for value in hopping_expand.values()]
        keys = set().union(*keys)

        new_hopping_tmp = {
            key: [
                v
                for hop_dict in hopping_expand.values()
                for k, v in hop_dict.items()
                if k == key
            ]
            for key in keys
        }

        new_hopping = {
            key: (
                np.concatenate([v[0] for v in value]),
                np.concatenate([v[1] for v in value]),
            )
            for key, value in new_hopping_tmp.items()
        }
        return Cell(new_lat, new_pos, hopping=new_hopping, onsite=new_onsite)
