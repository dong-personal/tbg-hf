from scipy.spatial import cKDTree  # type: ignore
import numpy as np
from multiprocessing import Pool

import time


class cPKDTree:
    def __init__(
        self, data: np.ndarray, box, isperiodic: np.ndarray, dir="half", parallel=1
    ):
        time1 = time.time()
        self.points = data
        self.box = box
        self.dim = data.shape[1]
        self.isperiodic = isperiodic

        self.parallel = parallel
        self.tmp = None

        # self.isinbox()

        self.map = [np.zeros(self.dim, dtype=int)]
        map = []
        for i in range(self.dim):
            if self.isperiodic[i] == 0:
                map.append([0])
            else:
                map.append(list(range(-isperiodic[i], isperiodic[i] + 1)))
        self.map = np.stack(np.meshgrid(*map), axis=-1).reshape(-1, self.dim)
        # print("Number of images:", len(self.map))
        if dir == "half":
            plus = self.map[np.newaxis, :, :] + self.map[:, np.newaxis, :]
            isequal = np.where(np.all(plus == 0, axis=-1))
            self.map = np.delete(self.map, isequal[1][isequal[0] > isequal[1]], axis=0)
            # print("Number of images:", len(self.map))
        data_list = []
        for map in self.map:
            data_list.append(data + np.dot(map, self.box))
        self.othertree = self.parallel_multiprocess_map(cKDTree, data_list)
        for i, tree in enumerate(self.othertree):
            if np.array_equal(self.map[i], np.zeros(self.dim, dtype=int)):
                self.maintree = tree
                break
        print("init time:", time.time() - time1)

    def parallel_multiprocess_map(self, function, parameter):
        with Pool(processes=self.parallel) as p:
            r = p.map(function, parameter)
        return r

    def parallel_multiprocess_starmap(self, function, parameter):
        with Pool(processes=self.parallel) as p:
            r = p.starmap(function, parameter)
        return r

    def query_pairs(self, cutoff):

        time1 = time.time()
        pairs_lists = self.find_pairs(cutoff)
        print("find_pairs time:", time.time() - time1)
        time1 = time.time()

        dr_lists = [
            np.linalg.norm(
                self.points[index_list[:, 1]]
                - self.points[index_list[:, 0]]
                + np.dot(self.map[i], self.box),
                axis=1,
            )
            for i, index_list in enumerate(pairs_lists)
        ]
        print("pairs_with_distance time:", time.time() - time1)
        return {
            tuple(m): (np.array(p), dr_list)
            for m, p, dr_list in zip(self.map, pairs_lists, dr_lists)
        }
        # index_list = np.vstack(pairs_lists)
        # map_list = np.vstack(
        #     [
        #         self.map[i] + np.zeros((len(pairs_lists[i]), self.dim), dtype=int)
        #         for i in range(len(pairs_lists))
        #     ]
        # )
        # dr_list = np.linalg.norm(
        #     self.points[index_list[:, 1]]
        #     - self.points[index_list[:, 0]]
        #     + np.dot(map_list, self.box),
        #     axis=1,
        # )
        # pairs_with_distance = [
        #     (m, pair, dr) for m, pair, dr in zip(map_list, index_list, dr_list)
        # ]

        # return pairs_with_distance

    def _helper(self, i, cutoff):
        if np.array_equal(self.map[i], np.zeros(self.dim, dtype=int)):
            result = np.array(
                self.maintree.query_pairs(cutoff, p=2.0, output_type="ndarray"),
                dtype=int,
            )
        else:
            bp = self.maintree.query_ball_tree(self.othertree[i], cutoff, p=2.0)
            result = self.trim(bp)
        # print("Finished", self.map[i])
        return result

    def find_pairs(self, cutoff):

        if self.parallel > 1:
            parameter = [(i, cutoff) for i in range(len(self.othertree))]
            pairs = self.parallel_multiprocess_starmap(self._helper, parameter)

        else:
            pairs = []
            for i, tree in enumerate(self.othertree):
                if np.array_equal(self.map[i], np.zeros(self.dim, dtype=int)):

                    bp = np.array(
                        (
                            self.maintree.query_pairs(
                                cutoff, p=2.0, output_type="ndarray"
                            )
                        ),
                        dtype=int,
                    )
                    pairs.append(bp)

                else:
                    bp = self.maintree.query_ball_tree(tree, cutoff, p=2.0)
                    pairs.append(self.trim(bp))
                print("Finished", self.map[i])
        return pairs

    def isinbox(self):
        frac = np.dot(self.points, np.linalg.inv(self.box))
        if np.any((frac >= 1) | (frac < 0)):
            raise ValueError("Points are not in the box")

    @staticmethod
    def trim(return_value):
        # print(return_value)
        idx = np.concatenate(
            [
                (i + np.zeros(len(llist), dtype=int))
                for i, llist in enumerate(return_value)
            ]
        )
        jdx = np.concatenate(return_value).astype(int)
        result = np.vstack((idx, jdx)).T
        return result

    def query_ball_tree(self, othertree: cKDTree, r, p=2.0, sign=0):
        pair = {}
        if sign == -1:
            for i, tree in enumerate(self.othertree):
                pair[tuple(self.map[i])] = othertree.query_ball_tree(tree, r, p=p)
            return pair
        else:
            for i, tree in enumerate(self.othertree):
                pair[tuple(self.map[i])] = tree.query_ball_tree(othertree, r, p=p)
            return pair

    def query_ball_point(self, point: np.ndarray, r: float, p=2.0):
        pair = {}
        for i, tree in enumerate(self.othertree):
            pair[tuple(self.map[i])] = tree.query_ball_point(point, r, p)
        return pair


if __name__ == "__main__":
    data = np.random.rand(10000, 2)
    box = np.array([[1, 0], [0, 1]])
    periodic = np.array([1, 1])
    tree = cPKDTree(data, box, periodic, parallel=4)
    pairs = tree.query_pairs(0.07)

    # plt.scatter(data[:, 0], data[:, 1], c="b")
    # for rn, pair, dr in pairs:
    #     p1 = data[pair[0]]
    #     p2 = data[pair[1]] + np.dot(rn, box)
    #     plt.plot(
    #         [p1[0], p2[0]],
    #         [p1[1], p2[1]],
    #         "r-",
    #     )
    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.show()
