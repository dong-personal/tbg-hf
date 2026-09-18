import numpy as np
from scipy.sparse import coo_matrix

from types import FunctionType
from time import time

from .cell import Cell
from .mpi import MPIEnv

# from make_cell import *


def tuple_pack(func):
    def wrapper(t: tuple):
        return func(*t)

    return wrapper


def timelog(func):
    call_times = 0

    def wrapper(*args, **kwargs):
        nonlocal call_times

        print(f"Call {call_times} start")
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Call {call_times} took {end - start:.4f} seconds")
        call_times += 1
        return result

    return wrapper


class Solver:
    # fractional coordinates
    def __init__(self, cell: Cell) -> None:

        # sample
        self.cell = cell

    def ham_setter(self):
        hopping = np.concatenate([hop[1] for rn, hop in self.cell.hopping.items()])

        pos = self.cell.pos
        dr = []
        for rn, hop in self.cell.hopping.items():
            idx = hop[0]
            if rn == (0, 0, 0):
                dr.append(pos[idx[:, 1]] - pos[idx[:, 0]])
            else:
                tmp = pos[idx[:, 1]] - pos[idx[:, 0]] + np.array(rn)
                dr.append(tmp)

        dr = np.concatenate(dr)
        idx = np.concatenate(
            [hop[0][:, 0] for rn, hop in self.cell.hopping.items()], dtype=np.int32
        )
        jdx = np.concatenate(
            [hop[0][:, 1] for rn, hop in self.cell.hopping.items()], dtype=np.int32
        )

        if self.cell.onsite is None:
            onsite = np.zeros(pos.shape[0], dtype=np.complex128)
        elif not isinstance(self.cell.onsite, np.ndarray):
            onsite = np.zeros(pos.shape[0], dtype=np.complex128) + self.cell.onsite
        else:
            onsite = self.cell.onsite

        def set_ham_csr(kpoints: np.ndarray):
            phase = 2 * np.pi * np.dot(kpoints, dr.T)
            factor = np.cos(phase) + 1j * np.sin(phase)
            ham_half = factor * hopping
            orb_num = pos.shape[0]

            ham_coo_diag = coo_matrix(
                (
                    onsite,
                    (np.arange(orb_num), np.arange(orb_num)),  # type: ignore
                ),
                shape=(orb_num, orb_num),
            )

            ham_coo_half = coo_matrix(
                (ham_half, (idx, jdx)),
                shape=(orb_num, orb_num),
            )

            ham_coo = ham_coo_half + ham_coo_half.getH() + ham_coo_diag
            ham_csr = ham_coo.tocsr()
            ham_csr.eliminate_zeros()
            return ham_csr

        # hopping in cell
        return set_ham_csr

    @staticmethod
    def feast_init(eng_min, eng_max, predict_num, output=False):

        import feast

        fpm = np.zeros(128, dtype=np.int32)
        feast.feastinit(fpm)

        if output:
            fpm[0] = 1

        def eigensolver_feast(csr_matrix):
            indptr = csr_matrix.indptr + 1
            indices = csr_matrix.indices + 1
            data = csr_matrix.data

            a = feast.zfeast_hcsrev(
                indptr,
                indices,
                data,
                "F",
                csr_matrix.shape[0],
                eng_min,
                eng_max,
                predict_num,
                fpm,
            )
            if a[-1] != 0:
                raise RuntimeError(f"feast error {a[-1]}")
            # 0 eig, 1 eigvec, 2 num_eig, -1 flag
            # eigvec=[predict_num, num_orb]
            eignum = a[2]
            print(f"feast found {eignum} eigenvalues")
            return a[0][0:eignum], a[1][0:eignum]

        return eigensolver_feast

    @staticmethod
    def eigensolver_numpy(csr_matrix):
        eigval, eigvec = np.linalg.eigh(csr_matrix.toarray())
        return eigval, eigvec.T

    def solve(
        self,
        kpoints: np.ndarray,
        eigensolver: FunctionType = eigensolver_numpy,
        post_process: FunctionType = lambda x: x,
    ):

        # solver = Solver(self.cell)
        set_ham = self.ham_setter()

        f = lambda k: post_process((*eigensolver(set_ham(k)), k))

        f = timelog(f)

        with MPIEnv() as env:
            result = env.map(f, kpoints)
            result = env.allgather(result)
        return result

    @staticmethod
    @tuple_pack
    def dos_process(eigval, eigvec, k):
        return eigval

    @staticmethod
    @tuple_pack
    def band_process(eigval, eigvec, k):
        return eigval

    @staticmethod
    @tuple_pack
    def default_process(eigval, eigvec, k):
        return eigval, eigvec, k

    @staticmethod
    def ldos_processor(sites):
        @tuple_pack
        def ldos_process(eigval, eigvec, k):
            if isinstance(sites, dict):
                projection = {}
                for site in sites.keys():
                    proj = np.sum(np.abs(eigvec[:, sites[site]]) ** 2, axis=1)
                    projection[site] = proj

                return eigval, projection
            elif isinstance(sites, list):
                projection = []
                for site in sites:
                    proj = np.sum(np.abs(eigvec[:, site]) ** 2, axis=1)
                    projection.append(proj)

                return eigval, projection

        return ldos_process

    @staticmethod
    def op_process(op):
        @tuple_pack
        def process(eigval, eigvec, k):
            return op(eigval, eigvec, k)

        return process
