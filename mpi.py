from mpi4py import MPI
import sys
import io


class MPIEnv:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 防止重复初始化
        if self._initialized:
            return

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.process_num = self.comm.Get_size()

        self.stdout = None
        self.stderr = None
        # self.redirect_output()

        MPIEnv._initialized = True
        # print(f"Process {self.rank}/{self.process_num} initialized.")
        from mpi4py import get_config

        # print(get_config())

    def barrier(self):
        self.comm.Barrier()

    def kill(self):
        self.comm.Abort(99)

    def redirect_output(self):
        if self.rank != 0:
            self.stdout = io.StringIO()
            self.stderr = io.StringIO()
            sys.stdout = self.stdout
            sys.stderr = self.stderr

    def map(self, func, parameter_list):
        if self.process_num == 1:
            return [func(p) for p in parameter_list]

        n = len(parameter_list)

        n_per_process = n // self.process_num

        leftover = n % self.process_num

        if self.rank < leftover:
            this_parameters = parameter_list[
                self.rank * (n_per_process + 1) : (self.rank + 1) * (n_per_process + 1)
            ]
        else:
            this_parameters = parameter_list[
                self.rank * n_per_process
                + leftover : (self.rank + 1) * n_per_process
                + leftover
            ]

        n_per_process = len(this_parameters)

        result = []
        counter = 0
        try:
            for p in this_parameters:
                r = func(p)
                result.append(r)
                counter += 1
        except Exception as e:
            print(f"Error in process {self.rank} with {counter}th parameter")
            import traceback

            traceback.print_exc()

            self.kill()

        return result

    def gather(self, p):
        # self.__gather.append(p)
        result = self.comm.gather(p, root=0)
        if self.rank == 0:
            return [item for sublist in result for item in sublist]  # type: ignore
        else:
            return None

    def allgather(self, p):
        result = self.comm.allgather(p)
        return [item for sublist in result for item in sublist]

    def exit(self, remaining=0):
        self.barrier()  # Ensure all processes reach this point before exiting
        if self.rank != remaining:
            import sys

            sys.exit(0)  # 正常退出而不是


ENV = MPIEnv()
