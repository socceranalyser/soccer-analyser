import os

# Our matrices are tiny (~60x60); multithreaded BLAS only adds contention, especially
# when backtests run one process per league. Must be set before numpy is imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
