import os as _os
# Many ppbench processes run at once (tool episodes, shell-session calls, renders); numpy/BLAS/OpenMP otherwise start one
# thread per core in every process (127 threads each, load ~314 on 128 cores on 2026-09-14). Two threads per process.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_var, _os.environ.get("PPBENCH_THREADS", "2"))
# numpy asks for transparent huge pages on large arrays; with ~200 processes allocating voxel grids this caused ~900 direct
# compaction stalls per second (94% failing) and ~70% system CPU on 2026-09-15. Plain pages for every ppbench process.
_os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
