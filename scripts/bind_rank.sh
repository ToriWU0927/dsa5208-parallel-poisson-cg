#!/usr/bin/env bash
# Bind one MPI rank to OMP_NUM_THREADS consecutive cores (hwloc logical core
# numbering, i.e. packed socket by socket) and exec the program. Used with
#   mpirun --map-by ppr:R:node --bind-to none scripts/bind_rank.sh PROGRAM ...
# when Open MPI's own --map-by ...:PE=t binding cannot be used across nodes.
t=${OMP_NUM_THREADS:-1}
lr=${OMPI_COMM_WORLD_LOCAL_RANK:?not started by Open MPI mpirun}
first=$(( lr * t )); last=$(( first + t - 1 ))
exec hwloc-bind --cpubind "core:${first}-${last}" -- "$@"
