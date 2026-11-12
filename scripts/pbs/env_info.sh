# Sourced at the start of every PBS job: records the run environment for the
# "experimental environment" table of the report.
# Printed to the job output; collected manually, never submitted as a log.
echo "=== environment ==="
echo "job:      ${PBS_JOBID:-none}  queue: ${PBS_QUEUE:-none}"
echo "date:     $(date -Is)"
echo "nodes:    $(sort -u "${PBS_NODEFILE:-/dev/null}" 2>/dev/null | tr '\n' ' ')"
echo "cpu:      $(lscpu | sed -n 's/^Model name: *//p')"
echo "cores:    $(nproc) visible on $(hostname)"
echo "compiler: $(g++ --version | head -1)"
echo "mpi:      $(mpirun --version 2>/dev/null | head -1)"
echo "modules:  $(module -t list 2>&1 | tr '\n' ' ')"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "gpus:     $(nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tr '\n' ';')"
fi
echo "==================="
