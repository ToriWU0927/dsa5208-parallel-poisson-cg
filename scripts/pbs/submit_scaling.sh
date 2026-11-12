#!/usr/bin/env bash
# Submit one scaling experiment for every node count it needs.
#   scripts/pbs/submit_scaling.sh e2 [nodes...]     (default nodes: 1 2 4)
# e5 and e6 only need 1 and 4 nodes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
EXP="${1:?usage: submit_scaling.sh e2|e3|e5|e6 [nodes...]}"; shift
if [[ $# -gt 0 ]]; then NODELIST="$*"
elif [[ $EXP == e5 || $EXP == e6 ]]; then NODELIST="1 4"
else NODELIST="1 2 4"; fi
for n in $NODELIST; do
    # the 1-node job holds the slow small-p runs (p = 1 at 512^3), so it gets more time
    # multi-node configurations take < 2 min each, so a hang is cut off after 5 min
    wt=00:40:00; to=300; [[ $n == 1 ]] && { wt=02:00:00; to=600; }
    qsub -N "${EXP}_n${n}" -v EXP="$EXP",RUN_TIMEOUT="$to" -r n -k oed \
         -l select="${n}":ncpus=72:mpiprocs=72:mem=400gb -l place=scatter:excl \
         -l walltime="$wt" scripts/pbs/vanda_scaling.pbs
done
