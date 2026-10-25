#!/bin/bash
# Submit the X1 chain (retrieval evaluation of the indices) as SLURM jobs with afterok dependencies.
#   bash slurm/x1_debug.sh                     # debug: <= 200 queries per benchmark, gold papers + 4k distractors
#   SKIP="benchmarks corpus" bash slurm/x1_debug.sh
#   PROFILE=full bash slurm/x1_debug.sh        # full: every linked query, gold + 200k distractors
# Login node first (once): python -c "from alexandria.config import load_config; from alexandria.x1.benchmarks import fetch; fetch(load_config('x1','debug'))"
set -euo pipefail
cd "$(dirname "$0")/.."
ALEX=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/envs/Alex/bin/alexandria
PROFILE=${PROFILE:-debug}
GPU_GRES=${GPU_GRES:-gpu:2g.20gb:1}
SKIP=" ${SKIP:-} "
P=(--profile "$PROFILE"
   --set site.slurm.cpu.partition=grete:preemptible --set site.slurm.gpu.partition=grete:preemptible
   --set site.slurm.debug.partition=grete:preemptible --set "site.slurm.gpu.gres=${GPU_GRES}"
   --set site.slurm.gpu.cpus=16 --set site.slurm.gpu.mem=96G --set site.slurm.gpu.debug_time=04:00:00)
# CPU stages read the release-wide id lookup and many shards: more CPUs, memory, time
BIG=(--set site.slurm.debug.cpus=16 --set site.slurm.debug.mem=96G --set site.slurm.debug.debug_time=04:00:00)
FORCE=${FORCE:---force}
jid() { grep -oE 'job [0-9]+' | tail -1 | grep -oE '[0-9]+'; }
sub() { local label=$1; shift
        if [[ "$SKIP" == *" $label "* ]]; then echo ""; else $ALEX submit "$@" | jid; fi; }
after() { for j in "$@"; do [ -n "$j" ] && printf -- '--after %s ' "$j"; done; }

BENCH=$(sub benchmarks x1 benchmarks "${P[@]}" "${BIG[@]}" $FORCE);                     echo "x1 benchmarks    ${BENCH:-skipped}"
CORP=$(sub corpus x1 corpus "${P[@]}" "${BIG[@]}" $FORCE $(after "$BENCH"));              echo "x1 corpus        ${CORP:-skipped}"
INDEX=$(sub index x1 index "${P[@]}" $FORCE $(after "$CORP"));                            echo "x1 index         ${INDEX:-skipped}"
SEARCH=$(sub search x1 search "${P[@]}" $FORCE $(after "$INDEX" "$BENCH"));               echo "x1 search        ${SEARCH:-skipped}"
DIAG=$(sub ws3-diagnosis x1 ws3-diagnosis "${P[@]}" $FORCE);                              echo "x1 ws3-diagnosis ${DIAG:-skipped}"
RES=$(sub report x1 results report "${P[@]}" "${BIG[@]}" $FORCE --one-job $(after "$SEARCH" "$DIAG")); echo "x1 results+pdf   ${RES:-skipped}"
