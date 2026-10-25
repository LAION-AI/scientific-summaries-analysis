#!/bin/bash
# Submit the X4 chain (field-level & cross-paper questions over the WS3 index) as SLURM jobs with afterok dependencies.
#   bash slurm/x4_debug.sh                     # debug: ~200 candidates -> ~100-150 verified items, 2 models x 6 arms
#   SKIP="items" bash slurm/x4_debug.sh        # reuse the items, rerun answer/grade/results
#   PROFILE=full bash slurm/x4_debug.sh
# Needs ws3/index of the same profile (bash slurm/ws3_debug.sh).
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
# 7-8B models on a 20 GB slice: fp8 weights, 14k context
GEN=(--set site.llm.quantization=fp8 --set site.llm.max_model_len=14336)
FORCE=${FORCE:---force}
jid() { grep -oE 'job [0-9]+' | tail -1 | grep -oE '[0-9]+'; }
sub() { local label=$1; shift
        if [[ "$SKIP" == *" $label "* ]]; then echo ""; else $ALEX submit "$@" | jid; fi; }
after() { for j in "$@"; do [ -n "$j" ] && printf -- '--after %s ' "$j"; done; }

ITEMS=$(sub items x4 items "${P[@]}" "${GEN[@]}" $FORCE);                                  echo "x4 items        ${ITEMS:-skipped}"
ANS=$(sub answer x4 answer "${P[@]}" "${GEN[@]}" $FORCE --array 2 $(after "$ITEMS"));       echo "x4 answer       ${ANS:-skipped}"
GRADE=$(sub grade x4 grade "${P[@]}" "${GEN[@]}" $FORCE $(after "$ANS"));                   echo "x4 grade        ${GRADE:-skipped}"
RES=$(sub report x4 results report "${P[@]}" $FORCE --one-job $(after "$GRADE"));           echo "x4 results+pdf  ${RES:-skipped}"
