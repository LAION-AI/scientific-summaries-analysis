#!/bin/bash
# Submit the X6 chain (token & compute efficiency) as SLURM jobs with afterok dependencies.
#   bash slurm/x6_debug.sh                          # debug: WS2 bench (2 models), RAG budgets, qwen2.5-0.5b training
#   SKIP="inference rag" bash slurm/x6_debug.sh     # reuse measurements, rerun the rest
#   PROFILE=full bash slurm/x6_debug.sh
# Needs (same profile): ws2 questions / contexts / grade, ws3 benchmarks + index, x2 questions + index,
# ws4 data / decontam / probe, x1 index + results.
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
# 7-8B models on a 20 GB slice: fp8 weights, 14k context (the trained 0.5B model is probed in bf16)
GEN=(--set site.llm.quantization=fp8 --set site.llm.max_model_len=14336)
FORCE=${FORCE:---force}
jid() { grep -oE 'job [0-9]+' | tail -1 | grep -oE '[0-9]+'; }
sub() { local label=$1; shift
        if [[ "$SKIP" == *" $label "* ]]; then echo ""; else $ALEX submit "$@" | jid; fi; }
after() { for j in "$@"; do [ -n "$j" ] && printf -- '--after %s ' "$j"; done; }

INF=$(sub inference x6 inference "${P[@]}" "${GEN[@]}" $FORCE --array 2);         echo "x6 inference        ${INF:-skipped}"
RAG=$(sub rag x6 rag-budget "${P[@]}" "${GEN[@]}" $FORCE --array 2);              echo "x6 rag-budget       ${RAG:-skipped}"
TRN=$(sub train x6 train-efficiency "${P[@]}" "${GEN[@]}" $FORCE);                 echo "x6 train-efficiency ${TRN:-skipped}"
STO=$(sub storage x6 storage "${P[@]}" $FORCE);                                    echo "x6 storage          ${STO:-skipped}"
RES=$(sub report x6 results report "${P[@]}" $FORCE --one-job $(after "$INF" "$RAG" "$TRN" "$STO")); echo "x6 results+pdf      ${RES:-skipped}"
