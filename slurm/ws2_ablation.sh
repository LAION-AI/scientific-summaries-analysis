#!/bin/bash
# WS2 structure ablation as a SLURM chain (afterok): is it the schema or just "a summary" that helps?
#   bash slurm/ws2_ablation.sh                 # debug: the existing 100 WS2 papers / 418 questions, 4 answer models
#   PROFILE=full GPU_GRES=gpu:3g.40gb:1 LLM_QUANT=null LLM_LEN=null CTX_TOKENS=16000 bash slurm/ws2_ablation.sh
# Needs WS2 papers / questions / ku (and the existing answers of the 5 base arms, which are reused: only the new arms
# are answered, into parts/<model>__ablation.parquet). Settings follow slurm/ws2_ws3_debug.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
ALEX=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/envs/Alex/bin/alexandria
PROFILE=${PROFILE:-debug}
GPU_GRES=${GPU_GRES:-gpu:2g.20gb:1}
LLM_QUANT=${LLM_QUANT:-fp8}
LLM_LEN=${LLM_LEN:-14336}
CTX_TOKENS=${CTX_TOKENS:-12000}
ARMS='[ku_local, prose_local, ku_minus_context, ku_minus_methods, ku_minus_results, ku_minus_claims, ku_minus_limitations, ku_minus_takeaways, ku_results_claims, ku_flat]'
SKIP=" ${SKIP:-} "
P=(--profile "$PROFILE"
   --set site.slurm.cpu.partition=grete:preemptible --set site.slurm.gpu.partition=grete:preemptible
   --set site.slurm.debug.partition=grete:preemptible --set "site.slurm.gpu.gres=${GPU_GRES}"
   --set site.slurm.gpu.cpus=8 --set site.slurm.gpu.mem=64G --set site.slurm.gpu.debug_time=06:00:00
   --set "site.llm.quantization=${LLM_QUANT}" --set "site.llm.max_model_len=${LLM_LEN}"
   --set "ws2.contexts.max_context_tokens=${CTX_TOKENS}")
jid() { grep -oE 'job [0-9]+' | tail -1 | grep -oE '[0-9]+'; }
sub() { local label=$1; shift
        if [[ "$SKIP" == *" $label "* ]]; then echo ""; else $ALEX submit "$@" | jid; fi; }
after() { for j in "$@"; do [ -n "$j" ] && printf -- '--after %s ' "$j"; done; }

GEN=$(sub gen ws2 ablation-gen "${P[@]}" --force);                                              echo "ws2 ablation-gen ${GEN:-skipped}"
CTX=$(sub contexts ws2 contexts "${P[@]}" --force $(after "$GEN"));                             echo "ws2 contexts     ${CTX:-skipped}"
ANS=$(sub answer ws2 answer "${P[@]}" --set "ws2.answer.arms=${ARMS}" --set ws2.answer.part_suffix=__ablation \
      --force --array 4 $(after "$CTX"));                                                        echo "ws2 answer       ${ANS:-skipped} (new arms only, array)"
GR=$(sub grade ws2 grade "${P[@]}" --force $(after "$ANS"));                                    echo "ws2 grade        ${GR:-skipped}"
RES=$(sub report ws2 results report "${P[@]}" --force --one-job $(after "$GR"));                echo "ws2 results+pdf  ${RES:-skipped}"
