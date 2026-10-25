#!/bin/bash
# Submit the X5 chain (knowledge conflict) as SLURM jobs with afterok dependencies.
#   bash slurm/x5_debug.sh                         # debug: 150 WS2 + 24 X2 items, qwen2.5-7b + olmo3-7b
#   SKIP="items counterfactual" bash slurm/x5_debug.sh   # reuse stages whose outputs are still valid
# Needs ws2 questions/ku, ws3 index and (optional, post-cutoff items) x2 questions/summarize/index.
set -euo pipefail
cd "$(dirname "$0")/.."
ALEX=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/envs/Alex/bin/alexandria
PROFILE=${PROFILE:-debug}
GEN_GRES=${GEN_GRES:-gpu:2g.20gb:1}          # 7-8B editor / answerers / judge (fp8, 14k context on 20 GB)
MODELS=${MODELS:-2}                          # array size = number of x5.answer.models
SKIP=" ${SKIP:-} "
P=(--profile "$PROFILE"
   --set site.slurm.cpu.partition=grete:preemptible --set site.slurm.gpu.partition=grete:preemptible
   --set site.slurm.debug.partition=grete:preemptible --set site.slurm.gpu.cpus=8 --set site.slurm.gpu.mem=64G
   --set site.slurm.gpu.debug_time=06:00:00)
GEN=(--set "site.slurm.gpu.gres=${GEN_GRES}" --set site.llm.quantization=fp8 --set site.llm.max_model_len=14336)
FORCE=${FORCE:---force}
jid() { grep -oE 'job [0-9]+' | tail -1 | grep -oE '[0-9]+'; }
sub() { local label=$1; shift
        if [[ "$SKIP" == *" $label "* ]]; then echo ""; else $ALEX submit "$@" | jid; fi; }
after() { for j in "$@"; do [ -n "$j" ] && printf -- '--after %s ' "$j"; done; }

ITEMS=$(sub items x5 items "${P[@]}" $FORCE);                                                 echo "x5 items          ${ITEMS:-skipped}"
CF=$(sub counterfactual x5 counterfactual "${P[@]}" "${GEN[@]}" $FORCE $(after "$ITEMS"));   echo "x5 counterfactual ${CF:-skipped}"
ANS=$(sub answer x5 answer "${P[@]}" "${GEN[@]}" $FORCE --array "$MODELS" $(after "$CF"));   echo "x5 answer         ${ANS:-skipped} (array $MODELS)"
GR=$(sub grade x5 grade "${P[@]}" "${GEN[@]}" $FORCE $(after "$ANS"));                        echo "x5 grade          ${GR:-skipped}"
RES=$(sub report x5 results report "${P[@]}" $FORCE --one-job $(after "$GR"));               echo "x5 results+pdf    ${RES:-skipped}"
