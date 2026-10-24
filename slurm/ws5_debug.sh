#!/bin/bash
# Submit the WS5 chain (teacher distillation) as SLURM jobs with afterok dependencies.
#   bash slurm/ws5_debug.sh                        # debug: teacher Qwen3.5-27B (40 GB slice), 0.5B student, 24/4/12 papers
#   SKIP="papers teacher" bash slurm/ws5_debug.sh  # reuse stages whose outputs are still valid
#   PROFILE=full TRAIN_GRES=gpu:3g.40gb:1 bash slurm/ws5_debug.sh
# Optional Claude teacher (provider anthropic): run `alexandria ws5 teacher` on a LOGIN node (internet + key), then
#   SKIP="papers teacher" ... bash slurm/ws5_debug.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ALEX=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/envs/Alex/bin/alexandria
PROFILE=${PROFILE:-debug}
GEN_GRES=${GEN_GRES:-gpu:2g.20gb:1}          # 7-8B generator / judge / answerer (fp8, 14k context on 20 GB)
TRAIN_GRES=${TRAIN_GRES:-gpu:2g.20gb:1}      # student training + generation
TEACH_GRES=${TEACH_GRES:-gpu:3g.40gb:1}      # open-weight teacher Qwen3.5-27B-FP8 (31 GB) needs a 40 GB slice ...
TEACH_OFFLOAD=${TEACH_OFFLOAD:-0}            # ... or a 20 GB slice with TEACH_OFFLOAD=16 GB of weights in CPU memory (slower)
TEACH_SRC=${TEACH_SRC:-32000}                # paper tokens given to the teacher
TRAIN_ARRAY=${TRAIN_ARRAY:-3}
SKIP=" ${SKIP:-} "
P=(--profile "$PROFILE"
   --set site.slurm.cpu.partition=grete:preemptible --set site.slurm.gpu.partition=grete:preemptible
   --set site.slurm.debug.partition=grete:preemptible --set site.slurm.gpu.cpus=8 --set site.slurm.gpu.mem=64G
   --set site.slurm.gpu.debug_time=06:00:00)
GEN=(--set "site.slurm.gpu.gres=${GEN_GRES}" --set site.llm.quantization=fp8 --set site.llm.max_model_len=14336)
TRN=(--set "site.slurm.gpu.gres=${TRAIN_GRES}")
FORCE=${FORCE:---force}
jid() { grep -oE 'job [0-9]+' | tail -1 | grep -oE '[0-9]+'; }
sub() { local label=$1; shift
        if [[ "$SKIP" == *" $label "* ]]; then echo ""; else $ALEX submit "$@" | jid; fi; }
after() { for j in "$@"; do [ -n "$j" ] && printf -- '--after %s ' "$j"; done; }

PAPERS=$(sub papers ws5 papers "${P[@]}" $FORCE);                                                echo "ws5 papers    ${PAPERS:-skipped}"
TEACH=$(sub teacher ws5 teacher "${P[@]}" --set "site.slurm.gpu.gres=${TEACH_GRES}" --set site.slurm.gpu.mem=96G --set "ws5.teacher.cpu_offload_gb=${TEACH_OFFLOAD}" --set "ws5.teacher.source_tokens=${TEACH_SRC}" $FORCE $(after "$PAPERS"));                 echo "ws5 teacher   ${TEACH:-skipped}"
VAL=$(sub validate ws5 validate "${P[@]}" $FORCE $(after "$TEACH"));                              echo "ws5 validate  ${VAL:-skipped}"
TRAIN=$(sub train ws5 train "${P[@]}" "${TRN[@]}" $FORCE --array "$TRAIN_ARRAY" $(after "$VAL")); echo "ws5 train     ${TRAIN:-skipped} (array $TRAIN_ARRAY)"
GENR=$(sub generate ws5 generate "${P[@]}" "${TRN[@]}" $FORCE --array "$TRAIN_ARRAY" $(after "$TRAIN" "$TEACH")); echo "ws5 generate  ${GENR:-skipped} (array $TRAIN_ARRAY)"
MCQ=$(sub mcq ws5 mcq "${P[@]}" "${GEN[@]}" $FORCE $(after "$PAPERS"));                           echo "ws5 mcq       ${MCQ:-skipped}"
JUDGE=$(sub judge ws5 judge "${P[@]}" "${GEN[@]}" $FORCE $(after "$GENR"));                       echo "ws5 judge     ${JUDGE:-skipped}"
ANS=$(sub answer ws5 answer "${P[@]}" "${GEN[@]}" $FORCE $(after "$GENR" "$MCQ"));                echo "ws5 answer    ${ANS:-skipped}"
RES=$(sub report ws5 results report "${P[@]}" $FORCE --one-job $(after "$JUDGE" "$ANS"));         echo "ws5 results+pdf ${RES:-skipped}"
