#!/bin/bash
# Submit the WS4 chain (training-signal study) as SLURM jobs with afterok dependencies.
#   bash slurm/ws4_debug.sh                         # debug: 0.5B model, 24 + 12 papers, 1 seed, 30 steps, lm-eval --limit 50
#   SKIP="papers probe" bash slurm/ws4_debug.sh     # reuse stages whose outputs are still valid
#   PROFILE=full TRAIN_GRES=gpu:3g.40gb:1 TRAIN_ARRAY=16 EVAL_ARRAY=16 bash slurm/ws4_debug.sh
# Login node first: `alexandria fetch-ws4`. Needs WS1 `features` (+ `quality`) of the same profile; `train-tools`
# also needs the WS3 index.
set -euo pipefail
cd "$(dirname "$0")/.."
ALEX=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/envs/Alex/bin/alexandria
PROFILE=${PROFILE:-debug}
GEN_GRES=${GEN_GRES:-gpu:2g.20gb:1}          # Qwen3-8B generator / judge (fp8, 14k context on 20 GB)
TRAIN_GRES=${TRAIN_GRES:-gpu:2g.20gb:1}      # training + eval of the trained model
TRAIN_ARRAY=${TRAIN_ARRAY:-4}
EVAL_ARRAY=${EVAL_ARRAY:-4}
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

PAPERS=$(sub papers ws4 papers "${P[@]}" $FORCE);                                              echo "ws4 papers       ${PAPERS:-skipped}"
PROBE=$(sub probe ws4 probe "${P[@]}" "${GEN[@]}" $FORCE $(after "$PAPERS"));                   echo "ws4 probe        ${PROBE:-skipped}"
DATA=$(sub data ws4 data "${P[@]}" $FORCE $(after "$PAPERS"));                                  echo "ws4 data         ${DATA:-skipped}"
QA=$(sub qa-data ws4 qa-data "${P[@]}" "${GEN[@]}" $FORCE $(after "$PAPERS"));                  echo "ws4 qa-data      ${QA:-skipped}"
DEC=$(sub decontam ws4 decontam "${P[@]}" $FORCE $(after "$DATA" "$QA" "$PROBE"));              echo "ws4 decontam     ${DEC:-skipped}"
SFT=$(sub train-sft ws4 train-sft "${P[@]}" "${TRN[@]}" $FORCE --array "$TRAIN_ARRAY" $(after "$DEC")); echo "ws4 train-sft    ${SFT:-skipped} (array $TRAIN_ARRAY)"
GRPO=$(sub train-grpo ws4 train-grpo "${P[@]}" "${TRN[@]}" $FORCE --array 3 $(after "$DEC"));  echo "ws4 train-grpo   ${GRPO:-skipped} (array 3)"
TOOLS=$(sub train-tools ws4 train-tools "${P[@]}" "${TRN[@]}" $FORCE $(after "$DEC"));          echo "ws4 train-tools  ${TOOLS:-skipped}"
EVAL=$(sub eval ws4 eval "${P[@]}" "${TRN[@]}" $FORCE --array "$EVAL_ARRAY" $(after "$SFT" "$GRPO" "$TOOLS" "$PROBE")); echo "ws4 eval         ${EVAL:-skipped} (array $EVAL_ARRAY)"
GRADE=$(sub grade ws4 grade "${P[@]}" "${GEN[@]}" $FORCE $(after "$EVAL"));                     echo "ws4 grade        ${GRADE:-skipped}"
RES=$(sub report ws4 results report "${P[@]}" $FORCE --one-job $(after "$GRADE"));              echo "ws4 results+pdf  ${RES:-skipped}"
