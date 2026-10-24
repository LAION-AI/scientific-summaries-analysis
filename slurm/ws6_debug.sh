#!/bin/bash
# Submit the WS6 chain (embeddings as input) as SLURM jobs with afterok dependencies.
#   bash slurm/ws6_debug.sh                        # debug: 0.5B frozen LLM, 48 + 12 projector papers, k in {1, 4}
#   SKIP="papers qa embed" bash slurm/ws6_debug.sh # reuse stages whose outputs are still valid
#   PROFILE=full TRAIN_GRES=gpu:3g.40gb:1 KS=3 bash slurm/ws6_debug.sh
# Needs WS1 `features`, WS2 `ku` + `questions`, WS3 `benchmarks` + `index` of the same profile; gte-modernbert in
# site.hf_home (login node: snapshot_download Alibaba-NLP/gte-modernbert-base at the pinned revision).
set -euo pipefail
cd "$(dirname "$0")/.."
ALEX=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/envs/Alex/bin/alexandria
PROFILE=${PROFILE:-debug}
GEN_GRES=${GEN_GRES:-gpu:2g.20gb:1}          # Qwen3-8B generator / judge (fp8, 14k context on 20 GB)
TRAIN_GRES=${TRAIN_GRES:-gpu:2g.20gb:1}      # embedder, projector training, eval
KS=${KS:-2}                                  # array tasks for train-projector (one per k)
EVAL_ARRAY=${EVAL_ARRAY:-3}
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

PAPERS=$(sub papers ws6 papers "${P[@]}" $FORCE);                                         echo "ws6 papers           ${PAPERS:-skipped}"
QA=$(sub qa ws6 qa "${P[@]}" "${GEN[@]}" $FORCE $(after "$PAPERS"));                        echo "ws6 qa               ${QA:-skipped}"
EMB=$(sub embed ws6 embed "${P[@]}" "${TRN[@]}" $FORCE $(after "$PAPERS"));                 echo "ws6 embed            ${EMB:-skipped}"
TP=$(sub train-projector ws6 train-projector "${P[@]}" "${TRN[@]}" $FORCE --array "$KS" $(after "$QA" "$EMB")); echo "ws6 train-projector  ${TP:-skipped} (array $KS)"
EV=$(sub eval ws6 eval "${P[@]}" "${TRN[@]}" $FORCE --array "$EVAL_ARRAY" $(after "$TP" "$EMB")); echo "ws6 eval             ${EV:-skipped} (array $EVAL_ARRAY)"
GR=$(sub grade ws6 grade "${P[@]}" "${GEN[@]}" $FORCE $(after "$EV"));                      echo "ws6 grade            ${GR:-skipped}"
RES=$(sub report ws6 results report "${P[@]}" $FORCE --one-job $(after "$GR"));             echo "ws6 results+pdf      ${RES:-skipped}"
