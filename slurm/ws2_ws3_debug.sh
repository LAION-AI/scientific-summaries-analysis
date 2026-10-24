#!/bin/bash
# Submit the whole WS2 + WS3 run as a chain of SLURM jobs (afterok dependencies).
#   bash slurm/ws2_ws3_debug.sh                          # debug: 100 papers / 100 items per benchmark
#   SKIP="questions pilot" bash slurm/ws2_ws3_debug.sh   # reuse stages whose outputs are still valid
#   PROFILE=full GPU_GRES=gpu:3g.40gb:1 LLM_QUANT=null LLM_LEN=null CTX_TOKENS=16000 bash slurm/ws2_ws3_debug.sh
# Needs WS1 `features` + `quality` of the same profile.
# Login node first: `alexandria fetch-benchmarks` (benchmark files, PubMedQA PMID metadata, SciFact corpus).
# GPU stages run on 20 GB MIG slices of grete:preemptible with fp8 weights and a 14k context (recorded in every
# manifest as llm_overrides); set GPU_GRES / LLM_* to change that, e.g. GPU_GRES=gpu:3g.40gb:1 LLM_QUANT=null.
set -euo pipefail
cd "$(dirname "$0")/.."
ALEX=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/envs/Alex/bin/alexandria
GPU_GRES=${GPU_GRES:-gpu:2g.20gb:1}
LLM_QUANT=${LLM_QUANT:-fp8}
LLM_LEN=${LLM_LEN:-14336}           # OLMo-3-7B (no GQA) fits ~14.5k tokens of KV cache on a 20 GB slice
PROFILE=${PROFILE:-debug}
CTX_TOKENS=${CTX_TOKENS:-12000}     # WS2 context cap (fits LLM_LEN with the prompt); full runs: 16000
SKIP=" ${SKIP:-} "
P=(--profile "$PROFILE"
   --set site.slurm.cpu.partition=grete:preemptible --set site.slurm.gpu.partition=grete:preemptible
   --set site.slurm.debug.partition=grete:preemptible --set "site.slurm.gpu.gres=${GPU_GRES}"
   --set site.slurm.gpu.cpus=8 --set site.slurm.gpu.mem=64G --set site.slurm.gpu.debug_time=06:00:00
   --set "site.llm.quantization=${LLM_QUANT}" --set "site.llm.max_model_len=${LLM_LEN}"
   --set "ws2.contexts.max_context_tokens=${CTX_TOKENS}")
# benchmarks builds the release-wide id lookup once (gold-paper linking; cached afterwards): more CPUs, memory, time
BIG=(--set site.slurm.debug.cpus=16 --set site.slurm.debug.mem=64G --set site.slurm.debug.debug_time=04:00:00)
FORCE=${FORCE:---force}
jid() { grep -oE 'job [0-9]+' | tail -1 | grep -oE '[0-9]+'; }
# sub <stage-label> <ws> <stage...> [opts]: submit unless skipped; prints the job id ("" when skipped)
sub() { local label=$1; shift
        if [[ "$SKIP" == *" $label "* ]]; then echo ""; else $ALEX submit "$@" | jid; fi; }
after() { for j in "$@"; do [ -n "$j" ] && printf -- '--after %s ' "$j"; done; }

PAPERS=$(sub papers ws2 papers "${P[@]}" $FORCE);                                                  echo "ws2 papers      ${PAPERS:-skipped}"
QUEST=$(sub questions ws2 questions "${P[@]}" $FORCE $(after "$PAPERS"));                           echo "ws2 questions   ${QUEST:-skipped}"
KU=$(sub ku ws2 ku "${P[@]}" $FORCE $(after "$PAPERS"));                                            echo "ws2 ku          ${KU:-skipped} (one KU per paper, KU-Fixes)"
PREP=$(sub contexts ws2 contexts "${P[@]}" $FORCE $(after "$QUEST" "$KU"));                         echo "ws2 contexts    ${PREP:-skipped}"
ANS=$(sub answer ws2 answer "${P[@]}" $FORCE --array 4 $(after "$PREP"));                           echo "ws2 answer      ${ANS:-skipped} (array: one model per task)"
GR2=$(sub grade2 ws2 grade "${P[@]}" $FORCE $(after "$ANS"));                                       echo "ws2 grade       ${GR2:-skipped}"
RES2=$(sub report2 ws2 results report "${P[@]}" $FORCE --one-job $(after "$GR2"));                  echo "ws2 results+pdf ${RES2:-skipped}"
BENCH=$(sub benchmarks ws3 benchmarks "${P[@]}" "${BIG[@]}" $FORCE $(after "$QUEST" "$PAPERS"));    echo "ws3 benchmarks  ${BENCH:-skipped} (+ gold-paper linking)"
INDEX=$(sub index ws3 index "${P[@]}" $FORCE $(after "$BENCH"));                                    echo "ws3 index       ${INDEX:-skipped} (corpus + gold papers)"
PILOT=$(sub pilot ws3 pilot "${P[@]}" $FORCE --array 4 $(after "$BENCH"));                          echo "ws3 pilot       ${PILOT:-skipped}"
AG=$(sub agent ws3 agent "${P[@]}" $FORCE --array 4 $(after "$INDEX" "$BENCH" "$PREP"));            echo "ws3 agent       ${AG:-skipped} (array)"
GR3=$(sub grade3 ws3 grade "${P[@]}" $FORCE $(after "$AG"));                                        echo "ws3 grade       ${GR3:-skipped}"
RES3=$(sub report3 ws3 results report "${P[@]}" $FORCE --one-job $(after "$GR3" "$PILOT"));         echo "ws3 results+pdf ${RES3:-skipped}"
