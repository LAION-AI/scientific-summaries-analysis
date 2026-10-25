#!/bin/bash
# Submit the X2 chain (post-cutoff benchmark) as SLURM jobs with afterok dependencies.
# First, on a LOGIN node (internet): alexandria x2 papers --profile debug        (arXiv API + arxiv.org/html)
#   bash slurm/x2_debug.sh                          # debug: 24 papers, Qwen3-8B summariser, gpt-oss-20b questions
#   PROFILE=full SUM_GRES=gpu:3g.40gb:1 bash slurm/x2_debug.sh   # full: Qwen3.5-27B summariser (WS5 teacher)
# Needs the WS3 index of the same profile (the incremental update starts from it).
set -euo pipefail
cd "$(dirname "$0")/.."
ALEX=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/envs/Alex/bin/alexandria
PROFILE=${PROFILE:-debug}
SUM_GRES=${SUM_GRES:-gpu:2g.20gb:1}          # debug summariser Qwen3-8B (fp8); full: 3g.40gb for Qwen3.5-27B-FP8
GEN_GRES=${GEN_GRES:-gpu:2g.20gb:1}
SKIP=" ${SKIP:-} "
P=(--profile "$PROFILE"
   --set site.slurm.cpu.partition=grete:preemptible --set site.slurm.gpu.partition=grete:preemptible
   --set site.slurm.debug.partition=grete:preemptible --set site.slurm.gpu.cpus=8 --set site.slurm.gpu.mem=64G
   --set site.slurm.gpu.debug_time=06:00:00)
FP8=(--set site.llm.quantization=fp8 --set site.llm.max_model_len=14336)   # 7-8B models on 20 GB slices (as WS3)
if [ "$PROFILE" = full ]; then SUMQ=(); else SUMQ=("${FP8[@]}"); fi
FORCE=${FORCE:---force}
jid() { grep -oE 'job [0-9]+' | tail -1 | grep -oE '[0-9]+'; }
sub() { local label=$1; shift
        if [[ "$SKIP" == *" $label "* ]]; then echo ""; else $ALEX submit "$@" | jid; fi; }
after() { for j in "$@"; do [ -n "$j" ] && printf -- '--after %s ' "$j"; done; }

SUM=$(sub summarize x2 summarize "${P[@]}" --set "site.slurm.gpu.gres=${SUM_GRES}" "${SUMQ[@]}" $FORCE);   echo "x2 summarize ${SUM:-skipped}"
QUE=$(sub questions x2 questions "${P[@]}" --set "site.slurm.gpu.gres=${GEN_GRES}" $FORCE);              echo "x2 questions ${QUE:-skipped} (gpt-oss-20b, MXFP4)"
VER=$(sub verify x2 verify "${P[@]}" $FORCE $(after "$QUE"));                                            echo "x2 verify    ${VER:-skipped}"
IDX=$(sub index x2 index "${P[@]}" --set "site.slurm.gpu.gres=${GEN_GRES}" $FORCE $(after "$SUM"));      echo "x2 index     ${IDX:-skipped}"
EVA=$(sub eval x2 eval "${P[@]}" --set "site.slurm.gpu.gres=${GEN_GRES}" "${FP8[@]}" $FORCE --array 4 $(after "$IDX" "$QUE")); echo "x2 eval      ${EVA:-skipped} (array 4)"
GRA=$(sub grade x2 grade "${P[@]}" --set "site.slurm.gpu.gres=${GEN_GRES}" "${FP8[@]}" $FORCE $(after "$EVA")); echo "x2 grade     ${GRA:-skipped}"
RES=$(sub report x2 results report "${P[@]}" $FORCE --one-job $(after "$GRA" "$VER"));                    echo "x2 results+pdf ${RES:-skipped}"
