#!/bin/bash
# Submit the X3 chain (Alexandria vs OpenScholar vs PaperQA2) as SLURM jobs with afterok dependencies.
#   bash slurm/x3_debug.sh                    # debug: 50 items per benchmark
#   PROFILE=full bash slurm/x3_debug.sh       # every item
# Needs: the WS3 index + benchmarks of the same profile, ScholarQABench single-paper files in $STORE/benchmarks/
# scholarqabench, the Alex-pqa venv (paper-qa), and the models in the HF cache (login node).
set -euo pipefail
cd "$(dirname "$0")/.."
ALEX=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/envs/Alex/bin/alexandria
PROFILE=${PROFILE:-debug}
GRES=${GRES:-gpu:2g.20gb:1}
SKIP=" ${SKIP:-} "
P=(--profile "$PROFILE"
   --set site.slurm.cpu.partition=grete:preemptible --set site.slurm.gpu.partition=grete:preemptible
   --set site.slurm.debug.partition=grete:preemptible --set site.slurm.gpu.cpus=8 --set site.slurm.gpu.mem=64G
   --set site.slurm.gpu.debug_time=06:00:00 --set "site.slurm.gpu.gres=${GRES}")
FP8=(--set site.llm.quantization=fp8 --set site.llm.max_model_len=14336)   # 7-8B models on 20 GB slices (as WS3)
FORCE=${FORCE:---force}
jid() { grep -oE 'job [0-9]+' | tail -1 | grep -oE '[0-9]+'; }
sub() { local label=$1; shift
        if [[ "$SKIP" == *" $label "* ]]; then echo ""; else $ALEX submit "$@" | jid; fi; }
after() { for j in "$@"; do [ -n "$j" ] && printf -- '--after %s ' "$j"; done; }

ITEMS=$(sub items x3 items "${P[@]}" $FORCE);                                                   echo "x3 items       ${ITEMS:-skipped}"
CORP=$(sub pqa-corpus x3 pqa-corpus "${P[@]}" $FORCE);                                          echo "x3 pqa-corpus  ${CORP:-skipped}"
ALX=$(sub alexandria x3 alexandria "${P[@]}" "${FP8[@]}" $FORCE $(after "$ITEMS"));            echo "x3 alexandria  ${ALX:-skipped}"
OS=$(sub openscholar x3 openscholar "${P[@]}" "${FP8[@]}" $FORCE $(after "$ITEMS"));           echo "x3 openscholar ${OS:-skipped}"
PQA=$(sub paperqa x3 paperqa "${P[@]}" $FORCE $(after "$ITEMS" "$CORP"));                      echo "x3 paperqa     ${PQA:-skipped}"
GR=$(sub grade x3 grade "${P[@]}" "${FP8[@]}" $FORCE $(after "$ALX" "$OS" "$PQA"));            echo "x3 grade       ${GR:-skipped}"
RES=$(sub report x3 results report "${P[@]}" $FORCE --one-job $(after "$GR"));                 echo "x3 results+pdf ${RES:-skipped}"
