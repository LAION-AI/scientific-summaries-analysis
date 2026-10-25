#!/bin/bash
# Submit the S1 chain (safety checks and evaluation) as SLURM jobs with afterok dependencies.
#   bash slurm/s1_debug.sh                       # debug: WS1 sample (480 records), small WMDP / Do-Not-Answer subsets
#   SKIP="pii injection" bash slurm/s1_debug.sh  # reuse stages whose outputs are still valid
#   PROFILE=full bash slurm/s1_debug.sh          # large random record sample, every WMDP / Do-Not-Answer item
# Needs the WS3 index + benchmarks of the same profile (injection-live, uplift) and, in the HF cache (login node):
# Qwen3Guard-Gen-8B, granite-guardian-3.3-8b, cais/wmdp, LibrAI/do-not-answer.
set -euo pipefail
cd "$(dirname "$0")/.."
ALEX=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/envs/Alex/bin/alexandria
PROFILE=${PROFILE:-debug}
GRES=${GRES:-gpu:2g.20gb:1}
ARRAY=${ARRAY:-2}                               # one array task per answer model (injection-live, uplift)
SKIP=" ${SKIP:-} "
P=(--profile "$PROFILE"
   --set site.slurm.cpu.partition=grete:preemptible --set site.slurm.gpu.partition=grete:preemptible
   --set site.slurm.debug.partition=grete:preemptible --set site.slurm.gpu.cpus=8 --set site.slurm.gpu.mem=64G
   --set site.slurm.gpu.debug_time=06:00:00 --set "site.slurm.gpu.gres=${GRES}")
FP8=(--set site.llm.quantization=fp8 --set site.llm.max_model_len=14336)   # 7-8B models on 20 GB slices (as WS3)
BIG=(--set site.slurm.debug.cpus=16 --set site.slurm.debug.mem=64G --set site.slurm.debug.debug_time=02:00:00)
FORCE=${FORCE:---force}
jid() { grep -oE 'job [0-9]+' | tail -1 | grep -oE '[0-9]+'; }
sub() { local label=$1; shift
        if [[ "$SKIP" == *" $label "* ]]; then echo ""; else $ALEX submit "$@" | jid; fi; }
after() { for j in "$@"; do [ -n "$j" ] && printf -- '--after %s ' "$j"; done; }

PII=$(sub pii s1 pii "${P[@]}" "${BIG[@]}" $FORCE);                                              echo "s1 pii            ${PII:-skipped}"
INJ=$(sub injection s1 injection "${P[@]}" "${BIG[@]}" $FORCE);                                  echo "s1 injection      ${INJ:-skipped}"
RET=$(sub retractions s1 retractions "${P[@]}" "${BIG[@]}" $FORCE);                              echo "s1 retractions    ${RET:-skipped}"
GRD=$(sub guard s1 guard "${P[@]}" "${FP8[@]}" $FORCE);                                          echo "s1 guard          ${GRD:-skipped}"
DU=$(sub dual-use s1 dual-use "${P[@]}" "${FP8[@]}" $FORCE);                                     echo "s1 dual-use       ${DU:-skipped}"
LIVE=$(sub injection-live s1 injection-live "${P[@]}" "${FP8[@]}" $FORCE --array "$ARRAY");     echo "s1 injection-live ${LIVE:-skipped} (array $ARRAY)"
UP=$(sub uplift s1 uplift "${P[@]}" "${FP8[@]}" $FORCE --array "$ARRAY");                        echo "s1 uplift         ${UP:-skipped} (array $ARRAY)"
RES=$(sub report s1 results report "${P[@]}" $FORCE --one-job $(after "$PII" "$INJ" "$RET" "$GRD" "$DU" "$LIVE" "$UP")); echo "s1 results+pdf    ${RES:-skipped}"
