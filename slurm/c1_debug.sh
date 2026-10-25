#!/bin/bash
# Submit C1 (regulatory & ethics compliance matrix) as one CPU-sized job on a GPU partition (CPU partitions are closed).
#   bash slurm/c1_debug.sh                 # debug profile (the stored sample; seconds)
#   PROFILE=full bash slurm/c1_debug.sh    # full: scans up to c1.max_records records
#   SKIP_TDM=1 bash slurm/c1_debug.sh      # reuse the TDM check (it needs the internet: runs HERE, on the login node)
# Reads (same profile, whatever has run): s1 pii / guard / dual-use / injection / injection-live / results,
# ws1 quality / licenses / describe / provenance / datasheet, x6 results. Rerun it after any of these change.
set -euo pipefail
cd "$(dirname "$0")/.."
ALEX=/mnt/vast-kisski/projects/kisski-l3s-llm4sci/gollam/envs/Alex/bin/alexandria
PROFILE=${PROFILE:-debug}
P=(--profile "$PROFILE" --set site.slurm.cpu.partition=grete:preemptible --set site.slurm.debug.partition=grete:preemptible)
if [ -z "${SKIP_TDM:-}" ]; then $ALEX c1 tdm --profile "$PROFILE" --force; fi     # compute nodes are offline
$ALEX submit c1 evidence review matrix report "${P[@]}" --force --one-job ${AFTER:+--after $AFTER}
