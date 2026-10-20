#!/usr/bin/env bash
# Retarget the green and yellow LAFAN1 clips to g1d and open the browser panel, or, without a display,
# keep the frame cache current for a panel elsewhere (scripts/retarget_version.sh).
#
#   VERSION=v1 NUM_WORKERS=64 scripts/lafan/retarget.sh   start version v1, or resume it after Ctrl-C
#   scripts/lafan/retarget.sh --panel                     only open the panel
exec "$(dirname "$0")/../retarget_version.sh" lafan "$@"
