#!/usr/bin/env bash
# Retarget the green and yellow BONES-SEED clips to g1d and open the browser panel
# (version folders, config snapshot and resume rules: scripts/retarget_version.sh).
#
#   VERSION=v2 scripts/bones_seed/retarget.sh   start version v2, or resume it after Ctrl-C
#   scripts/bones_seed/retarget.sh --panel      only open the panel
#   VERSION=v3 NOTE="arm jump fix" scripts/bones_seed/retarget.sh num_workers=12
exec "$(dirname "$0")/../retarget_version.sh" bones_seed "$@"
