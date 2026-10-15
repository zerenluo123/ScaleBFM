#!/usr/bin/env bash
# Retarget the green and yellow BONES-SEED clips to g1d and open the browser panel.
#
#   VERSION=v2 scripts/bones_seed/retarget.sh   start version v2, or resume it after Ctrl-C
#   scripts/bones_seed/retarget.sh --panel      only open the panel
#
# The source dataset is read-only; each batch run is a version folder $OUT/vN holding
# bvh/<date>/<clip>.pkl, run.json (fps, robot, date, git commit, note) and, under config/ and
# assets/, copies of the robot, correspondence, loader and MJCF files the run started with. Clips
# already converted are skipped. Resuming a version is refused when any of those files changed since
# it started, so one version never mixes two configs; the changes are printed and the next free
# version is suggested. A new version must be the next number. The panel runs in its own session:
# Ctrl-C or the end of the run does not close it, and only one panel is opened. Only one run per
# version is allowed. Clips moved to the OOD zone in the panel (vN/ood/bvh/...) count as converted
# and are never converted again. Extra hydra overrides are passed through and recorded in run.json, e.g.
#   VERSION=v3 NOTE="arm jump fix" scripts/bones_seed/retarget.sh num_workers=12
set -euo pipefail
cd "$(dirname "$0")/../.."

BONES=/home/zerenluo/Space/bfm_datasets/bones-seed
OUT=${OUT:-/home/zerenluo/Space/bfm_datasets/bone-seed_retarget}
TARGET_FPS=${TARGET_FPS:-30}
LISTS=${LISTS:-[$OUT/survey/green.txt,$OUT/survey/yellow.txt]}
SNAPSHOT=(config/robot/unitree_g1d.yaml config/correspondence/gmr/soma_to_unitree_g1d.yaml
          config/loader/soma.yaml config/loader/base.yaml assets/unitree_g1d/g1d_mocap.xml)

# Files of $SNAPSHOT that differ from the copies a version started with (all of them if it has none).
changed_files() {
  local f
  for f in "${SNAPSHOT[@]}"; do
    cmp -s "$f" "$OUT/$1/$f" || echo "$f"
  done
}

LATEST=$(find "$OUT" -mindepth 2 -maxdepth 2 -path "$OUT/v*/run.json" -printf '%h\n' | sed 's|.*/v||' | grep -E '^[0-9]+$' | sort -n | tail -1 || true)
NEXT=v$(( ${LATEST:-0} + 1 ))
if [[ "${1:-}" == "--panel" ]]; then
  VERSION=${VERSION:-v${LATEST:-1}}
elif [[ -z "${VERSION:-}" ]]; then
  echo "Set VERSION: VERSION=v${LATEST:-1} resumes the latest version, VERSION=$NEXT starts a new one"
  exit 1
fi
[[ "$VERSION" =~ ^v[0-9]+$ ]] || { echo "VERSION must look like v2, got '$VERSION'"; exit 1; }
RUN_DIR=$OUT/$VERSION
if [[ ! -f "$RUN_DIR/run.json" && "${1:-}" != "--panel" && "$VERSION" != "$NEXT" ]]; then
  echo "$VERSION does not exist yet; a new version must be the next number, $NEXT"
  exit 1
fi

python -c "import mink, PySide6" || { echo "Activate the env first: conda activate scaleretarget"; exit 1; }

open_panel() {
  if pgrep -f "^[^ ]*python[0-9.]* scripts/bones_seed/browse.py" >/dev/null; then
    echo "[panel] already open; pick $VERSION under 结果版本 (the list refreshes by itself)"
  else
    setsid nohup python scripts/bones_seed/browse.py --version "$VERSION" >"$OUT/panel.log" 2>&1 </dev/null &
    echo "[panel] opened on $VERSION (log: $OUT/panel.log)"
  fi
}
[[ "${1:-}" == "--panel" ]] && { open_panel; exit 0; }

mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/.retarget.lock"
if ! flock -n 9; then
  echo "[retarget] a run of $VERSION is already in progress; not starting another"
  open_panel
  exit 0
fi

if [[ ! -f "$RUN_DIR/run.json" ]]; then
  mkdir -p "$RUN_DIR/bvh"
  cp --parents "${SNAPSHOT[@]}" "$RUN_DIR/"
  RUN_JSON="$RUN_DIR/run.json" FPS=$TARGET_FPS NOTE=${NOTE:-} \
  GIT=$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . 2>/dev/null || echo "+uncommitted") \
  python - "$@" <<'EOF'
import json, os, sys, time
info = {"fps": int(os.environ["FPS"]), "robot": "unitree_g1d", "created": time.strftime("%Y-%m-%d %H:%M"),
        "git": os.environ["GIT"], "note": os.environ["NOTE"], "overrides": sys.argv[1:]}
json.dump(info, open(os.environ["RUN_JSON"], "w"), ensure_ascii=False, indent=2)
EOF
  echo "[retarget] new version $VERSION: $RUN_DIR"
else
  FPS_RUN=$(python -c "import json; print(json.load(open('$RUN_DIR/run.json'))['fps'])")
  [[ "$FPS_RUN" == "$TARGET_FPS" ]] || { echo "[retarget] $VERSION was run at $FPS_RUN fps, not $TARGET_FPS"; exit 1; }
  if [[ ! -d "$RUN_DIR/config" ]]; then
    echo "[retarget] $VERSION has no copy of the config it was run with, so it cannot be resumed."
    echo "           Start a new version: VERSION=$NEXT scripts/bones_seed/retarget.sh"
    exit 1
  fi
  changed=$(changed_files "$VERSION")
  if [[ -n "$changed" ]]; then
    echo "[retarget] the config changed since $VERSION started, so it is not resumed (it would mix two configs):"
    for f in $changed; do
      echo
      diff -u --label "$VERSION/$f" --label "$f (now)" "$RUN_DIR/$f" "$f" | head -80 || true
    done
    echo
    echo "           Start a new version with the current config: VERSION=$NEXT scripts/bones_seed/retarget.sh"
    echo "           or restore the files above to resume $VERSION."
    exit 1
  fi
  mkdir -p "$RUN_DIR/bvh"
fi

open_panel
# Temporary files of clips that were being written when a previous run was interrupted.
find "$RUN_DIR/bvh" -name "*.tmp" -delete

LOGURU_LEVEL=${LOGURU_LEVEL:-INFO} python retarget.py +robot=unitree_g1d +loader=soma \
  multi_process=true num_workers=${NUM_WORKERS:-18} \
  data_path=$BONES/soma_uniform/bvh \
  "loader.config.file_list=$LISTS" \
  loader.config.read_only_root=$BONES \
  loader.config.target_fps=$TARGET_FPS \
  "++loader.config.done_dirs=[$RUN_DIR/ood]" \
  output_dir=$RUN_DIR "$@"
echo "[retarget] finished; the panel stays open until you close it"
