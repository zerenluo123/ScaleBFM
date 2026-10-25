"""Play one source motion in MuJoCo with the IK targets drawn as frames, as in `retarget.py`.

The targets are recomputed from the source file with the same loader, root height filter, scaling
and offsets as the retargeter. The robot comes from a retargeted pkl, or is hidden when no pkl is
given so that only the reference is shown.

    python -m scaleretarget.viewer.clip_player --source X.bvh [--pkl X.pkl] --robot unitree_g1d --loader soma

The official BONES-SEED G1 retarget of the same clip plays with --official_csv bones-seed/g1/csv/X.csv
--robot unitree_g1, read the same way as utils/convert_bones_to_ours.py (every 4th row of 120 fps).

Keys (also shown in the window): space pause, R restart, right/left arrow pause and step one frame,
up arrow jump to the next marked spike, C camera follow on/off, T transparent robot.

--spikes "113→114, 230→232" marks frame ranges (1-based, as shown in the window); playback starts
one second before the first one.

--live retargets the source on the spot with the current config instead of reading a pkl, and marks
its own arm-joint jumps (> 20 rad/s after frame 5), e.g. to compare an IK change with an older pkl.
--official_csv without --spikes marks the official retarget's own jumps the same way.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import joblib
import mujoco as mj
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from loguru import logger

from scaleretarget.utils.arm_jumps import JUMP_SPEED, arm_jump_ranges
from scaleretarget.utils.convert_bones_to_ours import euler_xyz_degrees_to_quaternion
from scaleretarget.viewer.robot_motion_viewer import RobotMotionViewer

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config")
KEY_RIGHT, KEY_LEFT, KEY_UP = 262, 263, 265
MAX_SPIKES_SHOWN = 6


def load_config(robot, loader, overrides=(), snapshot=None):
    """Compose the retarget config; with snapshot (a version folder), the config files it saved replace
    the current ones and the robot overrides of its run.json are applied, so the targets match what that
    version was run with."""
    if snapshot is not None:
        run_json = os.path.join(snapshot, "run.json")
        saved_overrides = json.load(open(run_json)).get("overrides", []) if os.path.isfile(run_json) else []
        robot_overrides = [o for o in saved_overrides if o.lstrip("+").startswith("robot.")]
        overrides = list(dict.fromkeys([*robot_overrides, *overrides]))
    with tempfile.TemporaryDirectory() as tmp:
        config_dir = os.path.abspath(CONFIG_DIR)
        saved = snapshot is not None and os.path.isdir(os.path.join(snapshot, "config"))
        if saved:
            shutil.copytree(config_dir, tmp, dirs_exist_ok=True)
            shutil.copytree(os.path.join(snapshot, "config"), tmp, dirs_exist_ok=True)
            config_dir = tmp
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose("base", overrides=[f"+robot={robot}", f"+loader={loader}", "data_path=unused",
                                                "enable_viewer=false", *overrides])
    if saved:
        # The correspondence is named by path (relative to the ScaleRetarget folder), not composed.
        copy = os.path.join(snapshot, config.retargeter.config.ik_config)
        if os.path.isfile(copy):
            config.retargeter.config.ik_config = copy
    return config


def load_official_csv(path, source_fps=120, target_fps=30):
    data = np.atleast_2d(np.genfromtxt(path, delimiter=",", skip_header=1))[:: source_fps // target_fps]
    return {
        "root_pos": data[:, 1:4] / 100.0,
        "root_rot": euler_xyz_degrees_to_quaternion(data[:, 4:7]),
        "dof_pos": np.deg2rad(data[:, 7:]),
        "fps": target_fps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--official_csv", default=None, help="official BONES-SEED G1 csv; use with --robot unitree_g1")
    ap.add_argument("--live", action="store_true", help="retarget the source now with the current config")
    ap.add_argument("--robot", default="unitree_g1d")
    ap.add_argument("--loader", default="soma")
    ap.add_argument("--title", default="")
    ap.add_argument("--spikes", default="", help="frame ranges to mark, e.g. '113→114, 230→232' (1-based)")
    ap.add_argument("--spike_label", default="spikes")
    ap.add_argument("--override", nargs="*", default=[], help="hydra overrides the pkl was retargeted with")
    ap.add_argument("--config_snapshot", default=None, help="version folder whose saved config the pkl was retargeted with")
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    config = load_config(args.robot, args.loader, args.override, args.config_snapshot)
    loader = instantiate(config.loader)
    retargeter = instantiate(config.retargeter)

    motion = joblib.load(args.pkl) if args.pkl else None
    if args.official_csv:
        motion = load_official_csv(args.official_csv)
    if motion is not None:
        # Resample the source to the pkl's rate so targets and robot frames line up.
        loader.target_fps = int(round(motion["fps"]))
    frames, extras = loader._load_sample(args.source)
    retargeter.update(extras)

    if args.live:
        mode = "retargeted live (current config)"
        print(f"[player] retargeting {len(frames)} frames ...", flush=True)
        retargeter.show_progress = False
        qpos = retargeter.retarget([dict(f) for f in frames])
        root_pos, root_rot, dof_pos = retargeter.base.split_qpos(qpos)
        targets = retargeter.compute_targets(frames)
        args.spikes = ", ".join(arm_jump_ranges(retargeter.model, dof_pos, retargeter.base.num_root_qpos, extras["fps"]))
        args.spike_label = f"live jumps > {JUMP_SPEED:.0f} rad/s"
    elif motion is not None:
        mode = "official BONES G1" if args.official_csv else "retargeted"
        root_pos, root_rot, dof_pos = motion["root_pos"], motion["root_rot"][:, [3, 0, 1, 2]], motion["dof_pos"]
        targets = retargeter.compute_targets(frames)
        if args.official_csv and not args.spikes:
            args.spikes = ", ".join(arm_jump_ranges(retargeter.model, dof_pos, retargeter.base.num_root_qpos, motion["fps"]))
            args.spike_label = f"official jumps > {JUMP_SPEED:.0f} rad/s"
    else:
        mode = "reference only"
        targets = retargeter.compute_targets(frames)
        root_name = retargeter.human_root_name
        root_pos = [t[root_name][0] for t in targets]
        root_rot = [t[root_name][1] for t in targets]

    num_frames = min(len(targets), len(root_pos))
    # The overlay font is ASCII only.
    spike_ranges = [s.strip().replace("→", "->") for s in args.spikes.split(",") if s.strip()]
    preroll = int(round(extras["fps"]))
    spike_starts = [max(0, int(s.split("->")[0]) - 1 - preroll) for s in spike_ranges]
    state = {"frame": spike_starts[0] if spike_starts else 0, "paused": False, "transparent": False, "spike": 0}

    # Every letter is also a built-in MuJoCo shortcut (C contact points, R reflection, T transparency),
    # so these handlers only change state that the built-ins do not touch.
    def key_callback(keycode):
        if keycode == ord(" "):
            state["paused"] = not state["paused"]
        elif keycode == ord("R"):
            state["frame"] = 0
        elif keycode in (KEY_RIGHT, KEY_LEFT):
            # While playing, the frame shown is already one ahead of the last drawn one.
            back = 0 if state["paused"] else 1
            state["paused"] = True
            step = 1 if keycode == KEY_RIGHT else -1
            state["frame"] = (state["frame"] - back + step) % num_frames
        elif keycode == KEY_UP and spike_starts:
            state["spike"] = (state["spike"] + 1) % len(spike_starts)
            state["frame"] = spike_starts[state["spike"]]
        elif keycode == ord("C"):
            viewer.camera_follow = not viewer.camera_follow
        elif keycode == ord("T") and mode != "reference only":
            state["transparent"] = not state["transparent"]
            viewer.model.geom_rgba[robot_geoms, 3] = 0.3 if state["transparent"] else robot_alpha

    viewer = RobotMotionViewer(robot_config=config.robot, key_callback=key_callback)
    viewer.update(fps=extras["fps"])
    robot_geoms = viewer.model.geom_bodyid != 0
    robot_alpha = viewer.model.geom_rgba[robot_geoms, 3].copy()
    if mode == "reference only":
        # Keep the floor (world body) and hide the robot; it only carries the camera along the root target.
        viewer.model.geom_rgba[robot_geoms, 3] = 0.0
        dof_pos = [np.zeros(viewer.model.nq - viewer.base.num_root_qpos)] * num_frames

    title = args.title or os.path.basename(args.source)
    while viewer.viewer.is_running():
        i = state["frame"]
        follow = "on" if viewer.camera_follow else "off (mouse: drag rotate / pan, scroll zoom)"
        keys = [
            ("Space", "pause / play" + ("   [paused]" if state["paused"] else "")),
            ("R", "restart from frame 1"),
            ("Left / Right", "pause and step one frame back / forward"),
        ]
        if spike_starts:
            keys.append(("Up", f"jump to 1 s before the next spike ({state['spike'] + 1}/{len(spike_starts)})"))
        keys.append(("C", f"camera follow: {follow}"))
        if mode != "reference only":
            keys.append(("T", "transparent robot: " + ("on" if state["transparent"] else "off")))
        status = f"{mode}   frame {i + 1}/{num_frames}   {extras['fps']:.1f} fps"
        if spike_ranges:
            shown = ", ".join(spike_ranges[:MAX_SPIKES_SHOWN])
            more = f" (+{len(spike_ranges) - MAX_SPIKES_SHOWN} more)" if len(spike_ranges) > MAX_SPIKES_SHOWN else ""
            status += f"\n{args.spike_label} at frames {shown}{more}"
        viewer.viewer.set_texts([
            (mj.mjtFontScale.mjFONTSCALE_150, mj.mjtGridPos.mjGRID_TOPLEFT, title + ("\n" if spike_ranges else ""),
             status),
            (mj.mjtFontScale.mjFONTSCALE_100, mj.mjtGridPos.mjGRID_BOTTOMLEFT,
             "\n".join(k for k, _ in keys), "\n".join(v for _, v in keys)),
        ])
        viewer.step(root_pos=root_pos[i], root_rot=root_rot[i], dof_pos=dof_pos[i], human_motion_data=targets[i])
        if state["paused"]:
            time.sleep(0.02)
        else:
            state["frame"] = (i + 1) % num_frames
    viewer.close()


if __name__ == "__main__":
    main()
