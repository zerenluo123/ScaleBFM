"""Version folders of a batch retarget and the per-pkl frame / arm speed cache the panel and the jump
audit read.

Each version folder vN holds bvh/<rel>.pkl, run.json (fps, robot, ...) and the caches
vN/frame_counts.csv and vN/ood/frame_counts.csv. Scanning opens only new or rewritten pkls. On a
remote machine, build_cache --loop keeps the caches current, so a panel reading them over a network
mount (--cache_only) never walks the folders itself.
"""

import json
import os
import re
import time
from functools import partial
from multiprocessing import Pool

import joblib
import mujoco as mj
import pandas as pd
from omegaconf import OmegaConf

from scaleretarget.pipeline.dataset import ROOT
from scaleretarget.utils.arm_jumps import arm_dof_columns, arm_speed, jump_ranges
from scaleretarget.utils.ood import OOD_DIR
from scaleretarget.utils.robot_base import RobotBase

VERSION_RE = re.compile(r"^v(\d+)$")


def list_versions(root):
    """Version folders vN under root that hold a bvh/ folder, in run order."""
    if not os.path.isdir(root):
        return []
    found = [d for d in os.listdir(root) if VERSION_RE.match(d) and os.path.isdir(os.path.join(root, d, "bvh"))]
    return sorted(found, key=lambda d: int(VERSION_RE.match(d).group(1)))


def run_info(version_dir):
    path = os.path.join(version_dir, "run.json")
    return json.load(open(path)) if os.path.isfile(path) else {}


def arm_columns(robot):
    cfg = OmegaConf.load(os.path.join(ROOT, "config", "robot", f"{robot}.yaml"))
    model = mj.MjModel.from_xml_path(os.path.join(ROOT, cfg.robot.robot_xml_path))
    return arm_dof_columns(model, RobotBase(model).num_root_qpos)


def read_pkl(path, columns):
    """(frames, fps, arm speed peak, ranges > 20 rad/s, ranges > 30 rad/s), or None while being written."""
    try:
        motion = joblib.load(path)
    except Exception:
        return None
    fps = float(motion["fps"])
    speed = arm_speed(motion["dof_pos"], columns, fps)
    vmax = float(speed.max()) if len(speed) else 0.0
    return len(motion["root_pos"]), fps, vmax, ", ".join(jump_ranges(speed, 20.0)), ", ".join(jump_ranges(speed, 30.0))


class FrameCounter:
    """Frame count, fps and arm joint speed spikes of every pkl under <version>/<sub>, cached in
    <version>/<cache> so a rescan only opens new or rewritten files. With cache_only, scan only rereads
    the cache, which something else (build_cache) keeps up to date."""

    FIELDS = ["frames", "fps", "mtime", "vmax", "gt20", "gt30"]

    def __init__(self, version_dir, columns, sub="bvh", cache="frame_counts.csv", cache_only=False):
        self.retarget_dir = os.path.join(version_dir, sub)
        self.cache_path = os.path.join(version_dir, cache)
        self.columns = columns
        self.cache_only = cache_only
        self.entries = self.read_cache()

    def read_cache(self):
        if not os.path.isfile(self.cache_path):
            return {}
        # round_trip: the default parser can miss the last digit of the mtime, which marks the pkl as rewritten.
        cache = pd.read_csv(self.cache_path, keep_default_na=False, float_precision="round_trip")
        if not set(self.FIELDS) <= set(cache.columns):
            return {}
        return {r.rel: (int(r.frames), float(r.fps), float(r.mtime), float(r.vmax), str(r.gt20), str(r.gt30))
                for r in cache.itertuples()}

    def scan(self, pool=None):
        if self.cache_only:
            self.entries = self.read_cache()
            return dict(self.entries)
        found, todo = {}, []
        if os.path.isdir(self.retarget_dir):
            for dirpath, _, files in os.walk(self.retarget_dir):
                for name in files:
                    if not name.endswith(".pkl"):
                        continue
                    path = os.path.join(dirpath, name)
                    rel = os.path.relpath(path, self.retarget_dir)[:-4]
                    mtime = os.path.getmtime(path)
                    entry = self.entries.get(rel)
                    if entry is not None and entry[2] == mtime:
                        found[rel] = entry
                    else:
                        todo.append((rel, path, mtime))
        reader = partial(read_pkl, columns=self.columns)
        paths = [p for _, p, _ in todo]
        results = pool.imap(reader, paths, chunksize=64) if pool is not None else map(reader, paths)
        for (rel, _, mtime), res in zip(todo, results):
            if res is not None:  # None: still being written, picked up by the next scan
                found[rel] = (res[0], res[1], mtime, *res[2:])
        changed = bool(todo) or len(found) != len(self.entries)
        self.entries = found
        if changed:
            frame = pd.DataFrame([(k, *v) for k, v in found.items()], columns=["rel", *self.FIELDS])
            try:
                frame.to_csv(self.cache_path + ".tmp", index=False)
                os.replace(self.cache_path + ".tmp", self.cache_path)
            except OSError as e:  # read-only version folder: the next scan rereads the new pkls
                print(f"[cache] not written: {e}", flush=True)
        return dict(found)


def counters(version_dir, columns, cache_only=False):
    """Frame counters of a version's dataset folder and of its OOD folder."""
    return (FrameCounter(version_dir, columns, cache_only=cache_only),
            FrameCounter(version_dir, columns, sub=os.path.join(OOD_DIR, "bvh"),
                         cache=os.path.join(OOD_DIR, "frame_counts.csv"), cache_only=cache_only))


def build_cache(version_dir, robot, workers=18, loop=0):
    """Fill the frame caches of a version with many processes; with loop > 0, rescan every loop seconds."""
    dataset, ood = counters(version_dir, arm_columns(run_info(version_dir).get("robot", robot)))
    with Pool(workers) as pool:
        while True:
            t0 = time.time()
            n, n_ood = len(dataset.scan(pool)), len(ood.scan(pool))
            print(f"[cache] {time.strftime('%H:%M:%S')} {version_dir}: {n:,} pkl, {n_ood:,} in OOD "
                  f"({time.time() - t0:.0f} s)", flush=True)
            if loop <= 0:
                return
            time.sleep(max(0.0, loop - (time.time() - t0)))
