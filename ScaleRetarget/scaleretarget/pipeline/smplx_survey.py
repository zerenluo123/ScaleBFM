"""Per-clip kinematic survey of an SMPL-X dataset (GRAB, OMOMO, AMASS, Embody3D) against the g1d workspace.

Loads each clip with the dataset's loader at 30 fps, as the batch run does (body shape fitted to the G1,
pelvis at the clip's own height), and writes one row of workspace metrics (workspace.py) per clip, plus
the frame count and time at the capture rate, the clip's body (gender, betas[0]; unused by the retarget)
and the steps where the reference arm pops (ref_pops.py; the clip is reloaded at its capture rate only
when an arm flips). Zones are assigned afterwards (the dataset's classify.py).

The bones come from the dataset's g1d correspondence table; the four datasets share the original G1 one.
"""

import os
import time
from multiprocessing import Pool

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from scaleretarget.pipeline.dataset import ROOT
from scaleretarget.pipeline.ref_pops import frame_rotations, reference_pops, table_arm_bones
from scaleretarget.pipeline.workspace import clip_metrics, longest_run, rotation_offsets

FPS = 30
HANDS = ("left_wrist", "right_wrist")
_ctx = {}


def read_header(path):
    """Frame count and time at the capture rate, the clip's gender and betas[0], and, where the data marks
    its frames as tracked or not (Embody3D), the share of untracked frames and their longest run."""
    data = joblib.load(path) if path.endswith(".pkl") else np.load(path, allow_pickle=True)
    rate = float(np.asarray(data["mocap_frame_rate"]).item())
    header = {"frame_count": int(len(data["pose_body"])), "frame_time": 1.0 / rate,
              "gender": str(np.asarray(data["gender"]).item()), "beta0": float(np.asarray(data["betas"]).reshape(-1)[0])}
    if "tracked" in data:
        lost = ~np.asarray(data["tracked"], bool)
        header |= {"frac_untracked": float(lost.mean()), "untracked_run_s": longest_run(lost) / rate}
    return header


def _init(dataset):
    from hydra.utils import instantiate
    from loguru import logger
    from scaleretarget.viewer.clip_player import load_config
    logger.remove()
    os.chdir(ROOT)
    loader = instantiate(load_config(dataset.robot, dataset.loader).loader)
    loader.target_fps = FPS
    table = yaml.safe_load(open(dataset.correspondence))["ik_match_table"]
    _ctx.update(dataset=dataset, loader=loader, hips=table["pelvis_proxy"][0], chest=table["torso_link"][0],
                arm_bones=table_arm_bones(table), offsets=rotation_offsets(dataset.correspondence))


def _native_frames(path, rate):
    loader = _ctx["loader"]
    loader.target_fps = rate
    try:
        return loader._load_sample(path)[0]
    finally:
        loader.target_fps = FPS


def survey_clip(path):
    dataset = _ctx["dataset"]
    row = {"rel": os.path.relpath(path, dataset.bvh_root).removesuffix(dataset.suffix)}
    try:
        row |= read_header(path)
        frames, extras = _ctx["loader"]._load_sample(path)
        pos = lambda b: np.array([f[b][0] for f in frames])
        rot = lambda b: R.from_quat(np.array([f[b][1] for f in frames]), scalar_first=True)
        hands_z = np.stack([pos(h)[:, 2] for h in HANDS], 1)
        row |= clip_metrics(pos(_ctx["hips"]), rot(_ctx["hips"]), rot(_ctx["chest"]), hands_z, extras["fps"], _ctx["offsets"])
        native = lambda: frame_rotations(_native_frames(path, 1.0 / row["frame_time"]))
        row |= reference_pops(native, frames, extras["fps"], row["frame_time"], _ctx["arm_bones"])
        row["error"] = ""
    except Exception as e:  # noqa: BLE001 - one bad file should not kill the sweep
        row["error"] = repr(e)
    return row


def run(dataset, paths, out, workers):
    """Survey paths (files under dataset.bvh_root) into the parquet out. Every worker fits the body shape
    once when it starts (about a minute)."""
    print(f"{len(paths)} clips in {dataset.bvh_root}")
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(dataset,)) as pool:
        rows = list(pool.imap_unordered(survey_clip, paths, chunksize=4))
    df = pd.DataFrame(rows).sort_values("rel")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_parquet(out)
    print(f"{(time.time() - t0) / 60:.1f} min, errors: {int((df.error != '').sum())}, written {out}")
