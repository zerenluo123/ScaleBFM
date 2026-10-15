"""Clips taken out of a batch-retarget version folder as out of distribution (OOD).

A version folder vN holds the dataset under vN/bvh/<rel>.pkl. OOD clips move to vN/ood/bvh/<rel>.pkl,
logged in vN/ood/ood.csv with the zone they came from and why, so the dataset folder only holds the
clips meant for training. Passing vN/ood as loader.config.done_dirs keeps batch runs from converting
them again; restoring moves the pkl back.
"""

import os
import time

import pandas as pd

OOD_DIR = "ood"
OOD_FIELDS = ["rel", "zone", "reason", "time"]


def ood_pkl(version_dir, rel):
    return os.path.join(version_dir, OOD_DIR, "bvh", rel + ".pkl")


def dataset_pkl(version_dir, rel):
    return os.path.join(version_dir, "bvh", rel + ".pkl")


def find_pkl(version_dir, rel):
    """The clip's pkl in the dataset or in the OOD folder; None if it was not converted."""
    for path in (dataset_pkl(version_dir, rel), ood_pkl(version_dir, rel)):
        if os.path.isfile(path):
            return path
    return None


def read_ood_log(version_dir):
    path = os.path.join(version_dir, OOD_DIR, "ood.csv")
    return pd.read_csv(path, keep_default_na=False) if os.path.isfile(path) else pd.DataFrame(columns=OOD_FIELDS)


def _write_ood_log(version_dir, log):
    path = os.path.join(version_dir, OOD_DIR, "ood.csv")
    log.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)


def move_to_ood(version_dir, rel, zone, reason=""):
    """Move a converted clip from vN/bvh/ to vN/ood/bvh/ and log where it came from."""
    dst = ood_pkl(version_dir, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.replace(dataset_pkl(version_dir, rel), dst)
    log = read_ood_log(version_dir)
    entry = pd.DataFrame([{"rel": rel, "zone": zone, "reason": reason, "time": time.strftime("%Y-%m-%d %H:%M")}])
    _write_ood_log(version_dir, pd.concat([log[log.rel != rel], entry], ignore_index=True))


def restore_from_ood(version_dir, rel):
    """Move a clip from vN/ood/bvh/ back to vN/bvh/; refused if vN/bvh/ already holds it."""
    dst = dataset_pkl(version_dir, rel)
    if os.path.exists(dst):
        raise FileExistsError(f"{dst} already exists")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.replace(ood_pkl(version_dir, rel), dst)
    log = read_ood_log(version_dir)
    _write_ood_log(version_dir, log[log.rel != rel])
