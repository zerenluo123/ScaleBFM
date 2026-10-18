"""Re-measure the numbers in config/loader/hiphi.yaml and the torso_link offsets of the HiPHI
correspondences, and print them next to the current ones.

reference_distances           lengths (cm) each HiPHI segment is rescaled to on its rest pose, under the
                              HiPHI names. Limbs and UpLeg (from the mid hip joints, where HiPHI's Hips is;
                              SOMA's is above and behind): the SOMA Uniform joint distances, 90th percentile
                              over the frames of random BONES-SEED motions (SomaLoader), all frames alike for
                              these fixed lengths. Arm (mid hip joints to shoulder, which moves with the
                              spine and shoulder blades): the rest-pose (T-pose) length at which the HiPHI
                              walks have the median over frames of the SOMA walks
hips_offset                   SOMA Uniform's Hips relative to its mid hip joints (cm), median over the same
                              motions, turned into the HiPHI Hips frame through the two pelvis_proxy
                              rotation offsets; the HiPHI loader moves its Hips there
planted_ankle_height          height (m) of the ankle a SOMA Uniform motion stands on with a straight leg
                              (planted_ankle_height in the HiPHI loader), median over the same motions
typical_planted_ankle_height  the same for HiPHI, median over the surveyed motions that stand
                              (survey.py records the loader's ground shift)
torso_link rotation offset    the HiPHI rest-pose offset (every bone frame is the world frame there, so it
                              is the pelvis_proxy one) pitched forward so that, on walking motions, the
                              torso target leans ahead of the trunk line (mid hip joints -> mid upper arms)
                              by as much as SOMA Chest's does (difference of the medians over upright
                              frames). Not from Hips: SOMA's sits ~8 cm above and ~3 cm behind the hip
                              joints, HiPHI's between them, which alone tilts a Hips-based line ~3 deg.
torso_link position offset    puts the torso target where it is relative to the shoulder targets with SOMA
                              (median over random motions, in the torso target frame); the same for G1 and
                              G1-D, whose torso and arm targets share one scale

    python scripts/hiphi/calibrate_offsets.py [--motions 200] [--workers 18]
"""

import argparse
import glob
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from dataset import CORRESPONDENCE, DATASET, ROOT, read_metadata

BONES = os.environ.get("BONES_SEED", "/home/zerenluo/Space/bfm_datasets/bones-seed")
BONES_BVH = BONES + "/soma_uniform/bvh"
BONES_METADATA = f"{BONES}/metadata/seed_metadata_v004.parquet"
CORRESPONDENCES = {"soma": f"{ROOT}/config/correspondence/gmr/soma_to_unitree_g1d.yaml", "hiphi": CORRESPONDENCE}
SHOULDER_LINKS = ("left_shoulder_roll_link", "right_shoulder_roll_link")
HIP_JOINTS = {"soma": ("LeftLeg", "RightLeg"), "hiphi": ("LeftUpLeg", "RightUpLeg")}
HIPHI_WALKS = ("walk", "amble", "stride", "march")
UPRIGHT_HIPS_Z = 0.75  # m; walking frames, not crouches
HIP_CENTER = "HipCenter"  # mid SOMA hip joints
# HiPHI name -> (SOMA joint, SOMA joint the distance is measured from)
SOMA_SEGMENTS = {
    f"{s}{h}": (f"{s}{b}", f"{s}{p}" if p != HIP_CENTER else HIP_CENTER)
    for s in ("Left", "Right")
    for h, b, p in [("Arm", "Arm", HIP_CENTER), ("ForeArm", "ForeArm", "Arm"), ("Hand", "Hand", "ForeArm"),
                    ("UpLeg", "Leg", HIP_CENTER), ("Leg", "Shin", "Leg"), ("Foot", "Foot", "Shin")]
}
_ctx = {}


def soma_motion(path):
    """Per-segment distances (cm) of every frame, the straight-leg planted ankle height (m) or None, and the
    median position (cm) of Hips relative to the mid hip joints in the pelvis_proxy target frame."""
    if "loader" not in _ctx:
        from hydra.utils import instantiate
        from loguru import logger
        from scaleretarget.viewer.clip_player import load_config
        logger.remove()
        _ctx["loader"] = instantiate(load_config("unitree_g1d", "soma").loader)
        pelvis = yaml.safe_load(open(CORRESPONDENCES["soma"]))["ik_match_table"]["pelvis_proxy"][4]
        _ctx["pelvis"] = R.from_quat(pelvis, scalar_first=True)
    from scaleretarget.loader.hiphi_loader import STRAIGHT_LEG
    frames, extras = _ctx["loader"]._load_sample(path)
    center = np.mean([[f[j][0] for f in frames] for j in HIP_JOINTS["soma"]], axis=0)
    pos = lambda b: center if b == HIP_CENTER else np.array([f[b][0] for f in frames])
    distances = {h: np.linalg.norm(pos(b) - pos(p), axis=1) * 100 for h, (b, p) in SOMA_SEGMENTS.items()}
    pelvis = R.from_quat(np.array([f["Hips"][1] for f in frames]), scalar_first=True) * _ctx["pelvis"]
    hips = np.median(pelvis.inv().apply(pos("Hips") - center), axis=0) * 100
    leg = np.mean([(np.percentile(distances[f"{s}Leg"], 90) + np.percentile(distances[f"{s}Foot"], 90)) / 100
                   for s in ("Left", "Right")])
    ankle = np.stack([pos("LeftFoot"), pos("RightFoot")], 1)
    hip = np.stack([pos("LeftLeg"), pos("RightLeg")], 1)
    lower = ankle[..., 2].argmin(1)
    frame = np.arange(len(frames))
    standing = np.linalg.norm(hip[frame, lower] - ankle[frame, lower], axis=-1) >= STRAIGHT_LEG * leg
    planted = np.percentile(ankle[frame, lower][standing, 2], 10) if standing.sum() >= int(extras["fps"]) else None
    return distances, planted, hips


def hiphi_trunk(path):
    """Per side, the raw mid hip joints (Hips) to shoulder distance of every frame over that of the rest pose."""
    from scaleretarget.loader.hiphi_loader import rest_pose
    from scaleretarget.utils.lafan_vendor.extract import read_bvh_multi_channel
    from scaleretarget.utils.lafan_vendor.utils import quat_fk
    bvh = read_bvh_multi_channel(path)
    _, positions = quat_fk(bvh.quats, bvh.pos, bvh.parents)
    rest = rest_pose(bvh)
    out = {}
    for side in ("Left", "Right"):
        arm = bvh.bones.index(f"{side}Arm")
        out[f"{side}Arm"] = np.linalg.norm(positions[:, arm] - positions[:, 0], axis=-1) / np.linalg.norm(rest[arm])
    return out


def walk_motions(args):
    """Non-mirrored walks: BONES-SEED SOMA Uniform paths, HiPHI paths, and all unpacked HiPHI motions."""
    bones = pd.read_parquet(BONES_METADATA)
    walks = bones[(bones.category == "Basic Locomotion Neutral") & ~bones.is_mirror
                  & bones.content_short_description.str.contains(r"\bwalk", case=False)]
    soma_walks = list(BONES + "/" + walks.sample(min(args.motions, len(walks)), random_state=0).move_soma_uniform_path)
    hiphi = read_metadata()
    hiphi["path"] = DATASET.bvh_root + "/" + hiphi.rel + ".bvh"
    hiphi = hiphi[~hiphi.mirrored & hiphi.path.map(os.path.exists)]
    walks = hiphi[(hiphi.frame == "Self_motion") & hiphi.lu.isin(HIPHI_WALKS)]
    hiphi_walks = list(walks.sample(min(args.motions, len(walks)), random_state=0).path)
    return soma_walks, hiphi_walks, hiphi


def trunk_reference(args, soma_walks, hiphi_walks):
    """reference_distances for Arm (see the module doc)."""
    with ProcessPoolExecutor(args.workers) as ex:
        soma = [r[0] for r in ex.map(soma_motion, soma_walks, chunksize=2)]
        hiphi = list(ex.map(hiphi_trunk, hiphi_walks, chunksize=2))
    return {name: np.median(np.concatenate([d[name] for d in soma])) / np.median(np.concatenate([d[name] for d in hiphi]))
            for name in ("LeftArm", "RightArm")}


def torso_motion(job):
    """With torso_link rotation offset `offset` (wxyz): median over upright frames of how far (deg) the torso
    target leans ahead of the trunk line (nan if under a second upright), and the median position (m) of the
    mid shoulder target relative to the torso bone, in the torso target frame (robot scale)."""
    dataset, path, offset = job
    if dataset not in _ctx:
        from hydra.utils import instantiate
        from loguru import logger
        from scaleretarget.viewer.clip_player import load_config
        logger.remove()
        _ctx[dataset] = instantiate(load_config("unitree_g1d", dataset).loader)
    corr = yaml.safe_load(open(CORRESPONDENCES[dataset]))
    table, scale = corr["ik_match_table"], corr["human_scale_table"]
    rot = lambda q: R.from_quat(q, scalar_first=True)
    frames, extras = _ctx[dataset]._load_sample(path)
    bone = table["torso_link"][0]
    if bone not in frames[0]:
        return np.nan, np.full(3, np.nan)
    pos = lambda b: np.array([f[b][0] for f in frames])
    quat = lambda b: rot(np.array([f[b][1] for f in frames]))
    torso = quat(bone) * rot(offset)

    heading = (quat("Hips") * rot(table["pelvis_proxy"][4])).apply([1.0, 0.0, 0.0])
    heading[:, 2] = 0
    heading /= np.linalg.norm(heading, axis=1, keepdims=True)
    pitch = lambda v: np.degrees(np.arctan2((v * heading).sum(1), v[:, 2]))
    hip_joints = (pos(HIP_JOINTS[dataset][0]) + pos(HIP_JOINTS[dataset][1])) / 2
    trunk = (pos(table[SHOULDER_LINKS[0]][0]) + pos(table[SHOULDER_LINKS[1]][0])) / 2 - hip_joints
    lean = pitch(torso.apply([0.0, 0.0, 1.0])) - pitch(trunk)
    upright = pos("Hips")[:, 2] > UPRIGHT_HIPS_Z

    # torso and arm targets are scaled about Hips by the same factor, so their differences scale with it
    assert all(scale[table[link][0]] == scale[bone] for link in SHOULDER_LINKS)
    shoulders = np.mean([scale[bone] * pos(table[link][0]) + (quat(table[link][0]) * rot(table[link][4])).apply(table[link][3])
                         for link in SHOULDER_LINKS], axis=0)
    relative = torso.inv().apply(shoulders - scale[bone] * pos(bone))
    return (np.median(lean[upright]) if upright.sum() >= int(extras["fps"]) else np.nan), np.median(relative, axis=0)


def torso_offsets(args, soma_walks, hiphi_walks, hiphi):
    """torso_link rotation and position offsets for HiPHI, measured against SOMA (see the module doc)."""
    rot = lambda q: R.from_quat(q, scalar_first=True)
    soma_table = yaml.safe_load(open(CORRESPONDENCES["soma"]))["ik_match_table"]
    hiphi_table = yaml.safe_load(open(CORRESPONDENCE))["ik_match_table"]
    soma_offset = soma_table["torso_link"][4]
    rest = rot(hiphi_table["pelvis_proxy"][4])

    rest_wxyz = rest.as_quat(scalar_first=True)
    with ProcessPoolExecutor(args.workers) as ex:
        soma_lean = [r[0] for r in ex.map(torso_motion, [("soma", p, soma_offset) for p in soma_walks])]
        hiphi_lean = [r[0] for r in ex.map(torso_motion, [("hiphi", p, rest_wxyz) for p in hiphi_walks])]
    forward = np.nanmedian(soma_lean) - np.nanmedian(hiphi_lean)
    offset = (rest * R.from_euler("y", forward, degrees=True)).as_quat(scalar_first=True)

    random.seed(1)
    soma_random = random.sample(sorted(glob.glob(f"{BONES_BVH}/*/*.bvh")), args.motions)
    hiphi_random = list(hiphi.sample(min(args.motions, len(hiphi)), random_state=1).path)
    with ProcessPoolExecutor(args.workers) as ex:
        soma_rel = np.nanmedian([r[1] for r in ex.map(torso_motion, [("soma", p, soma_offset) for p in soma_random], chunksize=2)], axis=0)
        hiphi_rel = np.nanmedian([r[1] for r in ex.map(torso_motion, [("hiphi", p, offset) for p in hiphi_random], chunksize=2)], axis=0)
    position = np.asarray(soma_table["torso_link"][3]) + hiphi_rel - soma_rel
    position[1] = 0.0  # left-right symmetric (every motion has its mirror)
    return {"bone": hiphi_table["torso_link"][0], "forward": forward, "rotation": offset, "position": position,
            "walks": (len(soma_walks), len(hiphi_walks)), "lean": (np.nanmedian(soma_lean), np.nanmedian(hiphi_lean)),
            "current": hiphi_table["torso_link"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motions", type=int, default=200, help="random BONES-SEED motions")
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()
    os.chdir(ROOT)
    current = yaml.safe_load(open(os.path.join(ROOT, "config/loader/hiphi.yaml")))["loader"]["config"]

    random.seed(0)
    paths = random.sample(sorted(glob.glob(f"{BONES_BVH}/*/*.bvh")), args.motions)
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(soma_motion, paths, chunksize=2))
    soma_walks, hiphi_walks, hiphi = walk_motions(args)
    trunk = trunk_reference(args, soma_walks, hiphi_walks)
    print(f"SOMA Uniform, {len(paths)} motions; walks SOMA / HiPHI: {len(soma_walks)} / {len(hiphi_walks)}")
    print("reference_distances (cm):  measured / current")
    for name in SOMA_SEGMENTS:
        if name in trunk:
            measured = trunk[name]
        else:
            measured = np.percentile(np.concatenate([d[name] for d, _, _ in results]), 90)
        print(f"  {name:13s} {measured:6.2f} / {current['reference_distances'][name]:.2f}")
    planted = [p for _, p, _ in results if p is not None]
    print(f"planted_ankle_height (m): {np.median(planted):.3f} / {current['planted_ankle_height']}  "
          f"({len(planted)} motions stand)")
    hips = np.median([h for _, _, h in results], axis=0)
    hips[1] = 0.0  # left-right symmetric
    hiphi_pelvis = R.from_quat(yaml.safe_load(open(CORRESPONDENCE))["ik_match_table"]["pelvis_proxy"][4], scalar_first=True)
    print(f"SOMA Hips from the mid hip joints (cm, forward / left / up): {np.round(hips, 2).tolist()}")
    print(f"hips_offset (cm, HiPHI Hips frame): {np.round(hiphi_pelvis.apply(hips), 2).tolist()} / "
          f"{current.get('hips_offset')}")

    zones = DATASET.load_zones()
    stands = zones[zones.stands.fillna(False).astype(bool)]
    ankle = current["planted_ankle_height"] - stands.ground_shift
    print(f"HiPHI, {len(zones)} surveyed motions, {len(stands)} stand")
    print(f"typical_planted_ankle_height (m): {ankle.median():.3f} / {current['typical_planted_ankle_height']}  "
          f"(10-90% {ankle.quantile(0.1) * 100:.1f}-{ankle.quantile(0.9) * 100:.1f} cm)")

    torso = torso_offsets(args, soma_walks, hiphi_walks, hiphi)
    print(f"torso_link ({torso['bone']}), walking motions SOMA / HiPHI: {torso['walks'][0]} / {torso['walks'][1]}")
    print(f"  torso target ahead of the trunk line (deg): SOMA Chest {torso['lean'][0]:+.2f}, "
          f"HiPHI {torso['bone']} at the rest-pose offset {torso['lean'][1]:+.2f} -> pitch forward {torso['forward']:+.2f}")
    print(f"  rotation offset: {np.round(torso['rotation'], 4).tolist()} / {torso['current'][4]}")
    print(f"  position offset: {np.round(torso['position'], 4).tolist()} / {torso['current'][3]}")


if __name__ == "__main__":
    main()
