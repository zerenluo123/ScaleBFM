import numpy as np
from scipy.spatial.transform import Rotation as R

from scaleretarget.loader.base_loader import BaseLoader
from scaleretarget.utils.lafan_vendor.extract import read_bvh_multi_channel
from scaleretarget.utils.lafan_vendor.utils import quat_fk, quat_mul
from scaleretarget.utils.resampling import resample_bvh_motion


def rest_pose(bvh_data):
    """Joint positions with every joint rotation zero (HiPHI: an upright T-pose), root at the origin.

    HiPHI writes a translation for every joint. The limbs keep one length throughout (not always the header
    OFFSET), which is used. The solver shortens the spine, neck and head joints by up to ~1 cm each with the
    pose (crouched motions most), so their per-motion lengths differ by ~3 cm over the trunk between motions
    of one skeleton; they take the header OFFSET, about their longest, the same for every motion of it.
    """
    local = bvh_data.pos[0].copy()
    varies = np.ptp(bvh_data.pos, axis=0).max(axis=-1) > 1e-3
    local[varies] = bvh_data.offsets[varies]
    local[0] = 0.0
    quats = np.zeros((1, len(bvh_data.bones), 4))
    quats[..., 0] = 1.0
    return quat_fk(quats, local[None], bvh_data.parents)[1][0]


def segment_ratios(bones, parents, rest, reference):
    """Per-joint factor on its local translation that rescales the skeleton to the reference distances.

    reference[bone] is the distance from bone to its nearest reference ancestor (or the root), compared with
    the same distance on the rest pose, so that a performer's proportions do not depend on what the motion
    does (the trunk's mid hip joints to shoulder distance does). A reference bone's factor applies to the
    joints from that ancestor down to it: joints above reference bones take the mean factor of those below
    them, the others their parent's factor. Returns the factors and the per-bone distance ratios.
    """
    own = {}
    for bone, distance in reference.items():
        i = bones.index(bone)
        top = parents[i]
        while parents[top] >= 0 and bones[top] not in reference:
            top = parents[top]
        own[i] = distance / np.linalg.norm(rest[i] - rest[top])
    below = {i: [] for i in range(len(bones))}
    for i, r in own.items():
        j = parents[i]
        while j >= 0 and j not in own:
            below[j].append(r)
            j = parents[j]
    ratios = np.ones(len(bones))
    for i in range(len(bones)):  # BVH order: parents come first
        if i in own:
            ratios[i] = own[i]
        elif below[i]:
            ratios[i] = np.mean(below[i])
        elif parents[i] >= 0:
            ratios[i] = ratios[parents[i]]
    return ratios, {bones[i]: r for i, r in own.items()}


STRAIGHT_LEG = 0.97  # hip-to-ankle distance over thigh + shin length from which a leg counts as straight


def planted_ankle_height(positions, bones, leg_length, min_frames):
    """Height of the ankle the motion stands on with a straight leg, None if it does so for under min_frames.

    10th percentile over the frames where the lower ankle's leg is straight. Restricting to those frames
    leaves out sitting and kneeling (ankle on the floor) and crouching on the toes (ankle lifted).
    """
    feet = np.array([bones.index(f"{s}Foot") for s in ("Left", "Right")])
    hips = np.array([bones.index(f"{s}UpLeg") for s in ("Left", "Right")])
    lower = positions[:, feet, 2].argmin(axis=1)
    frame = np.arange(len(positions))
    ankle = positions[frame, feet[lower]]
    standing = np.linalg.norm(positions[frame, hips[lower]] - ankle, axis=-1) >= STRAIGHT_LEG * leg_length
    return np.percentile(ankle[standing, 2], 10) if standing.sum() >= min_frames else None


# Bones whose children all lie on their up axis (y), so that a turn about it moves no joint; a foot's one
# child, the toe, leaves it free to turn about the ankle-to-toe axis.
TRUNK = ("Spine", "Spine1", "Spine2", "Spine3", "Neck", "Neck1", "Head")
LEGS = ("LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg")
FEET = ("LeftFoot", "RightFoot")
ARMS = ("LeftArm", "RightArm")
Y = np.array([0.0, 1.0, 0.0])


def quarter_turns(angle):
    """Nearest multiple of 90 deg to angle (rad), in quarter turns 0..3."""
    return int(np.round(angle / (np.pi / 2))) % 4


def half_turns(angle):
    """A half turn (2 quarter turns) for an angle (rad) beyond 90 deg either way, else 0."""
    return 2 * int(abs(angle) > np.pi / 2)


def angle_about(axis, v):
    """Angle (rad) of v about the unit axis, from the side of it +y lies on."""
    e = Y - (Y @ axis) * axis
    e /= np.linalg.norm(e)
    return np.arctan2(np.cross(e, v) @ axis, e @ v)


def canonical_frames(bvh_data):
    """Turn the bone frames that some HiPHI captures set a quarter or half turn off the others, about axes
    the joint positions do not fix, back to the common convention, in place. Returns {bone: quarter turns}.

    Those captures (all of performers A116-A132) set the trunk and neck bones (TRUNK) half a turn off about
    their up axis: the shoulder line lies along -x of their frames instead of +x (Hips and Spine4 are not, so
    Spine, Spine4 and Neck show ~180 deg local rotations), and the torso target, on Spine2, faces backwards.
    The thighs and shins (LEGS) are half a turn off about theirs (the hip line along -x), so that the knee
    targets point backwards, the feet (FEET) half a turn off about the ankle-to-toe axis (the shin on the -y
    side of it), and the upper arms a quarter turn off about theirs (the elbow bends towards +y of the arm
    frame instead of +z).

    Half turns: over the dataset, summed over the frames, the shoulder line in a trunk bone lies within 90 deg
    of +x (a twisted spine, a turned head) or within 30 deg of -x, the hip line in a thigh or shin within 60
    deg of +x (hip and knee rotation; the knee bends about it) or near -x, and the shin in a foot within 90 deg
    of the +y side (ankle inversion) or within 60 deg of the -y side, so a bone beyond 90 deg is turned by half
    a turn. Upper arms: the elbow bend direction summed over the frames (weighted by how far the elbow is
    bent), rounded to the nearest quarter turn, only where the trunk is turned; elsewhere it is either that of
    a twisted upper arm or, for an arm kept straight, noise. The children take the inverse turn in their local
    rotation and translation, so every joint position and every other bone's frame stays the same.
    """
    index = {name: i for i, name in enumerate(bvh_data.bones)}
    quats, pos = quat_fk(bvh_data.quats, bvh_data.pos, bvh_data.parents)
    frame = lambda b: R.from_quat(quats[:, index[b]].reshape(-1, 4), scalar_first=True)  # noqa: E731
    at = lambda b: pos[:, index[b]]  # noqa: E731
    turns = {}
    shoulders = at("LeftArm") - at("RightArm")
    for bone in (b for b in TRUNK if b in index):
        v = frame(bone).inv().apply(shoulders).sum(0)
        turns[bone] = (half_turns(np.arctan2(-v[2], v[0])), Y)
    if any(k for k, _ in turns.values()):
        for bone in ARMS:
            side = bone.removesuffix("Arm")
            upper = at(f"{side}ForeArm") - at(bone)
            upper /= np.linalg.norm(upper, axis=1, keepdims=True)
            lower = at(f"{side}Hand") - at(f"{side}ForeArm")
            bend = lower - np.sum(lower * upper, axis=1, keepdims=True) * upper
            v = frame(bone).inv().apply(bend).sum(0)
            turns[bone] = (-quarter_turns(np.arctan2(v[1], v[2])) % 4, np.array([1.0, 0.0, 0.0]))
    hips = at("LeftUpLeg") - at("RightUpLeg")
    for bone in LEGS:
        v = frame(bone).inv().apply(hips).sum(0)
        turns[bone] = (half_turns(np.arctan2(-v[2], v[0])), Y)
    for bone in FEET:
        side = bone.removesuffix("Foot")
        toe = bvh_data.pos[0, index[f"{side}ToeBase"]]
        axis = toe / np.linalg.norm(toe)
        v = frame(bone).inv().apply(at(f"{side}Leg") - at(bone)).sum(0)
        turns[bone] = (half_turns(angle_about(axis, v)), axis)
    children = {i: [c for c, p in enumerate(bvh_data.parents) if p == i] for i in range(len(bvh_data.bones))}
    done = {}
    for bone, (k, axis) in turns.items():
        if k == 0:
            continue
        turn = R.from_rotvec(axis * k * np.pi / 2)
        i = index[bone]
        bvh_data.quats[:, i] = (R.from_quat(bvh_data.quats[:, i], scalar_first=True) * turn).as_quat(scalar_first=True)
        for c in children[i]:
            bvh_data.quats[:, c] = (turn.inv() * R.from_quat(bvh_data.quats[:, c], scalar_first=True)).as_quat(scalar_first=True)
            bvh_data.pos[:, c] = turn.inv().apply(bvh_data.pos[:, c])
            bvh_data.offsets[c] = turn.inv().apply(bvh_data.offsets[c])
        done[bone] = k
    return done


def remove_forearm_twist(bvh_data, global_quats):
    """Move the forearm's twist about its own axis (relative to the upper arm) into the hand, in place.

    HiPHI splits pronation between forearm and hand (forearm twist std ~30 deg), while SOMA keeps the
    forearm a pure swing of the upper arm, as the G1 elbow hinge is. Forearm direction, hand orientation
    and all positions are unchanged.
    """
    index = {name: i for i, name in enumerate(bvh_data.bones)}
    for side in ("Left", "Right"):
        arm, forearm, hand = (index[f"{side}{b}"] for b in ("Arm", "ForeArm", "Hand"))
        axis = bvh_data.offsets[hand] / np.linalg.norm(bvh_data.offsets[hand])
        upper = R.from_quat(global_quats[:, arm], scalar_first=True)
        lower = R.from_quat(global_quats[:, forearm], scalar_first=True)
        rel = (upper.inv() * lower).as_quat()  # xyzw
        twist = np.concatenate([(rel[:, :3] @ axis)[:, None] * axis, rel[:, 3:]], axis=1)
        twist = R.from_quat(twist / np.linalg.norm(twist, axis=1, keepdims=True))
        global_quats[:, forearm] = (lower * twist.inv()).as_quat(scalar_first=True)


class HiphiLoader(BaseLoader):
    """HiPHI BVH (Noitom): Y-up, centimeters, Hips root, one skeleton per actor.

    The skeleton is rescaled segment by segment, measured on its rest pose, to the BONES-SEED SOMA Uniform
    proportions (config reference_distances), the hips trajectory with the leg length, so the SOMA scale
    tables carry over.
    HiPHI's Hips is at the mid hip joints, SOMA's above and behind them; after rescaling, the Hips point
    (not the bones below it) is moved there (config hips_offset), as the robot root and the scaling about
    it follow it. The whole motion is then shifted vertically so that the ankle it stands on sits at the height the SOMA
    Uniform skeleton has, planted_ankle_height; motions that never stand on a straight leg take the
    typical HiPHI height instead.
    """

    LEGS = [[f"{s}UpLeg", f"{s}Leg", f"{s}Foot"] for s in ("Left", "Right")]

    def _load_sample(self, sample_path):

        bvh_data = read_bvh_multi_channel(sample_path)

        aligned_fps = resample_bvh_motion(bvh_data, self.target_fps)
        turned = canonical_frames(bvh_data)

        reference = self.config.reference_distances
        ratios, own = segment_ratios(bvh_data.bones, bvh_data.parents, rest_pose(bvh_data), reference)
        ratios[0] = np.mean([sum(reference[b] for b in leg) / sum(reference[b] / own[b] for b in leg) for leg in self.LEGS])
        bvh_data.pos = bvh_data.pos * ratios[None, :, None]

        global_data = quat_fk(bvh_data.quats, bvh_data.pos, bvh_data.parents)
        remove_forearm_twist(bvh_data, global_data[0])

        # Same Y-up to Z-up rotation as SomaLoader.
        rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

        positions = global_data[1] @ rotation_matrix.T / 100  # cm to m
        hips_offset = R.from_quat(global_data[0][:, 0], scalar_first=True).apply(list(self.config.hips_offset))
        positions[:, 0] += hips_offset @ rotation_matrix.T / 100
        leg_length = np.mean([reference[leg[1]] + reference[leg[2]] for leg in self.LEGS]) / 100
        planted = planted_ankle_height(positions, bvh_data.bones, leg_length, int(aligned_fps))
        shift = self.config.planted_ankle_height - (self.config.typical_planted_ankle_height if planted is None else planted)
        positions[:, :, 2] += shift

        frames = []
        for frame in range(bvh_data.pos.shape[0]):
            result = {}
            for i, bone in enumerate(bvh_data.bones):
                orientation = quat_mul(rotation_quat, global_data[0][frame, i])
                result[bone] = (positions[frame, i], orientation)
            frames.append(result)

        extras = {
            "fps": aligned_fps,
            "ground_shift": shift,
            "stands": planted is not None,
            "turned_frames": turned,
        }

        return frames, extras
