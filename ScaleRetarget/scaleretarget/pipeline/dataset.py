"""What the pipeline needs to know about a dataset: where it is, how clips are named, and its zone table.

A dataset's source is read-only. Everything derived from it goes under out: survey/ (kinematic survey,
zone table, green / yellow / red lists, reference twist flips) and one folder per batch-retarget
version, vN/ (bvh/<rel>.pkl, ood/, run.json, config snapshot). A clip is identified by rel, its path
under bvh_root (the batch run's data_path) without suffix, which is also its path under vN/bvh/.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Dataset:
    name = ""  # shown in the panel title
    key = ""  # its folder under scripts/
    loader = ""  # hydra loader config
    robot = "unitree_g1d"  # robot of the batch runs
    source = ""  # read-only root
    bvh_root = ""  # data_path of the batch runs
    suffix = ".bvh"  # of the clip files under bvh_root
    out = ""
    zones_file = "zones.parquet"
    # Config files copied into each version folder; a version is resumed only while they are unchanged.
    snapshot = ()
    # Official G1 retarget as <root>/<rel>.csv; without one the reference is our standard G1 retarget, live.
    official_csv_root = None
    mirror_suffix = ""
    group_label = "分组"  # panel column of the group field

    @property
    def survey_dir(self):
        return os.path.join(self.out, "survey")

    @property
    def zones(self):
        return os.path.join(self.survey_dir, self.zones_file)

    @property
    def twist_flips(self):
        return os.path.join(self.survey_dir, "ref_twist_flips.csv")

    def bvh_path(self, rel):
        return os.path.join(self.bvh_root, rel + self.suffix)

    def twin(self, rel):
        """The mirrored clip of rel, or the original of a mirrored one."""
        s = self.mirror_suffix
        return rel.removesuffix(s) if rel.endswith(s) else rel + s

    def load_zones(self):
        """The zone table, with at least rel, zone, reason, filename, group (e.g. capture date), category,
        description, seconds, span_expected (time from first to last source frame, which sets the resampled
        frame count), session and actor (clips of one actor in one capture)."""
        raise NotImplementedError

    def expected_frames(self, span, fps):
        """Frame count of the loader's resampling to fps of a clip spanning span seconds."""
        return (span * fps).round().astype(int).clip(lower=1) + 1

    def version_dir(self, version):
        return os.path.join(self.out, version)

    def shell(self):
        """Shell variables for scripts/retarget_version.sh."""
        snapshot = " ".join(self.snapshot)
        return "\n".join([
            f"DATASET_NAME='{self.name}'", f"SOURCE='{self.source}'", f"DATA_PATH='{self.bvh_root}'",
            f"OUT='{self.out}'", f"LOADER='{self.loader}'", f"ROBOT='{self.robot}'", f"SNAPSHOT=({snapshot})",
            f"LISTS='[{self.survey_dir}/green.txt,{self.survey_dir}/yellow.txt]'",
        ])
