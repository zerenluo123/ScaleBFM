"""Compare two batch-retarget versions of a dataset (scaleretarget/pipeline/versions.py): a table of where every
clip sits per zone (dataset / OOD / not converted), written to <dst>/version_report.md and printed.
--carry_ood first puts every OOD clip of src into the OOD of dst (moved if dst converted it, hard-linked
from src if not); without --apply it only counts. A re-run should end with the same counts as before.

    python scripts/version_report.py lafan --src v1 --dst v2 [--carry_ood [--apply]]

scripts/retarget_version.sh writes the report against the previous version at the end of every run.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="folder under scripts/, e.g. lafan, hiphi, bones_seed")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--carry_ood", action="store_true", help="put the OOD clips of src into the OOD of dst first")
    ap.add_argument("--apply", action="store_true", help="with --carry_ood: move / link; without it only count")
    args = ap.parse_args()
    sys.path.insert(0, os.path.join(HERE, args.dataset))
    from dataset import DATASET
    from scaleretarget.pipeline.versions import carry_ood, report

    for v in (args.src, args.dst):
        if not os.path.isdir(DATASET.version_dir(v)):
            raise SystemExit(f"{DATASET.version_dir(v)} does not exist")
    if args.carry_ood:
        moved, linked, already = carry_ood(DATASET, args.src, args.dst, args.apply)
        verb = "" if args.apply else " (dry run; --apply to do it)"
        print(f"[carry_ood] {args.src} → {args.dst}: move {moved} converted clips, hard-link {linked} not converted, "
              f"{already} already in OOD{verb}")
    text, same = report(DATASET, args.src, args.dst)
    print(text)
    print(f"[report] {os.path.join(DATASET.version_dir(args.dst), 'version_report.md')}")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
