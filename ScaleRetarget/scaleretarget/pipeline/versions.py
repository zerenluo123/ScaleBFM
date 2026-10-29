"""Keep two batch-retarget versions of a dataset in step: where every clip sits (dataset bvh/, OOD, or not
converted) per zone, and the OOD decisions of the earlier version carried into the later one.

report() writes <dst>/version_report.md: per zone of the current zone table, how many clips each version
holds in bvh/, in OOD and not converted, the OOD reasons, and every clip whose place differs. carry_ood()
puts each OOD clip of src into the OOD of dst with its log row: a clip dst converted is moved there
(utils/ood.py); one dst has not converted is hard-linked from src, so a carry before the batch run also
keeps the run from converting it (done_dirs). Clips already in the OOD of dst are left alone.
"""

import os

import pandas as pd

from scaleretarget.utils.ood import _write_ood_log, dataset_pkl, move_to_ood, ood_pkl, read_ood_log

ZONE_LABELS = {"green": "绿区", "yellow": "黄区", "red": "红区"}
PLACE_LABELS = {"bvh": "数据区", "ood": "OOD", "none": "未转换"}
MAX_DIFFS = 200


def places(dataset, version, rels):
    """rel -> 'bvh', 'ood' or 'none' in a version."""
    vdir = dataset.version_dir(version)
    return {rel: "bvh" if os.path.isfile(dataset_pkl(vdir, rel)) else "ood" if os.path.isfile(ood_pkl(vdir, rel)) else "none"
            for rel in rels}


def converted_rels(dataset, version):
    """Every rel with a pkl in a version's bvh/ or OOD, also clips the zone table no longer lists."""
    vdir = dataset.version_dir(version)
    out = set()
    for sub in (os.path.join(vdir, "bvh"), os.path.join(vdir, "ood", "bvh")):
        for root, _, files in os.walk(sub):
            out.update(os.path.relpath(os.path.join(root, f), sub)[:-4] for f in files if f.endswith(".pkl"))
    return out


def report(dataset, src, dst):
    """Markdown table comparing the two versions; also returns whether every clip sits in the same place."""
    zones = dataset.load_zones()
    zone_of = dict(zip(zones.rel, zones.zone))
    rels = sorted(set(zones.rel) | converted_rels(dataset, src) | converted_rels(dataset, dst))
    a, b = places(dataset, src, rels), places(dataset, dst, rels)
    df = pd.DataFrame({"rel": rels, "zone": [zone_of.get(r, "missing") for r in rels],
                       src: [a[r] for r in rels], dst: [b[r] for r in rels]})

    lines = [f"# {dataset.name}: {src} → {dst}", "",
             f"| 分区（当前分区表） | 条数 | {src} 数据区 | {src} OOD | {src} 未转换 | {dst} 数据区 | {dst} OOD | {dst} 未转换 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    zone_rows = [(z, df.zone == z) for z in ("green", "yellow", "red")]
    if (df.zone == "missing").any():
        zone_rows.append(("missing", df.zone == "missing"))
    zone_rows.append(("total", df.zone != "missing"))
    for z, mask in zone_rows:
        part = df[mask]
        cells = [str(int((part[v] == p).sum())) for v in (src, dst) for p in ("bvh", "ood", "none")]
        label = {"missing": "不在分区表", "total": "合计"}.get(z, ZONE_LABELS.get(z, z))
        lines.append(f"| {label} | {len(part)} | " + " | ".join(cells) + " |")

    reasons = pd.concat([read_ood_log(dataset.version_dir(v)).assign(version=v) for v in (src, dst)])
    if len(reasons):
        lines += ["", f"| OOD 理由 | {src} | {dst} |", "|---|---:|---:|"]
        table = reasons.groupby(["reason", "version"]).size().unstack(fill_value=0).reindex(columns=[src, dst], fill_value=0)
        for reason, row in table.iterrows():
            lines.append(f"| {reason} | {row[src]} | {row[dst]} |")

    diff = df[df[src] != df[dst]]
    same = diff.empty
    lines += ["", "结论：每段的位置两版完全一致。" if same else f"结论：{len(diff)} 段位置不同。"]
    if not same:
        lines += ["", f"| 片段 | 分区 | {src} | {dst} |", "|---|---|---|---|"]
        for r in diff.head(MAX_DIFFS).itertuples():
            lines.append(f"| {r.rel} | {ZONE_LABELS.get(r.zone, r.zone)} | {PLACE_LABELS[getattr(r, src)]} | {PLACE_LABELS[getattr(r, dst)]} |")
        if len(diff) > MAX_DIFFS:
            lines.append(f"| …… 另外 {len(diff) - MAX_DIFFS} 段 | | | |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(dataset.version_dir(dst), "version_report.md"), "w") as f:
        f.write(text)
    return text, same


def carry_ood(dataset, src, dst, apply=False):
    """Put every OOD clip of src into the OOD of dst; returns (moved, linked, already) counts."""
    sdir, ddir = dataset.version_dir(src), dataset.version_dir(dst)
    log = read_ood_log(sdir)
    moved, linked, already = [], [], 0
    for r in log.itertuples():
        if os.path.isfile(ood_pkl(ddir, r.rel)):
            already += 1
        elif os.path.isfile(dataset_pkl(ddir, r.rel)):
            moved.append(r)
        elif os.path.isfile(ood_pkl(sdir, r.rel)):
            linked.append(r)
    if apply:
        for r in moved:
            move_to_ood(ddir, r.rel, r.zone, r.reason)
        for r in linked:
            target = ood_pkl(ddir, r.rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.link(ood_pkl(sdir, r.rel), target)
        if linked:
            rows = log[log.rel.isin([r.rel for r in linked])]
            _write_ood_log(ddir, pd.concat([read_ood_log(ddir), rows]).drop_duplicates("rel", keep="last"))
    return len(moved), len(linked), already
