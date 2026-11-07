"""Sanity check of the centroid convention on every downloaded case.

For each case, compare the ground-truth JSON centroid (voxel -> mm through the mask
affine) with the centre of mass of the same label in the ground-truth mask. If the
orientation/voxel convention were wrong, distances would be tens to hundreds of mm.

Writes results/sanity/centroid_vs_mask.csv and results/sanity/centroid_vs_mask_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
import pandas as pd

from bonelm import io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/verse20")
    ap.add_argument("--out", default="results/sanity")
    ap.add_argument(
        "--from-csv",
        action="store_true",
        help="only rebuild the summary from an existing centroid_vs_mask.csv",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        summarize(pd.read_csv(out / "centroid_vs_mask.csv"), out)
        return

    rows = []
    for c in io.find_cases(args.data_root):
        msk = nib.load(c.mask)
        json_codes, _ = io.load_centroid_json(c.ctd)
        img_codes = nio.aff2axcodes(msk.affine)
        gt = io.load_gt_centroids_mm(c.ctd, msk)
        com = io.mask_centroids_mm(msk)
        sp = nib.affines.voxel_sizes(msk.affine)
        for lab in sorted(set(gt) | set(com)):
            r = dict(
                split=c.split,
                subject=c.subject,
                series=c.series,
                label=lab,
                vertebra=io.VERSE_LABELS.get(lab, str(lab)),
                json_axcodes="".join(json_codes),
                img_axcodes="".join(img_codes),
                spacing=" x ".join(f"{s:.2f}" for s in sp),
                in_json=lab in gt,
                in_mask=lab in com,
            )
            if lab in gt and lab in com:
                # displacement expressed in RAS mm, so the AP component has a fixed meaning
                d = com[lab] - gt[lab]
                r.update(
                    dist_mm=float(np.linalg.norm(d)),
                    d_R=float(d[0]),
                    d_A=float(d[1]),
                    d_S=float(d[2]),
                )
            rows.append(r)
        print(f"{c.split}/{c.series}: {len(gt)} json, {len(com)} mask labels")

    df = pd.DataFrame(rows)
    df.to_csv(out / "centroid_vs_mask.csv", index=False)
    summarize(df, out)


def summarize(df: pd.DataFrame, out: Path) -> None:
    """Write the JSON summary and a per-group table (C1 / other cervical / thoracic / lumbar)."""
    both = df.dropna(subset=["dist_mm"]).copy()
    summary = {
        "n_cases": int(df.series.nunique()),
        "n_cases_json_axcodes_differ_from_image": int(
            df.groupby("series")
            .apply(lambda g: (g.json_axcodes != g.img_axcodes).any(), include_groups=False)
            .sum()
        ),
        "n_vertebrae_compared": len(both),
        "n_labels_json_only": int((df.in_json & ~df.in_mask).sum()),
        "n_labels_mask_only": int((~df.in_json & df.in_mask).sum()),
        "dist_mm": {
            q: float(both.dist_mm.quantile(p))
            for q, p in [
                ("min", 0),
                ("p25", 0.25),
                ("median", 0.5),
                ("p75", 0.75),
                ("p95", 0.95),
                ("max", 1),
            ]
        },
        "mean_offset_RAS_mm (mask COM - json)": [
            float(both[k].mean()) for k in ("d_R", "d_A", "d_S")
        ],
    }
    (out / "centroid_vs_mask_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # C1 (atlas) has no vertebral body, so it is reported separately
    both["group"] = ["C1 (atlas)" if lab == 1 else io.region_of(int(lab)) for lab in both.label]
    tab = (
        both.groupby("group")
        .agg(
            n=("dist_mm", "size"),
            median_dist_mm=("dist_mm", "median"),
            p95_dist_mm=("dist_mm", lambda s: s.quantile(0.95)),
            median_d_R=("d_R", "median"),
            median_d_A=("d_A", "median"),
            median_d_S=("d_S", "median"),
        )
        .round(2)
    )
    tab.to_csv(out / "centroid_vs_mask_by_region.csv")
    print(tab.to_string())


if __name__ == "__main__":
    main()
