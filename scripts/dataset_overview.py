"""Dataset overview: annotated vertebrae per split, region and Genant fracture status.

Joins the GT centroid JSONs with the OSF Genant CSV on (series, vertebra label).
Writes results/sanity/dataset_summary.csv, results/sanity/fracture_join.csv and
results/figures/dataset_overview.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bonelm import io  # noqa: E402

STATUS_COLORS = {
    "healthy": "#8ecae6",
    "below_threshold": "#ffb703",
    "fractured": "#d62828",
    "ungraded": "#bdbdbd",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/verse20")
    ap.add_argument(
        "--fracture-csv", default="data/fracture/Genant_Fractures_thoracolumbar_reference.csv"
    )
    ap.add_argument("--min-grade", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)

    rows = []
    for c in io.find_cases(args.data_root):
        _, ctds = io.load_centroid_json(c.ctd)
        rows += [dict(split=c.split, subject=c.subject, series=c.series, label=lab) for lab in ctds]
    v = pd.DataFrame(rows)
    g = io.load_genant_csv(args.fracture_csv)
    v = v.merge(g[["series", "label", "grade"]], on=["series", "label"], how="left")
    v["vertebra"] = v.label.map(io.VERSE_LABELS)
    v["region"] = v.label.map(io.region_of)
    v["status"] = [io.fracture_status(x, args.min_grade) for x in v.grade]
    v["series_in_csv"] = v.series.isin(set(g.series))
    (out / "sanity").mkdir(parents=True, exist_ok=True)
    v.to_csv(out / "sanity" / "fracture_join.csv", index=False)

    summ = (
        v.groupby("split")
        .agg(
            scans=("series", "nunique"),
            patients=("subject", "nunique"),
            vertebrae=("label", "size"),
        )
        .join(pd.crosstab(v.split, v.region))
        .join(pd.crosstab(v.split, v.status))
    )
    summ.to_csv(out / "sanity" / "dataset_summary.csv")
    print(summ.to_string())
    print("\nraw grade codes:\n", v.grade.value_counts(dropna=False).sort_index().to_string())

    # T13 (label 28) sits between T12 and L1
    order = sorted(v.label.unique(), key=lambda x: 19.5 if x == 28 else x)
    splits = sorted(v.split.unique())
    fig, axes = plt.subplots(
        len(splits), 1, figsize=(11, 2.4 * len(splits)), sharex=True, facecolor="white"
    )
    for ax, sp in zip(axes, splits, strict=True):
        ct = pd.crosstab(v[v.split == sp].label, v[v.split == sp].status)
        ct = ct.reindex(index=order, columns=list(STATUS_COLORS), fill_value=0)
        bottom = None
        for st, col in STATUS_COLORS.items():
            ax.bar(range(len(order)), ct[st], bottom=bottom, color=col, label=st, width=0.8)
            bottom = ct[st] if bottom is None else bottom + ct[st]
        s = summ.loc[sp]
        ax.set_title(
            f"{sp}: {s.scans} scans, {s.patients} patients, {s.vertebrae} vertebrae",
            fontsize=9,
            loc="left",
        )
        ax.set_ylabel("vertebrae", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(range(len(order)), [io.VERSE_LABELS[x] for x in order], fontsize=7)
    axes[0].legend(
        fontsize=7,
        ncol=4,
        frameon=False,
        loc="upper left",
        title=f"Genant status (fractured = grade >= {args.min_grade})",
        title_fontsize=7,
    )
    fig.tight_layout()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figures" / "dataset_overview.png", dpi=150)
    print("saved", out / "figures" / "dataset_overview.png")


if __name__ == "__main__":
    main()
