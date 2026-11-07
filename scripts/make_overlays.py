"""Overlay figure: GT centroids (JSON) and GT-mask centres of mass on sagittal CT slices.

Uses every case whose CT has been downloaded (``fetch_verse.py --part ct``), up to --n.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import nibabel.orientations as nio  # noqa: E402

from bonelm import io, viz  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/verse20")
    ap.add_argument("--split", default="01_training")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="results/figures/overlay_gt_centroids.png")
    args = ap.parse_args()

    cases = [c for c in io.find_cases(args.data_root, args.split) if c.ct.exists()][: args.n]
    if not cases:
        raise SystemExit("no CTs found; run scripts/fetch_verse.py --part ct first")
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 7.5), facecolor="white")
    axes = [axes] if len(cases) == 1 else list(axes)
    for ax, c in zip(axes, cases, strict=True):
        ct, msk = nib.load(c.ct), nib.load(c.mask)
        sp = nib.affines.voxel_sizes(ct.affine)
        viz.sagittal_overlay(
            ax,
            ct,
            msk,
            {
                "GT centroid (JSON)": io.load_gt_centroids_mm(c.ctd, ct),
                "GT mask centre of mass": io.mask_centroids_mm(msk),
            },
            title=f"{c.series}  ({''.join(nio.aff2axcodes(ct.affine))}, "
            f"{sp[0]:.2f}x{sp[1]:.2f}x{sp[2]:.2f} mm)",
        )
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("saved", args.out, [c.series for c in cases])


if __name__ == "__main__":
    main()
