"""Figures: centroid overlays on sagittal CT slices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from bonelm.io import VERSE_LABELS, mm_to_voxel

HU_WINDOW = (-400.0, 1200.0)


def sagittal_overlay(
    ax: plt.Axes,
    ct: nib.spatialimages.SpatialImage,
    mask: nib.spatialimages.SpatialImage | None,
    points_mm: dict[str, dict[int, np.ndarray]],
    title: str = "",
    slab_mm: float = 15.0,
) -> None:
    """Draw a sagittal CT slice through the median x of the first point set.

    ``points_mm`` maps a legend name to ``{label: xyz_mm}``; the first set is labelled.
    Everything is reoriented to RAS so the slice is a true sagittal plane; points are
    projected with the inverse affine, and points further than ``slab_mm`` from the
    slice are drawn faded.
    """
    ct = nib.as_closest_canonical(ct)
    arr = np.asanyarray(ct.dataobj)
    sx, sy, sz = nib.affines.voxel_sizes(ct.affine)
    names = list(points_mm)
    ref = np.array(list(points_mm[names[0]].values()))
    i = int(np.clip(np.round(np.median(mm_to_voxel(ct.affine, ref)[:, 0])), 0, arr.shape[0] - 1))

    sl = arr[i].astype(np.float32).T  # rows = S (k), cols = A (j)
    extent = (0, arr.shape[1] * sy, 0, arr.shape[2] * sz)
    ax.imshow(
        sl,
        cmap="gray",
        vmin=HU_WINDOW[0],
        vmax=HU_WINDOW[1],
        origin="lower",
        extent=extent,
        interpolation="bilinear",
    )
    if mask is not None:
        m = np.asanyarray(nib.as_closest_canonical(mask).dataobj)[i].T
        ax.contour(
            np.linspace(0, extent[1], m.shape[1]),
            np.linspace(0, extent[3], m.shape[0]),
            (m > 0).astype(float),
            levels=[0.5],
            colors="#4cc9f0",
            linewidths=0.6,
        )

    styles = [
        dict(marker="o", s=28, c="#ff3b30", edgecolors="white", linewidths=0.6),
        dict(marker="x", s=30, c="#ffd60a", linewidths=1.2),
    ]
    for n, name in enumerate(names):
        pts = points_mm[name]
        if not pts:
            continue
        labs = sorted(pts)
        vox = mm_to_voxel(ct.affine, np.array([pts[k] for k in labs]))
        near = np.abs(vox[:, 0] - i) * sx <= slab_mm
        st = styles[n % len(styles)]
        ax.scatter(vox[near, 1] * sy, vox[near, 2] * sz, label=name, zorder=3, **st)
        ax.scatter(vox[~near, 1] * sy, vox[~near, 2] * sz, alpha=0.35, zorder=3, **st)
        if n == 0:
            for lab, v in zip(labs, vox, strict=True):
                ax.annotate(
                    VERSE_LABELS.get(lab, str(lab)),
                    (v[1] * sy, v[2] * sz),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=7,
                    color="white",
                    va="center",
                    zorder=4,
                )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("anterior  (mm)", fontsize=8)
    ax.set_ylabel("superior  (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
