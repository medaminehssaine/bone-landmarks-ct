"""Loading VerSe cases: CT, vertebra masks, centroid JSON, fracture grades.

Coordinate conventions (checked against anjany/verse ``utils/data_utilities.py``):

* The subject-based VerSe centroid JSON starts with ``{"direction": [a, b, c]}``, the
  axis codes of the voxel grid the coordinates live in. ``X, Y, Z`` are *voxel*
  coordinates along array axes 0, 1, 2 of that grid (not mm).
* In practice ``direction`` equals the axis codes of the CT/mask affine, so the values
  are voxel indices of the image array. If they ever differ, we permute/flip into the
  image grid; a flip maps index ``i`` to ``n - 1 - i`` (nibabel convention). The VerSe
  helper ``reorient_centroids_to`` uses ``n - i``, which is one voxel off; we do not
  copy that.
* Everything downstream is in **world mm**: ``affine @ [i, j, k, 1]`` (nibabel / RAS+).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
import pandas as pd

# VerSe label index -> name. 26/27 (sacrum, coccyx) are not annotated in VerSe.
VERSE_LABELS: dict[int, str] = {
    **{i: f"C{i}" for i in range(1, 8)},
    **{i + 7: f"T{i}" for i in range(1, 13)},
    **{i + 19: f"L{i}" for i in range(1, 7)},
    26: "Sacrum",
    27: "Coccyx",
    28: "T13",
}
NAME_TO_LABEL: dict[str, int] = {v: k for k, v in VERSE_LABELS.items()}


def region_of(label: int) -> str:
    """Return 'cervical', 'thoracic' or 'lumbar' for a VerSe vertebra label."""
    if 1 <= label <= 7:
        return "cervical"
    if 8 <= label <= 19 or label == 28:
        return "thoracic"
    if 20 <= label <= 25:
        return "lumbar"
    return "other"


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

_CTD_SUFFIX = "_seg-subreg_ctd.json"
_MSK_SUFFIX = "_seg-vert_msk.nii.gz"


@dataclass(frozen=True)
class Case:
    """One VerSe image series (a patient may have several)."""

    split: str  # '01_training' | '02_validation' | '03_test'
    subject: str  # e.g. 'sub-verse588' (patient-level cluster id)
    series: str  # e.g. 'verse588', 'verse512', 'gl003' (matches fracture CSV ids)
    ct: Path
    mask: Path
    ctd: Path


def series_id(prefix: str) -> str:
    """Series id from a BIDS file prefix: 'split-verseNNN' wins, else the subject id."""
    m = re.search(r"split-(verse\d+)", prefix)
    if m:
        return m.group(1)
    m = re.match(r"sub-([a-z]+\d+)", prefix)
    if not m:
        raise ValueError(f"cannot parse series id from {prefix!r}")
    return m.group(1)


def find_cases(data_root: str | Path, split: str | None = None) -> list[Case]:
    """List all cases under ``data_root/<split>/derivatives`` that have a centroid JSON.

    CT paths are returned even if the CT has not been downloaded (check ``case.ct.exists()``).
    """
    root = Path(data_root)
    splits = [split] if split else sorted(p.name for p in root.iterdir() if p.is_dir())
    cases: list[Case] = []
    for sp in splits:
        for ctd in sorted((root / sp / "derivatives").glob(f"*/*{_CTD_SUFFIX}")):
            prefix = ctd.name[: -len(_CTD_SUFFIX)]
            subject = ctd.parent.name
            cases.append(
                Case(
                    split=sp,
                    subject=subject,
                    series=series_id(prefix),
                    ct=root / sp / "rawdata" / subject / f"{prefix}_ct.nii.gz",
                    mask=ctd.parent / f"{prefix}{_MSK_SUFFIX}",
                    ctd=ctd,
                )
            )
    return cases


# ---------------------------------------------------------------------------
# Centroids
# ---------------------------------------------------------------------------


def load_centroid_json(path: str | Path) -> tuple[tuple[str, str, str], dict[int, np.ndarray]]:
    """Read a VerSe subject-based centroid JSON.

    Returns ``(axcodes, {label: voxel_xyz})``. Entries with NaN coordinates are skipped.
    """
    with open(path) as f:
        items = json.load(f)
    axcodes: tuple[str, str, str] | None = None
    ctds: dict[int, np.ndarray] = {}
    for d in items:
        if "direction" in d:
            axcodes = tuple(d["direction"])  # type: ignore[assignment]
            continue
        xyz = np.array([d["X"], d["Y"], d["Z"]], dtype=float)
        if np.isnan(xyz).any():
            continue
        ctds[int(d["label"])] = xyz
    if axcodes is None:
        raise ValueError(f"{path}: no 'direction' entry")
    return axcodes, ctds


def reorient_voxel_coords(
    xyz: np.ndarray,
    from_axcodes: tuple[str, ...],
    to_axcodes: tuple[str, ...],
    to_shape: tuple[int, ...],
) -> np.ndarray:
    """Map voxel coords from a grid with ``from_axcodes`` into the image grid ``to_axcodes``.

    ``to_shape`` is the shape of the target (image) array. Works on (3,) or (N, 3).
    """
    pts = np.atleast_2d(np.asarray(xyz, dtype=float))
    if tuple(from_axcodes) == tuple(to_axcodes):
        return pts.reshape(np.shape(xyz)).copy()
    # ornt_transform(start, end)[i] = (axis in end, flip) for axis i of start
    trans = nio.ornt_transform(nio.axcodes2ornt(from_axcodes), nio.axcodes2ornt(to_axcodes))
    out = np.empty_like(pts)
    for src_ax, (dst_ax, flip) in enumerate(trans.astype(int)):
        v = pts[:, src_ax]
        out[:, dst_ax] = (to_shape[dst_ax] - 1) - v if flip == -1 else v
    return out.reshape(np.shape(xyz))


def voxel_to_mm(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """Voxel index coordinates -> world mm via the 4x4 affine. Accepts (3,) or (N, 3)."""
    return nib.affines.apply_affine(affine, np.asarray(ijk, dtype=float))


def mm_to_voxel(affine: np.ndarray, xyz_mm: np.ndarray) -> np.ndarray:
    """World mm -> (fractional) voxel index coordinates."""
    return nib.affines.apply_affine(np.linalg.inv(affine), np.asarray(xyz_mm, dtype=float))


def load_gt_centroids_mm(
    ctd_path: str | Path, ref_img: nib.spatialimages.SpatialImage
) -> dict[int, np.ndarray]:
    """Ground-truth centroids in world mm, using ``ref_img`` (the CT or mask) for the grid."""
    axcodes, ctds = load_centroid_json(ctd_path)
    img_axcodes = nio.aff2axcodes(ref_img.affine)
    out = {}
    for label, xyz in ctds.items():
        ijk = reorient_voxel_coords(xyz, axcodes, img_axcodes, ref_img.shape[:3])
        out[label] = voxel_to_mm(ref_img.affine, ijk)
    return out


def mask_centroids_mm(
    mask_img: nib.spatialimages.SpatialImage, labels: list[int] | None = None
) -> dict[int, np.ndarray]:
    """Centre of mass of each label in a label map, in world mm.

    Labels absent from the mask are simply not returned (caller handles missing ones).
    Uses ``np.bincount`` over foreground voxels (``scipy.ndimage.center_of_mass`` with a
    label list is orders of magnitude slower on 512x512xN volumes).
    """
    data = np.asanyarray(mask_img.dataobj)
    ijk = np.nonzero(data)
    if len(ijk[0]) == 0:
        return {}
    lab = data[ijk].astype(np.int64)
    if lab.min() < 0:
        raise ValueError("negative labels in mask")
    count = np.bincount(lab)
    sums = [np.bincount(lab, weights=ax.astype(np.float64)) for ax in ijk]
    wanted = set(labels) if labels is not None else None
    out = {}
    for v in np.flatnonzero(count):
        if wanted is not None and int(v) not in wanted:
            continue
        c = np.array([s[v] / count[v] for s in sums])
        out[int(v)] = voxel_to_mm(mask_img.affine, c)
    return out


# ---------------------------------------------------------------------------
# Fracture grades (Genant), OSF node 4skx2
# ---------------------------------------------------------------------------

GENANT_GRADES = {0: "none", 1: "mild", 2: "moderate", 3: "severe"}


def load_genant_csv(path: str | Path) -> pd.DataFrame:
    """Long table ``series, label, vertebra, grade`` from the VerSe Genant CSV.

    IDs in the CSV are zero-padded series numbers ('004' -> 'verse004'). Empty cells
    (vertebra not graded / not in FOV) are dropped. Codes outside 0-3 are kept as-is;
    use :func:`fracture_status` to decide how to treat them.
    """
    df = pd.read_csv(path, sep=";", dtype={"ID": str})
    long = df.melt(id_vars="ID", var_name="vertebra", value_name="grade").dropna()
    long["series"] = "verse" + long["ID"].str.zfill(3)
    long["label"] = long["vertebra"].map(NAME_TO_LABEL)
    long["grade"] = long["grade"].astype(int)
    return long[["series", "label", "vertebra", "grade"]].reset_index(drop=True)


def fracture_status(grade: float | None, min_grade: int = 1) -> str:
    """Map a Genant grade to 'healthy' (0), 'fractured' (>= min_grade), 'below_threshold'
    (0 < grade < min_grade) or 'ungraded' (missing, or a non-Genant code such as 5 or 8
    whose meaning is not documented with the CSV)."""
    if grade is None or (isinstance(grade, float) and np.isnan(grade)):
        return "ungraded"
    g = int(grade)
    if g not in GENANT_GRADES:
        return "ungraded"
    if g == 0:
        return "healthy"
    return "fractured" if g >= min_grade else "below_threshold"
