"""Tests for affine handling, centroid loading and fracture-grade parsing."""

import json

import nibabel as nib
import nibabel.orientations as nio
import numpy as np
import pytest

from bonelm import io


def _affine(spacing, origin, flips=(1, 1, 1)):
    aff = np.diag([*(np.array(spacing) * np.array(flips)), 1.0])
    aff[:3, 3] = origin
    return aff


def test_voxel_to_mm_known_spacing():
    aff = _affine((0.5, 0.8, 3.0), (-100.0, 20.0, 5.0))
    ijk = np.array([[0, 0, 0], [10, 10, 10], [2.5, 0, 1]])
    expected = np.array([[-100, 20, 5], [-95, 28, 35], [-98.75, 20, 8]])
    np.testing.assert_allclose(io.voxel_to_mm(aff, ijk), expected)
    np.testing.assert_allclose(io.mm_to_voxel(aff, expected), ijk)


def test_mask_centroid_anisotropic_and_flipped():
    # LPS-like flips, very anisotropic voxels: centroid must land at the known mm point
    aff = _affine((0.7, 0.7, 2.5), (50.0, 80.0, -30.0), flips=(-1, -1, 1))
    data = np.zeros((40, 30, 20), dtype=np.uint8)
    data[10:13, 5:9, 4:6] = 7  # centre voxel (11, 6.5, 4.5)
    data[30, 20, 15] = 20
    img = nib.Nifti1Image(data, aff)
    c = io.mask_centroids_mm(img)
    np.testing.assert_allclose(c[7], [50 - 0.7 * 11, 80 - 0.7 * 6.5, -30 + 2.5 * 4.5])
    np.testing.assert_allclose(c[20], [50 - 0.7 * 30, 80 - 0.7 * 20, -30 + 2.5 * 15])
    assert set(c) == {7, 20}


def test_distances_invariant_to_array_orientation():
    """The same anatomy stored in two array orientations must give the same mm distances."""
    aff = _affine((0.9, 0.9, 3.0), (0, 0, 0), flips=(-1, 1, -1))
    data = np.zeros((32, 28, 16), dtype=np.uint8)
    data[3:6, 4:7, 2:4] = 1
    data[20:24, 15:18, 10:12] = 2
    img_a = nib.Nifti1Image(data, aff)
    img_b = nib.as_closest_canonical(img_a)  # permuted/flipped array, same world space
    assert nio.aff2axcodes(img_a.affine) != nio.aff2axcodes(img_b.affine)
    ca, cb = io.mask_centroids_mm(img_a), io.mask_centroids_mm(img_b)
    for k in (1, 2):
        np.testing.assert_allclose(ca[k], cb[k], atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(ca[1] - ca[2]), np.linalg.norm(cb[1] - cb[2]))


@pytest.mark.parametrize("to_codes", [("P", "I", "R"), ("R", "A", "S"), ("I", "P", "L")])
def test_reorient_voxel_coords_matches_nibabel(to_codes):
    """A marked voxel must be found at the mapped index after nibabel reorients the array."""
    aff = _affine((1.0, 1.2, 2.0), (0, 0, 0), flips=(-1, -1, 1))  # LPS
    shape = (9, 11, 7)
    data = np.zeros(shape, dtype=np.uint8)
    src = np.array([2, 8, 1])
    data[tuple(src)] = 1
    from_codes = nio.aff2axcodes(aff)
    trans = nio.ornt_transform(nio.axcodes2ornt(from_codes), nio.axcodes2ornt(to_codes))
    moved = nio.apply_orientation(data, trans)
    dst = io.reorient_voxel_coords(src, from_codes, to_codes, moved.shape)
    assert moved[tuple(np.round(dst).astype(int))] == 1
    # and the world position is unchanged
    new_aff = aff @ nio.inv_ornt_aff(trans, shape)
    np.testing.assert_allclose(io.voxel_to_mm(new_aff, dst), io.voxel_to_mm(aff, src))


def test_load_gt_centroids_mm(tmp_path):
    aff = _affine((0.8, 0.8, 1.5), (-10.0, -20.0, 100.0), flips=(1, -1, 1))
    img = nib.Nifti1Image(np.zeros((50, 50, 40), dtype=np.int16), aff)
    codes = list(nio.aff2axcodes(aff))
    p = tmp_path / "x_seg-subreg_ctd.json"
    p.write_text(
        json.dumps(
            [
                {"direction": codes},
                {"label": 20, "X": 10.0, "Y": 5.0, "Z": 2.0},
                {"label": 21, "X": float("nan"), "Y": 1.0, "Z": 1.0},
            ]
        )
    )
    out = io.load_gt_centroids_mm(p, img)
    assert set(out) == {20}
    np.testing.assert_allclose(out[20], [-10 + 8.0, -20 - 4.0, 100 + 3.0])


def test_series_id():
    assert io.series_id("sub-verse500_dir-ax") == "verse500"
    assert io.series_id("sub-verse588") == "verse588"
    assert io.series_id("sub-verse400_split-verse012_dir-sag") == "verse012"
    assert io.series_id("sub-gl003_dir-ax") == "gl003"


def test_genant_csv(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("ID;T12;L1;L2;T13\n004;0;3;;\n510;;1;5;0\n")
    df = io.load_genant_csv(p)
    got = {(r.series, r.vertebra): r.grade for r in df.itertuples()}
    assert got == {
        ("verse004", "T12"): 0,
        ("verse004", "L1"): 3,
        ("verse510", "L1"): 1,
        ("verse510", "L2"): 5,
        ("verse510", "T13"): 0,
    }
    assert dict(zip(df.vertebra, df.label, strict=True)) == {
        "T12": 19,
        "L1": 20,
        "L2": 21,
        "T13": 28,
    }


def test_fracture_status():
    assert io.fracture_status(0) == "healthy"
    assert io.fracture_status(2) == "fractured"
    assert io.fracture_status(1, min_grade=2) == "below_threshold"
    assert io.fracture_status(5) == "ungraded"
    assert io.fracture_status(float("nan")) == "ungraded"


def test_region_of():
    assert io.region_of(1) == "cervical"
    assert io.region_of(19) == "thoracic"
    assert io.region_of(28) == "thoracic"
    assert io.region_of(25) == "lumbar"
