"""Unit and integration tests for the 3D CBCT segmentation, nerve-distance, and mesh pipeline."""

from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
from cbct_pipeline import (
    parse_nifti_bytes,
    write_nifti_gz_bytes,
    postprocess_and_compute_findings,
    export_cbct_meshes,
)


def _build_synthetic_cbct_labelmap() -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Creates a 40x40x40 3D labelmap with 0.5 mm isotropic spacing where:
    - Right hemimandible (x < 20):
      - Mandibular canal (label=5) spans x=[6..10], y=[15..20], z=[6..10] (top surface at z=10)
      - Lower teeth (label=4) spans x=[6..12], y=[15..22], z=[14..22] (bottom surface at z=14)
      -> Exact vertical separation = (14 - 10) * 0.5 mm = 2.00 mm!
    - Left hemimandible (x >= 20):
      - Mandibular canal (label=5) spans x=[26..30], y=[15..20], z=[8..12] (top surface at z=12)
      - Lower teeth (label=4) spans x=[26..32], y=[15..22], z=[14..22] (bottom surface at z=14)
      -> Exact vertical separation = (14 - 12) * 0.5 mm = 1.00 mm!
    """
    vol = np.zeros((40, 40, 40), dtype=np.int16)
    # Maxilla (1) and Mandible (2) large enough to pass 60 mm^3 island filter (> 480 voxels at 0.125 mm^3/vox)
    vol[5:35, 10:30, 24:34] = 1
    vol[5:35, 10:30, 2:13] = 2
    vol[8:32, 12:26, 23:28] = 3

    # Right lower teeth (4) and right canal (5)
    vol[6:16, 12:24, 14:22] = 4  # 10 * 12 * 8 = 960 voxels (> 480)
    vol[7:10, 15:19, 6:11] = 5   # top surface at z = 10

    # Left lower teeth (4) and left canal (5)
    vol[24:34, 12:24, 14:22] = 4  # 10 * 12 * 8 = 960 voxels (> 480)
    vol[27:30, 15:19, 8:13] = 5   # top surface at z = 12

    return vol, (0.5, 0.5, 0.5)


def test_nifti_roundtrip_and_spacing():
    """Verify NIfTI-1 (.nii.gz) writer and parser preserve 3D array and physical spacing."""
    vol, spacing = _build_synthetic_cbct_labelmap()
    gz_bytes = write_nifti_gz_bytes(vol, spacing)
    parsed_vol, parsed_spacing = parse_nifti_bytes(gz_bytes, is_gzipped=True)

    assert parsed_vol.shape == (40, 40, 40)
    assert np.allclose(parsed_spacing, (0.5, 0.5, 0.5))
    assert np.array_equal(parsed_vol, vol)


def test_analytical_3d_nerve_distance_mm():
    """Verify exact 3D Euclidean distance (mm) between lower teeth (4) and mandibular canal (5)."""
    vol, spacing = _build_synthetic_cbct_labelmap()
    seg_clean, stats, findings = postprocess_and_compute_findings(vol, spacing, apply_island_filter=True)

    assert stats["mandible"]["voxel_count"] > 0
    assert stats["lower_teeth"]["voxel_count"] > 0
    assert stats["mandibular_canal"]["voxel_count"] > 0
    assert len(findings) == 2

    by_hemi = {f["hemimandible"]: f for f in findings}
    assert by_hemi["right"]["min_distance_mm"] == pytest.approx(2.0, abs=0.01)
    assert by_hemi["left"]["min_distance_mm"] == pytest.approx(1.0, abs=0.01)
    assert len(by_hemi["right"]["bbox_3d_voxel"]) == 6


def test_cbct_mesh_stl_and_glb_export(tmp_path: Path):
    """Verify 3D mesh export produces valid binary .glb (glTF 2.0) and .stl files."""
    vol, spacing = _build_synthetic_cbct_labelmap()
    glb_url, stl_urls = export_cbct_meshes(vol, spacing, tmp_path, "test_cbct")

    glb_file = tmp_path / Path(glb_url).name
    assert glb_file.exists()
    raw_glb = glb_file.read_bytes()
    assert raw_glb[:4] == b"glTF"
    assert len(stl_urls) == 5
    for cname, url in stl_urls.items():
        stl_file = tmp_path / Path(url).name
        assert stl_file.exists()
        assert stl_file.stat().st_size > 84


def test_cbct_unsegmented_volume_without_weights_returns_503(monkeypatch):
    """Verify raw unsegmented CBCT volume fails closed with HTTP 503 when nnU-Net weights are absent."""
    monkeypatch.delenv("VISCAN_CBCT_WEIGHTS_DIR", raising=False)
    # Create an intensity volume with values outside {0..5} so it requires live neural segmentation
    rng = np.random.default_rng(42)
    raw_hu_vol = rng.integers(-500, 1800, size=(16, 16, 16), dtype=np.int16)
    nii_bytes = write_nifti_gz_bytes(raw_hu_vol, (0.5, 0.5, 0.5))

    client = TestClient(app_module.app)
    resp = client.post(
        "/analyze",
        files={"file": ("raw_cbct_scan.nii.gz", nii_bytes, "application/gzip")},
    )
    assert resp.status_code == 503
    assert "DentalSegmentator" in resp.json()["detail"]


def test_cbct_analyze_endpoint_returns_3d_schema():
    """Verify uploading a 3D CBCT NIfTI to POST /analyze returns the full 3D CBCT schema and meshes."""
    vol, spacing = _build_synthetic_cbct_labelmap()
    nii_bytes = write_nifti_gz_bytes(vol, spacing)

    client = TestClient(app_module.app)
    resp = client.post(
        "/analyze",
        files={"file": ("synthetic_cbct.nii.gz", nii_bytes, "application/gzip")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["modality"] == "CBCT"
    assert data["model"]["pretrained_as_is"] is True
    assert data["mesh_glb_url"].endswith(".glb")
    assert len(data["findings"]) == 2
    assert "disclaimer" in data


def test_different_inputs_produce_distinct_3d_glb_meshes(tmp_path: Path):
    """Verify different CBCT volumes and different 2D OPG images produce distinct 3D .glb meshes."""
    from PIL import Image
    from cbct_pipeline import export_opg_3d_relief_glb

    vol1, spacing = _build_synthetic_cbct_labelmap()
    vol2 = vol1.copy()
    vol2[20:, :, :] = 0  # Right hemimandible only

    glb1_url, _ = export_cbct_meshes(vol1, spacing, tmp_path, "cbct_1")
    glb2_url, _ = export_cbct_meshes(vol2, spacing, tmp_path, "cbct_2")
    b1 = (tmp_path / Path(glb1_url).name).read_bytes()
    b2 = (tmp_path / Path(glb2_url).name).read_bytes()
    assert b1 != b2

    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    img1 = Image.fromarray(rng1.integers(20, 220, size=(200, 400, 3), dtype=np.uint8))
    img2 = Image.fromarray(rng2.integers(20, 220, size=(200, 400, 3), dtype=np.uint8))
    opg1_url = export_opg_3d_relief_glb(img1, [{"class": "caries", "bbox": [30, 40, 80, 90]}], tmp_path, "opg_1")
    opg2_url = export_opg_3d_relief_glb(img2, [{"class": "impacted_tooth", "bbox": [120, 50, 190, 140]}], tmp_path, "opg_2")
    ob1 = (tmp_path / Path(opg1_url).name).read_bytes()
    ob2 = (tmp_path / Path(opg2_url).name).read_bytes()
    assert ob1 != ob2
    assert ob1 != b1
