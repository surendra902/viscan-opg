"""Unit and integration tests for the 3D CBCT segmentation, provenance, nerve-distance, and mesh pipeline."""

import hashlib
import gzip
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module
import cbct_pipeline
from imaging import ImageProcessingError
from cbct_pipeline import (
    export_cbct_meshes,
    is_cbct_bytes,
    load_cbct_input,
    parse_nifti_bytes,
    postprocess_and_compute_findings,
    write_nifti_gz_bytes,
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


def _make_bundle(
    vol_seg: np.ndarray,
    hu_seed: int,
    spacing=(0.5, 0.5, 0.5),
    provenance: dict | None = None,
) -> bytes:
    """Zip bundle = grayscale volume (random HU, outside 0..5) + label map, like the demo bundles."""
    rng = np.random.default_rng(hu_seed)
    hu = rng.integers(-500, 1800, size=vol_seg.shape, dtype=np.int16)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Scan_Volume.nii.gz", write_nifti_gz_bytes(hu, spacing))
        zf.writestr("Scan_Segmentation.nii.gz", write_nifti_gz_bytes(vol_seg.astype(np.uint8), spacing))
        if provenance is not None:
            zf.writestr("provenance.json", json.dumps(provenance))
    return buf.getvalue()


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


def test_content_routing_recognizes_nifti_and_cbct_zip_only():
    """CBCT routing is content-aware and must not treat every ZIP (or PNG) as a volume."""
    vol, spacing = _build_synthetic_cbct_labelmap()
    nii = write_nifti_gz_bytes(vol, spacing)

    png_buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(png_buf, format="PNG")
    png = png_buf.getvalue()

    zip_with_nii = io.BytesIO()
    with zipfile.ZipFile(zip_with_nii, "w") as zf:
        zf.writestr("scan.nii.gz", nii)

    zip_with_png = io.BytesIO()
    with zipfile.ZipFile(zip_with_png, "w") as zf:
        zf.writestr("scan.png", png)

    assert is_cbct_bytes(nii, "no_extension") is True
    assert is_cbct_bytes(png, "scan.png") is False
    assert is_cbct_bytes(zip_with_nii.getvalue(), "bundle.dat") is True
    assert is_cbct_bytes(zip_with_png.getvalue(), "bundle.dat") is False


def test_cbct_unsegmented_volume_without_weights_returns_503(monkeypatch):
    """Verify raw unsegmented CBCT volume fails closed with HTTP 503 when nnU-Net weights are absent."""
    monkeypatch.delenv("VISCAN_CBCT_WEIGHTS_DIR", raising=False)
    rng = np.random.default_rng(42)
    raw_hu_vol = rng.integers(-500, 1800, size=(16, 16, 16), dtype=np.int16)
    nii_bytes = write_nifti_gz_bytes(raw_hu_vol, (0.5, 0.5, 0.5))

    client = TestClient(app_module.app)
    resp = client.post(
        "/analyze",
        files={"file": ("raw_cbct_scan.nii.gz", nii_bytes, "application/gzip")},
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "DentalSegmentator" in detail
    assert "No demo data was substituted" in detail


def test_cbct_analyze_endpoint_returns_3d_schema():
    """Uploading a 5-class label map must be labeled as upload-supplied, never as model inference."""
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
    assert data["segmentation_source"] == "uploaded_labelmap"
    assert data["model"]["pretrained_as_is"] is True
    assert data["model"]["inference_ran_in_this_request"] is False
    assert data["three_d_available"] is True
    assert data["volume_metadata"]["has_grayscale_volume"] is False
    assert data["original_image_note"] is not None
    assert data["timings_ms"]["segmentation"] is None
    assert "memory" in data
    assert data["mesh_glb_url"].endswith(".glb")
    assert len(data["findings"]) == 2
    assert "disclaimer" in data


def test_2d_opg_response_never_advertises_3d(monkeypatch):
    """A 2D radiograph must return no mesh and an explicit no-3D note."""
    import torch

    class _EmptyDetector:
        def __call__(self, images):
            return [{
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }]

    monkeypatch.setattr(app_module.service, "model", _EmptyDetector())
    monkeypatch.setattr(app_module.service.input_gate, "validate", lambda img: (True, 1.0, "ok"))

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(20, 20, 20)).save(buf, format="PNG")
    client = TestClient(app_module.app)
    resp = client.post("/analyze", files={"file": ("opg.png", buf.getvalue(), "image/png")})

    assert resp.status_code == 200
    data = resp.json()
    assert data["modality"] == "OPG"
    assert data["three_d_available"] is False
    assert "mesh_glb_url" not in data
    assert "CBCT" in data["three_d_note"]
    assert data["findings"] == []


def test_bundle_declared_provenance_is_reported_but_not_trusted(tmp_path: Path):
    """A user-supplied provenance.json cannot make labels look like live model inference."""
    from service import DentalAnalysisService

    seg, _ = _build_synthetic_cbct_labelmap()
    bundle = _make_bundle(
        seg, 7,
        provenance={
            "segmentation_source": "demo_reference_labelmap",
            "model_inference_ran": False,
        },
    )
    svc = DentalAnalysisService(weights_path=None, results_dir=str(tmp_path))
    report = svc.analyze_cbct_bytes(bundle, "bundle.zip")

    assert report["segmentation_source"] == "uploaded_labelmap"
    assert "Bundle declaration (unverified): demo_reference_labelmap." in report["segmentation_source_detail"]
    assert report["model"]["inference_ran_in_this_request"] is False
    assert report["demo"] is False


def test_demo_endpoint_uses_server_side_provenance(monkeypatch, tmp_path: Path):
    """The demo endpoint must label the Post bundle as a reference mask, not as live inference."""
    seg, _ = _build_synthetic_cbct_labelmap()
    bundle_path = tmp_path / "PostDentalSurgery_bundle.zip"
    bundle_path.write_bytes(_make_bundle(seg, 11))
    monkeypatch.setattr(app_module, "_scan_bundles", lambda: {"PostDentalSurgery": bundle_path})

    client = TestClient(app_module.app)
    resp = client.post("/analyze/sample_cbct?scan=PostDentalSurgery")

    assert resp.status_code == 200
    data = resp.json()
    assert data["demo"] is True
    assert data["segmentation_source"] == "demo_reference_labelmap"
    assert data["model"]["inference_ran_in_this_request"] is False
    assert "reference mask" in data["segmentation_source_detail"]


def test_raw_cbct_model_path_and_distinct_meshes(monkeypatch, tmp_path: Path):
    """
    Raw volumes must take the model path and produce per-upload meshes.
    The predictor is stubbed, so this proves wiring/provenance, not nnU-Net accuracy.
    """
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    monkeypatch.setenv("VISCAN_CBCT_WEIGHTS_DIR", str(weights_dir))

    seg_a, _ = _build_synthetic_cbct_labelmap()
    seg_b = seg_a.copy()
    seg_b[24:34, 12:24, 14:22] = 0
    seg_b[27:30, 15:19, 8:13] = 0

    def fake_predict(vol_xyz, spacing_xyz, weights_path):
        return seg_b.copy() if int(vol_xyz[0, 0, 0]) == 200 else seg_a.copy()

    monkeypatch.setattr(app_module.service, "_run_live_nnunet_cbct", fake_predict)

    rng = np.random.default_rng(5)
    raw_a = rng.integers(-500, 1800, size=(40, 40, 40), dtype=np.int16)
    raw_a[0, 0, 0] = 100
    raw_b = rng.integers(-500, 1800, size=(40, 40, 40), dtype=np.int16)
    raw_b[0, 0, 0] = 200

    client = TestClient(app_module.app)
    ra = client.post(
        "/analyze",
        files={"file": ("raw_a.nii.gz", write_nifti_gz_bytes(raw_a, (0.5, 0.5, 0.5)), "application/gzip")},
    )
    rb = client.post(
        "/analyze",
        files={"file": ("raw_b.nii.gz", write_nifti_gz_bytes(raw_b, (0.5, 0.5, 0.5)), "application/gzip")},
    )

    assert ra.status_code == 200 and rb.status_code == 200
    da, db = ra.json(), rb.json()
    for data in (da, db):
        assert data["segmentation_source"] == "model_inference"
        assert data["model"]["inference_ran_in_this_request"] is True
        assert data["timings_ms"]["segmentation"] is not None
        assert data["three_d_available"] is True

    def glb_sha(url):
        return hashlib.sha256((app_module.RESULTS_DIR / Path(url).name).read_bytes()).hexdigest()

    assert glb_sha(da["mesh_glb_url"]) != glb_sha(db["mesh_glb_url"])
    assert len(da["findings"]) == 2 and len(db["findings"]) == 1


def _write_dicom_slice(index: int) -> bytes:
    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    ds = pydicom.dataset.Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID

    ds.Modality = "CT"
    ds.PatientName = "PHI_TEST_PATIENT"
    ds.PatientID = "PHI12345"
    ds.PatientBirthDate = "19800101"
    ds.InstitutionName = "PHI_HOSPITAL"
    ds.Rows = 8
    ds.Columns = 8
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.ImagePositionPatient = [0.0, 0.0, float(index)]
    ds.InstanceNumber = index + 1
    ds.PixelData = np.full((8, 8), index * 100, dtype=np.int16).tobytes()

    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def test_dicom_zip_strips_patient_identifiers(monkeypatch, tmp_path: Path):
    """DICOM patient identifiers must not appear in ingestion metadata or the JSON report."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(3):
            zf.writestr(f"slice_{i}.dcm", _write_dicom_slice(i))
    zip_bytes = buf.getvalue()

    vol, seg, spacing, info = load_cbct_input(zip_bytes, "series.zip")
    assert seg is None
    assert vol.shape == (8, 8, 3)
    assert spacing == (1.0, 1.0, 1.0)
    serialized = json.dumps(info)
    for phi in ("PHI_TEST_PATIENT", "PHI12345", "PHI_HOSPITAL", "19800101"):
        assert phi not in serialized

    from service import DentalAnalysisService

    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    monkeypatch.setenv("VISCAN_CBCT_WEIGHTS_DIR", str(weights_dir))

    def fake_predict(vol_xyz, spacing_xyz, weights_path):
        seg = np.zeros(vol_xyz.shape, dtype=np.uint8)
        seg[:, :, 0:2] = 1
        return seg

    svc = DentalAnalysisService(weights_path=None, results_dir=str(tmp_path / "results"))
    monkeypatch.setattr(svc, "_run_live_nnunet_cbct", fake_predict)
    report = svc.analyze_cbct_bytes(zip_bytes, "series.zip")

    assert report["segmentation_source"] == "model_inference"
    assert report["volume_metadata"]["has_grayscale_volume"] is True
    serialized_report = json.dumps(report)
    for phi in ("PHI_TEST_PATIENT", "PHI12345", "PHI_HOSPITAL", "19800101"):
        assert phi not in serialized_report


def test_different_scans_produce_different_slices_models_and_findings(tmp_path: Path):
    """Regression: two different scans must never yield identical slices, 3D models or measurements."""
    from service import DentalAnalysisService

    seg_a, _ = _build_synthetic_cbct_labelmap()
    seg_b = seg_a.copy()
    seg_b[24:34, 12:24, 14:22] = 0  # remove left lower teeth
    seg_b[27:30, 15:19, 8:13] = 0   # and left canal -> only right-side finding remains

    svc = DentalAnalysisService(weights_path=None, results_dir=str(tmp_path))
    ra = svc.analyze_cbct_bytes(_make_bundle(seg_a, 1), "a_bundle.zip")
    rb = svc.analyze_cbct_bytes(_make_bundle(seg_b, 2), "b_bundle.zip")

    def sha(url):
        return hashlib.sha256((tmp_path / Path(url).name).read_bytes()).hexdigest()

    for key in ("original_image_url", "annotated_image_url", "mesh_glb_url"):
        assert sha(ra[key]) != sha(rb[key]), key
    assert len(ra["findings"]) == 2 and len(rb["findings"]) == 1
    for r in (ra, rb):
        assert r["segmentation_source"] == "uploaded_labelmap"
        for f in r["findings"]:
            assert f"{f['min_distance_mm']:.2f} mm" in r["summary"]
    assert ra["summary"] != rb["summary"]


def test_gzip_nifti_decompression_is_bounded(monkeypatch):
    """A gzip bomb must be rejected by the decompression cap, not decompressed in full."""
    monkeypatch.setattr(cbct_pipeline, "MAX_NIFTI_DECOMPRESSED_BYTES", 4096)
    bomb = gzip.compress(b"\x00" * 100_000)
    with pytest.raises(ImageProcessingError, match="safety limit"):
        parse_nifti_bytes(bomb)


def test_zip_archive_uncompressed_size_is_bounded(monkeypatch):
    """A ZIP declaring more uncompressed bytes than the cap is rejected before any member read."""
    monkeypatch.setattr(cbct_pipeline, "MAX_ZIP_UNCOMPRESSED_BYTES", 4096)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("scan.nii", b"\x00" * 100_000)
    with pytest.raises(ImageProcessingError, match="safety limit"):
        load_cbct_input(buf.getvalue(), "bomb.zip")
