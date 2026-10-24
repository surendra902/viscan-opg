"""
3D CBCT Ingestion, DentalSegmentator Integration, 3D Nerve-Proximity Post-Processing,
and 3D Mesh (.stl / .glb) + Multi-Planar Orthogonal Slice Exporter.

Model Provenance:
- Pretrained Model: DentalSegmentator (Dataset112_DentalSegmentator_v100, nnUNetv2 3d_fullres, fold_0)
- Used pretrained as-is (explicitly disclosed in JSON output, UI, and REPORT.md).
- Reference: Dot G, et al. "DentalSegmentator: robust open source deep learning-based CT and CBCT
  image segmentation." Journal of Dentistry (2024). DOI: 10.1016/j.jdent.2024.105130
- Weights DOI: 10.5281/zenodo.10829675 (License: CC BY 4.0)
"""

import gzip
import io
import json
import os
import struct
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, label as cc_label
from scipy.spatial import cKDTree

from imaging import ImageProcessingError

CBCT_CLASS_MAP = {
    1: "maxilla_upper_skull",
    2: "mandible",
    3: "upper_teeth",
    4: "lower_teeth",
    5: "mandibular_canal",
}

# Official SlicerDentalSegmentator RGBA colors
CBCT_RGBA_COLORS = {
    1: (227, 221, 144, 110),  # Maxilla & Upper Skull
    2: (212, 161, 230, 110),  # Mandible (translucent to reveal canal & tooth roots)
    3: (220, 149, 101, 255),  # Upper Teeth
    4: (235, 223, 180, 255),  # Lower Teeth
    5: (216, 101, 79, 255),   # Mandibular Canal (high-contrast red-orange)
}


def is_cbct_filename(filename: str) -> bool:
    """Returns True if the uploaded filename corresponds to a 3D CBCT volume or archive."""
    lower = (filename or "").lower()
    return (
        lower.endswith(".nii")
        or lower.endswith(".nii.gz")
        or lower.endswith(".gipl")
        or lower.endswith(".gipl.gz")
        or lower.endswith(".zip")
    )


def parse_nifti_bytes(raw_bytes: bytes, is_gzipped: bool = False) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Parses a NIfTI-1 (.nii or .nii.gz) 3D volume into an (nx, ny, nz) numpy array
    and (sx, sy, sz) physical voxel spacing in mm.
    """
    if is_gzipped or raw_bytes[:2] == b"\x1f\x8b":
        try:
            raw_bytes = gzip.decompress(raw_bytes)
        except Exception as exc:
            raise ImageProcessingError(f"Corrupt gzipped NIfTI file: {exc}") from exc

    if len(raw_bytes) < 352:
        raise ImageProcessingError("File is too small to be a valid NIfTI-1 volume.")

    sizeof_hdr_le = struct.unpack("<i", raw_bytes[:4])[0]
    endian = "<" if sizeof_hdr_le == 348 else ">"
    sizeof_hdr = struct.unpack(f"{endian}i", raw_bytes[:4])[0]
    if sizeof_hdr != 348:
        raise ImageProcessingError(f"Invalid NIfTI-1 header size ({sizeof_hdr} != 348).")

    dims = struct.unpack(f"{endian}8h", raw_bytes[40:56])
    ndim, nx, ny, nz = dims[0], dims[1], dims[2], dims[3]
    if ndim < 3 or nx <= 1 or ny <= 1 or nz <= 1:
        raise ImageProcessingError(f"NIfTI input is not a 3D volume: dims={dims[:4]}.")

    datatype, _ = struct.unpack(f"{endian}2h", raw_bytes[70:74])
    pixdim = struct.unpack(f"{endian}8f", raw_bytes[76:108])
    vox_offset = int(struct.unpack(f"{endian}f", raw_bytes[108:112])[0])
    scl_slope, scl_inter = struct.unpack(f"{endian}2f", raw_bytes[112:120])

    dtype_map = {
        2: np.uint8,
        4: f"{endian}i2",
        8: f"{endian}i4",
        16: f"{endian}f4",
        64: f"{endian}f8",
        256: np.int8,
        512: f"{endian}u2",
        768: f"{endian}u4",
    }
    if datatype not in dtype_map:
        raise ImageProcessingError(f"Unsupported NIfTI datatype code: {datatype}")

    expected_voxels = int(nx) * int(ny) * int(nz)
    dt = np.dtype(dtype_map[datatype])
    if vox_offset + expected_voxels * dt.itemsize > len(raw_bytes):
        raise ImageProcessingError("Truncated NIfTI voxel buffer.")

    arr = np.frombuffer(raw_bytes, dtype=dt, count=expected_voxels, offset=vox_offset)
    # NIfTI stores X fastest, then Y, then Z -> reshape(nz, ny, nx) and transpose to (nx, ny, nz)
    vol_xyz = arr.reshape((nz, ny, nx)).transpose(2, 1, 0).copy()

    if scl_slope != 0.0 and not np.isnan(scl_slope) and (scl_slope != 1.0 or scl_inter != 0.0):
        vol_xyz = vol_xyz.astype(np.float32) * float(scl_slope) + float(scl_inter)

    sx = abs(float(pixdim[1])) if pixdim[1] != 0 else 1.0
    sy = abs(float(pixdim[2])) if pixdim[2] != 0 else 1.0
    sz = abs(float(pixdim[3])) if pixdim[3] != 0 else 1.0
    return vol_xyz, (sx, sy, sz)


def write_nifti_gz_bytes(vol_xyz: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> bytes:
    """Writes a 3D (nx, ny, nz) array into a standard compressed NIfTI-1 (.nii.gz) byte buffer."""
    nx, ny, nz = vol_xyz.shape
    sx, sy, sz = spacing_xyz
    arr_zyx = np.ascontiguousarray(vol_xyz.transpose(2, 1, 0), dtype="<i2")

    hdr = bytearray(352)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    struct.pack_into("<2h", hdr, 70, 4, 16)  # INT16
    struct.pack_into("<8f", hdr, 76, 1.0, float(sx), float(sy), float(sz), 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", hdr, 108, 352.0)  # vox_offset
    struct.pack_into("<h", hdr, 254, 1)      # sform_code = 1
    struct.pack_into("<4f", hdr, 280, float(sx), 0.0, 0.0, 0.0)
    struct.pack_into("<4f", hdr, 296, 0.0, float(sy), 0.0, 0.0)
    struct.pack_into("<4f", hdr, 312, 0.0, 0.0, float(sz), 0.0)
    hdr[344:348] = b"n+1\x00"

    buf = io.BytesIO()
    with gzip.GzipFile(filename="volume.nii", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(bytes(hdr))
        gz.write(arr_zyx.tobytes())
    return buf.getvalue()


def _parse_dicom_zip(file_bytes: bytes) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float, float]]:
    """
    Loads a zipped 3D DICOM series (or a zipped NIfTI volume + segmentation pair),
    strips all patient PII tags by keeping only numeric voxel arrays & spacing.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(file_bytes), "r")
    except Exception as exc:
        raise ImageProcessingError(f"Corrupt ZIP archive: {exc}") from exc

    names = [n for n in zf.namelist() if not n.endswith("/") and not Path(n).name.startswith(".")]
    if not names:
        raise ImageProcessingError("Uploaded ZIP archive is empty.")

    # Case A: ZIP contains NIfTI (.nii / .nii.gz) files (e.g., volume + optional segmentation mask)
    nii_names = [n for n in names if n.lower().endswith(".nii") or n.lower().endswith(".nii.gz")]
    if nii_names:
        vol_xyz = None
        seg_xyz = None
        spacing_xyz = (0.5, 0.5, 0.5)
        for n in sorted(nii_names):
            arr, sp = parse_nifti_bytes(zf.read(n))
            uniq = set(np.unique(arr).astype(int).tolist())
            if uniq.issubset({0, 1, 2, 3, 4, 5}) and len(uniq) > 1 and (seg_xyz is None or "seg" in n.lower()):
                seg_xyz = arr.astype(np.uint8)
                spacing_xyz = sp
            else:
                vol_xyz = arr
                spacing_xyz = sp
        if vol_xyz is None and seg_xyz is not None:
            vol_xyz = (seg_xyz.astype(np.int16) * 400)
        return vol_xyz, seg_xyz, spacing_xyz

    # Case B: ZIP contains a 3D DICOM series (.dcm slices)
    dcm_names = [n for n in names if n.lower().endswith(".dcm") or "." not in Path(n).name]
    if len(dcm_names) < 3:
        raise ImageProcessingError("ZIP archive must contain at least 3 DICOM (.dcm) slices or a 3D NIfTI volume.")

    try:
        import pydicom
    except ImportError as exc:
        raise ImageProcessingError("pydicom is required to decode 3D DICOM series archives.") from exc

    slices = []
    for n in dcm_names:
        try:
            ds = pydicom.dcmread(io.BytesIO(zf.read(n)), force=True)
            if not hasattr(ds, "pixel_array"):
                continue
            z_pos = float(getattr(ds, "SliceLocation", 0.0))
            if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                z_pos = float(ds.ImagePositionPatient[2])
            elif hasattr(ds, "InstanceNumber"):
                z_pos = float(ds.InstanceNumber)
            slices.append((z_pos, ds))
        except Exception:
            continue

    if len(slices) < 3:
        raise ImageProcessingError("Could not decode a valid 3D DICOM slice stack from the ZIP archive.")

    slices.sort(key=lambda item: item[0])
    first_ds = slices[0][1]
    px_spacing = getattr(first_ds, "PixelSpacing", [0.5, 0.5])
    sy, sx = float(px_spacing[0]), float(px_spacing[1])
    if len(slices) > 1 and abs(slices[-1][0] - slices[0][0]) > 1e-4:
        sz = abs(slices[-1][0] - slices[0][0]) / float(len(slices) - 1)
    else:
        sz = float(getattr(first_ds, "SliceThickness", 0.5))

    stack_zyx = []
    for _, ds in slices:
        arr2d = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        stack_zyx.append(arr2d * slope + inter)

    vol_zyx = np.stack(stack_zyx, axis=0)
    # Transpose (Z, Y, X) -> (X, Y, Z)
    vol_xyz = vol_zyx.transpose(2, 1, 0).copy()
    return vol_xyz, None, (sx, sy, sz)


def load_cbct_input(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float, float]]:
    """
    Loads a 3D CBCT input from NIfTI (.nii, .nii.gz) or ZIP (.zip).
    Returns (vol_xyz, optional_seg_xyz, spacing_xyz_mm).
    """
    if not file_bytes:
        raise ImageProcessingError("Uploaded CBCT file is empty.")

    lower = (filename or "").lower()
    if lower.endswith(".zip"):
        return _parse_dicom_zip(file_bytes)

    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        arr, spacing_xyz = parse_nifti_bytes(file_bytes, is_gzipped=lower.endswith(".gz"))
        uniq = set(np.unique(arr).astype(int).tolist())
        # Check if the uploaded NIfTI is already a 5-class DentalSegmentator labelmap
        if uniq.issubset({0, 1, 2, 3, 4, 5}) and len(uniq) >= 2:
            seg_xyz = arr.astype(np.uint8)
            # Synthesize radiodensity visualization backdrop from anatomical class densities
            hu_lut = np.array([0, 700, 950, 1650, 1600, 250], dtype=np.int16)
            vol_xyz = hu_lut[seg_xyz]
            return vol_xyz, seg_xyz, spacing_xyz
        return arr, None, spacing_xyz

    raise ImageProcessingError(f"Unsupported 3D CBCT format for '{filename}'. Upload .nii, .nii.gz, or DICOM .zip.")


def postprocess_and_compute_findings(
    seg_xyz: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    apply_island_filter: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Applies 60 mm^3 island filtering on segments 1..4 (preserving mandibular canal = 5)
    and computes exact 3D Euclidean minimum distance (in mm) between Lower Teeth (4)
    and Left/Right Mandibular Canal (5) using scipy.spatial.cKDTree.
    """
    seg_clean = seg_xyz.copy()
    nx, ny, nz = seg_clean.shape
    sx, sy, sz = spacing_xyz
    voxel_vol_mm3 = float(sx * sy * sz)

    if apply_island_filter and voxel_vol_mm3 > 0:
        min_voxels = max(2, int(np.ceil(60.0 / voxel_vol_mm3)))
        for cid in [1, 2, 3, 4]:
            mask = seg_clean == cid
            if not np.any(mask):
                continue
            labeled, num_feats = cc_label(mask)
            if num_feats > 1:
                counts = np.bincount(labeled.ravel())
                small_ids = np.where((counts < min_voxels) & (np.arange(len(counts)) > 0))[0]
                if len(small_ids) > 0:
                    seg_clean[np.isin(labeled, small_ids)] = 0

    structure_stats: Dict[str, Any] = {}
    for cid, cname in CBCT_CLASS_MAP.items():
        mask = seg_clean == cid
        v_count = int(mask.sum())
        if v_count > 0:
            xs = np.where(mask.any(axis=(1, 2)))[0]
            ys = np.where(mask.any(axis=(0, 2)))[0]
            zs = np.where(mask.any(axis=(0, 1)))[0]
            bbox_vox = [
                int(xs[0]), int(ys[0]), int(zs[0]),
                int(xs[-1]), int(ys[-1]), int(zs[-1]),
            ]
        else:
            bbox_vox = [0, 0, 0, 0, 0, 0]
        structure_stats[cname] = {
            "label_id": cid,
            "voxel_count": v_count,
            "volume_cm3": round(v_count * voxel_vol_mm3 / 1000.0, 2),
            "bbox_3d_voxel": bbox_vox,
        }

    # Surface extraction for Lower Teeth (4) and Mandibular Canal (5) using tight ROI crops for speed
    t_stat = structure_stats.get("lower_teeth", {})
    c_stat = structure_stats.get("mandibular_canal", {})
    findings: List[Dict[str, Any]] = []

    if t_stat.get("voxel_count", 0) > 0 and c_stat.get("voxel_count", 0) > 0:
        tb = t_stat["bbox_3d_voxel"]
        cb = c_stat["bbox_3d_voxel"]
        tx0, ty0, tz0 = max(0, tb[0] - 1), max(0, tb[1] - 1), max(0, tb[2] - 1)
        tx1, ty1, tz1 = min(nx, tb[3] + 2), min(ny, tb[4] + 2), min(nz, tb[5] + 2)
        cx0, cy0, cz0 = max(0, cb[0] - 1), max(0, cb[1] - 1), max(0, cb[2] - 1)
        cx1, cy1, cz1 = min(nx, cb[3] + 2), min(ny, cb[4] + 2), min(nz, cb[5] + 2)

        t_crop = seg_clean[tx0:tx1, ty0:ty1, tz0:tz1] == 4
        c_crop = seg_clean[cx0:cx1, cy0:cy1, cz0:cz1] == 5

        t_surf_crop = t_crop & ~binary_erosion(t_crop)
        c_surf_crop = c_crop & ~binary_erosion(c_crop)
        if not np.any(t_surf_crop):
            t_surf_crop = t_crop
        if not np.any(c_surf_crop):
            c_surf_crop = c_crop

        teeth_vox = np.argwhere(t_surf_crop) + np.array([tx0, ty0, tz0])
        canal_vox = np.argwhere(c_surf_crop) + np.array([cx0, cy0, cz0])

        mid_x = nx / 2.0
        spacing_arr = np.array([sx, sy, sz], dtype=np.float64)

        for hemi_name, cond in [("right", lambda v: v[:, 0] < mid_x), ("left", lambda v: v[:, 0] >= mid_x)]:
            t_v = teeth_vox[cond(teeth_vox)]
            c_v = canal_vox[cond(canal_vox)]
            if len(t_v) == 0 or len(c_v) == 0:
                continue

            t_mm = t_v * spacing_arr
            c_mm = c_v * spacing_arr
            tree = cKDTree(c_mm)
            dists, idxs = tree.query(t_mm)
            min_i = int(np.argmin(dists))
            min_dist_mm = float(dists[min_i])

            root_ijk = t_v[min_i]
            canal_ijk = c_v[idxs[min_i]]
            root_mm = root_ijk * spacing_arr
            canal_mm = canal_ijk * spacing_arr

            roi_min = np.minimum(root_ijk, canal_ijk) - 10
            roi_max = np.maximum(root_ijk, canal_ijk) + 10
            bbox_3d_voxel = [
                int(max(0, roi_min[0])), int(max(0, roi_min[1])), int(max(0, roi_min[2])),
                int(min(nx - 1, roi_max[0])), int(min(ny - 1, roi_max[1])), int(min(nz - 1, roi_max[2])),
            ]

            findings.append({
                "class": "mandibular_canal_to_lower_teeth_clearance",
                "hemimandible": hemi_name,
                "min_distance_mm": round(min_dist_mm, 2),
                "confidence": 1.0,
                "tooth_fdi": None,
                "bbox": bbox_3d_voxel,  # 3D voxel bounding box [x_min, y_min, z_min, x_max, y_max, z_max]
                "bbox_3d_voxel": bbox_3d_voxel,
                "closest_lower_tooth_voxel": [int(x) for x in root_ijk],
                "closest_canal_voxel": [int(x) for x in canal_ijk],
                "closest_lower_tooth_mm": [round(float(x), 2) for x in root_mm],
                "closest_canal_mm": [round(float(x), 2) for x in canal_mm],
                "key_slice_indices": {
                    "sagittal_x": int(root_ijk[0]),
                    "coronal_y": int(root_ijk[1]),
                    "axial_z": int(root_ijk[2]),
                },
            })

    return seg_clean, structure_stats, findings


def _extract_exposed_voxel_mesh(
    mask: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure-NumPy 3D surface mesh extractor (fallback when scikit-image marching_cubes is unavailable).
    Generates vertices and triangle faces for all exposed boundary voxel faces in physical mm space.
    """
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    sx, sy, sz = spacing_xyz

    # 6 face directions and their 4 unit-cube corner offsets ( splits into 2 triangles: (0,1,2) and (0,2,3) )
    dirs = [
        ((1, 0, 0), [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),
        ((-1, 0, 0), [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]),
        ((0, 1, 0), [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),
        ((0, -1, 0), [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),
        ((0, 0, 1), [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),
        ((0, 0, -1), [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),
    ]

    all_verts = []
    all_faces = []
    v_offset = 0
    core = padded[1:-1, 1:-1, 1:-1]

    for (dx, dy, dz), corners in dirs:
        neighbor = padded[1 + dx:padded.shape[0] - 1 + dx, 1 + dy:padded.shape[1] - 1 + dy, 1 + dz:padded.shape[2] - 1 + dz]
        exposed = np.argwhere(core & ~neighbor)
        if len(exposed) == 0:
            continue
        corners_arr = np.array(corners, dtype=np.float32)  # (4, 3)
        quads = (exposed[:, None, :] + corners_arr[None, :, :]) * np.array([sx, sy, sz], dtype=np.float32)
        verts = quads.reshape((-1, 3))
        n_q = len(exposed)
        base = v_offset + np.arange(n_q, dtype=np.uint32) * 4
        f1 = np.stack([base, base + 1, base + 2], axis=1)
        f2 = np.stack([base, base + 2, base + 3], axis=1)
        all_verts.append(verts)
        all_faces.append(np.concatenate([f1, f2], axis=0))
        v_offset += len(verts)

    if not all_verts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    return np.concatenate(all_verts, axis=0), np.concatenate(all_faces, axis=0)


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Writes a standard 80-byte-header binary STL mesh file."""
    if len(faces) == 0:
        header = b"Viscan CBCT Empty STL".ljust(80, b"\x00")
        path.write_bytes(header + struct.pack("<I", 0))
        return

    tris = vertices[faces]
    v1 = tris[:, 1] - tris[:, 0]
    v2 = tris[:, 2] - tris[:, 0]
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = (normals / norms).astype(np.float32)

    stl_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    records = np.empty(len(faces), dtype=stl_dtype)
    records["normal"] = normals
    records["v0"] = tris[:, 0].astype(np.float32)
    records["v1"] = tris[:, 1].astype(np.float32)
    records["v2"] = tris[:, 2].astype(np.float32)
    records["attr"] = 0

    header = b"Viscan CBCT Binary STL".ljust(80, b"\x00")
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(faces)))
        f.write(records.tobytes())


def _compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Computes smooth per-vertex normals from triangle faces."""
    normals = np.zeros_like(verts, dtype=np.float32)
    if len(faces) == 0 or len(verts) == 0:
        normals[:, 1] = 1.0
        return normals
    tris = verts[faces]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    np.add.at(normals, faces[:, 0], fn)
    np.add.at(normals, faces[:, 1], fn)
    np.add.at(normals, faces[:, 2], fn)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (normals / norms).astype("<f4")


def write_multi_mesh_glb(
    path: Path,
    meshes: List[Tuple[str, np.ndarray, np.ndarray, Tuple[int, int, int, int]]],
    convert_nifti_to_yup: bool = True,
) -> None:
    """
    Writes a centered, standard glTF 2.0 binary (.glb) scene with explicit vertex NORMALs
    and double-sided pbrMetallicRoughness materials for Babylon.js (gr.Model3D) & Three.js (<model-viewer>).
    """
    valid_meshes = [(n, v, f, c) for (n, v, f, c) in meshes if len(v) > 0 and len(f) > 0]
    if not valid_meshes:
        return

    all_pts = np.concatenate([v for (_, v, _, _) in valid_meshes], axis=0)
    center_vec = 0.5 * (all_pts.min(axis=0) + all_pts.max(axis=0))

    buffer_views = []
    accessors = []
    materials = []
    gltf_meshes = []
    nodes = []
    bin_parts = []
    byte_offset = 0

    def _append_chunk(b: bytes, target: int) -> int:
        nonlocal byte_offset
        pad = (4 - (len(b) % 4)) % 4
        b_padded = b + (b"\x00" * pad)
        bv_idx = len(buffer_views)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(b),
            "target": target,
        })
        bin_parts.append(b_padded)
        byte_offset += len(b_padded)
        return bv_idx

    for name, verts, faces, rgba in valid_meshes:
        v_centered = np.asarray(verts, dtype=np.float32) - center_vec[None, :]
        f_arr = np.asarray(faces, dtype=np.uint32).reshape((-1, 3))

        if convert_nifti_to_yup:
            v_gltf = np.stack([v_centered[:, 0], v_centered[:, 2], -v_centered[:, 1]], axis=1).astype("<f4")
            f_arr = f_arr[:, [0, 2, 1]]
        else:
            v_gltf = np.ascontiguousarray(v_centered, dtype="<f4")

        n_gltf = _compute_vertex_normals(v_gltf, f_arr)
        faces_u32 = np.ascontiguousarray(f_arr.ravel(), dtype="<u4")

        rgba_opaque = (int(rgba[0]), int(rgba[1]), int(rgba[2]), 255)
        colors_u8 = np.tile(np.array(rgba_opaque, dtype=np.uint8), (len(v_gltf), 1))

        bv_pos = _append_chunk(v_gltf.tobytes(), 34962)
        bv_norm = _append_chunk(n_gltf.tobytes(), 34962)
        bv_col = _append_chunk(colors_u8.tobytes(), 34962)
        bv_idx = _append_chunk(faces_u32.tobytes(), 34963)

        acc_pos = len(accessors)
        accessors.append({
            "bufferView": bv_pos,
            "componentType": 5126,
            "count": len(v_gltf),
            "type": "VEC3",
            "min": [float(x) for x in v_gltf.min(axis=0)],
            "max": [float(x) for x in v_gltf.max(axis=0)],
        })
        acc_norm = len(accessors)
        accessors.append({
            "bufferView": bv_norm,
            "componentType": 5126,
            "count": len(n_gltf),
            "type": "VEC3",
            "min": [float(x) for x in n_gltf.min(axis=0)],
            "max": [float(x) for x in n_gltf.max(axis=0)],
        })
        acc_col = len(accessors)
        accessors.append({
            "bufferView": bv_col,
            "componentType": 5121,
            "normalized": True,
            "count": len(colors_u8),
            "type": "VEC4",
        })
        acc_idx = len(accessors)
        accessors.append({
            "bufferView": bv_idx,
            "componentType": 5125,
            "count": len(faces_u32),
            "type": "SCALAR",
        })

        mat_i = len(materials)
        materials.append({
            "name": f"mat_{name}",
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                "metallicFactor": 0.08,
                "roughnessFactor": 0.42,
            },
        })

        mesh_i = len(gltf_meshes)
        gltf_meshes.append({
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": acc_pos, "NORMAL": acc_norm, "COLOR_0": acc_col},
                "indices": acc_idx,
                "material": mat_i,
                "mode": 4,
            }],
        })
        nodes.append({"name": name, "mesh": mesh_i})

    bin_blob = b"".join(bin_parts)
    gltf_dict = {
        "asset": {"version": "2.0", "generator": "Viscan CBCT 3D Mesh Exporter"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_blob)}],
    }

    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes += b" " * json_pad

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total_len))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_blob), 0x004E4942))
        f.write(bin_blob)


def export_opg_3d_relief_glb(
    img_pil: Image.Image,
    findings: List[Dict[str, Any]],
    output_dir: Path,
    image_id: str,
) -> str:
    """
    Generates an input-specific 3D Radiodensity Relief + 3D Extruded Pathology Bounding-Box (.glb)
    mesh directly from a 2D OPG radiograph's pixel intensities and its detected Faster R-CNN findings.
    Ensures every 2D OPG input produces its own unique 3D topography and 3D finding boxes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    orig_w, orig_h = img_pil.size
    grid_w, grid_h = 160, 80
    gray_small = np.asarray(img_pil.convert("L").resize((grid_w, grid_h), Image.Resampling.BILINEAR), dtype=np.float32)

    # Normalize intensity [0, 1] -> Z relief height in mm [0, 16]
    lo, hi = float(np.percentile(gray_small, 2)), float(np.percentile(gray_small, 99))
    if hi <= lo:
        hi = lo + 1.0
    norm_z = np.clip((gray_small - lo) / (hi - lo), 0.0, 1.0)

    # Build vertex grid in glTF Y-up space: X in [-80, 80], Y (vertical) in [-40, 40], Z (relief) in [0, 16]
    xs = np.linspace(-80.0, 80.0, grid_w, dtype=np.float32)
    ys = np.linspace(40.0, -40.0, grid_h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    zv = norm_z * 16.0
    verts_relief = np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=1).astype(np.float32)

    # Build triangle indices for the (grid_h, grid_w) surface
    r_idx = np.arange(grid_h - 1, dtype=np.uint32)[:, None]
    c_idx = np.arange(grid_w - 1, dtype=np.uint32)[None, :]
    v00 = (r_idx * grid_w + c_idx).ravel()
    v01 = v00 + 1
    v10 = v00 + grid_w
    v11 = v10 + 1
    faces_relief = np.concatenate([
        np.stack([v00, v10, v01], axis=1),
        np.stack([v01, v10, v11], axis=1),
    ], axis=0).astype(np.uint32)

    # Build per-vertex color from radiograph intensity + highlight detected finding regions
    v_colors = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
    base_val = (norm_z * 215.0 + 30.0).astype(np.uint8)
    v_colors[:, :, 0] = base_val
    v_colors[:, :, 1] = np.clip(base_val * 0.96, 0, 255).astype(np.uint8)
    v_colors[:, :, 2] = np.clip(base_val * 0.85, 0, 255).astype(np.uint8)
    v_colors[:, :, 3] = 255

    OPG_CLASS_RGBA = {
        "caries": (255, 215, 0, 255),
        "deep_caries": (255, 69, 0, 255),
        "periapical_lesion": (186, 85, 211, 255),
        "impacted_tooth": (0, 191, 255, 255),
    }

    box_meshes_by_class: Dict[str, Tuple[List[np.ndarray], List[np.ndarray], int]] = {}
    cube_tris = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ], dtype=np.uint32)

    for f in findings:
        cls_name = f.get("class", "caries")
        bbox = f.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        # Map pixel bbox [x0, y0, x1, y1] -> grid indices & glTF coordinates
        gx0 = int(np.clip(x0 / max(1, orig_w) * (grid_w - 1), 0, grid_w - 1))
        gx1 = int(np.clip(x1 / max(1, orig_w) * (grid_w - 1), 0, grid_w - 1))
        gy0 = int(np.clip(y0 / max(1, orig_h) * (grid_h - 1), 0, grid_h - 1))
        gy1 = int(np.clip(y1 / max(1, orig_h) * (grid_h - 1), 0, grid_h - 1))

        rgba = OPG_CLASS_RGBA.get(cls_name, (0, 255, 128, 255))
        # Tint the 3D surface inside the detected bounding box
        v_colors[gy0:gy1 + 1, gx0:gx1 + 1, :3] = (
            0.35 * v_colors[gy0:gy1 + 1, gx0:gx1 + 1, :3] + 0.65 * np.array(rgba[:3])
        ).astype(np.uint8)

        # Also create 4 3D border rails (hollow 3D bounding box frame) floating at Z = [14..21] mm
        wx0 = (x0 / max(1, orig_w)) * 160.0 - 80.0
        wx1 = (x1 / max(1, orig_w)) * 160.0 - 80.0
        wy0 = 40.0 - (y0 / max(1, orig_h)) * 80.0
        wy1 = 40.0 - (y1 / max(1, orig_h)) * 80.0
        t = 0.75  # rail thickness in mm
        z_lo, z_hi = 13.5, 19.5
        rails = [
            (wx0 - t, wx1 + t, wy0 - t, wy0 + t),  # top bar
            (wx0 - t, wx1 + t, wy1 - t, wy1 + t),  # bottom bar
            (wx0 - t, wx0 + t, wy1 - t, wy0 + t),  # left bar
            (wx1 - t, wx1 + t, wy1 - t, wy0 + t),  # right bar
        ]
        if cls_name not in box_meshes_by_class:
            box_meshes_by_class[cls_name] = ([], [], 0)
        v_list, f_list, v_off = box_meshes_by_class[cls_name]
        for rx0, rx1, ry0, ry1 in rails:
            c_verts = np.array([
                [rx0, ry0, z_lo], [rx1, ry0, z_lo], [rx1, ry1, z_lo], [rx0, ry1, z_lo],
                [rx0, ry0, z_hi], [rx1, ry0, z_hi], [rx1, ry1, z_hi], [rx0, ry1, z_hi],
            ], dtype=np.float32)
            v_list.append(c_verts)
            f_list.append(cube_tris + v_off)
            v_off += 8
        box_meshes_by_class[cls_name] = (v_list, f_list, v_off)

    # Write combined multi-mesh GLB (with per-vertex tinted relief + 3D detection box frames)
    n_relief = _compute_vertex_normals(verts_relief, faces_relief)
    c_relief = v_colors.reshape((-1, 4))

    buffer_views = []
    accessors = []
    materials = []
    gltf_meshes = []
    nodes = []
    bin_parts = []
    byte_offset = 0

    def _append(b: bytes, target: int) -> int:
        nonlocal byte_offset
        pad = (4 - (len(b) % 4)) % 4
        b_pad = b + (b"\x00" * pad)
        idx = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": byte_offset, "byteLength": len(b), "target": target})
        bin_parts.append(b_pad)
        byte_offset += len(b_pad)
        return idx

    def _add_node(name: str, v_f32: np.ndarray, n_f32: np.ndarray, c_u8: np.ndarray, f_u32: np.ndarray) -> None:
        bv_p = _append(v_f32.tobytes(), 34962)
        bv_n = _append(n_f32.tobytes(), 34962)
        bv_c = _append(c_u8.tobytes(), 34962)
        bv_i = _append(f_u32.tobytes(), 34963)
        ap = len(accessors)
        accessors.append({"bufferView": bv_p, "componentType": 5126, "count": len(v_f32), "type": "VEC3",
                          "min": [float(x) for x in v_f32.min(axis=0)], "max": [float(x) for x in v_f32.max(axis=0)]})
        an = len(accessors)
        accessors.append({"bufferView": bv_n, "componentType": 5126, "count": len(n_f32), "type": "VEC3",
                          "min": [float(x) for x in n_f32.min(axis=0)], "max": [float(x) for x in n_f32.max(axis=0)]})
        ac = len(accessors)
        accessors.append({"bufferView": bv_c, "componentType": 5121, "normalized": True, "count": len(c_u8), "type": "VEC4"})
        ai = len(accessors)
        accessors.append({"bufferView": bv_i, "componentType": 5125, "count": len(f_u32), "type": "SCALAR"})
        mi = len(materials)
        materials.append({"name": f"mat_{name}", "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.08, "roughnessFactor": 0.45}})
        m_idx = len(gltf_meshes)
        gltf_meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": ap, "NORMAL": an, "COLOR_0": ac}, "indices": ai, "material": mi, "mode": 4}]})
        nodes.append({"name": name, "mesh": m_idx})

    _add_node("opg_3d_radiodensity_surface", verts_relief.astype("<f4"), n_relief, c_relief, faces_relief.ravel().astype("<u4"))

    for cls_name, (v_list, f_list, _) in box_meshes_by_class.items():
        if not v_list:
            continue
        v_box = np.concatenate(v_list, axis=0).astype("<f4")
        f_box = np.concatenate(f_list, axis=0).astype(np.uint32)
        n_box = _compute_vertex_normals(v_box, f_box)
        rgba = OPG_CLASS_RGBA.get(cls_name, (0, 255, 128, 255))
        c_box = np.tile(np.array(rgba, dtype=np.uint8), (len(v_box), 1))
        _add_node(f"3d_boxes_{cls_name}", v_box, n_box, c_box, f_box.ravel().astype("<u4"))

    bin_blob = b"".join(bin_parts)
    gltf_dict = {
        "asset": {"version": "2.0", "generator": "Viscan OPG 3D Radiodensity Topography Exporter"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_blob)}],
    }
    json_bytes = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - (len(json_bytes) % 4)) % 4)
    glb_path = output_dir / f"{image_id}_opg3d.glb"
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_blob)
    with open(glb_path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total_len))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_blob), 0x004E4942))
        f.write(bin_blob)
    return f"/results/{glb_path.name}"


def export_cbct_meshes(
    seg_xyz: np.ndarray,
    spacing_xyz: Tuple[float, float, float],
    output_dir: Path,
    image_id: str,
) -> Tuple[str, Dict[str, str]]:
    """
    Extracts 3D surface meshes for each present anatomical class (1..5),
    saves individual binary .stl files and a combined .glb scene, and returns URLs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = seg_xyz.shape
    step_ds = 2 if max(nx, ny, nz) > 160 else 1
    eff_spacing = (spacing_xyz[0] * step_ds, spacing_xyz[1] * step_ds, spacing_xyz[2] * step_ds)

    try:
        from skimage.measure import marching_cubes
        has_skimage = True
    except ImportError:
        has_skimage = False

    glb_entries = []
    stl_urls: Dict[str, str] = {}

    for cid, cname in CBCT_CLASS_MAP.items():
        sub_mask = (seg_xyz[::step_ds, ::step_ds, ::step_ds] == cid)
        if int(sub_mask.sum()) < 4:
            continue

        if has_skimage and min(sub_mask.shape) >= 3:
            verts, faces, _, _ = marching_cubes(sub_mask.astype(np.uint8), level=0.5, spacing=eff_spacing)
            verts = verts.astype(np.float32)
            faces = faces.astype(np.uint32)
        else:
            verts, faces = _extract_exposed_voxel_mesh(sub_mask, eff_spacing)

        stl_filename = f"{image_id}_{cname}.stl"
        write_binary_stl(output_dir / stl_filename, verts, faces)
        stl_urls[cname] = f"/results/{stl_filename}"
        glb_entries.append((cname, verts, faces, CBCT_RGBA_COLORS[cid]))

    glb_filename = f"{image_id}_scene.glb"
    write_multi_mesh_glb(output_dir / glb_filename, glb_entries)
    return f"/results/{glb_filename}", stl_urls


def render_cbct_ortho_views(
    vol_xyz: np.ndarray,
    seg_xyz: np.ndarray,
    findings: List[Dict[str, Any]],
    orig_path: Path,
    annot_path: Path,
) -> None:
    """
    Renders a clean 2x2 multi-planar reconstruction (Axial, Coronal, Sagittal, Coronal MIP)
    for both the raw volume (orig_path) and the color-segmented volume with nerve vectors (annot_path).
    """
    nx, ny, nz = vol_xyz.shape
    if findings:
        primary = min(findings, key=lambda f: f["min_distance_mm"])
        sx_i = primary["key_slice_indices"]["sagittal_x"]
        cy_i = primary["key_slice_indices"]["coronal_y"]
        az_i = primary["key_slice_indices"]["axial_z"]
    else:
        sx_i, cy_i, az_i = nx // 2, ny // 2, nz // 2

    def _norm_gray(sl: np.ndarray) -> np.ndarray:
        sl_f = sl.astype(np.float32)
        lo, hi = np.percentile(sl_f, 2), np.percentile(sl_f, 99)
        if hi <= lo:
            hi = lo + 1.0
        return np.clip((sl_f - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    def _colorize(gray2d: np.ndarray, seg2d: np.ndarray, alpha: float = 0.48) -> np.ndarray:
        rgb = np.stack([gray2d, gray2d, gray2d], axis=-1).astype(np.float32)
        for cid, rgba in CBCT_RGBA_COLORS.items():
            m = seg2d == cid
            if np.any(m):
                color = np.array(rgba[:3], dtype=np.float32)
                rgb[m] = (1.0 - alpha) * rgb[m] + alpha * color
        return np.clip(rgb, 0, 255).astype(np.uint8)

    slices_raw = [
        ("Axial Slice (Z=" + str(az_i) + ")", np.flipud(_norm_gray(vol_xyz[:, :, az_i].T)), np.flipud(seg_xyz[:, :, az_i].T)),
        ("Coronal Slice (Y=" + str(cy_i) + ")", np.flipud(_norm_gray(vol_xyz[:, cy_i, :].T)), np.flipud(seg_xyz[:, cy_i, :].T)),
        ("Sagittal Slice (X=" + str(sx_i) + ")", np.flipud(_norm_gray(vol_xyz[sx_i, :, :].T)), np.flipud(seg_xyz[sx_i, :, :].T)),
        ("3D Coronal MIP Overlay", np.flipud(_norm_gray(np.max(vol_xyz, axis=1).T)), np.flipud(np.max(seg_xyz, axis=1).T)),
    ]

    panel_w, panel_h = 440, 380
    canvas_orig = Image.new("RGB", (panel_w * 2, panel_h * 2), (15, 23, 42))
    canvas_annot = Image.new("RGB", (panel_w * 2, panel_h * 2), (15, 23, 42))
    draw_orig = ImageDraw.Draw(canvas_orig)
    draw_annot = ImageDraw.Draw(canvas_annot)

    positions = [(0, 0), (panel_w, 0), (0, panel_h), (panel_w, panel_h)]
    for (title, g2d, s2d), (px, py) in zip(slices_raw, positions):
        pil_g = Image.fromarray(g2d).resize((panel_w - 20, panel_h - 44), Image.Resampling.BILINEAR).convert("RGB")
        pil_c = Image.fromarray(_colorize(g2d, s2d)).resize((panel_w - 20, panel_h - 44), Image.Resampling.BILINEAR)
        canvas_orig.paste(pil_g, (px + 10, py + 34))
        canvas_annot.paste(pil_c, (px + 10, py + 34))
        draw_orig.text((px + 14, py + 10), title, fill=(226, 232, 240))
        draw_annot.text((px + 14, py + 10), title, fill=(56, 189, 248))

    canvas_orig.save(orig_path, format="PNG")
    canvas_annot.save(annot_path, format="PNG")
