# CBCT Demo Provenance

These two bundles are the only CBCT scans the server will label as demos. They are
never substituted for a user upload. Each bundle contains a grayscale volume, a
5-class label map, and a `provenance.json` that is treated as an unverified
declaration when someone re-uploads the bundle.

| Scan | Label source in the bundle | How the labels were produced | Reported `segmentation_source` |
|---|---|---|---|
| `PostDentalSurgery` | Published SlicerDentalSegmentator reference mask, after this repo's 60 mm³ island filter | Published reference segmentation; no model inference in this demo | `demo_reference_labelmap` |
| `PreDentalSurgery` | DentalSegmentator model output supplied with the bundle | nnU-Net v2 `3d_fullres` fold_0 weights; the run that produced it is not documented in an executed notebook in this repo | `demo_model_labelmap` |

The scans are the paired Pre/Post volumes distributed with 3D Slicer's
**CBCT Dental Surgery** sample data (also used by the SlicerDentalSegmentator
project). This document does not claim that they are the same patient or that
the volume differences are clinically meaningful.

## Bundle integrity

| Bundle | Bytes | SHA-256 |
|---|---:|---|
| `PostDentalSurgery_bundle.zip` | 35,593,920 | `dd7f8519687ce067ce6ff5cb512b19e87fa21732e4f85a1907f97c09f3917b90` |
| `PreDentalSurgery_bundle.zip` | 24,786,261 | `d290fbe39f3ed3cc392664ccbdae2294e4abe0d204e4d6e561b2052ad6a7a8ff` |

Rebuild and regenerate every artifact with:

```bash
python build_cbct_bundles.py
python generate_cbct_samples.py
```

`generate_cbct_samples.py` runs the real service on each bundle, so the PNG, STL,
GLB and JSON files in this directory are derived from the bundle contents rather
than hand-copied from another scan.

## Mesh exporter used for these artifacts

`export_cbct_meshes` uses `skimage.measure.marching_cubes` when `scikit-image` is
importable and otherwise falls back to `_extract_exposed_voxel_mesh`, a pure-NumPy
exposed-voxel mesher. The artifacts in this directory were produced on a machine
without `scikit-image`, so every STL/GLB here comes from the fallback path and the
`Meshing` timings below measure that path. The HF Space image installs
`scikit-image` from `requirements.txt`, so its exports use Marching Cubes; the
shipped GLB files (28.7 MB and 35.1 MB) are larger than the Marching Cubes output
the live Space produces for the same bundle.

## External model measurements (Kaggle Tesla T4)

The `PostDentalSurgery` column is transcribed from the executed
`kaggle_cbct_notebook.ipynb`. The `PreDentalSurgery` column is transcribed from
the `run_metrics.json` supplied inside the Pre bundle; that run has no executed
notebook in this repository and cannot be re-inspected here. None of these
numbers were measured on the machine that generated the sample artifacts.

| Metric | `PostDentalSurgery` | `PreDentalSurgery` |
|---|---:|---:|
| Model initialization | 2,131.9 ms | not recorded |
| 3D sliding-window inference (252 tiles) | 131,407.1 ms | 115,902.3 ms |
| Peak GPU VRAM allocated | 4,936.0 MB | 4,724.9 MB |
| CPU RSS during the run | 2,641.8 MB (+1,270.8 MB) | not recorded |
| Post-processing + `cKDTree` distance | 5,816.3 ms | not recorded |
| Meshing | 938.6 ms | not recorded |

The Post run also compared its **model prediction** with the official reference
mask shipped in this demo bundle. These Dice values describe the model, not the
demo label map:

| Class | Dice (model prediction vs reference) |
|---|---:|
| `maxilla_upper_skull` | 0.9768 |
| `mandible` | 0.9849 |
| `upper_teeth` | 0.9867 |
| `lower_teeth` | 0.9745 |
| `mandibular_canal` | 0.8968 |
| **Mean** | **0.9639** |

No reference mask was used for `PreDentalSurgery`, so no Dice is reported for it.

## Local sample-artifact run

The artifacts in this directory were regenerated on a local CPU from the
pre-segmented bundles (no nnU-Net segmentation stage ran). The timings below are
end-to-end service timings for loading, post-processing, MPR rendering and mesh
export only:

| Scan | Load | Post-processing | Rendering | Meshing | Total |
|---|---:|---:|---:|---:|---:|
| `PostDentalSurgery` | 2,427.9 ms | 2,454.9 ms | 269.0 ms | 4,479.5 ms | 9,631.3 ms |
| `PreDentalSurgery` | 1,989.4 ms | 4,018.7 ms | 252.2 ms | 4,178.3 ms | 10,438.6 ms |

The JSON reports record `model.inference_ran_in_this_request = false`,
`timings_ms.segmentation = null`, and `memory.gpu_peak_allocated_mb = null` for
these demo requests.
