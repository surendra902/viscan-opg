# CBCT Demo Provenance

These two bundles are the only CBCT scans the server will label as demos. They are
never substituted for a user upload. Each bundle contains a grayscale volume, a
5-class label map, and a `provenance.json` that is treated as an unverified
declaration when someone re-uploads the bundle.

| Scan | Label source in the bundle | How the labels were produced | Reported `segmentation_source` |
|---|---|---|---|
| `PostDentalSurgery` | Official SlicerDentalSegmentator reference mask | Published reference segmentation; no model inference in this demo | `demo_reference_labelmap` |
| `PreDentalSurgery` | DentalSegmentator model output | nnU-Net v2 `3d_fullres` fold_0 run in the Kaggle scan-2 notebook (Tesla T4) | `demo_model_labelmap` |

The scans are the paired Pre/Post volumes distributed with 3D Slicer's
**CBCT Dental Surgery** sample data (also used by the SlicerDentalSegmentator
project). This document does not claim that they are the same patient or that
the volume differences are clinically meaningful.

## Bundle integrity

| Bundle | Bytes | SHA-256 |
|---|---:|---|
| `PostDentalSurgery_bundle.zip` | 35,593,875 | `fc07b2bd71a29b09d5aef4223dee5b4668c5c79322a7f5dde7b51a618263e0e2` |
| `PreDentalSurgery_bundle.zip` | 24,786,219 | `c0c8465417bf029870801bcae25ce2d6638cf07e6c5dabd06c0a806a0629bbf0` |

Rebuild and regenerate every artifact with:

```bash
python build_cbct_bundles.py
python generate_cbct_samples.py
```

`generate_cbct_samples.py` runs the real service on each bundle, so the PNG, STL,
GLB and JSON files in this directory are derived from the bundle contents rather
than hand-copied from another scan.

## External model measurements (Kaggle Tesla T4)

These numbers were measured in the `kaggle_cbct_notebook.ipynb` execution, not on
the machine that generated the sample artifacts.

| Metric | `PostDentalSurgery` | `PreDentalSurgery` |
|---|---:|---:|
| Model initialization | 2,131.9 ms | not recorded |
| 3D sliding-window inference (252 tiles) | 131,407.1 ms | 115,902.3 ms |
| Peak GPU VRAM allocated | 4,936.0 MB | 4,724.9 MB |
| CPU RSS during the run | 2,641.8 MB (+1,270.8 MB) | not recorded |
| Post-processing + `cKDTree` distance | 5,816.3 ms | not recorded |
| Marching Cubes meshing | 938.6 ms | not recorded |

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
| `PostDentalSurgery` | 1,871.9 ms | 2,294.6 ms | 246.1 ms | 5,061.0 ms | 9,473.5 ms |
| `PreDentalSurgery` | 1,806.7 ms | 3,248.2 ms | 247.0 ms | 4,442.8 ms | 9,744.7 ms |

The JSON reports record `model.inference_ran_in_this_request = false`,
`timings_ms.segmentation = null`, and `memory.gpu_peak_allocated_mb = null` for
these demo requests.
