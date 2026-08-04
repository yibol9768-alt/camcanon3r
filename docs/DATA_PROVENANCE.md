# Data provenance

## ETH3D high-resolution multi-view training data

Downloaded: 2026-08-03  
Source index: <https://www.eth3d.net/datasets>  
Format documentation: <https://www.eth3d.net/documentation>

The first ground-truth pilot uses the official `office` training scene. Data
archives and extracted data are intentionally excluded from Git.

| Archive | Official URL | Bytes | SHA-256 |
|---|---|---:|---|
| `office_dslr_jpg.7z` | <https://www.eth3d.net/data/office_dslr_jpg.7z> | 231,734,054 | `bc2bfdf784b0cb8139832b8339ad533be022901a635969fefe9d8bc2b9fff07d` |
| `office_dslr_undistorted.7z` | <https://www.eth3d.net/data/office_dslr_undistorted.7z> | 288,765,137 | `316c0c10c79cc173e4b5c26102fed3caedbde7e1865fd896ae15423f0f8cf04c` |
| `office_dslr_depth.7z` | <https://www.eth3d.net/data/office_dslr_depth.7z> | 441,352,937 | `aa2e421079eff69a25332e41c81f2f7ecfea54de97bd30f49e2435cc7c700d8b` |

Machine-local storage on `my5090`:

- archives: `C:\Users\liuyibo\camcanon3r-data\eth3d`;
- extracted scene: `E:\camcanon3r-data\eth3d\office`;
- WSL view: `/mnt/e/camcanon3r-data/eth3d/office`.

The extracted scene contains 26 original DSLR images, 26 pre-undistorted
images, the corresponding COLMAP text calibration/poses, and 26 rendered depth
maps. The official documentation states that depth files retain the `.JPG`
names but are little-endian float32 row-major dumps, with positive infinity for
invalid pixels. They match the original 6048x4032 distorted images, **not** the
pre-undistorted 6221x4146 images. CamCanon3R therefore uses:

- original JPEG + `THIN_PRISM_FISHEYE` calibration for pixel-aligned depth;
- pre-undistorted JPEG + `PINHOLE` calibration only for the separate clean
  camera-pose evaluation.

No ETH3D file is redistributed. Dataset terms and citation requirements must be
checked again before releasing a derived benchmark package.

## Frozen full-training-scene expansion

The confirmatory expansion includes all 13 official high-resolution
multi-view training scenes, rather than selecting scenes after viewing model
outcomes. The frozen manifest is `configs/eth3d_training_archives.json` and
records official URLs plus live `Content-Length` values observed on
2026-08-03. It consists of the two official all-scene DSLR JPEG archives and
the 13 scene-specific rendered-depth archives (16,873,121,856 bytes total).

On my5090, archives and the machine-local SHA-256 report live under
`/mnt/e/camcanon3r-data/eth3d_archives`. Downloads are serialized, resumable,
length checked, hashed after completion, and run at low CPU/I/O priority
through the command-scoped proxy. Download completion does not authorize
automatic extraction or preprocessing while another project owns the machine.

The benchmark-scale four-view evaluation freezes the first four
lexicographically sorted DSLR filenames within every scene before any model
outcome is inspected. Selected raw images, pre-undistorted images, calibration,
and matching raw depth are extracted to
`/mnt/e/camcanon3r-data/eth3d_selected`. The machine-local
`selection_report.json` records archive provenance, member byte lengths,
per-file SHA-256 values, and exact selected filenames.

Acquisition completed on 2026-08-04 with all 15 archives present at their
frozen byte lengths (16,873,121,856 bytes total). The report stores hashes
computed from the downloaded local files; because the frozen manifest does not
publish official digests, these hashes establish local identity and
reproducibility rather than independent upstream authenticity. The frozen
four-view extraction completed at `2026-08-04T05:02:15.442011+00:00` and a
separate strict audit rehashed all 234 payload files, matched the exact reported
path set and sizes across 13 scenes, and found no extra payload files.

## Frozen office four-view pilot

The first confirmatory set is `DSC_0219` through `DSC_0222`. Live validation on
2026-08-03 established that:

- raw and pre-undistorted COLMAP extrinsics are bit-identical for all four
  selected views (maximum absolute difference `0.0`);
- the raw camera is `THIN_PRISM_FISHEYE`, 6048x4032;
- the pre-undistorted camera is `PINHOLE`, 6221x4146;
- each selected raw depth file is exactly 97,542,144 bytes, equal to
  `6048 * 4032 * sizeof(float32)`.

The two evaluation paths are intentionally separate:

```bash
# Raw image prediction: pose plus pixel-aligned raw depth.
python scripts/evaluate_eth3d.py PREDICTION.npz \
  /mnt/e/camcanon3r-data/eth3d/office/dslr_calibration_jpg \
  /mnt/e/camcanon3r-data/eth3d/office/ground_truth_depth/dslr_images

# Pre-undistorted prediction: pose only; raw depth must not be attached.
python scripts/evaluate_eth3d.py PREDICTION.npz \
  /mnt/e/camcanon3r-data/eth3d/office/dslr_calibration_undistorted \
  --skip-depth
```

The `--skip-depth` branch reports `"depth": null` so downstream aggregation
cannot silently confuse missing depth with a zero error.
