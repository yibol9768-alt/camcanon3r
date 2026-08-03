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
