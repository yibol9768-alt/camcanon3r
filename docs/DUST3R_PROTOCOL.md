# Frozen DUSt3R confirmatory protocol

DUSt3R is the pre-registered second model for testing whether preprocessing
non-equivariance generalizes beyond VGGT. It is confirmatory evidence, not a
replacement for the ETH3D ground-truth accuracy experiment.

## Provenance and license boundary

- Official repository: `https://github.com/naver/dust3r`
- Frozen commit: `4c24a6ebf04809f2cfe59915e51779c8984aaa40`
- Checkpoint: `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`
- Code license: CC BY-NC-SA 4.0, non-commercial use only
- Checkpoint use additionally inherits the public training-data and base-model
  license terms listed by DUSt3R.

The external repository and weights are machine-local and excluded from this
MIT repository. The setup script requires explicit acknowledgement of the
non-commercial boundary. No DUSt3R source or weight file is redistributed.
The my5090 Python environment is frozen in
`configs/dust3r_requirements_lock.txt`; PyTorch 2.11.0 and torchvision 0.26.0
use the CUDA 12.8 wheels. The model snapshot is accepted only at exactly
2,284,790,056 bytes with SHA-256
`7c300a89534113436bde52732d3151212bcbd90f0aa3c8d1496f86d84bfe4b42`.

## Exact preprocessing contract

The official 512-pixel loader resizes the long edge, independently rounds both
dimensions, and center-crops them to patch multiples. A square image becomes a
512x384 tensor unless the checkpoint declares `square_ok`. CamCanon3R mirrors
these operations in `plan_dust3r_preprocessing`, verifies the logged size
against every loaded tensor, and stores three matrices per view:

1. protocol source-to-prepared affine;
2. DUSt3R prepared-to-tensor affine;
3. their exact source-to-tensor composition.

DUSt3R returns camera-to-world poses. The adapter converts them to the same
world-to-camera convention used by VGGT and ETH3D, while retaining the original
camera-to-world matrices in the archive. Depth, intrinsics, point maps,
confidence, and spatial transforms use the common prediction schema.

## Frozen inference settings

- four sorted views;
- complete symmetrized pair graph;
- batch size 1;
- point-cloud optimizer initialized with MST;
- cosine schedule, 300 iterations, learning rate 0.01;
- seed 17;
- one checkpoint load reused across a resumable sweep.

Cross-transform disagreement only supports a non-equivariance claim. Accuracy
degradation still requires pose or depth ground truth, and multi-model
generality is promoted only if the frozen DUSt3R matrix independently passes
the thresholds in the claim-evidence map.
