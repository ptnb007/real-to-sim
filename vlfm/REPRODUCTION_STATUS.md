# VLFM reproduction status

Target machine: Windows 10, WSL2 Ubuntu 24.04, NVIDIA GeForce RTX 5060
(8 GB VRAM, compute capability 12.0).

## Completed

- Imported a working Ubuntu 24.04 WSL2 distribution at `D:\WSL\Ubuntu-24.04`.
- Created the `vlfm` Python 3.9 Conda environment.
- Installed Habitat-Sim 0.2.4 display/Bullet and the VLFM Habitat-Lab and
  Habitat-Baselines releases.
- Installed PyTorch 2.7.0 + CUDA 12.8 and verified a real CUDA matrix
  multiplication on the RTX 5060.
- Built the GroundingDINO deformable-attention CUDA extension for `sm_120`.
- Installed VLFM and generated `data/dummy_policy.pth`.
- Downloaded and verified MobileSAM, GroundingDINO, YOLOv7, PointNav, and
  BLIP2-ITM model assets.
- Downloaded HM3D ObjectNav train, val, and val-mini episode definitions.
- Downloaded and installed the complete HM3D v0.2 validation split, including
  Habitat scenes, configs, semantic annotations, and semantic configs.
- Loaded all four model servers concurrently on the 8 GB GPU and completed
  real requests through every REST endpoint without an out-of-memory error.
- Verified Habitat-Sim camera rendering under WSLg by selecting the default
  Mesa EGL device (`gpu_device_id=-1`). A 64x64 RGBA observation rendered
  successfully with llvmpipe.
- Started a real VLFM val-mini job through dataset initialization, worker
  creation, renderer initialization, and all policy modes.
- Completed one real HM3D val-mini ObjectNav episode successfully. The agent
  searched for a toilet, detected it, navigated to it, and called stop.

Verified concurrent smoke-test outputs:

- GroundingDINO request completed.
- BLIP2-ITM cosine score: approximately `0.429` for the synthetic test image.
- MobileSAM mask: `16380` pixels for the synthetic square.
- YOLOv7 request completed.

## Verified single-episode result

Run date: 2026-08-20. Dataset: HM3D ObjectNav v1 val-mini. Episode count: 1.

- Elapsed episode time: 33.14 seconds
- Final step: 58
- Average reward: 6.6503
- Distance to goal: 0.0457
- Success: 1.0000
- SPL: 0.7376
- SoftSPL: 0.7297
- Target detected: 1.0000
- Stop called: 1.0000

This is a functional reproduction smoke test, not the aggregate paper result.
The single episode ran with Habitat rendered by Mesa llvmpipe under WSLg while
the four learned visual services ran on the RTX 5060 through CUDA.

With the four VLM services running, repeat the episode from the repository root
with:

```bash
bash scripts/run_val_mini_wsl.sh
```

Generate the composite visualization video with:

```bash
bash scripts/run_val_mini_video_wsl.sh
```

The verified example video is stored at
`results/val_mini_video/vlfm_val_mini_demo.mp4`. It contains the egocentric
camera view together with VLFM's obstacle/trajectory and language-value maps.

## Compatibility changes

The paper pins PyTorch 1.12.1 with CUDA 11.3, which predates NVIDIA Blackwell
GPUs. This environment retains Python 3.9 and the paper's Habitat versions but
uses PyTorch 2.7.0/CUDA 12.8, the first PyTorch release line with Blackwell
support.

Minimal source compatibility changes:

- GroundingDINO uses `value.scalar_type()` in two CUDA dispatch calls instead
  of the removed PyTorch 1.x `value.type()` form.
- YOLOv7 explicitly passes `weights_only=False` when loading the trusted
  official checkpoint, matching PyTorch's pre-2.6 behavior.
- VLFM's trainer likewise loads the locally generated dummy policy with
  `weights_only=False` because it contains an OmegaConf configuration object.
- spaCy is pinned to 3.7.5 and setuptools to 69.5.1 for Python 3.9 and legacy
  LAVIS/GroundingDINO compatibility.
- WSLg uses Habitat-Sim's display/Bullet package and the validation launcher
  sets `habitat.simulator.habitat_sim_v0.gpu_device_id=-1`. This uses Mesa's
  default EGL renderer because WSL does not expose the RTX GPU as a native EGL
  device. The VLM models still run on the RTX 5060 through CUDA.
- PointNav checkpoint loading explicitly uses `weights_only=False`, matching
  PyTorch's pre-2.6 behavior for the trusted official checkpoint containing an
  OmegaConf configuration.

## Repeatable checks

With the four servers running:

```bash
python scripts/smoke_test_vlm_services.py
```

Asset check that permits only the credential-gated scene files to be missing:

```bash
python scripts/check_reproduction_assets.py --allow-missing-scenes
```
