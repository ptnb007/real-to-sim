# CRUZR → Isaac Sim migration work

## Current scope

This directory implements the first independently runnable part of the requested
migration: Isaac physical rigid-body poses → a renderer-neutral Gaussian binding
stream. It does not replace the existing Motrix project or other robot projects.

The original CRUZR source/assets have not been located. The supplied presentation
is evidence of the prior demo, not a source-code distribution. No CRUZR robot,
41-link binding, 3DGS image, navigation or grasping result is claimed by this code.

## Run on this workstation

`run_smoke.bat` uses the installed Isaac Sim 5.1 at `D:\isaacsim-5.1.0`.
Set `ISAACSIM_ROOT` to override. The validation uses actual gravity/contact physics
on a 20 cm cube and writes `results/verification.json`, `poses.jsonl`, and a USD.
The cube is a test fixture, not a CRUZR substitute. It does not render a camera image.

Run `python -m unittest discover -s cruzr_isaac -v` from the parent directory to
check coordinate composition, covariance rotation and millimetre conversion.

## Required CRUZR handoff

- `replay_cruzr.py`, dependencies/version information and renderer source.
- `cruzr_full.xml`, included XMLs and all referenced robot meshes.
- `cruzr_static_environment.xml` and its scene assets.
- `body_gaussians.json`, `cruzr_bind_manifest.json`, all 41 link PLYs and `L2.ply`.
- Existing URDF/USD if available. MJCF is not silently treated as URDF.

Review the original binding schema and whether PLYs already have bind transforms
or mm→m conversion applied. Convert exactly once. Preserve physical link names
during model import, then audit joint axes/limits, inertias and collision shapes.

## Explicit bridge binding schema

Create a JSON object with `schema: "cruzr-isaac-bindings/v1"` and a nonempty
`links` array. Each item requires:

- `name`: actual source link name.
- `prim_path`: absolute USD path of that physical rigid link.
- `ply`: PLY path, relative to the binding JSON or absolute.
- `asset_unit_to_m`: 0.001 only for unconverted millimetre coordinates, else 1.0.
- `T_link_asset`: 4×4 rigid transform mapping metre-scaled asset coordinates to
  link-local coordinates, rows stored as arrays. No reflection, shear or scale.

No guessed CRUZR binding file is included. The original manifest must be reviewed.

With a validated metre-unit, Z-up USD and normalized bindings:

```bat
"D:\isaacsim-5.1.0\python.bat" run_isaac_bridge.py --stage robot.usd --bindings bindings.json --output results_robot
```

The stream uses wxyz quaternions internally and outputs row-major matrices for
column vectors: `T_world_asset = T_world_link @ T_link_asset`.
The renderer must apply `unit_to_m` to means and its square to covariance, then
apply the rigid transform to both means and covariance. View-dependent SH colour
requires the renderer's own corresponding local view-direction handling.

## Remaining integration gates

1. Obtain and audit original CRUZR project; preserve the working Motrix baseline.
2. Import CRUZR into Isaac and verify real joints/contacts before appearance work.
3. Connect this stream directly to the actual GS renderer; JSONL is a debug transport,
   not a demonstrated real-time IPC solution. Confirm all 41 bindings and camera
   intrinsics/extrinsics, shared depth/occlusion and robot/background alignment.
4. Export camera calibration/poses and validate RGB/depth separately. A Gaussian
   renderer's depth is not automatically metric sensor depth.
5. Validate GS and mesh scene alignment. Prepare FAR Planner integration only after
   odometry, point cloud, TF and velocity-command contracts are known.

P6000 server access/model is unverified. Workstation GPU is RTX 5060, 8 GB.
