# Room-scale human search experiment

This isolated experiment reuses the existing VLFM ITMPolicyV2 exploration,
BLIP2 value map, MobileSAM object point cloud, and pretrained PointNav controller.
An adapter uses the literal goal `person` and GroundingDINO instead of the
original six-category HM3D goal-ID lookup. The baseline files are unchanged.

## Run on the configured WSL host

```bash
bash /mnt/c/Users/woom/Desktop/woom/vlfm/scripts/run_person_search_wsl.sh \
  --run-name my_person_run --max-steps 200
```

Choose a new run name each time. Existing results are never overwritten. The
wrapper starts three model services on dedicated ports 12281-12283 and stops only
the processes it created, on both success and failure. It does not touch other
GPU applications or the original VLFM server ports.

To inspect placement without loading models:

```bash
cd /home/woom/vlfm
/home/woom/miniconda3/envs/vlfm/bin/python \
  /mnt/c/Users/woom/Desktop/woom/vlfm/scripts/person_search_demo.py --prepare \
  --output /mnt/c/Users/woom/Desktop/woom/vlfm/results/person_search/cesium_prepare
```

## Integrity and evaluation

- Scene: HM3D validation scene `00800-TEEsavR23oF`.
- One static 1.7 m Cesium Man avatar; source and CC BY license are in
  `data/person_search/README.md`.
- Target placement is seeded, on the starting floor, with floor clearance and
  a reachable path. Static-object navmesh rebuilding prevents walking through it.
- Preparation-camera probes are **not** performed during actual navigation.
- Policy inputs are explicitly restricted to RGB, depth, GPS, compass, heading,
  and the string `person`. No target coordinates or semantic images reach it.
- Semantic instance ID 10001 is used only by the evaluator to validate that a
  detection box overlaps the inserted avatar (at least 50 target pixels).
- Custom success: policy explicitly calls STOP, it has detected the inserted
  avatar, the most recent matching detection is within five steps, and final
  horizontal distance to the avatar center is at most 1.2 m.
- This is NOT the original HM3D toilet success metric, a full benchmark,
  photorealistic human recognition, dynamic search, or humanoid locomotion.

Each run keeps setup coordinates (evaluation ground truth), per-step detections
and trajectory, metrics, model logs, navigation log, and an MP4. Failed runs are
kept too. Playback runs at 8 frames/s; this is not wall-clock inference speed.

## Verified result — 2026-09-03

`results/person_search/run_03` completed one custom episode successfully:

- 72 recorded steps: 12 initialization, 47 exploration, 13 target-navigation.
- The person was invisible at the initial pose. First evaluator-confirmed
  detection was at zero-based step 59; explicit STOP was at step 71.
- Travel distance: 8.250 m. Final horizontal target-center distance: 0.956 m.
- Detection confidence threshold: 0.65. Zero frames with retained detections
  but no box matching the inserted person under the overlap criterion above.
  This is a frame-level diagnostic, not a general detector precision score.
- Zero forward actions with less than 0.05 m displacement. This is not a
  collision-rate or dynamic-obstacle-avoidance benchmark.
- Navigation loop time: 41.65 seconds, excluding model-service startup.
- Policy received no target coordinates or semantic instance images. A separate
  read-only audit recomputed distance/path length from the recorded trajectory,
  checked detection scores, explicit STOP and mode transitions, and passed.
- Model services were stopped after completion. GPU usage returned to
  1183 MiB total device usage; other desktop applications were not stopped.

The earlier `run_02` also reached the person but used threshold 0.35 and
included false-positive detections of room decorations. The 0.65 threshold was
chosen after inspecting that run. Therefore `run_03` is a tuned demonstration
in the same scene, **not held-out evidence of generalization**. `run_01` failed
due to an evaluation-mask shape mismatch; its logs remain available.

The avatar is a stylized static T-pose figure, not a real trapped person. The
video shows simulated RGB-D observations and learned-policy actions. This
milestone establishes a single-person search-and-approach loop, not multi-person
coverage, rescue assessment, humanoid control, or manipulation.

Read-only evidence audit (does not start GPU models):

```bash
python scripts/verify_person_search.py results/person_search/run_03
```

## Realistically textured avatar option

The default `--asset cesium` preserves the original stylized baseline. To use
the MIT-licensed Microsoft Rocketbox adult with face, skin, clothing and hair
textures, select `--asset rocketbox`. The model is a static virtual person;
no claim of a real person's scan or real rescue capability is made.

```bash
bash /mnt/c/Users/woom/Desktop/woom/vlfm/scripts/run_person_search_wsl.sh \
  --run-name my_rocketbox_run --asset rocketbox --yaw 2.3 --max-steps 200
```

The existing confidence threshold remains 0.65. Pose/orientation and asset hash
are recorded in each run. Preparation is isolated in its own output directory
and never runs inside the policy episode:

```bash
cd /home/woom/vlfm
/home/woom/miniconda3/envs/vlfm/bin/python \
  /mnt/c/Users/woom/Desktop/woom/vlfm/scripts/person_search_demo.py \
  --prepare --asset rocketbox --yaw 2.3 \
  --output /mnt/c/Users/woom/Desktop/woom/vlfm/results/person_search/my_rocketbox_prepare
```

See `data/person_search/rocketbox/README.md` for source, licensing, conversion
instructions and the distinction between realistic textures and a real person.
The `rocketbox_prepare_01` image documents an intermediate arm-pose conversion
error; `rocketbox_prepare_02` uses the corrected relaxed standing pose. Neither
is a navigation episode. Original Cesium results remain unchanged.

### Verified Rocketbox result — 2026-09-03

`results/person_search/rocketbox_run_01` successfully completed the same custom
search task with the textured adult, yaw 2.3 radians and threshold 0.65:

- 70 recorded steps: 12 initialization, 47 exploration, 11 target-navigation.
- Person not visible initially. First confirmed detection: zero-based step 59.
- Explicit STOP at step 69, at 0.956 m horizontal distance to target center.
- Travel distance 8.250 m; navigation-loop time 41.64 s excluding service startup.
- Zero unmatched-detection frames under the evaluation criterion; zero blocked
  forward steps under the displacement diagnostic. Neither is a general safety
  or detector-accuracy claim.
- Independent artifact audit passed, including starting pose, path length,
  confidence filtering, target overlap, STOP, mode transitions and asset hash.
- All model services owned by this run exited after completion.

The MP4 contains the 70 recorded frames at 8 fps (8.75 s playback), not a
real-time speed demonstration. `person_detected.jpg` shows the first confirmed
full-body detection, and `final.jpg` shows the close stopping position.

This is one static virtual avatar in the previously used room with a selected
orientation, not an unseen-room benchmark or evidence of real rescue readiness.
