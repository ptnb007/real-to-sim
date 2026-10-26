# Multi-room, multi-pose person-search experiment

This isolated experiment extends the single-person adapter without changing the
original VLFM baseline. It inserts the same realistically textured adult in
three static poses (standing, seated, lying) at safe, reachable locations
assigned to three distinct HM3D semantic region IDs.

The policy input allow-list is RGB, depth, GPS, compass, heading and the text
`person`. Simulator coordinates, semantic images, region IDs and pose labels are
evaluation-only. GroundingDINO detects `person`; MobileSAM builds the object
point cloud; VLFM's BLIP2 value map selects unexplored frontiers; the learned
PointNav policy executes movement.

After a policy STOP is independently confirmed against one inserted semantic
instance and a 1.25 m radius, the evaluator records the pose/room and removes
that already-found avatar. This explicit evaluation intervention prevents the
single-target VLFM object map from repeatedly returning to the same person. It
does **not** pass the positions of remaining people to the policy. The policy
then performs twelve 30-degree turns to scan that room and resumes exploration.

Two completion levels are reported separately. **Room-scan completion** means
that all three people were confirmed in distinct rooms and every found room
received a twelve-view 360-degree scan. **Whole-scene frontier completion** is
stricter: room scanning is complete and VLFM's map reports no remaining
frontier of at least 1.5 square metres. Neither metric means every physical
surface was imaged or proves generalization to unseen buildings.

Run preparation without GPU models:

```bash
cd /home/woom/vlfm
/home/woom/miniconda3/envs/vlfm/bin/python \
  /mnt/c/Users/woom/Desktop/woom/vlfm/scripts/multi_person_search_demo.py \
  --prepare --output /mnt/c/Users/woom/Desktop/woom/vlfm/results/multi_person_search/my_prepare
```

Run the managed experiment (dedicated ports 12381-12383):

```bash
bash /mnt/c/Users/woom/Desktop/woom/vlfm/scripts/run_multi_person_search_wsl.sh \
  --run-name my_run --max-steps 750 --person-confidence 0.58
```

The manager stops only the model processes it starts. Every run name must be
new, and all failed runs are retained rather than overwritten.

Because the baseline object point-cloud map can retain a one-frame hypothesis
indefinitely, this extension discards a target hypothesis after twelve frames
without any repeated person detection. This anti-stale rule uses only detector
history, never evaluator semantics or the coordinates of inserted people.

## Verified run (run_04)

The policy found seated, standing and lying people at steps 91, 134 and 584,
respectively, in three distinct semantic regions. It visited seven regions,
travelled 81.56 m, and completed all three twelve-view room scans. The 750-step
budget ended while a map frontier still remained, so room-scan completion
passed while whole-scene frontier completion did not.
