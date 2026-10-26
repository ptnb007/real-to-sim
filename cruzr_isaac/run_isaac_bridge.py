"""Isaac Sim 5.1 -> renderer-neutral per-link pose stream.

--smoke validates actual falling-body physics and pose export without CRUZR.
--stage + --bindings exports live rigid-body poses from a supplied robot stage.
This does not implement a Gaussian renderer or claim CRUZR migration is complete.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

from pose_bridge import load_bindings, make_frame, matrix

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--stage", type=Path)
parser.add_argument("--bindings", type=Path)
parser.add_argument("--frames", type=int, default=120)
parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
args, kit_args = parser.parse_known_args()
if args.frames < 60:
    parser.error("At least 60 frames required")
if args.smoke == bool(args.stage):
    parser.error("Choose either --smoke or --stage")
if not args.smoke and (not args.bindings or not args.stage.is_file()):
    parser.error("--stage must exist and --bindings is required")
# Validate assets before starting the expensive runtime.
bindings = None if args.smoke else load_bindings(args.bindings)
runtime = (args.output.resolve().parent / "runtime").as_posix()
for folder in ("cache", "data", "logs"):
    Path(runtime, folder).mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], *kit_args, "--/app/vulkan=false",
            f"--/app/tokens/cache={runtime}/cache",
            f"--/app/tokens/data={runtime}/data",
            f"--/app/tokens/logs={runtime}/logs",
            f"--/log/file={runtime}/logs/bridge.log"]
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting",
                     "multi_gpu": False, "width": 640, "height": 480})
exit_code = 0
try:
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.prims import SingleRigidPrim
    from isaacsim.core.utils.stage import open_stage
    from pxr import Gf, PhysicsSchemaTools, UsdGeom, UsdPhysics

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "verification.json"
    # Invalidate any previous success before running.
    report_path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    if args.stage:
        if not open_stage(str(args.stage.resolve())):
            raise RuntimeError("Could not open supplied USD")
        stage = omni.usd.get_context().get_stage()
        if abs(UsdGeom.GetStageMetersPerUnit(stage)-1.0) > 1e-8:
            raise ValueError("Stage must be in metres; explicit conversion required")
        if UsdGeom.GetStageUpAxis(stage) != "Z":
            raise ValueError("Expected Z-up stage; explicit conversion required")
    world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=1/60)
    stage = omni.usd.get_context().get_stage()
    if args.smoke:
        PhysicsSchemaTools.addGroundPlane(stage, "/World/Ground", "Z", 10.,
                                         Gf.Vec3f(0), Gf.Vec3f(.4))
        cube = world.scene.add(DynamicCuboid(prim_path="/World/BridgeTestBody", name="bridge_test",
                               position=np.array([0., 0., 1.]), size=.2, mass=1.))
        bindings = [{"name": "physics_test_body", "prim_path": "/World/BridgeTestBody",
                     "ply": None, "asset_unit_to_m": 1.,
                     "T_link_asset": matrix([0,0,0], [1,0,0,0])}]
        # Camera convention is stated explicitly: USD cameras look along local -Z.
        camera = UsdGeom.Camera.Define(stage, "/World/BridgeTestCamera")
        camera.AddTranslateOp().Set(Gf.Vec3d(2, 2, 2))
        camera.GetFocalLengthAttr().Set(24.)
    bodies = {}
    for item in bindings:
        prim = stage.GetPrimAtPath(item["prim_path"])
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Expected physical rigid link: {item['prim_path']}")
        bodies[item["name"]] = SingleRigidPrim(item["prim_path"], name="export_"+item["name"])
    world.reset()
    for body in bodies.values():
        body.initialize()
    start, end = None, None
    with (output / "poses.jsonl").open("w", encoding="utf-8") as stream:
        for frame in range(args.frames):
            world.step(render=False)
            poses = {}
            for name, body in bodies.items():
                p, q = body.get_world_pose()
                poses[name] = ([float(v) for v in p], [float(v) for v in q])
            packet = make_frame(frame, float(world.current_time), bindings, poses)
            if start is None:
                start = packet
            end = packet
            stream.write(json.dumps(packet, allow_nan=False)+"\n")
    checks = {"finite_pose_stream": True, "frames": args.frames, "links": len(bindings)}
    if args.smoke:
        z_start = start["links"][0]["T_world_asset"][2][3]
        z_end = end["links"][0]["T_world_asset"][2][3]
        checks.update(initial_z_m=z_start, final_z_m=z_end,
                      fell_under_gravity=z_end < z_start-.2,
                      rested_on_ground=abs(z_end-.1) < .03)
        if not checks["fell_under_gravity"] or not checks["rested_on_ground"]:
            raise RuntimeError(f"Physics smoke validation failed: {checks}")
        stage.GetRootLayer().Export(str(output / "bridge_smoke.usda"))
    result = {"status": "passed", "mode": "physics_smoke" if args.smoke else "supplied_robot_pose_export",
              "checks": checks, "scope": "Physics + pose stream only. No CRUZR model or 3DGS rendering validated."}
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("BRIDGE_VERIFIED " + json.dumps(result), flush=True)
except BaseException as exc:
    exit_code = 1
    traceback.print_exc()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verification.json").write_text(json.dumps({"status": "failed", "error": str(exc)}, indent=2), encoding="utf-8")
finally:
    app.close()
if exit_code:
    raise SystemExit(exit_code)
