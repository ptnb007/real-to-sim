"""Actual Isaac physics + upstream GPU Gaussian updates; reports render separately."""
import bootstrap
import json
import sys
import traceback
from pathlib import Path

out = bootstrap.ROOT / 'results'
out.mkdir(exist_ok=True)
runtime = bootstrap.ROOT / 'runtime'
for name in ('cache', 'data', 'logs'):
    (runtime / name).mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], '--/app/vulkan=false',
            *[f'--/app/tokens/{name}={(runtime/name).as_posix()}' for name in ('cache', 'data', 'logs')],
            f'--/log/file={(runtime/"logs/integration.log").as_posix()}']
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'multi_gpu': False, 'width': 320, 'height': 240})
report = {'physics_and_sync': 'running', 'rendered': False, 'fixture': 'synthetic Gaussian cube; not a reconstructed robot'}
(out/'integration.json').write_text(json.dumps(report), encoding='utf-8')
exit_code = 0
try:
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from PIL import Image
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from gaussian_renderer import GaussianData, save_ply
    from pxr import Gf, UsdGeom, PhysicsSchemaTools
    from isaac_adapter import IsaacGaussianAdapter

    # A local-frame Gaussian cube with an asymmetric marker tests rotation as well as translation.
    local = np.array([[x,y,z] for x in (-.1,0,.1) for y in (-.1,0,.1) for z in (-.1,0,.1)], dtype=np.float32)
    local = np.vstack([local, [.15, .025, .03]]).astype(np.float32)
    background = np.array([[x,y,0.] for x in np.linspace(-1,1,7) for y in np.linspace(-1,1,7)], dtype=np.float32)
    for name, xyz, color in [('cube', local, [.9,.25,.1]), ('background', background, [.2,.4,.7])]:
        n = len(xyz)
        data = GaussianData(xyz, np.tile([1.,0,0,0], (n,1)).astype('float32'),
                            np.full((n,3), .025, dtype='float32'), np.full(n,.95,dtype='float32'),
                            np.tile((np.array(color)-.5)/.28209479177387814, (n,1)).astype('float32'))
        save_ply(data, out/f'{name}.ply')
    world = World(stage_units_in_meters=1., physics_dt=1/120, rendering_dt=1/60)
    stage = omni.usd.get_context().get_stage()
    PhysicsSchemaTools.addGroundPlane(stage, '/World/Ground', 'Z', 10., Gf.Vec3f(0), Gf.Vec3f(.4))
    cube = world.scene.add(DynamicCuboid(prim_path='/World/Cube', name='cube', position=np.array([0.,0.,1.]),
                                      orientation=Rotation.from_euler('z',35,degrees=True).as_quat()[[3,0,1,2]],
                                      size=.2, mass=1.))
    camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    camera.AddTranslateOp().Set(Gf.Vec3d(0,0,3))
    camera_orientation = camera.AddOrientOp()
    camera_orientation.Set(Gf.Quatf(1., Gf.Vec3f(0)))
    camera.GetFocalLengthAttr().Set(24.)
    camera.GetVerticalApertureAttr().Set(24.)
    camera.GetHorizontalApertureAttr().Set(32.)
    world.reset()
    cube.set_angular_velocity(np.array([0.,0.,1.]))
    adapter = IsaacGaussianAdapter({'background':str(out/'background.ply'), 'cube':str(out/'cube.ply')}, {'cube':cube})
    r = adapter.renderer
    start, end = r.gaussian_start_indices['cube'], r.gaussian_end_indices['cube']
    original = r.gau_ori_xyz_all_cu[start:end].cpu().numpy().copy()
    static = r.gaussians.xyz[:start].clone()
    frames, max_error, max_rotation_error = [], 0., 0.
    for frame in range(120):
        world.step(render=False)
        adapter.update()
        p, q = cube.get_world_pose()
        expected = Rotation.from_quat(q, scalar_first=True).apply(original) + p
        actual = r.gaussians.xyz[start:end].cpu().numpy()
        max_error = max(max_error, float(np.max(np.abs(expected-actual))))
        rotations = r.gaussians.rot[start:end].cpu().numpy()
        max_rotation_error = max(max_rotation_error, float(np.max(1-np.abs(rotations @ q))))
        frames.append({'frame':frame, 'position':p.tolist(), 'quaternion_wxyz':q.tolist()})
    assert max_error < 1e-5, max_error
    assert max_rotation_error < 1e-5, max_rotation_error
    assert torch.equal(static, r.gaussians.xyz[:start])
    assert frames[0]['position'][2] > .9 and .08 < frames[-1]['position'][2] < .13
    # Independent projection check: USD -Z forward, +Y up -> CV +Z forward, +Y down.
    cv = np.diag([1.,-1.,-1.])
    points = np.array([[0,0,0],[.1,0,0],[0,.1,0.]])
    camera_pos, camera_rotation, height, width, fovy = adapter.camera_parameters(camera,320,240)
    camera_points = (points-camera_pos[0]) @ camera_rotation.reshape(3,3) @ cv
    focal = height/(2*np.tan(np.radians(fovy[0])/2))
    pixels = focal * camera_points[:,:2]/camera_points[:,2:] + [width/2,height/2]
    np.testing.assert_allclose(pixels, [[160,120],[168,120],[160,112]])
    camera_orientation.Set(Gf.Quatf(float(np.sqrt(.5)), Gf.Vec3f(0,float(np.sqrt(.5)),0)))
    cp, cr, _, _, _ = adapter.camera_parameters(camera,320,240)
    side_points = np.array([[-3,0,3],[-3,0,2.9],[-3,.1,3.]])
    side_cv = (side_points-cp[0]) @ cr.reshape(3,3) @ cv
    side_pixels = focal * side_cv[:,:2]/side_cv[:,2:] + [160,120]
    np.testing.assert_allclose(side_pixels, [[160,120],[168,120],[160,112]], atol=2e-5)
    camera_orientation.Set(Gf.Quatf(1., Gf.Vec3f(0)))
    report.update(physics_and_sync='passed', frames=len(frames), max_position_error_m=max_error,
                  max_quaternion_dot_error=max_rotation_error, static_background_unchanged=True,
                  camera_projection='passed (identity and 90-degree yaw)', initial_height_m=frames[0]['position'][2],
                  final_height_m=frames[-1]['position'][2], torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0))
    (out/'live_poses.json').write_text(json.dumps(frames, indent=2), encoding='utf-8')
    try:
        rgb, depth = adapter.render_usd_camera(camera,320,240)
        assert list(rgb.shape)==[1,240,320,3] and torch.isfinite(rgb).all()
        Image.fromarray((rgb[0].clamp(0,1).cpu().numpy()*255).astype('uint8')).save(out/'integration.png')
        report['rendered'] = True
    except Exception:
        report['render_error'] = traceback.format_exc()
        exit_code = 2
    stage.GetRootLayer().Export(str(out/'integration.usda'))
except Exception:
    report.update(physics_and_sync='failed', error=traceback.format_exc())
    exit_code = 1
finally:
    (out/'integration.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    print('GS_ISAAC_RESULT '+json.dumps(report), flush=True)
    app.close()
sys.exit(exit_code)
