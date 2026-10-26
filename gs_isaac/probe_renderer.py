"""Try upstream GS rendering on an actual bundled navigation scene."""
import bootstrap
import json
import traceback
import numpy as np
import torch
from PIL import Image
from gaussian_renderer import GSRenderer

out = bootstrap.ROOT / 'results'
out.mkdir(exist_ok=True)
scene = bootstrap.ROOT.parent / 'third_party/gs_playground/demo/navigation/nav_scene_1/3dgs/point_cloud.ply'
report = {'scene': str(scene), 'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'rendered': False}
try:
    renderer = GSRenderer({'background': str(scene)})
    report['gaussians'] = len(renderer.gaussians)
    report['bounds'] = [renderer.gaussians.xyz.amin(0).tolist(), renderer.gaussians.xyz.amax(0).tolist()]
    rgb, depth = renderer.render_batch(np.array([[0., 0., 2.]]), np.eye(3).reshape(1, 9), 240, 320, np.array([60.]))
    Image.fromarray((rgb[0].clamp(0, 1).cpu().numpy()*255).astype('uint8')).save(out / 'static_scene.png')
    report.update(rendered=True, rgb_shape=list(rgb.shape), depth_shape=list(depth.shape))
except Exception:
    report['error'] = traceback.format_exc()
(out / 'renderer_probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['rendered'] else 2)
