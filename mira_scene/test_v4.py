import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

repo = Path('/home/woom/Mira-Scene')
workspace = Path('/mnt/c/Users/woom/Desktop/woom')
output = workspace / 'mira_scene/results/v4_test'
output.mkdir(parents=True, exist_ok=True)
cases = output / 'cases'
config = yaml.safe_load((repo / 'infer_scripts/config/example.yaml').read_text())
for entry in config['environments'].values():
    entry['python'] = sys.executable
config['external']['moge']['repo'] = str(repo / 'external/MoGe')
config['depth'] = {'method': 'moge', 'enable_metric': False}
config['floor'] = {'generate_texture': False, 'require_generated': False}
config_path = output / 'test_config.yaml'
config_path.write_text(yaml.safe_dump(config))
env = dict(os.environ, PYTHONPATH=str(repo / 'external/MoGe'))
results = []
def run(name, arguments):
    p = subprocess.run([sys.executable, *arguments], cwd=repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (output / (name + '.log')).write_text(p.stdout)
    results.append({'test': name, 'exit_code': p.returncode})
    print(name, p.returncode, p.stdout[-1800:], flush=True)

for name, filename in [('v4_eye', 'eye_level.png'), ('v4_overview', 'preview.png')]:
    folder = cases / name / 'input'
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(workspace / 'astra_scene/focus_v4' / filename, folder / 'scene.png')
run('full_preflight', ['infer_scripts/pipeline.py', '--input', str(cases / 'v4_eye/input/scene.png'), '--output', str(output / 'preflight'), '--config', str(config_path), '--preflight-only'])
for name in ['v4_eye', 'v4_overview']:
    run(name + '_depth', ['infer_scripts/1_estimation_depth.py', '--data-root', str(cases), '--case', name, '--method', 'moge', '--device', 'cuda'])
    path = cases / name / 'depth/moge/depth.npy'
    if path.exists():
        d = np.load(path)
        valid = np.isfinite(d) & (d > 0)
        fig, ax = plt.subplots(figsize=(10, 8))
        p = ax.imshow(d, cmap='turbo_r', vmin=np.percentile(d[valid], 2), vmax=np.percentile(d[valid], 98))
        ax.set_title(name + ' / relative depth (NOT calibrated meters)')
        ax.axis('off')
        fig.colorbar(p, ax=ax)
        fig.savefig(output / (name + '_depth.png'))
        plt.close(fig)

# Conservative visible floor patch in the inspected 1200x1000 eye-level render.
mask = Image.new('L', Image.open(cases / 'v4_eye/input/scene.png').size)
ImageDraw.Draw(mask).polygon([(90, 840), (220, 795), (475, 895), (515, 970), (155, 970)], fill=255)
mask.save(cases / 'v4_eye/input/floor_mask.png')
run('v4_eye_floor', ['infer_scripts/4_estimate_floor.py', '--demo_dir', str(cases), '--case', 'v4_eye', '--depth_method', 'moge'])
(output / 'test_results.json').write_text(json.dumps(results, indent=2))
