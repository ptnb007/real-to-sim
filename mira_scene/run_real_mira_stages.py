import os
import json
import subprocess
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='/mnt/c/Users/woom/Desktop/woom/mira_scene/results/real_lab_mira')
opts = parser.parse_args()
root = Path(opts.root)
repo = Path('/home/woom/Mira-Scene')
cases = root / 'cases'
results = []
jobs = [
    ('ccm', '/home/woom/mira_envs/ccm-rtx5060/bin/python',
     ['infer_scripts/2_inference_CCM.py', '--demo_dir', str(cases),
      '--ckpt_dir', str(repo / 'checkpoints/mira-scene/pipeline'),
      '--use_cropped_condition', '--num_inference_steps', '30']),
    ('depth', '/home/woom/mira_envs/geometry-rtx5060/bin/python',
     ['infer_scripts/1_estimation_depth.py', '--data-root', str(cases), '--method', 'moge', '--device', 'cuda']),
]
for name, python, args in jobs:
    env = dict(os.environ)
    env['PYTHONPATH'] = str(repo / 'external/MoGe')
    with (root / (name + '.log')).open('w') as log:
        proc = subprocess.run([python, '-u', *args], cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    results.append({'stage':name, 'exit_code':proc.returncode, 'input':str(cases)})
    (root / 'stage_status.json').write_text(json.dumps(results, indent=2))
    print(name, proc.returncode, flush=True)
