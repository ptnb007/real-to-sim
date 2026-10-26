"""Bundle source, upstream snapshots, and actual validation evidence."""
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parent.parent
task = root/'gs_isaac'
result = json.loads((task/'results/integration.json').read_text(encoding='utf-8'))
assert result['physics_and_sync'] == 'passed'
assert result['frames'] == 120
assert result['camera_projection'] == 'passed (identity and 90-degree yaw)'
destination = root/'GS_Playground_Isaac_Adapter.zip'
files = [p for p in task.iterdir() if p.suffix in ('.py','.bat','.md','.log')]
files += [p for p in (task/'results').iterdir() if p.is_file()]
files += [task/'prebuilt_source.json']
files += [p for p in (task/'prebuilt').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
renderer = root/'third_party/GaussianRenderer'
files += [p for p in (renderer/'src').rglob('*.py')]
files += [renderer/name for name in ('LICENSE','README.md','AGENTS.md','pyproject.toml')]
playground = root/'third_party/gs_playground'
files += [playground/name for name in ('LICENSE','README.md','pyproject.toml')]
files += [p for p in (playground/'demo/navigation/nav_scene_1').rglob('*') if p.is_file()]
files += [playground/'demo/navigation/robot_locomotion.py',playground/'demo/live_demo/replay.py']
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(set(files)):
        archive.write(file, file.relative_to(root).as_posix())
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
print(json.dumps({'path':str(destination),'bytes':destination.stat().st_size,'files':len(files),
                  'physics_and_sync':result['physics_and_sync'],'rendered':result['rendered']}))
