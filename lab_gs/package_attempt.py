import json
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parent
destination=root.parent/'Lab_Region_GS_Attempt.zip'
files=[p for p in root.iterdir() if p.suffix in ('.py','.bat','.md','.json','.log')]
for folder in ('input','partial_gs','sparse','sparse_independent'):
    files += [p for p in (root/folder).rglob('*') if p.is_file()]
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
    for file in sorted(set(files)):
        archive.write(file,file.relative_to(root).as_posix())
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
print(json.dumps({'file':str(destination),'bytes':destination.stat().st_size,'visual_quality':'failed','full_region_complete':False}))
