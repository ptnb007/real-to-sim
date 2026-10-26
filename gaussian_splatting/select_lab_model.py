import json
import os
from pathlib import Path
from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary, read_points3D_binary
root = Path(os.environ.get('GS_DATA_ROOT', '/home/woom/gaussian_splatting_demo/lab_video_20260929'))
models=[]
for path in sorted((root/'sparse').iterdir()):
    if not (path/'images.bin').is_file(): continue
    images=read_images_binary(path/'images.bin')
    points=read_points3D_binary(path/'points3D.bin')
    models.append({'path':str(path.relative_to(root)), 'registered_images':len(images), 'points':len(points), 'mean_point_error_px':sum(p.error for p in points.values())/max(len(points),1)})
best=max(models,key=lambda x:(x['registered_images'],x['points']))
if best['registered_images']<10 or best['points']<100: raise RuntimeError('Insufficient reconstruction')
report={'input_frames':len(list((root/'images').rglob('*.jpg'))),'selected':best,'models':models,'metric_scale_calibrated':False}
(root/'reconstruction_report.json').write_text(json.dumps(report,indent=2))
(root/'selected_model.txt').write_text(best['path'])
print(json.dumps(report,indent=2))
