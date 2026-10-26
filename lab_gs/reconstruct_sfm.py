"""Recover measured camera poses before attempting any Gaussian training."""
import json
from pathlib import Path
from PIL import Image, ImageOps
import pycolmap
import sqlite3

root=Path(__file__).resolve().parent
images=root/'images'
images.mkdir(exist_ok=True)
source=[]
for i,p in enumerate(sorted((root/'input').glob('*.jpg'))):
    name=f'lab_{i+1:02d}.jpg'
    im=ImageOps.exif_transpose(Image.open(p)).convert('RGB')
    im.resize((1280,960),Image.Resampling.LANCZOS).save(images/name,quality=96)
    source.append({'image':name,'source':p.name})
(root/'source_map.json').write_text(json.dumps(source,indent=2))
db=root/'sfm.db'
if not db.exists():
    pycolmap.extract_features(str(db),str(images),camera_mode=pycolmap.CameraMode.SINGLE,
                            camera_model='SIMPLE_RADIAL',device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(str(db),device=pycolmap.Device.cpu)
options=pycolmap.IncrementalPipelineOptions()
options.min_model_size=3
options.num_threads=8
options.random_seed=42
options.max_runtime_seconds=180
sparse=root/'sparse'
sparse.mkdir(exist_ok=True)
models=pycolmap.incremental_mapping(str(db),str(images),str(sparse),options=options)
report={'input_images':len(source),'models':[], 'status':'no_connected_model'}
for idx,rec in models.items():
    report['models'].append({'id':idx,'registered_images':rec.num_reg_images(),
                            'points3D':rec.num_points3D(),'mean_reprojection_error_px':rec.compute_mean_reprojection_error(),
                            'images':[im.name for im in rec.images.values() if im.has_pose]})
if report['models']:
    report['status']='poses_recovered'
    report['best_model']=max(report['models'],key=lambda x:x['registered_images'])['id']
with sqlite3.connect(db) as connection:
    report['verified_pairs']=connection.execute('SELECT COUNT(*) FROM two_view_geometries WHERE rows>=15').fetchone()[0]
(root/'sfm_report.json').write_text(json.dumps(report,indent=2))
print('SFM_RESULT '+json.dumps(report),flush=True)
