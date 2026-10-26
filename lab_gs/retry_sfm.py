"""Second independent calibration hypothesis: no shared-intrinsics assumption."""
import json
from pathlib import Path
import pycolmap

root=Path(__file__).resolve().parent
db=root/'sfm_independent.db'
if not db.exists():
    pycolmap.extract_features(str(db),str(root/'input'),camera_mode=pycolmap.CameraMode.PER_IMAGE,
                            camera_model='SIMPLE_RADIAL',device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(str(db),device=pycolmap.Device.cpu)
opt=pycolmap.IncrementalPipelineOptions()
opt.min_model_size=3
opt.num_threads=8
opt.random_seed=42
opt.max_runtime_seconds=180
out=root/'sparse_independent'
out.mkdir(exist_ok=True)
models=pycolmap.incremental_mapping(str(db),str(root/'input'),str(out),options=opt)
report={'input_images':9,'camera_hypothesis':'PER_IMAGE, original resolution','models':[]}
for idx,rec in models.items():
    report['models'].append({'id':idx,'registered_images':rec.num_reg_images(),'points3D':rec.num_points3D(),
                            'mean_reprojection_error_px':rec.compute_mean_reprojection_error(),
                            'images':[im.name for im in rec.images.values() if im.has_pose]})
(root/'sfm_independent_report.json').write_text(json.dumps(report,indent=2))
print('SFM_RESULT '+json.dumps(report),flush=True)
