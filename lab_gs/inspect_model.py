import pycolmap
from pathlib import Path
rec=pycolmap.Reconstruction(str(Path(__file__).parent/'sparse_independent/0'))
for key,im in rec.images.items():
    if im.has_pose:
        print(im.name,im.cam_from_world(),rec.cameras[im.camera_id])
        print('K',rec.cameras[im.camera_id].calibration_matrix())
        break
print(pycolmap.undistort_images.__doc__)
