"""Render three training-camera previews from this project's trusted checkpoint."""
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
import sys
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
config,pipeline,checkpoint,step=eval_setup(Path(sys.argv[1]),test_mode='val')
out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
cameras=pipeline.datamanager.train_dataset.cameras
ids=[0,len(cameras)//2,len(cameras)-1]
with torch.no_grad():
 for i in ids:
  cam=cameras[i:i+1].to(pipeline.device)
  rgb=pipeline.model.get_outputs_for_camera(cam)['rgb'].clamp(0,1).cpu().numpy()
  Image.fromarray((rgb*255).astype(np.uint8)).save(out/f'camera_{i:03d}.png')
(out/'preview_metadata.json').write_text(json.dumps({'checkpoint':str(checkpoint),'step':step,'camera_indices':ids,'note':'Training-camera previews, not held-out evaluation'},indent=2))
