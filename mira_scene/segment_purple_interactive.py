"""SAM3 box-guided mask for the previously inspected real video frame."""
import sys, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
repo = Path('/home/woom/Mira-Scene')
sys.path.insert(0, str(repo))
from infer_scripts.segmentation.backends.sam3 import Sam3Backend
root = Path('/mnt/c/Users/woom/Desktop/woom')
source = root / 'astra_scene/video_review/frame_00.jpg'
out = root / 'mira_scene/results/real_lab_mira/purple_mask_review'
out.mkdir(exist_ok=True)
backend = Sam3Backend(repo/'external/sam3', repo/'checkpoints/sam3/sam3.pt')
with torch.inference_mode():
    masks, scores = backend.interactive(source, box=[245,0,623,473],
        points=[[335,140],[503,100],[335,370],[720,300]], labels=[1,1,1,0])
for i, (mask, score) in enumerate(zip(masks,scores)):
    Image.fromarray(mask.astype('uint8')*255).save(out/f'candidate_{i}.png')
    rgb=np.array(Image.open(source).convert('RGB'))
    rgb[mask]=(rgb[mask]*.5+np.array([30,180,255])*.5).astype('uint8')
    Image.fromarray(rgb).save(out/f'candidate_{i}_review.jpg')
(out/'scores.json').write_text(json.dumps(scores))
print(scores, flush=True)
