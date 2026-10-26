import sys
import json
from pathlib import Path
import torch
from PIL import Image
import numpy as np
sys.path.insert(0, '/home/woom/Mira-Scene')
from infer_scripts.segmentation.backends.sam3 import Sam3Backend
root = Path('/mnt/c/Users/woom/Desktop/woom')
out = root / 'mira_scene/results/v4_sam3'
out.mkdir(parents=True, exist_ok=True)
source = root / 'astra_scene/focus_v4/eye_level.png'
backend = Sam3Backend(Path('/home/woom/Mira-Scene/external/sam3'), Path('/home/woom/Mira-Scene/checkpoints/sam3/sam3.pt'))
records = []
with torch.inference_mode():
    for label in ['table covered with a gray tablecloth', 'desk', 'white folding screen', 'purple partition']:
        masks, scores = backend.text(source, label)
        print(label, scores, flush=True)
        rgb = np.array(Image.open(source).convert('RGB'))
        for i, (mask, score) in enumerate(zip(masks, scores)):
            name = label.replace(' ', '_') + f'_{i:02d}'
            Image.fromarray(mask.astype('uint8') * 255).save(out / (name + '.png'))
            rgb[mask] = (rgb[mask] * .5 + np.array([30, 180, 255]) * .5).astype('uint8')
            records.append({'name': name, 'prompt': label, 'score': score, 'pixels': int(mask.sum())})
        Image.fromarray(rgb).save(out / (label.replace(' ', '_') + '_overlay.jpg'))
(out / 'detections.json').write_text(json.dumps(records, indent=2))
