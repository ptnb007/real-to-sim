"""Prepare SAM3 instance masks for the robot/divider Mira reconstruction."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

repo = Path('/home/woom/Mira-Scene')
sys.path.insert(0, str(repo))
from infer_scripts.segmentation.backends.sam3 import Sam3Backend

case = Path('/mnt/c/Users/woom/Desktop/woom/mira_scene/results/real_lab_mira/cases/robot_divider_scene')
image_path = case / 'input/scene.png'
output = case / 'input'
rgb = np.asarray(Image.open(image_path).convert('RGB'))
backend = Sam3Backend(repo / 'external/sam3', repo / 'checkpoints/sam3/sam3.pt')
prompts = [
    ('robot', ['robot']),
    ('divider', ['purple folding room divider', 'white folding room divider', 'folding privacy screen']),
    ('table', ['table covered with black cloth']),
    ('floor', ['floor']),
]
records = []
idx = 0
overlay = rgb.copy()
palette = np.array([[30, 180, 255], [255, 120, 40], [80, 220, 100], [250, 80, 180]], dtype=np.uint8)
# Clear only masks generated in this dedicated, newly-created case directory.
for stale_mask in output.glob('mask_*.png'):
    stale_mask.unlink()
with torch.inference_mode():
    for category, prompt_variants in prompts:
        candidates = []
        for prompt in prompt_variants:
            masks, scores = backend.text(image_path, prompt)
            print(category, prompt, [float(x) for x in scores], flush=True)
            candidates.extend((mask, float(score), prompt) for mask, score in zip(masks, scores) if score >= 0.45 and int(mask.sum()) >= 300)
            if candidates and category != 'divider':
                break
        kept = [(mask, score, prompt) for mask, score, prompt in candidates]
        if category == 'floor':
            if kept:
                mask = np.logical_or.reduce([m for m, _, _ in kept])
                Image.fromarray(mask.astype(np.uint8) * 255).save(output / 'floor_mask.png')
            continue
        for mask, score, prompt in kept:
            name = f'mask_{idx:03d}.png'
            Image.fromarray(mask.astype(np.uint8) * 255).save(output / name)
            overlay[mask] = (overlay[mask].astype(np.float32) * 0.55 + palette[idx % len(palette)] * 0.45).astype(np.uint8)
            records.append({'category': category, 'prompt': prompt, 'score': score, 'mask': name, 'pixels': int(mask.sum())})
            idx += 1
Image.fromarray(overlay).save(case / 'sam3_review.png')
(case / 'sam3_masks.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
print('MASKS_READY', len(records), flush=True)
