"""Real-photo Mira preparation. Outputs remain distinct from synthetic v4 tests."""
import json
import sys
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path('/mnt/c/Users/woom/Desktop/woom')
REPO = Path('/home/woom/Mira-Scene')
sys.path.insert(0, str(REPO))
from infer_scripts.segmentation.backends.sam3 import Sam3Backend

OUT = ROOT / 'mira_scene/results/real_lab_mira'
sources = {
    'purple_table': ROOT / 'astra_scene/video_review/frame_00.jpg',
    'white_divider': ROOT / 'gaussian_splatting/input_photos_20260930/5473f5c5d4281be7c15cc84b828c47d6.jpg',
}
backend = Sam3Backend(REPO / 'external/sam3', REPO / 'checkpoints/sam3/sam3.pt')
for name, source in sources.items():
    folder = OUT / 'cases' / name / 'input'
    folder.mkdir(parents=True, exist_ok=True)
    im = Image.open(source).convert('RGB')
    im.save(folder / 'scene.png')
    records = []
    exclusion = np.zeros((im.height, im.width), dtype=bool)
    for prompt in ['robot', 'person']:
        with torch.inference_mode():
            masks, scores = backend.text(source, prompt)
        for mask, score in zip(masks, scores):
            if score >= .45:
                exclusion |= mask
    Image.fromarray(exclusion.astype('uint8') * 255).save(folder / 'excluded_robot_person.png')
    prompts = (['table covered with gray cloth', 'folding room divider', 'floor']
               if name == 'purple_table' else ['white folding room divider', 'floor'])
    idx = 0
    overlay = np.array(im)
    for prompt in prompts:
        with torch.inference_mode():
            masks, scores = backend.text(source, prompt)
        print(name, prompt, scores, flush=True)
        if prompt == 'floor' or 'divider' in prompt:
            selected = [mask for mask, score in zip(masks, scores) if score >= .45]
            if selected:
                masks = [np.logical_or.reduce(selected)]
                scores = [max(scores)]
        for mask, score in zip(masks, scores):
            if score < .45:
                continue
            mask = mask & ~exclusion
            if mask.sum() < 1000:
                continue
            if prompt == 'floor':
                filename = 'floor_mask.png'
            else:
                filename = f'mask_{idx:03d}.png'
                idx += 1
            Image.fromarray(mask.astype('uint8') * 255).save(folder / filename)
            overlay[mask] = (overlay[mask] * .55 + np.array([30,180,255]) * .45).astype('uint8')
            records.append({'prompt':prompt, 'score':float(score), 'mask':filename, 'pixels':int(mask.sum())})
    Image.fromarray(overlay).save(folder.parent / 'segmentation_review.jpg')
    (folder.parent / 'provenance.json').write_text(json.dumps({
        'source':str(source), 'input_type':'real photo/video frame',
        'method':'official SAM3 text segmentation; not VLM scene graph',
        'not_reconstructed':['excluded robot/person regions; no invented background'],
        'objects':records}, indent=2))
print('REAL_LAB_SEGMENTATION_COMPLETE', flush=True)
