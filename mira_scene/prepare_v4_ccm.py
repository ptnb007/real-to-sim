"""Prepare traceable SAM3 masks; no hand-drawn geometry is called CCM output."""
from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image

root = Path('/mnt/c/Users/woom/Desktop/woom')
source = root / 'mira_scene/results/v4_sam3'
case = root / 'mira_scene/results/v4_ccm/cases/v4_eye/input'
case.mkdir(parents=True, exist_ok=True)
shutil.copy2(root / 'astra_scene/focus_v4/eye_level.png', case / 'scene.png')
names = ['table_covered_with_a_gray_tablecloth_00',
         'table_covered_with_a_gray_tablecloth_01', 'folding_room_divider_00',
         'television_00']
manifest = {'input_type': 'synthetic v4 Blender render, NOT real photograph',
            'segmentation': 'official SAM3 text prompts; human visual review',
            'limitations': ['table masks predominantly cover cloth; legs omitted',
                            'television mask may omit stand',
                            'white screen panel masks require instance consolidation'],
            'objects': []}
for i, name in enumerate(names):
    shutil.copy2(source / (name + '.png'), case / f'mask_{i:03d}.png')
    manifest['objects'].append({'index': i, 'source_mask': name})
shutil.copy2(source / 'floor_00.png', case / 'floor_mask.png')
(case.parent / 'provenance.json').write_text(json.dumps(manifest, indent=2))
print(case)
