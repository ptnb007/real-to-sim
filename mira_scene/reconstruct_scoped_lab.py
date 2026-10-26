"""Official SAM3 preprocessing for the scoped, robot-free lab Mira cases."""
import sys, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch

REPO=Path('/home/woom/Mira-Scene')
ROOT=Path('/mnt/c/Users/woom/Desktop/woom/mira_scene/results/lab_between_dividers')
sys.path.insert(0,str(REPO))
from infer_scripts.segmentation.backends.sam3 import Sam3Backend
backend=Sam3Backend(REPO/'external/sam3',REPO/'checkpoints/sam3/sam3.pt')
jobs=[('tables_purple',ROOT/'cleaned/07.png',['table covered with gray cloth','purple folding room divider']),
      ('white_boundary',next((ROOT/'originals').glob('09_*.jpg')),['white folding room divider'])]
for name,source,prompts in jobs:
    case=ROOT/'cases'/name
    folder=case/'input'; folder.mkdir(parents=True,exist_ok=True)
    rgb=Image.open(source).convert('RGB'); rgb.save(folder/'scene.png')
    records=[]; idx=0
    for prompt in prompts+['floor']:
        with torch.inference_mode(): masks,scores=backend.text(source,prompt)
        selected=[(np.asarray(m,dtype=bool),float(s)) for m,s in zip(masks,scores) if float(s)>=.45]
        if 'divider' in prompt or prompt=='floor':
            selected=[(np.logical_or.reduce([m for m,s in selected]),max(s for m,s in selected))] if selected else []
        for mask,score in selected:
            if mask.sum()<1000: continue
            filename='floor_mask.png' if prompt=='floor' else f'mask_{idx:03d}.png'
            if prompt!='floor': idx+=1
            Image.fromarray(mask.astype('uint8')*255).save(folder/filename)
            records.append({'prompt':prompt,'score':score,'mask':filename,'pixels':int(mask.sum())})
    (case/'provenance.json').write_text(json.dumps({'source':str(source),'image_inpainted':name=='tables_purple',
       'scope':'tables, divider, floor; other scene content excluded by masks',
       'scale':'uncalibrated','objects':records},indent=2))
    print(name,records,flush=True)
