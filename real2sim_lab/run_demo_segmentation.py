"""Actual SAM3 text-prompt segmentation on the GS-Real2Sim bundled demo."""
import json
import sys
import time
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path[:0]=[str(root/'deps'),str(root.parent/'third_party/sam3'),str(root.parent/'third_party/iopath')]
import sam3_windows_compat
import torch
import numpy as np
from PIL import Image,ImageDraw
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

out=root/'demo_segmentation'
(out/'masks').mkdir(parents=True,exist_ok=True)
image=Image.open(root.parent/'third_party/gs-real2sim/example/example.png').convert('RGB')
image.save(out/'input_image.png')
start=time.time()
model=build_sam3_image_model(device='cuda',checkpoint_path=str(root/'checkpoints/sam3/sam3.pt'),
                             load_from_HF=False,enable_inst_interactivity=False,compile=False)
processor=Sam3Processor(model,device='cuda',confidence_threshold=.4)
report={'stage':'SAM3 segmentation only','objects':[],'generated_mesh':False,'generated_gs':False,
        'compatibility':'CPU/OpenCV EDT helper; upstream USE_PERFLIB=0; neural model unmodified'}
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    state=processor.set_image(image)
    preview=image.copy()
    draw=ImageDraw.Draw(preview)
    for prompt,color in [('pot',(255,70,70)),('bottle',(50,220,90)),('towel',(40,130,255))]:
        result=processor.set_text_prompt(prompt=prompt,state=state)
        masks=result['masks'].detach().cpu().numpy()
        scores=result['scores'].float().detach().cpu().numpy()
        entry={'prompt':prompt,'count':len(masks),'scores':scores.tolist()}
        for i,mask in enumerate(masks):
            mask=mask.squeeze()>0
            Image.fromarray((mask*255).astype('uint8')).save(out/'masks'/f'{prompt}_{i}.png')
            overlay=Image.new('RGB',image.size,color)
            blended=Image.blend(preview,overlay,.4)
            preview.paste(blended,(0,0),Image.fromarray((mask*255).astype('uint8')))
            yy,xx=np.where(mask)
            if len(xx):
                draw=ImageDraw.Draw(preview)
                draw.text((int(xx.min()),max(0,int(yy.min())-15)),prompt,fill=color)
        report['objects'].append(entry)
preview.save(out/'sam3_demo_masks.png')
report['elapsed_s']=time.time()-start
report['peak_gpu_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
(out/'report.json').write_text(json.dumps(report,indent=2))
print('SEGMENTATION_RESULT '+json.dumps(report),flush=True)
