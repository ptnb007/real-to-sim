"""Render real bundled 3DGS assets from four calibrated perspective views."""
import bootstrap
import json
import numpy as np
from PIL import Image, ImageDraw
from gaussian_renderer import GSRenderer

out = bootstrap.ROOT/'results'
scene = bootstrap.ROOT.parent/'third_party/gs_playground/demo/navigation/nav_scene_1/3dgs/point_cloud.ply'
renderer = GSRenderer({'background':str(scene)})
positions = np.array([[0,0,1.45]]*4,dtype=np.float32)
targets = np.array([[-2,0,1.2],[0,2,1.2],[2,0,1.2],[0,-2,1.2]],dtype=np.float32)
rotations = []
for p,t in zip(positions,targets):
    forward=t-p
    forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1.])
    right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    rotations.append(np.column_stack([right,up,-forward]))
rgb,depth=renderer.render_batch(positions,np.array(rotations).reshape(4,9),720,960,np.full(4,70.))
assert np.isfinite(rgb.cpu().numpy()).all()
sheet=Image.new('RGB',(1920,1488),'#101827')
draw=ImageDraw.Draw(sheet)
for i,frame in enumerate(rgb):
    img=Image.fromarray((frame.clamp(0,1).cpu().numpy()*255).astype('uint8'))
    img.save(out/f'scene_view_{i+1}.png')
    x,y=(i%2)*960,(i//2)*744
    sheet.paste(img,(x,y+24))
    draw.text((x+12,y+5),f'GS-Playground | actual 3DGS render | view {i+1}',fill='white')
sheet.save(out/'GS_Playground_Views.png')
(out/'views.json').write_text(json.dumps({'rendered':True,'resolution':[960,720],
    'positions':positions.tolist(),'targets':targets.tolist(),'gaussians':len(renderer.gaussians),
    'note':'Official static navigation scene; not an Isaac RGB capture or completed navigation task.'},indent=2))
print('Rendered four 960x720 views')
