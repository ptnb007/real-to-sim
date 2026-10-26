"""Measured five-view 3DGS draft. Does not claim the disconnected white-board area."""
import json
import os
import sys
import time
from pathlib import Path

root=Path(__file__).resolve().parent
shared=root.parent/'gs_isaac'
sys.path[:0]=[str(shared/'prebuilt'),str(shared/'deps'),str(root.parent/'third_party/GaussianRenderer/src')]
os.environ['TORCH_COMPILE_DISABLE']='1'
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
import pycolmap
from scipy.spatial import cKDTree
from gsplat import rasterization,DefaultStrategy
from gaussian_renderer import GaussianData,save_ply

torch.manual_seed(42)
np.random.seed(42)
out=root/'partial_gs'
out.mkdir(exist_ok=True)
rec=pycolmap.Reconstruction(str(root/'sparse_independent/0'))
images=[im for im in rec.images.values() if im.has_pose]
centers=np.array([im.projection_center() for im in images])
center=centers.mean(0)
scale=np.max(np.linalg.norm(centers-center,axis=1))
points=[p for p in rec.points3D.values() if p.error<2 and p.track.length()>=3]
xyz=(np.array([p.xyz for p in points])-center)/scale
colors=np.array([p.color for p in points],dtype=np.float32)/255
neighbors=cKDTree(xyz).query(xyz,k=min(4,len(xyz)))[0][:,1:]
sizes=np.maximum(np.sqrt((neighbors**2).mean(1))*.5,.005)
n=len(xyz)
params=torch.nn.ParameterDict({
    'means':torch.nn.Parameter(torch.tensor(xyz,dtype=torch.float32,device='cuda')),
    'scales':torch.nn.Parameter(torch.tensor(np.log(sizes[:,None].repeat(3,1)),dtype=torch.float32,device='cuda')),
    'quats':torch.nn.Parameter(torch.tensor(np.tile([1.,0,0,0],(n,1)),dtype=torch.float32,device='cuda')),
    'opacities':torch.nn.Parameter(torch.full((n,),-2.1972246,device='cuda')),
    'rgb':torch.nn.Parameter(torch.logit(torch.tensor(colors,dtype=torch.float32,device='cuda').clamp(.01,.99))),
})
lrs={'means':.00016,'scales':.005,'quats':.001,'opacities':.05,'rgb':.025}
optimizers={k:torch.optim.Adam([v],lr=lrs[k],eps=1e-15) for k,v in params.items()}
strategy=DefaultStrategy(refine_start_iter=100,refine_stop_iter=2200,refine_every=50,reset_every=3000,
                         prune_scale3d=.5,grow_grad2d=.0002)
strategy.check_sanity(params,optimizers)
state=strategy.initialize_state(scene_scale=1.)
frames=[]
# Map SIMPLE_RADIAL images to pinhole views before training, using the solved radial coefficient.
import cv2
for im in images:
    cam=rec.cameras[im.camera_id]
    K=cam.calibration_matrix()
    raw=np.array(Image.open(root/'input'/im.name).convert('RGB'))
    k=cam.params[3] if cam.model==pycolmap.CameraModelId.SIMPLE_RADIAL else 0.
    raw=cv2.undistort(raw,K,np.array([k,0,0,0.]))
    h,w=raw.shape[:2]
    target=cv2.resize(raw,(640,480),interpolation=cv2.INTER_AREA)
    K=K.copy()
    K[0,:]*=640/w
    K[1,:]*=480/h
    transform=np.eye(4)
    transform[:3,:]=im.cam_from_world().matrix()
    transform[:3,3]=(transform[:3,:3]@center+transform[:3,3])/scale
    frames.append({'name':im.name,'target':torch.tensor(target/255,dtype=torch.float32,device='cuda'),
                   'K':torch.tensor(K,dtype=torch.float32,device='cuda')[None],
                   'view':torch.tensor(transform,dtype=torch.float32,device='cuda')[None]})

def render(frame):
    return rasterization(means=params['means'],quats=params['quats'],scales=params['scales'].exp(),
        opacities=params['opacities'].sigmoid(),colors=params['rgb'].sigmoid(),viewmats=frame['view'],Ks=frame['K'],
        width=640,height=480,sh_degree=None,packed=False,near_plane=.01,far_plane=100.)

def ssim_loss(a,b):
    a=a.permute(2,0,1)[None]
    b=b.permute(2,0,1)[None]
    avg=lambda v:F.avg_pool2d(v,7,stride=1,padding=3)
    ma,mb=avg(a),avg(b)
    va,vb=avg(a*a)-ma*ma,avg(b*b)-mb*mb
    cov=avg(a*b)-ma*mb
    return 1-(((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean()

start=time.time()
losses=[]
for step in range(3000):
    frame=frames[np.random.randint(len(frames))]
    rgb,alpha,info=render(frame)
    strategy.step_pre_backward(params,optimizers,state,step,info)
    loss=.8*F.l1_loss(rgb[0],frame['target'])+.2*ssim_loss(rgb[0],frame['target'])
    if not torch.isfinite(loss):
        raise RuntimeError(f'Nonfinite loss at step {step}')
    loss.backward()
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    strategy.step_post_backward(params,optimizers,state,step,info)
    if len(params['means'])>40000:
        strategy.refine_stop_iter=step
    if step%250==0 or step==2999:
        record={'step':step+1,'loss':loss.item(),'gaussians':len(params['means']),'elapsed_s':time.time()-start}
        losses.append(record)
        print(json.dumps(record),flush=True)
        torch.save({k:v.detach().cpu() for k,v in params.items()},out/'checkpoint.pt')

with torch.no_grad():
    data=GaussianData(params['means'].cpu().numpy(),F.normalize(params['quats'],dim=-1).cpu().numpy(),
        params['scales'].exp().cpu().numpy(),params['opacities'].sigmoid().cpu().numpy(),
        ((params['rgb'].sigmoid()-.5)/.28209479177387814).cpu().numpy()[:,None,:])
    save_ply(data,out/'lab_partial.ply')
    canvas=Image.new('RGB',(1280,len(frames)*504),'#17202e')
    draw=ImageDraw.Draw(canvas)
    for i,frame in enumerate(frames):
        rgb,_,_=render(frame)
        pred=Image.fromarray((rgb[0].clamp(0,1).cpu().numpy()*255).astype('uint8'))
        target=Image.fromarray((frame['target'].cpu().numpy()*255).astype('uint8'))
        pred.save(out/f'training_view_{i+1}.png')
        canvas.paste(target,(0,i*504+24))
        canvas.paste(pred,(640,i*504+24))
        draw.text((8,i*504+5),'SOURCE | '+frame['name'][:10],fill='white')
        draw.text((648,i*504+5),'3DGS DRAFT | training view (not held out)',fill='white')
    canvas.save(out/'partial_training_comparison.png')
report={'status':'partial_draft_only','registered_images':len(frames),'input_images':9,
        'initial_sparse_points':n,'gaussians':len(params['means']),'iterations':3000,
        'elapsed_s':time.time()-start,'normalized_center':center.tolist(),'normalized_scale':float(scale),
        'metric_scale_calibrated':False,'white_partition_registered':False,
        'limitations':['White partition views are disconnected. Requested full region is not reconstructed.',
                      'Only training-view previews; no held-out validation.',
                      'People and robot are not semantically removed. No collision mesh.'],
        'losses':losses}
(out/'report.json').write_text(json.dumps(report,indent=2))
print('TRAIN_RESULT '+json.dumps(report),flush=True)
