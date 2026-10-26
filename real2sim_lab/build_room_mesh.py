"""Extract a triangle mesh from the exact GS scene shown in the supplied image."""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'gs_isaac/prebuilt'),str(ROOT/'real2sim_lab/deps'),str(ROOT/'gs_isaac/deps'),str(ROOT/'third_party/GaussianRenderer/src')]
os.environ['TORCH_COMPILE_DISABLE']='1'
import numpy as np
import torch
import open3d as o3d
from PIL import Image,ImageDraw
from gaussian_renderer import GSRenderer
from gsplat.rendering import rasterization

OUT=ROOT/'real2sim_lab/room_mesh'; OUT.mkdir(exist_ok=True)
scene=ROOT/'third_party/gs_playground/demo/navigation/nav_scene_1/3dgs/point_cloud.ply'
start=time.time(); renderer=GSRenderer({'background':str(scene)}); g=renderer.gaussians
g.sh=g.sh.reshape(len(g),-1,3).contiguous()
W,H=640,480; f=H/(2*np.tan(np.deg2rad(70)/2))
K=np.array([[f,0,W/2],[0,f,H/2],[0,0,1]],np.float32)
intr=o3d.camera.PinholeCameraIntrinsic(W,H,f,f,W/2,H/2)
volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=.025,sdf_trunc=.10,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
def camera(pos,direction):
    forward=np.array(direction,dtype=float); forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1.]); right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    c2w=np.eye(4); c2w[:3,:3]=np.column_stack([right,down,forward]);c2w[:3,3]=pos
    return c2w
poses=[]
for z in [.85,1.45,1.95]:
    for pitch in [-35,0,35]:
        for yaw in np.arange(0,360,30):
            a,p=np.deg2rad([yaw,pitch]); poses.append(camera([0,0,z],[np.cos(a)*np.cos(p),np.sin(a)*np.cos(p),np.sin(p)]))
with torch.inference_mode():
    for i,c2w in enumerate(poses):
        renders,alpha,_=rasterization(means=g.xyz,quats=g.rot,scales=g.scale,opacities=g.opacity,colors=g.sh,viewmats=torch.tensor(np.linalg.inv(c2w)[None],device='cuda',dtype=torch.float32),Ks=torch.tensor(K[None],device='cuda'),width=W,height=H,sh_degree=int(np.sqrt(g.sh.shape[1]))-1,render_mode='RGB+ED',packed=False)
        rgb=(renders[0,:,:,:3].clamp(0,1).cpu().numpy()*255).astype('uint8')
        depth=renders[0,:,:,3].cpu().numpy().astype('float32'); a=alpha[0,:,:,0].cpu().numpy()
        depth[(a<.95)|(~np.isfinite(depth))|(depth<.15)|(depth>10)]=0
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb),o3d.geometry.Image(depth),depth_scale=1.,depth_trunc=10.,convert_rgb_to_intensity=False)
        volume.integrate(rgbd,intr,np.linalg.inv(c2w))
        if i%12==0: print('fused',i+1,'/',len(poses),flush=True)
mesh=volume.extract_triangle_mesh();mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
labels,counts,_=mesh.cluster_connected_triangles();labels=np.asarray(labels);counts=np.asarray(counts)
mesh.remove_triangles_by_mask(counts[labels]<100);mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals()
assert len(mesh.triangles)>1000
o3d.io.write_triangle_mesh(str(OUT/'room_mesh.ply'),mesh)
o3d.io.write_triangle_mesh(str(OUT/'room_mesh.obj'),mesh,write_vertex_colors=True)
print('mesh',len(mesh.vertices),len(mesh.triangles),flush=True)
# Render actual mesh triangles, without Gaussian rendering, at original four cameras.
ray_scene=o3d.t.geometry.RaycastingScene();ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
verts=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles);colors=np.asarray(mesh.vertex_colors)
sheet=Image.new('RGB',(1280,1008),'#142033'); draw=ImageDraw.Draw(sheet)
directions=[[-2,0,-.25],[0,2,-.25],[2,0,-.25],[0,-2,-.25]]
for i,d in enumerate(directions):
    pose=camera([0,0,1.45],d)
    rays=o3d.t.geometry.RaycastingScene.create_rays_pinhole(K.astype('float64'),np.linalg.inv(pose),W,H)
    result=ray_scene.cast_rays(rays);ids=result['primitive_ids'].numpy();uv=result['primitive_uvs'].numpy();valid=ids<len(faces)
    pixels=np.full((H,W,3),.08,np.float32);fids=faces[ids[valid]];b=uv[valid];weights=np.stack([1-b.sum(1),b[:,0],b[:,1]],1)
    col=(colors[fids]*weights[:,:,None]).sum(1)
    normals=result['primitive_normals'].numpy()[valid];light=.6+.4*np.abs(normals@np.array([.3,-.5,.81]))
    pixels[valid]=col*light[:,None]
    img=Image.fromarray((np.clip(pixels,0,1)*255).astype('uint8'));img.save(OUT/f'mesh_view_{i+1}.png')
    x,y=i%2*640,i//2*504;sheet.paste(img,(x,y+24));draw.text((x+10,y+5),f'Extracted triangle mesh | view {i+1}',fill='white')
sheet.save(OUT/'mesh_preview.png')
# Explicit static triangle collision surface, in unchanged source coordinates.
import runpy
runpy.run_path(str(ROOT/'real2sim_lab/finalize_room_mesh.py'),run_name='__main__')
report={'source':str(scene.relative_to(ROOT)),'method':'3DGS expected-depth + alpha >= .95 + TSDF fusion','views':len(poses),'voxel_source_units':.025,'vertices':len(verts),'triangles':len(faces),'bounds':np.stack([verts.min(0),verts.max(0)]).tolist(),'elapsed_seconds':time.time()-start,'watertight':mesh.is_watertight(),'self_intersecting':mesh.is_self_intersecting(),'notes':['Geometry extracted from original GS, not inferred from screenshot.','No SAM3D object generation in this result.','Source scale retained; real-world dimensions unverified.','Static scene surface; individual objects not separated, missing backsides possible.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
