"""Export view-specific RGB-D point clouds from actual MoGe camera point maps."""
from pathlib import Path
import json, struct, zlib, base64
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT/'mira_scene/results/real_lab_mira'
OUT = BASE/'rgbd_pointcloud'
OUT.mkdir(parents=True, exist_ok=True)

def load_mask(path, shape):
    return np.asarray(Image.open(path).convert('L').resize((shape[1],shape[0]),Image.Resampling.NEAREST)) > 127

records=[]
preview_data=[]
for case, label in [('purple_table','视频帧：桌子和紫色挡板'),('white_divider','照片：白色挡板')]:
    cdir=BASE/'cases'/case
    folder=cdir/'input'
    rgb=np.asarray(Image.open(folder/'scene.png').convert('RGB'))
    pts=np.load(cdir/'depth/moge/camera_pts_map.npy').astype(np.float32)
    valid=np.load(cdir/'depth/moge/valid_mask.npy').astype(bool)
    masks=[]
    for mask_path in sorted(folder.glob('mask_*.png')):
        masks.append(load_mask(mask_path,rgb.shape[:2]))
    # The earlier coarse text prompt split the accordion screen into panels;
    # re-union those genuine SAM3 masks for a single coherent instance.
    archive=folder/'panel_masks_before_merge'
    if archive.is_dir():
        masks.extend(load_mask(p,rgb.shape[:2]) for p in archive.glob('mask_*.png'))
    if case=='purple_table':
        # Interactive SAM3 box-and-point prompt, visually reviewed candidate 2.
        masks.append(load_mask(BASE/'purple_mask_review/candidate_2.png',rgb.shape[:2]))
    objmask=np.logical_or.reduce(masks) if masks else np.ones(rgb.shape[:2],bool)
    # White-divider depth preview covers the whole image; show its whole valid cloud.
    keep=(np.ones_like(objmask) if case=='white_divider' else objmask) & valid & np.isfinite(pts).all(axis=-1)
    h,w=keep.shape
    # Uniformly thin dense pixels to a practical interactive/rendered cloud.
    stride=max(1,int(np.ceil(max(h,w)/800)))
    yy,xx=np.mgrid[0:h:stride,0:w:stride]
    sel=keep[::stride,::stride]
    xyz=pts[::stride,::stride][sel]
    col=rgb[::stride,::stride][sel]
    center=np.median(xyz,axis=0)
    scale=max(float(np.max(np.abs(xyz-center))),1e-6)
    norm=np.clip((xyz-center)/scale,-1,1)
    # Native binary PLY, camera coordinates remain in MoGe's relative scale.
    ply=OUT/f'{case}.ply'
    with ply.open('wb') as f:
        header=("ply\nformat binary_little_endian 1.0\n"+f"element vertex {len(xyz)}\n"+
                "property float x\nproperty float y\nproperty float z\n"+
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        f.write(header.encode('ascii'))
        for p,c in zip(xyz,col): f.write(struct.pack('<fffBBB',*map(float,p),*map(int,c)))
    # Compact viewer payload: uniform sample, uint16 xyz + uint8 RGB.
    rng=np.random.default_rng(42)
    chosen=rng.choice(len(norm),min(16000,len(norm)),replace=False)
    view_norm,view_col=norm[chosen],col[chosen]
    packed=np.empty((len(view_norm),9),dtype=np.uint8)
    packed[:,:6]=np.rint((view_norm+1)*32767.5).astype('<u2').view(np.uint8).reshape(-1,6)
    packed[:,6:]=view_col
    payload=base64.b64encode(zlib.compress(packed.tobytes(),6)).decode('ascii')
    records.append({'name':case,'label':label,'count':len(view_norm),'full_count':len(xyz),'ply':ply.name,'payload':payload,
                    'center':center.tolist(),'scale':scale,'source':str(cdir/'input/scene.png')})
    preview_data.append((label,xyz,col))

# Standalone WebGL viewer with no external script or CDN dependency. The two
# camera views are switchable but are never shown as a registered shared frame.
data=json.dumps(records,separators=(',',':'))
html='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实照片 RGB-D 彩色点云</title>
<style>body{margin:0;background:#141820;color:#e8edf5;font:15px system-ui}header{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:12px 16px}h1{font-size:18px;font-weight:500;margin:0}select,button{font:inherit;color:inherit;background:#252d39;border:1px solid #424d5c;border-radius:6px;padding:7px 10px}#view{height:min(72vh,680px);min-height:380px}canvas{width:100%;height:100%;display:block}#note{padding:0 16px;color:#b6c0ce;font-size:13px}</style>
<header><h1>真实照片 RGB + MoGe 深度：3D 彩色点云</h1><label>视角 <select id="case"></select></label><button id="reset">重置视角</button><span id="count"></span></header>
<div id="view" aria-label="可旋转、平移和缩放的彩色点云"><canvas id="canvas"></canvas></div><p id="note">拖动旋转 · 右键平移 · 滚轮缩放。显示的是各自照片坐标下的相对尺度点云，多个视角之间尚未配准，深度未标定为米。</p>
<script>
const records=DATA, menu=document.getElementById('case'), canvas=document.getElementById('canvas');
for(const r of records){const o=document.createElement('option');o.value=r.name;o.textContent=r.label;menu.append(o)}
const gl=canvas.getContext('webgl',{antialias:true});let program,posBuf,colBuf,count=0,yaw=.38,pitch=.28,radius=2.8,target=[0,0,0],drag=null;
const status=document.getElementById('count');
if(!gl){status.textContent='此浏览器未启用 WebGL';throw new Error('WebGL unavailable')}
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s}
const vs=shader(gl.VERTEX_SHADER,`attribute vec3 p;attribute vec3 c;uniform mat4 mvp;varying vec3 color;void main(){gl_Position=mvp*vec4(p,1.0);gl_PointSize=4.0;color=c;}`);
const fs=shader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 color;void main(){vec2 q=gl_PointCoord-vec2(.5);if(dot(q,q)>.25)discard;gl_FragColor=vec4(color,1.0);}`);
program=gl.createProgram();gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program));
posBuf=gl.createBuffer();colBuf=gl.createBuffer();gl.useProgram(program);gl.enable(gl.DEPTH_TEST);gl.clearColor(.078,.094,.125,1);
function decode64(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a}
async function show(){try{status.textContent='正在载入点云…';const r=records.find(x=>x.name===menu.value),z=decode64(r.payload);const ds=new DecompressionStream('deflate');const raw=new Uint8Array(await new Response(new Blob([z]).stream().pipeThrough(ds)).arrayBuffer());const n=r.count,xyz=new Float32Array(n*3),rgb=new Uint8Array(n*3),dv=new DataView(raw.buffer,raw.byteOffset,raw.byteLength);for(let i=0;i<n;i++){xyz[i*3]=dv.getUint16(i*9,true)/65535*2-1;xyz[i*3+1]=dv.getUint16(i*9+2,true)/65535*2-1;xyz[i*3+2]=dv.getUint16(i*9+4,true)/65535*2-1;rgb[i*3]=raw[i*9+6];rgb[i*3+1]=raw[i*9+7];rgb[i*3+2]=raw[i*9+8]}gl.bindBuffer(gl.ARRAY_BUFFER,posBuf);gl.bufferData(gl.ARRAY_BUFFER,xyz,gl.STATIC_DRAW);gl.bindBuffer(gl.ARRAY_BUFFER,colBuf);gl.bufferData(gl.ARRAY_BUFFER,rgb,gl.STATIC_DRAW);count=n;status.textContent=n.toLocaleString()+' 点 · '+r.label;resetView()}catch(e){status.textContent='点云载入失败：'+e.message;console.error(e)}}
function resetView(){yaw=.38;pitch=.28;radius=2.8;target=[0,0,0]}
function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)o[c*4+r]=a[r]*b[c*4]+a[4+r]*b[c*4+1]+a[8+r]*b[c*4+2]+a[12+r]*b[c*4+3];return o}
function norm(v){const l=Math.hypot(...v)||1;return v.map(x=>x/l)}function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
function matrices(aspect){const f=1/Math.tan(55*Math.PI/360),near=.01,far=100,proj=new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);const cp=Math.cos(pitch),eye=[target[0]+radius*Math.sin(yaw)*cp,target[1]+radius*Math.sin(pitch),target[2]+radius*Math.cos(yaw)*cp],z=norm([eye[0]-target[0],eye[1]-target[1],eye[2]-target[2]]),x=norm(cross([0,1,0],z)),y=cross(z,x),view=new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);return mul(proj,view)}
function draw(){const w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(devicePixelRatio||1,2);if(canvas.width!==Math.floor(w*dpr)||canvas.height!==Math.floor(h*dpr)){canvas.width=Math.floor(w*dpr);canvas.height=Math.floor(h*dpr)}gl.viewport(0,0,canvas.width,canvas.height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);if(count){gl.useProgram(program);gl.bindBuffer(gl.ARRAY_BUFFER,posBuf);let a=gl.getAttribLocation(program,'p');gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,colBuf);a=gl.getAttribLocation(program,'c');gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,3,gl.UNSIGNED_BYTE,true,0,0);gl.uniformMatrix4fv(gl.getUniformLocation(program,'mvp'),false,matrices(canvas.width/canvas.height));gl.drawArrays(gl.POINTS,0,count)}requestAnimationFrame(draw)}
canvas.addEventListener('contextmenu',e=>e.preventDefault());canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,button:e.button};canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;if(drag.button===2){target[0]-=dx*.003*radius;target[1]+=dy*.003*radius}else{yaw-=dx*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*.008))}});canvas.addEventListener('wheel',e=>{radius=Math.max(1.2,Math.min(9,radius*Math.exp(e.deltaY*.001)));e.preventDefault()},{passive:false});menu.onchange=show;document.getElementById('reset').onclick=resetView;show();draw();
</script>'''.replace('DATA',data)
(OUT/'index.html').write_text(html,encoding='utf-8')
(OUT/'manifest.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='payload'} for r in records],indent=2,ensure_ascii=False),encoding='utf-8')
fig=plt.figure(figsize=(12,5.6),dpi=150)
for i,(label,xyz,col) in enumerate(preview_data,1):
    ax=fig.add_subplot(1,2,i,projection='3d')
    rng=np.random.default_rng(123+i); ids=rng.choice(len(xyz),min(10000,len(xyz)),replace=False)
    q=xyz[ids]-np.median(xyz,axis=0)
    ax.scatter(q[:,0],-q[:,2],q[:,1],s=.28,c=col[ids]/255.,depthshade=False)
    ax.set_title(('Purple table + divider' if i == 1 else 'White folding divider'))
    ax.set_xlabel('camera X (relative)'); ax.set_ylabel('depth (relative)'); ax.set_zlabel('up Y (relative)')
    ax.view_init(elev=21,azim=-65); ax.set_box_aspect((1,1,0.65))
fig.tight_layout(); fig.savefig(OUT/'pointcloud_preview.png',bbox_inches='tight'); plt.close(fig)
print(json.dumps([{k:v for k,v in r.items() if k!='payload'} for r in records],indent=2,ensure_ascii=False))
