"""Photo-referenced, uncalibrated lab proxy. No generative 3D reconstruction or measured geometry.
Rebuild: python source/build_scene.py   (numpy, scipy, pillow, opencv-python, trimesh)
The environment is a partial observed corner, NOT an inferred closed room.
All dimensions below are modelling assumptions in metres, based on table height = 0.75 m.
"""
from pathlib import Path
import json, math, base64, struct, shutil, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import cv2
import trimesh
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parents[1]
TEX=ROOT/'textures'; TEX.mkdir(exist_ok=True)
REF=ROOT/'references'; REF.mkdir(exist_ok=True)
rng=np.random.default_rng(79)
PHOTOS=[('2977152208755455d27cbf9ab8d2ca20.jpg','reference_01.jpg'),('19b809ba81c1d19208135c437f6621be.jpg','reference_02.jpg'),('a8a69213cf59c409cb2a51c53b447cd2.jpg','reference_03.jpg'),('eb3db14e3c9723ea27fd55691b6edad6.jpg','reference_04.jpg')]
photos=[]
for fn,out in PHOTOS:
 p=ROOT.parent/fn
 if not p.exists(): p=REF/out
 im=Image.open(p).convert('RGB').resize((1408,1056))
 im.save(REF/out,quality=88)
 photos.append(im)

def saveim(name,im):
 im.convert('RGB').save(TEX/(name+'.jpg'),quality=91)
 return name+'.jpg'
def crop_quad(name,img,pts,size):
 a=np.asarray(img)
 H=cv2.getPerspectiveTransform(np.float32(pts),np.float32([[0,0],[size[0]-1,0],[size[0]-1,size[1]-1],[0,size[1]-1]]))
 return saveim(name,Image.fromarray(cv2.warpPerspective(a,H,size)))
# Rectified image regions retain actual signs / brands, rather than inventing their text.
crop_quad('bag_front',photos[0],[(251,507),(422,491),(434,637),(255,672)],(512,512))
crop_quad('lost_found',photos[0],[(622,263),(689,265),(689,320),(624,318)],(512,416))
poster_quads=[[(5,10),(87,16),(85,153),(7,142)],[(274,8),(321,9),(325,170),(275,166)],[(641,3),(724,2),(724,185),(640,179)],[(757,7),(861,8),(863,181),(756,181)],[(898,9),(975,12),(975,186),(898,189)],[(1038,14),(1136,16),(1134,193),(1037,194)]]
for i,q in enumerate(poster_quads): crop_quad('poster_%02d'%i,photos[3],q,(384,560))
# Photo-sampled, de-lit cloth pattern, blended with a tiny textile weave.
a=np.asarray(photos[3].crop((393,607,573,737)).resize((512,512)),dtype=np.float32)
g=a.mean(2); blur=cv2.GaussianBlur(g,(0,0),24)
g=(g-blur)*.24+75
Y,X=np.mgrid[:512,:512]; weave=1.3*np.sin(X*np.pi)+1.2*np.sin(Y*np.pi*.95)
c=np.stack([g+weave,g+weave+2,g+weave+3],-1)
saveim('cloth',Image.fromarray(np.uint8(np.clip(c,0,255))))
# Wood-laminate material synthesized from a real floor colour patch; not a measured texture scan.
N=1024; Y,X=np.mgrid[:N,:N]
base=np.zeros((N,N,3),np.float32)
for row in range(8):
 y0=row*128
 tint=np.array([133,109,86])+rng.uniform(-7,7)
 coarse=cv2.resize(rng.normal(0,1,(32,512)).astype(np.float32),(N,128),interpolation=cv2.INTER_CUBIC)
 waves=np.sin(np.arange(128)[:,None]*.42+np.sin(np.arange(N)[None,:]*.009+row)*2)*2.3
 grain=coarse*4+waves+rng.normal(0,.8,(128,N))
 base[y0:y0+128]=tint+grain[:,:,None]
 base[y0:y0+2]*=.69
 for col in range(0,N,384):
  c=(col+row*141)%N; base[y0:y0+128,c:c+2]*=.80
saveim('wood_floor',Image.fromarray(np.uint8(np.clip(base,0,255))))
# Other low-frequency procedural materials provide editable, non-view-dependent sides.
noise=rng.normal(0,2.0,(256,256,1))
saveim('paper',Image.fromarray(np.uint8(np.clip(np.array([171,134,82])+noise,0,255))))
canvas=Image.new('RGB',(512,200),(234,229,213));d=ImageDraw.Draw(canvas)
def font_at(size,bold=False):
 for candidate in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf','/System/Library/Fonts/Helvetica.ttc']:
  try:return ImageFont.truetype(candidate,size)
  except OSError:pass
 return ImageFont.load_default(size=size)
font=font_at(62,True)
d.text((94,46),'NUS',fill=(32,39,57),font=font)
d.rectangle((22,50,72,134),outline=(32,39,57),width=4)
d.line((22,93,72,93),fill=(182,135,48),width=6)
saveim('tote_label',canvas)
canvas=Image.new('RGB',(400,160),(26,112,118));d=ImageDraw.Draw(canvas)
f=font_at(113)
d.text((62,4),'EXIT',fill='white',font=f)
saveim('exit',canvas)
# Finely grained dark monitor glass, with a very faint neutral reflection.
Y,X=np.mgrid[:384,:640]; g=18+5*np.exp(-((X-190)**2/18000+(Y-80)**2/4800))
saveim('screen',Image.fromarray(np.uint8(np.stack([g,g+3,g+4],-1))))
# Hazard tape texture (only the strips visible in the photographs are modelled).
Y,X=np.mgrid[:64,:512]; mask=((X+Y)//48)%2
im=np.zeros((64,512,3),np.uint8);im[:]=[225,182,44];im[mask==0]=[29,30,29]
saveim('hazard',Image.fromarray(im))

materials=[]; matindex={}
def mat(name,color,rough=.8,metal=0,texture=None):
 i=len(materials);matindex[name]=i
 materials.append({'name':name,'color':color,'roughness':rough,'metallic':metal,'texture':texture})
 return i
mat('Paint warm white',[.84,.83,.79],.93)
mat('Blind ivory',[.90,.885,.835],.9)
mat('Ceiling',[.79,.79,.76],.9)
mat('Dark trim',[.11,.125,.13],.65)
mat('Glass board',[.16,.185,.19],.5)
mat('Silver',[.51,.55,.56],.35,.7)
mat('Tablecloth',[1,1,1],.97,texture='cloth.jpg')
mat('Partition',[.57,.545,.563],.96)
mat('Black rubber',[.028,.033,.036],.87)
mat('Wood',[1,1,1],.83,texture='wood_floor.jpg')
mat('Paper',[1,1,1],.95,texture='paper.jpg')
mat('Bag front',[1,1,1],.95,texture='bag_front.jpg')
mat('White canvas',[.78,.765,.707],.97)
mat('Tote label',[1,1,1],.98,texture='tote_label.jpg')
mat('Red canvas',[.68,.048,.055],.98)
mat('Blue plastic',[.055,.205,.55],.52)
mat('Grey plastic',[.21,.26,.30],.6)
mat('White plastic',[.90,.883,.82],.69)
mat('Black plastic',[.035,.042,.047],.55)
mat('Pink tape',[.60,.425,.414],.9)
mat('Hazard tape',[1,1,1],.9,texture='hazard.jpg')
mat('Screen glass',[1,1,1],.18,.15,texture='screen.jpg')
mat('Lost and Found sign',[1,1,1],.86,texture='lost_found.jpg')
mat('Exit sign',[1,1,1],.62,texture='exit.jpg')
mat('Cardboard',[.50,.355,.195],.95)
mat('Tan tape',[.72,.645,.49],.98)
mat('Brown part',[.18,.108,.061],.65)
mat('Indicator red',[.9,.03,.015],.4)
for i in range(6):mat('Poster %02d'%i,[1,1,1],.93,texture='poster_%02d.jpg'%i)
mat('Lamp diffuser',[.98,.974,.91],.77)
mat('Tube wood',[.48,.365,.225],.94)

objects=[]; meshes=[]; lookup={}
def obj(id,label,category='props',draggable=True,note='',parent=None):
 o={'id':id,'label':label,'category':category,'draggable':draggable,'note':note,'parent':parent,'estimated':True}
 objects.append(o);lookup[id]=o;return id

def mesh(gid,name,verts,faces,material,uv=None,collider=False,smooth=True):
 v=np.asarray(verts,dtype=np.float32);f=np.asarray(faces,dtype=np.int32)
 tm=trimesh.Trimesh(vertices=v,faces=f,process=False)
 if uv is None:
  uv=np.column_stack((v[:,0],v[:,1]))
 uv=np.asarray(uv,dtype=np.float32)
 if not smooth:
  v=v[f.reshape(-1)];uv=uv[f.reshape(-1)]
  f=np.arange(len(v),dtype=np.int32).reshape(-1,3)
  tm=trimesh.Trimesh(vertices=v,faces=f,process=False)
 meshes.append({'object':gid,'name':name,'v':v,'f':f,'n':np.asarray(tm.vertex_normals,dtype=np.float32),'uv':uv,'mat':matindex[material] if isinstance(material,str) else material,'collider':bool(collider)})

def box(gid,name,c,s,material,collider=False,angle=0):
 c=np.array(c,float);sx,sy,sz=np.array(s,float)/2
 verts=[];faces=[];uv=[]
 quads=[([[-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]]),([[sx,-sy,-sz],[-sx,-sy,-sz],[-sx,sy,-sz],[sx,sy,-sz]]),([[sx,-sy,sz],[sx,-sy,-sz],[sx,sy,-sz],[sx,sy,sz]]),([[-sx,-sy,-sz],[-sx,-sy,sz],[-sx,sy,sz],[-sx,sy,-sz]]),([[-sx,sy,sz],[sx,sy,sz],[sx,sy,-sz],[-sx,sy,-sz]]),([[-sx,-sy,-sz],[sx,-sy,-sz],[sx,-sy,sz],[-sx,-sy,sz]])]
 for q in quads:
  k=len(verts);verts.extend(q);uv.extend([[0,0],[1,0],[1,1],[0,1]]);faces.extend([[k,k+1,k+2],[k,k+2,k+3]])
 v=np.array(verts);R=np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
 mesh(gid,name,v@R.T+c,faces,material,uv,collider,smooth=False)

def quad(gid,name,pts,material,uv=None,collider=False):
 mesh(gid,name,pts,[[0,1,2],[0,2,3]],material,uv if uv is not None else [[0,0],[1,0],[1,1],[0,1]],collider)

def cylinder(gid,name,a,b,r,material,collider=False,sections=16,r2=None):
 a=np.array(a,float);b=np.array(b,float);direction=b-a;h=np.linalg.norm(direction)
 axis=direction/h;u=np.cross(axis,[0,0,1])
 if np.linalg.norm(u)<.01:u=np.cross(axis,[1,0,0])
 u=u/np.linalg.norm(u);v=np.cross(axis,u)
 vs=[];uv=[];fs=[]
 for j,(p,rad) in enumerate([(a,r),(b,r if r2 is None else r2)]):
  for i in range(sections):
   t=2*np.pi*i/sections;vs.append(p+rad*(u*np.cos(t)+v*np.sin(t)));uv.append([i/sections,j])
 for i in range(sections):
  k=(i+1)%sections;fs.extend([[i,k,k+sections],[i,k+sections,i+sections]])
 vs.extend([a,b]);uv.extend([[.5,.5],[.5,.5]])
 for i in range(sections):
  k=(i+1)%sections;fs.extend([[2*sections,k,i],[2*sections+1,i+sections,k+sections]])
 mesh(gid,name,vs,fs,material,uv,collider)

def tube(gid,name,points,r,material,steps=60,sides=8,collider=False):
 pts=np.array(points,float);t=np.linspace(0,1,len(pts));ts=np.linspace(0,1,steps)
 curve=CubicSpline(t,pts,axis=0)(ts);tang=np.gradient(curve,axis=0);vs=[];uv=[];fs=[]
 for j,(p,di) in enumerate(zip(curve,tang)):
  di=di/np.linalg.norm(di);u=np.cross(di,[0,1,0])
  if np.linalg.norm(u)<.01:u=np.cross(di,[1,0,0])
  u/=np.linalg.norm(u);v=np.cross(di,u)
  for i in range(sides):
   th=2*np.pi*i/sides;vs.append(p+r*(u*np.cos(th)+v*np.sin(th)));uv.append([i/sides,j/(steps-1)])
 for j in range(steps-1):
  for i in range(sides):
   a=j*sides+i;b=j*sides+(i+1)%sides;c=b+sides;d=a+sides
   fs.extend([[a,b,c],[a,c,d]])
 mesh(gid,name,vs,fs,material,uv,collider)

def ring(gid,name,c,rad,minor,material,segments=32):
 p=[]
 for i in range(segments+1):
  a=2*np.pi*i/segments;p.append([c[0]+rad*np.cos(a),c[1],c[2]+rad*np.sin(a)])
 tube(gid,name,p,minor,material,segments*2,10,False)

def drape(gid,c,w,d,h,drop,seed):
 # Upper cloth surface is a real mesh; sides are drapes, not solid collision boxes.
 nx,nz=48,36;vs=[];uv=[];fs=[]
 for j in range(nz+1):
  for i in range(nx+1):
   x=-w/2+w*i/nx;z=-d/2+d*j/nz
   ripple=.0018*np.sin(x*31+z*11+seed)+.002*np.sin(z*41+x*4)
   vs.append([c[0]+x,h+.008+ripple,c[1]+z]);uv.append([x*1.5,z*1.5])
 for j in range(nz):
  for i in range(nx):
   a=j*(nx+1)+i;b=a+1;dd=a+nx+1;cc=dd+1;fs.extend([[a,dd,b],[b,dd,cc]])
 mesh(gid,'cloth_top',vs,fs,'Tablecloth',uv)
 for side in range(4):
  N=64;M=18;vs=[];uv=[];fs=[]
  length=w if side<2 else d
  for j in range(M+1):
   t=j/M
   for i in range(N+1):
    along=-length/2+length*i/N
    wav=(.003+.018*t)*np.sin(along*23+seed)+(t**2)*.022*np.sin(along*9+seed)
    hem=.035*np.sin(along*5+seed)*t**5
    out=.012+.027*t+wav
    if side<2: x=along;z=(-1 if side==0 else 1)*(d/2+out)
    else:x=(-1 if side==2 else 1)*(w/2+out);z=along
    vs.append([c[0]+x,h+.01-drop*t+hem,c[1]+z]);uv.append([along*1.8,t*1.1])
  for j in range(M):
   for i in range(N):
    a=j*(N+1)+i;b=a+1;cc=a+N+2;dd=a+N+1
    fs.extend([[a,b,cc],[a,cc,dd]])
  mesh(gid,'cloth_drape_%d'%side,vs,fs,'Tablecloth',uv)

def table(id,label,c,w,d):
 gid=obj(id,label,'furniture',True,'桌高暂定 0.75 m；尺寸未实测；桌布为静态视觉网格。')
 box(gid,'tabletop',[c[0],.719,c[1]],[w,.062,d],'Dark trim',True)
 for sx in [-1,1]:
  for sz in [-1,1]:
   box(gid,'leg',[c[0]+sx*(w/2-.09),.354,c[1]+sz*(d/2-.12)],[.05,.708,.05],'Silver',True)
   box(gid,'foot',[c[0]+sx*(w/2-.09),.014,c[1]+sz*(d/2-.12)],[.075,.028,.08],'Black rubber',True)
 drape(gid,c,w+.02,d+.02,.75,.57,4 if id=='front_table' else 8)
 return gid

# Visible room corner. Open edges indicate missing capture coverage, not new doors/windows.
gid=obj('floor','木纹地板 · 可见区域','architecture',False,'仅为照片覆盖区的裁切边界；不是实测房间轮廓。')
box(gid,'floor_slab',[0,-.065,.20],[6.50,.13,6.00],'Dark trim',True)
quad(gid,'laminate_surface',[[-3.25,.003,-2.8],[-3.25,.003,3.2],[3.25,.003,3.2],[3.25,.003,-2.8]],'Wood',[[0,0],[0,2.8],[3.0,2.8],[3.0,0]])
arch=obj('walls','墙体与出口门','architecture',False,'依照片建立后墙、右墙。其余边界未知，未闭合成完整房间。')
# Actual door location represented in the observed back wall; no new opening is invented.
box(arch,'back_wall_left',[-.675,1.48,-2.85],[5.15,2.96,.10],'Paint warm white',True)
box(arch,'back_wall_right',[3.15,1.48,-2.85],[.2,2.96,.10],'Paint warm white',True)
box(arch,'door_lintel',[2.475,2.66,-2.85],[1.15,.60,.10],'Paint warm white',True)
box(arch,'closed_exit_door',[2.475,1.15,-2.815],[1.13,2.30,.055],'Ceiling',True)
for x in [1.897,3.053]:box(arch,'door_jamb',[x,1.18,-2.77],[.035,2.36,.04],'Silver')
box(arch,'door_frame_header',[2.475,2.355,-2.77],[1.19,.035,.04],'Silver')
box(arch,'door_panic_bar',[2.475,1.065,-2.751],[.83,.025,.035],'Silver')
box(arch,'door_closer',[2.64,2.20,-2.73],[.22,.09,.055],'Silver')
cylinder(arch,'closer_arm',[2.63,2.225,-2.69],[2.36,2.30,-2.695],.009,'Silver')
box(arch,'exit_backing',[2.46,2.575,-2.74],[.40,.225,.06],'Silver')
quad(arch,'exit_face',[[2.27,2.47,-2.705],[2.65,2.47,-2.705],[2.65,2.68,-2.705],[2.27,2.68,-2.705]],'Exit sign')
box(arch,'right_wall',[3.30,1.48,.20],[.10,2.96,6.0],'Paint warm white',True)
box(arch,'back_skirting',[0,.055,-2.785],[6.5,.11,.033],'Dark trim')
box(arch,'right_skirting',[3.236,.055,.20],[.033,.11,6.0],'Dark trim')
# Two roller blinds and an existing high grille on the right wall.
blind=obj('blinds_vent','卷帘与空调风口','architecture',False,'位置与外观参考照片；尺寸近似。')
for z,l in [(-1.61,2.25),(1.03,2.94)]:
 box(blind,'blind_panel',[3.218,1.89,z],[.025,1.40,l],'Blind ivory')
 box(blind,'blind_bottom_bar',[3.199,1.18,z],[.038,.027,l+.018],'Silver')
 cylinder(blind,'blind_roller',[3.193,2.61,z-l/2],[3.193,2.61,z+l/2],.024,'Blind ivory')
 box(blind,'window_reveal_edge',[3.239,1.88,z+l/2+.015],[.02,1.45,.020],'Silver')
box(blind,'hvac_bulkhead',[3.064,2.805,.14],[.365,.31,4.15],'Paint warm white',True)
box(blind,'hvac_dark_grille',[2.871,2.82,.14],[.008,.205,3.87],'Dark trim')
for z in np.arange(-1.78,2.075,.048):box(blind,'grille_vertical',[2.86,2.82,z],[.014,.214,.010],'Blind ivory')
for y in [2.734,2.779,2.824,2.869,2.914]:box(blind,'grille_horizontal',[2.85,y,.14],[.016,.008,3.89],'Blind ivory')
# Back display board, actual image-cropped posters, and shallow vertical slats.
wallposter=obj('poster_wall','海报展示墙','architecture',False,'海报正面取自原始照片；背面和厚度为近似。')
box(wallposter,'poster_backboard',[-.73,1.90,-2.76],[4.98,1.62,.048],'Glass board')
for x in np.arange(-3.20,1.77,.055):box(wallposter,'backboard_rib',[x,1.90,-2.721],[.012,1.60,.012],'Dark trim')
for i,x in enumerate(np.linspace(-2.88,1.37,6)):
 w=.53;h=.96
 box(wallposter,'poster_frame_%d'%i,[x,1.98,-2.693],[w+.026,h+.026,.013],'Silver')
 quad(wallposter,'poster_%d'%i,[[x-w/2,1.98-h/2,-2.683],[x+w/2,1.98-h/2,-2.683],[x+w/2,1.98+h/2,-2.683],[x-w/2,1.98+h/2,-2.683]],'Poster %02d'%i)
box(wallposter,'bottom_board_rail',[-.72,1.044,-2.71],[5.01,.075,.14],'Dark trim')
# Ceiling is included in the model but hidden in the default cutaway viewer.
roof=obj('ceiling','吊顶与现有灯具','ceiling',False,'默认剖开查看；并非现实中拆除吊顶。网格间距、遮挡区为近似。')
box(roof,'ceiling_panel',[0,2.99,.2],[6.50,.06,6.0],'Ceiling')
for x in np.arange(-3.25,3.3,.65):box(roof,'ceiling_grid_x',[x,2.952,.2],[.012,.016,6.0],'Paint warm white')
for z in np.arange(-2.8,3.3,.65):box(roof,'ceiling_grid_z',[0,2.952,z],[6.5,.016,.012],'Paint warm white')
for x,z in [(-1.30,-1.5),(1.3,-.2),(-1.30,1.1)]:box(roof,'existing_light_panel',[x,2.935,z],[.60,.018,.60],'Lamp diffuser')
box(roof,'spotlight_track',[-.65,2.878,-2.22],[4.9,.026,.035],'Paint warm white')
for x in [-2.8,-1.75,-.65,.45,1.42]:
 cylinder(roof,'light_pendant',[x,2.88,-2.22],[x,2.78,-2.22],.014,'White plastic')
 cylinder(roof,'spotlight_body',[x,2.79,-2.22],[x,2.65,-2.24],.041,'White plastic',sections=20)
# Portable panel partition, not a structural wall.
gid=obj('partition','Lost and Found 折叠隔断','furniture',True,'面板分缝与标牌来自照片；宽度与厚度近似。')
for i in range(7):
 x=-1.80+(i+.5)*.465
 box(gid,'partition_panel_%d'%i,[x,.975,-1.845],[.453,1.95,.055],'Partition',True)
box(gid,'sign_back',[.73,1.44,-1.810],[.315,.255,.007],'White plastic')
quad(gid,'sign_face',[[.5725,1.3125,-1.805],[.8875,1.3125,-1.805],[.8875,1.5675,-1.805],[.5725,1.5675,-1.805]],'Lost and Found sign')
front=table('front_table','前方长桌与深灰桌布',(-1.52,.98),1.25,2.43)
rear=table('rear_table','后方工作桌与深灰桌布',(.47,-1.225),2.37,.89)
# Storage containers under the rear table.
def bin_obj(id,label,c,w,d,h,color,lid=False,parent=None):
 gid=obj(id,label,'storage',True,'静态容器代理；壁厚与遮挡处未测量。',parent)
 box(gid,'base',[c[0],c[1]+.018,c[2]],[w,.036,d],color,True)
 for sx in [-1,1]:box(gid,'side',[c[0]+sx*(w/2-.015),c[1]+h/2,c[2]],[.03,h,d],color,True)
 for sz in [-1,1]:box(gid,'side',[c[0],c[1]+h/2,c[2]+sz*(d/2-.015)],[w,h,.03],color,True)
 for sx in [-1,1]:box(gid,'rim',[c[0]+sx*w/2,c[1]+h,c[2]],[.035,.025,d+.035],color)
 for sz in [-1,1]:box(gid,'rim',[c[0],c[1]+h,c[2]+sz*d/2],[w+.035,.025,.035],color)
 if lid:box(gid,'lid',[c[0],c[1]+h+.025,c[2]],[w+.04,.045,d+.04],color,True)
 return gid
bin_obj('grey_bin','桌下灰色收纳箱',(-.32,.014,-1.14),.46,.42,.21,'Grey plastic')
bin_obj('black_bin','桌下黑色收纳箱',(.39,.014,-1.35),.78,.42,.32,'Black plastic',True)
bin_obj('blue_bin','桌下蓝色收纳箱',(1.12,.014,-1.33),.47,.38,.29,'Blue plastic',True)
for x in np.arange(.915,1.335,.057):box('blue_bin','plastic_rib',[x,.162,-1.127],[.012,.252,.009],'Blue plastic')
# Trash bin in the gap between the tables.
gid=obj('waste_bin','桌间黑色垃圾桶','storage',True,'桶内小物仅做近似，不是完整重建。')
cylinder(gid,'bin_body',[-.93,.024,-.75],[-.93,.52,-.75],.185,'Black plastic',True,32,.21)
cylinder(gid,'bin_lid',[-.93,.512,-.75],[-.93,.54,-.75],.224,'Black rubber',False,32)
box(gid,'cloth_discard',[-.94,.546,-.75],[.18,.025,.13],'White canvas',angle=.25)
# Brown paper shopping bag (front texture from photo), white/red cloth bags remain.
gid=obj('paper_bag','UNIQLO 牛皮纸袋','props',True,'正面为照片纹理；侧面、背面与折痕为近似；无柔体物理。',front)
bx,bz=-1.57,.14;by=.765;w,d,h=.39,.18,.40
box(gid,'paper_base',[bx,by+.005,bz],[w,.010,d],'Paper',True)
# Open paper bag panels with slightly uneven top edge.
for side in [-1,1]:
 nx,ny=24,24;vs=[];uv=[];fs=[]
 for j in range(ny+1):
  t=j/ny
  for i in range(nx+1):
   a=i/nx;x=bx-w/2+w*a;yy=by+h*t+(.008*np.sin(a*21)+.004*np.sin(a*59))*t**5
   zz=bz+side*(d/2+.008*np.sin(a*15+t*13)*np.sin(np.pi*t)+.010*t)
   vs.append([x,yy,zz]);uv.append([a,t])
 for j in range(ny):
  for i in range(nx):
   k=j*(nx+1)+i;fs.extend([[k,k+1,k+nx+2],[k,k+nx+2,k+nx+1]])
 mesh(gid,'bag_panel',vs,fs,'Bag front' if side==1 else 'Paper',uv)
for sx in [-1,1]:
 quad(gid,'bag_gusset',[[bx+sx*w/2,by,bz-d/2],[bx+sx*w/2,by,bz+d/2],[bx+sx*w/2,by+h,bz+d/2+.01],[bx+sx*w/2,by+h,bz-d/2-.01]],'Paper')
for xx in [-.10,.03,.135]:box(gid,'packing_tape',[bx+xx,by+h-.014,bz+d/2+.014],[.036,.066,.004],'Tan tape',angle=.02)
# Proxy collision volume is separate: not a physically accurate deformable bag.
# Leave the top open: no solid cube intersects the paper surface.
quad(gid,'bag_inner_base',[[bx-w/2,by+.006,bz-d/2],[bx-w/2,by+.006,bz+d/2],[bx+w/2,by+.006,bz+d/2],[bx+w/2,by+.006,bz-d/2]],'Brown part')
# Cloth tote body, hand-shaped curved meshes rather than image planes.
def clothbag(id,label,x,z,red=False):
 gid=obj(id,label,'props',True,'外形为静态近似；不含布料动力学。',front)
 w=.31;d=.12;h=.24;by=.766;matn='White canvas'
 for side in [-1,1]:
  vs=[];uv=[];fs=[];nx=24;ny=18
  for j in range(ny+1):
   t=j/ny
   for i in range(nx+1):
    u=i/nx;xx=x+(u-.5)*w*(.78+.22*np.sin(np.pi*t/2))
    yy=by+h*t-.016*np.sin(u*np.pi)*t
    zz=z+side*(d/2+.018*np.sin(np.pi*u)*np.sin(np.pi*t))+.007*np.sin(u*27+t*15)
    vs.append([xx,yy,zz]);uv.append([u,t])
  for j in range(ny):
   for i in range(nx):
    k=j*(nx+1)+i;fs.extend([[k,k+1,k+nx+2],[k,k+nx+2,k+nx+1]])
  mesh(gid,'tote_cloth',vs,fs,matn,uv)
 for side in [-1,1]:
  tube(gid,'tote_handle',[[x-.09,by+h-.02,z+side*d/2],[x-.12,by+h+.095,z+side*.07],[x,by+h+.15,z+side*.06],[x+.12,by+h+.08,z+side*.06],[x+.09,by+h-.02,z+side*d/2]],.011,'Red canvas' if red else matn,36,8)
 if not red:
  quad(gid,'nus_front',[[x-.13,by+.04,z+.084],[x+.13,by+.04,z+.084],[x+.13,by+.14,z+.084],[x-.13,by+.14,z+.084]],'Tote label')
 return gid
clothbag('white_tote','白色 NUS 布袋',-1.225,.16)
clothbag('red_white_bag','红白布袋',-1.87,.10,True)
# Small existing parts on the rear table. No extra household objects have been introduced.
gid=obj('white_parts','桌面白色环形与齿形小件','props',True,'可独立移动的静态网格；形状近似。抓取前需重建精细形状并设置刚体参数。',rear)
ring(gid,'white_ring',[.80,.786,-1.20],.066,.009,'White plastic')
ring(gid,'white_ring_inner',[.82,.789,-1.195],.037,.006,'White plastic')
for xx,zz,rr in [(.68,-1.16,.024),(.97,-1.17,.024),(.83,-1.25,.022)]:
 cylinder(gid,'gear_core',[xx,.761,zz],[xx,.787,zz],rr,'White plastic',False,24)
 for i in range(12):
  ang=2*np.pi*i/12;box(gid,'gear_tooth',[xx+(rr+.002)*np.cos(ang),.775,zz+(rr+.002)*np.sin(ang)],[.013,.026,.010],'White plastic',angle=-ang)
for x in [.76,.82,.87]:cylinder(gid,'pin',[x,.772,-1.22],[x,.815,-1.22],.006,'White plastic')
gid=obj('brown_part','桌面棕色圆柱小件','props',True,'静态小件；质量与精确形状未知。',rear)
cylinder(gid,'round_part',[-.28,.760,-1.27],[-.28,.792,-1.27],.025,'Brown part',False,24)
# TVs on mobile stands. Right monitor faces into the room, left monitor faces the camera.
def monitor(id,label,x,z,width,height,angle=0):
 gid=obj(id,label,'equipment',True,'屏幕、支架、底座按照片建立；背部连接件近似。')
 def tr(p):
  a=np.array(p,float);R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
  return a@R.T+np.array([x,0,z])
 def bb(name,c,s,m,coll=False):box(gid,name,tr(c),s,m,coll,angle)
 bb('tv_base',[0,.048,0],[.83,.065,.51],'Black plastic',True)
 bb('tv_column',[0,.68,-.045],[.115,1.24,.115],'Dark trim',True)
 bb('mount_plate',[0,1.54,-.075],[.36,.35,.08],'Black plastic')
 bb('screen_frame',[0,1.70,0],[width+.034,height+.034,.062],'Black plastic',True)
 pts=[[-width/2,1.70-height/2,.034],[width/2,1.70-height/2,.034],[width/2,1.70+height/2,.034],[-width/2,1.70+height/2,.034]]
 quad(gid,'screen_surface',[tr(p) for p in pts],'Screen glass')
 for a in [-.34,.34]:
  for b in [-.19,.19]:
   p=tr([a,.024,b]);cylinder(gid,'wheel',p+[-.035,0,0],p+[.035,0,0],.033,'Black rubber')
 return gid
monitor('left_monitor','左侧显示屏及移动支架',-2.66,-1.61,1.42,.80,.02)
monitor('right_monitor','右侧显示屏及移动支架',2.99,.73,1.47,.83,-np.pi/2)
# Low shelf under left screen, folded rods / frames / boxes (not robots).
gid=obj('equipment_rack','左后方物料架与折叠杆件','storage',True,'保留非机器人设备与杂物；被遮挡的小件简化为代理。')
for x in [-3.02,-2.23]:
 for z in [-2.51,-1.96]:box(gid,'rack_post',[x,.57,z],[.035,1.14,.035],'Dark trim',True)
for y in [.12,.51,.91]:box(gid,'rack_shelf',[-2.625,y,-2.235],[.90,.025,.62],'Dark trim',True)
for i in range(15):
 y=.17+i*.047
 cylinder(gid,'stored_wooden_rod',[-3.10,y,-2.02],[-2.06,y,-2.02],.018,'Tube wood')
for x in [-2.96,-2.72,-2.44,-2.20]:box(gid,'folding_frame_bar',[x,.67,-1.92],[.024,1.08,.025],'Dark trim')
box(gid,'cardboard_box',[-2.62,1.05,-2.21],[.48,.29,.39],'Cardboard',True)
box(gid,'box_tape',[-2.62,1.202,-2.21],[.06,.006,.39],'Tan tape')
box(gid,'cart_shelf',[-2.52,.44,-1.35],[.58,.045,.40],'Dark trim',True)
for x in [-2.77,-2.27]:
 for z in [-1.51,-1.18]:box(gid,'cart_leg',[x,.24,z],[.026,.45,.026],'Dark trim')
box(gid,'small_control_box',[-2.53,.52,-1.34],[.24,.12,.20],'Black plastic',True)
# Floor cardboard carton to the left of the partition.
gid=obj('floor_carton','隔断旁纸箱','storage',True,'箱体与开口为近似。')
box(gid,'carton_base',[-1.99,.015,-2.07],[.34,.03,.29],'Cardboard',True)
for sx in [-1,1]:box(gid,'carton_side',[-1.99+sx*.165,.16,-2.07],[.012,.31,.29],'White canvas',True)
for sz in [-1,1]:box(gid,'carton_side',[-1.99,.16,-2.07+sz*.14],[.34,.31,.014],'White canvas',True)
box(gid,'open_cardboard_flap',[-2.07,.36,-2.14],[.17,.19,.009],'Cardboard',angle=.1)
# Three blue pedals and network / power components remain near the right monitor.
gid=obj('network_power','路由器、插线板与地面线缆','equipment',True,'线缆只有视觉网格，不是软体；插头、走线与长度近似。')
box(gid,'router',[2.69,.16,.61],[.25,.055,.21],'Black plastic',True)
for x,z in [(2.58,.53),(2.80,.53),(2.58,.70),(2.80,.70)]:cylinder(gid,'antenna',[x,.18,z],[x-.025,.44,z],.008,'Black plastic',sections=10)
box(gid,'power_strip',[2.55,.038,1.05],[.12,.055,.30],'White plastic',True)
for z in [.95,1.04,1.13]:
 box(gid,'plug',[2.55,.083,z],[.068,.055,.064],'White plastic')
 box(gid,'socket_slot',[2.552,.112,z],[.011,.004,.026],'Dark trim')
box(gid,'power_brick',[2.04,.029,.52],[.20,.058,.105],'Black plastic',True)
box(gid,'red_LED',[1.936,.033,.52],[.003,.010,.017],'Indicator red')
paths=[([[2.95,1.52,.69],[2.94,.8,.68],[2.84,.06,.61],[2.48,.02,.78],[2.55,.03,.95]],.004,'Black rubber'),([[2.89,1.15,1.10],[2.93,.68,1.05],[2.90,.03,1.27],[2.58,.025,1.27],[2.53,.07,1.13]],.003,'White plastic'),([[2.15,.035,.52],[2.40,.014,.47],[2.50,.014,.70],[2.22,.014,.90],[2.08,.014,.68],[2.4,.02,.65],[2.51,.04,1.03]],.004,'Black rubber'),([[2.76,.16,.66],[2.77,.035,.91],[2.59,.02,1.21],[2.72,.02,1.3],[2.82,.04,1.24]],.003,'Black rubber'),([[2.04,.03,.54],[1.79,.02,.72],[1.92,.018,1.01],[2.27,.018,1.10],[2.31,.02,.72],[2.71,.08,.59]],.004,'Black rubber')]
for i,(p,r,m) in enumerate(paths):tube(gid,'cable_%02d'%i,p,r,m,60,6)
gid=obj('blue_pedals','三只蓝色脚踏开关','props',True,'外形和尺寸近似；未绑定交互开关。')
for i in range(3):
 x=2.41+i*.18;z=1.34
 box(gid,'pedal_base',[x,.023,z],[.145,.028,.18],'Black rubber',True)
 # Slightly sloping pedal top using an explicit wedge.
 vs=[[x-.069,.035,z-.082],[x+.069,.035,z-.082],[x+.069,.07,z+.082],[x-.069,.07,z+.082],[x-.069,.02,z+.082],[x+.069,.02,z+.082]]
 fs=[[0,1,2],[0,2,3],[3,2,5],[3,5,4],[0,3,4],[1,5,2]]
 mesh(gid,'blue_pedal',vs,fs,'Blue plastic',collider=True,smooth=False)
# Existing sockets on right wall, kept without inventing new fixture positions.
gid=obj('wall_sockets','墙面电源插座','architecture',False,'位置取近似；不是电气施工图。')
for z,y in [(-1.52,1.05),(-.05,1.05),(1.00,1.06),(2.41,1.04)]:
 box(gid,'socket_plate',[3.202,y,z],[.04,.11,.15],'White plastic')
 for dz,dy in [(0,.018),(-.028,-.020),(.028,-.020)]:box(gid,'socket_hole',[3.178,y+dy,z+dz],[.003,.014,.008],'Dark trim')
box(gid,'surface_cable_trunking',[3.205,1.16,.20],[.04,.025,5.98],'Ceiling')
# Office chair at edge of photo 3, empty (no person geometry).
gid=obj('office_chair','右侧黑色办公椅','furniture',True,'按可见椅背、扶手及底盘近似；未保留人物。')
cx,cz=2.21,2.43
cylinder(gid,'chair_post',[cx,.10,cz],[cx,.49,cz],.03,'Silver')
box(gid,'chair_seat',[cx,.48,cz],[.47,.085,.47],'Black rubber',True,angle=-.22)
box(gid,'chair_back',[cx+.05,.80,cz+.245],[.43,.57,.054],'Dark trim',True,angle=-.22)
for sx in [-1,1]:
 box(gid,'armrest',[cx+sx*.27,.66,cz],[.063,.046,.34],'Black plastic')
 box(gid,'armrest_support',[cx+sx*.27,.568,cz+.06],[.025,.16,.035],'Dark trim')
for i in range(5):
 a=2*np.pi*i/5;end=np.array([cx+.32*np.cos(a),.08,cz+.32*np.sin(a)])
 cylinder(gid,'chair_leg',[cx,.10,cz],end,.018,'Dark trim')
 cylinder(gid,'caster',end+[-.033,-.027,0],end+[.033,-.027,0],.038,'Black rubber')
# Partial foreground chair / bin visible in reference 4, retained as simplified objects.
gid=obj('red_chair','前缘红色椅子','furniture',True,'仅依据照片中露出的部分近似；不包含人物。')
box(gid,'red_back',[-1.53,.58,2.79],[.53,.35,.045],'Red canvas',True,angle=.04)
box(gid,'seat',[-1.53,.42,2.55],[.47,.058,.44],'Grey plastic',True)
for x in [-1.73,-1.33]:
 for z in [2.38,2.73]:cylinder(gid,'chair_leg',[x,.02,z],[x,.40,z],.015,'Silver')
bin_obj('foreground_tray','前缘灰色塑料周转盆',(-2.06,.44,2.61),.57,.43,.15,'Grey plastic')
# Existing floor tape only. Do not replace it with the invented large rectangular zone.
gid=obj('floor_markings','原有黄黑胶带、粉色胶带与白标记','markings',False,'胶带位置与长度依据可见部分近似；未添加新的工作区边框。')
quad(gid,'rear_hazard',[[-.83,.007,-.675],[1.77,.007,-.675],[1.77,.007,-.620],[-.83,.007,-.620]],'Hazard tape',[[0,0],[5,0],[5,1],[0,1]])
# Second short tape segment near the storage gap, visible in reference 4.
quad(gid,'side_hazard',[[-1.04,.007,-.72],[-1.28,.007,-.62],[-1.255,.007,-.57],[-1.015,.007,-.67]],'Hazard tape')
a=np.array([-.97,.008,-.28]);b=np.array([.24,.008,3.15]);di=b-a;di[1]=0;per=np.array([di[2],0,-di[0]]);per=per/np.linalg.norm(per)*.035
quad(gid,'pink_tape',[a-per,b-per,b+per,a+per],'Pink tape')
for x,z in [(.27,-.41),(1.35,-.37),(-.52,-.39)]:box(gid,'small_white_tape',[x,.009,z],[.042,.003,.022],'White canvas')

# Merge small components by object/material/collider, keeping semantic instances separate.
merged=[]
for ob in objects:
 keys=[]
 for m in meshes:
  if m['object']==ob['id'] and (m['mat'],m['collider']) not in keys:keys.append((m['mat'],m['collider']))
 for mi,co in keys:
  parts=[m for m in meshes if m['object']==ob['id'] and m['mat']==mi and m['collider']==co]
  vv=[];nn=[];uu=[];ff=[];offset=0
  for p in parts:
   vv.append(p['v']);nn.append(p['n']);uu.append(p['uv']);ff.append(p['f']+offset);offset+=len(p['v'])
  merged.append({'object':ob['id'],'name':ob['id']+'__'+str(mi)+('_collider' if co else '_visual'),'v':np.concatenate(vv),'n':np.concatenate(nn),'uv':np.concatenate(uu),'f':np.concatenate(ff),'mat':mi,'collider':co})
meshes=merged
# Group bounds in the authored (Y-up) space; do not represent these as measured dimensions.
for ob in objects:
 vv=np.concatenate([m['v'] for m in meshes if m['object']==ob['id']])
 ob['bounds']=[vv.min(0).tolist(),vv.max(0).tolist()]
# GLB writer: fully embedded textures, named groups, indexed triangle meshes.
blob=bytearray();views=[];accessors=[]
def append_bytes(data,target=None):
 while len(blob)%4:blob.append(0)
 off=len(blob);blob.extend(data);d={'buffer':0,'byteOffset':off,'byteLength':len(data)}
 if target:d['target']=target
 views.append(d);return len(views)-1

def accessor(a,typ,comp=5126,bounds=False):
 a=np.asarray(a,dtype='<f4' if comp==5126 else '<u4')
 v=append_bytes(a.tobytes(),34963 if typ=='SCALAR' and comp==5125 else 34962)
 d={'bufferView':v,'componentType':comp,'count':len(a),'type':typ}
 if bounds:d.update(min=a.min(0).astype(float).tolist(),max=a.max(0).astype(float).tolist())
 accessors.append(d);return len(accessors)-1
images=[];textures=[];texindex={}
for ma in materials:
 if ma['texture'] and ma['texture'] not in texindex:
  texindex[ma['texture']]=len(textures)
  v=append_bytes((TEX/ma['texture']).read_bytes());images.append({'bufferView':v,'mimeType':'image/jpeg','name':ma['texture']})
  textures.append({'source':len(images)-1,'sampler':0})
gmats=[]
for ma in materials:
 pbr={'baseColorFactor':ma['color']+[1],'roughnessFactor':ma['roughness'],'metallicFactor':ma['metallic']}
 if ma['texture']:pbr['baseColorTexture']={'index':texindex[ma['texture']]}
 gmats.append({'name':ma['name'],'doubleSided':True,'pbrMetallicRoughness':pbr})
gmeshes=[];nodes=[{'name':'Photo_referenced_lab_proxy','children':[],'extras':{'estimatedScale':True,'referenceTableHeightM':.75,'reconstructionType':'photo-referenced manual approximate scene, not a calibrated scan','robotRemoved':True,'partialCoverage':True}}]
objnode={}
for ob in objects:
 idx=len(nodes);objnode[ob['id']]=idx
 nodes.append({'name':ob['id'],'children':[],'extras':ob})
# Flat groups make translations transparent. Parent IDs are metadata for carrying tabletop props.
nodes[0]['children']=list(objnode.values())
for m in meshes:
 pos=accessor(m['v'],'VEC3',bounds=True);norm=accessor(m['n'],'VEC3');uv=accessor(np.column_stack((m['uv'][:,0],1-m['uv'][:,1])),'VEC2');idx=accessor(m['f'].reshape(-1),'SCALAR',5125)
 gmeshes.append({'name':m['name'],'primitives':[{'attributes':{'POSITION':pos,'NORMAL':norm,'TEXCOORD_0':uv},'indices':idx,'material':m['mat'],'mode':4}],'extras':{'staticCollider':m['collider']}})
 ni=len(nodes);nodes.append({'name':m['name'],'mesh':len(gmeshes)-1,'extras':{'objectId':m['object'],'staticCollider':m['collider']}});nodes[objnode[m['object']]]['children'].append(ni)
doc={'asset':{'version':'2.0','generator':'Photo-referenced lab proxy builder; uncalibrated'},'scene':0,'scenes':[{'name':'Robot-free lab | approximate partial scene','nodes':[0]}],'nodes':nodes,'meshes':gmeshes,'materials':gmats,'textures':textures,'images':images,'samplers':[{'magFilter':9729,'minFilter':9987,'wrapS':10497,'wrapT':10497}],'buffers':[{'byteLength':len(blob)}],'bufferViews':views,'accessors':accessors}
j=json.dumps(doc,ensure_ascii=False,separators=(',',':')).encode();j+=b' '*((-len(j))%4);blob+=b'\0'*((-len(blob))%4)
glb=struct.pack('<III',0x46546c67,2,12+8+len(j)+8+len(blob))+struct.pack('<II',len(j),0x4e4f534a)+j+struct.pack('<II',len(blob),0x004e4942)+blob
(ROOT/'robot_lab.glb').write_bytes(glb)
# Native USD scene. Convert (X,Y,Z)_glTF to (X,-Z,Y)_USD; metres and Z-up.
def vec(v):return f'({float(v[0]):.6f}, {float(v[1]):.6f}, {float(v[2]):.6f})'
def farray(a):return ', '.join(vec(v) for v in a)
R=np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=float)
L=['#usda 1.0','(','    defaultPrim = "World"','    metersPerUnit = 1','    upAxis = "Z"','    doc = "Uncalibrated photo-referenced partial scene. No robot. Static collision geometry. Not a measured reconstruction."',')','', 'def Xform "World"','{','    custom bool reconstruction:scaleCalibrated = false','    custom double reconstruction:assumedTableHeightMeters = 0.75','    custom string reconstruction:method = "Photo-referenced approximate modelling; unseen geometry simplified."','    double3 xformOp:scale = (1, 1, 1)','    uniform token[] xformOpOrder = ["xformOp:scale"]','    def Scope "Looks"','    {']
for i,ma in enumerate(materials):
 L+=['        def Material "M%d"'%i,'        {','            token outputs:surface.connect = </World/Looks/M%d/PBR.outputs:surface>'%i,'            def Shader "PBR"','            {','                uniform token info:id = "UsdPreviewSurface"','                color3f inputs:diffuseColor = '+vec(ma['color']),'                float inputs:roughness = %.5f'%ma['roughness'],'                float inputs:metallic = %.5f'%ma['metallic'],'                token outputs:surface']
 if ma['texture']:L+=['                color3f inputs:diffuseColor.connect = </World/Looks/M%d/Texture.outputs:rgb>'%i]
 L+=['            }']
 if ma['texture']:
  L+=['            def Shader "UV"','            {','                uniform token info:id = "UsdPrimvarReader_float2"','                token inputs:varname = "st"','                float2 outputs:result','            }','            def Shader "Texture"','            {','                uniform token info:id = "UsdUVTexture"','                asset inputs:file = @textures/'+ma['texture']+'@','                token inputs:sourceColorSpace = "sRGB"','                token inputs:wrapS = "repeat"','                token inputs:wrapT = "repeat"','                float2 inputs:st.connect = </World/Looks/M%d/UV.outputs:result>'%i,'                float3 outputs:rgb','            }']
 L+=['        }']
L+=['    }','    def PhysicsScene "Physics"','    {','        vector3f physics:gravityDirection = (0, 0, -1)','        float physics:gravityMagnitude = 9.81','    }','    def DistantLight "Key"','    {','        float inputs:intensity = 1400','        color3f inputs:color = (1, 0.98, 0.94)','        float inputs:angle = 20','        float3 xformOp:rotateXYZ = (-28, 20, -25)','        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]','    }','    def DomeLight "Fill"','    {','        float inputs:intensity = 700','        color3f inputs:color = (0.91, 0.95, 1)','    }']
for ob in objects:
 L+=['    def Xform "'+ob['id']+'"','    {','        custom string userProperties:label = '+json.dumps(ob['label'],ensure_ascii=False),'        custom bool reconstruction:estimated = true','        custom string reconstruction:note = '+json.dumps(ob['note'],ensure_ascii=False),'        double3 xformOp:translate = (0, 0, 0)','        uniform token[] xformOpOrder = ["xformOp:translate"]']
 for m in [a for a in meshes if a['object']==ob['id']]:
  v=m['v']@R.T;n=m['n']@R.T
  apis=['MaterialBindingAPI']+(['PhysicsCollisionAPI','PhysicsMeshCollisionAPI'] if m['collider'] else [])
  L+=['        def Mesh "'+m['name']+'" (','            prepend apiSchemas = '+json.dumps(apis), '        )','        {','            uniform token subdivisionScheme = "none"','            uniform bool doubleSided = true','            point3f[] points = ['+farray(v)+']','            int[] faceVertexCounts = ['+', '.join(['3']*len(m['f']))+']','            int[] faceVertexIndices = ['+', '.join(str(int(q)) for q in m['f'].ravel())+']','            normal3f[] normals = ['+farray(n)+'] (','                interpolation = "vertex"','            )','            texCoord2f[] primvars:st = ['+', '.join('(%.6f, %.6f)'%tuple(uv) for uv in m['uv'])+'] (','                interpolation = "vertex"','            )','            float3[] extent = ['+vec(v.min(0))+', '+vec(v.max(0))+']','            rel material:binding = </World/Looks/M%d>'%m['mat']]
  if m['collider']:L+=['            bool physics:collisionEnabled = true','            uniform token physics:approximation = "none"']
  L+=['        }']
 L+=['    }']
L+=['}']
(ROOT/'robot_lab.usda').write_text('\n'.join(L),encoding='utf-8')
# Saved source mesh records support programmatic audits without decoding the glTF.
manifest={'description':'Photo-referenced approximate partial laboratory scene, not a calibrated 3D reconstruction','robots':0,'scale_calibrated':False,'assumed_table_height_m':.75,'up_axis_glb':'Y','up_axis_usd':'Z','usd_meters_per_unit':1,'objects':objects,'materials':materials,'mesh_count':len(meshes),'triangle_count':int(sum(len(m['f']) for m in meshes)),'vertex_count':int(sum(len(m['v']) for m in meshes)),'static_collision_mesh_count':sum(m['collider'] for m in meshes),'physics_runtime_validated':False,'limitations':['No calibrated intrinsics, depth, camera poses or real measured lengths were supplied.','Rear/hidden surfaces and small clutter are approximate.','Partial model boundaries are capture extent, not real room dimensions.','No articulated robot; no human geometry.','No cloth/cable dynamics; no calibrated rigid-body targets.','Web viewer is not a physics simulation.','USD authored with static collision meshes; actual Isaac Sim execution has not been tested.']}
(ROOT/'scene_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in manifest.items() if k not in ['objects','materials','limitations']},indent=2))
print('GLB',len(glb),'USD', (ROOT/'robot_lab.usda').stat().st_size)
