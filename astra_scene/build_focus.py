"""Restricted photo-reference model: purple/white partitions and table zone."""
from pathlib import Path
import math, json, random
SOURCE=Path(__file__).with_name('build_photo_lab.py')
# Reuse material and primitive helpers only; do not build the previous room.
exec(compile(SOURCE.read_text(encoding='utf-8').split('# Estimated 7.2')[0],str(SOURCE),'exec'))
OUT=ROOT/'focus_v4'; OUT.mkdir(exist_ok=True)
random.seed(21)
def surface(material,scale,strength,distance):
    nodes=material.node_tree.nodes; links=material.node_tree.links
    noise=nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value=scale; noise.inputs['Detail'].default_value=2
    bump=nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value=strength; bump.inputs['Distance'].default_value=distance
    links.new(noise.outputs['Fac'],bump.inputs['Height']);links.new(bump.outputs['Normal'],nodes.get('Principled BSDF').inputs['Normal'])
surface(dark,240,.22,.0008)
dark.node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.92
dark.node_tree.nodes.get('Principled BSDF').inputs['Sheen Weight'].default_value=.2
surface(mauve,130,.16,.0005);surface(paper,95,.25,.0007);surface(white,170,.12,.0004)
surface(wood,100,.2,.0008)
box('Local_floor',(0,0,-.06),(4.8,4.8,.12),wood,True)
# Narrow joints and subtly different timber strips, rather than a single stretched grain.
for row in range(20):
    y=-2.4+(row+.5)*.24
    offset=(row%3)*.4
    for col in range(5):
        lo=max(-2.4,-3.6+offset+col*1.2); hi=min(2.4,-2.4+offset+col*1.2)
        if hi<=lo:continue
        m=wood.copy();r=m.node_tree.nodes.get('Color Ramp')
        if r:
            factor=random.uniform(.88,1.06)
            for e in r.color_ramp.elements:e.color=tuple(v*factor for v in e.color[:3])+(1,)
        box('Timber_plank',((lo+hi)/2,y,.003),(hi-lo-.002,.238,.006),m,False,.0006)
# The purple partition and white operator screen bound opposite sides of this local zone.
for i in range(7):
    x=-1.65+i*.49
    o=box('Purple_panel',(x,1.68,1.0),(.483,.055,2.0),mauve,True,.006)
    o.rotation_euler.z=math.radians(2 if i%2 else -2)
box('Sign',(.75,1.643,1.42),(.36,.012,.21),paper)
text('Lost and Found',(.75,1.631,1.40),.045)
for i in range(5):
    x=-.2+i*.49
    o=box('White_panel',(x,-1.82,1.0),(.483,.045,2),paper,True,.006)
    o.rotation_euler.z=math.radians(3 if i%2 else -3)
    # Shallow raised horizontal texture visible on the white folding screen.
    for j in range(34): box('White_panel_ridge',(x,-1.793,.04+j*.058),(.465,.002,.0025),white)
# Small table against the purple screen, and the long foreground demonstration table.
table('Rear_table',.80,1.08,1.8,.82)
box('Rear_storage',(.85,1.07,.17),(.68,.5,.32),blue,True,.025)
table('Front_table',-1.0,-.17,1.12,2.5)
# Replace simple flat skirts with a continuous draped, lightly wrinkled cloth mesh.
for o in list(bpy.data.objects):
    if o.name.startswith('Front_table_cloth'): bpy.data.objects.remove(o,do_unlink=True)
verts=[]; faces=[]; nx=60; ny=90
for iy in range(ny+1):
    v=-1.63+3.26*iy/ny
    for ix in range(nx+1):
        u=-.89+1.78*ix/nx
        ex=max(abs(u)-.56,0); ey=max(abs(v)-1.25,0)
        x=-1+max(-.56,min(.56,u))+math.copysign(.035*(1-math.exp(-ex*12)),u)
        y=-.17+max(-1.25,min(1.25,v))+math.copysign(.035*(1-math.exp(-ey*12)),v)
        drop=max(ex,ey)
        z=.815-drop+.0008*math.sin(u*26+v*15)
        if drop==0:
            z+=.003*math.exp(-((u-.10-.07*math.sin(v*2))/.018)**2)
            z+=.002*math.exp(-((v+.38)/.027)**2)
        if drop>0:
            x+=(.004*math.sin(v*23)+.003*math.sin(v*39+.6))*min(drop*5,1)
            y+=(.004*math.sin(u*22)+.002*math.sin(u*41))*min(drop*5,1)
        verts.append((x,y,z))
for iy in range(ny):
    for ix in range(nx):
        a=iy*(nx+1)+ix; faces.append((a,a+1,a+nx+2,a+nx+1))
mesh=bpy.data.meshes.new('Draped_cloth'); mesh.from_pydata(verts,[],faces); mesh.update()
ob=bpy.data.objects.new('Grey_tablecloth',mesh); bpy.context.collection.objects.link(ob); ob.data.materials.append(dark)
for p in mesh.polygons:p.use_smooth=True
solid=ob.modifiers.new('Fabric thickness','SOLIDIFY'); solid.thickness=.0015
box('Local_wall',(2.35,0,1.35),(.08,3.6,2.7),white,True)
tv('Wall_display',1.91,-.33,1.02)
table('Operator_table',1.3,-1.37,1.05,.60)
box('Monitor',(1.32,-1.4,1.09),(.45,.045,.3),black)
box('Bin',(-.02,1.0,.29),(.32,.34,.58),black,True,.045)
box('Pink_floor_tape',(-.25,-.2,.004),(.045,2.8,.006),mauve)
for y in [-1,1]:
    bpy.ops.object.light_add(type='AREA',location=(0,y,3.0)); bpy.context.object.data.energy=150; bpy.context.object.data.shape='RECTANGLE';bpy.context.object.data.size=1.2;bpy.context.object.data.size_y=.6
bpy.ops.object.light_add(type='AREA',location=(-3,-3,2.5));fill=bpy.context.object;fill.data.energy=110;fill.data.size=4
fill.rotation_euler=(Vector((0,0,1))-fill.location).to_track_quat('-Z','Y').to_euler()
bpy.context.scene.world.use_nodes=True
bpy.context.scene.world.node_tree.nodes.get('Background').inputs[0].default_value=(.55,.58,.65,1)
bpy.context.scene.world.node_tree.nodes.get('Background').inputs[1].default_value=.35
def camera(eye,target):
    bpy.ops.object.camera_add(location=eye); c=bpy.context.object
    c.rotation_euler=(Vector(target)-c.location).to_track_quat('-Z','Y').to_euler(); c.data.lens=43; return c
sc=bpy.context.scene; sc.camera=camera((-6,-5,5.7),(0,0,.75))
sc.render.engine='CYCLES'; sc.cycles.samples=64; sc.cycles.use_denoising=True;sc.cycles.device='CPU'
sc.render.resolution_x=1200;sc.render.resolution_y=1000;sc.render.resolution_percentage=100
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'focused_zone.blend'))
bpy.ops.export_scene.gltf(filepath=str(OUT/'focused_zone.glb'),export_format='GLB')
bpy.ops.wm.usd_export(filepath=str(OUT/'focused_visual.usdc'),root_prim_path='/World')
stage=Usd.Stage.CreateNew(str(OUT/'focused_zone.usda')); w=UsdGeom.Xform.Define(stage,'/World'); stage.SetDefaultPrim(w.GetPrim())
w.GetPrim().GetReferences().AddReference('focused_visual.usdc','/World'); UsdGeom.SetStageMetersPerUnit(stage,1); UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
for i,c in enumerate(colliders):
    p=UsdGeom.Cube.Define(stage,f'/World/Collision/C{i}');p.CreateSizeAttr(1)
    p.AddTranslateOp().Set(Gf.Vec3d(*c['center']));p.AddScaleOp().Set(Gf.Vec3f(*c['size']));p.CreateVisibilityAttr('invisible');UsdPhysics.CollisionAPI.Apply(p.GetPrim())
stage.GetRootLayer().Save()
sc.render.filepath=str(OUT/'preview.png'); bpy.ops.render.render(write_still=True)
sc.camera=camera((-3.1,-2.9,1.7),(.4,.7,1.05));sc.camera.data.lens=28
sc.render.filepath=str(OUT/'eye_level.png');bpy.ops.render.render(write_still=True)
(OUT/'notes.txt').write_text('材质光照第四版：桌布微表面和不规则褶皱、木地板拼缝和色差、挡板凹凸、柔和顶灯。照片参考近似建模，尺寸未校准。预览为 Blender Cycles 渲染；USD/GLB 导出不保证保留程序化材质节点，Isaac 外观待单独验证。未验证物理运行。',encoding='utf-8')
