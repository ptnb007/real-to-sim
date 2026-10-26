"""Photo-guided approximate lab layout. Not photogrammetry or measured geometry."""
import bpy, math, json, sys
from pathlib import Path
from mathutils import Vector
from pxr import Usd, UsdGeom, UsdPhysics, Gf

ROOT = Path(__file__).resolve().parent
V2 = '--video-v2' in sys.argv
DEST = ROOT / 'video_v2' if V2 else ROOT
OUT = DEST / 'output_mesh'
USD = DEST / 'isaac_usd'
OUT.mkdir(exist_ok=True, parents=True); USD.mkdir(exist_ok=True, parents=True)
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
bpy.context.scene.unit_settings.system = 'METRIC'
def mat(name, color, rough=.65, metal=0):
    m=bpy.data.materials.new(name); m.diffuse_color=(*color,1); m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF'); p.inputs['Base Color'].default_value=(*color,1)
    p.inputs['Roughness'].default_value=rough; p.inputs['Metallic'].default_value=metal
    return m
white=mat('Warm plaster',(.77,.76,.70)); dark=mat('Charcoal textile',(.105,.115,.11))
black=mat('Black plastic',(.018,.023,.026)); mauve=mat('Lavender folding screen',(.49,.44,.47))
steel=mat('Steel',(.32,.35,.36),.28,.7); red=mat('Red chair trim',(.58,.025,.035))
blue=mat('Storage blue',(.045,.12,.28)); paper=mat('Poster paper',(.84,.82,.74))
tan=mat('Paper bag',(.48,.30,.12)); glass=mat('Display dark glass',(.016,.035,.044),.18)
wood=mat('Wood floor',(.34,.22,.13))
n=wood.node_tree.nodes; l=wood.node_tree.links
tex=n.new('ShaderNodeTexNoise'); tex.inputs['Scale'].default_value=5; tex.inputs['Detail'].default_value=2
mapping=n.new('ShaderNodeVectorMath'); mapping.operation='MULTIPLY'; mapping.inputs[1].default_value=(1,45,3)
coord=n.new('ShaderNodeTexCoord'); l.new(coord.outputs['Generated'],mapping.inputs[0]); l.new(mapping.outputs[0],tex.inputs['Vector'])
ramp=n.new('ShaderNodeValToRGB'); ramp.color_ramp.elements[0].color=(.12,.065,.025,1); ramp.color_ramp.elements[1].color=(.43,.29,.16,1)
l.new(tex.outputs['Fac'],ramp.inputs[0]); l.new(ramp.outputs[0],n.get('Principled BSDF').inputs['Base Color'])
colliders=[]
def box(name, loc, size, material, collision=False, bevel=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc); o=bpy.context.object; o.name=name; o.dimensions=size
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True); o.data.materials.append(material)
    if bevel:
        mod=o.modifiers.new('Soft edges','BEVEL'); mod.width=bevel; mod.segments=2
        o.modifiers.new('Normals','WEIGHTED_NORMAL')
    if collision: colliders.append({'name':name,'center':list(loc),'size':list(size)})
    return o
def text(label,loc,size=.1,rotation=(math.pi/2,0,0), material=black):
    bpy.ops.object.text_add(location=loc,rotation=rotation); o=bpy.context.object; o.name='Label_'+label
    o.data.body=label; o.data.size=size; o.data.align_x='CENTER'; o.data.materials.append(material)
def table(name,x,y,w,d):
    box(name+'_top',(x,y,.76),(w,d,.09),dark,True,.02)
    for sx in [-1,1]:
        for sy in [-1,1]: box(name+'_leg',(x+sx*(w/2-.1),y+sy*(d/2-.1),.36),(.055,.055,.72),steel,True)
    for sy in [-1,1]: box(name+'_cloth',(x,y+sy*d/2,.49),(w,.025,.5),dark)
def chair(name,x,y):
    box(name+'_seat',(x,y,.47),(.48,.47,.1),black,True,.04)
    box(name+'_red_back',(x,y+.21,.85),(.51,.09,.65),red,True,.04)
    box(name+'_back',(x,y+.15,.85),(.46,.035,.57),black)
    box(name+'_stem',(x,y,.23),(.055,.055,.4),steel)
    box(name+'_base',(x,y,.06),(.55,.48,.08),black,True)
def tv(name,x,y,w=1.3):
    box(name+'_base',(x,y,.06),(.8,.52,.1),black,True,.03)
    box(name+'_stand',(x,y,.82),(.1,.1,1.55),steel,True)
    box(name+'_frame',(x,y,1.65),(w,.09,w*.58),black,True,.015)
    box(name+'_screen',(x,y-.05,1.65),(w-.055,.012,w*.58-.055),glass)

# Estimated 7.2 x 5.6 m workspace. Front and left are open for cutaway viewing.
box('Floor',(0,-1.5 if V2 else 0,-.09),(7.2,8.6 if V2 else 5.6,.18),wood,True)
box('Poster_wall',(0,2.8,1.5),(7.2,.12,3),white,True)
box('Right_wall',(3.6,-1.5 if V2 else 0,1.5),(.12,8.6 if V2 else 5.6,3),white,True)
box('Poster_backing',(0,2.72,1.85),(6.8,.045,1.55),steel)
for i in range(9):
    x=-3.1+i*.74
    box('Poster',(x,2.685,1.9),(.57,.015,.94),paper)
    for j in range(7): box('Poster_line',(x,2.674,2.17-j*.085),(.44,.008,.014),steel)
    box('Poster_blue_footer',(x,2.67,1.46),(.56,.012,.055),blue)
for i in range(7): box('Folding_screen',(.15+i*.46,1.28,1.05),(.45,.06,2.1),mauve,True,.008)
box('Lost_found_sign',(1.65,1.239,1.5),(.4,.015,.23),paper)
text('Lost and Found',(1.65,1.225,1.47),.055)
table('Central_worktable',-.65,-.15,1.25,2.75)
table('Rear_worktable',2.1,.65,1.75,.85)
box('Under_table_bin',(2.1,.65,.17),(.8,.58,.3),blue,True,.035)
box('Paper_bag',(-.65,.65,.98),(.32,.20,.38),tan,False,.015)
text('LAB',(-.65,.544,.98),.09,material=paper)
box('Black_bin',(.52,.7,.3),(.38,.4,.6),black,True,.05)
tv('Large_display',-2.35,1.75,1.5)
tv('Side_display',2.9,-1.05 if V2 else -.15,1.1)
table('Operator_desk',2.75,-1.75,1.35,.65)
box('Desktop_monitor',(2.75,-1.7,1.2),(.58,.05,.38),black)
chair('Operator_chair',2.4,-2.3); chair('Visitor_chair',.25,-1.85)
for i in range(3):
    box('Storage_crate',(-3+i*.55,2,.3),(.45,.5,.6),blue,True,.02)
for i in range(10): box('Stored_frame',(-1.5,2.38,.32+i*.07),(1.45,.25,.045),tan)
for i in range(3):
    y=-1.65+i*1.65
    box('Vent_housing',(3.48,y,2.65),(.18,1.4,.4),white)
    for j in range(12): box('Vent_slot',(3.377,y-.6+j*.11,2.65),(.012,.018,.29),black)
for x in [-2,0,2]:
    box('Floor_tape',(x,-2.65,.006),(.025,.25,.009),mauve)
if V2:
    # Video end reveals the aisle continuing beyond the foreground table.
    for i in range(2):
        y=-3.5-i*1.5
        table('Extended_wall_desk',2.9,y,1.1,1.25)
        chair('Extended_chair',2.0,y)
        box('Extended_monitor',(3.0,y+.25,1.05),(.45,.07,.32),black)
        box('Under_desk_computer',(3.0,y,.3),(.25,.4,.55),black,True)
    for i in range(5):
        box('White_operator_screen',(1.45+i*.4,-2.8,1.05),(.39,.055,2.1),paper,True,.006)
    box('Tape_long_aisle',(.7,-2.25,.006),(.05,5.4,.009),mauve)
    box('Tape_cross_aisle',(1.65,-.55,.007),(1.9,.05,.009),mauve)
    box('Red_bag',(-.88,.52,.93),(.21,.28,.3),red,False,.035)
    box('White_bag',(-.42,.54,.94),(.24,.22,.32),paper,False,.035)
    for i in range(3): box('Blue_foot_pedal',(2.55+i*.18,-1.42,.035),(.14,.19,.06),blue)
    table('Left_storage_bench',-2.7,-3.9,1.1,2.0)
    box('Floor_storage_bin',(-2.6,-5.1,.22),(.75,.48,.44),blue,True,.03)
    bpy.ops.object.light_add(type='AREA', location=(0,-4,4)); bpy.context.object.data.energy=500; bpy.context.object.data.size=5
for y in [-2,0,2]:
    bpy.ops.object.light_add(type='AREA', location=(0,y,4)); o=bpy.context.object; o.data.energy=450; o.data.shape='DISK'; o.data.size=5
bpy.context.scene.world.color=(.35,.35,.35)
def cam(name,eye,target,lens=35):
    bpy.ops.object.camera_add(location=eye); c=bpy.context.object; c.name=name
    c.rotation_euler=(Vector(target)-c.location).to_track_quat('-Z','Y').to_euler(); c.data.lens=lens; return c
overview=cam('Overview',(-10,-14,11) if V2 else (-8,-10,8),(0,-1.5,.7) if V2 else (0,0,.7),43)
inside=cam('Walkthrough',(-2.3,-2.6,1.65),(.7,.9,1.0),22)
aisle=cam('Aisle',(.9,-5.5,1.65),(1.0,.4,.9),24) if V2 else inside
sc=bpy.context.scene; sc.camera=overview; sc.render.engine='CYCLES'; sc.cycles.samples=16; sc.cycles.device='CPU'
sc.render.resolution_x=1200; sc.render.resolution_y=900; sc.render.resolution_percentage=100
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'lab_photo_reference.blend'))
bpy.ops.export_scene.gltf(filepath=str(OUT/'lab_photo_reference.glb'),export_format='GLB')
bpy.ops.wm.obj_export(filepath=str(OUT/'lab_photo_reference.obj'),export_materials=True)
bpy.ops.wm.usd_export(filepath=str(USD/'lab_visual.usdc'),root_prim_path='/World',export_materials=True)

# Separate analytic collision proxies for static furniture and architecture.
stage=Usd.Stage.CreateNew(str(USD/'lab_physics.usda'))
world=UsdGeom.Xform.Define(stage,'/World'); stage.SetDefaultPrim(world.GetPrim())
world.GetPrim().GetReferences().AddReference('lab_visual.usdc','/World')
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage,1)
physics=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene'); physics.CreateGravityDirectionAttr((0,0,-1)); physics.CreateGravityMagnitudeAttr(9.81)
for i,c in enumerate(colliders):
    p=UsdGeom.Cube.Define(stage,f'/World/Colliders/C{i:03d}'); p.CreateSizeAttr(1)
    p.AddTranslateOp().Set(Gf.Vec3d(*c['center'])); p.AddScaleOp().Set(Gf.Vec3f(*c['size']))
    p.CreateVisibilityAttr('invisible'); UsdPhysics.CollisionAPI.Apply(p.GetPrim())
stage.GetRootLayer().Save()
combined=Usd.Stage.CreateNew(str(USD/'lab_with_r1pro.usda'))
w=UsdGeom.Xform.Define(combined,'/World'); combined.SetDefaultPrim(w.GetPrim()); w.GetPrim().GetReferences().AddReference('lab_physics.usda','/World')
UsdGeom.SetStageUpAxis(combined,UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(combined,1)
robot=UsdGeom.Xform.Define(combined,'/World/R1Pro'); robot.GetPrim().GetReferences().AddReference('../../../isaacsim/galaxea_r1pro_scene.usda' if V2 else '../../isaacsim/galaxea_r1pro_scene.usda','/r1pro_urdf')
robot.GetPrim().GetAttribute('xformOp:translate').Set(Gf.Vec3d(2.65,-.15,.02) if V2 else Gf.Vec3d(1.6,-.55,.02))
combined.GetRootLayer().Save()
report={'type':'photo-guided approximate editable scene','measured_scale':False,'assumed_room_m':[7.2,5.6,3],'photos':5,'collision_boxes':len(colliders),'robot_reference_valid':combined.GetPrimAtPath('/World/R1Pro/base_link').IsValid(),'physics_runtime_tested':False,'limitations':['Unseen room boundaries estimated','People and loose cables omitted','Robot placement requires simulation validation']}
if V2:
    report.update(version=2,video_seconds=16.57,assumed_room_m=[7.2,8.6,3],note='Extended region is inferred from video, not metrically reconstructed')
(DEST/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for camera,name in [(overview,'overview'),(inside,'walkthrough')]+([(aisle,'aisle')] if V2 else []):
    sc.camera=camera; sc.render.filepath=str(OUT/(name+'.png')); bpy.ops.render.render(write_still=True)
print('LAB_BUILD_DONE',json.dumps(report))
