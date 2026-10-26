import bpy,json,math
from pathlib import Path
from mathutils import Vector
SRC=Path(r'C:\Users\woom\Documents\xwechat_files\wxid_3fg5lyf1apf612_e732\msg\file\2026-09\astra_lab_vertex_color.ply')
OUT=Path(__file__).resolve().parent/'real_ply';OUT.mkdir(exist_ok=True)
bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
bpy.ops.wm.ply_import(filepath=str(SRC));obj=bpy.context.object;obj.name='Captured_lab_mesh'
attrs=list(obj.data.color_attributes)
mat=bpy.data.materials.new('Original_vertex_colors');mat.use_nodes=True
nodes=mat.node_tree.nodes;nodes.clear();output=nodes.new('ShaderNodeOutputMaterial');emit=nodes.new('ShaderNodeEmission');attr=nodes.new('ShaderNodeVertexColor');attr.layer_name=attrs[0].name if attrs else ''
mat.node_tree.links.new(attr.outputs['Color'],emit.inputs['Color']);mat.node_tree.links.new(emit.outputs[0],output.inputs['Surface']);obj.data.materials.clear();obj.data.materials.append(mat)
points=[Vector(v) for v in obj.bound_box];lo=Vector([min(p[i] for p in points) for i in range(3)]);hi=Vector([max(p[i] for p in points) for i in range(3)]);center=(lo+hi)/2;radius=max(hi-lo)
report={'vertices':len(obj.data.vertices),'faces':len(obj.data.polygons),'color_attributes':[a.name for a in attrs],'bounds':[list(lo),list(hi)],'source':str(SRC),'scale_verified':False}
(OUT/'inspection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
sc=bpy.context.scene;sc.render.engine='CYCLES';sc.cycles.samples=8;sc.cycles.device='CPU';sc.render.resolution_x=1200;sc.render.resolution_y=900;sc.render.resolution_percentage=100
sc.view_settings.view_transform='Standard';sc.world.color=(.08,.08,.08)
for index,direction in enumerate([(-1,-1,.8),(1,1,.8),(0,-.25,1.6)]):
    bpy.ops.object.camera_add(location=center+Vector(direction)*radius*.8);c=bpy.context.object;c.rotation_euler=(center-c.location).to_track_quat('-Z','Y').to_euler();c.data.type='ORTHO';c.data.ortho_scale=radius*1.25;sc.camera=c
    sc.render.filepath=str(OUT/f'view_{index}.png');bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'original_color_mesh.blend'))
print(json.dumps(report))
