import bpy,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'real2sim_lab/room_mesh'
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.usd_import(filepath=str(out/'room_mesh.usda'))
meshes=[o for o in bpy.data.objects if o.type=='MESH'];assert len(meshes)==1
m=meshes[0].data
r=json.loads((out/'report.json').read_text());assert len(m.vertices)==r['vertices'] and len(m.polygons)==r['triangles']
r['usd_blender_import_verified']=True
(out/'report.json').write_text(json.dumps(r,indent=2));print('USD_VALIDATED',len(m.vertices),len(m.polygons))
