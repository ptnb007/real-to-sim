import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'real2sim_lab/deps')]
import numpy as np
import open3d as o3d
out=ROOT/'real2sim_lab/room_mesh'
mesh=o3d.io.read_triangle_mesh(str(out/'room_mesh.ply'))
verts=np.asarray(mesh.vertices); faces=np.asarray(mesh.triangles);colors=np.asarray(mesh.vertex_colors)
assert np.isfinite(verts).all() and faces.max()<len(verts) and faces.min()>=0
def tuples(a):return ',\n'.join('('+', '.join(f'{float(x):.7g}' for x in row)+')' for row in a)
usd='#usda 1.0\n(\n defaultPrim = "Room"\n upAxis = "Z"\n)\ndef Mesh "Room" (\n prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n)\n{\n'
usd+=' point3f[] points = ['+tuples(verts)+']\n'
usd+=' int[] faceVertexCounts = ['+','.join(['3']*len(faces))+']\n'
usd+=' int[] faceVertexIndices = ['+','.join(map(str,faces.ravel()))+']\n'
usd+=' uniform token subdivisionScheme = "none"\n color3f[] primvars:displayColor = ['+tuples(colors)+'] (\n interpolation = "vertex"\n)\n'
usd+=' bool physics:collisionEnabled = true\n uniform token physics:approximation = "none"\n}\n'
(out/'room_mesh.usda').write_text(usd,encoding='utf-8')
r={'source':'third_party/gs_playground/demo/navigation/nav_scene_1/3dgs/point_cloud.ply','method':'GS expected-depth and alpha>=0.95, TSDF fusion','views':108,'voxel_source_units':.025,'vertices':len(verts),'triangles':len(faces),'bounds':np.stack([verts.min(0),verts.max(0)]).tolist(),'watertight':mesh.is_watertight(),'edge_manifold':mesh.is_edge_manifold(),'notes':['Original GS geometry extraction; not screenshot-only inference or SAM3D output.','Static surface, object instances not separated.','Real-world scale and physics behavior unverified.']}
(out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
