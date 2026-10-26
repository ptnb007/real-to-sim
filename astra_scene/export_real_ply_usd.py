import bpy
from pathlib import Path
from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf
ROOT=Path(__file__).resolve().parent/'real_ply'
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'original_color_mesh.blend'))
obj=bpy.data.objects['Captured_lab_mesh']
stage=Usd.Stage.CreateNew(str(ROOT/'captured_lab.usdc'))
world=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(world.GetPrim())
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1)
mesh=UsdGeom.Mesh.Define(stage,'/World/CapturedLab')
mesh.CreatePointsAttr([tuple(v.co) for v in obj.data.vertices])
mesh.CreateFaceVertexCountsAttr([len(p.vertices) for p in obj.data.polygons])
mesh.CreateFaceVertexIndicesAttr([i for p in obj.data.polygons for i in p.vertices])
mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
color=obj.data.color_attributes[0]
colors=[Gf.Vec3f(*c.color[:3]) for c in color.data]
interpolation='vertex' if color.domain=='POINT' else 'faceVarying'
mesh.CreateDisplayColorPrimvar(interpolation).Set(colors)
material=UsdShade.Material.Define(stage,'/World/Materials/OriginalColors')
reader=UsdShade.Shader.Define(stage,'/World/Materials/OriginalColors/Reader');reader.CreateIdAttr('UsdPrimvarReader_float3');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('displayColor');reader.CreateOutput('result',Sdf.ValueTypeNames.Float3)
shader=UsdShade.Shader.Define(stage,'/World/Materials/OriginalColors/Surface');shader.CreateIdAttr('UsdPreviewSurface')
shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(),'result')
shader.CreateOutput('surface',Sdf.ValueTypeNames.Token);material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
stage.GetRootLayer().customLayerData={'cameraSettings':{'Perspective':{'position':Gf.Vec3d(-9,-10,7),'target':Gf.Vec3d(0,-1.5,1)}}}
stage.GetRootLayer().Save()
check=Usd.Stage.Open(str(ROOT/'captured_lab.usdc'))
assert len(UsdGeom.Mesh(check.GetPrimAtPath('/World/CapturedLab')).GetPointsAttr().Get())==149583
print('USD_VERIFIED',ROOT/'captured_lab.usdc',len(colors),interpolation)
