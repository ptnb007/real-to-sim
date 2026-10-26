from pathlib import Path
from pxr import Usd, UsdGeom, UsdLux, Gf
folder=Path(__file__).resolve().parent/'focus_v4'
stage=Usd.Stage.CreateNew(str(folder/'focused_zone_isaac.usda'))
stage.GetRootLayer().subLayerPaths=['focused_zone.usda']
stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1)
for p in stage.Traverse():
    if 'Light' in p.GetTypeName():
        UsdLux.LightAPI(p).CreateIntensityAttr(0)
    if p.IsA(UsdGeom.Mesh):UsdGeom.Mesh(p).CreateDoubleSidedAttr(True)
dome=UsdLux.DomeLight.Define(stage,'/World/IsaacLighting/Environment')
dome.CreateIntensityAttr(650);dome.CreateColorAttr(Gf.Vec3f(1,.97,.94))
sun=UsdLux.DistantLight.Define(stage,'/World/IsaacLighting/SoftKey')
sun.CreateIntensityAttr(1800);sun.CreateAngleAttr(12)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-25,-30,0))
eye=Gf.Vec3d(-6,-5,5.7);target=Gf.Vec3d(0,0,.75)
camera=UsdGeom.Camera.Define(stage,'/World/ReviewCamera')
camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye,target,Gf.Vec3d(0,0,1)).GetInverse())
camera.CreateClippingRangeAttr(Gf.Vec2f(.05,100));camera.CreateFocalLengthAttr(43)
stage.GetRootLayer().customLayerData={'cameraSettings':{'Perspective':{'position':eye,'target':target},'boundCamera':'/World/ReviewCamera'}}
stage.GetRootLayer().Save()
check=Usd.Stage.Open(str(folder/'focused_zone_isaac.usda'))
assert check.GetPrimAtPath('/World/ReviewCamera').IsValid()
assert UsdLux.DomeLight(check.GetPrimAtPath('/World/IsaacLighting/Environment')).GetIntensityAttr().Get()==650
print('Saved lighting and camera fix:',folder/'focused_zone_isaac.usda')
