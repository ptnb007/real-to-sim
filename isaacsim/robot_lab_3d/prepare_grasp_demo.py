"""Create a non-destructive grasp-demo stage and gripper-patched R1 Pro USD.

The source URDF/USD are left untouched. The 50 mm finger travel comes from a
separate R1 Pro simulation model and is explicitly a demo compatibility patch;
it is not a factory calibration for this exact robot revision.
"""

from pathlib import Path
import re
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
ISAAC = HERE.parent
SOURCE_ROBOT = ISAAC / "galaxea_r1pro_scene.usda"
PATCHED_ROBOT = HERE / "galaxea_r1pro_gripper_demo.usda"
SCENE = HERE / "robot_lab_grasp_demo.usda"
SOURCE_URDF = ISAAC.parent / "third_party" / "galaxea_urdf" / "R1Pro" / "urdf_r1pro_g1z_2026" / "urdf" / "r1pro_2026.urdf"
PATCHED_URDF = HERE / "r1pro_grasp_demo.urdf"
LEFT_IK_CONFIG = HERE / "grasp_demo_robot_description.yaml"
RIGHT_IK_CONFIG = HERE / "grasp_demo_robot_description_right.yaml"
VISUAL_LAYER = HERE / "robot_visual_materials.usda"
DEMO_ARM_LIMITS = [(-1.5, 1.0), (-1.5, 0.5), (-1.5, 1.5), (-2.0, 0.5),
                   (-2.8, 0.8), (-1.0, 1.0), (-2.0, 2.0)]


def write_visual_materials() -> None:
    """Bind renderer-independent surface materials only to robot visuals."""
    palette = {
        "Silver": ((0.52, 0.56, 0.61), 0.72, 0.30),
        "Graphite": ((0.035, 0.042, 0.052), 0.30, 0.40),
        "Rubber": ((0.012, 0.015, 0.019), 0.0, 0.75),
        "Sensor": ((0.018, 0.025, 0.033), 0.18, 0.20),
    }
    chunks = ['#usda 1.0\n\nover "World"\n{\n    def Scope "RobotRenderMaterials"\n    {']
    for name, (color, metal, rough) in palette.items():
        chunks.append(f'''        def Material "{name}"
        {{
            token outputs:surface.connect = </World/RobotRenderMaterials/{name}/Surface.outputs:surface>
            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = {color}
                float inputs:metallic = {metal}
                float inputs:roughness = {rough}
                token outputs:surface
            }}
        }}''')
    chunks.append('    }\n    over "GalaxeaR1Pro"\n    {\n        over "Robot"\n        {')
    for link in ET.parse(PATCHED_URDF).getroot().findall("link"):
        name = link.attrib["name"]
        material = "Silver"
        if "wheel" in name:
            material = "Rubber"
        elif any(part in name for part in ("camera", "head", "lidar", "gmsl", "d405", "imu")):
            material = "Sensor"
        elif "gripper_link" in name or "steer" in name or name in ("torso_link1", "torso_link3"):
            material = "Graphite"
        elif re.search(r"arm_link[2467]$", name):
            material = "Graphite"
        chunks.append(f'''            over "{name}"
            {{
                over "visuals" (
                    prepend apiSchemas = ["MaterialBindingAPI"]
                )
                {{
                    rel material:binding = </World/RobotRenderMaterials/{material}> (
                        bindMaterialAs = "strongerThanDescendants"
                    )
                }}
            }}''')
    chunks.append('        }\n    }\n}\n')
    VISUAL_LAYER.write_text("\n".join(chunks), encoding="utf-8")


def write_robot_patch_layer() -> None:
    if not SOURCE_ROBOT.is_file():
        raise FileNotFoundError(SOURCE_ROBOT)
    overrides = []
    for side in ("left", "right"):
        for finger, low, high in ((1, 0.0, 0.05), (2, -0.05, 0.0)):
            name = f"{side}_gripper_finger_joint{finger}"
            overrides.append(
                f'''            over "{name}"
            {{
                float physics:lowerLimit = {low:.6f}
                float physics:upperLimit = {high:.6f}
                float physxJoint:maxJointVelocity = 0.200000
                float drive:linear:physics:maxForce = 40.000000
                float drive:linear:physics:damping = 20.000000
            }}'''
            )
    for name in [f"torso_joint{i}" for i in range(1, 5)] + [
        f"{side}_arm_joint{i}" for side in ("left", "right") for i in range(1, 8)
    ]:
        overrides.append(
            f'''            over "{name}"
            {{
                float drive:angular:physics:damping = 45.000000
            }}'''
        )
    PATCHED_ROBOT.write_text(
        '''#usda 1.0
(
    defaultPrim = "r1pro_urdf"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "Non-destructive simulation compatibility layer; 50 mm finger travel is a demo assumption cross-checked against a separate R1 Pro simulation model."
)

def Xform "r1pro_urdf" (
    prepend references = @../galaxea_r1pro_scene.usda@</r1pro_urdf>
)
{
    over "joints"
    {
'''
        + "\n".join(overrides)
        + '''
    }
}
''',
        encoding="utf-8",
    )


def write_urdf_ik_copy() -> None:
    """Give Lula valid demo-only gripper limits; keep the source URDF intact."""
    if not SOURCE_URDF.is_file():
        raise FileNotFoundError(SOURCE_URDF)
    source = SOURCE_URDF.read_text(encoding="utf-8")
    for side in ("left", "right"):
        for finger, low, high in ((1, 0.0, 0.05), (2, -0.05, 0.0)):
            joint = f"{side}_gripper_finger_joint{finger}"
            pattern = re.compile(
                rf'(<joint\s+name="{re.escape(joint)}"\s+type="prismatic">.*?<limit\s+)'
                rf'lower="[^"]*"\s+upper="[^"]*"\s+effort="[^"]*"\s+velocity="[^"]*"(\s*/>)',
                re.DOTALL,
            )
            source, count = pattern.subn(
                rf'\g<1>lower="{low:g}" upper="{high:g}" effort="40" velocity="0.2"\g<2>',
                source,
                count=1,
            )
            if count != 1:
                raise RuntimeError(f"Could not patch the demo URDF limits for {joint}")
    PATCHED_URDF.write_text(source, encoding="utf-8")
    tree = ET.parse(PATCHED_URDF)
    for joint in tree.getroot().findall("joint"):
        name = joint.attrib["name"]
        if name.startswith("right_arm_joint"):
            lo, hi = DEMO_ARM_LIMITS[int(name.rsplit("joint", 1)[1]) - 1]
            limit = joint.find("limit")
            limit.set("lower", str(max(lo, float(limit.attrib["lower"]))))
            limit.set("upper", str(min(hi, float(limit.attrib["upper"]))))
            limit.set("velocity", "0.6")
    tree.write(PATCHED_URDF, encoding="utf-8", xml_declaration=True)


def write_right_ik_config() -> None:
    """Use the arm plus live torso joints so Lula tracks the moving shoulder mount."""
    active = [f"right_arm_joint{i}" for i in range(1, 8)]
    rules = [
        f"  - {{name: {name}, rule: fixed, value: {dict(zip([f'torso_joint{i}' for i in range(1, 5)], [0.5, -0.8, -0.3, 0.0])).get(name, 0.0)}}}"
        for name in [f"torso_joint{i}" for i in range(1, 5)] + [f"steer_motor_joint{i}" for i in range(1, 4)]
        + [f"wheel_motor_joint{i}" for i in range(1, 4)]
        + [f"left_arm_joint{i}" for i in range(1, 8)]
        + [f"right_gripper_finger_joint{i}" for i in (1, 2)]
    ]
    content = [
        "api_version: 1.0",
        "root_link: base_link",
        "cspace:",
        *[f"  - {name}" for name in active],
        "default_q: [-0.5, -0.46, 0.65, -1.23, -2.27, -0.37, -1.44]",
        f"acceleration_limits: [{', '.join('2.0' for _ in active)}]",
        f"jerk_limits: [{', '.join('80.0' for _ in active)}]",
        "cspace_to_urdf_rules:",
        *rules,
    ]
    RIGHT_IK_CONFIG.write_text("\n".join(content) + "\n", encoding="utf-8")


def main() -> None:
    write_robot_patch_layer()
    write_urdf_ik_copy()
    write_right_ik_config()
    write_visual_materials()

    SCENE.write_text(
        '''#usda 1.0
(
    defaultPrim = "World"
    subLayers = [@robot_visual_materials.usda@]
    metersPerUnit = 1
    upAxis = "Z"
    doc = "Approximate photo-referenced lab plus an explicitly assumed can and R1 Pro grasp-demo setup."
)

def Xform "World"
{
    def SphereLight "RobotSoftKey"
    {
        float inputs:intensity = 1800
        float inputs:radius = 0.35
        bool inputs:normalize = true
        color3f inputs:color = (1, 0.97, 0.93)
        double3 xformOp:translate = (1.3, -0.6, 2.2)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def SphereLight "RobotSoftFill"
    {
        float inputs:intensity = 650
        float inputs:radius = 0.5
        bool inputs:normalize = true
        color3f inputs:color = (0.90, 0.95, 1.0)
        double3 xformOp:translate = (-0.6, -0.3, 1.8)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
        def "Physics_Materials"
        {
            def Material "grasp_target_material" (
                prepend apiSchemas = ["PhysicsMaterialAPI"]
            )
            {
                float physics:dynamicFriction = 0.8
                float physics:restitution = 0.0
                float physics:staticFriction = 1.0
                uniform token physxMaterial:frictionCombineMode = "max"
                bool physxMaterial:improvePatchFriction = true
            }
        }

    def Xform "Environment" (
        prepend references = @robot_lab.usda@</World>
    )
    {
        # Keep the measured/photogrammetric room reconstruction fixed.
        double3 xformOp:translate = (0.300000, -0.480000, 0.000000)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        over "brown_part"
        {
            token visibility = "invisible"
        }

        # The tabletop is also represented by the dedicated thin support
        # proxy below. Disable only the coincident triangle tabletop collider
        # to avoid duplicate coplanar PhysX contacts; keep it visible.
        over "rear_table"
        {
            over "rear_table__3_collider"
            {
                bool physics:collisionEnabled = false
            }
        }
    }

    # Stable, hidden collision proxy exactly over the reconstructed tabletop.
    # The photo mesh remains untouched; this thin box only supplies a robust
    # static PhysX support surface for the dynamic grasp target.
    def Cube "GraspTableSupport" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        double3 xformOp:translate = (0.770000, 0.745000, 0.719000)
        double3 xformOp:scale = (2.370000, 0.890000, 0.062000)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        token visibility = "invisible"
        bool physics:collisionEnabled = true
    }

    def Xform "DemoCan" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        custom string reconstruction:note = "Proxy for the small brown cylinder already reconstructed on the rear tabletop; dimensions copied from its mesh bounds, mass is an illustrative assumption."
        # Move the target toward the table's front edge, within the robot's
        # reachable workspace, rather than leaving it deep on the tabletop.
        double3 xformOp:translate = (0.800000, 0.350000, 0.770000)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 0.035000
        bool physics:kinematicEnabled = false
        float physics:linearDamping = 0.05
        float physics:angularDamping = 0.05

        def Cylinder "Can" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]
        )
        {
            rel material:binding:physics = </World/Physics_Materials/grasp_target_material> (
                bindMaterialAs = "strongerThanDescendants"
            )
            uniform token axis = "Z"
            double radius = 0.025
            double height = 0.032
            bool physics:collisionEnabled = true
            float3[] extent = [(-0.025, -0.025, -0.016), (0.025, 0.025, 0.016)]
            color3f[] primvars:displayColor = [(0.68, 0.70, 0.72)] (
                interpolation = "constant"
            )
        }
    }

    def Xform "GalaxeaR1Pro"
    {
        custom string reconstruction:placementNote = "Demo placement is calibrated from the physical USD: chassis heading and right-arm reach face world +Y toward the table."
        # Keep the chassis fully in the aisle in front of the table. The
        # tabletop support begins near Y=0.245; the base center stays behind it.
        double3 xformOp:translate = (0.500000, -0.020000, 0.010000)
        float3 xformOp:rotateXYZ = (0, 0, 90)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]

        def Xform "Robot" (
            prepend references = @fixed_base_r1pro.usda@</r1pro_urdf>
        )
        {
            # Bind a matching high-friction PhysX material to the actual
            # contact fingers, not merely to the dynamic target.
            over "right_gripper_finger_link1" (
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {
                rel material:binding:physics = </World/Physics_Materials/grasp_target_material> (
                    bindMaterialAs = "strongerThanDescendants"
                )
            }
            over "right_gripper_finger_link2" (
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {
                rel material:binding:physics = </World/Physics_Materials/grasp_target_material> (
                    bindMaterialAs = "strongerThanDescendants"
                )
            }
        }
    }
}
''',
        encoding="utf-8",
    )
    print(f"Wrote {PATCHED_ROBOT}")
    print(f"Wrote {PATCHED_URDF}")
    print(f"Wrote {SCENE}")


if __name__ == "__main__":
    main()
