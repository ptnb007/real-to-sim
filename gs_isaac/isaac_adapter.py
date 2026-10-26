"""Single-scene Isaac Sim -> upstream GaussianRenderer bridge.

Assets must already use metres and the same Z-up world/link frames.
USD camera poses are OpenGL style (-Z forward, +Y up). Upstream y_up=True
flips the camera's Y/Z columns before converting camera-to-world to view.
"""
import bootstrap
import numpy as np
from gaussian_renderer import GSRenderer


class IsaacGaussianAdapter:
    def __init__(self, models, bodies):
        self.bodies = dict(bodies)
        self.renderer = GSRenderer(models)
        self.renderer.set_objects_mapping([
            (name, self.renderer.gaussian_start_indices[name], self.renderer.gaussian_end_indices[name])
            for name in self.bodies
        ])

    def update(self):
        if not self.bodies:
            return
        poses = [body.get_world_pose() for body in self.bodies.values()]
        self.renderer.update_gaussian_properties(
            np.asarray([p for p, q in poses], dtype=np.float32),
            np.asarray([q for p, q in poses], dtype=np.float32), scalar_first=True,
        )

    @staticmethod
    def camera_parameters(camera, width, height):
        from pxr import UsdGeom
        transform = np.asarray(UsdGeom.Xformable(camera.GetPrim()).ComputeLocalToWorldTransform(0)).T
        rotation = transform[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        fovy = np.degrees(2 * np.arctan(float(camera.GetVerticalApertureAttr().Get()) /
                                      (2 * float(camera.GetFocalLengthAttr().Get()))))
        expected_aspect = camera.GetHorizontalApertureAttr().Get() / camera.GetVerticalApertureAttr().Get()
        if not np.isclose(expected_aspect, width / height, atol=1e-4):
            raise ValueError('Camera aperture aspect must match image dimensions (square pixels).')
        return (
            transform[None, :3, 3], rotation.reshape(1, 9), height, width,
            np.array([fovy], dtype=np.float32),
        )

    def render_usd_camera(self, camera, width, height):
        return self.renderer.render_batch(*self.camera_parameters(camera, width, height), y_up=True)
