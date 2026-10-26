"""Synchronized raw-data recorder for the R1 Pro grasp episode.

The recorder writes the original simulation streams to MCAP. Array messages use
NumPy's self-describing NPY payload; state messages use JSON. The MCAP file is
the source of truth and is converted to HDF5 only after the run finishes.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np


class GraspDataRecorder:
    def __init__(self, output_root: Path, world, robot, stage, sample_every: int = 6):
        self.output_root = Path(output_root) / "datasets" / "grasp_episode_001"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.world = world
        self.robot = robot
        self.stage = stage
        self.sample_every = int(sample_every)
        self.frame = 0
        self.sample = 0
        self.dt = 1.0 / 60.0
        self.previous_position = None
        self.previous_orientation = None
        self.previous_time = None
        self.channels = {}
        self.schemas = {}
        self.camera = None
        self.lidar = None
        self.latest_lidar_points = None
        self.closed = False

        third_party = str(Path(__file__).resolve().parent / "third_party")
        if third_party not in sys.path:
            sys.path.append(third_party)
        from mcap.writer import CompressionType, Writer

        self.mcap_path = self.output_root / "grasp_episode_001.mcap"
        self.writer = Writer(str(self.mcap_path), compression=CompressionType.ZSTD)
        self.writer.start(profile="r1pro-grasp", library="isaac-sim-5.1-data-recorder")
        self._register_channels()

    def _register(self, topic: str, schema_name: str, encoding: str, schema: dict) -> None:
        schema_id = self.writer.register_schema(schema_name, "jsonschema", json.dumps(schema).encode("utf-8"))
        self.channels[topic] = self.writer.register_channel(topic, encoding, schema_id)

    def _register_channels(self) -> None:
        array_schema = {"type": "object", "description": "NumPy NPY array payload"}
        state_schema = {"type": "object"}
        for topic in ("/camera/rgb", "/camera/depth", "/lidar/points"):
            self._register(topic, topic.strip("/").replace("/", "_"), "application/x-npy", array_schema)
        for topic in ("/robot/pose", "/odom", "/joint_states", "/clock"):
            self._register(topic, topic.strip("/").replace("/", "_"), "json", state_schema)

    def initialize_sensors(self) -> None:
        import omni.replicator.core as rep
        from isaacsim.sensors.rtx import LidarRtx

        camera_prim = rep.create.camera(
            position=(1.65, -0.25, 1.35),
            look_at=(0.80, 0.35, 0.82),
            name="DatasetCamera",
        )
        self.render_product = rep.create.render_product(camera_prim, (320, 180))
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self.rgb_annotator.attach([self.render_product])
        self.depth_annotator.attach([self.render_product])

        self.lidar = LidarRtx(
            prim_path="/World/DatasetLidar",
            name="dataset_lidar",
            position=np.asarray([0.50, -0.02, 1.25]),
            orientation=np.asarray([1.0, 0.0, 0.0, 0.0]),
            config_file_name="Example_Rotary",
        )
        self.lidar.initialize()
        self.lidar.attach_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")

        metadata = {
            "episode": "grasp_episode_001",
            "simulation": "Isaac Sim 5.1",
            "physics_hz": 60,
            "sample_hz": 60 // self.sample_every,
            "topics": list(self.channels),
            "units": {"depth": "m", "position": "m", "joint_position": "rad_or_m", "timestamp": "ns"},
            "frames": {"world": "Isaac world", "robot_base": "Galaxea R1 Pro articulation root"},
        }
        self.writer.add_metadata("episode_manifest", {k: json.dumps(v) if not isinstance(v, str) else v for k, v in metadata.items()})

    @staticmethod
    def _npy_bytes(value) -> bytes:
        stream = io.BytesIO()
        np.save(stream, np.asarray(value), allow_pickle=False)
        return stream.getvalue()

    def _write(self, topic: str, timestamp_ns: int, payload: bytes) -> None:
        self.writer.add_message(self.channels[topic], timestamp_ns, payload, timestamp_ns)

    def _json(self, topic: str, timestamp_ns: int, value: dict) -> None:
        self._write(topic, timestamp_ns, json.dumps(value, separators=(",", ":")).encode("utf-8"))

    def step(self) -> None:
        self.frame += 1
        # The RTX LiDAR publishes one firing slice per rendered physics frame.
        # Read it at 60 Hz and combine six slices into each 10 Hz dataset scan.
        lidar_frame = self.lidar.get_current_frame()
        lidar_entry = lidar_frame.get("IsaacExtractRTXSensorPointCloudNoAccumulator", {})
        lidar_slice = np.asarray(lidar_entry.get("data", np.empty((0, 3))), dtype=np.float32)
        if lidar_slice.size:
            self.latest_lidar_points = lidar_slice.reshape((-1, 3)).copy()
        if self.frame % self.sample_every:
            return
        # The rotary sensor completes scans more slowly than the 10 Hz master
        # state clock. Start the episode after its first full scan, then use the
        # latest complete scan (standard sample-and-hold synchronization).
        if self.latest_lidar_points is None:
            return
        timestamp_s = self.frame * self.dt
        timestamp_ns = int(round(timestamp_s * 1_000_000_000))

        rgb = np.asarray(self.rgb_annotator.get_data())
        depth = np.asarray(self.depth_annotator.get_data(), dtype=np.float32)
        lidar_points = self.latest_lidar_points

        position, orientation = self.robot.get_world_pose()
        position = np.asarray(position, dtype=np.float64)
        orientation = np.asarray(orientation, dtype=np.float64)
        if self.previous_position is None:
            linear_velocity = np.zeros(3)
            angular_velocity = np.zeros(3)
        else:
            elapsed = max(timestamp_s - self.previous_time, 1e-9)
            linear_velocity = (position - self.previous_position) / elapsed
            angular_velocity = np.asarray(self.robot.get_angular_velocity(), dtype=np.float64)
        self.previous_position = position.copy()
        self.previous_orientation = orientation.copy()
        self.previous_time = timestamp_s

        joint_positions = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        joint_velocities = np.asarray(self.robot.get_joint_velocities(), dtype=np.float64)
        joint_efforts = np.asarray(self.robot.get_measured_joint_efforts(), dtype=np.float64)

        self._write("/camera/rgb", timestamp_ns, self._npy_bytes(rgb))
        self._write("/camera/depth", timestamp_ns, self._npy_bytes(depth))
        self._write("/lidar/points", timestamp_ns, self._npy_bytes(lidar_points))
        self._json("/robot/pose", timestamp_ns, {"position_m": position.tolist(), "orientation_wxyz": orientation.tolist()})
        self._json("/odom", timestamp_ns, {
            "position_m": position.tolist(), "orientation_wxyz": orientation.tolist(),
            "linear_velocity_mps": linear_velocity.tolist(), "angular_velocity_radps": angular_velocity.tolist(),
        })
        self._json("/joint_states", timestamp_ns, {
            "name": list(self.robot.dof_names), "position": joint_positions.tolist(),
            "velocity": joint_velocities.tolist(), "effort": joint_efforts.tolist(),
        })
        self._json("/clock", timestamp_ns, {"sim_time_s": timestamp_s, "frame": self.frame, "sample": self.sample})
        self.sample += 1

    def close(self) -> dict:
        if not self.closed:
            self.writer.finish()
            self.closed = True
        manifest = {
            "status": "recorded",
            "episode": "grasp_episode_001",
            "mcap": str(self.mcap_path),
            "samples": self.sample,
            "topics": list(self.channels),
            "sample_hz": 60 // self.sample_every,
        }
        (self.output_root / "raw_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
