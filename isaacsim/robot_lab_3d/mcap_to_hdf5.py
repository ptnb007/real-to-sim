"""Convert the recorded grasp MCAP into one aligned HDF5 episode."""

import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party"))

import h5py
import numpy as np
from mcap.reader import make_reader

EPISODE = HERE / "datasets" / "grasp_episode_001"
MCAP = EPISODE / "grasp_episode_001.mcap"
HDF5 = EPISODE / "grasp_episode_001.hdf5"


def main() -> None:
    records = {}
    with MCAP.open("rb") as stream:
        reader = make_reader(stream)
        for schema, channel, message in reader.iter_messages():
            records.setdefault(channel.topic, []).append((message.log_time, message.data))

    required = ["/camera/rgb", "/camera/depth", "/lidar/points", "/robot/pose", "/odom", "/joint_states", "/clock"]
    missing = [topic for topic in required if not records.get(topic)]
    if missing:
        raise RuntimeError(f"MCAP is missing required topics: {missing}")

    clock = [(ts, json.loads(data)) for ts, data in records["/clock"]]
    master_timestamps = np.asarray([ts for ts, _ in clock], dtype=np.int64)
    count = len(master_timestamps)

    def exact(topic):
        values = {ts: data for ts, data in records[topic]}
        absent = [int(ts) for ts in master_timestamps if int(ts) not in values]
        if absent:
            raise RuntimeError(f"{topic} has {len(absent)} missing synchronized samples")
        return [values[int(ts)] for ts in master_timestamps]

    rgb = [np.load(io.BytesIO(data), allow_pickle=False) for data in exact("/camera/rgb")]
    depth = [np.load(io.BytesIO(data), allow_pickle=False) for data in exact("/camera/depth")]
    lidar = [np.load(io.BytesIO(data), allow_pickle=False) for data in exact("/lidar/points")]
    poses = [json.loads(data) for data in exact("/robot/pose")]
    odom = [json.loads(data) for data in exact("/odom")]
    joints = [json.loads(data) for data in exact("/joint_states")]

    lidar_total_points = int(sum(len(points) for points in lidar))
    lidar_nonempty_samples = int(sum(bool(len(points)) for points in lidar))
    if lidar_total_points < 1000 or lidar_nonempty_samples < int(0.9 * count):
        raise RuntimeError(
            f"LiDAR validation failed: {lidar_total_points} total points across "
            f"{lidar_nonempty_samples}/{count} non-empty synchronized samples"
        )

    with h5py.File(HDF5, "w") as handle:
        episode = handle.create_group("episodes/grasp_episode_001")
        episode.attrs["source_mcap"] = MCAP.name
        episode.attrs["timestamp_unit"] = "ns"
        episode.attrs["depth_unit"] = "m"
        episode.create_dataset("timestamps", data=master_timestamps, compression="gzip")
        episode.create_dataset("rgb", data=np.stack(rgb), compression="gzip", chunks=(1,) + rgb[0].shape)
        episode.create_dataset("depth", data=np.stack(depth), compression="gzip", chunks=(1,) + depth[0].shape)
        lidar_group = episode.create_group("lidar")
        lidar_group.create_dataset("offsets", data=np.cumsum([0] + [len(points) for points in lidar]), compression="gzip")
        lidar_group.create_dataset("points", data=np.concatenate(lidar, axis=0) if any(len(p) for p in lidar) else np.empty((0, 3)), compression="gzip")
        pose_group = episode.create_group("robot_pose")
        pose_group.create_dataset("position_m", data=np.asarray([p["position_m"] for p in poses]))
        pose_group.create_dataset("orientation_wxyz", data=np.asarray([p["orientation_wxyz"] for p in poses]))
        odom_group = episode.create_group("odometry")
        for key in ("position_m", "orientation_wxyz", "linear_velocity_mps", "angular_velocity_radps"):
            odom_group.create_dataset(key, data=np.asarray([item[key] for item in odom]))
        joint_group = episode.create_group("joint_state")
        joint_group.create_dataset("names", data=np.asarray(joints[0]["name"], dtype=h5py.string_dtype("utf-8")))
        for key in ("position", "velocity", "effort"):
            joint_group.create_dataset(key, data=np.asarray([item[key] for item in joints]), compression="gzip")

    report = {
        "status": "passed", "source_mcap": str(MCAP), "hdf5": str(HDF5), "samples": count,
        "topics": required, "timestamp_start_ns": int(master_timestamps[0]), "timestamp_end_ns": int(master_timestamps[-1]),
        "rgb_shape": [count] + list(rgb[0].shape), "depth_shape": [count] + list(depth[0].shape),
        "lidar_total_points": lidar_total_points, "lidar_nonempty_samples": lidar_nonempty_samples,
        "joint_count": len(joints[0]["name"]),
    }
    (EPISODE / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
