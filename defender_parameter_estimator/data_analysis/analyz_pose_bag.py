#!/usr/bin/env python3
"""
analyze_pose_bag.py
-------------------
Extracts /holoocean/PoseSensor from a ROS 2 Iron MCAP bag
and reports timing, rate, and basic pose statistics.

Checks:
- Message count, start/end time
- Mean Δt, std Δt (to get Hz)
- Mean and std of position (x, y, z)
- Mean and std of orientation (quaternion)
- Optional: prints orientation rate check (if we decide later to use for gravity compensation)

Usage:
    python3 analyze_pose_bag.py <path_to_ros2_bag>
"""

import sys
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped

def analyze_pose(bag_path: str, topic_name: str = "/holoocean/PoseSensor"):
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic_name not in topics:
        raise RuntimeError(f"Topic '{topic_name}' not found in bag. Available topics:\n" +
                           "\n".join(topics.keys()))

    times = []
    positions = []
    quats = []

    print(f"[INFO] Reading {topic_name} from {bag_path} ...")

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != topic_name:
            continue

        msg = deserialize_message(data, PoseStamped)
        t_sec = t * 1e-9
        times.append(t_sec)
        positions.append([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        quats.append([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])

    times = np.array(times)
    positions = np.array(positions)
    quats = np.array(quats)

    if len(times) < 2:
        print("Not enough PoseSensor messages to analyze.")
        return

    dt = np.diff(times)
    mean_dt = np.mean(dt)
    hz = 1.0 / mean_dt if mean_dt > 0 else float("nan")

    print(f"\n=== Pose Topic: {topic_name} ===")
    print(f"Messages: {len(times)}")
    print(f"Start time: {times[0]:.3f} s, End time: {times[-1]:.3f} s")
    print(f"Mean Δt: {mean_dt*1000:.2f} ms  (~{hz:.1f} Hz)")
    print(f"Δt std: {np.std(dt)*1000:.2f} ms\n")

    def stats(arr, label):
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        print(f"{label} mean [x y z]: {mean}")
        print(f"{label} std  [x y z]: {std}\n")

    stats(positions, "Position (m)")
    stats(quats[:, :3], "Orientation quaternion (x,y,z)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_pose_bag.py <path_to_ros2_bag> [topic_name]")
        sys.exit(1)

    bag_path = sys.argv[1].rstrip("/")
    topic = sys.argv[2] if len(sys.argv) > 2 else "/holoocean/PoseSensor"
    analyze_pose(bag_path, topic)
