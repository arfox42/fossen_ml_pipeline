#!/usr/bin/env python3
"""
analyze_dvl_bag.py
------------------
Extract and analyze /holoocean/DVLSensorVelocity (or /holoocean/VelocitySensor)
from a ROS2 Iron MCAP bag.

Computes:
- Timestamp statistics (start, end, rate)
- Mean and std of linear velocities
- Detects frame convention (by checking sign of steady-state Z)

Usage:
    python3 analyze_dvl_bag.py <path_to_ros2_bag> [topic_name]
"""

import sys
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import TwistWithCovarianceStamped

def analyze_dvl(bag_path: str, topic_name: str = "/holoocean/DVLSensorVelocity"):
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic_name not in topics:
        raise RuntimeError(f"Topic '{topic_name}' not found. Available topics:\n" +
                           "\n".join(topics.keys()))

    times = []
    lin_vels = []
    ang_vels = []

    print(f"[INFO] Reading {topic_name} from {bag_path} ...")

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != topic_name:
            continue

        msg = deserialize_message(data, TwistWithCovarianceStamped)
        times.append(t * 1e-9)

        lin_vels.append([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        ang_vels.append([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])

    times = np.array(times)
    lin_vels = np.array(lin_vels)
    ang_vels = np.array(ang_vels)

    if len(times) < 2:
        print("Not enough messages for analysis.")
        return

    dt = np.diff(times)
    mean_dt = np.mean(dt)
    hz = 1.0 / mean_dt if mean_dt > 0 else float("nan")

    print(f"\n=== DVL Topic: {topic_name} ===")
    print(f"Messages: {len(times)}")
    print(f"Start time: {times[0]:.3f} s, End time: {times[-1]:.3f} s")
    print(f"Mean Δt: {mean_dt*1000:.2f} ms (~{hz:.1f} Hz)")
    print(f"Δt std: {np.std(dt)*1000:.2f} ms\n")

    def stats(arr, label):
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        print(f"{label} mean [x y z]: {mean}")
        print(f"{label} std  [x y z]: {std}\n")

    stats(lin_vels, "Linear velocity (m/s)")
    stats(ang_vels, "Angular velocity (rad/s)")

    # Optional heuristic: detect vertical direction convention
    mean_z = np.mean(lin_vels[:, 2])
    if abs(mean_z) < 1e-3:
        frame_hint = "Undetermined (hovering or symmetric data)"
    elif mean_z > 0:
        frame_hint = "Likely NED (+Z down)"
    else:
        frame_hint = "Likely NWU (+Z up)"
    print(f"[CHECK] Vertical axis convention: {frame_hint}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_dvl_bag.py <path_to_ros2_bag> [topic_name]")
        sys.exit(1)

    bag_path = sys.argv[1]
    topic = sys.argv[2] if len(sys.argv) > 2 else "/holoocean/DVLSensorVelocity"
    analyze_dvl(bag_path, topic)
