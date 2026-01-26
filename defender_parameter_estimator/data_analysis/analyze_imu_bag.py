#!/usr/bin/env python3
"""
Extract and analyze /holoocean/IMUSensor messages from a ROS2 Iron MCAP bag.
Print timestamps, rates, and basic statistics.
"""

import sys
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu

def analyze_imu(bag_path: str, topic_name: str = "/holoocean/IMUSensor"):
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic_name not in topics:
        print(f"[ERROR] Topic '{topic_name}' not found in bag.")
        print("Available topics:")
        for t in topics.keys():
            print("  ", t)
        sys.exit(1)

    times = []
    lin_accels = []
    ang_vels = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != topic_name:
            continue

        msg = deserialize_message(data, Imu)
        times.append(t * 1e-9)  # convert ns to seconds
        lin_accels.append([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        ang_vels.append([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

    times = np.array(times)
    lin_accels = np.array(lin_accels)
    ang_vels = np.array(ang_vels)

    # === Timing stats ===
    if len(times) < 2:
        print("Not enough messages to analyze.")
        return

    dt = np.diff(times)
    mean_dt = np.mean(dt)
    hz = 1.0 / mean_dt if mean_dt > 0 else float('nan')

    print(f"\n=== IMU Topic: {topic_name} ===")
    print(f"Messages: {len(times)}")
    print(f"Start time: {times[0]:.3f} s, End time: {times[-1]:.3f} s")
    print(f"Mean Δt: {mean_dt*1000:.2f} ms  (~{hz:.1f} Hz)")
    print(f"Δt std: {np.std(dt)*1000:.2f} ms\n")

    # === Basic data stats ===
    def stats(arr, label):
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        print(f"{label} mean [x y z]: {mean}")
        print(f"{label} std  [x y z]: {std}\n")

    stats(lin_accels, "Linear accel (m/s²)")
    stats(ang_vels, "Angular vel (rad/s)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_imu_bag.py <path_to_ros2_bag>")
        sys.exit(1)

    analyze_imu(sys.argv[1])
