#!/usr/bin/env python3
"""
analyze_wrench_bag.py
---------------------
Analyze /defender/test_runner_wrench_newton in a ROS 2 Iron MCAP bag.

Reports:
- Message count, start/end time
- Mean Δt, std Δt (Hz)
- Mean/std of forces [X Y Z] (N) and torques [K M N] (N·m)

Usage:
    python3 analyze_wrench_bag.py <path_to_ros2_bag> [topic_name]
"""

import sys
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import WrenchStamped

def analyze_wrench(bag_path: str, topic_name: str = "/defender/test_runner_wrench_newton"):
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
    forces = []
    torques = []

    print(f"[INFO] Reading {topic_name} from {bag_path} ...")

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != topic_name:
            continue

        msg = deserialize_message(data, WrenchStamped)
        times.append(t * 1e-9)
        forces.append([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        torques.append([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

    times = np.array(times)
    if len(times) < 2:
        print("[ERROR] Not enough wrench messages to analyze.")
        return
    forces = np.array(forces)
    torques = np.array(torques)

    dt = np.diff(times)
    mean_dt = dt.mean()
    hz = 1.0 / mean_dt if mean_dt > 0 else float("nan")

    print(f"\n=== Wrench Topic: {topic_name} ===")
    print(f"Messages: {len(times)}")
    print(f"Start time: {times[0]:.3f} s, End time: {times[-1]:.3f} s")
    print(f"Mean Δt: {mean_dt*1000:.2f} ms (~{hz:.1f} Hz)")
    print(f"Δt std: {dt.std()*1000:.2f} ms\n")

    def stats(arr, label):
        m, s = arr.mean(axis=0), arr.std(axis=0)
        print(f"{label} mean [X Y Z]: {m}")
        print(f"{label} std  [X Y Z]: {s}\n")

    stats(forces, "Forces (N)")
    stats(torques, "Torques (N·m)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_wrench_bag.py <path_to_ros2_bag> [topic_name]")
        sys.exit(1)

    bag = sys.argv[1].rstrip("/")
    topic = sys.argv[2] if len(sys.argv) > 2 else "/defender/test_runner_wrench_newton"
    analyze_wrench(bag, topic)
