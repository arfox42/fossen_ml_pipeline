#!/usr/bin/env python3
"""
check_imu_dvl_timestamps.py
---------------------------
Compares timestamps between /holoocean/IMUSensor and /holoocean/DVLSensorVelocity
in a ROS 2 Iron MCAP bag.

Reports:
- Start and end times
- Mean Δt for each
- Average time offset between nearest samples
- Whether timestamps match exactly (within tolerance)

Usage:
    python3 check_imu_dvl_timestamps.py <path_to_ros2_bag>
"""

import sys
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped

def read_timestamps(bag_path, topic, msg_type):
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    times = []
    while reader.has_next():
        tname, data, t = reader.read_next()
        if tname != topic:
            continue
        _ = deserialize_message(data, msg_type)
        times.append(t * 1e-9)
    return np.array(times)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 check_imu_dvl_timestamps.py <path_to_ros2_bag>")
        sys.exit(1)

    bag_path = sys.argv[1].rstrip("/")

    imu_topic = "/holoocean/IMUSensor"
    dvl_topic = "/holoocean/DVLSensorVelocity"

    print(f"[INFO] Reading timestamps for {imu_topic} and {dvl_topic} ...")

    t_imu = read_timestamps(bag_path, imu_topic, Imu)
    t_dvl = read_timestamps(bag_path, dvl_topic, TwistWithCovarianceStamped)

    print(f"\nIMU: {len(t_imu)} samples, {t_imu[0]:.3f}s → {t_imu[-1]:.3f}s")
    print(f"DVL: {len(t_dvl)} samples, {t_dvl[0]:.3f}s → {t_dvl[-1]:.3f}s")

    dt_imu = np.diff(t_imu).mean() if len(t_imu) > 1 else np.nan
    dt_dvl = np.diff(t_dvl).mean() if len(t_dvl) > 1 else np.nan
    print(f"Mean Δt IMU: {dt_imu*1000:.2f} ms  (~{1/dt_imu:.1f} Hz)")
    print(f"Mean Δt DVL: {dt_dvl*1000:.2f} ms  (~{1/dt_dvl:.1f} Hz)\n")

    # Align comparison
    t_min = max(t_imu[0], t_dvl[0])
    t_max = min(t_imu[-1], t_dvl[-1])
    imu_mask = (t_imu >= t_min) & (t_imu <= t_max)
    dvl_mask = (t_dvl >= t_min) & (t_dvl <= t_max)
    t_imu_sync = t_imu[imu_mask]
    t_dvl_sync = t_dvl[dvl_mask]

    diffs = []
    for t in t_imu_sync:
        idx = np.argmin(np.abs(t_dvl_sync - t))
        diffs.append(t_dvl_sync[idx] - t)
    diffs = np.array(diffs)

    print(f"Average |Δt| between nearest IMU–DVL sample: {np.mean(np.abs(diffs))*1000:.2f} ms")
    print(f"Std of |Δt|: {np.std(np.abs(diffs))*1000:.2f} ms")

    tol = 1e-3  # 1 ms tolerance
    matching = np.sum(np.abs(diffs) < tol)
    print(f"Samples matching within 1 ms: {matching}/{len(diffs)} "
          f"({100*matching/len(diffs):.1f}%)")

    if np.mean(np.abs(diffs)) < 0.001:
        print("[CHECK] IMU and DVL timestamps are synchronized (likely same sim clock).")
    else:
        print("[CHECK] IMU and DVL are NOT perfectly aligned; interpolation needed.")

if __name__ == "__main__":
    main()
