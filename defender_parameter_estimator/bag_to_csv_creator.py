#!/usr/bin/env python3
"""
bag_to_csv_creator.py
---------------------
Extracts IMU (for ν̇) and DVL (for ν) from a ROS 2 (Iron) MCAP bag.
Rotates IMU (body-NWU → body-NED) and combines both into one synchronized file: csv_full.

Header order:
time	u_dot	v_dot	w_dot	p_dot	q_dot	r_dot	u	v	w	p	q	r	x	y	z	phi	theta	psi	X	Y	Z	K	M	N	norm_dof	norm_value

Usage:
    python3 bag_to_csv_creator.py <path_to_ros2_bag>
"""

import sys
import os
import numpy as np
import math
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseStamped, WrenchStamped

# === NWU → NED rotation matrix (Fossen convention) ===
R_NWU_to_NED = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

def read_imu_ned(bag_path: str, topic_name: str = "/holoocean/IMUSensor"):
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    timestamps, accel_ned, gyro_ned = [], [], []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, Imu)
        t_sec = t * 1e-9
        timestamps.append(t_sec)

        accel_nwu = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        gyro_nwu = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        accel_ned.append(R_NWU_to_NED @ accel_nwu)
        gyro_ned.append(R_NWU_to_NED @ gyro_nwu)

    timestamps = np.array(timestamps)
    accel_ned = np.array(accel_ned)
    gyro_ned = np.array(gyro_ned)

    # Differentiate gyro → angular acceleration
    ang_accel = np.zeros_like(gyro_ned)
    if len(timestamps) > 2:
        dt = np.diff(timestamps)
        ang_accel[1:-1] = (gyro_ned[2:] - gyro_ned[:-2]) / (timestamps[2:] - timestamps[:-2])[:, None]
        ang_accel[0] = (gyro_ned[1] - gyro_ned[0]) / dt[0]
        ang_accel[-1] = (gyro_ned[-1] - gyro_ned[-2]) / dt[-1]

    return timestamps, accel_ned, gyro_ned, ang_accel


def read_dvl_ned(bag_path: str, topic_name: str = "/holoocean/DVLSensorVelocity"):
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    timestamps, vel_ned = [], []
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, TwistWithCovarianceStamped)
        t_sec = t * 1e-9
        timestamps.append(t_sec)
        vel_ned.append([
            msg.twist.twist.linear.x,  # surge (u)
            msg.twist.twist.linear.y,  # sway (v)
            msg.twist.twist.linear.z   # heave (w)
        ])

    return np.array(timestamps), np.array(vel_ned)

def read_pose_nwu(bag_path: str, topic_name: str = "/holoocean/PoseSensor"):
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    timestamps, positions, quats = [], [], []

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, PoseStamped)
        t_sec = t * 1e-9
        timestamps.append(t_sec)
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

    return np.array(timestamps), np.array(positions), np.array(quats)

def quat_to_euler_nwu(x, y, z, w):
    """Convert quaternion (NWU) to Euler angles (radians)."""
    # roll (phi)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    phi = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (theta)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        theta = math.copysign(math.pi / 2, sinp)
    else:
        theta = math.asin(sinp)

    # yaw (psi)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    psi = math.atan2(siny_cosp, cosy_cosp)

    return phi, theta, psi

def read_wrench_newton(bag_path: str, topic_name: str = "/defender/test_runner_wrench_newton"):
    """Read body-frame forces and torques (N, N·m) from the wrench topic."""
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    timestamps, forces, torques = [], [], []

    SAMPLE_DELAY = 4   # number of samples wrench leads physics
    DT = 0.03          # approximate sim timestep [s] (29.8 ms)

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != topic_name:
            continue

        msg = deserialize_message(data, WrenchStamped)
        t_sec = t * 1e-9

        # --- apply known delay correction (simulation → physics lag) ---
        t_sec += SAMPLE_DELAY * DT

        timestamps.append(t_sec)
        forces.append([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        torques.append([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

    if not timestamps:
        raise RuntimeError(f"No messages found on {topic_name}")

    return np.array(timestamps), np.hstack((np.array(forces), np.array(torques)))




def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bag_to_csv_creator.py <path_to_ros2_bag>")
        sys.exit(1)
    bag_path = sys.argv[1].rstrip("/")

    print(f"[INFO] Processing bag: {bag_path}")

    # === Read IMU (body NWU → NED) ===
    t_imu, accel_ned, gyro_ned, ang_accel = read_imu_ned(bag_path)

    # === Read DVL (body NED) ===
    t_dvl, vel_ned = read_dvl_ned(bag_path)

    # === Read PoseSensor (global NWU) ===
    t_pose, pos_nwu, quat_nwu = read_pose_nwu(bag_path)
    euler_nwu = np.array([quat_to_euler_nwu(*q) for q in quat_nwu])

    # === Read Wrench (body NED, 15.9 Hz) ===
    t_wrench, wrench_data = read_wrench_newton(bag_path)

    # Convert pose to NED
    pos_ned = pos_nwu.copy()
    pos_ned[:, 1] *= -1
    pos_ned[:, 2] *= -1
    euler_ned = euler_nwu.copy()
    euler_ned[:, 1] *= -1
    euler_ned[:, 2] *= -1

    # === Time alignment (use IMU as reference) ===
    if not (len(t_imu) == len(t_dvl) == len(t_pose)) or not (
        np.allclose(t_imu, t_dvl, atol=1e-3) and np.allclose(t_imu, t_pose, atol=1e-3)
    ):
        print("[WARN] Timestamps differ slightly — aligning all data to IMU timeline.")

        def align_to(ref_t, src_t, src_data):
            aligned = np.zeros((len(ref_t), src_data.shape[1]))
            for i, t in enumerate(ref_t):
                idx = np.argmin(np.abs(src_t - t))
                aligned[i] = src_data[idx]
            return aligned

        vel_ned = align_to(t_imu, t_dvl, vel_ned)
        pos_ned = align_to(t_imu, t_pose, pos_ned)
        euler_ned = align_to(t_imu, t_pose, euler_ned)
        quat_nwu = align_to(t_imu, t_pose, quat_nwu)

        # Zero-order hold for slower wrench stream
        wrench_aligned = np.zeros((len(t_imu), wrench_data.shape[1]))
        j = 0
        for i, t in enumerate(t_imu):
            while j + 1 < len(t_wrench) and t_wrench[j + 1] <= t:
                j += 1
            wrench_aligned[i] = wrench_data[j]
        t_common = t_imu
    else:
        # Always align (ZOH) to IMU length to avoid index mismatch
        wrench_aligned = np.zeros((len(t_imu), wrench_data.shape[1]))
        j = 0
        for i, t in enumerate(t_imu):
            while j + 1 < len(t_wrench) and t_wrench[j + 1] <= t:
                j += 1
            wrench_aligned[i] = wrench_data[j]
        t_common = t_imu

    # === Gravity compensation (inside loop per quaternion) ===
    g_world_ned = np.array([0.0, 0.0, -9.8])  # +Down in NED
    accel_corrected = np.zeros_like(accel_ned)

    for i in range(len(t_common)):
        qx, qy, qz, qw = quat_nwu[i]
        R_b2nwu = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)]
        ])
        R_b2ned = R_NWU_to_NED @ R_b2nwu  # body → world (NED)
        g_body_ned = R_b2ned.T @ g_world_ned  # gravity vector in body NED
        accel_corrected[i] = accel_ned[i] + g_body_ned  # ADD gravity

    # === Write output CSV ===
    out_csv = os.path.join(bag_path, "csv_full")
    header = (
        'time\t'
        'u_dot\tv_dot\tw_dot\tp_dot\tq_dot\tr_dot\t'
        'u\tv\tw\tp\tq\tr\t'
        'x\ty\tz\tphi\ttheta\tpsi\t'
        'X\tY\tZ\tK\tM\tN\t'
        'norm_dof\tnorm_value\n'
    )

    print(f"[INFO] Writing combined data to {out_csv}")

    with open(out_csv, "w") as f:
        f.write(header)
        for i in range(len(t_common)):
            f.write(
                f"{t_common[i]:.9f}\t"
                f"{accel_corrected[i,0]:.6f}\t{accel_corrected[i,1]:.6f}\t{accel_corrected[i,2]:.6f}\t"
                f"{ang_accel[i,0]:.6f}\t{ang_accel[i,1]:.6f}\t{ang_accel[i,2]:.6f}\t"
                f"{vel_ned[i,0]:.6f}\t{vel_ned[i,1]:.6f}\t{vel_ned[i,2]:.6f}\t"
                f"{gyro_ned[i,0]:.6f}\t{gyro_ned[i,1]:.6f}\t{gyro_ned[i,2]:.6f}\t"
                f"{pos_ned[i,0]:.6f}\t{pos_ned[i,1]:.6f}\t{pos_ned[i,2]:.6f}\t"
                f"{euler_ned[i,0]:.6f}\t{euler_ned[i,1]:.6f}\t{euler_ned[i,2]:.6f}\t"
                f"{wrench_aligned[i,0]:.6f}\t{wrench_aligned[i,1]:.6f}\t{wrench_aligned[i,2]:.6f}\t"
                f"{wrench_aligned[i,3]:.6f}\t{wrench_aligned[i,4]:.6f}\t{wrench_aligned[i,5]:.6f}\t"
                + "\t".join(["0.000000"] * 2) + "\n"
            )

    print(f"[DONE] Exported {len(t_common)} samples to csv_full")
    print(f"[CHECK] Mean corrected accel (NED): {np.mean(accel_corrected, axis=0)} m/s²")
    print(f"[CHECK] Mean velocity (NED): {np.mean(vel_ned, axis=0)} m/s")
    print(f"[CHECK] Mean position (NED): {np.mean(pos_ned, axis=0)} m")



if __name__ == "__main__":
    main()
