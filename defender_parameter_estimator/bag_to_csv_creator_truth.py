#!/usr/bin/env python3
"""
bag_to_csv_truth_accel.py
-------------------------
Extracts truth ν̇ from /defender/accel (AccelStamped),
ν from DVL, pqr from IMU gyro, pose from PoseSensor, and wrench,
aligns them to the /defender/accel timeline, and writes a TSV-like CSV.

Header order:
time    u_dot v_dot w_dot p_dot q_dot r_dot
        u v w p q r
        x y z phi theta psi
        X Y Z K M N
        norm_dof norm_value
"""

import sys
import os
import math
import numpy as np

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message

from geometry_msgs.msg import AccelStamped, TwistWithCovarianceStamped, PoseStamped, WrenchStamped
from sensor_msgs.msg import Imu


# === NWU → NED rotation matrix (Fossen convention) ===
R_NWU_to_NED = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])


def _make_reader(bag_path: str):
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    return reader


def read_accel_truth_ned(bag_path: str, topic_name: str = "/defender/accel"):
    """
    Read truth accelerations (m/s², rad/s²) from the dynamics node.
    Assumed already in body-NED.
    """
    reader = _make_reader(bag_path)

    t, lin, ang = [], [], []
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, AccelStamped)
        t_sec = stamp * 1e-9
        t.append(t_sec)
        lin.append([msg.accel.linear.x, msg.accel.linear.y, msg.accel.linear.z])
        ang.append([msg.accel.angular.x, msg.accel.angular.y, msg.accel.angular.z])

    if not t:
        raise RuntimeError(f"No messages found on {topic_name}")

    return np.asarray(t), np.asarray(lin), np.asarray(ang)


def read_dvl_vel_ned(bag_path: str, topic_name: str = "/holoocean/DVLSensorVelocity", dvl_gain: float = 1.0):
    """
    Read DVL body velocities (assumed body-NED).
    Optionally apply a scalar gain correction.
    """
    reader = _make_reader(bag_path)

    t, vel = [], []
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, TwistWithCovarianceStamped)
        t_sec = stamp * 1e-9
        t.append(t_sec)
        vel.append([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])

    if not t:
        raise RuntimeError(f"No messages found on {topic_name}")

    t = np.asarray(t)
    vel = np.asarray(vel) * float(dvl_gain)
    return t, vel


def read_imu_gyro_ned(bag_path: str, topic_name: str = "/holoocean/IMUSensor"):
    """
    Read angular velocity from IMU and rotate body-NWU -> body-NED.
    Returns: t_imu, gyro_ned (Nx3) for [p q r]
    """
    reader = _make_reader(bag_path)

    t, gyro = [], []
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, Imu)
        t_sec = stamp * 1e-9
        t.append(t_sec)

        gyro_nwu = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=float)
        gyro.append(R_NWU_to_NED @ gyro_nwu)

    if not t:
        raise RuntimeError(f"No IMU messages found on {topic_name}")

    return np.asarray(t), np.asarray(gyro)


def read_pose_nwu(bag_path: str, topic_name: str = "/holoocean/PoseSensor"):
    """
    Read pose in world-NWU (HoloOcean). We’ll convert to world-NED after.
    """
    reader = _make_reader(bag_path)

    t, pos, quat = [], [], []
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, PoseStamped)
        t_sec = stamp * 1e-9
        t.append(t_sec)
        pos.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        quat.append([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

    if not t:
        raise RuntimeError(f"No Pose messages found on {topic_name}")

    return np.asarray(t), np.asarray(pos), np.asarray(quat)


def quat_to_euler_nwu(x, y, z, w):
    """
    Convert quaternion (NWU) → Euler angles (radians), 3-2-1 (roll, pitch, yaw).
    """
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    phi = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    theta = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    psi = math.atan2(siny_cosp, cosy_cosp)

    return phi, theta, psi


def read_wrench_newton(bag_path: str, topic_name: str = "/defender/test_runner_wrench_newton",
                       sample_delay: int = 0, dt: float = 0.03):
    """
    Read body wrench (N, N·m).
    If you still have a known sim timing lead/lag, you can shift the timestamps:
        t += sample_delay * dt
    """
    reader = _make_reader(bag_path)

    t, wrench = [], []
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, WrenchStamped)
        t_sec = stamp * 1e-9 + float(sample_delay) * float(dt)
        t.append(t_sec)
        wrench.append([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ])

    if not t:
        raise RuntimeError(f"No wrench messages found on {topic_name}")

    return np.asarray(t), np.asarray(wrench)


def align_nearest(ref_t: np.ndarray, src_t: np.ndarray, src_data: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbor alignment. Good for same-ish rate streams.
    """
    out = np.zeros((len(ref_t), src_data.shape[1]), dtype=float)
    for i, tr in enumerate(ref_t):
        idx = int(np.argmin(np.abs(src_t - tr)))
        out[i] = src_data[idx]
    return out


def align_zoh(ref_t: np.ndarray, src_t: np.ndarray, src_data: np.ndarray) -> np.ndarray:
    """
    Zero-order hold alignment. Good for slower streams (wrench).
    Assumes src_t is sorted.
    """
    out = np.zeros((len(ref_t), src_data.shape[1]), dtype=float)
    j = 0
    for i, tr in enumerate(ref_t):
        while j + 1 < len(src_t) and src_t[j + 1] <= tr:
            j += 1
        out[i] = src_data[j]
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 bag_to_csv_truth_accel.py <path_to_ros2_bag_folder>")
        sys.exit(1)

    bag_path = sys.argv[1].rstrip("/")
    if not os.path.isdir(bag_path):
        print(f"[ERROR] Bag folder not found: {bag_path}")
        sys.exit(1)
    if not os.path.isfile(os.path.join(bag_path, "metadata.yaml")):
        print(f"[ERROR] No metadata.yaml inside: {bag_path}")
        sys.exit(1)

    # ----------------------------
    # Topics / knobs
    # ----------------------------
    TOPIC_ACCEL = "/defender/accel"
    TOPIC_DVL   = "/holoocean/DVLSensorVelocity"
    TOPIC_IMU   = "/holoocean/IMUSensor"
    TOPIC_POSE  = "/holoocean/PoseSensor"
    TOPIC_WRENCH = "/defender/test_runner_wrench_newton"

    DVL_GAIN = 1.0            # if you want: ~2.13
    WRENCH_SAMPLE_DELAY = 0   # keep your old 4 if still needed
    WRENCH_DT = 0.03

    print(f"[INFO] Opening bag: {bag_path}")

    # ----------------------------
    # Read streams
    # ----------------------------
    t_accel, lin_acc, ang_acc = read_accel_truth_ned(bag_path, TOPIC_ACCEL)
    print(f"[INFO] /defender/accel samples: {len(t_accel)}")

    t_dvl, vel = read_dvl_vel_ned(bag_path, TOPIC_DVL, dvl_gain=DVL_GAIN)
    print(f"[INFO] DVL samples: {len(t_dvl)} (gain={DVL_GAIN})")

    t_imu, gyro = read_imu_gyro_ned(bag_path, TOPIC_IMU)
    print(f"[INFO] IMU gyro samples: {len(t_imu)}")

    t_pose, pos_nwu, quat_nwu = read_pose_nwu(bag_path, TOPIC_POSE)
    print(f"[INFO] Pose samples: {len(t_pose)}")

    t_wrench, wrench = read_wrench_newton(bag_path, TOPIC_WRENCH,
                                          sample_delay=WRENCH_SAMPLE_DELAY,
                                          dt=WRENCH_DT)
    print(f"[INFO] Wrench samples: {len(t_wrench)} (delay={WRENCH_SAMPLE_DELAY}*{WRENCH_DT}s)")

    # ----------------------------
    # Convert pose NWU -> NED
    # ----------------------------
    euler_nwu = np.asarray([quat_to_euler_nwu(*q) for q in quat_nwu])

    pos_ned = pos_nwu.copy()
    pos_ned[:, 1] *= -1
    pos_ned[:, 2] *= -1

    euler_ned = euler_nwu.copy()
    euler_ned[:, 1] *= -1
    euler_ned[:, 2] *= -1

    # ----------------------------
    # Align everything to accel timeline
    # ----------------------------
    ref_t = t_accel

    vel_aligned   = align_nearest(ref_t, t_dvl, vel)
    gyro_aligned  = align_nearest(ref_t, t_imu, gyro)
    pos_aligned   = align_nearest(ref_t, t_pose, pos_ned)
    euler_aligned = align_nearest(ref_t, t_pose, euler_ned)
    wrench_aligned = align_zoh(ref_t, t_wrench, wrench)

    # ----------------------------
    # Assemble output
    # ----------------------------
    out_csv = os.path.join(bag_path, "csv_full_truth")
    header = (
        "time\t"
        "u_dot\tv_dot\tw_dot\tp_dot\tq_dot\tr_dot\t"
        "u\tv\tw\tp\tq\tr\t"
        "x\ty\tz\tphi\ttheta\tpsi\t"
        "X\tY\tZ\tK\tM\tN\t"
        "norm_dof\tnorm_value\n"
    )

    print(f"[INFO] Writing → {out_csv}")

    with open(out_csv, "w") as f:
        f.write(header)
        for i in range(len(ref_t)):
            # nudot = [u_dot v_dot w_dot p_dot q_dot r_dot]
            nudot = np.hstack((lin_acc[i], ang_acc[i]))
            nu    = np.hstack((vel_aligned[i], gyro_aligned[i]))
            eta   = np.hstack((pos_aligned[i], euler_aligned[i]))
            tau   = wrench_aligned[i]

            f.write(
                f"{ref_t[i]:.9f}\t"
                f"{nudot[0]:.6f}\t{nudot[1]:.6f}\t{nudot[2]:.6f}\t{nudot[3]:.6f}\t{nudot[4]:.6f}\t{nudot[5]:.6f}\t"
                f"{nu[0]:.6f}\t{nu[1]:.6f}\t{nu[2]:.6f}\t{nu[3]:.6f}\t{nu[4]:.6f}\t{nu[5]:.6f}\t"
                f"{eta[0]:.6f}\t{eta[1]:.6f}\t{eta[2]:.6f}\t{eta[3]:.6f}\t{eta[4]:.6f}\t{eta[5]:.6f}\t"
                f"{tau[0]:.6f}\t{tau[1]:.6f}\t{tau[2]:.6f}\t{tau[3]:.6f}\t{tau[4]:.6f}\t{tau[5]:.6f}\t"
                f"0.000000\t0.000000\n"
            )

    print(f"[DONE] Wrote {len(ref_t)} rows to {out_csv}")
    print(f"[CHECK] mean nudot = {np.mean(np.hstack((lin_acc, ang_acc)), axis=0)}")
    print(f"[CHECK] mean nu    = {np.mean(np.hstack((vel_aligned, gyro_aligned)), axis=0)}")
    print(f"[CHECK] mean pos   = {np.mean(pos_aligned, axis=0)}")


if __name__ == "__main__":
    main()
