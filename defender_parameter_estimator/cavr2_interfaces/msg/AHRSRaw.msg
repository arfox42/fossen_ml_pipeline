# Translation of vr::dds_types::AHRSRaw from DDS XML.
# This message contains raw data from the AHRS sensor, including flags
# to indicate which fields contain valid data.

# --- Data Validity Flags ---
bool has_quat      # Indicates if the 'quat' field contains valid data
bool has_euler     # Indicates if the 'euler' field contains valid data
bool has_lin_acc   # Indicates if the 'lin_acc' field contains valid data
bool has_ang_vel   # Indicates if the 'ang_vel' field contains valid data
bool has_mag_field # Indicates if the 'mag_field' field contains valid data
bool has_lin_vel   # Indicates if the 'lin_vel' field contains valid data

# --- Data Fields (using your newly created nested types) ---
Quaternion quat    # Orientation quaternion (W, X, Y, Z)
VecRPY euler       # Orientation Euler angles (Roll, Pitch, Yaw in radians)
VecXYZ lin_acc     # Linear acceleration
VecXYZ ang_vel     # Angular velocity
VecXYZ mag_field   # Magnetic field vector
VecXYZ lin_vel     # Linear velocity