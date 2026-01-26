#An array of poses that represents a Path for a robot to follow
std_msgs/Header header
geometry_msgs/PoseStamped[] poses_leg1 # normally want to send 2 legs (for continuity of execution)
geometry_msgs/PoseStamped[] poses_leg2