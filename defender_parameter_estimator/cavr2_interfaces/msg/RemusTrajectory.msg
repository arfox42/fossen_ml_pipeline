# Intended as a temporary trajectory msg format
# that is a modiication of the ROS trajectory_msgs/JointTrajectoryPoint.msg
# for the remus vehicle

#Right now it is 2 poses for each twist,accel,altitude,mode
#Consider revising in the future - possible errors

#Figure out a way to use PoseWithCovariance is deliberately
#chosen to have the pose with a designated positional uncertainty as a goal

std_msgs/Header header
geometry_msgs/PoseStamped[] poses
geometry_msgs/Twist[] twists
float64[] altitudes
string[] modes
#eventually include duration - see trajectory_msgs

