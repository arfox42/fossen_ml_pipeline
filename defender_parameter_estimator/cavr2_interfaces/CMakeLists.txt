cmake_minimum_required(VERSION 3.8)
project(cavr2_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
	"msg/CtrlCmd.msg"
        "msg/CtrlLL.msg"
        "msg/CtrlHL.msg"
        "msg/Heartbeat.msg"
        "msg/MissionControl.msg"
        "msg/MissionWaypoint.msg"
        "msg/MissionWaypoints.msg"
        "msg/Path.msg"
        "msg/RemusAdcp.msg"
        "msg/RemusAcommsRangerCcl.msg"
        "msg/RemusBattery.msg"
        "msg/RemusCtd.msg"
        "msg/RemusDepthOverride.msg"
        "msg/RemusDivePlaneCmd.msg"
        "msg/RemusFinControl.msg"
        "msg/RemusFluorometer.msg"
        "msg/RemusFullOverride.msg"
        "msg/RemusHover.msg"
        "msg/RemusModemMessage.msg"
        "msg/RemusNavError.msg"
        "msg/RemusNavFix.msg"
        "msg/RemusRedirect.msg"
        "msg/RemusRequest.msg"
        "msg/RemusReturnClosest.msg"
        "msg/RemusRoute.msg"
        "msg/RemusRudderCmd.msg"
        "msg/RemusRun.msg"
        "msg/RemusSetDepth.msg"
        "msg/RemusSetDestination.msg"
        "msg/RemusSetHeading.msg"
        "msg/RemusSetPosition.msg"
        "msg/RemusSetSpeed.msg"
	"msg/RemusState.msg"
        "msg/RemusStatus.msg"
        "msg/RemusStatusResponse.msg"
        "msg/RemusThrusterCmds.msg"
	"msg/RemusTime.msg"
        "msg/RemusTrajectory.msg"
        "msg/RemusUsbl.msg"
        "msg/RemusVersion.msg"
        "msg/RemusVersionRequest.msg"
        "msg/RemusXBodyControl.msg"
        "msg/TeledyneSetDepth.msg"
	"msg/DefenderPWMForces.msg"



  ###########################################################################
  # VideoRay DDS → ROS 2 IDL Definitions
  #
  # NOTE:
  #   ROS 2 requires that each IDL file contain exactly ONE struct.
  #   The VR AHRSRaw message depends on:
  #       - Quaternion.idl
  #       - VecXYZ.idl
  #       - VecRPY.idl
  #       - AHRSRaw.idl
  #
  #   These files define the DDS message types needed to subscribe
  #   to VideoRay's native DDS topics directly from ROS 2.
  ###########################################################################
        "msg/Quaternion.msg"
        "msg/VecXYZ.msg"
        "msg/VecRPY.msg"
        "msg/AHRSRaw.msg"
  ###########################################################################




	"action/PathSegments.action"
	"action/PathGoal.action"
	"action/MissionGoal.action"
	"action/MissionObjective.action"
	"srv/GetMissionStatus.srv"
	"srv/ModifyMissionStatus.srv"
	"srv/GetVehicleStatus.srv"
	"srv/SetVehicleStatus.srv"
	"srv/GetPlatformStatus.srv"
	"srv/SetPlatformStatus.srv"
	"srv/SetPlatformStatusBit.srv"
	"srv/SetPlatformStatusBitArray.srv"
	"srv/SetVehicleStatusBit.srv"
	"srv/SetVehicleStatusBitArray.srv"
	"srv/UnsetPlatformStatusBit.srv"
	"srv/UnsetPlatformStatusBitArray.srv"
	"srv/UnsetVehicleStatusBit.srv"
	"srv/UnsetVehicleStatusBitArray.srv"
	"srv/RunTest.srv"

        DEPENDENCIES geometry_msgs std_msgs nav_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()


