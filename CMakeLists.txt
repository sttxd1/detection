cmake_minimum_required(VERSION 3.8)
project(detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
# find_package(PCL REQUIRED)
# find_package(pcl_conversions REQUIRED)


# # Include directories
# include_directories(
#   include
#   ${PCL_INCLUDE_DIRS}
# )

# # Declare a C++ executable
# add_executable(
#   point_cloud_processing_node src/human_detection_pcd.cpp
#   main_ground_based_people_detection src/main_ground_based_people_detection.cpp
# )

# # Specify libraries to link a library or executable target against
# target_link_libraries(
#   point_cloud_processing_node
#   main_ground_based_people_detection
#   ${PCL_LIBRARIES}
# )

# # Add dependencies to exported targets, like ROS messages or services
# ament_target_dependencies(
#   point_cloud_processing_node
#   rclcpp
#   sensor_msgs
#   pcl_conversions
# )

# ament_target_dependencies(
#   main_ground_based_people_detection
#   rclcpp
#   sensor_msgs
#   pcl_conversions
# )
# find_package(PCL REQUIRED)
# include_directories(${PCL_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})
# add_executable (ground_based_rgbd_people_detector src/main_ground_based_people_detection.cpp)
# target_link_libraries (ground_based_rgbd_people_detector ${PCL_LIBRARIES})

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# add_executable(oakd_detection src/oakd_detection.cpp)
# target_include_directories(oakd_detection PUBLIC
#   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   $<INSTALL_INTERFACE:include>)
# target_compile_features(oakd_detection PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

# install(TARGETS oakd_detection
#   DESTINATION lib/${PROJECT_NAME})

# Install targets
# install(TARGETS
#   point_cloud_processing_node
#   main_ground_based_people_detection
#   DESTINATION lib/${PROJECT_NAME}
# )

# Install Python executables
install(
  PROGRAMS
  src/oakd_yolo.py
  src/human_detection_pcd.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch config
  DESTINATION share/${PROJECT_NAME}
)



ament_package()