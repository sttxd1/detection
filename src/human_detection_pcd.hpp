#ifndef POINT_CLOUD_PROCESSING_NODE_HPP_
#define POINT_CLOUD_PROCESSING_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>

using sensor_msgs::msg::PointCloud2;
using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

class PointCloudProcessingNode : public rclcpp::Node
{
public:
    PointCloudProcessingNode();

private:
    void point_cloud_callback(const PointCloud2::SharedPtr msg);
    PointCloud ros_to_pcl(const PointCloud2::SharedPtr& ros_cloud);
    PointCloud2 pcl_to_ros(const PointCloud& pcl_cloud);
    PointCloud preprocess_point_cloud(const PointCloud& pcl_data);

    rclcpp::Subscription<PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<PointCloud2>::SharedPtr publisher_;
};

#endif // POINT_CLOUD_PROCESSING_NODE_HPP_
