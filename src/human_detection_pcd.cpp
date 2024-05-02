#include "human_detection_pcd.hpp"

PointCloudProcessingNode::PointCloudProcessingNode() : Node("point_cloud_processing_node")
{
    auto qos_profile = rclcpp::QoS(rclcpp::QoSInitialization(
                                        RMW_QOS_POLICY_HISTORY_KEEP_LAST, 10),
                                    rmw_qos_profile_sensor_data);
    subscription_ = this->create_subscription<PointCloud2>(
                        "/stereo/points",
                        qos_profile,
                        std::bind(&PointCloudProcessingNode::point_cloud_callback, this, std::placeholders::_1));
    publisher_ = this->create_publisher<PointCloud2>("/processed_points", 10);
}

void PointCloudProcessingNode::point_cloud_callback(const PointCloud2::SharedPtr msg)
{
    RCLCPP_INFO(this->get_logger(), "Getting pc data");
    auto pcl_data = ros_to_pcl(msg);
    auto processed_data = preprocess_point_cloud(pcl_data);
    auto processed_msg = pcl_to_ros(processed_data);
    publisher_->publish(processed_msg);
}

PointCloud PointCloudProcessingNode::ros_to_pcl(const PointCloud2::SharedPtr& ros_cloud)
{
    PointCloud pcl_cloud;
    pcl_conversions::fromROSMsg(*ros_cloud, pcl_cloud);
    return pcl_cloud;
}

PointCloud2 PointCloudProcessingNode::pcl_to_ros(const PointCloud& pcl_cloud)
{
    PointCloud2 ros_cloud;
    pcl_conversions::toROSMsg(pcl_cloud, ros_cloud);
    ros_cloud.header.stamp = this->get_clock()->now();
    ros_cloud.header.frame_id = "oak-d_frame";
    return ros_cloud;
}

PointCloud PointCloudProcessingNode::preprocess_point_cloud(const PointCloud& pcl_data)
{
    PointCloud cloud_filtered;
    pcl::VoxelGrid<pcl::PointXYZ> vox;
    vox.setInputCloud(pcl_data.makeShared());
    vox.setLeafSize(0.01f, 0.01f, 0.01f);
    vox.filter(cloud_filtered);
    return cloud_filtered;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudProcessingNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
