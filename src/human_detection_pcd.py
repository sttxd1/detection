import rclpy
from rclpy.node import Node

import sensor_msgs_py.point_cloud2 as pc2
import struct
import pcl
from sensor_msgs.msg import PointCloud2, PointField

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


class PointCloudProcessingNode(Node):
    def __init__(self):
        super().__init__('point_cloud_processing_node')
        qos_profile = QoSProfile(
                                    reliability=ReliabilityPolicy.BEST_EFFORT,
                                    durability=DurabilityPolicy.VOLATILE,
                                    history=HistoryPolicy.KEEP_LAST,
                                    depth=10  
                                )
        self.subscription = self.create_subscription(PointCloud2, '/stereo/points', self.point_cloud_callback, qos_profile)
        self.publisher = self.create_publisher(PointCloud2, '/processed_points', 10)

    def point_cloud_callback(self, msg):
        print("Getting pc data")
        # Convert ROS2 PointCloud2 to PCL data
        pcl_data = self.ros_to_pcl(msg)
        # Preprocess the point cloud
        processed_data = self.preprocess_point_cloud(pcl_data)
        # Convert back to ROS2 PointCloud2
        processed_msg = self.pcl_to_ros(processed_data)
        # Publish or use the processed data
        self.publisher.publish(processed_msg)

    def ros_to_pcl(self, ros_cloud):
        points_list = []

        for data in pc2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([data[0], data[1], data[2]])

        pcl_cloud = pcl.PointCloud()
        pcl_cloud.from_list(points_list)

        return pcl_cloud

    def pcl_to_ros(self, pcl_cloud):
        ros_cloud = PointCloud2()
        ros_cloud.header.stamp = self.get_clock().now().to_msg()
        ros_cloud.header.frame_id = 'oak-d_frame' 

        ros_cloud.height = 1  # Unordered point cloud
        ros_cloud.width = len(pcl_cloud.to_list())

        ros_cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        ros_cloud.is_bigendian = False
        ros_cloud.point_step = 12  # Float32 for each x, y, z
        ros_cloud.row_step = ros_cloud.point_step * ros_cloud.width
        ros_cloud.is_dense = True  # No invalid points

        buffer = []

        for point in pcl_cloud:
            buffer.extend(struct.pack('fff', point[0], point[1], point[2]))

        ros_cloud.data = bytearray(buffer)

        return ros_cloud


    def preprocess_point_cloud(self, pcl_data):
        vox = pcl_data.make_voxel_grid_filter()
        vox.set_leaf_size(0.01, 0.01, 0.01)  
        cloud_filtered = vox.filter()
        return cloud_filtered


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
