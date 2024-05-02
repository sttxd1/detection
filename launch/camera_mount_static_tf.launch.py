from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    ld = LaunchDescription()

    # node1 = Node(package = "tf2_ros", 
    #             executable = "static_transform_publisher",
    #             arguments = ["-10.5784", "-7.97458", "0.0", "0", "0", "0.603978", "0.797001", "map", "tripod"], # Roll: 0, Pitch: 0, Yaw: 1.2969659 (74.31deg)
    #             output="screen"
    #             )

    # node2 = Node(package = "tf2_ros", 
    #             executable = "static_transform_publisher",
    #             arguments = ["0.05", "0.01", "1", "0", "0.1736482", "0", "0.9848078", "tripod", "oak-d-base-frame"], # Roll: 0, Pitch: 0.3490659 (20deg), Yaw: 0  
    #             output="screen"
    #             )

    node3 = Node(package = "tf2_ros", 
                executable = "static_transform_publisher",
                arguments = ["0.05", "0.01", "1.0", "0", "0.1736482", "0", "0.9848077", "map", "oak-d-base-frame"],
                output="screen"
                )



    ld.add_action(node3)
    # ld.add_action(node2)
    # ld.add_action(node3)

    return ld