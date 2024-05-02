#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2 

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from tf2_ros import Buffer, TransformListener
# from tf2_ros.transformations import quaternion_from_euler
from geometry_msgs.msg import TransformStamped#, PointStamped
from tf2_geometry_msgs import PointStamped
import numpy as np


class YoloObjectDetection(Node):
    def __init__(self):
        super().__init__('yolo_object_detection')
        self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw', self.yolo_rgb_callback, 10)
        # self.stereo_sub = self.create_subscription(Image, '/oak/stereo/image_raw', self.stereo_callback, 10)
        #self.stereo_sub = self.create_subscription(Image, '/stereo/converted_depth', self.stereo_callback, 10)
        self.yolo_pub = self.create_publisher(Image, 'yolo_img', 10)
        self.publisher_ = self.create_publisher(Marker, 'visualization_marker', 10)
        self.cv_bridge = CvBridge()
        self.yolo = YOLO('yolov8n.pt').to('cuda')
        self.current_frame_yolo_results = None
        self.detected_human_img_coords = None
        self.detected_car_img_coords = None
        # self.marker_pub_class = MarkerPublisher)

        # Initialize TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.listener = TransformListener(self.tf_buffer, self)

        # object coordinates in sensor frame
        self.object_point = PointStamped()

        # Timer to periodically try the transformation
        #self.timer = self.create_timer(1.0, self.transform_object)
        self.rgb_intrinsic_mat = np.array([[1007.03765,    0.,  693.05655],
                                        [0.     , 1007.59267,  356.9163],
                                        [0.     ,    0.     ,    1.]
                                        ])
        
    def publish_marker(self, point_msg):
        # point_msg.point.x 
        # point_msg.point.y 
        # point_msg.point.z

        marker = Marker()
        marker.header.frame_id = "oak-d-base-frame" #"map"
        marker.type = marker.SPHERE
        marker.action = marker.ADD

        # Set the pose of the marker. This is a full 6DOF pose relative to the frame/time specified in the header
        marker.pose.position.x = point_msg.point.x  #1.0  
        marker.pose.position.y = point_msg.point.y #2.0  
        marker.pose.position.z = point_msg.point.z #location[2] #3.0  

        print("Drawing visualization with coordinates: ", (point_msg.point.x, point_msg.point.y, point_msg.point.z))

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set the scale of the marker -- 1x1x1 here means 1m on a side
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        # Set the color 
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.publisher_.publish(marker)

    def transform_object(self):
        try:
            # Check if the transformation is available
            now = rclpy.time.Time()
            self.tf_buffer.can_transform('map', self.object_point.header.frame_id, now)

            # Transform the object point to the map frame
            object_in_map = self.tf_buffer.transform(self.object_point, 'map', timeout=rclpy.duration.Duration(seconds=1))
            self.get_logger().info(f"Object in map frame: {object_in_map.point.x}, {object_in_map.point.y}, {object_in_map.point.z}")
            return object_in_map
            #return self.object_point

        except Exception as e:
            self.get_logger().info(f"Could not transform object: {str(e)}")

    def pixel_to_world(self, x, y, depth):
        # Inverse of the camera matrix
        inv_camera_matrix = np.linalg.inv(self.rgb_intrinsic_mat)

        # Homogeneous coordinates of the pixel
        pixel_homog = np.array([x, y, 1])

        # Convert to normalized camera coordinates
        normalized_camera_coord = inv_camera_matrix.dot(pixel_homog)

        # Scale with the depth
        world_coord = normalized_camera_coord * depth

        # Use TF to transform to map frame based on camera location
        return world_coord

    def yolo_rgb_callback(self, msg):
        # cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
        #print("Shape of rgb image: ", cv_image.shape)
        #print("Type of rgb image: ", cv_image.dtype)
        #results = self.yolo(cv_image)

        # the [0] is necessary because for some reason it's a list of of one list with the below results [[boxes, keypoints, ...]]
        results = self.yolo.predict(source=cv_image, show=False,device='cuda')[0] # list of results [boxes, keypoints, masks, names, orig_img, orig_shape, path, probs, save_dir, speed]
        self.current_frame_yolo_results = results

        # boxes = results.boxes  # Boxes object for bbox outputs
        # img_shape = results.orig_shape
        # masks = results.masks  # Masks object for segmentation masks outputs
        # keypoints = results.keypoints  # Keypoints object for pose outputs
        # probs = results.probs  # Probs object for classification outputs

        # iterate through all the detection boxes in the current frame
        for box in results.boxes.cpu().numpy():
            object_id = int(box.cls[0])
            object_confidence = box.conf

            # cls = int(box.cls[0])
            # print("cls is: ", (cls, object_id))
            # If label id is human and confidence is above some threshold
            if object_id == 0 and object_confidence >= 0.60:
                x1, y1, x2, y2 = box.xyxy[0] # x1, y1 are top left of bounding box, width, height
                #print("xyxy are: ",  (x1,y1,x2,y2))
               # print("Type of x1 is: ", type(x1))
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                #print("Center of detection is: ", (x_center, y_center))
                
                self.detected_human_img_coords = [int(x_center), int(y_center)]



                r = box.xyxy[0].astype(int)
                cv2.rectangle(cv_image, r[:2], r[2:], (0, 0, 255), 5)
                cv2.circle(cv_image, (self.detected_human_img_coords[0], self.detected_human_img_coords[1]), 65535, (0, 0, 255), 5)
                

            if object_id == 2 and object_confidence >= 0.50:
                x3, y3, x4, y4 = box.xyxy[0] # x1, y1 are top left of bounding box, width, height
                #print("xyxy are: ",  (x1,y1,x2,y2))
               # print("Type of x1 is: ", type(x1))
                x_center = (x3 + x4) / 2.0
                y_center = (y3 + y4) / 2.0
                #print("Center of detection is: ", (x_center, y_center))
                
                self.detected_car_img_coords = [int(x_center), int(y_center)]



                r = box.xyxy[0].astype(int)
                cv2.rectangle(cv_image, r[:2], r[2:], (255, 0, 0), 5)
                cv2.circle(cv_image, (self.detected_car_img_coords[0], self.detected_car_img_coords[1]), 65535, (255, 0, 0), 5)

            # else:
            #     yolo_msg = self.cv_bridge.cv2_to_imgmsg(cv_image)
            #     self.yolo_pub.publish(yolo_msg)


                # cv2.imshow("frame", cv_image)
                # cv2.waitKey(1)
        yolo_msg = self.cv_bridge.cv2_to_imgmsg(cv_image)
        self.yolo_pub.publish(yolo_msg)    
        
        #self.detected_human_img_coords = None



    def stereo_callback(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        cv_image = cv_image / 1000 # convert from mm to m
        is_valid = False

        if self.detected_human_img_coords is not None and self.detected_human_img_coords[0] < 720 and self.detected_human_img_coords[1] < 1280:
            is_valid = True

        if self.detected_human_img_coords is not None and is_valid:
            #cv2.circle(cv_image, (self.detected_human_img_coords[0], self.detected_human_img_coords[1]), 65535, 1, 2)


            roi_x, roi_y = self.detected_human_img_coords[0], self.detected_human_img_coords[1]
            roi = cv2.blur(cv_image, (30,30))#cv_image[roi_x, roi_y]
            print("roi has shape and avg value: ", (roi.shape, np.average(roi)))
            print("original values at x, y: ", cv_image[self.detected_human_img_coords[0], self.detected_human_img_coords[1]])
            #print("x and y values: ", (self.detected_human_img_coords[0], self.detected_human_img_coords[1]))
            print("blurred values at x,y: ", (roi[self.detected_human_img_coords[0], self.detected_human_img_coords[1]]))

            # cv2.imshow("Depth image", cv_image)
            # cv2.waitKey(1)


            # print("Shape of depth: ", cv_image.shape) #720, 1280
            # print("Type: ", cv_image.dtype) # uint16
            # print("Depth data at location of person: ", cv_image[self.detected_human_img_coords[0], self.detected_human_img_coords[1]])
            depth_measurement = roi[self.detected_human_img_coords[0], self.detected_human_img_coords[1]]

            if np.isnan(depth_measurement):
                print("Is nan")
                return


            # object in sensor frame
            self.object_point.header.frame_id = "oak-d-base-frame"  
            self.object_point.header.stamp = self.get_clock().now().to_msg()
            #scaled_float_value = (10 * depth_measurement) / (65535)
            # print("Original float value: ", depth_measurement)
            # print("Scaled float value (0-10): ", scaled_float_value)
            # print("pixel points: ", (self.detected_human_img_coords[0], self.detected_human_img_coords[1]))


            point_in_cam_frame_3d = self.pixel_to_world(self.detected_human_img_coords[0], self.detected_human_img_coords[1], depth_measurement)
            print(point_in_cam_frame_3d)
            print(point_in_cam_frame_3d.shape)

            self.object_point.point.x = point_in_cam_frame_3d[0] #float(depth_measurement)   
            self.object_point.point.y = point_in_cam_frame_3d[1]  
            self.object_point.point.z = point_in_cam_frame_3d[2]  

            print("image point after conversion with matrix: ", (point_in_cam_frame_3d[0], point_in_cam_frame_3d[1], point_in_cam_frame_3d[2]))


            # convert to object in map frame
            #point_in_map = self.transform_object()
            point_in_map = self.object_point

            self.publish_marker(point_in_map)

        

        


def main(args=None):
    print("Startinc yolo detector")
    rclpy.init(args=args)
    yolo_node = YoloObjectDetection()
    rclpy.spin(yolo_node)
    yolo_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()





