#!/usr/bin/env python3

# Code copied from main depthai repo, depthai_helpers/projector_3d.py

import numpy as np
import cupy as cp  # Import CuPy for GPU acceleration
import open3d as o3d
from scipy.spatial.transform import Rotation
from open3d.pipelines.registration import registration_icp, TransformationEstimationPointToPoint
import time

class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        self.depth_map = None
        self.rgb = None
        self.pcl = o3d.geometry.PointCloud()

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Point Cloud")
        self.vis.add_geometry(self.pcl)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.4, origin=[0, 0, 0])
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.5)
        self.vis.add_geometry(origin)
        self.view_control = self.vis.get_view_control()
        self.view_control.set_constant_z_far(1000)
        self.isstarted = False

        self.reference_pcl = o3d.geometry.PointCloud()
        self.has_update_point_cloud = False
        self.first_time = time.time()

    def register_point_cloud(self, target_pcl):
        """
        Register the target point cloud to the reference point cloud using ICP.
        """
        # Use ICP to compute the transformation that aligns the target point cloud to the reference point cloud
        threshold = 0.10  # Set a threshold for the ICP algorithm # Default or original of 0.02
        trans_init = cp.eye(4)  # Initial transformation using CuPy
        # trans_init = np.eye(4)  # Initial transformation using CuPy
        registration_result = registration_icp(target_pcl, self.reference_pcl, threshold, cp.asnumpy(trans_init),
                                   TransformationEstimationPointToPoint())
        # registration_result = registration_icp(target_pcl, self.reference_pcl, threshold, trans_init,
        #                            TransformationEstimationPointToPoint())
        # Check the size of the correspondence set
        num_correspondences = len(registration_result.correspondence_set)
        print(f"Number of correspondences: {num_correspondences}")

        # Check the inlier RMSE
        print(f"Inlier RMSE: {registration_result.inlier_rmse}")

        transformation = registration_result.transformation
        fitness_score = registration_result.fitness
        print(f"fitness_score: {fitness_score}")

        if time.time() - self.first_time > 15: # Needs 10 seconds to pass to start recording
            if fitness_score > 0.75 or not self.has_update_point_cloud:
                print("Saving")
                # Update the reference point cloud with the registered new point cloud
                self.update_point_cloud(transformation, self.pcl)

        return registration_result

    def update_point_cloud(self, transformation, target_pcl):
        self.has_update_point_cloud = True
        """
        Update the reference point cloud with the registered target point cloud.
        """
        # Apply the transformation to the target point cloud
        target_pcl.transform(transformation)

        # Merge the transformed target point cloud with the reference point cloud
        self.reference_pcl += target_pcl

        # Update the pcl to be the same as reference_pcl
        self.pcl.points = self.reference_pcl.points
        self.pcl.colors = self.reference_pcl.colors

    def get_rotation_matrix(self, pitch, roll):
        """
        Convert pitch and roll angles to a rotation matrix.
        """
        # Convert pitch and roll to a rotation matrix
        rotation_pitch = Rotation.from_euler('x', pitch, degrees=True)
        rotation_roll = Rotation.from_euler('y', roll, degrees=True)

        # Combine the rotations
        rotation = rotation_pitch * rotation_roll
        rotation_matrix = rotation.as_matrix()

        # Convert the 3x3 rotation matrix to a 4x4 transformation matrix using CuPy
        transformation_matrix = cp.eye(4)
        # transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = cp.array(rotation_matrix)
        # transformation_matrix[:3, :3] = np.array(rotation_matrix)

        return cp.asnumpy(transformation_matrix)  # Convert back to numpy for Open3D compatibility
        # return transformation_matrix  # Convert back to numpy for Open3D compatibility

    def rgbd_to_projection(self, depth_map, rgb, transformation_matrix, downsample=True, remove_noise=True):
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth_map)

        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(rgb.shape) != 3), depth_trunc=20000, depth_scale=1000.0
        # )
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(rgb.shape) != 3), depth_trunc=1000, depth_scale=1000.0
        # )
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(rgb.shape) != 3), depth_trunc=100, depth_scale=1000.0
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)

        if downsample:
            pcd = pcd.voxel_down_sample(voxel_size=0.10) # I liked 0.135

        if remove_noise:
            pcd = pcd.remove_statistical_outlier(90, 0.05)[0]

        self.pcl.points = pcd.points
        self.pcl.colors = pcd.colors

        # # Convert pitch and roll from radians to degrees
        # pitch_deg = np.degrees(pitch)
        # roll_deg = np.degrees(roll)
        #
        # # # Debug: Print pitch and roll values in degrees
        # # print(f"Pitch (degrees): {pitch_deg}, Roll (degrees): {roll_deg}")
        #
        # # Get the rotation matrix using pitch and roll in degrees
        # rotation_matrix = self.get_rotation_matrix(pitch_deg, roll_deg)

        # Apply the rotation directly without translating the point cloud
        self.pcl.transform(transformation_matrix)

        # Register the transformed point cloud to the reference point cloud
        registration_result  = self.register_point_cloud(self.pcl)


        return self.pcl

    def visualize_pcd(self):
        self.vis.update_geometry(self.pcl)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()

