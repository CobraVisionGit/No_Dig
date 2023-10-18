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
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.5)
        self.vis.add_geometry(origin)
        self.view_control = self.vis.get_view_control()
        self.view_control.set_constant_z_far(1000)
        self.isstarted = False

        self.reference_pcl = o3d.geometry.PointCloud()
        self.temp_pcl = o3d.geometry.PointCloud()  # Create a member variable to hold the temporary point cloud
        self.vis.add_geometry(self.temp_pcl)

        self.has_update_point_cloud = False
        self.first_time = time.time()

        self.voxel_grid = VoxelGrid(0.20)

    def register_point_cloud(self, target_pcl, transformation_matrix):
        """
        Register the target point cloud to the reference point cloud using ICP.
        This method only estimates the pose (transformation).
        """
        threshold = 0.20
        trans_init = np.eye(4)
        converge_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200, # HAd set to 2000 before
            relative_fitness=1e-6,
            relative_rmse=1e-6
        )
        registration_result = registration_icp(
            target_pcl, self.reference_pcl, threshold, transformation_matrix,
            TransformationEstimationPointToPoint(),
            criteria=converge_criteria
        )
        transformation = registration_result.transformation

        # final_transformation = np.dot(transformation, transformation_matrix)
        # print(f"final_transformation:\n {final_transformation}")

        # if self.has_update_point_cloud:
        target_pcl.transform(transformation)

        fitness_score = registration_result.fitness
        print(f"fitness_score: {fitness_score}")

        if 999925 > (time.time() - self.first_time) > 8: # Needs 10 seconds to pass to start recording

            translation_magnitude = np.linalg.norm(transformation[:3, 3])
            # print(f"translation_magnitude: {translation_magnitude}")

            if (fitness_score > 0.60 and translation_magnitude > 0.10) or not self.has_update_point_cloud:
                print("Saving! - EEEEEEEEEEE")
                # Update the reference point cloud with the registered new point cloud
                self.update_point_cloud(transformation, self.pcl)

        return transformation


    def update_point_cloud(self, transformation, target_pcl):
        """
        Update the reference point cloud with the registered target point cloud.
        """
        rgb_o3d = o3d.geometry.Image(self.rgb)
        depth_o3d = o3d.geometry.Image(self.depth_map)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(self.rgb.shape) != 3), depth_trunc=20000, depth_scale=1000.0
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)

        # Perform voxel downsampling and statistical outlier removal
        target_pcl = pcd.voxel_down_sample(voxel_size=0.15)  # Adjust voxel size as needed
        target_pcl = pcd.remove_statistical_outlier(90, 0.05)[0]

        # Get the points and colors from the existing and new point clouds
        existing_points = np.asarray(self.reference_pcl.points)
        existing_colors = np.asarray(self.reference_pcl.colors)
        new_points = np.asarray(target_pcl.points)
        new_colors = np.asarray(target_pcl.colors)

        # Add the existing points to the voxel grid
        for point, color in zip(existing_points, existing_colors):
            self.voxel_grid.add_point(point, color)

        # Add the new points to the voxel grid, if there isn't already a point in the corresponding voxel
        for point, color in zip(new_points, new_colors):
            self.voxel_grid.add_point(point, color)

        # Update the point cloud with the points from the voxel grid
        self.reference_pcl.points = o3d.utility.Vector3dVector(self.voxel_grid.get_points())
        self.reference_pcl.colors = o3d.utility.Vector3dVector(self.voxel_grid.get_colors())

        self.has_update_point_cloud = True

        # Update the pcl to be the same as reference_pcl
        self.pcl.points = self.reference_pcl.points
        self.pcl.colors = self.reference_pcl.colors


    def filter_depth_range(self, depth_map, min_range=0, max_range=0):
        # Transfer your numpy array to the GPU
        depth_map_gpu = cp.array(depth_map)

        # Create a mask for values within the desired range
        mask = (depth_map_gpu >= min_range) & (depth_map_gpu <= max_range)

        # Set values outside the range to zero
        depth_map_gpu[~mask] = 0

        # Transfer the result back to CPU (if needed)
        depth_filtered = depth_map_gpu.get()

        return depth_filtered


    def rgbd_to_projection(self, depth_map, rgb, transformation_matrix, downsample=True, remove_noise=True):
        self.transformation_matrix = transformation_matrix
        self.depth_map = depth_map
        self.rgb = rgb

        if self.has_update_point_cloud: # Once the first set of pcl gets saved, now use shorter pcls for matching
            depth_map_filtered = self.filter_depth_range(depth_map.copy(), min_range=0, max_range=4000)

        # Filtered
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth_map)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(rgb.shape) != 3), depth_trunc=20000, depth_scale=1000.0
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)

        if downsample:
            pcd = pcd.voxel_down_sample(voxel_size=0.050) # I liked 0.135
        #
        if remove_noise:
            pcd = pcd.remove_statistical_outlier(30, 0.02)[0]

        self.pcl.points = pcd.points
        self.pcl.colors = pcd.colors


        # Apply the rotation directly without translating the point cloud
        self.pcl.transform(transformation_matrix)

        # Register the point cloud to the reference point cloud
        estimated_transformation = self.register_point_cloud(self.pcl, transformation_matrix)


        return self.pcl

    def visualize_pcd(self):
        # self.pcl = self.pcl.voxel_down_sample(voxel_size=0.050) # I liked 0.135
        # self.pcl = self.pcl.remove_statistical_outlier(90, 0.02)[0]

        # combined_points = np.concatenate((np.asarray(self.reference_pcl.points), np.asarray(self.pcl.points)), axis=0)
        # combined_colors = np.concatenate((np.asarray(self.reference_pcl.colors), np.asarray(self.pcl.colors)), axis=0)
        # self.temp_pcl.points = o3d.utility.Vector3dVector(combined_points)
        # self.temp_pcl.colors = o3d.utility.Vector3dVector(combined_colors)
        self.vis.update_geometry(self.pcl)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()

class VoxelGrid:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.grid = {}  # Dictionary to store points, keyed by voxel coordinates

    def _get_voxel_coords(self, point):
        # Convert a point's coordinates to voxel coordinates
        return tuple((point / self.voxel_size).astype(int))

    def add_point(self, point, color):
        voxel_coords = self._get_voxel_coords(point)
        if voxel_coords not in self.grid:
            self.grid[voxel_coords] = {
                'point': point,
                'color': color
            }

    def get_points(self):
        return [data['point'] for data in self.grid.values()]

    def get_colors(self):
        return [data['color'] for data in self.grid.values()]