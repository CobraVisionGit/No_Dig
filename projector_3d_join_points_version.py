#!/usr/bin/env python3

# Code copied from main depthai repo, depthai_helpers/projector_3d.py

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from open3d.pipelines.registration import registration_icp, TransformationEstimationPointToPoint

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
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.5, origin=[0, 0, 0])
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.5)
        self.vis.add_geometry(origin)
        self.view_control = self.vis.get_view_control()
        self.view_control.set_constant_z_far(1000)
        self.isstarted = False

        self.reference_pcl = o3d.geometry.PointCloud()

    def register_point_cloud(self, target_pcl):
        """
        Register the target point cloud to the reference point cloud using ICP.
        """
        # Use ICP to compute the transformation that aligns the target point cloud to the reference point cloud
        threshold = 0.02  # Set a threshold for the ICP algorithm
        trans_init = np.eye(4)  # Initial transformation
        reg_p2p = registration_icp(target_pcl, self.reference_pcl, threshold, trans_init,
                                   TransformationEstimationPointToPoint())
        return reg_p2p.transformation

    def update_point_cloud(self, transformation, target_pcl):
        """
        Update the reference point cloud with the registered target point cloud.
        """
        # Apply the transformation to the target point cloud
        target_pcl.transform(transformation)

        # Merge the transformed target point cloud with the reference point cloud
        self.reference_pcl += target_pcl

    def rgbd_to_projection(self, depth_map, rgb, downsample=False, remove_noise=False):
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth_map)

        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(rgb.shape) != 3), depth_trunc=20000, depth_scale=1000.0
        # )
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(rgb.shape) != 3), depth_trunc=2000, depth_scale=1000.0
        )
        # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        #     rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(rgb.shape) != 3), depth_trunc=200, depth_scale=1000.0
        # )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)

        if downsample:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)

        if remove_noise:
            pcd = pcd.remove_statistical_outlier(30, 0.1)[0]

        self.pcl.points = pcd.points
        self.pcl.colors = pcd.colors

        # Register the transformed point cloud to the reference point cloud
        transformation = self.register_point_cloud(self.pcl)

        # Update the reference point cloud with the registered new point cloud
        self.update_point_cloud(transformation, self.pcl)

        return self.pcl

    def visualize_pcd(self):
        self.vis.update_geometry(self.pcl)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()

