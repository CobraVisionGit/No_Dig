import open3d as o3d
from open3d.pipelines.registration import registration_ransac_based_on_feature_matching
from open3d.pipelines.registration import TransformationEstimationPointToPoint
from open3d.pipelines.registration import CorrespondenceCheckerBasedOnEdgeLength
from open3d.pipelines.registration import CorrespondenceCheckerBasedOnDistance
from open3d.pipelines.registration import CorrespondenceCheckerBasedOnNormal
import numpy as np
import time

class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        # Initialize the visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        # Initialize the point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.reference_pcd = o3d.geometry.PointCloud()
        self.temp_pcd = o3d.geometry.PointCloud()

        # Add the point cloud to the visualizer
        self.vis.add_geometry(self.reference_pcd)
        # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
        # self.vis.add_geometry(origin)

        render_option = self.vis.get_render_option()
        render_option.point_size = 10.0  # Set point size to 10.0

        self.had_reset_view = False
        self.view_counter = 0

        self.first_time = time.time()
        self.has_update_point_cloud = False

        self.voxel_grid = VoxelGrid(0.20)

    def update_point_cloud(self, rgb, points, transformation_matrix):
        if transformation_matrix is not None and points is not None:
            # Rescale points from mm to meters
            points = points / 1000.0
            transformation_matrix[:, 3][:3] = transformation_matrix[:, 3][:3] / 1000.0

            # print(f"points.shape: {points.shape}")
            if isinstance(points, np.ndarray):
                temp_pcl = o3d.geometry.PointCloud()
                temp_pcl.points = o3d.utility.Vector3dVector(points)
                points = temp_pcl

            # Update the point cloud data
            self.pcd.points = points.points

            colors = np.array([[0, 0, 0] for _ in range(len(points.points))])  # Red color for all points
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            # Apply the rotation directly without translating the point cloud
            self.pcd.transform(transformation_matrix)

            # Register the point cloud to the reference point cloud
            self.register_point_cloud(self.pcd, transformation_matrix)


            # # print(f"Before: {len(self.reference_pcd.points)}, {len(self.pcd.points)}")
            # combined_points = np.concatenate((np.asarray(self.reference_pcd.points), np.asarray(self.pcd.points)), axis=0)
            # combined_colors = np.concatenate((np.asarray(self.reference_pcd.colors), np.asarray(self.pcd.colors)), axis=0)
            # # print(f"After: {len(combined_points)}")
            # self.reference_pcd.points = o3d.utility.Vector3dVector(combined_points)
            # self.reference_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
            # # print(f"After: {len(self.reference_pcd.points)}")


            # Update the visualization
            # self.vis.update_geometry(self.reference_pcd)
            self.vis.update_geometry(self.reference_pcd)
            self.vis.poll_events()
            self.vis.update_renderer()

            if self.view_counter % 10 == 0 and self.view_counter < 1000000: # 30 # If it does not seem to reset view, increase the < n # The 'n' value
                self.vis.reset_view_point(True)
                # # Get the ViewControl object
                # view_ctl = self.vis.get_view_control()
                # # Set the front direction of the camera to rotate the view 180 degrees
                # front = [0, -1, 0]  # This vector points downwards
                # view_ctl.set_front(front)
                # # Change the field of view to invert the view
                # view_ctl.change_field_of_view(step=-180.0)

                self.had_reset_view = True

            self.view_counter += 1

    def register_point_cloud(self, target_pcd, transformation_matrix):
        """
        Register the target point cloud to the reference point cloud using RANSAC.
        This method only estimates the pose (transformation).
        """
        # transformation_matrix = np.eye(4)
        fitness_score = 0
        if len(target_pcd.points) == 0 or len(self.reference_pcd.points) == 0:
            # print("One or both point clouds are empty, cannot register.")
            # return np.eye(4)  # Return an identity matrix or handle this case as appropriate
            pass
        else:

            # Ensure normals are computed
            if target_pcd.has_normals() is False:
                target_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            if self.reference_pcd.has_normals() is False:
                self.reference_pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # Assuming FPFH feature is used, which needs to be computed in advance
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                self.reference_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))

            distance_threshold = 2   # Adjust this threshold as needed

            result_ransac = registration_ransac_based_on_feature_matching(
                target_pcd, self.reference_pcd, source_fpfh, target_fpfh, False,  # False is for mutual_filter
                distance_threshold,  # max_correspondence_distance should be a float
                TransformationEstimationPointToPoint(False),  # estimation_method
                4,  # ransac_n
                [CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 CorrespondenceCheckerBasedOnDistance(distance_threshold),
                 CorrespondenceCheckerBasedOnNormal(0.349066)],  # checkers
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000)  # criteria
            )

            transformation = result_ransac.transformation

            final_transformation = np.dot(transformation, transformation_matrix)

            target_pcd.transform(final_transformation)

            fitness_score = result_ransac.inlier_rmse
            if fitness_score > 0.20:
                print(f"fitness_score: {fitness_score}")

        if (time.time() - self.first_time) > 7: # Needs 10 seconds to pass to start recording

            # translation_magnitude = np.linalg.norm(transformation[:3, 3])
            # print(f"translation_magnitude: {translation_magnitude}")

            # if (fitness_score > 0.60 and translation_magnitude > 0.10) or not self.has_update_point_cloud:
            if (fitness_score > 0.20) or not self.has_update_point_cloud:
                print("Saving! - EEEEEEEEEEE")
                print(f"Amount of pcd points to save: {len(self.pcd.points)}")
                print(f"reference_pcd points size: {len(self.reference_pcd.points)}")
                # Update the reference point cloud with the registered new point cloud
                self.save_point_cloud(self.pcd)

    def save_point_cloud(self, target_pcd):
        target_pcd = target_pcd.remove_statistical_outlier(90, 0.05)[0]

        # Get the points and colors from the existing and new point clouds
        existing_points = np.asarray(self.reference_pcd.points)
        existing_colors = np.asarray(self.reference_pcd.colors)
        new_points = np.asarray(target_pcd.points)
        new_colors = np.asarray(target_pcd.colors)
        print(f"Existing points size: {len(existing_points)}")

        # Add the existing points to the voxel grid
        for point, color in zip(existing_points, existing_colors):
            self.voxel_grid.add_point(point, color)

        # Add the new points to the voxel grid, if there isn't already a point in the corresponding voxel
        for point, color in zip(new_points, new_colors):
            self.voxel_grid.add_point(point, color)

        # Update the point cloud with the points from the voxel grid
        self.reference_pcd.points = o3d.utility.Vector3dVector(self.voxel_grid.get_points())
        self.reference_pcd.colors = o3d.utility.Vector3dVector(self.voxel_grid.get_colors())
        print(f"reference_pcd points size: {len(self.reference_pcd.points)}")

        self.has_update_point_cloud = True

        # Update the pcl to be the same as reference_pcl
        self.pcd.points = self.reference_pcd.points
        self.pcd.colors = self.reference_pcd.colors
        print(f"pcd points size: {len(self.reference_pcd.points)}")

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
        color = [1., 0., 0.]
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