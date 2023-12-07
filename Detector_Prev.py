import torch
import torchvision.transforms
import numpy as np
import cv2
import time

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
from cv2 import aruco
import math


class TemporalSmoother:
    def __init__(self, window_size):
        self.window_size = window_size
        self.positions = []

    def update(self, new_position):
        self.positions.append(new_position)
        if len(self.positions) > self.window_size:
            self.positions.pop(0)
        return np.mean(self.positions, axis=0)


class Detector:
    def __init__(self):
        self.model = None  # model initialization
        self.classes = None
        self.min_score = 0.50
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.orb = cv2.ORB_create()

        # FLANN parameters for feature matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=12,
                            key_size=24,
                            multi_probe_level=2)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.environment_map = {}
        self.environment_map_list = []
        self.map_points = np.empty((0, 3), dtype=np.float32)
        self.map_points_list = []
        self.map_points_temporary = np.empty((0, 3), dtype=np.float32)
        self.map_points_now = np.empty((0, 3), dtype=np.float32)
        self.map_points_last = np.empty((0, 3), dtype=np.float32)
        self.map_points_reference = np.empty((0, 3), dtype=np.float32)
        self.camera_pose = None  # Initial camera pose is identity matrix
        self.camera_pose_list = []
        self.object_position = None  # World coordinates of the object
        self.initialized_object = False

        self.smoother = TemporalSmoother(window_size=5)

        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)  # aruco.Dictionary_get(aruco.DICT_4X4_50)
        self.param_markers = aruco.DetectorParameters()  # aruco.DetectorParameters_create()# Create the ArucoDetector object
        self.aruco_detector = aruco.ArucoDetector(self.marker_dict, self.param_markers)
        self.marker_corners, self.marker_ids = None, None
        # Define the marker size
        self.marker_size = 1  # Assume marker size is 1 unit
        # Create 3D object points
        self.marker_points = np.array([
            [0, self.marker_size, 0],
            [self.marker_size, self.marker_size, 0],
            [self.marker_size, 0, 0],
            [0, 0, 0]
        ], dtype=np.float32)
        self.saved_zones = {}
        self.aruco_rvecs_previous = []
        self.last_few_special_vectors = []

        print("Detector started")

    def set_intrinsics(self, intrinsics, dist_coeffs, intrinsics_rgb, dist_coeffs_rgb):
        self.intrinsics = intrinsics
        self.dist_coeffs = dist_coeffs
        self.intrinsics_rgb = intrinsics_rgb
        self.dist_coeffs_rgb = dist_coeffs_rgb

        # Extracting the intrinsic parameters from self.intrinsics
        self.intrinsics = np.array(self.intrinsics)
        self.intrinsics_rgb = np.array(self.intrinsics_rgb)
        self.dist_coeffs_rgb = np.array(self.dist_coeffs_rgb)
        self.fx, self.fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        self.cx, self.cy = self.intrinsics[0, 2], self.intrinsics[1, 2]

    def initialize_object(self, depth_frame):
        if not self.initialized_object:
            print("INITIALIZING OBJECT")
            # # Normalize the depth frame to the range 0-255
            # normalized_depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            # # Convert normalized depth frame to 8-bit unsigned integers
            # depth_frame_8u = np.uint8(normalized_depth_frame)
            # # Apply denoising
            # denoised_depth_frame_8u = cv2.fastNlMeansDenoising(depth_frame_8u, None, 30, 7, 21)
            # # If necessary, convert denoised depth frame back to original data type
            # depth_frame = denoised_depth_frame_8u.astype(depth_frame.dtype)

            # Assuming depth_frame is your depth map
            h, w = depth_frame.shape
            mid_pixel_coords = np.array([w // 2, h // 2])
            depth_value = depth_frame[mid_pixel_coords[1], mid_pixel_coords[0]]
            self.object_position = self.to_spatial(*mid_pixel_coords, depth_value)
            self.initialized_object = True

    def update_camera_pose(self, transformation_matrix):
        self.camera_pose = transformation_matrix  # Update camera pose
        self.camera_pose_list.append(self.camera_pose)

    def to_spatial(self, u, v, Z):
        if Z == 0:
            return None
        X_cam = (u - self.cx) * Z / self.fx
        Y_cam = (v - self.cy) * Z / self.fy
        Z_cam = Z
        return torch.tensor([X_cam, Y_cam, Z_cam], device=self.device)

    def to_image(self, X_cam, Y_cam, Z_cam):
        u = (X_cam * self.fx / Z_cam) + self.cx
        v = (Y_cam * self.fy / Z_cam) + self.cy
        return torch.tensor([u, v], device=self.device)

    def significant_pose_change(self, old_matrix, new_matrix, threshold=(1_000)):
        # This method checks whether there's a significant change in camera pose.
        # This is a simple placeholder. You might want to have a more sophisticated method.
        pose_change_value = np.linalg.norm(old_matrix - new_matrix)
        # if pose_change_value > 10_000_000:
        #     print("Crazy pose change values")
        #     return False
        # if pose_change_value > threshold:
        # print(f"significant pose change value: {pose_change_value}")
        return pose_change_value > threshold

    def update_map_points(self, new_spatial_coords):
        # Convert new_spatial_coords to CPU and then to numpy, if necessary
        new_spatial_coords_np = [coord.cpu().numpy() for coord in new_spatial_coords]

        # Now, new_spatial_coords_np is a list of numpy arrays. We want to concatenate these to self.map_points.
        # But first, we need to convert new_spatial_coords_np to a single numpy array.
        new_spatial_coords_np = np.array(new_spatial_coords_np)

        # Check if any of the new_spatial_coords_np already exist in self.map_points to avoid duplication
        # This assumes that self.map_points and new_spatial_coords_np have the same shape[-1] (e.g., both are Nx3 arrays)
        existing_coords_set = set(tuple(coord) for coord in self.map_points)
        new_coords_set = set(tuple(coord) for coord in new_spatial_coords_np)
        unique_new_coords_list = [coord for coord in new_coords_set - existing_coords_set]

        # Convert unique_new_coords_list to a 2D numpy array
        unique_new_coords = np.array(unique_new_coords_list).reshape(-1, 3)  # Assuming each coordinate has 3 elements

        # Now concatenate the unique new spatial coordinates to self.map_points
        self.map_points = np.concatenate((self.map_points, unique_new_coords), axis=0)

    def mapper(self, frame, depth_frame):
        # # Normalize the depth frame to the range 0-255
        # normalized_depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        # # Convert normalized depth frame to 8-bit unsigned integers
        # depth_frame_8u = np.uint8(normalized_depth_frame)
        # # Apply denoising
        # denoised_depth_frame_8u = cv2.fastNlMeansDenoising(depth_frame_8u, None, 30, 7, 21)
        # # If necessary, convert denoised depth frame back to original data type
        # depth_frame = denoised_depth_frame_8u.astype(depth_frame.dtype)

        transformation_matrix = None
        good_keypoints = None

        # Feature Extraction and Matching Section
        # ----------------------------------- Feature Extraction and Matching Start -----------------------------------
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask where the depth is zero
        depth_mask = depth_frame == 0

        # Set the corresponding pixels in gray_frame to a value
        # that's ignored by ORB (e.g., maximum value)
        # Assuming gray_frame is uint8, so max value is 255
        # if np.any(depth_mask == False):
        #     # print(f"depth_frame = : \n{depth_frame}")
        #     # print(f"depth_mask = : \n{depth_mask}")
        #     # print(f"Before: \n{gray_frame}")
        #     gray_frame[depth_mask] = 255
        #     # print(f"After: \n{gray_frame}")

        keypoints, descriptors = self.orb.detectAndCompute(gray_frame, None)
        # ----------------------------------- Feature Extraction and Matching End -----------------------------------

        # Convert data to PyTorch tensors and move to GPU
        depth_frame_t = torch.tensor(depth_frame.astype(np.float32), device=self.device)

        # Motion EstimationSection
        # ----------------------------------- Motion Estimation Start -----------------------------------
        # Convert the keypoints to spatial coordinates
        spatial_coords = [self.to_spatial(u, v, depth_frame_t[int(v), int(u)]) for kp in keypoints for u, v in [kp.pt]]

        # print(f"len(self.map_points): {len(self.map_points)}")

        if len(self.environment_map_list) == 0:
            environment_map_list_to_use = [self.environment_map]
        else:
            environment_map_list_to_use = self.environment_map_list

        break_index = -1
        # Matching with features in the map
        # (This is overly simplified and assumes features in the map are from a single previous frame)
        for environment_map_index, environment_map in reversed(list(enumerate(environment_map_list_to_use))):
            break_index += 1
            if break_index > 20:
                break
            # print(f"environment_map: {environment_map}, length of environment_map: {len(environment_map)}")
            if len(environment_map) == 0:
                print(f"CONTINUING! Empty environment_map.")
                continue
            # print(f"environment_map_index: {environment_map_index}")

            if len(environment_map['descriptors']) >= 2 and len(descriptors) >= 2:
                matches = self.flann.knnMatch(environment_map['descriptors'], descriptors, k=2)
            else:
                print("Not enough descriptors to perform matching.")
                matches = []  # or whatever is appropriate in your case

            # print(f"len(matches): {len(matches)}")

            good_matches = []
            for match in matches:
                if len(match) < 2:
                    continue  # Skip this match if there aren't 2 neighbors
                m, n = match
                if m.distance < 0.65 * n.distance:
                    good_matches.append(m)

            len_good_matches_before = len(matches)
            good_matches = [m for m in good_matches if
                            depth_frame[int(keypoints[m.trainIdx].pt[1]), int(keypoints[m.trainIdx].pt[0])] != 0]
            len_good_matches_after = len(good_matches)
            # if len_good_matches_before != len_good_matches_after:
            print(
                f"Index: {environment_map_index} out of {len(environment_map_list_to_use)}. Length of matches changed! Before: {len_good_matches_before}, After: {len_good_matches_after}")

            good_matches_required = 9
            if len(good_matches) < good_matches_required:
                # print(f"CONTINUING! Only have {len(good_matches)} good_matches")
                continue

            # print(f"len(good_matches): {len(good_matches)}")

            # Collect indices of good matches
            good_indices = [match.trainIdx for match in good_matches]

            # Filter keypoints and descriptors based on good matches
            good_keypoints = [keypoints[idx] for idx in good_indices]
            good_descriptors = descriptors[good_indices]

            if len(self.environment_map_list) == 0:
                map_points = self.map_points_temporary
            else:
                map_points = self.map_points_list[environment_map_index]

            if len(map_points) <= len(good_matches):
                print("Warming up!")
                obj_points, img_points = [], []
            else:
                if len(self.map_points) > 0:
                    obj_points = []  # 3D points in the world
                    img_points = []  # 2D points in the image

                    for match in good_matches:
                        img_idx = match.trainIdx  # index of the feature in the current image
                        map_idx = match.queryIdx  # index of the feature in the environment map

                        # assuming map_points is a list of 3D points corresponding to the environment_map['keypoints']
                        # print(f'map_idx: {map_idx}, len(self.map_points): {len(self.map_points)}, self.map_points: {self.map_points[:-3]}')
                        if len(self.environment_map_list) == 0:
                            obj_points.append(self.map_points_temporary[map_idx])
                        else:
                            # print(f"Length of envrionment_map_list: {len(self.environment_map_list)}. Length of map_points_list: {len(self.map_points_list)}")
                            map_points = self.map_points_list[environment_map_index]
                            map_point = map_points[map_idx]
                            obj_points.append(map_point)
                        # obj_points.append(self.map_points[map_idx])

                        # get the 2D point corresponding to this match in the current image
                        img_point = keypoints[img_idx].pt
                        img_points.append(img_point)

                    obj_points = np.array(obj_points, dtype=np.float32)
                    img_points = np.array(img_points, dtype=np.float32)
                else:
                    obj_points, img_points = [], []

            # Assume obj_points is a Nx3 array of 3D coordinates of feature points in the world frame
            # Assume img_points is a Nx2 array of 2D coordinates of the same points in the image frame
            # Assume camera_matrix and dist_coeffs are the intrinsic camera parameters and distortion coefficients

            if len(obj_points) >= good_matches_required and len(img_points) >= good_matches_required:
                ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.intrinsics_rgb, self.dist_coeffs_rgb)
                R, _ = cv2.Rodrigues(rvec)  # Gets rotation matrix

                # Create a 4x4 identity matrix
                transformation_matrix = np.eye(4)

                # Set the top-left 3x3 block to the rotation matrix
                transformation_matrix[:3, :3] = R

                # Set the top three elements of the rightmost column to the translation vector
                transformation_matrix[:3, 3] = tvec.squeeze()
                # print("BREAKING FREE")
                print(f"BREAKING FREE @ environment_map_index: {environment_map_index}")
                break  # Breaks out of the reverse for loop above
            else:
                print("Not enough point correspondences to estimate pose.")

        if good_keypoints is not None:
            # cv2.drawKeypoints(gray_frame, good_keypoints, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.drawKeypoints(gray_frame, good_keypoints, frame)

        # ----------------------------------- Motion Estimation End -----------------------------------

        if transformation_matrix is not None:
            # Check for significant change before processing new spatial coordinates
            if self.camera_pose_list[environment_map_index] is None or self.significant_pose_change(
                    self.camera_pose_list[environment_map_index], transformation_matrix) or len(
                    self.map_points_list) == 0:
                # Update the map points with the new spatial coordinates
                self.update_map_points(
                    spatial_coords)  # This line replaces the existing code for updating self.map_points

                # Update the map with good matches only
                # if len(good_matches) >= good_matches_required: # Not needed since transformation_matrix won't happen without this condition
                # Get the spatial coordinates of the good matches
                good_spatial_coords = [self.to_spatial(u, v, depth_frame_t[int(v), int(u)]) for match in good_matches
                                       for u, v in [keypoints[match.trainIdx].pt]]
                # Assuming good_spatial_coords is a list of tensors on a CUDA device
                good_spatial_coords_cpu_np = [coord.cpu().numpy() for coord in good_spatial_coords]

                # Now, good_spatial_coords_cpu_np is a list of numpy arrays. Convert this list to a single numpy array.
                # This step assumes that each element of good_spatial_coords_cpu_np is a 1D numpy array with the same shape.
                good_spatial_coords_np = np.array(good_spatial_coords_cpu_np).reshape(-1,
                                                                                      3)  # Adjust 3 to the correct number of elements per coordinate if necessary

                # Now you can assign this array to self.map_points_temporary
                self.map_points_temporary = good_spatial_coords_np
                self.map_points_list.append(self.map_points_temporary)

                # Update the map points with the spatial coordinates of the good matches
                self.update_map_points(good_spatial_coords)

                self.environment_map = {'keypoints': good_keypoints, 'descriptors': good_descriptors}
                self.environment_map_list.append(self.environment_map)

                self.update_camera_pose(transformation_matrix)  # Update camera pose

            if self.object_position is not None:
                # Step 3: Project the object's 3D world coordinates to 2D image coordinates
                # Assume self.object_position is a 3-element numpy array
                smoothed_position = self.smoother.update(
                    self.object_position.cpu().numpy())  # Update the smoother with the current object position
                object_position_homogeneous = np.append(smoothed_position, 1)  # Make it homogeneous
                # object_position_homogeneous = np.append(self.object_position.cpu().numpy(), 1)  # Make it homogeneous

                # Separate rotation and translation from the transformation matrix
                R = transformation_matrix[:3, :3]
                t = transformation_matrix[:3, 3]

                # Compute the projection matrix by combining the intrinsic matrix, rotation, and translation
                projection_matrix = np.dot(self.intrinsics,
                                           np.hstack((R, t.reshape(-1, 1))))  # Ensure t is a column vector

                # Project to 2D
                projected_position = np.dot(projection_matrix, object_position_homogeneous)
                projected_position /= projected_position[2]  # Divide by z to get image coordinates

                # Step 4: Draw the object in the 2D image
                projected_position = projected_position.astype(int)  # Convert to int for drawing
                cv2.circle(frame, tuple(projected_position[:2]), 20, (255, 0, 255),
                           -1)  # Draw a green circle at the projected position

            # self.map_points_last = self.map_points_now
            return frame, self.map_points_list[-1], transformation_matrix
        else:
            print("HERE ======================================== HERE")
            if len(self.map_points) == 0:
                try:
                    if len(good_matches) >= good_matches_required:
                        print("DOES IT EGVER MAKE IT HERE?")
                        # Get the spatial coordinates of the good matches
                        good_spatial_coords = [self.to_spatial(u, v, depth_frame_t[int(v), int(u)]) for match in
                                               good_matches
                                               for u, v in [keypoints[match.trainIdx].pt]]
                        # Assuming good_spatial_coords is a list of tensors on a CUDA device
                        good_spatial_coords_cpu_np = [coord.cpu().numpy() for coord in good_spatial_coords]

                        # Now, good_spatial_coords_cpu_np is a list of numpy arrays. Convert this list to a single numpy array.
                        # This step assumes that each element of good_spatial_coords_cpu_np is a 1D numpy array with the same shape.
                        good_spatial_coords_np = np.array(good_spatial_coords_cpu_np).reshape(-1,
                                                                                              3)  # Adjust 3 to the correct number of elements per coordinate if necessary

                        # Now you can assign this array to self.map_points_temporary
                        self.map_points_temporary = good_spatial_coords_np
                        self.map_points_list.append(self.map_points_temporary)

                        # Update the map points with the spatial coordinates of the good matches
                        self.update_map_points(good_spatial_coords)

                        # Collect indices of good matches
                        good_indices = [match.trainIdx for match in good_matches]

                        # Filter keypoints and descriptors based on good matches
                        good_keypoints = [keypoints[idx] for idx in good_indices]
                        good_descriptors = descriptors[good_indices]

                        # Update the environment map
                        self.environment_map = {'keypoints': good_keypoints, 'descriptors': good_descriptors}
                        self.environment_map_list.append(self.environment_map)

                        self.update_camera_pose(None)
                except Exception as e:
                    # This block will catch any other exceptions derived from the Exception base class
                    print(f"An error occurred: {e}")

                    print("THIS IS DESCRIPTOR", len(descriptors))
                    self.environment_map = {'keypoints': keypoints, 'descriptors': descriptors}

        return frame, None, None

    def mapper_for_o3d(self, frame, depth_frame):
        # # Normalize the depth frame to the range 0-255
        # normalized_depth_frame = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        # # Convert normalized depth frame to 8-bit unsigned integers
        # depth_frame_8u = np.uint8(normalized_depth_frame)
        # # Apply denoising
        # denoised_depth_frame_8u = cv2.fastNlMeansDenoising(depth_frame_8u, None, 30, 7, 21)
        # # If necessary, convert denoised depth frame back to original data type
        # depth_frame = denoised_depth_frame_8u.astype(depth_frame.dtype)

        transformation_matrix = None
        good_keypoints = None

        # Feature Extraction and Matching Section
        # ----------------------------------- Feature Extraction and Matching Start -----------------------------------
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a mask where the depth is zero
        depth_mask = depth_frame == 0

        # Set the corresponding pixels in gray_frame to a value
        # that's ignored by ORB (e.g., maximum value)
        # Assuming gray_frame is uint8, so max value is 255
        # if np.any(depth_mask == False):
        #     # print(f"depth_frame = : \n{depth_frame}")
        #     # print(f"depth_mask = : \n{depth_mask}")
        #     # print(f"Before: \n{gray_frame}")
        #     gray_frame[depth_mask] = 255
        #     # print(f"After: \n{gray_frame}")

        keypoints, descriptors = self.orb.detectAndCompute(gray_frame, None)
        # ----------------------------------- Feature Extraction and Matching End -----------------------------------

        # Convert data to PyTorch tensors and move to GPU
        depth_frame_t = torch.tensor(depth_frame.astype(np.float32), device=self.device)

        # Motion EstimationSection
        # ----------------------------------- Motion Estimation Start -----------------------------------
        # Convert the keypoints to spatial coordinates
        spatial_coords = [self.to_spatial(u, v, depth_frame_t[int(v), int(u)]) for kp in keypoints for u, v in [kp.pt]]

        # print(f"len(self.map_points): {len(self.map_points)}")

        if len(self.environment_map_list) == 0:
            environment_map_list_to_use = [self.environment_map]
        else:
            environment_map_list_to_use = self.environment_map_list

        break_index = -1
        # Matching with features in the map
        # (This is overly simplified and assumes features in the map are from a single previous frame)
        for environment_map_index, environment_map in reversed(list(enumerate(environment_map_list_to_use))):
            break_index += 1
            if break_index > 20:
                break
            # print(f"environment_map: {environment_map}, length of environment_map: {len(environment_map)}")
            if len(environment_map) == 0:
                print(f"CONTINUING! Empty environment_map.")
                continue
            # print(f"environment_map_index: {environment_map_index}")

            if descriptors is None or environment_map['descriptors'] is None:
                continue  # REMoVE?
            if len(environment_map['descriptors']) >= 2 and len(descriptors) >= 2:
                matches = self.flann.knnMatch(environment_map['descriptors'], descriptors, k=2)
            else:
                print("Not enough descriptors to perform matching.")
                matches = []  # or whatever is appropriate in your case

            # print(f"len(matches): {len(matches)}")

            good_matches = []
            for match in matches:
                if len(match) < 2:
                    continue  # Skip this match if there aren't 2 neighbors
                m, n = match
                # if m.distance < 0.65 * n.distance:
                #     good_matches.append(m)
                good_matches.append(m)  # Replacing above for open3d (o3d)

            len_good_matches_before = len(matches)
            good_matches = [m for m in good_matches if
                            depth_frame[int(keypoints[m.trainIdx].pt[1]), int(keypoints[m.trainIdx].pt[0])] != 0]
            len_good_matches_after = len(good_matches)
            # if len_good_matches_before != len_good_matches_after:
            # print(f"Index: {environment_map_index} out of {len(environment_map_list_to_use)}. Length of matches changed! Before: {len_good_matches_before}, After: {len_good_matches_after}")

            good_matches_required = 9
            if len(good_matches) < good_matches_required:
                # print(f"CONTINUING! Only have {len(good_matches)} good_matches")
                continue

            # print(f"len(good_matches): {len(good_matches)}")

            # Collect indices of good matches
            good_indices = [match.trainIdx for match in good_matches]

            # Filter keypoints and descriptors based on good matches
            good_keypoints = [keypoints[idx] for idx in good_indices]
            good_descriptors = descriptors[good_indices]

            if len(self.environment_map_list) == 0:
                map_points = self.map_points_temporary
            else:
                map_points = self.map_points_list[environment_map_index]

            if len(map_points) <= len(good_matches):
                # print("Warming up!")
                obj_points, img_points = [], []
            else:
                if len(self.map_points) > 0:
                    obj_points = []  # 3D points in the world
                    img_points = []  # 2D points in the image

                    for match in good_matches:
                        img_idx = match.trainIdx  # index of the feature in the current image
                        map_idx = match.queryIdx  # index of the feature in the environment map

                        # assuming map_points is a list of 3D points corresponding to the environment_map['keypoints']
                        # print(f'map_idx: {map_idx}, len(self.map_points): {len(self.map_points)}, self.map_points: {self.map_points[:-3]}')
                        if len(self.environment_map_list) == 0:
                            obj_points.append(self.map_points_temporary[map_idx])
                        else:
                            # print(f"Length of envrionment_map_list: {len(self.environment_map_list)}. Length of map_points_list: {len(self.map_points_list)}")
                            map_points = self.map_points_list[environment_map_index]
                            map_point = map_points[map_idx]
                            obj_points.append(map_point)
                        # obj_points.append(self.map_points[map_idx])

                        # get the 2D point corresponding to this match in the current image
                        img_point = keypoints[img_idx].pt
                        img_points.append(img_point)

                    obj_points = np.array(obj_points, dtype=np.float32)
                    img_points = np.array(img_points, dtype=np.float32)
                else:
                    obj_points, img_points = [], []

            # Assume obj_points is a Nx3 array of 3D coordinates of feature points in the world frame
            # Assume img_points is a Nx2 array of 2D coordinates of the same points in the image frame
            # Assume camera_matrix and dist_coeffs are the intrinsic camera parameters and distortion coefficients

            if len(obj_points) >= good_matches_required and len(img_points) >= good_matches_required:
                ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, self.intrinsics_rgb, self.dist_coeffs_rgb)
                R, _ = cv2.Rodrigues(rvec)  # Gets rotation matrix

                # Create a 4x4 identity matrix
                transformation_matrix = np.eye(4)

                # Set the top-left 3x3 block to the rotation matrix
                transformation_matrix[:3, :3] = R

                # Set the top three elements of the rightmost column to the translation vector
                transformation_matrix[:3, 3] = tvec.squeeze()
                # print("BREAKING FREE")
                # print(f"BREAKING FREE @ environment_map_index: {environment_map_index}")
                break  # Breaks out of the reverse for loop above
            else:
                # print("Not enough point correspondences to estimate pose.")
                pass

        if good_keypoints is not None:
            # cv2.drawKeypoints(gray_frame, good_keypoints, frame, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.drawKeypoints(gray_frame, good_keypoints, frame)

        # ----------------------------------- Motion Estimation End -----------------------------------

        if transformation_matrix is not None:
            # Check for significant change before processing new spatial coordinates
            if self.camera_pose_list[environment_map_index] is None or self.significant_pose_change(
                    self.camera_pose_list[environment_map_index], transformation_matrix) or len(
                    self.map_points_list) == 0:
                # Update the map points with the new spatial coordinates
                self.update_map_points(
                    spatial_coords)  # This line replaces the existing code for updating self.map_points

                # Update the map with good matches only
                # if len(good_matches) >= good_matches_required: # Not needed since transformation_matrix won't happen without this condition
                # Get the spatial coordinates of the good matches
                good_spatial_coords = [self.to_spatial(u, v, depth_frame_t[int(v), int(u)]) for match in good_matches
                                       for u, v in [keypoints[match.trainIdx].pt]]
                # Assuming good_spatial_coords is a list of tensors on a CUDA device
                good_spatial_coords_cpu_np = [coord.cpu().numpy() for coord in good_spatial_coords]

                # Now, good_spatial_coords_cpu_np is a list of numpy arrays. Convert this list to a single numpy array.
                # This step assumes that each element of good_spatial_coords_cpu_np is a 1D numpy array with the same shape.
                good_spatial_coords_np = np.array(good_spatial_coords_cpu_np).reshape(-1, 3)  # Adjust 3 to the correct number of elements per coordinate if necessary

                # Now you can assign this array to self.map_points_temporary
                self.map_points_temporary = good_spatial_coords_np
                self.map_points_list.append(self.map_points_temporary)

                # Update the map points with the spatial coordinates of the good matches
                self.update_map_points(good_spatial_coords)

                self.environment_map = {'keypoints': good_keypoints, 'descriptors': good_descriptors}
                self.environment_map_list.append(self.environment_map)

                self.update_camera_pose(transformation_matrix)  # Update camera pose

            if self.object_position is not None:
                # Step 3: Project the object's 3D world coordinates to 2D image coordinates
                # Assume self.object_position is a 3-element numpy array
                smoothed_position = self.smoother.update(
                    self.object_position.cpu().numpy())  # Update the smoother with the current object position
                object_position_homogeneous = np.append(smoothed_position, 1)  # Make it homogeneous
                # object_position_homogeneous = np.append(self.object_position.cpu().numpy(), 1)  # Make it homogeneous

                # Separate rotation and translation from the transformation matrix
                R = transformation_matrix[:3, :3]
                t = transformation_matrix[:3, 3]

                # Compute the projection matrix by combining the intrinsic matrix, rotation, and translation
                projection_matrix = np.dot(self.intrinsics,
                                           np.hstack((R, t.reshape(-1, 1))))  # Ensure t is a column vector

                # Project to 2D
                projected_position = np.dot(projection_matrix, object_position_homogeneous)
                projected_position /= projected_position[2]  # Divide by z to get image coordinates

                # Step 4: Draw the object in the 2D image
                projected_position = projected_position.astype(int)  # Convert to int for drawing
                cv2.circle(frame, tuple(projected_position[:2]), 20, (255, 0, 255),
                           -1)  # Draw a green circle at the projected position

            # self.map_points_last = self.map_points_now
            return frame, self.map_points_list[-1], transformation_matrix
        else:
            # print("HERE ======================================== HERE")
            if len(self.map_points) == 0:
                try:
                    if len(good_matches) >= good_matches_required:
                        print("DOES IT EGVER MAKE IT HERE?")
                        # Get the spatial coordinates of the good matches
                        good_spatial_coords = [self.to_spatial(u, v, depth_frame_t[int(v), int(u)]) for match in
                                               good_matches
                                               for u, v in [keypoints[match.trainIdx].pt]]
                        # Assuming good_spatial_coords is a list of tensors on a CUDA device
                        good_spatial_coords_cpu_np = [coord.cpu().numpy() for coord in good_spatial_coords]

                        # Now, good_spatial_coords_cpu_np is a list of numpy arrays. Convert this list to a single numpy array.
                        # This step assumes that each element of good_spatial_coords_cpu_np is a 1D numpy array with the same shape.
                        good_spatial_coords_np = np.array(good_spatial_coords_cpu_np).reshape(-1,
                                                                                              3)  # Adjust 3 to the correct number of elements per coordinate if necessary

                        # Now you can assign this array to self.map_points_temporary
                        self.map_points_temporary = good_spatial_coords_np
                        self.map_points_list.append(self.map_points_temporary)

                        # Update the map points with the spatial coordinates of the good matches
                        self.update_map_points(good_spatial_coords)

                        # Collect indices of good matches
                        good_indices = [match.trainIdx for match in good_matches]

                        # Filter keypoints and descriptors based on good matches
                        good_keypoints = [keypoints[idx] for idx in good_indices]
                        good_descriptors = descriptors[good_indices]

                        # Update the environment map
                        self.environment_map = {'keypoints': good_keypoints, 'descriptors': good_descriptors}
                        self.environment_map_list.append(self.environment_map)

                        self.update_camera_pose(None)
                except Exception as e:
                    # This block will catch any other exceptions derived from the Exception base class
                    print(f"An error occurred: {e}")

                    print("THIS IS DESCRIPTOR", len(descriptors))
                    self.environment_map = {'keypoints': keypoints, 'descriptors': descriptors}

        return frame, None, None

    def setup_YOLOv7(self):
        import numpy as np
        import time
        import sys
        import argparse
        from numpy import random
        # from models.experimental import attempt_load
        # from utils.datasets import LoadStreams
        from Utils import non_max_suppression, scale_coords  # , set_logging
        # from utils.torch_utils import time_synchronized
        import models as models

        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords

        self.iou_thres = 0.45
        self.augment = False
        self.agnostic_nms = False

        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = torch.load("y7-no_dig.pt", map_location=self.device)
        self.model = self.model['ema' if self.model.get('ema') else 'model'].float().fuse().eval()

        self.classes = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        print(self.classes)

        # Compatibility updates
        for m in self.model.modules():
            if type(m) in [torch.nn.Hardswish, torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            if type(m) is torch.nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility

        self.imgsz = 640  # check img_size

        if self.half:
            self.model.half()  # to FP16

        # Run inference
        if self.device.type != 'cpu':
            with torch.no_grad():
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                    next(self.model.parameters())))  # run once

    def put_text(self, image, label, start_point, font, fontScale, color, thickness):
        cv2.putText(image, label, start_point, font, fontScale, (0, 0, 0), max(round(thickness * 1.5), 3))
        cv2.putText(image, label, start_point, font, fontScale, color, thickness)

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def preprocess_frame_YOLOv7(self, input_frames):
        self.input_frames = input_frames

        # Convert single frame to list of frames for consistency
        self.frames = [input_frames] if isinstance(input_frames,
                                                   np.ndarray) and input_frames.ndim == 3 else input_frames

        def preprocess_frame(frame_temp):
            img = self.letterbox(frame_temp, self.imgsz, stride=32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            return img

        # Preprocess all frames
        batch_tensor = torch.cat([preprocess_frame(frame) for frame in self.frames])

        return batch_tensor

    def detect_YOLOv7(self, batch_tensor):
        with torch.no_grad():
            t1 = time.time()
            batch_preds = self.model(batch_tensor, augment=self.augment)[0]
            t2 = time.time()
            inference_time = round((t2 - t1) * 1000) / len(batch_tensor)
            # print("Inference Time:", inference_time, "ms")

            # Ensure batch_preds is always a list of tensors
            if batch_tensor.shape[0] == 1:  # Only one image
                batch_preds = [batch_preds]

            # Apply NMS to batch predictions
            t3 = time.time()
            batch_preds = [
                self.non_max_suppression(pred, self.min_score, self.iou_thres, classes=None,
                                         agnostic=self.agnostic_nms) for pred in batch_preds
            ]
            t4 = time.time()
            nms_time = round((t4 - t3) * 1000) / len(batch_tensor)
            # print("NMS Time:", nms_time, "ms")

        batch_coordinates, batch_scores, batch_class_indexes, batch_inference_times = [], [], [], []
        for pred in batch_preds:
            for i, det in enumerate(pred):
                coordinates, scores, class_indexes = [], [], []
                if len(det):
                    scaled_coords = self.scale_coords(batch_tensor.shape[2:], det[:, :4], self.frames[i].shape)
                    det[:, :4] = scaled_coords.round()
                    for *xyxy, conf, cls in reversed(det):
                        x1, y1, x2, y2 = map(int, xyxy)
                        coordinates.append([x1, y1, x2, y2])
                        scores.append(conf.item())
                        class_indexes.append(int(cls))

                coordinates, scores, class_indexes = np.array(coordinates), np.array(scores), np.array(class_indexes)

                batch_coordinates.append(coordinates)
                batch_scores.append(scores)
                batch_class_indexes.append(class_indexes)
                batch_inference_times.append(inference_time)

        if isinstance(self.input_frames,
                      np.ndarray) and self.input_frames.ndim == 3:  # If single frame passed instead of batched
            batch_coordinates, batch_scores = batch_coordinates[0], batch_scores[0]
            batch_class_indexes, batch_inference_times = batch_class_indexes[0], batch_inference_times[0]
        return batch_coordinates, batch_scores, batch_class_indexes, batch_inference_times

    def get_aruco_matrix(self, frame, depth_frame_t=None):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.marker_corners, self.marker_ids, reject = self.aruco_detector.detectMarkers(frame)
        if self.marker_ids is not None:
            marker_corners_temp, marker_ids_temp = [], []
            for marker_id_index, marker_id in enumerate(self.marker_ids):
                marker_id_temp = int(marker_id)  # Convert marker_id to int
                if marker_id_temp == 227:
                    marker_corners_temp.append(self.marker_corners[marker_id_index])
                    marker_ids_temp.append(marker_id)
            self.marker_corners, self.marker_ids = tuple(marker_corners_temp), np.array(marker_ids_temp)

        self.aruco_rvecs, self.aruco_tvecs = [], []
        for corners in self.marker_corners:
            corners_temp = corners[0]

            # Calculate the centroid
            # print(f"corners: {corners}")
            centroid = np.mean(corners_temp, axis=0)
            # print(f"centroid: {centroid}")

            # Function to calculate angle
            def calculate_angle(point):
                return math.atan2(point[1] - centroid[1], point[0] - centroid[0])

            # Sort corners based on angle
            sorted_corners = np.array( sorted(corners_temp, key=calculate_angle, reverse=True) , dtype=np.float32)
            # print(f"sorted_corners: {sorted_corners}")
            sorted_corners = np.array([sorted_corners], dtype=np.float32)

            aruco_ret, aruco_rvec, aruco_tvec = cv2.solvePnP(self.marker_points, corners, self.intrinsics_rgb, self.dist_coeffs_rgb)

            self.aruco_rvecs.append(aruco_rvec)
            self.aruco_tvecs.append(aruco_tvec)

        if len(self.marker_corners) > 0 and len(self.marker_ids) > 0:
            # Draw detected markers
            aruco.drawDetectedMarkers(frame, self.marker_corners, self.marker_ids)

            for i in range(len(self.aruco_rvecs)):
                cv2.drawFrameAxes(frame, self.intrinsics_rgb, self.dist_coeffs_rgb, self.aruco_rvecs[i], self.aruco_tvecs[i], 2 * self.marker_size)
                # time.sleep(0.3)

    def annotate_results(self, frame, coordinates, class_indexes, depth_frame):
        # print(f"frame shape: {frame.shape}")
        # print(f"depth_frame shape: {depth_frame.shape}")
        # ... [Annotating logic goes here]

        # depth_frame = depth_frame * 6/4.4

        # Convert data to PyTorch tensors and move to GPU
        depth_frame_t = torch.from_numpy(depth_frame.astype(np.float32)).to(self.device)

        self.get_aruco_matrix(frame, depth_frame_t=depth_frame_t)

        coords_T_list, center_xy_TL_list, center_xy_TR_list, center_xy_TB_list = [], [], [], []
        center_xy_bucket_list = []

        # Prepares font, thickness and other scaling relative to resolution
        height_image = frame.shape[0]
        fontScale = max((0.0007 * height_image + 0.1174), 0.5)
        txt_thk = max(round(0.0015 * height_image - 0.0183), 1)
        bb_thk = max(round(0.0007 * height_image + 1.4908), 1)
        y_adj_mult = max(round(0.0138 * height_image + 0.0183), 10)
        y_adj = max(round(0.0069 * height_image + 2.0183), 5)

        # Draws coordinates
        self.labels = ""
        self.violation_coords = []
        self.violation_coords_temp = []
        for coordinate_index, coordinate in enumerate(coordinates):
            should_draw = True
            should_draw_box = False
            label_current = self.classes[class_indexes[coordinate_index]]

            self.violation_coords.append(coordinate)
            # if active_point_current:
            self.violation_coords_temp.append(coordinate)

            center_xy = (int((coordinate[0] + coordinate[2]) / 2), int((coordinate[1] + coordinate[3]) / 2))
            # center_bottom_xy = ( int( (coordinate[0] + coordinate[2])/2 ), int( coordinate[3] ) )

            if label_current == "T":
                color = (255, 255, 255)
                coords_T_list.append(coordinate)
                should_draw_box = True
            elif label_current == "TL":
                color = (0, 0, 255)
                center_xy_TL_list.append(center_xy)
            elif label_current == "TR":
                color = (255, 0, 0)
                center_xy_TR_list.append(center_xy)
            elif label_current == "TB":
                color = (0, 255, 0)
                center_xy_TB_list.append(center_xy)
            elif label_current == "Bucket_Bottom" or label_current == "Bucket":
                if label_current == "Bucket_Bottom":
                    color = (0, 255, 255)
                    center_xy_bucket_list.append(center_xy)
                    should_draw = True
                    should_draw_box = True
                elif label_current == "Bucket":
                    color = (0, 255, 255)
                    center_xy_bucket_list.append(center_xy)
                    should_draw = True
                    should_draw_box = True
            else:
                color = (255, 255, 255)
                should_draw = False
                should_draw_box = False

            if should_draw:
                if should_draw_box:
                    self.put_text(frame, label_current,
                                  (int(coordinate[0]), int(coordinate[1])),
                                  cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, txt_thk
                                  )
                    cv2.rectangle(frame,
                                  (int(coordinate[0]) - 5, int(coordinate[1]) - y_adj),
                                  (int(coordinate[2]) + 5, int(coordinate[3]) + y_adj), color,
                                  bb_thk
                                  )
                else:
                    cv2.circle(frame, center_xy, 10, color, -1)

        # if len(coords_T_list) > 0 and len(center_xy_TL_list) > 0 and len(center_xy_TR_list) > 0 and len(center_xy_TB_list) > 0:
        # Get the height of the T detection
        t1 = time.time()
        # frame, transformation_matrix = self.get_anchor_points(frame, depth_frame, center_xy_T, center_xy_TL, center_xy_TR, center_xy_TB)
        frame, transformation_matrix = self.draw_no_dig(frame, depth_frame, coords_T_list, center_xy_TL_list,
                                                        center_xy_TR_list, center_xy_TB_list,
                                                        center_xy_bucket_list)
        t2 = time.time()
        print(f"draw_no_dig Time: {round((t2 - t1) * 1000)} ms!")

        self.aruco_rvecs_previous = self.aruco_rvecs
        self.aruco_tvecs_previous = self.aruco_tvecs

        if transformation_matrix is not None:
            return frame, self.map_points_now, transformation_matrix
        else:
            return frame, None, None

    def get_info_from_vehicle_coords(self, center_xy_bucket_list, depth_frame_t):

        center_spatial_bucket_list = []

        for center_xy_bucket in center_xy_bucket_list:
            # Convert other data to tensors
            center_bottom_vehicle_t = torch.tensor(center_xy_bucket, device=self.device, dtype=torch.float32)

            # Compute the spatial coordinates of the T mark
            coords_spatial_center_bottom_vehicle = self.to_spatial(*center_bottom_vehicle_t, depth_frame_t[
                center_bottom_vehicle_t[1].long() - 1, center_bottom_vehicle_t[0].long() - 1])
            if coords_spatial_center_bottom_vehicle is None:
                return []

            center_spatial_bucket_list.append(coords_spatial_center_bottom_vehicle)

        return center_spatial_bucket_list

    def get_info_from_coords(self, center_xy_TL, center_xy_TR, center_xy_TB, depth_frame_t):
        # Convert other data to tensors
        center_xy_TL_t = torch.tensor(center_xy_TL, device=self.device, dtype=torch.float32)
        center_xy_TR_t = torch.tensor(center_xy_TR, device=self.device, dtype=torch.float32)
        center_xy_TB_t = torch.tensor(center_xy_TB, device=self.device, dtype=torch.float32)

        # Compute the spatial coordinates of the T mark
        coords_TL = self.to_spatial(*center_xy_TL_t,
                                    depth_frame_t[center_xy_TL_t[1].long(), center_xy_TL_t[0].long()])
        coords_TR = self.to_spatial(*center_xy_TR_t,
                                    depth_frame_t[center_xy_TR_t[1].long(), center_xy_TR_t[0].long()])
        coords_TB = self.to_spatial(*center_xy_TB_t,
                                    depth_frame_t[center_xy_TB_t[1].long(), center_xy_TB_t[0].long()])
        if not all(coords is not None for coords in (coords_TL, coords_TR, coords_TB)):
            return None, None, None, None, None, None

        # Assuming center_xy_TL and center_xy_TR are torch tensors
        center_xy_TL_np = np.array(center_xy_TL)
        center_xy_TR_np = np.array(center_xy_TR)
        average_xy_TLTR_np = (center_xy_TL_np + center_xy_TR_np) / 2
        average_xy_TLTR = tuple(average_xy_TLTR_np)
        # Now convert the averaged image coordinates to spatial coordinates
        average_xy_TLTR_t = torch.tensor(average_xy_TLTR, device=self.device, dtype=torch.float32)
        center_spatial_TL_TB = self.to_spatial(*average_xy_TLTR_t,
                                               depth_frame_t[average_xy_TLTR_t[1].long(), average_xy_TLTR_t[0].long()])
        if center_spatial_TL_TB is None:
            return None, None, None, None, None, None

        # Compute two vectors lying in the plane
        vector1 = coords_TR - coords_TL
        vector2 = coords_TB - coords_TL

        # Compute the normal to the plane
        normal = torch.cross(vector1, vector2)
        normal = normal / torch.linalg.norm(normal)  # Normalize the normal

        return coords_TL, coords_TR, coords_TB, average_xy_TLTR, center_spatial_TL_TB, normal

    def get_normal_vector_from_coords(self, center_xy_TL, center_xy_TR, center_xy_TB, depth_frame_t):
        # Convert other data to tensors
        center_xy_TL_t = torch.tensor(center_xy_TL, device=self.device, dtype=torch.float32)
        center_xy_TR_t = torch.tensor(center_xy_TR, device=self.device, dtype=torch.float32)
        center_xy_TB_t = torch.tensor(center_xy_TB, device=self.device, dtype=torch.float32)

        if (center_xy_TL_t[0] < 0 or center_xy_TL_t[1] < 0 or center_xy_TR_t[0] < 0 or center_xy_TR_t[1] < 0 or center_xy_TB_t[0] < 0 or center_xy_TB_t[1] < 0
        or center_xy_TL_t[1] > (self.h-1) or center_xy_TL_t[0] > (self.w-1) or center_xy_TR_t[1] > (self.h-1) or center_xy_TR_t[0] > (self.w-1) or center_xy_TB_t[1] > (self.h-1) or center_xy_TB_t[0] > (self.w-1)
        ):
            return None

        if (
                    depth_frame_t[center_xy_TL_t[1].long(), center_xy_TL_t[0].long()] <= 5  # Arbitrary values for noisy depth that was removed
                and depth_frame_t[center_xy_TR_t[1].long(), center_xy_TR_t[0].long()] <= 5
                and depth_frame_t[center_xy_TB_t[1].long(), center_xy_TB_t[0].long()] <= 5
        ):
            return None

        # Compute the spatial coordinates of the T mark
        coords_TL = self.to_spatial(*center_xy_TL_t,
                                    depth_frame_t[center_xy_TL_t[1].long(), center_xy_TL_t[0].long()])
        coords_TR = self.to_spatial(*center_xy_TR_t,
                                    depth_frame_t[center_xy_TR_t[1].long(), center_xy_TR_t[0].long()])
        coords_TB = self.to_spatial(*center_xy_TB_t,
                                    depth_frame_t[center_xy_TB_t[1].long(), center_xy_TB_t[0].long()])
        if not all(coords is not None for coords in (coords_TL, coords_TR, coords_TB)):
            return None

        # Assuming center_xy_TL and center_xy_TR are torch tensors
        average_xy_TLTR_t = (center_xy_TL_t + center_xy_TR_t) / 2
        center_spatial_TL_TB = self.to_spatial(*average_xy_TLTR_t, depth_frame_t[average_xy_TLTR_t[1].long(), average_xy_TLTR_t[0].long()])
        if center_spatial_TL_TB is None:
            return None

        # Compute two vectors lying in the plane
        vector1 = coords_TR - coords_TL
        vector2 = coords_TB - coords_TL

        # Compute the normal to the plane
        normal = torch.cross(vector1, vector2)
        normal = normal / torch.linalg.norm(normal)  # Normalize the normal

        if len(self.last_few_special_vectors) > 0:
            mean_vector = torch.mean(torch.stack(self.last_few_special_vectors), dim=0)
            std_vector = torch.std(torch.stack(self.last_few_special_vectors), dim=0)
            lower_bound = mean_vector - 2 * std_vector
            upper_bound = mean_vector + 2 * std_vector

            if torch.all(normal >= lower_bound) and torch.all(normal <= upper_bound):  # Filters out outliers!
                self.last_few_special_vectors.append(normal)
                while len(self.last_few_special_vectors) > 3:
                    self.last_few_special_vectors.pop(0)

                average_normal = torch.mean(torch.stack(self.last_few_special_vectors), dim=0)

            else:  # I still want this added, I just don't want it in the returned average calculations!
                average_normal = torch.mean(torch.stack(self.last_few_special_vectors), dim=0)

                self.last_few_special_vectors.append(normal)
                while len(self.last_few_special_vectors) > 3:
                    self.last_few_special_vectors.pop(0)
        else:
            average_normal = normal
            self.last_few_special_vectors.append(normal)
            while len(self.last_few_special_vectors) > 3:
                self.last_few_special_vectors.pop(0)

        return average_normal

    def get_rvec_from_coords(self, corners, depth_frame):

        corners = corners[0]

        depth_frame_t = torch.from_numpy(depth_frame.astype(np.float32)).to(self.device)
        # Assume corners is a list or array with shape (4, 2),
        # where corners[0] is the top-left corner,
        # corners[1] is the top-right corner,
        # corners[2] is the bottom-right corner,
        # and corners[3] is the bottom-left corner.

        # Convert corners to tensors
        corners_t = torch.tensor(corners, device=self.device, dtype=torch.float32)

        # Compute the spatial coordinates of the corners
        coords_TL = self.to_spatial(*corners_t[0], depth_frame_t[corners_t[0][1].long(), corners_t[0][0].long()])
        coords_TR = self.to_spatial(*corners_t[1], depth_frame_t[corners_t[1][1].long(), corners_t[1][0].long()])
        coords_BL = self.to_spatial(*corners_t[3], depth_frame_t[corners_t[3][1].long(), corners_t[3][0].long()])

        # Compute two vectors lying in the plane
        vector1 = coords_TR - coords_TL
        vector2 = coords_BL - coords_TL

        # Normalize vector1 and vector2
        vector1 = vector1 / torch.linalg.norm(vector1)
        vector2 = vector2 / torch.linalg.norm(vector2)

        # Compute the normal to the plane
        normal = torch.cross(vector1, vector2)
        normal = normal / torch.linalg.norm(normal)  # Normalize the normal

        # Convert the vectors to numpy arrays
        vector1_np = vector1.cpu().numpy()
        vector2_np = vector2.cpu().numpy()
        normal_np = normal.cpu().numpy()

        # Create the rotation matrix
        rotation_matrix = np.column_stack((vector1_np, vector2_np, normal_np))

        # Convert the rotation matrix to a rotation vector
        rvec, _ = cv2.Rodrigues(rotation_matrix)

        return rvec

    def draw_zones_using_T_Marks(self, frame):
        for coords_T_index, coords_T in enumerate(self.coords_T_list):
            coords_T, center_xy_TL, center_xy_TR, center_xy_TB = coords_T, self.center_xy_TL_list_arranged[
                coords_T_index], self.center_xy_TR_list_arranged[coords_T_index], self.center_xy_TB_list_arranged[
                coords_T_index]

            coords_TL, coords_TR, coords_TB, average_xy_TLTR, center_spatial_TL_TB, normal = self.get_info_from_coords(
                center_xy_TL, center_xy_TR, center_xy_TB, self.depth_frame_t)
            if coords_TL is None:
                return frame

            center_spatial_bucket_list = self.get_info_from_vehicle_coords(self.center_xy_bucket_list,
                                                                                   self.depth_frame_t)

            if self.normal_special is not None:
                normal = self.normal_special

            if self.marker_center is not None:
                # Draw the center location and where it is pointing to relative to the ArUco marker
                # For simplicity, assume the red zone center is the same as the T detection center
                center_image_TL_TB = self.to_image(*center_spatial_TL_TB)
                if not torch.isnan(center_image_TL_TB).any():

                    red_zone_center = center_image_TL_TB

                    # Convert the tensor back to a numpy array on the CPU
                    red_zone_center_np = red_zone_center.cpu().numpy()

                    # Convert the numpy array to a tuple of integers
                    red_zone_center_tuple = tuple(map(int, red_zone_center_np))

                    # Draw red zone center and line to ArUco marker center
                    cv2.circle(frame, red_zone_center_tuple, 10, (0, 0, 255), -1)
                    cv2.line(frame, red_zone_center_tuple, tuple(self.marker_center.astype(int)), (0, 0, 255), 2)

                    distance = torch.linalg.norm(center_spatial_TL_TB - self.center_spatial_marker)
                    center_point = tuple(((np.array(self.marker_center.astype(int)) + np.array(red_zone_center_tuple)) / 2).astype(int))
                    self.put_text(frame, f"{round(float(distance*0.0032808), 1)} ft",
                                  center_point,
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
                                  )

            # Project the coordinates of the T mark onto the normal to get the height along the zed axis
            avg_height = torch.dot(center_spatial_TL_TB, normal)

            # Create an empty mask
            mask_2 = torch.zeros((self.h, self.w, 4), device=self.device, dtype=torch.uint8)

            # Create coordinate grids
            # Suggested code
            u, v = torch.meshgrid(torch.arange(self.w, device=self.device), torch.arange(self.h, device=self.device))
            Z = self.depth_frame_t[v, u]
            coords = torch.stack(((u - self.cx) * Z / self.fx,
                                  (v - self.cy) * Z / self.fy,
                                  Z), dim=-1)

            mask_Z = (abs(Z) >= 5)

            # Project the coordinates onto the normal to get the height along the zed axis
            zed = torch.sum(coords * normal, dim=-1)

            # Check if the zed coordinate (height) is within the desired height range
            within_height_range_2 = ((avg_height - self.height_tolerance_2 <= zed) &
                                     (zed <= avg_height + self.height_tolerance_2) &
                                     mask_Z)

            within_height_range_for_bucket = False
            condition_for_vehicle = False

            if center_spatial_bucket_list:
                center_spatial_bucket_tensor = torch.stack(center_spatial_bucket_list)
                bucket_center_height_locations = torch.matmul(center_spatial_bucket_tensor, normal)
                within_height_range_for_bucket = ((avg_height - self.height_tolerance_2 <= bucket_center_height_locations) &
                                                  (bucket_center_height_locations <= avg_height + self.height_tolerance_2))
                matched_indices = torch.nonzero(within_height_range_for_bucket, as_tuple=True)[0]
                if len(matched_indices) > 0:
                    center_xy_bucket = self.center_xy_bucket_list[matched_indices[0].item()]

            # Calculate the slope (m) and y-intercept (b) of the line
            x1, y1 = torch.tensor(center_xy_TL, device=self.device)
            x2, y2 = torch.tensor(center_xy_TR, device=self.device)
            m = torch.where(x2 != x1, (y2 - y1) / (x2 - x1), torch.tensor(float('nan'), device=self.device))
            b = y1 - m * x1

            # Determine the side of the line on which center_xy_TB lies
            x, y = torch.tensor(center_xy_TB, device=self.device)
            line_y = m * x + b
            above_line_tensor = torch.where(~torch.isnan(m), y > line_y, x > x1)

            # Ensure the tensor has the correct shape for broadcasting in subsequent operations
            above_line_tensor = above_line_tensor.unsqueeze(0).unsqueeze(0)

            # Check if the pixel is on the desired side of the line
            # Suggested code
            on_desired_side = torch.where(
                ~torch.isnan(m),
                (above_line_tensor & (v > m * u + b)) | (~above_line_tensor & (v < m * u + b)),
                (above_line_tensor & (u > x1)) | (~above_line_tensor & (u < x1))
            )

            # Ensure the conditions are boolean
            on_desired_side = on_desired_side.bool()

            # Combine the conditions
            condition_2 = within_height_range_2 & on_desired_side

            if within_height_range_for_bucket:
                condition_for_vehicle = condition_2[center_xy_bucket[0], center_xy_bucket[1]].bool()

            condition_2 = condition_2.t()

            # Expand the dimensions of the condition tensor to match the mask tensor
            condition_2_expanded = condition_2.unsqueeze(-1).expand_as(mask_2)

            # print(f"condition_for_vehicle: {condition_for_vehicle}")
            if condition_for_vehicle:
                color = [0, 0, 255, 50]
            else:
                color = [255, 0, 0, 50]

            # Set the mask values based on the condition
            color_tensor_2 = torch.tensor(color, device=self.device, dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
            # Set the mask values based on the condition for height_tolerance_1
            mask_2 = torch.where(condition_2_expanded, color_tensor_2, mask_2)

            # When you need to convert a tensor back to a NumPy array, first move it to the CPU
            mask_2_np = mask_2.cpu().numpy()

            # Blend the masks with the original image
            frame = cv2.addWeighted(frame, 1, mask_2_np[..., :3], 0.5, 0)

        return frame

    def draw_no_dig(self, frame, depth_frame, coords_T_list, center_xy_TL_list, center_xy_TR_list, center_xy_TB_list,
                    center_xy_bucket_list):
        # frame_copy = frame.copy()
        coords_T_list_temp = coords_T_list
        normal_special = None  # This is the normal vector that will go across the plane ground
        self.marker_center = None
        center_xy_to_use_list = []
        scaling_factor = 1
        # Convert height tolerance to millimeters
        # height_tolerance_1 = 0.0 / 0.0032808  # 0.8 feet to mm
        height_tolerance_2 = 0.5 / 0.0032808  # 1.0 feet to mm
        height_tolerance_2 = torch.tensor(height_tolerance_2, device='cuda', dtype=torch.float32)

        # Getting the dimensions of the frame
        h, w = frame.shape[:2]
        h_depth, w_depth = depth_frame.shape

        self.height_tolerance_2, self.h, self.w = height_tolerance_2, h_depth, w_depth

        # Convert data to PyTorch tensors and move to GPU
        # depth_frame_t = torch.from_numpy(depth_frame.astype(np.float32)).to(self.device)
        # Whenever you work with depth_map coordinates
        depth_frame_t = torch.from_numpy(depth_frame.astype(np.float32)).to(self.device) * scaling_factor

        center_xy_TL_list_arranged, center_xy_TR_list_arranged, center_xy_TB_list_arranged, t_indexes_to_pop = [], [], [], []
        widen_tolerance = 5
        for coords_T_index, coords_T in enumerate(coords_T_list):
            has_all_keypoints = False
            has_tl = False
            has_tr = False
            has_tb = False
            for center_xy_TL in center_xy_TL_list:
                if (coords_T[0] - widen_tolerance) < center_xy_TL[0] < (coords_T[2] + widen_tolerance) and (
                        coords_T[1] - widen_tolerance) < center_xy_TL[1] < (coords_T[3] + widen_tolerance):
                    center_xy_TL_list_arranged.append(center_xy_TL)
                    has_tl = True
            for center_xy_TR in center_xy_TR_list:
                if (coords_T[0] - widen_tolerance) < center_xy_TR[0] < (coords_T[2] + widen_tolerance) and (
                        coords_T[1] - widen_tolerance) < center_xy_TR[1] < (coords_T[3] + widen_tolerance):
                    center_xy_TR_list_arranged.append(center_xy_TR)
                    has_tr = True
            for center_xy_TB in center_xy_TB_list:
                if (coords_T[0] - widen_tolerance) < center_xy_TB[0] < (coords_T[2] + widen_tolerance) and (
                        coords_T[1] - widen_tolerance) < center_xy_TB[1] < (coords_T[3] + widen_tolerance):
                    center_xy_TB_list_arranged.append(center_xy_TB)
                    has_tb = True

            if has_tl and has_tr and has_tb:
                # print(f"coords_T_index {coords_T_index} is good!")
                pass  # Everythings all good!
            else:
                # IF it does not have all, discard whatever was appended
                if has_tl:
                    center_xy_TL_list_arranged.pop(-1)
                if has_tr:
                    center_xy_TR_list_arranged.pop(-1)
                if has_tb:
                    center_xy_TB_list_arranged.pop(-1)

                t_indexes_to_pop.append(coords_T_index)

        for t_index_to_pop in reversed(t_indexes_to_pop):
            # print(f"t_index_to_pop: {t_index_to_pop}")
            coords_T_list.pop(t_index_to_pop)

        if len(coords_T_list_temp) >= 3:
            max_distance = 0
            selected_triplet = None
            for i in range(len(coords_T_list_temp)):
                for j in range(i + 1, len(coords_T_list_temp)):
                    for k in range(j + 1, len(coords_T_list_temp)):
                        triplet = (coords_T_list_temp[i], coords_T_list_temp[j], coords_T_list_temp[k])
                        pairwise_sum = (np.linalg.norm(triplet[0] - triplet[1]) +
                                        np.linalg.norm(triplet[1] - triplet[2]) +
                                        np.linalg.norm(triplet[0] - triplet[2]))
                        if pairwise_sum > max_distance:
                            max_distance = pairwise_sum
                            selected_triplet = triplet

            center_xy_to_use_list = [(int((coord[0] + coord[2]) / 2), int((coord[1] + coord[3]) / 2)) for coord in
                                     selected_triplet]

            normal_special_temp = self.get_normal_vector_from_coords(center_xy_to_use_list[0], center_xy_to_use_list[1],
                                                                     center_xy_to_use_list[2], depth_frame_t)
            if normal_special_temp is not None:
                normal_special = normal_special_temp

        elif len(coords_T_list_temp) >= 0:
            for special_index in range(len(coords_T_list_temp)):
                coordinate = coords_T_list_temp[special_index]
                center_xy = (int((coordinate[0] + coordinate[2]) / 2), int((coordinate[1] + coordinate[3]) / 2))
                center_xy_to_use_list.append(center_xy)

        self.coords_T_list, self.center_xy_TL_list_arranged, self.center_xy_TR_list_arranged = coords_T_list, center_xy_TL_list_arranged, center_xy_TR_list_arranged
        self.center_xy_TB_list_arranged, self.center_xy_bucket_list, self.depth_frame_t = center_xy_TB_list_arranged, center_xy_bucket_list, depth_frame_t
        self.normal_special = normal_special

        # Check if there are detected ArUco markers
        if self.marker_ids is not None and len(self.marker_ids) > 0:
        # if self.marker_ids is not None and len(self.marker_ids) > 0:
            # Iterate through each detected marker
            for idx, marker_id in enumerate(self.marker_ids):
                marker_id = int(marker_id)  # Convert marker_id to int
                if marker_id == 227:

                    marker_center = np.mean(self.marker_corners[idx][0], axis=0)
                    marker_center_t = torch.from_numpy(marker_center).float().to(self.device)
                    center_spatial_marker = self.to_spatial(*marker_center_t, depth_frame_t[
                        marker_center_t[1].long(), marker_center_t[0].long()])
                    if center_spatial_marker is None:
                        transformation_matrix = None
                        return frame, transformation_matrix

                    self.marker_center = marker_center
                    self.center_spatial_marker = center_spatial_marker

                    aruco_rvec, aruco_tvec = self.aruco_rvecs[idx], self.aruco_tvecs[idx]

                    if marker_id not in self.saved_zones and len(coords_T_list) > 0:
                        self.saved_zones[marker_id] = {'offset_vector': [], 'coords_TL': [], 'coords_TR': [],
                                                       'coords_TB': [], 'center_spatial': [], 'normal': [],
                                                       'aruco_rvec': [], 'aruco_tvec': [], 'marker_center': []}

                        for coords_T_index, coords_T in enumerate(coords_T_list):
                            coords_T, center_xy_TL, center_xy_TR, center_xy_TB = coords_T, center_xy_TL_list_arranged[
                                coords_T_index], center_xy_TR_list_arranged[coords_T_index], center_xy_TB_list_arranged[
                                coords_T_index]

                            coords_TL, coords_TR, coords_TB, average_xy_TLTR, center_spatial_TL_TB, normal = self.get_info_from_coords(
                                center_xy_TL, center_xy_TR, center_xy_TB, depth_frame_t)
                            if coords_TL is None:
                                transformation_matrix = None
                                return frame, transformation_matrix

                            offset_vector = center_spatial_TL_TB - center_spatial_marker

                            # Save Zone Section
                            self.saved_zones[marker_id]['offset_vector'].append(offset_vector.tolist())
                            self.saved_zones[marker_id]['coords_TL'].append(coords_TL.tolist())
                            self.saved_zones[marker_id]['coords_TR'].append(coords_TR.tolist())
                            self.saved_zones[marker_id]['coords_TB'].append(coords_TB.tolist())
                            self.saved_zones[marker_id]['center_spatial'].append(center_spatial_TL_TB.tolist())
                            self.saved_zones[marker_id]['normal'].append(normal.tolist())
                            self.saved_zones[marker_id]['aruco_rvec'].append(aruco_rvec.tolist())
                            self.saved_zones[marker_id]['aruco_tvec'].append(aruco_tvec.tolist())
                            self.saved_zones[marker_id]['marker_center'].append(center_spatial_marker.tolist())
                    else:
                        saved_zone_indexes_to_not_draw = []
                        saved_zone_indexes_to_draw = []
                        potential_indexes_list = []
                        potential_offset_vector_list = []
                        potential_coords_TL_list = []
                        potential_coords_TR_list = []
                        potential_coords_TB_list = []
                        potential_center_spatial_list = []
                        potential_normal_list = []
                        potential_aruco_rvec_list = []
                        potential_aruco_tvec_list = []
                        potential_marker_center_list = []

                        for coords_T_index, coords_T in enumerate(coords_T_list):
                            coords_T, center_xy_TL, center_xy_TR, center_xy_TB = coords_T, center_xy_TL_list_arranged[
                                coords_T_index], center_xy_TR_list_arranged[coords_T_index], center_xy_TB_list_arranged[
                                coords_T_index]

                            coords_TL, coords_TR, coords_TB, average_xy_TLTR, center_spatial_TL_TB, normal = self.get_info_from_coords(
                                center_xy_TL, center_xy_TR, center_xy_TB, depth_frame_t)
                            if coords_TL is None:
                                transformation_matrix = None
                                return frame, transformation_matrix

                            # Check saved zones for a match with the current ArUco marker
                            saved_zone = self.saved_zones.get(marker_id)
                            # print(f"saved_zone: {saved_zone}")
                            if saved_zone is not None:

                                saved_zone_coords_TL_list = saved_zone['coords_TL']
                                saved_zone_coords_TR_list = saved_zone['coords_TR']
                                saved_zone_center_spatial_list = saved_zone['center_spatial']
                                saved_zone_offset_vector_list = saved_zone['offset_vector']

                                for saved_zone_center_spatial_index, saved_zone_center_spatial in enumerate(
                                        saved_zone_center_spatial_list):
                                    saved_zone_coords_TL = saved_zone_coords_TL_list[saved_zone_center_spatial_index]
                                    saved_zone_coords_TR = saved_zone_coords_TR_list[saved_zone_center_spatial_index]

                                    coords_TL_memory = saved_zone_coords_TL
                                    coords_TL_memory = torch.tensor(coords_TL_memory, device='cuda',
                                                                    dtype=torch.float32)
                                    coords_TR_memory = saved_zone_coords_TR
                                    coords_TR_memory = torch.tensor(coords_TR_memory, device='cuda',
                                                                    dtype=torch.float32)

                                    # Stack coords_TL and coords_TR along a new dimension
                                    coords_tensor_memory = torch.stack([coords_TL_memory, coords_TR_memory], dim=0)

                                    # Now compute the mean along the new dimension (dim=0)
                                    red_zone_center_spatial_memory = torch.mean(coords_tensor_memory, dim=0)
                                    distance = torch.linalg.norm(center_spatial_TL_TB - red_zone_center_spatial_memory)

                                    # If detected T mark is around 1 foot (assuming distance is in millimeters)
                                    # print(f"distance: {distance}")
                                    if abs(distance) < (1.5 / 0.0032808): # This determines how far T mark has to be from another
                                        saved_zone_indexes_to_draw.append(coords_T_index)
                                        saved_zone_indexes_to_not_draw.append(saved_zone_center_spatial_index)
                                        # print(f"distance: {distance}")
                                    else:
                                        # print(f"Distance that IS HERE: {distance}")
                                        # print(f"Length of saved_zone_center_spatial_list: {len(saved_zone_center_spatial_list)}")
                                        # Save Zone Section
                                        # Assuming center_xy_TL and center_xy_TR are torch tensors
                                        center_xy_TL_np = np.array(center_xy_TL)
                                        center_xy_TR_np = np.array(center_xy_TR)
                                        average_xy_TLTR_np = (center_xy_TL_np + center_xy_TR_np) / 2
                                        average_xy_TLTR = tuple(average_xy_TLTR_np)

                                        # Now convert the averaged frame coordinates to spatial coordinates
                                        average_xy_TLTR_t = torch.tensor(average_xy_TLTR, device=self.device,
                                                                         dtype=torch.float32)
                                        center_spatial_TL_TB = self.to_spatial(*average_xy_TLTR_t, depth_frame_t[
                                            average_xy_TLTR_t[1].long(), average_xy_TLTR_t[0].long()])
                                        if center_spatial_TL_TB is None:
                                            transformation_matrix = None
                                            return frame, transformation_matrix

                                        offset_vector = center_spatial_TL_TB - center_spatial_marker

                                        potential_indexes_list.append(coords_T_index)
                                        potential_offset_vector_list.append(offset_vector.tolist())
                                        potential_coords_TL_list.append(coords_TL.tolist())
                                        potential_coords_TR_list.append(coords_TR.tolist())
                                        potential_coords_TB_list.append(coords_TB.tolist())
                                        potential_center_spatial_list.append(center_spatial_TL_TB.tolist())
                                        potential_normal_list.append(normal.tolist())
                                        potential_aruco_rvec_list.append(aruco_rvec.tolist())
                                        potential_aruco_tvec_list.append(aruco_tvec.tolist())
                                        potential_marker_center_list.append(center_spatial_marker.tolist())

                        saved_zone = self.saved_zones.get(marker_id)
                        if saved_zone is not None:
                            saved_zone_coords_TL_list = saved_zone.get('coords_TL')
                            if saved_zone_coords_TL_list is not None:
                                saved_zone_offset_vector_list = saved_zone['offset_vector']
                                saved_zone_coords_TR_list = saved_zone['coords_TR']
                                saved_zone_coords_TB_list = saved_zone['coords_TB']
                                saved_zone_center_spatial_list = saved_zone['center_spatial']
                                saved_zone_normal_list = saved_zone['normal']
                                saved_zone_aruco_rvec_list = saved_zone['aruco_rvec']
                                saved_zone_aruco_tvec_list = saved_zone['aruco_tvec']
                                saved_zone_marker_center_list = saved_zone['marker_center']

                                if len(saved_zone_indexes_to_draw) > 0:
                                    should_draw_from_T = True  # This indicates if the script should draw using the T mark or saved points in memory

                                else:
                                    saved_zone_indexes_to_draw = []
                                    for saved_zone_index_to_draw in range(len(saved_zone_coords_TL_list)):
                                        saved_zone_indexes_to_draw.append(saved_zone_index_to_draw)
                                    should_draw_from_T = False

                                saved_index_to_use_list = []

                                if (len(saved_zone_center_spatial_list) + len(coords_T_list_temp) >= 3) and len(coords_T_list_temp) < 3 and not should_draw_from_T:

                                    saved_zone_index_to_not_use_list = []
                                    potential_index_to_use_list = []
                                    potential_center_xy_to_use_list = []
                                    for saved_zone_index_to_draw_index, saved_zone_index_to_draw in enumerate(saved_zone_indexes_to_draw):
                                        if len(center_xy_to_use_list) == 3:
                                            break

                                        saved_zone_coords_TL = saved_zone_coords_TL_list[
                                            saved_zone_index_to_draw_index]
                                        saved_zone_coords_TR = saved_zone_coords_TR_list[
                                            saved_zone_index_to_draw_index]

                                        coords_TL_memory = saved_zone_coords_TL
                                        coords_TL_memory = torch.tensor(coords_TL_memory, device='cuda',
                                                                        dtype=torch.float32)
                                        coords_TR_memory = saved_zone_coords_TR
                                        coords_TR_memory = torch.tensor(coords_TR_memory, device='cuda',
                                                                        dtype=torch.float32)

                                        # Stack coords_TL and coords_TR along a new dimension
                                        coords_tensor_memory = torch.stack([coords_TL_memory, coords_TR_memory], dim=0)

                                        # Now compute the mean along the new dimension (dim=0)
                                        red_zone_center_spatial_memory = torch.mean(coords_tensor_memory, dim=0)

                                        for center_xy in center_xy_to_use_list:
                                            center_xy_t = torch.tensor(center_xy, device=self.device, dtype=torch.float32)
                                            center_spatial_T = self.to_spatial(*center_xy_t, depth_frame_t[center_xy_t[1].long(), center_xy_t[0].long()])
                                            if center_spatial_T is None:
                                                break

                                            distance = torch.linalg.norm(center_spatial_T - red_zone_center_spatial_memory)

                                            # If detected T mark is around 1 foot (assuming distance is in millimeters)
                                            # print(f"distance: {distance}")
                                            if abs(distance) > (1.5 / 0.0032808):
                                                saved_zone_offset_vector = saved_zone_offset_vector_list[saved_zone_index_to_draw_index]
                                                saved_zone_offset_vector = torch.tensor(saved_zone_offset_vector, device='cuda', dtype=torch.float32)
                                                saved_zone_aruco_rvec = saved_zone_aruco_rvec_list[
                                                    saved_zone_index_to_draw_index]

                                                delta_rotation = aruco_rvec - saved_zone_aruco_rvec
                                                delta_rmat, _ = cv2.Rodrigues(delta_rotation)
                                                delta_rmat = torch.tensor(delta_rmat, device='cuda', dtype=torch.float32)
                                                center_spatial_TL_TB_memory = (delta_rmat @ (saved_zone_offset_vector)) + center_spatial_marker

                                                # Draw the center location and where it is pointing to relative to the ArUco marker
                                                # For simplicity, assume the red zone center is the same as the T detection center
                                                center_image_TL_TB_memory = self.to_image(*center_spatial_TL_TB_memory)


                                                coordinate = center_image_TL_TB_memory.cpu().numpy()
                                                center_xy = coordinate

                                                potential_index_to_use_list.append(saved_zone_index_to_draw_index)
                                                potential_center_xy_to_use_list.append(center_xy)

                                                if len(center_xy_to_use_list) >= 5:
                                                    break
                                            else:
                                                saved_zone_index_to_not_use_list.append(saved_zone_index_to_draw_index)

                                    for potential_index_to_use_inxex, potential_index_to_use in enumerate(potential_index_to_use_list):
                                        if potential_index_to_use in saved_zone_index_to_not_use_list:
                                            continue
                                        center_xy_to_use_list.append(potential_center_xy_to_use_list[potential_index_to_use_inxex])

                                        saved_index_to_use_list.append(potential_index_to_use)

                                    if len(center_xy_to_use_list) >= 3:
                                        max_distance = 0
                                        selected_triplet = None
                                        for i in range(len(center_xy_to_use_list)):
                                            for j in range(i + 1, len(center_xy_to_use_list)):
                                                for k in range(j + 1, len(center_xy_to_use_list)):
                                                    triplet = (center_xy_to_use_list[i], center_xy_to_use_list[j],
                                                               center_xy_to_use_list[k])
                                                    pairwise_sum = (np.linalg.norm(
                                                        np.array(triplet[0]) - np.array(triplet[1])) +
                                                                    np.linalg.norm(
                                                                        np.array(triplet[1]) - np.array(triplet[2])) +
                                                                    np.linalg.norm(
                                                                        np.array(triplet[0]) - np.array(triplet[2])))
                                                    if pairwise_sum > max_distance:
                                                        max_distance = pairwise_sum
                                                        selected_triplet = triplet

                                        center_xy_to_use_list = [(int(coord[0]), int(coord[1])) for coord in
                                                                 selected_triplet]

                                        normal_special_temp = self.get_normal_vector_from_coords(center_xy_to_use_list[0],
                                                                                            center_xy_to_use_list[1],
                                                                                            center_xy_to_use_list[2],
                                                                                            depth_frame_t)
                                        if normal_special_temp is not None:
                                            normal_special = normal_special_temp
                                            self.normal_special = normal_special

                                frame = self.draw_zones_using_T_Marks(frame)

                                if not should_draw_from_T:
                                    for saved_zone_index_to_draw_index, saved_zone_index_to_draw in enumerate(saved_zone_indexes_to_draw):
                                        center_spatial_TL_TB_memory = torch.tensor(saved_zone_center_spatial_list[saved_zone_index_to_draw_index], device='cuda', dtype=torch.float32)
                                        saved_zone_offset_vector = saved_zone_offset_vector_list[saved_zone_index_to_draw_index]
                                        # saved_zone_offset_vector = 0
                                        saved_zone_offset_vector = torch.tensor(saved_zone_offset_vector, device='cuda', dtype=torch.float32)
                                        saved_zone_coords_TL = saved_zone_coords_TL_list[saved_zone_index_to_draw_index]
                                        saved_zone_coords_TR = saved_zone_coords_TR_list[saved_zone_index_to_draw_index]
                                        saved_zone_coords_TB = saved_zone_coords_TB_list[saved_zone_index_to_draw_index]
                                        saved_zone_normal = saved_zone_normal_list[saved_zone_index_to_draw_index]
                                        normal_memory = torch.tensor(saved_zone_normal, device='cuda', dtype=torch.float32)
                                        saved_zone_aruco_rvec = saved_zone_aruco_rvec_list[saved_zone_index_to_draw_index]
                                        saved_zone_aruco_tvec = saved_zone_aruco_tvec_list[saved_zone_index_to_draw_index]
                                        saved_zone_marker_center = saved_zone_marker_center_list[saved_zone_index_to_draw_index]
                                        saved_zone_marker_center = torch.tensor(saved_zone_marker_center, device='cuda', dtype=torch.float32)

                                        delta_rotation = aruco_rvec - saved_zone_aruco_rvec
                                        delta_rmat, _ = cv2.Rodrigues(delta_rotation)
                                        delta_rmat = torch.tensor(delta_rmat, device='cuda', dtype=torch.float32)
                                        delta_translation = saved_zone_marker_center - center_spatial_TL_TB_memory
                                        delta_translation = aruco_tvec - saved_zone_aruco_tvec
                                        # delta_translation = torch.tensor(delta_translation, device='cuda',dtype=torch.float32)
                                        delta_translation = torch.tensor(delta_translation, device='cuda',
                                                                         dtype=torch.float32).squeeze()
                                        center_spatial_TL_TB_memory = (
                                                                                  delta_rmat @ saved_zone_offset_vector) + center_spatial_marker

                                        coords_TL_memory = saved_zone_coords_TL
                                        coords_TL_memory = torch.tensor(coords_TL_memory, device='cuda',
                                                                        dtype=torch.float32)
                                        coords_TR_memory = saved_zone_coords_TR
                                        coords_TR_memory = torch.tensor(coords_TR_memory, device='cuda',
                                                                        dtype=torch.float32)
                                        coords_TB_memory = saved_zone_coords_TB
                                        coords_TB_memory = torch.tensor(coords_TB_memory, device='cuda',
                                                                        dtype=torch.float32)

                                        coords_TL_memory = delta_rmat @ (
                                                    coords_TL_memory - saved_zone_marker_center) + center_spatial_marker
                                        center_xy_TL_memory = self.to_image(*coords_TL_memory)
                                        coords_TR_memory = delta_rmat @ (
                                                    coords_TR_memory - saved_zone_marker_center) + center_spatial_marker
                                        center_xy_TR_memory = self.to_image(*coords_TR_memory)
                                        coords_TB_memory = delta_rmat @ (
                                                    coords_TB_memory - saved_zone_marker_center) + center_spatial_marker
                                        center_xy_TB_memory = self.to_image(*coords_TB_memory)

                                        if normal_special is not None:
                                            normal_memory = normal_special
                                        else:
                                            scaling_factor_coords = 1.5
                                            # Assume center_xy_TL_memory, center_xy_TR_memory, and center_xy_TB_memory are your coordinates
                                            coords = torch.stack(
                                                [center_xy_TL_memory, center_xy_TR_memory, center_xy_TB_memory])

                                            # Find the centroid of the coordinates
                                            centroid = torch.mean(coords, dim=0)

                                            # Calculate the vector from the centroid to each coordinate
                                            vectors = coords - centroid

                                            # Scale the vectors to spread out the coordinates and increase the max radius
                                            scaled_vectors = vectors * scaling_factor_coords  # scaling_factor_coords > 1 will spread the coordinates out

                                            # Calculate the new coordinates
                                            new_coords = centroid + scaled_vectors

                                            normal_special_temp = self.get_normal_vector_from_coords(
                                                new_coords[0].cpu(),
                                                new_coords[1].cpu(),
                                                new_coords[2].cpu(),
                                                depth_frame_t)
                                            if normal_special_temp is not None:
                                                normal_memory = normal_special_temp
                                                self.normal_special = normal_special_temp
                                            else:
                                                normal_memory = delta_rmat @ normal_memory

                                        # Draw the center location and where it is pointing to relative to the ArUco marker
                                        # For simplicity, assume the red zone center is the same as the T detection center
                                        center_image_TL_TB_memory = self.to_image(*center_spatial_TL_TB_memory)
                                        red_zone_center_memory = center_image_TL_TB_memory

                                        # Convert the tensor back to a numpy array on the CPU
                                        red_zone_center_np_memory = red_zone_center_memory.cpu().numpy()
                                        # Convert the numpy array to a tuple of integers
                                        red_zone_center_tuple_memory = tuple(map(int, red_zone_center_np_memory))

                                        # Draw red zone center and line to ArUco marker center
                                        cv2.circle(frame, red_zone_center_tuple_memory, 10, (0, 0, 255), -1)
                                        cv2.line(frame, red_zone_center_tuple_memory, tuple(marker_center.astype(int)),
                                                 (0, 0, 255), 2)

                                        distance = torch.linalg.norm(
                                            center_spatial_TL_TB_memory - center_spatial_marker)
                                        center_point = tuple(
                                            ((np.array(marker_center.astype(int)) + np.array(
                                                red_zone_center_tuple_memory)) / 2).astype(int))
                                        self.put_text(frame, f"{round(float(distance * 0.0032808), 1)} ft",
                                                      center_point,
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
                                                      )

                                        avg_height = torch.dot(center_spatial_TL_TB_memory, normal_memory)

                                        # Create an empty mask
                                        mask_2 = torch.zeros((h, w, 4), device=self.device, dtype=torch.uint8)

                                        # Create coordinate grids
                                        u, v = torch.meshgrid(torch.arange(self.w, device=self.device),
                                                              torch.arange(self.h, device=self.device))
                                        Z = self.depth_frame_t[v, u]
                                        coords = torch.stack(((u - self.cx) * Z / self.fx,
                                                              (v - self.cy) * Z / self.fy,
                                                              Z), dim=-1)

                                        mask_Z = (abs(Z) >= 5)

                                        # Project the coordinates onto the normal to get the height along the zed axis
                                        zed = torch.sum(coords * normal_memory, dim=-1)

                                        # Check if the zed coordinate (height) is within the desired height range
                                        within_height_range_2 = ((avg_height - self.height_tolerance_2 <= zed) &
                                                                 (zed <= avg_height + self.height_tolerance_2) &
                                                                 mask_Z)

                                        center_spatial_bucket_list = self.get_info_from_vehicle_coords(
                                            center_xy_bucket_list,
                                            depth_frame_t)

                                        within_height_range_for_bucket = False
                                        condition_for_vehicle = False

                                        if center_spatial_bucket_list:
                                            center_spatial_bucket_tensor = torch.stack(center_spatial_bucket_list)
                                            bucket_center_height_locations = torch.matmul(center_spatial_bucket_tensor,
                                                                                          normal_memory)
                                            within_height_range_for_bucket = ((
                                                                                      avg_height - height_tolerance_2 <= bucket_center_height_locations) &
                                                                              (
                                                                                      bucket_center_height_locations <= avg_height + height_tolerance_2))
                                            matched_indices = \
                                                torch.nonzero(within_height_range_for_bucket, as_tuple=True)[0]
                                            if len(matched_indices) > 0:
                                                center_xy_bucket = center_xy_bucket_list[matched_indices[0].item()]

                                            # Calculate the slope (m) and y-intercept (b) of the line
                                        x1, y1 = torch.tensor(center_xy_TL_memory, device=self.device)
                                        x2, y2 = torch.tensor(center_xy_TR_memory, device=self.device)
                                        m = torch.where(x2 != x1, (y2 - y1) / (x2 - x1),
                                                        torch.tensor(float('nan'), device=self.device))
                                        b = y1 - m * x1

                                        # Determine the side of the line on which center_xy_TB_memory lies
                                        x, y = torch.tensor(center_xy_TB_memory, device=self.device)
                                        line_y = m * x + b
                                        above_line_tensor = torch.where(~torch.isnan(m), y > line_y, x > x1)

                                        # Ensure the tensor has the correct shape for broadcasting in subsequent operations
                                        above_line_tensor = above_line_tensor.unsqueeze(0).unsqueeze(0)

                                        # Check if the pixel is on the desired side of the line
                                        # Suggested code
                                        on_desired_side = torch.where(
                                            ~torch.isnan(m),
                                            (above_line_tensor & (v > m * u + b)) | (
                                                        ~above_line_tensor & (v < m * u + b)),
                                            (above_line_tensor & (u > x1)) | (~above_line_tensor & (u < x1))
                                        )

                                        # Ensure the conditions are boolean
                                        on_desired_side = on_desired_side.bool()

                                        # Combine the conditions
                                        condition_2 = within_height_range_2 & on_desired_side

                                        if within_height_range_for_bucket:
                                            condition_for_vehicle = condition_2[
                                                center_xy_bucket[0], center_xy_bucket[1]].bool()

                                        condition_2 = condition_2.t()

                                        # Expand the dimensions of the condition tensor to match the mask tensor
                                        condition_2_expanded = condition_2.unsqueeze(-1).expand_as(mask_2)

                                        # print(f"condition_for_vehicle: {condition_for_vehicle}")
                                        if condition_for_vehicle:
                                            color = [0, 0, 255, 50]
                                        else:
                                            color = [255, 0, 0, 50]

                                        # Set the mask values based on the condition
                                        color_tensor_2 = torch.tensor(color, device=self.device,
                                                                      dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
                                        # Set the mask values based on the condition for height_tolerance_1
                                        mask_2 = torch.where(condition_2_expanded, color_tensor_2, mask_2)

                                        # When you need to convert a tensor back to a NumPy array, first move it to the CPU
                                        mask_2_np = mask_2.cpu().numpy()

                                        # Blend the masks with the original image
                                        frame = cv2.addWeighted(frame, 1, mask_2_np[..., :3], 0.5, 0)
                                else:
                                    for saved_zone_center_spatial_index, saved_zone_center_spatial in enumerate(saved_zone_center_spatial_list):
                                        if saved_zone_center_spatial_index in saved_zone_indexes_to_not_draw:
                                            continue
                                        center_spatial_TL_TB_memory = torch.tensor(saved_zone_center_spatial_list[saved_zone_center_spatial_index], device='cuda', dtype=torch.float32)
                                        saved_zone_offset_vector = saved_zone_offset_vector_list[saved_zone_center_spatial_index]
                                        # saved_zone_offset_vector = 0
                                        saved_zone_offset_vector = torch.tensor(saved_zone_offset_vector, device='cuda', dtype=torch.float32)
                                        saved_zone_coords_TL = saved_zone_coords_TL_list[saved_zone_center_spatial_index]
                                        saved_zone_coords_TR = saved_zone_coords_TR_list[saved_zone_center_spatial_index]
                                        saved_zone_coords_TB = saved_zone_coords_TB_list[saved_zone_center_spatial_index]
                                        saved_zone_normal = saved_zone_normal_list[saved_zone_center_spatial_index]
                                        normal_memory = torch.tensor(saved_zone_normal, device='cuda', dtype=torch.float32)
                                        saved_zone_aruco_rvec = saved_zone_aruco_rvec_list[saved_zone_center_spatial_index]
                                        saved_zone_aruco_tvec = saved_zone_aruco_tvec_list[saved_zone_center_spatial_index]
                                        saved_zone_marker_center = saved_zone_marker_center_list[saved_zone_center_spatial_index]
                                        saved_zone_marker_center = torch.tensor(saved_zone_marker_center, device='cuda', dtype=torch.float32)

                                        delta_rotation = aruco_rvec - saved_zone_aruco_rvec
                                        delta_rmat, _ = cv2.Rodrigues(delta_rotation)
                                        delta_rmat = torch.tensor(delta_rmat, device='cuda', dtype=torch.float32)
                                        delta_translation = saved_zone_marker_center - center_spatial_TL_TB_memory
                                        delta_translation = aruco_tvec - saved_zone_aruco_tvec
                                        # delta_translation = torch.tensor(delta_translation, device='cuda',dtype=torch.float32)
                                        delta_translation = torch.tensor(delta_translation, device='cuda',
                                                                         dtype=torch.float32).squeeze()
                                        center_spatial_TL_TB_memory = (
                                                                                  delta_rmat @ saved_zone_offset_vector) + center_spatial_marker

                                        coords_TL_memory = saved_zone_coords_TL
                                        coords_TL_memory = torch.tensor(coords_TL_memory, device='cuda',
                                                                        dtype=torch.float32)
                                        coords_TR_memory = saved_zone_coords_TR
                                        coords_TR_memory = torch.tensor(coords_TR_memory, device='cuda',
                                                                        dtype=torch.float32)
                                        coords_TB_memory = saved_zone_coords_TB
                                        coords_TB_memory = torch.tensor(coords_TB_memory, device='cuda',
                                                                        dtype=torch.float32)

                                        coords_TL_memory = delta_rmat @ (
                                                    coords_TL_memory - saved_zone_marker_center) + center_spatial_marker
                                        center_xy_TL_memory = self.to_image(*coords_TL_memory)
                                        coords_TR_memory = delta_rmat @ (
                                                    coords_TR_memory - saved_zone_marker_center) + center_spatial_marker
                                        center_xy_TR_memory = self.to_image(*coords_TR_memory)
                                        coords_TB_memory = delta_rmat @ (
                                                    coords_TB_memory - saved_zone_marker_center) + center_spatial_marker
                                        center_xy_TB_memory = self.to_image(*coords_TB_memory)

                                        if normal_special is not None:
                                            normal_memory = normal_special
                                        else:
                                            scaling_factor_coords = 1.5
                                            # Assume center_xy_TL_memory, center_xy_TR_memory, and center_xy_TB_memory are your coordinates
                                            coords = torch.stack(
                                                [center_xy_TL_memory, center_xy_TR_memory, center_xy_TB_memory])

                                            # Find the centroid of the coordinates
                                            centroid = torch.mean(coords, dim=0)

                                            # Calculate the vector from the centroid to each coordinate
                                            vectors = coords - centroid

                                            # Scale the vectors to spread out the coordinates and increase the max radius
                                            scaled_vectors = vectors * scaling_factor_coords  # scaling_factor_coords > 1 will spread the coordinates out

                                            # Calculate the new coordinates
                                            new_coords = centroid + scaled_vectors

                                            normal_special_temp = self.get_normal_vector_from_coords(
                                                new_coords[0].cpu(),
                                                new_coords[1].cpu(),
                                                new_coords[2].cpu(),
                                                depth_frame_t)
                                            if normal_special_temp is not None:
                                                normal_memory = normal_special_temp
                                                self.normal_special = normal_special_temp
                                            else:
                                                normal_memory = delta_rmat @ normal_memory

                                        # Draw the center location and where it is pointing to relative to the ArUco marker
                                        # For simplicity, assume the red zone center is the same as the T detection center
                                        center_image_TL_TB_memory = self.to_image(*center_spatial_TL_TB_memory)
                                        red_zone_center_memory = center_image_TL_TB_memory

                                        # Convert the tensor back to a numpy array on the CPU
                                        red_zone_center_np_memory = red_zone_center_memory.cpu().numpy()
                                        # Convert the numpy array to a tuple of integers
                                        red_zone_center_tuple_memory = tuple(map(int, red_zone_center_np_memory))

                                        # Draw red zone center and line to ArUco marker center
                                        cv2.circle(frame, red_zone_center_tuple_memory, 10, (0, 0, 255), -1)
                                        cv2.line(frame, red_zone_center_tuple_memory, tuple(marker_center.astype(int)),
                                                 (0, 0, 255), 2)

                                        distance = torch.linalg.norm(
                                            center_spatial_TL_TB_memory - center_spatial_marker)
                                        center_point = tuple(
                                            ((np.array(marker_center.astype(int)) + np.array(
                                                red_zone_center_tuple_memory)) / 2).astype(int))
                                        self.put_text(frame, f"{round(float(distance * 0.0032808), 1)} ft",
                                                      center_point,
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
                                                      )

                                        avg_height = torch.dot(center_spatial_TL_TB_memory, normal_memory)

                                        # Create an empty mask
                                        mask_2 = torch.zeros((h, w, 4), device=self.device, dtype=torch.uint8)

                                        # Create coordinate grids
                                        u, v = torch.meshgrid(torch.arange(self.w, device=self.device),
                                                              torch.arange(self.h, device=self.device))
                                        Z = self.depth_frame_t[v, u]
                                        coords = torch.stack(((u - self.cx) * Z / self.fx,
                                                              (v - self.cy) * Z / self.fy,
                                                              Z), dim=-1)

                                        mask_Z = (abs(Z) >= 5)

                                        # Project the coordinates onto the normal to get the height along the zed axis
                                        zed = torch.sum(coords * normal_memory, dim=-1)

                                        # Check if the zed coordinate (height) is within the desired height range
                                        within_height_range_2 = ((avg_height - self.height_tolerance_2 <= zed) &
                                                                 (zed <= avg_height + self.height_tolerance_2) &
                                                                 mask_Z)

                                        center_spatial_bucket_list = self.get_info_from_vehicle_coords(
                                            center_xy_bucket_list,
                                            depth_frame_t)

                                        within_height_range_for_bucket = False
                                        condition_for_vehicle = False

                                        if center_spatial_bucket_list:
                                            center_spatial_bucket_tensor = torch.stack(center_spatial_bucket_list)
                                            bucket_center_height_locations = torch.matmul(center_spatial_bucket_tensor,
                                                                                          normal_memory)
                                            within_height_range_for_bucket = ((
                                                                                      avg_height - height_tolerance_2 <= bucket_center_height_locations) &
                                                                              (
                                                                                      bucket_center_height_locations <= avg_height + height_tolerance_2))
                                            matched_indices = \
                                                torch.nonzero(within_height_range_for_bucket, as_tuple=True)[0]
                                            if len(matched_indices) > 0:
                                                center_xy_bucket = center_xy_bucket_list[matched_indices[0].item()]

                                            # Calculate the slope (m) and y-intercept (b) of the line
                                        x1, y1 = torch.tensor(center_xy_TL_memory, device=self.device)
                                        x2, y2 = torch.tensor(center_xy_TR_memory, device=self.device)
                                        m = torch.where(x2 != x1, (y2 - y1) / (x2 - x1),
                                                        torch.tensor(float('nan'), device=self.device))
                                        b = y1 - m * x1

                                        # Determine the side of the line on which center_xy_TB_memory lies
                                        x, y = torch.tensor(center_xy_TB_memory, device=self.device)
                                        line_y = m * x + b
                                        above_line_tensor = torch.where(~torch.isnan(m), y > line_y, x > x1)

                                        # Ensure the tensor has the correct shape for broadcasting in subsequent operations
                                        above_line_tensor = above_line_tensor.unsqueeze(0).unsqueeze(0)

                                        # Check if the pixel is on the desired side of the line
                                        # Suggested code
                                        on_desired_side = torch.where(
                                            ~torch.isnan(m),
                                            (above_line_tensor & (v > m * u + b)) | (
                                                    ~above_line_tensor & (v < m * u + b)),
                                            (above_line_tensor & (u > x1)) | (~above_line_tensor & (u < x1))
                                        )

                                        # Ensure the conditions are boolean
                                        on_desired_side = on_desired_side.bool()

                                        # Combine the conditions
                                        condition_2 = within_height_range_2 & on_desired_side

                                        if within_height_range_for_bucket:
                                            condition_for_vehicle = condition_2[
                                                center_xy_bucket[0], center_xy_bucket[1]].bool()

                                        condition_2 = condition_2.t()

                                        # Expand the dimensions of the condition tensor to match the mask tensor
                                        condition_2_expanded = condition_2.unsqueeze(-1).expand_as(mask_2)

                                        # print(f"condition_for_vehicle: {condition_for_vehicle}")
                                        if condition_for_vehicle:
                                            color = [0, 0, 255, 50]
                                        else:
                                            color = [255, 0, 0, 50]

                                        # Set the mask values based on the condition
                                        color_tensor_2 = torch.tensor(color, device=self.device,
                                                                      dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
                                        # Set the mask values based on the condition for height_tolerance_1
                                        mask_2 = torch.where(condition_2_expanded, color_tensor_2, mask_2)

                                        # When you need to convert a tensor back to a NumPy array, first move it to the CPU
                                        mask_2_np = mask_2.cpu().numpy()

                                        # Blend the masks with the original image
                                        frame = cv2.addWeighted(frame, 1, mask_2_np[..., :3], 0.5, 0)


                        for potential_index_index, potential_index in enumerate(potential_indexes_list):
                            if potential_index in saved_zone_indexes_to_draw:
                                if should_draw_from_T:
                                    continue
                            self.saved_zones[marker_id]['offset_vector'].append(
                                potential_offset_vector_list[potential_index_index])
                            self.saved_zones[marker_id]['coords_TL'].append(
                                potential_coords_TL_list[potential_index_index])
                            self.saved_zones[marker_id]['coords_TR'].append(
                                potential_coords_TR_list[potential_index_index])
                            self.saved_zones[marker_id]['coords_TB'].append(
                                potential_coords_TB_list[potential_index_index])
                            self.saved_zones[marker_id]['center_spatial'].append(
                                potential_center_spatial_list[potential_index_index])
                            self.saved_zones[marker_id]['normal'].append(potential_normal_list[potential_index_index])
                            self.saved_zones[marker_id]['aruco_rvec'].append(
                                potential_aruco_rvec_list[potential_index_index])
                            self.saved_zones[marker_id]['aruco_tvec'].append(
                                potential_aruco_tvec_list[potential_index_index])
                            self.saved_zones[marker_id]['marker_center'].append(
                                potential_marker_center_list[potential_index_index])

        else:
            frame = self.draw_zones_using_T_Marks(frame)

        transformation_matrix = None
        return frame, transformation_matrix
