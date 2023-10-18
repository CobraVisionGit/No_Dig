#!/usr/bin/env python3

import time
from sys import maxsize

import cv2
import depthai as dai
import open3d as o3d

import os
import sys
sys.path.append('../')
from torchvision import models
import cv2
import json
import argparse
import numpy as np
from typing import Type, Tuple, List
from datetime import datetime
import time
from datetime import timedelta
from pprint import pprint
from termcolor import colored
import yaml
# import cupy as cp


from Detector import Detector

detector = Detector()
detector.setup_YOLOv7()

# User Parameters
COLOR = True
lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
toggle_open3d = False
toggle_detect = True
toggle_imu = False
toggle_isp = True

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

pipeline = dai.Pipeline()
if toggle_imu:
    imu = pipeline.create(dai.node.IMU)
    xlink_out_imu = pipeline.createXLinkOut()

    # Set xlink output stream name
    xlink_out_imu.setStreamName('imu')

    # Enable ACCELEROMETER_RAW and GYROSCOPE_RAW at 100 Hz rate
    imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200)
    # Set batch report settings
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    # Link IMU node to XLinkOut node
    imu.out.link(xlink_out_imu.input)

monoLeft = pipeline.create(dai.node.MonoCamera)
# monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)

monoRight = pipeline.create(dai.node.MonoCamera)
# monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(median)
# stereo.initialConfig.setConfidenceThreshold(80)
stereo.initialConfig.setConfidenceThreshold(200)

stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 100 # Default 400
config.postProcessing.thresholdFilter.maxRange = 20000 # Default 200000
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)
laser_dot_brightness = 1200

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# xout_disparity = pipeline.createXLinkOut()
# xout_disparity.setStreamName('disparity')
# stereo.disparity.link(xout_disparity.input)

xout_rect_left = pipeline.createXLinkOut()
xout_rect_left.setStreamName("rectified_left")
xout_rect_right = pipeline.createXLinkOut()
xout_rect_right.setStreamName("rectified_right")

if COLOR:
    xout_colorize = pipeline.createXLinkOut()
    xout_colorize.setStreamName("colorize")
    camRgb = pipeline.create(dai.node.ColorCamera)
    # camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    if toggle_isp:
        camRgb.setIspScale(1, 3)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.initialControl.setManualFocus(130)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    if toggle_isp:
        camRgb.isp.link(xout_colorize.input)
    else:
        camRgb.video.link(xout_colorize.input)
else:
    # stereo.rectifiedRight.link(xout_colorize.input)
    # pass

    xoutLeft = pipeline.create(dai.node.XLinkOut)
    xoutRight = pipeline.create(dai.node.XLinkOut)
    xoutLeft.setStreamName("left")
    xoutRight.setStreamName("right")

    stereo.syncedLeft.link(xoutLeft.input)
    stereo.syncedRight.link(xoutRight.input)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

stereo.rectifiedLeft.link(xout_rect_left.input)
stereo.rectifiedRight.link(xout_rect_right.input)

# Create a window
cv2.namedWindow('color', cv2.WND_PROP_FULLSCREEN)

# Set the window to full screen
cv2.setWindowProperty('color', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

from Host_Sync import HostSync

with dai.Device(pipeline) as device:

    device.setIrLaserDotProjectorBrightness(laser_dot_brightness)
    qs = []
    qs.append(device.getOutputQueue("depth", maxSize=1, blocking=False))
    if COLOR:
        qs.append(device.getOutputQueue("colorize", maxSize=1, blocking=False))
    else:
        qs.append(device.getOutputQueue("left", maxSize=1, blocking=False))
        qs.append(device.getOutputQueue("right", maxSize=1, blocking=False))
    qs.append(device.getOutputQueue("rectified_left", maxSize=1, blocking=False))
    qs.append(device.getOutputQueue("rectified_right", maxSize=1, blocking=False))

    if toggle_imu:
        qs.append( device.getOutputQueue("imu", maxSize=1, blocking=False) )  # Create output queue for IMU data

    if toggle_open3d:
        try:
            from projector_3d import PointCloudVisualizer
        except ImportError as e:
            raise ImportError(
                f"\033[1;5;31mError occured when importing PCL projector: {e}. Try disabling the point cloud \033[0m ")

    calibData = device.readCalibration()
    if COLOR:
        if toggle_isp:
            w, h = camRgb.getIspSize()
        else:
            w, h = camRgb.getResolutionSize()
        print(f"RGB FPS: {camRgb.getFps()}")
        print(f"monoRight FPS: {monoRight.getFps()}")
        print(f"w, h: {w}, {h}")
        if toggle_isp:
            intrinsics_rgb = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, dai.Size2f(w, h))
        else:
            intrinsics_rgb = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A)
        dist_coeffs_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
        print(f"intrinsics_rgb: {intrinsics_rgb}")
        print(f"dist_coeffs_rgb: {dist_coeffs_rgb}")

        w_mono, h_mono = monoRight.getResolutionSize()
        intrinsics_mono = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C)
        dist_coeffs_mono = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))
        print(f"intrinsics_mono: {intrinsics_mono}")
        print(f"dist_coeffs_mono: {dist_coeffs_mono}")


        # intrinsics_rgb = intrinsics_mono
        # dist_coeffs_rgb = dist_coeffs_mono
    else:
        w, h = monoRight.getResolutionSize()
        intrinsics_rgb = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C)
        dist_coeffs_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))

        w_mono, h_mono = w, h
        intrinsics_mono = intrinsics_rgb
        dist_coeffs_mono = dist_coeffs_rgb

    if toggle_open3d:
        pcl_converter = PointCloudVisualizer(intrinsics_rgb, w, h)

    detector.set_intrinsics(intrinsics_mono, dist_coeffs_mono, intrinsics_rgb, dist_coeffs_rgb)

    serial_no = device.getMxId()
    if COLOR:
        sync = HostSync(camRgb.getFps(), toggle_imu=toggle_imu, is_colored=COLOR)
    else:
        sync = HostSync(30, toggle_imu=toggle_imu, is_colored=COLOR)
    depth_vis, color, rect_left, rect_right = None, None, None, None

    # was_msgs_true = False
    was_msgs_true = False
    times_between_when_msgs_true = []
    while_index = -1

    while True:

        start_time_while_loop = time.time()
        for q in qs:
            new_msg = q.tryGet()
            if new_msg is not None:
                if "depthai.IMUData" in str(new_msg):
                    # index = 0
                    for packet in new_msg.packets:
                        # print(f"index: {index}")
                        accel_data = packet.acceleroMeter
                        gyro_data = packet.gyroscope

                        msgs = sync.add_msg("accel_data", accel_data)
                        msgs = sync.add_msg("gyro_data", gyro_data)
                        # break
                        # index += 1
                else:
                    msgs = sync.add_msg(q.getName(), new_msg)

                if msgs:
                    while_index += 1
                    was_msgs_true = True
                    # sync.display_movement() # Prints out translational movement in feet
                    start_time = time.time()

                    # print(f"msgs.keys(): {msgs.keys()}")
                    depth = msgs["depth"][1].getFrame()
                    if COLOR:
                        color = msgs["colorize"][1].getCvFrame()
                        color_frame_data_timestamp = msgs["colorize"][0]
                    else:
                        mono_left = msgs["left"][1].getCvFrame()
                        mono_right = msgs["right"][1].getCvFrame()
                        color = np.stack([mono_left] * 3, axis=-1)
                        color_frame_data_timestamp = msgs["left"][0]

                    if toggle_imu:
                        accel_data = msgs["accel_data"][1]
                        gyro_data = msgs["gyro_data"][1]
                        gyro_data_timestamp = msgs["gyro_data"][0]

                        transformation_matrix, rotation_matrix = sync.adjust_orientation(accel_data)

                    cv2.imwrite(f"Images/{while_index}.jpg", color)

                    input_frames = color.copy()
                    if toggle_detect:
                        batch_tensor = detector.preprocess_frame_YOLOv7(input_frames)
                        coordinates, scores, class_indexes, inference_time = detector.detect_YOLOv7(batch_tensor)
                        output_color, map_points, transformation_matrix = detector.annotate_results(input_frames, coordinates, class_indexes, depth)
                    else:
                        output_color = input_frames

                        if while_index > 5:
                            # if while_index > 10:
                            #     detector.initialize_object(depth.astype(np.float32))
                            output_color, map_points, transformation_matrix = detector.mapper_for_o3d(input_frames, depth)
                        else:
                            output_color = input_frames.copy()

                    cv2.putText(output_color, f"Timestamp: {color_frame_data_timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)


                    # rectified_left = msgs["rectified_left"][1].getCvFrame()
                    # rectified_right = msgs["rectified_right"][1].getCvFrame()
                    depth_vis = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depth_vis = cv2.equalizeHist(depth_vis)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_HOT)
                    # cv2.imshow("depth", depth_vis)
                    # cv2.imshow("color", color.copy())
                    cv2.imshow("color", output_color)
                    # cv2.imshow("rectified_left", rectified_left)
                    # cv2.imshow("rectified_right", rectified_right)
                    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    rows, cols = depth.shape

                    # Keeps only every other nth value in depth map
                    # depth = filter_depth(depth, 20)

                    if toggle_open3d:
                        if while_index > 5:
                            # pcl_converter.rgbd_to_projection(depth, rgb, transformation_matrix)
                            pcl_converter.update_point_cloud(rgb, map_points, transformation_matrix)
                            # pcl_converter.visualize_pcd()


                    end_time = time.time()
                    print(f"Time Lapsed inside msgs section: {round((end_time-start_time)*1000)} ms")

                else:
                    was_msgs_true = False

            else:
                was_msgs_true = False

        key = cv2.waitKey(1)
        if key == ord("s"):
            timestamp = str(int(time.time()))
            # cv2.imwrite(f"{serial_no}_{timestamp}_depth.png", depth_vis)
            # cv2.imwrite(f"{serial_no}_{timestamp}_color.png", color)
            # cv2.imwrite(f"{serial_no}_{timestamp}_rectified_left.png", rectified_left)
            # cv2.imwrite(f"{serial_no}_{timestamp}_rectified_right.png", rectified_right)
            if toggle_open3d:
                # o3d.io.write_point_cloud(f"{serial_no}_{timestamp}.pcd", pcl_converter.pcl, compressed=True)
                pass
        elif key == ord("q"):
            break


        end_time_while_loop = time.time()

        # print(f"Time Lapsed from while loop: {round((end_time_while_loop - start_time_while_loop) * 1000)} ms")


        if was_msgs_true: # This is used to get the time it takes to get msgs to be true. So this is time between these events
            end_time_while_loop_for_msgs = time.time()

            try:
                time_lapsed = round((end_time_while_loop_for_msgs - start_time_while_loop_for_msgs) * 1000)
                times_between_when_msgs_true.append(time_lapsed)
                # print(f"Time Lapsed since last time msgs == True: {time_lapsed} ms")

                added_time = 0
                for time_temp in times_between_when_msgs_true:
                    added_time += time_temp
                average_times_between_when_msgs_true = round( added_time / len(times_between_when_msgs_true) )
                # print(f"Average time Lapsed since last time msgs == True: {average_times_between_when_msgs_true} ms")

                while len(times_between_when_msgs_true) > 20:
                    times_between_when_msgs_true.pop(0)

            except Exception as e:
                print(f"An error occurred: {e}")

            start_time_while_loop_for_msgs = time.time()
