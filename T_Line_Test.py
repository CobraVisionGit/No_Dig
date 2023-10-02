#!/usr/bin/env python3

import time
from sys import maxsize

import cv2
import depthai as dai
import open3d as o3d

import os
import sys
sys.path.append('../')
import torch
import torchvision.transforms
from torchvision import models
import cv2
import json
import argparse
import numpy as np
from typing import Type, Tuple, List
from datetime import datetime
import time
from pprint import pprint
from termcolor import colored
import yaml

class Detector:
    def __init__(self):
        self.model = None  # model initialization
        self.classes = None
        self.min_score = 0.20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Detector started")

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
                    next( self.model.parameters() ) ) )  # run once

    def put_text(self, image, label, start_point, font, fontScale, color, thickness):
        cv2.putText(image, label, start_point, font, fontScale, (0, 0, 0), max(round(thickness * 1.5), 3) )
        cv2.putText(image, label, start_point, font, fontScale, color, thickness)

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
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
        self.frames = [input_frames] if isinstance(input_frames, np.ndarray) and input_frames.ndim == 3 else input_frames

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

        if isinstance(self.input_frames, np.ndarray) and self.input_frames.ndim == 3: # If single frame passed instead of batched
            batch_coordinates, batch_scores = batch_coordinates[0], batch_scores[0]
            batch_class_indexes, batch_inference_times = batch_class_indexes[0], batch_inference_times[0]
        return batch_coordinates, batch_scores, batch_class_indexes, batch_inference_times

    def annotate_results(self, frame, coordinates, class_indexes):
        # ... [Annotating logic goes here]
        center_xy_TL, center_xy_TR, center_xy_TB = None, None, None

        if len(coordinates) > 0:
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
                label_current = self.classes[ class_indexes[coordinate_index] ]

                self.violation_coords.append(coordinate)
                # if active_point_current:
                self.violation_coords_temp.append(coordinate)

                center_xy = ( int( (coordinate[0] + coordinate[2])/2 ), int( (coordinate[1] + coordinate[3])/2 ) )

                if label_current == "TL":
                    # color = (0, 0, 255)
                    color = (0, 0, 255)
                    center_xy_TL = center_xy
                elif label_current == "TR":
                    color = (255, 0, 0)
                    center_xy_TR = center_xy
                elif label_current == "TB":
                    color = (0, 255, 0)
                    center_xy_TB = center_xy
                else:
                    color = (255, 255, 255)

                self.put_text(frame, label_current,
                                       (int(coordinate[0]), int(coordinate[1]) ),
                                       cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, txt_thk)
                cv2.rectangle(frame,
                              (int(coordinate[0]) - 5, int(coordinate[1]) - y_adj),
                              (int(coordinate[2]) + 5, int(coordinate[3]) + y_adj), color,
                              bb_thk)

                if center_xy_TL is not None and center_xy_TR is not None and center_xy_TB is not None:
                    frame = self.draw_no_dig(frame, center_xy_TL, center_xy_TR, center_xy_TB)


        return frame

    def draw_no_dig(self, frame, center_xy_TL, center_xy_TR, center_xy_TB):
        # Load your image
        image = frame

        # Get the dimensions of the image
        height, width, _ = image.shape

        # Calculate the slope (m) and y-intercept (b) of the line
        x1, y1 = center_xy_TL
        x2, y2 = center_xy_TR
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
        else:
            m = None
            b = None

        # Create a mask with an alpha channel
        mask = np.zeros((height, width, 4), dtype=np.uint8)

        # Determine the side of the line on which center_xy_TB lies
        x, y = center_xy_TB
        if m is not None:
            line_y = m * x + b
            above_line = y < line_y
        else:
            above_line = x < x1  # For vertical line

        # Draw the filled polygon on the mask
        if above_line:  # If center_xy_TB is above the line
            if m is not None:
                pts = np.array([[0, int(m * 0 + b)], [width, int(m * width + b)], [width, height], [0, height]],
                               dtype=np.int32)
            else:
                pts = np.array([[0, 0], [width, 0], [0, height]], dtype=np.int32)
        else:  # If center_xy_TB is below the line
            if m is not None:
                pts = np.array([[0, 0], [width, 0], [width, int(m * width + b)], [0, int(m * 0 + b)]], dtype=np.int32)
            else:
                pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.int32)

        cv2.fillPoly(mask, [pts], (0, 0, 255, 127))  # Red color, 50% transparency

        # Blend the mask with the original image
        image = cv2.addWeighted(image, 1, mask[:, :, :3], 0.5, 0)

        return image

COLOR = True

lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(median)
# stereo.initialConfig.setConfidenceThreshold(255)

stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 200 # Default 400
config.postProcessing.thresholdFilter.maxRange = 2000000 # Default 200000
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# xout_disparity = pipeline.createXLinkOut()
# xout_disparity.setStreamName('disparity')
# stereo.disparity.link(xout_disparity.input)

xout_colorize = pipeline.createXLinkOut()
xout_colorize.setStreamName("colorize")
xout_rect_left = pipeline.createXLinkOut()
xout_rect_left.setStreamName("rectified_left")
xout_rect_right = pipeline.createXLinkOut()
xout_rect_right.setStreamName("rectified_right")

if COLOR:
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setIspScale(1, 3)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    camRgb.initialControl.setManualFocus(130)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    camRgb.isp.link(xout_colorize.input)
else:
    stereo.rectifiedRight.link(xout_colorize.input)

stereo.rectifiedLeft.link(xout_rect_left.input)
stereo.rectifiedRight.link(xout_rect_right.input)


class HostSync:
    def __init__(self):
        self.arrays = {}

    def add_msg(self, name, msg):
        if not name in self.arrays:
            self.arrays[name] = []
        # Add msg to array
        self.arrays[name].append({"msg": msg, "seq": msg.getSequenceNum()})

        synced = {}
        for name, arr in self.arrays.items():
            for i, obj in enumerate(arr):
                if msg.getSequenceNum() == obj["seq"]:
                    synced[name] = obj["msg"]
                    break
        # If there are 5 (all) synced msgs, remove all old msgs
        # and return synced msgs
        if len(synced) == 4:  # color, left, right, depth, nn
            # Remove old msgs
            for name, arr in self.arrays.items():
                for i, obj in enumerate(arr):
                    if obj["seq"] < msg.getSequenceNum():
                        arr.remove(obj)
                    else:
                        break
            return synced
        return False

detector = Detector()
detector.setup_YOLOv7()

with dai.Device(pipeline) as device:

    device.setIrLaserDotProjectorBrightness(1200)
    qs = []
    qs.append(device.getOutputQueue("depth", maxSize=1, blocking=False))
    qs.append(device.getOutputQueue("colorize", maxSize=1, blocking=False))
    qs.append(device.getOutputQueue("rectified_left", maxSize=1, blocking=False))
    qs.append(device.getOutputQueue("rectified_right", maxSize=1, blocking=False))

    try:
        from projector_3d import PointCloudVisualizer
    except ImportError as e:
        raise ImportError(
            f"\033[1;5;31mError occured when importing PCL projector: {e}. Try disabling the point cloud \033[0m ")

    calibData = device.readCalibration()
    if COLOR:
        w, h = camRgb.getIspSize()
        print(f"w, h: {w}, {h}")
        intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, dai.Size2f(w, h))
        print(f"intrinsics: {intrinsics}")
    else:
        w, h = monoRight.getResolutionSize()
        intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(w, h))
    pcl_converter = PointCloudVisualizer(intrinsics, w, h)

    serial_no = device.getMxId()
    sync = HostSync()
    depth_vis, color, rect_left, rect_right = None, None, None, None

    while True:
        for q in qs:
            new_msg = q.tryGet()
            if new_msg is not None:
                msgs = sync.add_msg(q.getName(), new_msg)
                if msgs:
                    depth = msgs["depth"].getFrame()
                    color = msgs["colorize"].getCvFrame()

                    input_frames = color.copy()
                    batch_tensor = detector.preprocess_frame_YOLOv7(input_frames)
                    coordinates, scores, class_indexes, inference_time = detector.detect_YOLOv7(batch_tensor)
                    color = detector.annotate_results(input_frames, coordinates, class_indexes)


                    rectified_left = msgs["rectified_left"].getCvFrame()
                    rectified_right = msgs["rectified_right"].getCvFrame()
                    depth_vis = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depth_vis = cv2.equalizeHist(depth_vis)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_HOT)
                    cv2.imshow("depth", depth_vis)
                    cv2.imshow("color", color)
                    cv2.imshow("rectified_left", rectified_left)
                    cv2.imshow("rectified_right", rectified_right)
                    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    pcl_converter.rgbd_to_projection(depth, rgb)
                    pcl_converter.visualize_pcd()

        key = cv2.waitKey(1)
        if key == ord("s"):
            timestamp = str(int(time.time()))
            cv2.imwrite(f"{serial_no}_{timestamp}_depth.png", depth_vis)
            cv2.imwrite(f"{serial_no}_{timestamp}_color.png", color)
            cv2.imwrite(f"{serial_no}_{timestamp}_rectified_left.png", rectified_left)
            cv2.imwrite(f"{serial_no}_{timestamp}_rectified_right.png", rectified_right)
            o3d.io.write_point_cloud(f"{serial_no}_{timestamp}.pcd", pcl_converter.pcl, compressed=True)
        elif key == ord("q"):
            break