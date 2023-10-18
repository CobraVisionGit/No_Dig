import numpy as np
from depthai_sdk import OakCamera
from depthai_sdk.classes import DetectionPacket
import numpy as np
import cv2, PIL
import tensorflow as tf
from cv2 import aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import math

c_x = 0

c_y = 0

tag_x = 0

tag_y = 0


# Current Version V7: Detects both T's and Aruco Markers. Next step to do: angle measurement
def find_angle(point1, point2):
    # Calculate the horizontal and vertical differences between the points
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]

    # Calculate the angle using arctan2
    angle_radians = math.atan2(delta_y, delta_x)

    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)

    # Ensure the angle is in the range [0, 360)
    if angle_degrees < 0:
        angle_degrees += 360

    return angle_degrees


def calculate_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    # Calculate the angle in radians
    angle_rad = np.arctan2(y2 - y1, x2 - x1)

    # Convert the angle from radians to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def calculate_Distance(x1, y1, x2, y2):
    # Calculate the squared differences
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the squared distance
    squared_distance = dx ** 2 + dy ** 2

    # Calculate the actual distance by taking the square root
    distance = math.sqrt(squared_distance)

    return distance


# Preprocessing image for classify function
def preprocess_image(image):
    # Resize the image to (180, 180)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (180, 180))

    # Normalize the pixel values to be in the range [0, 1]
    normalized_image = resized_image / 255.0

    # Add an extra dimension to represent the batch size (1 in this case)
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image


# Function to classify T's direction
def classify(image):
    model = tf.saved_model.load('T/')
    class_names = ['T_Down', 'T_Left', 'T_Right', 'T_Up', 'bucket']

    # Preprocess the input image
    input_image = preprocess_image(image)

    # Convert the input numpy array to a TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)

    # Run the inference
    output = model(input_tensor)

    # Get the predicted class index
    predicted_class_index = np.argmax(output, axis=1)[0]

    # Get the predicted class label
    predicted_label = class_names[predicted_class_index]
    Type = predicted_label
    return Type


def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def find_midpoint(x1, y1, x2, y2):
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2
    return midpoint_x, midpoint_y


# Function to find safe zone
def find_safe_regions(T_types, T_coordinates, distance_factor):
    # Sort T_coordinates based on x-coordinate
    T_coordinates_sorted = sorted(T_coordinates, key=lambda x: x[0])
    # Calculate the average distance between consecutive T's
    avg_distance = np.mean([calculate_distance(T_coordinates_sorted[i], T_coordinates_sorted[i + 1]) for i in
                            range(len(T_coordinates_sorted) - 1)])
    # Set a threshold distance for T's in the same safe region
    threshold_distance = avg_distance * distance_factor  # Adjust this factor as needed
    # Initialize the first safe region
    current_region = [T_coordinates_sorted[0]]
    safe_regions = []

    for i in range(1, len(T_coordinates_sorted)):
        if calculate_distance(T_coordinates_sorted[i], current_region[-1]) <= threshold_distance:
            # Add T to the current safe region
            current_region.append(T_coordinates_sorted[i])
        else:
            # Start a new safe region
            safe_regions.append(current_region)
            current_region = [T_coordinates_sorted[i]]
    # Add the last safe region
    safe_regions.append(current_region)

    return safe_regions


frame_count = 0


def draw_transparent_red_rectangle(image, pt1, pt2, alpha=0.4):
    """
    Draw a transparent red rectangle on the image using alpha blending.
    """
    overlay = image.copy()
    red_color = (0, 0, 255)  # Red color in BGR
    cv2.rectangle(overlay, pt1, pt2, red_color, -1)  # Draw the filled rectangle on the overlay
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)  # Perform alpha blending


c_x = 0
c_y = 0
tag_x = 0
tag_y = 0


def process_object(frame, det, label, T_coordinates, T_types, unique_T_types, bucket_coordinates):
    mylist1 = list(det.top_left)
    mylist2 = list(det.bottom_right)
    global specified_statement_executed

    if label == "T":
        image = frame[mylist1[1]:mylist2[1], mylist1[0]:mylist2[0]]
        Type = classify(image)
        # cv2.rectangle(frame, det.top_left, det.bottom_right, (255, 0, 0), 2)
        # cv2.putText(frame, str(Type), (mylist1[0] - 5, mylist1[1] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
        #  (0, 255, 0), 1, cv2.LINE_AA)

        center_x = (mylist1[0] + mylist2[0]) // 2
        center_y = (mylist1[1] + mylist2[1]) // 2
        global c_x
        global c_y
        c_x = center_x
        c_y = center_y
        T_coordinates.append((center_x, center_y))
        T_types.append(Type)
        unique_T_types.add(Type)
        # center_x, center_y: x, y for T
        if Type == "T_Up":
            line_start = (center_x - 200, center_y)
            line_end = (center_x + 200, center_y)
            rect_width = 100  # Adjusted width of the rectangle
            rect_center_x = (line_start[0] + line_end[0]) // 2
            rect_start = (rect_center_x - rect_width // 2, line_start[1] - 150)
            rect_end = (rect_center_x + rect_width // 2, line_start[1])
            # Draw the transparent red rectangle
            draw_transparent_red_rectangle(frame, rect_start, rect_end)
            specified_statement_executed = True
        elif Type == "T_Down":
            line_start = (center_x - 200, center_y)
            line_end = (center_x + 200, center_y)
            rect_width = 100  # Adjusted width of the rectangle
            rect_center_x = (line_start[0] + line_end[0]) // 2
            rect_start = (rect_center_x - rect_width // 2, center_y)
            rect_end = (rect_center_x + rect_width // 2, center_y + 125)  # Adjusted height
            draw_transparent_red_rectangle(frame, rect_start, rect_end)
            specified_statement_executed = True


        elif Type == "T_Left":
            line_start = (center_x, center_y - 250)
            line_end = ([center_x, center_y + 250])
            # cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
            rect_width = 200  # Adjusted width of the rectangle
            rect_center_y = (line_start[1] + line_end[1]) // 2
            rect_start = (line_start[0] - rect_width, rect_center_y - rect_width // 2)
            rect_end = (line_start[0], rect_center_y + rect_width // 2)
            draw_transparent_red_rectangle(frame, rect_start, rect_end)
            specified_statement_executed = True

        elif Type == "T_Right":
            line_start = (center_x, center_y - 250)
            line_end = ([center_x, center_y + 250])
            # cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
            rect_width = 200  # Adjusted width of the rectangle
            rect_center_y = (line_start[1] + line_end[1]) // 2
            rect_start = (line_start[0], rect_center_y - rect_width // 2)
            rect_end = (line_start[0] + rect_width, rect_center_y + rect_width // 2)
            # Draw he transparent red rectangle
            draw_transparent_red_rectangle(frame, rect_start, rect_end)

            specified_statement_executed = True

    elif label == "Bucket":
        bucket_coordinates.append((mylist1, mylist2))
        # cv2.rectangle(frame, det.top_left, det.bottom_right, (0, 255, 0), 2)
        # cv2.putText(frame, "Bucket", (mylist1[0], mylist1[1] - 10), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)


calib_data_path = "C:/work/No_Dig/Final_main - Copy/MultiMatrix.npz"

calib_data = np.load(calib_data_path)
print(calib_data.files)

cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
r_vectors = calib_data["rVector"]
t_vectors = calib_data["tVector"]

MARKER_SIZE = 8  # centimeters

marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)  # aruco.Dictionary_get(aruco.DICT_4X4_50)

param_markers = cv2.aruco.DetectorParameters()  # aruco.DetectorParameters_create()

# with OakCamera(replay='case1.jpg') as oak:
with OakCamera() as oak:
    color = oak.create_camera('color')
    det = oak.create_nn('best_openvino_2022.1_7shave.blob', color, nn_type='yolo')
    det.config_yolo(2, 4,
                    [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0,
                     373.0, 326.0],
                    {"side52": [0, 1, 2], "side26": [3, 4, 5], "side13": [6, 7, 8]},
                    0.5, 0.5)

    Labels = ["T", "Bucket"]
    last_frame_with_detection = None

    last_frame_with_detection = None
    display_locked_frame = False


    ###################

    ###########################
    def cb(packet: DetectionPacket, visualizer):
        global specified_statement_executed
        specified_statement_executed = False
        n = 0
        frame = packet.frame
        T_coordinates = []
        T_types = []
        bucket_coordinates = []  # List to store bucket coordinates
        unique_T_types = set()  # Keep track of unique T types
        det_num = 0
        ################################################################
        # ret, frame = packet.frame #cap.read()
        # if not ret:
        # break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_IDs, reject = aruco.detectMarkers(
            gray_frame, marker_dict, parameters=param_markers
        )
        if marker_corners:
            rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
                marker_corners, MARKER_SIZE, cam_mat, dist_coef
            )
            # rVec[i], tVec[i]: x, y for Tag

            # print (rVec, tVec)
            total_markers = range(0, marker_IDs.size)
            for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
                cv2.polylines(
                    frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA
                )
                corners = corners.reshape(4, 2)
                corners = corners.astype(int)
                top_right = corners[0].ravel()
                top_left = corners[1].ravel()
                bottom_right = corners[2].ravel()
                bottom_left = corners[3].ravel()
                global tag_x
                global tag_y

                tag_x, tag_y = find_midpoint(corners[1].ravel()[0], corners[1].ravel()[1], corners[2].ravel()[0],
                                             corners[2].ravel()[1])

                # print((corners[0].ravel()))
                # tag_x= corners[0].ravel()[0]#rVec[0]
                # tag_y=corners[0].ravel()[0]#tVec[0]

                # Calculating the distance
                distance = np.sqrt(
                    tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2
                )
                # Draw the pose of the marker
                point = cv2.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)

                cv2.putText(
                    frame,
                    f"id: {ids[0]} Dist: {round(distance, 2)}",
                    top_right,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.3,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"x:{round(tVec[i][0][0], 1)} y: {round(tVec[i][0][1], 1)} ",
                    bottom_right,
                    cv2.FONT_HERSHEY_PLAIN,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                # print(ids, "  ", corners)

        ###########################################################

        for det, d in zip(packet.detections, packet.img_detections.detections):
            n += 1
            label = Labels[int(det.label)]
            process_object(frame, det, label, T_coordinates, T_types, unique_T_types, bucket_coordinates)
            det_num += 1

            distance1 = calculate_distance((c_x, c_y), (tag_x, tag_y))  # calculate_distance((tag_x, tag_y), (c_x, c_y))
            # print(distance1) #dactual istance is distance1-40
            angle1 = find_angle((c_x, c_y), (tag_x, tag_y))
            print(angle1)
            # v2.resizeWindow(packet.name, 600, new_height)
            # frame=cv2.resize(frame, (500, 500))

        # if specified_statement_executed:
        # Display the frame if a specified statement was executed
        # v2.resizeWindow(packet.name, 600, new_height)
        frame = cv2.resize(frame, (500, 500))
        cv2.imshow(packet.name, frame)
        # else:
        #    pass

        # cv2.imshow(packet.name, frame)


    oak.visualize(det, callback=cb, fps=True)
    # oak.callback(imu.out.main, callback=cb2)

    oak.start(blocking=True)