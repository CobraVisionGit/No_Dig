import cv2
import matplotlib.pyplot as plt
import numpy as np

def save_aruco_marker(marker_id):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    marker_image = aruco_dict.drawMarker(marker_id, 1000)  # Set a high resolution by specifying a large marker size
    filename = f'Aruco_Marker_{marker_id}.png'
    cv2.imwrite(filename, marker_image)
    print(f'Saved {filename}')

# Save Aruco markers with IDs 227, 228, 229, and 230
for marker_id in [227, 228, 229, 230]:
    save_aruco_marker(marker_id)