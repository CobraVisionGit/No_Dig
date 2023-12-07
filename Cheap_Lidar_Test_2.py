import open3d as o3d
import matplotlib.pyplot as plt
from rplidar import RPLidar
import math
import numpy as np
from projector_3d_lidar import PointCloudVisualizer
import threading
from queue import Queue


PORT_NAME = 'COM3'

def lidar_to_pointcloud_360(points):
    """Convert LIDAR data to a point cloud."""
    # Assuming points is a list of (angle, distance) tuples
    angles = [point[1] for point in points]
    distances = [point[2] for point in points]

    # Convert to Cartesian coordinates
    x = [dist * math.cos(math.radians(angle)) for angle, dist in zip(angles, distances)]
    y = [dist * math.sin(math.radians(angle)) for angle, dist in zip(angles, distances)]
    z = [0] * len(x)  # All points are on the ground plane (z=0)

    return np.vstack((x, y, z)).T  # Nx3 array

def lidar_to_pointcloud_180(points, angle_range=(0, 180)):
    """Convert LIDAR data to a point cloud, filtering out points outside the specified angle range."""
    # Extracting angles and distances from points
    angles = [point[1] for point in points]  # point[1] should be the angle
    distances = [point[2] for point in points]  # point[2] should be the distance

    # Preparing lists to hold valid Cartesian coordinates
    x = []
    y = []
    z = []  # All points are on the ground plane (z=0)

    for angle, dist in zip(angles, distances):
        # Check if the angle is within the desired range
        if angle_range[0] <= angle <= angle_range[1]:
            # Angle is within range, so convert polar coordinates to Cartesian and add to list
            x.append(dist * math.cos(math.radians(angle)))
            y.append(dist * math.sin(math.radians(angle)))
            z.append(0)  # This assumes a flat ground plane; adjust if your environment is different

    # Return an Nx3 array of the valid points (each row is x, y, z)
    return np.vstack((x, y, z)).T


def lidar_data_generator(PORT_NAME='COM3'):
    lidar = RPLidar(PORT_NAME, 115200, 3)

    try:
        print('Recording data...')

        for scan in lidar.iter_scans():
            # Convert LIDAR data to a point cloud
            points = lidar_to_pointcloud_180(scan)
            yield points

    except KeyboardInterrupt:
        print('Stopping.')

    finally:
        print('Disconnecting from RPLIDAR...')
        lidar.stop()
        lidar.disconnect()


def lidar_data_collector(data_queue, PORT_NAME='COM3'):
    lidar = RPLidar(PORT_NAME, 115200, 3)

    try:
        print('Recording data...')

        for scan in lidar.iter_scans():
            # Convert LIDAR data to a point cloud
            points = lidar_to_pointcloud_180(scan)
            data_queue.put(points)  # Put the data into the queue instead of yielding

    except KeyboardInterrupt:
        print('Stopping.')

    finally:
        print('Disconnecting from RPLIDAR...')
        lidar.stop()
        lidar.disconnect()



def main():
    print('Recording data...')
    visualizer = PointCloudVisualizer()

    data_queue = Queue()  # Create a queue to hold the LIDAR data

    # Start the data collection thread
    data_thread = threading.Thread(target=lidar_data_collector, args=(data_queue, 'COM3'), daemon=True)
    data_thread.start()

    try:
        # This will continuously update the visualizer with new point clouds
        while True:  # Keep running until KeyboardInterrupt
            points = data_queue.get()  # This will block until data is available
            if points.size > 0:  # Make sure there's data to display
                visualizer.update_point_cloud(points)  # Update with actual point data
    except KeyboardInterrupt:
        print("Received a keyboard interrupt. Stopping visualization.")


if __name__ == '__main__':
    data_queue = Queue()

    main()
