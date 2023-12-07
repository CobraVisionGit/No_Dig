import matplotlib.pyplot as plt
from rplidar import RPLidar
import math
import numpy as np

PORT_NAME = 'COM3'
lidar = RPLidar(PORT_NAME, 115200, 3)


def plot_points(points):
    # Assuming points is a list of (angle, distance) tuples
    angles = [point[1] for point in points]
    distances = [point[2] for point in points]

    # Convert to Cartesian coordinates. Here we change the sign for the y-coordinate.
    x = [dist * math.cos(math.radians(angle)) for angle, dist in zip(angles, distances)]
    y = [-dist * math.sin(math.radians(angle)) for angle, dist in zip(angles, distances)]  # Note the negative sign here

    return x, y


try:
    print('Recording data...')

    # Set up the plot in interactive mode
    plt.ion()
    fig, ax = plt.subplots()

    # Create an empty scatter plot for the LIDAR points and initialize the arrow
    sc = ax.scatter([], [])  # Initial empty plot
    arrow = plt.Arrow(0, 0, 0, 0, width=1000, color='r')  # Dummy arrow, will be updated
    ax.add_patch(arrow)

    # Add a point to represent the LIDAR sensor's position
    sensor_scatter = ax.scatter([0], [0], c='blue', s=100)  # Blue point at the origin

    # Set constant plot limits to avoid rescaling
    ax.set_xlim(-10000, 10000)
    ax.set_ylim(-10000, 10000)

    for scan in lidar.iter_scans():
        x, y = plot_points(scan)

        # Get the first point's angle to update the arrow's direction
        # (assuming the first point in each scan is where the LIDAR is looking)
        if scan:
            _, angle, distance = scan[0]  # Extract the angle and distance
            arrow_dx = distance * math.cos(math.radians(angle))
            arrow_dy = distance * math.sin(math.radians(angle))

            # Remove the old arrow and add a new one
            arrow.remove()
            arrow = plt.Arrow(0, 0, arrow_dx, arrow_dy, width=1000, color='r')
            ax.add_patch(arrow)

        # Update scatter plot data
        sc.set_offsets(list(zip(x, y)))

        # Redraw the plot
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)  # Short pause to allow for updates

except KeyboardInterrupt:
    print('Stopping.')
finally:
    print('Disconnecting from RPLIDAR...')
    lidar.stop()
    lidar.disconnect()

# Keep the window open until it's closed by the user
plt.ioff()
plt.show()
