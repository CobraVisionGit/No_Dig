
from datetime import datetime
import time
from datetime import timedelta
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import ahrs
from ahrs.filters import Madgwick

class HostSync:
    def __init__(self, rgb_fps, toggle_imu=False, is_colored = False):
        self.is_colored = is_colored
        self.gyro_data = None
        self.madgwick_filter = Madgwick()
        self.current_quaternion = np.array([1., 0., 0., 0.])  # Initialize the quaternion attribute
        self.arrays = {}
        self.msgs = dict()
        min_time_diff = int(1000/rgb_fps/2)
        self.toggle_imu = toggle_imu
        self.last_time = None
        self.last_time_gyro = None
        self.last_time_accel = None
        self.time_delta = None
        self.time_delta_gyro = None
        self.time_delta_accel = None
        self.cumulative_yaw = 0.0
        self.pitch = 0.0  # Pitch value
        self.roll = 0.0  # Roll value
        self.x = 0
        self.y = 0
        self.z = 0
        self.vx = 0
        self.vy = 0
        self.vz = 0
        self.cumulative_x = 0.0  # Cumulative translation in x-direction
        self.cumulative_y = 0.0  # Cumulative translation in y-direction
        self.cumulative_z = 0.0  # Cumulative translation in z-direction

    def add_msg(self, name, msg, ts=None):
        if ts is None:
            ts = msg.getTimestamp()
            # print(f"ts: {ts}")

        if not name in self.msgs:
            self.msgs[name] = []

        self.msgs[name].append((ts, msg))

        if self.toggle_imu:
            # Cumulative calc section
            if name == 'gyro_data':
                self.adjust_cumulative_variables(msg, ts, "gyro_data")
            elif name == 'accel_data':
                self.adjust_cumulative_variables(msg, ts, 'accel_data')

        synced = {}
        for name, arr in self.msgs.items():
            # Go through all stored messages and calculate the time difference to the target msg.
            # Then sort these self.msgs to find a msg that's closest to the target time, and check
            # whether it's below 17ms which is considered in-sync.
            diffs = []
            for i, (msg_ts, msg) in enumerate(arr):
                diffs.append(abs(msg_ts - ts))
            if len(diffs) == 0: break
            diffsSorted = diffs.copy()
            diffsSorted.sort()
            dif = diffsSorted[0]

            if dif < timedelta(milliseconds=15):
                # print(f'Found synced {name} with ts {msg_ts}, target ts {ts}, diff {dif}, location {diffs.index(dif)}')
                # print(diffs)
                synced[name] = diffs.index(dif)

        if self.toggle_imu:
            if self.is_colored:
                sync_needed = 6
            else:
                sync_needed = 7
        else:
            if self.is_colored:
                sync_needed = 4
            else:
                sync_needed = 5

        if len(synced) == sync_needed:  # We have 6 synced self.msgs (depth', 'colorize', 'rectified_left', 'rectified_right', 'accel_data', 'gyro_data')
            # print('--------\Synced self.msgs! Target ts', ts, )
            # Remove older self.msgs
            for name, i in synced.items():
                self.msgs[name] = self.msgs[name][i:]
            ret = {}
            for name, arr in self.msgs.items():
                ret[name] = arr.pop(0)
                # print(f'{name} msg ts: {ret[name][0]}, diff {abs(ts - ret[name][0]).microseconds / 1000}ms')
            return ret
        return False

    def adjust_cumulative_variables(self, data, data_timestamp, imu_type):
        # print(f"data_timestamp: {data_timestamp}")
        # Calculate yaw using gyroscope data
        if self.last_time_gyro is not None:
            if imu_type == "gyro_data":
                self.gyro_data = data
                time_now = data_timestamp
                self.time_delta_gyro = time_now - self.last_time_gyro
                # yaw_rate = -(gyro_data.x + 0.0018) # Assuming gyro_data.z provides the rate of change of yaw
                data_x = data.x + 0.001571
                if abs(data_x) < 0.10:
                    data_x = 0
                # print(f"gyro_x: {data_x}")
                yaw_rate = -(data_x) # Assuming gyro_data.z provides the rate of change of yaw
                # print(f"self.time_delta: {self.time_delta}")
                yaw_change = yaw_rate * self.time_delta_gyro.total_seconds()
                self.cumulative_yaw += yaw_change

        self.last_time_gyro = data_timestamp

        if self.last_time_accel is not None:
            if imu_type == 'accel_data':
                time_now = data_timestamp
                # print(f"time_now: {time_now}")
                # print(f"last_time_accel: {self.last_time_accel}")
                self.time_delta_accel = time_now - self.last_time_accel
                # print(f"time_delta_accel: {self.time_delta_accel}")

                # Normalize accelerometer data
                # print(f"accel_data: {data.x}, {data.y}, {data.z}")
                ax, ay, az = data.z + 0.111643, data.y + 0.106552, data.x - 0.086204
                # ax, ay, az = data.z, data.y , data.x
                # print(f"ax: {ax}, ay: {ay}, az: {az}")

                gyro_data = np.array([self.gyro_data.x, self.gyro_data.y, self.gyro_data.z])
                accel_data_hehe = np.array([data.z, data.y, data.x])

                # Update the quaternion
                self.current_quaternion = self.madgwick_filter.updateIMU(q=self.current_quaternion, gyr=gyro_data, acc=accel_data_hehe)

                rotation = Rotation.from_quat(self.current_quaternion)
                rotation_matrix = rotation.as_matrix()
                gravity_sensor_frame = np.dot(rotation_matrix, np.array([0, 0, 9.81]))
                ax -= gravity_sensor_frame[0]
                ay -= gravity_sensor_frame[1]
                az -= gravity_sensor_frame[2]

                # # print(f"ax: {ax}, ay: {ay}, az: {az}")
                # if abs(ax) < 0.20:
                #     ax = 0
                # if abs(ay) < 0.20:
                #     ay = 0
                # if abs(az) < 0.20:
                #     az = 0
                # print(f"ax: {ax}, ay: {ay}, az: {az}")

                # Update velocity
                self.vx += ax * self.time_delta_accel.total_seconds()
                self.vy += ay * self.time_delta_accel.total_seconds()
                self.vz += az * self.time_delta_accel.total_seconds()

                # Update position
                self.cumulative_x += self.vx * self.time_delta_accel.total_seconds()
                self.cumulative_y += self.vy * self.time_delta_accel.total_seconds()
                self.cumulative_z += self.vz * self.time_delta_accel.total_seconds()

                # print(f"self.cumulative: {self.cumulative_x}, {self.cumulative_y}, {self.cumulative_z}")

        self.last_time_accel = data_timestamp

    def adjust_orientation(self, accel_data):
        # Normalize accelerometer data
        # ax, ay, az = accel_data.z-0.312, accel_data.y+0.027, accel_data.x+0.114
        ax, ay, az = accel_data.z + 0.111643, accel_data.y + 0.106552, accel_data.x - 0.086204
        # print(f"Before Norm: ax, ay, az: {ax, ay, az}")
        norm = np.sqrt(ax * ax + ay * ay + az * az)
        ax /= norm
        ay /= norm
        az /= norm
        # print(f"After Norm: ax, ay, az: {ax, ay, az}")

        if abs(ax) < 0.10:
            ax = 0
        if abs(ay) < 0.10:
            ay = 0
        if abs(az) < 0.10:
            az = 0
        # print(f"ax: {ax}, ay: {ay}, az: {az}")

        # Calculate pitch and roll based on accelerometer data
        # pitch = np.arctan2(accel_data.z, accel_data.x) # This also might work
        self.pitch = np.arctan2(ax, np.sqrt(ax * ax + az * az))
        self.roll = np.arctan2(ay, np.sqrt(ax * ax + az * az))


        # Convert the pitch, roll, and yaw to a rotation matrix
        rotation = Rotation.from_euler('xyz', [self.pitch, -self.cumulative_yaw, self.roll])

        # rotation = Rotation.from_euler('xz', [pitch, roll])
        rotation_matrix = rotation.as_matrix()

        # Convert the 3x3 rotation matrix to a 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix

        rotation_180_y = np.array([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        transformation_matrix = np.dot(transformation_matrix, rotation_180_y)

        # Update the translation in the transformation matrix
        transformation_matrix[2, 3] = self.cumulative_x
        transformation_matrix[0, 3] = self.cumulative_y
        transformation_matrix[1, 3] = self.cumulative_z


        return transformation_matrix, rotation_matrix

    def overlay_movement_on_frame(self, frame):
        # Convert meters to feet
        x_feet = self.cumulative_x * 3.28084
        y_feet = self.cumulative_y * 3.28084
        z_feet = self.cumulative_z * 3.28084

        text = f'Movement: X: {x_feet:.2f} ft, Y: {y_feet:.2f} ft, Z: {z_feet:.2f} ft'
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return frame

    def display_movement(self):
        # Convert meters to feet
        x_feet = self.cumulative_x * 3.28084
        y_feet = self.cumulative_y * 3.28084
        z_feet = self.cumulative_z * 3.28084

        if abs(x_feet) > 0.1 or abs(y_feet) > 0.1 or abs(z_feet) > 0.1:
            # print(f'Movement: X: {x_feet:.2f} ft, Y: {y_feet:.2f} ft, Z: {z_feet:.2f} ft')
            pass
