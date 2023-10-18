
# !/usr/bin/env python3

import cv2
import depthai as dai
import time
import math
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
imu = pipeline.create(dai.node.IMU)
xlinkOut = pipeline.create(dai.node.XLinkOut)

xlinkOut.setStreamName("imu")

# enable ACCELEROMETER_RAW at 500 hz rate
imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 500)
# enable GYROSCOPE_RAW at 400 hz rate
imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
# it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
# above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
imu.setBatchReportThreshold(1)
# maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
# if lower or equal to batchReportThreshold then the sending is always blocking on device
# useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
imu.setMaxBatchReports(10)

# Link plugins IMU -> XLINK
imu.out.link(xlinkOut.input)

# Initialize variables for calibration
calibration_duration = 180.0  # Calibration duration in seconds
calibration_data = []
gyro_calibration_data = []
gyro_mean_values = [0, 0, 0]  # Initialize to zeros

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:

    def timeDeltaToMilliS(delta) -> float:
        return delta.total_seconds( ) *1000

    # Output queue for imu bulk packets
    imuQueue = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
    baseTs = None
    calibration_start_time = time.time() # Record the calibration start time

    while True:
        imuData = imuQueue.get()  # blocking call, will wait until a new data has arrived

        imuPackets = imuData.packets
        for imuPacket in imuPackets:
            # Calibration Section: Accelerometer
            # print("Calibration Section: Accelerometer")
            acceleroValues = imuPacket.acceleroMeter

            # Collect calibration data if within calibration duration
            if time.time() - calibration_start_time < calibration_duration:
                calibration_data.append([acceleroValues.x, acceleroValues.y, acceleroValues.z])
            elif len(calibration_data) > 0:
                # Calculate mean accelerometer values for calibration
                mean_values = np.mean(np.array(calibration_data), axis=0)
                print(f"Calibration complete. Mean values: x: {mean_values[0]:.06f} y: {mean_values[1]:.06f} z: {mean_values[2]:.06f}")
                calibration_data = []  # Clear calibration data


            # Calibration Section: Gyroscope
            # print("Calibration Section: Gyroscope")
            gyroValues = imuPacket.gyroscope

            # Collect calibration data if within calibration duration
            if time.time() - calibration_start_time < calibration_duration:
                gyro_calibration_data.append([gyroValues.x, gyroValues.y, gyroValues.z])
            elif len(gyro_calibration_data) > 0:
                # Calculate mean gyroscope values for calibration
                gyro_mean_values = np.mean(np.array(gyro_calibration_data), axis=0)
                print(f"Gyro Calibration complete. Mean values: x: {gyro_mean_values[0]:.06f} y: {gyro_mean_values[1]:.06f} z: {gyro_mean_values[2]:.06f}")
                gyro_calibration_data = []  # Clear calibration data


            acceleroTs = acceleroValues.getTimestampDevice()
            gyroTs = gyroValues.getTimestampDevice()
            if baseTs is None:
                baseTs = acceleroTs if acceleroTs < gyroTs else gyroTs
            acceleroTs = timeDeltaToMilliS(acceleroTs - baseTs)
            gyroTs = timeDeltaToMilliS(gyroTs - baseTs)

            imuF = "{:.06f}"
            tsF  = "{:.03f}"

            # Correct the gyroscope readings
            corrected_gyro_x = gyroValues.x - gyro_mean_values[0]
            corrected_gyro_y = gyroValues.y - gyro_mean_values[1]
            corrected_gyro_z = gyroValues.z - gyro_mean_values[2]

            # print(f"Corrected Gyroscope [rad/s]: x: {imuF.format(corrected_gyro_x)} y: {imuF.format(corrected_gyro_y)} z: {imuF.format(corrected_gyro_z)} ")

            # print(f"Accelerometer timestamp: {tsF.format(acceleroTs)} ms")
            # print(f"Accelerometer [m/s^2]: x: {imuF.format(acceleroValues.x)} y: {imuF.format(acceleroValues.y)} z: {imuF.format(acceleroValues.z)}")
            # print(f"Gyroscope timestamp: {tsF.format(gyroTs)} ms")
            # print(f"Gyroscope [rad/s]: x: {imuF.format(gyroValues.x)} y: {imuF.format(gyroValues.y)} z: {imuF.format(gyroValues.z)} ")

        if cv2.waitKey(1) == ord('q'):
            break
