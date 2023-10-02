import depthai as dai

# Create a pipeline
pipeline = dai.Pipeline()
imu = pipeline.create(dai.node.IMU)
xlink_out = pipeline.create(dai.node.XLinkOut)

# Set xlink output stream name
xlink_out.setStreamName('imu')

# Enable ACCELEROMETER_RAW and GYROSCOPE_RAW at 100 Hz rate
imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 100)
# Set batch report settings
imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)

# Link IMU node to XLinkOut node
imu.out.link(xlink_out.input)

# Create device and start a pipeline
with dai.Device(pipeline) as device:
    # Get output queue
    imu_queue = device.getOutputQueue(name='imu', maxSize=10, blocking=False)

    while True:
        # Get IMU data
        imu_data = imu_queue.get()  # Blocking call, will wait until a new data has arrived
        # Now imu_data contains the IMU data, which you can access with imu_data.packets

        for packet in imu_data.packets:
            accel_data = packet.acceleroMeter
            gyro_data = packet.gyroscope

            # Now accel_data and gyro_data contain the accelerometer and gyroscope data respectively
            # They are in the format of dai.IMUReport - you can access the data with accel_data.x, accel_data.y, accel_data.z, etc.
            print(f"Accelerometer: {accel_data.x}, {accel_data.y}, {accel_data.z}")
            print(f"Gyroscope: {gyro_data.x}, {gyro_data.y}, {gyro_data.z}")
