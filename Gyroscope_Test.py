import depthai as dai
import numpy as np


# Create a pipeline
pipeline = dai.Pipeline()

# Create an IMU node
imu_node = pipeline.createIMU()
imu_node.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 400)  # 400 Hz sample rate

# Initialize the device with the created pipeline
device = dai.Device(pipeline)

# Create an IMU queue
imu_queue = device.getOutputQueue(name="imu", maxSize=8, blocking=False)

def get_accel_data():
    while True:
        imu_data = imu_queue.get()  # Get IMU packet
        for imu_packet in imu_data.packets:
            accel_data = imu_packet.acceleroMeter
            if accel_data:  # If accelerometer data is available
                return accel_data

def compute_roll_pitch(accel_data):
    roll = np.arctan2(accel_data.y, np.sqrt(accel_data.x**2 + accel_data.z**2))
    pitch = np.arctan2(-accel_data.x, np.sqrt(accel_data.y**2 + accel_data.z**2))
    return roll, pitch

if __name__ == "__main__":
    accel_data = get_accel_data()
    roll, pitch = compute_roll_pitch(accel_data)
    print(f"Roll: {np.degrees(roll)} degrees, Pitch: {np.degrees(pitch)} degrees")
