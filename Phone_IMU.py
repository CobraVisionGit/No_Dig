import asyncio
import websockets
import json
import math

# Conversion factor from meters to feet
m_to_ft = 3.28084

# Initial conditions
velocity = [0, 0, 0]  # Velocity in x, y, z directions
position = [0, 0, 0]  # Position in x, y, z directions
previous_time = None

# Orientation estimation
pitch = 0.0
roll = 0.0
alpha = 0.98  # Complementary filter coefficient


async def get_sensor_data():
    global velocity, position, previous_time, pitch, roll
    accel_uri = "ws://192.168.0.194:8080/sensor/connect?type=android.sensor.accelerometer"
    gyro_uri = "ws://192.168.0.194:8080/sensor/connect?type=android.sensor.gyroscope"
    mag_uri = "ws://192.168.0.194:8080/sensor/connect?type=android.sensor.magnetic_field"

    previous_accel = [0, 0, 0]

    while True:
        try:
            async with websockets.connect(accel_uri) as accel_websocket, \
                    websockets.connect(gyro_uri) as gyro_websocket, \
                    websockets.connect(mag_uri) as mag_websocket:

                while True:
                    accel_data = await accel_websocket.recv()
                    gyro_data = await gyro_websocket.recv()
                    # mag_data = await mag_websocket.recv()  # Uncomment if magnetometer data is needed

                    accel_json = json.loads(accel_data)
                    gyro_json = json.loads(gyro_data)
                    # mag_json = json.loads(mag_data)  # Uncomment if magnetometer data is needed

                    # Assuming the timestamp is consistent across sensors
                    current_time = accel_json.get('timestamp', None)

                    if previous_time is not None and current_time is not None:
                        delta_t = (current_time - previous_time) / 1_000_000_000  # Convert nanoseconds to seconds

                        # Update orientation estimation with a simple complementary filter
                        pitch_accel = math.atan2(accel_json['values'][0],
                                                 math.sqrt(accel_json['values'][1] ** 2 + accel_json['values'][2] ** 2))
                        roll_accel = math.atan2(accel_json['values'][1],
                                                math.sqrt(accel_json['values'][0] ** 2 + accel_json['values'][2] ** 2))
                        pitch_gyro = pitch + gyro_json['values'][0] * delta_t
                        roll_gyro = roll + gyro_json['values'][1] * delta_t
                        pitch = alpha * pitch_gyro + (1 - alpha) * pitch_accel
                        roll = alpha * roll_gyro + (1 - alpha) * roll_accel

                        # Subtract gravity component from accelerometer readings
                        accel = [
                            accel_json['values'][0] - math.sin(pitch),
                            accel_json['values'][1] - math.cos(pitch) * math.sin(roll),
                            accel_json['values'][2] - math.cos(pitch) * math.cos(roll)
                        ]
                        print(f"acceleration: {accel}")

                        # Update velocity and position
                        for i in range(3):
                            velocity[i] += 0.5 * (previous_accel[i] + accel[i]) * delta_t  # Trapezoidal integration
                            position[i] += 0.5 * (velocity[i] + velocity[i]) * delta_t  # Trapezoidal integration

                        # Convert velocity and position from meters to feet
                        velocity_ft = [v * m_to_ft for v in velocity]
                        position_ft = [p * m_to_ft for p in position]

                        print(f"Velocity: {velocity_ft}")
                        print(f"Position: {position_ft}")

                        previous_accel = accel
                    previous_time = current_time
        except (websockets.exceptions.ConnectionClosedError, ConnectionResetError) as e:
            print(f"Connection error: {e}. Reconnecting...")
            await asyncio.sleep(1)  # Wait for 1 second before attempting to reconnect


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(get_sensor_data())
