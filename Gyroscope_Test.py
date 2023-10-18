import websocket
import threading
import time

def on_message_accel(ws, message):
    print("Accelerometer: " + message)  # sensor data here in JSON format
    # time.sleep(1)

def on_message_gyro(ws, message):
    print("Gyroscope: " + message)  # sensor data here in JSON format
    time.sleep(1000)

def on_error(ws, error):
    print("### error ###")
    print(error)

def on_close(ws, close_code, reason):
    print("### closed ###")
    print("close code : ", close_code)
    print("reason : ", reason)

def on_open(ws):
    print("connection opened")

def run_accelerometer_websocket():
    ws = websocket.WebSocketApp(
        "ws://192.168.0.194:8080/sensor/connect?type=android.sensor.accelerometer",
        on_open=on_open,
        on_message=on_message_accel,  # Updated to the new callback function
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

def run_gyroscope_websocket():
    gs = websocket.WebSocketApp(
        "ws://192.168.0.194:8080/sensor/connect?type=android.sensor.gyroscope",
        on_open=on_open,
        on_message=on_message_gyro,  # Updated to the new callback function
        on_error=on_error,
        on_close=on_close
    )
    gs.run_forever()

if __name__ == "__main__":
    # Start a new thread for the accelerometer websocket
    accelerometer_thread = threading.Thread(target=run_accelerometer_websocket)
    accelerometer_thread.start()

    # Start a new thread for the gyroscope websocket
    gyroscope_thread = threading.Thread(target=run_gyroscope_websocket)
    gyroscope_thread.start()

    # Optionally wait for both threads to finish (this will block indefinitely in this case)
    accelerometer_thread.join()
    gyroscope_thread.join()
