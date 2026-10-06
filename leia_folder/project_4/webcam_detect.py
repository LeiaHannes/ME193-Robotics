# Runs the trained minifig detector on a live webcam feed and steers
# BigLegoRobot over MQTT so it turns to face the minifig.
# Press "q" in the video window (or Ctrl+C in the terminal) to quit.

import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# Folder that contains mqttlib.py. This matches the path in your MQTT example,
# so adjust it if this script lives in a different folder than that one.
LIB_DIR = Path(__file__).resolve().parent.parent.parent / "Public stuff" / "useful libraries"
sys.path.insert(0, str(LIB_DIR))

from mqttlib import MQTTClient

# Check the actual folder name after training -- if you ran training more than
# once, it may be green_minifig2, green_minifig3, etc.
WEIGHTS = Path(__file__).parent.parent.parent.parent.parent / "runs" / "detect" / "green_minifig-4" / "weights" / "best.pt"
CAMERA_INDEX = 0      # 0 = default webcam; try 1 or 2 if you have more than one camera
CONFIDENCE = 0.5      # ignore detections the model is less than 50% sure about

# MQTT setup
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "BigLegoRobot"
PUBLISH_INTERVAL = 0.1  # seconds between messages (at most 10 per second)

# Steering
TURN_SPEED = 1   # motor fraction, 0 to 1 (same units as the GUI's speed slider)
DEAD_BAND = 10     # pixels: if the minifig is this close to center, stop turning

model = YOLO(WEIGHTS)

# CAP_DSHOW makes webcams open faster and more reliably on Windows
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera {CAMERA_INDEX}. Is another app using it?")

last_publish = 0.0
last_seen = time.time()

try:
    with MQTTClient(broker=BROKER, port=PORT) as client:
        print(f"Connected to {BROKER}:{PORT}. Publishing to '{TOPIC}'")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Couldn't read a frame from the camera.")
                    break

                # Run detection on this frame
                results = model(frame, conf=CONFIDENCE, verbose=False)
                result = results[0]

                if len(result.boxes) > 0:
                    # Keep only the most confident box
                    best = result.boxes[int(result.boxes.conf.argmax())]
                    x1, y1, x2, y2 = best.xyxy[0].tolist()   # box corners in pixels
                    center_x = (x1 + x2) / 2

                    # Positive error = minifig is right of center, negative = left
                    frame_width = frame.shape[1]
                    error = center_x - frame_width / 2

                    if error < -DEAD_BAND:      # minifig on the left -> spin left
                        left_speed, right_speed = -TURN_SPEED, -TURN_SPEED
                    elif error > DEAD_BAND:     # minifig on the right -> spin right
                        left_speed, right_speed = TURN_SPEED, TURN_SPEED
                    else:                       # roughly centered -> hold still
                        left_speed, right_speed = 0, 0

                    message = json.dumps({"left": left_speed, "right": right_speed})
                    last_seen = time.time()
                else:
                    if time.time() - last_seen > 0.5:  # if we haven't seen the minifig for a while
                        message = "stop"
                    # Minifig not visible -> don't keep driving blind
                    #message = "stop"

                #### PASS COMMANDS THROUGH MQTT ###
                now = time.time()
                if now - last_publish >= PUBLISH_INTERVAL:
                    client.publish(TOPIC, message)
                    last_publish = now
                    print(f"Sent: {message}")
                ###################################

                # Draw boxes and labels on the frame and show it
                annotated = result.plot()

                # Shade the dead band region and draw the center line
                h, w = annotated.shape[:2]
                overlay = annotated.copy()
                cv2.rectangle(overlay, (w // 2 - DEAD_BAND, 0), (w // 2 + DEAD_BAND, h), (0, 255, 255), -1)
                annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)
                cv2.line(annotated, (w // 2, 0), (w // 2, h), (0, 255, 255), 2)

                cv2.imshow("Minifig detector (press q to quit)", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            # Make sure the robot stops when this script exits for any reason
            client.publish(TOPIC, "stop")
            time.sleep(0.2)  # give the message time to go out before disconnecting
            print("Sent stop")

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Always release the camera, even if something crashes
    cap.release()
    cv2.destroyAllWindows()