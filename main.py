import cv2
import mediapipe as mp
import math
import numpy as np

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ----------------------------
# Initialize Windows Volume
# ----------------------------
devices = AudioUtilities.GetSpeakers()

interface = devices.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(
    interface,
    POINTER(IAudioEndpointVolume)
)

minVol, maxVol, _ = volume.GetVolumeRange()

# ----------------------------
# MediaPipe Hand Landmarker
# ----------------------------
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ----------------------------
# Open Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            points = []

            # Draw Landmarks
            for landmark in hand:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                points.append((x, y))

                cv2.circle(frame, (x, y), 5, (0,255,0), -1)

            # Draw Connections
            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255,0,0),
                    2
                )

            # Thumb Tip
            thumb = points[4]

            # Index Finger Tip
            index = points[8]

            cv2.circle(frame, thumb, 10, (0,0,255), -1)
            cv2.circle(frame, index, 10, (255,0,255), -1)

            # Draw Line
            cv2.line(frame, thumb, index, (0,255,255), 3)

            # Calculate Distance
            distance = math.hypot(
                thumb[0]-index[0],
                thumb[1]-index[1]
            )

            # Convert Distance → Volume
            vol = np.interp(
                distance,
                [30, 200],
                [minVol, maxVol]
            )

            volume.SetMasterVolumeLevel(vol, None)

            # Convert Distance → Percentage
            volPercent = np.interp(
                distance,
                [30,200],
                [0,100]
            )

            # Display Distance
            cv2.putText(
                frame,
                f"Distance : {int(distance)} px",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            # Display Volume Percentage
            cv2.putText(
                frame,
                f"Volume : {int(volPercent)} %",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

            # Volume Bar
            bar = np.interp(
                distance,
                [30,200],
                [400,150]
            )

            cv2.rectangle(frame, (50,150), (85,400), (255,255,255), 2)
            cv2.rectangle(frame, (50,int(bar)), (85,400), (0,255,0), -1)

    cv2.imshow("Hand Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()