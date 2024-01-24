import mediapipe as mp
import cv2
import imageio
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
pathG = "output-onlinegiftools.gif"
pathN = "watertornado.gif"
gif = imageio.mimread(pathN)
Wind_tornado = [frame for frame in gif]
gif2 = imageio.mimread(pathG)
Water_tornado = [frame for frame in gif2]
numGif = 0

def wind_effect(img, tor, self):
    tor = cv2.cvtColor(tor, cv2.COLOR_BGR2RGB)
    tor = cv2.GaussianBlur(tor, (7, 7), 0)
    x1 = int(self.landmark[8].x * img.shape[1])
    y1 = int(self.landmark[8].y * img.shape[0])
    x2 = int(self.landmark[6].x * img.shape[1])
    y2 = int(self.landmark[5].y * img.shape[0])
    torY = int(abs(y2 - y1)) * 2
    torX = int((tor.shape[1] * torY / tor.shape[0]) / 2) * 2
    if y1 < y2 and abs(x2 - x1) < 40:
        if torX > 0:
            eff = cv2.resize(tor, (torX, torY))
            disX = round(torX / 2)
            ptsEff = np.float32([[0, 0], [torX, 0], [0, torY], [torX, torY]])
            ptsImg = np.float32([[x1 - disX, y1 - torY], [x1 + disX, y1 - torY], [x1 - disX, y1], [x1 + disX, y1]])
            matrix = cv2.getPerspectiveTransform(ptsEff, ptsImg)
            effs = cv2.warpPerspective(eff, matrix, (img.shape[1], img.shape[0]))
            img[:, :, :] = np.where(effs != effs[0, 0, :], effs, img[:, :, :])

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            result = results.multi_hand_landmarks[0]
            if numGif < len(Water_tornado):
                wind_effect(image, Wind_tornado[numGif], result)
                # try_effect(image, tornado[numGif])
                if len(results.multi_hand_landmarks) == 2:
                    result = results.multi_hand_landmarks[1]
                    wind_effect(image, Water_tornado[numGif], result)
                numGif += 1
            else:
                numGif = 0
                wind_effect(image, Wind_tornado[numGif], result)
                if len(results.multi_hand_landmarks) == 2:
                    result = results.multi_hand_landmarks[1]
                    wind_effect(image, Water_tornado[numGif], result)
            # for num, hand in enumerate(results.multi_hand_landmarks):
            #         mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
            #         mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            #         mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
            #         )

        cv2.imshow('Hand Effect', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
