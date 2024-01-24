import mediapipe as mp
import cv2 as cv
import numpy as np
import uuid
import os
import imageio
import math

# Import effect gif
eff_list = imageio.mimread("water_stream_v1.gif", '.gif')
for idx,eff in enumerate(eff_list):
    eff_list[idx] = cv.cvtColor(eff, cv.COLOR_RGB2BGR)


# Resolution
w_res = 640
h_res = 480
h_eff, w_eff, _ = eff_list[0].shape


# Get coordinates for perspective transform

def get_frame(point_A, point_B):
    hw_ratio = h_eff/w_eff
    # Calculating frame for gif file
    ABCD_1 = np.float32([[0,0],[0,0],[0,0],[0,0]])
    ABCD_2 = np.float32([point_A,point_B,[0,0],[0,0]])


    w_frame = round(math.dist(ABCD_2[1], ABCD_2[0]))
    h_frame = round(w_frame*hw_ratio)
    ABCD_1[0] = ABCD_2[0]
    ABCD_1[1] = ABCD_2[0] + [w_frame,0]
    ABCD_1[2] = ABCD_2[0] + [0,h_frame]
    ABCD_1[3] = ABCD_2[0] + [w_frame, h_frame]

    B_x_shift = ABCD_2[1,0] - ABCD_1[1,0]
    B_y_shift = ABCD_1[1,1] - ABCD_2[1,1]

    ABCD_2[2] = ABCD_1[2] + [round(B_y_shift*hw_ratio), round(B_x_shift*hw_ratio)]
    ABCD_2[3] = ABCD_2[2] + [ABCD_2[1,0] - ABCD_1[0,0], ABCD_2[1,1] - ABCD_1[0,1]]
    return ABCD_2


# Add effect to video frame

def add_eff(img, pts2, effect):
    # Taking effect
    pts1 = np.float32([[0,0], [w_eff,0], [0, h_eff], [w_eff, h_eff]])

    # Get matrix
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    # Get blank image with effect on it
    effect_img = cv.warpPerspective(effect, matrix, (w_res,h_res))

    # Merging effect onto image
    output = np.where(effect_img != effect_img[0, 0, 0], effect_img, img)
    return output


# Model for hand
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Check if hand is in position
def check_hand(results):
    points = []
    for hand in results:
            y_array = np.array([hand.landmark[4].y, hand.landmark[8].y, hand.landmark[12].y, hand.landmark[16].y, hand.landmark[20].y])*h_res
            if np.all(y_array>hand.landmark[0].y*h_res):
                x_array = np.array([hand.landmark[4].x, hand.landmark[8].x, hand.landmark[12].x, hand.landmark[16].x, hand.landmark[20].x])*w_res
                idxmin = np.argmin(x_array)
                idxmax = np.argmax(x_array)
                thres = max(y_array[idxmax], y_array[idxmin]) + (x_array[idxmax] - x_array[idxmin])/5
                if np.all(y_array < thres):
                    A = (int(x_array[idxmin]),int(y_array[idxmin]+(x_array[idxmax] - x_array[idxmin])/5.5))
                    B = (int(x_array[idxmax]),int(y_array[idxmax]+(x_array[idxmax] - x_array[idxmin])/5.5))
                    points.append(A)
                    points.append(B)
                    return points

# Define camera
cap = cv.VideoCapture(0)

# Variable to count gif
count = 0
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        # BGR 2 RGB
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # Rendering results
        if results.multi_hand_landmarks:
            coords = check_hand(results.multi_hand_landmarks)
            if coords:
                if count == 5:
                    count = 0
                pts2 = get_frame(coords[0], coords[1])
                image = add_eff(image, pts2,eff_list[count])
                count += 1
        cv.imshow('Hand Effect', image)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()




