import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import math
import imageio
import matplotlib.pyplot as plt
# from numpy.distutils.misc_util import get_frame
# from main import width, height
img_eff = imageio.get_reader('crystal.gif')
effect = [frame for frame in img_eff]
#print(effect[0].shape)
w_eff = effect[0].shape[1]
h_eff = effect[0].shape[0]
max_count = len(effect)
count = 0
count_div =0

w_res = 640
h_res = 480



def get_frame(point_1, point_2):
    # tìm vector chỉ phương, suy ra vector pháp tuyến, từ đó suy ra 4 điểm
    vector_u = (point_1[0] - point_2[0], point_1[1] - point_2[1])
    # print(vector_u)
    vector_1 = (int(-vector_u[1] / 2), int(vector_u[0] / 2))
    # print(vector_1)
    vector_2 = (int(vector_u[1] / 2), int(-vector_u[0] / 2))
    # print(vector_2)
    # tạo 1 hình vuông ABCD
    # point 1 là trung điểm AB, point 2 là trung điểm CD
    A = [point_1[0] + vector_2[0], point_1[1] + vector_2[1]]
    B = [point_1[0] + vector_1[0], point_1[1] + vector_1[1]]
    
    C = [point_2[0] + vector_2[0], point_2[1] + vector_2[1]]
    D = [point_2[0] + vector_1[0], point_2[1] + vector_1[1]]
    
    pts = np.float32([[A[0], A[1]], [B[0], B[1]], [C[0], C[1]], [D[0], D[1]]])
    return A, B, C, D, pts


def add_effect(pts2, pt1, pt2):
    global count, count_div
    # count_div +=1
    # if(count_div==2):
    #     count_div =0
    #     count +=1
    count+=1
    size_thres = int(math.dist(pt1, pt2))
    #print(size_thres)
    effect_temp=effect[count]
    pts1 = np.float32([[0, 0], [w_eff, 0], [0, h_eff], [w_eff, h_eff]])
    # print(pts1)
    # Get matrix
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # Get blank image with effect on it
    effect_copy=effect_temp[:,:,0:3]
    effect_copy=cv2.cvtColor(effect_copy,cv2.COLOR_RGB2BGR)
    effect_copy = cv2.GaussianBlur(effect_copy, (7, 7), 0)
    effect_img = cv2.warpPerspective(effect_copy, matrix, (w_res, h_res))
    thres_color = 30
    # Lồng effect vào ảnh
    image[:,:,:] = np.where(effect_img >=thres_color, effect_img, image[:,:,:])
    if(count== max_count -1): count =0
    
def dectect_gesture(hand_result):
    lmk = hand_result.landmark
    
    pt1 = (int(lmk[8].x * width), int(lmk[8].y * height))
    pt2 = (int(lmk[4].x * width), int(lmk[4].y * height))
    
    dist_thres = math.dist((lmk[5].x, lmk[5].y), (lmk[0].x, lmk[0].y))
    dist1 = math.dist((lmk[12].x, lmk[12].y), (lmk[0].x, lmk[0].y))
    dist2 = math.dist((lmk[16].x, lmk[16].y), (lmk[0].x, lmk[0].y))
    dist3 = math.dist((lmk[20].x, lmk[20].y), (lmk[0].x, lmk[0].y))
    dist4 = math.dist((lmk[8].x, lmk[8].y), (lmk[0].x, lmk[0].y))
    
    dist_thumb4 = math.dist((lmk[4].x, lmk[4].y), (lmk[5].x, lmk[5].y))
    dist_thumb3 = math.dist((lmk[3].x, lmk[3].y), (lmk[5].x, lmk[5].y)) 
    if dist3 < dist_thres and dist2 < dist_thres and dist1 < dist_thres and dist4 > dist_thres and dist_thumb4 > dist_thumb3:
        #print('OK')
        A, B, C, D, pts2 = get_frame(pt1, pt2)
        # print(pts2)
        # cv2.putText(image, 'A', A, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # cv2.putText(image, 'B', B, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # cv2.putText(image, 'C', C, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # cv2.putText(image, 'D', D, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # cv2.line(image, A, B, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.line(image, B, D, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.line(image, C, D, (0, 0, 255), 1, cv2.LINE_AA)
        # cv2.line(image, C, A, (0, 0, 255), 1, cv2.LINE_AA)
        
        add_effect(pts2, C, D)

    
width = 640  # kích thước của camera
height = 480

# #nhận dạng 
# recognizer = vision.GestureRecognizer.create_from_options(options) 
mp_drawing = mp.solutions.drawing_utils  # vẽ landmark function
mp_hands = mp.solutions.hands  # khởi tạo hand class
hands = mp_hands.Hands(static_image_mode=0, min_detection_confidence=0.8, min_tracking_confidence=0.5) 
# llandmarks point , chế độ ảnh tĩnh, video thì value 0

# dữ liệu từ camera

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_res)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_res)
while cap.isOpened():
    ret, frame = cap.read()
    # ret là giá trị bool, true là frame thành công,frame là từng khung hình từ camera
    # đổi bgr to rgb
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Lật ảnh
    image = cv2.flip(image, 1)
    # bật cờ
    image.flags.writeable = False
        
        # Detections
    results = hands.process(image)
        
        # Set flag to true
    image.flags.writeable = True
    # recognition_result = recognizer.recognize(image)
    
        # RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # print(image.shape) 
    if results.multi_hand_landmarks:
        # for num, hand in enumerate(results.multi_hand_landmarks):
        #     mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
        #                                 mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        #                                 mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
        #                                  )
            
        # print(f'{mp_hands.HandLandmark(0).name}:')
        # print(f'x: {results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark(0).value].x * width}')
        # print(f'y: {results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark(0).value].y * height}')
        #
        # in đỉnh ngón giữa
        # print(f'{mp_hands.HandLandmark(12).name}:')
        # print(f'x: {results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark(12).value].x * width}')
        # print(f'y: {results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark(12).value].y * height}')
    
        dectect_gesture(results.multi_hand_landmarks[0])
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()     
     
