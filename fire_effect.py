import mediapipe as mp
import cv2
import math 
import numpy as np
import imageio 

height = 480
weight = 640

def handdetect(result):
    lm = result.multi_hand_landmarks
    if len(result.multi_handedness) == 2:
        point_1 = np.array((lm[0].landmark[20].x, lm[0].landmark[20].y)) * [weight, height]
        point_2 = np.array((lm[0].landmark[19].x, lm[0].landmark[19].y))* (weight, height)
        wrist0 = np.array((lm[0].landmark[0].x,lm[0].landmark[0].y))* (weight, height)
        wrist1 = np.array((lm[1].landmark[0].x,lm[1].landmark[0].y))* (weight, height)                            
        max_dist = math.dist(point_1, point_2) #wrist_dist
        dist = math.dist(wrist0, wrist1)
        # tranh 2 tay trum rot xuong     
        ut_l = lm[0].landmark[20].y 
        ut_r = lm[1].landmark[20].y 
        ct_l = lm[0].landmark[0].y
        ct_r = lm[1].landmark[0].y
        #2 tay chap lai
        wrist2 = np.array((lm[0].landmark[0].x,lm[0].landmark[12].y))* (weight, height)
        wrist3 = np.array((lm[1].landmark[0].x,lm[1].landmark[12].y))* (weight, height)   
        dist23 =  math.dist(wrist2, wrist3)
        #khoang cach point 2 trai va phai 
        two_0 = np.array((lm[0].landmark[2].x,lm[0].landmark[2].y))* (weight, height)
        two_1 = np.array((lm[1].landmark[2].x,lm[1].landmark[2].y))* (weight, height)   
        disttwo =  math.dist(two_0, two_1)


        if dist <= 4* max_dist and ut_l < ct_l and ut_r < ct_r and dist23 > 2*max_dist:
            fire = 1 # bien co co lua
            x_insert =  int(lm[0].landmark[2].x  *weight ) if (lm[0].landmark[2].x > lm[1].landmark[2].x) else int(lm[1].landmark[2].x  *weight )
            y_insert =  int(lm[0].landmark[2].y  *height ) if (lm[0].landmark[2].y > lm[1].landmark[2].y) else int(lm[1].landmark[2].y  *height )
            x_insert = x_insert - int((disttwo - 6*max_dist)/2) 
            return fire,x_insert, y_insert, max_dist #vi tri chen lua
        else:
            fire = 0 # bien co khong co lua
            return fire, 0, 0, 0
    else:
        fire = 0 # bien co khong co lua

        return fire,0, 0, 0
        
       
# doc gif
gifs = imageio.mimread("blue_fame.gif") 
effects = [gif for gif in gifs]
numEff = 0 
        
def insert_gif(chen, w_max, hand , XR, YR, fire):
#    fire = remove_background(fire)
    fire = fire[:,:,0:3]
    fire = cv2.cvtColor(fire, cv2.COLOR_RGB2BGR)
    if chen == 1:
            effect = cv2.resize(fire, (int(w_max*6), int(w_max*9)))
            height, width, _ = effect.shape
#XR, YR la toa do bottom tay phai
            x = XR - width  # X-coordinate of the top-left corner
            y = YR - height  # Y-coordinate of the top-left corner
# Check if the position and dimensions exceed the boundaries of the bigger picture
            if x < 0 or y < 0 or x + width > hand.shape[1] or y + height > hand.shape[0]:
                print("Invalid position or dimensions for inserting the smaller picture.")
                exit()  
# Insert the fire picture into the hand picture    
#            hand[y:y+height, x:x+width] = effect  x
            hand[y:y+height, x:x+width] = np.where(effect > effect[0,0,:], effect,hand[y:y+height, x:x+width] )               
    else:
        hand = hand 
    return hand            
#detect right or left hand
             

              
      
  
# ham main 
    
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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

        
        # Rendering results

        if results.multi_hand_landmarks:
            flag, XR, YR, WMAX = handdetect(results)
                
            for num, hand in enumerate(results.multi_hand_landmarks):
#                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
#                                       mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                                       mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
#                                        )  
                if numEff < len(effects):
                    image = insert_gif(flag, WMAX, image, XR, YR, effects[numEff])
                    numEff += 1
                else:
                    numEff = 0
                    image = insert_gif(flag, WMAX, image, XR, YR, effects[numEff])
                                             
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break