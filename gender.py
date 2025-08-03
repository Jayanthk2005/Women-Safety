import cv2
import time
import mediapipe as mp
import requests

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)

def detect_gesture(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

    thumb_tip = landmarks[4]          
    pinky_finger_tip = landmarks[20]    
    index_finger_tip = landmarks[8]    
    middle_finger_tip = landmarks[12]    
    ring_finger_tip = landmarks[16]      

   
    thumb_raised = thumb_tip[1] < landmarks[2][1]         
    pinky_raised = pinky_finger_tip[1] < landmarks[18][1]  
    index_raised = index_finger_tip[1] >= landmarks[6][1]  
    middle_raised = middle_finger_tip[1] >= landmarks[10][1]  
    ring_raised = ring_finger_tip[1] >= landmarks[14][1]    

   
    return thumb_raised and pinky_raised and not index_raised and not middle_raised and not ring_raised


face_Proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
gender_Proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(face_model, face_Proto)
genderNet = cv2.dnn.readNet(gender_model, gender_Proto)

genderList = ['Male', 'Female']
Model_Mean = (78.4263377603, 87.7689143744, 114.895847746)


danger_start_time = None
danger_duration = 5
call_police_time = None
call_police_duration = 9  

while True:
    ret, frame = video.read()
    if not ret:
        break
    frameH, frameW = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bbox = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameW)
            y1 = int(detection[0, 0, i, 4] * frameH)
            x2 = int(detection[0, 0, i, 5] * frameW)
            y2 = int(detection[0, 0, i, 6] * frameH)
            bbox.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    danger_detected = False

   
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if detect_gesture(hand_landmarks):
                danger_detected = True
                if danger_start_time is None:
                    danger_start_time = time.time()  
            else:
                danger_start_time = None  

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    num_men = 0
    num_women = 0

    for box in bbox:
        x1, y1, x2, y2 = box
        face = frame[max(0, y1):min(y2, frame.shape[0] - 1),
                     max(0, x1):min(x2, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), Model_Mean, swapRB=False)
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        if gender == 'Male':
            num_men += 1
        else:
            num_women += 1

    if danger_detected and num_women > 0:
        cv2.putText(frame, "THE WOMEN IS IN DANGER", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
        if danger_start_time is not None and (time.time() - danger_start_time) > danger_duration:
            if call_police_time is None:
                call_police_time = time.time()  
            
            cv2.putText(frame, "CALL POLICE", (20, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

           
            if (time.time() - call_police_time) > call_police_duration:
                call_police_time = None  
                location_data = {
                    'location': {'latitude': 0.0, 'longitude': 0.0} 
                }
                try:
                    response = requests.post('http://localhost:5000/contact_police', json=location_data)
                    if response.status_code == 500:
                        print("Police contacted successfully!")
                    else:
                        print("Failed to contact police.")
                except Exception as e:
                    print(f"Error contacting police: {e}")

    cv2.putText(frame, f"Men: {num_men}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Women: {num_women}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Detector", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()