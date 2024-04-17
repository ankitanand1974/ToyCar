import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import threading
import time
from tracker import *
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize the Firebase Admin SDK
cred = credentials.Certificate('key.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://toycar-8266e-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# Reference to the 'traffic' node in the database
ref = db.reference('traffic')

# Function to upload data to Firebase
def upload_data(value1, value2):
    # Get the current data from Firebase
    current_data = ref.get() or {}
    print("\nLEFT signal:", value1)
    print("RIGHT signal:", value2)
    if error:
        print("Error:", error)

    # Check if the values are different from the current data
    if current_data != {'left_signal': value1, 'right_signal': value2}:
        # Update the 'traffic' node with the new values
        data = {'left_signal': value1, 'right_signal': value2}
        ref.set(data)
        print("Data uploaded to Firebase.")


# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Load YOLO model
model = YOLO('yolov8s.pt')

# Move the model to GPU
# model = model.to('cuda')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap1 = cv2.VideoCapture('1.mp4')
# cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture('2.mp4')
# cap2 = cv2.VideoCapture(1)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

tracker1 = Tracker()
tracker2 = Tracker()

cy1 = 184
cy2 = 209
offset = 8

upcar1 = {}
downcar1 = {}
countercarup1 = []
countercardown1 = []

left_signal="GREEN"
right_signal="GREEN"
error=""

upcar2 = {}
downcar2 = {}
countercarup2 = []
countercardown2 = []

import time

# Define the maximum waiting time for a side
MAX_WAIT_TIME = 30

# Global counters for cars entering and exiting the bridge
l_car_in = 0
l_car_out = 0
r_car_in = 0
r_car_out = 0


def signal_system(left_car_in, left_car_out, right_car_in, right_car_out):
    # Initialize the signals and counters if they don't exist
    if 'left_signal' not in signal_system.__dict__:
        signal_system.left_signal = 'GREEN'
        signal_system.right_signal = 'GREEN'
        signal_system.left_counter = 0
        signal_system.right_counter = 0

    # Update the counters based on the new input values
    signal_system.left_counter += left_car_in
    signal_system.left_counter -= right_car_out
    signal_system.right_counter += right_car_in
    signal_system.right_counter -= left_car_out

    # Check if the first vehicle has entered the bridge
    if signal_system.left_counter > 0 or signal_system.right_counter > 0:
        # Set the opposite side's signal to RED
        if signal_system.left_counter > 0:
            signal_system.right_signal = 'RED'
        else:
            signal_system.left_signal = 'RED'

    # Check if 5 vehicles have entered from the GREEN side and exited from the RED side
    if signal_system.left_signal == 'GREEN':
        if signal_system.right_counter >= 5:
            signal_system.left_signal = 'RED'
            signal_system.right_signal = 'GREEN'
    else:
        if signal_system.left_counter >= 5:
            signal_system.left_signal = 'GREEN'
            signal_system.right_signal = 'RED'

    return signal_system.left_signal, signal_system.right_signal



def upload_data_thread():
    global left_signal, right_signal,error
    while True:
        #left_signal, right_signal = signal_system(l_car_in, l_car_out, r_car_in, r_car_out)
        left_signal, right_signal = signal_system(l_car_in, l_car_out, r_car_in, r_car_out)
        upload_data(left_signal, right_signal)
        time.sleep(1)  # Wait for 1 second before uploading again

upload_thread = threading.Thread(target=upload_data_thread)
upload_thread.daemon = True  # Set the thread as daemon so it terminates when the main thread exits
upload_thread.start()


while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not (ret1 and ret2):
        break

    frame1 = cv2.resize(frame1, (510, 250))
    frame2 = cv2.resize(frame2, (510, 250))

    # Use the same model for both video streams on GPU
    results1 = model(frame1, verbose=False)
    results2 = model(frame2, verbose=False)

    a1 = results1[0].boxes.data.cpu().numpy()
    px1 = pd.DataFrame(a1).astype("float")

    a2 = results2[0].boxes.data.cpu().numpy()
    px2 = pd.DataFrame(a2).astype("float")

    list1 = []
    list2 = []

    # ... (the rest of the code remains the same)

    for index, row in px1.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list1.append([x1, y1, x2, y2])

    for index, row in px2.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list2.append([x1, y1, x2, y2])

    bbox_idx1 = tracker1.update(list1)
    bbox_idx2 = tracker2.update(list2)

    for bbox in bbox_idx1:
        x3, y3, x4, y4, id1 = bbox
        cx3 = int(x3 + x4) // 2
        cy3 = int(y3 + y4) // 2
        if cy1 < cy3 < cy2:
            if id1 not in upcar1 and id1 not in downcar1:
                upcar1[id1] = (cx3, cy3)
            if id1 in upcar1:
                cv2.circle(frame1, (cx3, cy3), 4, (255, 0, 0), -1)
                cv2.rectangle(frame1, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cvzone.putTextRect(frame1, f'{id1}', (x3, y3), 1, 1)
                if countercarup1.count(id1) == 0:
                    countercarup1.append(id1)
        elif cy3 > cy2:
            if id1 not in downcar1 and id1 not in upcar1:
                downcar1[id1] = (cx3, cy3)
            if id1 in downcar1:
                cv2.circle(frame1, (cx3, cy3), 4, (255, 0, 255), -1)
                cv2.rectangle(frame1, (x3, y3), (x4, y4), (255, 0, 0), 2)
                cvzone.putTextRect(frame1, f'{id1}', (x3, y3), 1, 1)
                if countercardown1.count(id1) == 0:
                    countercardown1.append(id1)

    for bbox in bbox_idx2:
        x3, y3, x4, y4, id2 = bbox
        cx3 = int(x3 + x4) // 2
        cy3 = int(y3 + y4) // 2
        if cy1 < cy3 < cy2:
            if id2 not in upcar2 and id2 not in downcar2:
                upcar2[id2] = (cx3, cy3)
            if id2 in upcar2:
                cv2.circle(frame2, (cx3, cy3), 4, (255, 0, 0), -1)
                cv2.rectangle(frame2, (x3, y3), (x4, y4), (255, 0, 255), 2)
                cvzone.putTextRect(frame2, f'{id2}', (x3, y3), 1, 1)
                if countercarup2.count(id2) == 0:
                    countercarup2.append(id2)
        elif cy3 > cy2:
            if id2 not in downcar2 and id2 not in upcar2:
                downcar2[id2] = (cx3, cy3)
            if id2 in downcar2:
                cv2.circle(frame2, (cx3, cy3), 4, (255, 0, 255), -1)
                cv2.rectangle(frame2, (x3, y3), (x4, y4), (255, 0, 0), 2)
                cvzone.putTextRect(frame2, f'{id2}', (x3, y3), 1, 1)
                if countercardown2.count(id2) == 0:
                    countercardown2.append(id2)

    cv2.line(frame1, (1, cy1), (508, cy1), (0, 255, 0), 2)
    cv2.line(frame1, (3, cy2), (506, cy2), (0, 0, 255), 2)
    cup1 = len(countercarup1)
    cdown1 = len(countercardown1)
    cvzone.putTextRect(frame1, f'upcar1:-{cup1}', (50, 60), 1, 2)
    cvzone.putTextRect(frame1, f'downcar1:-{cdown1}', (50, 160), 1, 2)
    cv2.line(frame2, (1, cy1), (508, cy1), (0, 255, 0), 2)
    cv2.line(frame2, (3, cy2), (506, cy2), (0, 0, 255), 2)
    cup2 = len(countercarup2)
    cdown2 = len(countercardown2)
    cvzone.putTextRect(frame2, f'upcar2:-{cup2}', (50, 60), 1, 2)
    cvzone.putTextRect(frame2, f'downcar2:-{cdown2}', (50,160), 1, 2)
    combined_frame = np.vstack((frame1, frame2))  # Stack frames vertically
    cv2.imshow("RGB", combined_frame)
    l_car_in = cup1
    l_car_out = cdown1
    r_car_in = cup2
    r_car_out = cdown2
    # upload_data(l_car_in, l_car_out, r_car_in, r_car_out)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()