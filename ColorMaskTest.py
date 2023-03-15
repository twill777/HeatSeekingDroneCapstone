from __future__ import print_function
import cv2
import argparse
import numpy as np
from pynput.mouse import Controller


max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

mouse = Controller()
centerCoords = [0, 0]
showColor = True
radius = 20

colorPicker = np.zeros((1, 1, 3), np.uint8)
colorPickerLow = np.zeros((1, 1, 3), np.uint8)

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)

def on_mouse(event,x,y,flags,param):
    global showColor

    if event == cv2.EVENT_MOUSEMOVE:
        centerCoords[0] = x
        centerCoords[1] = y

    if event == cv2.EVENT_LBUTTONDOWN and not showColor:
        showColor = True
        global low_H
        global low_S
        global low_V
        global high_H
        global high_S
        global high_V

        colorPicker_HSV = cv2.cvtColor(colorPicker, cv2.COLOR_BGR2HSV)
        
        margin = 15
        hHi,sHi,vHi = cv2.split(colorPicker_HSV)
        high_H = min(180,(int(np.max(hHi)-margin) * max_value_H) // max_value)
        high_S = min(255,int(np.max(sHi))-margin)
        high_V = min(255,(np.max(vHi))-margin)

        colorPickerLow_HSV = cv2.cvtColor(colorPickerLow, cv2.COLOR_BGR2HSV)
        hLo, sLo, vLo = cv2.split(colorPickerLow_HSV)
        low_H = max(0,(int(np.min(hLo)-margin) * max_value_H) // max_value)
        low_S = max(0,int(np.min(sLo))-margin)
        low_V = max(0,int(np.min(vLo))-margin)

        print(low_H, low_S, low_V)
        print(high_H, high_S, high_V)
        print('-')


parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)
cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

cap = cv2.VideoCapture("test_video5.mp4")
#cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
#if not cap.isOpened():
#    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    frame_width, frame_height = frame.shape[:2]

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    frame_threshold = cv2.cvtColor(frame_threshold, cv2.COLOR_GRAY2BGR)
    frame_filteredColor = cv2.bitwise_and(frame, frame_threshold)

    colorPicker = np.zeros((frame_width, frame_height, 3), np.uint8)
    mask = cv2.circle(colorPicker, centerCoords, radius, (255, 255, 255), -1)
    
    frame_blur = cv2.GaussianBlur(frame,(35,35),0)
    colorPickerLow = colorPicker.copy()
    colorPicker = cv2.bitwise_and(frame, mask)
    
    frame_flipped = cv2.bitwise_not(frame)
    colorPickerLow = cv2.bitwise_and(frame_flipped, mask)
    colorPickerLow = cv2.bitwise_not(colorPickerLow)


    h, s, v = cv2.split(colorPicker)
    mh = h[h > 0].mean()
    ms = s[s > 0].mean()
    mv = v[v > 0].mean()

    frame = cv2.circle(frame, centerCoords, radius, (mh, ms, mv), -1)
    #frame = cv2.circle(frame, centerCoords, radius, (255, 255, 255), -1)

    cv2.namedWindow(window_capture_name)
    cv2.setMouseCallback(window_capture_name, on_mouse, window_capture_name)
    cv2.setMouseCallback(window_capture_name, on_mouse, window_capture_name)

    if showColor:
        cv2.imshow(window_capture_name, frame_filteredColor)
    else:
        cv2.imshow(window_capture_name, frame)

    key = cv2.waitKey(30)
    if key == ord('q') or key == 27:
        break

    if key == ord(' '):
        showColor = not showColor
    if key == ord('s'):
        radius -= 1
    if key == ord('w'):
        radius += 1

cap.release()
cv2.destroyAllWindows()