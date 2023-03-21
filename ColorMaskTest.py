from __future__ import print_function
import cv2
import argparse
import numpy as np
from pynput.mouse import Controller
from TrackerProcessing import *

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
shownFrame = 0
radius = 20

marginh = 20
margins = 40
marginv = 20
hshift = 10
sshift = 15
vshift = 2

colorPicker = np.zeros((1, 1, 3), np.uint8)
colorPickerLow = np.zeros((1, 1, 3), np.uint8)

imageScaledown = 4
lastTargetPos = [0, 0]

def on_mouse(event,x,y,flags,param):
    global shownFrame

    # Update mouse coords whenever it moves for circle drawing
    if event == cv2.EVENT_MOUSEMOVE:
        centerCoords[0] = x
        centerCoords[1] = y

    # Set bounds based on colors where the mouse had clicked
    if event == cv2.EVENT_LBUTTONDOWN and shownFrame == 3:
        shownFrame = 1
        global low_H
        global low_S
        global low_V
        global high_H
        global high_S
        global high_V

        # Convert color picker to HSV
        colorPicker_HSV = cv2.cvtColor(colorPicker, cv2.COLOR_BGR2HSV)

        # Extract the mean h, s, and v channels from the masked circle
        h, s, v = cv2.split(colorPicker_HSV)
        mh = h[h > 0].mean()
        ms = s[s > 0].mean()
        mv = v[v > 0].mean()

        high_H = (mh + marginh) % max_value_H
        high_S = (ms + margins) % max_value
        high_V = (mv + marginv) % max_value

        low_H = (mh - marginh) % max_value_H
        low_S = (ms - margins) % max_value
        low_V = (mv - marginv) % max_value

#cap = cv2.VideoCapture("test_video5.mp4")
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    frame_width, frame_height = frame.shape[:2]

    frame_threshold = procInBounds(frame, low_H, low_S, low_V, high_H, high_S, high_V)
    frame_threshold_proc = procImg(frame_threshold)

    frame_threshold_color = cv2.cvtColor(frame_threshold, cv2.COLOR_GRAY2BGR)

    # Map color back onto thresholded frame
    frame_filteredColor = cv2.bitwise_and(frame, frame_threshold_color)

    # Create a black frame
    colorPicker = np.zeros((frame_width, frame_height, 3), np.uint8)

    # Draw a white circle over the mouse on the black frame
    mask = cv2.circle(colorPicker, centerCoords, radius, (255, 255, 255), -1)

    # Multiply the frame by the mask, so that colorPicker is a black frame with a circle window matching what's on "frame" at that point
    colorPicker = cv2.bitwise_and(frame, mask)
    
    # Invert the frame colors
    frame_flipped = cv2.bitwise_not(frame)

    # Extract the mean blue, red, and green channels from the masked circle
    b, g, r = cv2.split(colorPicker)
    mb = b[b > 0].mean()
    mg = g[g > 0].mean()
    mr = r[r > 0].mean()

    # Draw the mean color of the circle around the mouse onto the render frame
    frame = cv2.circle(frame, centerCoords, radius, (mb, mg, mr), -1)

    cv2.namedWindow(window_capture_name)
    cv2.setMouseCallback(window_capture_name, on_mouse, window_capture_name)
    cv2.setMouseCallback(window_capture_name, on_mouse, window_capture_name)

    if shownFrame == 0:
        cv2.imshow(window_capture_name, trackFrame(frame, frame_threshold_proc))
    elif shownFrame == 1:
        cv2.imshow(window_capture_name, frame_threshold_proc)
    elif shownFrame == 2:
        cv2.imshow(window_capture_name, frame_filteredColor)
    elif shownFrame == 3:
        cv2.imshow(window_capture_name, frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

    if key == ord(' '):
        shownFrame = shownFrame + 1 % 4
    if key == ord('1'):
        shownFrame = 0
    if key == ord('2'):
        shownFrame = 1
    if key == ord('3'):
        shownFrame = 2
    if key == ord('4'):
        shownFrame = 3

    if key == ord('h'):
        marginh -= hshift
        high_H = (high_H - hshift) % max_value
        low_H = (low_H + hshift) % max_value_H
    if key == ord('u'):
        marginh += hshift
        high_H = (high_H + hshift) % max_value
        low_H = (low_H - hshift) % max_value_H
    if key == ord('j'):
        margins -= sshift
        high_S = (high_S - sshift) % max_value
        low_S = (low_S + sshift) % max_value_H
    if key == ord('i'):
        margins += sshift
        high_S = (high_S + sshift) % max_value
        low_S = (low_S - sshift) % max_value_H
    if key == ord('k'):
        marginv -= vshift
        high_V = (high_V - vshift) % max_value
        low_V = (low_V + vshift) % max_value_H
    if key == ord('o'):
        marginv += vshift
        high_V = (high_V + vshift) % max_value
        low_V = (low_V - vshift) % max_value_H

    if key == ord('s'):
        radius -= 1
    if key == ord('w'):
        radius += 1

cap.release()
cv2.destroyAllWindows()