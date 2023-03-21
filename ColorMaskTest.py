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
shownFrame = 3
radius = 20

editLowBound = 0

baseMargins = [10,40,20]

marginh = baseMargins[0]
margins = baseMargins[1]
marginv = baseMargins[2]
hshift = 5
sshift = 5
vshift = 5

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
        shownFrame = 2
        global low_H
        global low_S
        global low_V
        global high_H
        global high_S
        global high_V
        global marginh
        global margins
        global marginv

        # Convert color picker to HSV
        colorPicker_HSV = cv2.cvtColor(colorPicker, cv2.COLOR_BGR2HSV)

        # Extract the mean h, s, and v channels from the masked circle
        h, s, v = cv2.split(colorPicker_HSV)
        mh = h[h > 0].mean()
        ms = s[s > 0].mean()
        mv = v[v > 0].mean()

        high_H = (mh + marginh) % max_value_H
        high_S = min(max_value, (ms + margins))
        high_V = min(max_value, (mv + marginv))

        low_H = (mh - marginh) % max_value_H
        low_S = max(0, (ms - margins))
        low_V = max(0, (mv - marginv))

        marginh = baseMargins[0]
        margins = baseMargins[1]
        marginv = baseMargins[2]

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
    framePicker = frame.copy()
    framePicker = cv2.circle(framePicker, centerCoords, radius, (mb, mg, mr), -1)

    cv2.namedWindow(window_capture_name)
    cv2.setMouseCallback(window_capture_name, on_mouse, window_capture_name)
    cv2.setMouseCallback(window_capture_name, on_mouse, window_capture_name)

    frame_threshold_output = cv2.cvtColor(frame_threshold_proc, cv2.COLOR_GRAY2BGR)

    for f in [frame_filteredColor, frame_threshold_output]:
        cv2.putText(f, 'High H: %.0f' % high_H, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255 * editLowBound, 255 * editLowBound, 255), 2)

        cv2.putText(f, 'Low H: %.0f' % low_H, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255 * (1 - editLowBound), 255 * (1 - editLowBound), 255), 2)

        cv2.putText(f, 'High S: %.0f' % high_S, (180, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255 * editLowBound, 255 * editLowBound, 255), 2)

        cv2.putText(f, 'Low S: %.0f' % low_S, (180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255 * (1 - editLowBound), 255 * (1 - editLowBound), 255), 2)

        cv2.putText(f, 'High V: %.0f' % high_V, (350, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255 * editLowBound, 255 * editLowBound, 255), 2)

        cv2.putText(f, 'Low V: %.0f' % low_V, (350, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255 * (1 - editLowBound), 255 * (1 - editLowBound), 255), 2)

    if shownFrame == 0:
        cv2.imshow(window_capture_name, trackFrame(frame, frame_threshold_proc))
    elif shownFrame == 1:
        cv2.imshow(window_capture_name, frame_threshold_output)
    elif shownFrame == 2:
        cv2.imshow(window_capture_name, frame_filteredColor)
    elif shownFrame == 3:
        cv2.imshow(window_capture_name, framePicker)

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
        high_H = (high_H - (hshift * (1 - editLowBound))) % max_value_H
        low_H = (low_H - (hshift * editLowBound)) % max_value_H
    if key == ord('u'):
        high_H = (high_H + (hshift * (1 - editLowBound))) % max_value_H
        low_H = (low_H + (hshift * editLowBound)) % max_value_H
    if key == ord('j'):
        high_S = max(low_S + 1, high_S - (sshift * (1 - editLowBound)))
        low_S = max(0, low_S - (sshift * editLowBound))
    if key == ord('i'):
        high_S = min(max_value, high_S + (sshift * (1 - editLowBound)))
        low_S = min(high_S - 1, low_S + (sshift * editLowBound))
    if key == ord('k'):
        high_V = max(low_V + 1, high_V - (vshift * (1 - editLowBound)))
        low_V = max(0, low_V - (vshift * editLowBound))
    if key == ord('o'):
        high_V = min(max_value, high_V + (vshift * (1 - editLowBound)))
        low_V = min(high_V - 1, low_V + (vshift * editLowBound))

    if key == ord('t'):
        editLowBound = 1 - editLowBound
    if key == ord('f'):
        temp = high_H
        high_H = low_H
        low_H = temp

    if key == ord('s'):
        radius -= 1
    if key == ord('w'):
        radius += 1

cap.release()
cv2.destroyAllWindows()