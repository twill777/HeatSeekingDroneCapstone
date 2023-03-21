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

imageScaledown = 4
lastTargetPos = [0, 0]

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

    # Update mouse coords whenever it moves for circle drawing
    if event == cv2.EVENT_MOUSEMOVE:
        centerCoords[0] = x
        centerCoords[1] = y

    # Set bounds based on colors where the mouse had clicked
    if event == cv2.EVENT_LBUTTONDOWN and not showColor:
        showColor = True
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

        marginh = 20
        margins = 40
        marginv = 20

        high_H = (mh + marginh) % max_value_H
        high_S = (ms + margins) % max_value
        high_V = (mv + marginv) % max_value

        low_H = (mh - marginh) % max_value_H
        low_S = (ms - margins) % max_value
        low_V = (mv - marginv) % max_value

        print(low_H, low_S, low_V)
        print(high_H, high_S, high_V)
        print('-')



def findAverageOfPixels(frame):
    extraScaledown = 4
    frame_height, frame_width = frame.shape[:2]
    frame = cv2.resize(frame, [frame_width//extraScaledown, frame_height//extraScaledown])
    frame_height, frame_width = frame.shape[:2]
    xAvg = 0
    yAvg = 0

    whiteCount = 0

    for y in range(frame_width):
        for x in range(frame_height):
            p = frame[x, y]
            if p > 0:
                xAvg += x
                yAvg += y
                whiteCount += 1

    if whiteCount == 0:
        whiteCount = 1

    return [(yAvg * extraScaledown) // whiteCount, (xAvg * extraScaledown) // whiteCount]


def trackFrame(frame, threshold):
    global lastTargetPos
    dragWeight = 0.1
    frame_height, frame_width = frame.shape[:2]

    threshold = cv2.resize(frame, [frame_width//imageScaledown, frame_height//imageScaledown])

    motionCenter = findAverageOfPixels(threshold)
    smoothCenter = motionCenter.copy()
    if lastTargetPos != [0, 0] and motionCenter != [0, 0]:
        smoothCenter[0] = (dragWeight * motionCenter[0]) + ((1-dragWeight) * lastTargetPos[0])
        smoothCenter[1] = (dragWeight * motionCenter[1]) + ((1-dragWeight) * lastTargetPos[1])
    lastTargetPos = smoothCenter

    cv2.line(frame, (int(smoothCenter[0]), 0), (int(smoothCenter[0]), frame_height//imageScaledown), (0, 255, 255), 2)
    cv2.line(frame, (0, int(smoothCenter[1])), (frame_width//imageScaledown, int(smoothCenter[1])), (0, 255, 255), 2)

    return frame


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

    # Convert to HSV for thresholding operation
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold based on bounds then convert to BGR
    if low_H > high_H:
        frame_threshold_low = cv2.inRange(frame_HSV, (0, low_S, low_V), (low_H, high_S, high_V))
        frame_threshold_high = cv2.inRange(frame_HSV, (high_H, low_S, low_V), (180, high_S, high_V))
        frame_threshold = cv2.addWeighted(frame_threshold_low, 1, frame_threshold_high, 1, 0)
        #frame_threshold = cv2.cvtColor(frame_threshold, cv2.COLOR_GRAY2BGR)
        #frame_threshold = cv2.bitwise_and(frame, frame_threshold)
        #frame_threshold = cv2.cvtColor(frame_threshold, cv2.COLOR_BGR2HSV)
        #frame_threshold = cv2.inRange(frame_threshold, (high_H, low_S, low_V), (180, high_S, high_V))
    else:
        frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    frame_threshold_color = cv2.cvtColor(frame_threshold, cv2.COLOR_GRAY2BGR)

    # Map color back onto thresholded frame
    frame_filteredColor = cv2.bitwise_and(frame, frame_threshold_color)

    # Create a black frame
    colorPicker = np.zeros((frame_width, frame_height, 3), np.uint8)

    # Draw a white circle over the mouse on the black frame
    mask = cv2.circle(colorPicker, centerCoords, radius, (255, 255, 255), -1)
    
    # Blur frame ( UNUSED)
    frame_blur = cv2.GaussianBlur(frame,(35,35),0)

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

    if showColor:
        cv2.imshow(window_capture_name, trackFrame(frame, frame_threshold))
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