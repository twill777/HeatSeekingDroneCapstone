import cv2

imageScaledown = 4
lastTargetPos = [0,0]
dragWeight = 0.15

#process a colored image
def procImgClr(frame, low_h, low_s, low_v, high_h, high_s, high_v):
    return procImg(procInBounds(frame, low_h, low_s, low_v, high_h, high_s, high_v))

#proecess a B/W image
def procImg(frame):
    frame = cv2.GaussianBlur(frame,(15,15),0)
    r, frame = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    frame = cv2.GaussianBlur(frame,(21,21),0)
    return frame


def procInBounds(frame, low_h, low_s, low_v, high_h, high_s, high_v):
    cv2.convertScaleAbs(frame, 2)

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold based on bounds then convert to BGR
    if low_h > high_h:
        frame_threshold_low = cv2.inRange(frame_HSV, (0, low_s, low_v), (low_h, high_s, high_v))
        frame_threshold_high = cv2.inRange(frame_HSV, (high_h, low_s, low_v), (180, high_s, high_v))
        frame_threshold = cv2.addWeighted(frame_threshold_low, 1, frame_threshold_high, 1, 0)
    else:
        frame_threshold = cv2.inRange(frame_HSV, (low_h, low_s, low_v), (high_h, high_s, high_v))

    return frame_threshold


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
    frame_height, frame_width = frame.shape[:2]

    threshold = cv2.resize(threshold, [frame_width//imageScaledown, frame_height//imageScaledown])
    frame = cv2.resize(frame, [frame_width // imageScaledown, frame_height // imageScaledown])

    motionCenter = findAverageOfPixels(threshold)
    smoothCenter = motionCenter.copy()
    if lastTargetPos != [0, 0] and motionCenter != [0, 0]:
        smoothCenter[0] = (dragWeight * motionCenter[0]) + ((1-dragWeight) * lastTargetPos[0])
        smoothCenter[1] = (dragWeight * motionCenter[1]) + ((1-dragWeight) * lastTargetPos[1])
    lastTargetPos = smoothCenter

    cv2.line(frame, (int(smoothCenter[0]), 0), (int(smoothCenter[0]), frame_height//imageScaledown), (0, 255, 255), 2)
    cv2.line(frame, (0, int(smoothCenter[1])), (frame_width//imageScaledown, int(smoothCenter[1])), (0, 255, 255), 2)

    frame = cv2.resize(frame, [frame_width, frame_height])

    return frame