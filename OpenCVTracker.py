import cv2
from matplotlib import pyplot as plt
import numpy as np

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
    frame_height, frame_width = frame.shape[:2]

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_mask = cv2.inRange(frame_HSV, (low_h, low_s, low_v), (high_h, high_s, high_v))

    return frame_mask


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


def drawHistogram(frame):
    frame_height, frame_width = frame.shape[:2]

    # Histogram x vals, y vals, and [xmax, ymax]
    histogramValues = [[], [], [0,0]]

    for y in range(frame_width):
        histogramValues[0].append(0)
    for x in range(frame_height):
        histogramValues[1].append(0)
            
    for y in range(frame_width):
        for x in range(frame_height):
            f = frame[x, y]
            histogramValues[0][y] += f
            histogramValues[1][x] += f
            if histogramValues[2][0] < f:
                histogramValues[2][0] = f
            if histogramValues[2][1] < f:
                histogramValues[2][1] = f

    histogramValues[0] = [x / (255 * frame_height) for x in histogramValues[0]]
    histogramValues[1] = [y / (255 * frame_height) for y in histogramValues[1]]

    return histogramValues

imageScaledown = 4
lastTargetPos = [0, 0]

def trackFrame(frame, low_h, low_s, low_v, high_h, high_s, high_v):
    global lastTargetPos
    dragWeight = 0.1
    frame_height, frame_width = frame.shape[:2]

    frame = cv2.resize(frame, [frame_width//imageScaledown, frame_height//imageScaledown])
    cleanFrame = procImgClr(frame, low_h, low_s, low_v, high_h, high_s, high_v)

    motionCenter = findAverageOfPixels(cleanFrame)
    smoothCenter = motionCenter.copy()
    if lastTargetPos != [0, 0] and motionCenter != [0, 0]:
        smoothCenter[0] = (dragWeight * motionCenter[0]) + ((1-dragWeight) * lastTargetPos[0])
        smoothCenter[1] = (dragWeight * motionCenter[1]) + ((1-dragWeight) * lastTargetPos[1])
    lastTargetPos = smoothCenter

    cv2.line(frame, (int(smoothCenter[0]), 0), (int(smoothCenter[0]), frame_height//imageScaledown), (0, 255, 255), 2)
    cv2.line(frame, (0, int(smoothCenter[1])), (frame_width//imageScaledown, int(smoothCenter[1])), (0, 255, 255), 2)

    return frame


def trackVideo(inName, outName):
    # Only run image processing on each nth frame
    frameSkips = 25

    video = None
    video = cv2.VideoCapture(inName)
    frame = None
    
    ret, frame = video.read()
    if not ret:
        print('cannot read the video')
    current_frame = 0

    while True:
        current_frame += 1
        if current_frame % frameSkips != 0:
            continue
        ret, frame = video.read()
        if frame is None:
            break
        if not ret:
            print('something went wrong')
            break
        
        cv2.imshow("Object Tracking", trackFrame(frame, 120, 120, 80, 170, 240, 255))

        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
            
    video.release()
    cv2.destroyAllWindows()


def slowTrackVideo(inName, outName):
    imageScaledown = 4
    dragWeight = 0.1

    # If the target is on this outer percent horizontally/vertically, move in that direction
    moveBounds = [0.25, 0.35]

    # Only run image processing on each nth frame
    frameSkips = 25

    video = None
    video = cv2.VideoCapture(inName)
    frame = None
    
    ret, frame = video.read()
    frame_height, frame_width = frame.shape[:2]
    frame = cv2.resize(frame, [frame_width//imageScaledown, frame_height//imageScaledown])
    
    cleanFrame = procImgClr(frame)

    lastTargetPos = [0, 0]
    lastObjDist = 0

    trackerOutput = cv2.VideoWriter(outName+'.mp4', 
                            cv2.VideoWriter_fourcc(*'mp4v'), 60.0, 
                            (frame_width//imageScaledown * 3, frame_height//imageScaledown * 3), True)
    if not ret:
        print('cannot read the video')

    mtInitialized = False

    current_frame = 0

    while True:
        current_frame += 1
        if current_frame % frameSkips != 0:
            continue
        ret, frame = video.read()
        if frame is None:
            break
        frame = cv2.resize(frame, [frame_width//imageScaledown, frame_height//imageScaledown])

        if not ret:
            print('something went wrong')
            break
        
        cleanFrame = procImgClr(frame)

        histo = drawHistogram(cleanFrame)
        motionCenter = findAverageOfPixels(cleanFrame)
        smoothCenter = motionCenter.copy()
        if lastTargetPos != [0, 0] and motionCenter != [0, 0]:
            smoothCenter[0] = (dragWeight * motionCenter[0]) + ((1-dragWeight) * lastTargetPos[0])
            smoothCenter[1] = (dragWeight * motionCenter[1]) + ((1-dragWeight) * lastTargetPos[1])
        lastTargetPos = smoothCenter

        timer = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Recreate processing frames
        cv2.convertScaleAbs(frame, 2)
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_mask = cv2.inRange(frame_HSV, (120, 120, 80), (170, 240, 255))
        frame1 = cv2.GaussianBlur(frame_mask,(15,15),0)
        r, frame2 = cv2.threshold(frame1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        frame3 = cv2.GaussianBlur(frame2,(21,21),0)
        # ---------------------------

        cv2.line(frame, (int(smoothCenter[0]), 0), (int(smoothCenter[0]), frame_height//imageScaledown), (0, 255, 255), 2)
        cv2.line(frame, (0, int(smoothCenter[1])), (frame_width//imageScaledown, int(smoothCenter[1])), (0, 255, 255), 2)

        cleanFrame = cv2.cvtColor(cleanFrame, cv2.COLOR_GRAY2BGR)
        frame_mask = cv2.cvtColor(frame_mask, cv2.COLOR_GRAY2BGR)

        bboxFrame = cleanFrame.copy()
        controlFrame = cleanFrame.copy()
        shape_height, shape_width = frame.shape[:2]

        first_coords = [0, 0]
        last_coords = [0, 0]

        # Compass directions - up, right, down, left
        controlOut = [0,0,0,0]

        if smoothCenter != [0,0]:
            xMoveZone = moveBounds[0] * shape_width
            controlOut[1] = max(smoothCenter[0] - (shape_width - xMoveZone), 0) / xMoveZone
            controlOut[3] = max(xMoveZone - smoothCenter[0], 0) / xMoveZone

            yMoveZone = moveBounds[1] * shape_height
            controlOut[0] = max(yMoveZone - smoothCenter[1], 0) / yMoveZone
            controlOut[2] = max(smoothCenter[1] - (shape_height - yMoveZone), 0) / yMoveZone

        cv2.putText(controlFrame, '%.2f' % controlOut[0], (shape_width//2 - 30,shape_height//10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)
        cv2.putText(controlFrame, '%.2f' % controlOut[1], (shape_width - shape_width//10 - 15,shape_height//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)
        cv2.putText(controlFrame, '%.2f' % controlOut[2], (shape_width//2 - 30,shape_height - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)
        cv2.putText(controlFrame, '%.2f' % controlOut[3], (10,shape_height//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)
    
        for x in range(shape_width):
            hx = int((histo[0][x] * frame_height))

            if hx > histo[2][0] * 0.1:
                last_coords[0] = x
                if first_coords[0] == 0:
                    first_coords[0] = x

            cv2.circle(bboxFrame, (x, hx), 1, (255, 255, 0), 2)

        for y in range(shape_height):
            hy = int((histo[1][y] * frame_width))

            if hy > histo[2][1] * 0.1:
                last_coords[1] = y
                if first_coords[1] == 0:
                    first_coords[1] = y

            cv2.circle(bboxFrame, (hy, y), 1, (0, 255, 255), 2)

        diffX = last_coords[0] - first_coords[0]
        diffY = last_coords[1] - first_coords[1]
        dist = max(diffX, diffY)
        if lastObjDist != 0:
            dist = int((dist * dragWeight) + (lastObjDist * (1-dragWeight)))
        lastObjDist = dist
       
        cv2.line(controlFrame, (int(moveBounds[0] * shape_width), 0), (int(moveBounds[0] * shape_width), frame_height//imageScaledown), (0, 100, 255), 2)
        cv2.line(controlFrame, (int(shape_width - (moveBounds[0] * shape_width)), 0), (int(shape_width - (moveBounds[0] * shape_width)), frame_height//imageScaledown), (0, 100, 255), 2)
        cv2.line(controlFrame, (0, int(moveBounds[1] * shape_height)), (frame_width//imageScaledown, int(moveBounds[1] * shape_height)), (0, 100, 255), 2)
        cv2.line(controlFrame, (0, int(shape_height - (moveBounds[1] * shape_height))), (frame_width//imageScaledown, int(shape_height - (moveBounds[1] * shape_height))), (0, 100, 255), 2)
        cv2.line(controlFrame, (int(smoothCenter[0]), 0), (int(smoothCenter[0]), frame_height//imageScaledown), (0, 255, 255), 2)
        cv2.line(controlFrame, (0, int(smoothCenter[1])), (frame_width//imageScaledown, int(smoothCenter[1])), (0, 255, 255), 2)

        p1 = (int(first_coords[0]), int(first_coords[1]))
        p2 = (int(last_coords[0]), int(last_coords[1]))
        cv2.rectangle(bboxFrame, p1, p2, (255,50,50), 2, 1)
        cv2.circle(bboxFrame, (int(smoothCenter[0]), int(smoothCenter[1])), dist // 2, (255, 100, 0), 2)
        cv2.putText(bboxFrame, '%.2f' % (dist / min(shape_width, shape_height)), (shape_width - 60, shape_height - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255,255,255),2)

        rad = int(np.sqrt(((motionCenter[0] - smoothCenter[0]) ** 2) + ((motionCenter[1] - smoothCenter[1]) ** 2)))

        cv2.line(cleanFrame, (int(motionCenter[0]), 0), (int(motionCenter[0]), frame_height//imageScaledown), (0, 0, 255), 2)
        cv2.line(cleanFrame, (0, int(motionCenter[1])), (frame_width//imageScaledown, int(motionCenter[1])), (0, 0, 255), 2)
        cv2.line(cleanFrame, (int(motionCenter[0]), int(motionCenter[1])), (int(smoothCenter[0]), int(smoothCenter[1])), (0, 255, 0), 2)
        cv2.circle(cleanFrame, (int(smoothCenter[0]), int(smoothCenter[1])), rad, (0, 255, 0), 2)

        row1 = np.hstack([frame, frame_HSV, frame_mask])
        row2 = np.hstack([frame1, frame2, frame3])
        row3 = np.hstack([cleanFrame, bboxFrame, controlFrame])

        row2 = cv2.cvtColor(row2, cv2.COLOR_GRAY2BGR)
        collage = np.vstack([row1, row2, row3])

        cv2.imshow("Object Tracking", collage)

        trackerOutput.write(collage)

        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
            
    video.release()
    trackerOutput.release()
    cv2.destroyAllWindows()

for i in range(6,7):
    inName = "test_video" + str(i) + ".mp4"
    outName = "Tracker_" + str(i) + ".mp4"
    trackVideo(inName, outName)