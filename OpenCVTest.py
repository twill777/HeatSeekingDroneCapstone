import cv2
from matplotlib import pyplot as plt
import numpy as np

#process a colored image
def procImgClr(frame):
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, frame)
    frame = procRed(frame)
    
    return procImg(frame)

#proecess a B/W image
def procImg(frame):
    frame = cv2.GaussianBlur(frame,(7,7),0)
    r, frame = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return frame


def procRed(frame):
    cv2.convertScaleAbs(frame, 2)
    frame_height, frame_width = frame.shape[:2]

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (120, 120, 80), (170, 240, 255))

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

OPENCV_OBJECT_TRACKERS = {
		"kcf": cv2.TrackerKCF_create
	}

keptFrames = 8
imageScaledown = 4
enableCVTracking = False

# Only run image processing on each nth frame
frameSkips = 25

video = None
video = cv2.VideoCapture("test_videoRed.MOV")
frame = None

t = OPENCV_OBJECT_TRACKERS["kcf"]

name = "CapstoneTracker"

tracker = t()
tracker2 = t()
mtTracker = t()
mtTracker2 = t()
    
ret, frame = video.read()
frame_height, frame_width = frame.shape[:2]
frame = cv2.resize(frame, [frame_width//imageScaledown, frame_height//imageScaledown])
    
mtFrame = procImgClr(frame)
cleanFrame = mtFrame.copy()
mtFrame = cv2.cvtColor(mtFrame, cv2.COLOR_GRAY2BGR)
    
lastFramesProc = []
lastFrames = []

lastTargetPos = [0, 0]
    
for i in range(keptFrames):
    lastFramesProc.append(cleanFrame.copy())
    lastFrames.append(frame.copy())

output = cv2.VideoWriter(name+'.mp4', 
                        cv2.VideoWriter_fourcc(*'mp4v'), 60.0, 
                        (frame_width//imageScaledown, frame_height//imageScaledown), True)
mtOutput = cv2.VideoWriter(name+'_MT.mp4', 
                        cv2.VideoWriter_fourcc(*'mp4v'), 60.0, 
                        (frame_width//imageScaledown, frame_height//imageScaledown), True)
if not ret:
    print('cannot read the video')

if enableCVTracking:
    # Select the bounding box in the first frame
    bbox1 = cv2.selectROI(frame, False)
    bbox2 = bbox1
    
    # Tracker 1 : tracks colored moving object
    ret = tracker.init(frame, bbox1)
    # Tracker 2 : tracks colored moving object, reinitializes to last frame
    ret = tracker2.init(frame, bbox2)

mtInitialized = False

retMT = False
retMT2 = False
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
        
    mtFrame = procImgClr(frame)
    cleanFrame = mtFrame.copy()

    for i in range(keptFrames):
        mtFrame = cleanFrame - lastFramesProc[i]

    mtFrame = cv2.medianBlur(mtFrame, 11)
    processLoops = 1
    for i in range(processLoops):
        mtFrame = procImg(mtFrame)

    for i in range(keptFrames-1,0,-1):
        lastFramesProc[i] = lastFramesProc[i - 1].copy()
        lastFrames[i] = lastFrames[i - 1].copy()
    lastFramesProc[0] = cleanFrame.copy()
    lastFrames[0] = frame.copy()

    motionCenter = findAverageOfPixels(cleanFrame)
    smoothCenter = motionCenter.copy()
    if lastTargetPos != [0, 0] and motionCenter != [0, 0]:
        smoothCenter[0] = (0.15 * motionCenter[0]) + (0.85 * lastTargetPos[0])
        smoothCenter[1] = (0.15 * motionCenter[1]) + (0.85 * lastTargetPos[1])
    lastTargetPos = smoothCenter

    mtFrame = cv2.cvtColor(mtFrame, cv2.COLOR_GRAY2BGR)
    timer = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if enableCVTracking:
        ret1, bbox1 = tracker.update(frame)
        ret2, bbox2 = tracker2.update(frame)
        tracker2 = t()
        ret = tracker2.init(lastFrames[1], bbox2)
        ret2, bbox2 = tracker2.update(frame)

        # +++ Motion Tracking +++
        if mtInitialized:
            retMT, bboxMT = mtTracker.update(mtFrame)

            retMT2, bboxMT2 = mtTracker2.update(mtFrame)
            mtTracker2 = t()
            ret = mtTracker2.init(lastFramesProc[1], bboxMT2)
            retMT2, bboxMT2 = mtTracker.update(mtFrame)
        # =======================

        elif current_frame >= keptFrames:
            bboxMT = bbox1
            bboxMT2 = bbox1
            # Tracker MT : tracks b/w moving object
            ret = mtTracker.init(mtFrame, bboxMT)
            mtInitialized = True
            # Tracker MT2 : tracks b/w moving object, reinitializes to last frame
            ret = mtTracker2.init(mtFrame, bboxMT2)
            mtInitialized = True

        if ret1:
            p1 = (int(bbox1[0]), int(bbox1[1]))
            p2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else:
            cv2.putText(frame, "Tracking 1   failure detected", (100,80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        if ret2:
            p1 = (int(bbox2[0]), int(bbox2[1]))
            p2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
        else:
            cv2.putText(frame, "Tracking 2   failure detected", (100,110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # +++ Motion Tracking +++
        if mtInitialized:
            if retMT:
                p1 = (int(bboxMT[0]), int(bboxMT[1]))
                p2 = (int(bboxMT[0] + bboxMT[2]), int(bboxMT[1] + bboxMT[3]))
                cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1)
                cv2.rectangle(mtFrame, p1, p2, (0,0,255), 2, 1)
            else:
                cv2.putText(frame, "Tracking MT  failure detected", (100,140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            if retMT2:
                p1 = (int(bboxMT2[0]), int(bboxMT2[1]))
                p2 = (int(bboxMT2[0] + bboxMT2[2]), int(bboxMT2[1] + bboxMT2[3]))
                cv2.rectangle(frame, p1, p2, (0,255,255), 2, 1)
                cv2.rectangle(mtFrame, p1, p2, (0,255,255), 2, 1)
            else:
                cv2.putText(frame, "Tracking MT2 failure detected", (100,170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        # =======================

    #cv2.line(frame, (int(motionCenter[0]), 0), (int(motionCenter[0]), frame_height//imageScaledown), (0, 0, 255), 3)
    #cv2.line(frame, (0, int(motionCenter[1])), (frame_width//imageScaledown, int(motionCenter[1])), (0, 0, 255), 3)
    cv2.line(frame, (int(smoothCenter[0]), 0), (int(smoothCenter[0]), frame_height//imageScaledown), (0, 255, 255), 3)
    cv2.line(frame, (0, int(smoothCenter[1])), (frame_width//imageScaledown, int(smoothCenter[1])), (0, 255, 255), 3)

    cv2.line(mtFrame, (int(motionCenter[0]), 0), (int(motionCenter[0]), frame_height//imageScaledown), (0, 0, 255), 3)
    cv2.line(mtFrame, (0, int(motionCenter[1])), (frame_width//imageScaledown, int(motionCenter[1])), (0, 0, 255), 3)

    cv2.putText(frame, name+" Tracker", (100,20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

    cv2.imshow("Tracking", frame)
    output.write(frame)

    cv2.imshow("Clean Frame", cleanFrame)
    cv2.imshow("Motion Tracker", mtFrame)
    mtOutput.write(mtFrame)

    k = cv2.waitKey(1) & 0xff
    if k == 27 : break
            
video.release()
output.release()
mtOutput.release()
cv2.destroyAllWindows()