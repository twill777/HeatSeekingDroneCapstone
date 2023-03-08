from pyparrot.Anafi import Anafi
from pyparrot.DroneVision import DroneVision
from pyparrot.Model import Model
import threading
import cv2
import time
from PyQt5.QtGui import QImage
import subprocess
import keyboard
import time
from pathlib import Path
import numpy as np
import asyncio

#Some global variables to be used
isAlive = False
imageFlag = 0
globalNum = 1


class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision

    def save_pictures(self, args):
        global globalNum, imageFlag
        print("saving pictures")
        img = self.vision.get_latest_valid_picture()

        # limiting the pictures to the first 10 just to limit the demo from writing out a ton of files
        if (img is not None):
            filename = "Anafi_Vision%06d.png" % self.index
            cv2.imwrite(filename, img)
            self.index += 1
            imageFlag = 1

def draw_current_photo():

    #Quick demo of returning an image to show in the user window.  Clearly one would want to make this a dynamic image
    
    image = cv2.imread('test_image_000123.png')

    if (image is not None):
        if len(image.shape) < 3 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, byteValue = image.shape
        byteValue = byteValue * width

        qimage = QImage(image, width, height, byteValue, QImage.Format_RGB888)

        return qimage
    else:
        return None
"""
async def read_stream(stream):
    frame_count = 0
    while True:
        data = await stream.read(1024)
        if not data:
            break
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(f"frame{frame_count}.jpg", frame)
        frame_count += 1


async def read_rtsp(rtsp_url):
    cmd = ["ffmpeg", "-i", rtsp_url, "-f", "image2pipe", "-pix_fmt", "bgr24", "-vcodec", "rawvideo", "-"]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    await read_stream(proc.stdout)


def run_rtsp_on_thread(rtsp_url):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(read_rtsp(rtsp_url))
    loop.close()

def main():
    rtsp_url = "rtsp://192.168.42.1/live"
    thread = threading.Thread(target=run_rtsp_on_thread, args=(rtsp_url,))
    thread.start()
    thread.join()
"""
if __name__ == "__main__":
    anafi = Anafi(drone_type="Anafi", ip_address="192.168.42.1")
    print("Connecting to drone")
    success = anafi.connect(5)

    if success:
        print("Connnected!")
        anafiVision = DroneVision(drone_object=anafi, model=Model.ANAFI)

        userVision = UserVision(anafiVision)
        anafiVision.set_user_callback_function(userVision.save_pictures,user_callback_args=None)
        #success = anafiVision.open_video()
        #subprocess.Popen("ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental rtsp://192.168.42.1/live ",
                        #shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        streamProc = subprocess.Popen("ffmpeg -i rtsp://192.168.42.1/live -r 15 Anafi_Vision%06d.png",
                         shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=10**8)
        """
        ffmpeg_cmd = ['ffmpeg',
                      '-i', 'rtsp://192.168.42.1/live',
                      '-r', '1',
                      '-pix_fmt', 'bgr24',
                      '-vcodec', 'rawvideo', '-an', '-sn', '-dn',
                      '-f', 'image2pipe', '-']

        ff_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
        while True:
                raw_frame = ff_proc.stdout.read(1280*720*3)
                frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((720, 1280, 3))
                filename = "Anafi_Vision%06d.png" % globalNum
                cv2.imwrite(filename, frame)
                globalNum += 1
                print("1")
        path = Path("Anafi_Vision%06d.png" % globalNum)
        """
        """
        rtsp_url = "rtsp://192.168.42.1/live"
        print("0")
        cap = cv2.VideoCapture(rtsp_url)
        print("0.5")

        while True:
            print("1")
            ret, frame = cap.read()
            print("2")
            if not ret:
                print("Error reading frame")
                break

            cv2.imshow("RTSP Stream", frame)
            print("3")
            cv2.waitKey(1)
            print("4")
        """
        main()

        autoFlag = 0
        delayFlag = 0
        stopFlag = 0
        if success:
            lastkey = None
            while True:
                """if path.exists():
                    img = cv2.imread("Anafi_Vision%06d.png" % globalNum)
                    imageFlag = 0
                    print("Anafi_Vision%06d.png" % globalNum)
                    print(img.shape)
                    globalNum += 1
                    img = 0
                """
                if keyboard.is_pressed('k'):
                    anafi.safe_takeoff(10)
                    print("Takeoff")
                    stopFlag = 1
                elif keyboard.is_pressed('l'):
                    anafi.safe_land(10)
                    print("Landing")
                    stopFlag = 1
                elif keyboard.is_pressed('y'):
                    anafi.fly_direct(0,25,0,0,0.1)
                    lastkey = 'y'
                    print("Forward")
                    stopFlag = 1
                elif keyboard.is_pressed('h'):
                    anafi.fly_direct(0,-25,0,0,0.1)
                    print("Backward")
                    lastkey = 'h'
                    stopFlag = 1
                elif keyboard.is_pressed('g'):
                    anafi.fly_direct(-25,0,0,0,0.1)
                    print("Left")
                    lastkey = 'g'
                    stopFlag = 1
                elif keyboard.is_pressed('j'):
                    anafi.fly_direct(25,0,0,0,0.1)
                    print("Right")
                    lastkey = 'j'
                    stopFlag = 1
                elif keyboard.is_pressed('1'):
                    anafi.fly_direct(0,0,-100,0,0.1)
                    print("Rotate Right")
                    lastkey = '1'
                    stopFlag = 1
                elif keyboard.is_pressed('2'):
                    anafi.fly_direct(0,0,100,0,0.1)
                    print("Rotate Left")
                    lastkey = '2'
                    stopFlag = 1
                elif keyboard.is_pressed('3'):
                    anafi.fly_direct(0, 0, 0,-20, 0.1)
                    print("Down")
                    lastkey = '3'
                    stopFlag = 1
                elif keyboard.is_pressed('4'):
                    anafi.fly_direct(0, 0, 0, 20, 0.1)
                    print("Up")
                    lastkey = '4'
                    stopFlag = 1
                elif keyboard.is_pressed('5'):
                    print("Autonomous Flying Engaged")
                    autoFlag = 1
                    delayFlag = 1
                    while autoFlag == 1:
                        if delayFlag == 1:
                            anafi.fly_direct(0, 0, 0, 0, 0.3)
                        if keyboard.is_pressed('l'):
                            anafi.safe_land(10)
                        elif keyboard.is_pressed('5'):
                            print("Autonomous Flying Disengaged")
                            anafi.fly_direct(0, 0, 0, 0, 0.3)
                            autoFlag = 0
                        else:
                            anafi.fly_direct(0, 0, 0, 0, 0)
                else:
                    if stopFlag:
                        anafi.fly_direct(0, 0, 0, 0, 0.01)
                        print("stop")
                        stopFlag = 0
                    anafi.fly_direct(0, 0, 0, 0, 0)

        anafi.disconnect()
    else:
        print("Error connecting to drone")
"""
anafi = Anafi(drone_type="Anafi", ip_address="192.168.42.1")
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('20230131_233910.mp4')
# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video file")

# Read until video is completed
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
"""
"""
print("connecting")
success = anafi.connect(10)
print(success)


print("sleeping")
anafi.smart_sleep(1)

#droneVision.open_video()
anafi.ask_for_state_update()
anafi.smart_sleep(20)
#droneVision.close_video()

print("DONE - disconnecting")
anafi.disconnect()
"""