import cv2, math
import urllib.request
import numpy as np
from roboflow import Roboflow

#  Define known variables
count = 0
KNOWN_FOCAL_LENGTH = 758.5 # average focal length obtained in calibration.py
KNOWN_DISTANCE_BETWEEN_CAMERA = 3.1 # distance of camera in cm
SCALE_FACTOR = 0.00171316

#  Define cameras left and right
cam1='http://172.20.10.3/cam-hi.jpg' #Left camera
cam2='http://172.20.10.2/cam-hi.jpg' #Right camera

#  Specify folders to save pictures taken by both cameras  
folderL = r'./images/Stereo/Real/L/'
folderR = r'./images/Stereo/Real/R/'
folderL = r'./images/Stereo/Real/L/'
folder = r'./images/Stereo/Real/'

#  Read the depth map xml file from calibration.py
cv_file = cv2.FileStorage("./MonoParam7.xml", cv2.FILE_STORAGE_READ)

Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()


def takeCurrentPic(L, R):
    cv2.imwrite(folderL+'imageL%d.png'%count,L)
    cv2.imwrite(folderR+'imageR%d.png'%count,R)
    print("pictures taken.")

def objectDetection():

    rf1 = Roboflow(api_key="QycZdcWgOjjB6XYdD6UG")
    project1 = rf1.workspace().project("flower-measurement")
    model1 = project1.version(1).model

    predictL = model1.predict(folderL+'imageL%d.png'%count).json()
    model1.predict(folderL+'imageL%d.png'%count).save(folderL+'predictL%d.png'%count)
    for pred in predictL['predictions']:
        x1 = int(pred['x'])
        y1 = int(pred['y'])
        width1 = pred['width']
        class1 = str(pred['class'])

    # print(width1)

    rf2 = Roboflow(api_key="QycZdcWgOjjB6XYdD6UG")
    project2 = rf2.workspace().project("flower-measurement")
    model2 = project2.version(1).model

    predictR = model2.predict(folderR+'imageR%d.png'%count).json()
    model2.predict(folderR+'imageR%d.png'%count).save(folderR+'predictR%d.png'%count)
    for pred in predictR['predictions']:
        x2 = int(pred['x'])
        y2 = int(pred['y'])
        width2 = pred['width']
        class2 = str(pred['class'])

    if 'width1' and 'width2' in locals():
        print(width1)
        print(width2)
        print("Object detected.")
        return x1, x2, y1, y2, class1, class2, width1, width2
    else:
        print("No object present")
        return None, None, None, None, None, None, None, None

def getDistance(disparity):
    try:
        Distance = (KNOWN_FOCAL_LENGTH) * (KNOWN_DISTANCE_BETWEEN_CAMERA / disparity)
        if Distance != float('inf'):
            return Distance
        else:
            return 0
    except:
        return "No object detected"
    
def getArea(width1,width2):
    pie = (22/7)
    width = (width1+width2)/2
    area = (1/4) * pie * pow(width,2)
    print("Area in pixel: %f" %area)

    return area

def showResults(classL,classR,xL, xR, yL, yR, distance, width,area):
    L = cv2.imread(folderL+'predictL%d.png'%count)
    R = cv2.imread(folderR+'predictR%d.png'%count)

    dL = "Distance = "+str(int(distance))+"cm"
    w = "width = "+str(int(width))+"px"
    actual_area = "area = "+str(int(area))+"cm^2"

    cv2.putText(L,classL,(int(xL),int(yL)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.putText(L,dL,(int(xL),int(yL+50)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.putText(L,w,(int(xL),int(yL+100)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.putText(L,actual_area,(int(xL),int(yL+150)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.putText(R,classR,(int(xR),int(yR)),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    
    LR = np.concatenate((L, R), axis=1) 

    cv2.imwrite(folder+'img%d.png'%count, LR)
    cv2.imshow("predicted values",LR)

while True:
#  Get images from both cameras and save it to a frame
    img_respL = urllib.request.urlopen(cam1)
    imgnpL = np.array(bytearray(img_respL.read()),dtype=np.uint8)
    frameL = cv2.imdecode(imgnpL,-1)
    frameL = cv2.rotate(frameL, cv2.ROTATE_180)

    img_respR = urllib.request.urlopen(cam2)
    imgnpR = np.array(bytearray(img_respR.read()),dtype=np.uint8)
    frameR = cv2.imdecode(imgnpR,-1)
    frameR  = cv2.rotate(frameR, cv2.ROTATE_180)

    #  Remap the frames using the depth map matrix
    imgL_remapped = cv2.remap(frameL, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    imgR_remapped = cv2.remap(frameR, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    #  Display the remapped left and right views on a window
    Hori = np.concatenate((imgL_remapped, imgR_remapped), axis=1) 
    cv2.imshow("live transmission", Hori)
    key=cv2.waitKey(33)

    if key==ord('k'):
        count = count +1
        #  1. Take current frame and save to folders. (including the models)
        takeCurrentPic(imgL_remapped, imgR_remapped)
        #  2. Use the model coordinates of lates taken picture
        xCenter_L, xCenter_R, yCenter_L, yCenter_R, classL, classR, width1, width2 = objectDetection()
        if width1 and width2 != None:
            #  4. Calculate disparity with center of both objects in each images images
            disparity = abs(xCenter_L - xCenter_R)
            print("disparity = %f"%disparity)
            #  4. Calculate distance using disparity
            distance = getDistance(disparity)
            print("Distance = %f"%distance)
            #  5. Calculate area of ROI
            area = getArea(width1,width2)
            avg_width = (width1 + width2)/2
            #  6. Write a formula to estimate the flower actual surface area
            actual_area = area * math.pow(SCALE_FACTOR*distance,2) #/ math.pow(distance,2)
            print("estimated area = %f"%actual_area)
            showResults(classL,classR,xCenter_L,xCenter_R,yCenter_L,yCenter_R,distance,avg_width,actual_area)
            #key = cv2.waitKey(0)

    elif key==ord('l'):
        cv2.destroyWindow("predicted values")

    elif key == ord('q'):
        break

cv2.destroyAllWindows()
print("Program closed")