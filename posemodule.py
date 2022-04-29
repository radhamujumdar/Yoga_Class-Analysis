import cv2
import mediapipe as mp
import math
from mediapipe.python._framework_bindings import packet
# from prediction import *

class poseDetector():

    def __init__(self, mode=False,upBody=False,maxHands=2, smooth=True,model_complexity=2,detectionCon=0.5, trackCon=0.5
    ):

        self.mode = mode
        self.upBody = upBody
        self.maxHands=maxHands
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.model_complexity=model_complexity
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.upBody,self.maxHands, self.smooth,self.model_complexity,self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3,final_label, draw=True):
        self.final_label=final_label
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        print(angle)

        def angle_matches():
            cv2.line(img, (x1, y1), (x2, y2), (0,128,0), 3)
            cv2.line(img, (x3, y3), (x2, y2), (0,128,0), 3)
            cv2.circle(img, (x1, y1), 10, (0,128,0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0,128,0), 2)
            cv2.circle(img, (x2, y2), 10, (0,128,0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0,128,0), 2)
            cv2.circle(img, (x3, y3), 10, (0,128,0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0,128,0), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,128, 0), 2)
        def angle_not_matches():
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (0, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)


        # Draw
        if draw:
            if self.final_label==0:
                if angle >= 220 and angle <=260:
                    angle_matches()
                else:
                    angle_not_matches()
            elif self.final_label==1:
                if angle >= 10 and angle <=40:
                    angle_matches()
                elif angle >= 315 and angle <=350:
                    angle_matches()
                else:
                    angle_not_matches()
            elif self.final_label==2:
                if angle >= 170 and angle <=200:
                    angle_matches()
                else:
                    angle_not_matches()
            elif self.final_label==3:
                if angle >= 165 and angle <=180:
                    angle_matches()
                elif angle>=168 and angle <=190:
                    angle_matches()
                else:
                    angle_not_matches()
            elif self.final_label==4:
                if angle >= 140 and angle <=170:
                    angle_matches()
                elif angle>=200 and angle<=230:
                    angle_matches()
                else:
                    angle_not_matches()


            elif self.final_label==5:
                if angle >= 12 and angle <=45:
                    angle_matches()
                else:
                    angle_not_matches()
            return angle

# def main():
#         #     cap = cv2.VideoCapture('PoseVideos/1.mp4')
#         detector = poseDetector()
#         # while True:
#         #         success, img = cap.read()
#         image=cv2.imread(found)
#         #         print(image.shape)
#         img = detector.findPose(image)
#         lmList = detector.findPosition(img, draw=False)
#             # if len(lmList) != 0:
#             #     print(lmList[14])
#             #     cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
#
#
#
#         cv2.imshow("Image", img)
#         cv2.waitKey(1)
#
#
# if __name__ == "__main__":
#   main()
