import cv2
import numpy as np

class CircleDetector:
    def detect_circles(self, frame):
        C_m = 30
        color = cv2.resize(frame, dsize=(int(frame.shape[1]), int(frame.shape[0])))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        frame = cv2.medianBlur(frame, ksize=5)
        frame = cv2.Canny(frame, threshold1=0.0, threshold2=100)
        mean_threshold = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, C_m)
        contours, _ = cv2.findContours(mean_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mean_threshold, contours, -1, (0, 0, 0), 3, 3)
        mean_threshold = cv2.bitwise_not(mean_threshold)
        cv2.drawContours(color, contours, -1, (0, 255, 0), 3)
        return color, mean_threshold

cap = cv2.VideoCapture(0)
cv2.namedWindow("original_video")
circle = CircleDetector()

while True:
    ret, img = cap.read()
    if not ret:
        break
    color_frame, contours_frame = circle.detect_circles(img)
    cv2.imshow("original_video", color_frame)
    cv2.imshow("contours only", contours_frame)
    if cv2.waitKey(1) == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
