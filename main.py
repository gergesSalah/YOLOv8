from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("D:/myWork/4th year-myWork/graduated project/Project development/model/code/YOLOv8/best (1).pt")
results = model.predict(source="abcess 1 (11).jpg",show=True, conf=0.5,stream=True,save=True)#imgsz=512    (29).jpg     61.jpg


for r in results:
    for c in r.boxes.cls:
        print(model.names[int(c)])

# img = model.predict(source="4.jpg",show=True, conf=0.5)
#
# igm3= cv2.imread("4.jpg",0)
# cv2.imshow("img3",img)
# cv2.waitKey(0)
