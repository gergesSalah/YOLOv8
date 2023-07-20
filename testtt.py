from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("D:/myWork/4th year-myWork/graduated project/Project development/model/code/YOLOv8/best (1).pt")

classes = model.get_classes()
print("classess",classes)

# detections = model.predict(source="4.jpg", show=True, conf=0.5)
#
# # Extract the numpy array from the DetectionPredictor object
# img = detections.image
#
# # Display the image using cv2.imshow()
# cv2.imshow("img3", img)
# cv2.waitKey(0)

# results = model("4.jpg")
# print(results[0])
# cv2.imshow("img",results)


# inputs = [img, img]  # list of np arrays
results = model('4.jpg')  # List of Results objects

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmenation masks outputs
    probs = result.probs  # Class probabilities for classification outputs
    print("the boxes is : ",boxes)
    print("the masks is : ",masks)
    print("the probs is : ",probs)