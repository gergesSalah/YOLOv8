from ultralytics import YOLO


model = YOLO("D:/myWork/4th year-myWork/graduated project/Project development/model/code/YOLOv8/best (1).pt")

class_names = model.names
print(class_names)

detections = model.predict(source="4.jpg", show=True, conf=0.5)

# Loop through the detections and print the predicted classes for each object
for detection in detections:
    print(class_names[detection.class_id])