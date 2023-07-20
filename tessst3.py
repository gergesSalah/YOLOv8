from ultralytics import YOLO

model = YOLO("D:/myWork/4th year-myWork/graduated project/Project development/model/code/YOLOv8/best (1).pt")

# Get the detection results for the image
detector = model.predict(source="4.jpg", show=True, conf=0.5)

# Draw bounding boxes on image
image_with_boxes = detector.draw_bboxes(image, predictions)

# Display image
cv2.imshow('Image', image_with_boxes)
cv2.waitKey(0)