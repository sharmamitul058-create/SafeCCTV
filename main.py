from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)
for result in results:
    for box in result.boxes:

        class_id = int(box.cls[0])

        if class_id == 0:
            print("Person detected!")

            print(class_id, confidence)

    annotated = results[0].plot()

    cv2.imshow("YOLO", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(model.names)