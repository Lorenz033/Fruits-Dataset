from ultralytics import YOLO
import cv2

# 1️⃣ Load your trained YOLO model
model = YOLO(r"C:\Users\John Lorenz\Downloads\best (2).pt")  # replace with your model path

# 2️⃣ Open external webcam (replace 1 with your webcam index)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open external webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 3️⃣ Run YOLO inference on the frame
    results = model(frame,verbose=False)

    # 4️⃣ Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # 5️⃣ Display the live feed with YOLO detections
    cv2.imshow("YOLO External Webcam Feed", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
