import cv2
from ultralytics import YOLO

# -----------------------------
# 1. Load YOLOv8 model
# -----------------------------
model = YOLO("golfballyolov8n.pt")  # trained to detect golf ball

# -----------------------------
# 2. Video paths
# -----------------------------
input_video_path = r"cq-8-ir.mov"
output_video_path = "cq-8-ir.mp4"

# -----------------------------
# 3. Setup video reader and writer
# -----------------------------
cap = cv2.VideoCapture(input_video_path)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# -----------------------------
# 4. Inference loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Access detections directly
    detections = results[0].boxes  # Bounding boxes + info

    # Example: Count number of golf balls detected
    golfball_count = len(detections)

    # Add custom label text on the frame
    cv2.putText(
        annotated_frame,
        f"Golfballs detected: {golfball_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Write annotated frame to output file
    out.write(annotated_frame)

    # Display in a window
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 5. Cleanup
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"[INFO] Inference complete. Saved output to: {output_video_path}")

