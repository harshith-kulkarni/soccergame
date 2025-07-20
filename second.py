import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import time

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOLOv11 model
try:
    model = YOLO("best.pt")
    model.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Print model class names to verify player class ID
print(f"Model class names: {model.names}")
# Dynamically select player class ID (modify if 'player' is not class 0)
player_class_id = 2  # Change to 1 or other if 'player' is not class 0 in model.names

# Initialize DeepSORT tracker with optimized parameters
tracker = DeepSort(max_age=70, n_init=1, embedder="mobilenet", max_iou_distance=0.6)

# Load video
cap = cv2.VideoCapture("15sec_input_720p.mp4")
if not cap.isOpened():
    print("Error: Could not open video. Check path: data/15sec_input_720p.mp4")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {width}x{height} @ {fps} FPS")

# Initialize video writer for output
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (960,540))

# Initialize tracking log
log = []

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame_start = time.time()

    # Optional: Resize frame for faster processing (uncomment if FPS is low)
    frame = cv2.resize(frame, (960,540))

    # Detect players with YOLOv11
    results = model(frame, conf=0.9, classes=[player_class_id], iou=0.5, device=device)
    detections = []
    for box in results[0].boxes:
        x, y, w, h = box.xywh[0].cpu().numpy()
        conf = box.conf.cpu().numpy().item()
        detections.append(([x - w/2, y - h/2, w, h], conf, player_class_id))
        # Debug: Log detection confidence
        print(f"Frame {frame_count}: Detection at ({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}) with conf={conf:.3f}")

    # Debug: Print number of detections
    print(f"Frame {frame_count}: Detected {len(detections)} players")

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Debug: Print number of tracks and their states
    confirmed_count = sum(1 for t in tracks if t.is_confirmed())
    print(f"Frame {frame_count}: Tracking {len(tracks)} players ({confirmed_count} confirmed)")

    # Annotate frame with bounding boxes and player ID tags
    for track in tracks:
        bbox = track.to_tlbr()
        player_id = track.track_id
        status = "Confirmed" if track.is_confirmed() else "Tentative"
        # Draw bounding box (green for confirmed, red for tentative)
        color = (0, 255, 0) if track.is_confirmed() else (0, 0, 255)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        # Add player ID tag with status
        cv2.putText(frame, f"Player_{player_id} ({status})", (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # Log tracking data
        log.append([frame_count, f"Player_{player_id}", status, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Write frame to output video
    out.write(frame)

    # Print FPS for debugging
    print(f"Frame {frame_count}: {1 / (time.time() - frame_start):.2f} FPS")

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()

# Save tracking log with status
np.savetxt("tracking_data.csv", log, delimiter=",", fmt="%s",
           header="frame,player_id,status,x_min,y_min,x_max,y_max", comments="")
print(f"Processing complete. Total time: {time.time() - start_time:.2f} seconds")
print("Output saved: output.mp4, tracking_data.csv")

# After the while loop, before releasing resources
total_detections = sum([int(row[0]) for row in log]) if log else 0
num_frames = frame_count
avg_detections_per_frame = len(log) / num_frames if num_frames else 0
avg_confidence = np.mean([float(row[4]) for row in log]) if log else 0

print(f"Total frames processed: {num_frames}")
print(f"Total detections: {len(log)}")
print(f"Average detections per frame: {avg_detections_per_frame:.2f}")
print(f"Average detection confidence: {avg_confidence:.3f}")