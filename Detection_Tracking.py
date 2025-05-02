import cv2
import numpy as np
np.float = float
import mss
import pygetwindow as gw
import time
import os
import sys
import firebase_admin
from firebase_admin import credentials, firestore, storage
import gc
import torch
import psutil

# YOLO + ByteTrack
from ultralytics import YOLO
sys.path.append(os.path.join(os.getcwd(), "ByteTrack"))
from yolox.tracker.byte_tracker import BYTETracker

#############################################
# 1. Locate the Reolink window
#############################################
candidate_windows = [w for w in gw.getAllWindows() if "Reolink" in w.title]
if not candidate_windows:
    print("Error: No window with 'Reolink' in its title found.")
    exit(1)
window = candidate_windows[0]

left = max(0, window.left)
top = max(0, window.top)
width = window.width
height = window.height

monitor = {"top": top, "left": left, "width": width, "height": height}
print(f"Capturing window '{window.title}' at {monitor}")

#############################################
# 2. Firebase Initialization
#############################################
firebase_key_path = "C:/Users/lukasmolvaer/Agrointel/firebase-key.json"
try:
    cred = credentials.Certificate(firebase_key_path)
    firebase_admin.initialize_app(cred, {'storageBucket': 'your-firebase-project-id.appspot.com'})
    db = firestore.client()
    bucket = storage.bucket()
    print("Firebase initialized with Firestore and Storage.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    exit(1)

#############################################
# 3. YOLO Model Setup
#############################################
MODEL_PATH = "C:/Users/lukasmolvaer/Agrointel/best.pt"
try:
    torch.cuda.empty_cache()
    model = YOLO(MODEL_PATH)
    model.fuse()
    if torch.cuda.is_available():
        model.to("cuda:0")
        print("YOLO model loaded on GPU.")
    else:
        model.to("cpu")
        print("CUDA not available, loaded YOLO on CPU.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

#############################################
# 4. ByteTrack Setup
#############################################
class BYTETrackerArgs:
    track_thresh = 0.25
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 3.0
    min_box_area = 1.0
    mot20 = False
byte_tracker = BYTETracker(BYTETrackerArgs())
print("ByteTrack tracker created.")

#############################################
# 5. Zone Definitions
#############################################
zones = {
    "eating": [np.array([[1,59],[1,281],[703,227],[1289,330],[1289,59]],dtype=np.int32)],
    "sleeping": [
        np.array([[211,600],[211,800],[1017,800],[1017,600]],dtype=np.int32),
        np.array([[1176,590],[1187,800],[1288,800],[1288,590]],dtype=np.int32)
    ]
}
zone_colors = {"eating":(0,0,255),"sleeping":(255,0,0)}

def detect_zone(point, zones_dict):
    for name, polys in zones_dict.items():
        for poly in polys:
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return name
    return None

# Adjust zones for half-res capture
for name, plist in zones.items():
    for i, poly in enumerate(plist):
        plist[i] = (poly // 2).astype(np.int32)

#############################################
# 6. Tracking Variables
#############################################
time_in_zone = {}
coordinates_per_cow = {}
coord_interval, snapshot_interval, batch_interval = 2, 1800, 30
last_coord = time.time()
last_upload = time.time()
batch = db.batch()
batch_count = 0
BATCH_SIZE = 500
frame_count = 0
buffer_w, buffer_h = width // 2, height // 2
last_saved_time = {}

print("Starting capture + tracking. Ctrl+C to stop.")
last_frame = time.time()

frame_buffer = np.empty((buffer_h, buffer_w, 3), dtype=np.uint8)

# Setup directory for saving cow crops
crop_save_dir = 'cattle_reid_dataset'
os.makedirs(crop_save_dir, exist_ok=True)

try:
    with mss.mss() as sct:
        while True:
            # Capture and downsample
            sct_img = sct.grab(monitor)
            view = np.ndarray((height, width, 4), dtype=np.uint8, buffer=sct_img.raw)
            small = view[::2, ::2, :3]
            np.copyto(frame_buffer, small)
            frame = frame_buffer
            del view, small, sct_img

            now = time.time()
            dt = now - last_frame
            last_frame = now
            frame_count += 1

            # YOLO detection
            res = model(frame)
            boxes = res[0].boxes.xyxy.cpu().numpy()
            confs = res[0].boxes.conf.cpu().numpy()
            cls_ids = res[0].boxes.cls.cpu().numpy().astype(int)
            mask = (cls_ids == 0)
            dets = np.hstack((boxes[mask], confs[mask].reshape(-1, 1))) if boxes.size else np.empty((0, 5))
            res[0].boxes = None
            del res

            # ByteTrack update
            tracks = byte_tracker.update(dets, img_info=frame.shape, img_size=frame.shape)
            used_ids = set()

            for tr in tracks:
                if tr.track_id > 30:
                    for candidate in range(1, 31):
                        if candidate not in used_ids:
                            tr.track_id = candidate
                            break
                used_ids.add(tr.track_id)

            collect = False
            if now - last_coord >= coord_interval:
                collect = True
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                print(f"Collecting coords at {ts}")

            for tr in tracks:
                x1_f, y1_f, x2_f, y2_f = tr.tlbr
                x1_i, y1_i, x2_i, y2_i = map(int, (x1_f, y1_f, x2_f, y2_f))
                cx, cy = int((x1_i + x2_i) / 2), int((y1_i + y2_i) / 2)
                cid = tr.track_id
                zone = detect_zone((cx, cy), zones)

                time_in_zone.setdefault(cid, {z: 0 for z in zones})
                coordinates_per_cow.setdefault(cid, [])
                if zone:
                    time_in_zone[cid][zone] += dt
                if collect:
                    coordinates_per_cow[cid].append({
                        "timestamp": ts,
                        "zone": zone or "Null Zone",
                        "coordinates": {"x": float(cx), "y": float(cy)}
                    })

                # Save cow crops (once every 2 seconds per cow)
                now_ts = time.time()
                if cid not in last_saved_time or now_ts - last_saved_time[cid] > 2.0:
                    cow_crop = frame[max(y1_i, 0):max(y2_i, 0), max(x1_i, 0):max(x2_i, 0)]
                    if cow_crop.size > 0 and cow_crop.shape[0] > 30 and cow_crop.shape[1] > 30:
                        cow_dir = os.path.join(crop_save_dir, f"cow_{cid}")
                        os.makedirs(cow_dir, exist_ok=True)
                        crop_filename = os.path.join(cow_dir, f"{time.time():.6f}.jpg")
                        cv2.imwrite(crop_filename, cow_crop)
                        last_saved_time[cid] = now_ts

                # Draw bounding box and ID
                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{cid}", (x1_i, y1_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if zone:
                    cv2.putText(
                        frame,
                        f"{zone}:{time_in_zone[cid][zone]:.1f}s",
                        (x1_i, y2_i + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

            if collect:
                last_coord = now

            # Draw zones
            for name, plist in zones.items():
                for poly in plist:
                    cv2.polylines(frame, [poly], True, zone_colors[name], 2)
                    cv2.putText(frame, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_colors[name], 2)

            # Periodic Firebase upload
            if now - last_upload >= batch_interval or batch_count >= BATCH_SIZE:
                for cid, tz in time_in_zone.items():
                    batch.set(db.collection("cows").document(str(cid)), {"time_in_zone": tz}, merge=True)
                for cid, coords in coordinates_per_cow.items():
                    for c in coords:
                        batch.set(
                            db.collection("cows").document(str(cid))
                            .collection("coordinates").document(c["timestamp"]),
                            c
                        )
                        batch_count += 1
                if batch_count > 0:
                    batch.commit()
                    batch = db.batch()
                    batch_count = 0
                coordinates_per_cow = {cid: [] for cid in coordinates_per_cow}
                last_upload = now
                gc.collect()

            # Snapshot upload
            if now - last_upload >= snapshot_interval:
                fn = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(fn, frame)
                try:
                    bucket.blob(f"cows/snapshots/{fn}").upload_from_filename(fn)
                    os.remove(fn)
                except Exception:
                    pass
                gc.collect()

            # Logs & cleanup
            if frame_count % 30 == 0:
                mem = psutil.virtual_memory()
                gpu_mem = torch.cuda.memory_allocated() / 1024**2
                print(f"{frame_count} frames. GPU {gpu_mem:.2f}MB, RAM {mem.used/1024**3:.2f}/{mem.total/1024**3:.2f}GB")
                gc.collect()
                torch.cuda.empty_cache()

            del frame, boxes, confs, cls_ids, dets, tracks
            gc.collect()

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    if time_in_zone or any(coordinates_per_cow.values()):
        for cid, tz in time_in_zone.items():
            batch.set(db.collection("cows").document(str(cid)), {"time_in_zone": tz}, merge=True)
        for cid, coords in coordinates_per_cow.items():
            for c in coords:
                batch.set(
                    db.collection("cows").document(str(cid))
                    .collection("coordinates").document(c["timestamp"]),
                    c
                )
        if batch_count > 0:
            batch.commit()
    torch.cuda.empty_cache()
    print("Program terminated.")
