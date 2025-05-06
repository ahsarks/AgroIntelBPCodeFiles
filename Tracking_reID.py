# (unchanged imports and definitions)
import cv2
import numpy as np
np.float = float
import time
import os
import sys
import firebase_admin
from firebase_admin import credentials, firestore, storage
import gc
import torch
import torch.nn.functional as F
import psutil
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import mss
import pygetwindow as gw
from concurrent.futures import ThreadPoolExecutor

from ultralytics import YOLO
sys.path.append(os.path.join(os.getcwd(), "ByteTrack"))
from yolox.tracker.byte_tracker import BYTETracker

# === Load ReID model and build embedding database ===
reid_model = resnet18(weights=None)
reid_model.fc = torch.nn.Linear(512, 128)
reid_model.load_state_dict(torch.load("cattle_reid.pth", map_location="cpu"))
reid_model.fc = torch.nn.Identity()
reid_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reid_model.to(device)

reid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

known_embeddings = {}
recognized_cows = set()
reid_dataset_path = "cattle_reid_dataset"

print("Loading cattle embeddings...")

def process_image(cow_id, img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = reid_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = reid_model(tensor).cpu().numpy()
        print(f"Loaded {img_path}")
        return (cow_id, embedding[0])
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

futures = []
with ThreadPoolExecutor(max_workers=8) as executor:
    for cow_id in os.listdir(reid_dataset_path):
        cow_path = os.path.join(reid_dataset_path, cow_id)
        if not os.path.isdir(cow_path):
            continue
        print(f"Processing {cow_id}")
        for img_name in os.listdir(cow_path):
            img_path = os.path.join(cow_path, img_name)
            futures.append(executor.submit(process_image, cow_id, img_path))

for future in futures:
    result = future.result()
    if result:
        cow_id, embedding = result
        known_embeddings[cow_id] = known_embeddings.get(cow_id, []) + [embedding]

for key in known_embeddings:
    known_embeddings[key] = np.mean(known_embeddings[key], axis=0, keepdims=True)

print("Finished loading all embeddings.")

def identify_cow(crop_img):
    try:
        img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        tensor = reid_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = reid_model(tensor).cpu().numpy()
    except Exception as e:
        print(f"ReID embedding error: {e}")
        return None

    best_score = -1
    best_id = None
    for cow_id, avg_emb in known_embeddings.items():
        score = cosine_similarity(embedding, avg_emb)[0][0]
        if score > best_score:
            best_score = score
            best_id = cow_id
    return best_id if best_score >= 0.85 else None

# === Firebase setup ===
firebase_key_path = "C:/Users/lukasmolvaer/Agrointel/firebase-key.json"
cred = credentials.Certificate(firebase_key_path)
firebase_admin.initialize_app(cred, {'storageBucket': 'your-firebase-project-id.appspot.com'})
db = firestore.client()
bucket = storage.bucket()

# === YOLO and ByteTrack setup ===
model = YOLO("C:/Users/lukasmolvaer/Agrointel/best.pt")
model.fuse()
model.to("cuda" if torch.cuda.is_available() else "cpu")

class BYTETrackerArgs:
    track_thresh = 0.25
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 3.0
    min_box_area = 1.0
    mot20 = False

byte_tracker = BYTETracker(BYTETrackerArgs())

# === Zone setup ===
zones = {
    "eating": [np.array([[1,59],[1,281],[703,227],[1289,330],[1289,59]],dtype=np.int32)],
    "sleeping": [
        np.array([[211,600],[211,800],[1017,800],[1017,600]],dtype=np.int32),
        np.array([[1176,590],[1187,800],[1288,800],[1288,590]],dtype=np.int32)]
}
zone_colors = {"eating":(0,0,255),"sleeping":(255,0,0)}

def detect_zone(point, zones_dict):
    for name, polys in zones_dict.items():
        for poly in polys:
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return name
    return None

for name, plist in zones.items():
    for i, poly in enumerate(plist):
        plist[i] = (poly // 2).astype(np.int32)

# === Locate Reolink window ===
candidate_windows = [w for w in gw.getAllWindows() if "Reolink" in w.title]
if not candidate_windows:
    print("Error: No window with 'Reolink' in its title found.")
    exit(1)
window = candidate_windows[0]
left, top = max(0, window.left), max(0, window.top)
width, height = window.width, window.height
monitor = {"top": top, "left": left, "width": width, "height": height}

# === Tracking Variables ===
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
frame_buffer = np.empty((buffer_h, buffer_w, 3), dtype=np.uint8)

# === ReID attempt tracking ===
track_id_to_cow_id = {}
track_attempt_counter = {}
already_recognized_tracks = set()
MAX_ATTEMPTS = 10

print("Starting capture + tracking. Ctrl+C to stop.")
last_frame = time.time()

try:
    with mss.mss() as sct:
        while True:
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

            res = model(frame)
            boxes = res[0].boxes.xyxy.cpu().numpy()
            confs = res[0].boxes.conf.cpu().numpy()
            cls_ids = res[0].boxes.cls.cpu().numpy().astype(int)
            mask = (cls_ids == 0)
            dets = np.hstack((boxes[mask], confs[mask].reshape(-1, 1))) if boxes.size else np.empty((0, 5))
            res[0].boxes = None
            del res

            tracks = byte_tracker.update(dets, img_info=frame.shape, img_size=frame.shape)
            used_ids = set()

            collect = False
            if now - last_coord >= coord_interval:
                collect = True
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                print(f"Collecting coords at {ts}")

            for tr in tracks:
                x1_f, y1_f, x2_f, y2_f = tr.tlbr
                x1_i, y1_i, x2_i, y2_i = map(int, (x1_f, y1_f, x2_f, y2_f))
                cx, cy = (x1_i + x2_i) // 2, (y1_i + y2_i) // 2

                cid = tr.track_id
                if cid not in already_recognized_tracks:
                    track_attempt_counter.setdefault(cid, 0)
                    if track_attempt_counter[cid] < MAX_ATTEMPTS:
                        cow_crop = frame[max(y1_i, 0):max(y2_i, 0), max(x1_i, 0):max(x2_i, 0)]
                        recognized_id = identify_cow(cow_crop)
                        if recognized_id:
                            cow_numeric_id = int(recognized_id.split('_')[-1])
                            track_id_to_cow_id[cid] = cow_numeric_id
                            recognized_cows.add(recognized_id)
                            already_recognized_tracks.add(cid)

                            os.makedirs("recognized_cows", exist_ok=True)
                            snapshot_filename = f"recognized_cows/{recognized_id}_{int(time.time())}.jpg"
                            cv2.imwrite(snapshot_filename, cow_crop)
                            print(f"Saved snapshot of {recognized_id} to {snapshot_filename}")

                        else:
                            track_attempt_counter[cid] += 1
                    else:
                        already_recognized_tracks.add(cid)

                if cid in track_id_to_cow_id:
                    tr.track_id = track_id_to_cow_id[cid]
                else:
                    if tr.track_id > 30:
                        for candidate in range(1, 31):
                            if candidate not in used_ids:
                                tr.track_id = candidate
                                break
                cid = tr.track_id
                used_ids.add(cid)

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

                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
                cv2.putText(frame, f"ID:{cid}", (x1_i, y1_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                if zone:
                    cv2.putText(frame, f"{zone}:{time_in_zone[cid][zone]:.1f}s", (x1_i, y2_i + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            if collect:
                last_coord = now

            for name, plist in zones.items():
                for poly in plist:
                    cv2.polylines(frame, [poly], True, zone_colors[name], 2)
                    cv2.putText(frame, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_colors[name], 2)

            if now - last_upload >= batch_interval or batch_count >= BATCH_SIZE:
                for cid, tz in time_in_zone.items():
                    batch.set(db.collection("cows").document(str(cid)), {"time_in_zone": tz}, merge=True)
                for cid, coords in coordinates_per_cow.items():
                    for c in coords:
                        batch.set(db.collection("cows").document(str(cid)).collection("coordinates").document(c["timestamp"]), c)
                        batch_count += 1
                if batch_count > 0:
                    batch.commit()
                    batch = db.batch()
                    batch_count = 0
                coordinates_per_cow = {cid: [] for cid in coordinates_per_cow}
                last_upload = now
                gc.collect()

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
                batch.set(db.collection("cows").document(str(cid)).collection("coordinates").document(c["timestamp"]), c)
        if batch_count > 0:
            batch.commit()
    torch.cuda.empty_cache()

    print("Program terminated.")
    if recognized_cows:
        print("Recognized cattle during session:")
        for cow in recognized_cows:
            print(f"- {cow}")
    else:
        print("No known cattle were recognized.")
