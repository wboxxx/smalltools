import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import shutil
import subprocess
from glob import glob

# === PARAMÈTRES ===
DOWNSCALE = 0.2
FRAME_SKIP = 15
MATCH_THRESHOLD = 0.85
MAX_GAP_BETWEEN_HITS = 8.0  # en secondes

# === CHEMINS ===
VB_PATH = "C:/Users/Vincent B/Videos/Brads/slowblues/"
# VIDEO_PATH = VB_PATH + "bcglsbpv2-003_hi.mp4"
VIDEO_PATH = VB_PATH + "bcglsbpv2-001_hi.mp4"
TEMPLATE_PATH = VB_PATH + "template_crop.png"
OUTPUT_JSON = VB_PATH + "T_appearance_info.json"
TMP_IMG_DIR = os.path.join(VB_PATH, "tmp_frames")

# === CHARGEMENT TEMPLATE ===
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
if DOWNSCALE != 1.0:
    template = cv2.resize(template, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
print(f"[DEBUG] template size: {template.shape[::-1]} (w x h)")

w, h = template.shape[::-1]

# === INFOS VIDÉO ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# === PRÉPARATION DOSSIER TEMPORAIRE ===
if os.path.exists(TMP_IMG_DIR):
    shutil.rmtree(TMP_IMG_DIR)
os.makedirs(TMP_IMG_DIR)

# === EXTRACTION VIA FFMPEG ===
print("🚀 Extraction des frames via ffmpeg...")
subprocess.run([
    "ffmpeg",
    "-i", VIDEO_PATH,
    "-vf", f"fps={fps / FRAME_SKIP}",
    os.path.join(TMP_IMG_DIR, "frame_%05d.jpg")
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# === ANALYSE DES FRAMES ===
hits = []
results = []
frame_paths = sorted(glob(os.path.join(TMP_IMG_DIR, "frame_*.jpg")))

for idx, img_path in enumerate(frame_paths):
    frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        continue
    if DOWNSCALE != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
    res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, _ = cv2.minMaxLoc(res)
    timestamp = idx * FRAME_SKIP / fps
    results.append((timestamp, score))
    if score > MATCH_THRESHOLD:
        hits.append((timestamp, score))
# === IDENTIFICATION DES FRONTS MONTANTS ET DURÉES ===
ON_THRESHOLD = MATCH_THRESHOLD
MIN_DURATION = 1.0  # secondes
detections = []
start_ts = None

for t, score in results:
    if score >= ON_THRESHOLD:
        if start_ts is None:
            start_ts = t  # Début de séquence
    else:
        if start_ts is not None:
            duration = t - start_ts
            if duration >= MIN_DURATION:
                detections.append({
                    "timestamp_sec": round(start_ts, 3),
                    "duration_sec": round(duration, 3),
                    "score": 1.0  # ou moyenne de score sur la zone si souhaité
                })
                print(f"🎯 Front montant à {start_ts:.3f}s — durée={duration:.3f}s")
            start_ts = None

# Cas où la vidéo se termine pendant une phase "on"
if start_ts is not None:
    duration = results[-1][0] - start_ts
    if duration >= MIN_DURATION:
        detections.append({
            "timestamp_sec": round(start_ts, 3),
            "duration_sec": round(duration, 3),
            "score": 1.0
        })
        print(f"🎯 Front montant à {start_ts:.3f}s — durée={duration:.3f}s")
    # === POST-PROCESSING : fusion des séquences trop proches
MIN_GAP_TO_SEPARATE = 4.0  # secondes
merged = []

for d in detections:
    if not merged:
        merged.append(d)
    else:
        last = merged[-1]
        gap = d["timestamp_sec"] - (last["timestamp_sec"] + last["duration_sec"])
        if gap < MIN_GAP_TO_SEPARATE:
            # Fusion : étendre la durée du précédent
            last["duration_sec"] = round((d["timestamp_sec"] + d["duration_sec"]) - last["timestamp_sec"], 3)
        else:
            merged.append(d)

# Remplace l'ancien tableau
detections = merged
# === CONVERSION EN DICTIONNAIRE {index: (timestamp, duration)}
onsets_dict = {
    idx: (d["timestamp_sec"], d["duration_sec"])
    for idx, d in enumerate(detections)
}

# 🧠 Vérification console
print("\n🗂️ Résumé des fronts montants :")
for idx, (ts, dur) in onsets_dict.items():
    print(f"  {idx}: start={ts:.3f}s, durée={dur:.3f}s")

# # === REGROUPEMENT DES PLAGES DE HITS ===
# groups = []
# current_group = []
# for ts, score in hits:
    # if not current_group or ts - current_group[-1][0] <= MAX_GAP_BETWEEN_HITS:
        # current_group.append((ts, score))
    # else:
        # groups.append(current_group)
        # current_group = [(ts, score)]
# if current_group:
    # groups.append(current_group)

# # === EXTRACTION DES DÉTECTIONS ===
# detections = []
# for group in groups:
    # start_ts = group[0][0]
    # end_ts = group[-1][0]
    # duration = round(end_ts - start_ts, 3)
    # first_score = group[0][1]
    # detections.append({
        # "timestamp_sec": round(start_ts, 3),
        # "score": round(first_score, 3),
        # "duration_sec": duration
    # })
    # print(f"🎯 Motif à {start_ts:.3f}s — durée={duration:.3f}s — score={first_score:.3f}")

# === EXPORT JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(detections, f, indent=2)

print(f"✅ {len(detections)} séquences enregistrées dans {OUTPUT_JSON}")

# === PLOT
df = pd.DataFrame(results, columns=["timestamp_sec", "score"])
plt.figure(figsize=(12, 4))
plt.plot(df["timestamp_sec"], df["score"], label="Score matchTemplate", alpha=0.4)
for d in detections:
    plt.axvspan(d["timestamp_sec"], d["timestamp_sec"] + d["duration_sec"], color="green", alpha=0.2)
    plt.axvline(d["timestamp_sec"], color="green", linestyle="--")
plt.xlabel("Temps (s)")
plt.ylabel("Score")
plt.title("Détections du motif (zones vertes = apparition)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === NETTOYAGE
shutil.rmtree(TMP_IMG_DIR)
print("🧹 Dossier temporaire supprimé.")
