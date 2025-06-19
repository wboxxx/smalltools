
import cv2
import numpy as np
import json
import csv
import os

import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
def show_template_frame_with_meta(image_path, meta_path):
    image = Image.open(image_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    x0, y0 = meta["x0"], meta["y0"]
    x1 = x0 + meta["width"]
    y1 = y0 + meta["height"]
    timestamp = meta["timestamp_sec"]
    motion = meta["motion_expected"]

    display_img = image.copy()
    draw = ImageDraw.Draw(display_img)
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

    root = tk.Tk()
    root.title("Template Frame + Meta")

    img_tk = ImageTk.PhotoImage(display_img)
    canvas = tk.Canvas(root, width=display_img.width, height=display_img.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=img_tk)

    text = f"🕒 {timestamp}s — ROI: {meta['width']}×{meta['height']} px\\nMotion expected: {'yes' if motion else 'no'}"
    canvas.create_rectangle(8, 8, 8 + 400, 38, fill="black", outline="black")  # fond noir derrière le texte
    canvas.create_text(10, 10, anchor="nw", text=text, fill="white", font=("Arial", 12))
    root.mainloop()


# === CONFIGURATION ===
VIDEO_PATH = "C:/Users/Vincent B/Videos/Brads/slowblues/bcglsbpv2-001_hi_clip.mp4"
TEMPLATE_IMAGE_PATH = "C:/Users/Vincent B/Videos/Brads/slowblues/template_crop.png"
TEMPLATE_META_PATH = "C:/Users/Vincent B/Videos/Brads/slowblues/template_meta.json"
OUTPUT_CSV_PATH = "C:/Users/Vincent B/Videos/Brads/slowblues/t_hsv_detections.csv"

# === CHARGEMENT DU MOTIF ET DES MÉTADONNÉES ===
template = cv2.imread(TEMPLATE_IMAGE_PATH)
if template is None:
    raise FileNotFoundError(f"❌ Template image not found: {TEMPLATE_IMAGE_PATH}")

with open(TEMPLATE_META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

x0, y0 = meta["x0"], meta["y0"]
w, h = meta["width"], meta["height"]
motion_expected = meta.get("motion_expected", False)

# === EXTRACTION DU PROFIL DE COULEUR HSV ===
template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
hue_mean = int(np.mean(template_hsv[:, :, 0]))
sat_mean = int(np.mean(template_hsv[:, :, 1]))
val_mean = int(np.mean(template_hsv[:, :, 2]))

# Tolérance couleur (ajustable)
H_TOL, S_TOL, V_TOL = 20, 50, 50
lower_bound = np.array([max(0, hue_mean - H_TOL), max(0, sat_mean - S_TOL), max(0, val_mean - V_TOL)])
upper_bound = np.array([min(179, hue_mean + H_TOL), min(255, sat_mean + S_TOL), min(255, val_mean + V_TOL)])

# === OUVERTURE VIDÉO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Impossible d’ouvrir la vidéo : {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = int(fps)  # une frame par seconde
# Optionnel : affichage diagnostic visuel
show_template_frame_with_meta("C:/Users/Vincent B/Videos/Brads/slowblues/template_fullframe.png", "C:/Users/Vincent B/Videos/Brads/slowblues/template_meta.json")

print(f"🔍 Lancement détection HSV — mode {'mobile' if motion_expected else 'fixe'}")
print(f"🎨 Profil HSV moyen : H={hue_mean}, S={sat_mean}, V={val_mean}")
print(f"🔢 Frames à analyser : {total_frames // step}")

results = []

for frame_idx in range(0, total_frames, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if motion_expected:
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            area = ww * hh
            if area >= w * h * 0.5:  # seuil : 50% taille du template
                timestamp = round(frame_idx / fps, 2)
                results.append([timestamp, x, y, ww, hh, area])
    else:
        roi = hsv[y0:y0 + h, x0:x0 + w]
        mask = cv2.inRange(roi, lower_bound, upper_bound)
        match_ratio = np.sum(mask > 0) / (w * h)
        if match_ratio > 0.5:  # seuil de match 50%
            timestamp = round(frame_idx / fps, 2)
            results.append([timestamp, x0, y0, w, h, match_ratio])

cap.release()

# === SAUVEGARDE CSV ===
with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp_sec", "x", "y", "width", "height", "score"])
    writer.writerows(results)

print(f"✅ Détection terminée : {len(results)} occurrences trouvées.")
print(f"📄 Résultats : {OUTPUT_CSV_PATH}")
