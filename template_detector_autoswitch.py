import cv2
import numpy as np
import json
import csv

# === CONFIGURATION ===
# === CONFIGURATION ===
VB_PATH= "C:/Users/Vincent B/Videos/Brads/slowblues/"
VIDEO_PATH = "C:/Users/Vincent B/Videos/Brads/slowblues/bcglsbpv2-001_hi_clip.mp4"
TEMPLATE_IMAGE_PATH = "C:/Users/Vincent B/Videos/Brads/slowblues/template_crop.png"
TEMPLATE_META_PATH = "C:/Users/Vincent B/Videos/Brads/slowblues/template_meta.json"
OUTPUT_CSV_PATH = "C:/Users/Vincent B/Videos/Brads/slowblues/t_hsv_detections.csv"

FRAME_IMAGE_PATH = "C:/Users/Vincent B/Videos/Brads/slowblues/template_fullframe.png"

def decide_strategy(template, meta):
    hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    mean_s = np.mean(hsv[:, :, 1])
    std_s = np.std(hsv[:, :, 1])
    w, h = meta["width"], meta["height"]

    orb = cv2.ORB_create()
    kp, _ = orb.detectAndCompute(template, None)
    num_kp = len(kp)

    print(f"[DEBUG] mean_s={mean_s:.2f}, std_s={std_s:.2f}, surface={w*h}, orb_kp={num_kp}")

    if mean_s < 20 and std_s < 20 and w * h < 2000:
        return "template_match"
    elif num_kp >= 10:
        return "orb"
    else:
        return "hsv"



# === CHARGEMENT TEMPLATE + MÉTA ===
template = cv2.imread(TEMPLATE_IMAGE_PATH)
frame_img = cv2.imread(FRAME_IMAGE_PATH)
with open(TEMPLATE_META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

x0, y0, w, h = meta["x0"], meta["y0"], meta["width"], meta["height"]
motion_expected = meta.get("motion_expected", False)

# === ANALYSE DU TEMPLATE POUR CHOIX DE STRATÉGIE ===
template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
mean_s = np.mean(template_hsv[:, :, 1])
std_s = np.std(template_hsv[:, :, 1])

orb = cv2.ORB_create()
kp, _ = orb.detectAndCompute(template, None)
num_kp = len(kp)
strategy = decide_strategy(template, meta)

print(f"[DEBUG] mean_s={mean_s:.2f}, std_s={std_s:.2f}, surface={w*h}, orb_kp={num_kp}")

print(f"🧠 Méthode choisie : {strategy} — motion_expected={motion_expected}")

# === OUVERTURE VIDÉO ===
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Impossible d’ouvrir la vidéo : {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = int(fps)  # 1 FPS

results = []

# === DÉTECTION SELON STRATÉGIE ===
for frame_idx in range(0, total_frames, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    if strategy == "template_match":
        if motion_expected:
            res = cv2.matchTemplate(gray_frame, gray_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.9:
                results.append([round(frame_idx / fps, 2), *max_loc, w, h, round(max_val, 3)])
        else:
            roi = gray_frame[y0:y0 + h, x0:x0 + w]
            res = cv2.matchTemplate(roi, gray_template, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]
            if score > 0.9:
                results.append([round(frame_idx / fps, 2), x0, y0, w, h, round(score, 3)])

    elif strategy == "hsv":
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(template_hsv)[:3]
        lower = np.array([max(0, mean_hsv[0]-15), max(0, mean_hsv[1]-50), max(0, mean_hsv[2]-50)])
        upper = np.array([min(179, mean_hsv[0]+15), min(255, mean_hsv[1]+50), min(255, mean_hsv[2]+50)])
        if motion_expected:
            mask = cv2.inRange(hsv_frame, lower, upper)
            if np.sum(mask) > w*h*100:
                results.append([round(frame_idx / fps, 2), 0, 0, 0, 0, np.sum(mask)])
        else:
            roi = hsv_frame[y0:y0 + h, x0:x0 + w]
            mask = cv2.inRange(roi, lower, upper)
            match_ratio = np.sum(mask > 0) / (w * h)
            if match_ratio > 0.5:
                results.append([round(frame_idx / fps, 2), x0, y0, w, h, round(match_ratio, 2)])

cap.release()

# === EXPORT CSV ===
with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_sec", "x", "y", "width", "height", "score"])
    writer.writerows(results)

print(f"✅ Détection terminée. {len(results)} résultats enregistrés dans {OUTPUT_CSV_PATH}")
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(VB_PATH + "t_hsv_detections.csv")
plt.plot(df["timestamp_sec"], df["score"], marker="o")
plt.xlabel("Temps (s)")
plt.ylabel("Score")
plt.title("Évolution du score de détection")
plt.grid(True)
# === PASSE DE RAFFINEMENT AUTOUR DU MEILLEUR SCORE ===
if len(df) > 0:
    best_row = df.loc[df["score"].idxmax()]
    best_time = best_row["timestamp_sec"]
    print(f"🎯 Best match initial à {best_time:.2f}s → score={best_row['score']}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    best_frame = int(best_time * fps)
    window = int(fps * 0.1)  # ±100ms → environ 6 frames à 60fps
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    refined = []
    for i in range(-window, window + 1):
        frame_idx = best_frame + i
        if frame_idx < 0:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        refined.append((frame_idx / fps, max_val))
    cap.release()

    refined_df = pd.DataFrame(refined, columns=["timestamp_sec", "score"])
    best_refined = refined_df.loc[refined_df["score"].idxmax()]
    best_time_refined = best_refined["timestamp_sec"]
    print(f"🔍 Raffiné : {best_time_refined:.3f}s → score = {best_refined['score']:.3f}")

    # Ajout sur le graphique
    plt.plot(refined_df["timestamp_sec"], refined_df["score"], "r.-", label="Raffinage 100ms")
    plt.axvline(best_time_refined, color="green", linestyle="--", label=f"Best @ {best_time_refined:.2f}s")
    plt.legend()
    plt.title("Évolution du score de détection (avec raffinage)")
    plt.show()

else:
    print("❌ Aucun résultat initial — pas de raffinement possible.")

plt.show()
