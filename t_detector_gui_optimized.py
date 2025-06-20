
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import time
import os
import subprocess
import sys
from PIL import Image, ImageDraw

output_path = os.path.expanduser("~/Documents/refined_t_timestamps.txt")
_MASK_STORE = os.path.expanduser("~/.t_detector_last_mask.txt")


def open_results_file(path, log_callback):
    """Try to open the results file with the default system application."""
    try:
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        log_callback(f"📂 Ouverture du fichier : {path}")
    except Exception as e:
        log_callback(f"⚠️ Impossible d'ouvrir automatiquement le fichier : {e}")


def generate_t_template_from_video(video_path, output_path="t_template_generated.png", y_ratio=0.75):
    """Generate a black T template from the first frame of a video."""
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"Impossible de lire la vidéo : {video_path}")

    height, width = frame.shape[:2]
    thickness = int(width * 0.01)
    y_bar = int(height * y_ratio)
    x_center = width // 2

    img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [(0, y_bar - thickness // 2), (width, y_bar + thickness // 2)],
        fill=(0, 0, 0, 255),
    )

    draw.rectangle(
        [(x_center - thickness // 2, 0), (x_center + thickness // 2, y_bar)],
        fill=(0, 0, 0, 255),
    )

    img.save(output_path)
    print(f"✅ Template T généré : {output_path}")
    return output_path

def detect_coarse_and_refined(video_path, template_path, threshold, scale_factor, log_callback):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    template_orig = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    _, mask_orig = cv2.threshold(template_orig, 250, 255, cv2.THRESH_BINARY_INV)

    if scale_factor != 1.0:
        template = cv2.resize(template_orig, (0, 0), fx=scale_factor, fy=scale_factor)
        mask = cv2.resize(mask_orig, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    else:
        template = template_orig
        mask = mask_orig

    w, h = template.shape[::-1]
    jump_frames = int(1 * fps)  # 30s
    cooldown_frames = int(120 * fps)  # skip 2 min after match

    coarse_matches = []
    frame_idx = 0
    start_time = time.time()

    log_callback("⏱ Première passe : détection rapide avec sauts de 1s...\n")
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale_factor != 1.0:
            gray = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)

        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val >= threshold:
            ts = frame_idx / fps
            coarse_matches.append(ts)
            log_callback(f"✅ Match détecté à {ts:.2f}s (score={max_val:.3f}) → saut de 2min")
            frame_idx += cooldown_frames
        else:
            frame_idx += jump_frames

    cap.release()
    log_callback(f"\n⏱ Fin première passe ({time.time() - start_time:.1f}s). Matches bruts : {len(coarse_matches)}\n")

    cap = cv2.VideoCapture(video_path)
    refined_matches = []
    log_callback("🔍 Deuxième passe : recherche précise de chaque début d'apparition...\n")

    for coarse_ts in coarse_matches:
        start_frame = int((coarse_ts - 180) * fps)
        end_frame = int(coarse_ts * fps)
        start_frame = max(start_frame, 0)

        best_score = -1
        best_frame = -1
        step = int(fps)  # 1s
        log_callback(f"\n📍 Recherche entre {start_frame/fps:.2f}s et {end_frame/fps:.2f}s...")

        for f in range(start_frame, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if scale_factor != 1.0:
                gray = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)

            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED, mask=mask)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val >= threshold and (best_score < 0 or max_val > best_score):
                best_score = max_val
                best_frame = f

        if best_frame >= 0:
            refined_ts = best_frame / fps
            refined_matches.append(refined_ts)
            log_callback(f"🎯 Match précis à {refined_ts:.2f}s (score={best_score:.3f})")

    cap.release()

    with open(output_path, "w") as f:
        for t in refined_matches:
            f.write(f"{t:.3f}\n")

    log_callback(f"\n✅ Détection terminée. Résultats enregistrés dans refined_t_timestamps.txt")
    open_results_file(output_path, log_callback)


def run_gui():
    def browse_video():
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            video_path.set(path)

    def browse_template():
        path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])
        if path:
            template_path.set(path)
            try:
                with open(_MASK_STORE, "w") as f:
                    f.write(path)
            except OSError:
                pass

    def generate_template():
        path = video_path.get()
        if not path:
            log_callback("⚠️ Sélectionnez d'abord une vidéo")
            return
        try:
            out = generate_t_template_from_video(path)
            template_path.set(out)
            try:
                with open(_MASK_STORE, "w") as f:
                    f.write(out)
            except OSError:
                pass
            log_callback(f"🆕 Template généré : {out}")
        except Exception as e:
            log_callback(f"⚠️ Erreur génération template : {e}")

    def launch_detection():
        detect_button.config(state=tk.DISABLED)
        log_text.delete(1.0, tk.END)
        log_callback("🔄 Lancement de la détection...\n")
        threshold = threshold_slider.get()
        scale_val = scale_var.get()
        scale_factor = {"1x": 1.0, "1/2": 0.5, "1/4": 0.25, "1/8": 0.125}[scale_val]

        def run_detection():
            detect_coarse_and_refined(
                video_path.get(),
                template_path.get(),
                threshold,
                scale_factor,
                log_callback,
            )
            root.after(0, lambda: detect_button.config(state=tk.NORMAL))

        threading.Thread(target=run_detection, daemon=True).start()

    def log_callback(msg):
        log_text.insert(tk.END, msg + "\n")
        log_text.see(tk.END)

    root = tk.Tk()
    root.title("Optimized Inverted T Detection")

    video_path = tk.StringVar()
    default_template = "inverted_t_template_1percent.png"
    if os.path.exists(_MASK_STORE):
        try:
            with open(_MASK_STORE, "r") as f:
                stored = f.read().strip()
                if stored:
                    default_template = stored
        except OSError:
            pass

    template_path = tk.StringVar(value=default_template)

    ttk.Label(root, text="🎞 Select Video File:").pack(anchor="w", padx=10)
    ttk.Entry(root, textvariable=video_path, width=60).pack(padx=10)
    ttk.Button(root, text="Browse", command=browse_video).pack(pady=2)

    ttk.Label(root, text="🖼 Select T Template (white areas ignored):").pack(anchor="w", padx=10)
    ttk.Entry(root, textvariable=template_path, width=60).pack(padx=10)
    ttk.Button(root, text="Browse", command=browse_template).pack(pady=2)
    ttk.Button(root, text="Generate from Video", command=generate_template).pack(pady=2)

    ttk.Label(root, text="🎚 Detection Threshold:").pack(anchor="w", padx=10, pady=(10, 0))
    threshold_slider = ttk.Scale(root, from_=0.5, to=0.99, orient="horizontal")
    threshold_slider.set(0.75)
    threshold_slider.pack(padx=10, fill="x")

    ttk.Label(root, text="📏 Downscale (accélère l'analyse):").pack(anchor="w", padx=10)
    scale_var = tk.StringVar(value="1/2")
    ttk.Combobox(root, textvariable=scale_var, values=["1x", "1/2", "1/4", "1/8"]).pack(padx=10)

    detect_button = ttk.Button(root, text="▶️ Launch Detection", command=launch_detection)
    detect_button.pack(pady=10)

    log_text = tk.Text(root, height=20, width=80)
    log_text.pack(padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
