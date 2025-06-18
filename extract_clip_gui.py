import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox


def parse_timestamp(ts_str: str) -> str:
    """Validate and normalize H:M:S timestamp to HH:MM:SS."""
    parts = ts_str.strip().split(":")
    if len(parts) != 3:
        raise ValueError("Le timestamp doit être au format H:M:S")
    try:
        h, m, s = [int(p) for p in parts]
    except ValueError as e:
        raise ValueError("Le timestamp doit contenir uniquement des nombres") from e
    if m < 0 or m >= 60 or s < 0 or s >= 60 or h < 0:
        raise ValueError("Valeurs de temps invalides")
    return f"{h:02}:{m:02}:{s:02}"


def extract_clip(video_path: str, start_ts: str, duration_s: int) -> str:
    """Extract clip using ffmpeg and return output file path."""
    base, ext = os.path.splitext(video_path)
    output_path = f"{base}_clip{ext}"
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        start_ts,
        "-i",
        video_path,
        "-t",
        str(duration_s),
        "-c",
        "copy",
        output_path,
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg n'est pas installé")
    except subprocess.CalledProcessError:
        raise RuntimeError("Échec de l'extraction")
    return output_path


def browse_file():
    path = filedialog.askopenfilename(
        title="Choisir un fichier vidéo",
        filetypes=[("Fichiers vidéo", "*.mp4 *.mkv *.mov *.m4v")],
    )
    if path:
        video_path.set(path)


def launch_extraction():
    path = video_path.get()
    if not path:
        messagebox.showerror("Erreur", "Aucun fichier sélectionné")
        return
    ts_input = timestamp_entry.get()
    try:
        ts_norm = parse_timestamp(ts_input)
    except ValueError as e:
        messagebox.showerror("Erreur", str(e))
        return
    duration = int(duration_var.get())
    try:
        out = extract_clip(path, ts_norm, duration)
        messagebox.showinfo("Succès", f"Clip extrait :\n{out}")
    except RuntimeError as e:
        messagebox.showerror("Erreur", str(e))


# UI setup
root = tk.Tk()
root.title("Extraction de clip vidéo")
root.geometry("400x200")
root.resizable(False, False)

video_path = tk.StringVar()

browse_btn = tk.Button(root, text="Choisir une vidéo", command=browse_file)
browse_btn.pack(pady=10)

entry_frame = tk.Frame(root)
entry_frame.pack(pady=5)

timestamp_label = tk.Label(entry_frame, text="Timestamp (H:M:S) :")
timestamp_label.grid(row=0, column=0, padx=5)

timestamp_entry = tk.Entry(entry_frame)
timestamp_entry.grid(row=0, column=1, padx=5)

duration_label = tk.Label(entry_frame, text="Durée :")
duration_label.grid(row=1, column=0, padx=5, pady=5)

duration_var = tk.StringVar(value="10")
duration_options = [str(i) for i in range(10, 70, 10)]
duration_menu = tk.OptionMenu(entry_frame, duration_var, *duration_options)
duration_menu.grid(row=1, column=1, padx=5, pady=5)

extract_btn = tk.Button(root, text="Extraire le clip", command=launch_extraction)
extract_btn.pack(pady=10)

root.mainloop()
