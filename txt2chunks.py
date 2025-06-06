import os
import re
from tkinter import Tk, filedialog
from pathlib import Path

def timestamp_to_seconds(ts):
    minutes, seconds = map(int, ts.strip("[]").split(":"))
    return minutes * 60 + seconds

def process_transcript(filepath, max_chunk_duration=120):
    file_path = Path(filepath)
    output_dir = file_path.with_suffix('')
    output_dir.mkdir(exist_ok=True)

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]

    timestamp_re = re.compile(r"^\[\d+:\d{2}\]$")
    chunks = []
    current_chunk = ""
    current_start = None
    current_end = None

    i = 0
    while i < len(lines):
        if timestamp_re.match(lines[i]):
            timestamp = lines[i]
            text = lines[i+1] if i + 1 < len(lines) else ''
            time_sec = timestamp_to_seconds(timestamp)

            if current_start is None:
                current_start = time_sec
                current_end = time_sec
            elif time_sec - current_start > max_chunk_duration:
                # Sauver le chunk précédent
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_start = time_sec

            current_chunk += f"{timestamp} {text} "
            current_end = time_sec
            i += 2
        else:
            i += 1

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Écriture fichiers
    for idx, chunk in enumerate(chunks, 1):
        chunk_path = output_dir / f"chunk_{idx:03}.txt"
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(chunk)

    print(f"[OK] {len(chunks)} chunks écrits dans {output_dir}")

def main():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Choisir un fichier transcript .txt",
        filetypes=[("Text files", "*.txt")]
    )
    if file_path:
        process_transcript(file_path)
    else:
        print("Aucun fichier sélectionné.")

if __name__ == "__main__":
    main()
