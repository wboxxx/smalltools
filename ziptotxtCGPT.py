import zipfile
import json
from pathlib import Path

# === CONFIGURATION ===
ZIP_FILE = Path(r"C:\Users\Vincent B\Downloads\vbchatgpt_250602.zip")  # <-- à adapter
OUTPUT_FILE = Path("chatgpt_full_history.txt")

def extract_conversations_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as archive:
        # Trouver conversations.json
        json_filename = next((f for f in archive.namelist() if f.endswith("conversations.json")), None)
        if not json_filename:
            raise FileNotFoundError("conversations.json introuvable dans l'archive.")

        with archive.open(json_filename) as f:
            data = json.load(f)
    return data

def format_conversation(title, messages):
    lines = [f"\n==== {title} ====\n"]
    for msg in messages:
        role = msg.get("author", {}).get("role", "unknown").upper()
        parts = msg.get("content", {}).get("parts", [""])
        text = parts[0] if isinstance(parts, list) and parts else ""
        lines.append(f"{role}:\n{text}\n")
    return "\n".join(lines)

def main():
    try:
        conversations = extract_conversations_from_zip(ZIP_FILE)
    except Exception as e:
        print(f"❌ Erreur de lecture du fichier ZIP : {e}")
        return

    all_text = []
    for convo in conversations:
        title = convo.get("title", "Untitled")
        mappings = convo.get("mapping", {})
        # Ordonner les messages par date
        raw_messages = []
        for m in mappings.values():
            msg = m.get("message")
            if msg and isinstance(msg, dict):
                raw_messages.append(msg)

        with_time = [m for m in raw_messages if isinstance(m.get("create_time"), (int, float))]
        without_time = [m for m in raw_messages if m not in with_time]

        messages = sorted(with_time, key=lambda m: m["create_time"]) + without_time
        all_text.append(format_conversation(title, messages))

    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(all_text))
        print(f"✅ Export terminé : {OUTPUT_FILE.resolve()}")
    except Exception as e:
        print(f"❌ Erreur lors de l’écriture du fichier : {e}")

if __name__ == "__main__":
    main()
