from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SUBJECTS = RAW / "data"
LABELS = RAW / "label_description.json"
OUTPUTS = ROOT / "outputs"
