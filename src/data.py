"""Download the original Kaggle archive and keep only its text CSV."""

from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.kaggle.com/api/v1/datasets/download/snehaanbhawal/resume-dataset"
SOURCE = "https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    archive = folder / "source.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        candidates = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one CSV, found {candidates}")
        with z.open(candidates[0]) as f:
            raw = pd.read_csv(f)
    df = raw.rename(columns={"Resume_str": "Resume"})[["Resume", "Category"]]
    if df.isna().any().any():
        raise ValueError("Source contains missing values; review before training")
    target = folder / "resumes.csv"
    df.to_csv(target, index=False)
    provenance = {
        "source": SOURCE,
        "download_url": URL,
        "synthetic": False,
        "archive_sha256": sha256(archive),
        "csv_sha256": sha256(target),
        "archive_member": candidates[0],
        "raw_rows": len(df),
        "raw_categories": df.Category.nunique(),
        "transformation": "Rename Resume_str to Resume; retain Resume and Category only",
    }
    (folder / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
