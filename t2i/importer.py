"""Import bereits erzeugter Bilder samt Prompts in die Datenbank.

    python -m t2i.importer [Quellordner]      (Standard: ~/tools/imagegen)

Zuordnung Bild -> Prompt über den Dateinamen (prompts-*.txt, Zeilen "name|prompt"). Nur eindeutige Namen werden
übernommen: kommt derselbe Name mit verschiedenen Prompts vor (Versionen v1..v10), wäre die Zuordnung geraten und
würde die Lerndaten verfälschen. Bilder ohne Prompt, Upscales (x2/x4), Masken und Prüf-Crops werden übersprungen.
Gleiche Bilder (SHA-1) werden nur einmal importiert; liegt eine Kopie in picks/final, wird sie als Favorit (★)
markiert, aber nicht als 👍 gewertet: ohne 👎-Gegenstück würde der Stil dieses einen Projekts die Prompt-Statistik prägen.
Modell, Seed und Schritte sind bei Altbildern unbekannt (model="imported", seed=-1, steps=0).
"""
import hashlib
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

from . import db
from .config import IMAGES_DIR

SKIP = re.compile(r"x2|x4|mask|check-|tmp")
FAVORITE_DIRS = {"picks", "picks2", "final"}


def load_prompts(root: Path) -> dict[str, str]:
    seen: dict[str, set[str]] = {}
    for f in [*root.glob("prompts-*.txt"), *root.glob("out/**/prompts-*.txt")]:
        for line in f.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "|" in line:
                name, _, prompt = line.partition("|")
                seen.setdefault(name.strip(), set()).add(prompt.strip())
    return {n: next(iter(p)) for n, p in seen.items() if len(p) == 1}


def main(root: Path) -> None:
    prompts = load_prompts(root)
    con = db.connect()
    groups: dict[str, list[Path]] = {}   # sha1 -> Dateien
    for p in sorted((root / "out").rglob("*.png")):
        if SKIP.search(p.relative_to(root).as_posix()) or p.stem not in prompts:
            continue
        groups.setdefault(hashlib.sha1(p.read_bytes()).hexdigest(), []).append(p)

    added = skipped = 0
    for sha, files in groups.items():
        name = f"import_{sha[:12]}.png"
        if (IMAGES_DIR / name).exists():
            skipped += 1
            continue
        src = files[0]
        shutil.copy2(src, IMAGES_DIR / name)
        w, h = Image.open(src).size
        fav = any(part in FAVORITE_DIRS for f in files for part in f.parts)
        db.insert_image(
            con, batch_id=f"import-{sha[:8]}", prompt=prompts[src.stem], model="imported", steps=0, guidance=0,
            width=w, height=h, seed=-1, file=name, created_at=src.stat().st_mtime,
            rating=0, favorite=int(fav),  # nur ★: ein Stil-Projekt ohne 👎 würde die Statistik einseitig prägen
            note=f"import: {src.relative_to(root).as_posix()}")
        added += 1
    print(f"{added} importiert, {skipped} schon vorhanden, {len(groups)} eindeutige Bilder mit Prompt.")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "tools" / "imagegen")
