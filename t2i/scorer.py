"""Lernschicht 2: Bewertungsmodell. Sagt vorher, ob dir ein Bild gefällt, bevor du es bewertet hast.

  Bild --CLIP ViT-B/32 (fest, nicht trainiert)--> 512-dim Vektor --logistische Regression (deine 👍/👎)--> Score 0..1

Nur die kleine Regression wird trainiert (Sekunden auf der CPU); CLIP bleibt unverändert. Damit das nicht auf
Rauschen lernt, gibt es Mindestdaten, eine Kreuzvalidierung und einen Permutationstest (Training auf durchgemischten
Bewertungen). Ist das Modell nicht signifikant besser als Zufall, bekommt es kein Vorsortieren.
"""
import os
import pickle
import threading

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # hf_xet hängt in diesem Netz (siehe Riddle_Games gotchas)

import numpy as np
from PIL import Image

from . import db
from .config import DATA_DIR, IMAGES_DIR

CLIP_NAME = "openai/clip-vit-base-patch32"     # MIT-Lizenz
MODEL_PATH = DATA_DIR / "scorer.pkl"
MIN_RATED = 20          # bewertete Bilder insgesamt
MIN_PER_CLASS = 5       # mindestens so viele 👍 und 👎
MIN_GAIN = 0.05         # CV-Genauigkeit muss die Raten-Basislinie um so viel schlagen ...
MAX_P = 0.02            # ... und im Permutationstest deutlich besser sein als Zufall (streng: ein falsches Vorsortieren
N_PERMUTATIONS = 200    #     wirkt wie Gewissheit; mit 200 Permutationen ist p >= 0.005 erreichbar)

_lock = threading.Lock()
_clip = None
_clf = None


def _load_clip():
    global _clip
    if _clip is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))   # iGPU-Generierung soll nicht verhungern
        _clip = (CLIPModel.from_pretrained(CLIP_NAME).eval(), CLIPProcessor.from_pretrained(CLIP_NAME))
    return _clip


def embed(files: list[str]) -> np.ndarray:
    """CLIP-Bildvektoren, L2-normiert, Form (n, 512)."""
    import torch
    model, proc = _load_clip()
    out = []
    for i in range(0, len(files), 16):
        imgs = [Image.open(IMAGES_DIR / f).convert("RGB") for f in files[i:i + 16]]
        with torch.no_grad():
            f = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
        if not torch.is_tensor(f):      # neuere transformers-Versionen liefern ein Output-Objekt
            f = f.pooler_output
        f = f / f.norm(dim=-1, keepdim=True)
        out.append(f.numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def ensure_embeddings(con, rows) -> None:
    have = {r[0] for r in con.execute("SELECT image_id FROM embeddings")}
    todo = [r for r in rows if r["id"] not in have]
    if not todo:
        return
    vecs = embed([r["file"] for r in todo])
    con.executemany("INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                    [(r["id"], v.tobytes()) for r, v in zip(todo, vecs)])
    con.commit()


def _vectors(con, ids: list[int]) -> np.ndarray:
    q = ",".join("?" for _ in ids)
    m = {r[0]: np.frombuffer(r[1], np.float32) for r in
         con.execute(f"SELECT image_id, vec FROM embeddings WHERE image_id IN ({q})", ids)}
    return np.stack([m[i] for i in ids])


def status(con) -> dict:
    rated = con.execute("SELECT SUM(rating=1) g, SUM(rating=-1) b FROM images").fetchone()
    good, bad = rated["g"] or 0, rated["b"] or 0
    info = {"good": good, "bad": bad, "min_rated": MIN_RATED, "min_per_class": MIN_PER_CLASS,
            "can_train": good + bad >= MIN_RATED and min(good, bad) >= MIN_PER_CLASS, "trained": MODEL_PATH.exists()}
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            meta = pickle.load(f)
        info.update({k: meta[k] for k in ("n_train", "cv_accuracy", "baseline", "p_value", "active")})
    return info


def train(con) -> dict:
    """Trainiert auf allen 👍/👎, bewertet danach alle unbewerteten Bilder. Gibt den Status zurück."""
    global _clf
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score

    with _lock:
        st = status(con)
        if not st["can_train"]:
            raise ValueError(f"Zu wenig Daten: {st['good']} 👍 / {st['bad']} 👎 "
                             f"(nötig: {MIN_RATED} gesamt, je Klasse mindestens {MIN_PER_CLASS})")
        rated = con.execute("SELECT id, file, rating FROM images WHERE rating != 0").fetchall()
        ensure_embeddings(con, rated)
        X = _vectors(con, [r["id"] for r in rated])
        y = np.array([r["rating"] == 1 for r in rated])

        folds = StratifiedKFold(n_splits=min(5, int(min(y.sum(), (~y).sum()))), shuffle=True, random_state=0)
        # Regularisierung per Kreuzvalidierung nach Log-Loss wählen (bewertet auch die Güte der Wahrscheinlichkeiten,
        # nicht nur die Rangfolge). CLIP-Vektoren sind normiert: mit C=1 würden die Scores sonst um 0.5 zusammengedrückt.
        best_ll, best_c = -np.inf, 1.0
        for c in (1.0, 10.0, 100.0):
            ll = float(cross_val_score(LogisticRegression(C=c, class_weight="balanced", max_iter=2000), X, y,
                                       cv=folds, scoring="neg_log_loss").mean())
            if ll > best_ll:
                best_ll, best_c = ll, c
        cv = float(cross_val_score(LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000), X, y,
                                   cv=folds, scoring="balanced_accuracy").mean())
        baseline = 0.5       # balanced_accuracy von "immer dasselbe raten"
        clf = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000).fit(X, y)
        # Permutationstest: mit durchgemischten Bewertungen neu trainieren. Nur wenn der echte Wert klar über dem
        # liegt, was Zufall erreicht, ist das Modell aktiv. Bei ~30 Bildern streut die CV-Genauigkeit auf Rauschen
        # allein um ±0.1 - eine feste Schwelle würde es hier fälschlich freischalten.
        _, _, p_value = permutation_test_score(
            LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000), X, y, cv=folds,
            scoring="balanced_accuracy", n_permutations=N_PERMUTATIONS, random_state=0)
        active = bool(cv >= baseline + MIN_GAIN and p_value < MAX_P)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"clf": clf, "n_train": len(y), "cv_accuracy": round(cv, 3), "baseline": baseline,
                         "p_value": round(float(p_value), 3), "active": active}, f)
        _clf = clf if active else None
        _rescore(con, _clf)
    return status(con)


def _rescore(con, clf) -> None:
    con.execute("UPDATE images SET score = NULL")
    if clf is not None:
        rows = con.execute("SELECT id, file FROM images WHERE rating = 0").fetchall()
        ensure_embeddings(con, rows)
        if rows:
            p = clf.predict_proba(_vectors(con, [r["id"] for r in rows]))[:, 1]
            con.executemany("UPDATE images SET score = ? WHERE id = ?",
                            [(round(float(s), 3), r["id"]) for s, r in zip(p, rows)])
    con.commit()


def score_new(con, image_id: int) -> None:
    """Nach einer Generierung aufrufen: bewertet das neue Bild, falls ein aktives Modell existiert."""
    global _clf
    if not MODEL_PATH.exists():
        return
    with _lock:
        if _clf is None:
            with open(MODEL_PATH, "rb") as f:
                meta = pickle.load(f)
            if not meta["active"]:
                return
            _clf = meta["clf"]
        row = con.execute("SELECT id, file FROM images WHERE id = ?", (image_id,)).fetchone()
        ensure_embeddings(con, [row])
        p = _clf.predict_proba(_vectors(con, [image_id]))[:, 1]
        con.execute("UPDATE images SET score = ? WHERE id = ?", (round(float(p[0]), 3), image_id))
        con.commit()
