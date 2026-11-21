"""Lernschicht 1: welche Prompt-Formulierungen und Einstellungen führen zu guten Bewertungen?

Kein Training, nur Statistik über die Bewertungen in der Datenbank:
  * Prompt wird an Kommas/Punkten/Semikolons in Phrasen zerlegt.
  * Je Phrase: Score = (gut - schlecht) / (n + K)  -> K dämpft kleine Stichproben (3 von 3 gut ist weniger
    überzeugend als 30 von 30).
  * Eine Phrase zählt nur, wenn sie in >= MIN_PROMPTS verschiedenen Prompts und bei >= MIN_N bewerteten Bildern
    vorkommt, sonst messen wir das Motiv statt den Stil.
"""
import re

K = 2.0
MIN_N = 3
MIN_PROMPTS = 2
THRESHOLD = 0.2

_SPLIT = re.compile(r"[,.;\n]+")


def phrases(prompt: str) -> set[str]:
    out = set()
    for p in _SPLIT.split(prompt.lower()):
        p = " ".join(p.split())
        if 3 <= len(p) <= 80:
            out.add(p)
    return out


def phrase_stats(con) -> list[dict]:
    rows = con.execute("SELECT prompt, rating FROM images WHERE rating != 0").fetchall()
    acc: dict[str, dict] = {}
    for r in rows:
        for ph in phrases(r["prompt"]):
            a = acc.setdefault(ph, {"phrase": ph, "n": 0, "good": 0, "bad": 0, "prompts": set()})
            a["n"] += 1
            a["good"] += r["rating"] == 1
            a["bad"] += r["rating"] == -1
            a["prompts"].add(r["prompt"])
    result = []
    for a in acc.values():
        if a["n"] < MIN_N or len(a["prompts"]) < MIN_PROMPTS:
            continue
        result.append({"phrase": a["phrase"], "n": a["n"], "good": a["good"], "bad": a["bad"],
                       "score": round((a["good"] - a["bad"]) / (a["n"] + K), 3)})
    return sorted(result, key=lambda x: -x["score"])


def model_stats(con) -> list[dict]:
    rows = con.execute(
        "SELECT model, COUNT(*) n, SUM(rating=1) good, SUM(rating=-1) bad, AVG(NULLIF(duration_s,0)) avg_s "
        "FROM images WHERE rating != 0 AND model != 'imported' GROUP BY model").fetchall()
    return [{"model": r["model"], "rated": r["n"], "good": r["good"], "bad": r["bad"],
             "good_rate": round(r["good"] / r["n"], 2), "avg_seconds": round(r["avg_s"] or 0, 1)} for r in rows]


def insights(con) -> dict:
    stats = phrase_stats(con)
    rated = con.execute("SELECT COUNT(*) FROM images WHERE rating != 0").fetchone()[0]
    return {
        "rated": rated,
        "helpful": [s for s in stats if s["score"] >= THRESHOLD][:10],
        "harmful": [s for s in reversed(stats) if s["score"] <= -THRESHOLD][:10],
        "models": model_stats(con),
        "ready": rated >= 20,   # darunter sind die Zahlen eher Rauschen
    }


def suggest(con, prompt: str) -> dict:
    """Phrasen, die sich bewährt haben und im Prompt fehlen (add), und Phrasen im Prompt, die schlecht abschneiden (avoid)."""
    stats = phrase_stats(con)
    have = phrases(prompt)
    return {
        "add": [s for s in stats if s["score"] >= THRESHOLD and s["phrase"] not in have][:5],
        "avoid": [s for s in stats if s["score"] <= -THRESHOLD and s["phrase"] in have][:5],
    }
