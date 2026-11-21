"""HTTP-API + Weboberfläche. Start: python -m t2i.server  (dann http://127.0.0.1:7860)"""
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db, insights, scorer
from .config import IMAGES_DIR, ROOT, discover_models
from .generator import Generator, Job

app = FastAPI(title="t2i")
gen = Generator()
con = db.connect()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str
    count: int = Field(4, ge=1, le=16)
    steps: int | None = Field(None, ge=1, le=100)
    guidance: float | None = Field(None, ge=0, le=20)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    negative: str = ""
    seed: int | None = None


class RateRequest(BaseModel):
    rating: int | None = Field(None, ge=-1, le=1)
    favorite: bool | None = None
    note: str | None = None


@app.get("/api/models")
def models():
    return discover_models()


@app.post("/api/generate")
def generate(req: GenerateRequest):
    models = discover_models()
    if req.model not in models:
        raise HTTPException(400, f"Unbekanntes Modell: {req.model}")
    d = models[req.model]["defaults"]
    job = Job(prompt=req.prompt.strip(), model=req.model, count=req.count,
              steps=req.steps or d["steps"],
              guidance=d["guidance"] if req.guidance is None else req.guidance,
              width=req.width, height=req.height, negative=req.negative.strip(), seed=req.seed)
    return gen.submit(job).public()


@app.get("/api/jobs")
def jobs():
    return [j.public() for j in gen.jobs.values() if j.status in ("queued", "loading", "running")]


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in gen.jobs:
        raise HTTPException(404)
    return gen.jobs[job_id].public()


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    gen.cancel(job_id)
    return {"ok": True}


@app.get("/api/images")
def images(filter: str = "all", limit: int = 60, offset: int = 0):
    return db.list_images(con, filter, min(limit, 200), offset)


@app.get("/api/images/{image_id}")
def image(image_id: int):
    r = db.get(con, image_id)
    if not r:
        raise HTTPException(404)
    return r


@app.post("/api/images/{image_id}/rate")
def rate(image_id: int, req: RateRequest):
    r = db.rate(con, image_id, req.rating, req.favorite, req.note)
    if not r:
        raise HTTPException(404)
    return r


@app.get("/api/stats")
def stats():
    return db.stats(con)


@app.get("/api/insights")
def get_insights():
    return insights.insights(con)


@app.get("/api/suggest")
def get_suggest(prompt: str):
    return insights.suggest(con, prompt)


@app.get("/api/scorer")
def scorer_status():
    return scorer.status(con)


@app.post("/api/scorer/train")
def scorer_train():
    try:
        return scorer.train(con)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/img/{name}")
def img(name: str):
    path = (IMAGES_DIR / name).resolve()
    if path.parent != IMAGES_DIR.resolve() or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("T2I_PORT", 7860)))
