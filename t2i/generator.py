"""Generator-Worker: ein Hintergrund-Thread arbeitet eine Job-Warteschlange ab.

Es ist immer nur ein Modell geladen (die iGPU teilt sich 16 GB mit dem System; FLUX int4 braucht allein ~12 GB).
Beim Modellwechsel wird das alte Pipeline-Objekt zuerst freigegeben.
"""
import gc
import queue
import random
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field

import openvino_genai as ov_genai
from PIL import Image

from . import db, scorer
from .config import DEVICE, IMAGES_DIR, OV_CACHE_DIR, discover_models


@dataclass
class Job:
    prompt: str
    model: str
    count: int
    steps: int
    guidance: float
    width: int
    height: int
    negative: str = ""
    seed: int | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"          # queued | loading | running | done | error | cancelled
    done: int = 0                   # fertige Bilder
    step: int = 0                   # Schritt im aktuellen Bild
    image_ids: list[int] = field(default_factory=list)
    error: str = ""
    cancel: bool = False

    def public(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "status", "prompt", "model", "count", "steps", "done", "step", "image_ids", "error")}


class Generator:
    def __init__(self):
        self.con = db.connect()
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[Job] = queue.Queue()
        self.pipe = None
        self.loaded_model: str | None = None
        threading.Thread(target=self._worker, daemon=True, name="generator").start()

    def submit(self, job: Job) -> Job:
        models = discover_models()
        if job.model not in models:
            raise ValueError(f"Unbekanntes Modell: {job.model}")
        if not models[job.model]["defaults"]["negative"]:
            job.negative = ""  # FLUX ignoriert Negativ-Prompts; nicht als Daten speichern, die nichts bewirkt haben
        job.width -= job.width % 64
        job.height -= job.height % 64
        self.jobs[job.id] = job
        self.queue.put(job)
        return job

    def cancel(self, job_id: str) -> None:
        if job_id in self.jobs:
            self.jobs[job_id].cancel = True

    def _load(self, name: str) -> None:
        if self.loaded_model == name:
            return
        self.pipe = None
        gc.collect()
        path = discover_models()[name]["path"]
        cfg = {"CACHE_DIR": str(OV_CACHE_DIR)} if DEVICE != "NPU" else {}
        self.pipe = ov_genai.Text2ImagePipeline(path, DEVICE, **cfg)
        self.loaded_model = name

    def _worker(self) -> None:
        while True:
            job = self.queue.get()
            if job.cancel:
                job.status = "cancelled"
                continue
            try:
                job.status = "loading"
                self._load(job.model)
                job.status = "running"
                self._run(job)
                job.status = "cancelled" if job.cancel else "done"
            except Exception as e:
                traceback.print_exc()
                job.status, job.error = "error", f"{type(e).__name__}: {e}"

    def _run(self, job: Job) -> None:
        base_seed = job.seed if job.seed is not None else random.randint(0, 2**31 - 1)

        def on_step(step, num_steps, latent):
            job.step = step + 1
            return job.cancel  # True bricht die Generierung ab

        for i in range(job.count):
            if job.cancel:
                return
            seed = base_seed + i
            job.step = 0
            kw = dict(width=job.width, height=job.height, num_inference_steps=job.steps,
                      guidance_scale=job.guidance, rng_seed=seed, callback=on_step)
            if job.negative:
                kw["negative_prompt"] = job.negative
            t0 = time.time()
            tensor = self.pipe.generate(job.prompt, **kw)
            if job.cancel:
                return
            dt = time.time() - t0
            fname = f"{time.strftime('%Y%m%d-%H%M%S')}_{job.id}_{i}.png"
            Image.fromarray(tensor.data[0]).save(IMAGES_DIR / fname)
            image_id = db.insert_image(
                self.con, batch_id=job.id, prompt=job.prompt, negative=job.negative, model=job.model,
                steps=job.steps, guidance=job.guidance, width=job.width, height=job.height, seed=seed,
                file=fname, duration_s=round(dt, 1))
            job.image_ids.append(image_id)
            job.done = i + 1
            try:
                scorer.score_new(self.con, image_id)
            except Exception:       # das Vorsortieren darf eine fertige Generierung nie scheitern lassen
                traceback.print_exc()
