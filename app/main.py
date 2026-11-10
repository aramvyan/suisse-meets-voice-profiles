"""Web API for the voice-profile prototype.

Run:  .venv/bin/uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import recognizer
from .engine import (MIN_ENROLL_SECONDS, ROOT, SAMPLE_RATE, VoiceEngine,
                     cluster_audio, load_audio)
from .store import CONSENT_HASH, CONSENT_TEXT, CONSENT_VERSION, TenantVault

# Two fictional customers, to show that their data never mixes.
TENANTS = {
    "alpine-dental": "Alpine Dental AG",
    "zurich-legal": "Zürich Legal GmbH",
}

app = FastAPI(title="Suisse Meets — Voice Profiles prototype")
engine = VoiceEngine()
engine_lock = threading.Lock()  # ONNX sessions are shared; serialize calls
STATIC = ROOT / "app" / "static"
DEMO = ROOT / "data" / "demo"


def vault(tenant: str) -> TenantVault:
    # In production the tenant comes from the authenticated session, never
    # from user input. Here it is the URL, restricted to known tenants.
    if tenant not in TENANTS:
        raise HTTPException(404, "unknown tenant")
    return TenantVault(tenant)


async def decode_upload(upload: UploadFile):
    try:
        return load_audio(await upload.read())  # in memory only
    except Exception:
        raise HTTPException(400, "could not decode audio")


# --- pages -------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/enroll")
def enroll_page():
    return FileResponse(STATIC / "enroll.html")


@app.get("/api/tenants")
def tenants():
    return [{"id": k, "name": v} for k, v in TENANTS.items()]


@app.get("/api/demo-files")
def demo_files():
    if not (DEMO / "manifest.json").exists():
        return {}
    return json.loads((DEMO / "manifest.json").read_text())


# --- admin: profiles, invites, audit ----------------------------------
class InviteIn(BaseModel):
    name: str
    email: str


@app.get("/api/t/{tenant}/profiles")
def profiles(tenant: str):
    return vault(tenant).list_profiles()


@app.post("/api/t/{tenant}/invites")
def invite(tenant: str, body: InviteIn):
    if not body.name.strip() or "@" not in body.email:
        raise HTTPException(400, "name and valid email required")
    token = vault(tenant).create_invite(body.name.strip(), body.email.strip().lower())
    # In production this link is emailed to the person; the admin never sees
    # the enrollment page, so the admin cannot consent on their behalf.
    return {"link": f"/enroll?tenant={tenant}&token={token}"}


@app.delete("/api/t/{tenant}/profiles/{profile_id}")
def admin_delete(tenant: str, profile_id: str):
    if not vault(tenant).delete_profile(profile_id, by="admin"):
        raise HTTPException(404, "no such profile")
    return {"deleted": True}


@app.get("/api/t/{tenant}/audit")
def audit(tenant: str):
    return vault(tenant).audit()


# --- the person: consent + enrollment + self-service delete -----------
@app.get("/api/invite/{tenant}/{token}")
def invite_info(tenant: str, token: str):
    inv = vault(tenant).peek_invite(token)
    if not inv:
        raise HTTPException(404, "invite link is invalid or expired")
    return {"name": inv["name"], "email": inv["email"], "organisation": TENANTS[tenant],
            "consent_text": CONSENT_TEXT, "consent_version": CONSENT_VERSION}


@app.post("/api/invite/{tenant}/{token}/enroll")
async def enroll(tenant: str, token: str, audio: UploadFile,
                 consent: bool = Form(False), consent_version: str = Form("")):
    v = vault(tenant)
    if not consent or consent_version != CONSENT_VERSION:
        raise HTTPException(400, "explicit consent to the current text is required")
    inv = v.take_invite(token)  # claimed atomically: one invite, one profile
    if not inv:
        raise HTTPException(404, "invite link is invalid or expired")
    # Consent is recorded BEFORE any voice processing starts.
    consent_id = v.log("consent_given", email=inv["email"], version=CONSENT_VERSION,
                       text_sha256=CONSENT_HASH, channel="personal invite link")
    try:
        samples = await decode_upload(audio)
        with engine_lock:
            turns = engine.diarize(samples)
        speech = sum(t.end - t.start for t in turns)
        if speech < MIN_ENROLL_SECONDS:
            raise HTTPException(400, f"only {speech:.0f}s of speech detected — "
                                     f"please speak for at least {MIN_ENROLL_SECONDS:.0f}s")
        per_cluster: dict[int, float] = {}
        for t in turns:
            per_cluster[t.cluster] = per_cluster.get(t.cluster, 0) + t.end - t.start
        main = max(per_cluster, key=per_cluster.get)
        if per_cluster[main] < 0.85 * speech:
            raise HTTPException(400, "more than one voice detected — please record alone")
        with engine_lock:
            vec = engine.embed(cluster_audio(samples, turns, main, max_seconds=60))
    except Exception as e:
        v.log("enrollment_failed", email=inv["email"], consent_id=consent_id,
              reason=getattr(e, "detail", "error"))
        v.restore_invite(token, inv)  # let the person try again
        raise
    pid = v.add_profile(inv["name"], inv["email"], vec, per_cluster[main], consent_id)
    # The raw recording is discarded here: it only lived in memory.
    # The manage token is the person's private delete link; in production it
    # is emailed to them, and they can also delete after logging in.
    return {"manage_token": v.create_manage_token(pid),
            "speech_seconds": round(per_cluster[main], 1)}


@app.delete("/api/me/{tenant}/{manage_token}")
def self_delete(tenant: str, manage_token: str):
    v = vault(tenant)
    profile_id = v.profile_for_manage_token(manage_token)
    if not profile_id or not v.delete_profile(profile_id, by="self"):
        raise HTTPException(404, "no such profile")
    return {"deleted": True}


# --- analysing a meeting ----------------------------------------------
def _chunks(start: float, end: float, max_len: float = 25.0):
    while start < end:
        yield start, min(end, start + max_len)
        start += max_len


@app.post("/api/t/{tenant}/analyze")
async def analyze(tenant: str, audio: UploadFile, threshold: float | None = Form(None)):
    v = vault(tenant)
    samples = await decode_upload(audio)
    thr = threshold if threshold is not None else recognizer.calibrated_threshold()
    t0 = time.time()
    with engine_lock:
        turns = engine.diarize(samples)
        clusters = sorted({t.cluster for t in turns})
        seconds = {c: sum(t.end - t.start for t in turns if t.cluster == c) for c in clusters}
        vecs = {c: engine.embed(cluster_audio(samples, turns, c)) for c in clusters}
        ids, names, mat = v.embeddings()  # this tenant's profiles only
        matches = recognizer.identify(vecs, seconds, ids, names, mat, thr)

        # Clusters recognised as the same person become one speaker.
        canonical = {}
        for c in clusters:
            pid = matches[c].profile_id
            canonical[c] = next((k for k in clusters if pid and matches[k].profile_id == pid), c)
        for t in turns:
            t.cluster = canonical[t.cluster]
        for c in clusters:
            k = canonical[c]
            if k != c:
                seconds[k] += seconds.pop(c)
                matches[k].score = max(matches[k].score, matches.pop(c).score)
        clusters = sorted(seconds)

        # Merge consecutive turns of the same speaker, then transcribe.
        merged = []
        for t in turns:
            if merged and merged[-1]["cluster"] == t.cluster and t.start - merged[-1]["end"] < 1.0:
                merged[-1]["end"] = t.end
            else:
                merged.append({"start": t.start, "end": t.end, "cluster": t.cluster})
        for seg in merged:
            texts = [engine.transcribe(samples[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)])
                     for a, b in _chunks(seg["start"], seg["end"])]
            seg["text"] = " ".join(x for x in texts if x)
    order = {c: i + 1 for i, c in enumerate(dict.fromkeys(t.cluster for t in turns))}
    # Each use of a profile is logged (ids only, no voice data).
    v.log("meeting_analyzed", speakers=len(clusters),
          profiles_used=",".join(m.profile_id for m in matches.values() if m.profile_id))
    return {
        "duration": round(len(samples) / SAMPLE_RATE, 1),
        "processing_seconds": round(time.time() - t0, 1),
        "threshold": thr,
        "speakers": [{"cluster": c, "anon": f"Speaker {order[c]}",
                      "name": matches[c].name, "profile_id": matches[c].profile_id,
                      "score": round(matches[c].score, 3),
                      "seconds": round(seconds[c], 1)} for c in clusters],
        "segments": [{**s, "start": round(s["start"], 1), "end": round(s["end"], 1)}
                     for s in merged],
    }


# --- evaluation results -----------------------------------------------
@app.get("/api/eval")
def evaluation():
    f = ROOT / "data" / "eval" / "metrics.json"
    if not f.exists():
        raise HTTPException(404, "run scripts/evaluate_ami.py first")
    return json.loads(f.read_text())


app.mount("/static", StaticFiles(directory=STATIC), name="static")
if DEMO.exists():
    app.mount("/demo", StaticFiles(directory=DEMO), name="demo")
