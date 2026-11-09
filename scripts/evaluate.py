"""Measure recognition quality and build demo files.

Data: Mini LibriSpeech dev-clean-2 + train-clean-5 (54 English speakers,
public domain).
Protocol:
  * speakers are split into a CALIBRATION half and a TEST half (disjoint
    people), so the threshold is never tuned on the people we report on;
  * in each half, 20 speakers are "enrolled" with ~20 s of speech (what the
    enrollment page asks for) and the rest stay unknown — guests, clients, people who
    declined consent;
  * enrollment clips and test clips of a speaker never overlap.
Tests:
  1. verification sweep  — every test clip vs every profile -> FAR / FRR / EER
  2. identification      — each clip: correct name / wrong name / unnamed
  3. synthetic meetings  — full pipeline (diarize -> fingerprint -> match),
                           time-weighted labelling accuracy, clean and noisy
  4. tenant isolation    — same voices, wrong tenant -> no names may appear

Run:  .venv/bin/python -m scripts.evaluate
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from app import recognizer, store
from app.engine import ROOT, SAMPLE_RATE, VoiceEngine, cluster_audio

LIBRI = ROOT / "data" / "raw" / "LibriSpeech"
OUT = ROOT / "data" / "eval"
DEMO = ROOT / "data" / "demo"
TARGET_FAR = 0.001  # per comparison; a meeting compares each voice with every profile
ENROLL_SECONDS = 20.0
N_ENROLLED = 20  # per half; the remaining speakers of the half stay unknown
rng = random.Random(7)

DEMO_NAMES = {"F": ["Anna Keller", "Lea Müller", "Sofia Bianchi"],
              "M": ["Marco Rossi", "Thomas Weber", "Luca Meier"]}


def load_speakers() -> dict[str, dict]:
    genders = {}
    meta = LIBRI / "SPEAKERS.TXT"
    if meta.exists():
        for line in meta.read_text().splitlines():
            if line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                genders[parts[0]] = parts[1]
    speakers = {}
    for subset in ("dev-clean-2", "train-clean-5"):
        for d in sorted((LIBRI / subset).iterdir()):
            files = sorted(d.rglob("*.flac"))
            if len(files) >= 8:  # need enrollment audio + several test clips
                speakers[d.name] = {"files": files, "gender": genders.get(d.name, "?")}
    return speakers


def read(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    assert sr == SAMPLE_RATE
    return audio


def split_speaker(files: list[Path]):
    """First utterances until ~20 s -> enrollment; the rest -> test clips."""
    enroll, test, total = [], [], 0.0
    for f in files:
        a = read(f)
        if total < ENROLL_SECONDS:
            enroll.append(a)
            total += len(a) / SAMPLE_RATE
        else:
            test.append(a)
    return np.concatenate(enroll), test


def clip(a: np.ndarray, max_s: float = 8.0) -> np.ndarray:
    # Meeting turns are short; cap test clips so we don't test on easy
    # 30-second monologues.
    n = int(max_s * SAMPLE_RATE)
    return a if len(a) <= n else a[:n]


def build_half(engine, speaker_ids, speakers):
    enrolled, unknown = speaker_ids[:N_ENROLLED], speaker_ids[N_ENROLLED:]
    profiles, clips = {}, []
    for sid in enrolled + unknown:
        enroll_audio, test_audio = split_speaker(speakers[sid]["files"])
        if sid in enrolled:
            profiles[sid] = engine.embed(enroll_audio)
        for a in test_audio[:12]:
            c = clip(a)
            if len(c) >= 2 * SAMPLE_RATE:
                clips.append({"speaker": sid, "known": sid in enrolled,
                              "vec": engine.embed(c), "audio": a})
    return profiles, clips


def scores_matrix(profiles, clips):
    ids = list(profiles)
    mat = np.stack([profiles[i] for i in ids])
    return ids, np.stack([c["vec"] for c in clips]) @ mat.T


def sweep(profiles, clips):
    ids, S = scores_matrix(profiles, clips)
    genuine, impostor = [], []
    for ci, c in enumerate(clips):
        for pi, pid in enumerate(ids):
            (genuine if c["speaker"] == pid else impostor).append(S[ci, pi])
    genuine, impostor = np.array(genuine), np.array(impostor)
    pts = []
    for t in np.round(np.arange(0.20, 0.901, 0.01), 2):
        pts.append({"threshold": float(t),
                    "far": float((impostor >= t).mean()),
                    "frr": float((genuine < t).mean())})
    eer_pt = min(pts, key=lambda p: abs(p["far"] - p["frr"]))
    return pts, {"threshold": eer_pt["threshold"], "rate": (eer_pt["far"] + eer_pt["frr"]) / 2}, \
        len(genuine), len(impostor)


def identification(profiles, clips, thr):
    ids, S = scores_matrix(profiles, clips)
    correct = wrong = known_unnamed = 0
    for ci, c in enumerate(clips):
        best = int(S[ci].argmax())
        pred = ids[best] if S[ci, best] >= thr else None
        truth = c["speaker"] if c["known"] else None
        if pred == truth:
            correct += 1
        elif pred is not None:
            wrong += 1
        else:
            known_unnamed += 1
    n = len(clips)
    return {"correct_rate": correct / n, "wrong_name_rate": wrong / n,
            "known_unnamed_rate": known_unnamed / n, "test_utterances": n}


def make_meeting(cast, clips_by_speaker, seconds=150.0):
    """Alternating turns of real utterances with short pauses. Returns audio
    plus a per-sample reference label array."""
    audio, labels, total = [], [], 0.0
    pause = np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)
    used = {s: list(clips_by_speaker[s]) for s in cast}
    for s in used.values():
        rng.shuffle(s)
    last = None
    while total < seconds:
        choices = [s for s in cast if used[s] and s != last]
        if not choices:
            break
        s = rng.choice(choices)
        a = used[s].pop()
        a = a[: int(rng.uniform(3, 12) * SAMPLE_RATE)]
        audio += [a, pause]
        labels += [speech_labels(a, cast.index(s)),
                   np.full(len(pause), -1, dtype=np.int16)]
        total += (len(a) + len(pause)) / SAMPLE_RATE
        last = s
    return np.concatenate(audio), np.concatenate(labels)


def speech_labels(a: np.ndarray, label: int) -> np.ndarray:
    """Reference labels for one utterance: audiobook clips contain pauses,
    and a pause is not speech the system could have attributed. Frames more
    than 35 dB below the utterance's loudest frame count as silence."""
    frame = int(0.03 * SAMPLE_RATE)
    out = np.full(len(a), label, dtype=np.int16)
    n = len(a) // frame
    if n == 0:
        return out
    e = 10 * np.log10((a[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-10)
    for i in np.where(e < e.max() - 35)[0]:
        out[i * frame:(i + 1) * frame] = -1
    return out


def add_noise(a: np.ndarray, snr_db: float) -> np.ndarray:
    p = np.mean(a ** 2)
    noise = np.random.default_rng(1).normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(a))
    return (a + noise).astype(np.float32)


def run_meeting(engine, audio, ids, names, mat, thr):
    turns = engine.diarize(audio)
    clusters = sorted({t.cluster for t in turns})
    secs = {c: sum(t.end - t.start for t in turns if t.cluster == c) for c in clusters}
    vecs = {c: engine.embed(cluster_audio(audio, turns, c)) for c in clusters}
    matches = recognizer.identify(vecs, secs, ids, names, mat, thr)
    hyp = np.full(len(audio), -1, dtype=np.int32)
    for t in turns:
        hyp[int(t.start * SAMPLE_RATE):int(t.end * SAMPLE_RATE)] = t.cluster
    return hyp, matches


def score_meeting(ref, hyp, matches, cast, known):
    speech = ref >= 0
    total = speech.sum()
    # Labelling accuracy: what name (or "unnamed") the reader sees.
    shown = np.array([None] * len(hyp), dtype=object)
    for c, m in matches.items():
        shown[hyp == c] = m.profile_id or "?"
    correct = wrong = 0
    for i, sid in enumerate(cast):
        idx = ref == i
        s = shown[idx]
        if sid in known:
            correct += (s == sid).sum()
        else:
            correct += ((s == "?") | (s == None)).sum()  # noqa: E711
        wrong += sum(1 for x in s if x not in (None, "?", sid))
    # Diarization error rate with greedy cluster->speaker mapping.
    mapping, used = {}, set()
    pairs = sorted(((((ref == i) & (hyp == c)).sum(), c, i) for c in matches
                    for i in range(len(cast))), reverse=True)
    for n, c, i in pairs:
        if c not in mapping and i not in used and n > 0:
            mapping[c], used = i, used | {i}
    mapped = np.array([mapping.get(h, -2) if h >= 0 else -1 for h in hyp])
    miss = (speech & (hyp < 0)).sum()
    fa = ((~speech) & (hyp >= 0)).sum()
    conf = (speech & (hyp >= 0) & (mapped != ref)).sum()
    return total, correct, wrong, miss + fa + conf


def meetings_eval(engine, profiles, clips, thr, n=8, snr=None):
    ids = list(profiles)
    names, mat = ids, np.stack([profiles[i] for i in ids])
    by_spk: dict[str, list] = {}
    for c in clips:
        by_spk.setdefault(c["speaker"], []).append(c["audio"])
    known_pool = [s for s in ids if s in by_spk]
    unknown_pool = [s for s in by_spk if s not in profiles]
    agg = np.zeros(4)
    minutes = 0.0
    for _ in range(n):
        cast = rng.sample(known_pool, 3) + rng.sample(unknown_pool, 1)
        audio, ref = make_meeting(cast, by_spk)
        if snr is not None:
            audio = add_noise(audio, snr)
        hyp, matches = run_meeting(engine, audio, ids, names, mat, thr)
        agg += score_meeting(ref, hyp, matches, cast, set(ids))
        minutes += len(audio) / SAMPLE_RATE / 60
    total, correct, wrong, der = agg
    return {"count": n, "total_minutes": round(minutes, 1), "speakers_per_meeting": 4,
            "attribution_accuracy": correct / total, "wrong_name_rate": wrong / total,
            "der": der / total}


def tenant_isolation(engine, test_profiles, clips, thr) -> int:
    """Enroll the test speakers in tenant A via the real vault code, then
    analyse their meeting as tenant B. Any of A's names appearing is a leak."""
    tmp = Path(tempfile.mkdtemp())
    old = store.TENANTS_DIR
    store.TENANTS_DIR = tmp
    try:
        a, b = store.TenantVault("tenant-a"), store.TenantVault("tenant-b")
        for sid, vec in test_profiles.items():
            a.add_profile(f"A-{sid}", f"{sid}@a.ch", vec, 20, "test")
        b.add_profile("B-only person", "x@b.ch", np.roll(vec, 7), 20, "test")
        by_spk: dict[str, list] = {}
        for c in clips:
            by_spk.setdefault(c["speaker"], []).append(c["audio"])
        leaks = 0
        for _ in range(3):
            cast = rng.sample([s for s in test_profiles if s in by_spk], 4)
            audio, _ = make_meeting(cast, by_spk, seconds=60)
            ids, names, mat = b.embeddings()
            _, matches = run_meeting(engine, audio, ids, names, mat, thr)
            leaks += sum(1 for m in matches.values() if m.name and m.name.startswith("A-"))
        return leaks
    finally:
        store.TENANTS_DIR = old
        shutil.rmtree(tmp)


def build_demo(speakers, test_ids, test_profiles, clips):
    """Enrollment samples + two demo meetings with fictional names."""
    if DEMO.exists():
        shutil.rmtree(DEMO)
    (DEMO / "enroll").mkdir(parents=True)
    (DEMO / "meetings").mkdir()
    by_spk: dict[str, list] = {}
    for c in clips:
        by_spk.setdefault(c["speaker"], []).append(c["audio"])
    pool = {"F": list(DEMO_NAMES["F"]), "M": list(DEMO_NAMES["M"])}
    cast = []
    for sid in [s for s in test_ids if s in test_profiles and s in by_spk]:
        g = speakers[sid]["gender"] if speakers[sid]["gender"] in pool else "F"
        if pool[g] and len(cast) < 4:
            cast.append((sid, pool[g].pop(0)))
    guest = next(s for s in test_ids if s not in test_profiles and s in by_spk)
    manifest = {"enroll_samples": [], "meetings": []}
    for sid, name in cast:
        audio, _ = split_speaker(speakers[sid]["files"])
        f = f"enroll/{name.split()[0].lower()}.wav"
        sf.write(DEMO / f, audio, SAMPLE_RATE)
        manifest["enroll_samples"].append({"file": f, "label": f"{name} (LibriSpeech speaker {sid})"})
    for i, (people, title) in enumerate([
            ([c[0] for c in cast] + [guest], "Team meeting — 4 colleagues + 1 guest"),
            ([cast[0][0], cast[1][0], guest], "Client call — 2 colleagues + 1 client")]):
        audio, _ = make_meeting(people, by_spk, seconds=120)
        f = f"meetings/meeting{i + 1}.wav"
        sf.write(DEMO / f, audio, SAMPLE_RATE)
        manifest["meetings"].append({"file": f, "title": title})
    (DEMO / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def main():
    engine = VoiceEngine(num_threads=4)
    speakers = load_speakers()
    ids = list(speakers)
    rng.shuffle(ids)
    half = len(ids) // 2
    cal_ids, test_ids = ids[:half], ids[half:2 * half]
    print(f"{len(ids)} speakers; calibration={len(cal_ids)} test={len(test_ids)}")

    cal_profiles, cal_clips = build_half(engine, cal_ids, speakers)
    pts, eer_cal, _, _ = sweep(cal_profiles, cal_clips)
    ok = [p for p in pts if p["far"] <= TARGET_FAR]
    thr = ok[0]["threshold"] if ok else 0.6
    print(f"calibrated threshold {thr} (EER on calibration {eer_cal['rate']:.3%})")

    test_profiles, test_clips = build_half(engine, test_ids, speakers)
    pts_t, eer_t, n_gen, n_imp = sweep(test_profiles, test_clips)
    at = next(p for p in pts_t if abs(p["threshold"] - thr) < 1e-6)
    ident = identification(test_profiles, test_clips, thr)
    print("identification", ident)
    meet = meetings_eval(engine, test_profiles, test_clips, thr)
    print("meetings clean", meet)
    noisy = meetings_eval(engine, test_profiles, test_clips, thr, n=4, snr=10)
    print("meetings noisy 10dB", noisy)
    leaks = tenant_isolation(engine, test_profiles, test_clips, thr)
    print("cross-tenant leaks", leaks)
    meet["cross_tenant_names_leaked"] = leaks
    meet["noisy_10db"] = noisy

    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {
        "dataset": f"Mini LibriSpeech (public-domain English audiobooks): {2 * half} speakers, "
                   f"split into {half} people for tuning the threshold and {half} different "
                   f"people for the reported results ({N_ENROLLED} enrolled + "
                   f"{half - N_ENROLLED} never enrolled in each group)",
        "models": "pyannote segmentation 3.0 + NVIDIA TitaNet-small + Whisper tiny.en, "
                  "all ONNX via sherpa-onnx, CPU only",
        "target_far": TARGET_FAR,
        "chosen_threshold": thr,
        "eer": eer_t,
        "sweep": pts_t,
        "verification": {"genuine_trials": n_gen, "impostor_trials": n_imp,
                         "far_at_chosen": at["far"], "frr_at_chosen": at["frr"]},
        "identification": {**ident, "enrolled_speakers": N_ENROLLED,
                           "unknown_speakers": half - N_ENROLLED,
                           "clip_seconds": "2–8 s each"},
        "meetings": meet,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    build_demo(speakers, test_ids, test_profiles, test_clips)
    print("wrote", OUT / "metrics.json", "and", DEMO)


if __name__ == "__main__":
    main()
