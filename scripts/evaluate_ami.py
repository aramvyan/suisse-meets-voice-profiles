"""Evaluation on real meetings: the AMI Meeting Corpus (CC BY 4.0).

Why AMI: the same 4 people meet several times, so we can do exactly what the
product does: learn voices in one meeting, recognise them in later meetings.
None of the three compared voice models lists AMI in its training data
(TitaNet's training data includes LibriSpeech, which is why the earlier
LibriSpeech test was dropped).

Protocol (leave-one-group-out cross-validation)
  * 4 groups of 4 people: ES2004, IS1009, TS3003, EN2002;
  * each group takes one turn as the CALIBRATION group: the voice model is
    chosen there by equal error rate on short clips, and the threshold by a
    sweep over its meetings (lowest wrong-name rate <= 1%, most correct);
    the other 3 groups (different people) are the TEST; results are pooled
    over the 4 turns, so every number comes from people not used for tuning;
  * voices are enrolled from each group's first meeting (a): ~20 s of each
    person, only where nobody else talks; tested on the later meetings;
  * guests: every meeting is run 4 times, each time one of the 4 people is
    NOT enrolled and must stay unnamed;
  * scoring per 10 ms of speech; overlapped speech (2+ people at once) is
    excluded from accuracy and reported separately.
  * the production setting (model + threshold) is then chosen the same way
    on all 4 groups; the cross-validated numbers estimate how it performs.

Audio: Mix-Headset (all headset mics summed), i.e. clean-ish close-talk.
Far-field room microphones are harder and are not tested here.

Run:  .venv/bin/python -m scripts.evaluate_ami
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

from app import recognizer, store
from app.engine import MODELS, ROOT, SAMPLE_RATE, VoiceEngine, cluster_audio

AMI = ROOT / "data" / "ami"
OUT = ROOT / "data" / "eval"
CACHE = AMI / "cache"
DEMO = ROOT / "data" / "demo"
FRAME = 0.01
ENROLL_SECONDS = 20.0
MAX_WRONG = 0.01  # threshold rule: lowest threshold with wrong names <= 1%

GROUPS = {"ES2004": ["b"], "IS1009": ["b", "c"], "TS3003": ["b", "c"], "EN2002": ["b", "c"]}

EMBED_MODELS = {
    "NVIDIA TitaNet-small": "nemo_en_titanet_small.onnx",
    "WeSpeaker ResNet34": "wespeaker_en_voxceleb_resnet34_LM.onnx",
    "3D-Speaker ERes2Net": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
}


# --- data ---------------------------------------------------------------
def read_rttm(meeting: str) -> list[tuple[float, float, str]]:
    segs = []
    for line in (AMI / "rttm" / f"{meeting}.rttm").read_text().splitlines():
        p = line.split()
        segs.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return segs


def read_audio(meeting: str) -> np.ndarray:
    a, sr = sf.read(AMI / "audio" / f"{meeting}.wav", dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    assert sr == SAMPLE_RATE, sr
    return a


def frame_labels(segs, n_frames, speakers):
    """Per 10 ms frame: speaker index, -1 silence, -2 overlap."""
    count = np.zeros(n_frames, dtype=np.int16)
    who = np.full(n_frames, -1, dtype=np.int16)
    for s, e, spk in segs:
        a, b = int(s / FRAME), min(n_frames, int(e / FRAME))
        count[a:b] += 1
        who[a:b] = speakers.index(spk)
    who[count > 1] = -2
    return who


def solo_segments(segs, speaker, min_len=1.0):
    """Stretches where only `speaker` talks."""
    others = [(s, e) for s, e, k in segs if k != speaker]
    out = []
    for s, e, k in segs:
        if k != speaker:
            continue
        cur = s
        for os_, oe in sorted(others):
            if oe <= cur or os_ >= e:
                continue
            if os_ - cur >= min_len:
                out.append((cur, os_))
            cur = max(cur, oe)
        if e - cur >= min_len:
            out.append((cur, e))
    return out


def enrollment_audio(audio, segs, speaker):
    parts, total = [], 0.0
    for s, e in solo_segments(segs, speaker, min_len=1.5):
        parts.append(audio[int(s * SAMPLE_RATE):int(e * SAMPLE_RATE)])
        total += e - s
        if total >= ENROLL_SECONDS:
            break
    n = int(30 * SAMPLE_RATE)  # one long monologue can overshoot; cap it
    return np.concatenate(parts)[:n], min(total, 30.0)


# --- embeddings -----------------------------------------------------------
class Embedders:
    def __init__(self):
        self.ex = {name: sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODELS / f), num_threads=4))
            for name, f in EMBED_MODELS.items()}

    def embed(self, name, samples):
        ex = self.ex[name]
        st = ex.create_stream()
        st.accept_waveform(SAMPLE_RATE, samples)
        st.input_finished()
        v = np.asarray(ex.compute(st), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


# --- one meeting: diarize once (cached), embed clusters with every model ----
def process_meeting(engine, emb, meeting):
    cache = CACHE / f"{meeting}.json"  # diarization is slow; cache the turns
    audio = read_audio(meeting)
    if cache.exists():
        turns = [recognizer_turn(*t) for t in json.loads(cache.read_text())]
    else:
        print(f"  diarizing {meeting} ({len(audio) / SAMPLE_RATE / 60:.1f} min)...", flush=True)
        turns = engine.diarize(audio)
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([(t.start, t.end, t.cluster) for t in turns]))
    clusters = sorted({t.cluster for t in turns})
    secs = {c: sum(t.end - t.start for t in turns if t.cluster == c) for c in clusters}
    vecs = {name: {c: emb.embed(name, cluster_audio(audio, turns, c)) for c in clusters}
            for name in EMBED_MODELS}
    n_frames = int(len(audio) / SAMPLE_RATE / FRAME)
    hyp = np.full(n_frames, -1, dtype=np.int32)
    for t in turns:
        hyp[int(t.start / FRAME):int(t.end / FRAME)] = t.cluster
    return {"turns": turns, "secs": secs, "vecs": vecs, "hyp": hyp,
            "minutes": len(audio) / SAMPLE_RATE / 60}


def recognizer_turn(start, end, cluster):
    from app.engine import Turn
    return Turn(float(start), float(end), int(cluster))


def profiles_for(emb, group):
    meeting = group + "a"
    audio, segs = read_audio(meeting), read_rttm(meeting)
    speakers = sorted({k for _, _, k in segs})
    profs = {name: {} for name in EMBED_MODELS}
    seconds = {}
    for spk in speakers:
        a, t = enrollment_audio(audio, segs, spk)
        seconds[spk] = t
        for name in EMBED_MODELS:
            profs[name][spk] = emb.embed(name, a)
    return speakers, profs, seconds


def score(proc, ref, speakers, enrolled, model, thr):
    ids = list(enrolled)
    mat = np.stack([enrolled[i] for i in ids])
    m = recognizer.identify(proc["vecs"][model], proc["secs"], ids, ids, mat, thr)
    shown = np.full(len(proc["hyp"]), None, dtype=object)
    for c, match in m.items():
        shown[proc["hyp"] == c] = match.profile_id or "?"
    n = min(len(ref), len(shown))
    ref, shown = ref[:n], shown[:n]
    single = ref >= 0
    correct = wrong = missed = 0
    for i, spk in enumerate(speakers):
        s = shown[ref == i]
        correct += int((s == (spk if spk in enrolled else "?")).sum())
        wrong += sum(1 for x in s if x not in (None, "?", spk))
        missed += int(sum(1 for x in s if x is None))
    return {"frames": int(single.sum()), "correct": correct, "wrong": wrong, "missed": missed}


def evaluate(meetings, emb, engine, model, thr, cache_proc):
    """All rotations (each person once a guest) over the given meetings."""
    agg = {"frames": 0, "correct": 0, "wrong": 0, "missed": 0}
    per_meeting = []
    for group, letters in meetings.items():
        if ("prof", group) not in cache_proc:
            cache_proc[("prof", group)] = profiles_for(emb, group)
        speakers, profs, _ = cache_proc[("prof", group)]
        for letter in letters:
            meeting = group + letter
            if meeting not in cache_proc:
                cache_proc[meeting] = process_meeting(engine, emb, meeting)
            proc = cache_proc[meeting]
            segs = read_rttm(meeting)
            ref = frame_labels(segs, len(proc["hyp"]), speakers)
            mt = {"frames": 0, "correct": 0, "wrong": 0, "missed": 0}
            for guest in speakers:
                enrolled = {s: profs[model][s] for s in speakers if s != guest}
                r = score(proc, ref, speakers, enrolled, model, thr)
                for k in mt:
                    mt[k] += r[k]
            for k in agg:
                agg[k] += mt[k]
            per_meeting.append({"meeting": meeting, "minutes": round(proc["minutes"], 1),
                                "correct": mt["correct"] / mt["frames"],
                                "wrong": mt["wrong"] / mt["frames"],
                                "overlap_share": float((ref == -2).sum() / max(1, (ref != -1).sum()))})
    f = agg["frames"]
    return {"correct": agg["correct"] / f, "wrong": agg["wrong"] / f,
            "missed": agg["missed"] / f, "frames": f, "per_meeting": per_meeting}


def clip_scores(emb, model, group, cache_proc):
    """Short real clips (2-8 s, one person talking) of one group vs that
    group's enrolled profiles: (genuine scores, impostor scores). Cached."""
    key = ("clips", group, model)
    if key in cache_proc:
        return cache_proc[key]
    if ("prof", group) not in cache_proc:
        cache_proc[("prof", group)] = profiles_for(emb, group)
    gen, imp = [], []
    speakers, profs, _ = cache_proc[("prof", group)]
    for letter in GROUPS[group]:
        audio, segs = read_audio(group + letter), read_rttm(group + letter)
        for spk in speakers:
            for s, e in solo_segments(segs, spk, min_len=2.0)[:25]:
                v = emb.embed(model, audio[int(s * SAMPLE_RATE):int(min(e, s + 8) * SAMPLE_RATE)])
                for other in speakers:
                    (gen if other == spk else imp).append(float(v @ profs[model][other]))
    cache_proc[key] = (gen, imp)
    return gen, imp


def eer_of(gen, imp):
    gen, imp = np.array(gen), np.array(imp)
    ts = np.arange(-0.2, 1.0, 0.005)
    best = min(ts, key=lambda t: abs((imp >= t).mean() - (gen < t).mean()))
    rate = ((imp >= best).mean() + (gen < best).mean()) / 2
    return {"eer": float(rate), "genuine": len(gen), "impostor": len(imp),
            "impostor_mean": float(imp.mean()), "genuine_mean": float(gen.mean())}


def pooled_eer(emb, model, groups, cache_proc):
    gen, imp = [], []
    for g in groups:
        a, b = clip_scores(emb, model, g, cache_proc)
        gen += a
        imp += b
    return eer_of(gen, imp)


def foreign_gallery(emb, engine, model, thr, cache_proc):
    """Each group's meetings analysed by a customer that has the OTHER groups'
    people enrolled (real, different voices), and none of the speakers.
    Every name shown is a false accept. Also checks the real vault code:
    profiles stored for customer A are not visible to customer B."""
    shown = clusters = 0
    for g in GROUPS:
        others = {f"{o}-{s}": cache_proc[("prof", o)][1][model][s]
                  for o in GROUPS if o != g for s in cache_proc[("prof", o)][0]}
        ids = list(others)
        mat = np.stack([others[i] for i in ids])
        for letter in GROUPS[g]:
            proc = cache_proc[g + letter]
            m = recognizer.identify(proc["vecs"][model], proc["secs"], ids, ids, mat, thr)
            shown += sum(1 for x in m.values() if x.name)
            clusters += sum(1 for c in m if proc["secs"][c] >= recognizer.MIN_CLUSTER_SECONDS)
    tmp = Path(tempfile.mkdtemp())
    old = store.TENANTS_DIR
    store.TENANTS_DIR = tmp
    try:
        a, b = store.TenantVault("tenant-a"), store.TenantVault("tenant-b")
        speakers, profs, _ = cache_proc[("prof", "IS1009")]
        for s in speakers:
            a.add_profile(s, f"{s}@a.ch", profs[model][s], 20, "test")
        visible_to_b = len(b.list_profiles())
    finally:
        store.TENANTS_DIR = old
        shutil.rmtree(tmp)
    return {"names_shown": shown, "speakers_checked": clusters, "gallery_size": 12,
            "profiles_visible_to_other_customer": visible_to_b}


def build_demo(cache_proc):
    """Real-meeting demo: enrollment samples from IS1009a and a 3-minute
    excerpt of IS1009b where all four people speak."""
    if DEMO.exists():
        shutil.rmtree(DEMO)
    (DEMO / "enroll").mkdir(parents=True)
    (DEMO / "meetings").mkdir()
    if ("prof", "IS1009") not in cache_proc:
        cache_proc[("prof", "IS1009")] = profiles_for(Embedders(), "IS1009")
    speakers, _, _ = cache_proc[("prof", "IS1009")]
    names = ["Anna Keller", "Marco Rossi", "Lea Müller", "Thomas Weber"]
    audio_a, segs_a = read_audio("IS1009a"), read_rttm("IS1009a")
    manifest = {"enroll_samples": [], "meetings": []}
    for spk, name in zip(speakers[:3], names):
        a, _ = enrollment_audio(audio_a, segs_a, spk)
        f = f"enroll/{name.split()[0].lower()}.wav"
        sf.write(DEMO / f, a, SAMPLE_RATE)
        manifest["enroll_samples"].append({"file": f, "label": f"{name} (AMI IS1009, speaker {spk})"})
    audio_b, segs_b = read_audio("IS1009b"), read_rttm("IS1009b")
    win, best, best_score = 180.0, 0.0, -1.0
    for start in np.arange(60, len(audio_b) / SAMPLE_RATE - win, 15):
        per = {s: 0.0 for s in speakers}
        for s, e, k in segs_b:
            per[k] += max(0.0, min(e, start + win) - max(s, start))
        sc = min(per.values())  # everyone must talk
        if sc > best_score:
            best, best_score = start, sc
    clip_ = audio_b[int(best * SAMPLE_RATE):int((best + win) * SAMPLE_RATE)]
    sf.write(DEMO / "meetings" / "meeting1.wav", clip_, SAMPLE_RATE)
    manifest["meetings"].append({"file": "meetings/meeting1.wav",
                                 "title": f"Real meeting (AMI IS1009b, 3 min) — 3 enrolled + 1 not enrolled ({speakers[3]})"})
    (DEMO / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def choose(emb, engine, groups, cache_proc, sweep_thr):
    """Model by clip EER, then threshold by meeting sweep, on `groups` only."""
    eers = {m: pooled_eer(emb, m, groups, cache_proc)["eer"] for m in EMBED_MODELS}
    model = min(eers, key=eers.get)
    calib = {g: GROUPS[g] for g in groups}
    pts = []
    for t in sweep_thr:
        r = evaluate(calib, emb, engine, model, float(t), cache_proc)
        pts.append({"threshold": float(t), "correct": r["correct"], "wrong": r["wrong"]})
    ok = [p for p in pts if p["wrong"] <= MAX_WRONG]
    best = max(ok, key=lambda p: p["correct"]) if ok else min(pts, key=lambda p: p["wrong"])
    return model, best["threshold"], eers


def pool(results):
    """Frame-weighted average of several evaluate() results."""
    tot = sum(r["frames"] for r in results)
    return {k: sum(r[k] * r["frames"] for r in results) / tot for k in ("correct", "wrong", "missed")}


def main():
    engine = VoiceEngine(num_threads=4)
    emb = Embedders()
    cache_proc: dict = {}
    sweep_thr = np.round(np.arange(0.30, 0.86, 0.025), 3)

    print("cross-validation: each group once as calibration, the other 3 as test", flush=True)
    folds, fold_results = [], []
    for g in GROUPS:
        model, thr, eers = choose(emb, engine, [g], cache_proc, sweep_thr)
        test = {o: GROUPS[o] for o in GROUPS if o != g}
        r = evaluate(test, emb, engine, model, thr, cache_proc)
        fold_results.append(r)
        folds.append({"calibration_group": g, "model": model, "threshold": thr,
                      "calibration_eers": eers, "test": {k: r[k] for k in ("correct", "wrong", "missed")},
                      "per_meeting": r["per_meeting"]})
        print(f"  calib {g}: {model} thr={thr} -> test correct={r['correct']:.3f} "
              f"wrong={r['wrong']:.4f}", flush=True)
    cv = pool(fold_results)
    print("cross-validated:", cv, flush=True)

    # Every model on every meeting with its own cross-validated thresholds,
    # so the model comparison is also on unseen people.
    models = {}
    for model in EMBED_MODELS:
        rs = []
        for g in GROUPS:
            calib = {g: GROUPS[g]}
            pts = [(float(t), evaluate(calib, emb, engine, model, float(t), cache_proc)) for t in sweep_thr]
            ok = [(t, r) for t, r in pts if r["wrong"] <= MAX_WRONG]
            t = max(ok, key=lambda x: x[1]["correct"])[0] if ok else min(pts, key=lambda x: x[1]["wrong"])[0]
            rs.append(evaluate({o: GROUPS[o] for o in GROUPS if o != g}, emb, engine, model, t, cache_proc))
        models[model] = {"test": pool(rs), "clips": pooled_eer(emb, model, list(GROUPS), cache_proc)}
        print(f"  {model}: {models[model]['test']} clipEER={models[model]['clips']['eer']:.3%}", flush=True)

    # Production setting: same procedure on all groups.
    prod_model, prod_thr, _ = choose(emb, engine, list(GROUPS), cache_proc, sweep_thr)
    print("production setting:", prod_model, prod_thr, flush=True)
    fg = foreign_gallery(emb, engine, prod_model, prod_thr, cache_proc)
    print("foreign gallery:", fg, flush=True)
    sweep = []
    for t in sweep_thr:
        rs = [evaluate({o: GROUPS[o] for o in GROUPS if o != g}, emb, engine, prod_model, float(t), cache_proc)
              for g in GROUPS]
        sweep.append({"threshold": float(t), **{k: v for k, v in pool(rs).items() if k != "missed"}})

    per_meeting = {}
    for f in folds:
        for p in f["per_meeting"]:
            per_meeting.setdefault(p["meeting"], []).append(p)
    per_meeting_avg = [{"meeting": k, "minutes": v[0]["minutes"],
                        "correct": float(np.mean([x["correct"] for x in v])),
                        "wrong": float(np.mean([x["wrong"] for x in v])),
                        "overlap_share": v[0]["overlap_share"]} for k, v in per_meeting.items()]
    minutes = sum(p["minutes"] for p in per_meeting_avg)
    metrics = {
        "dataset": "AMI Meeting Corpus (real meetings, CC BY 4.0), headset mix. 4 groups of 4 people, "
                   "7 later meetings. Voices enrolled from each group's first meeting. Leave-one-group-out: "
                   "model and threshold chosen on one group, results measured on the other three, all 4 "
                   "rotations pooled",
        "chosen_model": prod_model,
        "chosen_threshold": prod_thr,
        "max_wrong_rule": MAX_WRONG,
        "test_minutes": round(minutes, 1),
        "result": cv,
        "folds": [{k: f[k] for k in ("calibration_group", "model", "threshold", "test")} for f in folds],
        "per_meeting": per_meeting_avg,
        "clips": models[prod_model]["clips"],
        "test_sweep": sweep,
        "models": models,
        "foreign_gallery": fg,
        "limits": "Headset audio, not far-field room microphones. English only. 16 people in 4 groups; "
                  "results vary a lot between groups, see per meeting. Overlapped speech excluded.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    build_demo(cache_proc)
    print("wrote", OUT / "metrics.json")


if __name__ == "__main__":
    main()
