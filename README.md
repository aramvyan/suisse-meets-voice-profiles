# Voice Profiles — prototype for Suisse Meets (Challenge B)

Working prototype of automatic speaker recognition with explicit consent,
self-service deletion and strict per-customer isolation. Everything runs
locally on CPU — no audio is sent to any cloud service.

## What it shows

| Challenge question | Where in the prototype |
|---|---|
| How does a voice get in, and how is consent given? | Admin can only *invite*. The person opens their own link, reads a versioned consent text, ticks an unticked box, records ~20 s. Consent is logged **before** the profile exists. Raw audio is discarded; only a 192-number fingerprint is kept. Self-delete button, 12-month auto-expiry. |
| How does recognition work, how do we measure it? | Diarization (pyannote 3.0) → one fingerprint per anonymous speaker (NVIDIA TitaNet-small, picked by benchmark) → cosine match against the customer's profiles, one-to-one, with a threshold. Below threshold the speaker stays "Speaker N". `scripts/evaluate.py` tunes the threshold on one group of people and reports on a different group: FAR/FRR/EER, per-clip identification, full-pipeline meetings (clean + noisy), tenant-leak test. |
| How are customers kept separate? | Separate folder **and** separate encryption key per customer; no code path loads profiles across customers; tenant ids validated; tests prove a copied file can't be decrypted by another customer and the same voice is never named in another customer's meeting. |
| Build / test / launch | See the concept document. |

## Run it

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv sherpa-onnx fastapi "uvicorn[standard]" python-multipart numpy soundfile cryptography pytest httpx
./scripts/download.sh                    # models + Mini LibriSpeech (~400 MB)
.venv/bin/python -m scripts.evaluate     # metrics + demo audio (~15 min on CPU)
.venv/bin/python -m pytest -q            # privacy / isolation / consent tests
.venv/bin/uvicorn app.main:app --port 8000
```

Open http://localhost:8000.

## Demo flow (5 minutes)

1. **Team & consent** → invite "Anna Keller" → open the link → read consent →
   tick → use the demo sample "Anna Keller" (or your own mic) → profile created.
   Repeat for 2–3 colleagues.
2. **Analyse a meeting** → demo "Team meeting" → toggle *Before / With voice
   profiles*. The guest stays unnamed.
3. Switch customer to **Zürich Legal** → analyse the same meeting → nobody is
   named: Alpine Dental's voices don't exist there.
4. Back to Alpine Dental → delete Anna → analyse again → Anna is anonymous.
5. **Audit log** → consent given / profile created / deleted, no biometrics.
6. **Accuracy** → how the threshold was chosen and what the numbers are.

## Honest limits

* LibriSpeech is clean read speech. Real meetings have far-field mics,
  cross-talk and noise; the noisy-meeting test is a first proxy, real pilot
  data (with consent) is needed before launch.
* The encryption key sits next to the data here; production uses a KMS in a
  Swiss region with per-tenant keys.
* Overlapping speech is not handled specially.
* **No admin login** — it is a localhost demo. In production the tenant comes
  from the authenticated admin session (never from the URL), invite links go
  only to the invited email address, and the admin API never sees the token.
  Here the invite link is shown to the admin so the flow can be demoed on one
  machine.
