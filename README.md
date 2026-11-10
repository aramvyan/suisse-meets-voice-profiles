# Voice Profiles — prototype for Suisse Meets (Challenge B)

Working prototype of automatic speaker recognition with explicit consent,
self-service deletion and strict per-customer isolation. Everything runs
locally on CPU — no audio is sent to any cloud service.

## What it shows

| Challenge question | Where in the prototype |
|---|---|
| How does a voice get in, and how is consent given? | Admin can only *invite*. The person opens their own link, reads a versioned consent text, ticks an unticked box, records ~20 s. Consent is logged **before** the profile exists. Raw audio is discarded; only a 192-number fingerprint is kept. Self-delete button, 12-month auto-expiry. |
| How does recognition work, how do we measure it? | Diarization (pyannote 3.0) → one fingerprint per anonymous speaker → cosine match against the customer's profiles, with a threshold. Below threshold the speaker stays "Speaker N". `scripts/evaluate_ami.py` tests on real meetings (AMI corpus): voices enrolled from a group's first meeting, recognised in later meetings; model and threshold chosen on one group, results reported on three other groups; every meeting run with each person once as a non-enrolled guest. |
| How are customers kept separate? | Separate folder **and** separate encryption key per customer; no code path loads profiles across customers; tenant ids validated; tests prove a copied file can't be decrypted by another customer and the same voice is never named in another customer's meeting. |
| Build / test / launch | See the concept document. |

## Run it

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv sherpa-onnx fastapi "uvicorn[standard]" python-multipart numpy soundfile cryptography pytest httpx
./scripts/download.sh                    # models + 11 AMI meetings (~700 MB)
.venv/bin/python -m scripts.evaluate_ami # metrics + demo audio (~2 h on a laptop CPU)
.venv/bin/python -m pytest -q            # privacy / isolation / consent tests
.venv/bin/uvicorn app.main:app --port 8000
```

Open http://localhost:8000.

## Demo flow (5 minutes)

1. **Team & consent** → invite "Anna Keller" → open the link → read consent →
   tick → use the demo sample "Anna Keller" (or your own mic) → profile created.
   Repeat for the other two demo colleagues.
2. **Analyse a meeting** → the demo excerpt of a real AMI meeting → toggle
   *Before / With voice profiles*. The fourth person is not enrolled and stays unnamed.
3. Switch customer to **Zürich Legal** → analyse the same meeting → nobody is
   named: Alpine Dental's voices don't exist there.
4. Back to Alpine Dental → delete Anna → analyse again → Anna is anonymous.
5. **Audit log** → consent given / profile created / deleted, no biometrics.
6. **Accuracy** → how the threshold was chosen and what the numbers are.

## Honest limits

* AMI is tested with headset audio (clean, close microphones), in English.
  Far-field room microphones, phone codecs and Swiss languages are not
  tested yet; real pilot data (with consent) is needed before launch.
* The encryption key sits next to the data here; production uses a KMS in a
  Swiss region with per-tenant keys.
* Overlapping speech is not handled specially.
* **No admin login** — it is a localhost demo. In production the tenant comes
  from the authenticated admin session (never from the URL), invite links go
  only to the invited email address, and the admin API never sees the token.
  Here the invite link is shown to the admin so the flow can be demoed on one
  machine.
* The threshold slider on the analysis page is a demo control. In production
  the threshold is set by the server, never by the client.
* The enrollment page uses a fixed text; the random-sentence replay check from
  the concept is not built yet.

## Data and models

* Evaluation data: [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) (CC BY 4.0),
  reference speaker turns from [BUTSpeechFIT/AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup).
  Audio is downloaded by `scripts/download.sh` and not stored in this repository.
* Models: pyannote segmentation 3.0, NVIDIA TitaNet-small, WeSpeaker ResNet34,
  3D-Speaker ERes2Net and Whisper tiny.en, in the ONNX versions published by
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
