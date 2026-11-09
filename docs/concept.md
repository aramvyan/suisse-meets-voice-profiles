# Voice Profiles for Suisse Meets

Concept and plan. Challenge B. Aram Verdoyan, November 2026.

## Summary

Today a transcript says "Speaker 1" and "Speaker 2". Users rename the same colleagues in every meeting.

The idea: a person records their voice once, with their own explicit consent. After that, every meeting of their organisation shows their name automatically. On any device.

I built a working prototype to test the idea before writing this plan. It runs fully on a laptop CPU, with open models.

| What was tested | Result |
|---|---|
| Short voice clips (2-8 s), 20 known + 7 unknown people | 99.7% correct, 0.3% wrong name |
| Full pipeline on 8 synthetic meetings (20 min) | 91.8% of speech time correctly labelled, 2.0% under a wrong name |
| Same meetings with background noise | 89.9% correct, 1.1% wrong name |
| Same voices, analysed as another customer | 0 names leaked |

These numbers come from clean audiobook speech. Real meetings are harder. Section 3 explains how we measure on real audio before launch.

Three rules guide every decision below:

1. **Only the person can say yes.** Not the admin, not the boss.
2. **A wrong name is worse than no name.** When the system is not sure, it shows "Speaker 2", like today.
3. **One customer's voices never meet another customer's data.** Not in code, not in storage, not in logs.

---

## 1. How a voice gets into the system, and how consent works

### Enrollment flow

1. The customer admin turns the feature on for the organisation. It is **off by default**.
2. The admin **invites** colleagues. The admin cannot record or consent for anyone.
3. Each person gets a personal link by email. It works for 7 days and only once.
4. The person reads a short consent text. It says in plain words:
    - what is stored (a voice fingerprint, not the recording),
    - why (only to show their name in this organisation's transcripts),
    - where (Switzerland),
    - that saying no has no consequence for them,
    - how to delete it, and that it expires after 12 months.
5. The checkbox is **not** pre-ticked. There is a visible "No, thanks" button.
6. The person reads a short text aloud for about 20 seconds, in the browser or the app.
7. The system checks the sample: at least 8 seconds of speech, only one voice. If not, it asks to record again.
8. The consent record is saved **first**. Then the fingerprint is created. Then the recording is deleted.

### Second way to enroll: "This is me"

Recording a sample is extra work, and some people will skip it. So after a meeting, a person can open the transcript and mark **their own** segments with "This is me". The same consent screen comes first.

Risk: someone could claim another person's voice. So this path needs one short live check: the person says one sentence, and it must match the claimed segments. If not, nothing is saved.

### What is stored

| Stored | Not stored |
|---|---|
| Voice fingerprint: 192 numbers, encrypted | The enrollment recording (deleted right after) |
| Name, email, organisation | Fingerprints of guests or non-enrolled people |
| Consent record: who, when, which text version (hash), which channel | |
| Audit log: created, used, deleted. No biometric data in it | |

The audit log has no voice data. So we can keep it after deletion, as proof that consent was given and later withdrawn.

### Deleting

- **The person:** one click in their settings, and a delete link in every email we send them. Deletion is immediate.
- **The admin:** can delete any profile in the organisation.
- **Automatic:** after 12 months, unless the person renews it. We send a reminder 30 days before.
- **Leavers:** when a user is removed from the customer account, their profile is deleted the same moment.
- **Whole customer leaves:** we delete the customer's encryption key. All their profiles become unreadable at once, also in backups (crypto-shredding).

### People who did not enroll

Guests, clients and people who said no never get a profile. During a meeting the system computes a fingerprint for each anonymous speaker, compares it, and throws it away. It is never saved. These people stay "Speaker N". The meeting host can also switch off voice recognition for a single meeting, for example a sensitive client call.

### What voice profiles are not used for

- Not for login or authentication. A recorded voice could fool it, and the profile only exists to show a name.
- Not for emotion, mood, or performance analysis, and not for per-employee statistics like speaking time.
- Not for training our models.

These limits go into the consent text and the customer contract.

---

## 2. How recognition works

```
meeting audio
   │
   ▼
1. Diarization: split audio by voice  ──►  "Speaker 1", "Speaker 2", ...
   │                                         (anonymous, like today)
   ▼
2. One fingerprint per anonymous speaker (up to 30 s of their speech)
   │
   ▼
3. Compare with THIS customer's profiles (cosine similarity)
   │
   ▼
4. Decision:
   score ≥ threshold  → show the name
   score < threshold  → keep "Speaker N"
   one person can't take two speakers, unless the score is much higher
   (diarization sometimes splits one person in two)
```

**Models used in the prototype** (all open, all run on CPU, no cloud):

- pyannote segmentation 3.0: who speaks when,
- NVIDIA TitaNet-small: the voice fingerprint,
- Whisper tiny.en: the text (only for the demo; Suisse Meets has its own transcription).

I compared three fingerprint models on the same data before choosing:

| Model | Equal error rate | Avg score, different people | Avg score, same person |
|---|---|---|---|
| WeSpeaker ResNet34 | 2.48% | 0.57 | 0.89 |
| 3D-Speaker ERes2Net | 0.62% | 0.18 | 0.82 |
| **NVIDIA TitaNet-small** | **0.00%** | **0.07** | **0.88** |

With the first model, 22% of meeting speech got a wrong name. With TitaNet it was 2%. This is why the model choice must be made with data and repeated on real Suisse Meets audio.

### Where it runs

Recognition runs on the server, in Switzerland, in the same job that creates the transcript today. Phone, recorder and browser recordings all upload audio already. Profiles are never sent to devices. This keeps them in one place, so deletion is real and immediate.

### Different devices

A voice recorded on a phone sounds different from the same voice on a conference recorder. Two measures:

1. We measure accuracy **per device type**, not only in total.
2. A profile can hold up to 3 fingerprints from different devices. "This is me" from a recorder meeting adds a recorder fingerprint.

---

## 3. How we measure that it is right

### Before launch (offline)

We build a test set from real meetings of our own team and pilot customers, with everyone's consent. Then:

| Metric | What it answers | Target for launch |
|---|---|---|
| Wrong-name rate (share of speech time under a wrong name) | How often do we embarrass a user? | below 1% |
| Correct-label rate (name, or "unknown" for guests) | How much work do we save? | above 85% |
| False accept rate per comparison | Is the threshold safe? | below 0.1% |
| Results per device, language, gender | Does it work for everyone the same? | no group much worse than average |

**Rule of the test:** the threshold is tuned on one group of people and reported on a **different** group. The prototype does this: 27 people for tuning, 27 other people for the results. Otherwise the numbers look better than reality.

Languages: Swiss German, French and Italian speakers must be in the test set. Voice fingerprints mostly don't depend on language, but this must be checked, not assumed.

### After launch (online)

- **Correction rate:** how often users change an automatic name. This is the best real-world signal, and we get it for free.
- **Auto-named rate:** share of speakers that got a name.
- **Opt-in rate:** share of invited people who said yes. A low rate means the consent text or the value is unclear.
- All of it per device type and per customer, on a dashboard, with an alert when the correction rate goes up.

---

## 4. How each customer's profiles stay separate

Isolation is not one wall. It is several, so one mistake does not leak data.

1. **Tenant comes from login.** The matching job reads the customer id from the authenticated session or job record. Never from a URL or request field.
2. **Separate encryption key per customer.** Envelope encryption with a key service in a Swiss region. A file copied from customer A can't be read with customer B's key. The prototype has a test for exactly this.
3. **No shared search index.** The comparison function receives one customer's profiles and nothing else. There is no code path that loads profiles across customers.
4. **Database rules** (row-level security) as a second check, in case the code has a bug.
5. **Logs never contain fingerprints.** Only ids.
6. **Automated tests in CI:** enroll voices in customer A, analyse the same voices as customer B, expect zero names. This runs on every change.
7. **External penetration test** before general launch.

---

## 5. Legal frame

This is my reading of the sources, not legal advice. Phase 0 of the plan is a review by a Swiss data protection lawyer.

### The newest Swiss case: PostFinance voiceprints

PostFinance created voiceprints of callers on an opt-out basis. The FDPIC ruled this unlawful in May 2025. On 21 October 2026 the Federal Administrative Court confirmed it (A-4467/2025):

- creating a voiceprint needs **express consent**,
- a person who just **keeps talking** after being informed has **not** consented,
- voiceprints without consent must be deleted.

The judgment can still be appealed. But the direction is clear, and our design follows it: an active, separate "yes" before any profile exists. No opt-out, no "by continuing you agree".

Sources: [BVGer press release](https://www.bvger.ch/en/newsroom/media-releases/creation-of-voiceprints-requires-explicit-consent-of-affected-individuals-3903), [FDPIC conclusion](https://www.edoeb.admin.ch/en/conclusion-investigation-into-voice-recognition-postfinance).

### Switzerland: revised Data Protection Act (nDSG / revFADP, SR 235.1)

| Rule | What it means for us |
|---|---|
| Art. 5 lit. c no. 4: biometric data that uniquely identifies a person is **sensitive** | Voice profiles are sensitive data |
| Art. 6 para. 7 lit. a: consent for sensitive data must be **explicit** | Unticked checkbox, versioned text, logged |
| Art. 6 para. 4: destroy data when no longer needed | Recording deleted at once, profile expires after 12 months |
| Art. 7: privacy by design and by default | Feature off by default, store fingerprint only |
| Art. 9: processing by a processor | Customer contract (DPA) defines what we may do |
| Art. 22: impact assessment for large-scale processing of sensitive data | DPIA in Phase 0 |
| Art. 24: report a breach with high risk to the FDPIC; a processor reports to the controller | Incident process before launch |
| Art. 30 and 32: processing against a person's express wish breaks personality rights; a court can order deletion | One-click deletion, effective immediately |

Source: [Fedlex SR 235.1](https://www.fedlex.admin.ch/eli/cc/2022/491/en)

### EU customers: GDPR

| Rule | What it means for us |
|---|---|
| Art. 4(14) and Art. 9(1): biometric data used to uniquely identify a person is a special category | Same as Switzerland: sensitive |
| Art. 9(2)(a): allowed with explicit consent for a specified purpose | One purpose only: names in transcripts |
| Art. 7(3): withdrawing must be as easy as giving | Delete link in every email, one click |
| Art. 17: right to erasure | Immediate deletion, crypto-shredding for backups |
| Art. 28: processor contract | The customer is controller, Suisse Meets is processor. We stay processor only if we never use voices for our own goals, for example training. |
| Art. 35(3)(b): DPIA for large-scale special-category data | Same DPIA covers both laws |

Source: [GDPR on EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng)

### The hard question 1: can an employee really consent?

- **EU:** the EDPB says consent at work is usually not free, because of the power imbalance. It is possible only when refusing has **no negative effect at all** (Guidelines 05/2020 on consent, section 3.1.1).
- **Switzerland:** Art. 328b of the Code of Obligations limits what an employer may process about employees. The FDPIC says employee consent cannot override it. The Federal Supreme Court (4A_518/2020) reads it more flexibly. Lawyers disagree.

Our design makes refusing cost nothing:

- the request comes from the product, not from the manager,
- there is no "refused" status, no reminders, no list of who said no,
- people without a profile get the same transcripts as today,
- the feature only adds convenience. Nobody's work depends on it.

A Swiss employment lawyer must confirm this is enough.

### The hard question 2: guests' voices

To know who is speaking, the system compares **every** voice in the meeting with the profiles, including guests who never agreed to anything. The EDPB gives a similar example with video: a hotel uses face recognition to spot consenting VIP guests. The EDPB says this is unlawful unless **all** scanned guests consent, because everyone's face is compared (Guidelines 3/2019, section 5.1).

The same logic may apply to voices, even though the guest's fingerprint is never saved. Options, from light to strict:

1. Notice at the start of the meeting, host switch per meeting, fingerprint thrown away (the baseline design).
2. Compare only speakers when every participant is a known, enrolled colleague (internal meetings only).
3. "Suggest and confirm": the system shows "Is this Anna?" and a person confirms.

This is the biggest open legal question. The decision is for the lawyer and the business in Phase 0. The prototype already keeps guests anonymous and never stores their fingerprint. The meeting notice and host switch are part of the production build.

### EU AI Act (Regulation 2024/1689)

- "Remote biometric identification" means identifying people **without their active involvement** by comparing them with a reference database (Art. 3(41)). Such systems are **high-risk** (Annex III point 1(a)). Pure 1:1 verification is excluded.
- Identifying who speaks in a meeting, against a list of enrolled colleagues, may fall under this definition. Enrolment could count as "active involvement", but there is no official guidance yet.
- High-risk obligations for Annex III systems now apply from **2 December 2027** (moved by Regulation 2026/1744). They also reach Swiss providers when the output is used in the EU.
- If the feature is high-risk, it needs risk management, logging, human oversight and a conformity assessment. That is real cost. The plan: get a classification from the lawyer in Phase 0, and launch for Swiss customers first if EU classification is unclear.

Sources: [AI Act on EUR-Lex](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng), [Regulation 2026/1744](https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng)

### Workplace monitoring

Art. 26 of Ordinance 3 to the Labour Act (ArGV 3) forbids systems built to monitor employees' behaviour. Showing a name in minutes is not monitoring. But per-person analytics on top of it (speaking time per employee, participation scores) could be. So voice profiles are never used for per-employee statistics. This is written into the consent text and the contract. ([FDPIC on workplace monitoring](https://www.edoeb.admin.ch/en/monitoring-systems-in-the-workplace))

---

## 6. Build, test and launch, step by step

| Phase | Time | What | Gate to the next phase |
|---|---|---|---|
| 0. Legal | 2-3 weeks | DPIA, consent text, employee-consent question (Art. 328b CO), guest option (section 5), AI Act classification, customer contract (DPA) and privacy policy, with a Swiss data protection lawyer | Lawyer sign-off |
| 1. Real-data test | 2-3 weeks | Run the prototype pipeline on our own team's meetings (with consent), all three device types | Wrong-name rate below 1% on real audio |
| 2. Production build | 3-4 weeks | Enrollment in web and app, profile service with per-customer keys, deletion flows, audit log, admin switch, "wrong person" correction button, monitoring | All isolation and consent tests pass, security review done |
| 3. Closed beta | 4 weeks | 3-5 friendly customers, feature flag per customer, weekly review of correction rate and opt-in rate | Correction rate stable and low, no privacy incident |
| 4. Launch | ongoing | Opt-in for all customers. Recalibrate the threshold every quarter with new consented data | |

**Off switch:** one feature flag stops recognition for a customer, or for everyone, at once. Transcripts fall back to "Speaker N". Nothing else breaks.

---

## 7. Risks and open questions

| Risk | Plan |
|---|---|
| Employee consent may not count as free (EDPB, Art. 328b CO) | Request comes from the product, refusing costs nothing, no list of who said no. Swiss employment lawyer to confirm. |
| Guests' voices are compared, even if not stored (EDPB hotel example) | Three options in section 5, from notice to internal-only matching. Decided in Phase 0. |
| EU AI Act may classify it as high-risk from 2 Dec 2027 | Classification in Phase 0. Swiss customers first if unclear. |
| PostFinance judgment may be appealed and change | Our design already meets the strict reading: active express consent. |
| Real meetings are harder than test audio | Phase 1 gate on real data. Conservative threshold. |
| One person split into two speakers | Second matching pass with a stricter threshold. Remaining cases stay anonymous, not wrong. |
| Overlapping speech | Not handled in v1. Measure how often it happens in real meetings first. |
| Model works worse for some accents or voices | Measure per group before launch, block launch if one group is much worse. |

---

## Appendix: the prototype

Source code: `suisse-meets-voice-profiles` (Python, FastAPI, sherpa-onnx). Runs with one command on a laptop.

What it shows:

- Invite, consent, recording and self-delete pages.
- Meeting analysis with a "before / after" switch, and the guest staying anonymous.
- Two fictional customers. The same meeting gets names in one and no names in the other.
- Accuracy page: how the threshold was chosen, the model comparison, all test results.
- Audit log of consent and deletions.
- 25 automated tests for consent, deletion, encryption and isolation rules.
