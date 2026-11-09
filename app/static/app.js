const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const fmtDate = (iso) => iso ? new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) : "";
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

let tenant = null;
let calibrated = 0.55;
let lastResult = null;
let view = "after";

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

// ---------- tabs ----------
document.querySelectorAll("nav.tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("nav.tabs button").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll("main > section").forEach((s) => s.classList.toggle("hidden", s.id !== `tab-${b.dataset.tab}`));
  refresh();
}));

// ---------- tenant ----------
async function initTenants() {
  const list = await api("/api/tenants");
  $("#tenant").innerHTML = list.map((t) => `<option value="${t.id}">${esc(t.name)}</option>`).join("");
  tenant = list[0].id;
  $("#tenant").addEventListener("change", (e) => {
    tenant = e.target.value;
    lastResult = null;
    $("#result-panel").classList.add("hidden");
    $("#invite-result").classList.add("hidden");
    refresh();
  });
}

// ---------- team ----------
async function loadProfiles() {
  const list = await api(`/api/t/${tenant}/profiles`);
  if (!list.length) {
    $("#profiles").innerHTML = `<div class="empty">No voice profiles yet for this customer. Invite someone above.</div>`;
    return;
  }
  $("#profiles").innerHTML = `<table><thead><tr><th>Name</th><th>Email</th><th>Enrolled</th><th>Auto-deletes</th><th>Sample</th><th></th></tr></thead><tbody>${
    list.map((p) => `<tr>
      <td><b>${esc(p.name)}</b></td><td class="muted">${esc(p.email)}</td>
      <td>${fmtDate(p.enrolled_at)}</td><td>${fmtDate(p.expires_at)}</td>
      <td class="mono">${p.sample_seconds}s</td>
      <td style="text-align:right"><button class="btn danger" data-del="${p.id}">Delete</button></td></tr>`).join("")
  }</tbody></table>`;
  document.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true;
    await api(`/api/t/${tenant}/profiles/${b.dataset.del}`, { method: "DELETE" });
    loadProfiles();
  }));
}

$("#invite-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = $("#inv-name").value, email = $("#inv-email").value;
  const box = $("#invite-result");
  try {
    const { link } = await api(`/api/t/${tenant}/invites`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email }),
    });
    box.className = "notice ok";
    box.innerHTML = `Invite created. In production this link is emailed to <b>${esc(email)}</b> and expires in 7 days.<br>
      <a href="${link}" target="_blank" rel="noopener">Open as ${esc(name)} →</a>`;
    $("#invite-form").reset();
  } catch (err) {
    box.className = "notice err";
    box.textContent = err.message;
  }
});

// ---------- meeting ----------
const thr = $("#thr");
function showThr() {
  $("#thr-v").textContent = Number(thr.value).toFixed(2);
  $("#thr-note").textContent = Math.abs(thr.value - calibrated) < 0.005
    ? "(calibrated on test data)" : `(calibrated: ${calibrated.toFixed(2)})`;
}
thr.addEventListener("input", showThr);

async function initDemo() {
  try { calibrated = (await api("/api/eval")).chosen_threshold; } catch {}
  thr.value = calibrated;
  showThr();
  const demo = await api("/api/demo-files");
  for (const m of demo.meetings || []) {
    $("#demo-meeting").insertAdjacentHTML("beforeend",
      `<option value="${esc(m.file)}">${esc(m.title)}</option>`);
  }
}

$("#analyze-btn").addEventListener("click", async () => {
  const file = $("#meeting-file").files[0];
  const demo = $("#demo-meeting").value;
  if (!file && !demo) { $("#analyze-status").textContent = "Choose a file or a demo meeting first."; return; }
  const fd = new FormData();
  let audioUrl;
  if (file) { fd.append("audio", file); audioUrl = URL.createObjectURL(file); }
  else {
    const blob = await (await fetch(`/demo/${demo}`)).blob();
    fd.append("audio", blob, demo.split("/").pop());
    audioUrl = `/demo/${demo}`;
  }
  fd.append("threshold", thr.value);
  $("#analyze-btn").disabled = true;
  $("#analyze-status").textContent = "Processing locally… (diarization + fingerprints + transcription)";
  try {
    lastResult = await api(`/api/t/${tenant}/analyze`, { method: "POST", body: fd });
    lastResult.audioUrl = audioUrl;
    $("#analyze-status").textContent = "";
    renderResult();
  } catch (err) {
    $("#analyze-status").textContent = `Error: ${err.message}`;
  } finally {
    $("#analyze-btn").disabled = false;
  }
});

$("#v-before").addEventListener("click", () => { view = "before"; renderResult(); });
$("#v-after").addEventListener("click", () => { view = "after"; renderResult(); });

function renderResult() {
  const r = lastResult;
  if (!r) return;
  $("#v-before").classList.toggle("on", view === "before");
  $("#v-after").classList.toggle("on", view === "after");
  $("#result-panel").classList.remove("hidden");
  const named = r.speakers.filter((s) => s.name).length;
  $("#result-meta").textContent =
    `${fmtTime(r.duration)} of audio · ${r.speakers.length} voices found · ${named} recognised · threshold ${r.threshold.toFixed(2)} · processed in ${r.processing_seconds}s on this machine`;
  const byCluster = Object.fromEntries(r.speakers.map((s) => [s.cluster, s]));
  const label = (s) => (view === "after" && s.name) ? s.name : s.anon;
  $("#speakers").innerHTML = r.speakers.map((s) => {
    const status = view === "before" ? "" : s.name
      ? `<span class="pill ok">match ${s.score.toFixed(2)}</span>`
      : `<span class="pill warn">not recognised · best ${s.score.toFixed(2)}</span>`;
    return `<div class="spk"><div class="who">${esc(label(s))}</div>
      <div class="meta">${s.seconds}s of speech</div>${status}</div>`;
  }).join("");
  $("#audio-wrap").innerHTML = `<audio controls src="${r.audioUrl}" style="width:100%;margin-bottom:8px"></audio>`;
  $("#segments").innerHTML = r.segments.map((g) => {
    const s = byCluster[g.cluster];
    const anon = !(view === "after" && s.name);
    return `<div class="seg"><div class="t"><a href="#" data-seek="${g.start}">${fmtTime(g.start)}</a></div>
      <div class="n ${anon ? "anon" : ""}">${esc(label(s))}</div><div>${esc(g.text) || '<span class="muted">…</span>'}</div></div>`;
  }).join("");
  document.querySelectorAll("[data-seek]").forEach((a) => a.addEventListener("click", (e) => {
    e.preventDefault();
    const audio = $("#audio-wrap audio");
    audio.currentTime = Number(a.dataset.seek);
    audio.play();
  }));
}

// ---------- accuracy ----------
function sweepChart(sweep, chosen, eer) {
  const W = 640, H = 260, L = 46, R = 16, T = 14, B = 36;
  const x = (t) => L + (t - sweep[0].threshold) / (sweep.at(-1).threshold - sweep[0].threshold) * (W - L - R);
  const y = (v) => T + (1 - v) * (H - T - B);
  const line = (k) => sweep.map((p, i) => `${i ? "L" : "M"}${x(p.threshold).toFixed(1)},${y(p[k]).toFixed(1)}`).join("");
  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) =>
    `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/>
     <text x="${L - 8}" y="${y(v) + 4}" font-size="11" text-anchor="end" fill="var(--muted)">${v * 100}%</text>`).join("");
  const ticks = sweep.filter((_, i) => i % 10 === 0).map((p) =>
    `<text x="${x(p.threshold)}" y="${H - 14}" font-size="11" text-anchor="middle" fill="var(--muted)">${p.threshold.toFixed(1)}</text>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="False accept and false reject rate by threshold">
    ${grid}${ticks}
    <text x="${(W + L) / 2}" y="${H - 1}" font-size="11" text-anchor="middle" fill="var(--muted)">match threshold (cosine similarity)</text>
    <line x1="${x(chosen)}" x2="${x(chosen)}" y1="${T}" y2="${H - B}" stroke="var(--ink)" stroke-dasharray="4 3"/>
    <text x="${x(chosen) + 6}" y="${T + 12}" font-size="11" fill="var(--ink)">chosen ${chosen.toFixed(2)}</text>
    <path d="${line("far")}" fill="none" stroke="var(--accent)" stroke-width="2.2"/>
    <path d="${line("frr")}" fill="none" stroke="#2b5fad" stroke-width="2.2"/>
    <circle cx="${x(eer.threshold)}" cy="${y(eer.rate)}" r="4" fill="var(--ink)"/>
  </svg>
  <div class="row small" style="gap:18px;margin-top:4px">
    <span><b style="color:var(--accent)">━</b> False accept (wrong person named)</span>
    <span><b style="color:#2b5fad">━</b> False reject (known person left unnamed)</span>
    <span>● Equal error rate ${pct(eer.rate, 2)}</span>
  </div>`;
}

async function loadAccuracy() {
  let m;
  try { m = await api("/api/eval"); }
  catch { $("#accuracy").innerHTML = `<div class="panel"><div class="empty">No evaluation yet — run <span class="mono">scripts/evaluate.py</span>.</div></div>`; return; }
  const v = m.verification, id = m.identification, mt = m.meetings;
  $("#accuracy").innerHTML = `
  <div class="panel">
    <h2>How we know it works</h2>
    <p class="sub">${esc(m.dataset)}. The threshold was tuned on the first group only; every number below comes from the second group — people the system had never seen during tuning. Enrollment audio and test audio never overlap.</p>
    <div class="kpis">
      <div class="kpi"><div class="v">${pct(mt.attribution_accuracy)}</div><div class="l">of meeting speech time got the right label (name, or "unknown" for guests)</div></div>
      <div class="kpi"><div class="v">${pct(mt.wrong_name_rate, 2)}</div><div class="l">of speech time shown under a <b>wrong</b> person's name</div></div>
      <div class="kpi"><div class="v">${pct(v.far_at_chosen, 2)}</div><div class="l">false accept rate per comparison at threshold ${m.chosen_threshold.toFixed(2)}</div></div>
      <div class="kpi"><div class="v">${pct(m.eer.rate, 2)}</div><div class="l">equal error rate (standard voice-biometrics metric) — 0 errors in ${(v.genuine_trials + v.impostor_trials).toLocaleString()} comparisons; a small, clean test set, real meetings will be worse</div></div>
    </div>
  </div>
  <div class="panel">
    <h2>Choosing the threshold</h2>
    <p class="sub">Each point compares one voice sample with one profile: ${v.genuine_trials.toLocaleString()} same-person and ${v.impostor_trials.toLocaleString()} different-person comparisons. We pick the lowest threshold that keeps false accepts ≤ ${pct(m.target_far, 1)}.</p>
    ${sweepChart(m.sweep, m.chosen_threshold, m.eer)}
  </div>
  <div class="panel">
    <h2>Test results</h2>
    <table>
      <tr><th>Test</th><th>Setup</th><th>Result</th></tr>
      <tr><td><b>Single utterance identification</b></td>
        <td class="small">${id.enrolled_speakers} enrolled + ${id.unknown_speakers} never-seen speakers, ${id.test_utterances} test clips (${id.clip_seconds})</td>
        <td class="small">${pct(id.correct_rate)} correct · ${pct(id.wrong_name_rate, 2)} wrong name · ${pct(id.known_unnamed_rate)} known but left unnamed</td></tr>
      <tr><td><b>Synthetic meetings</b></td>
        <td class="small">${mt.count} meetings, ${mt.total_minutes} min total, ${mt.speakers_per_meeting} speakers each incl. unenrolled guests</td>
        <td class="small">${pct(mt.attribution_accuracy)} of time correctly labelled · ${pct(mt.wrong_name_rate, 2)} wrong name · diarization error ${pct(mt.der)}</td></tr>
      <tr><td><b>Same meetings + noise</b></td>
        <td class="small">${mt.noisy_10db.count} meetings with background noise at 10 dB SNR</td>
        <td class="small">${pct(mt.noisy_10db.attribution_accuracy)} correctly labelled · ${pct(mt.noisy_10db.wrong_name_rate, 2)} wrong name · diarization error ${pct(mt.noisy_10db.der)}</td></tr>
      <tr><td><b>Tenant isolation</b></td>
        <td class="small">Same voices enrolled in customer A, meetings analysed as customer B</td>
        <td class="small">${mt.cross_tenant_names_leaked === 0 ? '<span class="pill ok">0 names leaked</span>' : `<span class="pill red">${mt.cross_tenant_names_leaked} leaked</span>`}</td></tr>
    </table>
    <p class="small muted" style="margin-top:12px">Models: ${esc(m.models)}</p>
  </div>
  ${m.model_comparison ? `<div class="panel">
    <h2>Choosing the voice model</h2>
    <p class="sub">Three open models benchmarked on the same data. ${esc(m.model_comparison.setup)}. Lower equal error rate is better; a bigger gap between same-person and different-person scores is better.</p>
    <table><tr><th>Model</th><th>Equal error rate</th><th>Different people (avg score)</th><th>Same person (avg score)</th></tr>
    ${m.model_comparison.models.map((x) => `<tr><td><b>${esc(x.model)}</b> ${x.chosen ? '<span class="pill ok">chosen</span>' : ""}</td>
      <td class="mono">${pct(x.eer, 2)}</td><td class="mono">${x.impostor_mean_score.toFixed(2)}</td><td class="mono">${x.genuine_mean_score.toFixed(2)}</td></tr>`).join("")}
    </table></div>` : ""}`;
}

// ---------- audit ----------
async function loadAudit() {
  const list = await api(`/api/t/${tenant}/audit`);
  if (!list.length) { $("#audit").innerHTML = `<div class="empty">Nothing logged yet.</div>`; return; }
  const pill = (e) => e.startsWith("consent") ? "ok" : e.includes("delet") ? "red" : "warn";
  $("#audit").innerHTML = `<table><thead><tr><th>Time (UTC)</th><th>Event</th><th>Details</th></tr></thead><tbody>${
    list.map((a) => {
      const { id, at, event, ...rest } = a;
      return `<tr><td class="mono">${esc(at.replace("T", " ").replace("+00:00", ""))}</td>
        <td><span class="pill ${pill(event)}">${esc(event.replaceAll("_", " "))}</span></td>
        <td class="small mono">${esc(Object.entries(rest).map(([k, v]) => `${k}=${v}`).join("  "))}</td></tr>`;
    }).join("")
  }</tbody></table>`;
}

function refresh() {
  const tab = document.querySelector("nav.tabs button.active").dataset.tab;
  if (tab === "team") loadProfiles();
  if (tab === "accuracy") loadAccuracy();
  if (tab === "audit") loadAudit();
}

(async () => {
  await initTenants();
  await initDemo();
  refresh();
  window.addEventListener("focus", refresh); // pick up enrollments made in the other tab
})();
