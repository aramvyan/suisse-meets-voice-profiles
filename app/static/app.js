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
  // Fragments under 3 s can never be named (too little speech for a
  // reliable fingerprint); keep them in the transcript, not as cards.
  const main = r.speakers.filter((s) => s.seconds >= 3);
  const fragments = r.speakers.length - main.length;
  $("#result-meta").textContent =
    `${fmtTime(r.duration)} of audio · ${main.length} voices found${fragments ? ` (+${fragments} short fragments)` : ""} · ${named} recognised · threshold ${r.threshold.toFixed(2)} · processed in ${r.processing_seconds}s on this machine`;
  const byCluster = Object.fromEntries(r.speakers.map((s) => [s.cluster, s]));
  const label = (s) => (view === "after" && s.name) ? s.name : s.anon;
  $("#speakers").innerHTML = main.map((s) => {
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
function sweepChart(sweep, chosen) {
  const W = 640, H = 260, L = 46, R = 16, T = 14, B = 36;
  const t0 = sweep[0].threshold, t1 = sweep.at(-1).threshold;
  const x = (t) => L + (t - t0) / (t1 - t0) * (W - L - R);
  const y = (v) => T + (1 - v) * (H - T - B);
  const line = (k) => sweep.map((p, i) => `${i ? "L" : "M"}${x(p.threshold).toFixed(1)},${y(p[k]).toFixed(1)}`).join("");
  const grid = [0, 0.25, 0.5, 0.75, 1].map((v) =>
    `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/>
     <text x="${L - 8}" y="${y(v) + 4}" font-size="11" text-anchor="end" fill="var(--muted)">${v * 100}%</text>`).join("");
  const ticks = sweep.filter((_, i) => i % 4 === 0).map((p) =>
    `<text x="${x(p.threshold)}" y="${H - 14}" font-size="11" text-anchor="middle" fill="var(--muted)">${p.threshold.toFixed(2)}</text>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Correct and wrong labels by threshold">
    ${grid}${ticks}
    <text x="${(W + L) / 2}" y="${H - 1}" font-size="11" text-anchor="middle" fill="var(--muted)">match threshold (cosine similarity)</text>
    <line x1="${x(chosen)}" x2="${x(chosen)}" y1="${T}" y2="${H - B}" stroke="var(--ink)" stroke-dasharray="4 3"/>
    <text x="${x(chosen) + 6}" y="${T + 12}" font-size="11" fill="var(--ink)">chosen ${chosen.toFixed(2)}</text>
    <path d="${line("correct")}" fill="none" stroke="#2b5fad" stroke-width="2.2"/>
    <path d="${line("wrong")}" fill="none" stroke="var(--accent)" stroke-width="2.2"/>
  </svg>
  <div class="row small" style="gap:18px;margin-top:4px">
    <span><b style="color:#2b5fad">━</b> Correct label (name, or unnamed for a guest)</span>
    <span><b style="color:var(--accent)">━</b> Wrong person's name</span>
  </div>`;
}

async function loadAccuracy() {
  let m;
  try { m = await api("/api/eval"); }
  catch { $("#accuracy").innerHTML = `<div class="panel"><div class="empty">No evaluation yet. Run <span class="mono">scripts/evaluate_ami.py</span>.</div></div>`; return; }
  const r = m.result, c = m.clips, iso = m.tenant_isolation;
  $("#accuracy").innerHTML = `
  <div class="panel">
    <h2>How we know it works</h2>
    <p class="sub">${esc(m.dataset)}.</p>
    <div class="kpis">
      <div class="kpi"><div class="v">${pct(r.correct)}</div><div class="l">of speech time got the right label (the name, or no name for a guest)</div></div>
      <div class="kpi"><div class="v">${pct(r.wrong, 2)}</div><div class="l">of speech time shown under a <b>wrong</b> person's name</div></div>
      <div class="kpi"><div class="v">${pct(r.missed)}</div><div class="l">of speech the diarization missed (no label at all)</div></div>
      <div class="kpi"><div class="v">${pct(c.eer, 1)}</div><div class="l">equal error rate on ${c.genuine.toLocaleString()} short real clips vs profiles</div></div>
    </div>
    <p class="small muted">${esc(m.limits)}</p>
  </div>
  <div class="panel">
    <h2>Choosing the threshold</h2>
    <p class="sub">Rule: the threshold with the most correct labels while wrong names stay at or below ${pct(m.max_wrong_rule, 0)} on the calibration group. The chart shows every threshold on the test groups.</p>
    ${sweepChart(m.test_sweep, m.chosen_threshold)}
  </div>
  <div class="panel">
    <h2>Per meeting</h2>
    <p class="sub">Every meeting is run 4 times, each time with a different person not enrolled (the guest), and averaged over the rotations where it was a test meeting. One average hides a lot, so here is each meeting.</p>
    <table><tr><th>Meeting</th><th>Length</th><th>Correct</th><th>Wrong name</th><th>Overlapped speech</th></tr>
    ${m.per_meeting.map((p) => `<tr><td class="mono">${esc(p.meeting)}</td><td>${p.minutes} min</td><td>${pct(p.correct)}</td><td>${pct(p.wrong, 2)}</td><td>${pct(p.overlap_share)}</td></tr>`).join("")}
    </table>
  </div>
  <div class="panel">
    <h2>Choosing the voice model</h2>
    <p class="sub">Three open models, same meetings, same cross-validation. None of them lists AMI in its training data.</p>
    <table><tr><th>Model</th><th>Correct</th><th>Wrong name</th><th>Clip equal error rate</th><th>Score: same / different person</th></tr>
    ${Object.entries(m.models).map(([name, x]) => `<tr><td><b>${esc(name)}</b> ${name === m.chosen_model ? '<span class="pill ok">chosen</span>' : ""}</td>
      <td>${pct(x.test.correct)}</td><td>${pct(x.test.wrong, 2)}</td><td>${pct(x.clips.eer, 1)}</td>
      <td class="mono">${x.clips.genuine_mean.toFixed(2)} / ${x.clips.impostor_mean.toFixed(2)}</td></tr>`).join("")}
    </table>
  </div>
  <div class="panel">
    <h2>Each group once as the calibration group</h2>
    <p class="sub">The model and threshold are chosen on one group and measured on the three others. This is done 4 times; the numbers above are pooled.</p>
    <table><tr><th>Calibration group</th><th>Model chosen</th><th>Threshold</th><th>Correct (other groups)</th><th>Wrong name</th></tr>
    ${m.folds.map((f) => `<tr><td class="mono">${esc(f.calibration_group)}</td><td>${esc(f.model)}</td><td class="mono">${f.threshold.toFixed(3)}</td><td>${pct(f.test.correct)}</td><td>${pct(f.test.wrong, 2)}</td></tr>`).join("")}
    </table>
  </div>
  <div class="panel">
    <h2>Other customer's people</h2>
    <p class="sub">Each meeting analysed by a customer that has the ${m.foreign_gallery.gallery_size} people of the other groups enrolled, but none of the speakers. Every name shown is a false accept.</p>
    <p>${m.foreign_gallery.names_shown === 0 ? '<span class="pill ok">0 names shown</span>' : `<span class="pill red">${m.foreign_gallery.names_shown} name${m.foreign_gallery.names_shown === 1 ? "" : "s"} shown</span>`}
      <span class="small muted">out of ${m.foreign_gallery.speakers_checked} speakers checked · profiles of customer A visible to customer B: ${m.foreign_gallery.profiles_visible_to_other_customer}</span></p>
  </div>`;
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
