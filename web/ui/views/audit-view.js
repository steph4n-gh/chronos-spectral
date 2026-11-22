// ============================================================================
// CHRONOS-SPECTRAL v3.5 // Isolated Lazy Physics Integrity Audit A/B Bench
// 2-Row x 8-Axis Comparison Matrix, Dual Click/Keyboard-to-Seek Severity Timeline,
// FIFO Job Serialization (Slot A & Slot B never silently abort each other),
// "Audit Loaded Clip -> Slot A", Live HUD Telemetry Mirror & Zero Idle Wasm Cost.
// ============================================================================

const AUDIT_AXES = Object.freeze([
  { id: 'trajectory', short: 'TRAJ', title: 'Axis 0: Trajectory Dynamics' },
  { id: 'collision', short: 'COLL', title: 'Axis 1: Collision Restitution' },
  { id: 'depth_parallax', short: 'DEPTH', title: 'Axis 2: Depth & Parallax' },
  { id: 'fluid_div', short: 'DIV', title: 'Axis 3: Fluid Divergence' },
  { id: 'fluid_momentum', short: 'N-S', title: 'Axis 4: Fluid Navier-Stokes' },
  { id: 'bg_shimmer', short: 'BG', title: 'Axis 5: Static Background Shimmer' },
  { id: 'temporal_cadence', short: 'CAD', title: 'Axis 6: Temporal Cadence' },
  { id: 'av_sync', short: 'AV', title: 'Axis 7: AV Impact Sync' },
]);

const AXIS_TITLE_BY_ID = Object.freeze(
  Object.fromEntries(AUDIT_AXES.map((a) => [a.id, a.title]))
);

export function initAuditBench({ getActiveVideoSource, onSeekToTimeSec, onSeekToFraction, onAuditEvent, onBayMeterUpdate }) {
  const card = document.getElementById('physicsAuditCard');
  if (!card) return null;

  const slotEl = { A: document.getElementById('auditSlotA'), B: document.getElementById('auditSlotB') };
  const metaEl = { A: document.getElementById('auditMetaA'), B: document.getElementById('auditMetaB') };
  const gradeEl = { A: document.getElementById('auditGradeA'), B: document.getElementById('auditGradeB') };
  const fileEl = { A: document.getElementById('auditFileA'), B: document.getElementById('auditFileB') };
  const deltaEl = document.getElementById('auditDeltaLine');
  const tlCanvas = document.getElementById('auditTimelineCanvas');
  const tlCtx = tlCanvas ? tlCanvas.getContext('2d') : null;

  const slotPills = { A: new Map(), B: new Map() };
  card.querySelectorAll('.audit-pill[data-axis]').forEach((p) => {
    const s = p.dataset.slot === 'B' ? 'B' : 'A';
    slotPills[s].set(p.dataset.axis, p);
  });

  let auditWorker = null;
  let auditWorkerReadyPromise = null;
  let liveMode = false;
  let nextJobId = 1;
  const activeJobId = { A: 0, B: 0 };
  const results = { A: null, B: null };
  let runningJob = null; // { slot, jobId, payload, resolve, reject }
  const jobQueue = [];
  let tlCursorFrac = 0;

  function emitAuditState(patch) {
    if (typeof onAuditEvent === 'function') onAuditEvent(patch);
  }

  function paintTimeline() {
    if (!tlCtx || !tlCanvas) return;
    const w = tlCanvas.width;
    const h = tlCanvas.height;
    tlCtx.fillStyle = '#000000';
    tlCtx.fillRect(0, 0, w, h);
    tlCtx.fillStyle = '#2a2f36';
    tlCtx.fillRect(0, Math.floor(h / 2), w, 1);

    const drawTrack = (rec, y0, bandH, baseColor) => {
      if (!rec || !Array.isArray(rec.timeline) || rec.timeline.length === 0) return;
      const tl = rec.timeline;
      const n = tl.length;
      const step = w / n;
      for (let i = 0; i < n; i++) {
        const sev = tl[i];
        if (sev >= 0.5) tlCtx.fillStyle = '#f0506a';
        else if (sev > 0.1) tlCtx.fillStyle = '#f2b23e';
        else tlCtx.fillStyle = baseColor;
        tlCtx.fillRect(Math.floor(i * step), y0 + 2, Math.max(2, Math.ceil(step)), bandH - 4);
      }
    };
    drawTrack(results.A, 0, Math.floor(h / 2), '#3ecf8e');
    drawTrack(results.B, Math.floor(h / 2) + 1, Math.floor(h / 2) - 1, '#3fc8e4');
  }

  function renderPillsForSlot(slot, rec) {
    const map = slotPills[slot];
    if (!map) return;
    if (!rec || !Array.isArray(rec.axes)) {
      for (const el of map.values()) {
        el.dataset.state = 'pending';
        el.textContent = '--';
      }
      return;
    }
    for (const ax of rec.axes) {
      const el = map.get(ax.id);
      if (!el) continue;
      let st = 'na';
      let label = 'NA';
      if (ax.applicability === 'eligible') {
        if (ax.verdict === 'violated') { st = 'fail'; label = 'FAIL'; }
        else if (ax.verdict === 'warn') { st = 'warn'; label = 'WARN'; }
        else { st = 'pass'; label = 'PASS'; }
      }
      el.dataset.state = st;
      el.textContent = label;
      const axisName = AXIS_TITLE_BY_ID[ax.id] || ax.id;
      el.title = `Slot ${slot} · ${axisName}: ${ax.applicability} (${ax.verdict})`;
    }
  }

  function renderSummaryDelta() {
    if (liveMode) return;
    const a = results.A;
    const b = results.B;
    const fmt = (r) => {
      if (!r || !r.verdict) return '--';
      const iv = r.verdict.integrity === null ? 'NA' : r.verdict.integrity.toFixed(3);
      const rw = r.rlvr && r.rlvr.reward_milli !== null ? `, reward ${r.rlvr.reward_milli}` : '';
      return `${r.verdict.grade} (I=${iv}, ${r.verdict.violations} flags${rw})`;
    };
    if (a && b && a.verdict.integrity !== null && b.verdict.integrity !== null) {
      const d = b.verdict.integrity - a.verdict.integrity;
      deltaEl.textContent = `Delta Integrity (B - A): ${d >= 0 ? '+' : ''}${d.toFixed(3)} · A: ${fmt(a)} · B: ${fmt(b)}`;
      deltaEl.title = deltaEl.textContent;
      if (typeof onBayMeterUpdate === 'function') {
        onBayMeterUpdate(`A:${a.verdict.grade} B:${b.verdict.grade} d=${d >= 0 ? '+' : ''}${d.toFixed(2)}`);
      }
    } else if (a || b) {
      deltaEl.textContent = `A: ${fmt(a)} · B: ${fmt(b)} · Local Wasm32 SIMD128 (0 bytes uploaded)`;
      deltaEl.title = deltaEl.textContent;
      const active = a || b;
      if (typeof onBayMeterUpdate === 'function' && active?.verdict) {
        onBayMeterUpdate(`${a ? 'A' : 'B'}:${active.verdict.grade} I=${active.verdict.integrity === null ? 'NA' : active.verdict.integrity.toFixed(2)}`);
      }
    }
  }

  function pumpJobQueue() {
    if (runningJob || jobQueue.length === 0) return;
    const next = jobQueue.shift();
    runningJob = next;
    if (gradeEl[next.slot]) gradeEl[next.slot].textContent = 'INITIALIZING WASM...';
    emitAuditState({ slot: next.slot, jobId: next.jobId, patch: { phase: 'RUNNING', clipName: next.payload.clipName || '' } });
    ensureAuditWorker()
      .then((w) => {
        if (runningJob !== next) return;
        w.postMessage({ type: 'audit_run', slot: next.slot, jobId: next.jobId, ...next.payload });
      })
      .catch((err) => {
        if (runningJob === next) runningJob = null;
        if (gradeEl[next.slot]) gradeEl[next.slot].textContent = `ERROR · ${err.message || err}`;
        emitAuditState({ slot: next.slot, jobId: next.jobId, patch: { phase: 'ERROR' } });
        next.reject(err);
        pumpJobQueue();
      });
  }

  function ensureAuditWorker() {
    if (auditWorkerReadyPromise) return auditWorkerReadyPromise;
    emitAuditState({ worker: 'STARTING' });
    const w = new Worker('./worker.js?v=20261122v35');
    auditWorker = w;
    auditWorkerReadyPromise = new Promise((resolve, reject) => {
      let settledInit = false;
      const failInit = (msg) => {
        if (auditWorker === w) {
          auditWorker = null;
          auditWorkerReadyPromise = null;
        }
        emitAuditState({ worker: 'FAILED' });
        if (!settledInit) {
          settledInit = true;
          reject(new Error(msg));
        } else if (runningJob) {
          const cur = runningJob;
          runningJob = null;
          if (gradeEl[cur.slot]) gradeEl[cur.slot].textContent = `ERROR · ${msg}`;
          emitAuditState({ slot: cur.slot, jobId: cur.jobId, patch: { phase: 'ERROR' } });
          cur.reject(new Error(msg));
          pumpJobQueue();
        }
      };

      w.addEventListener('error', (ev) => {
        failInit(ev?.message || 'Audit worker crashed');
      });

      const onMsg = (ev) => {
        const m = ev.data || {};
        if (m.type === 'audit_ready') {
          settledInit = true;
          emitAuditState({ worker: 'READY' });
          resolve(w);
        } else if (m.type === 'audit_progress') {
          const s = m.slot === 'B' ? 'B' : 'A';
          if (m.jobId && m.jobId !== activeJobId[s]) return;
          if (gradeEl[s]) gradeEl[s].textContent = `AUDITING · ${m.frame}/${m.total}f`;
        } else if (m.type === 'audit_done') {
          const s = m.slot === 'B' ? 'B' : 'A';
          const jid = m.jobId ?? activeJobId[s];
          const cur = runningJob && runningJob.slot === s && runningJob.jobId === jid ? runningJob : null;
          if (cur) runningJob = null;
          if (jid >= activeJobId[s]) {
            const rec = m.record;
            results[s] = rec;
            if (metaEl[s]) metaEl[s].textContent = `${rec.clip.path} · ${rec.ingest.frames_committed}f (${rec.perf.wall_s}s)`;
            if (gradeEl[s]) {
              const iv = rec.verdict.integrity === null ? 'NA' : rec.verdict.integrity.toFixed(3);
              gradeEl[s].textContent = `${rec.verdict.grade} · I=${iv} · ${rec.verdict.violations} flags · ${rec.verdict.applicable_axes}/8 axes`;
            }
            renderPillsForSlot(s, rec);
            paintTimeline();
            renderSummaryDelta();
            emitAuditState({
              slot: s,
              jobId: jid,
              patch: { phase: 'DONE', clipName: rec.clip.path, grade: rec.verdict.grade, integrity: rec.verdict.integrity, record: rec },
            });
          }
          if (cur) cur.resolve(m.record);
          pumpJobQueue();
        } else if (m.type === 'audit_error') {
          const s = m.slot === 'B' ? 'B' : 'A';
          const jid = m.jobId ?? activeJobId[s];
          const cur = runningJob && runningJob.slot === s && runningJob.jobId === jid ? runningJob : null;
          if (cur) runningJob = null;
          if (gradeEl[s]) gradeEl[s].textContent = `ERROR · ${m.message}`;
          emitAuditState({ slot: s, jobId: jid, patch: { phase: 'ERROR' } });
          if (cur) {
            cur.reject(new Error(m.message || 'audit_error'));
            pumpJobQueue();
          } else if (!settledInit) {
            failInit(m.message || 'audit_error');
          }
        }
      };
      w.addEventListener('message', onMsg);
      w.postMessage({ type: 'audit_init', wasmUrl: './chronos_spectral.wasm?v=20261122v35' });
    });
    return auditWorkerReadyPromise;
  }

  function dispatchAudit(slot, payload) {
    const s = slot === 'B' ? 'B' : 'A';
    const jobId = nextJobId++;
    activeJobId[s] = jobId;
    // Drop any older queued (not-yet-running) job for the same slot so rapid clicks stay snappy.
    for (let i = jobQueue.length - 1; i >= 0; i--) {
      if (jobQueue[i].slot === s) {
        jobQueue[i].reject(new Error('superseded_by_newer_slot_job'));
        jobQueue.splice(i, 1);
      }
    }
    if (gradeEl[s]) gradeEl[s].textContent = runningJob ? 'QUEUED...' : 'INITIALIZING WASM...';
    return new Promise((resolve, reject) => {
      jobQueue.push({ slot: s, jobId, payload, resolve, reject });
      pumpJobQueue();
    });
  }

  ['A', 'B'].forEach((s) => {
    const btn = document.getElementById(`btnAuditChoose${s}`);
    if (btn && fileEl[s]) {
      btn.addEventListener('click', () => fileEl[s].click());
      fileEl[s].addEventListener('change', () => {
        const f = fileEl[s].files && fileEl[s].files[0];
        if (f) dispatchAudit(s, { file: f, clipName: f.name, profile: 'quick' }).catch(() => {});
        fileEl[s].value = '';
      });
    }
    const box = slotEl[s];
    if (box) {
      box.addEventListener('dragover', (ev) => { ev.preventDefault(); box.dataset.dropActive = 'true'; });
      box.addEventListener('dragleave', () => { delete box.dataset.dropActive; });
      box.addEventListener('drop', (ev) => {
        ev.preventDefault();
        delete box.dataset.dropActive;
        const f = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
        if (f) dispatchAudit(s, { file: f, clipName: f.name, profile: 'quick' }).catch(() => {});
      });
    }
  });

  const btnRef = document.getElementById('btnAuditSampleRef');
  if (btnRef) {
    btnRef.addEventListener('click', () => {
      dispatchAudit('A', { url: './samples/juggling_ballistic.mp4', clipName: 'juggling_ballistic.mp4', profile: 'quick', maxFrames: 75 }).catch(() => {});
    });
  }
  const btnStut = document.getElementById('btnAuditSampleStutter');
  if (btnStut) {
    btnStut.addEventListener('click', () => {
      dispatchAudit('B', { syntheticKind: 'stutter_freeze', clipName: 'synthetic_stutter_freeze', profile: 'quick', maxFrames: 48 }).catch(() => {});
    });
  }
  const btnCurrent = document.getElementById('btnAuditCurrent');
  if (btnCurrent) {
    btnCurrent.addEventListener('click', () => {
      const src = typeof getActiveVideoSource === 'function' ? getActiveVideoSource() : null;
      if (!src) return;
      if (src.file) {
        dispatchAudit('A', { file: src.file, clipName: src.file.name, profile: 'quick' }).catch(() => {});
      } else if (src.url) {
        const name = src.url.split('/').pop() || src.url;
        dispatchAudit('A', { url: src.url, clipName: name, profile: 'quick', maxFrames: 90 }).catch(() => {});
      }
    });
  }
  const btnLive = document.getElementById('btnAuditLiveToggle');
  if (btnLive) {
    btnLive.addEventListener('click', () => {
      liveMode = !liveMode;
      btnLive.setAttribute('aria-pressed', String(liveMode));
      btnLive.textContent = liveMode ? 'Live: ON' : 'Live: OFF';
      emitAuditState({ liveMirror: liveMode });
      if (liveMode) {
        deltaEl.textContent = 'Live mirror ON · Reading HUD worker telemetry (zero extra Wasm instance)';
      } else {
        renderSummaryDelta();
      }
    });
  }
  const btnExport = document.getElementById('btnAuditExportJson');
  if (btnExport) {
    btnExport.addEventListener('click', () => {
      const payload = { schema: 'chronos-audit-ab/1', slotA: results.A, slotB: results.B };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'chronos-audit-report.json';
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
  }

  function seekAuditedSlotA(frac) {
    const clamped = Math.max(0, Math.min(1, frac));
    tlCursorFrac = clamped;
    const rec = results.A;
    if (rec && rec.ingest && rec.clip && typeof onSeekToTimeSec === 'function') {
      const fps = Number(rec.clip.fps) > 1 ? Number(rec.clip.fps) : 30;
      const auditedDurSec = (Number(rec.ingest.frames_committed) || 0) / fps;
      if (auditedDurSec > 0) {
        onSeekToTimeSec(clamped * auditedDurSec);
        return;
      }
    }
    if (typeof onSeekToFraction === 'function') onSeekToFraction(clamped);
  }

  // Click upper band (Slot A) or use Left/Right arrows when focused to seek loaded video
  if (tlCanvas) {
    tlCanvas.tabIndex = 0;
    tlCanvas.addEventListener('click', (ev) => {
      const rect = tlCanvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      // Ignore lower-half clicks when Slot B is an independent synthetic/comparison clip
      if (results.B && ev.clientY - rect.top > rect.height / 2) return;
      seekAuditedSlotA((ev.clientX - rect.left) / rect.width);
    });
    tlCanvas.addEventListener('keydown', (ev) => {
      if (ev.key === 'ArrowRight') {
        ev.preventDefault();
        seekAuditedSlotA(tlCursorFrac + 0.05);
      } else if (ev.key === 'ArrowLeft') {
        ev.preventDefault();
        seekAuditedSlotA(tlCursorFrac - 0.05);
      }
    });
  }

  paintTimeline();

  function updateLiveMirror(summaryText) {
    if (!liveMode || !deltaEl || !summaryText) return;
    if (deltaEl.textContent !== summaryText) deltaEl.textContent = summaryText;
  }

  const probe = {
    workerReady: () => Boolean(auditWorker),
    updateLiveMirror,
    getState: () => {
      const primarySlot = results.B ? 'B' : 'A';
      const primaryMap = slotPills[primarySlot];
      return {
        workerReady: Boolean(auditWorker),
        liveMode,
        slotA: results.A,
        slotB: results.B,
        pillStates: Object.fromEntries(Array.from(primaryMap.entries()).map(([k, el]) => [k, el.dataset.state || 'pending'])),
        slotAPills: Object.fromEntries(Array.from(slotPills.A.entries()).map(([k, el]) => [k, el.dataset.state || 'pending'])),
        slotBPills: Object.fromEntries(Array.from(slotPills.B.entries()).map(([k, el]) => [k, el.dataset.state || 'pending'])),
      };
    },
    runUrl: (slot, url, opts = {}) => dispatchAudit(slot, { url, clipName: opts.clipName || url, ...opts }),
    runSynthetic: (slot, kind, opts = {}) => dispatchAudit(slot, { syntheticKind: kind, clipName: opts.clipName || kind, ...opts }),
    axes: AUDIT_AXES,
  };
  window.__chronosAuditProbe = probe;
  return probe;
}
