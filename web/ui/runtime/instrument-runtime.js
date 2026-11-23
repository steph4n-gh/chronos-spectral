// ============================================================================
// CHRONOS-SPECTRAL v3.5 // Canonical Instrument Runtime
// Extracted & upgraded from monolithic index.html script into a clean ES module
// governed by `fsm.js`, `constraints.js`, `reasons.js`, and `recipes/registry.js`.
// ============================================================================

import {
  BOS_OWNERS,
  computeBosOwner,
  createInitialInstrumentState,
  projectLegacyUiState,
  reduceInstrumentState,
} from '../core/fsm.js';
import {
  bosOwnerDisplayName,
  deriveControlStates,
  planIntent,
} from '../core/constraints.js';
import {
  BOS_REASON_BITS,
  DEPTH_REASON_BITS,
  ELASTO_REASON_BITS,
  RS_REASON_BITS,
  WIND_REASON_BITS,
  formatReasonSummary,
} from '../core/reasons.js';
import { RECIPES, RECIPE_BY_KEY, RECIPE_BY_URL } from '../recipes/registry.js';
import { createGatesView } from '../views/gates-view.js';
import { createBenchShell } from '../views/bench-shell.js';
import { initAuditBench } from '../views/audit-view.js';

export function initInstrumentRuntime() {
  const hudEl = document.getElementById('hudCanvas');
  const pipEl = document.getElementById('pipCanvas');
  const persistenceEl = document.getElementById('persistenceCanvas');
  const hilbertEl = document.getElementById('hilbertCanvas');
  const spectralEl = document.getElementById('spectralCanvas');
  const physicsPhaseEl = document.getElementById('physicsPhaseCanvas');
  const rosterSummaryEl = document.getElementById('rosterSummaryMeta');
  const rosterReacqEl = document.getElementById('rosterReacqVal');
  const rosterHoldChipEl = document.getElementById('rosterHoldChip');
  const rosterHoldEl = document.getElementById('rosterHoldVal');

  const rosterCardRefs = Array.from(document.querySelectorAll('.roster-card')).map((card, idx) => ({
    card,
    slotIdx: idx,
    canvas: card.querySelector('.slot-pip-canvas'),
    pinBtn: card.querySelector('.roster-pin-btn'),
    swatch: card.querySelector('.roster-swatch'),
    slotIdEl: card.querySelector('.roster-slot-id'),
    badgeEl: card.querySelector('.roster-state-badge'),
    lblSpd: card.querySelector('.rs-lbl-spd'),
    valSpd: card.querySelector('.rs-val-spd'),
    lblAng: card.querySelector('.rs-lbl-ang'),
    valAng: card.querySelector('.rs-val-ang'),
    valAge: card.querySelector('.rs-val-age'),
    valId: card.querySelector('.rs-val-id'),
    prevTid: -1,
    prevState: '',
    prevPinned: false,
    lastDomMs: 0,
  }));

  const hudOffscreen = hudEl.transferControlToOffscreen();
  const pipOffscreen = pipEl.transferControlToOffscreen();
  const persistenceOffscreen = persistenceEl.transferControlToOffscreen();
  const hilbertOffscreen = hilbertEl.transferControlToOffscreen();
  const spectralOffscreen = spectralEl.transferControlToOffscreen();
  const physicsPhaseOffscreen = physicsPhaseEl ? physicsPhaseEl.transferControlToOffscreen() : null;
  const slotPipOffscreens = rosterCardRefs.map((r) => r.canvas.transferControlToOffscreen());

  const bootQuery = new URLSearchParams(location.search);
  const bootPresetKey = bootQuery.get('preset') || bootQuery.get('recipe') || '';
  const hasValidBootPreset = Boolean(bootPresetKey && RECIPE_BY_KEY[bootPresetKey]);
  const isInitialDeckSurface = bootQuery.get('view') === 'deck';
  const bootIdleBench = !isInitialDeckSurface && !hasValidBootPreset;
  let isStaticHostMode = document.body.dataset.staticHost === 'true';
  let initialPresetBooted = false;
  let workerKernelReady = false;
  let pendingHeroSceneBeforeReady = hasValidBootPreset ? bootPresetKey : null;

  let instrumentState = createInitialInstrumentState({
    sourceKind: isInitialDeckSurface ? 'TESTBENCH' : 'IDLE',
    resMode: isInitialDeckSurface ? 0 : 1,
    welcomeVisible: bootIdleBench,
    hostCaps: {
      staticHost: isStaticHostMode,
      relayAvailable: !isStaticHostMode,
      webCodecs: typeof VideoDecoder === 'function' && typeof EncodedVideoChunk === 'function',
      webcamCapable: Boolean(navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function'),
      displayCaptureCapable: Boolean(navigator.mediaDevices && typeof navigator.mediaDevices.getDisplayMedia === 'function'),
      webAudioCapable: Boolean(window.AudioContext || window.webkitAudioContext),
      fullscreenCapable: Boolean(document.documentElement && typeof document.documentElement.requestFullscreen === 'function'),
    },
  });
  let gatesView = null;
  let benchShell = null;
  let lastGatesRenderMs = 0;

  function renderInstrumentViews(force = false) {
    const now = performance.now();
    if (!force && now - lastGatesRenderMs < 110) return;
    lastGatesRenderMs = now;
    if (benchShell) benchShell.render(instrumentState);
    if (gatesView) gatesView.render(instrumentState);
  }

  function dispatchInstrument(event, forceRender = true) {
    const next = reduceInstrumentState(instrumentState, event);
    if (next !== instrumentState) {
      instrumentState = next;
      const proj = projectLegacyUiState(instrumentState);
      if (typeof deckUiState !== 'undefined' && deckUiState) {
        window.__chronosUiState = Object.freeze({
          ...deckUiState,
          schema: proj.schema,
          bosOwner: proj.bosOwner,
          recipe: proj.recipe,
          ui: proj.ui,
          sourceDetail: proj.sourceDetail,
        });
      } else {
        window.__chronosUiState = proj;
      }
      renderInstrumentViews(forceRender);
    }
    return instrumentState;
  }

  const worker = new Worker('./worker.js?v=20261122v35');
  worker.addEventListener('error', (err) => {
    console.error('Worker uncaught error:', err.message || err);
    dispatchInstrument({ type: 'WORKER_FAULT' }, true);
    if (dom.pillKernel && dom.pillKernel.textContent.includes('BOOTING')) {
      dom.pillKernel.textContent = '✕ KERNEL WORKER ERROR';
    }
  });
  worker.addEventListener('messageerror', (err) => {
    console.error('Worker messageerror:', err);
  });
  const initTransfers = [hudOffscreen, pipOffscreen, persistenceOffscreen, hilbertOffscreen, spectralOffscreen];
  if (physicsPhaseOffscreen) initTransfers.push(physicsPhaseOffscreen);
  initTransfers.push(...slotPipOffscreens);
  worker.postMessage(
    {
      type: 'init',
      wasmUrl: './chronos_spectral.wasm?v=20261122v35',
      resMode: isInitialDeckSurface ? 0 : 1,
      startIdle: bootIdleBench,
      showcaseMinimalScopes: false,
      scopeVisibilityMask: 0x09,
      rawWipeRatio: 0.0,
      showHeroTargets: true,
      hudCanvas: hudOffscreen,
      pipCanvas: pipOffscreen,
      persistenceCanvas: persistenceOffscreen,
      hilbertCanvas: hilbertOffscreen,
      spectralCanvas: spectralOffscreen,
      physicsPhaseCanvas: physicsPhaseOffscreen,
      slotPipCanvases: slotPipOffscreens,
    },
    initTransfers
  );

  rosterCardRefs.forEach((ref) => {
    ref.pinBtn.addEventListener('click', () => {
      if (ref.pinBtn.getAttribute('aria-disabled') === 'true') return;
      const tid = Number(ref.card.dataset.tid) || 0;
      if (tid > 0) {
        worker.postMessage({ type: 'pin_target', trackId: tid });
      }
    });
  });

  function setTextIfChanged(el, next) {
    if (el && el.textContent !== next) el.textContent = next;
  }

  function updateSlotRosterUi(slotRoster) {
    if (!Array.isArray(slotRoster)) return;
    const nowMs = performance.now();
    let totalReacq = 0;
    let holdCount = 0;

    for (let i = 0; i < rosterCardRefs.length && i < slotRoster.length; i++) {
      const ref = rosterCardRefs[i];
      const s = slotRoster[i];
      if (!s) continue;
      if (s.state !== 'IDLE') totalReacq += s.reacq || 0;
      if (s.state === 'HOLD') holdCount++;

      const stateChanged =
        ref.prevTid !== s.tid ||
        ref.prevState !== s.state ||
        ref.prevPinned !== Boolean(s.isPinned);

      if (stateChanged) {
        ref.prevTid = s.tid;
        ref.prevState = s.state;
        ref.prevPinned = Boolean(s.isPinned);
        ref.card.dataset.state = s.state;
        ref.card.dataset.tid = String(s.tid || 0);
        ref.card.dataset.pinned = s.isPinned ? '1' : '0';

        const isOperable = s.tid > 0 && s.state !== 'IDLE';
        ref.pinBtn.setAttribute('aria-disabled', String(!isOperable));
        ref.pinBtn.setAttribute('aria-pressed', String(Boolean(s.isPinned)));
        const slotLabel = String(i + 1).padStart(2, '0');
        ref.pinBtn.setAttribute(
          'aria-label',
          isOperable
            ? `${s.isPinned ? 'Unpin' : 'Pin focus on'} Slot ${slotLabel} Target T${s.tid}`
            : `Slot ${slotLabel} standby`
        );
        ref.swatch.style.backgroundColor = s.state === 'IDLE' ? '#4c6180' : s.color;
        setTextIfChanged(ref.slotIdEl, s.tid > 0 ? `S${i + 1} · T${s.tid}` : `S${i + 1} · --`);
        const badgeTxt = s.isPinned && s.state !== 'PINNED' ? `PIN·${s.state.slice(0, 3)}` : s.state;
        setTextIfChanged(ref.badgeEl, badgeTxt);
      }

      // Throttle numeric metric mutations to ~4.5 Hz (220 ms) unless slot state/ID just changed
      if (!stateChanged && nowMs - ref.lastDomMs < 220) continue;
      ref.lastDomMs = nowMs;

      if (s.state === 'IDLE') {
        setTextIfChanged(ref.lblSpd, 'SPD');
        setTextIfChanged(ref.valSpd, `-- ${s.speedUnit}`);
        setTextIfChanged(ref.lblAng, 'HDG · ω');
        setTextIfChanged(ref.valAng, '--°');
        setTextIfChanged(ref.valAge, '--');
        setTextIfChanged(ref.valId, '-- · 0');
      } else {
        setTextIfChanged(ref.lblSpd, s.freqHz >= 0.35 ? 'SPD · HZ' : 'SPD');
        const spdTxt = s.speedUnit === 'OOP'
          ? 'OUT-PLANE'
          : s.speedUnit === 'm/s'
          ? `${s.speed.toFixed(1)} m/s`
          : `${Math.round(s.speed)} px/s`;
        setTextIfChanged(
          ref.valSpd,
          s.freqHz >= 0.35 && s.speedUnit !== 'OOP'
            ? `${spdTxt}·${s.freqHz.toFixed(1)}Hz`
            : spdTxt
        );

        if (s.isPivot) {
          setTextIfChanged(ref.lblAng, 'SWG · R');
          const swSign = s.angleDeg >= 0 ? '+' : '';
          const rTxt = s.rateUnit === 'm' ? `${s.rateOrRad.toFixed(2)}m` : `${Math.round(s.rateOrRad)}px`;
          setTextIfChanged(ref.valAng, `${swSign}${s.angleDeg.toFixed(1)}° ${rTxt}`);
        } else {
          setTextIfChanged(ref.lblAng, 'HDG · ω');
          const hSign = s.angleDeg >= 0 ? '+' : '';
          const wInt = Math.max(-999, Math.min(999, Math.round(s.rateOrRad)));
          const wSign = wInt >= 0 ? '+' : '';
          setTextIfChanged(ref.valAng, `${hSign}${Math.round(s.angleDeg)}° ${wSign}${wInt}°/s`);
        }

        const ageTxt = s.ageSec >= 10.0 ? `${s.ageSec.toFixed(0)}s` : `${s.ageSec.toFixed(1)}s`;
        setTextIfChanged(
          ref.valAge,
          s.coastFrames > 0 ? `${ageTxt} (${s.coastFrames}f)` : ageTxt
        );
        setTextIfChanged(ref.valId, `v₂:${s.padicVal} · ↻${s.reacq}`);
      }
    }

    if (rosterSummaryEl && nowMs - (rosterSummaryEl._lastMs || 0) >= 220) {
      rosterSummaryEl._lastMs = nowMs;
      if (rosterReacqEl) setTextIfChanged(rosterReacqEl, String(totalReacq));
      if (rosterHoldEl) setTextIfChanged(rosterHoldEl, String(holdCount));
      if (rosterHoldChipEl) {
        const nextActive = holdCount > 0 ? '1' : '0';
        if (rosterHoldChipEl.dataset.active !== nextActive) {
          rosterHoldChipEl.dataset.active = nextActive;
        }
      }
    }
  }

  // ========================================================================
  // Polyphonic Delayed-View Synchronized WebAudio Synthesizer
  // ========================================================================
  let audioCtx = null;
  let audioEnabled = false;
  let noiseBuffer = null;

  function getAudioOutputLatencySec() {
    if (!audioCtx) return 0.0;
    return (audioCtx.outputLatency || 0.0) + (audioCtx.baseLatency || 0.0);
  }

  function syncAudioLatencyToWorker() {
    worker.postMessage({
      type: 'params',
      audioOutLatencySec: audioEnabled ? getAudioOutputLatencySec() : 0.0,
    });
  }

  function ensureAudioContext() {
    if (!audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AC({ latencyHint: 'interactive' });
      noiseBuffer = audioCtx.createBuffer(1, Math.floor(audioCtx.sampleRate * 0.25), audioCtx.sampleRate);
      const data = noiseBuffer.getChannelData(0);
      for (let i = 0; i < data.length; i++) {
        data[i] = Math.random() * 2.0 - 1.0;
      }
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    syncAudioLatencyToWorker();
  }

  function playPhaseStrobePulse(resonant, tremorHz, amp) {
    if (!audioEnabled || !audioCtx) return;
    const t0 = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = resonant ? 'triangle' : 'sine';
    const baseFreq = resonant ? 528.0 : 264.0;
    osc.frequency.setValueAtTime(baseFreq + Math.min(80, tremorHz * 4), t0);
    osc.frequency.exponentialRampToValueAtTime(baseFreq * 0.5, t0 + 0.055);

    const peak = Math.min(0.18, 0.05 + amp * 0.08);
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.linearRampToValueAtTime(peak, t0 + 0.004);
    gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.06);

    osc.connect(gain).connect(audioCtx.destination);
    osc.start(t0);
    osc.stop(t0 + 0.065);
  }

  function playDelayedSyncPercussion(zone, leadSec, velocity, sentEpochMs) {
    if (!audioEnabled || !audioCtx) return;
    const nowEpochMs = performance.timeOrigin + performance.now();
    const msgDelaySec = typeof sentEpochMs === 'number'
      ? Math.max(0.0, Math.min(0.10, (nowEpochMs - sentEpochMs) * 0.001))
      : 0.0;
    const netLeadSec = Math.max(0.0, Math.min(0.24, leadSec - msgDelaySec - getAudioOutputLatencySec()));
    const when = audioCtx.currentTime + netLeadSec;
    const vel = Math.max(0.2, Math.min(1.0, velocity));

    if (zone === 0) {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(155.0, when);
      osc.frequency.exponentialRampToValueAtTime(42.0, when + 0.14);
      gain.gain.setValueAtTime(0.38 * vel, when);
      gain.gain.exponentialRampToValueAtTime(0.001, when + 0.22);
      osc.connect(gain).connect(audioCtx.destination);
      osc.start(when);
      osc.stop(when + 0.23);
    } else if (zone === 1) {
      const noise = audioCtx.createBufferSource();
      noise.buffer = noiseBuffer;
      const filter = audioCtx.createBiquadFilter();
      filter.type = 'highpass';
      filter.frequency.setValueAtTime(1100, when);
      const nGain = audioCtx.createGain();
      nGain.gain.setValueAtTime(0.26 * vel, when);
      nGain.gain.exponentialRampToValueAtTime(0.001, when + 0.13);
      noise.connect(filter).connect(nGain).connect(audioCtx.destination);
      noise.start(when);
      noise.stop(when + 0.14);

      const body = audioCtx.createOscillator();
      const bGain = audioCtx.createGain();
      body.type = 'triangle';
      body.frequency.setValueAtTime(210, when);
      body.frequency.exponentialRampToValueAtTime(115, when + 0.07);
      bGain.gain.setValueAtTime(0.22 * vel, when);
      bGain.gain.exponentialRampToValueAtTime(0.001, when + 0.08);
      body.connect(bGain).connect(audioCtx.destination);
      body.start(when);
      body.stop(when + 0.09);
    } else if (zone === 2) {
      const noise = audioCtx.createBufferSource();
      noise.buffer = noiseBuffer;
      const hp = audioCtx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.setValueAtTime(6200, when);
      const gain = audioCtx.createGain();
      gain.gain.setValueAtTime(0.22 * vel, when);
      gain.gain.exponentialRampToValueAtTime(0.001, when + 0.065);
      noise.connect(hp).connect(gain).connect(audioCtx.destination);
      noise.start(when);
      noise.stop(when + 0.07);
    } else {
      const carrier = audioCtx.createOscillator();
      const mod = audioCtx.createOscillator();
      const modGain = audioCtx.createGain();
      const outGain = audioCtx.createGain();
      carrier.type = 'sine';
      carrier.frequency.setValueAtTime(330, when);
      carrier.frequency.exponentialRampToValueAtTime(165, when + 0.16);
      mod.frequency.setValueAtTime(495, when);
      modGain.gain.setValueAtTime(140 * vel, when);
      modGain.gain.exponentialRampToValueAtTime(1, when + 0.12);
      outGain.gain.setValueAtTime(0.28 * vel, when);
      outGain.gain.exponentialRampToValueAtTime(0.001, when + 0.18);
      mod.connect(modGain).connect(carrier.frequency);
      carrier.connect(outGain).connect(audioCtx.destination);
      mod.start(when);
      carrier.start(when);
      mod.stop(when + 0.19);
      carrier.stop(when + 0.19);
    }
  }

  // ========================================================================
  // Epoch-Tagged Frame Pump, Worker frame_ack Backpressure & 3-Tier Live Suite
  // ========================================================================
  const videoEl = document.getElementById('hiddenVideoSource');
  const liveDialog = document.getElementById('liveStreamDialog');
  let activeMediaStream = null;
  let currentObjectUrl = null;
  let activeMseAbort = null;
  let activeMseMediaSource = null;
  let activeMseRetryTimer = 0;
  let activeLiveRetryCount = 0;
  let activeLiveConfig = null;
  let videoPumpActive = false;
  let activeSourceMode = 'testbench'; // 'testbench' | 'webcam' | 'video' | 'live'
  let activeVideoFps = 30.0;
  let lastMediaTime = -1;
  let lastPresentedFrames = -1;
  let pendingVideoGapMask = 0;
  let lastPumpPerfMs = 0;
  let pumpStartPerfMs = 0;
  let activeResMode = 0;
  let activeFrameW = 320;
  let activeFrameH = 180;
  let sourceEpoch = 1;
  let frameSeq = 0;
  let workerFrameBusy = false;
  let queuedPendingFrame = null;
  let pumpDroppedFrames = 0;
  let bitmapCaptureBusy = false;
  let bitmapCaptureToken = 0;
  let activeCaptureToken = 0;
  const fallbackCapCanvas = document.createElement('canvas');
  const fallbackCapCtx = fallbackCapCanvas.getContext('2d', { willReadFrequently: true });

  const dom = {
    pillKernel: document.getElementById('pillKernel'),
    pillArena: document.getElementById('pillArena'),
    pillPerf: document.getElementById('pillPerf'),
    pillFps: document.getElementById('pillFps'),
    pillStreamHealth: document.getElementById('pillStreamHealth'),
    inputResMeta: document.getElementById('inputResMeta'),
    hudDockSlot: document.getElementById('hudDockSlot'),
    hudPanel: document.getElementById('hudPanel'),
    btnHudHeroToggle: document.getElementById('btnHudHeroToggle'),
    btnHudFullscreen: document.getElementById('btnHudFullscreen'),
    chkHeroTargets: document.getElementById('chkHeroTargets'),
    chkKalman: document.getElementById('chkKalman'),
    chkAllOverlays: document.getElementById('chkAllOverlays'),
    chkEcaOverlay: document.getElementById('chkEcaOverlay'),
    hudSourceLabel: document.getElementById('hudSourceLabel'),
    mortonReadout: document.getElementById('mortonReadout'),
    lblPadicSwaps: document.getElementById('lblPadicSwaps'),
    kpiPadicSwaps: document.getElementById('kpiPadicSwaps'),
    kpiKalmanSwaps: document.getElementById('kpiKalmanSwaps'),
    lblClutterDb: document.getElementById('lblClutterDb'),
    kpiClutterDb: document.getElementById('kpiClutterDb'),
    lblWasmMs: document.getElementById('lblWasmMs'),
    kpiWasmMs: document.getElementById('kpiWasmMs'),
    kpiTargets: document.getElementById('kpiTargets'),
    lblCollisions: document.getElementById('lblCollisions'),
    kpiCollisions: document.getElementById('kpiCollisions'),
    persistSummaryMeta: document.getElementById('persistSummaryMeta'),
    kpiPersistSpan: document.getElementById('kpiPersistSpan'),
    kpiPersistActivePeak: document.getElementById('kpiPersistActivePeak'),
    kpiPersistUnique: document.getElementById('kpiPersistUnique'),
    kpiPersistRelocks: document.getElementById('kpiPersistRelocks'),
    kpiPersistLongest: document.getElementById('kpiPersistLongest'),
    kpiPersistSustained: document.getElementById('kpiPersistSustained'),
    hilbertMeta: document.getElementById('hilbertMeta'),
    spectralMeta: document.getElementById('spectralMeta'),
    fistaGapMeta: document.getElementById('fistaGapMeta'),
    physicsIdPanel: document.getElementById('physicsIdPanel'),
    physicsIdMeta: document.getElementById('physicsIdMeta'),
    physRegimeBadge: document.getElementById('physRegimeBadge'),
    physJumpBadge: document.getElementById('physJumpBadge'),
    physicsOdeEq: document.getElementById('physicsOdeEq'),
    kpiPhysicsR2: document.getElementById('kpiPhysicsR2'),
    kpiPhysicsConf: document.getElementById('kpiPhysicsConf'),
    kpiImmWeights: document.getElementById('kpiImmWeights'),
    kpiPhysicsBeta: document.getElementById('kpiPhysicsBeta'),
    kpiPhysicsThrust: document.getElementById('kpiPhysicsThrust'),
    kpiPhysicsDamping: document.getElementById('kpiPhysicsDamping'),
    kpiPhysicsEllipticT: document.getElementById('kpiPhysicsEllipticT'),
    kpiKuramotoSync: document.getElementById('kpiKuramotoSync'),
    kpiLyapunovMax: document.getElementById('kpiLyapunovMax'),
    kpiTaubinDelta: document.getElementById('kpiTaubinDelta'),
    kpiTaubinKappa: document.getElementById('kpiTaubinKappa'),
    kpiFistaKnots: document.getElementById('kpiFistaKnots'),
    kpiTwoBodyMassE: document.getElementById('kpiTwoBodyMassE'),
    kpiTwoBodyMomRes: document.getElementById('kpiTwoBodyMomRes'),
    kpiPhysicsConserv: document.getElementById('kpiPhysicsConserv'),
    continuumFieldPanel: document.getElementById('continuumFieldPanel'),
    continuumSummaryMeta: document.getElementById('continuumSummaryMeta'),
    swarmModeSelect: document.getElementById('swarmModeSelect'),
    fieldOverlayModeSelect: document.getElementById('fieldOverlayModeSelect'),
    bosModeSelect: document.getElementById('bosModeSelect'),
    metrologyModeSelect: document.getElementById('metrologyModeSelect'),
    hodgeModeSelect: document.getElementById('hodgeModeSelect'),
    rsVibroModeSelect: document.getElementById('rsVibroModeSelect'),
    depthModeSelect: document.getElementById('depthModeSelect'),
    windModeSelect: document.getElementById('windModeSelect'),
    elastoModeSelect: document.getElementById('elastoModeSelect'),
    probeExciterPreset: document.getElementById('probeExciterPreset'),
    btnProbeExciter: document.getElementById('btnProbeExciter'),
    inpRsDtRowUs: document.getElementById('inpRsDtRowUs'),
    inpRsExposureUs: document.getElementById('inpRsExposureUs'),
    selRsScanDir: document.getElementById('selRsScanDir'),
    chkCovEllipses: document.getElementById('chkCovEllipses'),
    chkNativeRoi: document.getElementById('chkNativeRoi'),
    rangeRoiX0: document.getElementById('rangeRoiX0'),
    rangeRoiY0: document.getElementById('rangeRoiY0'),
    valRoiX0: document.getElementById('valRoiX0'),
    valRoiY0: document.getElementById('valRoiY0'),
    roiGeomMeta: document.getElementById('roiGeomMeta'),
    acquisitionStatusBanner: document.getElementById('acquisitionStatusBanner'),
    kpiSwarmConfirmed: document.getElementById('kpiSwarmConfirmed'),
    kpiSwarmPeak: document.getElementById('kpiSwarmPeak'),
    kpiSwarmUnique: document.getElementById('kpiSwarmUnique'),
    kpiLatencySplit: document.getElementById('kpiLatencySplit'),
    kpiRigidBodies: document.getElementById('kpiRigidBodies'),
    kpiContinuumVort: document.getElementById('kpiContinuumVort'),
    kpiContinuumFtle: document.getElementById('kpiContinuumFtle'),
    kpiModalField: document.getElementById('kpiModalField'),
    swarmInspectReadout: document.getElementById('swarmInspectReadout'),
    expertDiagnosticsDrawer: document.getElementById('expertDiagnosticsDrawer'),
    expertDiagnosticsSummary: document.getElementById('expertDiagnosticsSummary'),
    diagAuctionSteps: document.getElementById('diagAuctionSteps'),
    diagRawResiduals: document.getElementById('diagRawResiduals'),
    diagValidityMask: document.getElementById('diagValidityMask'),
    diagFtleWindow: document.getElementById('diagFtleWindow'),
    diagSparseGraph: document.getElementById('diagSparseGraph'),
    diagKuramotoCom: document.getElementById('diagKuramotoCom'),
    diagFieldBackend: document.getElementById('diagFieldBackend'),
    diagMeanStrain: document.getElementById('diagMeanStrain'),
    diagWave1Ingest: document.getElementById('diagWave1Ingest'),
    diagWave1Stamps: document.getElementById('diagWave1Stamps'),
    diagRoiGeom: document.getElementById('diagRoiGeom'),
    diagBosState: document.getElementById('diagBosState'),
    diagMetrologyCov: document.getElementById('diagMetrologyCov'),
    diagWave2Stamps: document.getElementById('diagWave2Stamps'),
    diagRsWelch: document.getElementById('diagRsWelch'),
    diagHodge2Complex: document.getElementById('diagHodge2Complex'),
    diagOcclConservation: document.getElementById('diagOcclConservation'),
    diagWindTunnel: document.getElementById('diagWindTunnel'),
    diagElastography: document.getElementById('diagElastography'),
    kpiTremorHz: document.getElementById('kpiTremorHz'),
    kpiTremorPhi: document.getElementById('kpiTremorPhi'),
    lblApexKpi: document.getElementById('lblApexKpi'),
    kpiApexMs: document.getElementById('kpiApexMs'),
    lblReflexKpi: document.getElementById('lblReflexKpi'),
    kpiSaccadeMs: document.getElementById('kpiSaccadeMs'),
    lblKinAngle: document.getElementById('lblKinAngle'),
    kpiKinAngle: document.getElementById('kpiKinAngle'),
    lblKinSpeed: document.getElementById('lblKinSpeed'),
    kpiKinSpeed: document.getElementById('kpiKinSpeed'),
    lblKinDeflect: document.getElementById('lblKinDeflect'),
    kpiKinDeflect: document.getElementById('kpiKinDeflect'),
    lblKinRestGrav: document.getElementById('lblKinRestGrav'),
    kpiKinRestGrav: document.getElementById('kpiKinRestGrav'),
    calBadge: document.getElementById('calBadge'),
    calToolbar: document.getElementById('calToolbar'),
    calStatusLive: document.getElementById('calStatusLive'),
    btnCalRuler: document.getElementById('btnCalRuler'),
    btnCalQuad: document.getElementById('btnCalQuad'),
    btnCalAutoG: document.getElementById('btnCalAutoG'),
    btnCalPreset: document.getElementById('btnCalPreset'),
    btnCalClear: document.getElementById('btnCalClear'),
    calScaleForm: document.getElementById('calScaleForm'),
    calRulerM: document.getElementById('calRulerM'),
    calPlaneW: document.getElementById('calPlaneW'),
    calPlaneH: document.getElementById('calPlaneH'),
    calCoordForm: document.getElementById('calCoordForm'),
    calCoordMode: document.getElementById('calCoordMode'),
    labBadge: document.getElementById('labBadge'),
    labDescription: document.getElementById('labDescription'),
    btnLabStimulus: document.getElementById('btnLabStimulus'),
    phaseOffsetLabel: document.getElementById('phaseOffsetLabel'),
    recStatusLabel: document.getElementById('recStatusLabel'),
    benchScenarioField: document.getElementById('benchScenarioField'),
    benchScenarioSelect: document.getElementById('benchScenarioSelect'),
    liveStreamBar: document.getElementById('liveStreamBar'),
    liveTierBadge: document.getElementById('liveTierBadge'),
    liveStreamMeta: document.getElementById('liveStreamMeta'),
    liveCropSelect: document.getElementById('liveCropSelect'),
    liveDialogStatus: document.getElementById('liveDialogStatus'),
    tier3PresetSelect: document.getElementById('tier3PresetSelect'),
    tier3CustomHlsUrl: document.getElementById('tier3CustomHlsUrl'),
    tier2StreamUrl: document.getElementById('tier2StreamUrl'),
    btnFallbackTier2ToTier3: document.getElementById('btnFallbackTier2ToTier3'),
    videoTransportBar: document.getElementById('videoTransportBar'),
    vidTimeLabel: document.getElementById('vidTimeLabel'),
    videoScrubSlider: document.getElementById('videoScrubSlider'),
    btnFastPass: document.getElementById('btnFastPass'),
    btnVidPlayPause: document.getElementById('btnVidPlayPause'),
    btnFocusCycle: document.getElementById('btnFocusCycle'),
    btnOpenFullDeck: document.getElementById('btnOpenFullDeck'),
    btnReturnShowcase: document.getElementById('btnReturnShowcase'),
    btnShowcasePlayPause: document.getElementById('btnShowcasePlayPause'),
    btnShowcaseStepFwd: document.getElementById('btnShowcaseStepFwd'),
    hudCanvasStage: document.getElementById('hudCanvasStage'),
    hudWipeTrack: document.getElementById('hudWipeTrack'),
    hudWipeHandle: document.getElementById('hudWipeHandle'),
    showcaseWipeSlider: document.getElementById('showcaseWipeSlider'),
    showcaseWipeReadout: document.getElementById('showcaseWipeReadout'),
    showcaseToolbar: document.getElementById('showcaseToolbar'),
    showcaseToolbarStatus: document.getElementById('showcaseToolbarStatus'),
    scValKernel: document.getElementById('scValKernel'),
    scSubKernel: document.getElementById('scSubKernel'),
    scValTracks: document.getElementById('scValTracks'),
    scSubTracks: document.getElementById('scSubTracks'),
    scValPhysics: document.getElementById('scValPhysics'),
    scSubPhysics: document.getElementById('scSubPhysics'),
    scValInspect: document.getElementById('scValInspect'),
    scSubInspect: document.getElementById('scSubInspect'),
  };

  let videoScrubDragging = false;
  let latestTimelineEvents = [];
  let latestTimelineSpanSec = 0.0;
  let latestTimelineSeq = 0;
  let activeVideoFile = null;
  let activeVideoUrl = '';
  let fastPassRunning = false;
  let fastPassCancelRequested = false;

  function updateVideoTimeLabel() {
    const isVid = activeSourceMode === 'video';
    if (dom.btnShowcasePlayPause) {
      dom.btnShowcasePlayPause.setAttribute('aria-disabled', String(!isVid));
    }
    if (dom.btnShowcaseStepFwd) {
      dom.btnShowcaseStepFwd.setAttribute('aria-disabled', String(!isVid));
    }
    if (!isVid) {
      if (dom.btnShowcasePlayPause) {
        dom.btnShowcasePlayPause.textContent = '⏸ Live';
        dom.btnShowcasePlayPause.removeAttribute('aria-pressed');
      }
      if (dom.videoScrubSlider) {
        dom.videoScrubSlider.value = '0';
        dom.videoScrubSlider.disabled = true;
      }
      return;
    }
    const curNum = Number.isFinite(videoEl.currentTime) ? videoEl.currentTime : 0;
    const durNum = Number.isFinite(videoEl.duration) && videoEl.duration > 0 ? videoEl.duration : 0;
    const cur = curNum.toFixed(2);
    const dur = durNum > 0 ? durNum.toFixed(2) : '--';
    if (!fastPassRunning) {
      dom.vidTimeLabel.textContent = `${cur}s / ${dur}s`;
    }
    if (dom.videoScrubSlider) {
      dom.videoScrubSlider.disabled = durNum <= 0 || fastPassRunning;
      if (!videoScrubDragging && durNum > 0) {
        dom.videoScrubSlider.value = String(Math.max(0, Math.min(1000, Math.round((curNum / durNum) * 1000))));
      }
    }
    const playing = !videoEl.paused && !videoEl.ended;
    dom.btnVidPlayPause.classList.toggle('active', playing);
    dom.btnVidPlayPause.setAttribute('aria-pressed', String(playing));
    dom.btnVidPlayPause.textContent = playing ? '⏸ Pause' : '▶ Play';
    if (dom.btnShowcasePlayPause) {
      dom.btnShowcasePlayPause.textContent = playing ? '⏸ Pause' : '▶ Play';
    }
  }

  videoEl.addEventListener('timeupdate', updateVideoTimeLabel);
  videoEl.addEventListener('loadedmetadata', updateVideoTimeLabel);

  function mapSourceModeToFsm(mode) {
    if (mode === 'webcam') return 'WEBCAM';
    if (mode === 'video' || mode === 'live') return 'VIDEO';
    if (mode === 'testbench') return 'TESTBENCH';
    return 'IDLE';
  }

  const VALID_DECK_SOURCES = new Set(['IDLE', 'TESTBENCH', 'VIDEO', 'WEBCAM']);
  const VALID_DECK_ENGINES = new Set(['SWARM_OFF', 'AUTO_STANDBY', 'SWARM_ACTIVE']);
  const VALID_POINTER_TOOLS = new Set([
    'INSPECT_TARGET',
    'SEED_BOX',
    'CAL_RULER_P1',
    'CAL_RULER_P2',
    'CAL_QUAD_PT1',
    'CAL_QUAD_PT2',
    'CAL_QUAD_PT3',
    'CAL_QUAD_PT4',
    'CROP_ROI',
    'WIPE_DRAG',
  ]);
  const VALID_VIEWPORTS = new Set(['DOCKED', 'FULLSCREEN']);

  function deriveCalPointerTool(calArmed, calDraftCount) {
    if (calArmed === 'ruler') {
      return Number(calDraftCount) >= 1 ? 'CAL_RULER_P2' : 'CAL_RULER_P1';
    }
    if (calArmed === 'homography') {
      const pt = Math.min(4, Math.max(1, (Math.trunc(Number(calDraftCount) || 0)) + 1));
      return `CAL_QUAD_PT${pt}`;
    }
    return 'INSPECT_TARGET';
  }

  function normalizeInspectAck(raw) {
    const r = raw && typeof raw === 'object' ? raw : {};
    const seqNum = Number.isInteger(r.inspectSeq) ? r.inspectSeq : 0;
    const optNum = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
    const pxX = optNum(r.inspectReqPxX ?? r.inspectReqX);
    const pxY = optNum(r.inspectReqPxY ?? r.inspectReqY);
    return Object.freeze({
      inspectSeq: Math.max(0, seqNum),
      inspectReqNormX: optNum(r.inspectReqNormX),
      inspectReqNormY: optNum(r.inspectReqNormY),
      inspectReqPxX: pxX,
      inspectReqPxY: pxY,
      inspectStatus: typeof r.inspectStatus === 'string' ? r.inspectStatus : 'idle',
      inspectedTid: Number.isInteger(r.inspectedTid) ? Math.max(0, r.inspectedTid) : 0,
    });
  }

  function inspectAckEquals(a, b) {
    if (a === b) return true;
    if (!a || !b) return false;
    return (
      a.inspectSeq === b.inspectSeq &&
      a.inspectReqNormX === b.inspectReqNormX &&
      a.inspectReqNormY === b.inspectReqNormY &&
      a.inspectReqPxX === b.inspectReqPxX &&
      a.inspectReqPxY === b.inspectReqPxY &&
      a.inspectStatus === b.inspectStatus &&
      a.inspectedTid === b.inspectedTid
    );
  }

  function freezeDeckStateSnapshot(st) {
    return Object.freeze({
      seq: st.seq,
      source: st.source,
      engine: st.engine,
      overlays: Object.freeze({
        showHeroTargets: Boolean(st.overlays.showHeroTargets),
        showAllOverlays: Boolean(st.overlays.showAllOverlays),
        showEcaOverlay: Boolean(st.overlays.showEcaOverlay),
        showKalman: Boolean(st.overlays.showKalman),
        showCovEllipses: Boolean(st.overlays.showCovEllipses),
        fieldOverlayMode: Math.max(0, Math.min(15, Math.trunc(Number(st.overlays.fieldOverlayMode) || 0))),
      }),
      pointerTool: st.pointerTool,
      viewport: st.viewport,
      lastInspectAck: normalizeInspectAck(st.lastInspectAck),
    });
  }

  function transitionDeckState(state, event) {
    if (!event || typeof event.type !== 'string') return state;
    let nextSource = state.source;
    let nextEngine = state.engine;
    let nextOverlays = state.overlays;
    let nextPointerTool = state.pointerTool;
    let nextViewport = state.viewport;
    let nextInspectAck = state.lastInspectAck;

    if (event.type === 'SOURCE_CHANGED') {
      if (VALID_DECK_SOURCES.has(event.source)) {
        nextSource = event.source;
      }
    } else if (event.type === 'ENGINE_SYNC') {
      const computed = event.swarmActive
        ? 'SWARM_ACTIVE'
        : Number(event.swarmMode) === 1
        ? 'AUTO_STANDBY'
        : 'SWARM_OFF';
      if (VALID_DECK_ENGINES.has(computed)) {
        nextEngine = computed;
      }
    } else if (event.type === 'OVERLAY_SET') {
      const ov = {
        showHeroTargets: typeof event.showHeroTargets === 'boolean' ? event.showHeroTargets : state.overlays.showHeroTargets,
        showAllOverlays: typeof event.showAllOverlays === 'boolean' ? event.showAllOverlays : state.overlays.showAllOverlays,
        showEcaOverlay: typeof event.showEcaOverlay === 'boolean' ? event.showEcaOverlay : state.overlays.showEcaOverlay,
        showKalman: typeof event.showKalman === 'boolean' ? event.showKalman : state.overlays.showKalman,
        showCovEllipses: typeof event.showCovEllipses === 'boolean' ? event.showCovEllipses : state.overlays.showCovEllipses,
        fieldOverlayMode:
          typeof event.fieldOverlayMode === 'number' && Number.isFinite(event.fieldOverlayMode)
            ? Math.max(0, Math.min(15, Math.round(event.fieldOverlayMode)))
            : state.overlays.fieldOverlayMode,
      };
      if (
        ov.showHeroTargets !== state.overlays.showHeroTargets ||
        ov.showAllOverlays !== state.overlays.showAllOverlays ||
        ov.showEcaOverlay !== state.overlays.showEcaOverlay ||
        ov.showKalman !== state.overlays.showKalman ||
        ov.showCovEllipses !== state.overlays.showCovEllipses ||
        ov.fieldOverlayMode !== state.overlays.fieldOverlayMode
      ) {
        nextOverlays = ov;
      }
    } else if (event.type === 'CAL_TOOL_SYNC') {
      const derived = deriveCalPointerTool(event.calArmed, event.calDraftCount);
      if (derived !== 'INSPECT_TARGET' || state.pointerTool !== 'CROP_ROI') {
        nextPointerTool = derived;
      }
    } else if (event.type === 'POINTER_TOOL_SET') {
      if (VALID_POINTER_TOOLS.has(event.pointerTool)) {
        nextPointerTool = event.pointerTool;
      }
    } else if (event.type === 'FULLSCREEN_SYNC') {
      const vp = event.active ? 'FULLSCREEN' : 'DOCKED';
      if (VALID_VIEWPORTS.has(vp)) {
        nextViewport = vp;
      }
    } else if (event.type === 'INSPECT_ACK') {
      const candAck = normalizeInspectAck(event.ack);
      if (!inspectAckEquals(state.lastInspectAck, candAck)) {
        nextInspectAck = candAck;
      }
    } else {
      return state;
    }

    if (
      nextSource === state.source &&
      nextEngine === state.engine &&
      nextOverlays === state.overlays &&
      nextPointerTool === state.pointerTool &&
      nextViewport === state.viewport &&
      nextInspectAck === state.lastInspectAck
    ) {
      return state;
    }

    return freezeDeckStateSnapshot({
      seq: state.seq + 1,
      source: nextSource,
      engine: nextEngine,
      overlays: nextOverlays,
      pointerTool: nextPointerTool,
      viewport: nextViewport,
      lastInspectAck: nextInspectAck,
    });
  }

  let hudMaximizedFallback = false;
  let calSyncSeq = 0;
  let overlaySyncSeq = 0;
  let uiCalMinDraftCount = 0;
  let inspectReqSeq = 0;
  let deckUiState = freezeDeckStateSnapshot({
    seq: 0,
    source: isInitialDeckSurface ? 'TESTBENCH' : 'IDLE',
    engine: 'SWARM_OFF',
    overlays: {
      showHeroTargets: true,
      showAllOverlays: false,
      showEcaOverlay: false,
      showKalman: true,
      showCovEllipses: false,
      fieldOverlayMode: 0,
    },
    pointerTool: 'INSPECT_TARGET',
    viewport: 'DOCKED',
  });

  function renderDeckState(state) {
    const ov = state.overlays;
    if (dom.chkHeroTargets && dom.chkHeroTargets.checked !== ov.showHeroTargets) {
      dom.chkHeroTargets.checked = ov.showHeroTargets;
    }
    if (dom.chkAllOverlays && dom.chkAllOverlays.checked !== ov.showAllOverlays) {
      dom.chkAllOverlays.checked = ov.showAllOverlays;
    }
    if (dom.chkEcaOverlay && dom.chkEcaOverlay.checked !== ov.showEcaOverlay) {
      dom.chkEcaOverlay.checked = ov.showEcaOverlay;
    }
    if (dom.chkKalman && dom.chkKalman.checked !== ov.showKalman) {
      dom.chkKalman.checked = ov.showKalman;
    }
    if (dom.chkCovEllipses && dom.chkCovEllipses.checked !== ov.showCovEllipses) {
      dom.chkCovEllipses.checked = ov.showCovEllipses;
    }
    if (dom.fieldOverlayModeSelect && dom.fieldOverlayModeSelect.value !== String(ov.fieldOverlayMode)) {
      dom.fieldOverlayModeSelect.value = String(ov.fieldOverlayMode);
    }
    if (dom.btnHudHeroToggle) {
      const on = ov.showHeroTargets;
      const soloHidden = !on && state.engine !== 'SWARM_ACTIVE';
      dom.btnHudHeroToggle.classList.toggle('active', on);
      dom.btnHudHeroToggle.setAttribute('aria-pressed', String(on));
      dom.btnHudHeroToggle.dataset.hiddenSolo = String(soloHidden);
      setTextIfChanged(dom.btnHudHeroToggle, on ? 'Hero 16: ON' : 'HERO 16: HIDDEN');
      dom.btnHudHeroToggle.title = on
        ? 'Hide 16-slot Hero reticles, trails & Hermite bridges (H)'
        : soloHidden
        ? 'Hero 16 overlays hidden while Swarm is inactive -- click or press H to restore'
        : 'Show 16-slot Hero reticles & trails (H)';
    }

    const isFs = state.viewport === 'FULLSCREEN';
    const useFallbackMax = hudMaximizedFallback && isFs;
    if (dom.hudPanel) {
      if (dom.hudDockSlot && useFallbackMax && !dom.hudPanel.classList.contains('hud-maximized')) {
        const dockRect = dom.hudPanel.getBoundingClientRect();
        if (dockRect.height > 0) {
          dom.hudDockSlot.style.blockSize = `${dockRect.height}px`;
        }
      }
      dom.hudPanel.classList.toggle('hud-maximized', useFallbackMax);
      if (dom.hudDockSlot && !useFallbackMax) {
        dom.hudDockSlot.style.blockSize = '';
      }
      dom.hudPanel.dataset.source = state.source;
      dom.hudPanel.dataset.engine = state.engine;
      dom.hudPanel.dataset.pointerTool = state.pointerTool;
      dom.hudPanel.dataset.viewport = state.viewport;
    }
    if (dom.btnHudFullscreen) {
      dom.btnHudFullscreen.classList.toggle('active', isFs);
      dom.btnHudFullscreen.setAttribute('aria-pressed', String(isFs));
      setTextIfChanged(dom.btnHudFullscreen, isFs ? '✕ Exit Full' : '⛶ Fullscreen');
      dom.btnHudFullscreen.title = isFs
        ? 'Exit HUD fullscreen (F or Esc)'
        : 'Expand Delayed Wavefield HUD to fullscreen (F)';
    }
    if (hudEl) {
      hudEl.dataset.pointerTool = state.pointerTool;
    }
  }

  function dispatchDeckEvent(event) {
    const next = transitionDeckState(deckUiState, event);
    if (next !== deckUiState) {
      deckUiState = next;
      const proj = projectLegacyUiState(instrumentState);
      window.__chronosUiState = Object.freeze({
        ...deckUiState,
        schema: proj.schema,
        bosOwner: proj.bosOwner,
        recipe: proj.recipe,
        ui: proj.ui,
        sourceDetail: proj.sourceDetail,
      });
      renderDeckState(deckUiState);
      if (event.type === 'POINTER_TOOL_SET' || event.type === 'CAL_TOOL_SYNC') {
        dispatchInstrument({ type: 'SET_POINTER_TOOL', tool: deckUiState.pointerTool }, true);
      } else if (event.type === 'FULLSCREEN_SYNC') {
        dispatchInstrument({ type: 'SET_VIEWPORT', viewport: deckUiState.viewport }, true);
      } else {
        renderInstrumentViews(true);
      }
    }
    return deckUiState;
  }

  function beginOptimisticCalArm(nextArm) {
    calSyncSeq += 1;
    uiCalMinDraftCount = 0;
    dispatchDeckEvent({
      type: 'CAL_TOOL_SYNC',
      calArmed: nextArm,
      calDraftCount: 0,
    });
    return calSyncSeq;
  }

  function nextOverlaySyncSeq() {
    overlaySyncSeq += 1;
    return overlaySyncSeq;
  }

  function advanceOptimisticCalClick() {
    const pt = deckUiState.pointerTool;
    if (pt === 'CAL_RULER_P1') {
      uiCalMinDraftCount = Math.max(uiCalMinDraftCount, 1);
      dispatchDeckEvent({ type: 'CAL_TOOL_SYNC', calArmed: 'ruler', calDraftCount: 1 });
      return 'ruler';
    }
    if (pt === 'CAL_RULER_P2') {
      return 'ruler';
    }
    if (pt.startsWith('CAL_QUAD_PT')) {
      const curPt = Math.max(1, Math.min(4, Number(pt.slice(11)) || 1));
      if (curPt < 4) {
        uiCalMinDraftCount = Math.max(uiCalMinDraftCount, curPt);
        dispatchDeckEvent({ type: 'CAL_TOOL_SYNC', calArmed: 'homography', calDraftCount: curPt });
      }
      return 'homography';
    }
    return 'none';
  }

  renderDeckState(deckUiState);
  window.__chronosUiState = deckUiState;
  window.__chronosInspectState = deckUiState.lastInspectAck;

  const idleTransportRowEl = document.getElementById('idleTransportRow');
  function syncContextualSourceUi(mode, detail = {}) {
    activeSourceMode = mode;
    if (dom.benchScenarioField) dom.benchScenarioField.hidden = mode !== 'testbench';
    if (dom.videoTransportBar) dom.videoTransportBar.hidden = mode !== 'video';
    if (dom.liveStreamBar) dom.liveStreamBar.hidden = mode !== 'live';
    if (idleTransportRowEl) idleTransportRowEl.hidden = mode !== 'idle';
    if (dom.pillStreamHealth) {
      if (mode === 'testbench') {
        dom.pillStreamHealth.dataset.signal = 'lock';
        setTextIfChanged(dom.pillStreamHealth, '◆ SIGNAL GEN');
      } else if (mode === 'idle') {
        delete dom.pillStreamHealth.dataset.signal;
        setTextIfChanged(dom.pillStreamHealth, '○ STANDBY');
      }
    }
    const nextKind = detail.kind || (
      mode === 'idle' ? 'IDLE'
      : mode === 'webcam' ? 'WEBCAM'
      : mode === 'live' ? 'LIVE_MSE'
      : mode === 'video' ? (activeVideoFile ? 'FILE' : 'SAMPLE')
      : 'TESTBENCH'
    );
    const nextLabel = detail.label || (dom.hudSourceLabel ? dom.hudSourceLabel.textContent : '');
    if (dom.hudSourceLabel && nextLabel) dom.hudSourceLabel.title = nextLabel;
    dispatchInstrument({
      type: 'SOURCE_ACTIVATED',
      kind: nextKind,
      label: nextLabel,
      sampleUrl: activeVideoUrl || '',
      scenario: dom.benchScenarioSelect ? dom.benchScenarioSelect.value : 'pillar_crossing',
      originPreset: detail.originPreset !== undefined ? detail.originPreset : null,
      fps: activeVideoFps,
    }, true);
    dispatchDeckEvent({ type: 'SOURCE_CHANGED', source: mapSourceModeToFsm(mode) });
    syncStimulusButtonAvailability();
    updateVideoTimeLabel();
  }

  function flushPendingQueue() {
    if (queuedPendingFrame && queuedPendingFrame.bitmap && typeof queuedPendingFrame.bitmap.close === 'function') {
      queuedPendingFrame.bitmap.close();
    }
    queuedPendingFrame = null;
    workerFrameBusy = false;
    activeCaptureToken++;
    bitmapCaptureBusy = false;
  }

  let rsAudioCaptureCtx = null;
  let rsAudioCaptureSource = null;
  let rsAudioCaptureProc = null;
  let rsAudioCaptureGain = null;
  let rsAudioCaptureStream = null;
  let rsAudioCaptureTrack = null;
  let rsAudioCaptureStreamCleanup = null;
  let rsAudioCaptureTrackCleanup = null;
  let rsAudioCaptureRateHz = 48000.0;
  let rsAudioCaptureBaseUs = 0;
  let rsAudioCaptureCursor = 0;
  let rsAudioCaptureAnchored = false;
  let rsAudioAnchorRefineLeft = 0;
  let rsAudioTrackPollId = 0;
  let rsAudioMicReqStream = null;
  let rsAudioMicReqSeq = 0;
  let rsAudioMicInFlightSeq = 0;
  let rsSupplementalFallbackTrack = null;
  let rsOwnedWebcamAudioTracks = [];
  let rsVidClockOffsetUs = 0.0;
  let rsVidClockValid = false;
  let rsVidClockCaptureTimed = false;
  let rsAudioLastSentTEndUs = 0;
  const rsAudioSkewState = {
    warm: 0,
    tsCleanRun: 0,
    hasRefOrigin: false,
    refOriginUs: 0.0,
    hasTsOrigin: false,
    tsOriginSec: 0.0,
    hasCtOrigin: false,
    ctOriginSec: 0.0,
    kindSeen: new Uint8Array(4),
    kindOriginUs: new Float64Array(4),
    r0: 0.0,
    r1: 0.0,
    r2: 0.0,
    r3: 0.0,
    r4: 0.0,
    rCount: 0,
    t0: 0.0,
    t1: 0.0,
    t2: 0.0,
    t3: 0.0,
    t4: 0.0,
    tCount: 0,
    prevM: 0.0,
    prevM0: 0.0,
    prevT1: 0.0,
    prevT2: 0.0,
    maxStep2: 0.0,
    uAt6: 0.0,
    b12: new Float64Array(12),
    b12Idx: 0,
    b12Count: 0,
    b12Sum: 0.0,
    b16: new Float64Array(16),
    b16Idx: 0,
    b16Count: 0,
    b16Sum: 0.0,
    t12: new Float64Array(12),
    t12Idx: 0,
    t12Count: 0,
    t12Sum: 0.0,
    b36: new Float64Array(36),
    b36Idx: 0,
    b36Count: 0,
    b36Sum: 0.0,
    zRing: new Float64Array(512),
    zIdx: 0,
    zCount: 0,
    runMinT: 0.0,
    runMaxT: 0.0,
    hasPullInit: false,
    a3Min: 0.0,
    a3Max: 0.0,
    pullMinUs: 0.0,
    pullMaxUs: 0.0,
    prevPullStepUs: 0.0,
    pullUnitUs: 0.0,
    ctPullWarmMin: 0.0,
    ctPullWarmMax: 0.0,
    warmPullTot: 0,
    warmPullAtMin: 0,
    pRing: new Float64Array(96),
    pIdx: 0,
    pCount: 0,
    stripY: new Float64Array(505),
    stripPreMin: new Float64Array(313),
    stripPreMax: new Float64Array(313),
    stripSufMin: new Float64Array(313),
    stripSufMax: new Float64Array(313),
    uFirst: 0.0,
    baseU: 0.0,
    baseCount: 0,
    wFirst: 0.0,
    baseW: 0.0,
    baseWCount: 0,
    driftSign: 0,
  };

  function resetRsAudioSkewState() {
    rsAudioSkewState.warm = 0;
    rsAudioSkewState.tsCleanRun = 0;
    rsAudioSkewState.hasRefOrigin = false;
    rsAudioSkewState.refOriginUs = 0.0;
    rsAudioSkewState.hasTsOrigin = false;
    rsAudioSkewState.tsOriginSec = 0.0;
    rsAudioSkewState.hasCtOrigin = false;
    rsAudioSkewState.ctOriginSec = 0.0;
    rsAudioSkewState.kindSeen.fill(0);
    rsAudioSkewState.kindOriginUs.fill(0.0);
    rsAudioSkewState.r0 = 0.0;
    rsAudioSkewState.r1 = 0.0;
    rsAudioSkewState.r2 = 0.0;
    rsAudioSkewState.r3 = 0.0;
    rsAudioSkewState.r4 = 0.0;
    rsAudioSkewState.rCount = 0;
    rsAudioSkewState.t0 = 0.0;
    rsAudioSkewState.t1 = 0.0;
    rsAudioSkewState.t2 = 0.0;
    rsAudioSkewState.t3 = 0.0;
    rsAudioSkewState.t4 = 0.0;
    rsAudioSkewState.tCount = 0;
    rsAudioSkewState.prevM = 0.0;
    rsAudioSkewState.prevM0 = 0.0;
    rsAudioSkewState.prevT1 = 0.0;
    rsAudioSkewState.prevT2 = 0.0;
    rsAudioSkewState.maxStep2 = 0.0;
    rsAudioSkewState.uAt6 = 0.0;
    rsAudioSkewState.b12.fill(0.0);
    rsAudioSkewState.b12Idx = 0;
    rsAudioSkewState.b12Count = 0;
    rsAudioSkewState.b12Sum = 0.0;
    rsAudioSkewState.b16.fill(0.0);
    rsAudioSkewState.b16Idx = 0;
    rsAudioSkewState.b16Count = 0;
    rsAudioSkewState.b16Sum = 0.0;
    rsAudioSkewState.t12.fill(0.0);
    rsAudioSkewState.t12Idx = 0;
    rsAudioSkewState.t12Count = 0;
    rsAudioSkewState.t12Sum = 0.0;
    rsAudioSkewState.b36.fill(0.0);
    rsAudioSkewState.b36Idx = 0;
    rsAudioSkewState.b36Count = 0;
    rsAudioSkewState.b36Sum = 0.0;
    rsAudioSkewState.zRing.fill(0.0);
    rsAudioSkewState.zIdx = 0;
    rsAudioSkewState.zCount = 0;
    rsAudioSkewState.runMinT = 0.0;
    rsAudioSkewState.runMaxT = 0.0;
    rsAudioSkewState.hasPullInit = false;
    rsAudioSkewState.a3Min = 0.0;
    rsAudioSkewState.a3Max = 0.0;
    rsAudioSkewState.pullMinUs = 0.0;
    rsAudioSkewState.pullMaxUs = 0.0;
    rsAudioSkewState.prevPullStepUs = 0.0;
    rsAudioSkewState.pullUnitUs = 0.0;
    rsAudioSkewState.ctPullWarmMin = 0.0;
    rsAudioSkewState.ctPullWarmMax = 0.0;
    rsAudioSkewState.warmPullTot = 0;
    rsAudioSkewState.warmPullAtMin = 0;
    rsAudioSkewState.pRing.fill(0.0);
    rsAudioSkewState.pIdx = 0;
    rsAudioSkewState.pCount = 0;
    rsAudioSkewState.uFirst = 0.0;
    rsAudioSkewState.baseU = 0.0;
    rsAudioSkewState.baseCount = 0;
    rsAudioSkewState.wFirst = 0.0;
    rsAudioSkewState.baseW = 0.0;
    rsAudioSkewState.baseWCount = 0;
    rsAudioSkewState.driftSign = 0;
  }

  function med5Skew(a, b, c, d, e) {
    const arr = [a, b, c, d, e];
    arr.sort((x, y) => x - y);
    return arr[2];
  }

  function softThreshSkew(val, margin) {
    const av = Math.abs(val);
    if (av <= margin) return 0.0;
    return val < 0.0 ? -(av - margin) : (av - margin);
  }

  function clearRsAudioTrackPoll() {
    if (rsAudioTrackPollId) {
      clearInterval(rsAudioTrackPollId);
      rsAudioTrackPollId = 0;
    }
  }

  function stopRsSupplementalMic(streamForDetach = null) {
    const owned = rsOwnedWebcamAudioTracks;
    rsOwnedWebcamAudioTracks = [];
    rsSupplementalFallbackTrack = null;
    for (let i = 0; i < owned.length; i++) {
      const trk = owned[i];
      if (!trk) continue;
      if (streamForDetach && typeof streamForDetach.removeTrack === 'function') {
        try {
          streamForDetach.removeTrack(trk);
        } catch (_e) {}
      }
      try {
        if (typeof trk.stop === 'function') trk.stop();
      } catch (_e) {}
    }
  }

  function getLiveAudioTrack(stream) {
    if (!stream || typeof stream.getAudioTracks !== 'function') return null;
    const tracks = stream.getAudioTracks();
    if (tracks && tracks.length > 0) {
      for (let i = 0; i < tracks.length; i++) {
        const t = tracks[i];
        if (t && (!t.readyState || t.readyState === 'live')) {
          return t;
        }
      }
    }
    if (
      stream === activeMediaStream &&
      rsSupplementalFallbackTrack &&
      (!rsSupplementalFallbackTrack.readyState || rsSupplementalFallbackTrack.readyState === 'live')
    ) {
      return rsSupplementalFallbackTrack;
    }
    return null;
  }

  function stopRsLiveAudioCapture(unwatchStream = true) {
    if (unwatchStream) {
      clearRsAudioTrackPoll();
    }
    if (rsAudioCaptureTrackCleanup) {
      try {
        rsAudioCaptureTrackCleanup();
      } catch (_e) {}
      rsAudioCaptureTrackCleanup = null;
    }
    if (unwatchStream && rsAudioCaptureStreamCleanup) {
      try {
        rsAudioCaptureStreamCleanup();
      } catch (_e) {}
      rsAudioCaptureStreamCleanup = null;
      rsAudioCaptureStream = null;
    }
    rsAudioCaptureTrack = null;
    if (rsAudioCaptureProc) {
      try {
        rsAudioCaptureProc.onaudioprocess = null;
        rsAudioCaptureProc.disconnect();
      } catch (_e) {}
      rsAudioCaptureProc = null;
    }
    if (rsAudioCaptureSource) {
      try {
        rsAudioCaptureSource.disconnect();
      } catch (_e) {}
      rsAudioCaptureSource = null;
    }
    if (rsAudioCaptureGain) {
      try {
        rsAudioCaptureGain.disconnect();
      } catch (_e) {}
      rsAudioCaptureGain = null;
    }
    if (rsAudioCaptureCtx) {
      try {
        rsAudioCaptureCtx.close();
      } catch (_e) {}
      rsAudioCaptureCtx = null;
    }
    rsAudioCaptureBaseUs = 0;
    rsAudioCaptureCursor = 0;
    rsAudioCaptureAnchored = false;
    rsAudioAnchorRefineLeft = 0;
    rsAudioLastSentTEndUs = 0;
    resetRsAudioSkewState();
    if (unwatchStream) {
      rsAudioMicReqSeq++;
      rsAudioMicInFlightSeq = 0;
      stopRsSupplementalMic(activeMediaStream);
      rsAudioMicReqStream = null;
    }
  }

  function bindRsAudioStreamWatcher(stream) {
    if (rsAudioCaptureStream === stream) return;
    if (rsAudioCaptureStreamCleanup) {
      try {
        rsAudioCaptureStreamCleanup();
      } catch (_e) {}
      rsAudioCaptureStreamCleanup = null;
    }
    rsAudioCaptureStream = stream || null;
    if (!stream) return;
    const onStreamTrackChanged = () => syncRsLiveAudioCapture();
    const hasEv = typeof stream.addEventListener === 'function';
    if (hasEv) {
      stream.addEventListener('addtrack', onStreamTrackChanged);
      stream.addEventListener('removetrack', onStreamTrackChanged);
    }
    const hadOwnAdd = Object.prototype.hasOwnProperty.call(stream, 'addTrack');
    const hadOwnRemove = Object.prototype.hasOwnProperty.call(stream, 'removeTrack');
    const origAdd = typeof stream.addTrack === 'function' ? stream.addTrack : null;
    const origRemove = typeof stream.removeTrack === 'function' ? stream.removeTrack : null;
    if (origAdd) {
      stream.addTrack = function (...args) {
        const r = origAdd.apply(this, args);
        syncRsLiveAudioCapture();
        return r;
      };
    }
    if (origRemove) {
      stream.removeTrack = function (...args) {
        const r = origRemove.apply(this, args);
        syncRsLiveAudioCapture();
        return r;
      };
    }
    rsAudioCaptureStreamCleanup = () => {
      if (hasEv && typeof stream.removeEventListener === 'function') {
        stream.removeEventListener('addtrack', onStreamTrackChanged);
        stream.removeEventListener('removetrack', onStreamTrackChanged);
      }
      if (origAdd) {
        if (hadOwnAdd) stream.addTrack = origAdd;
        else delete stream.addTrack;
      }
      if (origRemove) {
        if (hadOwnRemove) stream.removeTrack = origRemove;
        else delete stream.removeTrack;
      }
    };
  }

  function requestRsLiveMicrophone(targetStream, reqEpoch) {
    if (
      !targetStream ||
      rsAudioMicInFlightSeq !== 0 ||
      rsAudioMicReqStream === targetStream ||
      activeSourceMode !== 'webcam' ||
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.getUserMedia !== 'function'
    ) {
      return;
    }
    rsAudioMicReqStream = targetStream;
    const myReqSeq = ++rsAudioMicReqSeq;
    rsAudioMicInFlightSeq = myReqSeq;
    Promise.resolve()
      .then(() => navigator.mediaDevices.getUserMedia({ audio: true, video: false }))
      .then((micStream) => {
        if (myReqSeq !== rsAudioMicInFlightSeq) {
          if (micStream && micStream !== activeMediaStream && typeof micStream.getTracks === 'function') {
            micStream.getTracks().forEach((t) => {
              if (t && typeof t.stop === 'function') t.stop();
            });
          }
          return;
        }
        rsAudioMicInFlightSeq = 0;
        const rmNow = dom.rsVibroModeSelect ? (parseInt(dom.rsVibroModeSelect.value || '0', 10) || 0) : 0;
        const stillValid =
          reqEpoch === sourceEpoch &&
          activeMediaStream === targetStream &&
          activeSourceMode === 'webcam' &&
          rmNow === 2;
        if (!stillValid) {
          if (micStream && micStream !== targetStream && typeof micStream.getTracks === 'function') {
            micStream.getTracks().forEach((t) => {
              if (t && typeof t.stop === 'function') t.stop();
            });
          }
          if (rsAudioMicReqStream === targetStream) {
            rsAudioMicReqStream = null;
          }
          return;
        }
        if (!micStream) return;
        if (micStream !== targetStream && typeof micStream.getVideoTracks === 'function') {
          const existingVid = typeof targetStream.getVideoTracks === 'function' ? targetStream.getVideoTracks() : [];
          micStream.getVideoTracks().forEach((vt) => {
            if (vt && !existingVid.includes(vt) && typeof vt.stop === 'function') vt.stop();
          });
        }
        let micTrack = getLiveAudioTrack(micStream);
        if (!micTrack && typeof micStream.getTracks === 'function') {
          const all = micStream.getTracks();
          for (let i = 0; i < all.length; i++) {
            const t = all[i];
            if (t && t.kind === 'audio' && (!t.readyState || t.readyState === 'live')) {
              micTrack = t;
              break;
            }
          }
        }
        if (!micTrack) return;
        if (micStream !== targetStream && !rsOwnedWebcamAudioTracks.includes(micTrack)) {
          rsOwnedWebcamAudioTracks.push(micTrack);
        }
        let inStream = false;
        if (typeof targetStream.getAudioTracks === 'function' && targetStream.getAudioTracks().includes(micTrack)) {
          inStream = true;
        } else if (micStream !== targetStream && typeof targetStream.addTrack === 'function') {
          try {
            targetStream.addTrack(micTrack);
          } catch (_e) {}
          inStream =
            typeof targetStream.getAudioTracks === 'function' &&
            targetStream.getAudioTracks().includes(micTrack);
        }
        rsSupplementalFallbackTrack = inStream ? null : micTrack;
        syncRsLiveAudioCapture();
      })
      .catch(() => {
        if (myReqSeq === rsAudioMicInFlightSeq) {
          rsAudioMicInFlightSeq = 0;
        }
      });
  }

  function syncRsLiveAudioCapture() {
    const rm = dom.rsVibroModeSelect ? (parseInt(dom.rsVibroModeSelect.value || '0', 10) || 0) : 0;
    if (!activeMediaStream || rm !== 2 || probeExciterActive) {
      stopRsLiveAudioCapture(true);
      return;
    }
    bindRsAudioStreamWatcher(activeMediaStream);
    const curTrack = getLiveAudioTrack(activeMediaStream);
    if (!curTrack) {
      stopRsLiveAudioCapture(false);
      requestRsLiveMicrophone(activeMediaStream, sourceEpoch);
      if (!rsAudioTrackPollId) {
        rsAudioTrackPollId = setInterval(syncRsLiveAudioCapture, 40);
      }
      return;
    }
    clearRsAudioTrackPoll();
    if (rsAudioCaptureCtx && rsAudioCaptureStream === activeMediaStream && rsAudioCaptureTrack === curTrack) {
      return;
    }
    if (rsAudioCaptureCtx) {
      stopRsLiveAudioCapture(false);
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    try {
      const ctx = new AC();
      let sourceStream = null;
      if (typeof MediaStream === 'function') {
        try {
          sourceStream = new MediaStream([curTrack]);
        } catch (_msErr) {
          sourceStream = null;
        }
      }
      if (!sourceStream) {
        sourceStream = {
          id: 'chronos-live-audio',
          active: true,
          getAudioTracks: () => [curTrack],
          getVideoTracks: () => [],
          getTracks: () => [curTrack],
        };
      }
      const src = ctx.createMediaStreamSource(sourceStream);
      const proc = ctx.createScriptProcessor(2048, 1, 1);
      const gain = ctx.createGain();
      gain.gain.value = 0.0;
      rsAudioCaptureRateHz = ctx.sampleRate || 48000.0;
      rsAudioCaptureStream = activeMediaStream;
      rsAudioCaptureTrack = curTrack;
      rsAudioCaptureBaseUs = 0;
      rsAudioCaptureCursor = 0;
      rsAudioCaptureAnchored = false;
      rsAudioAnchorRefineLeft = 0;
      resetRsAudioSkewState();
      const onTrackChanged = () => syncRsLiveAudioCapture();
      if (typeof curTrack.addEventListener === 'function') {
        curTrack.addEventListener('ended', onTrackChanged);
        rsAudioCaptureTrackCleanup = () => {
          if (typeof curTrack.removeEventListener === 'function') {
            curTrack.removeEventListener('ended', onTrackChanged);
          }
        };
      }
      const dtRowUs = dom.inpRsDtRowUs ? Number(dom.inpRsDtRowUs.value) : 40.0;
      const exposureUs = dom.inpRsExposureUs ? Number(dom.inpRsExposureUs.value) : 200.0;
      worker.postMessage({
        type: 'set_rs_calib',
        rsMode: 2,
        dtRowUs,
        exposureUs,
        audioRateHz: rsAudioCaptureRateHz,
        maxSkewUs: 500.0,
      });
      proc.onaudioprocess = (ev) => {
        const activeTrackNow = activeMediaStream ? getLiveAudioTrack(activeMediaStream) : null;
        if (activeMediaStream !== rsAudioCaptureStream || activeTrackNow !== rsAudioCaptureTrack) {
          syncRsLiveAudioCapture();
          return;
        }
        const ch = ev.inputBuffer && ev.inputBuffer.numberOfChannels > 0
          ? ev.inputBuffer.getChannelData(0)
          : null;
        if (!ch || ch.length === 0) return;
        const hasPumpRef = lastMediaTime >= 0 && lastPumpPerfMs > 0;
        if (!hasPumpRef) return;
        const nowPerf = performance.now();
        const mediaNowUs = rsVidClockValid
          ? Math.round(nowPerf * 1000.0 + rsVidClockOffsetUs)
          : Math.round(lastMediaTime * 1_000_000.0 + (nowPerf - lastPumpPerfMs) * 1000.0);
        if (!Number.isFinite(mediaNowUs) || mediaNowUs < 0) return;
        const samples = new Float32Array(ch.length);
        samples.set(ch);
        let outSamples = samples;
        let didResetBase = false;
        if (!rsAudioCaptureAnchored) {
          const availSamples = Math.floor((mediaNowUs * rsAudioCaptureRateHz) / 1_000_000.0);
          if (availSamples <= 0) {
            return;
          }
          if (availSamples < samples.length) {
            outSamples = samples.slice(samples.length - availSamples);
          }
          const outDurUs = Math.round((outSamples.length * 1_000_000.0) / rsAudioCaptureRateHz);
          const candidateBase = Math.max(0, mediaNowUs - outDurUs);
          const candidateEnd = candidateBase + outDurUs;
          rsAudioCaptureBaseUs =
            rsAudioLastSentTEndUs > 0 && candidateEnd <= rsAudioLastSentTEndUs
              ? rsAudioLastSentTEndUs
              : candidateBase;
          rsAudioCaptureCursor = 0;
          rsAudioCaptureAnchored = true;
          rsAudioAnchorRefineLeft = 4;
          didResetBase = true;
        } else if (rsAudioAnchorRefineLeft > 0) {
          rsAudioAnchorRefineLeft--;
          const availSamples = Math.floor((mediaNowUs * rsAudioCaptureRateHz) / 1_000_000.0);
          const estNextEndUs = rsAudioCaptureBaseUs + Math.round(((rsAudioCaptureCursor + outSamples.length) * 1_000_000.0) / rsAudioCaptureRateHz);
          const tentativeSkewUs = estNextEndUs - mediaNowUs;
          if (tentativeSkewUs > 200.0 && availSamples > 0) {
            if (tentativeSkewUs <= 800.0 && rsAudioCaptureBaseUs >= tentativeSkewUs) {
              const newBase = rsAudioCaptureBaseUs - Math.round(tentativeSkewUs);
              const newEnd = newBase + Math.round(((rsAudioCaptureCursor + outSamples.length) * 1_000_000.0) / rsAudioCaptureRateHz);
              if (newEnd > rsAudioLastSentTEndUs) {
                rsAudioCaptureBaseUs = newBase;
                didResetBase = true;
              }
            } else {
              let nextSlice = outSamples;
              if (availSamples < nextSlice.length) {
                nextSlice = nextSlice.slice(nextSlice.length - availSamples);
              }
              const outDurUs = Math.round((nextSlice.length * 1_000_000.0) / rsAudioCaptureRateHz);
              const newBase = Math.max(0, mediaNowUs - outDurUs);
              const newEnd = newBase + outDurUs;
              if (newEnd > rsAudioLastSentTEndUs) {
                outSamples = nextSlice;
                rsAudioCaptureBaseUs = newBase;
                rsAudioCaptureCursor = 0;
                didResetBase = true;
              }
            }
          }
        }
        const s = rsAudioSkewState;
        if (didResetBase) {
          resetRsAudioSkewState();
        }
        rsAudioCaptureCursor += outSamples.length;
        const safeRateHz = Math.max(8000.0, rsAudioCaptureRateHz);
        const cursorSec = rsAudioCaptureCursor / safeRateHz;
        const tEndUs = rsAudioCaptureBaseUs + Math.round((rsAudioCaptureCursor * 1_000_000.0) / rsAudioCaptureRateHz);
        if (!s.hasRefOrigin) {
          const candidateRefUs = rsVidClockValid
            ? rsVidClockOffsetUs
            : lastMediaTime * 1_000_000.0 - lastPumpPerfMs * 1000.0;
          s.refOriginUs = Number.isFinite(candidateRefUs) ? candidateRefUs : 0.0;
          s.hasRefOrigin = true;
        }
        const skewFromPerfMs = (perfMs) => {
          const mUs = Math.round(perfMs * 1000.0 + s.refOriginUs);
          return tEndUs - (Number.isFinite(mUs) ? mUs : mediaNowUs);
        };

        const raw0 = skewFromPerfMs(nowPerf);
        if (!s.kindSeen[0]) {
          s.kindOriginUs[0] = raw0 - s.prevM;
          s.kindSeen[0] = 1;
        }

        let hasCt = false;
        let raw3 = 0.0;
        if (typeof ctx.currentTime === 'number' && Number.isFinite(ctx.currentTime) && ctx.currentTime > 0) {
          if (!s.hasCtOrigin || Math.abs(cursorSec + s.ctOriginSec - ctx.currentTime) >= 10.0) {
            s.ctOriginSec = ctx.currentTime - cursorSec;
            s.hasCtOrigin = true;
          }
          const obsPullSec = ctx.currentTime - (cursorSec + s.ctOriginSec);
          raw3 = skewFromPerfMs(nowPerf - obsPullSec * 1000.0);
          hasCt = true;
          if (!s.kindSeen[3]) {
            s.kindOriginUs[3] = raw3 - s.prevM;
            s.kindSeen[3] = 1;
          }
        }

        let hasTs = false;
        let rawTs = 0.0;
        let tsKind = 1;
        if (typeof ctx.getOutputTimestamp === 'function') {
          try {
            const ts = ctx.getOutputTimestamp();
            if (
              ts &&
              typeof ts.performanceTime === 'number' &&
              Number.isFinite(ts.performanceTime) &&
              ts.performanceTime > 0 &&
              typeof ts.contextTime === 'number' &&
              Number.isFinite(ts.contextTime) &&
              ts.contextTime > 0
            ) {
              if (!s.hasTsOrigin || Math.abs(cursorSec + s.tsOriginSec - ts.contextTime) >= 10.0) {
                s.tsOriginSec = ts.contextTime - cursorSec;
                s.hasTsOrigin = true;
              }
              const raw1 = skewFromPerfMs(ts.performanceTime + (cursorSec + s.tsOriginSec - ts.contextTime) * 1000.0);
              if (!s.kindSeen[1]) {
                s.kindOriginUs[1] = raw1 - s.prevM;
                s.kindSeen[1] = 1;
              }
              if (
                ev &&
                typeof ev.playbackTime === 'number' &&
                Number.isFinite(ev.playbackTime) &&
                ev.playbackTime > 0 &&
                Math.abs(ev.playbackTime - ts.contextTime) < 10.0
              ) {
                const raw2 = skewFromPerfMs(ts.performanceTime + (ev.playbackTime - ts.contextTime) * 1000.0);
                if (!s.kindSeen[2]) {
                  s.kindOriginUs[2] = raw2 - s.prevM;
                  s.kindSeen[2] = 1;
                }
                rawTs = raw2;
                tsKind = 2;
              } else {
                rawTs = raw1;
                tsKind = 1;
              }
              hasTs = true;
            }
          } catch (_tsErr) {}
        }

        const aligned0 = raw0 - s.kindOriginUs[0];
        const aligned3 = hasCt ? raw3 - s.kindOriginUs[3] : aligned0;
        let alignedUs = aligned0;
        let trendInputUs = aligned0;
        if (hasTs) {
          alignedUs = rawTs - s.kindOriginUs[tsKind];
          trendInputUs = alignedUs;
          s.tsCleanRun++;
        } else {
          s.tsCleanRun = 0;
          if (hasCt) {
            alignedUs = aligned3;
          }
        }

        let estSkewUs = 0;
        if (!didResetBase) {
          s.warm++;
          const quantumUs = (128.0 * 1_000_000.0) / safeRateHz;
          const pull512Us = (512.0 * 1_000_000.0) / safeRateHz;
          const stepClamp = Math.max(3500.0, 1.35 * pull512Us);

          // 1) Hampel outlier guard + rate-scaled step clamp rejects >1.5-pull spikes without inflating [0, Q] random jitter
          s.r0 = s.r1;
          s.r1 = s.r2;
          s.r2 = s.r3;
          s.r3 = s.r4;
          s.r4 = alignedUs;
          if (s.rCount < 5) s.rCount++;
          let rawMed;
          if (s.rCount < 5) {
            const sl = [s.r0, s.r1, s.r2, s.r3, s.r4].slice(5 - s.rCount).sort((x, y) => x - y);
            rawMed = sl[sl.length >> 1];
          } else {
            rawMed = med5Skew(s.r0, s.r1, s.r2, s.r3, s.r4);
          }
          const despiked = Math.abs(alignedUs - rawMed) > 1.5 * pull512Us ? rawMed : alignedUs;
          const mClamped =
            s.warm <= 5
              ? despiked
              : s.prevM + Math.max(-stepClamp, Math.min(stepClamp, despiked - s.prevM));
          s.prevM = mClamped;
          if (s.warm === 6) s.uAt6 = mClamped;

          // 2) Fast offset branch u = Boxcar(12) * Boxcar(16) + linear unwrapped trend branch w36 = Boxcar(12)*Boxcar(36), v = Boxcar(48)(w36)
          if (s.b12Count < 12) {
            s.b12[s.b12Count++] = mClamped;
            s.b12Sum += mClamped;
          } else {
            s.b12Sum += mClamped - s.b12[s.b12Idx];
            s.b12[s.b12Idx] = mClamped;
            s.b12Idx = (s.b12Idx + 1) % 12;
          }
          const avg12 = s.b12Sum / s.b12Count;

          if (s.b16Count < 16) {
            s.b16[s.b16Count++] = avg12;
            s.b16Sum += avg12;
          } else {
            s.b16Sum += avg12 - s.b16[s.b16Idx];
            s.b16[s.b16Idx] = avg12;
            s.b16Idx = (s.b16Idx + 1) & 15;
          }
          const u = s.b16Sum / s.b16Count;

          s.t0 = s.t1;
          s.t1 = s.t2;
          s.t2 = s.t3;
          s.t3 = s.t4;
          s.t4 = trendInputUs;
          if (s.tCount < 5) s.tCount++;
          let rawMedT;
          if (s.tCount < 5) {
            const slT = [s.t0, s.t1, s.t2, s.t3, s.t4].slice(5 - s.tCount).sort((x, y) => x - y);
            rawMedT = slT[slT.length >> 1];
          } else {
            rawMedT = med5Skew(s.t0, s.t1, s.t2, s.t3, s.t4);
          }
          const despikedT = Math.abs(trendInputUs - rawMedT) > 1.5 * pull512Us ? rawMedT : trendInputUs;
          const mTrend =
            s.warm <= 5
              ? despikedT
              : s.prevM0 + Math.max(-stepClamp, Math.min(stepClamp, despikedT - s.prevM0));
          s.prevM0 = mTrend;
          if (s.warm >= 3) {
            const step2 = Math.abs(mTrend - 2.0 * s.prevT1 + s.prevT2);
            if (s.warm >= 8 && step2 > s.maxStep2) s.maxStep2 = step2;
          }
          s.prevT2 = s.prevT1;
          s.prevT1 = mTrend;
          if (s.warm === 1) {
            s.runMinT = mTrend;
            s.runMaxT = mTrend;
          } else {
            if (mTrend < s.runMinT) s.runMinT = mTrend;
            if (mTrend > s.runMaxT) s.runMaxT = mTrend;
          }
          s.zRing[s.zIdx] = mTrend;
          s.zIdx = (s.zIdx + 1) & 511;
          if (s.zCount < 512) s.zCount++;

          const pullStepUs = !hasTs && hasCt ? aligned3 - aligned0 : 0.0;
          const warmPullGate = s.maxStep2 <= 180.0 ? 250.0 : quantumUs + 150.0;
          if (!hasTs && hasCt) {
            if (!s.hasPullInit) {
              s.hasPullInit = true;
              s.a3Min = aligned3;
              s.a3Max = aligned3;
              s.pullMinUs = pullStepUs;
              s.pullMaxUs = pullStepUs;
              s.prevPullStepUs = pullStepUs;
              s.ctPullWarmMin = pullStepUs;
              s.ctPullWarmMax = pullStepUs;
              s.warmPullTot = 1;
              s.warmPullAtMin = 1;
            } else {
              if (aligned3 < s.a3Min) s.a3Min = aligned3;
              if (aligned3 > s.a3Max) s.a3Max = aligned3;
              if (pullStepUs < s.pullMinUs) s.pullMinUs = pullStepUs;
              if (pullStepUs > s.pullMaxUs) s.pullMaxUs = pullStepUs;
              const dp = Math.abs(pullStepUs - s.prevPullStepUs);
              if (dp >= 0.8 * quantumUs && (s.pullUnitUs === 0.0 || dp < s.pullUnitUs)) {
                s.pullUnitUs = dp;
              }
              s.prevPullStepUs = pullStepUs;
              if (s.warm <= 200 && Math.abs(aligned0) <= warmPullGate) {
                if (pullStepUs < s.ctPullWarmMin) {
                  s.ctPullWarmMin = pullStepUs;
                  s.warmPullAtMin = 1;
                } else if (pullStepUs <= s.ctPullWarmMin + 0.4 * quantumUs) {
                  s.warmPullAtMin++;
                }
                if (pullStepUs > s.ctPullWarmMax) s.ctPullWarmMax = pullStepUs;
                s.warmPullTot++;
              }
            }
            s.pRing[s.pIdx] = pullStepUs;
            s.pIdx = (s.pIdx + 1) % 96;
            if (s.pCount < 96) s.pCount++;
          }

          if (s.t12Count < 12) {
            s.t12[s.t12Count++] = mTrend;
            s.t12Sum += mTrend;
          } else {
            s.t12Sum += mTrend - s.t12[s.t12Idx];
            s.t12[s.t12Idx] = mTrend;
            s.t12Idx = (s.t12Idx + 1) % 12;
          }
          const tAvg12 = s.t12Sum / s.t12Count;

          if (s.b36Count < 36) {
            s.b36[s.b36Count++] = tAvg12;
            s.b36Sum += tAvg12;
          } else {
            s.b36Sum += tAvg12 - s.b36[s.b36Idx];
            s.b36[s.b36Idx] = tAvg12;
            s.b36Idx = (s.b36Idx + 1) % 36;
          }
          const w36 = s.b36Sum / s.b36Count;

          const baseGate = Math.max(2200.0, 0.85 * quantumUs);
          if (s.warm === 16 || (hasTs && s.tsCleanRun === 28 && Math.abs(u - s.uFirst) <= 4500.0)) {
            s.uFirst = u;
            s.baseU = u;
            s.baseCount = 1;
          } else if (s.warm > 16 && s.baseCount < 32 && Math.abs(u - s.uFirst) <= baseGate) {
            s.baseCount++;
            s.baseU += (u - s.baseU) / s.baseCount;
          }

          if (s.warm === 36) {
            s.wFirst = w36;
            s.baseW = w36;
            s.baseWCount = 1;
          } else if (s.warm > 36 && s.baseWCount < 72 && Math.abs(w36 - s.wFirst) <= baseGate) {
            s.baseWCount++;
            s.baseW += (w36 - s.baseW) / s.baseWCount;
          }

          let audioSkew = 0.0;
          if (s.warm >= 10 && s.warm < 24) {
            audioSkew = softThreshSkew(mClamped - s.uAt6, Math.max(16000.0, 1.5 * pull512Us));
          } else if (s.warm >= 24) {
            const cushionUs =
              s.tsCleanRun >= 64
                ? 1750.0
                : Math.max(9200.0, Math.min(36000.0, pull512Us + quantumUs + 450.0));
            const devU = u - s.baseU;
            const devW = w36 - s.baseW;
            const offSkew = softThreshSkew(devU, cushionUs);

            let pRecentMin = pullStepUs;
            if (!hasTs && hasCt) {
              for (let k = 0; k < s.pCount; k++) {
                const pv = s.pRing[k];
                if (pv < pRecentMin) pRecentMin = pv;
              }
            }
            const effPullPeriod = s.pullUnitUs > 0.0 ? Math.min(pull512Us + 50.0, s.pullUnitUs) : pull512Us;
            const spanA3 = s.hasPullInit ? s.a3Max - s.a3Min : 0.0;
            const hasNewPosPull = s.pullMaxUs > s.ctPullWarmMax + 0.75 * quantumUs;
            const warmBimodalCleared =
              s.ctPullWarmMax - s.ctPullWarmMin >= 0.75 * quantumUs &&
              s.warmPullTot >= 24 &&
              s.warmPullAtMin >= 0.12 * s.warmPullTot &&
              pRecentMin > s.ctPullWarmMin + 0.75 * quantumUs;
            const hasNewNegPull =
              s.ctPullWarmMax - s.ctPullWarmMin < 0.4 * quantumUs &&
              s.pullMinUs < s.ctPullWarmMin - 0.75 * quantumUs;
            const multiPullDispOnly =
              !hasTs &&
              hasCt &&
              spanA3 <= effPullPeriod + 25.0 &&
              ((devW < 0.0 && (hasNewPosPull || warmBimodalCleared)) || (devW > 0.0 && hasNewNegPull));

            let trendSkew = 0.0;
            if (!hasTs && hasCt && s.hasPullInit && spanA3 > effPullPeriod + 350.0) {
              const sgnCt = devW !== 0.0 ? (devW < 0.0 ? -1 : 1) : (aligned0 < 0.0 ? -1 : 1);
              trendSkew = softThreshSkew(sgnCt * (spanA3 - effPullPeriod), 350.0);
            }

            const certMargin = Math.max(1800.0, 0.55 * quantumUs + 350.0);
            if (multiPullDispOnly) {
              s.driftSign = 0;
            } else if (s.driftSign !== 0 && s.driftSign * devW <= certMargin) {
              s.driftSign = 0;
            }

            if (!multiPullDispOnly && !hasTs && s.driftSign === 0) {
              const stripCap =
                s.maxStep2 <= 180.0
                  ? Math.max(220.0, 2.2 * s.maxStep2 + 80.0)
                  : quantumUs + 15.0;
              let stripHitSpan = 0.0;

              if (s.zCount >= 309) {
                const W = 308;
                const zEnd = s.zRing[(s.zIdx + 511) & 511];
                const zStart = s.zRing[(s.zIdx + 511 - W) & 511];
                const spanZ = zEnd - zStart;
                const sgnZ = spanZ >= 0.0 ? 1 : -1;
                const runSpan = s.runMaxT - s.runMinT;
                const lo1 = sgnZ > 0 || s.maxStep2 <= 180.0 ? 14.2 : 19.65;
                const hi1 = 23.6;
                const reqSpan1 = sgnZ > 0 ? 5200.0 : (s.maxStep2 <= 180.0 ? 5550.0 : 6050.0);
                const minDevW1 = Math.max(2400.0, 5600.0 - quantumUs - 150.0);
                if (
                  runSpan >= reqSpan1 &&
                  sgnZ * devW >= minDevW1 &&
                  sgnZ * spanZ >= W * lo1 - quantumUs - 15.0
                ) {
                  const y = s.stripY;
                  for (let m = 0; m <= W; m++) {
                    y[m] = s.zRing[(s.zIdx + 511 - (W - m)) & 511];
                  }
                  const evalW = (bt) => {
                    let mn = 1e18;
                    let mx = -1e18;
                    for (let m = 0; m <= W; m++) {
                      const r = y[m] - bt * m;
                      if (r < mn) mn = r;
                      if (r > mx) mx = r;
                    }
                    return mx - mn;
                  };
                  const evalWF = (bt) => {
                    let mn = 1e18;
                    let mx = -1e18;
                    for (let m = 0; m <= W; m += 2) {
                      const r = y[m] - bt * m;
                      if (r < mn) mn = r;
                      if (r > mx) mx = r;
                    }
                    return mx - mn;
                  };

                  let a1 = lo1;
                  let b1 = hi1;
                  for (let it = 0; it < 12; it++) {
                    const m1 = a1 + (b1 - a1) / 3.0;
                    const m2 = b1 - (b1 - a1) / 3.0;
                    if (evalW(sgnZ * m1) <= evalW(sgnZ * m2)) b1 = m2;
                    else a1 = m1;
                  }
                  const mag1 = 0.5 * (a1 + b1);
                  if (evalW(sgnZ * mag1) <= stripCap) {
                    stripHitSpan = sgnZ * mag1 * W;
                  } else {
                    const reqSpanB2 =
                      sgnZ < 0 && s.maxStep2 > 180.0
                        ? Math.max(6500.0, 5000.0 + quantumUs + 25.0)
                        : 6500.0;
                    if (
                      runSpan >= reqSpanB2 &&
                      sgnZ * devW >= Math.max(3000.0, 6500.0 - quantumUs - 150.0)
                    ) {
                      let a2 = hi1;
                      let b2 = Math.max(hi1 + 0.6, (Math.abs(spanZ) + quantumUs) / W + 0.5);
                      for (let it = 0; it < 12; it++) {
                        const m1 = a2 + (b2 - a2) / 3.0;
                        const m2 = b2 - (b2 - a2) / 3.0;
                        if (evalWF(sgnZ * m1) <= evalWF(sgnZ * m2)) b2 = m2;
                        else a2 = m1;
                      }
                      const mag2 = 0.5 * (a2 + b2);
                      if (evalW(sgnZ * mag2) <= stripCap) {
                        const preMin = s.stripPreMin;
                        const preMax = s.stripPreMax;
                        const sufMin = s.stripSufMin;
                        const sufMax = s.stripSufMax;
                        preMin[0] = y[0];
                        preMax[0] = y[0];
                        for (let m = 1; m <= W; m++) {
                          preMin[m] = Math.min(preMin[m - 1], y[m]);
                          preMax[m] = Math.max(preMax[m - 1], y[m]);
                        }
                        sufMin[W] = y[W];
                        sufMax[W] = y[W];
                        for (let m = W - 1; m >= 0; m--) {
                          sufMin[m] = Math.min(sufMin[m + 1], y[m]);
                          sufMax[m] = Math.max(sufMax[m + 1], y[m]);
                        }
                        const Lr = 136;
                        const pCap = stripCap + 5.0;
                        let isShift =
                          sufMax[Lr] - sufMin[Lr] <= pCap ||
                          preMax[W - Lr] - preMin[W - Lr] <= pCap;
                        for (let m0 = 0; !isShift && m0 + Lr <= W; m0 += 2) {
                          if (preMax[m0] - preMin[m0] <= pCap && sufMax[m0 + Lr] - sufMin[m0 + Lr] <= pCap) {
                            isShift = true;
                          }
                        }
                        if (!isShift) {
                          stripHitSpan = sgnZ * mag2 * W;
                        }
                      }
                    }
                  }
                }
              }

              if (stripHitSpan === 0.0 && s.zCount >= 501) {
                const W2 = 500;
                const zEnd2 = s.zRing[(s.zIdx + 511) & 511];
                const zQ1 = s.zRing[(s.zIdx + 511 - 333) & 511];
                const zQ2 = s.zRing[(s.zIdx + 511 - 166) & 511];
                const zStart2 = s.zRing[(s.zIdx + 511 - W2) & 511];
                const spanZ2 = zEnd2 - zStart2;
                const sgnZ2 = spanZ2 >= 0.0 ? 1 : -1;
                const reqSpan2 =
                  sgnZ2 > 0
                    ? 5200.0
                    : (s.maxStep2 <= 180.0 ? 5550.0 : Math.max(6200.0, 5000.0 + quantumUs + 25.0));
                const thirds2 =
                  sgnZ2 * (zQ1 - zStart2) >= 167 * 9.5 - quantumUs - 20.0 &&
                  sgnZ2 * (zQ2 - zQ1) >= 167 * 9.5 - quantumUs - 20.0 &&
                  sgnZ2 * (zEnd2 - zQ2) >= 166 * 9.5 - quantumUs - 20.0;
                if (
                  s.runMaxT - s.runMinT >= reqSpan2 &&
                  thirds2 &&
                  sgnZ2 * devW >= 3600.0 &&
                  sgnZ2 * spanZ2 >= W2 * 9.5 - quantumUs - 15.0
                ) {
                  const y = s.stripY;
                  for (let m = 0; m <= W2; m++) {
                    y[m] = s.zRing[(s.zIdx + 511 - (W2 - m)) & 511];
                  }
                  const evalW2 = (bt) => {
                    let mn = 1e18;
                    let mx = -1e18;
                    for (let m = 0; m <= W2; m += 2) {
                      const r = y[m] - bt * m;
                      if (r < mn) mn = r;
                      if (r > mx) mx = r;
                    }
                    return mx - mn;
                  };
                  let aS = 9.5;
                  let bS = 20.0;
                  for (let it = 0; it < 12; it++) {
                    const m1 = aS + (bS - aS) / 3.0;
                    const m2 = bS - (bS - aS) / 3.0;
                    if (evalW2(sgnZ2 * m1) <= evalW2(sgnZ2 * m2)) bS = m2;
                    else aS = m1;
                  }
                  const magS = 0.5 * (aS + bS);
                  let mnF = 1e18;
                  let mxF = -1e18;
                  for (let m = 0; m <= W2; m++) {
                    const r = y[m] - sgnZ2 * magS * m;
                    if (r < mnF) mnF = r;
                    if (r > mxF) mxF = r;
                  }
                  if (mxF - mnF <= stripCap) {
                    stripHitSpan = sgnZ2 * magS * W2;
                  }
                }
              }

              if (stripHitSpan !== 0.0) {
                const tStrip = softThreshSkew(stripHitSpan, 2500.0);
                if (Math.abs(tStrip) > Math.abs(trendSkew)) trendSkew = tStrip;
              }
            }

            if (!multiPullDispOnly && Math.abs(trendSkew) > 500.0) {
              s.driftSign = trendSkew > 0.0 ? 1 : -1;
            }
            if (s.driftSign !== 0 && !multiPullDispOnly) {
              const certSkew = softThreshSkew(devW, certMargin);
              if (Math.abs(certSkew) > Math.abs(trendSkew)) trendSkew = certSkew;
            }

            audioSkew = Math.abs(trendSkew) > Math.abs(offSkew) ? trendSkew : offSkew;
          }

          if (s.warm >= 10) {
            const curVidRefUs = rsVidClockValid
              ? rsVidClockOffsetUs
              : lastMediaTime * 1_000_000.0 - lastPumpPerfMs * 1000.0;
            const vidRefDeltaUs = Number.isFinite(curVidRefUs) ? curVidRefUs - s.refOriginUs : 0.0;
            const vidCushionUs = rsVidClockValid && rsVidClockCaptureTimed ? 4500.0 : 17200.0;
            const vidSkewUs = s.warm >= 12 ? softThreshSkew(vidRefDeltaUs, vidCushionUs) : 0.0;
            estSkewUs = Math.round(audioSkew - vidSkewUs);
          }
        }
        rsAudioLastSentTEndUs = tEndUs;
        worker.postMessage(
          {
            type: 'push_audio_pcm',
            samples: outSamples,
            tEndUs,
            sampleRateHz: rsAudioCaptureRateHz,
            estSkewUs,
          },
          [outSamples.buffer]
        );
      };
      src.connect(proc);
      proc.connect(gain);
      gain.connect(ctx.destination);
      if (ctx.state === 'suspended') ctx.resume();
      rsAudioCaptureCtx = ctx;
      rsAudioCaptureSource = src;
      rsAudioCaptureProc = proc;
      rsAudioCaptureGain = gain;
    } catch (_e) {
      stopRsLiveAudioCapture();
    }
  }

  // ========================================================================
  // v3.3.0 Muxed MP4 Audio Feeder & WebAudio Probe Exciter (Elastography / ABOS)
  // ========================================================================
  let mp4AudioPcm = null;
  let mp4AudioRateHz = 48000.0;
  let mp4AudioCursor = 0;
  let mp4AudioPrimed = false;
  let mp4AudioDecodeToken = 0;

  function clearMp4AudioState() {
    mp4AudioDecodeToken++;
    mp4AudioPcm = null;
    mp4AudioRateHz = 48000.0;
    mp4AudioCursor = 0;
    mp4AudioPrimed = false;
  }

  async function decodeMp4AudioTrack(urlOrFile, expectedEpoch) {
    const myToken = ++mp4AudioDecodeToken;
    try {
      let rawBuf = null;
      if (typeof urlOrFile === 'string' && urlOrFile.length > 0) {
        const resp = await fetch(urlOrFile);
        if (!resp.ok) return false;
        rawBuf = await resp.arrayBuffer();
      } else if (urlOrFile && typeof urlOrFile.arrayBuffer === 'function') {
        rawBuf = await urlOrFile.arrayBuffer();
      } else {
        return false;
      }
      if (myToken !== mp4AudioDecodeToken || expectedEpoch !== sourceEpoch) return false;
      const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!OAC && !AC) return false;
      const decCtx = OAC ? new OAC(1, 1, 48000) : new AC();
      let decoded = null;
      try {
        decoded = await decCtx.decodeAudioData(rawBuf);
      } finally {
        if (!OAC && typeof decCtx.close === 'function') {
          try { await decCtx.close(); } catch (_e) {}
        }
      }
      if (!decoded || decoded.length === 0 || decoded.numberOfChannels < 1) return false;
      if (myToken !== mp4AudioDecodeToken || expectedEpoch !== sourceEpoch) return false;
      const n = decoded.length;
      const mono = new Float32Array(n);
      if (decoded.numberOfChannels === 1) {
        mono.set(decoded.getChannelData(0));
      } else {
        const ch0 = decoded.getChannelData(0);
        const ch1 = decoded.getChannelData(1);
        for (let i = 0; i < n; i++) {
          mono[i] = 0.5 * (ch0[i] + ch1[i]);
        }
      }
      mp4AudioPcm = mono;
      mp4AudioRateHz = decoded.sampleRate || 48000.0;
      mp4AudioCursor = 0;
      mp4AudioPrimed = false;
      const rm = dom.rsVibroModeSelect ? (parseInt(dom.rsVibroModeSelect.value || '0', 10) || 0) : 0;
      if (rm === 2 && Math.abs(mp4AudioRateHz - 48000.0) > 1.0) {
        const dtRowUs = dom.inpRsDtRowUs ? Number(dom.inpRsDtRowUs.value) : 35.0;
        const exposureUs = dom.inpRsExposureUs ? Number(dom.inpRsExposureUs.value) : 200.0;
        worker.postMessage({
          type: 'set_rs_calib',
          rsMode: 2,
          dtRowUs,
          exposureUs,
          audioRateHz: mp4AudioRateHz,
          maxSkewUs: 500.0,
        });
      }
      return true;
    } catch (_decodeErr) {
      return false;
    }
  }

  function pumpMp4AudioToWorker(mediaTimeSec) {
    if (!mp4AudioPcm || mp4AudioPcm.length === 0) return;
    if (!Number.isFinite(mediaTimeSec) || mediaTimeSec < 0) return;
    const targetSample = Math.min(
      mp4AudioPcm.length,
      Math.round((mediaTimeSec + 0.065) * mp4AudioRateHz)
    );
    if (!mp4AudioPrimed) {
      mp4AudioCursor = Math.max(0, Math.round((mediaTimeSec - 0.12) * mp4AudioRateHz));
      mp4AudioPrimed = true;
    }
    if (targetSample <= mp4AudioCursor) return;
    const slice = mp4AudioPcm.slice(mp4AudioCursor, targetSample);
    const tEndUs = Math.round((targetSample * 1_000_000.0) / mp4AudioRateHz);
    mp4AudioCursor = targetSample;
    worker.postMessage(
      {
        type: 'push_audio_pcm',
        samples: slice,
        tEndUs,
        sampleRateHz: mp4AudioRateHz,
        estSkewUs: 0.0,
      },
      [slice.buffer]
    );
  }

  let probeExciterCtx = null;
  let probeExciterOscs = [];
  let probeExciterActive = false;
  let probeExciterFreqs = [380, 860];
  let probeExciterRateHz = 48000.0;
  let probeExciterCursor = 0;
  let probeExciterPrimed = false;

  function parseProbeExciterFreqs() {
    const raw = dom.probeExciterPreset ? String(dom.probeExciterPreset.value || '380,860') : '380,860';
    const parsed = raw
      .split(',')
      .map((s) => Number(s.trim()))
      .filter((v) => Number.isFinite(v) && v > 1.0 && v <= 20000.0)
      .slice(0, 4);
    return parsed.length > 0 ? parsed : [380, 860];
  }

  function syncProbeExciterUi() {
    if (!dom.btnProbeExciter) return;
    dom.btnProbeExciter.classList.toggle('active', probeExciterActive);
    dom.btnProbeExciter.setAttribute('aria-pressed', String(probeExciterActive));
    setTextIfChanged(dom.btnProbeExciter, probeExciterActive ? '■ Stop' : '▶ Probe');
  }

  function pumpProbeExciterAudioToWorker(mediaTimeSec) {
    if (!probeExciterActive || mp4AudioPcm) return;
    if (!Number.isFinite(mediaTimeSec) || mediaTimeSec < 0) return;
    const rate = probeExciterRateHz;
    const targetSample = Math.round((mediaTimeSec + 0.065) * rate);
    if (!probeExciterPrimed || targetSample < probeExciterCursor) {
      probeExciterCursor = Math.max(0, Math.round((mediaTimeSec - 0.12) * rate));
      probeExciterPrimed = true;
    }
    if (targetSample - probeExciterCursor > Math.round(rate)) {
      probeExciterCursor = Math.max(0, targetSample - Math.round(0.2 * rate));
    }
    const n = targetSample - probeExciterCursor;
    if (n <= 0) return;
    const buf = new Float32Array(n);
    const invN = 1.0 / Math.max(1, probeExciterFreqs.length);
    const startSample = probeExciterCursor;
    for (let f = 0; f < probeExciterFreqs.length; f++) {
      const w = (2.0 * Math.PI * probeExciterFreqs[f]) / rate;
      for (let i = 0; i < n; i++) {
        buf[i] += invN * Math.sin(w * (startSample + i));
      }
    }
    probeExciterCursor = targetSample;
    const tEndUs = Math.round((targetSample * 1_000_000.0) / rate);
    worker.postMessage(
      { type: 'push_audio_pcm', samples: buf, tEndUs, sampleRateHz: rate, estSkewUs: 0.0 },
      [buf.buffer]
    );
  }

  function stopProbeExciter(resumeMic = true) {
    const wasActive = probeExciterActive;
    probeExciterActive = false;
    probeExciterCursor = 0;
    probeExciterPrimed = false;
    for (let i = 0; i < probeExciterOscs.length; i++) {
      try { probeExciterOscs[i].stop(); } catch (_e) {}
      try { probeExciterOscs[i].disconnect(); } catch (_e) {}
    }
    probeExciterOscs = [];
    if (probeExciterCtx) {
      try { probeExciterCtx.close(); } catch (_e) {}
      probeExciterCtx = null;
    }
    syncProbeExciterUi();
    if (wasActive && resumeMic) {
      syncRsLiveAudioCapture();
    }
  }

  function startProbeExciter(opts = null) {
    stopProbeExciter(false);
    const freqs = opts && Array.isArray(opts.freqs) && opts.freqs.length > 0
      ? opts.freqs
      : parseProbeExciterFreqs();
    probeExciterFreqs = freqs;
    probeExciterRateHz = 48000.0;
    probeExciterCursor = 0;
    probeExciterPrimed = false;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (AC) {
      try {
        const ctx = new AC({ sampleRate: 48000 });
        const master = ctx.createGain();
        master.gain.value = 0.18 / Math.max(1, freqs.length);
        master.connect(ctx.destination);
        const oscs = freqs.map((hz) => {
          const osc = ctx.createOscillator();
          osc.type = 'sine';
          osc.frequency.value = hz;
          osc.connect(master);
          osc.start();
          return osc;
        });
        if (ctx.state === 'suspended') ctx.resume();
        probeExciterCtx = ctx;
        probeExciterOscs = oscs;
      } catch (_audioErr) {}
    }
    probeExciterActive = true;
    stopRsLiveAudioCapture(true);
    const curEm = dom.elastoModeSelect ? (parseInt(dom.elastoModeSelect.value || '0', 10) || 0) : 0;
    const targetEm = opts && typeof opts.elastoMode === 'number'
      ? opts.elastoMode
      : (curEm > 0 ? curEm : (freqs.length === 1 && freqs[0] >= 1000 ? 4 : 1));
    if (dom.bosModeSelect && dom.bosModeSelect.value !== '0') {
      dom.bosModeSelect.value = '0';
      worker.postMessage({ type: 'set_bos_config', bosMode: 0, overlayKind: 0 });
    }
    if (dom.elastoModeSelect) dom.elastoModeSelect.value = String(targetEm);
    if (dom.rsVibroModeSelect && (targetEm === 1 || targetEm === 4)) {
      dom.rsVibroModeSelect.value = '2';
      sendRsControlToWorker(true);
    }
    worker.postMessage({ type: 'set_elasto_config', elastoMode: targetEm, lockFreqs: freqs });
    const targetOverlay = opts && typeof opts.overlayMode === 'number'
      ? opts.overlayMode
      : (targetEm === 4 ? 15 : 14);
    if (dom.fieldOverlayModeSelect) {
      dom.fieldOverlayModeSelect.value = String(targetOverlay);
      dispatchDeckEvent({ type: 'OVERLAY_SET', fieldOverlayMode: targetOverlay });
      worker.postMessage({ type: 'set_field_overlay_mode', fieldOverlayMode: targetOverlay, overlaySyncSeq: nextOverlaySyncSeq() });
    }
    syncProbeExciterUi();
  }

  async function activateLiveSoundProbeWebcam() {
    if (activeResMode === 0) {
      document.getElementById('btnResHd')?.click();
    }
    syncSwarmControlsForConfig(1, 0, 11, true);
    if (dom.probeExciterPreset) dom.probeExciterPreset.value = '110,220';
    applyWave1PresetToUiAndWorker({
      resMode: 1,
      swarmMode: 0,
      overlayMode: 11,
      bosMode: 0,
      hodgeMode: 0,
      windMode: 0,
      depthMode: 0,
      rsMode: 2,
      elastoMode: 1,
      elastoLockFreqs: [110, 220],
      rsCalib: { dtRowUs: 35.0, exposureUs: 500.0, scanDir: 0 },
      native1to1: true,
    });
    startProbeExciter({ freqs: [110, 220], elastoMode: 1, overlayMode: 11 });
    if (activeSourceMode !== 'webcam' || !activeMediaStream) {
      await startWebcamCapture({ highlightShowcase: 'live_sound_probe', preferMic: false });
    } else {
      setActiveShowcaseChip('live_sound_probe');
      dom.hudSourceLabel.textContent = 'SOURCE: LIVE WEBCAM · 110+220 Hz SOUND PROBE (RS COHERENCE + ELASTO)';
    }
  }

  function stopMediaCapture(preserveLiveRetry = false) {
    stopRsLiveAudioCapture();
    clearMp4AudioState();
    probeExciterCursor = 0;
    probeExciterPrimed = false;
    rsVidClockOffsetUs = 0.0;
    rsVidClockValid = false;
    rsVidClockCaptureTimed = false;
    sourceEpoch++;
    videoPumpActive = false;
    lastMediaTime = -1;
    lastPresentedFrames = -1;
    pendingVideoGapMask = 0;
    lastPumpPerfMs = 0;
    pumpDroppedFrames = 0;
    flushPendingQueue();
    if (!preserveLiveRetry) {
      clearTimeout(activeMseRetryTimer);
      activeMseRetryTimer = 0;
      activeLiveRetryCount = 0;
      activeLiveConfig = null;
    }
    if (activeMseAbort) {
      activeMseAbort.abort();
      activeMseAbort = null;
    }
    activeMseMediaSource = null;
    videoEl.pause();
    if (activeMediaStream) {
      activeMediaStream.getTracks().forEach((t) => t.stop());
      activeMediaStream = null;
    }
    if (currentObjectUrl) {
      URL.revokeObjectURL(currentObjectUrl);
      currentObjectUrl = null;
    }
    videoEl.srcObject = null;
    videoEl.removeAttribute('crossorigin');
    videoEl.removeAttribute('src');
    videoEl.load();
  }

  function dispatchBitmapFrameToWorker(item) {
    if (!item || item.epoch !== sourceEpoch) {
      if (item && item.bitmap && typeof item.bitmap.close === 'function') item.bitmap.close();
      return false;
    }
    if (!workerFrameBusy) {
      workerFrameBusy = true;
      try {
        worker.postMessage(
          {
            type: 'video_frame',
            epoch: item.epoch,
            seq: item.seq,
            bitmap: item.bitmap,
            dtSec: item.dtSec,
            mediaTimeSec: item.mediaTimeSec,
            tSrcUs: item.tSrcUs,
            captureAgeMs: item.captureAgeMs,
            enqueuedEpochMs: item.enqueuedEpochMs,
            gapMask: item.gapMask,
            geom: item.geom,
          },
          [item.bitmap]
        );
        return true;
      } catch (_postErr) {
        workerFrameBusy = false;
        if (item.bitmap && typeof item.bitmap.close === 'function') item.bitmap.close();
        pendingVideoGapMask = (pendingVideoGapMask | (item.gapMask || 0) | 1) & 0x3f;
        return false;
      }
    } else {
      if (queuedPendingFrame) {
        item.gapMask = ((item.gapMask || 0) | (queuedPendingFrame.gapMask || 0) | 1) & 0x3f;
        if (queuedPendingFrame.bitmap && typeof queuedPendingFrame.bitmap.close === 'function') {
          queuedPendingFrame.bitmap.close();
        }
        pumpDroppedFrames++;
      }
      queuedPendingFrame = item;
      return true;
    }
  }

  async function captureAndSendCurrentVideoFrame(
    dtSec,
    mediaTimeSec,
    expectedEpoch = sourceEpoch,
    metaExtras = null
  ) {
    const extras = metaExtras || {};
    const incomingGap = (extras.gapMask || 0) & 0x3f;
    if (videoEl.readyState < 2 || bitmapCaptureBusy || expectedEpoch !== sourceEpoch) {
      if (expectedEpoch === sourceEpoch) {
        const busyDropBit = bitmapCaptureBusy && videoEl.readyState >= 2 ? 1 : 0;
        const missedMask = (incomingGap | busyDropBit) & 0x3f;
        if (missedMask !== 0) {
          pendingVideoGapMask = (pendingVideoGapMask | missedMask) & 0x3f;
        }
      }
      return false;
    }
    bitmapCaptureBusy = true;
    const myCaptureToken = ++bitmapCaptureToken;
    activeCaptureToken = myCaptureToken;
    const effectiveGapMask = (pendingVideoGapMask | incomingGap) & 0x3f;
    pendingVideoGapMask = 0;
    const seq = ++frameSeq;
    const enqueuedEpochMs = performance.timeOrigin + performance.now();
    const tSrcUs = Number.isFinite(extras.tSrcUs) && extras.tSrcUs >= 0
      ? Math.round(extras.tSrcUs)
      : (Number.isFinite(mediaTimeSec) && mediaTimeSec >= 0
          ? Math.round(mediaTimeSec * 1_000_000)
          : null);
    const srcW = Math.max(0, Math.trunc(videoEl.videoWidth || 0));
    const srcH = Math.max(0, Math.trunc(videoEl.videoHeight || 0));
    const certifyVideo1to1 = (sampleW, sampleH) => {
      if (extras.sar1to1 === true) return true;
      if (typeof VideoFrame !== 'function') return false;
      let vf = null;
      try {
        vf = new VideoFrame(videoEl);
        return Boolean(
          vf?.visibleRect?.width > 0 &&
          vf.displayWidth === vf.visibleRect.width &&
          vf.displayHeight === vf.visibleRect.height &&
          sampleW === vf.visibleRect.width &&
          sampleH === vf.visibleRect.height
        );
      } catch (_vfErr) {
        return false;
      } finally {
        if (vf && typeof vf.close === 'function') {
          vf.close();
        }
      }
    };
    const commitCaptureMetadata = () => {
      if (typeof extras.presentedFrames === 'number' && Number.isFinite(extras.presentedFrames) && extras.presentedFrames >= 0) {
        lastPresentedFrames = extras.presentedFrames;
      }
      if (Number.isFinite(mediaTimeSec) && mediaTimeSec >= 0) {
        lastMediaTime = mediaTimeSec;
      }
      if (typeof extras.nowPerfMs === 'number' && Number.isFinite(extras.nowPerfMs) && extras.nowPerfMs > 0) {
        lastPumpPerfMs = extras.nowPerfMs;
      }
      // A capture timestamp is only implausible if it lies in the future. Do not compare it with the
      // rVFC `now` argument: that is the start of the rendering update, and a frame captured a few ms
      // after it (routine, and tens of ms under main-thread load) is still presented in that update.
      // Rejecting those frames swapped the reference to `now` and dragged rsVidClockOffsetUs by the
      // capture-to-`now` gap, which surfaced as false audio/video clock skew.
      const hasCaptureTime =
        typeof extras.captureTimeMs === 'number' &&
        Number.isFinite(extras.captureTimeMs) &&
        extras.captureTimeMs > 0 &&
        extras.captureTimeMs <= performance.now();
      const refPerf = hasCaptureTime
        ? extras.captureTimeMs
        : (typeof extras.nowPerfMs === 'number' && Number.isFinite(extras.nowPerfMs) && extras.nowPerfMs > 0
            ? extras.nowPerfMs
            : performance.now());
      if (Number.isFinite(mediaTimeSec) && mediaTimeSec >= 0 && Number.isFinite(refPerf) && refPerf > 0) {
        const obsOffsetUs = mediaTimeSec * 1_000_000.0 - refPerf * 1000.0;
        if (!rsVidClockValid || Math.abs(obsOffsetUs - rsVidClockOffsetUs) > 50_000.0) {
          rsVidClockOffsetUs = obsOffsetUs;
          rsVidClockValid = true;
        } else {
          rsVidClockOffsetUs += 0.15 * (obsOffsetUs - rsVidClockOffsetUs);
        }
        rsVidClockCaptureTimed = hasCaptureTime;
      }
    };
    let dispatched = false;
    try {
      const bitmap = await createImageBitmap(videoEl);
      if (expectedEpoch !== sourceEpoch) {
        bitmap.close();
        return false;
      }
      const preScaledFrame = !certifyVideo1to1(bitmap.width, bitmap.height);
      dispatched = dispatchBitmapFrameToWorker({
        epoch: expectedEpoch,
        seq,
        bitmap,
        dtSec,
        mediaTimeSec,
        tSrcUs,
        captureAgeMs: typeof extras.captureAgeMs === 'number' ? extras.captureAgeMs : -1.0,
        enqueuedEpochMs,
        gapMask: effectiveGapMask,
        geom: extras.geom || {
          visibleRect: {
            x: 0,
            y: 0,
            width: Math.max(1, bitmap.width || srcW || activeFrameW),
            height: Math.max(1, bitmap.height || srcH || activeFrameH),
          },
          preScaled: preScaledFrame,
        },
      });
      if (dispatched) {
        commitCaptureMetadata();
      } else if (expectedEpoch === sourceEpoch) {
        pendingVideoGapMask = (pendingVideoGapMask | effectiveGapMask | 1) & 0x3f;
      }
      return dispatched;
    } catch (_err) {
      if (expectedEpoch === sourceEpoch && !workerFrameBusy) {
        try {
          const capW = srcW > 0 ? srcW : activeFrameW;
          const capH = srcH > 0 ? srcH : activeFrameH;
          const fallbackPreScaled = srcW <= 0 || srcH <= 0 || !certifyVideo1to1(capW, capH);
          if (fallbackCapCanvas.width !== capW || fallbackCapCanvas.height !== capH) {
            fallbackCapCanvas.width = capW;
            fallbackCapCanvas.height = capH;
          }
          fallbackCapCtx.imageSmoothingEnabled = false;
          fallbackCapCtx.drawImage(videoEl, 0, 0, capW, capH);
          const img = fallbackCapCtx.getImageData(0, 0, capW, capH);
          const copy = new Uint8Array(img.data);
          workerFrameBusy = true;
          try {
            worker.postMessage(
              {
                type: 'video_rgba',
                epoch: expectedEpoch,
                seq,
                buffer: copy.buffer,
                dtSec,
                mediaTimeSec,
                tSrcUs,
                captureAgeMs: typeof extras.captureAgeMs === 'number' ? extras.captureAgeMs : -1.0,
                enqueuedEpochMs,
                gapMask: effectiveGapMask,
                fallbackScaled: fallbackPreScaled,
                geom: extras.geom || {
                  srcW: capW,
                  srcH: capH,
                  preScaled: fallbackPreScaled,
                },
              },
              [copy.buffer]
            );
            commitCaptureMetadata();
            dispatched = true;
            return true;
          } catch (_postErr) {
            workerFrameBusy = false;
            pendingVideoGapMask = (pendingVideoGapMask | effectiveGapMask | 1) & 0x3f;
          }
        } catch (_fallbackErr) {
          pendingVideoGapMask = (pendingVideoGapMask | effectiveGapMask | 1) & 0x3f;
        }
      } else if (expectedEpoch === sourceEpoch) {
        pendingVideoGapMask = (pendingVideoGapMask | effectiveGapMask | 1) & 0x3f;
      }
      return dispatched;
    } finally {
      if (activeCaptureToken === myCaptureToken) {
        bitmapCaptureBusy = false;
      }
    }
  }

  async function forwardUniqueVideoFrame(mediaTimeSec, nowPerfMs, pumpEpoch, rVfcMeta = null) {
    if (!videoPumpActive || pumpEpoch !== sourceEpoch || videoEl.readyState < 2 || videoEl.paused || videoEl.ended) {
      return false;
    }
    const hasValidPts = Number.isFinite(mediaTimeSec) && mediaTimeSec >= 0;
    const hasPresented = rVfcMeta && typeof rVfcMeta.presentedFrames === 'number' && Number.isFinite(rVfcMeta.presentedFrames);
    const presentedNow = hasPresented ? rVfcMeta.presentedFrames : -1;
    if (hasPresented) {
      if (lastPresentedFrames >= 0 && presentedNow === lastPresentedFrames) {
        return false;
      }
    } else if (hasValidPts && mediaTimeSec === lastMediaTime && (activeSourceMode === 'video' || activeSourceMode === 'live')) {
      return false;
    }

    let gapMask = 0;
    if (hasPresented && lastPresentedFrames >= 0 && presentedNow > lastPresentedFrames + 1) {
      gapMask |= 1; // FRAME_GAP_SEQ
    }

    if (activeMediaStream) {
      syncRsLiveAudioCapture();
    }
    // Detect <video loop> wrap-around or any backward seek (> 1 ms reversal) so tracks and PTS clocks reset cleanly
    if (hasValidPts && lastMediaTime >= 0 && mediaTimeSec + 1e-3 < lastMediaTime) {
      gapMask |= 8; // FRAME_GAP_SEEK_RESTART
      worker.postMessage({ type: 'lab_action', action: 'reset' });
      lastMediaTime = -1;
      lastPresentedFrames = -1;
      pendingVideoGapMask = 0;
      lastPumpPerfMs = 0;
      rsVidClockOffsetUs = 0.0;
      rsVidClockValid = false;
      rsVidClockCaptureTimed = false;
      rsAudioCaptureBaseUs = 0;
      rsAudioCaptureCursor = 0;
      rsAudioCaptureAnchored = false;
      rsAudioAnchorRefineLeft = 0;
      resetRsAudioSkewState();
      rsAudioLastSentTEndUs = 0;
      mp4AudioCursor = 0;
      mp4AudioPrimed = false;
      probeExciterCursor = 0;
      probeExciterPrimed = false;
    }

    if (hasValidPts && mp4AudioPcm && !activeMediaStream) {
      pumpMp4AudioToWorker(mediaTimeSec);
    } else if (hasValidPts && probeExciterActive && !mp4AudioPcm) {
      pumpProbeExciterAudioToWorker(mediaTimeSec);
    }

    const nomDt = 1.0 / Math.max(8.0, activeVideoFps);
    let dtSec = nomDt;
    if (hasValidPts && lastMediaTime >= 0 && mediaTimeSec > lastMediaTime) {
      const rawMediaDt = mediaTimeSec - lastMediaTime;
      dtSec = rawMediaDt > 0.5 ? nomDt : Math.max(0.002, rawMediaDt);
    } else if (lastPumpPerfMs > 0) {
      const rawPerfDt = (nowPerfMs - lastPumpPerfMs) * 0.001;
      dtSec = rawPerfDt > 0.5 ? nomDt : Math.max(0.002, rawPerfDt);
    }

    let captureAgeMs = -1.0;
    if (
      rVfcMeta &&
      typeof rVfcMeta.captureTime === 'number' &&
      Number.isFinite(rVfcMeta.captureTime) &&
      rVfcMeta.captureTime > 0 &&
      nowPerfMs >= rVfcMeta.captureTime
    ) {
      captureAgeMs = nowPerfMs - rVfcMeta.captureTime;
    }

    // Live MSE edge maintenance: preserve multi-segment HLS cushion (~10-22s) while bounding extreme drift (>35s)
    if (activeSourceMode === 'live' && activeMseMediaSource && videoEl.buffered.length > 0) {
      const bufEnd = videoEl.buffered.end(videoEl.buffered.length - 1);
      if (bufEnd - videoEl.currentTime > 35.0) {
        videoEl.currentTime = Math.max(0, bufEnd - 15.0);
      }
    }

    updateVideoTimeLabel();
    const captureTimeMs =
      rVfcMeta && typeof rVfcMeta.captureTime === 'number' && Number.isFinite(rVfcMeta.captureTime) && rVfcMeta.captureTime > 0
        ? rVfcMeta.captureTime
        : -1.0;
    return await captureAndSendCurrentVideoFrame(dtSec, hasValidPts ? mediaTimeSec : -1.0, pumpEpoch, {
      captureAgeMs,
      gapMask,
      presentedFrames: hasPresented ? presentedNow : -1,
      nowPerfMs,
      captureTimeMs,
    });
  }

  function startVideoPump() {
    videoPumpActive = true;
    lastMediaTime = -1;
    lastPresentedFrames = -1;
    pendingVideoGapMask = 0;
    lastPumpPerfMs = 0;
    rsVidClockOffsetUs = 0.0;
    rsVidClockValid = false;
    rsVidClockCaptureTimed = false;
    pumpStartPerfMs = performance.now();
    const myEpoch = sourceEpoch;

    if (typeof videoEl.requestVideoFrameCallback === 'function') {
      const onVideoFrame = (now, metadata) => {
        if (!videoPumpActive || myEpoch !== sourceEpoch) return;
        videoEl.requestVideoFrameCallback(onVideoFrame);
        forwardUniqueVideoFrame(metadata.mediaTime, now, myEpoch, metadata);
      };
      videoEl.requestVideoFrameCallback(onVideoFrame);
    } else {
      const onRafFallback = (now) => {
        if (!videoPumpActive || myEpoch !== sourceEpoch) return;
        requestAnimationFrame(onRafFallback);
        forwardUniqueVideoFrame(videoEl.currentTime, now, myEpoch, null);
      };
      requestAnimationFrame(onRafFallback);
    }
  }

  // ========================================================================
  // UI Bindings & Telemetry DOM Updates
  // ========================================================================
  let uiRecActive = false;
  let currentLabMode = 0;
  let uiCalArmed = 'none';

  function applyCalibrationUiState(m) {
    if (
      m &&
      typeof m.calSyncSeq === 'number' &&
      typeof calSyncSeq === 'number' &&
      m.calSyncSeq < calSyncSeq
    ) {
      return;
    }
    const nextArmed = m && m.calArmed ? m.calArmed : 'none';
    if (nextArmed === 'none' || nextArmed !== uiCalArmed) {
      uiCalMinDraftCount = 0;
    }
    uiCalArmed = nextArmed;
    lastCalArmedUi = uiCalArmed;
    const rawDraft = m && typeof m.calDraftCount === 'number' ? m.calDraftCount : 0;
    const effDraft = uiCalArmed === 'none' ? 0 : Math.max(rawDraft, uiCalMinDraftCount);
    dispatchDeckEvent({
      type: 'CAL_TOOL_SYNC',
      calArmed: uiCalArmed,
      calDraftCount: effDraft,
    });
    const rulerOn = uiCalArmed === 'ruler' || (uiCalArmed === 'none' && m.calMode === 'ruler');
    const quadOn = uiCalArmed === 'homography' || (uiCalArmed === 'none' && m.calMode === 'homography');
    dom.btnCalRuler.setAttribute('aria-pressed', String(rulerOn));
    dom.btnCalQuad.setAttribute('aria-pressed', String(quadOn));

    const isSurveyed = m.calProvenance === 'surveyed';
    let committedCal = { kind: 'PX', provenance: 'NONE', label: 'PX UNITS' };
    if (m.calMode === 'homography') {
      dom.calBadge.textContent = isSurveyed ? 'SURVEYED 4-PT H' : 'ASSUMED 4-PT H';
      committedCal = { kind: 'QUAD', provenance: isSurveyed ? 'SURVEYED' : 'ASSUMED', label: dom.calBadge.textContent };
    } else if (m.calMode === 'ruler') {
      dom.calBadge.textContent = isSurveyed ? 'SURVEYED RULER' : 'ASSUMED RULER';
      committedCal = { kind: 'RULER', provenance: isSurveyed ? 'SURVEYED' : 'ASSUMED', label: dom.calBadge.textContent };
    } else if (m.calMode === 'auto_g') {
      dom.calBadge.textContent = m.autoGLocked ? 'AUTO-g LOCKED' : 'AUTO-g 1/2 ARC';
      committedCal = { kind: 'AUTO_G', provenance: m.autoGLocked ? 'SURVEYED' : 'ASSUMED', label: dom.calBadge.textContent };
    } else {
      dom.calBadge.textContent = 'PX UNITS';
    }
    dom.calBadge.title = dom.calBadge.textContent;
    if (
      uiCalArmed === 'none' &&
      (instrumentState.calibration.committed.kind !== committedCal.kind ||
        instrumentState.calibration.committed.label !== committedCal.label)
    ) {
      dispatchInstrument({
        type: 'CAL_COMMITTED',
        sourceEpoch: instrumentState.source.epoch,
        continuityEpoch: instrumentState.runtime_health.continuityEpoch,
        committed: committedCal,
      }, false);
    }

    if (typeof m.summary === 'string') {
      dom.calStatusLive.textContent = m.summary;
    }

    const syncInputVal = (inputEl, numVal) => {
      if (typeof numVal === 'number' && Number.isFinite(numVal) && document.activeElement !== inputEl) {
        inputEl.value = numVal.toFixed(2);
        inputEl.setCustomValidity('');
        syncAria(inputEl, false);
      }
    };
    syncInputVal(dom.calRulerM, m.rulerMeters);
    syncInputVal(dom.calPlaneW, m.planeW);
    syncInputVal(dom.calPlaneH, m.planeH);
  }

  let lastRecStateKey = 'idle';
  function updateRecorderUi(active, frames, durationSec) {
    uiRecActive = Boolean(active);
    const btnRec = document.getElementById('btnRecToggle');
    btnRec.classList.toggle('active', uiRecActive);
    btnRec.setAttribute('aria-pressed', String(uiRecActive));
    const nextBtnTxt = uiRecActive ? '■ Pause Recording' : '● Start Recording';
    if (btnRec.textContent !== nextBtnTxt) btnRec.textContent = nextBtnTxt;
    const dur = typeof durationSec === 'number' ? durationSec.toFixed(1) : '0.0';
    const nextStatusTxt = uiRecActive
      ? `● REC ${frames} FRAMES (${dur}s)`
      : frames > 0
      ? `PAUSED (${frames} FRAMES / ${dur}s)`
      : 'IDLE (0 FRAMES)';
    if (dom.recStatusLabel.textContent !== nextStatusTxt) {
      dom.recStatusLabel.textContent = nextStatusTxt;
    }
    dispatchInstrument({ type: 'REC_STATE', active: uiRecActive, frames: Number(frames) || 0, durationSec: Number(durationSec) || 0 }, false);
    const stateKey = uiRecActive ? 'rec' : frames > 0 ? 'paused' : 'idle';
    if (stateKey !== lastRecStateKey) {
      lastRecStateKey = stateKey;
      const srEl = document.getElementById('recSrAnnounce');
      if (srEl) {
        srEl.textContent =
          stateKey === 'rec'
            ? 'Session recording started'
            : stateKey === 'paused'
            ? `Session recording paused at ${frames} frames`
            : 'Session recorder cleared';
      }
    }
  }

  function triggerBlobDownload(filename, mimeType, textContent) {
    window.__lastTelemetryExport = { filename, mimeType, length: textContent.length, preview: textContent.slice(0, 512), fullText: textContent };
    const blob = new Blob([textContent], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  function formatExportTimestamp() {
    return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  }

  let lastPersistDomMs = 0;
  let lastPersistPeakSeen = -1;
  let lastPersistUniqueSeen = -1;

  function formatClockMmSs(sec) {
    const s = Math.max(0, Math.floor(sec || 0));
    const m = Math.floor(s / 60);
    const rem = s % 60;
    return `${String(m).padStart(2, '0')}:${String(rem).padStart(2, '0')}`;
  }

  function updatePersistenceDomUi(p, activeTargets) {
    if (!p || !dom.kpiPersistSpan) return;
    const nowMs = performance.now();
    const peakChanged = p.peakSimultaneous !== lastPersistPeakSeen;
    const uniqChanged = p.uniqueIds !== lastPersistUniqueSeen;
    const resetOccurred = p.uniqueIds < lastPersistUniqueSeen || p.sessionElapsedSec < 0.25;
    if (!resetOccurred && !peakChanged && !uniqChanged && nowMs - lastPersistDomMs < 1000) {
      return;
    }
    lastPersistDomMs = nowMs;
    lastPersistPeakSeen = p.peakSimultaneous;
    lastPersistUniqueSeen = p.uniqueIds;

    setTextIfChanged(dom.persistSummaryMeta, `DORMANT ARCHIVE: ${p.dormantActive} / 32`);
    setTextIfChanged(dom.kpiPersistSpan, formatClockMmSs(p.sessionElapsedSec));
    setTextIfChanged(dom.kpiPersistActivePeak, `${activeTargets} / ${p.peakSimultaneous}`);
    setTextIfChanged(dom.kpiPersistUnique, String(p.uniqueIds));
    setTextIfChanged(dom.kpiPersistRelocks, String(p.totalRelocks));
    if (p.longestTenureTid > 0 && p.longestTenureFrames > 0) {
      const sec = p.longestTenureSec || 0;
      const durTxt = sec >= 60 ? formatClockMmSs(sec) : `${sec.toFixed(sec < 10 ? 1 : 0)}s`;
      setTextIfChanged(dom.kpiPersistLongest, `T${p.longestTenureTid} (${durTxt})`);
    } else {
      setTextIfChanged(dom.kpiPersistLongest, '--');
    }
    setTextIfChanged(dom.kpiPersistSustained, `${p.sustained10s} / ${p.sustained60s}`);
  }

  const PHYS_REGIME_LABELS = [
    'INERTIAL CV',
    'BALLISTIC',
    'AERO DRAG',
    'POWERED BOOST',
    'PARACHUTE',
    'PENDULUM HARM',
    'MICRO TREMOR',
    'COLLISIONAL',
    'CURVED CTRV',
  ];
  const PHYS_JUMP_LABELS = ['NONE', 'PEG JUMP', 'BURNOUT', 'CANOPY', 'COLLISION'];

  let lastPhysicsDomMs = 0;
  let lastPhysicsActiveSeen = null;
  let lastPhysicsRegimeSeen = -1;
  let lastPhysicsJumpSeen = -1;
  let lastPhysicsPairFrameSeen = -1;

  function fmtPhysicsCompact(v) {
    if (!Number.isFinite(v)) return '--';
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a < 1e-3 || a >= 10000) return v.toExponential(1);
    if (a < 0.1) return v.toFixed(4);
    if (a < 10) return v.toFixed(2);
    return v.toFixed(1);
  }

  function updatePhysicsIdDomUi(pid, m) {
    if (!pid || !dom.physicsOdeEq) return;
    const nowMs = performance.now();
    const has = Boolean(pid.hasPrimary);
    const regCode = Math.max(0, Math.min(8, Math.round(pid.physRegime || 0)));
    const jCode = Math.max(0, Math.min(4, Math.round(pid.constraintJumpCode || 0)));
    const pairFrame = Math.round(pid.lastPairFrame || 0);
    const resetOccurred = Boolean(m && m.sessionPersistence && m.sessionPersistence.sessionElapsedSec < 0.25);
    const stateChanged =
      has !== lastPhysicsActiveSeen ||
      regCode !== lastPhysicsRegimeSeen ||
      jCode !== lastPhysicsJumpSeen ||
      pairFrame !== lastPhysicsPairFrameSeen;

    if (!resetOccurred && !stateChanged && nowMs - lastPhysicsDomMs < 250) {
      return;
    }
    lastPhysicsDomMs = nowMs;
    lastPhysicsActiveSeen = has;
    lastPhysicsRegimeSeen = regCode;
    lastPhysicsJumpSeen = jCode;
    lastPhysicsPairFrameSeen = pairFrame;

    const knots = Math.max(0, Math.round(pid.fistaSharpKnots || 0));
    const r2 = Number.isFinite(pid.odeR2) ? pid.odeR2 : 0;
    const confPct = Math.round((Number.isFinite(pid.physRegimeConf) ? pid.physRegimeConf : 0) * 100);
    const unitTxt = pid.unitMode === 'metric' ? 'METRIC' : (pid.unitMode === 'out_of_plane' ? 'OUT-OF-PLANE' : 'PX');

    setTextIfChanged(dom.physicsIdMeta, `ODE R² ${has ? r2.toFixed(3) : '--'} | KNOTS ${knots} | ${unitTxt}`);

    const regName = (pid.physRegimeName || PHYS_REGIME_LABELS[regCode] || 'INERTIAL CV').replace(/_/g, ' ');
    setTextIfChanged(dom.physRegimeBadge, `R${regCode} · ${regName} (${confPct}%)`);
    if (dom.physRegimeBadge.dataset.active !== (has ? '1' : '0')) {
      dom.physRegimeBadge.dataset.active = has ? '1' : '0';
    }
    if (dom.physRegimeBadge.dataset.regime !== String(regCode)) {
      dom.physRegimeBadge.dataset.regime = String(regCode);
    }

    const jName = (pid.constraintJumpName || PHYS_JUMP_LABELS[jCode] || 'NONE').replace(/_/g, ' ');
    setTextIfChanged(dom.physJumpBadge, `J${jCode} · ${jName}`);
    if (dom.physJumpBadge.dataset.jump !== String(jCode)) {
      dom.physJumpBadge.dataset.jump = String(jCode);
    }

    const eq = has && pid.symbolicOde ? String(pid.symbolicOde) : "x'' = 0, y'' = 0";
    setTextIfChanged(dom.physicsOdeEq, eq);
    if (dom.physicsOdeEq.title !== eq) {
      dom.physicsOdeEq.title = eq;
    }

    const wCv = Number.isFinite(pid.immWcv) ? pid.immWcv : 0;
    const wBal = Number.isFinite(pid.immWbal) ? pid.immWbal : 0;
    const wTo = Number.isFinite(pid.immWturnOsc) ? pid.immWturnOsc : 0;
    setTextIfChanged(dom.kpiPhysicsR2, has ? r2.toFixed(3) : '--');
    setTextIfChanged(dom.kpiPhysicsConf, has ? `${confPct}% (R${regCode})` : '0%');
    setTextIfChanged(dom.kpiImmWeights, has ? `${wCv.toFixed(2)}/${wBal.toFixed(2)}/${wTo.toFixed(2)}` : '0.33/0.33/0.34');

    const disp = pid.display || {};
    const betaRaw = pid.aeroDragBeta || 0;
    const betaVal = Number.isFinite(disp.beta) ? disp.beta : betaRaw;
    const betaUnit = disp.betaUnit || 'px⁻¹';
    setTextIfChanged(dom.kpiPhysicsBeta, has && betaRaw > 1e-7 ? `${fmtPhysicsCompact(betaVal)} ${betaUnit}` : '--');

    const thrustRaw = pid.thrustAccel || 0;
    const thrustVal = Number.isFinite(disp.thrust) ? disp.thrust : thrustRaw;
    const thrustUnit = disp.thrustUnit || 'px/s²';
    setTextIfChanged(dom.kpiPhysicsThrust, has && thrustRaw > 1e-4 ? `${fmtPhysicsCompact(thrustVal)} ${thrustUnit}` : '--');

    const zeta = pid.dampingZeta || 0;
    const qFac = pid.qualityFactorQ || 0;
    const qTxt = qFac >= 998.5 ? 'INF' : qFac.toFixed(1);
    setTextIfChanged(dom.kpiPhysicsDamping, has && (zeta > 0 || qFac > 0) ? `${zeta.toFixed(3)} / ${qTxt}` : '-- / --');

    const tEll = pid.ellipticPeriodS || 0;
    setTextIfChanged(dom.kpiPhysicsEllipticT, has && tEll > 0 ? `${tEll.toFixed(3)} s` : '-- s');

    const oscN = Math.max(0, Math.round(pid.kuramotoOscCount || 0));
    const rSync = pid.kuramotoOrderR || 0;
    setTextIfChanged(dom.kpiKuramotoSync, oscN >= 2 ? `${rSync.toFixed(2)} / ${oscN}` : `-- / ${oscN}`);

    const lyap = pid.lyapunovDivRate || 0;
    setTextIfChanged(dom.kpiLyapunovMax, has && lyap > 0 ? `${lyap.toFixed(3)} s⁻¹` : '-- s⁻¹');

    const tdRaw = pid.taubinVsKasaDeltaPx || 0;
    const tdVal = Number.isFinite(disp.taubinDelta) ? disp.taubinDelta : tdRaw;
    const tdUnit = disp.taubinDeltaUnit || 'px';
    setTextIfChanged(dom.kpiTaubinDelta, has && tdRaw > 0 ? `${fmtPhysicsCompact(tdVal)} ${tdUnit}` : '-- px');

    const kapRaw = pid.taubinCurvatureInvPx || 0;
    const kapVal = Number.isFinite(disp.curvature) ? disp.curvature : kapRaw;
    const kapUnit = disp.curvatureUnit || 'px⁻¹';
    setTextIfChanged(dom.kpiTaubinKappa, has && kapRaw > 1e-6 ? `${fmtPhysicsCompact(kapVal)} ${kapUnit}` : '--');

    setTextIfChanged(dom.kpiFistaKnots, String(knots));

    const restE = pid.normalRestitutionE || 0;
    if (pairFrame > 0 && (pid.lastPairMassRatio || 0) > 0) {
      setTextIfChanged(dom.kpiTwoBodyMassE, `${pid.lastPairMassRatio.toFixed(2)} · ${restE.toFixed(2)}`);
      const retPct = Math.round((pid.lastPairEnergyRetention || 0) * 100);
      setTextIfChanged(dom.kpiTwoBodyMomRes, `${(pid.lastPairMomResid || 0).toFixed(3)} / ${retPct}%`);
    } else if (has && restE > 0) {
      setTextIfChanged(dom.kpiTwoBodyMassE, `-- · ${restE.toFixed(2)}`);
      setTextIfChanged(dom.kpiTwoBodyMomRes, '--');
    } else {
      setTextIfChanged(dom.kpiTwoBodyMassE, '--');
      setTextIfChanged(dom.kpiTwoBodyMomRes, '--');
    }

    const eVal = Number.isFinite(disp.energy) ? disp.energy : (pid.specificEnergy || 0);
    const lzVal = Number.isFinite(disp.angMom) ? disp.angMom : (pid.specificAngMom || 0);
    setTextIfChanged(dom.kpiPhysicsConserv, has ? `${fmtPhysicsCompact(eVal)} | ${fmtPhysicsCompact(lzVal)}` : '--');
  }

  let lastSwarmActive = false;
  let lastSwarmFrameW = 320;
  let lastSwarmFrameH = 180;
  let lastCalArmedUi = 'none';

  function formatValidityMaskLabel(mask) {
    const m = Number(mask) | 0;
    const flags = [];
    if (m & 1) flags.push('GRAD');
    if (m & 2) flags.push('FTLE');
    if (m & 4) flags.push('RIGID');
    if (m & 8) flags.push('MODAL');
    if (m & 16) flags.push('KURA');
    return flags.length ? flags.join(' · ') : 'NONE';
  }

  function fmtFixedOrDash(v, digits) {
    return typeof v === 'number' && Number.isFinite(v) ? v.toFixed(digits) : '--';
  }

  function fmtIntOrDash(v) {
    return typeof v === 'number' && Number.isFinite(v) ? Math.round(v).toLocaleString('en-US') : '--';
  }

  function publishInspectTelemetryState(c1k) {
    const ackSrc = c1k && typeof c1k === 'object' ? {
      inspectSeq: Number.isInteger(c1k.inspectSeq) ? c1k.inspectSeq : (Number.isInteger(c1k.inspectResSeq) ? c1k.inspectResSeq : 0),
      inspectReqNormX: c1k.inspectReqNormX,
      inspectReqNormY: c1k.inspectReqNormY,
      inspectReqPxX: c1k.inspectReqPxX ?? c1k.inspectReqX,
      inspectReqPxY: c1k.inspectReqPxY ?? c1k.inspectReqY,
      inspectStatus: c1k.inspectStatus,
      inspectedTid: c1k.inspectedTid,
    } : null;
    const inspReqSeq = c1k && Number.isInteger(c1k.inspectReqSeq) ? c1k.inspectReqSeq : (ackSrc ? ackSrc.inspectSeq : 0);
    if (typeof dispatchDeckEvent === 'function') {
      dispatchDeckEvent({ type: 'INSPECT_ACK', ack: ackSrc });
    }
    const ack = deckUiState && deckUiState.lastInspectAck
      ? deckUiState.lastInspectAck
      : normalizeInspectAck(ackSrc);
    if (dom.swarmInspectReadout) {
      const ds = dom.swarmInspectReadout.dataset || (dom.swarmInspectReadout.dataset = {});
      ds.inspectSeq = String(ack.inspectSeq);
      ds.inspectReqSeq = String(inspReqSeq);
      ds.inspectResSeq = String(ack.inspectSeq);
      ds.inspectStatus = ack.inspectStatus;
      ds.inspectTid = String(ack.inspectedTid);
      ds.inspectReqPxX = ack.inspectReqPxX !== null ? ack.inspectReqPxX.toFixed(3) : '';
      ds.inspectReqPxY = ack.inspectReqPxY !== null ? ack.inspectReqPxY.toFixed(3) : '';
      ds.inspectReqNormX = ack.inspectReqNormX !== null ? ack.inspectReqNormX.toFixed(6) : '';
      ds.inspectReqNormY = ack.inspectReqNormY !== null ? ack.inspectReqNormY.toFixed(6) : '';
    }
    if (typeof window !== 'undefined') {
      window.__chronosInspectState = ack;
    }
  }

  function updateContinuumDomUi(c1k, m) {
    if (!dom.continuumSummaryMeta) return;
    publishInspectTelemetryState(c1k);
    const kernelMs = Number(c1k && Number.isFinite(c1k.kernelComputeMedianMs) ? c1k.kernelComputeMedianMs : m.wasmMedianMs) || 0.0;
    const pipeMs = Number(c1k && Number.isFinite(c1k.partialPipelineLatencyMs) ? c1k.partialPipelineLatencyMs : m.partialPipelineLatencyMs) || kernelMs;
    setTextIfChanged(dom.kpiLatencySplit, `${kernelMs.toFixed(2)} ms | ${pipeMs.toFixed(1)} ms`);

    if (!c1k || !c1k.available) {
      setTextIfChanged(dom.continuumSummaryMeta, 'SWARM: STANDBY · LEGACY WASM');
      return;
    }

    const diagTag = c1k.diagnosticsValid === false ? ' · DIAG INVALID' : '';
    const modeTag = c1k.swarmActive
      ? `SWARM: ACTIVE · ${(c1k.fieldBackend || 'webgl2').toUpperCase()}${diagTag}`
      : (c1k.swarmMode === 2 ? `SWARM: FORCED WAIT${diagTag}` : (c1k.swarmMode === 1 ? `SWARM: AUTO STANDBY${diagTag}` : `SWARM: OFF · HERO 16${diagTag}`));
    setTextIfChanged(dom.continuumSummaryMeta, modeTag);
    setTextIfChanged(
      dom.kpiSwarmConfirmed,
      `${fmtIntOrDash(c1k.confirmedCount)} / ${fmtIntOrDash(c1k.coastingCount)}`
    );
    setTextIfChanged(dom.kpiSwarmPeak, fmtIntOrDash(c1k.peakSimultaneous));
    setTextIfChanged(dom.kpiSwarmUnique, fmtIntOrDash(c1k.uniqueTracks));

    const mask = Number.isInteger(c1k.validityMask) ? c1k.validityMask : 0;
    const hasValidatedTopBody = Boolean(
      (mask & 4) &&
      c1k.topRigidBodyValid &&
      Number.isInteger(c1k.rigidBodyCount) &&
      c1k.rigidBodyCount > 0 &&
      Number.isInteger(c1k.topRigidMembers) &&
      c1k.topRigidMembers >= 1 &&
      Number.isFinite(c1k.topRigidOmegaRadS)
    );
    if (hasValidatedTopBody) {
      const wSign = c1k.topRigidOmegaRadS >= 0 ? '+' : '';
      setTextIfChanged(dom.kpiRigidBodies, `${c1k.rigidBodyCount} | ${wSign}${c1k.topRigidOmegaRadS.toFixed(2)} rad/s`);
    } else if ((mask & 4) && Number.isInteger(c1k.rigidBodyCount) && c1k.rigidBodyCount === 0 && (c1k.validRigidBodyCount ?? 0) === 0) {
      setTextIfChanged(dom.kpiRigidBodies, '0 | -- rad/s');
    } else {
      setTextIfChanged(dom.kpiRigidBodies, '-- | -- rad/s');
    }

    const vortValid = Number.isFinite(c1k.maxVorticityS1) && Number.isFinite(c1k.rmsVorticityS1);
    if ((mask & 1) && vortValid) {
      setTextIfChanged(dom.kpiContinuumVort, `${c1k.maxVorticityS1.toFixed(2)} / ${c1k.rmsVorticityS1.toFixed(2)} s⁻¹`);
    } else if (c1k.swarmActive && vortValid) {
      setTextIfChanged(dom.kpiContinuumVort, `RAW ${c1k.maxVorticityS1.toFixed(2)} / ${c1k.rmsVorticityS1.toFixed(2)} s⁻¹`);
    } else {
      setTextIfChanged(dom.kpiContinuumVort, '-- / -- s⁻¹');
    }

    if ((mask & 2) && Number.isFinite(c1k.maxFtleS1) && c1k.maxFtleS1 > -900.0) {
      setTextIfChanged(dom.kpiContinuumFtle, `${c1k.maxFtleS1.toFixed(3)} s⁻¹ (${fmtIntOrDash(c1k.ftleCells)}c)`);
    } else if (c1k.swarmActive) {
      setTextIfChanged(dom.kpiContinuumFtle, `WARM ${fmtIntOrDash(c1k.ftleMatureTracks)}t`);
    } else {
      setTextIfChanged(dom.kpiContinuumFtle, '-- s⁻¹');
    }

    const modalValidNums = Number.isFinite(c1k.modalDomFreqHz) && Number.isFinite(c1k.modalR2);
    if ((mask & 8) && modalValidNums && c1k.modalDomFreqHz > 0.05) {
      setTextIfChanged(dom.kpiModalField, `${c1k.modalDomFreqHz.toFixed(2)} Hz | R₂ ${c1k.modalR2.toFixed(2)}`);
    } else if (c1k.swarmActive && modalValidNums && c1k.modalDomFreqHz > 0.05) {
      setTextIfChanged(dom.kpiModalField, `RAW ${c1k.modalDomFreqHz.toFixed(2)} Hz | R₂ ${c1k.modalR2.toFixed(2)}`);
    } else if (c1k.swarmActive) {
      setTextIfChanged(dom.kpiModalField, `WARM ${fmtIntOrDash(c1k.modalValidSensors)}s | R₂ ${fmtFixedOrDash(c1k.modalR2, 2)}`);
    } else {
      setTextIfChanged(dom.kpiModalField, '-- Hz | --');
    }

    if (c1k.inspectStatus === 'expired') {
      setTextIfChanged(dom.swarmInspectReadout, 'TRACK EXPIRED BEFORE CAUSAL FRAME · CLICK LIVE TRACK');
    } else if (c1k.inspectStatus === 'miss') {
      setTextIfChanged(dom.swarmInspectReadout, 'NO ACTIVE TRACK AT POINT · CLICK LIVE TRACK');
    } else if (c1k.inspectStatus === 'unavailable') {
      setTextIfChanged(dom.swarmInspectReadout, 'FIELD INSPECTION UNAVAILABLE (LEGACY WASM)');
    } else if (c1k.inspectedTrack && Number.isFinite(c1k.inspectedTrack.x) && Number.isFinite(c1k.inspectedTrack.y)) {
      const it = c1k.inspectedTrack;
      const bTxt = it.bodyId > 0 ? ` · B${it.bodyId}` : '';
      const spd = Number.isFinite(it.vx) && Number.isFinite(it.vy) ? fmtFixedOrDash(Math.hypot(it.vx, it.vy), 1) : '--';
      setTextIfChanged(
        dom.swarmInspectReadout,
        `INSP μT${it.tid}${bTxt} · (${it.x.toFixed(1)}, ${it.y.toFixed(1)}) · ${spd} px/s · FTLE ${fmtFixedOrDash(it.ftle, 2)}`
      );
    } else if (c1k.inspectedTid > 0) {
      const bTxt = c1k.inspectedBodyId > 0 ? ` · BODY #${c1k.inspectedBodyId}` : '';
      setTextIfChanged(dom.swarmInspectReadout, `INSP μT${c1k.inspectedTid}${bTxt}`);
    } else {
      setTextIfChanged(dom.swarmInspectReadout, 'CLICK HUD CANVAS TO INSPECT FIELD OR TRACK');
    }

    setTextIfChanged(dom.diagAuctionSteps, `${fmtIntOrDash(c1k.auctionSteps)} (↻${fmtIntOrDash(c1k.reacqEvents)})`);
    setTextIfChanged(dom.diagRawResiduals, `${fmtFixedOrDash(c1k.meanIcLkRes, 3)} | ${fmtFixedOrDash(c1k.topRigidSe2RmsePx, 2)} px`);
    setTextIfChanged(dom.diagValidityMask, formatValidityMaskLabel(mask));
    setTextIfChanged(dom.diagFtleWindow, `${fmtIntOrDash(c1k.ftleMatureTracks)} trk / ${fmtIntOrDash(c1k.ftleCells)} cells`);
    setTextIfChanged(dom.diagSparseGraph, `${fmtIntOrDash(c1k.knnEdgeCount)} / ${fmtIntOrDash(c1k.peakCount)}`);
    setTextIfChanged(dom.diagKuramotoCom, `${fmtFixedOrDash(c1k.kuramotoR, 2)} | ${fmtFixedOrDash(c1k.velDispersion, 1)} px/s`);
    setTextIfChanged(dom.diagFieldBackend, `${(c1k.fieldBackend || 'none').toUpperCase()} · ${fmtFixedOrDash(c1k.fieldRenderMs, 2)} ms`);
    setTextIfChanged(dom.diagMeanStrain, `${fmtFixedOrDash(c1k.meanStrain, 4)} s⁻¹`);
  }

  function applyResolutionUi(mode, w, h, pDim) {
    activeResMode = mode ? 1 : 0;
    activeFrameW = w || (activeResMode ? 640 : 320);
    activeFrameH = h || (activeResMode ? 360 : 180);
    const pipSz = pDim || (activeResMode ? 64 : 32);
    dom.inputResMeta.textContent = `${activeFrameW}×${activeFrameH} ${activeResMode ? 'HD' : 'SD'} RGBA`;
    pipEl.setAttribute('aria-label', `${pipSz}x${pipSz} raw vs Eulerian temporal-contrast picture-in-picture`);
    document.querySelectorAll('.res-btn').forEach((b) => {
      const on = Number(b.dataset.res) === activeResMode;
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', String(on));
    });
    if (instrumentState.resolution_roi.resMode !== activeResMode) {
      dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', resolution_roi: { resMode: activeResMode }, markPresetModified: false }, false);
    }
  }

  function updateWave1DomUi(m) {
    const w1 = m.wave1;
    const roi = m.roi;
    const bos = m.bos;
    const met = m.metrology;
    window.__chronosLastWave1Snap = {
      frameIdx: m.frameIdx,
      activeTargets: m.activeTargets,
      wave1: w1,
      roi,
      bos,
      metrology: met,
    };

    if (roi && dom.roiGeomMeta) {
      const roiTxt = roi.native1to1Active
        ? `1:1 NATIVE (${roi.cropX0},${roi.cropY0} ${roi.cropW}x${roi.cropH})`
        : (roi.fallbackScale ? 'FALLBACK RESCALE (SRC < ROI)' : `FIT ${roi.cropW}x${roi.cropH}->${roi.engineW}x${roi.engineH}`);
      setTextIfChanged(dom.roiGeomMeta, roiTxt);
      if (dom.valRoiX0 && dom.rangeRoiX0) {
        const pctX = Math.round(Number(dom.rangeRoiX0.value) || 0);
        setTextIfChanged(dom.valRoiX0, `${pctX}% (${roi.cropX0} px)`);
      }
      if (dom.valRoiY0 && dom.rangeRoiY0) {
        const pctY = Math.round(Number(dom.rangeRoiY0.value) || 0);
        setTextIfChanged(dom.valRoiY0, `${pctY}% (${roi.cropY0} px)`);
      }
      setTextIfChanged(dom.diagRoiGeom, roiTxt);
    }

    if (w1 && w1.available) {
      const dm = w1.dispMeta;
      const gapTag = dm && dm.frameGapFlag !== 0 && dm.gapLabels && dm.gapLabels.length
        ? dm.gapLabels.join('|')
        : 'OK';
      const seqTxt = dm ? `${dm.sourceEpoch}:${dm.sourceSeq}` : `0:${m.frameIdx || 0}`;
      setTextIfChanged(
        dom.diagWave1Ingest,
        `EP ${w1.continuityEpoch} · ${seqTxt} · ${gapTag} · ${w1.totalCommits}c`
      );
      const st = w1.stamps || {};
      const sc = (s) => (!s || !s.valid ? '-' : (s.held ? 'H' : 'V'));
      const stampTxt = `${sc(st.micro)}${sc(st.grad)}${sc(st.ftle)}${sc(st.modalField)}${sc(st.heroPhysics)}${sc(st.bos)} · ${w1.validRowCount}r`;
      setTextIfChanged(dom.diagWave1Stamps, stampTxt);
    }

    if (bos && dom.diagBosState) {
      const bosResTxt = bos.available && bos.state === 2
        ? `READY · ${bos.validDisp}/32768 · ${bos.pcgIters}it · res ${bos.trueRelRes.toExponential(1)}`
        : bos.available && bos.state === 3
        ? `DISTURBED · ${(bos.reasonFlags || []).slice(0, 3).join('+') || 'MOTION'}`
        : bos.available && bos.state === 1
        ? `WARMING ${bos.warmupCnt}/16`
        : (bos.available ? 'OFF' : 'STANDBY');
      setTextIfChanged(dom.diagBosState, bosResTxt);
    }

    if (met && dom.diagMetrologyCov) {
      const covTxt = met.available
        ? `${met.usableCovCount} usable · ${met.loomRejectedCount} loom · ${met.promotedCandidates} cand`
        : 'STANDBY';
      setTextIfChanged(dom.diagMetrologyCov, covTxt);
    }

    if (dom.acquisitionStatusBanner) {
      let severity = 'info';
      let bannerTxt = 'ACQUISITION STEADY · MONOTONIC PTS & 100% RING COMMIT';
      const dm = w1 && w1.dispMeta;
      const bosActive = bos && bos.available && bos.reqMode > 0;
      const w2 = m.wave2;
      const w2v = w2 && w2.vibro && w2.vibro.available ? w2.vibro : null;
      const w2vActive = Boolean(w2v && w2v.reqMode > 0);
      const w2h = w2 && w2.hodge && w2.hodge.available ? w2.hodge : null;
      const w2hActive = Boolean(w2h && w2h.reqMode > 0);
      const w2c = w2 && w2.conservation && w2.conservation.available ? w2.conservation : null;
      const w2w = w2 && w2.wind && w2.wind.available ? w2.wind : null;
      const w2wActive = Boolean(w2w && w2w.reqMode > 0);
      const w2e = w2 && w2.elasto && w2.elasto.available ? w2.elasto : null;
      const w2eActive = Boolean(w2e && w2e.reqMode > 0);
      const w2vNull = Boolean(w2v && (w2v.exposureNullFlag || ((w2v.reasonMask || 0) & 64) !== 0));
      const w2vHardFail = Boolean(w2v && (((w2v.reasonMask || 0) & ((1 << 4) | (1 << 5))) !== 0));
      if (bosActive && bos.state === 3) {
        severity = 'danger';
        const reasons = (bos.reasonFlags || []).join(', ') || 'CAMERA MOTION';
        bannerTxt = `⚠ TRIPOD FAILURE / DISTURBED (${reasons}) · SHIFT (${bos.shiftX.toFixed(2)}, ${bos.shiftY.toFixed(2)}) px · REF INVALIDATED`;
      } else if (w2eActive && w2e.state === 4) {
        severity = 'warn';
        const eWhy = formatReasonSummary(w2e.reasonMask || 0, ELASTO_REASON_BITS, 2).text || 'Check calibration or scratch mutex';
        bannerTxt = `✕ ELASTOGRAPHY REFUSED · ${eWhy}`;
      } else if (w2wActive && w2w.state === 4) {
        severity = 'warn';
        const wWhy = formatReasonSummary(w2w.reasonMask || 0, WIND_REASON_BITS, 2).text || 'Check flow continuum or scratch mutex';
        bannerTxt = `✕ WIND TUNNEL REFUSED · ${wWhy}`;
      } else if (w2vActive && (w2v.state === 4 || w2vNull)) {
        severity = w2vHardFail ? 'danger' : 'warn';
        const sincAbs = Number.isFinite(w2v.sincH) ? Math.abs(w2v.sincH).toFixed(3) : '--';
        const rTxt = (w2v.reasonFlags || []).slice(0, 3).join('+') || (w2vNull ? 'EXPOSURE_NULL' : 'DISTURBED');
        bannerTxt = w2vNull
          ? `⚠ RS EXPOSURE_NULL (|H|=${sincAbs} < 0.05) · ${rTxt}`
          : `⚠ RS VIBRO DISTURBED (${rTxt})`;
      } else if (w2hActive && w2h.state === 3) {
        severity = 'warn';
        bannerTxt = '⚠ HODGE PCG UNCONVERGED · FIELD INVALIDATED';
      } else if (w2c && w2c.status === 3) {
        severity = 'warn';
        bannerTxt = `⚠ OCCLUDED CONSERVATION VIOLATED · T${w2c.tidA || 0}-T${w2c.tidB || 0}`;
      } else if (dm && dm.frameGapFlag !== 0) {
        severity = 'warn';
        bannerTxt = `⚠ ACQUISITION GAP [${(dm.gapLabels || []).join('+')}] · EP ${w1.continuityEpoch} · PTS MONOTONIC GUARD`;
      } else if (roi && roi.fallbackScale) {
        severity = 'warn';
        bannerTxt = `⚠ NATIVE 1:1 UNAVAILABLE (${roi.srcW}x${roi.srcH} < ${roi.engineW}x${roi.engineH}) · RESCALED`;
      } else if (w2eActive && w2e.state === 1) {
        severity = 'warn';
        const eCalWhy = formatReasonSummary(w2e.reasonMask || 0, ELASTO_REASON_BITS, 2).text || 'Check RS timing and audio';
        bannerTxt = `◐ ELASTOGRAPHY CALIBRATION REQUIRED · ${eCalWhy}`;
      } else if (w2vActive && w2v.state === 1) {
        severity = 'warn';
        bannerTxt = `⚠ RS CALIBRATION REQUIRED (${(w2v.reasonFlags || []).slice(0, 2).join('+') || 'BAD_CALIB'})`;
      } else if (w2eActive && w2e.state === 3) {
        severity = 'ok';
        bannerTxt = w2e.mode === 4
          ? `◆ LOCK-IN ACOUSTIC BOS READY · ${(w2e.f0 || 0).toFixed(0)} Hz · |Phi| RMS ${(w2e.abosPhiRms || 0).toFixed(3)} · CURL ${(w2e.abosCurlRms || 0).toFixed(3)}`
          : `◆ ELASTOGRAPHY READY · ${w2e.classLabel || 'WAVE'} · a=${(w2e.alpha || 0).toFixed(2)} · c=${Math.round(w2e.medC || 0)} px/s · coh=${(w2e.meanCoh || 0).toFixed(2)}`;
      } else if (w2wActive && (w2w.state === 2 || w2w.state === 3)) {
        severity = 'ok';
        const loadPart = w2w.state === 3 ? `CL=${(w2w.cl || 0).toFixed(2)} CD=${(w2w.cd || 0).toFixed(2)} · ` : '';
        bannerTxt = `◆ POCKET WIND TUNNEL READY · ${loadPart}nu=${(w2w.nu || 0).toFixed(1)} px²/s · ${w2w.validCells || 0} CELLS`;
      } else if (bosActive && bos.state === 2) {
        severity = 'ok';
        bannerTxt = `◆ BOS SCHLIEREN READY · ${bos.validDisp}/32768 CELLS · CURL RMS ${bos.curlRms.toFixed(3)} · ${roi && roi.native1to1Active ? '1:1 NATIVE' : 'FIT'}`;
      } else if (w2vActive && w2v.state === 3) {
        severity = 'ok';
        const fTxt = Number.isFinite(w2v.domFreqHz) ? w2v.domFreqHz.toFixed(1) : '--';
        const gTxt = Number.isFinite(w2v.maxCohGamma2) ? w2v.maxCohGamma2.toFixed(3) : '--';
        bannerTxt = `◆ RS VIBROMETRY READY · K=${w2v.welchK || 0} · ${fTxt} Hz · g2=${gTxt}`;
      } else if (w2eActive && w2e.state === 2) {
        severity = 'info';
        bannerTxt = `ELASTOGRAPHY WARMING · SEGMENTS ${w2e.welchSegs || 0}/8 · LOCKING TONES`;
      } else if (w2wActive && w2w.state === 1) {
        severity = 'info';
        bannerTxt = `POCKET WIND TUNNEL WARMING · BUILDING HODGE PRESSURE FIELD`;
      } else if (bosActive && bos.state === 1) {
        severity = 'info';
        bannerTxt = `BOS REFERENCE WARMING (${bos.warmupCnt}/16) · KEEP CAMERA STATIC ON TRIPOD`;
      } else if (w2vActive && w2v.state === 2) {
        severity = 'info';
        bannerTxt = `RS WELCH WARMING · ${w2v.activeFramesInRun || 0}f · K=${w2v.welchK || 0}/8`;
      } else if (w2c && w2c.status === 2) {
        severity = 'info';
        bannerTxt = `◆ OCCLUDED CONSERVATION AMBIGUOUS (${(w2c.hypotheses || []).join(' / ') || 'H0_SWAP / H1_KEEP'})`;
      } else if (roi && roi.native1to1Active) {
        severity = 'ok';
        bannerTxt = `◆ NATIVE 1:1 ROI (${roi.cropX0},${roi.cropY0} ${roi.cropW}x${roi.cropH}) · ZERO IMPLICIT RESCALE`;
      }
      if (dom.acquisitionStatusBanner.dataset.severity !== severity) {
        dom.acquisitionStatusBanner.dataset.severity = severity;
      }
      setTextIfChanged(dom.acquisitionStatusBanner, bannerTxt);
    }
  }

  function updateWave2DomUi(m) {
    const w2 = m && m.wave2 ? m.wave2 : null;
    window.__chronosLastWave2Snap = w2;
    if (!w2) return;

    if (dom.diagWave2Stamps) {
      if (w2.available && w2.stamps) {
        const st = w2.stamps;
        const sc = (s) => (!s || !s.valid ? '-' : (s.held ? 'H' : 'V'));
        setTextIfChanged(
          dom.diagWave2Stamps,
          `${sc(st.vibro)}${sc(st.hodge)}${sc(st.conservation)}${sc(st.depth)}${sc(st.wind)}${sc(st.elasto)} · EP ${w2.continuityEpoch || 0}`
        );
      } else {
        setTextIfChanged(dom.diagWave2Stamps, '--');
      }
    }

    if (dom.diagRsWelch) {
      const v = w2.vibro;
      let rsTxt = 'OFF';
      if (v && v.available) {
        if (v.state === 3) {
          const fTxt = Number.isFinite(v.domFreqHz) ? v.domFreqHz.toFixed(1) : '--';
          if (v.reqMode === 1) {
            const aTxt = Number.isFinite(v.domAmpPx) ? v.domAmpPx.toFixed(2) : '--';
            rsTxt = `READY · RS-ONLY · ${fTxt}Hz · ${aTxt}px`;
          } else {
            const gTxt = Number.isFinite(v.maxCohGamma2) ? v.maxCohGamma2.toFixed(2) : '--';
            rsTxt = `READY · K=${v.welchK || 0} · ${fTxt}Hz · g2=${gTxt}`;
          }
        } else if (v.state === 2) {
          rsTxt = `WARMING · K=${v.welchK || 0}`;
        } else if (v.state === 1) {
          rsTxt = `UNCALIB · ${(v.reasonFlags || []).slice(0, 2).join('+') || 'BAD_CALIB'}`;
        } else if (v.state === 4) {
          rsTxt = `DISTURBED · ${(v.reasonFlags || []).slice(0, 2).join('+') || 'FAULT'}`;
        }
      }
      setTextIfChanged(dom.diagRsWelch, rsTxt);
    }

    if (dom.diagHodge2Complex) {
      const h = w2.hodge;
      let hTxt = 'OFF';
      if (h && h.available) {
        if (h.state === 2) {
          const ex = Number.isFinite(h.exactFrac) ? (h.exactFrac * 100).toFixed(0) : '--';
          const co = Number.isFinite(h.coexactFrac) ? (h.coexactFrac * 100).toFixed(0) : '--';
          const ha = Number.isFinite(h.harmFrac) ? (h.harmFrac * 100).toFixed(0) : '--';
          hTxt = `READY · ${ex}/${co}/${ha}% · b1=${h.beta1 || 0}`;
        } else if (h.state === 3) {
          hTxt = 'UNCONV';
        } else if (h.state === 1) {
          hTxt = 'STANDBY';
        }
      }
      setTextIfChanged(dom.diagHodge2Complex, hTxt);
    }

    if (dom.diagOcclConservation) {
      const c = w2.conservation;
      let cTxt = 'NONE';
      if (c && c.available && (c.status || 0) > 0) {
        const hyps = (c.hypotheses || []).join('/') || `T${c.tidA || 0}-T${c.tidB || 0}`;
        cTxt = `${c.statusLabel || 'NONE'} · ${hyps}`;
      }
      setTextIfChanged(dom.diagOcclConservation, cTxt);
    }

    if (dom.diagWindTunnel) {
      const w = w2.wind;
      let wTxt = 'OFF';
      if (w && w.available) {
        if (w.state === 3) {
          wTxt = `LOADS · CL=${(w.cl || 0).toFixed(2)} CD=${(w.cd || 0).toFixed(2)} · nu=${(w.nu || 0).toFixed(1)}`;
        } else if (w.state === 2) {
          wTxt = `FIELD · nu=${(w.nu || 0).toFixed(1)} · R2=${(w.nuR2 || 0).toFixed(2)} · ${w.validCells || 0}c`;
        } else if (w.state === 1) {
          wTxt = `WARMING · ${w.validCells || 0}c`;
        } else if (w.state === 4) {
          const wWhy = formatReasonSummary(w.reasonMask || 0, WIND_REASON_BITS, 2).text || 'CHECK FLOW';
          wTxt = `REFUSED · ${wWhy}`;
        }
      }
      setTextIfChanged(dom.diagWindTunnel, wTxt);
    }

    if (dom.diagElastography) {
      const e = w2.elasto;
      let eTxt = 'OFF';
      if (e && e.available) {
        if (e.state === 3) {
          if (e.mode === 4) {
            eTxt = `ABOS · ${(e.f0 || 0).toFixed(0)}Hz · |Phi|=${(e.abosPhiRms || 0).toFixed(3)} · curl=${(e.abosCurlRms || 0).toFixed(3)}`;
          } else {
            eTxt = `${e.classLabel || 'READY'} · a=${(e.alpha || 0).toFixed(2)} · c=${Math.round(e.medC || 0)} · coh=${(e.meanCoh || 0).toFixed(2)}`;
          }
        } else if (e.state === 2) {
          eTxt = `WARMING · seg=${e.welchSegs || 0}`;
        } else if (e.state === 1) {
          const eWhy = formatReasonSummary(e.reasonMask || 0, ELASTO_REASON_BITS, 2).text || 'CHECK CALIB';
          eTxt = `UNCALIB · ${eWhy}`;
        } else if (e.state === 4) {
          const eWhy = formatReasonSummary(e.reasonMask || 0, ELASTO_REASON_BITS, 2).text || 'CHECK MUTEX';
          eTxt = `REFUSED · ${eWhy}`;
        }
      }
      setTextIfChanged(dom.diagElastography, eTxt);
    }
  }

  let lastWorkerAckOrReject = null;
  let pendingWorkerAckWaiter = null;
  let nextW2ReqId = 1;
  const pendingW2Requests = new Map();

  function sendWave2Rpc(msgType, fields, timeoutMs = 4000) {
    return new Promise((resolve) => {
      const reqId = nextW2ReqId++;
      const timer = setTimeout(() => {
        pendingW2Requests.delete(reqId);
        resolve({ timeout: true, reqId });
      }, timeoutMs);
      pendingW2Requests.set(reqId, (ack) => {
        clearTimeout(timer);
        resolve(ack);
      });
      worker.postMessage({ type: msgType, reqId, wantAck: true, ...fields });
    });
  }

  window.__chronosWave1Probe = {
    postWorkerMsg: (payload, transfers = []) => {
      worker.postMessage(payload, transfers);
    },
    submitTestFrame: (payload, transfers = []) => new Promise((resolve) => {
      const timer = setTimeout(() => {
        pendingWorkerAckWaiter = null;
        resolve({ timeout: true });
      }, 4000);
      pendingWorkerAckWaiter = (msg) => {
        clearTimeout(timer);
        resolve(msg);
      };
      worker.postMessage(payload, transfers);
    }),
    getBusyState: () => ({ workerFrameBusy, bitmapCaptureBusy, lastPresentedFrames, pendingVideoGapMask, sourceEpoch, frameSeq, lastWorkerAckOrReject }),
    dispatchBitmapFrame: (item) => dispatchBitmapFrameToWorker(item),
    forwardVideoFrameForTest: (mt, perfMs, rVfcMeta) => forwardUniqueVideoFrame(mt, perfMs, sourceEpoch, rVfcMeta),
    setCaptureBusyForTest: (busy) => {
      if (busy) activeCaptureToken = ++bitmapCaptureToken;
      bitmapCaptureBusy = Boolean(busy);
    },
    simulateEpochFlushForTest: () => {
      sourceEpoch++;
      lastMediaTime = -1;
      lastPresentedFrames = -1;
      pendingVideoGapMask = 0;
      flushPendingQueue();
      return sourceEpoch;
    },
    getLastWave1: () => window.__chronosLastWave1Snap || null,
  };

  window.__chronosWave2Probe = {
    postWorkerMsg: (payload, transfers = []) => {
      worker.postMessage(payload, transfers);
    },
    getLastWave2: () => window.__chronosLastWave2Snap || null,
    submitAudioBlock: ({ samples, tEndUs, sampleRateHz, estSkewUs = 0.0 }) =>
      sendWave2Rpc('push_audio_pcm', { samples, tEndUs, sampleRateHz, estSkewUs }),
    evalOccludedPair: (input) =>
      sendWave2Rpc('eval_occluded_pair_test', { input }),
  };

  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'wave2_audio_ack' || m.type === 'wave2_cons_test_ack') {
      if (m.type === 'wave2_cons_test_ack' && m.conservation && window.__chronosLastWave2Snap) {
        window.__chronosLastWave2Snap.conservation = m.conservation;
        updateWave2DomUi({ wave2: window.__chronosLastWave2Snap });
      }
      const cb = typeof m.reqId === 'number' ? pendingW2Requests.get(m.reqId) : null;
      if (cb) {
        pendingW2Requests.delete(m.reqId);
        cb(m);
      }
      return;
    }
    if (m.type === 'frame_ack' || m.type === 'ingest_rejected') {
      lastWorkerAckOrReject = { ...m };
      if (pendingWorkerAckWaiter) {
        const cb = pendingWorkerAckWaiter;
        pendingWorkerAckWaiter = null;
        cb(m);
      }
      if (m.epoch === sourceEpoch) {
        workerFrameBusy = false;
        if (queuedPendingFrame) {
          const next = queuedPendingFrame;
          queuedPendingFrame = null;
          dispatchBitmapFrameToWorker(next);
        }
      }
      return;
    }
    if (m.type === 'config_ack') {
      dispatchInstrument({
        type: 'CONFIG_ACK',
        applied: m.applied !== false,
        appliedRevision: m.appliedRevision,
        sourceEpoch: m.sourceEpoch,
        workerEpoch: m.workerEpoch,
        bosOwner: m.bosOwner,
      }, true);
      return;
    }
    if (m.type === 'ready') {
      workerKernelReady = true;
      dom.pillKernel.textContent = '◆ KERNEL: 0 HEAP ALLOC / 0 IMPORTS';
      dom.pillKernel.title = dom.pillKernel.textContent;
      dom.pillKernel.dataset.signal = 'lock';
      dom.pillArena.textContent = `ARENA: ${m.pages} PAGES / ${(m.memoryBytes / 1048576).toFixed(2)} MB`;
      dom.pillArena.title = dom.pillArena.textContent;
      applyResolutionUi(m.resMode, m.frameW, m.frameH, m.pipDim);
      dispatchInstrument({ type: 'WORKER_READY' }, true);
      if (pendingHeroSceneBeforeReady) {
        initialPresetBooted = true;
        const targetScene = pendingHeroSceneBeforeReady;
        pendingHeroSceneBeforeReady = null;
        loadRecipe(targetScene, { settingsOnly: false });
      } else if (bootIdleBench && !initialPresetBooted) {
        syncContextualSourceUi('idle', { kind: 'IDLE', label: 'SOURCE: IDLE BENCH', originPreset: null });
        dom.hudSourceLabel.textContent = 'SOURCE: IDLE BENCH';
        dom.hudSourceLabel.title = 'SOURCE: IDLE BENCH';
      } else if (isInitialDeckSurface && !initialPresetBooted) {
        initialPresetBooted = true;
        dom.hudSourceLabel.textContent = 'SOURCE: BLIND PIXEL PILLAR TESTBENCH (getImageData ONLY)';
        dom.hudSourceLabel.title = dom.hudSourceLabel.textContent;
        syncContextualSourceUi('testbench', { kind: 'TESTBENCH', label: dom.hudSourceLabel.textContent, originPreset: 'bench_pillar_crossing' });
      } else {
        dispatchDeckEvent({ type: 'SOURCE_CHANGED', source: mapSourceModeToFsm(activeSourceMode) });
      }
      const overlayConverged = overlaySyncSeq === 0 || (typeof m.overlaySyncSeq === 'number' && m.overlaySyncSeq >= overlaySyncSeq);
      if (overlayConverged && typeof m.showHeroTargets === 'boolean') {
        dispatchDeckEvent({ type: 'OVERLAY_SET', showHeroTargets: m.showHeroTargets });
      }
      if (typeof publishInspectTelemetryState === 'function') {
        publishInspectTelemetryState(m.chronos1000 || null);
      }
    } else if (m.type === 'init_error') {
      dom.pillKernel.textContent = '⚠ KERNEL INIT FAILED';
      dom.pillKernel.dataset.signal = 'fault';
      dom.pillKernel.title = m.message || 'Failed to initialize Wasm kernel';
      dom.hudSourceLabel.textContent = `KERNEL ERROR: ${m.message || 'Wasm load failed'}`;
      dom.hudSourceLabel.title = dom.hudSourceLabel.textContent;
    } else if (m.type === 'swarm_config_sync') {
      const overlayConverged = overlaySyncSeq === 0 || (typeof m.overlaySyncSeq === 'number' && m.overlaySyncSeq >= overlaySyncSeq);
      if (typeof m.swarmMode === 'number' && dom.swarmModeSelect) {
        dom.swarmModeSelect.value = String(m.swarmMode);
      }
      if (overlayConverged && typeof m.fieldOverlayMode === 'number' && dom.fieldOverlayModeSelect) {
        dom.fieldOverlayModeSelect.value = String(m.fieldOverlayMode);
      }
      const patchOrg = {};
      const patchOv = {};
      if (typeof m.swarmMode === 'number') patchOrg.swarm = m.swarmMode;
      if (overlayConverged && typeof m.fieldOverlayMode === 'number') patchOv.field = m.fieldOverlayMode;
      if (overlayConverged && typeof m.showHeroTargets === 'boolean') patchOv.hero = m.showHeroTargets;
      if (Object.keys(patchOrg).length || Object.keys(patchOv).length) {
        dispatchInstrument({
          type: 'APPLY_CONFIG_PATCH',
          organs: patchOrg,
          overlays: patchOv,
          markPresetModified: instrumentState.source.presetModified,
        }, false);
      }
      dispatchDeckEvent({
        type: 'ENGINE_SYNC',
        swarmMode: typeof m.swarmMode === 'number' ? m.swarmMode : 0,
        swarmActive: Boolean(m.swarmActive),
      });
      if (overlayConverged) {
        dispatchDeckEvent({
          type: 'OVERLAY_SET',
          showHeroTargets: typeof m.showHeroTargets === 'boolean' ? m.showHeroTargets : undefined,
          fieldOverlayMode: typeof m.fieldOverlayMode === 'number' ? m.fieldOverlayMode : undefined,
        });
      }
    } else if (m.type === 'resolution_changed') {
      applyResolutionUi(m.resMode, m.frameW, m.frameH, m.pipDim);
    } else if (m.type === 'audio_strobe') {
      playPhaseStrobePulse(m.resonant, m.freqHz, m.amp);
    } else if (m.type === 'audio_percussion') {
      playDelayedSyncPercussion(m.zone, m.leadSec, m.velocity, m.sentEpochMs);
    } else if (m.type === 'telemetry') {
      const overlayConverged = overlaySyncSeq === 0 || (typeof m.overlaySyncSeq === 'number' && m.overlaySyncSeq >= overlaySyncSeq);
      lastSwarmActive = Boolean(m.chronos1000 && m.chronos1000.swarmActive);
      lastSwarmFrameW = m.frameW || lastSwarmFrameW;
      lastSwarmFrameH = m.frameH || lastSwarmFrameH;
      if (instrumentState.source.kind !== 'IDLE') {
        dispatchDeckEvent({ type: 'SOURCE_CHANGED', source: mapSourceModeToFsm(m.sourceMode) });
      }
      dispatchDeckEvent({
        type: 'ENGINE_SYNC',
        swarmMode: m.chronos1000 && typeof m.chronos1000.swarmMode === 'number'
          ? m.chronos1000.swarmMode
          : typeof m.swarmMode === 'number'
          ? m.swarmMode
          : 0,
        swarmActive: lastSwarmActive,
      });
      if (overlayConverged) {
        dispatchDeckEvent({
          type: 'OVERLAY_SET',
          showHeroTargets: typeof m.showHeroTargets === 'boolean' ? m.showHeroTargets : undefined,
          showAllOverlays: typeof m.showAllOverlays === 'boolean' ? m.showAllOverlays : undefined,
          fieldOverlayMode: m.chronos1000 && typeof m.chronos1000.fieldOverlayMode === 'number'
            ? m.chronos1000.fieldOverlayMode
            : undefined,
        });
      }
      const medMs = typeof m.wasmMedianMs === 'number' ? m.wasmMedianMs : m.wasmMeanMs;
      const budgetMs = m.resMode === 1
        ? (m.sourceMode === 'testbench' ? 1.0 : 2.2)
        : (m.sourceMode === 'testbench' ? 0.35 : 0.75);
      const underBudget = medMs < budgetMs;
      dom.pillPerf.textContent = `${underBudget ? '◆' : '▲'} KERNEL: ${medMs.toFixed(3)} ms`;
      dom.pillPerf.dataset.signal = underBudget ? 'lock' : 'warn';
      dom.pillFps.textContent = `FPS: ${m.fps.toFixed(0)}`;

      if (m.sourceMode !== 'testbench') {
        const camState = m.camMotionState || 0;
        const lagMs = Math.round(m.dispLagMs || 0);
        if (camState === 2) {
          dom.pillStreamHealth.dataset.signal = 'warn';
          dom.pillStreamHealth.textContent = `▲ CUT RESET · LAG ${lagMs}ms`;
        } else if (camState === 1) {
          dom.pillStreamHealth.dataset.signal = 'warn';
          dom.pillStreamHealth.textContent = `▲ PTZ PAN (${Math.round(m.camPanSpeedPxS || 0)} px/s) · LAG ${lagMs}ms`;
        } else {
          dom.pillStreamHealth.dataset.signal = 'lock';
          dom.pillStreamHealth.textContent = `◆ CAM LOCKED · LAG ${lagMs}ms · DROP ${pumpDroppedFrames}`;
        }
        if (activeSourceMode === 'live') {
          dom.liveStreamMeta.textContent = `F#${m.frameIdx} · LAG ${lagMs}ms · DISCONT ${m.discontCount || 0}`;
        }
      }

      if (m.sourceMode === 'testbench') {
        dom.lblPadicSwaps.textContent = 'Oracle ID Swaps';
        dom.kpiPadicSwaps.textContent = m.oracleSwaps != null ? `◆ ${m.oracleSwaps}` : '◆ N/A';
        dom.lblCollisions.textContent = 'Pillar Collisions';
        dom.kpiCollisions.textContent = m.oracleSwaps != null ? `${m.inPillarCollisions}` : 'N/A';
      } else {
        dom.lblPadicSwaps.textContent = '2-Adic Fiedler';
        dom.kpiPadicSwaps.textContent = `◆ ${m.padicSwaps}`;
        if (currentLabMode === 2) {
          dom.lblCollisions.textContent = 'L1 Impact Strikes';
          dom.kpiCollisions.textContent = `${m.strikeEvents || 0}`;
        } else {
          dom.lblCollisions.textContent = 'Hermite Bridges';
          dom.kpiCollisions.textContent = `${m.bridgedFrames || 0}f`;
        }
      }

      dom.kpiKalmanSwaps.textContent = `▲ ${m.kalmanSwaps}`;
      const evalPct = Math.round((m.clutterEvalRatio || 0) * 100);
      if (m.clutterDbValid) {
        dom.lblClutterDb.textContent = `ECA Clutter (${evalPct}%)`;
        dom.kpiClutterDb.textContent = `${m.clutterDb.toFixed(1)} dB`;
      } else {
        dom.lblClutterDb.textContent = 'ECA Clutter Supp.';
        dom.kpiClutterDb.textContent = 'WARMING';
      }
      dom.lblWasmMs.textContent = `Kernel (${m.wasmMeanMs.toFixed(2)} avg)`;
      dom.kpiWasmMs.textContent = `${medMs.toFixed(3)} ms`;
      dom.kpiTargets.textContent = `${m.activeTargets} / 16`;
      updatePersistenceDomUi(m.sessionPersistence, m.activeTargets);
      updatePhysicsIdDomUi(m.physicsId, m);
      updateContinuumDomUi(m.chronos1000, m);

      if (m.focusTrackId > 0) {
        dom.btnFocusCycle.textContent = `◎ Focus: PIN T${m.focusTrackId}`;
        dom.btnFocusCycle.classList.add('active');
      } else if (m.primaryTrackId > 0) {
        dom.btnFocusCycle.textContent = `◎ Focus: AUTO T${m.primaryTrackId}`;
        dom.btnFocusCycle.classList.remove('active');
      } else {
        dom.btnFocusCycle.textContent = '◎ Focus: AUTO';
        dom.btnFocusCycle.classList.remove('active');
      }

      const cleanSig = String(m.primarySigHex || '').replace(/^0x/i, '') || '--';
      dom.mortonReadout.textContent = `SIG: ${cleanSig} (v₂=${m.primaryPadicVal})`;
      dom.mortonReadout.title = `Binary-prefix signature ${cleanSig}, 2-adic valuation v₂ = ${m.primaryPadicVal}`;
      if (m.lookaheadK < 7) {
        dom.hilbertMeta.textContent = 'HILBERT GATED (K<7)';
        dom.kpiTremorHz.textContent = 'GATED (K<7)';
        dom.kpiTremorPhi.textContent = 'GATED';
      } else if (m.phaseValid && m.freqValid && m.tremorFreqHz > 0) {
        const cohPct = Math.round((m.tremorCoherence || 0) * 100);
        dom.hilbertMeta.textContent = `φ = ${m.tremorPhaseDeg.toFixed(0)}° | f = ${m.tremorFreqHz.toFixed(2)} Hz | COH ${cohPct}%`;
        dom.kpiTremorHz.textContent = `${m.tremorFreqHz.toFixed(2)} Hz`;
        dom.kpiTremorPhi.textContent = `${m.tremorPhaseDeg.toFixed(0)}°`;
      } else if (m.phaseValid) {
        dom.hilbertMeta.textContent = `φ = ${m.tremorPhaseDeg.toFixed(0)}° | APERIODIC MOTION`;
        dom.kpiTremorHz.textContent = 'APERIODIC';
        dom.kpiTremorPhi.textContent = `${m.tremorPhaseDeg.toFixed(0)}°`;
      } else {
        dom.hilbertMeta.textContent = 'STENCIL WARMING / HOLD';
        dom.kpiTremorHz.textContent = 'HOLD';
        dom.kpiTremorPhi.textContent = 'HOLD';
      }
      dom.spectralMeta.textContent = `λ₂ = ${m.lambda2.toFixed(4)} | Δλ = ${m.spectralGap.toFixed(3)}`;
      dom.fistaGapMeta.textContent = `DUAL GAP: ${m.fistaDualityGap.toFixed(3)}`;

      if (m.apexMs > 1.0) {
        dom.lblApexKpi.textContent = 'Ballistic T-Apex';
        dom.kpiApexMs.textContent = `${m.apexMs.toFixed(0)} ms`;
      } else if (currentLabMode === 0 && typeof m.apexErrPx === 'number' && m.apexErrPx >= 0) {
        dom.lblApexKpi.textContent = 'Last Apex Euclid Δ';
        dom.kpiApexMs.textContent = `${m.apexErrPx.toFixed(1)} px`;
      } else {
        dom.lblApexKpi.textContent = 'Ballistic T-Apex';
        dom.kpiApexMs.textContent = 'STEADY';
      }

      if (m.stimulusArmed) {
        dom.lblReflexKpi.textContent = 'Stimulus Armed';
        dom.kpiSaccadeMs.textContent = 'AWAITING';
      } else if (m.saccadeLatencyMs > 0) {
        dom.lblReflexKpi.textContent = 'Stimulus Onset Δt';
        dom.kpiSaccadeMs.textContent = `${m.saccadeLatencyMs.toFixed(1)} ms`;
      } else if (m.stepSettleMs > 0) {
        dom.lblReflexKpi.textContent = 'Step Settle Δt';
        dom.kpiSaccadeMs.textContent = `${m.stepSettleMs.toFixed(1)} ms`;
      } else {
        dom.lblReflexKpi.textContent = 'Onset / Settle Δt';
        dom.kpiSaccadeMs.textContent = 'READY';
      }

      const scaleQual = m.calProvenance === 'surveyed' ? 'Surveyed' : 'Assumed';
      if (m.activeTargets > 0) {
        if (m.pivotValid) {
          const signP = m.pendSwingDeg >= 0 ? '+' : '';
          dom.lblKinAngle.textContent = 'Pendulum Swing θ';
          dom.kpiKinAngle.textContent = `${signP}${m.pendSwingDeg.toFixed(1)}°`;
        } else {
          const signH = m.headingDeg >= 0 ? '+' : '';
          dom.lblKinAngle.textContent = 'Flight / Heading θ';
          dom.kpiKinAngle.textContent = `${signH}${m.headingDeg.toFixed(1)}°`;
        }
        if (m.metricValid < -0.5) {
          dom.lblKinSpeed.textContent = 'Perspective Guard';
          dom.kpiKinSpeed.textContent = 'OUT OF PLANE';
        } else if (m.calibrated) {
          dom.lblKinSpeed.textContent = `${scaleQual} Speed`;
          dom.kpiKinSpeed.textContent = `${m.speedPhys.toFixed(2)} m/s`;
        } else {
          dom.lblKinSpeed.textContent = 'Tangential Speed';
          dom.kpiKinSpeed.textContent = `${Math.round(m.speedPhys)} px/s`;
        }
      } else {
        dom.lblKinAngle.textContent = 'Flight / Swing θ';
        dom.kpiKinAngle.textContent = '--°';
        dom.lblKinSpeed.textContent = m.calibrated ? `${scaleQual} Speed` : 'Tangential Speed';
        dom.kpiKinSpeed.textContent = m.calibrated ? '-- m/s' : '-- px/s';
      }

      if (currentLabMode === 2) {
        if (m.lastStrikeValid) {
          const signD = m.lastDeflectDeg >= 0 ? '+' : '';
          if (m.scatterDeg > 2.0 && (m.scatterAgeFrames < 0 || m.scatterAgeFrames <= 90)) {
            dom.lblKinDeflect.textContent = 'Deflect Δθ / Scatter';
            dom.kpiKinDeflect.textContent = `${signD}${m.lastDeflectDeg.toFixed(0)}° / ${m.scatterDeg.toFixed(0)}°`;
          } else {
            dom.lblKinDeflect.textContent = 'Impact Deflection Δθ';
            dom.kpiKinDeflect.textContent = `${signD}${m.lastDeflectDeg.toFixed(1)}°`;
          }
        } else {
          dom.lblKinDeflect.textContent = 'Impact Deflection Δθ';
          dom.kpiKinDeflect.textContent = 'AWAITING';
        }
      } else if (m.pivotValid) {
        dom.lblKinDeflect.textContent = 'Tether Radius R';
        dom.kpiKinDeflect.textContent = m.calibrated && m.metricValid > 0.5
          ? `${m.pendRadiusPhys.toFixed(2)} m`
          : `${Math.round(m.pendRadiusPhys)} px`;
      } else {
        dom.lblKinDeflect.textContent = 'Angular Rate ω';
        dom.kpiKinDeflect.textContent = m.activeTargets > 0 ? `${Math.round(m.angVelDegS)}°/s` : '--°/s';
      }

      if (currentLabMode === 2) {
        const tidTag = m.lastStrikeValid && m.lastStrikeTid > 0 ? ` (T${m.lastStrikeTid})` : '';
        dom.lblKinRestGrav.textContent = `Speed Ratio v⁺/v⁻${tidTag}`;
        if (m.lastStrikeValid) {
          const ageSec = m.lastStrikeAgeFrames > 15 ? ` (${(m.lastStrikeAgeFrames / Math.max(8, m.fps)).toFixed(1)}s)` : '';
          dom.kpiKinRestGrav.textContent = `${m.lastRestitution.toFixed(2)}${ageSec}`;
        } else {
          dom.kpiKinRestGrav.textContent = 'READY';
        }
      } else if (m.sceneGravityConf >= 0.50 && m.sceneGravityPxS2 > 50) {
        dom.lblKinRestGrav.textContent = `Gravity ĝ (${m.sceneGravityArcs || 2} arcs)`;
        dom.kpiKinRestGrav.textContent = `${Math.round(m.sceneGravityPxS2)} px/s²`;
      } else if (m.sceneGravityPxS2 > 50 && (m.sceneGravityArcs || 0) === 1) {
        dom.lblKinRestGrav.textContent = 'Gravity ĝ (1/2 arcs)';
        dom.kpiKinRestGrav.textContent = `${Math.round(m.sceneGravityPxS2)} prov`;
      } else {
        dom.lblKinRestGrav.textContent = 'Metric Calibration';
        dom.kpiKinRestGrav.textContent = m.calibrated ? dom.calBadge.textContent : 'PX UNITS';
      }

      updateSlotRosterUi(m.slotRoster);
      updateRecorderUi(m.recActive, m.recFrames, m.recDurationSec);
      updateWave1DomUi(m);
      updateWave2DomUi(m);
      if (typeof updateShowcaseProofCards === 'function') {
        updateShowcaseProofCards(m, medMs);
      }
      syncRuntimeTelemetryToInstrument(m, medMs);
    } else if (m.type === 'cal_state') {
      applyCalibrationUiState(m);
    } else if (m.type === 'rec_state') {
      updateRecorderUi(m.recActive, m.recFrames, m.recDurationSec);
    } else if (m.type === 'timeline_events') {
      latestTimelineEvents = Array.isArray(m.events) ? m.events : [];
      latestTimelineSpanSec = typeof m.spanSec === 'number' && Number.isFinite(m.spanSec) ? m.spanSec : 0.0;
      latestTimelineSeq = typeof m.seq === 'number' && Number.isFinite(m.seq) ? m.seq : 0;
    } else if (m.type === 'seed_target_ack') {
      if (m.tid > 0 && dom.showcaseToolbarStatus) {
        dom.showcaseToolbarStatus.textContent = `Seeded user target T${m.tid} (${Math.round(m.w)}×${Math.round(m.h)} px)`;
      }
    } else if (m.type === 'fast_pass_progress') {
      if (dom.vidTimeLabel) {
        dom.vidTimeLabel.textContent = `FAST-PASS ${m.processed}/${m.total} · ${m.fps} fps (${m.codec || 'WC'})`;
      }
      if (dom.videoScrubSlider && m.total > 0) {
        dom.videoScrubSlider.value = String(Math.round((m.processed / m.total) * 1000));
      }
    } else if (m.type === 'fast_pass_done') {
      fastPassRunning = false;
      fastPassCancelRequested = false;
      if (dom.btnFastPass) {
        dom.btnFastPass.classList.remove('active');
        dom.btnFastPass.textContent = '⚡ Fast-Pass';
      }
      if (dom.vidTimeLabel) {
        dom.vidTimeLabel.textContent = m.label || `INDEXED ${m.processed || 0}f`;
      }
      if (dom.videoScrubSlider && activeSourceMode === 'video' && Number.isFinite(videoEl.duration) && videoEl.duration > 0) {
        dom.videoScrubSlider.disabled = false;
      }
    } else if (m.type === 'telemetry_export_data') {
      const { format, rows, cols, columns, buffer, meta, events, sessionConfig } = m;
      if (rows === 0) {
        dom.recStatusLabel.textContent = 'EMPTY BUFFER (RECORD FIRST)';
        return;
      }
      const data = new Float64Array(buffer);
      const stamp = formatExportTimestamp();
      if (format === 'json') {
        const records = new Array(rows);
        for (let r = 0; r < rows; r++) {
          const obj = {};
          const base = r * cols;
          for (let c = 0; c < cols; c++) {
            const v = data[base + c];
            obj[columns[c]] = !Number.isFinite(v) ? null : (Number.isInteger(v) ? v : Number(v.toFixed(6)));
          }
          records[r] = obj;
        }
        const payload = JSON.stringify(
          {
            exportedAt: new Date().toISOString(),
            meta,
            sessionConfig: sessionConfig || null,
            events: Array.isArray(events) ? events : latestTimelineEvents,
            frameCount: rows,
            columns,
            frames: records,
          },
          null,
          2
        );
        triggerBlobDownload(`chronos-telemetry-${stamp}.json`, 'application/json;charset=utf-8', payload);
        dom.recStatusLabel.textContent = `EXPORTED ${rows} FRAMES (JSON)`;
      } else {
        const lines = new Array(rows + 1);
        lines[0] = columns.join(',');
        for (let r = 0; r < rows; r++) {
          const base = r * cols;
          const rowStr = new Array(cols);
          for (let c = 0; c < cols; c++) {
            const v = data[base + c];
            rowStr[c] = !Number.isFinite(v) ? '' : (Number.isInteger(v) ? String(v) : v.toFixed(6));
          }
          lines[r + 1] = rowStr.join(',');
        }
        triggerBlobDownload(`chronos-telemetry-${stamp}.csv`, 'text/csv;charset=utf-8', lines.join('\n') + '\n');
        dom.recStatusLabel.textContent = `EXPORTED ${rows} FRAMES (CSV)`;
      }
    }
  };

  // Click-to-pin target on HUD Canvas + Accessible Keyboard Focus Cycle Button
  if (dom.swarmModeSelect) {
    dom.swarmModeSelect.addEventListener('change', () => {
      const mode = Math.max(0, Math.min(2, parseInt(dom.swarmModeSelect.value || '0', 10) || 0));
      dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { swarm: mode } }, true);
      dispatchDeckEvent({
        type: 'ENGINE_SYNC',
        swarmMode: mode,
        swarmActive: mode === 2 ? true : (mode === 0 ? false : lastSwarmActive),
      });
      worker.postMessage({ type: 'set_swarm_mode', swarmMode: mode });
    });
  }

  if (dom.fieldOverlayModeSelect) {
    dom.fieldOverlayModeSelect.addEventListener('change', () => {
      const om = Math.max(0, Math.min(15, parseInt(dom.fieldOverlayModeSelect.value || '0', 10) || 0));
      const patchOrg = {};
      let preemptedBos = false;
      if (om >= 5 && om <= 7 && dom.bosModeSelect && dom.bosModeSelect.value === '0') {
        if (dom.elastoModeSelect && dom.elastoModeSelect.value !== '0') {
          dom.elastoModeSelect.value = '0';
          patchOrg.elasto = 0;
          preemptedBos = true;
          worker.postMessage({ type: 'set_elasto_config', elastoMode: 0 });
        }
        if (dom.windModeSelect && dom.windModeSelect.value !== '0') {
          dom.windModeSelect.value = '0';
          patchOrg.wind = 0;
          preemptedBos = true;
          worker.postMessage({ type: 'set_wind_config', windMode: 0 });
        }
        dom.bosModeSelect.value = '1';
        patchOrg.bos = 1;
        worker.postMessage({ type: 'set_bos_config', bosMode: 1, overlayKind: om - 5, gaugeRoi: [4, 4, 18, 60] });
      } else if (om >= 8 && om <= 10 && dom.hodgeModeSelect && dom.hodgeModeSelect.value === '0') {
        dom.hodgeModeSelect.value = '1';
        patchOrg.hodge = 1;
        worker.postMessage({ type: 'set_hodge_config', hodgeMode: 1 });
      } else if (om === 11 && dom.rsVibroModeSelect && dom.rsVibroModeSelect.value === '0') {
        dom.rsVibroModeSelect.value = '2';
        patchOrg.rs = 2;
        sendRsControlToWorker(true);
      } else if (om === 12 && dom.depthModeSelect && dom.depthModeSelect.value === '0') {
        dom.depthModeSelect.value = '1';
        patchOrg.depth = 1;
        if (dom.swarmModeSelect && dom.swarmModeSelect.value === '0') {
          dom.swarmModeSelect.value = '2';
          patchOrg.swarm = 2;
          worker.postMessage({ type: 'set_swarm_mode', swarmMode: 2 });
        }
        worker.postMessage({ type: 'set_depth_config', depthMode: 1 });
      } else if (om === 13 && dom.windModeSelect && dom.windModeSelect.value === '0') {
        if (dom.bosModeSelect && dom.bosModeSelect.value !== '0') {
          dom.bosModeSelect.value = '0';
          patchOrg.bos = 0;
          preemptedBos = true;
          worker.postMessage({ type: 'set_bos_config', bosMode: 0, overlayKind: 0 });
        }
        dom.windModeSelect.value = '1';
        patchOrg.wind = 1;
        if (dom.swarmModeSelect && dom.swarmModeSelect.value === '0') {
          dom.swarmModeSelect.value = '2';
          patchOrg.swarm = 2;
          worker.postMessage({ type: 'set_swarm_mode', swarmMode: 2 });
        }
        if (dom.hodgeModeSelect && dom.hodgeModeSelect.value === '0') {
          dom.hodgeModeSelect.value = '1';
          patchOrg.hodge = 1;
          worker.postMessage({ type: 'set_hodge_config', hodgeMode: 1 });
        }
        worker.postMessage({ type: 'set_wind_config', windMode: 1 });
      } else if ((om === 14 || om === 15) && dom.elastoModeSelect && dom.elastoModeSelect.value === '0') {
        const targetEm = om === 15 ? 4 : 3;
        if (dom.bosModeSelect && dom.bosModeSelect.value !== '0') {
          dom.bosModeSelect.value = '0';
          patchOrg.bos = 0;
          preemptedBos = true;
          worker.postMessage({ type: 'set_bos_config', bosMode: 0, overlayKind: 0 });
        }
        if (dom.windModeSelect && dom.windModeSelect.value !== '0') {
          dom.windModeSelect.value = '0';
          patchOrg.wind = 0;
          preemptedBos = true;
          worker.postMessage({ type: 'set_wind_config', windMode: 0 });
        }
        dom.elastoModeSelect.value = String(targetEm);
        patchOrg.elasto = targetEm;
        if (targetEm === 4 && dom.rsVibroModeSelect && dom.rsVibroModeSelect.value === '0') {
          dom.rsVibroModeSelect.value = '2';
          patchOrg.rs = 2;
          sendRsControlToWorker(true);
        }
        worker.postMessage({ type: 'set_elasto_config', elastoMode: targetEm });
      }
      dispatchInstrument({
        type: 'APPLY_CONFIG_PATCH',
        organs: patchOrg,
        overlays: { field: om },
        recordUndo: preemptedBos,
        undoLabel: `Overlay ${om}`,
      }, true);
      dispatchDeckEvent({ type: 'OVERLAY_SET', fieldOverlayMode: om });
      worker.postMessage({ type: 'set_field_overlay_mode', fieldOverlayMode: om, overlaySyncSeq: nextOverlaySyncSeq() });
    });
  }

  // Explicit manual organ changes never silently kill a competing BOS scratch owner;
  // instead they update InstrumentState so the inline reason chip + 1-click atomic Fix button
  // and 5-slot Scratch Arena Bus show the exact owner and offer 1-click atomic handoff + Undo.
  if (dom.bosModeSelect) {
    dom.bosModeSelect.addEventListener('change', () => {
      const bm = Math.max(0, Math.min(2, parseInt(dom.bosModeSelect.value || '0', 10) || 0));
      dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { bos: bm } }, true);
      const om = dom.fieldOverlayModeSelect ? (parseInt(dom.fieldOverlayModeSelect.value || '0', 10) || 0) : 0;
      const okind = om >= 5 && om <= 7 ? om - 5 : 0;
      const payload = { type: 'set_bos_config', bosMode: bm, overlayKind: okind };
      if (bm > 0) payload.gaugeRoi = [4, 4, 18, 60];
      worker.postMessage(payload);
    });
  }

  if (dom.metrologyModeSelect) {
    dom.metrologyModeSelect.addEventListener('change', () => {
      const mm = Math.max(0, Math.min(2, parseInt(dom.metrologyModeSelect.value || '0', 10) || 0));
      const showEll = mm > 0 && dom.chkCovEllipses ? dom.chkCovEllipses.checked : false;
      dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { metrology: mm }, overlays: { covEllipses: showEll } }, true);
      worker.postMessage({ type: 'set_metrology_config', metrologyMode: mm, showCovEllipses: showEll });
    });
  }

  if (dom.hodgeModeSelect) {
    dom.hodgeModeSelect.addEventListener('change', () => {
      const hm = Math.max(0, Math.min(2, parseInt(dom.hodgeModeSelect.value || '0', 10) || 0));
      dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { hodge: hm } }, true);
      worker.postMessage({ type: 'set_hodge_config', hodgeMode: hm });
    });
  }

  if (dom.depthModeSelect) {
    dom.depthModeSelect.addEventListener('change', () => {
      const dm = Math.max(0, Math.min(4, parseInt(dom.depthModeSelect.value || '0', 10) || 0));
      dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { depth: dm } }, true);
      worker.postMessage({ type: 'set_depth_config', depthMode: dm });
    });
  }

  if (dom.windModeSelect) {
    dom.windModeSelect.addEventListener('change', () => {
      const wm = Math.max(0, Math.min(3, parseInt(dom.windModeSelect.value || '0', 10) || 0));
      dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { wind: wm } }, true);
      worker.postMessage({ type: 'set_wind_config', windMode: wm });
    });
  }

  if (dom.elastoModeSelect) {
    dom.elastoModeSelect.addEventListener('change', () => {
      const em = Math.max(0, Math.min(4, parseInt(dom.elastoModeSelect.value || '0', 10) || 0));
      dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { elasto: em } }, true);
      worker.postMessage({ type: 'set_elasto_config', elastoMode: em });
    });
  }

  if (dom.btnProbeExciter) {
    dom.btnProbeExciter.addEventListener('click', () => {
      if (probeExciterActive) stopProbeExciter();
      else startProbeExciter();
    });
  }

  if (dom.probeExciterPreset) {
    dom.probeExciterPreset.addEventListener('change', () => {
      if (probeExciterActive) startProbeExciter();
    });
  }

  function readRsUiScanDir() {
    const sdRaw = dom.selRsScanDir ? parseInt(dom.selRsScanDir.value, 10) : -1;
    return Number.isInteger(sdRaw) && sdRaw >= 0 && sdRaw <= 3 ? sdRaw : 0xffffffff;
  }

  function sendRsControlToWorker(autoEnableNative = false) {
    const rm = dom.rsVibroModeSelect
      ? Math.max(0, Math.min(2, parseInt(dom.rsVibroModeSelect.value || '0', 10) || 0))
      : 0;
    const dtRowUs = dom.inpRsDtRowUs ? Number(dom.inpRsDtRowUs.value) : 0.0;
    const exposureUs = dom.inpRsExposureUs ? Number(dom.inpRsExposureUs.value) : 0.0;
    const scanDir = readRsUiScanDir();
    if (rm > 0 && autoEnableNative && dom.chkNativeRoi && !dom.chkNativeRoi.checked) {
      dom.chkNativeRoi.checked = true;
    }
    const n1 = Boolean(dom.chkNativeRoi && dom.chkNativeRoi.checked);
    dispatchInstrument({
      type: 'APPLY_CONFIG_PATCH',
      organs: {
        rs: rm,
        rsCal: {
          ...instrumentState.organ_config.rsCal,
          rowUs: Number.isFinite(dtRowUs) ? dtRowUs : instrumentState.organ_config.rsCal.rowUs,
          exposureUs: Number.isFinite(exposureUs) ? exposureUs : instrumentState.organ_config.rsCal.exposureUs,
          scan: scanDir <= 3 ? scanDir : -1,
        },
      },
      resolution_roi: { native1to1: n1 },
    }, false);
    const xPct = dom.rangeRoiX0 ? Math.max(0, Math.min(100, Number(dom.rangeRoiX0.value) || 0)) : 50;
    const yPct = dom.rangeRoiY0 ? Math.max(0, Math.min(100, Number(dom.rangeRoiY0.value) || 0)) : 50;
    worker.postMessage({
      type: 'set_roi',
      native1to1: n1,
      normX: xPct / 100.0,
      normY: yPct / 100.0,
      scanDir,
    });
    worker.postMessage({
      type: 'set_rs_calib',
      rsMode: rm,
      dtRowUs,
      exposureUs,
      audioRateHz: rsAudioCaptureRateHz || 48000.0,
      maxSkewUs: 500.0,
    });
    syncRsLiveAudioCapture();
  }

  if (dom.rsVibroModeSelect) {
    dom.rsVibroModeSelect.addEventListener('change', () => {
      const rm = Math.max(0, Math.min(2, parseInt(dom.rsVibroModeSelect.value || '0', 10) || 0));
      dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { rs: rm } }, true);
      sendRsControlToWorker(rm > 0);
    });
  }
  if (dom.inpRsDtRowUs) {
    dom.inpRsDtRowUs.addEventListener('change', () => {
      const v = Number(dom.inpRsDtRowUs.value);
      if (Number.isFinite(v)) {
        dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { rsCal: { ...instrumentState.organ_config.rsCal, rowUs: v } } }, false);
      }
      const rm = dom.rsVibroModeSelect ? (parseInt(dom.rsVibroModeSelect.value || '0', 10) || 0) : 0;
      if (rm > 0) sendRsControlToWorker(false);
    });
  }
  if (dom.inpRsExposureUs) {
    dom.inpRsExposureUs.addEventListener('change', () => {
      const v = Number(dom.inpRsExposureUs.value);
      if (Number.isFinite(v)) {
        dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { rsCal: { ...instrumentState.organ_config.rsCal, exposureUs: v } } }, false);
      }
      const rm = dom.rsVibroModeSelect ? (parseInt(dom.rsVibroModeSelect.value || '0', 10) || 0) : 0;
      if (rm > 0) sendRsControlToWorker(false);
    });
  }
  if (dom.selRsScanDir) {
    dom.selRsScanDir.addEventListener('change', () => {
      const sdRaw = parseInt(dom.selRsScanDir.value, 10);
      dispatchInstrument({
        type: 'APPLY_CONFIG_PATCH',
        organs: { rsCal: { ...instrumentState.organ_config.rsCal, scan: Number.isInteger(sdRaw) ? sdRaw : -1 } },
      }, false);
      const rm = dom.rsVibroModeSelect ? (parseInt(dom.rsVibroModeSelect.value || '0', 10) || 0) : 0;
      if (rm > 0) {
        sendRsControlToWorker(false);
      } else {
        worker.postMessage({ type: 'set_roi', scanDir: readRsUiScanDir() });
      }
    });
  }

  if (dom.chkCovEllipses) {
    dom.chkCovEllipses.addEventListener('change', (e) => {
      const on = Boolean(e.target.checked);
      dispatchDeckEvent({ type: 'OVERLAY_SET', showCovEllipses: on });
      if (on && dom.metrologyModeSelect && dom.metrologyModeSelect.value === '0') {
        dom.metrologyModeSelect.value = '2';
        dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { metrology: 2 }, overlays: { covEllipses: true } }, true);
        worker.postMessage({ type: 'set_metrology_config', metrologyMode: 2, showCovEllipses: true });
      } else {
        dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', overlays: { covEllipses: on } }, true);
        worker.postMessage({ type: 'set_metrology_config', showCovEllipses: on });
      }
    });
  }

  function sendRoiControlToWorker() {
    const n1 = Boolean(dom.chkNativeRoi && dom.chkNativeRoi.checked);
    const xPct = dom.rangeRoiX0 ? Math.max(0, Math.min(100, Number(dom.rangeRoiX0.value) || 0)) : 50;
    const yPct = dom.rangeRoiY0 ? Math.max(0, Math.min(100, Number(dom.rangeRoiY0.value) || 0)) : 50;
    if (dom.rangeRoiX0) {
      dom.rangeRoiX0.setAttribute('aria-valuetext', `${xPct}% horizontal crop offset`);
      syncRangeFallbackProgress(dom.rangeRoiX0);
    }
    if (dom.rangeRoiY0) {
      dom.rangeRoiY0.setAttribute('aria-valuetext', `${yPct}% vertical crop offset`);
      syncRangeFallbackProgress(dom.rangeRoiY0);
    }
    dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', resolution_roi: { native1to1: n1, roiX0: xPct, roiY0: yPct } }, true);
    worker.postMessage({
      type: 'set_roi',
      native1to1: n1,
      normX: xPct / 100.0,
      normY: yPct / 100.0,
    });
  }

  if (dom.chkNativeRoi) {
    dom.chkNativeRoi.addEventListener('change', sendRoiControlToWorker);
  }
  const chkAutoTrackRoi = document.getElementById('chkAutoTrackRoi');
  if (chkAutoTrackRoi) {
    chkAutoTrackRoi.addEventListener('change', () => {
      worker.postMessage({ type: 'set_auto_track_roi', enabled: Boolean(chkAutoTrackRoi.checked) });
    });
  }
  const btnToolCropRoi = document.getElementById('btnToolCropRoi');
  const btnResetRoiCrop = document.getElementById('btnResetRoiCrop');
  function setCropRoiToolActive(on) {
    dispatchDeckEvent({ type: 'POINTER_TOOL_SET', pointerTool: on ? 'CROP_ROI' : 'INSPECT_TARGET' });
    if (btnToolCropRoi) {
      btnToolCropRoi.classList.toggle('active', on);
      btnToolCropRoi.setAttribute('aria-pressed', String(on));
    }
    if (!on) {
      worker.postMessage({ type: 'roi_box_preview', box: null });
    }
  }
  if (btnToolCropRoi) {
    btnToolCropRoi.addEventListener('click', () => {
      setCropRoiToolActive(deckUiState.pointerTool !== 'CROP_ROI');
    });
  }
  if (btnResetRoiCrop) {
    btnResetRoiCrop.addEventListener('click', () => {
      setCropRoiToolActive(false);
      if (dom.chkNativeRoi) dom.chkNativeRoi.checked = false;
      if (chkAutoTrackRoi) chkAutoTrackRoi.checked = false;
      dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', resolution_roi: { native1to1: false, autoTrackRoi: false, roiX0: 50, roiY0: 50 } }, true);
      worker.postMessage({ type: 'set_auto_track_roi', enabled: false });
      worker.postMessage({ type: 'set_roi', native1to1: false, normX: 0.5, normY: 0.5 });
      worker.postMessage({ type: 'set_crop', cropNorm: [0.0, 0.0, 1.0, 1.0] });
    });
  }
  if (dom.rangeRoiX0) {
    dom.rangeRoiX0.addEventListener('input', sendRoiControlToWorker);
  }
  if (dom.rangeRoiY0) {
    dom.rangeRoiY0.addEventListener('input', sendRoiControlToWorker);
  }

  function syncSwarmControlsForConfig(resTarget, swarmTarget, overlayTarget, postSwarmMode = false) {
    if (typeof resTarget === 'number' && resTarget === 1 && activeResMode === 0) {
      document.getElementById('btnResHd')?.click();
    }
    if (typeof swarmTarget === 'number' && dom.swarmModeSelect) {
      dom.swarmModeSelect.value = String(swarmTarget);
      dispatchDeckEvent({
        type: 'ENGINE_SYNC',
        swarmMode: swarmTarget,
        swarmActive: swarmTarget === 2 ? true : (swarmTarget === 0 ? false : lastSwarmActive),
      });
      if (postSwarmMode) {
        worker.postMessage({ type: 'set_swarm_mode', swarmMode: swarmTarget });
      }
    }
    if (typeof overlayTarget === 'number' && dom.fieldOverlayModeSelect) {
      dom.fieldOverlayModeSelect.value = String(overlayTarget);
      dispatchDeckEvent({ type: 'OVERLAY_SET', fieldOverlayMode: overlayTarget });
      worker.postMessage({ type: 'set_field_overlay_mode', fieldOverlayMode: overlayTarget, overlaySyncSeq: nextOverlaySyncSeq() });
    }
  }

  let seedDragState = null;
  let suppressNextHudClick = false;
  let suppressClickTimer = 0;

  function normHudCoordsFromPointer(ev) {
    const rect = hudEl.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    return {
      nx: Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width)),
      ny: Math.max(0, Math.min(1, (ev.clientY - rect.top) / rect.height)),
    };
  }

  hudEl.addEventListener('pointerdown', (ev) => {
    if (ev.button !== 0 || ev.target !== hudEl) return;
    const ptTool = deckUiState.pointerTool;
    if (ptTool === 'CAL_RULER_P1' || ptTool === 'CAL_RULER_P2' || ptTool.startsWith('CAL_QUAD_PT')) {
      return;
    }
    const pt = normHudCoordsFromPointer(ev);
    if (!pt) return;
    seedDragState = {
      pointerId: ev.pointerId,
      mode: ptTool === 'CROP_ROI' ? 'roi' : 'seed',
      startX: ev.clientX,
      startY: ev.clientY,
      nx0: pt.nx,
      ny0: pt.ny,
      nx1: pt.nx,
      ny1: pt.ny,
      active: false,
    };
  });

  hudEl.addEventListener('pointermove', (ev) => {
    if (!seedDragState || ev.pointerId !== seedDragState.pointerId) return;
    const dist = Math.hypot(ev.clientX - seedDragState.startX, ev.clientY - seedDragState.startY);
    if (!seedDragState.active && dist >= 6) {
      seedDragState.active = true;
      try { hudEl.setPointerCapture(ev.pointerId); } catch (_err) {}
    }
    if (!seedDragState.active) return;
    const pt = normHudCoordsFromPointer(ev);
    if (!pt) return;
    seedDragState.nx1 = pt.nx;
    seedDragState.ny1 = pt.ny;
    worker.postMessage({
      type: seedDragState.mode === 'roi' ? 'roi_box_preview' : 'seed_box_preview',
      box: {
        x0: seedDragState.nx0,
        y0: seedDragState.ny0,
        x1: seedDragState.nx1,
        y1: seedDragState.ny1,
      },
    });
  });

  function finishSeedDrag(ev, commit) {
    if (!seedDragState || ev.pointerId !== seedDragState.pointerId) return;
    const st = seedDragState;
    seedDragState = null;
    if (!st.active) return;
    try { hudEl.releasePointerCapture(ev.pointerId); } catch (_err) {}
    if (commit) {
      const pt = normHudCoordsFromPointer(ev);
      const nx1 = pt ? pt.nx : st.nx1;
      const ny1 = pt ? pt.ny : st.ny1;
      const normX = Math.min(st.nx0, nx1);
      const normY = Math.min(st.ny0, ny1);
      const normW = Math.abs(nx1 - st.nx0);
      const normH = Math.abs(ny1 - st.ny0);
      if (st.mode === 'roi') {
        worker.postMessage({ type: 'roi_box_preview', box: null });
        if (normW >= 0.05 && normH >= 0.05) {
          worker.postMessage({ type: 'set_crop', cropNorm: [normX, normY, normW, normH] });
        }
        setCropRoiToolActive(false);
      } else if (normW >= 0.006 && normH >= 0.006) {
        worker.postMessage({ type: 'seed_target', normX, normY, normW, normH });
      } else {
        worker.postMessage({ type: 'seed_box_preview', box: null });
      }
      suppressNextHudClick = true;
      clearTimeout(suppressClickTimer);
      suppressClickTimer = setTimeout(() => { suppressNextHudClick = false; }, 140);
    } else {
      worker.postMessage({ type: st.mode === 'roi' ? 'roi_box_preview' : 'seed_box_preview', box: null });
    }
  }

  hudEl.addEventListener('pointerup', (ev) => finishSeedDrag(ev, true));
  hudEl.addEventListener('pointercancel', (ev) => finishSeedDrag(ev, false));

  hudEl.addEventListener('click', (ev) => {
    if (suppressNextHudClick) {
      suppressNextHudClick = false;
      clearTimeout(suppressClickTimer);
      return;
    }
    const rect = hudEl.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const normX = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
    const normY = Math.max(0, Math.min(1, (ev.clientY - rect.top) / rect.height));
    const ptTool = deckUiState.pointerTool;
    if (ptTool === 'CAL_RULER_P1' || ptTool === 'CAL_RULER_P2' || ptTool.startsWith('CAL_QUAD_PT')) {
      const expectedCalArmed = advanceOptimisticCalClick();
      worker.postMessage({
        type: 'canvas_click',
        normX,
        normY,
        intent: 'calibration',
        expectedCalArmed,
      });
      return;
    }
    if (ptTool === 'CROP_ROI') {
      return;
    }
    if (deckUiState.engine === 'SWARM_ACTIVE' || lastSwarmActive || ev.shiftKey) {
      const nextInspectSeq = typeof inspectReqSeq === 'number' ? (inspectReqSeq += 1) : 1;
      worker.postMessage({
        type: 'inspect_point',
        inspectSeq: nextInspectSeq,
        normX,
        normY,
      });
      return;
    }
    worker.postMessage({ type: 'canvas_click', normX, normY });
  });

  dom.btnFocusCycle.addEventListener('click', () => {
    worker.postMessage({ type: 'cycle_focus' });
  });

  // Spatial Resolution Switching (320x180 SD vs 640x360 HD)
  document.querySelectorAll('.res-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const mode = Number(btn.dataset.res) ? 1 : 0;
      if (mode === activeResMode) return;
      applyResolutionUi(mode, mode ? 640 : 320, mode ? 360 : 180, mode ? 64 : 32);
      lastMediaTime = -1;
      lastPumpPerfMs = 0;
      worker.postMessage({ type: 'set_resolution', resMode: mode });
    });
  });

  // Testbench Scenario Selector
  dom.benchScenarioSelect.addEventListener('change', () => {
    const sc = dom.benchScenarioSelect.value;
    const isC1k = sc === 'vortex_swarm_1200' || sc === 'murmuration_1200' || sc === 'modal_beam_1152';
    if (isC1k) {
      syncSwarmControlsForConfig(1, 2, undefined, false);
    }
    worker.postMessage({ type: 'set_scenario', scenario: sc });
    if (sc === 'ballistic_cascade' && currentLabMode !== 0) {
      document.querySelector('.lab-btn[data-lab="0"]')?.click();
    } else if (sc === 'harmonic_tremor' && currentLabMode !== 1) {
      document.querySelector('.lab-btn[data-lab="1"]')?.click();
    } else if (sc === 'percussion_ricochet' && currentLabMode !== 2) {
      document.querySelector('.lab-btn[data-lab="2"]')?.click();
    }
  });

  // Source Switching + 3-Tier Live Stream Suite
  const btnTestbench = document.getElementById('btnSrcTestbench');
  const btnWebcam = document.getElementById('btnSrcWebcam');
  const btnLiveStream = document.getElementById('btnSrcLiveStream');
  const mp4Input = document.getElementById('mp4Input');
  const sampleVideoSelect = document.getElementById('sampleVideoSelect');

  function setSourceButtons(activeBtn) {
    [btnTestbench, btnWebcam, btnLiveStream].forEach((b) => {
      const on = b === activeBtn;
      b.classList.toggle('active', on);
      b.setAttribute('aria-pressed', String(on));
    });
  }

  function setLiveDialogBanner(msg, isError = false) {
    dom.liveDialogStatus.textContent = msg;
    dom.liveDialogStatus.classList.toggle('error', Boolean(isError));
  }

  // Native <dialog> close handling & light-dismiss fallback (modern-web-guidance)
  liveDialog.addEventListener('close', () => {
    const _action = liveDialog.returnValue;
    liveDialog.returnValue = '';
  });

  if (!('closedBy' in HTMLDialogElement.prototype)) {
    liveDialog.addEventListener('click', (event) => {
      if (event.target !== liveDialog) return;
      const rect = liveDialog.getBoundingClientRect();
      const isDialogContent = (
        rect.top <= event.clientY &&
        event.clientY <= rect.top + rect.height &&
        rect.left <= event.clientX &&
        event.clientX <= rect.left + rect.width
      );
      if (isDialogContent) return;
      liveDialog.close();
    });
  }

  let userPickedTier3Preset = false;
  dom.tier3PresetSelect.addEventListener('change', () => {
    userPickedTier3Preset = true;
  });

  function enableStaticHostMode() {
    if (isStaticHostMode) return;
    isStaticHostMode = true;
    document.body.dataset.staticHost = 'true';
    if (btnLiveStream) {
      btnLiveStream.title = 'Live HLS FFmpeg relay requires local scripts/dev_server.mjs (Tab Capture & Tier-2 CORS remain available)';
    }
    document.querySelectorAll('.showcase-btn[data-showcase="live_i80"]').forEach((btn) => {
      btn.title = 'Requires local node scripts/dev_server.mjs (FFmpeg HLS relay)';
    });
  }

  if (isStaticHostMode) {
    isStaticHostMode = false;
    enableStaticHostMode();
  }

  async function refreshTier3LivePresets(forceRefresh = false) {
    if (isStaticHostMode && !forceRefresh) return;
    try {
      const url = forceRefresh ? '/api/live-presets?refresh=1' : '/api/live-presets';
      const resp = await fetch(url, { cache: 'no-store' });
      if (!resp.ok) {
        if (resp.status === 404 || resp.status === 405) enableStaticHostMode();
        return;
      }
      const payload = await resp.json();
      if (!payload || !Array.isArray(payload.presets) || payload.presets.length === 0) return;
      const prevVal = dom.tier3PresetSelect.value;
      dom.tier3PresetSelect.innerHTML = '';

      const realCams = payload.presets.filter((p) => p.category !== 'local_sim' && p.kind !== 'simulated');
      const simRelays = payload.presets.filter((p) => p.category === 'local_sim' || p.kind === 'simulated');

      const makeOpt = (p) => {
        const opt = document.createElement('option');
        const isSim = p.category === 'local_sim' || p.kind === 'simulated';
        const name = p.label || p.title || p.id;
        opt.value = p.id;
        opt.dataset.lab = String(p.mode ?? p.labMode ?? 0);
        opt.dataset.fps = String(p.fpsHint ?? 30);
        const badge = isSim
          ? '[SYNTHETIC TEST]'
          : (p.online ? '[LIVE ONLINE]' : '[OFFLINE]');
        opt.textContent = `${badge} ${name}`;
        return opt;
      };

      if (realCams.length > 0) {
        const grpLive = document.createElement('optgroup');
        grpLive.label = 'Live Public Highway Cameras (Caltrans Real-Time HLS)';
        for (const p of realCams) grpLive.appendChild(makeOpt(p));
        dom.tier3PresetSelect.appendChild(grpLive);
      }
      if (simRelays.length > 0) {
        const grpSim = document.createElement('optgroup');
        grpSim.label = 'Synthetic Offline Test Relays (CI / Cut Stress)';
        for (const p of simRelays) grpSim.appendChild(makeOpt(p));
        dom.tier3PresetSelect.appendChild(grpSim);
      }

      const firstOnlineCam = realCams.find((p) => p.online);
      if (userPickedTier3Preset && prevVal && Array.from(dom.tier3PresetSelect.options).some((o) => o.value === prevVal)) {
        dom.tier3PresetSelect.value = prevVal;
      } else if (firstOnlineCam) {
        dom.tier3PresetSelect.value = firstOnlineCam.id;
      }
    } catch (_err) {
      enableStaticHostMode();
    }
  }

  if (!isStaticHostMode) {
    setTimeout(() => { refreshTier3LivePresets(false); }, 250);
  }

  function openLiveStreamModal() {
    dom.btnFallbackTier2ToTier3.hidden = true;
    if (isStaticHostMode) {
      setLiveDialogBanner('Static CDN host detected: Tier-3 FFmpeg HLS relay requires running local `node scripts/dev_server.mjs`. Bundled MP4s, swarms, webcam, local MP4 drop, and Tier-2 CORS streams run 100% in your browser.');
    } else {
      refreshTier3LivePresets(false);
    }
    if (!liveDialog.open) liveDialog.showModal();
  }

  btnLiveStream.addEventListener('click', openLiveStreamModal);
  document.getElementById('btnReopenLiveModal').addEventListener('click', openLiveStreamModal);
  document.getElementById('btnRefreshTier3Presets').addEventListener('click', () => {
    if (isStaticHostMode) {
      setLiveDialogBanner('Static CDN host: Tier-3 FFmpeg relay requires local `node scripts/dev_server.mjs`. Use Tier-1 Tab Capture, Tier-2 Direct CORS URL, Webcam, or bundled MP4s.');
      return;
    }
    setLiveDialogBanner('Probing Caltrans HLS camera endpoints...');
    refreshTier3LivePresets(true).then(() => {
      setLiveDialogBanner('Camera health probe updated. Select any [LIVE ONLINE] highway camera and click Connect.');
    });
  });

  dom.liveCropSelect.addEventListener('change', () => {
    const parts = dom.liveCropSelect.value.split(',').map(Number);
    if (parts.length === 4 && parts.every(Number.isFinite)) {
      worker.postMessage({ type: 'set_crop', cropNorm: parts });
    }
  });

  document.getElementById('btnStopLiveFeed').addEventListener('click', () => {
    btnTestbench.click();
  });

  // Tier 1: Browser Tab / Window Capture via getDisplayMedia
  document.getElementById('btnStartTier1Capture').addEventListener('click', async () => {
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getDisplayMedia !== 'function') {
      setLiveDialogBanner('getDisplayMedia is unavailable in this browser context. Use Tier 3 fMP4 MSE Relay instead.', true);
      return;
    }
    const reqEpoch = sourceEpoch;
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: { ideal: 30, max: 60 } },
        audio: false,
      });
      if (reqEpoch !== sourceEpoch) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      stopMediaCapture();
      const myEpoch = sourceEpoch;
      sampleVideoSelect.value = '';
      if (activeResMode === 0) document.getElementById('btnResHd')?.click();
      activeMediaStream = stream;
      activeVideoFps = 30.0;
      videoEl.loop = false;
      videoEl.srcObject = stream;
      await videoEl.play();
      if (myEpoch !== sourceEpoch) return;
      const vTrack = stream.getVideoTracks()[0];
      if (vTrack) {
        vTrack.addEventListener('ended', () => {
          if (activeSourceMode === 'live' && activeMediaStream === stream) btnTestbench.click();
        }, { once: true });
      }
      setSourceButtons(btnLiveStream);
      setActiveShowcaseChip('');
      syncContextualSourceUi('live', { kind: 'LIVE_TAB' });
      dom.liveTierBadge.textContent = 'TIER 1 · TAB / WINDOW CAPTURE';
      dom.hudSourceLabel.textContent = `SOURCE: LIVE TAB CAPTURE (${vTrack?.label || 'DISPLAY ROI'})`;
      worker.postMessage({ type: 'set_source', sourceMode: 'live', fpsHint: 30 });
      dom.liveCropSelect.dispatchEvent(new Event('change'));
      startVideoPump();
      if (liveDialog.open) liveDialog.close('tier1');
    } catch (err) {
      if (stream && activeMediaStream !== stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      setLiveDialogBanner(`Tab capture cancelled or denied: ${err.message || err}`, true);
    }
  });

  // Helper: Parse ISO-BMFF avcC box inside fMP4 init segment (ftyp + moov) to construct exact MSE codec string
  function findAvcCodecMime(u8) {
    for (let i = 0; i + 11 < u8.length; i++) {
      if (u8[i] === 0x61 && u8[i + 1] === 0x76 && u8[i + 2] === 0x63 && u8[i + 3] === 0x43) {
        const prof = u8[i + 5].toString(16).padStart(2, '0').toUpperCase();
        const comp = u8[i + 6].toString(16).padStart(2, '0').toUpperCase();
        const lvl = u8[i + 7].toString(16).padStart(2, '0').toUpperCase();
        return `video/mp4; codecs="avc1.${prof}${comp}${lvl}"`;
      }
    }
    return null;
  }

  function appendBytes(acc, chunk) {
    if (acc.length === 0) return chunk;
    const merged = new Uint8Array(acc.length + chunk.length);
    merged.set(acc, 0);
    merged.set(chunk, acc.length);
    return merged;
  }

  // Tier 3: Zero-Dependency Same-Origin FFmpeg fMP4 -> Native MediaSource (MSE) Relay
  async function startTier3MseStream(cfg, isAutoRetry = false) {
    if (typeof MediaSource === 'undefined') {
      setLiveDialogBanner('MediaSource Extensions (MSE) unsupported in this browser. Use Tier 1 Tab Capture.', true);
      return;
    }
    stopMediaCapture(isAutoRetry);
    activeLiveConfig = cfg;
    sampleVideoSelect.value = '';
    if (activeResMode === 0) document.getElementById('btnResHd')?.click();
    if (typeof cfg.targetLab === 'number' && currentLabMode !== cfg.targetLab) {
      document.querySelector(`.lab-btn[data-lab="${cfg.targetLab}"]`)?.click();
    }

    const myEpoch = sourceEpoch;
    const abortCtrl = new AbortController();
    activeMseAbort = abortCtrl;
    const ms = new MediaSource();
    activeMseMediaSource = ms;
    currentObjectUrl = URL.createObjectURL(ms);
    activeVideoFps = cfg.fpsHint || 30.0;
    videoEl.loop = false;
    videoEl.src = currentObjectUrl;

    setSourceButtons(btnLiveStream);
    setActiveShowcaseChip(cfg.showcaseId || '');
    syncContextualSourceUi('live', { kind: 'LIVE_MSE' });
    dom.liveTierBadge.textContent = 'TIER 3 · FFmpeg fMP4 MSE';
    dom.liveStreamMeta.textContent = 'CONNECTING LIVE HLS...';
    dom.hudSourceLabel.textContent = `SOURCE: LIVE fMP4 MSE (${cfg.label.toUpperCase()})`;
    if (!isAutoRetry) {
      worker.postMessage({ type: 'set_source', sourceMode: 'live', fpsHint: activeVideoFps });
      dom.liveCropSelect.dispatchEvent(new Event('change'));
    }

    ms.addEventListener('sourceopen', async () => {
      if (myEpoch !== sourceEpoch) return;
      let sb = null;
      const appendQueue = [];
      let queuedBytes = 0;
      const MAX_QUEUE_BYTES = 16 * 1024 * 1024;
      let initAccum = new Uint8Array(0);
      let playbackStarted = false;
      let healthySpanReached = false;
      let retryResetDone = false;
      let streamEnded = false;
      let fatalMseError = null;
      const mseSessionStartMs = performance.now();
      const minStartBufferSec = cfg.presetId && cfg.presetId.startsWith('sim_') ? 0.45 : 1.20;

      const maybeCloseMseStream = () => {
        if (streamEnded && appendQueue.length === 0 && (!sb || !sb.updating) && ms.readyState === 'open') {
          try {
            ms.endOfStream();
          } catch (_eosErr) { /* ignore */ }
        }
      };

      const maybeStartMsePlayback = () => {
        if (!sb || sb.buffered.length === 0 || myEpoch !== sourceEpoch) return;
        const start0 = sb.buffered.start(0);
        const endLast = sb.buffered.end(sb.buffered.length - 1);
        const spanSec = endLast - start0;
        if (!streamEnded && spanSec >= minStartBufferSec && performance.now() - mseSessionStartMs >= 15000) {
          healthySpanReached = true;
        }
        if (!playbackStarted) {
          if (spanSec < minStartBufferSec && !streamEnded) return;
          if (spanSec <= 0.02) return;
          playbackStarted = true;
          if (videoEl.currentTime + 0.05 < start0) {
            videoEl.currentTime = start0 + 0.02;
          }
          safePlayVideo().then((ok) => {
            if (ok && myEpoch === sourceEpoch) {
              if (healthySpanReached && !streamEnded && !retryResetDone) {
                activeLiveRetryCount = 0;
                retryResetDone = true;
              }
              startVideoPump();
            }
          });
        } else if (healthySpanReached && !streamEnded && !retryResetDone && !videoEl.paused) {
          activeLiveRetryCount = 0;
          retryResetDone = true;
        }
      };

      const pumpSourceBuffer = () => {
        if (!sb || sb.updating || myEpoch !== sourceEpoch || ms.readyState !== 'open') return;
        if (videoEl.currentTime > 12.0 && sb.buffered.length > 0) {
          const start0 = sb.buffered.start(0);
          const evictTo = videoEl.currentTime - 8.0;
          if (evictTo - start0 > 2.0) {
            try {
              sb.remove(start0, evictTo);
              return;
            } catch (_e) {
              // Continue to append if remove is rejected
            }
          }
        }
        if (appendQueue.length > 0) {
          let nextChunk;
          if (appendQueue.length === 1) {
            nextChunk = appendQueue.shift();
          } else {
            let totalBytes = 0;
            for (let i = 0; i < appendQueue.length; i++) totalBytes += appendQueue[i].byteLength;
            nextChunk = new Uint8Array(totalBytes);
            let off = 0;
            for (let i = 0; i < appendQueue.length; i++) {
              nextChunk.set(appendQueue[i], off);
              off += appendQueue[i].byteLength;
            }
            appendQueue.length = 0;
          }
          queuedBytes = Math.max(0, queuedBytes - nextChunk.byteLength);
          try {
            sb.appendBuffer(nextChunk);
          } catch (appendErr) {
            if (myEpoch !== sourceEpoch || abortCtrl.signal.aborted) return;
            if (appendErr && appendErr.name === 'QuotaExceededError' && sb.buffered.length > 0) {
              appendQueue.unshift(nextChunk);
              queuedBytes += nextChunk.byteLength;
              const s0 = sb.buffered.start(0);
              const sEnd = sb.buffered.end(sb.buffered.length - 1);
              const evictTarget = Math.min(sEnd - 0.5, videoEl.currentTime - 1.5);
              if (evictTarget > s0 + 0.2 && !sb.updating) {
                try {
                  sb.remove(s0, evictTarget);
                  return;
                } catch (_remErr) {
                  // Fall through to stream recovery
                }
              }
            }
            fatalMseError = appendErr;
            abortCtrl.abort();
          }
        } else {
          maybeCloseMseStream();
        }
      };

      try {
        const qs = cfg.customUrl
          ? `url=${encodeURIComponent(cfg.customUrl)}`
          : `preset=${encodeURIComponent(cfg.presetId)}`;
        const resp = await fetch(`/api/live-stream?${qs}`, { signal: abortCtrl.signal });
        if (!resp.ok || !resp.body) {
          let errDetail = `HTTP ${resp.status}`;
          try {
            const j = await resp.json();
            if (j && j.error) errDetail = j.error;
          } catch (_e) { /* ignore */ }
          const httpErr = new Error(errDetail);
          httpErr.httpStatus = resp.status;
          throw httpErr;
        }

        if (liveDialog.open) liveDialog.close('tier3');

        const reader = resp.body.getReader();
        while (myEpoch === sourceEpoch) {
          if (ms.readyState !== 'open' || videoEl.error || fatalMseError) {
            abortCtrl.abort();
            break;
          }
          const { value, done } = await reader.read();
          if (done || myEpoch !== sourceEpoch) break;
          if (ms.readyState !== 'open' || videoEl.error || fatalMseError) {
            abortCtrl.abort();
            break;
          }
          if (!value || value.byteLength === 0) continue;

          if (!sb) {
            initAccum = appendBytes(initAccum, value);
            const detectedMime = findAvcCodecMime(initAccum);
            if (!detectedMime && initAccum.byteLength < 65536) continue;
            const mime = detectedMime || 'video/mp4; codecs="avc1.42E01F"';
            if (!MediaSource.isTypeSupported(mime)) {
              throw new Error(`Unsupported fMP4 codec: ${mime}`);
            }
            sb = ms.addSourceBuffer(mime);
            sb.mode = 'segments';
            sb.addEventListener('error', () => {
              if (myEpoch === sourceEpoch && !fatalMseError) {
                fatalMseError = new Error('MSE SourceBuffer decode error');
                abortCtrl.abort();
              }
            });
            sb.addEventListener('updateend', () => {
              maybeStartMsePlayback();
              pumpSourceBuffer();
            });
            appendQueue.push(initAccum);
            queuedBytes += initAccum.byteLength;
            initAccum = new Uint8Array(0);
            pumpSourceBuffer();
          } else {
            if (queuedBytes + value.byteLength > MAX_QUEUE_BYTES) {
              fatalMseError = new Error('MSE append queue overflow');
              abortCtrl.abort();
              break;
            }
            appendQueue.push(value);
            queuedBytes += value.byteLength;
            pumpSourceBuffer();
          }
        }
        streamEnded = true;
        if (!fatalMseError && !videoEl.error) {
          maybeStartMsePlayback();
          maybeCloseMseStream();
        }

        // Reconnect automatically up to 3 times if continuous stream ended or glitched while still in live mode
        if (myEpoch === sourceEpoch && activeSourceMode === 'live' && activeLiveConfig) {
          if (activeLiveRetryCount < 3) {
            activeLiveRetryCount++;
            let extraPlayDrainMs = 0;
            if (!fatalMseError && !videoEl.error && playbackStarted && sb && sb.buffered.length > 0) {
              const remSec = sb.buffered.end(sb.buffered.length - 1) - videoEl.currentTime;
              if (remSec > 0.25) {
                extraPlayDrainMs = Math.min(15000, Math.round(remSec * 1000));
              }
            }
            const delayMs = 900 * activeLiveRetryCount + extraPlayDrainMs;
            dom.liveStreamMeta.textContent = `RECONNECTING (${activeLiveRetryCount}/3)...`;
            activeMseRetryTimer = setTimeout(() => {
              if (activeSourceMode === 'live' && activeLiveConfig) {
                startTier3MseStream(activeLiveConfig, true);
              }
            }, delayMs);
          } else {
            dom.liveStreamMeta.textContent =
              fatalMseError || videoEl.error ? 'STREAM ERROR (3/3)' : 'STREAM ENDED · CLICK RETRY';
          }
        }
      } catch (err) {
        if ((abortCtrl.signal.aborted && !fatalMseError) || myEpoch !== sourceEpoch) return;
        const realErr = fatalMseError || err;
        const status = realErr && typeof realErr.httpStatus === 'number' ? realErr.httpStatus : 0;
        const nonRetryable4xx = status >= 400 && status < 500 && status !== 408 && status !== 429;
        if (
          !nonRetryable4xx &&
          myEpoch === sourceEpoch &&
          activeSourceMode === 'live' &&
          activeLiveConfig &&
          activeLiveRetryCount < 3
        ) {
          activeLiveRetryCount++;
          const delayMs = 900 * activeLiveRetryCount;
          dom.liveStreamMeta.textContent = `RECONNECTING (${activeLiveRetryCount}/3)...`;
          activeMseRetryTimer = setTimeout(() => {
            if (activeSourceMode === 'live' && activeLiveConfig) {
              startTier3MseStream(activeLiveConfig, true);
            }
          }, delayMs);
          return;
        }
        setLiveDialogBanner(`Tier 3 relay error: ${realErr.message || realErr}. Ensure dev_server.mjs + ffmpeg are running.`, true);
        dom.liveStreamMeta.textContent = activeLiveRetryCount >= 3 ? 'STREAM ERROR (3/3)' : 'STREAM ERROR';
      }
    }, { once: true });
  }

  document.getElementById('btnStartTier3Mse').addEventListener('click', () => {
    if (isStaticHostMode) {
      setLiveDialogBanner('Static CDN host: Tier-3 FFmpeg HLS relay requires running local `node scripts/dev_server.mjs`.', true);
      return;
    }
    const customRaw = dom.tier3CustomHlsUrl.value.trim();
    if (customRaw) {
      if (!/^https?:\/\/.+/i.test(customRaw)) {
        dom.tier3CustomHlsUrl.setCustomValidity('Enter a valid http:// or https:// URL.');
        syncAria(dom.tier3CustomHlsUrl, true);
        dom.tier3CustomHlsUrl.focus();
        return;
      }
      dom.tier3CustomHlsUrl.setCustomValidity('');
      syncAria(dom.tier3CustomHlsUrl, false);
      startTier3MseStream({
        presetId: '',
        customUrl: customRaw,
        label: 'Custom HLS Relay',
        fpsHint: 30,
        targetLab: currentLabMode,
      });
      return;
    }
    const opt = dom.tier3PresetSelect.selectedOptions[0];
    const presetId = dom.tier3PresetSelect.value || 'sim_live_cut_demo';
    const cleanLabel = opt ? opt.textContent.replace(/^\[[^\]]+\]\s*/, '') : presetId;
    const targetLab = opt ? parseInt(opt.dataset.lab || '0', 10) : 0;
    const fpsHint = opt ? parseFloat(opt.dataset.fps || '30') : 30;
    startTier3MseStream({ presetId, customUrl: '', label: cleanLabel, fpsHint, targetLab });
  });

  // Tier 2: Direct CORS-Verified Stream URL with 1-px getImageData Readback Probe
  document.getElementById('btnStartTier2Url').addEventListener('click', async () => {
    const rawUrl = dom.tier2StreamUrl.value.trim();
    if (!rawUrl || !/^https?:\/\/.+/i.test(rawUrl)) {
      dom.tier2StreamUrl.setCustomValidity('Enter a valid http:// or https:// stream URL.');
      syncAria(dom.tier2StreamUrl, true);
      dom.tier2StreamUrl.focus();
      return;
    }
    dom.tier2StreamUrl.setCustomValidity('');
    syncAria(dom.tier2StreamUrl, false);
    dom.btnFallbackTier2ToTier3.hidden = true;
    setLiveDialogBanner('Probing direct CORS stream & verifying untainted RGBA pixel readback...');

    const reqEpoch = sourceEpoch;
    const probeVid = document.createElement('video');
    probeVid.muted = true;
    probeVid.playsInline = true;
    probeVid.crossOrigin = 'anonymous';
    try {
      probeVid.src = rawUrl;
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('Stream load timed out')), 5000);
        probeVid.addEventListener('loadeddata', () => { clearTimeout(timer); resolve(); }, { once: true });
        probeVid.addEventListener('error', () => { clearTimeout(timer); reject(new Error('Unsupported codec or CORS blocked')); }, { once: true });
      });
      if (reqEpoch !== sourceEpoch) return;

      fallbackCapCanvas.width = 4;
      fallbackCapCanvas.height = 4;
      fallbackCapCtx.drawImage(probeVid, 0, 0, 4, 4);
      fallbackCapCtx.getImageData(0, 0, 1, 1); // Throws SecurityError if CORS-tainted!

      probeVid.pause();
      probeVid.removeAttribute('src');
      probeVid.load();

      stopMediaCapture();
      const myEpoch = sourceEpoch;
      sampleVideoSelect.value = '';
      if (activeResMode === 0) document.getElementById('btnResHd')?.click();
      videoEl.crossOrigin = 'anonymous';
      videoEl.loop = true;
      videoEl.src = rawUrl;
      await safePlayVideo();
      if (myEpoch !== sourceEpoch) return;

      setSourceButtons(btnLiveStream);
      setActiveShowcaseChip('');
      syncContextualSourceUi('live', { kind: 'LIVE_URL' });
      dom.liveTierBadge.textContent = 'TIER 2 · DIRECT CORS URL';
      dom.hudSourceLabel.textContent = 'SOURCE: LIVE DIRECT CORS STREAM';
      worker.postMessage({ type: 'set_source', sourceMode: 'live', fpsHint: 30 });
      dom.liveCropSelect.dispatchEvent(new Event('change'));
      startVideoPump();
      if (liveDialog.open) liveDialog.close('tier2');
    } catch (err) {
      try {
        probeVid.pause();
        probeVid.removeAttribute('src');
        probeVid.load();
      } catch (_e) { /* ignore */ }
      if (reqEpoch !== sourceEpoch) return;
      dom.btnFallbackTier2ToTier3.hidden = false;
      setLiveDialogBanner(
        `Direct CORS / codec check failed (${err.message || err}). Click "Relay URL via Tier 3 FFmpeg Proxy" below or use Tier 1 Tab Capture.`,
        true
      );
    }
  });

  dom.btnFallbackTier2ToTier3.addEventListener('click', () => {
    const rawUrl = dom.tier2StreamUrl.value.trim();
    if (!rawUrl) return;
    dom.tier3CustomHlsUrl.value = rawUrl;
    startTier3MseStream({
      presetId: '',
      customUrl: rawUrl,
      label: 'Relayed Stream',
      fpsHint: 30,
      targetLab: currentLabMode,
    });
  });

  btnTestbench.addEventListener('click', () => {
    initialPresetBooted = true;
    stopProbeExciter(false);
    stopMediaCapture();
    sampleVideoSelect.value = '';
    setSourceButtons(btnTestbench);
    setActiveShowcaseChip('');
    syncContextualSourceUi('testbench', { kind: 'TESTBENCH', originPreset: null });
    dom.hudSourceLabel.textContent = 'SOURCE: BLIND PIXEL PILLAR TESTBENCH (getImageData ONLY)';
    dom.hudSourceLabel.title = dom.hudSourceLabel.textContent;
    worker.postMessage({ type: 'set_source', sourceMode: 'testbench' });
  });

  async function startWebcamCapture(opts = null) {
    const reqEpoch = sourceEpoch;
    const highlightShowcase = opts && typeof opts.highlightShowcase === 'string' ? opts.highlightShowcase : '';
    let stream = null;
    try {
      const wantAudio =
        !(opts && opts.preferMic === false) &&
        !probeExciterActive &&
        Boolean(dom.rsVibroModeSelect && dom.rsVibroModeSelect.value === '2');
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { ideal: 60 } },
          audio: wantAudio,
        });
      } catch (audioErr) {
        if (!wantAudio) throw audioErr;
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { ideal: 60 } },
          audio: false,
        });
      }
      if (reqEpoch !== sourceEpoch) {
        stream.getTracks().forEach((t) => t.stop());
        return false;
      }
      stopMediaCapture();
      const myEpoch = sourceEpoch;
      sampleVideoSelect.value = '';
      activeMediaStream = stream;
      if (wantAudio && typeof stream.getAudioTracks === 'function') {
        const audTracks = stream.getAudioTracks();
        for (let i = 0; i < audTracks.length; i++) {
          const atr = audTracks[i];
          if (atr && !rsOwnedWebcamAudioTracks.includes(atr)) {
            rsOwnedWebcamAudioTracks.push(atr);
          }
        }
      }
      rsAudioMicReqStream = wantAudio ? stream : null;
      const vidTrack = typeof stream.getVideoTracks === 'function' ? stream.getVideoTracks()[0] : null;
      const trackFps = vidTrack && typeof vidTrack.getSettings === 'function' ? Number(vidTrack.getSettings().frameRate) : NaN;
      activeVideoFps = Number.isFinite(trackFps) && trackFps >= 8.0 && trackFps <= 240.0 ? trackFps : 30.0;
      videoEl.srcObject = stream;
      await videoEl.play();
      if (myEpoch !== sourceEpoch) return false;
      initialPresetBooted = true;
      dispatchInstrument({ type: 'DISMISS_WELCOME' }, true);
      setSourceButtons(btnWebcam);
      setActiveShowcaseChip(highlightShowcase);
      syncContextualSourceUi('webcam', { kind: 'WEBCAM', originPreset: highlightShowcase || null });
      dom.hudSourceLabel.textContent = highlightShowcase === 'live_sound_probe'
        ? 'SOURCE: LIVE WEBCAM · 110+220 Hz SOUND PROBE (RS COHERENCE + ELASTO)'
        : 'SOURCE: LIVE WEBCAM OPTICAL FEED (DEDUPLICATED)';
      dom.hudSourceLabel.title = dom.hudSourceLabel.textContent;
      worker.postMessage({ type: 'set_source', sourceMode: 'webcam', fpsHint: Math.round(activeVideoFps) });
      syncRsLiveAudioCapture();
      startVideoPump();
      return true;
    } catch (err) {
      if (stream && activeMediaStream !== stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      if (reqEpoch === sourceEpoch) {
        dom.hudSourceLabel.textContent = `WEBCAM UNAVAILABLE: ${err.message}`;
      }
      return false;
    }
  }

  btnWebcam.addEventListener('click', () => {
    startWebcamCapture();
  });

  let showcaseInspectSubNote = 'Zero upload · Drag any MP4 onto canvas';

  function updateShowcaseProofCards(m, medMs) {
    if (!dom.scValKernel) return;
    const msTxt = typeof medMs === 'number' && Number.isFinite(medMs) ? `${medMs.toFixed(2)} ms / frame` : '-- ms / frame';
    setTextIfChanged(dom.scValKernel, msTxt);
    setTextIfChanged(dom.scSubKernel, `2.13 MiB · 0 heap allocs · ${m.resMode ? '640×360 HD' : '320×180 SD'}`);

    const c1k = m.chronos1000;
    const bos = m.bos;
    const w2 = m.wave2;
    const w2w = w2 && w2.wind && w2.wind.available && w2.wind.reqMode > 0 ? w2.wind : null;
    const w2e = w2 && w2.elasto && w2.elasto.available && w2.elasto.reqMode > 0 ? w2.elasto : null;
    const shownSwaps = m.sourceMode === 'testbench' ? (m.oracleSwaps ?? 0) : (m.padicSwaps ?? 0);
    if (w2e) {
      setTextIfChanged(dom.scValTracks, `${w2e.mode === 4 ? 'ABOS' : 'ELASTO'} ${w2e.stateLabel} · ${w2e.validCells || 0} cells`);
      setTextIfChanged(dom.scSubTracks, `${w2e.classLabel || 'NONE'} · segs=${w2e.welchSegs || 0} · b1=${w2e.beta1 || 0}`);
      const elStat = w2e.state === 3
        ? (w2e.mode === 4
            ? `|Phi| ${(w2e.abosPhiRms || 0).toFixed(3)} · curl ${(w2e.abosCurlRms || 0).toFixed(3)}`
            : `c=${Math.round(w2e.medC || 0)} px/s · a=${(w2e.alpha || 0).toFixed(2)}`)
        : `STATE ${w2e.stateLabel}`;
      setTextIfChanged(dom.scValPhysics, elStat);
      setTextIfChanged(dom.scSubPhysics, `${(w2e.f0 || 0).toFixed(1)} Hz · coh ${(w2e.meanCoh || 0).toFixed(2)} · aniso ${(w2e.anisotropy || 1).toFixed(2)}`);
      setTextIfChanged(dom.scValInspect, w2e.mode === 4 ? '256x128 Lock-In Acoustic BOS' : '64x36 Wave Speed & Power Flow');
    } else if (w2w) {
      setTextIfChanged(dom.scValTracks, `WIND ${w2w.stateLabel} · ${w2w.validCells || 0} cells · b1=${w2w.beta1 || 0}`);
      setTextIfChanged(dom.scSubTracks, `Uinf ${(w2w.uInf || 0).toFixed(1)} px/s · div ${(w2w.divFrac || 0).toFixed(2)}`);
      const wStat = w2w.state === 3
        ? `CL ${(w2w.cl || 0).toFixed(2)} · CD ${(w2w.cd || 0).toFixed(2)}`
        : `nu ${(w2w.nu || 0).toFixed(1)} px²/s`;
      setTextIfChanged(dom.scValPhysics, wStat);
      setTextIfChanged(dom.scSubPhysics, `KJ CL ${(w2w.clKj || 0).toFixed(2)} · nu R² ${(w2w.nuR2 || 0).toFixed(2)} · Cp [${(w2w.cpMin || 0).toFixed(1)}, ${(w2w.cpMax || 0).toFixed(1)}]`);
      setTextIfChanged(dom.scValInspect, '64x36 Optical Cp + Navier-Stokes');
    } else if (bos && bos.available && bos.reqMode > 0) {
      const usable = m.metrology ? m.metrology.usableCovCount : 0;
      const heroCands = m.metrology ? m.metrology.promotedCandidates : 0;
      setTextIfChanged(dom.scValTracks, `BOS ${bos.stateLabel} · ${bos.validDisp}/32768 cells`);
      setTextIfChanged(dom.scSubTracks, `${usable} 2x2 cov · ${heroCands} hero promo · 1:1 ${m.roi && m.roi.native1to1Active ? 'ON' : 'OFF'}`);
      const bosStat = bos.state === 3
        ? `TRIPOD DISTURBED (${(bos.reasonFlags || []).slice(0, 2).join('+') || 'MOTION'})`
        : bos.state === 2
        ? `RMS ${bos.rmsDisp.toFixed(3)} px · curl ${bos.curlRms.toFixed(3)}`
        : `WARM ${bos.warmupCnt}/16 frames`;
      setTextIfChanged(dom.scValPhysics, bosStat);
      setTextIfChanged(dom.scSubPhysics, `Neumann Phi · PCG ${bos.pcgIters} iters · res ${bos.trueRelRes.toExponential(1)}`);
      setTextIfChanged(dom.scValInspect, m.roi && m.roi.native1to1Active ? `1:1 Native ROI (${m.roi.cropW}x${m.roi.cropH})` : 'Click canvas to inspect micro-track');
    } else if (c1k && c1k.swarmActive) {
      setTextIfChanged(dom.scValTracks, `${fmtIntOrDash(c1k.confirmedCount)} active (peak ${fmtIntOrDash(c1k.peakSimultaneous)})`);
      setTextIfChanged(dom.scSubTracks, `unique ${fmtIntOrDash(c1k.uniqueTracks)} · coast ${fmtIntOrDash(c1k.coastingCount)} · IDSW ${shownSwaps}`);
      const hasFtle = Number.isFinite(c1k.maxFtleS1) && c1k.maxFtleS1 > -900.0;
      const vortVal = Number.isFinite(c1k.maxVorticityS1) ? c1k.maxVorticityS1.toFixed(1) : '0.0';
      setTextIfChanged(dom.scValPhysics, `curl ${vortVal} s⁻¹${hasFtle ? ` · FTLE ${c1k.maxFtleS1.toFixed(2)}` : ''}`);
      setTextIfChanged(dom.scSubPhysics, `64×36 Eulerian flow · ${(c1k.fieldBackend || 'webgl2').toUpperCase()}`);
      const insp = dom.swarmInspectReadout ? dom.swarmInspectReadout.textContent : '';
      if (insp && !insp.startsWith('CLICK HUD')) {
        setTextIfChanged(dom.scValInspect, insp);
      } else {
        setTextIfChanged(dom.scValInspect, 'Click any bird on canvas to inspect');
      }
    } else {
      setTextIfChanged(dom.scValTracks, `${m.activeTargets} / 16 locks · ${m.bridgedFrames || 0}f bridges`);
      const calLbl = dom.calBadge ? dom.calBadge.textContent : 'PX UNITS';
      setTextIfChanged(dom.scSubTracks, `${calLbl} · 2-adic swaps ◆ ${shownSwaps}`);
      const regTxt = dom.physRegimeBadge ? dom.physRegimeBadge.textContent : 'STANDBY';
      const r2Txt = dom.kpiPhysicsR2 ? dom.kpiPhysicsR2.textContent : '--%';
      setTextIfChanged(dom.scValPhysics, `${regTxt}${r2Txt && r2Txt !== '--%' ? ` · R² ${r2Txt}` : ''}`);
      const odeTxt = dom.physicsOdeEq && dom.physicsOdeEq.textContent ? dom.physicsOdeEq.textContent : 'Weak-form STRidge/AICc + K=8 FISTA';
      setTextIfChanged(dom.scSubPhysics, odeTxt);
      const spdTxt = dom.kpiKinSpeed ? dom.kpiKinSpeed.textContent : '';
      if (m.focusTrackId > 0) {
        setTextIfChanged(dom.scValInspect, `Pinned Track T${m.focusTrackId} · ${spdTxt}`);
      } else if (m.primaryTrackId > 0) {
        setTextIfChanged(dom.scValInspect, `Auto Focus T${m.primaryTrackId} · ${spdTxt}`);
      } else {
        setTextIfChanged(dom.scValInspect, 'Click any moving target on canvas');
      }
    }
    setTextIfChanged(dom.scSubInspect, showcaseInspectSubNote);
  }

  function setShowcaseWipePercent(pct, notifyWorker = true) {
    const clamped = Math.max(0, Math.min(100, Math.round(Number(pct) || 0)));
    if (dom.hudCanvasStage) {
      dom.hudCanvasStage.style.setProperty('--wipe-pct', String(clamped));
    }
    if (dom.showcaseWipeSlider) {
      dom.showcaseWipeSlider.value = String(clamped);
      dom.showcaseWipeSlider.style.setProperty('--attr-min', dom.showcaseWipeSlider.min || '0');
      dom.showcaseWipeSlider.style.setProperty('--attr-max', dom.showcaseWipeSlider.max || '100');
      dom.showcaseWipeSlider.style.setProperty('--control-value', String(clamped));
      dom.showcaseWipeSlider.setAttribute('aria-valuetext', `${clamped}% raw video, ${100 - clamped}% spectral wavefield`);
    }
    if (dom.showcaseWipeReadout) {
      dom.showcaseWipeReadout.textContent = clamped === 0
        ? '0% RAW · FULL SPECTRAL VIEW'
        : clamped === 100
        ? '100% RAW VIDEO · 0% SPECTRAL'
        : `${clamped}% RAW · ${100 - clamped}% SPECTRAL`;
    }
    dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', overlays: { wipePct: clamped }, markPresetModified: false }, false);
    if (notifyWorker) {
      worker.postMessage({ type: 'params', rawWipeRatio: clamped / 100.0 });
    }
  }

  function setHeroScenePressed(activeSceneKey) {
    document.querySelectorAll('.hero-scene-btn').forEach((b) => {
      b.setAttribute('aria-pressed', String(Boolean(activeSceneKey) && b.dataset.heroScene === activeSceneKey));
    });
  }

  async function safePlayVideo() {
    try {
      await videoEl.play();
      return true;
    } catch (err) {
      if (err && (err.name === 'AbortError' || err.name === 'NotAllowedError')) return false;
      throw err;
    }
  }

  async function loadLocalVideoFile(file) {
    if (!file) return;
    if (file.type && !file.type.startsWith('video/') && !/\.(mp4|webm|mov|m4v|mkv|ogg)$/i.test(file.name || '')) {
      if (dom.acquisitionStatusBanner) {
        dom.acquisitionStatusBanner.dataset.severity = 'danger';
        dom.acquisitionStatusBanner.textContent = `Unsupported file "${file.name}". Drop an MP4, WebM, or MOV video.`;
      }
      if (dom.showcaseToolbarStatus) dom.showcaseToolbarStatus.textContent = `Unsupported file type: ${file.name}`;
      return;
    }
    initialPresetBooted = true;
    const prevUrl = currentObjectUrl;
    const nextObjectUrl = URL.createObjectURL(file);
    stopMediaCapture();
    const myEpoch = sourceEpoch;
    sampleVideoSelect.value = '';
    setSourceButtons(null);
    setActiveShowcaseChip('');
    if (activeResMode === 0) {
      document.getElementById('btnResHd')?.click();
    }
    showcaseInspectSubNote = 'Loaded local file in WebAssembly (0 bytes uploaded)';
    if (dom.showcaseToolbarStatus) {
      dom.showcaseToolbarStatus.textContent = `Loaded local video ${file.name}`;
    }
    activeVideoFps = 30.0;
    activeVideoFile = file;
    activeVideoUrl = '';
    currentObjectUrl = nextObjectUrl;
    if (prevUrl && prevUrl !== nextObjectUrl) {
      try { URL.revokeObjectURL(prevUrl); } catch (_) {}
    }
    videoEl.src = currentObjectUrl;
    videoEl.loop = true;
    let playOk = false;
    try {
      playOk = await safePlayVideo();
    } catch (err) {
      if (dom.acquisitionStatusBanner) {
        dom.acquisitionStatusBanner.dataset.severity = 'danger';
        dom.acquisitionStatusBanner.textContent = `Could not decode "${file.name}": ${err?.message || 'unsupported codec'}`;
      }
      if (dom.showcaseToolbarStatus) dom.showcaseToolbarStatus.textContent = `Decode failed: ${file.name}`;
      return;
    }
    if (!playOk || myEpoch !== sourceEpoch) return;
    // Preserve user's current organs, overlays, and calibration preferences on file drop!
    syncContextualSourceUi('video', { kind: 'FILE', label: `SOURCE: MP4 FILE (${file.name})`, originPreset: null });
    dom.hudSourceLabel.textContent = `SOURCE: MP4 FILE (${file.name})`;
    dom.hudSourceLabel.title = dom.hudSourceLabel.textContent;
    const srcPayload = { type: 'set_source', sourceMode: 'video', fpsHint: 30 };
    const curEmLocal = dom.elastoModeSelect ? (parseInt(dom.elastoModeSelect.value || '0', 10) || 0) : 0;
    const curRmLocal = dom.rsVibroModeSelect ? (parseInt(dom.rsVibroModeSelect.value || '0', 10) || 0) : 0;
    if (curEmLocal === 1 || curEmLocal === 4 || curRmLocal === 2) {
      await decodeMp4AudioTrack(file, myEpoch);
      if (myEpoch !== sourceEpoch) return;
    }
    worker.postMessage(srcPayload);
    startVideoPump();
  }

  mp4Input.addEventListener('change', async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    await loadLocalVideoFile(file);
    mp4Input.value = '';
  });

  const showcaseClipMap = {
    model_rockets: { url: './samples/model_rocket_full_flights.mp4', cal: 'preset' },
    rocket_vtvl: { url: './samples/rocket_vtvl_full_flight.mp4', cal: 'preset' },
    water_rockets: { url: './samples/water_rocket_parachute_flights.mp4', cal: 'preset' },
    cologne_motorway: { url: './samples/cologne_motorway_occlusion.mp4', cal: 'preset' },
    rally_pan: { url: './samples/rally_ego_pan_tracking.mp4', cal: 'none' },
    chaos_pendulum: { url: './samples/chaotic_magnetic_pendulum.mp4', cal: 'preset' },
    coupled_metronomes: { url: './samples/coupled_metronomes_sync.mp4', cal: 'preset' },
    newtons_cradle: { url: './samples/newtons_cradle_impulses.mp4', cal: 'preset' },
    starling_1000: { url: './samples/starling_murmuration_1000.mp4', cal: 'none', resMode: 1, swarmMode: 2, overlayMode: 1, bosMode: 0, metrologyMode: 0, showCovEllipses: false },
    bos_plume: { url: './samples/bos_thermal_plume.mp4', cal: 'none', resMode: 1, swarmMode: 2, overlayMode: 5, bosMode: 1, bosGauge: [8, 8, 36, 120], native1to1: true, metrologyMode: 2, showCovEllipses: true },
    bos_handheld_failure: { url: './samples/bos_handheld_failure.mp4', cal: 'none', resMode: 1, swarmMode: 2, overlayMode: 5, bosMode: 1, bosGauge: [8, 8, 36, 120], native1to1: true, metrologyMode: 2, showCovEllipses: true },
    perspective_flyby_decoy: { url: './samples/perspective_flyby_decoy.mp4', cal: 'none', resMode: 1, swarmMode: 2, overlayMode: 0, bosMode: 0, native1to1: true, metrologyMode: 2, showCovEllipses: true },
    depth_parallax: { url: './samples/depth_parallax_dolly.mp4', cal: 'none', resMode: 1, swarmMode: 2, overlayMode: 12, depthMode: 1, depthCalib: { fPx: 500.0, speedMS: 0.30, baselineM: 0.0, maxDispPx: 32.0, stereoSign: 0 } },
    depth_foe_ttc: { url: './samples/depth_foe_approach.mp4', cal: 'none', resMode: 1, swarmMode: 2, overlayMode: 12, depthMode: 1, depthCalib: { fPx: 500.0, speedMS: 0.8522, baselineM: 0.0, maxDispPx: 32.0, stereoSign: 0 } },
    hodge_cylinder: { url: './samples/hodge_cylinder_circulation.mp4', cal: 'none', resMode: 1, swarmMode: 2, overlayMode: 10, hodgeMode: 1 },
    rs_vibro_247hz: { url: './samples/rs_vibrometry_247hz.mp4', cal: 'none', resMode: 1, swarmMode: 0, overlayMode: 11, rsMode: 1, rsCalib: { dtRowUs: 40.0, exposureUs: 200.0, scanDir: 0 }, native1to1: true },
    fovea_720p: { url: './samples/fovea_720p_micro_detail.mp4', cal: 'none', resMode: 1, swarmMode: 1, overlayMode: 0, native1to1: true },
    wind_lifting_cyl: { url: './samples/wind_lifting_cylinder_cp.mp4', cal: 'none', resMode: 1, swarmMode: 2, overlayMode: 13, hodgeMode: 1, windMode: 1 },
    wind_lamb_oseen: { url: './samples/wind_lamb_oseen_viscosity.mp4', cal: 'none', resMode: 1, swarmMode: 2, overlayMode: 13, hodgeMode: 1, windMode: 1 },
    wind_daphnia_piv: { url: './samples/real_daphnia_laser_piv.mp4', cal: 'preset', resMode: 1, swarmMode: 2, overlayMode: 13, hodgeMode: 1, windMode: 1 },
    wind_smoke_wing: { url: './samples/real_smoke_airfoil_cambridge.mp4', cal: 'none', resMode: 1, swarmMode: 2, overlayMode: 13, hodgeMode: 1, windMode: 1 },
    wind_karman_mri: { url: './samples/real_karman_mri_cylinder.mp4', cal: 'none', resMode: 0, swarmMode: 0, overlayMode: 13, hodgeMode: 0, windMode: 3 },
    elasto_crack: { url: './samples/elasto_hidden_crack_panel.mp4', cal: 'none', resMode: 1, swarmMode: 0, overlayMode: 14, rsMode: 2, elastoMode: 1, elastoLockFreqs: [380, 860], rsCalib: { dtRowUs: 35.0, exposureUs: 200.0, scanDir: 0 }, native1to1: true },
    elasto_guitar: { url: './samples/elasto_guitar_soundboard.mp4', cal: 'none', resMode: 1, swarmMode: 0, overlayMode: 14, rsMode: 2, elastoMode: 1, elastoLockFreqs: [247.5, 495], rsCalib: { dtRowUs: 35.0, exposureUs: 200.0, scanDir: 0 }, native1to1: true },
    elasto_gelatin: { url: './samples/elasto_gelatin_phantom.mp4', cal: 'none', resMode: 1, swarmMode: 0, overlayMode: 14, rsMode: 2, elastoMode: 1, elastoLockFreqs: [110, 220], rsCalib: { dtRowUs: 35.0, exposureUs: 250.0, scanDir: 0 }, native1to1: true },
    acoustic_bos: { url: './samples/acoustic_bos_standing_wave.mp4', cal: 'none', resMode: 1, swarmMode: 0, overlayMode: 15, rsMode: 2, elastoMode: 4, elastoLockFreqs: [1200], rsCalib: { dtRowUs: 35.0, exposureUs: 180.0, scanDir: 0 }, native1to1: true },
    elasto_fork_real: { url: './samples/real_tuning_fork_ripples.mp4', cal: 'none', resMode: 1, swarmMode: 0, overlayMode: 14, elastoMode: 3 },
    elasto_glass_real: { url: './samples/real_glass_faraday_waves.mp4', cal: 'none', resMode: 1, swarmMode: 0, overlayMode: 14, elastoMode: 3 },
    elasto_chladni: { url: './samples/real_chladni_plate_audio.mp4', cal: 'none', resMode: 1, swarmMode: 0, overlayMode: 14, elastoMode: 3 },
    elasto_oce_real: { url: './samples/real_shear_wave_oce.mp4', cal: 'none', resMode: 1, swarmMode: 0, overlayMode: 14, elastoMode: 3 },
    vortex_1200: { benchScenario: 'vortex_swarm_1200', resMode: 1, swarmMode: 2, overlayMode: 2, bosMode: 0, metrologyMode: 0, showCovEllipses: false },
    juggling: { url: './samples/juggling_ballistic.mp4', cal: 'auto_g' },
    gantry: { url: './samples/roundabout_occlusion.mp4', cal: 'none' },
    galileo: { url: './samples/galileo_pendulum_peg.mp4', cal: 'preset' },
    metronome: { url: './samples/harmonic_metronome_phase.mp4', cal: 'none' },
    billiards: { url: './samples/carom_billiards_impulses.mp4', cal: 'preset' },
  };

  let pendingShowcaseCal = '';

  function syncSelectsFromOrgans(org, ov, native1to1) {
    if (dom.swarmModeSelect) dom.swarmModeSelect.value = String(org.swarm ?? 0);
    if (dom.bosModeSelect) dom.bosModeSelect.value = String(org.bos ?? 0);
    if (dom.metrologyModeSelect) dom.metrologyModeSelect.value = String(org.metrology ?? 0);
    if (dom.hodgeModeSelect) dom.hodgeModeSelect.value = String(org.hodge ?? 0);
    if (dom.rsVibroModeSelect) dom.rsVibroModeSelect.value = String(org.rs ?? 0);
    if (dom.depthModeSelect) dom.depthModeSelect.value = String(org.depth ?? 0);
    if (dom.windModeSelect) dom.windModeSelect.value = String(org.wind ?? 0);
    if (dom.elastoModeSelect) dom.elastoModeSelect.value = String(org.elasto ?? 0);
    if (dom.fieldOverlayModeSelect && ov && typeof ov.field === 'number') {
      dom.fieldOverlayModeSelect.value = String(ov.field);
    }
    if (dom.chkNativeRoi && typeof native1to1 === 'boolean') dom.chkNativeRoi.checked = native1to1;
    if (org.rsCal) {
      if (dom.inpRsDtRowUs && typeof org.rsCal.rowUs === 'number') dom.inpRsDtRowUs.value = String(org.rsCal.rowUs);
      if (dom.inpRsExposureUs && typeof org.rsCal.exposureUs === 'number') dom.inpRsExposureUs.value = String(org.rsCal.exposureUs);
      if (dom.selRsScanDir && typeof org.rsCal.scan === 'number') dom.selRsScanDir.value = String(org.rsCal.scan);
    }
  }

  function postAtomicConfigToWorker(nextState, extra = {}) {
    const org = nextState.organ_config;
    const ov = nextState.overlays;
    const res = nextState.resolution_roi;
    const scanDir = org.rsCal && typeof org.rsCal.scan === 'number' && org.rsCal.scan >= 0 && org.rsCal.scan <= 3
      ? org.rsCal.scan
      : undefined;
    const ovSyncSeq = nextOverlaySyncSeq();
    worker.postMessage({
      type: 'apply_instrument_config',
      txId: nextState.revision,
      nextRevision: nextState.revision,
      sourceEpoch: nextState.source.epoch,
      workerEpoch: nextState.runtime_health.workerEpoch,
      completeConfig: {
        resMode: res.resMode,
        labMode: nextState.lab_mode,
        resetBosRef: Boolean(extra.resetBosRef === true),
        overlaySyncSeq: ovSyncSeq,
        organs: {
          ...org,
          rsCal: {
            ...org.rsCal,
            rateHz: rsAudioCaptureRateHz || org.rsCal.rateHz || 48000.0,
            maxSkewUs: org.rsCal.maxSkewUs || 500.0,
          },
        },
        overlays: ov,
      },
    });
    const roiMsg = { type: 'set_roi', native1to1: Boolean(res.native1to1) };
    if (scanDir !== undefined) roiMsg.scanDir = scanDir;
    worker.postMessage(roiMsg);
    syncRsLiveAudioCapture();
  }

  function syncLabAndLookaheadControls(st) {
    if (st.lab_mode !== currentLabMode) {
      currentLabMode = st.lab_mode;
      const copy = LAB_COPY[currentLabMode] || LAB_COPY[0];
      if (dom.labBadge) dom.labBadge.textContent = copy.badge;
      if (dom.labDescription) dom.labDescription.innerHTML = copy.html;
      document.querySelectorAll('.lab-btn').forEach((b) => {
        const on = Number(b.dataset.lab) === currentLabMode;
        b.classList.toggle('active', on);
        b.setAttribute('aria-pressed', String(on));
      });
      syncStimulusButtonAvailability();
    }
    if (sliderK && Number(sliderK.value) !== st.organ_config.lookahead) {
      sliderK.value = String(st.organ_config.lookahead);
      const effRate = (activeSourceMode === 'video' || activeSourceMode === 'live') ? activeVideoFps : 60.0;
      const vk = document.getElementById('valK');
      if (vk) vk.textContent = `${st.organ_config.lookahead} frames (${Math.round((st.organ_config.lookahead * 1000) / effRate)} ms)`;
    }
    const chkStr = document.getElementById('chkStrobe');
    if (chkStr) chkStr.checked = Boolean(st.overlays.strobe);
  }

  function commitAtomicPlan(plan, opts = {}) {
    const prevBos = instrumentState.organ_config.bos;
    const st = dispatchInstrument({
      type: 'APPLY_CONFIG_PATCH',
      organs: plan.organs,
      overlays: plan.overlays,
      lab_mode: plan.lab_mode,
      resolution_roi: opts.resolution_roi,
      recordUndo: Boolean(opts.recordUndo),
      undoLabel: opts.undoLabel || 'Previous setup',
      trackPending: true,
      markPresetModified: opts.markPresetModified,
      originPreset: opts.originPreset,
    }, true);
    syncSelectsFromOrgans(st.organ_config, st.overlays, st.resolution_roi.native1to1);
    dispatchDeckEvent({
      type: 'ENGINE_SYNC',
      swarmMode: st.organ_config.swarm,
      swarmActive: st.organ_config.swarm === 2 ? true : (st.organ_config.swarm === 0 ? false : lastSwarmActive),
    });
    dispatchDeckEvent({
      type: 'OVERLAY_SET',
      showHeroTargets: st.overlays.hero,
      showCovEllipses: st.overlays.covEllipses,
      fieldOverlayMode: st.overlays.field,
    });
    syncLabAndLookaheadControls(st);
    const shouldResetBosRef = opts.resetBosRef !== undefined
      ? Boolean(opts.resetBosRef)
      : (prevBos === 0 && st.organ_config.bos > 0);
    postAtomicConfigToWorker(st, { resetBosRef: shouldResetBosRef });
    return st;
  }

  function applyWave1PresetToUiAndWorker(cfg) {
    const nextElastoMode = cfg && typeof cfg.elastoMode === 'number' ? cfg.elastoMode : 0;
    const nextBosMode = nextElastoMode > 0 ? 0 : (cfg && typeof cfg.bosMode === 'number' ? cfg.bosMode : 0);
    const nextMetMode = cfg && typeof cfg.metrologyMode === 'number' ? cfg.metrologyMode : 0;
    const nextShowCov = Boolean(cfg && cfg.showCovEllipses);
    const nextNativeRoi = Boolean(cfg && cfg.native1to1);
    const nextWindMode = cfg && typeof cfg.windMode === 'number' ? cfg.windMode : 0;
    const nextHodgeMode = cfg && typeof cfg.hodgeMode === 'number' ? cfg.hodgeMode : ((nextWindMode === 1 || nextWindMode === 2) ? 1 : 0);
    const nextRsMode = cfg && typeof cfg.rsMode === 'number'
      ? cfg.rsMode
      : (nextElastoMode === 1 || nextElastoMode === 4 ? 2 : (nextElastoMode === 2 ? 1 : 0));
    const nextDepthMode = cfg && typeof cfg.depthMode === 'number' ? cfg.depthMode : 0;
    const nextSwarmMode = cfg && typeof cfg.swarmMode === 'number' ? cfg.swarmMode : instrumentState.organ_config.swarm;
    const nextRsCalib = cfg && cfg.rsCalib ? {
      rowUs: cfg.rsCalib.dtRowUs ?? 40.0,
      exposureUs: cfg.rsCalib.exposureUs ?? 200.0,
      scan: cfg.rsCalib.scanDir ?? 0,
      rateHz: rsAudioCaptureRateHz || 48000.0,
      maxSkewUs: 500.0,
    } : instrumentState.organ_config.rsCal;
    const nextDepthCalib = cfg && cfg.depthCalib ? {
      fPx: cfg.depthCalib.fPx ?? 0.0,
      speedMS: cfg.depthCalib.speedMS ?? 0.0,
      baselineM: cfg.depthCalib.baselineM ?? 0.0,
      maxDispPx: cfg.depthCalib.maxDispPx ?? 32.0,
      stereoSign: cfg.depthCalib.stereoSign ?? 0,
    } : { fPx: 0.0, speedMS: 0.0, baselineM: 0.0, maxDispPx: 32.0, stereoSign: 0 };
    const nextElastoLockFreqs = cfg && Array.isArray(cfg.elastoLockFreqs) ? cfg.elastoLockFreqs : [];

    let nextField = typeof (cfg && cfg.overlayMode) === 'number'
      ? cfg.overlayMode
      : (dom.fieldOverlayModeSelect ? (parseInt(dom.fieldOverlayModeSelect.value || '0', 10) || 0) : 0);
    if (typeof (cfg && cfg.overlayMode) !== 'number') {
      const orphanedBos = nextField >= 5 && nextField <= 7 && nextBosMode === 0;
      const orphanedHodge = nextField >= 8 && nextField <= 10 && nextHodgeMode === 0;
      const orphanedRs = nextField === 11 && nextRsMode === 0;
      const orphanedDepth = nextField === 12 && nextDepthMode === 0;
      const orphanedWind = nextField === 13 && nextWindMode === 0;
      const orphanedElasto = (nextField === 14 || nextField === 15) && nextElastoMode === 0;
      if (orphanedBos || orphanedHodge || orphanedRs || orphanedDepth || orphanedWind || orphanedElasto) {
        nextField = 0;
      }
    }

    const plan = {
      organs: {
        ...instrumentState.organ_config,
        swarm: nextSwarmMode,
        bos: nextBosMode,
        hodge: nextHodgeMode,
        metrology: nextMetMode,
        rs: nextRsMode,
        depth: nextDepthMode,
        wind: nextWindMode,
        elasto: nextElastoMode,
        bosGauge: cfg && Array.isArray(cfg.bosGauge) ? cfg.bosGauge : [0, 0, 0, 0],
        rsCal: nextRsCalib,
        depthCal: nextDepthCalib,
        fluidDensity: cfg && typeof cfg.fluidDensity === 'number' ? cfg.fluidDensity : 0.0,
        lockHz: nextElastoLockFreqs,
      },
      overlays: {
        ...instrumentState.overlays,
        covEllipses: nextShowCov,
        field: nextField,
        hero: cfg && typeof cfg.hero === 'boolean' ? cfg.hero : instrumentState.overlays.hero,
      },
      lab_mode: cfg && typeof cfg.labMode === 'number' ? cfg.labMode : instrumentState.lab_mode,
    };
    commitAtomicPlan(plan, {
      resolution_roi: { native1to1: nextNativeRoi },
      resetBosRef: nextBosMode > 0,
      markPresetModified: false,
    });
  }

  let pendingShowcaseEntryActive = false;
  let pendingRecipeResMode = null;

  sampleVideoSelect.addEventListener('change', async () => {
    const url = sampleVideoSelect.value;
    const selOpt = sampleVideoSelect.selectedOptions[0];
    const reqSwarm = selOpt && selOpt.dataset.swarm ? parseInt(selOpt.dataset.swarm, 10) : NaN;
    const hadExplicitRecipeRes = pendingRecipeResMode !== null;
    const explicitRecipeRes = pendingRecipeResMode;
    pendingRecipeResMode = null;
    if (url && Number.isFinite(reqSwarm) && !pendingShowcaseEntryActive) {
      syncSwarmControlsForConfig(1, reqSwarm, undefined, false);
    }
    if (!pendingShowcaseEntryActive && url) {
      initialPresetBooted = true;
      const matchedEntry = Object.values(showcaseClipMap).find((c) => c && c.url === url) || {};
      if (typeof matchedEntry.overlayMode === 'number') {
        syncSwarmControlsForConfig(matchedEntry.resMode ?? 1, matchedEntry.swarmMode ?? reqSwarm, matchedEntry.overlayMode, false);
      }
      applyWave1PresetToUiAndWorker(matchedEntry);
    }
    pendingShowcaseEntryActive = false;
    const queuedCal = pendingShowcaseCal;
    pendingShowcaseCal = '';
    if (!url) return;
    const opt = sampleVideoSelect.selectedOptions[0];
    const fpsHint = parseFloat(opt.dataset.fps || '30');
    const targetLab = parseInt(opt.dataset.lab || '0', 10);

    stopMediaCapture();
    const myEpoch = sourceEpoch;
    setSourceButtons(null);
    const matchingBtn = Array.from(document.querySelectorAll('.showcase-btn')).find((b) => {
      const cfg = showcaseClipMap[b.dataset.showcase];
      return cfg && cfg.url === url;
    });
    setActiveShowcaseChip(matchingBtn ? matchingBtn.dataset.showcase : '');
    if (hadExplicitRecipeRes) {
      if (activeResMode !== explicitRecipeRes) {
        document.getElementById(explicitRecipeRes === 1 ? 'btnResHd' : 'btnResSd')?.click();
      }
    } else if (activeResMode === 0) {
      document.getElementById('btnResHd')?.click();
    }
    activeVideoFps = fpsHint;
    activeVideoFile = null;
    activeVideoUrl = url;
    videoEl.src = url;
    videoEl.loop = true;
    const curEmSample = dom.elastoModeSelect ? (parseInt(dom.elastoModeSelect.value || '0', 10) || 0) : 0;
    const curRmSample = dom.rsVibroModeSelect ? (parseInt(dom.rsVibroModeSelect.value || '0', 10) || 0) : 0;
    const audioDecodePromise =
      curEmSample === 1 || curEmSample === 4 || curRmSample === 2
        ? decodeMp4AudioTrack(url, myEpoch)
        : null;
    const playOk = await safePlayVideo();
    if (!playOk) {
      if (myEpoch === sourceEpoch && url.includes('starling_murmuration_1000')) {
        showcaseInspectSubNote = 'Autoplay blocked by power mode · live 1,208-target flock running';
        syncSwarmControlsForConfig(1, 2, 1, false);
        btnTestbench.click();
        dom.benchScenarioSelect.value = 'murmuration_1200';
        worker.postMessage({ type: 'set_scenario', scenario: 'murmuration_1200' });
        setShowcaseWipePercent(28);
        setHeroTargetsVisible(false);
      }
      return;
    }
    if (audioDecodePromise) {
      await audioDecodePromise;
    }
    if (myEpoch !== sourceEpoch) return;

    const targetBtn = document.querySelector(`.lab-btn[data-lab="${targetLab}"]`);
    if (targetBtn && currentLabMode !== targetLab) {
      targetBtn.click();
    }
    const matchedPresetKey = matchingBtn ? matchingBtn.dataset.showcase : (RECIPE_BY_URL[url]?.key || null);
    const sampleLabel = `SOURCE: SAMPLE MP4 (${opt.textContent.split(' (')[0].toUpperCase()})`;
    syncContextualSourceUi('video', { kind: 'SAMPLE', label: sampleLabel, originPreset: matchedPresetKey });
    dom.hudSourceLabel.textContent = sampleLabel;
    dom.hudSourceLabel.title = sampleLabel;
    const sourceMsg = { type: 'set_source', sourceMode: 'video', fpsHint };
    if (Number.isFinite(reqSwarm)) {
      sourceMsg.swarmOverride = reqSwarm;
    }
    worker.postMessage(sourceMsg);
    if (queuedCal === 'preset') {
      document.getElementById('btnCalPreset')?.click();
    } else if (queuedCal === 'auto_g') {
      document.getElementById('btnCalAutoG')?.click();
    }
    startVideoPump();
  });

  // 1-Click Featured Showcases (Auto-Switch HD + Mode + Stream/Clip + Deterministic Post-Source Metric Preset)
  function setActiveShowcaseChip(id) {
    document.querySelectorAll('.showcase-btn').forEach((btn) => {
      const on = btn.dataset.showcase === id;
      btn.classList.toggle('active', on);
      btn.setAttribute('aria-pressed', String(on));
    });
  }

  function activateShowcaseEntry(key) {
    pendingShowcaseCal = '';
    if (key === 'live_sound_probe') {
      activateLiveSoundProbeWebcam();
      return;
    }
    stopProbeExciter(false);
    setActiveShowcaseChip(key);
    if (activeResMode === 0) {
      document.getElementById('btnResHd')?.click();
    }
    if (key === 'live_i80') {
      if (isStaticHostMode) {
        openLiveStreamModal();
        return;
      }
      dom.tier3CustomHlsUrl.value = '';
      const i80Opt = dom.tier3PresetSelect.querySelector('option[value="caltrans_d4_i80_ashby"]');
      const i80Offline = i80Opt && i80Opt.textContent.includes('[OFFLINE]');
      let chosenId = 'caltrans_d4_i80_ashby';
      let chosenLabel = 'Caltrans D4: I-80 @ Ashby Ave (Berkeley Overpass · 1080p)';
      if (i80Offline) {
        const altOpt = Array.from(dom.tier3PresetSelect.options).find((o) => o.textContent.includes('[LIVE ONLINE]'));
        if (altOpt) {
          chosenId = altOpt.value;
          chosenLabel = altOpt.textContent.replace(/^\[[^\]]+\]\s*/, '');
        }
      }
      if (dom.tier3PresetSelect.querySelector(`option[value="${chosenId}"]`)) {
        dom.tier3PresetSelect.value = chosenId;
      }
      startTier3MseStream({
        presetId: chosenId,
        customUrl: '',
        label: chosenLabel,
        fpsHint: 30,
        targetLab: 0,
        showcaseId: 'live_i80',
      });
      return;
    }
    const cfg = showcaseClipMap[key];
    if (!cfg) return;
    applyWave1PresetToUiAndWorker(cfg);
    if (cfg.benchScenario) {
      stopMediaCapture();
      sampleVideoSelect.value = '';
      dom.benchScenarioSelect.value = cfg.benchScenario;
      btnTestbench.click();
      dom.benchScenarioSelect.value = cfg.benchScenario;
      syncSwarmControlsForConfig(cfg.resMode, cfg.swarmMode, cfg.overlayMode, false);
      worker.postMessage({ type: 'set_scenario', scenario: cfg.benchScenario });
      setActiveShowcaseChip(key);
      return;
    }
    if (typeof cfg.resMode === 'number' || typeof cfg.swarmMode === 'number' || typeof cfg.overlayMode === 'number') {
      syncSwarmControlsForConfig(cfg.resMode, cfg.swarmMode, cfg.overlayMode, false);
    }
    pendingShowcaseCal = cfg.cal;
    pendingShowcaseEntryActive = true;
    sampleVideoSelect.value = cfg.url;
    sampleVideoSelect.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function loadRecipe(key, opts = {}) {
    const rec = RECIPE_BY_KEY[key];
    if (!rec) return false;
    const settingsOnly = Boolean(opts.settingsOnly);
    initialPresetBooted = true;
    if (!workerKernelReady && !settingsOnly) {
      pendingHeroSceneBeforeReady = key;
      dispatchInstrument({ type: 'DISMISS_WELCOME' }, true);
      return true;
    }

    // 1. Resolution target (Settings Only preserves the user's current resolution!)
    if (!settingsOnly && typeof rec.resMode === 'number' && rec.resMode !== activeResMode) {
      document.getElementById(rec.resMode === 1 ? 'btnResHd' : 'btnResSd')?.click();
    }
    if (settingsOnly && !workerKernelReady) {
      if (dom.showcaseToolbarStatus) dom.showcaseToolbarStatus.textContent = 'Waiting for Wasm kernel before applying settings';
      return false;
    }

    // 2. Build complete clean organ + overlay config from recipe over clean defaults
    const cleanOrg = {
      ...instrumentState.organ_config,
      swarm: rec.organs.swarm ?? 0,
      bos: rec.organs.bos ?? 0,
      hodge: rec.organs.hodge ?? 0,
      metrology: rec.organs.metrology ?? 0,
      rs: rec.organs.rs ?? 0,
      depth: rec.organs.depth ?? 0,
      wind: rec.organs.wind ?? 0,
      elasto: rec.organs.elasto ?? 0,
      lookahead: rec.organs.lookahead ?? 8,
      bosGauge: rec.organs.bosGauge ? rec.organs.bosGauge.slice() : [0, 0, 0, 0],
      rsCal: rec.organs.rsCal ? { ...instrumentState.organ_config.rsCal, ...rec.organs.rsCal } : instrumentState.organ_config.rsCal,
      depthCal: rec.organs.depthCal ? { ...rec.organs.depthCal } : { fPx: 0.0, speedMS: 0.0, baselineM: 0.0, maxDispPx: 32.0, stereoSign: 0 },
      fluidDensity: rec.organs.fluidDensity ?? 0.0,
      lockHz: rec.organs.lockHz ? rec.organs.lockHz.slice() : [],
    };
    const nextOv = {
      ...instrumentState.overlays,
      field: rec.overlays.field ?? 0,
      hero: rec.overlays.hero !== undefined ? Boolean(rec.overlays.hero) : true,
      covEllipses: Boolean(rec.overlays.covEllipses),
      strobe: Boolean(rec.overlays.strobe),
    };

    commitAtomicPlan(
      { organs: cleanOrg, overlays: nextOv, lab_mode: rec.labMode ?? 0 },
      {
        resolution_roi: { native1to1: Boolean(rec.native1to1) },
        recordUndo: true,
        undoLabel: settingsOnly ? `Before ${rec.key} settings` : `Before ${rec.key}`,
        markPresetModified: false,
        originPreset: settingsOnly ? instrumentState.source.originPreset : rec.key,
      }
    );

    if (rec.bayTab && benchShell) {
      benchShell.setBayTab(rec.bayTab);
    }

    if (typeof rec.overlays.wipePct === 'number') {
      setShowcaseWipePercent(rec.overlays.wipePct, true);
    } else if (!settingsOnly) {
      setShowcaseWipePercent(0, true);
    }

    if (settingsOnly) {
      if (rec.cal === 'auto_g') document.getElementById('btnCalAutoG')?.click();
      else if (rec.cal === 'preset' && activeVideoUrl) document.getElementById('btnCalPreset')?.click();
      // If applied settings need 48 kHz muxed audio on an already-loaded local MP4/sample, decode it now!
      const needsAudioTrack = cleanOrg.rs === 2 || cleanOrg.elasto === 1 || cleanOrg.elasto === 4;
      if (needsAudioTrack && activeSourceMode === 'video' && (activeVideoFile || activeVideoUrl)) {
        decodeMp4AudioTrack(activeVideoFile || activeVideoUrl, sourceEpoch).catch(() => {});
      }
      if (dom.showcaseToolbarStatus) {
        dom.showcaseToolbarStatus.textContent = `Applied settings from ${rec.title} onto active source`;
      }
      return true;
    }

    dispatchInstrument({ type: 'DISMISS_WELCOME' }, true);
    if (rec.source.kind === 'webcam') {
      activateLiveSoundProbeWebcam();
      return true;
    }
    if (rec.source.kind === 'live_mse') {
      activateShowcaseEntry('live_i80');
      return true;
    }
    if (rec.source.kind === 'bench') {
      stopProbeExciter(false);
      stopMediaCapture();
      sampleVideoSelect.value = '';
      dom.benchScenarioSelect.value = rec.source.scenario;
      setSourceButtons(btnTestbench);
      setActiveShowcaseChip(rec.key);
      syncContextualSourceUi('testbench', { kind: 'TESTBENCH', originPreset: rec.key });
      dom.hudSourceLabel.textContent = `SOURCE: SIGNAL GEN (${rec.source.scenario.toUpperCase()})`;
      worker.postMessage({ type: 'set_source', sourceMode: 'testbench' });
      worker.postMessage({ type: 'set_scenario', scenario: rec.source.scenario });
      return true;
    }
    if (rec.source.kind === 'sample') {
      stopProbeExciter(false);
      setActiveShowcaseChip(rec.key);
      pendingShowcaseCal = rec.cal || 'none';
      pendingShowcaseEntryActive = true;
      pendingRecipeResMode = typeof rec.resMode === 'number' ? rec.resMode : null;
      sampleVideoSelect.value = rec.source.url;
      sampleVideoSelect.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    }
    return false;
  }

  function activateHeroScene(sceneKey) {
    loadRecipe(sceneKey, { settingsOnly: false });
  }

  document.title = 'CHRONOS-SPECTRAL // Optical Physics Instrument';
  setShowcaseWipePercent(0, false);
  if (dom.showcaseWipeSlider) {
    dom.showcaseWipeSlider.addEventListener('input', () => {
      setShowcaseWipePercent(dom.showcaseWipeSlider.value, true);
    });
  }
  if (dom.hudWipeHandle) {
    if (dom.hudWipeTrack) dom.hudWipeTrack.hidden = false;
    let wipeGrabOffsetPx = 0;
    let wipeStartClientX = 0;
    let wipeDragged = false;

    const announceWipePosition = () => {
      if (!dom.showcaseToolbarStatus || !dom.showcaseWipeSlider) return;
      const pct = Math.max(0, Math.min(100, Math.round(Number(dom.showcaseWipeSlider.value) || 0)));
      dom.showcaseToolbarStatus.textContent =
        pct === 0
          ? 'Split-wipe divider docked left (full spectral view)'
          : pct === 100
          ? 'Split-wipe divider docked right (full raw video)'
          : `Split-wipe divider moved to ${pct}% raw video, ${100 - pct}% spectral wavefield`;
    };

    dom.hudWipeHandle.addEventListener('pointerdown', (ev) => {
      if (ev.button !== 0) return;
      const rect = hudEl.getBoundingClientRect();
      if (rect.width <= 0) return;
      const curPct = dom.showcaseWipeSlider ? (Number(dom.showcaseWipeSlider.value) || 0) : 28;
      wipeGrabOffsetPx = ev.clientX - (rect.left + (curPct / 100) * rect.width);
      wipeStartClientX = ev.clientX;
      wipeDragged = false;
      dom.hudWipeHandle.setPointerCapture(ev.pointerId);
    });

    dom.hudWipeHandle.addEventListener('pointermove', (ev) => {
      if (!dom.hudWipeHandle.hasPointerCapture(ev.pointerId)) return;
      if (!wipeDragged && Math.abs(ev.clientX - wipeStartClientX) < 4) return;
      wipeDragged = true;
      dom.hudWipeHandle.classList.add('dragging');
      const rect = hudEl.getBoundingClientRect();
      if (rect.width <= 0) return;
      const nextPct = Math.max(
        0,
        Math.min(100, Math.round(((ev.clientX - wipeGrabOffsetPx - rect.left) / rect.width) * 100))
      );
      setShowcaseWipePercent(nextPct, true);
    });

    dom.hudWipeHandle.addEventListener('lostpointercapture', () => {
      const didDrag = wipeDragged;
      wipeDragged = false;
      dom.hudWipeHandle.classList.remove('dragging');
      if (didDrag) announceWipePosition();
    });

    dom.hudWipeHandle.addEventListener('keydown', (ev) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(ev.key)) return;
      ev.preventDefault();
      const curPct = dom.showcaseWipeSlider ? (Number(dom.showcaseWipeSlider.value) || 0) : 28;
      const step = ev.shiftKey ? 10 : 2;
      const nextPct =
        ev.key === 'Home'
          ? 0
          : ev.key === 'End'
          ? 100
          : ev.key === 'ArrowLeft'
          ? Math.max(0, curPct - step)
          : Math.min(100, curPct + step);
      setShowcaseWipePercent(nextPct, true);
      announceWipePosition();
    });
  }
  if (dom.btnShowcasePlayPause) {
    dom.btnShowcasePlayPause.addEventListener('click', () => {
      if (dom.btnShowcasePlayPause.getAttribute('aria-disabled') === 'true') return;
      dom.btnVidPlayPause.click();
    });
  }
  if (dom.btnShowcaseStepFwd) {
    dom.btnShowcaseStepFwd.addEventListener('click', () => {
      if (dom.btnShowcaseStepFwd.getAttribute('aria-disabled') === 'true') return;
      stepVideoByFrames(1);
    });
  }
  document.querySelectorAll('.hero-scene-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      activateHeroScene(btn.dataset.heroScene);
    });
  });
  const btnHeroDropMp4 = document.getElementById('btnHeroDropMp4');
  if (btnHeroDropMp4) {
    btnHeroDropMp4.addEventListener('click', () => {
      mp4Input.click();
    });
  }
  const btnHeroWebcam = document.getElementById('btnHeroWebcam');
  if (btnHeroWebcam) {
    btnHeroWebcam.addEventListener('click', () => {
      stopProbeExciter(false);
      setHeroScenePressed('');
      syncSwarmControlsForConfig(1, 1, 0, true);
      setHeroTargetsVisible(true);
      setShowcaseWipePercent(20);
      showcaseInspectSubNote = 'Live webcam pixels streaming into local Wasm';
      if (dom.showcaseToolbarStatus) {
        dom.showcaseToolbarStatus.textContent = 'Requesting live webcam feed';
      }
      btnWebcam.click();
    });
  }
  const btnHeroSoundProbe = document.getElementById('btnHeroSoundProbe');
  if (btnHeroSoundProbe) {
    btnHeroSoundProbe.addEventListener('click', () => {
      setHeroScenePressed('');
      setHeroTargetsVisible(false);
      setShowcaseWipePercent(28);
      showcaseInspectSubNote = 'Live webcam + 110/220 Hz lock-in sound probe (RS coherence + elastography)';
      if (dom.showcaseToolbarStatus) {
        dom.showcaseToolbarStatus.textContent = 'Starting live webcam sound probe (110+220 Hz)';
      }
      activateLiveSoundProbeWebcam();
    });
  }
  if (dom.btnOpenFullDeck) {
    dom.btnOpenFullDeck.addEventListener('click', () => setSurfaceMode('deck'));
  }
  if (dom.btnReturnShowcase) {
    dom.btnReturnShowcase.addEventListener('click', () => setSurfaceMode('showcase'));
  }

  // Drag-and-drop MP4/WebM directly onto Showcase canvas or dock (zero bytes uploaded)
  const dropTargets = [dom.hudPanel, document.getElementById('showcaseActionDock')].filter(Boolean);
  let dropDepth = 0;
  dropTargets.forEach((el) => {
    el.addEventListener('dragenter', (ev) => {
      ev.preventDefault();
      dropDepth++;
      if (dom.hudPanel) dom.hudPanel.dataset.dropActive = 'true';
    });
    el.addEventListener('dragover', (ev) => {
      ev.preventDefault();
      if (ev.dataTransfer) ev.dataTransfer.dropEffect = 'copy';
    });
    el.addEventListener('dragleave', () => {
      dropDepth = Math.max(0, dropDepth - 1);
      if (dropDepth === 0 && dom.hudPanel) delete dom.hudPanel.dataset.dropActive;
    });
    el.addEventListener('drop', async (ev) => {
      ev.preventDefault();
      dropDepth = 0;
      if (dom.hudPanel) delete dom.hudPanel.dataset.dropActive;
      const file = ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files[0];
      if (file) {
        await loadLocalVideoFile(file);
      }
    });
  });

  // Showcase Toolbar roving-tabindex fallback when native focusgroup is unavailable (modern-web-guidance: toolbar)
  if (dom.showcaseToolbar && !('focusGroup' in HTMLElement.prototype)) {
    const bar = dom.showcaseToolbar;
    const barItems = Array.from(bar.querySelectorAll('button'));
    barItems.forEach((b, i) => {
      b.tabIndex = i ? -1 : 0;
    });
    bar.addEventListener('focusin', ({ target }) => {
      if (barItems.includes(target)) {
        barItems.forEach((b) => {
          b.tabIndex = b === target ? 0 : -1;
        });
      }
    });
    bar.addEventListener('keydown', (ev) => {
      const idx = barItems.indexOf(document.activeElement);
      if (idx < 0) return;
      const rtl = getComputedStyle(bar).direction === 'rtl';
      const steps = { ArrowRight: rtl ? -1 : 1, ArrowLeft: rtl ? 1 : -1 };
      let nextIdx = -1;
      if (Object.hasOwn(steps, ev.key)) {
        nextIdx = (idx + steps[ev.key] + barItems.length) % barItems.length;
      } else if (ev.key === 'Home') {
        nextIdx = 0;
      } else if (ev.key === 'End') {
        nextIdx = barItems.length - 1;
      }
      if (nextIdx >= 0) {
        ev.preventDefault();
        barItems[nextIdx].focus();
      }
    });
  }

  // Preset buttons are wired inside `createBenchShell` via declarative `loadRecipe`.

  // MP4 Frame-Accurate Transport Controls
  dom.btnVidPlayPause.addEventListener('click', async () => {
    if (activeSourceMode !== 'video') return;
    if (videoEl.paused || videoEl.ended) {
      await safePlayVideo();
    } else {
      videoEl.pause();
    }
    updateVideoTimeLabel();
  });

  async function stepVideoByFrames(deltaFrames) {
    if (activeSourceMode !== 'video' || videoEl.readyState < 2) return;
    videoEl.pause();
    const frameStepSec = 1.0 / Math.max(8.0, activeVideoFps);
    const dur = Number.isFinite(videoEl.duration) ? videoEl.duration : 30.0;
    const targetTime = Math.max(0.0, Math.min(dur - 0.002, videoEl.currentTime + deltaFrames * frameStepSec));
    if (deltaFrames < 0) {
      worker.postMessage({ type: 'lab_action', action: 'reset' });
    }
    await new Promise((resolve) => {
      const onSeeked = () => {
        videoEl.removeEventListener('seeked', onSeeked);
        resolve();
      };
      videoEl.addEventListener('seeked', onSeeked, { once: true });
      videoEl.currentTime = targetTime;
    });
    lastMediaTime = videoEl.currentTime;
    updateVideoTimeLabel();
    await captureAndSendCurrentVideoFrame(frameStepSec, videoEl.currentTime, sourceEpoch);
  }

  document.getElementById('btnVidStepBack').addEventListener('click', () => stepVideoByFrames(-1));
  document.getElementById('btnVidStepFwd').addEventListener('click', () => stepVideoByFrames(1));
  document.getElementById('btnVidReplay').addEventListener('click', async () => {
    if (activeSourceMode !== 'video') return;
    worker.postMessage({ type: 'lab_action', action: 'reset' });
    lastMediaTime = -1;
    videoEl.currentTime = 0;
    await safePlayVideo();
    updateVideoTimeLabel();
  });

  async function seekVideoToTime(targetSec) {
    if (activeSourceMode !== 'video' || !Number.isFinite(videoEl.duration) || videoEl.duration <= 0) return;
    const clamped = Math.max(0.0, Math.min(Math.max(0.0, videoEl.duration - 0.01), targetSec));
    const wasPaused = videoEl.paused || videoEl.ended;
    worker.postMessage({ type: 'lab_action', action: 'reset' });
    await new Promise((resolve) => {
      videoEl.addEventListener('seeked', () => resolve(), { once: true });
      videoEl.currentTime = clamped;
    });
    lastMediaTime = videoEl.currentTime;
    updateVideoTimeLabel();
    if (wasPaused) {
      const frameStepSec = 1.0 / Math.max(8.0, activeVideoFps);
      await captureAndSendCurrentVideoFrame(frameStepSec, videoEl.currentTime, sourceEpoch);
    }
  }

  if (dom.videoScrubSlider) {
    dom.videoScrubSlider.addEventListener('pointerdown', () => { videoScrubDragging = true; });
    dom.videoScrubSlider.addEventListener('input', () => {
      if (activeSourceMode !== 'video' || !Number.isFinite(videoEl.duration) || videoEl.duration <= 0) return;
      const frac = Math.max(0, Math.min(1, Number(dom.videoScrubSlider.value) / 1000));
      const target = frac * videoEl.duration;
      dom.vidTimeLabel.textContent = `${target.toFixed(2)}s / ${videoEl.duration.toFixed(2)}s`;
    });
    dom.videoScrubSlider.addEventListener('change', async () => {
      videoScrubDragging = false;
      if (activeSourceMode !== 'video' || !Number.isFinite(videoEl.duration) || videoEl.duration <= 0) return;
      const frac = Math.max(0, Math.min(1, Number(dom.videoScrubSlider.value) / 1000));
      await seekVideoToTime(frac * videoEl.duration);
    });
    window.addEventListener('pointerup', () => { videoScrubDragging = false; });
  }

  if (persistenceEl) {
    persistenceEl.addEventListener('click', async (ev) => {
      if (activeSourceMode !== 'video' || !Number.isFinite(videoEl.duration) || videoEl.duration <= 0) return;
      const rect = persistenceEl.getBoundingClientRect();
      if (rect.width <= 0) return;
      const canvasX = ((ev.clientX - rect.left) / rect.width) * 420;
      const frac = Math.max(0, Math.min(1, (canvasX - 10) / 400));
      let targetSec = frac * videoEl.duration;
      if (latestTimelineEvents.length > 0) {
        const denomSeq = Math.max(1, latestTimelineSeq);
        const denomSpan = Math.max(0.05, latestTimelineSpanSec, videoEl.duration);
        let bestDist = 0.035;
        let bestEv = null;
        for (let i = 0; i < latestTimelineEvents.length; i++) {
          const e = latestTimelineEvents[i];
          const ef = e.seq > 0 ? Math.max(0, Math.min(1, e.seq / denomSeq)) : Math.max(0, Math.min(1, e.tSec / denomSpan));
          const d = Math.abs(ef - frac);
          if (d < bestDist) {
            bestDist = d;
            bestEv = e;
          }
        }
        if (bestEv && Number.isFinite(bestEv.tSec)) {
          targetSec = bestEv.tSec;
        }
      }
      await seekVideoToTime(targetSec);
    });
  }

  // Worker-resident zero-copy WebCodecs VideoDecoder Fast-Pass (H.264 / H.265 / VP9 / AV1 + fMP4)
  async function runWebCodecsFastPass() {
    if (fastPassRunning) {
      fastPassCancelRequested = true;
      worker.postMessage({ type: 'worker_fast_pass_cancel' });
      return;
    }
    if (activeSourceMode !== 'video' || (!activeVideoFile && !activeVideoUrl)) return;
    if (typeof VideoDecoder !== 'function' || typeof EncodedVideoChunk !== 'function') {
      dom.vidTimeLabel.textContent = 'WEBCODECS UNAVAILABLE (USE PLAYBACK)';
      return;
    }
    fastPassRunning = true;
    fastPassCancelRequested = false;
    const myEpoch = sourceEpoch;
    if (dom.btnFastPass) {
      dom.btnFastPass.classList.add('active');
      dom.btnFastPass.textContent = '■ Cancel Pass';
    }
    videoEl.pause();
    updateVideoTimeLabel();
    dom.vidTimeLabel.textContent = 'WORKER WEBCODECS INDEXING...';
    const resolvedUrl = activeVideoUrl
      ? new URL(activeVideoUrl, window.location.href).href
      : null;
    worker.postMessage({
      type: 'worker_fast_pass_start',
      file: activeVideoFile || null,
      url: activeVideoFile ? null : resolvedUrl,
      epoch: myEpoch,
      fps: activeVideoFps,
    });
  }

  if (dom.btnFastPass) {
    dom.btnFastPass.addEventListener('click', () => {
      runWebCodecsFastPass();
    });
  }

  // Audio & Reset Buttons
  const btnAudio = document.getElementById('btnAudioToggle');
  btnAudio.addEventListener('click', () => {
    ensureAudioContext();
    audioEnabled = !audioEnabled;
    syncAudioLatencyToWorker();
    btnAudio.classList.toggle('active', audioEnabled);
    btnAudio.setAttribute('aria-pressed', String(audioEnabled));
    btnAudio.textContent = audioEnabled ? '♪ WebAudio Synth: ON' : '♪ Enable WebAudio Synth';
  });

  document.getElementById('btnResetArena').addEventListener('click', () => {
    worker.postMessage({ type: 'lab_action', action: 'reset' });
  });

  // Lab Modes (A, B, C)
  const LAB_COPY = [
    {
      badge: 'MODE A',
      html: '<strong>Mode A · Ballistic &amp; Hermite:</strong> Fits quadratic gravity priors, bridges occlusions via Hermite splines, and plots tangent flight angles &amp; apex reticles.',
      btn: '⚡ Launch Ballistic Arc Impulse',
      videoBtn: '⚡ Ballistic Impulse (Testbench Only)',
      action: 'ballistic_toss',
      simOnly: true,
    },
    {
      badge: 'MODE B',
      html: '<strong>Mode B · Pendulum &amp; Hilbert:</strong> Solves 2×2 algebraic pivots for swing θ, equalizes 15-tap Hilbert phase, and timestamps onset latency.',
      btn: '⚡ Arm & Trigger Saccade Stimulus',
      videoBtn: '⚡ Arm Stimulus Onset Timer',
      action: 'saccade_step',
      simOnly: false,
    },
    {
      badge: 'MODE C',
      html: '<strong>Mode C · Impulses &amp; Scatter:</strong> Measures pre/post-impact deflection Δθ, ratio v⁺/v⁻, and 2-body scatter synchronized with audio at t − K.',
      btn: '⚡ Snap Air-Percussion L1 Strike',
      videoBtn: '⚡ Synthetic Snap (Testbench Only)',
      action: 'air_strike',
      simOnly: true,
    },
  ];

  function syncStimulusButtonAvailability() {
    const info = LAB_COPY[currentLabMode];
    const disabledInVideo = activeSourceMode !== 'testbench' && info.simOnly;
    dom.btnLabStimulus.setAttribute('aria-disabled', String(disabledInVideo));
    dom.btnLabStimulus.textContent = activeSourceMode === 'testbench' ? info.btn : info.videoBtn;
  }

  document.querySelectorAll('.lab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      currentLabMode = Number(btn.dataset.lab);
      document.querySelectorAll('.lab-btn').forEach((b) => {
        const on = b === btn;
        b.classList.toggle('active', on);
        b.setAttribute('aria-pressed', String(on));
      });
      const info = LAB_COPY[currentLabMode];
      dom.labBadge.textContent = info.badge;
      dom.labDescription.innerHTML = info.html;
      syncStimulusButtonAvailability();
      if (currentLabMode === 2 && !audioEnabled) {
        ensureAudioContext();
        audioEnabled = true;
        syncAudioLatencyToWorker();
        btnAudio.classList.add('active');
        btnAudio.setAttribute('aria-pressed', 'true');
        btnAudio.textContent = '♪ WebAudio Synth: ON';
      }
      dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', lab_mode: currentLabMode }, true);
      worker.postMessage({ type: 'params', labMode: currentLabMode });
    });
  });

  dom.btnLabStimulus.addEventListener('click', () => {
    if (dom.btnLabStimulus.getAttribute('aria-disabled') === 'true') return;
    worker.postMessage({ type: 'lab_action', action: LAB_COPY[currentLabMode].action });
  });

  // Phase-Lock Offset Buttons
  document.querySelectorAll('.phase-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const rad = parseFloat(btn.dataset.phase);
      document.querySelectorAll('.phase-btn').forEach((b) => {
        const on = b === btn;
        b.classList.toggle('active', on);
        b.setAttribute('aria-pressed', String(on));
      });
      dom.phaseOffsetLabel.textContent = btn.textContent.toUpperCase();
      worker.postMessage({ type: 'params', phaseOffsetRad: rad });
    });
  });

  // Hyperparameter Sliders & Toggles
  const sliderK = document.getElementById('sliderK');
  const sliderEca = document.getElementById('sliderEca');
  const sliderLambda = document.getElementById('sliderLambda');
  const sliderGain = document.getElementById('sliderGain');

  function syncRangeFallbackProgress(input) {
    if (!CSS.supports('accent-color', '#00e5ff')) {
      const min = parseFloat(input.min) || 0;
      const max = parseFloat(input.max) || 100;
      const val = parseFloat(input.value) || 0;
      const pct = ((val - min) / (max - min)) * 100;
      input.style.setProperty('--progress', `${pct}%`);
    }
  }

  [sliderK, sliderEca, sliderLambda, sliderGain, dom.rangeRoiX0, dom.rangeRoiY0].filter(Boolean).forEach(syncRangeFallbackProgress);

  sliderK.addEventListener('input', () => {
    const k = parseInt(sliderK.value, 10);
    const effRate = (activeSourceMode === 'video' || activeSourceMode === 'live') ? activeVideoFps : 60.0;
    document.getElementById('valK').textContent = `${k} frames (${Math.round((k * 1000) / effRate)} ms)`;
    syncRangeFallbackProgress(sliderK);
    dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { lookahead: k } }, true);
    worker.postMessage({ type: 'params', lookaheadK: k });
  });

  sliderEca.addEventListener('input', () => {
    const v = parseFloat(sliderEca.value);
    document.getElementById('valEca').textContent = v.toFixed(3);
    syncRangeFallbackProgress(sliderEca);
    dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { ecaRate: v } }, false);
    worker.postMessage({ type: 'params', ecaAdaptRate: v });
  });

  sliderLambda.addEventListener('input', () => {
    const v = parseFloat(sliderLambda.value);
    document.getElementById('valLambda').textContent = v.toFixed(2);
    syncRangeFallbackProgress(sliderLambda);
    dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { fistaLambda: v } }, false);
    worker.postMessage({ type: 'params', fistaLambda: v });
  });

  sliderGain.addEventListener('input', () => {
    const v = parseFloat(sliderGain.value);
    document.getElementById('valGain').textContent = `${v.toFixed(0)}x`;
    syncRangeFallbackProgress(sliderGain);
    dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', organs: { eulerianGain: v } }, false);
    worker.postMessage({ type: 'params', eulerianGain: v });
  });

  function setHeroTargetsVisible(nextVal) {
    const visible = Boolean(nextVal);
    dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', overlays: { hero: visible } }, false);
    dispatchDeckEvent({ type: 'OVERLAY_SET', showHeroTargets: visible });
    worker.postMessage({ type: 'params', showHeroTargets: visible, overlaySyncSeq: nextOverlaySyncSeq() });
  }

  if (dom.chkHeroTargets) {
    dom.chkHeroTargets.addEventListener('change', (e) => {
      setHeroTargetsVisible(e.target.checked);
    });
  }

  if (dom.btnHudHeroToggle) {
    dom.btnHudHeroToggle.addEventListener('click', () => {
      setHeroTargetsVisible(!deckUiState.overlays.showHeroTargets);
    });
  }

  async function toggleHudFullscreen(forceActive) {
    const wantActive = typeof forceActive === 'boolean'
      ? forceActive
      : deckUiState.viewport !== 'FULLSCREEN';
    const fsElem = document.fullscreenElement || document.webkitFullscreenElement;
    if (wantActive) {
      if (fsElem === dom.hudPanel) {
        hudMaximizedFallback = false;
        dispatchDeckEvent({ type: 'FULLSCREEN_SYNC', active: true });
        return;
      }
      if (dom.hudPanel && typeof dom.hudPanel.requestFullscreen === 'function') {
        try {
          await dom.hudPanel.requestFullscreen();
          return;
        } catch (_err) {
          // Fall back to fixed viewport CSS .hud-maximized when browser blocks Fullscreen API
        }
      } else if (dom.hudPanel && typeof dom.hudPanel.webkitRequestFullscreen === 'function') {
        try {
          dom.hudPanel.webkitRequestFullscreen();
          return;
        } catch (_err) {
          // Fall back to .hud-maximized
        }
      }
      hudMaximizedFallback = true;
      dispatchDeckEvent({ type: 'FULLSCREEN_SYNC', active: true });
    } else {
      hudMaximizedFallback = false;
      if (fsElem) {
        try {
          if (typeof document.exitFullscreen === 'function') {
            await document.exitFullscreen();
            return;
          }
          if (typeof document.webkitExitFullscreen === 'function') {
            document.webkitExitFullscreen();
            return;
          }
        } catch (_err) {
          // Clear fallback below
        }
      }
      dispatchDeckEvent({ type: 'FULLSCREEN_SYNC', active: false });
    }
  }

  function handleFullscreenChange() {
    const fsElem = document.fullscreenElement || document.webkitFullscreenElement;
    const nativeActive = Boolean(dom.hudPanel && fsElem === dom.hudPanel);
    if (nativeActive) {
      hudMaximizedFallback = false;
    }
    dispatchDeckEvent({ type: 'FULLSCREEN_SYNC', active: nativeActive || hudMaximizedFallback });
  }

  document.addEventListener('fullscreenchange', handleFullscreenChange);
  document.addEventListener('webkitfullscreenchange', handleFullscreenChange);

  if (dom.btnHudFullscreen) {
    dom.btnHudFullscreen.addEventListener('click', () => {
      toggleHudFullscreen();
    });
  }

  document.getElementById('chkKalman').addEventListener('change', (e) => {
    dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', overlays: { kalman: e.target.checked } }, false);
    dispatchDeckEvent({ type: 'OVERLAY_SET', showKalman: e.target.checked });
    worker.postMessage({ type: 'params', showKalmanBaseline: e.target.checked, overlaySyncSeq: nextOverlaySyncSeq() });
  });
  document.getElementById('chkAllOverlays').addEventListener('change', (e) => {
    dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', overlays: { allCallouts: e.target.checked } }, false);
    dispatchDeckEvent({ type: 'OVERLAY_SET', showAllOverlays: e.target.checked });
    worker.postMessage({ type: 'params', showAllOverlays: e.target.checked, overlaySyncSeq: nextOverlaySyncSeq() });
  });
  document.getElementById('chkEcaOverlay').addEventListener('change', (e) => {
    dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', overlays: { eca: e.target.checked } }, false);
    dispatchDeckEvent({ type: 'OVERLAY_SET', showEcaOverlay: e.target.checked });
    worker.postMessage({ type: 'params', showEcaOverlay: e.target.checked, overlaySyncSeq: nextOverlaySyncSeq() });
  });
  document.getElementById('chkStrobe').addEventListener('change', (e) => {
    dispatchInstrument({ type: 'APPLY_CONFIG_PATCH', overlays: { strobe: e.target.checked } }, true);
    worker.postMessage({ type: 'params', strobeEnabled: e.target.checked });
  });

  // ========================================================================
  // Accessible Validation & Polite Error Live Region (modern-web-guidance)
  // ========================================================================
  const linkError = (el, shown) => {
    const errorId = el.dataset?.error;
    if (!errorId) return;
    const ids = (el.getAttribute('aria-describedby') || '')
      .split(/\s+/)
      .filter((id) => id && id !== errorId);
    if (shown) ids.push(errorId);
    if (ids.length) el.setAttribute('aria-describedby', ids.join(' '));
    else el.removeAttribute('aria-describedby');
  };

  const announcer = document.createElement('div');
  announcer.setAttribute('aria-live', 'polite');
  announcer.classList.add('visually-hidden');
  document.body.append(announcer);
  let announceTimer;
  let announcedField;

  const spokenText = (node) => {
    const copy = node.cloneNode(true);
    copy.querySelectorAll('[aria-hidden="true"]').forEach((n) => n.remove());
    copy.querySelectorAll('[role="img"][aria-label]').forEach((n) => n.replaceWith(`${n.getAttribute('aria-label')}: `));
    return copy.textContent.replace(/\s+/g, ' ').trim();
  };

  const announceError = (el) => {
    const error = el.dataset?.error ? document.getElementById(el.dataset.error) : null;
    if (!error || liveDialog.open) return;
    const label = el.labels?.[0] ? `${spokenText(el.labels[0])}: ` : '';
    clearTimeout(announceTimer);
    announcer.textContent = '';
    announcedField = el;
    announceTimer = setTimeout(() => {
      if (!liveDialog.open) announcer.textContent = label + spokenText(error);
    }, 500);
  };

  const flagged = new WeakSet();
  const announced = new WeakSet();
  const supportsUserInvalid = CSS.supports('selector(:user-invalid)');
  function syncAria(el, announce = false) {
    if (!el || !el.matches?.('input, select, textarea')) return;
    const invalid = supportsUserInvalid
      ? (el.matches(':user-invalid') || !el.validity.valid && flagged.has(el))
      : el.classList.contains('user-invalid-fallback');
    if (invalid) {
      if (announce && !announced.has(el)) {
        announceError(el);
        announced.add(el);
      }
      el.setAttribute('aria-invalid', 'true');
      linkError(el, true);
      flagged.add(el);
    } else if (flagged.has(el) || el.getAttribute('aria-invalid') !== 'true') {
      el.setAttribute('aria-invalid', 'false');
      linkError(el, false);
      flagged.delete(el);
      announced.delete(el);
      if (announcedField === el) {
        clearTimeout(announceTimer);
        announcer.textContent = '';
      }
    }
  }

  document.querySelectorAll('input, select, textarea').forEach((el) => syncAria(el));
  document.addEventListener('blur', (e) => syncAria(e.target, true), true);
  document.addEventListener('input', (e) => {
    if (e.target.hasAttribute?.('aria-invalid')) syncAria(e.target);
  });
  document.addEventListener('change', (e) => syncAria(e.target, true), true);
  document.addEventListener('invalid', (e) => {
    flagged.add(e.target);
    e.target.classList.add('user-invalid-fallback');
    announced.add(e.target);
    syncAria(e.target);
  }, true);

  // Metric & Perspective Calibration Toolbar + Scale & Keyboard Coordinate Forms
  if (!('focusGroup' in HTMLElement.prototype)) {
    const toolbarItems = Array.from(dom.calToolbar.querySelectorAll('button'));
    toolbarItems.forEach((b, i) => {
      b.tabIndex = i ? -1 : 0;
    });
    dom.calToolbar.addEventListener('focusin', ({ target }) => {
      if (toolbarItems.includes(target)) {
        toolbarItems.forEach((b) => {
          b.tabIndex = b === target ? 0 : -1;
        });
      }
    });
    dom.calToolbar.addEventListener('keydown', (ev) => {
      const idx = toolbarItems.indexOf(document.activeElement);
      if (idx < 0) return;
      const vertical = dom.calToolbar.getAttribute('aria-orientation') === 'vertical';
      const rtl = getComputedStyle(dom.calToolbar).direction === 'rtl';
      const steps = vertical
        ? { ArrowDown: 1, ArrowUp: -1 }
        : { ArrowRight: rtl ? -1 : 1, ArrowLeft: rtl ? 1 : -1 };
      let nextIdx = -1;
      if (Object.hasOwn(steps, ev.key)) {
        nextIdx = (idx + steps[ev.key] + toolbarItems.length) % toolbarItems.length;
      } else if (ev.key === 'Home') {
        nextIdx = 0;
      } else if (ev.key === 'End') {
        nextIdx = toolbarItems.length - 1;
      }
      if (nextIdx >= 0) {
        ev.preventDefault();
        toolbarItems[nextIdx].focus();
      }
    });
  }

  const calScaleInputs = [dom.calRulerM, dom.calPlaneW, dom.calPlaneH];

  function parseAndValidateScaleField(inputEl, announce = false) {
    const raw = inputEl.value.trim();
    const num = Number(raw);
    if (!raw || !/^\d+(\.\d+)?$/.test(raw) || !Number.isFinite(num) || num < 0.05 || num > 500) {
      inputEl.setCustomValidity('Enter positive meters between 0.05 and 500.');
      flagged.add(inputEl);
      inputEl.classList.add('user-invalid-fallback');
      syncAria(inputEl, announce);
      return null;
    }
    inputEl.setCustomValidity('');
    inputEl.classList.remove('user-invalid-fallback');
    syncAria(inputEl, false);
    return num;
  }

  function readValidatedCalibrationDims(routeFocusOnFail) {
    let firstBad = null;
    const vals = [];
    for (const inp of calScaleInputs) {
      const v = parseAndValidateScaleField(inp, !routeFocusOnFail);
      if (v === null && !firstBad) {
        firstBad = inp;
      }
      vals.push(v);
    }
    if (firstBad) {
      dom.calStatusLive.textContent = 'INVALID SCALE · ENTER POSITIVE METERS (0.05–500 m)';
      if (routeFocusOnFail) firstBad.focus();
      return null;
    }
    return { rulerMeters: vals[0], planeW: vals[1], planeH: vals[2] };
  }

  calScaleInputs.forEach((inp) => {
    inp.addEventListener('input', () => {
      inp.setCustomValidity('');
      inp.classList.remove('user-invalid-fallback');
      syncAria(inp, false);
    });
    inp.addEventListener('blur', () => {
      const v = parseAndValidateScaleField(inp, true);
      if (v === null) {
        dom.calStatusLive.textContent = 'INVALID SCALE · ENTER POSITIVE METERS (0.05–500 m)';
        return;
      }
      const dims = readValidatedCalibrationDims(false);
      if (dims) {
        worker.postMessage({ type: 'cal_control', action: 'stage_dims', ...dims });
      }
    });
    inp.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' && ev.isComposing) {
        ev.preventDefault();
      }
    });
  });

  dom.calScaleForm.addEventListener('submit', (ev) => {
    ev.preventDefault();
    const dims = readValidatedCalibrationDims(true);
    if (dims) {
      worker.postMessage({ type: 'cal_control', action: 'update_dims', ...dims });
    }
  });

  dom.calCoordMode.addEventListener('change', () => {
    const isQuad = dom.calCoordMode.value === 'homography';
    document.querySelectorAll('.coord-quad-only').forEach((el) => {
      el.hidden = !isQuad;
    });
  });

  dom.calCoordForm.addEventListener('submit', (ev) => {
    ev.preventDefault();
    const dims = readValidatedCalibrationDims(true);
    if (!dims) return;
    const coordMode = dom.calCoordMode.value === 'homography' ? 'homography' : 'ruler';
    const count = coordMode === 'homography' ? 4 : 2;
    const pts = [];
    for (let i = 1; i <= count; i++) {
      const xEl = document.getElementById(`coordX${i}`);
      const yEl = document.getElementById(`coordY${i}`);
      const vx = Number(xEl.value.trim());
      const vy = Number(yEl.value.trim());
      if (!Number.isFinite(vx) || vx < 0 || vx > 1 || !Number.isFinite(vy) || vy < 0 || vy > 1) {
        dom.calStatusLive.textContent = 'COORDINATES MUST BE IN [0.00, 1.00] · PRIOR CALIBRATION PRESERVED';
        (Number.isFinite(vx) && vx >= 0 && vx <= 1 ? yEl : xEl).focus();
        return;
      }
      pts.push([vx, vy]);
    }
    worker.postMessage({
      type: 'cal_control',
      action: 'set_coords',
      coordMode,
      pts,
      ...dims,
    });
  });

  function getActivePresetKey() {
    if (activeSourceMode === 'video' && sampleVideoSelect.value) {
      return sampleVideoSelect.value.split('/').pop().replace(/\.[^.]+$/, '');
    }
    if (activeSourceMode === 'testbench') return 'testbench';
    return 'custom_uncalibrated';
  }

  dom.btnCalRuler.addEventListener('click', () => {
    const dims = readValidatedCalibrationDims(true);
    if (!dims) return;
    const disarm = uiCalArmed === 'ruler' || deckUiState.pointerTool.startsWith('CAL_RULER_');
    const nextArm = disarm ? 'none' : 'ruler';
    uiCalArmed = nextArm;
    lastCalArmedUi = nextArm;
    const syncSeq = typeof beginOptimisticCalArm === 'function' ? beginOptimisticCalArm(nextArm) : 0;
    dom.btnCalRuler.setAttribute('aria-pressed', String(nextArm === 'ruler'));
    if (nextArm === 'ruler') dom.btnCalQuad.setAttribute('aria-pressed', 'false');
    worker.postMessage({
      type: 'cal_control',
      action: disarm ? 'cancel_arm' : 'arm_ruler',
      syncSeq,
      ...dims,
    });
  });

  dom.btnCalQuad.addEventListener('click', () => {
    const dims = readValidatedCalibrationDims(true);
    if (!dims) return;
    const disarm = uiCalArmed === 'homography' || deckUiState.pointerTool.startsWith('CAL_QUAD_');
    const nextArm = disarm ? 'none' : 'homography';
    uiCalArmed = nextArm;
    lastCalArmedUi = nextArm;
    const syncSeq = typeof beginOptimisticCalArm === 'function' ? beginOptimisticCalArm(nextArm) : 0;
    dom.btnCalQuad.setAttribute('aria-pressed', String(nextArm === 'homography'));
    if (nextArm === 'homography') dom.btnCalRuler.setAttribute('aria-pressed', 'false');
    worker.postMessage({
      type: 'cal_control',
      action: disarm ? 'cancel_arm' : 'arm_homography',
      syncSeq,
      ...dims,
    });
  });

  dom.btnCalAutoG.addEventListener('click', () => {
    uiCalArmed = 'none';
    lastCalArmedUi = 'none';
    const syncSeq = typeof beginOptimisticCalArm === 'function' ? beginOptimisticCalArm('none') : 0;
    worker.postMessage({ type: 'cal_control', action: 'set_auto_g', syncSeq });
  });

  dom.btnCalPreset.addEventListener('click', () => {
    uiCalArmed = 'none';
    lastCalArmedUi = 'none';
    const syncSeq = typeof beginOptimisticCalArm === 'function' ? beginOptimisticCalArm('none') : 0;
    worker.postMessage({ type: 'cal_control', action: 'apply_preset', preset: getActivePresetKey(), syncSeq });
  });

  dom.btnCalClear.addEventListener('click', () => {
    uiCalArmed = 'none';
    lastCalArmedUi = 'none';
    const syncSeq = typeof beginOptimisticCalArm === 'function' ? beginOptimisticCalArm('none') : 0;
    dom.btnCalRuler.setAttribute('aria-pressed', 'false');
    dom.btnCalQuad.setAttribute('aria-pressed', 'false');
    worker.postMessage({ type: 'cal_control', action: 'clear', syncSeq });
  });

  function handleBenchEscape(ev) {
    if (hudMaximizedFallback || deckUiState.viewport === 'FULLSCREEN') {
      ev.preventDefault();
      toggleHudFullscreen(false);
    }
    if (uiCalArmed !== 'none' || deckUiState.pointerTool.startsWith('CAL_')) {
      ev.preventDefault();
      uiCalArmed = 'none';
      lastCalArmedUi = 'none';
      const syncSeq = typeof beginOptimisticCalArm === 'function' ? beginOptimisticCalArm('none') : 0;
      dom.btnCalRuler.setAttribute('aria-pressed', 'false');
      dom.btnCalQuad.setAttribute('aria-pressed', 'false');
      worker.postMessage({ type: 'cal_control', action: 'cancel_arm', syncSeq });
      dispatchInstrument({ type: 'ESCAPE_PRESS' }, true);
      return;
    }
    if (deckUiState.pointerTool !== 'INSPECT_TARGET' || instrumentState.pointer_tool !== 'INSPECT_TARGET') {
      ev.preventDefault();
      setCropRoiToolActive(false);
      dispatchInstrument({ type: 'ESCAPE_PRESS' }, true);
    }
  }

  // ---- Mount v3.5 Views (GatesView, BenchShell, AuditBench) & Live Meter Sync ----
  const bayMeterTrack = document.getElementById('bayMeterTrack');
  const bayMeterPhase = document.getElementById('bayMeterPhase');
  const bayMeterPhysics = document.getElementById('bayMeterPhysics');
  const bayMeterField = document.getElementById('bayMeterField');
  const bayMeterAudit = document.getElementById('bayMeterAudit');

  let lastMeterSyncMs = 0;
  function syncRuntimeTelemetryToInstrument(m, medMs) {
    const now = performance.now();
    if (now - lastMeterSyncMs >= 180) {
      lastMeterSyncMs = now;
      const shownSwaps = m.sourceMode === 'testbench' ? (m.oracleSwaps ?? 0) : (m.padicSwaps ?? 0);
      setTextIfChanged(bayMeterTrack, `${m.activeTargets || 0}/16 · ${shownSwaps} sw`);
      setTextIfChanged(bayMeterPhase, m.lookaheadK < 7 ? 'K<7 GATED' : (m.freqValid && m.tremorFreqHz > 0 ? `${m.tremorFreqHz.toFixed(1)} Hz` : 'STANDBY'));
      const regTxt = dom.physRegimeBadge ? dom.physRegimeBadge.textContent : 'STANDBY';
      setTextIfChanged(bayMeterPhysics, regTxt);
      const owner = computeBosOwner(instrumentState.organ_config);
      const c1k = m.chronos1000;
      setTextIfChanged(
        bayMeterField,
        owner !== BOS_OWNERS.NONE ? bosOwnerDisplayName(owner) : (c1k && c1k.swarmActive ? `SWARM ${c1k.confirmedCount || 0}` : 'STANDBY')
      );
      if (typeof auditBenchHandle !== 'undefined' && auditBenchHandle && auditBenchHandle.updateLiveMirror) {
        const w2Live = m.wave2 || {};
        const act = m.activeTargets || 0;
        const windSt = w2Live.wind && w2Live.wind.state ? ` · Wind ${w2Live.wind.stateLabel || w2Live.wind.state}` : '';
        const consSt = w2Live.conservation && w2Live.conservation.status ? ` · Cons ${w2Live.conservation.statusLabel || w2Live.conservation.status}` : '';
        auditBenchHandle.updateLiveMirror(`Live HUD Mirror · ${act}/16 locks · ${(medMs || 0).toFixed(2)} ms${windSt}${consSt} (0 extra Wasm)`);
      }
    }

    const w2 = m.wave2 || {};
    const bos = m.bos || {};
    const met = m.metrology || {};
    const roi = m.roi || {};
    if (m.camMotionState === 2 && instrumentState.runtime_health.camMotionState !== 2) {
      dispatchInstrument({ type: 'SCENE_CUT' }, false);
    }
    dispatchInstrument({
      type: 'RUNTIME_TELEMETRY',
      health: {
        srcW: roi.srcW || m.frameW || 0,
        srcH: roi.srcH || m.frameH || 0,
        fps: m.fps || 0,
        effectiveK: m.lookaheadK ?? instrumentState.organ_config.lookahead,
        camMotionState: m.camMotionState || 0,
        swarmActive: Boolean(m.chronos1000 && m.chronos1000.swarmActive),
        kernelTickMs: medMs || 0,
        activeTargets: m.activeTargets || 0,
        audioPresent: Boolean(probeExciterActive || rsAudioCaptureTrack || (w2.vibro && (w2.vibro.welchK || 0) > 0)),
        products: {
          bos: { state: bos.state || 0, reasonMask: bos.reasonMask || 0, valid: bos.state === 2 },
          hodge: { state: w2.hodge?.state || 0, reasonMask: 0, valid: w2.hodge?.state === 2 },
          rs: { state: w2.vibro?.state || 0, reasonMask: w2.vibro?.reasonMask || 0, valid: w2.vibro?.state === 3 },
          depth: { state: w2.depth?.state || 0, reasonMask: w2.depth?.reasonMask || 0, valid: (w2.depth?.state || 0) >= 2 && (w2.depth?.state || 0) !== 4 && (w2.depth?.state || 0) !== 5 },
          wind: { state: w2.wind?.state || 0, reasonMask: w2.wind?.reasonMask || 0, valid: w2.wind?.state === 2 || w2.wind?.state === 3, provenance: w2.wind?.provenance || 0 },
          elasto: { state: w2.elasto?.state || 0, reasonMask: w2.elasto?.reasonMask || 0, valid: w2.elasto?.state === 3 },
          metrology: { state: met.available ? 2 : 0, usable: met.usableCovCount || 0, loom: met.loomRejectedCount || 0, cand: met.promotedCandidates || 0 },
        },
      },
    }, false);
  }

  function applyPlannedIntent(intent, label) {
    if (!intent) return null;
    if (intent.kind === 'SET_RES') {
      document.getElementById(intent.resMode === 1 ? 'btnResHd' : 'btnResSd')?.click();
      return instrumentState;
    }
    const plan = planIntent(instrumentState, intent);
    return commitAtomicPlan(plan, { recordUndo: true, undoLabel: label || plan.summary });
  }

  function undoLastConfig() {
    if (!instrumentState.undoConfig) return false;
    const prevBos = instrumentState.organ_config.bos;
    const st = dispatchInstrument({ type: 'UNDO_CONFIG' }, true);
    syncSelectsFromOrgans(st.organ_config, st.overlays, st.resolution_roi.native1to1);
    dispatchDeckEvent({
      type: 'ENGINE_SYNC',
      swarmMode: st.organ_config.swarm,
      swarmActive: st.organ_config.swarm === 2 ? true : (st.organ_config.swarm === 0 ? false : lastSwarmActive),
    });
    dispatchDeckEvent({
      type: 'OVERLAY_SET',
      showHeroTargets: st.overlays.hero,
      showCovEllipses: st.overlays.covEllipses,
      fieldOverlayMode: st.overlays.field,
    });
    syncLabAndLookaheadControls(st);
    postAtomicConfigToWorker(st, { resetBosRef: prevBos === 0 && st.organ_config.bos > 0 });
    return true;
  }

  function exportCurrentRecipeJson() {
    const st = instrumentState;
    const payload = {
      schema: 'chronos-recipe/1',
      exportedAt: new Date().toISOString(),
      source: { kind: st.source.kind, label: st.source.label, url: st.source.sampleUrl || null, scenario: st.source.scenario },
      resMode: st.resolution_roi.resMode,
      native1to1: st.resolution_roi.native1to1,
      labMode: st.lab_mode,
      organs: st.organ_config,
      overlays: st.overlays,
      bayTab: st.workspace_layout.bayTab,
    };
    const stamp = typeof formatExportTimestamp === 'function' ? formatExportTimestamp() : String(Date.now());
    triggerBlobDownload(`chronos-recipe-${stamp}.json`, 'application/json;charset=utf-8', JSON.stringify(payload, null, 2));
    if (dom.recStatusLabel) dom.recStatusLabel.textContent = 'EXPORTED RECIPE JSON';
  }

  gatesView = createGatesView({
    onApplyIntent: (intent, label) => applyPlannedIntent(intent, label),
    onTakeBosOwner: (owner) => applyPlannedIntent({ kind: 'TAKE_BOS_OWNER', owner }, `Scratch -> ${bosOwnerDisplayName(owner)}`),
    onUndo: () => undoLastConfig(),
    onStartProbe: () => { if (!probeExciterActive) startProbeExciter(); },
  });

  benchShell = createBenchShell({
    getState: () => instrumentState,
    dispatch: (ev) => dispatchInstrument(ev, true),
    onScopeVisibilityMask: (mask) => worker.postMessage({ type: 'set_scope_visibility', mask }),
    onLoadRecipe: (key, opts) => loadRecipe(key, opts),
    onOpenFile: () => mp4Input?.click(),
    onStartWebcam: () => {
      btnWebcam?.click();
    },
    onOpenLiveStream: () => openLiveStreamModal(),
    onStartSignalGen: () => {
      dispatchInstrument({ type: 'DISMISS_WELCOME' }, true);
      btnTestbench?.click();
    },
    onSelectPointerTool: (tool) => {
      if (tool === 'INSPECT_TARGET') {
        if (uiCalArmed !== 'none') {
          uiCalArmed = 'none';
          lastCalArmedUi = 'none';
          worker.postMessage({ type: 'cal_control', action: 'cancel_arm', syncSeq: beginOptimisticCalArm('none') });
        }
        setCropRoiToolActive(false);
        dispatchDeckEvent({ type: 'POINTER_TOOL_SET', pointerTool: 'INSPECT_TARGET' });
        dispatchInstrument({ type: 'SET_POINTER_TOOL', tool: 'INSPECT_TARGET' }, true);
      } else if (tool === 'SEED_BOX') {
        setCropRoiToolActive(false);
        dispatchDeckEvent({ type: 'POINTER_TOOL_SET', pointerTool: 'SEED_BOX' });
        dispatchInstrument({ type: 'SET_POINTER_TOOL', tool: 'SEED_BOX' }, true);
      } else if (tool === 'WIPE_DRAG') {
        setCropRoiToolActive(false);
        const cur = dom.showcaseWipeSlider ? (Number(dom.showcaseWipeSlider.value) || 0) : 0;
        if (cur === 0) setShowcaseWipePercent(28, true);
        dispatchDeckEvent({ type: 'POINTER_TOOL_SET', pointerTool: 'WIPE_DRAG' });
        dispatchInstrument({ type: 'SET_POINTER_TOOL', tool: 'WIPE_DRAG' }, true);
        dom.hudWipeHandle?.focus();
      }
    },
    onTakeBosScratch: (owner) => applyPlannedIntent({ kind: 'TAKE_BOS_OWNER', owner }, `Scratch -> ${bosOwnerDisplayName(owner)}`),
    onApplyIntent: (intent, label) => applyPlannedIntent(intent, label),
    onPlanIntent: (intent) => planIntent(instrumentState, intent),
    onUndo: () => undoLastConfig(),
    onExportRecipe: () => exportCurrentRecipeJson(),
    onToggleHero: () => setHeroTargetsVisible(!deckUiState.overlays.showHeroTargets),
    onToggleFullscreen: () => toggleHudFullscreen(),
    onCycleFocus: () => worker.postMessage({ type: 'cycle_focus' }),
    onToggleRec: () => document.getElementById('btnRecToggle')?.click(),
    onPlayPause: () => dom.btnVidPlayPause?.click(),
    onStepFrame: (d) => stepVideoByFrames(d),
    onEscape: (ev) => handleBenchEscape(ev),
  });

  if (window.__chronosUi) {
    window.__chronosUi.getConstraintMap = () => Object.fromEntries(deriveControlStates(instrumentState));
  }

  const auditBenchHandle = initAuditBench({
    getActiveVideoSource: () => {
      if (activeSourceMode !== 'video') return null;
      return { file: activeVideoFile, url: activeVideoUrl };
    },
    onSeekToTimeSec: (sec) => {
      if (activeSourceMode === 'video' && Number.isFinite(sec)) {
        seekVideoToTime(sec);
      }
    },
    onSeekToFraction: (frac) => {
      if (activeSourceMode === 'video' && Number.isFinite(videoEl.duration) && videoEl.duration > 0) {
        seekVideoToTime(frac * videoEl.duration);
      }
    },
    onAuditEvent: (patch) => dispatchInstrument({ type: 'AUDIT_EVENT', ...patch }, false),
    onBayMeterUpdate: (txt) => setTextIfChanged(bayMeterAudit, txt),
  });

  renderInstrumentViews(true);

  // Session Telemetry Recorder Controls
  document.getElementById('btnRecToggle').addEventListener('click', () => {
    worker.postMessage({
      type: 'rec_control',
      action: uiRecActive ? 'stop' : 'start',
      clear: false,
    });
  });
  document.getElementById('btnRecClear').addEventListener('click', () => {
    worker.postMessage({ type: 'rec_control', action: 'clear' });
  });
  document.getElementById('btnExportCsv').addEventListener('click', () => {
    worker.postMessage({ type: 'export_telemetry', format: 'csv' });
  });
  document.getElementById('btnExportJson').addEventListener('click', () => {
    worker.postMessage({ type: 'export_telemetry', format: 'json' });
  });

}
