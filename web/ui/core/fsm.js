// ============================================================================
// CHRONOS-SPECTRAL v3.5 // Canonical 100% Instrument Finite State Machine
// Pure zero-DOM ES module shared identically by browser runtime & Node verifier.
// ============================================================================

export const BOS_OWNERS = Object.freeze({
  NONE: 'NONE',
  DEPTH_STEREO: 'DEPTH_STEREO',
  BOS_THERMAL: 'BOS_THERMAL',
  WIND_TUNNEL: 'WIND_TUNNEL',
  ELASTO_ACOUSTIC_BOS: 'ELASTO_ACOUSTIC_BOS',
  ELASTO_SURFACE: 'ELASTO_SURFACE',
});

// Strict kernel hardware scratch priority (`src/lib.rs:14558-14577`).
// Reserve auto-fused depth (mode 4) across its entire selection so a late
// stereo reference can never preempt an active Wind or Elastography run mid-stream.
export const BOS_PRIORITY_ORDER = Object.freeze([
  BOS_OWNERS.DEPTH_STEREO,
  BOS_OWNERS.BOS_THERMAL,
  BOS_OWNERS.WIND_TUNNEL,
  BOS_OWNERS.ELASTO_ACOUSTIC_BOS,
  BOS_OWNERS.ELASTO_SURFACE,
]);

export const POINTER_TOOLS = Object.freeze([
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

export const RACK_TABS = Object.freeze(['engines', 'acquire', 'calibrate', 'record']);
export const BAY_TABS = Object.freeze(['track', 'phase', 'physics', 'field', 'audit']);

export function clampInt(val, lo, hi, fallback) {
  const n = Number(val);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(lo, Math.min(hi, Math.round(n)));
}

export function clampNum(val, lo, hi, fallback) {
  const n = Number(val);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(lo, Math.min(hi, n));
}

export function createDefaultOrgans() {
  return {
    swarm: 0,
    bos: 0,
    hodge: 0,
    metrology: 0,
    rs: 0,
    depth: 0,
    wind: 0,
    elasto: 0,
    lookahead: 8,
    bosGauge: [0, 0, 0, 0],
    rsCal: { rowUs: 40.0, exposureUs: 200.0, scan: 0, rateHz: 48000.0, maxSkewUs: 500.0 },
    depthCal: { fPx: 0.0, speedMS: 0.0, baselineM: 0.0, maxDispPx: 32.0, stereoSign: 0 },
    fluidDensity: 0.0,
    lockHz: [],
    ecaRate: 0.035,
    fistaLambda: 1.45,
    eulerianGain: 25,
  };
}

export function createDefaultOverlays() {
  return {
    field: 0,
    hero: true,
    kalman: true,
    allCallouts: false,
    eca: false,
    covEllipses: false,
    strobe: true,
    phase: 'LAG',
    wipePct: 0,
  };
}

export function createDefaultRuntimeHealth(hostCaps = {}) {
  return {
    workerState: 'BOOTING', // BOOTING | READY | FAILED
    workerEpoch: 0,
    appliedRevision: 0,
    continuityEpoch: 1,
    geometryEpoch: 0,
    srcW: 0,
    srcH: 0,
    fps: 0,
    seekable: false,
    nativeEligible: false,
    foveaEligible: false,
    effectiveK: 8,
    camMotionState: 0, // 0=locked, 1=pan, 2=cut
    panStreak: 0,
    roiSlewActive: false,
    swarmActive: false,
    audioPresent: false,
    audioRoute: 'NONE', // NONE | MUXED | MIC | CAPTURE_TRACK | PROBE | BENCH
    stereoRefFresh: false,
    externalDepthFresh: false,
    observedBosOwner: BOS_OWNERS.NONE,
    kernelTickMs: 0,
    activeTargets: 0,
    products: {
      bos: { state: 0, reasonMask: 0, valid: false },
      hodge: { state: 0, reasonMask: 0, valid: false },
      rs: { state: 0, reasonMask: 0, valid: false },
      depth: { state: 0, reasonMask: 0, valid: false },
      wind: { state: 0, reasonMask: 0, valid: false, provenance: 0 },
      elasto: { state: 0, reasonMask: 0, valid: false },
      metrology: { state: 0, usable: 0, loom: 0, cand: 0 },
    },
    host: {
      staticHost: Boolean(hostCaps.staticHost),
      relayAvailable: hostCaps.relayAvailable !== undefined ? Boolean(hostCaps.relayAvailable) : !hostCaps.staticHost,
      webCodecs: hostCaps.webCodecs !== undefined ? Boolean(hostCaps.webCodecs) : true,
      webcamCapable: hostCaps.webcamCapable !== undefined ? Boolean(hostCaps.webcamCapable) : true,
      displayCaptureCapable: hostCaps.displayCaptureCapable !== undefined ? Boolean(hostCaps.displayCaptureCapable) : true,
      webAudioCapable: hostCaps.webAudioCapable !== undefined ? Boolean(hostCaps.webAudioCapable) : true,
      fullscreenCapable: hostCaps.fullscreenCapable !== undefined ? Boolean(hostCaps.fullscreenCapable) : true,
    },
  };
}

/**
 * Exact deterministic reservation list of BOS scratch claimants from `organs`.
 * Matches `src/lib.rs:14558-14577` priority order, reserving auto-fused depth
 * (`depth === 4`) across its full selection so a late stereo ref never causes
 * a surprise mid-stream conflict.
 */
export function listBosClaimants(organs) {
  const out = [];
  if (!organs) return out;
  if (organs.depth === 2 || organs.depth === 4) out.push(BOS_OWNERS.DEPTH_STEREO);
  if (organs.bos === 1 || organs.bos === 2) out.push(BOS_OWNERS.BOS_THERMAL);
  if (organs.wind === 1 || organs.wind === 2 || organs.wind === 3) out.push(BOS_OWNERS.WIND_TUNNEL);
  if (organs.elasto === 4) out.push(BOS_OWNERS.ELASTO_ACOUSTIC_BOS);
  if (organs.elasto === 1 || organs.elasto === 2 || organs.elasto === 3) out.push(BOS_OWNERS.ELASTO_SURFACE);
  return out;
}

/**
 * Kernel winner under the exact priority order in `src/lib.rs:14564-14577`.
 * When `stereoRefFresh` is provided as `false` for `depth === 4`, computes the
 * *actual kernel tick owner* (for Gate/E2E parity checks). Without `runtime`,
 * computes the *reserved owner*.
 */
export function computeBosOwner(organs, runtime = null) {
  if (!organs) return BOS_OWNERS.NONE;
  const stereoClaims = runtime
    ? (organs.depth === 2 || (organs.depth === 4 && organs.depthCal.fPx >= 10 && organs.depthCal.baselineM >= 1e-4 && runtime.stereoRefFresh))
    : (organs.depth === 2 || organs.depth === 4);
  if (stereoClaims) return BOS_OWNERS.DEPTH_STEREO;
  if (organs.bos === 1 || organs.bos === 2) return BOS_OWNERS.BOS_THERMAL;
  if (organs.wind === 1 || organs.wind === 2 || organs.wind === 3) return BOS_OWNERS.WIND_TUNNEL;
  if (organs.elasto === 4) return BOS_OWNERS.ELASTO_ACOUSTIC_BOS;
  if (organs.elasto === 1 || organs.elasto === 2 || organs.elasto === 3) return BOS_OWNERS.ELASTO_SURFACE;
  return BOS_OWNERS.NONE;
}

export function isExclusiveBosReservation(organs) {
  return listBosClaimants(organs).length <= 1;
}

export function createInitialInstrumentState(opts = {}) {
  const organs = { ...createDefaultOrgans(), ...(opts.organs || {}) };
  const overlays = { ...createDefaultOverlays(), ...(opts.overlays || {}) };
  const sourceKind = opts.sourceKind || 'IDLE';
  const state = {
    schema: 2,
    revision: 0,
    source: {
      kind: sourceKind, // IDLE | FILE | SAMPLE | WEBCAM | LIVE_TAB | LIVE_URL | LIVE_MSE | TESTBENCH
      epoch: 0,
      lifecycle: sourceKind === 'IDLE' ? 'IDLE' : 'READY',
      label: opts.sourceLabel || '',
      sampleUrl: opts.sampleUrl || '',
      scenario: opts.scenario || 'pillar_crossing',
      originPreset: opts.originPreset || null,
      presetModified: false,
    },
    transport: {
      kind: sourceKind === 'IDLE' ? 'IDLE' : (sourceKind === 'FILE' || sourceKind === 'SAMPLE' ? 'MEDIA' : 'CONTINUOUS'),
      phase: sourceKind === 'IDLE' ? 'STOPPED' : 'PLAYING',
      ptsUs: 0,
      durationUs: null,
      fastPassRunning: false,
      liveCrop: 1.0,
    },
    resolution_roi: {
      resMode: opts.resMode === 1 ? 1 : 0, // 0=SD 320x180, 1=HD 640x360
      native1to1: Boolean(opts.native1to1),
      autoTrackRoi: false,
      roiX0: 0,
      roiY0: 0,
      cropNorm: [0, 0, 1, 1],
    },
    organ_config: organs,
    bos_owner: computeBosOwner(organs),
    lab_mode: clampInt(opts.labMode ?? 0, 0, 2, 0), // 0=A, 1=B, 2=C
    overlays,
    calibration: {
      committed: { kind: 'PX', provenance: 'NONE', label: 'PX UNITS' },
      segment: 1,
      draft: null, // { kind: 'RULER'|'QUAD', points: [], bindingEpoch, bindingGeom }
    },
    pointer_tool: 'INSPECT_TARGET',
    pointer_binding: { sourceEpoch: 0, geometryEpoch: 0, generation: 0 },
    audio_pipeline: {
      route: 'NONE',
      synthOutput: false,
      probeOutput: false,
      probePreset: '380,860',
    },
    recorder: {
      active: false,
      frames: 0,
      durationSec: 0,
    },
    audit_bench: {
      worker: 'COLD', // COLD | STARTING | READY | FAILED
      workerEpoch: 0,
      liveMirror: false,
      A: { phase: 'EMPTY', jobId: null, clipName: '', grade: null, integrity: null, record: null },
      B: { phase: 'EMPTY', jobId: null, clipName: '', grade: null, integrity: null, record: null },
    },
    workspace_layout: {
      rackTab: opts.rackTab || 'engines',
      bayTab: opts.bayTab || 'track',
      dialog: 'NONE', // NONE | LIVE_SOURCE | PRESETS | SHORTCUTS
      viewport: 'DOCKED', // DOCKED | WIDE | FULLSCREEN
      welcomeVisible: sourceKind === 'IDLE' && opts.welcomeVisible !== false,
      pinnedTrack: null, // { sourceEpoch, tid }
      inspectedTrack: null,
      coordForm: 'ruler',
      dropTarget: null,
    },
    visibility_policy: opts.visibilityPolicy === 'APPLICABLE_ONLY' ? 'APPLICABLE_ONLY' : 'ALL_CONSTRAINED',
    runtime_health: createDefaultRuntimeHealth(opts.hostCaps || {}),
    pending: null, // { txId, workerEpoch, baseRevision, nextRevision, completeConfig }
    undoConfig: null, // 1-level undo snapshot { label, organ_config, overlays, lab_mode }
  };
  return state;
}

export function cloneConfigSnapshot(state) {
  return {
    organ_config: {
      ...state.organ_config,
      bosGauge: state.organ_config.bosGauge.slice(),
      rsCal: { ...state.organ_config.rsCal },
      depthCal: { ...state.organ_config.depthCal },
      lockHz: state.organ_config.lockHz.slice(),
    },
    overlays: { ...state.overlays },
    resolution_roi: { native1to1: state.resolution_roi.native1to1 },
    lab_mode: state.lab_mode,
  };
}

export function legacySourceTag(kind) {
  if (kind === 'IDLE') return 'IDLE';
  if (kind === 'TESTBENCH') return 'TESTBENCH';
  if (kind === 'WEBCAM') return 'WEBCAM';
  return 'VIDEO';
}

/**
 * Legacy-compatible `window.__chronosUiState` snapshot preserving every v3.x key
 * while adding v3.5 schema-2 instrument state fields.
 */
export function projectLegacyUiState(state) {
  const swMode = state.organ_config.swarm;
  const engine = state.runtime_health.swarmActive
    ? 'SWARM_ACTIVE'
    : (swMode === 0 ? 'SWARM_OFF' : 'AUTO_STANDBY');
  return Object.freeze({
    schema: 2,
    seq: state.revision,
    source: legacySourceTag(state.source.kind),
    sourceDetail: state.source.kind,
    engine,
    overlays: Object.freeze({
      showHeroTargets: state.overlays.hero,
      showAllOverlays: state.overlays.allCallouts,
      showEcaOverlay: state.overlays.eca,
      showKalman: state.overlays.kalman,
      showCovEllipses: state.overlays.covEllipses,
      fieldOverlayMode: state.overlays.field,
    }),
    pointerTool: state.pointer_tool,
    viewport: state.workspace_layout.viewport === 'FULLSCREEN' ? 'FULLSCREEN' : 'DOCKED',
    bosOwner: state.bos_owner,
    recipe: Object.freeze({
      key: state.source.originPreset,
      modified: state.source.presetModified,
    }),
    ui: Object.freeze({
      rackTab: state.workspace_layout.rackTab,
      bayTab: state.workspace_layout.bayTab,
      visibilityPolicy: state.visibility_policy,
      dialog: state.workspace_layout.dialog,
    }),
    lastInspectAck: state.workspace_layout.inspectedTrack,
  });
}

function withDerived(next) {
  next.bos_owner = computeBosOwner(next.organ_config);
  return next;
}

/**
 * Pure deterministic reducer. Every UI gesture, preset load, worker ACK,
 * telemetry update, calibration cut, Esc press, or audit job event flows here.
 */
export function reduceInstrumentState(state, event) {
  if (!event || typeof event.type !== 'string') return state;

  switch (event.type) {
    case 'WORKER_READY': {
      return withDerived({
        ...state,
        runtime_health: {
          ...state.runtime_health,
          workerState: 'READY',
          workerEpoch: state.runtime_health.workerEpoch + 1,
          appliedRevision: state.revision,
        },
        pending: null,
      });
    }

    case 'WORKER_FAULT': {
      return withDerived({
        ...state,
        runtime_health: {
          ...state.runtime_health,
          workerState: 'FAILED',
        },
        pending: null,
      });
    }

    case 'SET_VISIBILITY_POLICY': {
      const pol = event.policy === 'APPLICABLE_ONLY' ? 'APPLICABLE_ONLY' : 'ALL_CONSTRAINED';
      if (pol === state.visibility_policy) return state;
      return { ...state, visibility_policy: pol };
    }

    case 'SET_RACK_TAB': {
      const tab = RACK_TABS.includes(event.tab) ? event.tab : state.workspace_layout.rackTab;
      if (tab === state.workspace_layout.rackTab) return state;
      return { ...state, workspace_layout: { ...state.workspace_layout, rackTab: tab } };
    }

    case 'SET_BAY_TAB': {
      const tab = BAY_TABS.includes(event.tab) ? event.tab : state.workspace_layout.bayTab;
      if (tab === state.workspace_layout.bayTab) return state;
      return { ...state, workspace_layout: { ...state.workspace_layout, bayTab: tab } };
    }

    case 'SET_DIALOG': {
      const dlg = ['NONE', 'LIVE_SOURCE', 'PRESETS', 'SHORTCUTS'].includes(event.dialog)
        ? event.dialog
        : 'NONE';
      if (dlg === state.workspace_layout.dialog) return state;
      return { ...state, workspace_layout: { ...state.workspace_layout, dialog: dlg } };
    }

    case 'SET_VIEWPORT': {
      const vp = ['DOCKED', 'WIDE', 'FULLSCREEN'].includes(event.viewport) ? event.viewport : 'DOCKED';
      if (vp === state.workspace_layout.viewport) return state;
      return { ...state, workspace_layout: { ...state.workspace_layout, viewport: vp } };
    }

    case 'DISMISS_WELCOME': {
      if (!state.workspace_layout.welcomeVisible) return state;
      return { ...state, workspace_layout: { ...state.workspace_layout, welcomeVisible: false } };
    }

    case 'SET_POINTER_TOOL': {
      const tool = POINTER_TOOLS.includes(event.tool) ? event.tool : 'INSPECT_TARGET';
      const nextGen = state.pointer_binding.generation + 1;
      const isCalTool = tool.startsWith('CAL_');
      return {
        ...state,
        pointer_tool: tool,
        pointer_binding: {
          sourceEpoch: state.source.epoch,
          geometryEpoch: state.runtime_health.geometryEpoch,
          generation: nextGen,
        },
        calibration: {
          ...state.calibration,
          draft: isCalTool
            ? {
                kind: tool.startsWith('CAL_RULER') ? 'RULER' : 'QUAD',
                points: Array.isArray(event.points) ? event.points.slice() : [],
                bindingEpoch: state.source.epoch,
                bindingGeom: state.runtime_health.geometryEpoch,
              }
            : null,
        },
      };
    }

    case 'ESCAPE_PRESS': {
      // Layered Escape: 1) cancel armed calibration/non-inspect tool preserving committed scale;
      // 2) close top-layer dialog; 3) exit fullscreen/wide viewport.
      if (state.pointer_tool !== 'INSPECT_TARGET') {
        return {
          ...state,
          pointer_tool: 'INSPECT_TARGET',
          pointer_binding: {
            ...state.pointer_binding,
            generation: state.pointer_binding.generation + 1,
          },
          calibration: {
            ...state.calibration,
            draft: null, // Committed calibration strictly preserved!
          },
        };
      }
      if (state.workspace_layout.dialog !== 'NONE') {
        return {
          ...state,
          workspace_layout: { ...state.workspace_layout, dialog: 'NONE' },
        };
      }
      if (state.workspace_layout.viewport !== 'DOCKED') {
        return {
          ...state,
          workspace_layout: { ...state.workspace_layout, viewport: 'DOCKED' },
        };
      }
      return state;
    }

    case 'APPLY_CONFIG_PATCH': {
      // Atomic configuration transition (used by manual controls, 1-click fixes, and recipes).
      const prevUndo = event.recordUndo
        ? { label: event.undoLabel || 'Previous setup', ...cloneConfigSnapshot(state) }
        : state.undoConfig;
      const nextOrg = {
        ...state.organ_config,
        ...(event.organs || {}),
        rsCal: { ...state.organ_config.rsCal, ...(event.organs?.rsCal || {}) },
        depthCal: { ...state.organ_config.depthCal, ...(event.organs?.depthCal || {}) },
      };
      nextOrg.swarm = clampInt(nextOrg.swarm, 0, 2, 0);
      nextOrg.bos = clampInt(nextOrg.bos, 0, 2, 0);
      nextOrg.hodge = clampInt(nextOrg.hodge, 0, 2, 0);
      nextOrg.metrology = clampInt(nextOrg.metrology, 0, 2, 0);
      nextOrg.rs = clampInt(nextOrg.rs, 0, 2, 0);
      nextOrg.depth = clampInt(nextOrg.depth, 0, 4, 0);
      nextOrg.wind = clampInt(nextOrg.wind, 0, 3, 0);
      nextOrg.elasto = clampInt(nextOrg.elasto, 0, 4, 0);
      nextOrg.lookahead = clampInt(nextOrg.lookahead, 0, 11, 8);
      nextOrg.ecaRate = clampNum(nextOrg.ecaRate, 0.005, 0.15, 0.025);
      nextOrg.fistaLambda = clampNum(nextOrg.fistaLambda, 0.1, 5.0, 0.9);
      nextOrg.eulerianGain = clampInt(nextOrg.eulerianGain, 5, 60, 25);

      const nextOv = { ...state.overlays, ...(event.overlays || {}) };
      nextOv.field = clampInt(nextOv.field, 0, 15, 0);
      nextOv.wipePct = clampInt(nextOv.wipePct, 0, 100, 0);

      const nextRes = {
        ...state.resolution_roi,
        ...(event.resolution_roi || {}),
      };
      nextRes.resMode = nextRes.resMode === 1 ? 1 : 0;

      const nextLab = event.lab_mode !== undefined ? clampInt(event.lab_mode, 0, 2, state.lab_mode) : state.lab_mode;
      const nextRev = state.revision + 1;
      const geomChanged =
        nextRes.resMode !== state.resolution_roi.resMode ||
        nextRes.native1to1 !== state.resolution_roi.native1to1 ||
        nextRes.roiX0 !== state.resolution_roi.roiX0 ||
        nextRes.roiY0 !== state.resolution_roi.roiY0;

      const nextState = withDerived({
        ...state,
        revision: nextRev,
        organ_config: nextOrg,
        overlays: nextOv,
        resolution_roi: nextRes,
        lab_mode: nextLab,
        source: {
          ...state.source,
          presetModified: event.markPresetModified !== undefined
            ? Boolean(event.markPresetModified)
            : (state.source.originPreset ? true : state.source.presetModified),
          originPreset: event.originPreset !== undefined ? event.originPreset : state.source.originPreset,
        },
        runtime_health: {
          ...state.runtime_health,
          geometryEpoch: geomChanged ? state.runtime_health.geometryEpoch + 1 : state.runtime_health.geometryEpoch,
        },
        pending: event.trackPending
          ? {
              txId: nextRev,
              workerEpoch: state.runtime_health.workerEpoch,
              baseRevision: state.revision,
              nextRevision: nextRev,
              sourceEpoch: state.source.epoch,
            }
          : state.pending,
        undoConfig: prevUndo,
      });
      return nextState;
    }

    case 'UNDO_CONFIG': {
      if (!state.undoConfig) return state;
      const snap = state.undoConfig;
      const nextRev = state.revision + 1;
      const nextNative = snap.resolution_roi ? Boolean(snap.resolution_roi.native1to1) : state.resolution_roi.native1to1;
      const geomChanged = nextNative !== state.resolution_roi.native1to1;
      return withDerived({
        ...state,
        revision: nextRev,
        organ_config: {
          ...snap.organ_config,
          bosGauge: snap.organ_config.bosGauge.slice(),
          rsCal: { ...snap.organ_config.rsCal },
          depthCal: { ...snap.organ_config.depthCal },
          lockHz: snap.organ_config.lockHz.slice(),
        },
        overlays: { ...snap.overlays },
        resolution_roi: {
          ...state.resolution_roi,
          native1to1: nextNative,
        },
        lab_mode: snap.lab_mode,
        undoConfig: null,
        runtime_health: {
          ...state.runtime_health,
          geometryEpoch: geomChanged ? state.runtime_health.geometryEpoch + 1 : state.runtime_health.geometryEpoch,
        },
        pending: {
          txId: nextRev,
          workerEpoch: state.runtime_health.workerEpoch,
          baseRevision: state.revision,
          nextRevision: nextRev,
          sourceEpoch: state.source.epoch,
        },
      });
    }

    case 'CONFIG_ACK': {
      // Reject unapplied ACKs or stale ACKs from an older worker, source epoch, or revision.
      if (event.applied === false) return state;
      if (event.workerEpoch !== undefined && event.workerEpoch !== state.runtime_health.workerEpoch) {
        return state;
      }
      if (event.sourceEpoch !== undefined && event.sourceEpoch !== state.source.epoch) {
        return state;
      }
      const applied = clampInt(event.appliedRevision ?? state.revision, 0, 1e9, state.revision);
      if (applied < state.runtime_health.appliedRevision) return state;
      const pendingCleared = !state.pending || applied >= state.pending.nextRevision;
      return {
        ...state,
        runtime_health: {
          ...state.runtime_health,
          appliedRevision: applied,
          observedBosOwner: event.bosOwner || state.runtime_health.observedBosOwner,
        },
        pending: pendingCleared ? null : state.pending,
      };
    }

    case 'SOURCE_ACTIVATED': {
      const nextEpoch = state.source.epoch + 1;
      const kind = event.kind || 'TESTBENCH';
      const isMedia = kind === 'FILE' || kind === 'SAMPLE';
      return withDerived({
        ...state,
        revision: state.revision + 1,
        source: {
          kind,
          epoch: nextEpoch,
          lifecycle: kind === 'IDLE' ? 'IDLE' : 'READY',
          label: event.label || '',
          sampleUrl: event.sampleUrl || '',
          scenario: event.scenario || state.source.scenario,
          originPreset: event.originPreset !== undefined ? event.originPreset : null,
          presetModified: false,
        },
        transport: {
          kind: kind === 'IDLE' ? 'IDLE' : (isMedia ? 'MEDIA' : 'CONTINUOUS'),
          phase: kind === 'IDLE' ? 'STOPPED' : 'PLAYING',
          ptsUs: 0,
          durationUs: event.durationUs ?? null,
          fastPassRunning: false,
          liveCrop: 1.0,
        },
        pointer_tool: 'INSPECT_TARGET',
        pointer_binding: {
          sourceEpoch: nextEpoch,
          geometryEpoch: state.runtime_health.geometryEpoch + 1,
          generation: state.pointer_binding.generation + 1,
        },
        calibration: {
          committed: { kind: 'PX', provenance: 'NONE', label: 'PX UNITS' },
          segment: state.calibration.segment + 1,
          draft: null,
        },
        workspace_layout: {
          ...state.workspace_layout,
          welcomeVisible: kind === 'IDLE',
          pinnedTrack: null,
          inspectedTrack: null,
          dialog: 'NONE',
        },
        runtime_health: {
          ...state.runtime_health,
          geometryEpoch: state.runtime_health.geometryEpoch + 1,
          srcW: event.srcW || 0,
          srcH: event.srcH || 0,
          fps: event.fps || 0,
          seekable: isMedia,
          camMotionState: 0,
          panStreak: 0,
        },
      });
    }

    case 'CAL_COMMITTED': {
      // Stale calibration ACKs from a prior source epoch or pre-cut continuity epoch are strictly no-op.
      if (event.sourceEpoch !== undefined && event.sourceEpoch !== state.source.epoch) {
        return state;
      }
      if (event.continuityEpoch !== undefined && event.continuityEpoch !== state.runtime_health.continuityEpoch) {
        return state;
      }
      return {
        ...state,
        pointer_tool: 'INSPECT_TARGET',
        calibration: {
          committed: event.committed || { kind: 'PX', provenance: 'NONE', label: 'PX UNITS' },
          segment: state.calibration.segment + 1,
          draft: null,
        },
      };
    }

    case 'REC_STATE': {
      return {
        ...state,
        recorder: {
          active: event.active !== undefined ? Boolean(event.active) : state.recorder.active,
          frames: event.frames !== undefined ? Math.max(0, Math.trunc(Number(event.frames) || 0)) : state.recorder.frames,
          durationSec: event.durationSec !== undefined ? Math.max(0, Number(event.durationSec) || 0) : state.recorder.durationSec,
        },
      };
    }

    case 'TRANSPORT_STATE': {
      return {
        ...state,
        transport: {
          ...state.transport,
          ...(event.transport || {}),
        },
      };
    }

    case 'SCENE_CUT': {
      // Hard scene cut clears committed calibration AND any armed calibration draft,
      // incrementing the calibration segment exactly once per cut.
      const hadCalOrDraft = state.calibration.committed.kind !== 'PX' || state.calibration.draft !== null;
      return {
        ...state,
        pointer_tool: state.pointer_tool.startsWith('CAL_') ? 'INSPECT_TARGET' : state.pointer_tool,
        pointer_binding: state.pointer_tool.startsWith('CAL_')
          ? { ...state.pointer_binding, generation: state.pointer_binding.generation + 1 }
          : state.pointer_binding,
        calibration: {
          committed: { kind: 'PX', provenance: 'NONE', label: 'PX UNITS' },
          segment: hadCalOrDraft ? state.calibration.segment + 1 : state.calibration.segment,
          draft: null,
        },
        runtime_health: {
          ...state.runtime_health,
          continuityEpoch: state.runtime_health.continuityEpoch + 1,
          camMotionState: 2,
          panStreak: 0,
        },
      };
    }

    case 'RUNTIME_TELEMETRY': {
      if (event.workerEpoch !== undefined && event.workerEpoch < state.runtime_health.workerEpoch) {
        return state;
      }
      const patch = event.health || {};
      const nextW = patch.srcW !== undefined ? patch.srcW : state.runtime_health.srcW;
      const nextH = patch.srcH !== undefined ? patch.srcH : state.runtime_health.srcH;
      const engW = state.resolution_roi.resMode === 1 ? 640 : 320;
      const engH = state.resolution_roi.resMode === 1 ? 360 : 180;
      const nativeEligible = nextW >= engW && nextH >= engH && nextW > 0 && nextH > 0;
      const foveaEligible = nextW > engW && nextH > engH && !state.resolution_roi.native1to1;
      return {
        ...state,
        runtime_health: {
          ...state.runtime_health,
          ...patch,
          nativeEligible,
          foveaEligible,
          products: {
            ...state.runtime_health.products,
            ...(patch.products || {}),
          },
        },
      };
    }

    case 'AUDIO_STATE': {
      return {
        ...state,
        audio_pipeline: {
          ...state.audio_pipeline,
          ...(event.audio || {}),
        },
        runtime_health: {
          ...state.runtime_health,
          audioPresent: event.audioPresent !== undefined ? Boolean(event.audioPresent) : state.runtime_health.audioPresent,
          audioRoute: event.route !== undefined ? event.route : state.runtime_health.audioRoute,
        },
      };
    }

    case 'AUDIT_EVENT': {
      // Strictly isolated from HUD source, organs, audio, and recorder!
      const ab = { ...state.audit_bench };
      if (event.worker !== undefined) ab.worker = event.worker;
      if (event.workerEpoch !== undefined) ab.workerEpoch = event.workerEpoch;
      if (event.liveMirror !== undefined) ab.liveMirror = Boolean(event.liveMirror);
      if (event.slot === 'A' || event.slot === 'B') {
        const curSlot = ab[event.slot];
        if (event.jobId !== undefined && curSlot.jobId !== null && event.jobId < curSlot.jobId) {
          return state; // Drop stale audit completion
        }
        ab[event.slot] = {
          ...curSlot,
          ...(event.jobId !== undefined ? { jobId: event.jobId } : {}),
          ...(event.patch || {}),
        };
      }
      return { ...state, audit_bench: ab };
    }

    default:
      return state;
  }
}
