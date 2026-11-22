// ============================================================================
// CHRONOS-SPECTRAL Dedicated V8 Web Worker Isolate
// Manages Zero-Allocation Wasm SIMD128 Arena Views, Pre-Assignment Shifted
// Lanczos Fiedler Eigensolver, Multi-Scenario Blind Pixel Testbench, and
// OffscreenCanvas Non-Causal Proof HUD & Split Raw/Eulerian PiP.
// ============================================================================

const MAX_FRAME_W = 640;
const MAX_FRAME_H = 360;
const MAX_FRAME_BYTES = MAX_FRAME_W * MAX_FRAME_H * 4;
const RING_LEN = 12;
const MAX_TARGETS = 16;
const GRAPH_NODES = 32;
const TELEM_PEAKS = 928;
const TELEM_EXCUR = 992;
const TELEM_KINETIC = 1120;
const TELEM_GLOBAL = 1248;
const TELEM_PERSIST = 1264;
const TELEM_PHYSICS_START = 1280;
const TELEM_PHYSICS_STRIDE = 16;
const TELEM_ENSEMBLE_START = 1536;
const TELEMETRY_LEN = 1568;
const MAX_PIP_BYTES = 64 * 64 * 4;

let resMode = 0; // 0: 320x180 SD (32x32 patch/PiP), 1: 640x360 HD (64x64 foveated patch/PiP)
let frameW = 320;
let frameH = 180;
let framePixels = frameW * frameH;
let frameBytes = framePixels * 4;
let pipDim = 32;
let resScale = 1;

let wasmExports = null;
let wasmMemory = null;

// Pre-allocated permanent views into 448-page (28.00 MiB = 29,360,128 B) static Wasm linear memory
const MAX_WASM_PAGES = 448;
const ringSlotU8 = new Array(RING_LEN);
const ringSlotByteOffsets = new Int32Array(RING_LEN).fill(-1);
const ringSlotImgData = new Array(RING_LEN);
const ecaOverlayBackingBuf = new Uint8ClampedArray(MAX_FRAME_BYTES);
let ecaResView = null;
let pipImgData = null;
// (Shifted Lanczos Fiedler runs in-kernel via chronos_solve_fiedler_lanczos)
let fiedlerView = null;
let telemView = null;
let slotMetaView = null;
const slotPipImgData = new Array(MAX_TARGETS).fill(null);

// Swarm & Wave-1 ABI-v2 Constants
const MAX_MICRO = 3072;
const MICRO_STRIDE = 8; // [track_id, pos_x, pos_y, vel_x, vel_y, state, body_id, ftle_val]
const MAX_RIGID_BODIES = 32;
const BODY_STRIDE = 8; // [body_id, member_count, cx, cy, vx, vy, omega_rad_s, se2_rmse_px]
const FIELD_GRID_W = 64;
const FIELD_GRID_H = 36;
const FIELD_CELLS = FIELD_GRID_W * FIELD_GRID_H; // 2304
const FIELD_STRIDE = 6; // [u_px_s, v_px_s, vorticity_s1, divergence_s1, ftle_s1, valid_weight]
const SWARM_TELEM_LEN = 64;

const RING_META_STRIDE = 24;
const ROW_LUMA_H = 360;
const WAVE1_TELEM_LEN = 192;
const HERO_CAND_LEN = 4 + 16 * 8;
const BOS_W = 256;
const BOS_H = 128;
const BOS_N = BOS_W * BOS_H; // 32768
const BOS_TELEM_LEN = 32;
const BOS_REASON_NAMES = [
  'ANCHOR_SUP',
  'EXT_PAN_CUT',
  'INT_SHIFT',
  'PHOTOM_MSE',
  'DISCONT',
  'NO_GAUGE',
  'NONCONV',
  'LOW_TEX',
];
const BOS_STATE_LABELS = ['OFF', 'WARM', 'READY', 'DISTURBED'];
const GAP_FLAG_NAMES = ['SEQ_GAP', 'NONINC_PTS', 'EXCESS_DT', 'SEEK_RESTART', 'GEOM_CHANGE', 'SCENE_CUT', 'WRAP'];

let microTracksView = null;
let continuumFieldView = null;
let rigidBodiesView = null;
let modalFieldView = null;
let swarmTelemView = null;

// Wave-1 core ABI views
let ringMetaView = null;
let ringMetaF32View = null;
let rowLumaView = null;
let wave1TelemView = null;
let wave1TelemF32View = null;

// Wave-1 metrology & BOS optional ABI views (populated when merged exports exist)
let microCovView = null;
let microResidualView = null;
let microLoomView = null;
let microQualityView = null;
let heroCandidatesView = null;
let heroCandidatesF32View = null;
let bosTelemView = null;
let bosTelemF32View = null;
let bosPlanesView = null;
let bosValidView = null;
let bosRgbaView = null;

let wave1CoreAbiValid = false;
let wave1MetrologyAbiValid = false;
let wave1BosAbiValid = false;

// Wave-2 ABI-v2 constants & views (Vibrometry, Hodge 2-complex, Occluded Conservation)
const WAVE2_TELEM_LEN = 64;
const VIBRO_TELEM_LEN = 32;
const VIBRO_PLANES_LEN = 7040;
const AUDIO_STAGING_LEN = 4096;
const HODGE_TELEM_LEN = 32;
const HODGE_PLANES_LEN = 16128;
const HODGE_INJECT_LEN = 6912;
const CONSERVATION_LEN = 32;

const VIBRO_STATE_LABELS = ['OFF', 'UNCALIB', 'WARMING', 'READY', 'DISTURBED'];
const RS_REASON_NAMES = [
  'BAD_CALIB',
  'NO_SCAN_DIR',
  'NOT_NATIVE_1TO1',
  'FRAME_GAP',
  'CAMERA_MOTION',
  'CLOCK_SKEW',
  'EXPOSURE_NULL',
  'LOW_GRAD',
  'AUDIO_UNDERRUN',
  'AXIS_TRANSPOSED',
  'NO_PTS',
];
const HODGE_STATE_LABELS = ['OFF', 'STANDBY', 'READY', 'UNCONV'];
const OCCL_STATUS_LABELS = ['NONE', 'UNIQUE', 'AMBIGUOUS', 'VIOLATED'];
const OCCL_HYP_NAMES = ['H0_KEEP', 'H0_SWAP', 'H1_KEEP', 'H1_SWAP'];
const DEPTH_TELEM_LEN = 32;
const DEPTH_FIELD_LEN = 64 * 36 * 2;
const DEPTH_TRACKS_LEN = 3072 * 2;
const DEPTH_HERO_LEN = 16 * 4;
const DEPTH_INGEST_LEN = 128 * 64;
const DEPTH_STATE_LABELS = [
  'OFF',
  'WARMING',
  'READY_PARALLAX',
  'READY_TTC',
  'REFUSED_ROTATION',
  'REFUSED_NO_BASELINE',
  'READY_STEREO',
  'READY_EXTERNAL',
];
const DEPTH_PROV_LABELS = [
  'NONE',
  'MONO_REL',
  'MONO_METRIC',
  'TTC_FREE',
  'SURVEYED_STEREO',
  'SURVEYED_LIDAR',
  'ASSUMED_NEURAL_PRIOR',
];

let wave2TelemView = null;
let wave2TelemF32View = null;
let vibroTelemView = null;
let vibroPlanesView = null;
let audioStagingView = null;
let hodgeTelemView = null;
let hodgePlanesView = null;
let hodgeInjectView = null;
let conservationTelemView = null;
let conservationInputView = null;

let wave2CoreAbiValid = false;
let wave2VibroAbiValid = false;
let wave2HodgeAbiValid = false;
let wave2ConsAbiValid = false;
let depthAbiValid = false;
let depthTelemView = null;
let depthTelemF32View = null;
let depthFieldView = null;
let depthTracksView = null;
let depthTrackFlagsView = null;
let depthHeroView = null;
let depthIngestView = null;
let depthReqMode = 0;

// Wave D2 (Pocket Wind Tunnel) & Wave D1 (kHz Optical Elastography & Lock-In Acoustic BOS) ABI-v3
const WIND_TELEM_LEN = 32;
const WIND_PLANES_LEN = 3 * FIELD_CELLS; // [Cp, p/rho, omega_filt]
const WIND_STATE_LABELS = ['OFF', 'WARMING', 'READY_FIELD', 'READY_LOADS', 'REFUSED'];
const WIND_REASON_NAMES = [
  'NO_CONTINUUM',
  'HODGE_UNCONV',
  'MUTEX_CONFLICT',
  'HIGH_DIV',
  'NONCONSERV_FORCING',
  'NO_STEADY_FREESTREAM',
  'NO_VOID_FOR_LOADS',
  'LOW_PRESSURE_QUALITY',
];
let windAbiValid = false;
let windTelemView = null;
let windTelemF32View = null;
let windPlanesView = null;
let windReqMode = 0;
let windFluidDensity = 0.0;

const ELASTO_TELEM_LEN = 32;
const ELASTO_SURF_PLANES_LEN = 6 * FIELD_CELLS; // [c, coh, div, amp, phase, phi]
const ELASTO_STATE_LABELS = ['OFF', 'UNCALIB', 'WARMING', 'READY', 'REFUSED'];
const ELASTO_CLASS_LABELS = ['NONE', 'KIRCHHOFF_PLATE', 'MEMBRANE_SHEAR', 'DISPERSIVE'];
let elastoAbiValid = false;
let elastoTelemView = null;
let elastoTelemF32View = null;
let elastoPlanesView = null;
let elastoValidView = null;
let elastoReqMode = 0;
let elastoLockFreqs = [];

// Dual-Resolution 1:1 Fovea ABI views (160x160 lazy-chroma tile)
const FOVEA_DIM = 160;
let foveaRgbaView = null;
let foveaReqView = null;
let foveaAbiValid = false;
let foveaStageCanvas = null;
let foveaStageCtx = null;
let roiBoxPreview = null; // { x0, y0, x1, y1 } normalized [0..1]
let autoTrackRoiEnabled = false;
let roiSlewPending = false;
let workerFastPassCancel = false;
let workerFastPassRunning = false;

const rsCalibConfig = {
  mode: 0,
  dtRowUs: 0.0,
  exposureUs: 0.0,
  audioRateHz: 48000.0,
  maxSkewUs: 500.0,
};
let hodgeReqMode = 0;
let manualConsTestActive = false;
const w2ScratchU32 = new Uint32Array(1);
const w2ScratchF32 = new Float32Array(w2ScratchU32.buffer);

function u32BitsToF32(u) {
  w2ScratchU32[0] = u >>> 0;
  return w2ScratchF32[0];
}

let nativeRoiEnabled = false;
let roiOffsetNormX = 0.5;
let roiOffsetNormY = 0.5;
let roiUseExplicitPx = false;
let roiExplicitX0 = 0;
let roiExplicitY0 = 0;
let roiOrientation = 0;
let roiScanDir = 0xffffffff;
let workerExpectedEngineFrame = 0;
const rgbaStageScratch = new Uint8Array(640 * 360 * 4);
let lastIngestGeom = {
  srcW: 320,
  srcH: 180,
  cropX0: 0,
  cropY0: 0,
  cropW: 320,
  cropH: 180,
  orientation: 0,
  scanDir: 0xffffffff,
  engineW: 320,
  engineH: 180,
  dstX: 0,
  dstY: 0,
  dstW: 320,
  dstH: 180,
  native1to1: 0,
  fallbackScale: false,
};
let microMetrologyMode = 0;
let showCovEllipses = false;
let bosReqMode = 0;
let bosOverlayKind = 0; // 0=|eps|, 1=phi, 2=curl
let bosGaugeRoi = [0, 0, 0, 0];
let workerLocalEpoch = 1;
let workerLocalSeq = 0;
let workerSimTimeSec = 0.0;

let userSwarmMode = 0; // Persistent user selection: 0=off, 1=auto, 2=forced
let swarmMode = 0; // Active mode applied to Wasm (includes dense scenario/sample overrides)
let denseSwarmOverrideActive = false;
let chronos1000AbiValid = false;
let fieldOverlayMode = 0; // 0=off, 1=velocity, 2=vorticity, 3=ftle, 4=modal, 5=bos_|eps|, 6=bos_phi, 7=bos_curl
let inspectedMicroTid = 0;
let lastInspectStatus = 'idle';
let lastInspectReqSeq = 0;
let lastInspectResSeq = 0;
let lastInspectReqX = null;
let lastInspectReqY = null;
let lastInspectReqNormX = null;
let lastInspectReqNormY = null;

function clearActiveInspectionSelection() {
  inspectedMicroTid = 0;
  lastInspectStatus = 'idle';
  lastInspectReqX = null;
  lastInspectReqY = null;
  lastInspectReqNormX = null;
  lastInspectReqNormY = null;
}
let lastFieldRenderMs = 0.0;
let lastSwarmCopyMs = 0.0;
let lastFieldBackend = 'none';
let lastDtSecUsed = 1.0 / 30.0;
let ingestElapsedMs = 0.0;
let latestCausalTimeMs = 0.0;
let latestCausalPtsMs = -1.0;

function clearWave1Views() {
  wave1CoreAbiValid = false;
  wave1MetrologyAbiValid = false;
  wave1BosAbiValid = false;
  ringMetaView = null;
  ringMetaF32View = null;
  rowLumaView = null;
  wave1TelemView = null;
  wave1TelemF32View = null;
  microCovView = null;
  microResidualView = null;
  microLoomView = null;
  microQualityView = null;
  heroCandidatesView = null;
  heroCandidatesF32View = null;
  bosTelemView = null;
  bosTelemF32View = null;
  bosPlanesView = null;
  bosValidView = null;
  bosRgbaView = null;
  clearWave2Views();
}

function clearWave2Views() {
  wave2CoreAbiValid = false;
  wave2VibroAbiValid = false;
  wave2HodgeAbiValid = false;
  wave2ConsAbiValid = false;
  wave2TelemView = null;
  wave2TelemF32View = null;
  vibroTelemView = null;
  vibroPlanesView = null;
  audioStagingView = null;
  hodgeTelemView = null;
  hodgePlanesView = null;
  hodgeInjectView = null;
  conservationTelemView = null;
  conservationInputView = null;
  foveaRgbaView = null;
  foveaReqView = null;
  foveaAbiValid = false;
  depthAbiValid = false;
  depthTelemView = null;
  depthTelemF32View = null;
  depthFieldView = null;
  depthTracksView = null;
  depthTrackFlagsView = null;
  depthHeroView = null;
  depthIngestView = null;
  windAbiValid = false;
  windTelemView = null;
  windTelemF32View = null;
  windPlanesView = null;
  elastoAbiValid = false;
  elastoTelemView = null;
  elastoTelemF32View = null;
  elastoPlanesView = null;
  elastoValidView = null;
}

function refreshWave2Views() {
  clearWave2Views();
  if (!wasmMemory || !wasmExports) return false;
  const buf = wasmMemory.buffer;
  const bufLen = buf.byteLength;

  if (typeof wasmExports.chronos_get_wave2_telemetry_ptr === 'function') {
    const w2Ptr = wasmExports.chronos_get_wave2_telemetry_ptr();
    if ((w2Ptr & 3) === 0 && validatePtrSpan(bufLen, w2Ptr, WAVE2_TELEM_LEN * 4)) {
      wave2TelemView = new Uint32Array(buf, w2Ptr, WAVE2_TELEM_LEN);
      wave2TelemF32View = new Float32Array(buf, w2Ptr, WAVE2_TELEM_LEN);
      wave2CoreAbiValid = true;
    }
  }

  if (
    typeof wasmExports.chronos_set_rs_calib === 'function' &&
    typeof wasmExports.chronos_get_audio_staging_ptr === 'function' &&
    typeof wasmExports.chronos_push_audio_block === 'function' &&
    typeof wasmExports.chronos_get_vibro_telemetry_ptr === 'function' &&
    typeof wasmExports.chronos_get_vibro_planes_ptr === 'function'
  ) {
    const asPtr = wasmExports.chronos_get_audio_staging_ptr();
    const vtPtr = wasmExports.chronos_get_vibro_telemetry_ptr();
    const vpPtr = wasmExports.chronos_get_vibro_planes_ptr();
    if (
      (asPtr & 3) === 0 && validatePtrSpan(bufLen, asPtr, AUDIO_STAGING_LEN * 4) &&
      (vtPtr & 3) === 0 && validatePtrSpan(bufLen, vtPtr, VIBRO_TELEM_LEN * 4) &&
      (vpPtr & 3) === 0 && validatePtrSpan(bufLen, vpPtr, VIBRO_PLANES_LEN * 4)
    ) {
      audioStagingView = new Float32Array(buf, asPtr, AUDIO_STAGING_LEN);
      vibroTelemView = new Uint32Array(buf, vtPtr, VIBRO_TELEM_LEN);
      vibroPlanesView = new Float32Array(buf, vpPtr, VIBRO_PLANES_LEN);
      wave2VibroAbiValid = true;
      if (rsCalibConfig.mode > 0) {
        wasmExports.chronos_set_rs_calib(
          rsCalibConfig.mode,
          rsCalibConfig.dtRowUs,
          rsCalibConfig.exposureUs,
          rsCalibConfig.audioRateHz,
          rsCalibConfig.maxSkewUs
        );
      }
    }
  }

  if (
    typeof wasmExports.chronos_set_hodge_mode === 'function' &&
    typeof wasmExports.chronos_get_hodge_telemetry_ptr === 'function' &&
    typeof wasmExports.chronos_get_hodge_planes_ptr === 'function' &&
    typeof wasmExports.chronos_get_hodge_inject_ptr === 'function' &&
    typeof wasmExports.chronos_hodge_solve_injected === 'function'
  ) {
    const htPtr = wasmExports.chronos_get_hodge_telemetry_ptr();
    const hpPtr = wasmExports.chronos_get_hodge_planes_ptr();
    const hiPtr = wasmExports.chronos_get_hodge_inject_ptr();
    if (
      (htPtr & 3) === 0 && validatePtrSpan(bufLen, htPtr, HODGE_TELEM_LEN * 4) &&
      (hpPtr & 3) === 0 && validatePtrSpan(bufLen, hpPtr, HODGE_PLANES_LEN * 4) &&
      (hiPtr & 3) === 0 && validatePtrSpan(bufLen, hiPtr, HODGE_INJECT_LEN * 4)
    ) {
      hodgeTelemView = new Uint32Array(buf, htPtr, HODGE_TELEM_LEN);
      hodgePlanesView = new Float32Array(buf, hpPtr, HODGE_PLANES_LEN);
      hodgeInjectView = new Float32Array(buf, hiPtr, HODGE_INJECT_LEN);
      wave2HodgeAbiValid = true;
      wasmExports.chronos_set_hodge_mode(hodgeReqMode);
    }
  }

  if (
    typeof wasmExports.chronos_get_conservation_telemetry_ptr === 'function' &&
    typeof wasmExports.chronos_get_conservation_input_ptr === 'function' &&
    typeof wasmExports.chronos_eval_occluded_pair === 'function'
  ) {
    const ctPtr = wasmExports.chronos_get_conservation_telemetry_ptr();
    const ciPtr = wasmExports.chronos_get_conservation_input_ptr();
    if (
      (ctPtr & 3) === 0 && validatePtrSpan(bufLen, ctPtr, CONSERVATION_LEN * 4) &&
      (ciPtr & 3) === 0 && validatePtrSpan(bufLen, ciPtr, CONSERVATION_LEN * 4)
    ) {
      conservationTelemView = new Float32Array(buf, ctPtr, CONSERVATION_LEN);
      conservationInputView = new Float32Array(buf, ciPtr, CONSERVATION_LEN);
      wave2ConsAbiValid = true;
    }
  }

  if (
    typeof wasmExports.chronos_get_fovea_rgba_ptr === 'function' &&
    typeof wasmExports.chronos_set_fovea_meta === 'function' &&
    typeof wasmExports.chronos_get_fovea_request_ptr === 'function'
  ) {
    const fPtr = wasmExports.chronos_get_fovea_rgba_ptr(0);
    const rPtr = wasmExports.chronos_get_fovea_request_ptr();
    if (
      validatePtrSpan(bufLen, fPtr, FOVEA_DIM * FOVEA_DIM * 4) &&
      (rPtr & 3) === 0 && validatePtrSpan(bufLen, rPtr, 8 * 4)
    ) {
      foveaRgbaView = new Uint8Array(buf, fPtr, FOVEA_DIM * FOVEA_DIM * 4);
      foveaReqView = new Uint32Array(buf, rPtr, 8);
      foveaAbiValid = true;
    }
  }

  if (
    typeof wasmExports.chronos_set_depth_mode === 'function' &&
    typeof wasmExports.chronos_get_depth_telemetry_ptr === 'function' &&
    typeof wasmExports.chronos_get_depth_field_ptr === 'function' &&
    typeof wasmExports.chronos_get_depth_tracks_ptr === 'function' &&
    typeof wasmExports.chronos_get_depth_track_flags_ptr === 'function' &&
    typeof wasmExports.chronos_get_depth_hero_ptr === 'function' &&
    typeof wasmExports.chronos_get_depth_ingest_ptr === 'function'
  ) {
    const dtP = wasmExports.chronos_get_depth_telemetry_ptr();
    const dfP = wasmExports.chronos_get_depth_field_ptr();
    const dkP = wasmExports.chronos_get_depth_tracks_ptr();
    const flP = wasmExports.chronos_get_depth_track_flags_ptr();
    const dhP = wasmExports.chronos_get_depth_hero_ptr();
    const diP = wasmExports.chronos_get_depth_ingest_ptr();
    if (
      (dtP & 3) === 0 && validatePtrSpan(bufLen, dtP, DEPTH_TELEM_LEN * 4) &&
      (dfP & 3) === 0 && validatePtrSpan(bufLen, dfP, DEPTH_FIELD_LEN * 4) &&
      (dkP & 3) === 0 && validatePtrSpan(bufLen, dkP, DEPTH_TRACKS_LEN * 4) &&
      validatePtrSpan(bufLen, flP, 3072) &&
      (dhP & 3) === 0 && validatePtrSpan(bufLen, dhP, DEPTH_HERO_LEN * 4) &&
      (diP & 3) === 0 && validatePtrSpan(bufLen, diP, DEPTH_INGEST_LEN * 4)
    ) {
      depthTelemView = new Uint32Array(buf, dtP, DEPTH_TELEM_LEN);
      depthTelemF32View = new Float32Array(buf, dtP, DEPTH_TELEM_LEN);
      depthFieldView = new Float32Array(buf, dfP, DEPTH_FIELD_LEN);
      depthTracksView = new Float32Array(buf, dkP, DEPTH_TRACKS_LEN);
      depthTrackFlagsView = new Uint8Array(buf, flP, 3072);
      depthHeroView = new Float32Array(buf, dhP, DEPTH_HERO_LEN);
      depthIngestView = new Float32Array(buf, diP, DEPTH_INGEST_LEN);
      depthAbiValid = true;
      wasmExports.chronos_set_depth_mode(depthReqMode);
    }
  }

  if (
    typeof wasmExports.chronos_set_wind_mode === 'function' &&
    typeof wasmExports.chronos_set_wind_fluid_density === 'function' &&
    typeof wasmExports.chronos_get_wind_telemetry_ptr === 'function' &&
    typeof wasmExports.chronos_get_wind_planes_ptr === 'function'
  ) {
    const wtP = wasmExports.chronos_get_wind_telemetry_ptr();
    const wpP = wasmExports.chronos_get_wind_planes_ptr();
    if (
      (wtP & 3) === 0 && validatePtrSpan(bufLen, wtP, WIND_TELEM_LEN * 4) &&
      (wpP & 3) === 0 && validatePtrSpan(bufLen, wpP, WIND_PLANES_LEN * 4)
    ) {
      windTelemView = new Uint32Array(buf, wtP, WIND_TELEM_LEN);
      windTelemF32View = new Float32Array(buf, wtP, WIND_TELEM_LEN);
      windPlanesView = new Float32Array(buf, wpP, WIND_PLANES_LEN);
      windAbiValid = true;
      wasmExports.chronos_set_wind_mode(windReqMode);
      wasmExports.chronos_set_wind_fluid_density(windFluidDensity);
    }
  }

  if (
    typeof wasmExports.chronos_set_elasto_mode === 'function' &&
    typeof wasmExports.chronos_set_elasto_lock_freqs === 'function' &&
    typeof wasmExports.chronos_get_elasto_telemetry_ptr === 'function' &&
    typeof wasmExports.chronos_get_elasto_planes_ptr === 'function' &&
    typeof wasmExports.chronos_get_elasto_valid_ptr === 'function'
  ) {
    const etP = wasmExports.chronos_get_elasto_telemetry_ptr();
    const epP = wasmExports.chronos_get_elasto_planes_ptr();
    const evP = wasmExports.chronos_get_elasto_valid_ptr();
    if (
      (etP & 3) === 0 && validatePtrSpan(bufLen, etP, ELASTO_TELEM_LEN * 4) &&
      (epP & 3) === 0 && validatePtrSpan(bufLen, epP, 6 * BOS_N * 4) &&
      validatePtrSpan(bufLen, evP, BOS_N)
    ) {
      elastoTelemView = new Uint32Array(buf, etP, ELASTO_TELEM_LEN);
      elastoTelemF32View = new Float32Array(buf, etP, ELASTO_TELEM_LEN);
      elastoPlanesView = new Float32Array(buf, epP, 6 * BOS_N);
      elastoValidView = new Uint8Array(buf, evP, BOS_N);
      elastoAbiValid = true;
      wasmExports.chronos_set_elasto_mode(elastoReqMode);
      if (elastoLockFreqs.length > 0) {
        wasmExports.chronos_set_elasto_lock_freqs(
          elastoLockFreqs.length,
          elastoLockFreqs[0] || 0,
          elastoLockFreqs[1] || 0,
          elastoLockFreqs[2] || 0,
          elastoLockFreqs[3] || 0
        );
      } else {
        wasmExports.chronos_set_elasto_lock_freqs(0, 0, 0, 0, 0);
      }
    }
  }
  return wave2CoreAbiValid;
}

function validatePtrSpan(bufLen, ptr, byteLen) {
  return Number.isInteger(ptr) && ptr > 0 && ptr + byteLen <= bufLen;
}

function refreshWave1Views() {
  clearWave1Views();
  if (!wasmMemory || !wasmExports) return false;
  const buf = wasmMemory.buffer;
  const bufLen = buf.byteLength;

  if (
    typeof wasmExports.chronos_set_ingest_meta === 'function' &&
    typeof wasmExports.chronos_get_ring_meta_ptr === 'function' &&
    typeof wasmExports.chronos_get_row_luma_ptr === 'function' &&
    typeof wasmExports.chronos_get_wave1_telemetry_ptr === 'function'
  ) {
    const rmPtr = wasmExports.chronos_get_ring_meta_ptr();
    const rlPtr = wasmExports.chronos_get_row_luma_ptr();
    const w1Ptr = wasmExports.chronos_get_wave1_telemetry_ptr();
    if (
      (rmPtr & 3) === 0 && validatePtrSpan(bufLen, rmPtr, RING_LEN * RING_META_STRIDE * 4) &&
      (rlPtr & 3) === 0 && validatePtrSpan(bufLen, rlPtr, RING_LEN * ROW_LUMA_H * 4) &&
      (w1Ptr & 3) === 0 && validatePtrSpan(bufLen, w1Ptr, WAVE1_TELEM_LEN * 4)
    ) {
      ringMetaView = new Uint32Array(buf, rmPtr, RING_LEN * RING_META_STRIDE);
      ringMetaF32View = new Float32Array(buf, rmPtr, RING_LEN * RING_META_STRIDE);
      rowLumaView = new Float32Array(buf, rlPtr, RING_LEN * ROW_LUMA_H);
      wave1TelemView = new Uint32Array(buf, w1Ptr, WAVE1_TELEM_LEN);
      wave1TelemF32View = new Float32Array(buf, w1Ptr, WAVE1_TELEM_LEN);
      wave1CoreAbiValid = true;
    }
  }

  if (
    typeof wasmExports.chronos_set_micro_metrology_mode === 'function' &&
    typeof wasmExports.chronos_get_micro_cov_ptr === 'function' &&
    typeof wasmExports.chronos_get_micro_residual_ptr === 'function' &&
    typeof wasmExports.chronos_get_micro_loom_ptr === 'function' &&
    typeof wasmExports.chronos_get_micro_quality_ptr === 'function' &&
    typeof wasmExports.chronos_get_hero_candidates_ptr === 'function'
  ) {
    const covP = wasmExports.chronos_get_micro_cov_ptr();
    const resP = wasmExports.chronos_get_micro_residual_ptr();
    const loomP = wasmExports.chronos_get_micro_loom_ptr();
    const qualP = wasmExports.chronos_get_micro_quality_ptr();
    const candP = wasmExports.chronos_get_hero_candidates_ptr();
    if (
      (covP & 3) === 0 && validatePtrSpan(bufLen, covP, 3 * MAX_MICRO * 4) &&
      (resP & 3) === 0 && validatePtrSpan(bufLen, resP, 3 * MAX_MICRO * 4) &&
      (loomP & 3) === 0 && validatePtrSpan(bufLen, loomP, MAX_MICRO * 4) &&
      (qualP & 3) === 0 && validatePtrSpan(bufLen, qualP, MAX_MICRO * 4) &&
      (candP & 3) === 0 && validatePtrSpan(bufLen, candP, HERO_CAND_LEN * 4)
    ) {
      microCovView = new Float32Array(buf, covP, 3 * MAX_MICRO);
      microResidualView = new Float32Array(buf, resP, 3 * MAX_MICRO);
      microLoomView = new Float32Array(buf, loomP, MAX_MICRO);
      microQualityView = new Uint32Array(buf, qualP, MAX_MICRO);
      heroCandidatesView = new Uint32Array(buf, candP, HERO_CAND_LEN);
      heroCandidatesF32View = new Float32Array(buf, candP, HERO_CAND_LEN);
      wave1MetrologyAbiValid = true;
      wasmExports.chronos_set_micro_metrology_mode(microMetrologyMode);
    }
  }

  if (
    typeof wasmExports.chronos_set_bos_mode === 'function' &&
    typeof wasmExports.chronos_get_bos_telemetry_ptr === 'function' &&
    typeof wasmExports.chronos_get_bos_planes_ptr === 'function' &&
    typeof wasmExports.chronos_get_bos_valid_ptr === 'function' &&
    typeof wasmExports.chronos_get_bos_rgba_ptr === 'function'
  ) {
    const btP = wasmExports.chronos_get_bos_telemetry_ptr();
    const bpP = wasmExports.chronos_get_bos_planes_ptr();
    const bvP = wasmExports.chronos_get_bos_valid_ptr();
    const brP = wasmExports.chronos_get_bos_rgba_ptr();
    if (
      (btP & 3) === 0 && validatePtrSpan(bufLen, btP, BOS_TELEM_LEN * 4) &&
      (bpP & 3) === 0 && validatePtrSpan(bufLen, bpP, 6 * BOS_N * 4) &&
      validatePtrSpan(bufLen, bvP, BOS_N) &&
      validatePtrSpan(bufLen, brP, BOS_N * 4)
    ) {
      bosTelemView = new Uint32Array(buf, btP, BOS_TELEM_LEN);
      bosTelemF32View = new Float32Array(buf, btP, BOS_TELEM_LEN);
      bosPlanesView = new Float32Array(buf, bpP, 6 * BOS_N);
      bosValidView = new Uint8Array(buf, bvP, BOS_N);
      bosRgbaView = new Uint8ClampedArray(buf, brP, BOS_N * 4);
      wave1BosAbiValid = true;
      wasmExports.chronos_set_bos_mode(bosReqMode);
      if (typeof wasmExports.chronos_set_bos_overlay_kind === 'function') {
        wasmExports.chronos_set_bos_overlay_kind(bosOverlayKind);
      }
      if (typeof wasmExports.chronos_set_bos_gauge_roi === 'function') {
        wasmExports.chronos_set_bos_gauge_roi(
          bosGaugeRoi[0],
          bosGaugeRoi[1],
          bosGaugeRoi[2],
          bosGaugeRoi[3]
        );
      }
    }
  }
  refreshWave2Views();
  return wave1CoreAbiValid;
}

function hasChronos1000Exports() {
  return Boolean(
    chronos1000AbiValid &&
    wasmExports &&
    typeof wasmExports.chronos_get_micro_tracks_ptr === 'function' &&
    typeof wasmExports.chronos_get_continuum_field_ptr === 'function' &&
    typeof wasmExports.chronos_get_modal_field_ptr === 'function' &&
    typeof wasmExports.chronos_get_rigid_bodies_ptr === 'function' &&
    typeof wasmExports.chronos_get_swarm_telemetry_ptr === 'function' &&
    typeof wasmExports.chronos_set_swarm_mode === 'function' &&
    typeof wasmExports.chronos_get_swarm_mode === 'function' &&
    typeof wasmExports.chronos_inspect_field_point === 'function' &&
    typeof wasmExports.chronos_set_focus_target === 'function'
  );
}

function clearChronos1000Views() {
  chronos1000AbiValid = false;
  microTracksView = null;
  continuumFieldView = null;
  rigidBodiesView = null;
  modalFieldView = null;
  swarmTelemView = null;
}

function refreshChronos1000Views() {
  if (
    !wasmMemory ||
    !wasmExports ||
    typeof wasmExports.chronos_get_micro_tracks_ptr !== 'function' ||
    typeof wasmExports.chronos_get_continuum_field_ptr !== 'function' ||
    typeof wasmExports.chronos_get_modal_field_ptr !== 'function' ||
    typeof wasmExports.chronos_get_rigid_bodies_ptr !== 'function' ||
    typeof wasmExports.chronos_get_swarm_telemetry_ptr !== 'function' ||
    typeof wasmExports.chronos_set_swarm_mode !== 'function' ||
    typeof wasmExports.chronos_get_swarm_mode !== 'function' ||
    typeof wasmExports.chronos_inspect_field_point !== 'function' ||
    typeof wasmExports.chronos_set_focus_target !== 'function'
  ) {
    clearChronos1000Views();
    return false;
  }
  const buf = wasmMemory.buffer;
  const specs = [
    { ptr: wasmExports.chronos_get_micro_tracks_ptr(), len: MAX_MICRO * MICRO_STRIDE },
    { ptr: wasmExports.chronos_get_continuum_field_ptr(), len: FIELD_CELLS * FIELD_STRIDE },
    { ptr: wasmExports.chronos_get_modal_field_ptr(), len: FIELD_CELLS },
    { ptr: wasmExports.chronos_get_rigid_bodies_ptr(), len: MAX_RIGID_BODIES * BODY_STRIDE },
    { ptr: wasmExports.chronos_get_swarm_telemetry_ptr(), len: SWARM_TELEM_LEN },
  ];
  for (let i = 0; i < specs.length; i++) {
    const p = specs[i].ptr;
    const byteLen = specs[i].len * 4;
    if (!Number.isInteger(p) || p <= 0 || (p & 3) !== 0 || p + byteLen > buf.byteLength) {
      clearChronos1000Views();
      return false;
    }
    for (let j = i + 1; j < specs.length; j++) {
      const q = specs[j].ptr;
      const qByteLen = specs[j].len * 4;
      if (p < q + qByteLen && q < p + byteLen) {
        clearChronos1000Views();
        return false;
      }
    }
  }
  microTracksView = new Float32Array(buf, specs[0].ptr, specs[0].len);
  continuumFieldView = new Float32Array(buf, specs[1].ptr, specs[1].len);
  modalFieldView = new Float32Array(buf, specs[2].ptr, specs[2].len);
  rigidBodiesView = new Float32Array(buf, specs[3].ptr, specs[3].len);
  swarmTelemView = new Float32Array(buf, specs[4].ptr, specs[4].len);
  chronos1000AbiValid = true;
  wasmExports.chronos_set_swarm_mode(swarmMode);
  return true;
}

function isChronos1000Scenario(sc) {
  return sc === 'vortex_swarm_1200' || sc === 'murmuration_1200' || sc === 'modal_beam_1152';
}

// Pre-allocated ring-aligned snapshot buffers so t-effK HUD renders stay locked to dispSlot
const swarmRingValid = new Uint8Array(RING_LEN);
const ringIngestFrameId = new Uint32Array(RING_LEN);
const ringIngestFrameValid = new Uint8Array(RING_LEN);
const ringIngestTimeMs = new Float64Array(RING_LEN).fill(-1.0);
const ringIngestPtsMs = new Float64Array(RING_LEN).fill(-1.0);
const swarmRingMicro = Array.from({ length: RING_LEN }, () => new Float32Array(MAX_MICRO * MICRO_STRIDE));
const swarmRingField = Array.from({ length: RING_LEN }, () => new Float32Array(FIELD_CELLS * FIELD_STRIDE));
const swarmRingBodies = Array.from({ length: RING_LEN }, () => new Float32Array(MAX_RIGID_BODIES * BODY_STRIDE));
const swarmRingModal = Array.from({ length: RING_LEN }, () => new Float32Array(FIELD_CELLS));
const swarmRingTelem = Array.from({ length: RING_LEN }, () => new Float32Array(SWARM_TELEM_LEN));

// Wave-1 worker-resident delayed rings (zero 640x360 Wasm overlay allocations)
const covRingBuf = Array.from({ length: RING_LEN }, () => new Float32Array(MAX_MICRO * 3));
const qualRingBuf = Array.from({ length: RING_LEN }, () => new Uint32Array(MAX_MICRO));
const covRingValid = new Uint8Array(RING_LEN);
const bosRingRgba = Array.from({ length: RING_LEN }, () => new Uint8ClampedArray(BOS_N * 4));
const bosRingTelem = Array.from({ length: RING_LEN }, () => new Float64Array(20));
const bosRingValid = new Uint8Array(RING_LEN);
let bosOverlayCanvas = null;
let bosOverlayCtx = null;
let bosOverlayImgData = null;

// Wave-2 64x36 overlay canvas & delayed-display ring buffers
const w2CoreRing = Array.from({ length: RING_LEN }, () => new Uint32Array(52));
const w2CoreRingF32 = w2CoreRing.map((u) => new Float32Array(u.buffer));
const w2CoreRingValid = new Uint8Array(RING_LEN);
const vibroRingTelem = Array.from({ length: RING_LEN }, () => new Uint32Array(VIBRO_TELEM_LEN));
const vibroRingPlanes = Array.from({ length: RING_LEN }, () => new Float32Array(VIBRO_PLANES_LEN));
const vibroRingValid = new Uint8Array(RING_LEN);
const hodgeRingTelem = Array.from({ length: RING_LEN }, () => new Uint32Array(HODGE_TELEM_LEN));
const hodgeRingPlanes = Array.from({ length: RING_LEN }, () => new Float32Array(HODGE_PLANES_LEN));
const hodgeRingValid = new Uint8Array(RING_LEN);
const consRingTelem = Array.from({ length: RING_LEN }, () => new Float32Array(CONSERVATION_LEN));
const consRingValid = new Uint8Array(RING_LEN);
const depthRingTelem = Array.from({ length: RING_LEN }, () => new Uint32Array(DEPTH_TELEM_LEN));
const depthRingTelemF32 = depthRingTelem.map((u) => new Float32Array(u.buffer));
const depthRingField = Array.from({ length: RING_LEN }, () => new Float32Array(DEPTH_FIELD_LEN));
const depthRingStereoRgba = Array.from({ length: RING_LEN }, () => new Uint8ClampedArray(BOS_N * 4));
const depthRingValid = new Uint8Array(RING_LEN);
const windRingTelem = Array.from({ length: RING_LEN }, () => new Uint32Array(WIND_TELEM_LEN));
const windRingTelemF32 = windRingTelem.map((u) => new Float32Array(u.buffer));
const windRingPlanes = Array.from({ length: RING_LEN }, () => new Float32Array(WIND_PLANES_LEN));
const windRingValid = new Uint8Array(RING_LEN);
const elastoRingTelem = Array.from({ length: RING_LEN }, () => new Uint32Array(ELASTO_TELEM_LEN));
const elastoRingTelemF32 = elastoRingTelem.map((u) => new Float32Array(u.buffer));
const elastoRingPlanes = Array.from({ length: RING_LEN }, () => new Float32Array(ELASTO_SURF_PLANES_LEN));
const elastoRingValidMask = Array.from({ length: RING_LEN }, () => new Uint8Array(FIELD_CELLS));
const elastoRingAbosRgba = Array.from({ length: RING_LEN }, () => new Uint8ClampedArray(BOS_N * 4));
const elastoRingValid = new Uint8Array(RING_LEN);
let w2OverlayCanvas = null;
let w2OverlayCtx = null;
let w2OverlayImgData = null;
let tbRsFrameSeq = 0;
let tbRsAudioSampleCursor = 0;
let tbRsWasActive = false;
let tbRsTexW = 0;
let tbRsTexH = 0;
let tbRsSinA = null;
let tbRsCosA = null;
let tbRsCosB = null;
let tbRsSinB = null;
let tbRsSinC = null;
let tbRsCosC = null;
let tbRsFrameBuf = null;

// OffscreenCanvases & 2D contexts (pre-allocated once at boot)
let hudCanvas = null;
let hudCtx = null;
let pipCanvas = null;
let pipCtx = null;
let hilbertCanvas = null;
let hilbertCtx = null;
let spectralCanvas = null;
let spectralCtx = null;
let persistenceCanvas = null;
let persistenceCtx = null;
let physicsPhaseCanvas = null;
let physicsPhaseCtx = null;
const slotPipCanvases = new Array(MAX_TARGETS).fill(null);
const slotPipCtxs = new Array(MAX_TARGETS).fill(null);
let inWorkerRafTick = false;
let hudPresentRafId = 0;
let hudPresentWatchdogId = 0;
let hudSettleTimerId = 0;
let pendingHudPresentDirty = false;
let pendingHudPresentSlot = -1;
let lastCommittedHudSlot = -1;
let pendingHudSlotFast = true;
let pendingHudFullHist = false;

// Pre-allocated Per-Slot Sticky Hold-on-Loss & EMA-Smoothed Roster State (zero runtime GC)
const SLOT_HOLD_FRAMES = 48; // ~1.5s graceful hold when a target exits or briefly drops
const PINNED_HOLD_FRAMES = 180; // Match Wasm 180-frame user-pin retention window
const slotHoldLeft = new Int16Array(MAX_TARGETS);
const slotHoldTid = new Int32Array(MAX_TARGETS);
const slotIdleDrawn = new Uint8Array(MAX_TARGETS);
const slotHoldVeilDrawn = new Uint8Array(MAX_TARGETS);
const slotSpeedPillOn = new Uint8Array(MAX_TARGETS);
const slotSmoothedSpeed = new Float32Array(MAX_TARGETS);
const slotSmoothedHeading = new Float32Array(MAX_TARGETS);
const slotSmoothedOmega = new Float32Array(MAX_TARGETS);
const slotSmoothedEca = new Float32Array(MAX_TARGETS);
const slotLastAgeFrames = new Int32Array(MAX_TARGETS);
const slotLastCoast = new Int32Array(MAX_TARGETS);
const slotLastReacq = new Int32Array(MAX_TARGETS);
const slotLastVal = new Int32Array(MAX_TARGETS);
const slotLastSigHi = new Int32Array(MAX_TARGETS);
const slotLastSigLo = new Int32Array(MAX_TARGETS);
const slotLastFreqHz = new Float32Array(MAX_TARGETS);
const slotLastMetricValid = new Float32Array(MAX_TARGETS);
const slotLastCalSegId = new Int32Array(MAX_TARGETS).fill(-1);
const slotLastPivotValid = new Uint8Array(MAX_TARGETS);
const slotLastPendSwing = new Float32Array(MAX_TARGETS);
const slotLastPendRad = new Float32Array(MAX_TARGETS);
let camPanStreakWorker = 0;

// Internal staging & simulation OffscreenCanvases
let dispStageCanvas = null;
let dispStageCtx = null;
let pipStageCanvas = null;
let pipStageCtx = null;
let simCanvas = null;
let simCtx = null;
let ecaOverlayImgData = null;
let ecaOverlayBuf = null;
let ecaStageCanvas = null;
let ecaStageCtx = null;

// Runtime state
let running = false;
let sourceMode = 'testbench'; // 'testbench' | 'webcam' | 'video'
let benchScenario = 'pillar_crossing'; // 'pillar_crossing' | 'short_gap_bridge' | 'ballistic_cascade' | 'harmonic_tremor' | 'percussion_ricochet'
let showEcaOverlay = false;
let showKalmanBaseline = true;
let showHeroTargets = true;
let showAllOverlays = false;
let lastOverlaySyncSeq = 0;
let rawWipeRatio = 0.0;
let showcaseMinimalScopes = false;
let strobeEnabled = true;
const cropNorm = new Float64Array([0.0, 0.0, 1.0, 1.0]); // [x, y, w, h] in [0..1]

let lookaheadK = 8;
let ecaAdaptRate = 0.035;
let fistaLambda = 1.45;
let eulerianGain = 25.0;
let phaseOffsetRad = -Math.PI * 0.5;
let labMode = 0; // 0: Mode A (Ballistic/Bridge), 1: Mode B (Tremor/Reflex), 2: Mode C (Impulses)
let audioOutLatencySec = 0.0;

// Timing & Rolling Benchmark Telemetry (pre-allocated ring + median sort buffer)
const TIME_HIST_LEN = 120;
const wasmTimeHist = new Float32Array(TIME_HIST_LEN);
const wasmTimeSortBuf = new Float32Array(TIME_HIST_LEN);
let timeHistIdx = 0;
let lastRafTs = 0;
let smoothFps = 60.0;
let sourceFps = 60.0;
let tickCounter = 0;
let strobeFlashAlpha = 0.0;
const padFlashAlpha = new Float32Array([0, 0, 0, 0]);
let totalBridgedEvents = 0;
let totalStrikeEvents = 0;

// Pre-allocated Mode C Localized Impact Shockwave Pool (synchronized to t - K presentation)
const SHOCK_POOL = 24;
const shockTriggerFrame = new Uint32Array(SHOCK_POOL);
const shockTriggerArmed = new Uint8Array(SHOCK_POOL);
const shockX = new Float32Array(SHOCK_POOL);
const shockY = new Float32Array(SHOCK_POOL);
const shockZone = new Uint8Array(SHOCK_POOL);
const shockVel = new Float32Array(SHOCK_POOL);
const shockInDeg = new Float32Array(SHOCK_POOL);
const shockDeflectDeg = new Float32Array(SHOCK_POOL);
const shockRestitution = new Float32Array(SHOCK_POOL);
const shockAge = new Int16Array(SHOCK_POOL).fill(-1);
let shockWriteHead = 0;
let lastStrikeDeflectDeg = 0.0;
let lastStrikeRestitution = 0.0;
let lastStrikeTid = 0;
let lastStrikeFrame = -1;
let lastStrikeValid = false;
let lastScatterDeg = 0.0;
let lastScatterFrame = -1;
const slotLastStrikeX = new Float32Array(MAX_TARGETS);
const slotLastStrikeY = new Float32Array(MAX_TARGETS);
const prevRectHeading = new Float32Array(MAX_TARGETS);
const prevRectTid = new Int32Array(MAX_TARGETS);
const prevRectFrame = new Uint32Array(MAX_TARGETS);
const prevRectValid = new Uint8Array(MAX_TARGETS);
const slotRectAngVel = new Float32Array(MAX_TARGETS);

// Pre-allocated Mode A Frozen Parabolic Apex Reticle & Live Accuracy Score Buffers
const apexLockTid = new Int32Array(MAX_TARGETS);
const apexLockPredX = new Float32Array(MAX_TARGETS);
const apexLockPredY = new Float32Array(MAX_TARGETS);
const apexLockMinY = new Float32Array(MAX_TARGETS);
const apexLockAtMinX = new Float32Array(MAX_TARGETS);
const apexLockErrPx = new Float32Array(MAX_TARGETS).fill(-1.0);
const apexLockHold = new Int16Array(MAX_TARGETS);
let lastScoredApexErrPx = -1.0;

// Hilbert & Spectral Scope circular histories (pre-allocated, including track ID continuity)
const SCOPE_HIST = 128;
const histTremorR = new Float32Array(SCOPE_HIST);
const histTremorH = new Float32Array(SCOPE_HIST);
const histPhaseValid = new Uint8Array(SCOPE_HIST);
const histStrobe = new Uint8Array(SCOPE_HIST);
const histPipTid = new Int32Array(SCOPE_HIST).fill(-1);
const histLambda2 = new Float32Array(SCOPE_HIST);
let scopeWriteIdx = 0;

// Pre-allocated Zero-GC Phase-Space Portrait & Conservation Scope Buffers (15 Hz OffscreenCanvas)
const PHYS_REGIME_NAMES = [
  'INERTIAL CV',
  'BALLISTIC',
  'BALLISTIC + AERO DRAG',
  'POWERED BOOST',
  'PARACHUTE / TERMINAL',
  'PENDULUM / HARMONIC',
  'MICRO-TREMOR',
  'IMPULSIVE COLLISIONAL',
  'CURVED CTRV',
];
const JUMP_CODE_NAMES = [
  'NONE',
  'GALILEO PEG JUMP',
  'BOOST BURNOUT',
  'PARACHUTE DEPLOY',
  'COLLISION IMPULSE',
];
const PHASE_HIST = 140;
const phaseHistX = new Float32Array(PHASE_HIST);
const phaseHistY = new Float32Array(PHASE_HIST);
const phaseHistVx = new Float32Array(PHASE_HIST);
const phaseHistVy = new Float32Array(PHASE_HIST);
const phaseHistE = new Float32Array(PHASE_HIST);
const phaseHistEnsK = new Float32Array(PHASE_HIST);
const phaseHistKurR = new Float32Array(PHASE_HIST);
const phaseHistKurN = new Uint8Array(PHASE_HIST);
const phaseHistRegime = new Uint8Array(PHASE_HIST);
const phaseHistJump = new Uint8Array(PHASE_HIST);
const phaseHistTid = new Int32Array(PHASE_HIST).fill(-1);
let phaseHistWrite = 0;
let phaseHistCount = 0;
let lastPhysicsPhaseRenderMs = 0.0;

// Pre-allocated HUD label collision stagger buffers (zero per-frame allocation)
const MAX_HUD_LABELS = 24;
const hudLabelX = new Float32Array(MAX_HUD_LABELS);
const hudLabelY = new Float32Array(MAX_HUD_LABELS);
const hudLabelW = new Float32Array(MAX_HUD_LABELS);

// Pre-allocated 44-frame (~1.5s) persistent per-track comet trail buffers (zero runtime GC)
const TRAIL_SLOTS = 64;
const TRAIL_LEN = 44;
const trailOwnerTid = new Int32Array(TRAIL_SLOTS).fill(-1);
const trailHead = new Uint8Array(TRAIL_SLOTS);
const trailCount = new Uint8Array(TRAIL_SLOTS);
const trailLastFrame = new Uint32Array(TRAIL_SLOTS);
const trailX = new Float32Array(TRAIL_SLOTS * TRAIL_LEN);
const trailY = new Float32Array(TRAIL_SLOTS * TRAIL_LEN);

// Pre-allocated Zero-GC Frame-Accurate Telemetry Session Recorder (18,000 frames = 5 min @ 60 Hz, 56 cols)
const REC_COLS = 56;
const REC_MAX_FRAMES = 18000;
const REC_COLUMNS = [
  'frame_idx', 'elapsed_sec', 'frame_w', 'frame_h', 'lookahead_k', 'lab_mode',
  'wasm_tick_ms', 'active_targets', 'clutter_db', 'clutter_eval_ratio',
  'padic_swaps', 'oracle_swaps', 'kalman_swaps', 'in_pillar_collisions',
  'fiedler_lambda2', 'spectral_gap', 'fista_duality_gap',
  'primary_track_id', 'primary_sig_u32', 'primary_padic_val',
  'disp_raw_x', 'disp_raw_y', 'disp_fista_x', 'disp_fista_y',
  'disp_kalman_x', 'disp_kalman_y', 'disp_vis', 'disp_bridged',
  'tremor_r', 'tremor_h', 'tremor_phase_deg', 'tremor_freq_hz', 'tremor_amp',
  'phase_valid', 'strobe_fired', 'perc_mask', 'apex_ms', 'saccade_latency_ms',
  'heading_deg', 'speed_phys', 'ang_vel_deg_s', 'pend_swing_deg',
  'strike_deflect_deg', 'strike_speed_ratio', 'metric_x_m', 'metric_y_m',
  'cal_segment_id', 'metric_valid',
  'phys_regime', 'phys_regime_conf', 'ode_r2', 'aero_drag_beta',
  'thrust_accel', 'damping_zeta', 'kuramoto_order_r', 'lyapunov_div_rate',
];
const recSlab = new Float64Array(REC_MAX_FRAMES * REC_COLS);
let recActive = false;
let recHead = 0;
let recCount = 0;
let recTotalWritten = 0;
let recElapsedSec = 0.0;
let calSegmentId = 0;
let recCalSegments = [];

// Pre-allocated Zero-GC Multi-Minute Tracking Continuity & Custody Timeline State
const CONT_MAX_TRACKS = 128;
const CONT_MAX_SEGS = 24;
const CONT_HIST_BINS = 240;
const contTid = new Int32Array(CONT_MAX_TRACKS);
const contSlot = new Int16Array(CONT_MAX_TRACKS).fill(-1);
const contFirstSeq = new Int32Array(CONT_MAX_TRACKS);
const contLastSeq = new Int32Array(CONT_MAX_TRACKS);
const contTotalFrames = new Int32Array(CONT_MAX_TRACKS);
const contTotalSec = new Float32Array(CONT_MAX_TRACKS);
const contRelocks = new Int32Array(CONT_MAX_TRACKS);
const contSegCount = new Uint8Array(CONT_MAX_TRACKS);
const contSegStart = new Int32Array(CONT_MAX_TRACKS * CONT_MAX_SEGS);
const contSegEnd = new Int32Array(CONT_MAX_TRACKS * CONT_MAX_SEGS);
const contWasActive = new Uint8Array(CONT_MAX_TRACKS);
const contBinMaxActive = new Uint8Array(CONT_HIST_BINS);
const contBinFrames = new Uint32Array(CONT_HIST_BINS);
const contRankIdx = new Int32Array(CONT_MAX_TRACKS);
let contCount = 0;
let persistSampleSeq = 0;
let sessionElapsedSec = 0.0;
let lastPersistRenderMs = 0.0;
let contBinStrideFrames = 15;
let contBinHead = 0;

// Bounded Physics Event Timeline & Live Drag-Box Seeding State (v3.2.0)
const REC_EVENTS_MAX = 2048;
let recEvents = [];
let recEventsSpanSec = 0.0;
let lastTimelineEventsEmitMs = 0.0;
let lastTimelineEventsCount = 0;
let prevBosEventState = 0;
const prevSlotJumpEventCode = new Int8Array(MAX_TARGETS);
const prevSlotApexVal = new Float32Array(MAX_TARGETS);
const prevSlotApexEventFrame = new Uint32Array(MAX_TARGETS);
let lastCutEventFrame = 0xffffffff;
let seedBoxPreview = null; // { x0, y0, x1, y1 } normalized [0..1]
let fastPassActive = false;
const EVENT_TICK_COLORS = {
  strike: '#ff5d6c',
  scene_cut: '#00f0ff',
  apex: '#ffd166',
  bos_disturbed: '#ff922b',
  jump: '#c77dff',
};
const JUMP_EVENT_LABELS = ['', 'GALILEO PEG', 'BOOST BURNOUT', 'CHUTE DEPLOY', 'COLLISION'];

function resetRecEvents() {
  recEvents = [];
  recEventsSpanSec = 0.0;
  lastTimelineEventsEmitMs = 0.0;
  lastTimelineEventsCount = 0;
  prevBosEventState = 0;
  prevSlotJumpEventCode.fill(0);
  prevSlotApexVal.fill(0);
  prevSlotApexEventFrame.fill(0);
  lastCutEventFrame = 0xffffffff;
  self.postMessage({
    type: 'timeline_events',
    events: [],
    spanSec: 0.0,
    seq: 0,
  });
}

function pushRecEvent(kind, tSec, frame, tid, label, x, y) {
  const safeT = Number.isFinite(tSec) && tSec >= 0.0 ? Number(tSec.toFixed(4)) : 0.0;
  if (recEvents.length >= REC_EVENTS_MAX) {
    recEvents.shift();
  }
  recEvents.push({
    kind,
    tSec: safeT,
    frame: frame >>> 0,
    seq: persistSampleSeq,
    tid: tid | 0,
    label: label || kind.toUpperCase(),
    x: Number.isFinite(x) ? Number(x.toFixed(2)) : 0.0,
    y: Number.isFinite(y) ? Number(y.toFixed(2)) : 0.0,
  });
  if (safeT > recEventsSpanSec) {
    recEventsSpanSec = safeT;
  }
}

function emitTimelineEventsIfNeeded(force = false) {
  const nowMs = performance.now();
  if (!force && recEvents.length === lastTimelineEventsCount) return;
  if (!force && nowMs - lastTimelineEventsEmitMs < 350.0) return;
  lastTimelineEventsEmitMs = nowMs;
  lastTimelineEventsCount = recEvents.length;
  self.postMessage({
    type: 'timeline_events',
    events: recEvents,
    spanSec: Math.max(sessionElapsedSec, recEventsSpanSec),
    seq: persistSampleSeq,
  });
}

function renderRoiBoxOverlay(W, H) {
  if (!hudCtx) return;
  if (roiBoxPreview) {
    const x0 = Math.max(0, Math.min(1, roiBoxPreview.x0)) * W;
    const y0 = Math.max(0, Math.min(1, roiBoxPreview.y0)) * H;
    const x1 = Math.max(0, Math.min(1, roiBoxPreview.x1)) * W;
    const y1 = Math.max(0, Math.min(1, roiBoxPreview.y1)) * H;
    const rx = Math.min(x0, x1);
    const ry = Math.min(y0, y1);
    const rw = Math.abs(x1 - x0);
    const rh = Math.abs(y1 - y0);
    if (rw >= 2 && rh >= 2) {
      hudCtx.save();
      hudCtx.fillStyle = 'rgba(0, 229, 255, 0.14)';
      hudCtx.fillRect(rx, ry, rw, rh);
      hudCtx.strokeStyle = '#00e5ff';
      hudCtx.lineWidth = 1.8;
      hudCtx.setLineDash([6, 3]);
      hudCtx.strokeRect(rx + 0.5, ry + 0.5, rw, rh);
      hudCtx.setLineDash([]);
      hudCtx.fillStyle = 'rgba(6, 10, 18, 0.88)';
      const tagY = ry >= 18 ? ry - 16 : ry + 4;
      hudCtx.fillRect(rx, tagY, 110, 14);
      hudCtx.fillStyle = '#00e5ff';
      hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText('DRAG ZOOM ROI', rx + 5, tagY + 10.5);
      hudCtx.restore();
    }
  }
  const hasZoomCrop =
    nativeRoiEnabled ||
    cropNorm[0] > 0.001 ||
    cropNorm[1] > 0.001 ||
    cropNorm[2] < 0.999 ||
    cropNorm[3] < 0.999;
  if (hasZoomCrop && sourceMode !== 'testbench' && lastIngestGeom && lastIngestGeom.srcW > 0 && lastIngestGeom.srcH > 0) {
    const mw = 84;
    const mh = Math.max(36, Math.round((mw * lastIngestGeom.srcH) / lastIngestGeom.srcW));
    const mx = W - mw - 12;
    const my = 40;
    const rx = mx + (lastIngestGeom.cropX0 / lastIngestGeom.srcW) * mw;
    const ry = my + (lastIngestGeom.cropY0 / lastIngestGeom.srcH) * mh;
    const rw = Math.max(4, (lastIngestGeom.cropW / lastIngestGeom.srcW) * mw);
    const rh = Math.max(4, (lastIngestGeom.cropH / lastIngestGeom.srcH) * mh);
    hudCtx.save();
    hudCtx.fillStyle = 'rgba(5, 10, 18, 0.82)';
    hudCtx.fillRect(mx, my, mw, mh);
    hudCtx.strokeStyle = 'rgba(138, 173, 214, 0.55)';
    hudCtx.lineWidth = 1;
    hudCtx.strokeRect(mx + 0.5, my + 0.5, mw - 1, mh - 1);
    hudCtx.fillStyle = 'rgba(0, 229, 255, 0.22)';
    hudCtx.fillRect(rx, ry, rw, rh);
    hudCtx.strokeStyle = autoTrackRoiEnabled ? '#ffd166' : '#00e5ff';
    hudCtx.lineWidth = 1.4;
    hudCtx.strokeRect(rx + 0.5, ry + 0.5, Math.max(2, rw - 1), Math.max(2, rh - 1));
    hudCtx.fillStyle = '#a8c0df';
    hudCtx.font = '700 8px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(lastIngestGeom.native1to1 ? '1:1 ROI' : 'ZOOM ROI', mx + 3, my + mh - 3);
    hudCtx.restore();
  }
}

function renderSeedBoxOverlay(W, H) {
  renderRoiBoxOverlay(W, H);
  if (!seedBoxPreview || !hudCtx) return;
  const x0 = Math.max(0, Math.min(1, seedBoxPreview.x0)) * W;
  const y0 = Math.max(0, Math.min(1, seedBoxPreview.y0)) * H;
  const x1 = Math.max(0, Math.min(1, seedBoxPreview.x1)) * W;
  const y1 = Math.max(0, Math.min(1, seedBoxPreview.y1)) * H;
  const rx = Math.min(x0, x1);
  const ry = Math.min(y0, y1);
  const rw = Math.abs(x1 - x0);
  const rh = Math.abs(y1 - y0);
  if (rw < 2 || rh < 2) return;
  hudCtx.save();
  hudCtx.fillStyle = 'rgba(0, 240, 255, 0.12)';
  hudCtx.fillRect(rx, ry, rw, rh);
  hudCtx.strokeStyle = '#00f0ff';
  hudCtx.lineWidth = 1.6;
  hudCtx.setLineDash([5, 3]);
  hudCtx.strokeRect(rx + 0.5, ry + 0.5, rw, rh);
  hudCtx.setLineDash([]);
  hudCtx.strokeStyle = '#ffd166';
  hudCtx.lineWidth = 2.0;
  const cLen = Math.min(10, rw * 0.35, rh * 0.35);
  hudCtx.beginPath();
  hudCtx.moveTo(rx, ry + cLen); hudCtx.lineTo(rx, ry); hudCtx.lineTo(rx + cLen, ry);
  hudCtx.moveTo(rx + rw - cLen, ry); hudCtx.lineTo(rx + rw, ry); hudCtx.lineTo(rx + rw, ry + cLen);
  hudCtx.moveTo(rx, ry + rh - cLen); hudCtx.lineTo(rx, ry + rh); hudCtx.lineTo(rx + cLen, ry + rh);
  hudCtx.moveTo(rx + rw - cLen, ry + rh); hudCtx.lineTo(rx + rw, ry + rh); hudCtx.lineTo(rx + rw, ry + rh - cLen);
  hudCtx.stroke();
  hudCtx.fillStyle = 'rgba(6, 10, 18, 0.85)';
  const tagY = ry >= 18 ? ry - 16 : ry + 4;
  hudCtx.fillRect(rx, tagY, 92, 14);
  hudCtx.fillStyle = '#00f0ff';
  hudCtx.font = '600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  hudCtx.fillText('SEED TARGET BOX', rx + 4, tagY + 10.5);
  hudCtx.restore();
}

// ============================================================================
// Pre-Allocated Metric Ruler, Auto-g Scale, & 4-Point Plane Homography Engine
// Stored in normalized [0..1] viewport coordinates so SD <-> HD is automatic!
// ============================================================================
const GRAVITY_M_S2 = 9.80665;
const HOMOG_MIN_W = 0.14;
const AUTO_G_LOCK_CONF = 0.50;
const CORNER_TAGS = ['TL', 'TR', 'BR', 'BL'];
let calMode = 'none';       // 'none' | 'ruler' | 'auto_g' | 'homography'
let calProvenance = 'none'; // 'none' | 'assumed' | 'surveyed'
let calArmed = 'none';      // 'none' | 'ruler' | 'homography'
let calPtCount = 0;
let calDraftCount = 0;
let lastCalSyncSeq = 0;
let calRulerMeters = 1.0;
let calPlaneW = 2.84;
let calPlaneH = 1.42;
let calDraftRulerMeters = 1.0;
let calDraftPlaneW = 2.84;
let calDraftPlaneH = 1.42;
let calPresetName = '';
let calAutoGNormH = 0.0;    // scene_gravity_px_s2 / frameH (resolution-invariant)
let calAutoGLocked = false; // true only when >= 2 consistent free-fall arcs lock
let lastLambda2 = 0.0;
const calPtsNormX = new Float64Array(4);
const calPtsNormY = new Float64Array(4);
const calDraftPtsNormX = new Float64Array(4);
const calDraftPtsNormY = new Float64Array(4);
const calHomography = new Float64Array(9);    // H: (nx, ny, 1)^T -> ~ (X_m, Y_m, 1)^T
const calHomographyInv = new Float64Array(9); // H^-1: (X_m, Y_m, 1)^T -> ~ (nx, ny, 1)^T
let calHomographyValid = false;

// Pre-allocated solver & output scratchpad (zero runtime heap allocations)
const homMatA = new Float64Array(64);
const homVecB = new Float64Array(8);
const homSol = new Float64Array(8);
const homTargetX = new Float64Array(4);
const homTargetY = new Float64Array(4);
const homScratchMat = new Float64Array(9);
const homScratchInv = new Float64Array(9);
const calOutPt = new Float64Array(3);      // [Xm, Ym, metricValid: 1=m, 0=px, -1=out_of_plane]
const calOutNorm = new Float64Array(2);
const calOutVel = new Float64Array(3);     // [speedPhys, rectifiedHeadingDeg, metricValid]
const calOutStrike = new Float64Array(3);  // [rectInDeg, rectDeflectDeg, rectSpeedRatio]
const calOutScatter = new Float64Array(3); // [rectTh1Deg, rectTh2Deg, rectScatterDeg]
const calOutPend = new Float64Array(3);    // [swingDeg, radiusPhys, metricValid]

function wrapDeg180(deg) {
  let d = ((deg + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
  if (d === -180.0) d = 180.0;
  return d;
}

function invert3x3(m, out) {
  const a00 = m[0], a01 = m[1], a02 = m[2];
  const a10 = m[3], a11 = m[4], a12 = m[5];
  const a20 = m[6], a21 = m[7], a22 = m[8];

  const b01 = a22 * a11 - a12 * a21;
  const b11 = -a22 * a10 + a12 * a20;
  const b21 = a21 * a10 - a11 * a20;
  const det = a00 * b01 + a01 * b11 + a02 * b21;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-11) return false;
  const invDet = 1.0 / det;

  out[0] = b01 * invDet;
  out[1] = (-a22 * a01 + a02 * a21) * invDet;
  out[2] = (a12 * a01 - a02 * a11) * invDet;
  out[3] = b11 * invDet;
  out[4] = (a22 * a00 - a02 * a20) * invDet;
  out[5] = (-a12 * a00 + a02 * a10) * invDet;
  out[6] = b21 * invDet;
  out[7] = (-a21 * a00 + a01 * a20) * invDet;
  out[8] = (a11 * a00 - a01 * a10) * invDet;
  return true;
}

function solveHomography4Pt(planeW, planeH, ptsX = calPtsNormX, ptsY = calPtsNormY, commit = true) {
  const Wp = Math.max(0.01, planeW);
  const Hp = Math.max(0.01, planeH);

  for (let i = 0; i < 4; i++) {
    if (!Number.isFinite(ptsX[i]) || !Number.isFinite(ptsY[i])) return false;
  }

  // Enforce strictly convex clockwise TL -> TR -> BR -> BL screen-space quad (+x right, +y down)
  // and minimum normalized viewport area (>= 0.1% of frame, rejecting 1px/collinear/mirrored/bowtie quads).
  let signedArea2 = 0.0;
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) & 3;
    const k = (i + 2) & 3;
    const ux = ptsX[j] - ptsX[i];
    const uy = ptsY[j] - ptsY[i];
    const vx = ptsX[k] - ptsX[j];
    const vy = ptsY[k] - ptsY[j];
    const cr = ux * vy - uy * vx;
    if (!Number.isFinite(cr) || cr <= 1e-5) return false;
    signedArea2 += ptsX[i] * ptsY[j] - ptsX[j] * ptsY[i];
  }
  if (!Number.isFinite(signedArea2) || signedArea2 < 2e-3) return false;

  // Closed-form projective rectangle basis solve (zero h33 = 1 singularity even when the
  // vanishing horizon passes directly through image (0, 0)):
  // Map unit square (u, v) = (X / Wp, Y / Hp) corners:
  //   BL (0,0) -> p3,  BR (1,0) -> lam2 * p2,  TL (0,1) -> lam0 * p0,  TR (1,1) -> lam1 * p1
  // where lam2 * (p2 - p1) + lam0 * (p0 - p1) = p3 - p1 and lam1 = lam2 + lam0 - 1.
  const dx2 = ptsX[2] - ptsX[1];
  const dy2 = ptsY[2] - ptsY[1];
  const dx0 = ptsX[0] - ptsX[1];
  const dy0 = ptsY[0] - ptsY[1];
  const sx = ptsX[3] - ptsX[1];
  const sy = ptsY[3] - ptsY[1];
  const det = dx2 * dy0 - dx0 * dy2;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-11) return false;

  const lam2 = (sx * dy0 - dx0 * sy) / det;
  const lam0 = (dx2 * sy - sx * dy2) / det;
  const lam1 = lam2 + lam0 - 1.0;
  if (!(lam0 > 1e-4 && lam1 > 1e-4 && lam2 > 1e-4)) return false;

  // Build H_inv (metric [X, Y, 1]^T -> w_px * [x, y, 1]^T) directly from [c0/Wp, c1/Hp, c2]:
  const invWp = 1.0 / Wp;
  const invHp = 1.0 / Hp;
  homScratchInv[0] = (lam2 * ptsX[2] - ptsX[3]) * invWp;
  homScratchInv[1] = (lam0 * ptsX[0] - ptsX[3]) * invHp;
  homScratchInv[2] = ptsX[3];
  homScratchInv[3] = (lam2 * ptsY[2] - ptsY[3]) * invWp;
  homScratchInv[4] = (lam0 * ptsY[0] - ptsY[3]) * invHp;
  homScratchInv[5] = ptsY[3];
  homScratchInv[6] = (lam2 - 1.0) * invWp;
  homScratchInv[7] = (lam0 - 1.0) * invHp;
  homScratchInv[8] = 1.0;

  if (!invert3x3(homScratchInv, homScratchMat)) return false;

  // Projective depth at corner i is proportional to 1 / lam_i > 0.
  // Align sign so all 4 corners have w_i > 0, then scale H so min(w_corners) = 1.0
  // (guarantees w >= 1.0 > HOMOG_MIN_W throughout the entire convex ground quad while
  // bounding extreme horizon-grazing foreshortening ratios <= 250x).
  let posCount = 0;
  let negCount = 0;
  for (let i = 0; i < 4; i++) {
    const w = homScratchMat[6] * ptsX[i] + homScratchMat[7] * ptsY[i] + homScratchMat[8];
    if (!Number.isFinite(w)) return false;
    if (w > 1e-8) posCount++;
    else if (w < -1e-8) negCount++;
  }
  if (negCount === 4) {
    for (let k = 0; k < 9; k++) homScratchMat[k] = -homScratchMat[k];
  } else if (posCount !== 4) {
    return false;
  }

  let minW = Infinity;
  let maxW = 0.0;
  for (let i = 0; i < 4; i++) {
    const w = homScratchMat[6] * ptsX[i] + homScratchMat[7] * ptsY[i] + homScratchMat[8];
    if (w < minW) minW = w;
    if (w > maxW) maxW = w;
  }
  if (!(minW > 1e-8) || maxW / minW > 250.0) return false;

  const invMinW = 1.0 / minW;
  for (let k = 0; k < 9; k++) homScratchMat[k] *= invMinW;
  if (!invert3x3(homScratchMat, homScratchInv)) return false;

  for (let i = 0; i < 4; i++) {
    const w = homScratchMat[6] * ptsX[i] + homScratchMat[7] * ptsY[i] + homScratchMat[8];
    if (w <= HOMOG_MIN_W) return false;
  }

  if (commit) {
    calHomography.set(homScratchMat);
    calHomographyInv.set(homScratchInv);
  }
  return true;
}

function isCameraMotionSuspendingCal() {
  return !!(telemView && sourceMode !== 'testbench' && Math.round(telemView[TELEM_GLOBAL + 9]) > 0);
}

function getEffectivePxPerMeter() {
  if (isCameraMotionSuspendingCal()) return 0.0;
  if (calMode === 'ruler' && calPtCount >= 2) {
    const dx = (calPtsNormX[1] - calPtsNormX[0]) * frameW;
    const dy = (calPtsNormY[1] - calPtsNormY[0]) * frameH;
    const spanPx = Math.hypot(dx, dy);
    return spanPx > 1.0 ? spanPx / Math.max(0.01, calRulerMeters) : 0.0;
  }
  if (calMode === 'auto_g') {
    if (calAutoGLocked && calAutoGNormH > 0.05) {
      return (calAutoGNormH * frameH) / GRAVITY_M_S2;
    }
  }
  return 0.0;
}

function isMetricCalibrated() {
  if (isCameraMotionSuspendingCal()) return false;
  if (calMode === 'homography') return calHomographyValid;
  if (calMode === 'ruler' || calMode === 'auto_g') return getEffectivePxPerMeter() > 0.5;
  return false;
}

function safeFinite(v, fallback = 0.0) {
  return Number.isFinite(v) ? v : fallback;
}

function computeLocalHomographyPpm(px, py) {
  if (isCameraMotionSuspendingCal()) return -1.0;
  if (calMode !== 'homography' || !calHomographyValid) return 0.0;
  const nx = px / frameW;
  const ny = py / frameH;
  const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
  if (!(w > HOMOG_MIN_W)) return -1.0;
  const X = (calHomography[0] * nx + calHomography[1] * ny + calHomography[2]) / w;
  const Y = (calHomography[3] * nx + calHomography[4] * ny + calHomography[5]) / w;
  const j00 = ((calHomography[0] - X * calHomography[6]) / w) / frameW;
  const j01 = ((calHomography[1] - X * calHomography[7]) / w) / frameH;
  const j10 = ((calHomography[3] - Y * calHomography[6]) / w) / frameW;
  const j11 = ((calHomography[4] - Y * calHomography[7]) / w) / frameH;
  const det = Math.abs(j00 * j11 - j01 * j10);
  if (!Number.isFinite(det) || det < 1e-12) return 0.0;
  const ppm = 1.0 / Math.sqrt(det);
  return Number.isFinite(ppm) ? ppm : 0.0;
}

function formatCompactPhysicsVal(v, digits = 2) {
  if (!Number.isFinite(v) || Math.abs(v) < 1e-9) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e4) return v.toExponential(2);
  return v.toFixed(digits);
}

function formatSymbolicOdeString(p) {
  if (!p.hasPrimary) {
    if (p.kuramotoOscCount >= 2 && p.kuramotoOrderR > 0.05) {
      return `Σθ̇ᵢ = ω + (K/N)Σsin(θⱼ-θᵢ) [R=${p.kuramotoOrderR.toFixed(2)}, N=${p.kuramotoOscCount}]`;
    }
    return 'Awaiting confirmed target dynamics...';
  }
  const betaStr = `${formatCompactPhysicsVal(p.betaDisp, 3)} ${p.betaUnit}`;
  const thrustStr = `${formatCompactPhysicsVal(p.thrustDisp, 2)} ${p.thrustUnit}`;
  const kappaStr = `${formatCompactPhysicsVal(p.curvatureDisp, 3)} ${p.curvatureUnit}`;
  const zetaStr = p.dampingZeta.toFixed(3);
  const qStr = p.qualityFactorQ >= 998.5 ? '>=999' : (p.qualityFactorQ > 0 ? p.qualityFactorQ.toFixed(1) : '--');
  const tStr = p.ellipticPeriodS > 1e-3 ? `${p.ellipticPeriodS.toFixed(3)}s` : '--';

  switch (p.physRegime) {
    case 0:
      return `dr/dt = v (${p.speedStr}), d²r/dt² ≈ 0 [R²=${p.odeR2.toFixed(2)}]`;
    case 1:
      return `d²y/dt² = +g (${p.gStr}) [R²=${p.odeR2.toFixed(2)}]`;
    case 2:
      return `d²r/dt² = g - β‖v‖v (β=${betaStr}, g=${p.gStr})`;
    case 3:
      return `d²r/dt² = a_th + g - β‖v‖v (a_th=${thrustStr}, β=${betaStr})`;
    case 4:
      return `dv_y/dt = g - βv_y² (β=${betaStr}, g=${p.gStr})`;
    case 5:
      return `θ̈ + (g/L)sinθ + 2ζω₀θ̇ = 0 (T=${tStr}, ζ=${zetaStr}, Q=${qStr})`;
    case 6: {
      const fTag = p.tremorFreqHz > 0.05 ? `f=${p.tremorFreqHz.toFixed(2)}Hz, ` : '';
      return `q̈ + 2ζω₀q̇ + ω₀²q + αq³ = 0 (${fTag}ζ=${zetaStr}, Q=${qStr})`;
    }
    case 7: {
      const eStr = p.normalRestitutionE > 0.01 ? p.normalRestitutionE.toFixed(3) : '--';
      const mTag = p.lastPairMassRatio > 0 && p.lastPairFrame > 0 ? `, m₂/m₁=${p.lastPairMassRatio.toFixed(2)}` : '';
      return `Δvₙ⁺ = -eₙΔvₙ⁻ (eₙ=${eStr}${mTag})`;
    }
    case 8:
      return `ψ̇ = ω (${p.angVelDegS.toFixed(1)}°/s), d²r/dt² = ω×v (κ=${kappaStr})`;
    default:
      return `dr/dt = v (${p.speedStr})`;
  }
}

function mapPixelToMetric(px, py) {
  if (isCameraMotionSuspendingCal()) {
    calOutPt[0] = 0.0;
    calOutPt[1] = 0.0;
    calOutPt[2] = -1.0;
    return calOutPt;
  }
  if (calMode === 'homography' && calHomographyValid) {
    const nx = px / frameW;
    const ny = py / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (w > HOMOG_MIN_W) {
      calOutPt[0] = (calHomography[0] * nx + calHomography[1] * ny + calHomography[2]) / w;
      calOutPt[1] = (calHomography[3] * nx + calHomography[4] * ny + calHomography[5]) / w;
      calOutPt[2] = 1.0;
      return calOutPt;
    }
    calOutPt[0] = 0.0;
    calOutPt[1] = 0.0;
    calOutPt[2] = -1.0;
    return calOutPt;
  }
  const ppm = getEffectivePxPerMeter();
  if (ppm > 0.5) {
    if (calMode === 'ruler' && calPtCount >= 2) {
      const ox = calPtsNormX[0] * frameW;
      const oy = calPtsNormY[0] * frameH;
      calOutPt[0] = (px - ox) / ppm;
      calOutPt[1] = (oy - py) / ppm;
    } else {
      calOutPt[0] = px / ppm;
      calOutPt[1] = (frameH - py) / ppm;
    }
    calOutPt[2] = 1.0;
    return calOutPt;
  }
  calOutPt[0] = 0.0;
  calOutPt[1] = 0.0;
  calOutPt[2] = 0.0;
  return calOutPt;
}

function mapMetricToNorm(Xm, Ym) {
  const w = calHomographyInv[6] * Xm + calHomographyInv[7] * Ym + calHomographyInv[8];
  const safeW = Math.abs(w) < 1e-5 ? 1e-5 : w;
  calOutNorm[0] = (calHomographyInv[0] * Xm + calHomographyInv[1] * Ym + calHomographyInv[2]) / safeW;
  calOutNorm[1] = (calHomographyInv[3] * Xm + calHomographyInv[4] * Ym + calHomographyInv[5]) / safeW;
  return calOutNorm;
}

function mapVelocityToPhysical(px, py, speedPxS, headingDeg) {
  if (isCameraMotionSuspendingCal()) {
    calOutVel[0] = 0.0;
    calOutVel[1] = headingDeg;
    calOutVel[2] = -1.0;
    return calOutVel;
  }
  if (calMode === 'homography' && calHomographyValid) {
    const nx = px / frameW;
    const ny = py / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (w > HOMOG_MIN_W) {
      const X = (calHomography[0] * nx + calHomography[1] * ny + calHomography[2]) / w;
      const Y = (calHomography[3] * nx + calHomography[4] * ny + calHomography[5]) / w;
      const rad = headingDeg * (Math.PI / 180.0);
      const dnx = (speedPxS * Math.cos(rad)) / frameW;
      const dny = (-speedPxS * Math.sin(rad)) / frameH;
      const dw = calHomography[6] * dnx + calHomography[7] * dny;
      const dX = ((calHomography[0] * dnx + calHomography[1] * dny) - X * dw) / w;
      const dY = ((calHomography[3] * dnx + calHomography[4] * dny) - Y * dw) / w;
      const spdM = Math.hypot(dX, dY);
      calOutVel[0] = spdM;
      calOutVel[1] = spdM > 1e-4 ? Math.atan2(dY, dX) * (180.0 / Math.PI) : headingDeg;
      calOutVel[2] = 1.0;
      return calOutVel;
    }
    // Never fall back to raw px/s disguised as m/s above horizon / outside valid frustum!
    calOutVel[0] = 0.0;
    calOutVel[1] = headingDeg;
    calOutVel[2] = -1.0;
    return calOutVel;
  }
  const ppm = getEffectivePxPerMeter();
  if (ppm > 0.5) {
    calOutVel[0] = speedPxS / ppm;
    calOutVel[1] = headingDeg;
    calOutVel[2] = 1.0;
  } else {
    calOutVel[0] = speedPxS;
    calOutVel[1] = headingDeg;
    calOutVel[2] = 0.0;
  }
  return calOutVel;
}

function rectifyStrikeKinematics(sxPx, syPx, inDeg, deflectDeg, restitution) {
  if (calMode === 'homography' && calHomographyValid && !isCameraMotionSuspendingCal() && restitution > 0.01) {
    const nx = sxPx / frameW;
    const ny = syPx / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (w > HOMOG_MIN_W) {
      const X = (calHomography[0] * nx + calHomography[1] * ny + calHomography[2]) / w;
      const Y = (calHomography[3] * nx + calHomography[4] * ny + calHomography[5]) / w;

      const inRad = inDeg * (Math.PI / 180.0);
      const inDnx = Math.cos(inRad) / frameW;
      const inDny = -Math.sin(inRad) / frameH;
      const inDw = calHomography[6] * inDnx + calHomography[7] * inDny;
      const inDX = ((calHomography[0] * inDnx + calHomography[1] * inDny) - X * inDw) / w;
      const inDY = ((calHomography[3] * inDnx + calHomography[4] * inDny) - Y * inDw) / w;

      const outDeg = inDeg + deflectDeg;
      const outRad = outDeg * (Math.PI / 180.0);
      const outDnx = (restitution * Math.cos(outRad)) / frameW;
      const outDny = (-restitution * Math.sin(outRad)) / frameH;
      const outDw = calHomography[6] * outDnx + calHomography[7] * outDny;
      const outDX = ((calHomography[0] * outDnx + calHomography[1] * outDny) - X * outDw) / w;
      const outDY = ((calHomography[3] * outDnx + calHomography[4] * outDny) - Y * outDw) / w;

      const inNorm = Math.hypot(inDX, inDY);
      const outNorm = Math.hypot(outDX, outDY);
      if (inNorm > 1e-6 && outNorm > 1e-6) {
        const rInDeg = Math.atan2(inDY, inDX) * (180.0 / Math.PI);
        const rOutDeg = Math.atan2(outDY, outDX) * (180.0 / Math.PI);
        calOutStrike[0] = rInDeg;
        calOutStrike[1] = wrapDeg180(rOutDeg - rInDeg);
        calOutStrike[2] = Math.min(2.5, outNorm / inNorm);
        return calOutStrike;
      }
    }
  }
  calOutStrike[0] = inDeg;
  calOutStrike[1] = deflectDeg;
  calOutStrike[2] = restitution;
  return calOutStrike;
}

function rectifyScatterKinematics(scxPx, scyPx, th1Deg, th2Deg, rawScatterDeg) {
  if (calMode === 'homography' && calHomographyValid && !isCameraMotionSuspendingCal()) {
    const nx = scxPx / frameW;
    const ny = scyPx / frameH;
    const w = calHomography[6] * nx + calHomography[7] * ny + calHomography[8];
    if (w > HOMOG_MIN_W) {
      const X = (calHomography[0] * nx + calHomography[1] * ny + calHomography[2]) / w;
      const Y = (calHomography[3] * nx + calHomography[4] * ny + calHomography[5]) / w;

      const r1 = th1Deg * (Math.PI / 180.0);
      const dnx1 = Math.cos(r1) / frameW;
      const dny1 = -Math.sin(r1) / frameH;
      const dw1 = calHomography[6] * dnx1 + calHomography[7] * dny1;
      const dX1 = ((calHomography[0] * dnx1 + calHomography[1] * dny1) - X * dw1) / w;
      const dY1 = ((calHomography[3] * dnx1 + calHomography[4] * dny1) - Y * dw1) / w;

      const r2 = th2Deg * (Math.PI / 180.0);
      const dnx2 = Math.cos(r2) / frameW;
      const dny2 = -Math.sin(r2) / frameH;
      const dw2 = calHomography[6] * dnx2 + calHomography[7] * dny2;
      const dX2 = ((calHomography[0] * dnx2 + calHomography[1] * dny2) - X * dw2) / w;
      const dY2 = ((calHomography[3] * dnx2 + calHomography[4] * dny2) - Y * dw2) / w;

      if (Math.hypot(dX1, dY1) > 1e-6 && Math.hypot(dX2, dY2) > 1e-6) {
        const a1 = Math.atan2(dY1, dX1) * (180.0 / Math.PI);
        const a2 = Math.atan2(dY2, dX2) * (180.0 / Math.PI);
        calOutScatter[0] = a1;
        calOutScatter[1] = a2;
        calOutScatter[2] = Math.abs(wrapDeg180(a1 - a2));
        return calOutScatter;
      }
    }
  }
  calOutScatter[0] = th1Deg;
  calOutScatter[1] = th2Deg;
  calOutScatter[2] = rawScatterDeg;
  return calOutScatter;
}

function computePendulumKinematics(fx, fy, pivX, pivY) {
  const dx = fx - pivX;
  const dyAbs = Math.max(1e-3, Math.abs(fy - pivY));
  const swingDeg = Math.atan2(dx, dyAbs) * (180.0 / Math.PI);
  const rPx = Math.hypot(dx, fy - pivY);
  if (calMode === 'homography' && calHomographyValid) {
    const p0 = mapPixelToMetric(pivX, pivY);
    const p0x = p0[0], p0y = p0[1], p0v = p0[2];
    const p1 = mapPixelToMetric(fx, fy);
    calOutPend[0] = swingDeg;
    if (p0v > 0.5 && p1[2] > 0.5) {
      calOutPend[1] = Math.hypot(p1[0] - p0x, p1[1] - p0y);
      calOutPend[2] = 1.0;
    } else {
      calOutPend[1] = 0.0;
      calOutPend[2] = -1.0;
    }
    return calOutPend;
  }
  const ppm = getEffectivePxPerMeter();
  calOutPend[0] = swingDeg;
  if (ppm > 0.5) {
    calOutPend[1] = rPx / ppm;
    calOutPend[2] = 1.0;
  } else {
    calOutPend[1] = rPx;
    calOutPend[2] = 0.0;
  }
  return calOutPend;
}

function formatCalibrationSummary(overrideNotice) {
  if (overrideNotice) return overrideNotice;
  if (calArmed === 'ruler') {
    return `ARMED 2-PT RULER (${calDraftCount}/2 CLICKS · L=${calDraftRulerMeters.toFixed(2)} m · ESC PRESERVES PRIOR)`;
  }
  if (calArmed === 'homography') {
    return `ARMED 4-PT PLANE (${calDraftCount}/4 [${CORNER_TAGS[Math.min(3, calDraftCount)]}] · ${calDraftPlaneW.toFixed(2)}×${calDraftPlaneH.toFixed(2)} m · ESC PRESERVES)`;
  }
  if (isCameraMotionSuspendingCal() && calMode !== 'none') {
    return 'CAMERA MOTION — IMAGE-PLANE CALIBRATION SUSPENDED';
  }
  const provTag = calProvenance === 'assumed' ? 'ASSUMED SCALE' : 'SURVEYED';
  if (calMode === 'ruler') {
    const ppm = getEffectivePxPerMeter();
    return `2-PT RULER [${provTag}]: ${calRulerMeters.toFixed(2)} m (${Math.round(ppm)} px/m)`;
  }
  if (calMode === 'auto_g') {
    const ppm = getEffectivePxPerMeter();
    const arcs = telemView ? Math.round(telemView[TELEM_GLOBAL + 11]) : 0;
    if (calAutoGLocked && calAutoGNormH > 0.05) {
      const gPx = Math.round(calAutoGNormH * frameH);
      return `AUTO-g LOCKED (${Math.max(2, arcs)} ARCS): ĝ=${gPx} px/s² → ${Math.round(ppm)} px/m`;
    }
    if (telemView && telemView[TELEM_GLOBAL + 0] > 50.0) {
      return `AUTO-g PROVISIONAL (${Math.max(1, arcs)}/2 ARCS · ĝ=${Math.round(telemView[TELEM_GLOBAL + 0])} px/s² · AWAITING 2ND ARC)`;
    }
    return 'AUTO-g ARMED · AWAITING >= 2 PARABOLIC FLIGHT ARCS (g = 9.81 m/s²)';
  }
  if (calMode === 'homography' && calHomographyValid) {
    return `4-PT HOMOGRAPHY [${provTag}]: ${calPlaneW.toFixed(2)}×${calPlaneH.toFixed(2)} m [RECTIFIED]`;
  }
  return 'UNCALIBRATED · PIXEL UNITS (px · px/s)';
}

function getCalibrationSnapshot() {
  return {
    segmentId: calSegmentId,
    mode: calMode,
    provenance: calProvenance,
    armed: calArmed,
    ptCount: calPtCount,
    rulerMeters: calRulerMeters,
    planeW: calPlaneW,
    planeH: calPlaneH,
    presetName: calPresetName,
    autoGLocked: calAutoGLocked,
    pxPerMeter: Number(getEffectivePxPerMeter().toFixed(4)),
    ptsNorm: calPtCount > 0
      ? Array.from({ length: calPtCount }, (_, i) => [
          Number(calPtsNormX[i].toFixed(5)),
          Number(calPtsNormY[i].toFixed(5)),
        ])
      : [],
    homographyValid: calHomographyValid,
    homography: calHomographyValid ? Array.from(calHomography) : null,
  };
}

function recordCalibrationSegmentTransition(reason) {
  calSegmentId++;
  if (recActive) {
    recCalSegments.push({
      segmentId: calSegmentId,
      absStartFrame: recTotalWritten,
      startRow: recCount,
      elapsedSec: Number(recElapsedSec.toFixed(4)),
      reason,
      sourceMode,
      calibration: getCalibrationSnapshot(),
    });
  }
}

function emitCalibrationState(overrideNotice) {
  self.postMessage({
    type: 'cal_state',
    calSegmentId,
    calMode,
    calProvenance,
    calArmed,
    calPtCount,
    calDraftCount,
    calSyncSeq: lastCalSyncSeq,
    rulerMeters: calArmed === 'ruler' ? calDraftRulerMeters : calRulerMeters,
    planeW: calArmed === 'homography' ? calDraftPlaneW : calPlaneW,
    planeH: calArmed === 'homography' ? calDraftPlaneH : calPlaneH,
    presetName: calPresetName,
    autoGLocked: calAutoGLocked,
    calibrated: isMetricCalibrated(),
    ptsNorm: calPtCount > 0
      ? Array.from({ length: calPtCount }, (_, i) => [calPtsNormX[i], calPtsNormY[i]])
      : [],
    summary: formatCalibrationSummary(overrideNotice),
  });
}

function clearCalibrationState() {
  calMode = 'none';
  calProvenance = 'none';
  calArmed = 'none';
  calPtCount = 0;
  calDraftCount = 0;
  calPresetName = '';
  calAutoGNormH = 0.0;
  calAutoGLocked = false;
  calHomographyValid = false;
  calHomography.fill(0.0);
  calHomographyInv.fill(0.0);
}

function applySceneCalibrationPreset(presetKey) {
  clearCalibrationState();
  calPresetName = presetKey || '';
  if (presetKey === 'carom_billiards_impulses') {
    // Assumed illustrative Carom table inner plane (2.84 m x 1.42 m)
    calPlaneW = 2.84;
    calPlaneH = 1.42;
    calPtsNormX[0] = 0.18; calPtsNormY[0] = 0.21;
    calPtsNormX[1] = 0.82; calPtsNormY[1] = 0.21;
    calPtsNormX[2] = 0.94; calPtsNormY[2] = 0.87;
    calPtsNormX[3] = 0.06; calPtsNormY[3] = 0.87;
    calPtCount = 4;
    calHomographyValid = solveHomography4Pt(calPlaneW, calPlaneH);
    calMode = calHomographyValid ? 'homography' : 'none';
    calProvenance = calHomographyValid ? 'assumed' : 'none';
  } else if (presetKey === 'roundabout_occlusion') {
    // Assumed illustrative multi-lane highway deck (20.0 m x 32.0 m)
    calPlaneW = 20.00;
    calPlaneH = 32.00;
    calPtsNormX[0] = 0.20; calPtsNormY[0] = 0.17;
    calPtsNormX[1] = 0.78; calPtsNormY[1] = 0.17;
    calPtsNormX[2] = 0.93; calPtsNormY[2] = 0.90;
    calPtsNormX[3] = 0.07; calPtsNormY[3] = 0.90;
    calPtCount = 4;
    calHomographyValid = solveHomography4Pt(calPlaneW, calPlaneH);
    calMode = calHomographyValid ? 'homography' : 'none';
    calProvenance = calHomographyValid ? 'assumed' : 'none';
  } else if (presetKey === 'juggling_ballistic') {
    // Newtonian Auto-g (9.80665 m/s^2) — requires >= 2 independent free-fall arcs before locking!
    calMode = 'auto_g';
    calProvenance = 'assumed';
    if (telemView && telemView[TELEM_GLOBAL + 1] >= AUTO_G_LOCK_CONF && telemView[TELEM_GLOBAL + 0] > 50.0) {
      calAutoGNormH = telemView[TELEM_GLOBAL + 0] / frameH;
      calAutoGLocked = true;
    }
  } else if (presetKey === 'galileo_pendulum_peg') {
    // Assumed Galileo pendulum stand chord (0.65 m)
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 0.65;
    calPtsNormX[0] = 0.50; calPtsNormY[0] = 0.14;
    calPtsNormX[1] = 0.50; calPtsNormY[1] = 0.78;
    calPtCount = 2;
  } else if (presetKey === 'harmonic_metronome_phase') {
    // Assumed Nikko mechanical metronome tower body (0.20 m)
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 0.20;
    calPtsNormX[0] = 0.50; calPtsNormY[0] = 0.18;
    calPtsNormX[1] = 0.50; calPtsNormY[1] = 0.84;
    calPtCount = 2;
  } else if (presetKey === 'cologne_motorway_occlusion') {
    // Assumed illustrative multi-lane A4 motorway deck (28.0 m x 48.0 m)
    calPlaneW = 28.00;
    calPlaneH = 48.00;
    calPtsNormX[0] = 0.22; calPtsNormY[0] = 0.16;
    calPtsNormX[1] = 0.78; calPtsNormY[1] = 0.16;
    calPtsNormX[2] = 0.94; calPtsNormY[2] = 0.90;
    calPtsNormX[3] = 0.06; calPtsNormY[3] = 0.90;
    calPtCount = 4;
    calHomographyValid = solveHomography4Pt(calPlaneW, calPlaneH);
    calMode = calHomographyValid ? 'homography' : 'none';
    calProvenance = calHomographyValid ? 'assumed' : 'none';
  } else if (presetKey === 'chaotic_magnetic_pendulum') {
    // Assumed magnetic pendulum suspension chord (0.35 m)
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 0.35;
    calPtsNormX[0] = 0.50; calPtsNormY[0] = 0.12;
    calPtsNormX[1] = 0.50; calPtsNormY[1] = 0.80;
    calPtCount = 2;
  } else if (presetKey === 'coupled_metronomes_sync') {
    // Assumed coupled metronome platform span (0.45 m)
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 0.45;
    calPtsNormX[0] = 0.22; calPtsNormY[0] = 0.58;
    calPtsNormX[1] = 0.78; calPtsNormY[1] = 0.58;
    calPtCount = 2;
  } else if (presetKey === 'newtons_cradle_impulses') {
    // Assumed 5-sphere Newton's cradle span (0.18 m)
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 0.18;
    calPtsNormX[0] = 0.28; calPtsNormY[0] = 0.62;
    calPtsNormX[1] = 0.72; calPtsNormY[1] = 0.62;
    calPtCount = 2;
  } else if (presetKey === 'real_daphnia_laser_piv') {
    // Surveyed laboratory laser-sheet PIV FOV (32 mm horizontal span)
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 0.032;
    calPtsNormX[0] = 0.0; calPtsNormY[0] = 0.50;
    calPtsNormX[1] = 1.0; calPtsNormY[1] = 0.50;
    calPtCount = 2;
  } else if (
    presetKey === 'model_rocket_full_flights' ||
    presetKey === 'rocket_vtvl_full_flight' ||
    presetKey === 'water_rocket_parachute_flights'
  ) {
    // Assumed full-flight vertical ascent/descent span for real-camera rocketry showcases
    const rocketRulerPresets = {
      model_rocket_full_flights: [15.00, 0.88, 0.12],
      rocket_vtvl_full_flight: [10.00, 0.92, 0.16],
      water_rocket_parachute_flights: [12.00, 0.90, 0.12],
    };
    const [meters, y0, y1] = rocketRulerPresets[presetKey];
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = meters;
    calPtsNormX[0] = 0.50; calPtsNormY[0] = y0;
    calPtsNormX[1] = 0.50; calPtsNormY[1] = y1;
    calPtCount = 2;
  } else if (presetKey === 'testbench') {
    // Synthetic testbench 1.00 m optical bench reference
    calMode = 'ruler';
    calProvenance = 'assumed';
    calRulerMeters = 1.00;
    calPtsNormX[0] = 0.20; calPtsNormY[0] = 0.50;
    calPtsNormX[1] = 0.80; calPtsNormY[1] = 0.50;
    calPtCount = 2;
  }
  // Custom uploaded videos, webcams, and live feeds remain UNCALIBRATED (pixel units) until calibrated!
  recordCalibrationSegmentTransition(`preset:${calPresetName || 'uncalibrated'}`);
  emitCalibrationState();
}

// ============================================================================
// Blind Pixel Pillar Testbench Scenarios & 1-to-1 Ground-Truth Swap Oracle
// ============================================================================

const PILLARS_WIDE = [
  { x0: 88, x1: 136, y0: 14, y1: 166 },
  { x0: 184, x1: 232, y0: 14, y1: 166 },
];
const PILLARS_SHORT_GAP = [
  { x0: 102, x1: 118, y0: 14, y1: 166 },
  { x0: 202, x1: 218, y0: 14, y1: 166 },
];
const PILLARS_SWARM = [
  { x0: 108, x1: 124, y0: 14, y1: 166 },
  { x0: 196, x1: 212, y0: 14, y1: 166 },
];
const PILLARS_NONE = [];

function getActivePillars() {
  if (benchScenario === 'pillar_crossing') return PILLARS_WIDE;
  if (benchScenario === 'short_gap_bridge') return PILLARS_SHORT_GAP;
  if (benchScenario === 'multi_target_swarm') return PILLARS_SWARM;
  return PILLARS_NONE;
}

function getActiveSimDiscCount() {
  return benchScenario === 'multi_target_swarm' ? 12 : 4;
}

// ============================================================================
// CHRONOS-1000 High-Density Blind Scene Generators (Matched to benchmark_1000.mjs Q1-Q3)
// ============================================================================

const C1K_REF_W = 640.0;
const C1K_REF_H = 360.0;
const C1K_BG_U32 = (new Uint8Array(new Uint32Array([0x11223344]).buffer)[0] === 0x44)
  ? 0xFF160E0A
  : 0x0A0E16FF;

let c1kPixelBuf = new Uint8ClampedArray(640 * 360 * 4);
let c1kPixelU32 = new Uint32Array(c1kPixelBuf.buffer);
let c1kPixelImgData = new ImageData(c1kPixelBuf, 640, 360);

const C1K_MAX_PTS = 1208;
const c1kX = new Float32Array(C1K_MAX_PTS);
const c1kY = new Float32Array(C1K_MAX_PTS);
const c1kVx = new Float32Array(C1K_MAX_PTS);
const c1kVy = new Float32Array(C1K_MAX_PTS);
const c1kBaseX = new Float32Array(C1K_MAX_PTS);
const c1kBaseY = new Float32Array(C1K_MAX_PTS);
const c1kModeShape = new Float32Array(C1K_MAX_PTS);
const c1kPeak = new Float32Array(C1K_MAX_PTS);
const c1kTintR = new Float32Array(C1K_MAX_PTS);
const c1kTintG = new Float32Array(C1K_MAX_PTS);
const c1kTintB = new Float32Array(C1K_MAX_PTS);
const c1kBoidCellHead = new Int16Array(60 * 35);
const c1kBoidCellNext = new Int16Array(1100);
const c1kVortexList = [];
const c1kRigidDefs = [
  { cx: 140.0, cy: 135.0, vx: 16.0, vy: 8.0, omega: 0.90, theta: 0.0 },
  { cx: 330.0, cy: 235.0, vx: 8.0, vy: -10.0, omega: -0.70, theta: 0.0 },
  { cx: 505.0, cy: 135.0, vx: -14.0, vy: 12.0, omega: 0.60, theta: 0.0 },
];
const c1kPolyOffsetsX = new Float32Array(36);
const c1kPolyOffsetsY = new Float32Array(36);
let c1kSimCount = 0;
let c1kTimeSec = 0.0;

(function initC1kPolyOffsets() {
  for (let k = 0; k < 24; k++) {
    const a = (2.0 * Math.PI * k) / 24.0;
    const r = 42.0 + ((k & 1) === 0 ? 2.0 : -2.0);
    c1kPolyOffsetsX[k] = r * Math.cos(a);
    c1kPolyOffsetsY[k] = r * Math.sin(a);
  }
  for (let k = 0; k < 12; k++) {
    const a = (2.0 * Math.PI * k) / 12.0 + Math.PI / 12.0;
    c1kPolyOffsetsX[24 + k] = 22.0 * Math.cos(a);
    c1kPolyOffsetsY[24 + k] = 22.0 * Math.sin(a);
  }
})();

function seedC1kParticleVisual(slot, seedId) {
  const h1 = Math.abs(Math.sin(seedId * 12.9898 + 0.5) * 43758.5453) % 1.0;
  const h2 = Math.abs(Math.sin(seedId * 78.233 + 1.9) * 43758.5453) % 1.0;
  c1kPeak[slot] = 212.0 + h1 * 38.0;
  c1kTintR[slot] = 0.88 + 0.22 * h2;
  c1kTintG[slot] = 0.94 + 0.12 * (1.0 - h1);
  c1kTintB[slot] = 1.05 - 0.18 * h2;
}

function ensureC1kPixelSurface(w, h) {
  if (c1kPixelImgData.width === w && c1kPixelImgData.height === h) return;
  c1kPixelBuf = new Uint8ClampedArray(w * h * 4);
  c1kPixelU32 = new Uint32Array(c1kPixelBuf.buffer);
  c1kPixelImgData = new ImageData(c1kPixelBuf, w, h);
}

function c1kVortexVelocity(x, y, out) {
  let u = 0.0;
  let v = 0.0;
  const invRc2 = 1.0 / (48.0 * 48.0);
  for (let i = 0; i < c1kVortexList.length; i++) {
    const vo = c1kVortexList[i];
    const dx = x - vo.cx;
    const dy = y - vo.cy;
    const r2 = dx * dx + dy * dy;
    if (r2 > 1e-6) {
      const f = (vo.k / r2) * (1.0 - Math.exp(-r2 * invRc2));
      u += -dy * f;
      v += dx * f;
    }
  }
  out[0] = u;
  out[1] = v;
}

const _c1kK1 = new Float32Array(2);
const _c1kK2 = new Float32Array(2);
const _c1kK3 = new Float32Array(2);
const _c1kK4 = new Float32Array(2);

function resetChronos1000Scenario() {
  c1kTimeSec = 0.0;
  c1kSimCount = 0;
  if (benchScenario === 'vortex_swarm_1200') {
    c1kVortexList.length = 0;
    const K = 4600.0;
    const bases = [
      [210.0, 180.0, K],
      [430.0, 180.0, -K],
    ];
    for (const mx of [-1, 0, 1]) {
      for (const my of [-1, 0, 1]) {
        const sign = (mx !== 0 ? -1 : 1) * (my !== 0 ? -1 : 1);
        for (const [bx, by, bk] of bases) {
          const vx = mx === 0 ? bx : mx === -1 ? 2.0 * 14.0 - bx : 2.0 * 626.0 - bx;
          const vy = my === 0 ? by : my === -1 ? 2.0 * 14.0 - by : 2.0 * 346.0 - by;
          c1kVortexList.push({ cx: vx, cy: vy, k: bk * sign });
        }
      }
    }
    const nx = 48;
    const ny = 25;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i;
        const h1 = Math.sin(idx * 12.9898 + 1.23) * 43758.5453;
        const h2 = Math.sin(idx * 78.233 + 4.56) * 43758.5453;
        c1kX[idx] = 22.0 + ((i + 0.5) * (618.0 - 22.0)) / nx + (h1 - Math.floor(h1) - 0.5) * 3.0;
        c1kY[idx] = 20.0 + ((j + 0.5) * (340.0 - 20.0)) / ny + (h2 - Math.floor(h2) - 0.5) * 3.0;
        seedC1kParticleVisual(idx, idx + 1);
      }
    }
    c1kSimCount = 1200;
  } else if (benchScenario === 'murmuration_1200') {
    c1kRigidDefs[0] = { cx: 140.0, cy: 135.0, vx: 16.0, vy: 8.0, omega: 0.90, theta: 0.0 };
    c1kRigidDefs[1] = { cx: 330.0, cy: 235.0, vx: 8.0, vy: -10.0, omega: -0.70, theta: 0.0 };
    c1kRigidDefs[2] = { cx: 505.0, cy: 135.0, vx: -14.0, vy: 12.0, omega: 0.60, theta: 0.0 };
    let bSeed = 0;
    for (let b = 0; b < 3; b++) {
      for (let k = 0; k < 36; k++) {
        seedC1kParticleVisual(b * 36 + k, 1 + b * 40 + k);
        bSeed++;
      }
    }
    const candX = [];
    const candY = [];
    const nx = 50;
    const ny = 28;
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i;
        const h1 = Math.sin(idx * 17.13 + 0.7) * 43758.5453;
        const h2 = Math.sin(idx * 91.7 + 2.1) * 43758.5453;
        const px = 22.0 + ((i + 0.5) * 596.0) / nx + (h1 - Math.floor(h1) - 0.5) * 3.0;
        const py = 20.0 + ((j + 0.5) * 320.0) / ny + (h2 - Math.floor(h2) - 0.5) * 3.0;
        let ok = true;
        for (let b = 0; b < 3; b++) {
          if (Math.hypot(px - c1kRigidDefs[b].cx, py - c1kRigidDefs[b].cy) < 58.0) {
            ok = false;
            break;
          }
        }
        if (ok) {
          candX.push(px);
          candY.push(py);
        }
      }
    }
    const step = candX.length / 1100.0;
    for (let m = 0; m < 1100; m++) {
      const si = Math.floor(m * step);
      const px = candX[si];
      c1kX[108 + m] = px;
      c1kY[108 + m] = candY[si];
      c1kVx[108 + m] = 32.0;
      c1kVy[108 + m] = 6.0 * Math.sin(0.02 * px);
      seedC1kParticleVisual(108 + m, 200 + m);
    }
    c1kSimCount = 1208;
  } else if (benchScenario === 'modal_beam_1152') {
    const nx = 48;
    const ny = 24;
    for (let j = 0; j < ny; j++) {
      const y0 = 72.0 + (j * (288.0 - 72.0)) / (ny - 1);
      for (let i = 0; i < nx; i++) {
        const idx = j * nx + i;
        const x0 = 44.0 + (i * (596.0 - 44.0)) / (nx - 1);
        const u = (x0 - 320.0) / 276.0;
        c1kBaseX[idx] = x0;
        c1kBaseY[idx] = y0;
        c1kModeShape[idx] = Math.sin(0.5 * Math.PI * u);
        c1kX[idx] = x0;
        c1kY[idx] = y0;
        seedC1kParticleVisual(idx, idx + 1);
      }
    }
    c1kSimCount = 1152;
  }
}

function stepAndRenderChronos1000Scene(dtSec) {
  const dt = Math.min(0.05, Math.max(0.005, dtSec));
  const sx = frameW / C1K_REF_W;
  const sy = frameH / C1K_REF_H;
  ensureC1kPixelSurface(frameW, frameH);
  c1kPixelU32.fill(C1K_BG_U32);

  if (benchScenario === 'vortex_swarm_1200') {
    for (let i = 0; i < 1200; i++) {
      const x = c1kX[i];
      const y = c1kY[i];
      c1kVortexVelocity(x, y, _c1kK1);
      c1kVortexVelocity(x + 0.5 * dt * _c1kK1[0], y + 0.5 * dt * _c1kK1[1], _c1kK2);
      c1kVortexVelocity(x + 0.5 * dt * _c1kK2[0], y + 0.5 * dt * _c1kK2[1], _c1kK3);
      c1kVortexVelocity(x + dt * _c1kK3[0], y + dt * _c1kK3[1], _c1kK4);
      let nx = x + ((_c1kK1[0] + 2.0 * _c1kK2[0] + 2.0 * _c1kK3[0] + _c1kK4[0]) / 6.0) * dt;
      let ny = y + ((_c1kK1[1] + 2.0 * _c1kK2[1] + 2.0 * _c1kK3[1] + _c1kK4[1]) / 6.0) * dt;
      if (nx < 14.0) nx = 14.0;
      else if (nx > 626.0) nx = 626.0;
      if (ny < 14.0) ny = 14.0;
      else if (ny > 346.0) ny = 346.0;
      c1kX[i] = nx;
      c1kY[i] = ny;
    }
  } else if (benchScenario === 'murmuration_1200') {
    const t = c1kTimeSec;
    for (let b = 0; b < 3; b++) {
      const bd = c1kRigidDefs[b];
      bd.cx += bd.vx * dt;
      bd.cy += bd.vy * dt;
      bd.theta += bd.omega * dt;
      if (bd.cx < 62.0) { bd.cx = 62.0; bd.vx = Math.abs(bd.vx); }
      else if (bd.cx > 578.0) { bd.cx = 578.0; bd.vx = -Math.abs(bd.vx); }
      if (bd.cy < 62.0) { bd.cy = 62.0; bd.vy = Math.abs(bd.vy); }
      else if (bd.cy > 298.0) { bd.cy = 298.0; bd.vy = -Math.abs(bd.vy); }
      const ct = Math.cos(bd.theta);
      const st = Math.sin(bd.theta);
      for (let k = 0; k < 36; k++) {
        const lx = c1kPolyOffsetsX[k];
        const ly = c1kPolyOffsetsY[k];
        c1kX[b * 36 + k] = bd.cx + lx * ct - ly * st;
        c1kY[b * 36 + k] = bd.cy + lx * st + ly * ct;
      }
    }
    c1kBoidCellHead.fill(-1);
    for (let m = 0; m < 1100; m++) {
      const idx = 108 + m;
      const gx = Math.max(0, Math.min(59, Math.floor(c1kX[idx] / 12.0)));
      const gy = Math.max(0, Math.min(34, Math.floor(c1kY[idx] / 12.0)));
      const cIdx = gy * 60 + gx;
      c1kBoidCellNext[m] = c1kBoidCellHead[cIdx];
      c1kBoidCellHead[cIdx] = m;
    }
    for (let m = 0; m < 1100; m++) {
      const idx = 108 + m;
      const bx = c1kX[idx];
      const by = c1kY[idx];
      let ux = 36.0 * Math.cos(0.45 * t) + 18.0 * Math.sin(0.012 * by + 1.2 * t);
      let uy = 22.0 * Math.sin(0.65 * t) + 14.0 * Math.cos(0.009 * bx - 0.9 * t);
      if (bx < 40.0) ux += (40.0 - bx) * 2.5;
      if (bx > 600.0) ux -= (bx - 600.0) * 2.5;
      if (by < 36.0) uy += (36.0 - by) * 2.5;
      if (by > 324.0) uy -= (by - 324.0) * 2.5;
      for (let b = 0; b < 3; b++) {
        const dx = bx - c1kRigidDefs[b].cx;
        const dy = by - c1kRigidDefs[b].cy;
        const d = Math.hypot(dx, dy);
        if (d < 58.0 && d > 1e-3) {
          const push = (58.0 - d) * 4.5;
          ux += (dx / d) * push;
          uy += (dy / d) * push;
        }
      }
      const gx = Math.max(0, Math.min(59, Math.floor(bx / 12.0)));
      const gy = Math.max(0, Math.min(34, Math.floor(by / 12.0)));
      for (let oy = -1; oy <= 1; oy++) {
        const yy = gy + oy;
        if (yy < 0 || yy >= 35) continue;
        for (let ox = -1; ox <= 1; ox++) {
          const xx = gx + ox;
          if (xx < 0 || xx >= 60) continue;
          let other = c1kBoidCellHead[yy * 60 + xx];
          while (other !== -1) {
            if (other !== m) {
              const jIdx = 108 + other;
              const dx = bx - c1kX[jIdx];
              const dy = by - c1kY[jIdx];
              const d2 = dx * dx + dy * dy;
              if (d2 < 90.25 && d2 > 1e-4) {
                const d = Math.sqrt(d2);
                const push = (9.5 - d) * 6.0;
                ux += (dx / d) * push;
                uy += (dy / d) * push;
              }
            }
            other = c1kBoidCellNext[other];
          }
        }
      }
      let nvx = c1kVx[idx] * 0.65 + ux * 0.35;
      let nvy = c1kVy[idx] * 0.65 + uy * 0.35;
      const spd = Math.hypot(nvx, nvy);
      if (spd > 75.0) {
        nvx *= 75.0 / spd;
        nvy *= 75.0 / spd;
      }
      c1kVx[idx] = nvx;
      c1kVy[idx] = nvy;
    }
    for (let m = 0; m < 1100; m++) {
      const idx = 108 + m;
      let nbx = Math.min(624.0, Math.max(16.0, c1kX[idx] + c1kVx[idx] * dt));
      let nby = Math.min(344.0, Math.max(16.0, c1kY[idx] + c1kVy[idx] * dt));
      for (let b = 0; b < 3; b++) {
        const dx = nbx - c1kRigidDefs[b].cx;
        const dy = nby - c1kRigidDefs[b].cy;
        const d = Math.hypot(dx, dy);
        if (d < 50.0 && d > 1e-3) {
          nbx = c1kRigidDefs[b].cx + (dx / d) * 50.0;
          nby = c1kRigidDefs[b].cy + (dy / d) * 50.0;
        }
      }
      c1kX[idx] = nbx;
      c1kY[idx] = nby;
    }
  } else if (benchScenario === 'modal_beam_1152') {
    const phase = Math.sin(2.0 * Math.PI * 3.00 * c1kTimeSec);
    for (let i = 0; i < 1152; i++) {
      c1kX[i] = c1kBaseX[i];
      c1kY[i] = c1kBaseY[i] + 1.25 * c1kModeShape[i] * phase;
    }
  }

  c1kTimeSec += dt;

  // Stamp textured Gaussian sub-pixel blobs matching benchmark_1000.mjs appearance
  const invTwoSig2 = 1.0 / (2.0 * (0.85 * sx) * (0.85 * sx));
  const r2Max = 4.84 * sx * sx;
  const radPx = Math.max(1, Math.ceil(2.2 * sx));
  for (let i = 0; i < c1kSimCount; i++) {
    const cx = c1kX[i] * sx;
    const cy = c1kY[i] * sy;
    const x0 = Math.max(0, Math.floor(cx) - radPx);
    const x1 = Math.min(frameW - 1, Math.floor(cx) + radPx);
    const y0 = Math.max(0, Math.floor(cy) - radPx);
    const y1 = Math.min(frameH - 1, Math.floor(cy) + radPx);
    const pk = c1kPeak[i];
    const tR = c1kTintR[i];
    const tG = c1kTintG[i];
    const tB = c1kTintB[i];
    for (let py = y0; py <= y1; py++) {
      const dy = py - cy;
      for (let px = x0; px <= x1; px++) {
        const dx = px - cx;
        const r2 = dx * dx + dy * dy;
        if (r2 > r2Max) continue;
        const g = Math.exp(-r2 * invTwoSig2);
        const off = (py * frameW + px) * 4;
        const r = Math.min(255, Math.round(10 + pk * tR * g));
        const gr = Math.min(255, Math.round(14 + pk * tG * g));
        const b = Math.min(255, Math.round(22 + pk * tB * g));
        if (r > c1kPixelBuf[off]) c1kPixelBuf[off] = r;
        if (gr > c1kPixelBuf[off + 1]) c1kPixelBuf[off + 1] = gr;
        if (b > c1kPixelBuf[off + 2]) c1kPixelBuf[off + 2] = b;
      }
    }
  }

  simCtx.putImageData(c1kPixelImgData, 0, 0);
  return c1kPixelBuf;
}

const simDiscs = [
  { id: 0,  x: 52,  y: 56,  baseX: 66,  baseY: 56,  vx: 3.0,  vy: 0.0, ballistic: false, r: '#f83a44', core: '#ffe66e' },
  { id: 1,  x: 172, y: 56,  baseX: 254, baseY: 56,  vx: -3.0, vy: 0.0, ballistic: false, r: '#1ee1fa', core: '#0f2d55' },
  { id: 2,  x: 148, y: 124, baseX: 66,  baseY: 124, vx: 3.0,  vy: 0.0, ballistic: false, r: '#32f270', core: '#ebffeb' },
  { id: 3,  x: 268, y: 124, baseX: 254, baseY: 124, vx: -3.0, vy: 0.0, ballistic: false, r: '#e840f8', core: '#5af5ff' },
  { id: 4,  x: 54,  y: 78,  baseX: 54,  baseY: 78,  vx: 2.9,  vy: 0.0, ballistic: false, r: '#ff922b', core: '#fff3bf' },
  { id: 5,  x: 266, y: 78,  baseX: 266, baseY: 78,  vx: -2.9, vy: 0.0, ballistic: false, r: '#38d9a9', core: '#0b4f4a' },
  { id: 6,  x: 88,  y: 104, baseX: 88,  baseY: 104, vx: 3.0,  vy: 0.0, ballistic: false, r: '#a9e34b', core: '#1e3a10' },
  { id: 7,  x: 232, y: 104, baseX: 232, baseY: 104, vx: -3.0, vy: 0.0, ballistic: false, r: '#748ffc', core: '#eef2ff' },
  { id: 8,  x: 44,  y: 130, baseX: 44,  baseY: 130, vx: 3.1,  vy: 0.0, ballistic: false, r: '#f06595', core: '#fff0f6' },
  { id: 9,  x: 276, y: 130, baseX: 276, baseY: 130, vx: -3.1, vy: 0.0, ballistic: false, r: '#ffd43b', core: '#3b2400' },
  { id: 10, x: 68,  y: 154, baseX: 68,  baseY: 154, vx: 3.2,  vy: 0.0, ballistic: false, r: '#66d9e8', core: '#0c3547' },
  { id: 11, x: 252, y: 154, baseX: 252, baseY: 154, vx: -3.2, vy: 0.0, ballistic: false, r: '#c0eb75', core: '#26380a' },
];

// Pre-allocated model rocket multi-stage buffers (0: boost, 1: coast/apogee, 2: drogue recovery)
const rocketPhase = new Uint8Array(4);
const rocketBoostTicks = new Uint16Array(4);
const SMOKE_PUFFS = 40;
const smokeX = new Float32Array(SMOKE_PUFFS);
const smokeY = new Float32Array(SMOKE_PUFFS);
const smokeR = new Float32Array(SMOKE_PUFFS);
const smokeAlpha = new Float32Array(SMOKE_PUFFS);
let smokeWriteIdx = 0;

// Pre-allocated 65,536-ID ownership table (eliminates & 255 ID wrap aliasing)
const MAX_ORACLE_IDS = 65536;
const oracleTrackOwner = new Int16Array(MAX_ORACLE_IDS).fill(-1);
const oracleCandTrackSlot = new Int32Array(MAX_TARGETS * MAX_TARGETS);
const oracleCandDiscIdx = new Int32Array(MAX_TARGETS * MAX_TARGETS);
const oracleCandDist = new Float32Array(MAX_TARGETS * MAX_TARGETS);
let oraclePadicSwaps = 0;
let inPillarCollisions = 0;
let simTimeSec = 0.0;
let pendingSaccadeFrames = 0;
let pendingAirSnapFrames = 0;

const TARGET_COLORS = [
  '#00f0ff', '#ff4d6d', '#38f27a', '#ffb703',
  '#c77dff', '#4cc9f0', '#f72585', '#aacc00',
  '#ff7b00', '#06d6a0', '#e0aaff', '#ffd166',
  '#72efdd', '#ff8fab', '#9ef01a', '#80ffdb',
];

const ZONE_COLORS = ['#00f0ff', '#ffb703', '#38f27a', '#ff4d6d'];

function clearVisualQueues() {
  cancelPendingHudRender();
  swarmRingValid.fill(0);
  covRingValid.fill(0);
  bosRingValid.fill(0);
  w2CoreRingValid.fill(0);
  vibroRingValid.fill(0);
  hodgeRingValid.fill(0);
  consRingValid.fill(0);
  depthRingValid.fill(0);
  windRingValid.fill(0);
  elastoRingValid.fill(0);
  manualConsTestActive = false;
  tbRsFrameSeq = 0;
  tbRsAudioSampleCursor = 0;
  ringIngestFrameId.fill(0);
  ringIngestFrameValid.fill(0);
  ringIngestTimeMs.fill(-1.0);
  ringIngestPtsMs.fill(-1.0);
  clearActiveInspectionSelection();
  lastSwarmCopyMs = 0.0;
  lastFieldRenderMs = 0.0;
  ingestElapsedMs = 0.0;
  latestCausalTimeMs = 0.0;
  latestCausalPtsMs = -1.0;
  shockTriggerFrame.fill(0);
  shockTriggerArmed.fill(0);
  shockAge.fill(-1);
  shockInDeg.fill(0.0);
  shockDeflectDeg.fill(0.0);
  shockRestitution.fill(0.0);
  lastStrikeDeflectDeg = 0.0;
  lastStrikeRestitution = 0.0;
  lastStrikeTid = 0;
  lastStrikeFrame = -1;
  lastStrikeValid = false;
  lastScatterDeg = 0.0;
  lastScatterFrame = -1;
  slotLastStrikeX.fill(0.0);
  slotLastStrikeY.fill(0.0);
  prevRectHeading.fill(0.0);
  prevRectTid.fill(0);
  prevRectFrame.fill(0);
  prevRectValid.fill(0);
  slotRectAngVel.fill(0.0);
  padFlashAlpha.fill(0.0);
  apexLockTid.fill(0);
  apexLockErrPx.fill(-1.0);
  apexLockHold.fill(0);
  lastScoredApexErrPx = -1.0;
  histTremorR.fill(0.0);
  histTremorH.fill(0.0);
  histPhaseValid.fill(0);
  histStrobe.fill(0);
  histPipTid.fill(-1);
  histLambda2.fill(0.0);
  scopeWriteIdx = 0;
  strobeFlashAlpha = 0.0;
  totalBridgedEvents = 0;
  totalStrikeEvents = 0;
  trailOwnerTid.fill(-1);
  trailHead.fill(0);
  trailCount.fill(0);
  trailLastFrame.fill(0);
  slotHoldLeft.fill(0);
  slotHoldTid.fill(0);
  slotIdleDrawn.fill(0);
  slotHoldVeilDrawn.fill(0);
  slotSpeedPillOn.fill(0);
  slotSmoothedSpeed.fill(0.0);
  slotSmoothedHeading.fill(0.0);
  slotSmoothedOmega.fill(0.0);
  slotSmoothedEca.fill(0.0);
  slotLastAgeFrames.fill(0);
  slotLastCoast.fill(0);
  slotLastReacq.fill(0);
  slotLastVal.fill(0);
  slotLastSigHi.fill(0);
  slotLastSigLo.fill(0);
  slotLastFreqHz.fill(0.0);
  slotLastMetricValid.fill(0.0);
  slotLastPivotValid.fill(0);
  slotLastPendSwing.fill(0.0);
  slotLastPendRad.fill(0.0);
  phaseHistWrite = 0;
  phaseHistCount = 0;
  phaseHistTid.fill(-1);
  phaseHistX.fill(0.0);
  phaseHistY.fill(0.0);
  phaseHistVx.fill(0.0);
  phaseHistVy.fill(0.0);
  phaseHistE.fill(0.0);
  phaseHistEnsK.fill(0.0);
  phaseHistKurR.fill(0.0);
  phaseHistKurN.fill(0);
  phaseHistRegime.fill(0);
  phaseHistJump.fill(0);
  lastPhysicsPhaseRenderMs = 0.0;
}

function resetPersistenceState() {
  contTid.fill(0);
  contSlot.fill(-1);
  contFirstSeq.fill(0);
  contLastSeq.fill(0);
  contTotalFrames.fill(0);
  contTotalSec.fill(0.0);
  contRelocks.fill(0);
  contSegCount.fill(0);
  contSegStart.fill(0);
  contSegEnd.fill(0);
  contWasActive.fill(0);
  contBinMaxActive.fill(0);
  contBinFrames.fill(0);
  contCount = 0;
  persistSampleSeq = 0;
  sessionElapsedSec = 0.0;
  lastPersistRenderMs = 0.0;
  contBinStrideFrames = 15;
  contBinHead = 0;
  resetRecEvents();
}

function updatePersistenceState(fpsDt) {
  if (!telemView) return;
  persistSampleSeq++;
  sessionElapsedSec += fpsDt;
  const seq = persistSampleSeq;
  const activeNow = Math.max(0, Math.min(MAX_TARGETS, Math.round(telemView[3])));

  for (let m = 0; m < MAX_TARGETS; m++) {
    const b = 32 + m * 56;
    if (telemView[b] <= 0.5) continue;
    const tid = Math.round(telemView[b + 1]);
    if (tid <= 0) continue;

    let idx = -1;
    for (let t = 0; t < contCount; t++) {
      if (contTid[t] === tid) {
        idx = t;
        break;
      }
    }
    if (idx < 0) {
      if (contCount < CONT_MAX_TRACKS) {
        idx = contCount++;
      } else {
        let evictIdx = -1;
        let evictFrames = 1e9;
        for (let t = 0; t < CONT_MAX_TRACKS; t++) {
          if (contLastSeq[t] < seq - 2 && contTotalFrames[t] < evictFrames) {
            evictFrames = contTotalFrames[t];
            evictIdx = t;
          }
        }
        if (evictIdx < 0) {
          for (let t = 0; t < CONT_MAX_TRACKS; t++) {
            if (contTotalFrames[t] < evictFrames) {
              evictFrames = contTotalFrames[t];
              evictIdx = t;
            }
          }
        }
        idx = evictIdx >= 0 ? evictIdx : 0;
      }
      contTid[idx] = tid;
      contSlot[idx] = m;
      contFirstSeq[idx] = seq;
      contLastSeq[idx] = seq;
      contTotalFrames[idx] = 1;
      contTotalSec[idx] = fpsDt;
      contRelocks[idx] = 0;
      contSegCount[idx] = 1;
      contSegStart[idx * CONT_MAX_SEGS] = seq;
      contSegEnd[idx * CONT_MAX_SEGS] = seq;
      contWasActive[idx] = 1;
    } else {
      const gap = seq - contLastSeq[idx];
      contSlot[idx] = m;
      contLastSeq[idx] = seq;
      contTotalFrames[idx]++;
      contTotalSec[idx] += fpsDt;
      if (!contWasActive[idx] && gap > 6) {
        contRelocks[idx]++;
        const sc = contSegCount[idx];
        if (sc < CONT_MAX_SEGS) {
          contSegStart[idx * CONT_MAX_SEGS + sc] = seq;
          contSegEnd[idx * CONT_MAX_SEGS + sc] = seq;
          contSegCount[idx] = sc + 1;
        } else {
          contSegEnd[idx * CONT_MAX_SEGS + CONT_MAX_SEGS - 1] = seq;
        }
      } else {
        const sc = contSegCount[idx];
        if (sc > 0) {
          contSegEnd[idx * CONT_MAX_SEGS + (sc - 1)] = seq;
        }
      }
      contWasActive[idx] = 1;
    }
  }

  for (let t = 0; t < contCount; t++) {
    if (contLastSeq[t] !== seq) contWasActive[t] = 0;
  }

  let binIdx = Math.floor(seq / contBinStrideFrames);
  if (binIdx >= CONT_HIST_BINS) {
    const half = CONT_HIST_BINS >> 1;
    for (let k = 0; k < half; k++) {
      contBinMaxActive[k] = Math.max(contBinMaxActive[k * 2], contBinMaxActive[k * 2 + 1]);
      contBinFrames[k] = contBinFrames[k * 2];
    }
    contBinMaxActive.fill(0, half);
    contBinFrames.fill(0, half);
    contBinStrideFrames *= 2;
    contBinHead = half;
    binIdx = Math.min(CONT_HIST_BINS - 1, Math.floor(seq / contBinStrideFrames));
  }
  if (binIdx >= contBinHead) contBinHead = binIdx + 1;
  contBinFrames[binIdx]++;
  if (activeNow > contBinMaxActive[binIdx]) contBinMaxActive[binIdx] = activeNow;
}

function formatPersistDurationSec(sec) {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return m > 0 ? `${m}m ${String(rem).padStart(2, '0')}s` : `${sec.toFixed(sec < 10 ? 1 : 0)}s`;
}

function renderPersistenceTimeline(force = false) {
  if (!persistenceCtx || !persistenceCanvas) return;
  const nowMs = performance.now();
  if (!force && nowMs - lastPersistRenderMs < 500.0) return;
  lastPersistRenderMs = nowMs;

  const W = persistenceCanvas.width;
  const H = persistenceCanvas.height;
  const ctx = persistenceCtx;

  ctx.fillStyle = '#070b13';
  ctx.fillRect(0, 0, W, H);

  const sparkX0 = 10;
  const sparkY0 = 20;
  const sparkW = W - 20;
  const sparkH = 44;

  ctx.fillStyle = '#9db4d4';
  ctx.font = '600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  const binsUsed = Math.max(1, contBinHead);
  const spanTxt = formatPersistDurationSec(sessionElapsedSec);
  ctx.fillText(`CONCURRENCY TIMELINE (0..${MAX_TARGETS} LOCKS | SPAN ${spanTxt})`, sparkX0, 13);

  ctx.fillStyle = '#0b1220';
  ctx.fillRect(sparkX0, sparkY0, sparkW, sparkH);
  ctx.strokeStyle = '#192438';
  ctx.lineWidth = 1;
  ctx.strokeRect(sparkX0 + 0.5, sparkY0 + 0.5, sparkW - 1, sparkH - 1);

  const midY = sparkY0 + sparkH * 0.5;
  ctx.strokeStyle = 'rgba(100, 128, 168, 0.25)';
  ctx.beginPath();
  ctx.moveTo(sparkX0, midY);
  ctx.lineTo(sparkX0 + sparkW, midY);
  ctx.stroke();

  if (persistSampleSeq > 0) {
    const barW = Math.max(1.5, sparkW / Math.max(16, binsUsed));
    for (let b = 0; b < binsUsed; b++) {
      const val = contBinMaxActive[b];
      if (val <= 0) continue;
      const frac = Math.min(1.0, val / MAX_TARGETS);
      const bh = Math.max(2, Math.round(frac * (sparkH - 4)));
      const bx = sparkX0 + (b / Math.max(1, binsUsed)) * sparkW;
      ctx.fillStyle = val >= 9 ? '#38f27a' : '#00f0ff';
      ctx.fillRect(bx, sparkY0 + sparkH - 2 - bh, Math.max(1.5, barW - 0.5), bh);
    }
  }

  // Color-coded event markers along sparkline top & canvas top edge (clickable to seek)
  if (recEvents.length > 0) {
    const denomSeq = Math.max(1, persistSampleSeq);
    const denomSpan = Math.max(0.05, sessionElapsedSec, recEventsSpanSec);
    for (let ei = 0; ei < recEvents.length; ei++) {
      const ev = recEvents[ei];
      const frac = ev.seq > 0
        ? Math.max(0, Math.min(1, ev.seq / denomSeq))
        : Math.max(0, Math.min(1, ev.tSec / denomSpan));
      const ex = sparkX0 + frac * sparkW;
      ctx.fillStyle = EVENT_TICK_COLORS[ev.kind] || '#ffd166';
      ctx.fillRect(ex - 0.75, sparkY0 + 1, 1.8, 6);
      ctx.fillRect(ex - 0.75, 0, 1.8, 3);
    }
  }

  const laneHeaderY = sparkY0 + sparkH + 15;
  ctx.fillStyle = '#9db4d4';
  ctx.fillText('TOP-10 TARGET CUSTODY SWIMLANES (LOCKED vs RE-ENGAGED)', sparkX0, laneHeaderY);

  const swimX0 = 56;
  const swimW = W - swimX0 - 48;
  const swimTop = laneHeaderY + 6;
  const rowH = 11.5;

  if (contCount === 0 || persistSampleSeq <= 0) {
    ctx.fillStyle = '#5c7394';
    ctx.fillText('AWAITING MULTI-SECOND TARGET CUSTODY...', swimX0, swimTop + 42);
    return;
  }

  for (let i = 0; i < contCount; i++) contRankIdx[i] = i;
  const topN = Math.min(10, contCount);
  for (let i = 0; i < topN; i++) {
    let best = i;
    for (let j = i + 1; j < contCount; j++) {
      const a = contRankIdx[j];
      const b = contRankIdx[best];
      if (
        contTotalFrames[a] > contTotalFrames[b] ||
        (contTotalFrames[a] === contTotalFrames[b] && contTid[a] < contTid[b])
      ) {
        best = j;
      }
    }
    if (best !== i) {
      const tmp = contRankIdx[i];
      contRankIdx[i] = contRankIdx[best];
      contRankIdx[best] = tmp;
    }
  }

  const maxSeq = Math.max(1, persistSampleSeq);
  for (let r = 0; r < topN; r++) {
    const ti = contRankIdx[r];
    const tid = contTid[ti];
    const col = TARGET_COLORS[Math.max(0, tid - 1) % TARGET_COLORS.length];
    const ry = swimTop + r * rowH;

    ctx.fillStyle = contWasActive[ti] ? col : '#7f95b5';
    ctx.font = '600 8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(`T${tid}`, sparkX0, ry + 8);

    ctx.fillStyle = '#0d1524';
    ctx.fillRect(swimX0, ry + 2, swimW, 6);

    const sc = Math.min(CONT_MAX_SEGS, contSegCount[ti]);
    if (sc > 1) {
      ctx.strokeStyle = '#ffd166';
      ctx.lineWidth = 1;
      for (let sIdx = 1; sIdx < sc; sIdx++) {
        const gStart = swimX0 + (contSegEnd[ti * CONT_MAX_SEGS + sIdx - 1] / maxSeq) * swimW;
        const gEnd = swimX0 + (contSegStart[ti * CONT_MAX_SEGS + sIdx] / maxSeq) * swimW;
        ctx.beginPath();
        ctx.moveTo(gStart, ry + 5);
        ctx.lineTo(Math.max(gStart + 1, gEnd), ry + 5);
        ctx.stroke();
      }
    }
    ctx.fillStyle = col;
    for (let sIdx = 0; sIdx < sc; sIdx++) {
      const sx0 = swimX0 + (contSegStart[ti * CONT_MAX_SEGS + sIdx] / maxSeq) * swimW;
      const sx1 = swimX0 + (contSegEnd[ti * CONT_MAX_SEGS + sIdx] / maxSeq) * swimW;
      ctx.fillRect(sx0, ry + 2, Math.max(2.0, sx1 - sx0), 6);
    }

    ctx.fillStyle = contWasActive[ti] ? '#d9e6f7' : '#7f95b5';
    ctx.fillText(formatPersistDurationSec(contTotalSec[ti]), swimX0 + swimW + 5, ry + 8);
  }
}

function resetTestbenchState() {
  workerExpectedEngineFrame = 0;
  const s = resScale;
  if (isChronos1000Scenario(benchScenario)) {
    resetChronos1000Scenario();
  } else if (benchScenario === 'ballistic_cascade') {
    simDiscs[0].x = 42 * s;  simDiscs[0].y = 146 * s; simDiscs[0].baseY = 146 * s; simDiscs[0].vx = 2.6 * s;  simDiscs[0].vy = -6.8 * s; simDiscs[0].ballistic = true;
    simDiscs[1].x = 278 * s; simDiscs[1].y = 138 * s; simDiscs[1].baseY = 146 * s; simDiscs[1].vx = -2.8 * s; simDiscs[1].vy = -5.6 * s; simDiscs[1].ballistic = true;
    simDiscs[2].x = 112 * s; simDiscs[2].y = 148 * s; simDiscs[2].baseY = 148 * s; simDiscs[2].vx = 2.2 * s;  simDiscs[2].vy = -6.2 * s; simDiscs[2].ballistic = true;
    simDiscs[3].x = 210 * s; simDiscs[3].y = 144 * s; simDiscs[3].baseY = 148 * s; simDiscs[3].vx = -2.4 * s; simDiscs[3].vy = -7.0 * s; simDiscs[3].ballistic = true;
  } else if (benchScenario === 'model_rocket_flight') {
    const laneX = [56, 124, 196, 264];
    const initY = [140, 74, 42, 92];
    const initVy = [-2.4, -3.8, -0.8, 1.9];
    const initPhase = [0, 1, 1, 2];
    const initTicks = [4, 16, 28, 40];
    for (let i = 0; i < 4; i++) {
      simDiscs[i].baseX = laneX[i] * s;
      simDiscs[i].x = laneX[i] * s;
      simDiscs[i].y = initY[i] * s;
      simDiscs[i].baseY = 152 * s;
      simDiscs[i].vx = (i % 2 === 0 ? 0.35 : -0.35) * s;
      simDiscs[i].vy = initVy[i] * s;
      simDiscs[i].ballistic = true;
      rocketPhase[i] = initPhase[i];
      rocketBoostTicks[i] = initTicks[i];
    }
    smokeAlpha.fill(0.0);
    smokeWriteIdx = 0;
  } else if (benchScenario === 'harmonic_tremor') {
    simDiscs[0].x = 76 * s;  simDiscs[0].y = 56 * s;  simDiscs[0].baseX = 76 * s;  simDiscs[0].baseY = 56 * s;  simDiscs[0].vx = 0; simDiscs[0].vy = 0; simDiscs[0].ballistic = false;
    simDiscs[1].x = 244 * s; simDiscs[1].y = 56 * s;  simDiscs[1].baseX = 244 * s; simDiscs[1].baseY = 56 * s;  simDiscs[1].vx = 0; simDiscs[1].vy = 0; simDiscs[1].ballistic = false;
    simDiscs[2].x = 76 * s;  simDiscs[2].y = 124 * s; simDiscs[2].baseX = 76 * s;  simDiscs[2].baseY = 124 * s; simDiscs[2].vx = 0; simDiscs[2].vy = 0; simDiscs[2].ballistic = false;
    simDiscs[3].x = 244 * s; simDiscs[3].y = 124 * s; simDiscs[3].baseX = 244 * s; simDiscs[3].baseY = 124 * s; simDiscs[3].vx = 0; simDiscs[3].vy = 0; simDiscs[3].ballistic = false;
  } else if (benchScenario === 'percussion_ricochet') {
    simDiscs[0].x = 58 * s;  simDiscs[0].y = 44 * s;  simDiscs[0].baseY = 44 * s;  simDiscs[0].vx = 4.6 * s;  simDiscs[0].vy = 3.4 * s;  simDiscs[0].ballistic = false;
    simDiscs[1].x = 262 * s; simDiscs[1].y = 52 * s;  simDiscs[1].baseY = 52 * s;  simDiscs[1].vx = -4.4 * s; simDiscs[1].vy = 3.7 * s;  simDiscs[1].ballistic = false;
    simDiscs[2].x = 74 * s;  simDiscs[2].y = 136 * s; simDiscs[2].baseY = 136 * s; simDiscs[2].vx = 4.2 * s;  simDiscs[2].vy = -3.8 * s; simDiscs[2].ballistic = false;
    simDiscs[3].x = 246 * s; simDiscs[3].y = 128 * s; simDiscs[3].baseY = 128 * s; simDiscs[3].vx = -4.5 * s; simDiscs[3].vy = -3.3 * s; simDiscs[3].ballistic = false;
  } else if (benchScenario === 'multi_target_swarm') {
    const laneY = [26, 52, 78, 104, 130, 154];
    const xPairs = [
      [38, 282],
      [76, 244],
      [54, 266],
      [88, 232],
      [44, 276],
      [68, 252],
    ];
    for (let lane = 0; lane < 6; lane++) {
      const spd = (2.7 + 0.1 * lane) * s;
      const ly = laneY[lane] * s;
      const a = simDiscs[lane * 2];
      const b = simDiscs[lane * 2 + 1];
      a.x = xPairs[lane][0] * s; a.y = ly; a.baseY = ly; a.vx = spd;  a.vy = 0; a.ballistic = false;
      b.x = xPairs[lane][1] * s; b.y = ly; b.baseY = ly; b.vx = -spd; b.vy = 0; b.ballistic = false;
    }
  } else {
    const spd = benchScenario === 'short_gap_bridge' ? 3.2 * s : 3.0 * s;
    simDiscs[0].x = 52 * s;  simDiscs[0].y = 56 * s;  simDiscs[0].baseY = 56 * s;  simDiscs[0].vx = spd;  simDiscs[0].vy = 0; simDiscs[0].ballistic = false;
    simDiscs[1].x = 172 * s; simDiscs[1].y = 56 * s;  simDiscs[1].baseY = 56 * s;  simDiscs[1].vx = -spd; simDiscs[1].vy = 0; simDiscs[1].ballistic = false;
    simDiscs[2].x = 148 * s; simDiscs[2].y = 124 * s; simDiscs[2].baseY = 124 * s; simDiscs[2].vx = spd;  simDiscs[2].vy = 0; simDiscs[2].ballistic = false;
    simDiscs[3].x = 268 * s; simDiscs[3].y = 124 * s; simDiscs[3].baseY = 124 * s; simDiscs[3].vx = -spd; simDiscs[3].vy = 0; simDiscs[3].ballistic = false;
  }

  oracleTrackOwner.fill(-1);
  clearVisualQueues();
  resetPersistenceState();
  oraclePadicSwaps = 0;
  inPillarCollisions = 0;
  simTimeSec = 0.0;
  pendingSaccadeFrames = 0;
  pendingAirSnapFrames = 0;
  renderAllSlotPips(false);
  renderPersistenceTimeline(true);
  renderPhysicsPhaseCanvas(true);
}

function stepAndRenderBlindPillarScene(dtSec) {
  if (isChronos1000Scenario(benchScenario)) {
    return stepAndRenderChronos1000Scene(dtSec);
  }
  const ctx = simCtx;
  const s = resScale;
  const stepScale = Math.min(2.0, Math.max(0.25, dtSec * 60.0));
  simTimeSec += dtSec;

  // Deterministic Mode B / Mode C testbench stimuli (zero setTimeout jitter)
  if (pendingSaccadeFrames > 0) {
    pendingSaccadeFrames--;
    if (pendingSaccadeFrames === 0) {
      const d0 = simDiscs[0];
      if (benchScenario === 'harmonic_tremor') {
        d0.baseX = d0.baseX < 160 * s ? d0.baseX + 32 * s : d0.baseX - 32 * s;
      } else {
        d0.x = d0.x < 160 * s ? Math.min(84 * s, d0.x + 30 * s) : Math.max(145 * s, d0.x - 30 * s);
      }
    }
  }
  if (pendingAirSnapFrames > 0) {
    pendingAirSnapFrames--;
    if (pendingAirSnapFrames === 0 && benchScenario !== 'percussion_ricochet' && benchScenario !== 'harmonic_tremor' && benchScenario !== 'model_rocket_flight') {
      const nDiscsSnap = getActiveSimDiscCount();
      for (let i = 0; i < nDiscsSnap; i++) {
        simDiscs[i].vx = Math.sign(simDiscs[i].vx || 1) * 3.0 * s;
      }
    }
  }

  // 1. Static structured aerospace grid clutter (tests uncensored >60 dB ECA cancellation)
  ctx.fillStyle = '#121722';
  ctx.fillRect(0, 0, frameW, frameH);

  const gridStep = 24 * s;
  ctx.strokeStyle = '#1d2536';
  ctx.lineWidth = s;
  ctx.beginPath();
  for (let x = gridStep; x < frameW; x += gridStep) {
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, frameH);
  }
  for (let y = gridStep; y < frameH; y += gridStep) {
    ctx.moveTo(0, y + 0.5);
    ctx.lineTo(frameW, y + 0.5);
  }
  ctx.stroke();

  if (benchScenario === 'model_rocket_flight') {
    // Launch pads, guide rails, and dissipating exhaust smoke puffs behind active boosters
    ctx.fillStyle = '#1b2434';
    ctx.fillRect(0, 156 * s, frameW, frameH - 156 * s);
    ctx.strokeStyle = 'rgba(130, 165, 210, 0.35)';
    ctx.lineWidth = Math.max(1, s);
    const laneX = [56, 124, 196, 264];
    for (let i = 0; i < laneX.length; i++) {
      ctx.beginPath();
      ctx.moveTo(laneX[i] * s, 156 * s);
      ctx.lineTo(laneX[i] * s, 126 * s);
      ctx.stroke();
    }
    for (let p = 0; p < SMOKE_PUFFS; p++) {
      const a = smokeAlpha[p];
      if (a <= 0.015) continue;
      smokeY[p] += 0.14 * s * stepScale;
      smokeR[p] += 0.05 * s * stepScale;
      smokeAlpha[p] = Math.max(0.0, a - 0.022 * stepScale);
      ctx.fillStyle = `rgba(112, 130, 154, ${smokeAlpha[p].toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(smokeX[p], smokeY[p], smokeR[p], 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // 2. Advance disc kinematics & collisions
  const tSec = simTimeSec;
  const wallMargin = 18 * s;
  const pillars = getActivePillars();
  const nActiveDiscs = getActiveSimDiscCount();

  for (let i = 0; i < nActiveDiscs; i++) {
    const d = simDiscs[i];
    if (benchScenario === 'model_rocket_flight') {
      if (rocketPhase[i] === 0) {
        // Powered boost phase (ay < 0 upward thrust)
        const thrust = (0.34 + 0.03 * (i % 3)) * s * stepScale;
        d.vy -= thrust;
        d.y += d.vy * stepScale;
        rocketBoostTicks[i]++;
        if ((rocketBoostTicks[i] & 1) === 0) {
          smokeX[smokeWriteIdx] = d.x + (Math.sin(tSec * 9.0 + i) * 1.1 * s);
          smokeY[smokeWriteIdx] = d.y + 7.5 * s;
          smokeR[smokeWriteIdx] = 1.8 * s;
          smokeAlpha[smokeWriteIdx] = 0.08;
          smokeWriteIdx = (smokeWriteIdx + 1) % SMOKE_PUFFS;
        }
        if (rocketBoostTicks[i] >= 12 || d.vy <= -5.2 * s) {
          rocketPhase[i] = 1;
        }
      } else if (rocketPhase[i] === 1) {
        // Newtonian free-fall coast + apogee rollover (ay = +g > 0)
        const vyPrev = d.vy;
        d.vy += 0.20 * s * stepScale;
        d.y += 0.5 * (vyPrev + d.vy) * stepScale;
        if (d.vy >= 2.2 * s) {
          rocketPhase[i] = 2;
        }
      } else {
        // Drogue parachute recovery descent toward pad
        d.vy = Math.min(d.vy + 0.04 * s * stepScale, (2.15 + 0.15 * (i % 2)) * s);
        d.y += d.vy * stepScale;
      }

      // Gentle in-lane windward arc (preserves non-degenerate 2D motion without crossing lanes)
      d.x += d.vx * stepScale;
      const laneOffset = d.x - d.baseX;
      if (laneOffset < -16 * s) {
        d.x = d.baseX - 16 * s;
        d.vx = Math.abs(d.vx);
      } else if (laneOffset > 16 * s) {
        d.x = d.baseX + 16 * s;
        d.vx = -Math.abs(d.vx);
      }

      if (d.y < 16 * s) {
        d.y = 16 * s;
        d.vy = Math.abs(d.vy) * 0.35;
        rocketPhase[i] = 1;
      }
      if (d.y >= 152 * s) {
        d.y = 152 * s;
        d.vy = -(1.25 + 0.15 * (i % 3)) * s;
        rocketPhase[i] = 0;
        rocketBoostTicks[i] = 0;
      }
      continue;
    }

    if (benchScenario === 'harmonic_tremor' && !d.ballistic) {
      const phase = 2.0 * Math.PI * 8.5 * tSec + i * 0.9;
      const amp = 4.2 * s;
      d.x = d.baseX + amp * Math.cos(phase);
      d.y = d.baseY - amp * Math.sin(phase);
      continue;
    }

    d.x += d.vx * stepScale;

    if (d.ballistic) {
      const vyPrev = d.vy;
      d.vy += 0.22 * s * stepScale;
      d.y += 0.5 * (vyPrev + d.vy) * stepScale;
      if (d.y >= 152 * s) {
        d.y = 152 * s;
        if (benchScenario === 'ballistic_cascade') {
          d.vy = -(5.8 + (i % 3) * 0.55) * s;
        } else {
          d.vy = -Math.abs(d.vy) * 0.86;
          if (Math.abs(d.vy) < 1.2 * s) {
            d.ballistic = false;
            d.vy = 0;
            d.y = d.baseY;
          }
        }
      }
      if (d.y < 16 * s) {
        d.y = 16 * s;
        d.vy = Math.abs(d.vy) * 0.5;
      }
    } else if (benchScenario === 'percussion_ricochet') {
      d.y += d.vy * stepScale;
      if (d.y < wallMargin) {
        d.y = wallMargin;
        d.vy = Math.abs(d.vy);
      } else if (d.y > frameH - wallMargin) {
        d.y = frameH - wallMargin;
        d.vy = -Math.abs(d.vy);
      }
    } else {
      const tremor = 0.95 * s * Math.sin(2.0 * Math.PI * 9.5 * tSec + i * 1.3);
      d.y = d.baseY + tremor;
    }

    if (d.x < wallMargin) {
      d.x = wallMargin;
      d.vx = Math.abs(d.vx);
    } else if (d.x > frameW - wallMargin) {
      d.x = frameW - wallMargin;
      d.vx = -Math.abs(d.vx);
    }
  }

  // Pairwise elastic collisions
  const hitThresh = 14.5 * s;
  if (benchScenario === 'percussion_ricochet') {
    for (let i = 0; i < nActiveDiscs; i++) {
      for (let j = i + 1; j < nActiveDiscs; j++) {
        const a = simDiscs[i];
        const b = simDiscs[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.hypot(dx, dy);
        if (dist > 1e-3 && dist < 16.0 * s) {
          const nx = dx / dist;
          const ny = dy / dist;
          const dvx = a.vx - b.vx;
          const dvy = a.vy - b.vy;
          const relN = dvx * nx + dvy * ny;
          if (relN > 0) {
            a.vx -= relN * nx;
            a.vy -= relN * ny;
            b.vx += relN * nx;
            b.vy += relN * ny;
          }
        }
      }
    }
  } else if (benchScenario !== 'ballistic_cascade' && benchScenario !== 'harmonic_tremor' && benchScenario !== 'model_rocket_flight') {
    const nLanes = benchScenario === 'multi_target_swarm' ? 6 : 2;
    for (let lane = 0; lane < nLanes; lane++) {
      const a = simDiscs[lane * 2];
      const b = simDiscs[lane * 2 + 1];
      if (a.ballistic || b.ballistic) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      if (Math.abs(dx) < hitThresh && Math.abs(dy) < hitThresh && (a.vx - b.vx) * dx > 0) {
        const tmp = a.vx;
        a.vx = b.vx;
        b.vx = tmp;
        const midX = 0.5 * (a.x + b.x);
        for (let p = 0; p < pillars.length; p++) {
          if (midX >= pillars[p].x0 * s && midX <= pillars[p].x1 * s) {
            inPillarCollisions++;
            break;
          }
        }
      }
    }
  }

  // 3. Draw physical textured targets
  if (benchScenario === 'model_rocket_flight') {
    for (let i = 0; i < nActiveDiscs; i++) {
      const d = simDiscs[i];
      const ang = Math.atan2(d.vy, d.vx || 0.001) + Math.PI * 0.5;
      ctx.save();
      ctx.translate(d.x, d.y);
      ctx.rotate(rocketPhase[i] === 2 ? Math.PI * 0.04 * Math.sin(tSec * 4.0 + i) : ang * 0.35);

      if (rocketPhase[i] === 2) {
        // Deployed drogue chute above nosecone (compact so centroid stays on fuselage)
        ctx.fillStyle = 'rgba(255, 107, 53, 0.72)';
        ctx.beginPath();
        ctx.arc(0, -7.8 * s, 4.4 * s, Math.PI, 0);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = 'rgba(215, 228, 245, 0.45)';
        ctx.lineWidth = Math.max(1, 0.7 * s);
        ctx.beginPath();
        ctx.moveTo(-3.6 * s, -7.8 * s);
        ctx.lineTo(0, -3.5 * s);
        ctx.lineTo(3.6 * s, -7.8 * s);
        ctx.stroke();
      }

      // High-contrast fuselage body + bright beacon core (matches PD32 16x16 / 32x32 aperture)
      ctx.fillStyle = '#f4f8ff';
      ctx.fillRect(-3.0 * s, -3.5 * s, 6.0 * s, 7.5 * s);

      // Painted nosecone
      ctx.fillStyle = d.r;
      ctx.beginPath();
      ctx.moveTo(0, -7.5 * s);
      ctx.lineTo(3.2 * s, -3.5 * s);
      ctx.lineTo(-3.2 * s, -3.5 * s);
      ctx.closePath();
      ctx.fill();

      // Tail fins
      ctx.fillStyle = d.r;
      ctx.beginPath();
      ctx.moveTo(-3.0 * s, 1.5 * s);
      ctx.lineTo(-5.0 * s, 4.8 * s);
      ctx.lineTo(-3.0 * s, 4.0 * s);
      ctx.closePath();
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(3.0 * s, 1.5 * s);
      ctx.lineTo(5.0 * s, 4.8 * s);
      ctx.lineTo(3.0 * s, 4.0 * s);
      ctx.closePath();
      ctx.fill();

      // High-contrast avionics core at center of mass
      ctx.fillStyle = d.core;
      ctx.beginPath();
      ctx.arc(0, 0, 2.2 * s, 0, Math.PI * 2);
      ctx.fill();

      if (rocketPhase[i] === 0) {
        // Compact thrust exhaust plume during boost
        ctx.fillStyle = '#ffb703';
        ctx.beginPath();
        ctx.moveTo(-2.0 * s, 4.0 * s);
        ctx.lineTo(0, 8.8 * s + 1.4 * s * Math.sin(tSec * 42.0 + i));
        ctx.lineTo(2.0 * s, 4.0 * s);
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    }
  } else {
    for (let i = 0; i < nActiveDiscs; i++) {
      const d = simDiscs[i];
      ctx.fillStyle = d.r;
      ctx.beginPath();
      ctx.arc(d.x, d.y, 7.6 * s, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = d.core;
      ctx.beginPath();
      ctx.arc(d.x, d.y, 3.4 * s, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // 4. Draw 100% OPAQUE occlusion pillars ON TOP (wipes out underlying disc pixels completely)
  for (let p = 0; p < pillars.length; p++) {
    const pil = pillars[p];
    const px0 = pil.x0 * s;
    const py0 = pil.y0 * s;
    const pw = (pil.x1 - pil.x0) * s;
    const ph = (pil.y1 - pil.y0) * s;
    ctx.fillStyle = '#222a3a';
    ctx.fillRect(px0, py0, pw, ph);
    ctx.strokeStyle = '#36435c';
    ctx.lineWidth = s;
    ctx.strokeRect(px0 + 0.5 * s, py0 + 0.5 * s, pw - s, ph - s);
  }

  return ctx.getImageData(0, 0, frameW, frameH).data;
}

/// Strict 1-to-1 Nearest-Match Ground-Truth Identity Oracle (no & 255 wrap, no multi-disc aliasing).
function auditOracleSwaps() {
  const s = resScale;
  const pillars = getActivePillars();
  const nActiveDiscs = getActiveSimDiscCount();
  let nPairs = 0;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] < 0.5) continue;
    const latestVis = telemView[base + 44 + 11];
    if (latestVis < 0.5) continue;

    const ix = telemView[base + 8];
    const iy = telemView[base + 9];

    for (let i = 0; i < nActiveDiscs; i++) {
      const d = simDiscs[i];
      let occluded = false;
      for (let p = 0; p < pillars.length; p++) {
        if (d.x > (pillars[p].x0 - 6) * s && d.x < (pillars[p].x1 + 6) * s) {
          occluded = true;
          break;
        }
      }
      if (occluded) continue;

      const dist = Math.hypot(ix - d.x, iy - d.y);
      if (dist < 16.0 * s) {
        oracleCandTrackSlot[nPairs] = m;
        oracleCandDiscIdx[nPairs] = i;
        oracleCandDist[nPairs] = dist;
        nPairs++;
      }
    }
  }

  for (let a = 0; a < nPairs; a++) {
    for (let b = a + 1; b < nPairs; b++) {
      if (oracleCandDist[b] < oracleCandDist[a]) {
        const td = oracleCandDist[a]; oracleCandDist[a] = oracleCandDist[b]; oracleCandDist[b] = td;
        const tm = oracleCandTrackSlot[a]; oracleCandTrackSlot[a] = oracleCandTrackSlot[b]; oracleCandTrackSlot[b] = tm;
        const ti = oracleCandDiscIdx[a]; oracleCandDiscIdx[a] = oracleCandDiscIdx[b]; oracleCandDiscIdx[b] = ti;
      }
    }
  }

  let usedTracks = 0;
  let usedDiscs = 0;
  for (let k = 0; k < nPairs; k++) {
    const m = oracleCandTrackSlot[k];
    const di = oracleCandDiscIdx[k];
    if ((usedTracks & (1 << m)) !== 0 || (usedDiscs & (1 << di)) !== 0) continue;
    usedTracks |= 1 << m;
    usedDiscs |= 1 << di;

    const base = 32 + m * 56;
    const tid = Math.round(telemView[base + 1]);
    if (tid >= 0 && tid < MAX_ORACLE_IDS) {
      const discId = simDiscs[di].id;
      const prevOwner = oracleTrackOwner[tid];
      if (prevOwner === -1) {
        oracleTrackOwner[tid] = discId;
      } else if (prevOwner !== discId) {
        oraclePadicSwaps++;
        oracleTrackOwner[tid] = discId;
      }
    }
  }
}

// ============================================================================
// OffscreenCanvas Aerospace HUD & Telemetry Scope Renderers
// ============================================================================

function toHex8(sigHi, sigLo) {
  const u32 = (((sigHi | 0) << 16) | (sigLo | 0)) >>> 0;
  return '0x' + u32.toString(16).toUpperCase().padStart(8, '0');
}

function resolvePrimarySlot() {
  if (!telemView) return -1;
  const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
  if (focusTid > 0) {
    for (let m = 0; m < MAX_TARGETS; m++) {
      const base = 32 + m * 56;
      if (telemView[base] > 0.5 && Math.round(telemView[base + 1]) === focusTid) {
        return m;
      }
    }
  }
  return Math.round(telemView[8]);
}

function updateFrozenApexLocks(uDisp) {
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] < 0.5) {
      apexLockTid[m] = 0;
      apexLockHold[m] = 0;
      apexLockErrPx[m] = -1.0;
      continue;
    }
    const tid = Math.round(telemView[base + 1]);
    const apexFrames = telemView[base + 17];
    const apexX = telemView[base + 15];
    const apexY = telemView[base + 16];
    const curX = telemView[base + 20 + 11];
    const curY = telemView[base + 32 + 11];
    const dispY = telemView[base + 32 + uDisp];

    if (apexFrames >= 0.75) {
      if (apexLockTid[m] !== tid || apexLockHold[m] <= 0 || apexLockErrPx[m] >= 0.0) {
        // Lock new ballistic throw prediction while ascending!
        apexLockTid[m] = tid;
        apexLockPredX[m] = apexX;
        apexLockPredY[m] = apexY;
        apexLockMinY[m] = curY;
        apexLockAtMinX[m] = curX;
        apexLockErrPx[m] = -1.0;
        apexLockHold[m] = Math.max(16, lookaheadK + 12);
      } else if (apexFrames >= 1.25) {
        // Early ascent wide-chord refinement; freezes completely before peak!
        apexLockPredX[m] = 0.65 * apexLockPredX[m] + 0.35 * apexX;
        apexLockPredY[m] = 0.65 * apexLockPredY[m] + 0.35 * apexY;
        apexLockHold[m] = Math.max(apexLockHold[m], lookaheadK + 12);
      }
    }

    if (apexLockTid[m] === tid && apexLockHold[m] > 0) {
      if (curY < apexLockMinY[m]) {
        apexLockMinY[m] = curY;
        apexLockAtMinX[m] = curX;
      }
      // Score full Euclidean peak accuracy once projectile turns downward near predicted peak!
      if (apexLockErrPx[m] < 0.0 && apexFrames <= 0.05 && curY > apexLockMinY[m] + 2.5 * resScale) {
        const err = Math.hypot(
          apexLockAtMinX[m] - apexLockPredX[m],
          apexLockMinY[m] - apexLockPredY[m]
        );
        if (err <= 9.5 * resScale) {
          apexLockErrPx[m] = err;
          lastScoredApexErrPx = err;
        } else {
          apexLockTid[m] = 0;
          apexLockHold[m] = 0;
          apexLockErrPx[m] = -1.0;
          continue;
        }
      }
      if (apexLockErrPx[m] >= 0.0 && dispY > apexLockMinY[m] + 3.0 * resScale) {
        apexLockHold[m]--;
      } else if (apexFrames <= 0.05 && curY <= apexLockMinY[m] + 2.5 * resScale) {
        apexLockHold[m]--;
      }
    } else if (apexLockTid[m] !== tid) {
      apexLockTid[m] = 0;
      apexLockHold[m] = 0;
      apexLockErrPx[m] = -1.0;
    }
  }
}

function drawHudArrow(ctx, x0, y0, x1, y1, headLen) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.hypot(dx, dy);
  if (len < 2.0) return;
  const ang = Math.atan2(dy, dx);
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.lineTo(x1 - headLen * Math.cos(ang - 0.46), y1 - headLen * Math.sin(ang - 0.46));
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - headLen * Math.cos(ang + 0.46), y1 - headLen * Math.sin(ang + 0.46));
  ctx.stroke();
}

function reserveHudLabelSlot(W, H, prefX, prefY, boxW, placedCount) {
  let lx = Math.max(8, Math.min(W - boxW - 8, prefX));
  let ly = Math.max(38, Math.min(H - 54, prefY));
  for (let pass = 0; pass < 4; pass++) {
    let hit = false;
    for (let k = 0; k < placedCount; k++) {
      const overlapX = lx < hudLabelX[k] + hudLabelW[k] + 4 && lx + boxW + 4 > hudLabelX[k];
      const overlapY = Math.abs(ly - hudLabelY[k]) < 21;
      if (overlapX && overlapY) {
        hit = true;
        break;
      }
    }
    if (!hit) break;
    ly = ly + 22 < H - 54 ? ly + 22 : Math.max(38, ly - 22);
  }
  if (placedCount < MAX_HUD_LABELS) {
    hudLabelX[placedCount] = lx;
    hudLabelY[placedCount] = ly;
    hudLabelW[placedCount] = boxW;
  }
  calOutPt[0] = lx;
  calOutPt[1] = ly;
  return calOutPt;
}

function renderCalibrationOverlay(W, H) {
  if (calMode === 'none' && calArmed === 'none') return;
  hudCtx.save();

  if (calMode === 'ruler' && calPtCount >= 2 && calArmed === 'none') {
    const x0 = calPtsNormX[0] * W;
    const y0 = calPtsNormY[0] * H;
    const x1 = calPtsNormX[1] * W;
    const y1 = calPtsNormY[1] * H;
    const dx = x1 - x0;
    const dy = y1 - y0;
    const len = Math.max(1.0, Math.hypot(dx, dy));
    const nx = -dy / len;
    const ny = dx / len;

    hudCtx.strokeStyle = 'rgba(255, 209, 102, 0.85)';
    hudCtx.lineWidth = 1.8;
    hudCtx.setLineDash([6, 4]);
    hudCtx.beginPath();
    hudCtx.moveTo(x0, y0);
    hudCtx.lineTo(x1, y1);
    hudCtx.stroke();
    hudCtx.setLineDash([]);

    hudCtx.beginPath();
    hudCtx.moveTo(x0 - nx * 8, y0 - ny * 8);
    hudCtx.lineTo(x0 + nx * 8, y0 + ny * 8);
    hudCtx.moveTo(x1 - nx * 8, y1 - ny * 8);
    hudCtx.lineTo(x1 + nx * 8, y1 + ny * 8);
    hudCtx.stroke();

    const mx = 0.5 * (x0 + x1);
    const my = 0.5 * (y0 + y1);
    const rTxt = `📏 ${calRulerMeters.toFixed(2)} m`;
    hudCtx.fillStyle = 'rgba(6, 10, 18, 0.86)';
    hudCtx.fillRect(mx - 36, my - 10, 72, 18);
    hudCtx.strokeStyle = 'rgba(255, 209, 102, 0.65)';
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeRect(mx - 36, my - 10, 72, 18);
    hudCtx.fillStyle = '#ffd166';
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(rTxt, mx - 30, my + 3);
  } else if (calMode === 'homography' && calHomographyValid && calArmed === 'none') {
    // Perspective plane boundary + 4x4 iso-metric grid lines mapped back through H^-1
    hudCtx.strokeStyle = 'rgba(0, 229, 255, 0.22)';
    hudCtx.lineWidth = 1.0;
    hudCtx.setLineDash([4, 4]);
    for (let g = 1; g <= 3; g++) {
      const u = (g / 4.0) * calPlaneW;
      const pA = mapMetricToNorm(u, 0.0);
      const ax = pA[0] * W, ay = pA[1] * H;
      const pB = mapMetricToNorm(u, calPlaneH);
      const bx = pB[0] * W, by = pB[1] * H;
      hudCtx.beginPath();
      hudCtx.moveTo(ax, ay);
      hudCtx.lineTo(bx, by);
      hudCtx.stroke();

      const v = (g / 4.0) * calPlaneH;
      const pC = mapMetricToNorm(0.0, v);
      const cx = pC[0] * W, cy = pC[1] * H;
      const pD = mapMetricToNorm(calPlaneW, v);
      const dx = pD[0] * W, dy = pD[1] * H;
      hudCtx.beginPath();
      hudCtx.moveTo(cx, cy);
      hudCtx.lineTo(dx, dy);
      hudCtx.stroke();
    }
    hudCtx.setLineDash([]);

    hudCtx.strokeStyle = 'rgba(0, 229, 255, 0.58)';
    hudCtx.lineWidth = 1.8;
    hudCtx.beginPath();
    for (let i = 0; i < 4; i++) {
      const qx = calPtsNormX[i] * W;
      const qy = calPtsNormY[i] * H;
      if (i === 0) hudCtx.moveTo(qx, qy);
      else hudCtx.lineTo(qx, qy);
    }
    hudCtx.closePath();
    hudCtx.stroke();

    hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    for (let i = 0; i < 4; i++) {
      const qx = calPtsNormX[i] * W;
      const qy = calPtsNormY[i] * H;
      hudCtx.fillStyle = '#00e5ff';
      hudCtx.beginPath();
      hudCtx.arc(qx, qy, 3.6, 0, Math.PI * 2);
      hudCtx.fill();
      hudCtx.fillStyle = 'rgba(6, 10, 18, 0.82)';
      hudCtx.fillRect(qx + 5, qy - 12, 20, 13);
      hudCtx.fillStyle = '#8ef9f3';
      hudCtx.fillText(CORNER_TAGS[i], qx + 7, qy - 2);
    }
  }

  // Interactive click-capture crosshairs & guidance banner when armed (draft points keep prior cal intact on Esc!)
  if (calArmed !== 'none') {
    for (let i = 0; i < calDraftCount; i++) {
      const px = calDraftPtsNormX[i] * W;
      const py = calDraftPtsNormY[i] * H;
      hudCtx.strokeStyle = '#ffd166';
      hudCtx.lineWidth = 2.0;
      hudCtx.beginPath();
      hudCtx.arc(px, py, 7, 0, Math.PI * 2);
      hudCtx.moveTo(px - 12, py); hudCtx.lineTo(px + 12, py);
      hudCtx.moveTo(px, py - 12); hudCtx.lineTo(px, py + 12);
      hudCtx.stroke();
      if (i > 0) {
        hudCtx.setLineDash([4, 3]);
        hudCtx.beginPath();
        hudCtx.moveTo(calDraftPtsNormX[i - 1] * W, calDraftPtsNormY[i - 1] * H);
        hudCtx.lineTo(px, py);
        hudCtx.stroke();
        hudCtx.setLineDash([]);
      }
    }
    const prompt = calArmed === 'ruler'
      ? `📏 CLICK RULER ENDPOINT ${calDraftCount + 1}/2 (${calRulerMeters.toFixed(2)} m · ESC PRESERVES PRIOR)`
      : `⬠ CLICK PLANE CORNER ${calDraftCount + 1}/4 [${CORNER_TAGS[Math.min(3, calDraftCount)]}] (${calPlaneW.toFixed(2)}×${calPlaneH.toFixed(2)} m · ESC PRESERVES)`;
    const bw = 504;
    const bx = Math.round((W - bw) * 0.5);
    hudCtx.fillStyle = 'rgba(8, 13, 24, 0.92)';
    hudCtx.fillRect(bx, 40, bw, 24);
    hudCtx.strokeStyle = '#ffd166';
    hudCtx.lineWidth = 1.5;
    hudCtx.strokeRect(bx, 40, bw, 24);
    hudCtx.fillStyle = '#ffd166';
    hudCtx.font = '700 10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(prompt, bx + 12, 56);
  }

  hudCtx.restore();
}

// ============================================================================
// CHRONOS-1000 Field Layer Compositor (WebGL2 Instanced + Batched 2D Fallback)
// ============================================================================

const fieldTexRgba = new Uint8Array(FIELD_CELLS * 4);
const fieldTexGlPremul = new Uint8Array(FIELD_CELLS * 4);
const microInstBuf = new Float32Array(MAX_MICRO * 6);
const bodyMemHead = new Int16Array(MAX_RIGID_BODIES + 1);
const bodyMemNext = new Int16Array(MAX_MICRO);
const hullPtsX = new Float32Array(MAX_MICRO);
const hullPtsY = new Float32Array(MAX_MICRO);
const hullIdx = new Int32Array(MAX_MICRO);
const hullPoly = new Int32Array(MAX_MICRO * 2);

let fieldGlCanvas = null;
let fieldGlCtx = null;
let fieldGlBroken = false;
let fieldGlContextLost = false;
let fieldGlQuadProg = null;
let fieldGlQuadVao = null;
let fieldGlTex = null;
let fieldGlStemProg = null;
let fieldGlStemVao = null;
let fieldGlStemUInv = null;
let fieldGlPipProg = null;
let fieldGlPipVao = null;
let fieldGlPipUInv = null;
let fieldGlPipUHalf = null;
let fieldGlInstVbo = null;

function resetFieldWebGlResources() {
  fieldGlCtx = null;
  fieldGlQuadProg = null;
  fieldGlQuadVao = null;
  fieldGlTex = null;
  fieldGlStemProg = null;
  fieldGlStemVao = null;
  fieldGlStemUInv = null;
  fieldGlPipProg = null;
  fieldGlPipVao = null;
  fieldGlPipUInv = null;
  fieldGlPipUHalf = null;
  fieldGlInstVbo = null;
}

let field2dCanvas = null;
let field2dCtx = null;
let field2dImgData = null;

function compileFieldShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(sh) || 'shader compile error';
    gl.deleteShader(sh);
    throw new Error(info);
  }
  return sh;
}

function linkFieldProgram(gl, vsSrc, fsSrc) {
  const vs = compileFieldShader(gl, gl.VERTEX_SHADER, vsSrc);
  const fs = compileFieldShader(gl, gl.FRAGMENT_SHADER, fsSrc);
  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(prog) || 'program link error';
    gl.deleteProgram(prog);
    throw new Error(info);
  }
  return prog;
}

function ensureFieldWebGl2(W, H) {
  if (fieldGlBroken || fieldGlContextLost) return false;
  if (fieldGlCtx) {
    if (fieldGlCtx.isContextLost()) {
      fieldGlContextLost = true;
      resetFieldWebGlResources();
      return false;
    }
    if (fieldGlCanvas.width !== W || fieldGlCanvas.height !== H) {
      fieldGlCanvas.width = W;
      fieldGlCanvas.height = H;
    }
    return true;
  }
  try {
    if (!fieldGlCanvas) {
      fieldGlCanvas = new OffscreenCanvas(W, H);
      fieldGlCanvas.addEventListener('webglcontextlost', (ev) => {
        ev.preventDefault();
        fieldGlContextLost = true;
        resetFieldWebGlResources();
      });
      fieldGlCanvas.addEventListener('webglcontextrestored', () => {
        fieldGlContextLost = false;
        resetFieldWebGlResources();
      });
    } else if (fieldGlCanvas.width !== W || fieldGlCanvas.height !== H) {
      fieldGlCanvas.width = W;
      fieldGlCanvas.height = H;
    }
    const gl = fieldGlCanvas.getContext('webgl2', {
      alpha: true,
      antialias: true,
      depth: false,
      stencil: false,
      premultipliedAlpha: true,
      preserveDrawingBuffer: false,
    });
    if (!gl || gl.isContextLost()) {
      if (!gl) fieldGlBroken = true;
      return false;
    }

    const quadVs = `#version 300 es
      out vec2 vUv;
      void main() {
        vec2 p = vec2(
          (gl_VertexID == 1 || gl_VertexID == 3 || gl_VertexID == 4) ? 1.0 : -1.0,
          (gl_VertexID == 2 || gl_VertexID == 4 || gl_VertexID == 5) ? 1.0 : -1.0
        );
        vUv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
        gl_Position = vec4(p, 0.0, 1.0);
      }`;
    const quadFs = `#version 300 es
      precision mediump float;
      uniform sampler2D uFieldTex;
      in vec2 vUv;
      out vec4 fragColor;
      void main() {
        fragColor = texture(uFieldTex, vUv);
      }`;

    const stemVs = `#version 300 es
      layout(location = 0) in vec2 aPos;
      layout(location = 1) in vec2 aVel;
      layout(location = 2) in vec2 aMeta;
      uniform vec2 uInvFrame;
      out vec4 vColor;
      void main() {
        vec2 p = aPos * uInvFrame;
        float spd = length(aVel);
        vec2 stepPx = spd > 1e-3 ? aVel * min(0.16, 22.0 / spd) : vec2(0.0);
        vec2 uv = p + (gl_VertexID == 1 ? stepPx * uInvFrame : vec2(0.0));
        gl_Position = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
        float code = aMeta.x;
        float insp = aMeta.y;
        vec3 col = insp > 0.5 ? vec3(0.99, 0.88, 0.28)
          : (code > 2.5 ? vec3(1.0, 0.62, 0.18)
          : (code > 1.5 ? vec3(0.96, 0.62, 0.04) : vec3(0.0, 0.90, 1.0)));
        float alpha = (code > 1.5 && code < 2.5) ? 0.36 : 0.72;
        vColor = vec4(col * alpha, alpha);
      }`;
    const stemFs = `#version 300 es
      precision mediump float;
      in vec4 vColor;
      out vec4 fragColor;
      void main() {
        fragColor = vColor;
      }`;

    const pipVs = `#version 300 es
      layout(location = 0) in vec2 aPos;
      layout(location = 1) in vec2 aVel;
      layout(location = 2) in vec2 aMeta;
      uniform vec2 uInvFrame;
      uniform vec2 uHalfNdc;
      out vec2 vLocal;
      out vec2 vMeta;
      void main() {
        vec2 uv = aPos * uInvFrame;
        vec2 q = vec2(
          (gl_VertexID & 1) == 0 ? -1.0 : 1.0,
          (gl_VertexID & 2) == 0 ? -1.0 : 1.0
        );
        float scale = aMeta.y > 0.5 ? 1.65 : (aMeta.x > 2.5 ? 1.15 : 1.0);
        vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) + q * uHalfNdc * scale;
        gl_Position = vec4(ndc, 0.0, 1.0);
        vLocal = q;
        vMeta = aMeta;
      }`;
    const pipFs = `#version 300 es
      precision mediump float;
      in vec2 vLocal;
      in vec2 vMeta;
      out vec4 fragColor;
      void main() {
        float d = abs(vLocal.x) + abs(vLocal.y);
        float cov = 1.0 - smoothstep(0.68, 1.02, d);
        if (cov <= 0.01) discard;
        float code = vMeta.x;
        float insp = vMeta.y;
        vec3 col = insp > 0.5 ? vec3(0.99, 0.88, 0.28)
          : (code > 2.5 ? vec3(1.0, 0.62, 0.18)
          : (code > 1.5 ? vec3(0.96, 0.62, 0.04) : vec3(0.0, 0.90, 1.0)));
        float alpha = ((code > 1.5 && code < 2.5) ? 0.55 : 0.92) * cov;
        fragColor = vec4(col * alpha, alpha);
      }`;

    fieldGlQuadProg = linkFieldProgram(gl, quadVs, quadFs);
    fieldGlStemProg = linkFieldProgram(gl, stemVs, stemFs);
    fieldGlPipProg = linkFieldProgram(gl, pipVs, pipFs);

    fieldGlStemUInv = gl.getUniformLocation(fieldGlStemProg, 'uInvFrame');
    fieldGlPipUInv = gl.getUniformLocation(fieldGlPipProg, 'uInvFrame');
    fieldGlPipUHalf = gl.getUniformLocation(fieldGlPipProg, 'uHalfNdc');

    fieldGlQuadVao = gl.createVertexArray();
    gl.bindVertexArray(fieldGlQuadVao);
    gl.useProgram(fieldGlQuadProg);
    gl.uniform1i(gl.getUniformLocation(fieldGlQuadProg, 'uFieldTex'), 0);

    fieldGlTex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, fieldGlTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA8, FIELD_GRID_W, FIELD_GRID_H, 0,
      gl.RGBA, gl.UNSIGNED_BYTE, fieldTexRgba
    );

    fieldGlInstVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, fieldGlInstVbo);
    gl.bufferData(gl.ARRAY_BUFFER, microInstBuf.byteLength, gl.DYNAMIC_DRAW);

    const setupInstVao = (vao) => {
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, fieldGlInstVbo);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 24, 0);
      gl.vertexAttribDivisor(0, 1);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 24, 8);
      gl.vertexAttribDivisor(1, 1);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 24, 16);
      gl.vertexAttribDivisor(2, 1);
    };

    fieldGlStemVao = gl.createVertexArray();
    setupInstVao(fieldGlStemVao);
    fieldGlPipVao = gl.createVertexArray();
    setupInstVao(fieldGlPipVao);
    gl.bindVertexArray(null);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    fieldGlCtx = gl;
    return true;
  } catch (_err) {
    fieldGlBroken = true;
    fieldGlCtx = null;
    return false;
  }
}

function packFieldOverlayTexture(dispField, dispModal, validityMask, dispSwarmActive = true) {
  const mask = validityMask | 0;
  const active = Boolean(dispSwarmActive);
  const modeAllowed =
    !active ? false
      : fieldOverlayMode === 1 ? true
      : fieldOverlayMode === 2 ? (mask & 1) !== 0
      : fieldOverlayMode === 3 ? (mask & 2) !== 0
      : fieldOverlayMode === 4 ? Boolean(dispModal && (mask & 8) !== 0)
      : false;

  if (!modeAllowed) {
    fieldTexRgba.fill(0);
    fieldTexGlPremul.fill(0);
    return 1.0;
  }

  let maxMag = 0.0;
  for (let c = 0; c < FIELD_CELLS; c++) {
    const b = c * FIELD_STRIDE;
    const w = dispField[b + 5];
    let m = 0.0;
    if (fieldOverlayMode === 1 || fieldOverlayMode === 2) {
      if (!Number.isFinite(w) || w <= 0.0) continue;
      if (fieldOverlayMode === 1) {
        const u = dispField[b + 0];
        const v = dispField[b + 1];
        if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
        m = Math.hypot(u, v);
      } else {
        const vort = dispField[b + 2];
        if (!Number.isFinite(vort)) continue;
        m = Math.abs(vort);
      }
    } else if (fieldOverlayMode === 3) {
      const ftle = dispField[b + 4];
      if (!Number.isFinite(ftle) || ftle <= -900.0) continue;
      m = Math.max(0.0, ftle);
    } else {
      continue;
    }
    if (Number.isFinite(m) && m > maxMag) maxMag = m;
  }

  const normMax = fieldOverlayMode === 1
    ? Math.max(24.0, maxMag)
    : (fieldOverlayMode === 2 ? Math.max(0.5, maxMag) : Math.max(0.25, maxMag));

  for (let c = 0; c < FIELD_CELLS; c++) {
    const b = c * FIELD_STRIDE;
    const p = c * 4;
    const w = dispField[b + 5];
    let r = 0;
    let g = 0;
    let bl = 0;
    let a = 0;

    if (fieldOverlayMode === 4) {
      const phi = dispModal[c];
      if (Number.isFinite(phi) && phi >= -3.15 && phi <= 3.15) {
        r = Math.round(127.5 + 118.0 * Math.cos(phi));
        g = Math.round(127.5 + 118.0 * Math.cos(phi - 2.0943951));
        bl = Math.round(127.5 + 118.0 * Math.cos(phi + 2.0943951));
        a = 145;
      }
    } else if (fieldOverlayMode === 3) {
      const ftle = dispField[b + 4];
      if (Number.isFinite(ftle) && ftle > -900.0) {
        const wGain = Number.isFinite(w) && w > 0.0 ? Math.min(1.0, Math.max(0.25, w)) : 0.85;
        const t = Math.min(1.0, Math.max(0.0, ftle) / normMax);
        r = Math.round(70 + t * 185);
        g = Math.round(24 + t * t * 205);
        bl = Math.round(135 + (t < 0.6 ? t * 160 : (1.0 - t) * 240));
        a = Math.round((t < 0.08 ? 4 : 26 + t * 154) * wGain);
      }
    } else if (Number.isFinite(w) && w > 0.0) {
      const wGain = Math.min(1.0, Math.max(0.25, w));
      if (fieldOverlayMode === 1) {
        const u = dispField[b + 0];
        const v = dispField[b + 1];
        if (Number.isFinite(u) && Number.isFinite(v)) {
          const t = Math.min(1.0, Math.hypot(u, v) / normMax);
          r = Math.round(t < 0.55 ? 8 + t * 40 : 30 + (t - 0.55) * 490);
          g = Math.round(35 + t * 205);
          bl = Math.round(90 + (t < 0.65 ? t * 240 : (1.0 - t) * 420));
          a = Math.round((32 + t * 138) * wGain);
        }
      } else if (fieldOverlayMode === 2) {
        const vort = dispField[b + 2];
        if (Number.isFinite(vort)) {
          const s = Math.max(-1.0, Math.min(1.0, vort / normMax));
          const mag = Math.abs(s);
          if (s >= 0.0) {
            r = Math.round(180 + mag * 75);
            g = Math.round(70 + mag * 110);
            bl = 36;
          } else {
            r = 44;
            g = Math.round(115 + mag * 115);
            bl = 255;
          }
          a = Math.round((mag < 0.06 ? 6 : 24 + mag * 150) * wGain);
        }
      }
    }

    fieldTexRgba[p] = r;
    fieldTexRgba[p + 1] = g;
    fieldTexRgba[p + 2] = bl;
    fieldTexRgba[p + 3] = a;
    if (a <= 0) {
      fieldTexGlPremul[p] = 0;
      fieldTexGlPremul[p + 1] = 0;
      fieldTexGlPremul[p + 2] = 0;
      fieldTexGlPremul[p + 3] = 0;
    } else {
      const aF = a / 255.0;
      fieldTexGlPremul[p] = Math.round(r * aF);
      fieldTexGlPremul[p + 1] = Math.round(g * aF);
      fieldTexGlPremul[p + 2] = Math.round(bl * aF);
      fieldTexGlPremul[p + 3] = a;
    }
  }
  return normMax;
}

function computeConvexHullCount(n) {
  if (n < 3) return 0;
  for (let i = 0; i < n; i++) hullIdx[i] = i;
  for (let i = 1; i < n; i++) {
    const key = hullIdx[i];
    const kx = hullPtsX[key];
    const ky = hullPtsY[key];
    let j = i - 1;
    while (j >= 0) {
      const cj = hullIdx[j];
      const cx = hullPtsX[cj];
      if (cx < kx || (cx === kx && hullPtsY[cj] <= ky)) break;
      hullIdx[j + 1] = cj;
      j--;
    }
    hullIdx[j + 1] = key;
  }
  let k = 0;
  for (let i = 0; i < n; i++) {
    const idx = hullIdx[i];
    while (k >= 2) {
      const a = hullPoly[k - 2];
      const b = hullPoly[k - 1];
      const cross =
        (hullPtsX[b] - hullPtsX[a]) * (hullPtsY[idx] - hullPtsY[a]) -
        (hullPtsY[b] - hullPtsY[a]) * (hullPtsX[idx] - hullPtsX[a]);
      if (cross > 0.0) break;
      k--;
    }
    hullPoly[k++] = idx;
  }
  const lowerK = k + 1;
  for (let i = n - 2; i >= 0; i--) {
    const idx = hullIdx[i];
    while (k >= lowerK) {
      const a = hullPoly[k - 2];
      const b = hullPoly[k - 1];
      const cross =
        (hullPtsX[b] - hullPtsX[a]) * (hullPtsY[idx] - hullPtsY[a]) -
        (hullPtsY[b] - hullPtsY[a]) * (hullPtsX[idx] - hullPtsX[a]);
      if (cross > 0.0) break;
      k--;
    }
    hullPoly[k++] = idx;
  }
  return Math.max(0, k - 1);
}

function renderFieldFallback2D(W, H, visCount, sx, sy) {
  if (fieldOverlayMode >= 1 && fieldOverlayMode <= 4) {
    if (!field2dCanvas || field2dCanvas.width !== FIELD_GRID_W || field2dCanvas.height !== FIELD_GRID_H) {
      field2dCanvas = new OffscreenCanvas(FIELD_GRID_W, FIELD_GRID_H);
      field2dCtx = field2dCanvas.getContext('2d');
      field2dImgData = new ImageData(FIELD_GRID_W, FIELD_GRID_H);
    }
    field2dImgData.data.set(fieldTexRgba);
    field2dCtx.putImageData(field2dImgData, 0, 0);
    hudCtx.save();
    hudCtx.imageSmoothingEnabled = fieldOverlayMode !== 4;
    if (fieldOverlayMode !== 4) hudCtx.imageSmoothingQuality = 'low';
    hudCtx.drawImage(field2dCanvas, 0, 0, W, H);
    hudCtx.restore();
  }

  if (visCount > 0) {
    hudCtx.save();
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeStyle = 'rgba(0, 229, 255, 0.55)';
    hudCtx.beginPath();
    for (let i = 0; i < visCount; i++) {
      const b = i * 6;
      const px = microInstBuf[b] * sx;
      const py = microInstBuf[b + 1] * sy;
      const vx = microInstBuf[b + 2];
      const vy = microInstBuf[b + 3];
      const spd = Math.hypot(vx, vy);
      if (spd > 2.0) {
        const gain = Math.min(0.16, 22.0 / spd);
        hudCtx.moveTo(px, py);
        hudCtx.lineTo(px + vx * gain * sx, py + vy * gain * sy);
      }
    }
    hudCtx.stroke();

    hudCtx.fillStyle = 'rgba(0, 229, 255, 0.90)';
    for (let i = 0; i < visCount; i++) {
      const b = i * 6;
      if (microInstBuf[b + 4] === 1.0) {
        hudCtx.fillRect(microInstBuf[b] * sx - 1.5, microInstBuf[b + 1] * sy - 1.5, 3, 3);
      }
    }
    hudCtx.fillStyle = 'rgba(255, 158, 44, 0.95)';
    for (let i = 0; i < visCount; i++) {
      const b = i * 6;
      if (microInstBuf[b + 4] === 3.0) {
        hudCtx.fillRect(microInstBuf[b] * sx - 1.8, microInstBuf[b + 1] * sy - 1.8, 3.6, 3.6);
      }
    }
    hudCtx.fillStyle = 'rgba(245, 158, 11, 0.55)';
    for (let i = 0; i < visCount; i++) {
      const b = i * 6;
      if (microInstBuf[b + 4] === 2.0) {
        hudCtx.fillRect(microInstBuf[b] * sx - 1.0, microInstBuf[b + 1] * sy - 1.0, 2, 2);
      }
    }
    hudCtx.restore();
  }
}

function renderChronos1000FieldLayer(W, H, dispSlot, effK, sx, sy) {
  const ringOk = dispSlot >= 0 && dispSlot < RING_LEN && swarmRingValid[dispSlot] === 1;
  let dispMicro = null;
  let dispField = null;
  let dispBodies = null;
  let dispModal = null;
  let dispSwarm = null;

  if (ringOk) {
    dispMicro = swarmRingMicro[dispSlot];
    dispField = swarmRingField[dispSlot];
    dispBodies = swarmRingBodies[dispSlot];
    dispModal = swarmRingModal[dispSlot];
    dispSwarm = swarmRingTelem[dispSlot];
  } else if (hasChronos1000Exports() && effK === 0 && swarmTelemView && swarmTelemView[1] > 0.5) {
    dispMicro = microTracksView;
    dispField = continuumFieldView;
    dispBodies = rigidBodiesView;
    dispModal = modalFieldView;
    dispSwarm = swarmTelemView;
  } else {
    lastFieldRenderMs = 0.0;
    return;
  }

  if (!dispSwarm || (dispSwarm[1] <= 0.5 && fieldOverlayMode === 0)) {
    lastFieldRenderMs = 0.0;
    return;
  }

  const t0 = performance.now();
  bodyMemHead.fill(-1);
  let visCount = 0;
  let inspFound = false;
  let inspX = 0.0;
  let inspY = 0.0;
  let inspVx = 0.0;
  let inspVy = 0.0;
  let inspBody = 0;
  let inspFtle = 0.0;

  for (let i = 0; i < MAX_MICRO; i++) {
    const base = i * MICRO_STRIDE;
    const st = intOrNull(dispMicro[base + 5], 1, 2);
    if (st === null) continue;
    const px = dispMicro[base + 1];
    const py = dispMicro[base + 2];
    const vx = dispMicro[base + 3];
    const vy = dispMicro[base + 4];
    if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(vx) || !Number.isFinite(vy)) {
      continue;
    }
    const tid = intOrNull(dispMicro[base + 0], 1, 16777215) ?? 0;
    const bId = intOrNull(dispMicro[base + 6], 0, MAX_RIGID_BODIES) ?? 0;
    const isInsp = tid > 0 && tid === inspectedMicroTid;
    const mOff = visCount * 6;
    microInstBuf[mOff] = px;
    microInstBuf[mOff + 1] = py;
    microInstBuf[mOff + 2] = vx;
    microInstBuf[mOff + 3] = vy;
    microInstBuf[mOff + 4] = st === 1 ? (bId > 0 ? 3.0 : 1.0) : 2.0;
    microInstBuf[mOff + 5] = isInsp ? 1.0 : 0.0;
    visCount++;

    if (st === 1 && bId >= 1 && bId <= MAX_RIGID_BODIES) {
      bodyMemNext[i] = bodyMemHead[bId];
      bodyMemHead[bId] = i;
    }
    if (isInsp) {
      const rawFtle = dispMicro[base + 7];
      inspFound = true;
      inspX = px;
      inspY = py;
      inspVx = vx;
      inspVy = vy;
      inspBody = bId;
      inspFtle = Number.isFinite(rawFtle) && rawFtle > -900.0 ? rawFtle : null;
    }
  }

  const dispSwarmActive = Boolean(dispSwarm && dispSwarm[1] > 0.5);
  const maskVal = intOrNull(dispSwarm[16], 0, 31) ?? 0;
  const eulerianFieldMode = fieldOverlayMode >= 1 && fieldOverlayMode <= 4;
  let fieldNormMax = 1.0;
  if (eulerianFieldMode && dispField) {
    fieldNormMax = packFieldOverlayTexture(dispField, dispModal, maskVal, dispSwarmActive);
  }

  if (ensureFieldWebGl2(W, H)) {
    const gl = fieldGlCtx;
    gl.viewport(0, 0, W, H);
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (eulerianFieldMode && dispField) {
      gl.useProgram(fieldGlQuadProg);
      gl.bindVertexArray(fieldGlQuadVao);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, fieldGlTex);
      const texFilter = fieldOverlayMode === 4 ? gl.NEAREST : gl.LINEAR;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, texFilter);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, texFilter);
      gl.texSubImage2D(
        gl.TEXTURE_2D, 0, 0, 0, FIELD_GRID_W, FIELD_GRID_H,
        gl.RGBA, gl.UNSIGNED_BYTE, fieldTexGlPremul
      );
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    if (visCount > 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, fieldGlInstVbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, microInstBuf, 0, visCount * 6);

      gl.useProgram(fieldGlStemProg);
      gl.uniform2f(fieldGlStemUInv, 1.0 / frameW, 1.0 / frameH);
      gl.bindVertexArray(fieldGlStemVao);
      gl.drawArraysInstanced(gl.LINES, 0, 2, visCount);

      gl.useProgram(fieldGlPipProg);
      gl.uniform2f(fieldGlPipUInv, 1.0 / frameW, 1.0 / frameH);
      gl.uniform2f(fieldGlPipUHalf, 4.2 / W, 4.2 / H);
      gl.bindVertexArray(fieldGlPipVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, visCount);
    }

    gl.bindVertexArray(null);
    hudCtx.drawImage(fieldGlCanvas, 0, 0, W, H);
    lastFieldBackend = 'webgl2';
  } else {
    renderFieldFallback2D(W, H, visCount, sx, sy);
    lastFieldBackend = 'canvas2d';
  }

  // Rigid-motion SE(2) convex hulls & angular velocity labels (gated strictly by validity bit 2)
  if (dispBodies && (maskVal & 4) !== 0) {
    hudCtx.save();
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    for (let b = 0; b < MAX_RIGID_BODIES; b++) {
      const rb = parseRigidBodyDescriptor(dispBodies, b * BODY_STRIDE);
      if (!rb) {
        continue;
      }
      const bId = rb.bodyId;
      const memCount = rb.memberCount;
      const omega = rb.omegaRadS;
      const cx = rb.cx * sx;
      const cy = rb.cy * sy;

      let nPts = 0;
      let mIdx = bodyMemHead[bId];
      let minY = cy - 22.0;
      while (mIdx !== -1 && nPts < MAX_MICRO) {
        const px = dispMicro[mIdx * MICRO_STRIDE + 1] * sx;
        const py = dispMicro[mIdx * MICRO_STRIDE + 2] * sy;
        hullPtsX[nPts] = px;
        hullPtsY[nPts] = py;
        if (py < minY) minY = py;
        nPts++;
        mIdx = bodyMemNext[mIdx];
      }

      const hLen = computeConvexHullCount(nPts);
      if (hLen >= 3) {
        hudCtx.beginPath();
        hudCtx.moveTo(hullPtsX[hullPoly[0]], hullPtsY[hullPoly[0]]);
        for (let k = 1; k < hLen; k++) {
          hudCtx.lineTo(hullPtsX[hullPoly[k]], hullPtsY[hullPoly[k]]);
        }
        hudCtx.closePath();
        hudCtx.fillStyle = 'rgba(255, 158, 44, 0.10)';
        hudCtx.fill();
        hudCtx.strokeStyle = 'rgba(255, 178, 72, 0.88)';
        hudCtx.lineWidth = 1.6;
        hudCtx.stroke();
      }

      // Rotational spin arc at body centroid
      hudCtx.strokeStyle = '#ff9e2c';
      hudCtx.lineWidth = 1.6;
      hudCtx.beginPath();
      const arcR = 11.0;
      if (omega >= 0) {
        hudCtx.arc(cx, cy, arcR, -Math.PI * 0.75, Math.PI * 0.55, false);
      } else {
        hudCtx.arc(cx, cy, arcR, Math.PI * 0.75, -Math.PI * 0.55, true);
      }
      hudCtx.stroke();

      const lbl = `B${bId} · ω=${omega >= 0 ? '+' : ''}${omega.toFixed(2)} rad/s · N=${memCount}`;
      const lw = hudCtx.measureText(lbl).width + 10;
      const lx = Math.max(8, Math.min(W - lw - 8, cx - lw * 0.5));
      const ly = Math.max(42, Math.min(H - 26, minY - 18));
      hudCtx.fillStyle = 'rgba(8, 16, 30, 0.88)';
      hudCtx.fillRect(lx, ly, lw, 16);
      hudCtx.strokeStyle = 'rgba(255, 158, 44, 0.75)';
      hudCtx.lineWidth = 1.0;
      hudCtx.strokeRect(lx, ly, lw, 16);
      hudCtx.fillStyle = '#ffd18a';
      hudCtx.fillText(lbl, lx + 5, ly + 11);
    }
    hudCtx.restore();
  }

  // Inspected point/track diamond reticle + callout
  if (inspFound) {
    const ix = inspX * sx;
    const iy = inspY * sy;
    const spd = Math.hypot(inspVx, inspVy);
    hudCtx.save();
    hudCtx.strokeStyle = '#fde047';
    hudCtx.lineWidth = 1.8;
    hudCtx.beginPath();
    hudCtx.moveTo(ix, iy - 10);
    hudCtx.lineTo(ix + 10, iy);
    hudCtx.lineTo(ix, iy + 10);
    hudCtx.lineTo(ix - 10, iy);
    hudCtx.closePath();
    hudCtx.stroke();
    const tag = `μT${inspectedMicroTid}${inspBody > 0 ? ` · B${inspBody}` : ''} · ${spd.toFixed(1)} px/s | FTLE ${inspFtle !== null ? inspFtle.toFixed(2) : '--'}`;
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    const tw = hudCtx.measureText(tag).width + 10;
    const tx = Math.max(8, Math.min(W - tw - 8, ix + 14));
    const ty = Math.max(44, Math.min(H - 26, iy - 16));
    hudCtx.fillStyle = 'rgba(10, 18, 32, 0.90)';
    hudCtx.fillRect(tx, ty, tw, 16);
    hudCtx.strokeStyle = '#fde047';
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeRect(tx, ty, tw, 16);
    hudCtx.fillStyle = '#fef08a';
    hudCtx.fillText(tag, tx + 5, ty + 11);
    hudCtx.restore();
  }

  // Non-color Field Overlay legend caption (top-left clear of header strip)
  if (eulerianFieldMode) {
    let cap = '';
    if (fieldOverlayMode === 1) cap = `FIELD: VELOCITY |v| (0..${fieldNormMax.toFixed(0)} px/s)`;
    else if (fieldOverlayMode === 2) cap = `FIELD: VORTICITY CURL v (±${fieldNormMax.toFixed(2)} s⁻¹)`;
    else if (fieldOverlayMode === 3) cap = `FIELD: FTLE 16F RIDGES (0..${fieldNormMax.toFixed(2)} s⁻¹)`;
    else cap = 'FIELD: MODAL PHASE ([-π..+π] rad)';
    if (!dispSwarmActive) cap += ' · STANDBY';
    hudCtx.save();
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    const cw = hudCtx.measureText(cap).width + 12;
    hudCtx.fillStyle = 'rgba(7, 13, 24, 0.84)';
    hudCtx.fillRect(10, 44, cw, 17);
    hudCtx.strokeStyle = 'rgba(0, 229, 255, 0.45)';
    hudCtx.lineWidth = 1;
    hudCtx.strokeRect(10, 44, cw, 17);
    hudCtx.fillStyle = '#bae6fd';
    hudCtx.fillText(cap, 16, 56);
    hudCtx.restore();
  }

  lastFieldRenderMs = performance.now() - t0;
}

function invalidateWave2RingStamp(stampOffsetWords) {
  for (let s = 0; s < RING_LEN; s++) {
    w2CoreRing[s][4 + stampOffsetWords] = 0;
  }
}

function propagateWave2FaultToRing(ringValid, ringTelem, ringPlanes, liveTelem) {
  if (!liveTelem) return;
  for (let s = 0; s < RING_LEN; s++) {
    if (ringValid[s] === 1) {
      ringTelem[s].set(liveTelem);
      ringPlanes[s].fill(0.0);
    }
  }
}

function syncWave2RingEnabledBits() {
  if (!wave2CoreAbiValid || !wave2TelemView) return;
  const bits = wave2TelemView[2] >>> 0;
  for (let s = 0; s < RING_LEN; s++) {
    if (w2CoreRingValid[s] === 1) {
      w2CoreRing[s][2] = bits;
    }
  }
}

function resolveWave2RingOrLive(dispSlot, effK, ringValid, ringTelem, ringPlanes, liveOk, liveTelem, livePlanes) {
  const delayedRing = hasAnyRenderedFrame() && effK > 0;
  if (dispSlot >= 0 && dispSlot < RING_LEN && ringValid[dispSlot] === 1) {
    return { t: ringTelem[dispSlot], p: ringPlanes[dispSlot] };
  }
  if (delayedRing) {
    return null;
  }
  if (liveOk && liveTelem && livePlanes) {
    return { t: liveTelem, p: livePlanes };
  }
  return null;
}

function resolveWave2ConsArr(dispSlot, effK) {
  if (manualConsTestActive && wave2ConsAbiValid && conservationTelemView) {
    return conservationTelemView;
  }
  if (hasAnyRenderedFrame() && effK > 0) {
    return (dispSlot >= 0 && dispSlot < RING_LEN && consRingValid[dispSlot] === 1)
      ? consRingTelem[dispSlot]
      : null;
  }
  if (dispSlot >= 0 && dispSlot < RING_LEN && consRingValid[dispSlot] === 1) {
    return consRingTelem[dispSlot];
  }
  const st = wave2CoreAbiValid ? decodeWave2Stamp(32) : null;
  if (wave2ConsAbiValid && conservationTelemView && st && st.valid) {
    return conservationTelemView;
  }
  return null;
}

function drawHudCaptionBox(x, y, text, borderColor, bgColor, textColor) {
  hudCtx.save();
  hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  const cw = hudCtx.measureText(text).width + 12;
  hudCtx.fillStyle = bgColor;
  hudCtx.fillRect(x, y, cw, 17);
  hudCtx.strokeStyle = borderColor;
  hudCtx.lineWidth = 1;
  hudCtx.strokeRect(x, y, cw, 17);
  hudCtx.fillStyle = textColor;
  hudCtx.fillText(text, x + 6, y + 12);
  hudCtx.restore();
}

function colorizeAbosToRgba(planes, validMask, dstRgba) {
  let maxPhi = 0.0;
  for (let i = 0; i < BOS_N; i++) {
    if (validMask[i]) {
      const v = planes[i];
      if (Number.isFinite(v) && v > maxPhi) maxPhi = v;
    }
  }
  const invMax = 1.0 / Math.max(1e-4, maxPhi);
  for (let i = 0; i < BOS_N; i++) {
    const o = i * 4;
    if (!validMask[i]) {
      dstRgba[o] = 0; dstRgba[o + 1] = 0; dstRgba[o + 2] = 0; dstRgba[o + 3] = 0;
      continue;
    }
    const t = Math.max(0.0, Math.min(1.0, planes[i] * invMax));
    if (t < 0.06) {
      dstRgba[o] = 0; dstRgba[o + 1] = 0; dstRgba[o + 2] = 0; dstRgba[o + 3] = 0;
      continue;
    }
    let r, g, b;
    if (t < 0.5) {
      const s = t * 2.0;
      r = 12 + 30 * s;
      g = 56 + 170 * s;
      b = 148 + 107 * s;
    } else {
      const s = (t - 0.5) * 2.0;
      r = 42 + 213 * s;
      g = 226 + 24 * s;
      b = 255 - 165 * s;
    }
    dstRgba[o] = r | 0;
    dstRgba[o + 1] = g | 0;
    dstRgba[o + 2] = b | 0;
    dstRgba[o + 3] = Math.round(35 + 175 * t);
  }
}

function renderWave2Overlays(W, H, dispSlot, effK) {
  const isHodgeMode = fieldOverlayMode >= 8 && fieldOverlayMode <= 10;
  const isVibroMode = fieldOverlayMode === 11;
  const isDepthMode = fieldOverlayMode === 12;
  const isWindMode = fieldOverlayMode === 13;
  const isElastoSurfMode = fieldOverlayMode === 14;
  const isAcousticBosMode = fieldOverlayMode === 15;

  if (isHodgeMode && w2OverlayCtx && w2OverlayImgData) {
    const src = resolveWave2RingOrLive(
      dispSlot,
      effK,
      hodgeRingValid,
      hodgeRingTelem,
      hodgeRingPlanes,
      wave2HodgeAbiValid,
      hodgeTelemView,
      hodgePlanesView
    );
    const liveHodgeNonReady = hodgeReqMode > 0 && wave2HodgeAbiValid && hodgeTelemView && hodgeTelemView[0] !== 2;
    const ht = src ? src.t : (liveHodgeNonReady ? hodgeTelemView : null);
    const hp = src ? src.p : null;
    const hState = ht ? (ht[0] >>> 0) : 0;

    if (hp && hState === 2) {
      const uOff = fieldOverlayMode === 8 ? 0 : (fieldOverlayMode === 9 ? FIELD_CELLS * 2 : FIELD_CELLS * 4);
      const vOff = uOff + FIELD_CELLS;
      let maxMag = 0.0;
      for (let c = 0; c < FIELD_CELLS; c++) {
        const u = hp[uOff + c];
        const v = hp[vOff + c];
        if (Number.isFinite(u) && Number.isFinite(v)) {
          const m = Math.hypot(u, v);
          if (m > maxMag) maxMag = m;
        }
      }
      const normMax = Math.max(0.25, maxMag);
      const px = w2OverlayImgData.data;
      for (let c = 0; c < FIELD_CELLS; c++) {
        const o = c * 4;
        const u = hp[uOff + c];
        const v = hp[vOff + c];
        if (!Number.isFinite(u) || !Number.isFinite(v)) {
          px[o] = 0; px[o + 1] = 0; px[o + 2] = 0; px[o + 3] = 0;
          continue;
        }
        const t = Math.min(1.0, Math.hypot(u, v) / normMax);
        if (t < 0.02) {
          px[o] = 0; px[o + 1] = 0; px[o + 2] = 0; px[o + 3] = 0;
          continue;
        }
        if (fieldOverlayMode === 8) {
          px[o] = Math.round(14 + t * 116);
          px[o + 1] = Math.round(115 + t * 140);
          px[o + 2] = Math.round(205 + t * 50);
        } else if (fieldOverlayMode === 9) {
          px[o] = Math.round(28 + t * 227);
          px[o + 1] = Math.round(150 + t * 85);
          px[o + 2] = Math.round(185 - t * 145);
        } else {
          px[o] = Math.round(135 + t * 120);
          px[o + 1] = Math.round(40 + t * 115);
          px[o + 2] = Math.round(205 + t * 50);
        }
        px[o + 3] = Math.round(28 + t * 158);
      }
      w2OverlayCtx.putImageData(w2OverlayImgData, 0, 0);
      hudCtx.save();
      hudCtx.imageSmoothingEnabled = true;
      hudCtx.imageSmoothingQuality = 'low';
      hudCtx.drawImage(w2OverlayCanvas, 0, 0, W, H);
      hudCtx.restore();
    }

    const stLabel = HODGE_STATE_LABELS[Math.max(0, Math.min(3, hState))] || 'OFF';
    let hCap =
      fieldOverlayMode === 8
        ? 'HODGE EXACT |grad phi| (64x36)'
        : fieldOverlayMode === 9
        ? 'HODGE COEXACT |curl* psi| (64x36)'
        : 'HODGE HARMONIC |h| (64x36)';
    hCap += ` · ${stLabel}`;
    if (ht && hState === 2) {
      const ex = u32BitsToF32(ht[12]);
      const co = u32BitsToF32(ht[13]);
      const ha = u32BitsToF32(ht[14]);
      const beta1 = ht[7] >>> 0;
      const vAct = ht[3] >>> 0;
      hCap += ` · Ex ${(ex * 100).toFixed(0)}% Co ${(co * 100).toFixed(0)}% Ha ${(ha * 100).toFixed(0)}% · b1=${beta1} V=${vAct}`;
    } else if (hState === 3) {
      hCap += ' [UNCONVERGED]';
    }
    drawHudCaptionBox(
      10,
      44,
      hCap,
      hState === 3 ? '#ff4d6d' : 'rgba(56, 242, 122, 0.55)',
      hState === 3 ? 'rgba(38, 12, 16, 0.88)' : 'rgba(7, 13, 24, 0.84)',
      hState === 3 ? '#fecdd3' : '#a7f3d0'
    );
  } else if (isVibroMode && w2OverlayCtx && w2OverlayImgData) {
    const src = resolveWave2RingOrLive(
      dispSlot,
      effK,
      vibroRingValid,
      vibroRingTelem,
      vibroRingPlanes,
      wave2VibroAbiValid,
      vibroTelemView,
      vibroPlanesView
    );
    const liveVibroNonReady = rsCalibConfig.mode > 0 && wave2VibroAbiValid && vibroTelemView && vibroTelemView[0] !== 3;
    const vt = src ? src.t : (liveVibroNonReady ? vibroTelemView : null);
    const vp = src ? src.p : null;
    const vState = vt ? (vt[0] >>> 0) : 0;
    const vReason = vt ? (vt[2] >>> 0) : 0;

    const vMode = vt ? (vt[1] >>> 0) : (rsCalibConfig.mode >>> 0);
    const isRsOnly = vMode === 1;
    if (vp && vState === 3) {
      const px = w2OverlayImgData.data;
      if (isRsOnly) {
        const domAmp = vt ? u32BitsToF32(vt[12]) : 0.0;
        let maxAmp = Math.max(0.05, Number.isFinite(domAmp) ? domAmp : 0.0);
        for (let c = 0; c < FIELD_CELLS; c++) {
          const a = vp[c];
          if (Number.isFinite(a) && a > maxAmp) maxAmp = a;
        }
        const invMax = 1.0 / maxAmp;
        for (let c = 0; c < FIELD_CELLS; c++) {
          const o = c * 4;
          const amp = vp[c];
          if (!Number.isFinite(amp) || amp <= 0.015) {
            px[o] = 0; px[o + 1] = 0; px[o + 2] = 0; px[o + 3] = 0;
            continue;
          }
          const t = Math.min(1.0, Math.max(0.0, amp * invMax));
          px[o] = Math.round(28 + t * 227);
          px[o + 1] = Math.round(185 + t * 45 - t * t * 65);
          px[o + 2] = Math.round(235 - t * 170);
          px[o + 3] = Math.round(45 + t * 150);
        }
      } else {
        const gCrit = vt ? Math.max(0.05, Math.min(0.99, u32BitsToF32(vt[7]))) : 0.45;
        for (let c = 0; c < FIELD_CELLS; c++) {
          const o = c * 4;
          const coh = vp[FIELD_CELLS + c];
          if (!Number.isFinite(coh) || coh <= 0.02) {
            px[o] = 0; px[o + 1] = 0; px[o + 2] = 0; px[o + 3] = 0;
            continue;
          }
          const t = Math.min(1.0, Math.max(0.0, coh));
          const coupled = t >= gCrit;
          px[o] = coupled ? Math.round(24 + t * 110) : 54;
          px[o + 1] = coupled ? Math.round(190 + t * 65) : Math.round(90 + t * 60);
          px[o + 2] = coupled ? Math.round(160 + t * 95) : 140;
          px[o + 3] = coupled ? Math.round(65 + t * 135) : Math.round(16 + t * 48);
        }
      }
      w2OverlayCtx.putImageData(w2OverlayImgData, 0, 0);
      hudCtx.save();
      hudCtx.imageSmoothingEnabled = true;
      hudCtx.imageSmoothingQuality = 'low';
      hudCtx.drawImage(w2OverlayCanvas, 0, 0, W, H);
      hudCtx.restore();
    }

    const stLabel = VIBRO_STATE_LABELS[Math.max(0, Math.min(4, vState))] || 'OFF';
    let vCap = isRsOnly
      ? `RS VIBROMETRY AMP (64x36) · ${stLabel}`
      : `RS WELCH COH g2 (64x36) · ${stLabel}`;
    if (vt && vState >= 2) {
      const kSeg = vt[3] >>> 0;
      const domF = u32BitsToF32(vt[5]);
      const maxC = u32BitsToF32(vt[6]);
      const sincH = u32BitsToF32(vt[8]);
      const domAmp = u32BitsToF32(vt[12]);
      if (isRsOnly) {
        vCap += ` · ${domF.toFixed(1)}Hz · amp=${domAmp.toFixed(2)}px · H=${sincH.toFixed(2)}`;
      } else {
        vCap += ` · K=${kSeg} · ${domF.toFixed(1)}Hz · g2=${maxC.toFixed(3)} · H=${sincH.toFixed(2)}`;
      }
    }
    if (vReason !== 0) {
      const flags = [];
      for (let b = 0; b < RS_REASON_NAMES.length; b++) {
        if (vReason & (1 << b)) flags.push(RS_REASON_NAMES[b]);
      }
      if (flags.length > 0) vCap += ` [${flags.slice(0, 2).join('+')}]`;
    }
    const isWarn = vState === 4 || vState === 1 || (vReason & (1 << 6)) !== 0;
    drawHudCaptionBox(
      10,
      44,
      vCap,
      isWarn ? '#ff4d6d' : 'rgba(0, 229, 255, 0.55)',
      isWarn ? 'rgba(38, 12, 16, 0.88)' : 'rgba(7, 13, 24, 0.84)',
      isWarn ? '#fecdd3' : '#bae6fd'
    );
  } else if (isDepthMode && depthAbiValid && depthTelemView && depthTelemF32View && depthFieldView) {
    const delayedDepthRing = hasAnyRenderedFrame() && effK > 0;
    const hasSlotDepth = dispSlot >= 0 && dispSlot < RING_LEN && depthRingValid[dispSlot] === 1;
    const dTelemU = hasSlotDepth ? depthRingTelem[dispSlot] : (!delayedDepthRing ? depthTelemView : null);
    const dTelemF = hasSlotDepth ? depthRingTelemF32[dispSlot] : (!delayedDepthRing ? depthTelemF32View : null);
    const dField = hasSlotDepth ? depthRingField[dispSlot] : (!delayedDepthRing ? depthFieldView : null);
    const dStereoRgba = hasSlotDepth ? depthRingStereoRgba[dispSlot] : (!delayedDepthRing ? bosRgbaView : null);
    const dState = dTelemU ? Math.max(0, Math.min(7, dTelemU[2] >>> 0)) : 0;
    const dProv = dTelemU ? Math.max(0, Math.min(6, dTelemU[4] >>> 0)) : 0;
    const foeInliers = dTelemU ? (dTelemU[8] >>> 0) : 0;
    const ttcCnt = dTelemU ? (dTelemU[7] >>> 0) : 0;
    const validCells = dTelemU ? (dTelemU[9] >>> 0) : 0;
    const stereoNodes = dTelemU ? (dTelemU[13] >>> 0) : 0;
    const foeX = dTelemF ? dTelemF[22] : 0.0;
    const foeY = dTelemF ? dTelemF[23] : 0.0;
    const minTtc = dTelemF ? dTelemF[24] : 0.0;
    const bestR2 = dTelemF ? dTelemF[25] : 0.0;
    const medZ = dTelemF ? dTelemF[29] : 0.0;
    const minZ = dTelemF ? dTelemF[30] : 0.0;
    const maxZ = dTelemF ? dTelemF[31] : 0.0;

    if (dState === 6 && bosOverlayCtx && bosOverlayImgData && dStereoRgba) {
      bosOverlayImgData.data.set(dStereoRgba);
      bosOverlayCtx.putImageData(bosOverlayImgData, 0, 0);
      hudCtx.save();
      hudCtx.imageSmoothingEnabled = true;
      hudCtx.imageSmoothingQuality = 'low';
      hudCtx.drawImage(bosOverlayCanvas, 0, 0, W, H);
      hudCtx.restore();
    } else if (dField && validCells > 0 && w2OverlayCtx && w2OverlayImgData) {
      let minIz = Infinity;
      let maxIz = 0.0;
      for (let c = 0; c < FIELD_CELLS; c++) {
        const iz = dField[c * 2];
        const cf = dField[c * 2 + 1];
        if (cf > 0.0 && Number.isFinite(iz) && iz > 1e-6) {
          if (iz < minIz) minIz = iz;
          if (iz > maxIz) maxIz = iz;
        }
      }
      const invSpan = 1.0 / Math.max(1e-4, maxIz - minIz);
      const px = w2OverlayImgData.data;
      for (let c = 0; c < FIELD_CELLS; c++) {
        const o = c * 4;
        const iz = dField[c * 2];
        const cf = dField[c * 2 + 1];
        if (cf <= 0.0 || !Number.isFinite(iz) || iz <= 1e-6) {
          px[o] = 0; px[o + 1] = 0; px[o + 2] = 0; px[o + 3] = 0;
          continue;
        }
        const t = Math.max(0.0, Math.min(1.0, (iz - minIz) * invSpan));
        let r, g, b;
        if (t < 0.35) {
          const s = t / 0.35;
          r = 18 + 18 * s; g = 44 + 132 * s; b = 118 + 106 * s;
        } else if (t < 0.70) {
          const s = (t - 0.35) / 0.35;
          r = 36 + 206 * s; g = 176 + 42 * s; b = 224 - 154 * s;
        } else {
          const s = (t - 0.70) / 0.30;
          r = 242 + 13 * s; g = 218 - 138 * s; b = 70 - 38 * s;
        }
        px[o] = r | 0;
        px[o + 1] = g | 0;
        px[o + 2] = b | 0;
        px[o + 3] = Math.round(70 + 135 * Math.min(1.0, cf));
      }
      w2OverlayCtx.putImageData(w2OverlayImgData, 0, 0);
      hudCtx.save();
      hudCtx.imageSmoothingEnabled = true;
      hudCtx.imageSmoothingQuality = 'low';
      hudCtx.drawImage(w2OverlayCanvas, 0, 0, W, H);
      hudCtx.restore();
    }

    if (foeInliers >= 3 && Number.isFinite(foeX) && Number.isFinite(foeY)) {
      const fx = foeX * (W / Math.max(1, frameW));
      const fy = foeY * (H / Math.max(1, frameH));
      hudCtx.save();
      hudCtx.strokeStyle = '#fbbf24';
      hudCtx.lineWidth = 1.5;
      hudCtx.beginPath();
      hudCtx.arc(fx, fy, 11, 0, Math.PI * 2);
      hudCtx.moveTo(fx - 16, fy); hudCtx.lineTo(fx + 16, fy);
      hudCtx.moveTo(fx, fy - 16); hudCtx.lineTo(fx, fy + 16);
      hudCtx.stroke();
      hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillStyle = '#fde68a';
      hudCtx.fillText(
        ttcCnt > 0 && Number.isFinite(minTtc) ? `FoE · TTC ${minTtc.toFixed(2)}s (R2 ${bestR2.toFixed(3)})` : 'FoE',
        fx + 14,
        fy - 6
      );
      hudCtx.restore();
    }

    const stLbl = DEPTH_STATE_LABELS[dState] || 'OFF';
    const provLbl = DEPTH_PROV_LABELS[dProv] || 'NONE';
    let dCap = `DEPTH (${provLbl}) · ${stLbl}`;
    if (dState === 6) {
      dCap += ` · ${stereoNodes}/${BOS_N} · Zmed ${medZ.toFixed(2)}m [${minZ.toFixed(2)}..${maxZ.toFixed(2)}m]`;
    } else if (dState === 2 || dState === 7) {
      dCap += ` · ${validCells}/${FIELD_CELLS} cells`;
      if (Number.isFinite(minZ) && minZ > 0 && Number.isFinite(maxZ) && maxZ > 0) {
        dCap += ` · [${minZ.toFixed(2)}..${maxZ.toFixed(2)}]`;
      }
    }
    if (ttcCnt > 0 && Number.isFinite(minTtc) && minTtc > 0) {
      dCap += ` · minTTC ${minTtc.toFixed(2)}s (R2=${bestR2.toFixed(3)})`;
    }
    const refused = dState === 4 || dState === 5;
    const isNeural = dProv === 6;
    drawHudCaptionBox(
      10,
      44,
      dCap,
      refused ? '#ff4d6d' : (isNeural ? '#f59e0b' : 'rgba(56, 242, 122, 0.55)'),
      refused ? 'rgba(38, 12, 16, 0.88)' : 'rgba(7, 13, 24, 0.84)',
      refused ? '#fecdd3' : (isNeural ? '#fde68a' : '#a7f3d0')
    );
  } else if (isWindMode && windAbiValid && windTelemView && windTelemF32View && windPlanesView) {
    const delayedRing = hasAnyRenderedFrame() && effK > 0;
    const hasSlot = dispSlot >= 0 && dispSlot < RING_LEN && windRingValid[dispSlot] === 1;
    const wtU = hasSlot ? windRingTelem[dispSlot] : (!delayedRing ? windTelemView : null);
    const wtF = hasSlot ? windRingTelemF32[dispSlot] : (!delayedRing ? windTelemF32View : null);
    const wp = hasSlot ? windRingPlanes[dispSlot] : (!delayedRing ? windPlanesView : null);
    const wState = wtU ? Math.max(0, Math.min(4, wtU[0] >>> 0)) : 0;
    const wReason = wtU ? (wtU[2] >>> 0) : 0;
    const beta1 = wtU ? (wtU[5] >>> 0) : 0;
    const nu = wtF ? wtF[11] : 0.0;
    const reNum = wtF ? wtF[13] : 0.0;
    const cpMin = wtF ? wtF[14] : -1.0;
    const cpMax = wtF ? wtF[15] : 1.0;
    const cl = wtF ? wtF[18] : 0.0;
    const cd = wtF ? wtF[19] : 0.0;
    const etaNs = wtF ? wtF[21] : 0.0;
    const holeCx = wtF ? wtF[23] : 0.0;
    const holeCy = wtF ? wtF[24] : 0.0;

    if (wp && (wState === 2 || wState === 3) && w2OverlayCtx && w2OverlayImgData) {
      const span = Math.max(0.35, Math.max(Math.abs(cpMin), Math.abs(cpMax)));
      const invSpan = 1.0 / span;
      const px = w2OverlayImgData.data;
      for (let c = 0; c < FIELD_CELLS; c++) {
        const o = c * 4;
        const cp = wp[c];
        if (!Number.isFinite(cp) || Math.abs(cp) < 1e-4) {
          px[o] = 0; px[o + 1] = 0; px[o + 2] = 0; px[o + 3] = 0;
          continue;
        }
        const s = Math.max(-1.0, Math.min(1.0, cp * invSpan));
        const a = Math.abs(s);
        if (s < 0) {
          // Suction low-pressure zone: cyan-blue
          px[o] = Math.round(14 + (1.0 - a) * 32);
          px[o + 1] = Math.round(135 + a * 95);
          px[o + 2] = 255;
        } else {
          // Stagnation high-pressure zone: amber-crimson
          px[o] = 255;
          px[o + 1] = Math.round(190 - a * 120);
          px[o + 2] = Math.round(60 - a * 40);
        }
        px[o + 3] = Math.round(28 + a * 162);
      }
      w2OverlayCtx.putImageData(w2OverlayImgData, 0, 0);
      hudCtx.save();
      hudCtx.imageSmoothingEnabled = true;
      hudCtx.imageSmoothingQuality = 'low';
      hudCtx.drawImage(w2OverlayCanvas, 0, 0, W, H);
      hudCtx.restore();

      // Draw aerodynamic lift/drag resultant arrow on the primary obstacle void
      if (wState === 3) {
        const rawCx = holeCx > 1.0 ? holeCx : (hodgeTelemView ? u32BitsToF32(hodgeTelemView[23]) : 0.0);
        const rawCy = holeCy > 1.0 ? holeCy : (hodgeTelemView ? u32BitsToF32(hodgeTelemView[24]) : 0.0);
        const vcx = rawCx * (W / Math.max(1, frameW));
        const vcy = rawCy * (H / Math.max(1, frameH));
        if (vcx > 4 && vcy > 4 && vcx < W - 4 && vcy < H - 4) {
          const fMag = Math.hypot(cd, cl);
          const scale = Math.min(58, Math.max(18, fMag * 12.0));
          const fx = fMag > 1e-4 ? (cd / fMag) * scale : 0;
          const fy = fMag > 1e-4 ? (-cl / fMag) * scale : 0;
          hudCtx.save();
          hudCtx.strokeStyle = '#ffd166';
          hudCtx.fillStyle = '#ffd166';
          hudCtx.lineWidth = 2.4;
          hudCtx.beginPath();
          hudCtx.arc(vcx, vcy, 7, 0, Math.PI * 2);
          hudCtx.stroke();
          hudCtx.beginPath();
          hudCtx.moveTo(vcx, vcy);
          hudCtx.lineTo(vcx + fx, vcy + fy);
          hudCtx.stroke();
          hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
          hudCtx.fillText(`CL=${cl.toFixed(2)} CD=${cd.toFixed(2)}`, vcx + 10, vcy - 10);
          hudCtx.restore();
        }
      }
    }

    const stLbl = WIND_STATE_LABELS[wState] || 'OFF';
    let wCap = `WIND TUNNEL Cp (64x36) · ${stLbl}`;
    if (wState === 3) {
      wCap += ` · CL=${cl.toFixed(2)} CD=${cd.toFixed(2)} · b1=${beta1}`;
      if (reNum > 0.5) wCap += ` · Re=${reNum.toFixed(0)}`;
      wCap += ` · eta=${etaNs.toFixed(2)}`;
    } else if (wState === 2) {
      wCap += ` · nu=${nu.toFixed(1)} px2/s · eta=${etaNs.toFixed(2)}`;
    } else if (wState === 4 && wReason !== 0) {
      const fl = [];
      for (let b = 0; b < WIND_REASON_NAMES.length; b++) {
        if (wReason & (1 << b)) fl.push(WIND_REASON_NAMES[b]);
      }
      if (fl.length > 0) wCap += ` [${fl.slice(0, 2).join('+')}]`;
    }
    const isWarn = wState === 4;
    drawHudCaptionBox(
      10,
      44,
      wCap,
      isWarn ? '#ff4d6d' : 'rgba(255, 209, 102, 0.65)',
      isWarn ? 'rgba(38, 12, 16, 0.88)' : 'rgba(7, 13, 24, 0.86)',
      isWarn ? '#fecdd3' : '#fef08a'
    );
  } else if (
    (isElastoSurfMode || isAcousticBosMode) &&
    elastoAbiValid &&
    elastoTelemView &&
    elastoTelemF32View &&
    elastoPlanesView &&
    elastoValidView
  ) {
    const delayedRing = hasAnyRenderedFrame() && effK > 0;
    const hasSlot = dispSlot >= 0 && dispSlot < RING_LEN && elastoRingValid[dispSlot] === 1;
    const etU = hasSlot ? elastoRingTelem[dispSlot] : (!delayedRing ? elastoTelemView : null);
    const etF = hasSlot ? elastoRingTelemF32[dispSlot] : (!delayedRing ? elastoTelemF32View : null);
    const eState = etU ? Math.max(0, Math.min(4, etU[0] >>> 0)) : 0;
    const eMode = etU ? (etU[1] >>> 0) : elastoReqMode;
    const eReason = etU ? (etU[2] >>> 0) : 0;
    const eClass = etU ? Math.max(0, Math.min(3, etU[3] >>> 0)) : 0;
    const eValid = etU ? (etU[4] >>> 0) : 0;
    const beta1 = etU ? (etU[7] >>> 0) : 0;
    const f0 = etF ? etF[8] : 0.0;
    const f1 = etF ? etF[9] : 0.0;
    const medC = etF ? etF[12] : 0.0;
    const alpha = etF ? etF[13] : 0.0;
    const aniso = etF ? etF[14] : 1.0;
    const meanCoh = etF ? etF[15] : 0.0;
    const srcX = etF ? etF[16] : 0.0;
    const srcY = etF ? etF[17] : 0.0;
    const sinkX = etF ? etF[18] : 0.0;
    const sinkY = etF ? etF[19] : 0.0;
    const curlRms = etF ? etF[24] : 0.0;
    const phiRms = etF ? etF[25] : 0.0;
    const cP10 = etF ? etF[26] : 0.0;
    const cP90 = etF ? etF[27] : 0.0;

    if (isAcousticBosMode && eMode === 4 && eState === 3 && bosOverlayCtx && bosOverlayImgData) {
      if (hasSlot) {
        bosOverlayImgData.data.set(elastoRingAbosRgba[dispSlot]);
      } else {
        colorizeAbosToRgba(elastoPlanesView, elastoValidView, bosOverlayImgData.data);
      }
      bosOverlayCtx.putImageData(bosOverlayImgData, 0, 0);
      hudCtx.save();
      hudCtx.imageSmoothingEnabled = true;
      hudCtx.imageSmoothingQuality = 'low';
      hudCtx.drawImage(bosOverlayCanvas, 0, 0, W, H);
      hudCtx.restore();
    } else if (isElastoSurfMode && eMode !== 4 && eState === 3 && w2OverlayCtx && w2OverlayImgData) {
      const ep = hasSlot ? elastoRingPlanes[dispSlot] : elastoPlanesView;
      const ev = hasSlot ? elastoRingValidMask[dispSlot] : elastoValidView;
      const loC = cP10 > 1e-3 ? cP10 * 0.85 : medC * 0.5;
      const hiC = Math.max(loC + 1e-3, cP90 > loC ? cP90 * 1.15 : medC * 1.5);
      const invSpan = 1.0 / (hiC - loC);
      const px = w2OverlayImgData.data;
      for (let c = 0; c < FIELD_CELLS; c++) {
        const o = c * 4;
        const cv = ep[c];
        if (!ev[c] || !Number.isFinite(cv) || cv <= 0.0) {
          px[o] = 0; px[o + 1] = 0; px[o + 2] = 0; px[o + 3] = 0;
          continue;
        }
        const t = Math.max(0.0, Math.min(1.0, (cv - loC) * invSpan));
        // Plasma-like stiffness speed map: deep violet (soft) -> magenta -> amber-gold (stiff)
        let r, g, b;
        if (t < 0.5) {
          const s = t * 2.0;
          r = 42 + 168 * s;
          g = 24 + 44 * s;
          b = 148 + 42 * s;
        } else {
          const s = (t - 0.5) * 2.0;
          r = 210 + 45 * s;
          g = 68 + 166 * s;
          b = 190 - 140 * s;
        }
        px[o] = r | 0;
        px[o + 1] = g | 0;
        px[o + 2] = b | 0;
        px[o + 3] = Math.round(70 + 130 * (0.35 + 0.65 * t));
      }
      w2OverlayCtx.putImageData(w2OverlayImgData, 0, 0);
      hudCtx.save();
      hudCtx.imageSmoothingEnabled = true;
      hudCtx.imageSmoothingQuality = 'low';
      hudCtx.drawImage(w2OverlayCanvas, 0, 0, W, H);
      hudCtx.restore();

      const sx = W / Math.max(1, frameW);
      const sy = H / Math.max(1, frameH);
      hudCtx.save();
      hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      if (srcX > 4 && srcY > 4) {
        const pxX = srcX * sx;
        const pxY = srcY * sy;
        hudCtx.strokeStyle = '#38f27a';
        hudCtx.fillStyle = '#a7f3d0';
        hudCtx.lineWidth = 2.0;
        hudCtx.beginPath();
        hudCtx.arc(pxX, pxY, 9, 0, Math.PI * 2);
        hudCtx.moveTo(pxX - 13, pxY); hudCtx.lineTo(pxX + 13, pxY);
        hudCtx.moveTo(pxX, pxY - 13); hudCtx.lineTo(pxX, pxY + 13);
        hudCtx.stroke();
        hudCtx.fillText('+ SRC', pxX + 12, pxY - 6);
      }
      if (sinkX > 4 && sinkY > 4) {
        const kxX = sinkX * sx;
        const kxY = sinkY * sy;
        hudCtx.strokeStyle = '#ff4d6d';
        hudCtx.fillStyle = '#fecdd3';
        hudCtx.lineWidth = 2.0;
        hudCtx.beginPath();
        hudCtx.arc(kxX, kxY, 9, 0, Math.PI * 2);
        hudCtx.moveTo(kxX - 8, kxY - 8); hudCtx.lineTo(kxX + 8, kxY + 8);
        hudCtx.moveTo(kxX + 8, kxY - 8); hudCtx.lineTo(kxX - 8, kxY + 8);
        hudCtx.stroke();
        hudCtx.fillText('- SINK', kxX + 12, kxY + 12);
      }
      hudCtx.restore();
    }

    const stLbl = ELASTO_STATE_LABELS[eState] || 'OFF';
    let eCap = isAcousticBosMode
      ? `ACOUSTIC BOS (256x128) · ${stLbl}`
      : `ELASTOGRAPHY c(x,y) (64x36) · ${stLbl}`;
    if (eState === 3) {
      if (eMode === 4) {
        eCap += ` · ${f0.toFixed(0)}Hz · |Phi|=${phiRms.toFixed(2)} · curl=${curlRms.toFixed(3)} · ${eValid} nodes`;
      } else {
        const clsLbl = ELASTO_CLASS_LABELS[eClass] || 'NONE';
        const fTxt = f1 > 1.0 ? `${f0.toFixed(0)}+${f1.toFixed(0)}Hz` : `${f0.toFixed(1)}Hz`;
        eCap += ` · ${fTxt} · ${clsLbl}${alpha > 0.05 ? ` (a=${alpha.toFixed(2)})` : ''} · c=${medC.toFixed(0)} · coh=${meanCoh.toFixed(2)}`;
        if (beta1 > 0) eCap += ` · b1=${beta1}`;
        if (Math.abs(aniso - 1.0) > 0.15) eCap += ` · aniso=${aniso.toFixed(2)}`;
      }
    } else if (eState === 4 || eState === 1) {
      eCap += ` [0x${eReason.toString(16)}]`;
    }
    const isWarn = eState === 4 || eState === 1;
    drawHudCaptionBox(
      10,
      44,
      eCap,
      isWarn ? '#ff4d6d' : 'rgba(199, 125, 255, 0.65)',
      isWarn ? 'rgba(38, 12, 16, 0.88)' : 'rgba(7, 13, 24, 0.86)',
      isWarn ? '#fecdd3' : '#e9d5ff'
    );
  }

  const consArr = resolveWave2ConsArr(dispSlot, effK);
  const consStatus = consArr ? (Math.round(consArr[0]) | 0) : 0;
  if (consStatus >= 1 && consStatus <= 3) {
    const mask = Math.round(consArr[1]) >>> 0;
    const tidA = Math.round(consArr[14]) | 0;
    const tidB = Math.round(consArr[15]) | 0;
    const hyps = [];
    for (let b = 0; b < OCCL_HYP_NAMES.length; b++) {
      if (mask & (1 << b)) hyps.push(OCCL_HYP_NAMES[b]);
    }
    const stLbl = OCCL_STATUS_LABELS[consStatus] || 'NONE';
    const badgeY = (fieldOverlayMode > 0) ? 65 : 44;
    const borderC = consStatus === 1 ? '#38f27a' : (consStatus === 2 ? '#ffb703' : '#ff4d6d');
    const textC = consStatus === 1 ? '#a7f3d0' : (consStatus === 2 ? '#fde68a' : '#fecdd3');
    drawHudCaptionBox(
      10,
      badgeY,
      `OCCL T${tidA}-T${tidB} · ${stLbl} · ${hyps.length > 0 ? hyps.join('/') : 'NONE'}`,
      borderC,
      consStatus === 3 ? 'rgba(38, 12, 16, 0.88)' : 'rgba(7, 13, 24, 0.84)',
      textC
    );
  }
}

function renderWave1Overlays(W, H, dispSlot, effK, sx, sy) {
  const isBosMode = fieldOverlayMode >= 5 && fieldOverlayMode <= 7;
  if (isBosMode && bosOverlayCtx && bosOverlayImgData) {
    let dispBosRgba = null;
    let dispBosState = 0;
    let dispBosReason = 0;
    let dispBosRelRes = 0.0;
    let dispBosIters = 0;
    let dispBosValidDisp = 0;
    if (dispSlot >= 0 && dispSlot < RING_LEN && bosRingValid[dispSlot] === 1) {
      dispBosRgba = bosRingRgba[dispSlot];
      const bt = bosRingTelem[dispSlot];
      dispBosState = bt[2] | 0;
      dispBosReason = bt[3] | 0;
      dispBosValidDisp = bt[4] | 0;
      dispBosIters = bt[8] | 0;
      dispBosRelRes = bt[12];
    } else if (effK === 0 && wave1BosAbiValid && bosRgbaView && bosTelemView) {
      dispBosRgba = bosRgbaView;
      dispBosState = bosTelemView[2] | 0;
      dispBosReason = bosTelemView[3] | 0;
      dispBosValidDisp = bosTelemView[4] | 0;
      dispBosIters = bosTelemView[8] | 0;
      dispBosRelRes = bosTelemF32View[16];
    }

    if (dispBosRgba && dispBosState >= 2) {
      bosOverlayImgData.data.set(dispBosRgba);
      bosOverlayCtx.putImageData(bosOverlayImgData, 0, 0);
      hudCtx.save();
      hudCtx.imageSmoothingEnabled = fieldOverlayMode !== 7;
      hudCtx.imageSmoothingQuality = 'low';
      hudCtx.drawImage(bosOverlayCanvas, 0, 0, W, H);
      hudCtx.restore();
    }

    let bosCap =
      fieldOverlayMode === 5
        ? 'BOS REFRACTION |eps| (256x128)'
        : fieldOverlayMode === 6
        ? 'BOS NEUMANN POTENTIAL Phi (256x128)'
        : 'BOS CURL CHECK (dv/dx - du/dy)';
    const stLabel = BOS_STATE_LABELS[Math.max(0, Math.min(3, dispBosState))] || 'OFF';
    bosCap += ` · ${stLabel}`;
    if (dispBosState === 2) {
      bosCap += ` · ${dispBosValidDisp}/${BOS_N} · PCG ${dispBosIters} (${Number.isFinite(dispBosRelRes) ? dispBosRelRes.toExponential(1) : '--'})`;
    } else if (dispBosState === 3 && dispBosReason !== 0) {
      const reasons = [];
      for (let b = 0; b < BOS_REASON_NAMES.length; b++) {
        if (dispBosReason & (1 << b)) reasons.push(BOS_REASON_NAMES[b]);
      }
      bosCap += ` [${reasons.slice(0, 2).join('+')}]`;
    }
    hudCtx.save();
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    const cw = hudCtx.measureText(bosCap).width + 12;
    hudCtx.fillStyle = dispBosState === 3 ? 'rgba(38, 12, 16, 0.88)' : 'rgba(7, 13, 24, 0.84)';
    hudCtx.fillRect(10, 44, cw, 17);
    hudCtx.strokeStyle = dispBosState === 3 ? '#ff4d6d' : 'rgba(56, 242, 122, 0.55)';
    hudCtx.lineWidth = 1;
    hudCtx.strokeRect(10, 44, cw, 17);
    hudCtx.fillStyle = dispBosState === 3 ? '#fecdd3' : '#a7f3d0';
    hudCtx.fillText(bosCap, 16, 56);
    hudCtx.restore();
  }

  renderWave2Overlays(W, H, dispSlot, effK);

  if (showCovEllipses) {
    let dispMicro = null;
    let dispCov = null;
    let dispQual = null;
    if (dispSlot >= 0 && dispSlot < RING_LEN && swarmRingValid[dispSlot] === 1 && covRingValid[dispSlot] === 1) {
      dispMicro = swarmRingMicro[dispSlot];
      dispCov = covRingBuf[dispSlot];
      dispQual = qualRingBuf[dispSlot];
    } else if (effK === 0 && wave1MetrologyAbiValid && microTracksView && microCovView && microQualityView) {
      dispMicro = microTracksView;
      dispCov = microCovView;
      dispQual = microQualityView;
    }

    if (dispMicro && dispCov && dispQual) {
      hudCtx.save();
      const k95 = 2.447747; // sqrt(chi^2_2(0.95) = 5.99146)
      for (let i = 0; i < MAX_MICRO; i++) {
        const qual = dispQual[i];
        if ((qual & (1 << 3)) === 0) continue; // cov_usable bit 3
        const mb = i * MICRO_STRIDE;
        const st = intOrNull(dispMicro[mb + 5], 1, 2);
        if (st === null) continue;
        const px = dispMicro[mb + 1];
        const py = dispMicro[mb + 2];
        const cxx = dispCov[i];
        const cxy = dispCov[MAX_MICRO + i];
        const cyy = dispCov[2 * MAX_MICRO + i];
        if (
          !Number.isFinite(px) ||
          !Number.isFinite(py) ||
          !Number.isFinite(cxx) ||
          !Number.isFinite(cxy) ||
          !Number.isFinite(cyy)
        ) {
          continue;
        }
        const tr = cxx + cyy;
        const diff = cxx - cyy;
        const disc = Math.hypot(diff, 2.0 * cxy);
        const l1 = Math.max(1e-4, 0.5 * (tr + disc));
        const l2 = Math.max(1e-4, 0.5 * (tr - disc));
        const ang = 0.5 * Math.atan2(2.0 * cxy, diff);
        const rx = Math.min(36.0 * sx, Math.max(1.5, Math.sqrt(l1) * k95 * sx));
        const ry = Math.min(36.0 * sy, Math.max(1.5, Math.sqrt(l2) * k95 * sy));
        const isHeroCand = (qual & ((1 << 18) | (1 << 19))) !== 0;
        hudCtx.strokeStyle = isHeroCand ? 'rgba(251, 191, 36, 0.92)' : 'rgba(56, 189, 248, 0.65)';
        hudCtx.lineWidth = isHeroCand ? 1.5 : 0.9;
        hudCtx.beginPath();
        hudCtx.ellipse(px * sx, py * sy, rx, ry, ang, 0, Math.PI * 2);
        hudCtx.stroke();
      }
      hudCtx.restore();
    }
  }
}

function renderMainHud(dispSlot) {
  if (!hudCtx) return;
  const W = hudCanvas.width;   // 960
  const H = hudCanvas.height;  // 540
  const sx = W / frameW;       // 3.0 in SD, 1.5 in HD
  const sy = H / frameH;       // 3.0 in SD, 1.5 in HD
  const fIdx = currentCausalEngineFrameU32();
  const effK = Math.round(telemView[1]);
  const dispFrame = (fIdx - effK) >>> 0;
  const metricOn = isMetricCalibrated();
  const spdUnit = metricOn ? 'm/s' : 'px/s';

  // Activate any Mode C localized shockwaves whose scheduled presentation frame has arrived at t - K
  for (let q = 0; q < SHOCK_POOL; q++) {
    if (shockTriggerArmed[q] === 1 && ((fIdx - shockTriggerFrame[q]) >>> 0) < 0x80000000) {
      shockAge[q] = 0;
      padFlashAlpha[shockZone[q] & 3] = 1.0;
      shockTriggerArmed[q] = 0;
    }
  }

  // 1. Blit delayed display frame `t - K` from static Wasm ring buffer
  dispStageCtx.putImageData(ringSlotImgData[dispSlot], 0, 0);
  hudCtx.imageSmoothingEnabled = false;
  hudCtx.drawImage(dispStageCanvas, 0, 0, W, H);

  // 2. Optional Live ECA Motion-Residual Overlay (explicitly labeled as frame `t`)
  if (showEcaOverlay && ecaResView) {
    for (let i = 0; i < framePixels; i++) {
      const res = ecaResView[i];
      const off = i * 4;
      if (res > 0.025) {
        const v = Math.min(1.0, res * 3.2);
        ecaOverlayBuf[off] = (v * 30) | 0;
        ecaOverlayBuf[off + 1] = (v * 235) | 0;
        ecaOverlayBuf[off + 2] = (v * 255) | 0;
        ecaOverlayBuf[off + 3] = (v * 185) | 0;
      } else {
        ecaOverlayBuf[off + 3] = 0;
      }
    }
    ecaStageCtx.putImageData(ecaOverlayImgData, 0, 0);
    hudCtx.drawImage(ecaStageCanvas, 0, 0, W, H);
  }

  // 3. In Testbench mode, outline active opaque occlusion zones cleanly on HUD
  if (sourceMode === 'testbench') {
    const pillars = getActivePillars();
    if (pillars.length > 0) {
      hudCtx.save();
      hudCtx.strokeStyle = 'rgba(116, 152, 210, 0.50)';
      hudCtx.setLineDash([6, 4]);
      hudCtx.lineWidth = 1.5;
      for (let p = 0; p < pillars.length; p++) {
        const pil = pillars[p];
        const px = pil.x0 * resScale * sx;
        const py = pil.y0 * resScale * sy;
        const pw = (pil.x1 - pil.x0) * resScale * sx;
        const ph = (pil.y1 - pil.y0) * resScale * sy;
        hudCtx.strokeRect(px, py, pw, ph);
        hudCtx.fillStyle = 'rgba(140, 175, 235, 0.78)';
        hudCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText(`OCCLUDER P${p + 1}`, px + 6, py + 15);
      }
      hudCtx.restore();
    }
  }

  // 3a. Continuum Heatmap, 1,000+ Micro-Tracks, Rigid-Motion Hulls, BOS & 2x2 Covariance Overlays
  renderChronos1000FieldLayer(W, H, dispSlot, effK, sx, sy);
  renderWave1Overlays(W, H, dispSlot, effK, sx, sy);

  // 3b. Metric Ruler or 4-Point Perspective Homography Floor Grid Overlay
  renderCalibrationOverlay(W, H);
  renderSeedBoxOverlay(W, H);

  let placedLabels = 0;

  // 4. Mode C: Quadrant Voice Grid + Localized Impact Shockwaves + CAD Deflection & Scatter Protractors
  if (labMode === 2) {
    const labels = ['Q0: SUB-KICK 808', 'Q1: CRISP SNARE', 'Q2: METALLIC FM HAT', 'Q3: HARMONIC TOM'];
    hudCtx.save();
    hudCtx.strokeStyle = 'rgba(0, 240, 255, 0.16)';
    hudCtx.lineWidth = 1;
    hudCtx.beginPath();
    hudCtx.moveTo(W * 0.5, 0); hudCtx.lineTo(W * 0.5, H);
    hudCtx.moveTo(0, H * 0.5); hudCtx.lineTo(W, H * 0.5);
    hudCtx.stroke();

    for (let z = 0; z < 4; z++) {
      const qx = (z & 1) ? W * 0.5 : 0;
      const qy = (z & 2) ? H * 0.5 : 0;
      if (padFlashAlpha[z] > 0.01) {
        hudCtx.strokeStyle = ZONE_COLORS[z];
        hudCtx.globalAlpha = padFlashAlpha[z] * 0.65;
        hudCtx.lineWidth = 2.5;
        hudCtx.strokeRect(qx + 3, qy + 3, W * 0.5 - 6, H * 0.5 - 6);
        hudCtx.globalAlpha = 1.0;
        padFlashAlpha[z] *= 0.82;
      }
      const badgeY = z >= 2 ? H * 0.5 - 52 : 40;
      hudCtx.fillStyle = 'rgba(6, 10, 18, 0.72)';
      hudCtx.fillRect(qx + 10, qy + badgeY, 138, 18);
      hudCtx.fillStyle = ZONE_COLORS[z];
      hudCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(labels[z], qx + 15, qy + badgeY + 13);
    }

    // Localized concentric shockwave rings + CAD deflection & restitution protractors!
    for (let q = 0; q < SHOCK_POOL; q++) {
      if (shockAge[q] >= 0) {
        const age = shockAge[q];
        const cx = shockX[q] * sx;
        const cy = shockY[q] * sy;
        const zCol = ZONE_COLORS[shockZone[q] & 3];

        hudCtx.save();
        if (age <= 15) {
          const prog = age / 15.0;
          const alpha = Math.max(0.0, 1.0 - prog);
          const rOuter = 10.0 + prog * 54.0 * (0.65 + 0.5 * shockVel[q]);
          const rInner = 4.0 + prog * 28.0;

          hudCtx.globalAlpha = alpha;
          hudCtx.strokeStyle = zCol;
          hudCtx.lineWidth = 2.6 * (1.0 - 0.5 * prog);
          hudCtx.beginPath();
          hudCtx.arc(cx, cy, rOuter, 0, Math.PI * 2);
          hudCtx.stroke();

          hudCtx.strokeStyle = '#ffffff';
          hudCtx.lineWidth = 1.4;
          hudCtx.beginPath();
          hudCtx.arc(cx, cy, rInner, 0, Math.PI * 2);
          hudCtx.moveTo(cx - 8, cy); hudCtx.lineTo(cx + 8, cy);
          hudCtx.moveTo(cx, cy - 8); hudCtx.lineTo(cx, cy + 8);
          hudCtx.stroke();
        }

        // CAD Deflection Protractor & Restitution readout (held 26 frames for clear reading)
        if (shockRestitution[q] > 0.04 && Math.abs(shockDeflectDeg[q]) >= 2.5) {
          const fade = age <= 18 ? 0.96 : Math.max(0.0, 1.0 - (age - 18) / 9.0);
          hudCtx.globalAlpha = fade;

          // Canvas rays follow actual 2D screen geometry; numeric angle & restitution rectify via J_H!
          const rawInCanvas = -shockInDeg[q] * (Math.PI / 180.0);
          const rawOutCanvas = -(shockInDeg[q] + shockDeflectDeg[q]) * (Math.PI / 180.0);
          const rectSt = rectifyStrikeKinematics(
            shockX[q],
            shockY[q],
            shockInDeg[q],
            shockDeflectDeg[q],
            shockRestitution[q]
          );
          const dispDefl = rectSt[1];
          const dispRest = rectSt[2];

          // Incident continuation dashed reference ray
          hudCtx.strokeStyle = 'rgba(226, 238, 255, 0.68)';
          hudCtx.lineWidth = 1.5;
          hudCtx.setLineDash([4, 3]);
          hudCtx.beginPath();
          hudCtx.moveTo(cx, cy);
          hudCtx.lineTo(cx + Math.cos(rawInCanvas) * 34.0, cy + Math.sin(rawInCanvas) * 34.0);
          hudCtx.stroke();
          hudCtx.setLineDash([]);

          // Outgoing deflected velocity vector
          hudCtx.strokeStyle = zCol;
          hudCtx.lineWidth = 2.2;
          drawHudArrow(
            hudCtx,
            cx,
            cy,
            cx + Math.cos(rawOutCanvas) * 42.0,
            cy + Math.sin(rawOutCanvas) * 42.0,
            7.0
          );

          // Signed CAD deflection arc
          const dArc = Math.atan2(Math.sin(rawOutCanvas - rawInCanvas), Math.cos(rawOutCanvas - rawInCanvas));
          hudCtx.fillStyle = 'rgba(255, 209, 102, 0.20)';
          hudCtx.beginPath();
          hudCtx.moveTo(cx, cy);
          hudCtx.arc(cx, cy, 22.0, rawInCanvas, rawInCanvas + dArc, dArc < 0);
          hudCtx.closePath();
          hudCtx.fill();
          hudCtx.strokeStyle = '#ffd166';
          hudCtx.lineWidth = 1.6;
          hudCtx.beginPath();
          hudCtx.arc(cx, cy, 22.0, rawInCanvas, rawInCanvas + dArc, dArc < 0);
          hudCtx.stroke();

          const signStr = dispDefl >= 0 ? '+' : '';
          const cadTxt = `⚡ Δθ=${signStr}${dispDefl.toFixed(1)}° | v⁺/v⁻=${dispRest.toFixed(2)}`;
          const bw = 176;
          const pos = reserveHudLabelSlot(W, H, cx + 14, cy - 26, bw, placedLabels);
          placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
          hudCtx.fillStyle = 'rgba(6, 10, 18, 0.88)';
          hudCtx.fillRect(pos[0], pos[1], bw, 18);
          hudCtx.strokeStyle = zCol;
          hudCtx.lineWidth = 1.0;
          hudCtx.strokeRect(pos[0], pos[1], bw, 18);
          hudCtx.fillStyle = '#ffd166';
          hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
          hudCtx.fillText(cadTxt, pos[0] + 5, pos[1] + 12);
        } else if (age <= 10) {
          const bw = 110;
          const pos = reserveHudLabelSlot(W, H, cx + 12, cy - 22, bw, placedLabels);
          placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
          hudCtx.fillStyle = 'rgba(6, 10, 18, 0.84)';
          hudCtx.fillRect(pos[0], pos[1], bw, 16);
          hudCtx.fillStyle = zCol;
          hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
          hudCtx.fillText(`⚡ L1 IMPACT Q${shockZone[q]}`, pos[0] + 4, pos[1] + 11);
        }
        hudCtx.restore();

        shockAge[q]++;
        if (shockAge[q] > 26) shockAge[q] = -1;
      }
    }

    // Two-Body Elastic Scatter Wedge (90-deg carom billiard / disc pair collision proof!)
    const rawScDeg = telemView[TELEM_GLOBAL + 2];
    const scEventFrame = Math.round(telemView[TELEM_GLOBAL + 7]);
    if (rawScDeg > 2.0 && scEventFrame > 0 && Math.abs(dispFrame - scEventFrame) <= 24) {
      const scx = telemView[TELEM_GLOBAL + 3] * sx;
      const scy = telemView[TELEM_GLOBAL + 4] * sy;
      const th1Canvas = -telemView[TELEM_GLOBAL + 5] * (Math.PI / 180.0);
      const th2Canvas = -telemView[TELEM_GLOBAL + 6] * (Math.PI / 180.0);
      const rectSc = rectifyScatterKinematics(
        telemView[TELEM_GLOBAL + 3],
        telemView[TELEM_GLOBAL + 4],
        telemView[TELEM_GLOBAL + 5],
        telemView[TELEM_GLOBAL + 6],
        rawScDeg
      );
      const dispScDeg = rectSc[2];
      const dSc = Math.atan2(Math.sin(th2Canvas - th1Canvas), Math.cos(th2Canvas - th1Canvas));

      hudCtx.save();
      hudCtx.fillStyle = 'rgba(0, 240, 255, 0.22)';
      hudCtx.beginPath();
      hudCtx.moveTo(scx, scy);
      hudCtx.arc(scx, scy, 34.0, th1Canvas, th1Canvas + dSc, dSc < 0);
      hudCtx.closePath();
      hudCtx.fill();

      hudCtx.strokeStyle = '#00f0ff';
      hudCtx.lineWidth = 2.0;
      drawHudArrow(hudCtx, scx, scy, scx + Math.cos(th1Canvas) * 44.0, scy + Math.sin(th1Canvas) * 44.0, 6.5);
      drawHudArrow(hudCtx, scx, scy, scx + Math.cos(th2Canvas) * 44.0, scy + Math.sin(th2Canvas) * 44.0, 6.5);

      const scTxt = `⟁ CONTACT SCATTER ${dispScDeg.toFixed(1)}°`;
      const bw = 166;
      const pos = reserveHudLabelSlot(W, H, scx - 62, scy + 18, bw, placedLabels);
      placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
      hudCtx.fillStyle = 'rgba(6, 10, 18, 0.90)';
      hudCtx.fillRect(pos[0], pos[1], bw, 18);
      hudCtx.strokeStyle = '#00f0ff';
      hudCtx.lineWidth = 1.2;
      hudCtx.strokeRect(pos[0], pos[1], bw, 18);
      hudCtx.fillStyle = '#8ef9f3';
      hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(scTxt, pos[0] + 6, pos[1] + 12);
      hudCtx.restore();
    }

    hudCtx.restore();
  }

  const uDisp = (RING_LEN - 1) - effK;
  const effFps = Math.max(8.0, sourceMode === 'testbench' ? 60.0 : sourceFps);
  const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
  const primarySlot = resolvePrimarySlot();
  if (Math.round(telemView[TELEM_GLOBAL + 9]) === 2) {
    trailOwnerTid.fill(-1);
    trailCount.fill(0);
    trailLastFrame.fill(0);
  }

  updateFrozenApexLocks(uDisp);

  // 5. Render Target Trajectories, Hermite Spline Bridges, Frozen Apex Reticles, Protractors & Kalman Baselines
  if (showHeroTargets) {
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] < 0.5) continue;

    const kbase = TELEM_KINETIC + m * 8;
    const tid = Math.round(telemView[base + 1]);
    const isPrimary = m === primarySlot;
    const isPinned = focusTid !== 0 && tid === focusTid;
    const color = TARGET_COLORS[Math.max(0, tid - 1) % TARGET_COLORS.length];
    const fxPx = telemView[base + 2];
    const fyPx = telemView[base + 3];
    const fx = fxPx * sx;
    const fy = fyPx * sy;
    const rawDispX = telemView[base + 4] * sx;
    const rawDispY = telemView[base + 5] * sy;
    const kx = telemView[base + 6] * sx;
    const ky = telemView[base + 7] * sy;
    const dispVis = telemView[base + 10] > 0.5;
    const isBridged = telemView[base + 11] > 0.5;
    const sigHex = toHex8(telemView[base + 12], telemView[base + 13]);
    const padicVal = Math.round(telemView[base + 14]);
    const apexFrames = telemView[base + 17];
    const fiedlerCoord = telemView[base + 19];
    const headX = telemView[base + 20 + 11] * sx;
    const headY = telemView[base + 32 + 11] * sy;

    const rawHeadingDeg = telemView[kbase + 0];
    const rawSpeedPxS = telemView[kbase + 1];
    const pivXPx = telemView[kbase + 6];
    const pivYPx = telemView[kbase + 7];
    const hasPivot = pivXPx > -9.0e5 && pivYPx > -9.0e5;

    // Suppress static background scale wake clutter in Mode B real-camera clips unless Show All is active
    if (
      labMode === 1 &&
      sourceMode !== 'testbench' &&
      !showAllOverlays &&
      !isPrimary &&
      !isPinned &&
      !hasPivot &&
      rawSpeedPxS < 20.0 * resScale
    ) {
      continue;
    }

    // Update & render 44-frame (~1.5s) persistent comet trail for smooth visual trajectory history
    const ts = tid & (TRAIL_SLOTS - 1);
    const tBase = ts * TRAIL_LEN;
    if (dispVis || isBridged) {
      if (
        trailOwnerTid[ts] !== tid ||
        trailCount[ts] === 0 ||
        ((dispFrame - trailLastFrame[ts]) >>> 0) > 14
      ) {
        trailOwnerTid[ts] = tid;
        trailHead[ts] = 1;
        trailCount[ts] = 1;
        trailLastFrame[ts] = dispFrame;
        trailX[tBase] = fxPx;
        trailY[tBase] = fyPx;
      } else if (dispFrame !== trailLastFrame[ts]) {
        const prevIdx = (trailHead[ts] + TRAIL_LEN - 1) % TRAIL_LEN;
        const jumpPx = Math.hypot(fxPx - trailX[tBase + prevIdx], fyPx - trailY[tBase + prevIdx]);
        if (jumpPx > 52.0 * resScale) {
          trailHead[ts] = 1;
          trailCount[ts] = 1;
          trailX[tBase] = fxPx;
          trailY[tBase] = fyPx;
        } else {
          const wr = trailHead[ts];
          trailX[tBase + wr] = fxPx;
          trailY[tBase + wr] = fyPx;
          trailHead[ts] = (wr + 1) % TRAIL_LEN;
          if (trailCount[ts] < TRAIL_LEN) trailCount[ts] += 1;
        }
        trailLastFrame[ts] = dispFrame;
      }
    }

    hudCtx.save();

    const nTrail = trailOwnerTid[ts] === tid ? trailCount[ts] : 0;
    if (nTrail >= 3) {
      const oldest = (trailHead[ts] + TRAIL_LEN - nTrail) % TRAIL_LEN;
      const split1 = Math.max(1, Math.floor(nTrail * 0.42));
      const split2 = Math.max(split1 + 1, Math.floor(nTrail * 0.74));
      hudCtx.strokeStyle = color;
      hudCtx.lineCap = 'round';
      hudCtx.lineJoin = 'round';

      for (let band = 0; band < 3; band++) {
        const i0 = band === 0 ? 0 : (band === 1 ? split1 - 1 : split2 - 1);
        const i1 = band === 0 ? split1 : (band === 1 ? split2 : nTrail - 1);
        if (i1 <= i0) continue;
        hudCtx.globalAlpha = (isPrimary ? 1.0 : 0.78) * (band === 0 ? 0.18 : (band === 1 ? 0.36 : 0.60));
        hudCtx.lineWidth = (isPrimary ? 1.15 : 0.95) * (band === 0 ? 1.2 : (band === 1 ? 1.8 : 2.4));
        hudCtx.beginPath();
        for (let k = i0; k <= i1; k++) {
          const idx = tBase + ((oldest + k) % TRAIL_LEN);
          const tx = trailX[idx] * sx;
          const ty = trailY[idx] * sy;
          if (k === i0) hudCtx.moveTo(tx, ty);
          else hudCtx.lineTo(tx, ty);
        }
        hudCtx.stroke();
      }
      hudCtx.globalAlpha = 1.0;
    }

    // 5a. Virtual Pendulum / Metronome Pivot, Plumb Reference, Tether & Swing Wedge
    if (hasPivot && (isPrimary || isPinned || labMode === 1)) {
      const pivX = pivXPx * sx;
      const pivY = pivYPx * sy;
      const pendKin = computePendulumKinematics(fxPx, fyPx, pivXPx, pivYPx);
      const swingDeg = pendKin[0];
      const radPhys = pendKin[1];
      const plumbAng = fy >= pivY ? Math.PI * 0.5 : -Math.PI * 0.5;
      const tethAng = Math.atan2(fy - pivY, fx - pivX);
      const tethHudLen = Math.hypot(fx - pivX, fy - pivY);
      const wedgeR = Math.min(54.0, Math.max(22.0, tethHudLen * 0.36));

      // Plumb vertical reference dashed ray
      hudCtx.strokeStyle = 'rgba(168, 192, 223, 0.45)';
      hudCtx.lineWidth = 1.3;
      hudCtx.setLineDash([4, 4]);
      hudCtx.beginPath();
      hudCtx.moveTo(pivX, Math.max(38, Math.min(H - 38, pivY)));
      hudCtx.lineTo(pivX, fy >= pivY ? Math.min(H - 36, pivY + tethHudLen * 0.95) : Math.max(38, pivY - tethHudLen * 0.95));
      hudCtx.stroke();

      // Rigid tether ray from virtual pivot to FISTA bob/weight
      hudCtx.strokeStyle = '#38f27a';
      hudCtx.lineWidth = 1.8;
      hudCtx.setLineDash([6, 4]);
      hudCtx.beginPath();
      hudCtx.moveTo(pivX, pivY);
      hudCtx.lineTo(fx, fy);
      hudCtx.stroke();
      hudCtx.setLineDash([]);

      // Swing angle sector wedge (when pivot is near/inside viewport)
      if (pivY >= -30 && pivY <= H + 30 && Math.abs(swingDeg) >= 0.8) {
        const dSw = Math.atan2(Math.sin(tethAng - plumbAng), Math.cos(tethAng - plumbAng));
        hudCtx.fillStyle = 'rgba(56, 242, 122, 0.18)';
        hudCtx.beginPath();
        hudCtx.moveTo(pivX, pivY);
        hudCtx.arc(pivX, pivY, wedgeR, plumbAng, plumbAng + dSw, dSw < 0);
        hudCtx.closePath();
        hudCtx.fill();
        hudCtx.strokeStyle = '#38f27a';
        hudCtx.lineWidth = 1.6;
        hudCtx.beginPath();
        hudCtx.arc(pivX, pivY, wedgeR, plumbAng, plumbAng + dSw, dSw < 0);
        hudCtx.stroke();
      }

      // Pivot bullseye ring (or top-edge directional marker if ceiling mount is above camera frame)
      if (pivY >= 40 && pivY <= H - 36 && pivX >= 8 && pivX <= W - 8) {
        hudCtx.strokeStyle = '#38f27a';
        hudCtx.lineWidth = 2.0;
        hudCtx.beginPath();
        hudCtx.arc(pivX, pivY, 6.5, 0, Math.PI * 2);
        hudCtx.moveTo(pivX - 11, pivY); hudCtx.lineTo(pivX + 11, pivY);
        hudCtx.moveTo(pivX, pivY - 11); hudCtx.lineTo(pivX, pivY + 11);
        hudCtx.stroke();
        hudCtx.fillStyle = 'rgba(6, 10, 18, 0.84)';
        hudCtx.fillRect(pivX + 9, pivY - 8, 56, 15);
        hudCtx.fillStyle = '#38f27a';
        hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText('◎ PIVOT', pivX + 12, pivY + 3);
      } else if (pivY < 40) {
        const tClip = Math.abs(fy - pivY) > 1e-3 ? (44.0 - pivY) / (fy - pivY) : 0.0;
        const edgeX = Math.max(24, Math.min(W - 96, pivX + tClip * (fx - pivX)));
        hudCtx.fillStyle = 'rgba(6, 10, 18, 0.86)';
        hudCtx.fillRect(edgeX - 8, 38, 86, 15);
        hudCtx.fillStyle = '#38f27a';
        hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText('▲ VIRT PIVOT', edgeX - 4, 49);
      }

      const swSign = swingDeg >= 0 ? '+' : '';
      const rStr = metricOn ? `${radPhys.toFixed(2)}m` : `${Math.round(radPhys)}px`;
      const swTxt = `θ=${swSign}${swingDeg.toFixed(1)}° | R=${rStr}`;
      const bw = 138;
      const pos = reserveHudLabelSlot(W, H, fx + 30, fy + 16, bw, placedLabels);
      placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
      hudCtx.fillStyle = 'rgba(6, 10, 18, 0.88)';
      hudCtx.fillRect(pos[0], pos[1], bw, 17);
      hudCtx.strokeStyle = 'rgba(56, 242, 122, 0.65)';
      hudCtx.lineWidth = 1.0;
      hudCtx.strokeRect(pos[0], pos[1], bw, 17);
      hudCtx.fillStyle = '#38f27a';
      hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(swTxt, pos[0] + 5, pos[1] + 12);
    }

    // 5b. Past Dual-FISTA Ribbon [0 .. uDisp]
    hudCtx.lineWidth = isPrimary ? 2.8 : 2.2;
    hudCtx.strokeStyle = color;
    hudCtx.beginPath();
    for (let u = 0; u <= uDisp; u++) {
      const px = telemView[base + 20 + u] * sx;
      const py = telemView[base + 32 + u] * sy;
      if (u === 0) hudCtx.moveTo(px, py);
      else hudCtx.lineTo(px, py);
    }
    hudCtx.stroke();

    // 5c. Future Non-Causal Lookahead / Hermite Bridge Ribbon [uDisp .. 11]
    if (uDisp < RING_LEN - 1) {
      hudCtx.lineWidth = isBridged ? 3.0 : 2.0;
      hudCtx.strokeStyle = isBridged ? '#ffd166' : color;
      hudCtx.setLineDash([5, 4]);
      hudCtx.beginPath();
      for (let u = uDisp; u < RING_LEN; u++) {
        const px = telemView[base + 20 + u] * sx;
        const py = telemView[base + 32 + u] * sy;
        if (u === uDisp) hudCtx.moveTo(px, py);
        else hudCtx.lineTo(px, py);
      }
      hudCtx.stroke();
      hudCtx.setLineDash([]);

      hudCtx.fillStyle = isBridged ? '#ffd166' : color;
      hudCtx.beginPath();
      hudCtx.arc(headX, headY, isBridged ? 4.2 : 3.2, 0, Math.PI * 2);
      hudCtx.fill();

      // Explicit Non-Causal Bridge Proof: show unrevised causal coasting ghost vs t NOW re-emergence anchor!
      if (isBridged) {
        const ghostDist = Math.hypot(rawDispX - fx, rawDispY - fy);
        if (ghostDist > 3.5) {
          hudCtx.strokeStyle = 'rgba(255, 107, 107, 0.72)';
          hudCtx.lineWidth = 1.3;
          hudCtx.setLineDash([2, 3]);
          hudCtx.strokeRect(rawDispX - 10, rawDispY - 10, 20, 20);
          hudCtx.setLineDash([]);
        }
        hudCtx.fillStyle = '#ffd166';
        hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText('t NOW', headX + 6, headY - 5);
      }
    }

    // 5d. Tangent Velocity Vector & Flight-Angle Protractor (θ(t-K) & speed in m/s or px/s)
    if (rawSpeedPxS >= 14.0 * resScale) {
      const velCanvasAng = -rawHeadingDeg * (Math.PI / 180.0);
      const arrowLen = Math.min(54.0, Math.max(26.0, (rawSpeedPxS / resScale) * 0.22));
      const tipX = fx + Math.cos(velCanvasAng) * arrowLen;
      const tipY = fy + Math.sin(velCanvasAng) * arrowLen;
      const showFullProtractor = (isPrimary || isPinned || (labMode === 0 && apexLockHold[m] > 0)) && !hasPivot;

      if (showFullProtractor) {
        // Horizontal 0-deg datum reference tick
        const refDir = Math.cos(velCanvasAng) >= 0 ? 1.0 : -1.0;
        const refAng = refDir > 0 ? 0.0 : Math.PI;
        hudCtx.strokeStyle = 'rgba(168, 192, 223, 0.55)';
        hudCtx.lineWidth = 1.2;
        hudCtx.setLineDash([3, 3]);
        hudCtx.beginPath();
        hudCtx.moveTo(fx, fy);
        hudCtx.lineTo(fx + refDir * 32.0, fy);
        hudCtx.stroke();
        hudCtx.setLineDash([]);

        const dVelArc = Math.atan2(Math.sin(velCanvasAng - refAng), Math.cos(velCanvasAng - refAng));
        if (Math.abs(dVelArc) > 0.04) {
          hudCtx.fillStyle = 'rgba(0, 229, 255, 0.16)';
          hudCtx.beginPath();
          hudCtx.moveTo(fx, fy);
          hudCtx.arc(fx, fy, 20.0, refAng, refAng + dVelArc, dVelArc < 0);
          hudCtx.closePath();
          hudCtx.fill();
          hudCtx.strokeStyle = '#00e5ff';
          hudCtx.lineWidth = 1.4;
          hudCtx.beginPath();
          hudCtx.arc(fx, fy, 20.0, refAng, refAng + dVelArc, dVelArc < 0);
          hudCtx.stroke();
        }

        hudCtx.strokeStyle = '#00e5ff';
        hudCtx.lineWidth = 2.1;
        drawHudArrow(hudCtx, fx, fy, tipX, tipY, 6.5);

        const physVel = mapVelocityToPhysical(fxPx, fyPx, rawSpeedPxS, rawHeadingDeg);
        const dispSpd = physVel[0];
        const dispHeading = physVel[1];
        const vValid = physVel[2];
        const thSign = dispHeading >= 0 ? '+' : '';
        const kinTxt = vValid < -0.5
          ? `∠${thSign}${dispHeading.toFixed(1)}° · OUT-OF-PLANE`
          : `∠${thSign}${dispHeading.toFixed(1)}° · ${vValid > 0.5 ? dispSpd.toFixed(2) + 'm/s' : Math.round(dispSpd) + 'px/s'}`;
        const bw = vValid < -0.5 ? 168 : 148;
        const pos = reserveHudLabelSlot(W, H, fx + 30, fy + 12, bw, placedLabels);
        placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
        hudCtx.fillStyle = 'rgba(6, 10, 18, 0.88)';
        hudCtx.fillRect(pos[0], pos[1], bw, 17);
        hudCtx.strokeStyle = vValid < -0.5 ? 'rgba(255, 183, 3, 0.72)' : 'rgba(0, 229, 255, 0.58)';
        hudCtx.lineWidth = 1.0;
        hudCtx.strokeRect(pos[0], pos[1], bw, 17);
        hudCtx.fillStyle = vValid < -0.5 ? '#ffd166' : '#8ef9f3';
        hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText(kinTxt, pos[0] + 5, pos[1] + 12);
      } else if (labMode !== 1) {
        hudCtx.strokeStyle = color;
        hudCtx.globalAlpha = 0.78;
        hudCtx.lineWidth = 1.6;
        drawHudArrow(hudCtx, fx, fy, tipX, tipY, 5.5);
        hudCtx.globalAlpha = 1.0;
      }
    }

    // 5e. Kinematic-Only 2x2 Kalman Baseline Marker (shown always on testbench; on real feeds when diverging)
    const kalDist = Math.hypot(kx - fx, ky - fy);
    const minKalShow = sourceMode === 'testbench' ? 0.0 : 11.0;
    if (showKalmanBaseline && kalDist >= minKalShow && kalDist <= 165.0) {
      hudCtx.strokeStyle = 'rgba(255, 89, 94, 0.85)';
      hudCtx.lineWidth = 1.4;
      hudCtx.setLineDash([3, 3]);
      hudCtx.strokeRect(kx - 13, ky - 13, 26, 26);
      if (kalDist > 6.0) {
        hudCtx.beginPath();
        hudCtx.moveTo(fx, fy);
        hudCtx.lineTo(kx, ky);
        hudCtx.stroke();
      }
      hudCtx.setLineDash([]);
      if (kalDist > 16.0) {
        hudCtx.fillStyle = '#ff7b82';
        hudCtx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText(`KAL-${tid}`, kx - 14, ky + 23);
      }
    }

    // 5f. Mode A: Frozen Parabolic Apex Reticle & Live Peak Accuracy Score (APEX Δ = X.X px)
    if (labMode === 0 && apexLockTid[m] === tid && apexLockHold[m] > 0) {
      const predX = apexLockPredX[m] * sx;
      const predY = apexLockPredY[m] * sy;
      const scored = apexLockErrPx[m] >= 0.0;
      const arcCol = scored ? '#38f27a' : '#ffd166';

      if (!scored || fy > predY + 6.0) {
        hudCtx.strokeStyle = arcCol;
        hudCtx.lineWidth = 1.8;
        hudCtx.setLineDash([4, 4]);
        hudCtx.beginPath();
        hudCtx.moveTo(fx, fy);
        hudCtx.quadraticCurveTo(0.5 * (fx + predX), predY, predX, predY);
        hudCtx.stroke();
        hudCtx.setLineDash([]);
      }

      hudCtx.strokeStyle = arcCol;
      hudCtx.lineWidth = 2.0;
      hudCtx.beginPath();
      hudCtx.arc(predX, predY, 10, 0, Math.PI * 2);
      hudCtx.moveTo(predX - 15, predY); hudCtx.lineTo(predX + 15, predY);
      hudCtx.moveTo(predX, predY - 15); hudCtx.lineTo(predX, predY + 15);
      hudCtx.stroke();

      const badgeText = scored
        ? `APEX Δ=${apexLockErrPx[m].toFixed(1)}px`
        : `APEX ${Math.max(8, apexFrames * (1000.0 / effFps)).toFixed(0)}ms`;
      const bw = 114;
      const pos = reserveHudLabelSlot(W, H, predX + 13, predY - 18, bw, placedLabels);
      placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
      hudCtx.fillStyle = 'rgba(6, 10, 18, 0.86)';
      hudCtx.fillRect(pos[0], pos[1], bw, 17);
      hudCtx.fillStyle = arcCol;
      hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(badgeText, pos[0] + 4, pos[1] + 12);
    }

    // 5g. Primary 2-Adic + Fiedler + FISTA Corner-Bracket Reticle at `t - K`
    const boxR = isPrimary ? 30 : 26;
    const bLen = isPrimary ? 9 : 8;
    hudCtx.strokeStyle = isBridged ? '#ffd166' : color;
    hudCtx.lineWidth = isPrimary ? 2.6 : 2.0;
    hudCtx.beginPath();
    hudCtx.moveTo(fx - boxR, fy - boxR + bLen); hudCtx.lineTo(fx - boxR, fy - boxR); hudCtx.lineTo(fx - boxR + bLen, fy - boxR);
    hudCtx.moveTo(fx + boxR - bLen, fy - boxR); hudCtx.lineTo(fx + boxR, fy - boxR); hudCtx.lineTo(fx + boxR, fy - boxR + bLen);
    hudCtx.moveTo(fx - boxR, fy + boxR - bLen); hudCtx.lineTo(fx - boxR, fy + boxR); hudCtx.lineTo(fx - boxR + bLen, fy + boxR);
    hudCtx.moveTo(fx + boxR - bLen, fy + boxR); hudCtx.lineTo(fx + boxR, fy + boxR); hudCtx.lineTo(fx + boxR, fy + boxR - bLen);
    hudCtx.stroke();

    if (isPinned) {
      hudCtx.strokeStyle = '#ffffff';
      hudCtx.lineWidth = 1.2;
      hudCtx.strokeRect(fx - boxR - 4, fy - boxR - 4, (boxR + 4) * 2, (boxR + 4) * 2);
    }

    const coastNow = slotMetaView ? Math.round(slotMetaView[m * 8 + 1]) : 0;
    const showFullCallout =
      showAllOverlays ||
      isPrimary ||
      isPinned ||
      sourceMode === 'testbench' ||
      (isBridged && coastNow >= 5) ||
      (labMode === 0 && apexLockHold[m] > 0);
    if (showFullCallout) {
      const fSign = fiedlerCoord >= 0 ? 'F+' : 'F-';
      const stateTag = isPinned
        ? 'PINNED'
        : isBridged
        ? 'HERMITE-BRIDGE'
        : dispVis
        ? (isPrimary ? 'FOCUS' : 'LOCK')
        : 'COAST';
      const labelTop = `T${tid} [${stateTag}] v2:${padicVal} ${fSign}`;
      const posTop = reserveHudLabelSlot(W, H, fx - 26, fy > 68 ? fy - 46 : fy + 28, 162, placedLabels);
      placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);

      hudCtx.fillStyle = 'rgba(7, 11, 18, 0.85)';
      hudCtx.fillRect(posTop[0], posTop[1], 162, 19);
      hudCtx.fillStyle = isBridged ? '#ffd166' : color;
      hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(labelTop, posTop[0] + 4, posTop[1] + 13);

      hudCtx.fillStyle = 'rgba(205, 224, 245, 0.82)';
      hudCtx.font = '500 9px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(sigHex, fx + 29, fy - 2);
    } else {
      if (rawSpeedPxS >= 22.0 * resScale) slotSpeedPillOn[m] = 1;
      else if (rawSpeedPxS < 14.0 * resScale) slotSpeedPillOn[m] = 0;

      let compactTxt = isBridged ? `T${tid}·BRG` : `T${tid}`;
      let compactW = isBridged ? 54 : 34;
      if (slotSpeedPillOn[m]) {
        const secVel = mapVelocityToPhysical(fxPx, fyPx, rawSpeedPxS, rawHeadingDeg);
        const smSpd = slotSmoothedSpeed[m] > 0.01 ? slotSmoothedSpeed[m] : secVel[0];
        if (secVel[2] > 0.5) {
          compactTxt = isBridged
            ? `T${tid}·BRG·${smSpd.toFixed(1)}m/s`
            : `T${tid} · ${smSpd.toFixed(1)}m/s`;
          compactW = isBridged ? 102 : 78;
        } else if (secVel[2] > -0.5) {
          compactTxt = isBridged
            ? `T${tid}·BRG·${Math.round(smSpd)}px/s`
            : `T${tid} · ${Math.round(smSpd)}px/s`;
          compactW = isBridged ? 106 : 82;
        }
      }
      const posSec = reserveHudLabelSlot(W, H, fx - boxR, fy - boxR - 17, compactW, placedLabels);
      placedLabels = Math.min(MAX_HUD_LABELS, placedLabels + 1);
      hudCtx.fillStyle = 'rgba(7, 11, 18, 0.84)';
      hudCtx.fillRect(posSec[0], posSec[1], compactW, 14);
      hudCtx.fillStyle = isBridged ? '#ffd166' : color;
      hudCtx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
      hudCtx.fillText(compactTxt, posSec[0] + 4, posSec[1] + 10);
    }
    hudCtx.restore();
  }
  }

  // 6. Present-Time Phase-Locked Strobe Border Pulse (Mode B Biophysical Entrainment)
  if (labMode === 1 && strobeEnabled && strobeFlashAlpha > 0.02) {
    const isPosLead = phaseOffsetRad > 0;
    const rgb = isPosLead ? '255, 183, 3' : '0, 240, 255';
    hudCtx.save();
    hudCtx.strokeStyle = `rgba(${rgb}, ${strobeFlashAlpha.toFixed(3)})`;
    hudCtx.lineWidth = 10;
    hudCtx.strokeRect(5, 5, W - 10, H - 10);
    hudCtx.fillStyle = `rgba(${rgb}, ${(strobeFlashAlpha * 0.9).toFixed(3)})`;
    hudCtx.font = '700 12px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(
      isPosLead ? '⚡ STROBE CUE [+π/2 QUADRATURE LEAD]' : '◆ STROBE CUE [−π/2 QUADRATURE LAG]',
      18,
      H - 18
    );
    hudCtx.restore();
  }
  if (strobeFlashAlpha > 0.001) {
    strobeFlashAlpha *= 0.78;
  }

  // 7. Top-left HUD Status Strip + Top-right Focus Hint + Camera Motion Guard + Bottom Calibration Strip
  const clutterReady = telemView[22] > 0.5;
  const bgMaskPct = Math.round(telemView[23] * 100);
  const clutterStr = clutterReady ? `${telemView[4].toFixed(1)}dB (${bgMaskPct}%bg)` : 'WARMING';
  const ecaTag = showEcaOverlay ? ' | ECA@t' : '';
  const shownSwaps = sourceMode === 'testbench' ? oraclePadicSwaps : Math.round(telemView[5]);

  if (rawWipeRatio <= 0.002) {
    hudCtx.fillStyle = 'rgba(6, 10, 18, 0.82)';
    hudCtx.fillRect(10, 10, 598, 24);
    hudCtx.fillStyle = '#8ef9f3';
    hudCtx.font = '600 10.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(
      `VIEW [t-${effK}] ${frameW}x${frameH} | ECA ${clutterStr} | IDSW:${shownSwaps} | KAL:${Math.round(telemView[6])}${ecaTag}`,
      16,
      26
    );
  }

  const camState = Math.round(telemView[TELEM_GLOBAL + 9]);
  if (camState > 0 && sourceMode !== 'testbench') {
    const camMsg = camState === 2
      ? '⚡ SCENE CUT DETECTED · ECA SUBSPACE & TRACKS AUTO-RESET'
      : `⚠ CAMERA PTZ PAN (${Math.round(telemView[TELEM_GLOBAL + 10])} px/s) · TRACKING & SCALE HELD`;
    const bw = 468;
    const bx = Math.round((W - bw) * 0.5);
    hudCtx.fillStyle = 'rgba(26, 12, 16, 0.92)';
    hudCtx.fillRect(bx, 40, bw, 22);
    hudCtx.strokeStyle = camState === 2 ? '#00f0ff' : '#ffb703';
    hudCtx.lineWidth = 1.4;
    hudCtx.strokeRect(bx, 40, bw, 22);
    hudCtx.fillStyle = camState === 2 ? '#8ef9f3' : '#ffd166';
    hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(camMsg, bx + 12, 55);
  }

  const pTid = primarySlot >= 0 ? Math.round(telemView[32 + primarySlot * 56 + 1]) : 0;
  const focusStr = focusTid !== 0
    ? `◎ FOCUS: T${focusTid} [PINNED · CLICK TO UNPIN]`
    : pTid > 0
    ? `◎ FOCUS: T${pTid} [AUTO · CLICK PATCH TO PIN]`
    : '◎ AWAITING MOTION PATCH';
  hudCtx.fillStyle = 'rgba(6, 10, 18, 0.82)';
  hudCtx.fillRect(W - 266, 10, 256, 24);
  hudCtx.fillStyle = focusTid !== 0 ? '#ffd166' : '#a8c0df';
  hudCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  hudCtx.fillText(focusStr, W - 258, 26);

  // Bottom-left Scene Gravity Prior badge (Provisional on 1 arc vs Locked >= 2 arcs)
  const sceneGPx = telemView[TELEM_GLOBAL + 0];
  const sceneGConf = telemView[TELEM_GLOBAL + 1];
  const sceneGArcs = Math.round(telemView[TELEM_GLOBAL + 11]);
  if (labMode === 0 && sceneGConf >= 0.20 && sceneGPx > 50.0) {
    const locked = sceneGConf >= AUTO_G_LOCK_CONF;
    const gStatus = locked ? `LOCKED · ${Math.max(2, sceneGArcs)} ARCS` : `PROV · ${Math.max(1, sceneGArcs)}/2 ARCS`;
    hudCtx.fillStyle = 'rgba(6, 10, 18, 0.84)';
    hudCtx.fillRect(10, H - 32, 264, 22);
    hudCtx.strokeStyle = locked ? 'rgba(56, 242, 122, 0.58)' : 'rgba(255, 209, 102, 0.48)';
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeRect(10, H - 32, 264, 22);
    hudCtx.fillStyle = locked ? '#38f27a' : '#ffd166';
    hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(`ĝ = ${Math.round(sceneGPx)} px/s² (${gStatus})`, 16, H - 17);
  }

  // Bottom-right Metric / Homography Calibration badge with explicit Provenance tag
  if (calMode !== 'none') {
    const pBadge = calProvenance === 'assumed' ? 'ASSUMED' : 'SURVEYED';
    const calShort = isCameraMotionSuspendingCal()
      ? '⚠ CALIBRATION HELD (CAM MOTION)'
      : calMode === 'homography' && calHomographyValid
      ? `⬠ PLANE H [${pBadge}]: ${calPlaneW.toFixed(2)}×${calPlaneH.toFixed(2)}m`
      : calMode === 'auto_g'
      ? (calAutoGLocked
          ? `⚡ AUTO-g [LOCKED]: ${Math.round(getEffectivePxPerMeter())} px/m`
          : `⚡ AUTO-g [PROVISIONAL · px/s]`)
      : `📏 RULER [${pBadge}]: ${calRulerMeters.toFixed(2)}m (${Math.round(getEffectivePxPerMeter())}px/m)`;
    const cw = 324;
    hudCtx.fillStyle = 'rgba(6, 10, 18, 0.86)';
    hudCtx.fillRect(W - cw - 10, H - 32, cw, 22);
    hudCtx.strokeStyle = calProvenance === 'assumed' ? 'rgba(255, 209, 102, 0.55)' : 'rgba(0, 229, 255, 0.55)';
    hudCtx.lineWidth = 1.0;
    hudCtx.strokeRect(W - cw - 10, H - 32, cw, 22);
    hudCtx.fillStyle = calProvenance === 'assumed' ? '#ffd166' : '#00e5ff';
    hudCtx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    hudCtx.fillText(calShort, W - cw - 2, H - 17);
  }

  if (rawWipeRatio > 0.002) {
    const wipeX = Math.round(Math.max(0, Math.min(W, rawWipeRatio * W)));
    if (wipeX > 0) {
      hudCtx.save();
      hudCtx.beginPath();
      hudCtx.rect(0, 0, wipeX, H);
      hudCtx.clip();
      hudCtx.imageSmoothingEnabled = false;
      hudCtx.drawImage(dispStageCanvas, 0, 0, W, H);
      if (wipeX >= 124) {
        const badgeW = Math.min(168, wipeX - 16);
        hudCtx.fillStyle = 'rgba(5, 8, 15, 0.86)';
        hudCtx.fillRect(10, 10, badgeW, 24);
        hudCtx.strokeStyle = 'rgba(236, 244, 255, 0.38)';
        hudCtx.lineWidth = 1;
        hudCtx.strokeRect(10, 10, badgeW, 24);
        hudCtx.fillStyle = '#ecf4ff';
        hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
        hudCtx.fillText(`RAW PIXELS [t-${effK}]`, 18, 26);
      }
      hudCtx.restore();

      if (wipeX < W) {
        const rightGap = W - 276 - (wipeX + 10);
        if (rightGap >= 156) {
          const specTxt = rightGap >= 336
            ? `SPECTRAL [t-${effK}] | ECA ${clutterStr} | IDSW:${shownSwaps}`
            : `SPECTRAL [t-${effK}] | IDSW:${shownSwaps}`;
          const sw = Math.min(rightGap, Math.max(156, specTxt.length * 6.4 + 16));
          hudCtx.save();
          hudCtx.fillStyle = 'rgba(6, 10, 18, 0.86)';
          hudCtx.fillRect(wipeX + 10, 10, sw, 24);
          hudCtx.strokeStyle = 'rgba(0, 229, 255, 0.42)';
          hudCtx.lineWidth = 1;
          hudCtx.strokeRect(wipeX + 10, 10, sw, 24);
          hudCtx.fillStyle = '#8ef9f3';
          hudCtx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
          hudCtx.fillText(specTxt, wipeX + 18, 26);
          hudCtx.restore();
        }
        hudCtx.save();
        hudCtx.strokeStyle = '#00e5ff';
        hudCtx.lineWidth = 2;
        hudCtx.beginPath();
        hudCtx.moveTo(wipeX + 0.5, 0);
        hudCtx.lineTo(wipeX + 0.5, H);
        hudCtx.stroke();
        hudCtx.restore();
      }
    }
  }
}

/// Side-by-Side Split PiP: Left = 1x Raw Stabilized Crop from `rgba_ring[dispSlot]`,
/// Right = 25x Eulerian Temporal-Contrast Magnified Residual (`chronos_get_pip_rgba_ptr()`).
/// Zero extra Wasm memory bytes and zero per-frame allocations.
function renderEulerianPip() {
  if (!pipCtx || !pipImgData) return;
  const W = pipCanvas.width;   // 248
  const H = pipCanvas.height;  // 144

  pipCtx.fillStyle = '#070b13';
  pipCtx.fillRect(0, 0, W, H);

  const pSlot = resolvePrimarySlot();
  const boxSize = Math.min(H - 42, Math.floor((W - 20) * 0.5));
  const leftX = 6;
  const rightX = W - boxSize - 6;
  const topY = 19;

  pipCtx.fillStyle = '#9db4d4';
  pipCtx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
  pipCtx.fillText('1x RAW [t−K]', leftX + 2, 13);
  pipCtx.fillStyle = '#00f0ff';
  pipCtx.fillText(`${eulerianGain.toFixed(0)}x TEMPORAL Δ`, rightX + 2, 13);

  if (pSlot >= 0 && telemView[32 + pSlot * 56] > 0.5) {
    const b = 32 + pSlot * 56;
    const fx = telemView[b + 2];
    const fy = telemView[b + 3];
    const halfP = pipDim * 0.5;
    const cropX = Math.max(0, Math.min(frameW - pipDim, Math.round(fx - halfP)));
    const cropY = Math.max(0, Math.min(frameH - pipDim, Math.round(fy - halfP)));

    pipCtx.imageSmoothingEnabled = false;
    // Left viewport: 1x Raw stabilized crop directly from `dispStageCanvas` (`rgba_ring[dispSlot]`)
    pipCtx.drawImage(dispStageCanvas, cropX, cropY, pipDim, pipDim, leftX, topY, boxSize, boxSize);

    // Right viewport: 25x Eulerian temporal-contrast residual from Wasm SIMD128 PiP buffer
    pipStageCtx.putImageData(pipImgData, 0, 0);
    pipCtx.drawImage(pipStageCanvas, 0, 0, pipDim, pipDim, rightX, topY, boxSize, boxSize);

    pipCtx.strokeStyle = 'rgba(168, 192, 223, 0.55)';
    pipCtx.lineWidth = 1;
    pipCtx.strokeRect(leftX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);

    pipCtx.strokeStyle = 'rgba(0, 240, 255, 0.70)';
    pipCtx.strokeRect(rightX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
    pipCtx.beginPath();
    pipCtx.moveTo(rightX + boxSize * 0.5, topY);
    pipCtx.lineTo(rightX + boxSize * 0.5, topY + boxSize);
    pipCtx.moveTo(rightX, topY + boxSize * 0.5);
    pipCtx.lineTo(rightX + boxSize, topY + boxSize * 0.5);
    pipCtx.stroke();

    const pTid = Math.round(telemView[b + 1]);
    const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
    const modeTag = focusTid === pTid ? 'PINNED' : 'AUTO';
    pipCtx.fillStyle = 'rgba(6, 10, 18, 0.88)';
    pipCtx.fillRect(4, H - 19, W - 8, 15);
    pipCtx.fillStyle = focusTid === pTid ? '#ffd166' : '#00f0ff';
    pipCtx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
    pipCtx.fillText(`T${pTid} [${modeTag}] | ${pipDim}x${pipDim} FISTA STABILIZED`, 8, H - 8);
  } else {
    pipCtx.strokeStyle = '#1d293d';
    pipCtx.strokeRect(leftX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
    pipCtx.strokeRect(rightX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
    pipCtx.fillStyle = '#9db4d4';
    pipCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
    pipCtx.fillText('AWAITING PATCH LOCK', 12, H - 8);
  }
}

/// Updates EMA-smoothed per-slot kinematics & sticky hold snapshot before HUD and Roster rendering.
function updateSlotRosterEma() {
  if (!telemView) return;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const b = 32 + m * 56;
    if (telemView[b] <= 0.5) continue;

    const kb = TELEM_KINETIC + m * 8;
    const tid = Math.round(telemView[b + 1]);
    const fxPx = telemView[b + 2];
    const fyPx = telemView[b + 3];
    const physVel = mapVelocityToPhysical(fxPx, fyPx, telemView[kb + 1], telemView[kb + 0]);
    const rawSpd = physVel[0];
    const rawHdg = physVel[1];
    const rawOmega = (calMode === 'homography' && calHomographyValid && physVel[2] > 0.5)
      ? slotRectAngVel[m]
      : telemView[kb + 2];
    const rawEca = slotMetaView ? Math.min(100.0, Math.max(0.0, slotMetaView[m * 8 + 2] * 100.0)) : 0.0;

    const sameTid = slotHoldTid[m] === tid;
    const unitChanged = Math.round(slotLastMetricValid[m]) !== Math.round(physVel[2]);
    const segChanged = slotLastCalSegId[m] !== calSegmentId;
    if (!sameTid || unitChanged || segChanged) {
      slotSmoothedSpeed[m] = rawSpd;
      slotSmoothedHeading[m] = rawHdg;
      slotSmoothedOmega[m] = rawOmega;
      slotSmoothedEca[m] = rawEca;
    } else {
      slotSmoothedSpeed[m] = 0.78 * slotSmoothedSpeed[m] + 0.22 * rawSpd;
      slotSmoothedHeading[m] = wrapDeg180(
        slotSmoothedHeading[m] + 0.24 * wrapDeg180(rawHdg - slotSmoothedHeading[m])
      );
      slotSmoothedOmega[m] = 0.78 * slotSmoothedOmega[m] + 0.22 * rawOmega;
      slotSmoothedEca[m] = 0.80 * slotSmoothedEca[m] + 0.20 * rawEca;
    }

    const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
    slotHoldTid[m] = tid;
    slotHoldLeft[m] = focusTid !== 0 && tid === focusTid ? PINNED_HOLD_FRAMES : SLOT_HOLD_FRAMES;
    slotHoldVeilDrawn[m] = 0;
    slotLastVal[m] = Math.round(telemView[b + 14]);
    slotLastSigHi[m] = Math.round(telemView[b + 12]);
    slotLastSigLo[m] = Math.round(telemView[b + 13]);
    slotLastMetricValid[m] = physVel[2];
    slotLastCalSegId[m] = calSegmentId;

    if (slotMetaView) {
      slotLastAgeFrames[m] = Math.round(slotMetaView[m * 8 + 0]);
      slotLastCoast[m] = Math.round(slotMetaView[m * 8 + 1]);
      slotLastReacq[m] = Math.round(slotMetaView[m * 8 + 3]);
      slotLastFreqHz[m] = slotMetaView[m * 8 + 4];
    }

    if (telemView[kb + 6] > -9.0e5 && telemView[kb + 7] > -9.0e5) {
      const pend = computePendulumKinematics(fxPx, fyPx, telemView[kb + 6], telemView[kb + 7]);
      slotLastPivotValid[m] = 1;
      slotLastPendSwing[m] = sameTid && !segChanged
        ? 0.76 * slotLastPendSwing[m] + 0.24 * pend[0]
        : pend[0];
      slotLastPendRad[m] = sameTid && !segChanged
        ? 0.82 * slotLastPendRad[m] + 0.18 * pend[1]
        : pend[1];
    } else {
      slotLastPivotValid[m] = 0;
      slotLastPendSwing[m] = 0.0;
      slotLastPendRad[m] = 0.0;
    }
  }
}

/// Renders all 16 per-slot Side-by-Side Split `1x RAW [t-K] | 25x TEMPORAL Δ` OffscreenCanvases
/// with exact integer scaling (`128x128` per half: `4x` for `32x32` SD, `2x` for `64x64` HD)
/// and a calm 48-frame (~1.5s) translucent hold-on-loss veil so slots never blink on exit.
function renderAllSlotPips(advanceHold = true) {
  const hasTelem = Boolean(telemView);
  const primarySlot = hasTelem ? resolvePrimarySlot() : -1;
  const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
  const gainTxt = `${eulerianGain.toFixed(0)}x Δ`;

  for (let m = 0; m < MAX_TARGETS; m++) {
    const active = hasTelem && telemView[32 + m * 56] > 0.5;
    const ctx = slotPipCtxs[m];
    const imgData = slotPipImgData[m];

    if (!active && slotHoldLeft[m] > 0) {
      if (ctx && !slotHoldVeilDrawn[m]) {
        slotHoldVeilDrawn[m] = 1;
        const W = slotPipCanvases[m].width;
        const H = slotPipCanvases[m].height;
        const boxSize = Math.min(H - 20, Math.floor((W - 14) * 0.5));
        const leftX = 4;
        const topY = 14;
        ctx.fillStyle = 'rgba(6, 10, 18, 0.48)';
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = 'rgba(7, 11, 18, 0.90)';
        ctx.fillRect(leftX + 3, topY + boxSize - 17, W - leftX * 2 - 6, 15);
        ctx.fillStyle = '#9db4d4';
        ctx.font = '700 9px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.fillText(`HOLD T${slotHoldTid[m]} · LAST LOCK`, leftX + 7, topY + boxSize - 6);
      }
      if (advanceHold) {
        slotHoldLeft[m]--;
      }
      continue;
    }
    slotHoldVeilDrawn[m] = 0;

    if (!ctx) continue;
    const W = slotPipCanvases[m].width;   // 276
    const H = slotPipCanvases[m].height;  // 148
    const boxSize = Math.min(H - 20, Math.floor((W - 14) * 0.5)); // 128 = 32*4 = 64*2
    const leftX = 4;
    const rightX = W - boxSize - 4;       // 144
    const topY = 14;

    if (active && imgData) {
      const b = 32 + m * 56;
      const tid = Math.round(telemView[b + 1]);
      const fx = telemView[b + 2];
      const fy = telemView[b + 3];
      const dispVis = telemView[b + 10] > 0.5;
      const isBridged = telemView[b + 11] > 0.5;
      const isPinned = focusTid !== 0 && tid === focusTid;
      const isPrimary = m === primarySlot;
      const color = TARGET_COLORS[Math.max(0, tid - 1) % TARGET_COLORS.length];

      slotIdleDrawn[m] = 0;

      ctx.fillStyle = '#070b13';
      ctx.fillRect(0, 0, W, H);

      ctx.fillStyle = '#9db4d4';
      ctx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText('1x RAW [t−K]', leftX + 2, 10);
      ctx.fillStyle = '#00f0ff';
      ctx.fillText(gainTxt, rightX + 2, 10);

      const halfP = pipDim * 0.5;
      const cropX = Math.max(0, Math.min(frameW - pipDim, Math.round(fx - halfP)));
      const cropY = Math.max(0, Math.min(frameH - pipDim, Math.round(fy - halfP)));

      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(dispStageCanvas, cropX, cropY, pipDim, pipDim, leftX, topY, boxSize, boxSize);

      pipStageCtx.putImageData(imgData, 0, 0);
      ctx.drawImage(pipStageCanvas, 0, 0, pipDim, pipDim, rightX, topY, boxSize, boxSize);

      ctx.lineWidth = 1;
      ctx.strokeStyle = isBridged ? '#ffd166' : (!dispVis ? 'rgba(255, 209, 102, 0.78)' : color);
      ctx.strokeRect(leftX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
      if (isPinned || isPrimary) {
        ctx.strokeStyle = isPinned ? '#ffffff' : 'rgba(0, 240, 255, 0.85)';
        ctx.strokeRect(leftX - 1.5, topY - 1.5, boxSize + 3, boxSize + 3);
      }

      ctx.strokeStyle = 'rgba(0, 240, 255, 0.72)';
      ctx.strokeRect(rightX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
      ctx.beginPath();
      ctx.moveTo(rightX + boxSize * 0.5, topY);
      ctx.lineTo(rightX + boxSize * 0.5, topY + boxSize);
      ctx.moveTo(rightX, topY + boxSize * 0.5);
      ctx.lineTo(rightX + boxSize, topY + boxSize * 0.5);
      ctx.stroke();
    } else if (!slotIdleDrawn[m]) {
      slotIdleDrawn[m] = 1;
      ctx.fillStyle = '#060a12';
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = '#162033';
      ctx.lineWidth = 1;
      ctx.strokeRect(leftX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
      ctx.strokeRect(rightX + 0.5, topY + 0.5, boxSize - 1, boxSize - 1);
      ctx.fillStyle = '#5c7394';
      ctx.font = '600 9px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText('1x RAW', leftX + 3, 10);
      ctx.fillText('Δ STANDBY', rightX + 3, 10);
      ctx.fillStyle = '#4c6180';
      ctx.fillText(`SLOT ${String(m + 1).padStart(2, '0')} IDLE`, W * 0.5 - 32, topY + boxSize * 0.5 + 3);
    }
  }
}

function renderHilbertScope() {
  if (!hilbertCtx) return;
  const W = hilbertCanvas.width;
  const H = hilbertCanvas.height;

  hilbertCtx.fillStyle = '#080c14';
  hilbertCtx.fillRect(0, 0, W, H);

  const pSize = Math.min(H - 28, 140);
  const cx = 14 + pSize * 0.5;
  const cy = H * 0.5 + 6;
  const rad = pSize * 0.43;

  // Auto-range scale across clean physical amplitude tiers (handles ±0.8px micro-tremor to ±28px metronomes!)
  let peakAbs = 0.0;
  for (let k = 0; k < 64; k++) {
    const idx = (scopeWriteIdx + SCOPE_HIST - 64 + k) % SCOPE_HIST;
    if (histPhaseValid[idx]) {
      const ar = Math.abs(histTremorR[idx]);
      const ah = Math.abs(histTremorH[idx]);
      if (ar > peakAbs) peakAbs = ar;
      if (ah > peakAbs) peakAbs = ah;
    }
  }
  const peakNorm = peakAbs / resScale;
  let rangeBasePx = 1.5;
  if (peakNorm > 10.0) rangeBasePx = 28.0;
  else if (peakNorm > 3.6) rangeBasePx = 12.0;
  else if (peakNorm > 1.35) rangeBasePx = 4.0;
  const fullScalePx = rangeBasePx * resScale;
  const polarScale = rad / fullScalePx;
  const invWaveScale = 1.0 / fullScalePx;

  hilbertCtx.strokeStyle = '#1d293d';
  hilbertCtx.lineWidth = 1;
  hilbertCtx.beginPath();
  hilbertCtx.arc(cx, cy, rad, 0, Math.PI * 2);
  hilbertCtx.arc(cx, cy, rad * 0.5, 0, Math.PI * 2);
  hilbertCtx.moveTo(cx - rad, cy); hilbertCtx.lineTo(cx + rad, cy);
  hilbertCtx.moveTo(cx, cy - rad); hilbertCtx.lineTo(cx, cy + rad);
  hilbertCtx.stroke();

  hilbertCtx.fillStyle = '#738db5';
  hilbertCtx.font = '600 8px ui-monospace, SFMono-Regular, Menlo, monospace';
  hilbertCtx.fillText(`±${fullScalePx.toFixed(fullScalePx < 4 ? 1 : 0)}px`, cx - rad + 2, cy + rad + 10);

  const tx = cx + Math.cos(phaseOffsetRad) * rad;
  const ty = cy - Math.sin(phaseOffsetRad) * rad;
  hilbertCtx.strokeStyle = phaseOffsetRad > 0 ? '#ffb703' : '#00f0ff';
  hilbertCtx.setLineDash([3, 3]);
  hilbertCtx.beginPath();
  hilbertCtx.moveTo(cx, cy);
  hilbertCtx.lineTo(tx, ty);
  hilbertCtx.stroke();
  hilbertCtx.setLineDash([]);

  hilbertCtx.strokeStyle = '#38f27a';
  hilbertCtx.lineWidth = 1.6;
  hilbertCtx.beginPath();
  let polarPenDown = false;
  let prevTid = -1;
  for (let k = 0; k < 48; k++) {
    const idx = (scopeWriteIdx + SCOPE_HIST - 48 + k) % SCOPE_HIST;
    const tidHere = histPipTid[idx];
    if (!histPhaseValid[idx] || tidHere < 0) {
      polarPenDown = false;
      prevTid = -1;
      continue;
    }
    const rx = cx + Math.max(-rad, Math.min(rad, histTremorR[idx] * polarScale));
    const hy = cy - Math.max(-rad, Math.min(rad, histTremorH[idx] * polarScale));
    if (!polarPenDown || tidHere !== prevTid) {
      hilbertCtx.moveTo(rx, hy);
      polarPenDown = true;
    } else {
      hilbertCtx.lineTo(rx, hy);
    }
    prevTid = tidHere;
  }
  hilbertCtx.stroke();

  const curIdx = (scopeWriteIdx + SCOPE_HIST - 1) % SCOPE_HIST;
  const headX = cx + Math.max(-rad, Math.min(rad, histTremorR[curIdx] * polarScale));
  const headY = cy - Math.max(-rad, Math.min(rad, histTremorH[curIdx] * polarScale));
  if (histPhaseValid[curIdx]) {
    hilbertCtx.fillStyle = '#ffffff';
    hilbertCtx.beginPath();
    hilbertCtx.arc(headX, headY, 3.2, 0, Math.PI * 2);
    hilbertCtx.fill();
  }

  const waveX0 = 26 + pSize;
  const waveW = W - waveX0 - 10;
  const waveMidY = H * 0.5 + 6;
  const waveAmp = (H - 40) * 0.42;

  hilbertCtx.strokeStyle = '#182234';
  hilbertCtx.beginPath();
  hilbertCtx.moveTo(waveX0, waveMidY);
  hilbertCtx.lineTo(waveX0 + waveW, waveMidY);
  hilbertCtx.stroke();

  hilbertCtx.strokeStyle = 'rgba(255, 183, 3, 0.38)';
  for (let i = 0; i < SCOPE_HIST; i++) {
    const idx = (scopeWriteIdx + i) % SCOPE_HIST;
    if (histStrobe[idx]) {
      const wx = waveX0 + (i / (SCOPE_HIST - 1)) * waveW;
      hilbertCtx.beginPath();
      hilbertCtx.moveTo(wx, 22);
      hilbertCtx.lineTo(wx, H - 10);
      hilbertCtx.stroke();
    }
  }

  // R(t) trace (breaks cleanly across target switches)
  hilbertCtx.strokeStyle = '#00f0ff';
  hilbertCtx.lineWidth = 1.5;
  hilbertCtx.beginPath();
  let rPenDown = false;
  prevTid = -1;
  for (let i = 0; i < SCOPE_HIST; i++) {
    const idx = (scopeWriteIdx + i) % SCOPE_HIST;
    const tidHere = histPipTid[idx];
    if (tidHere < 0) {
      rPenDown = false;
      prevTid = -1;
      continue;
    }
    const wx = waveX0 + (i / (SCOPE_HIST - 1)) * waveW;
    const wy = waveMidY - Math.max(-1, Math.min(1, histTremorR[idx] * invWaveScale)) * waveAmp;
    if (!rPenDown || tidHere !== prevTid) {
      hilbertCtx.moveTo(wx, wy);
      rPenDown = true;
    } else {
      hilbertCtx.lineTo(wx, wy);
    }
    prevTid = tidHere;
  }
  hilbertCtx.stroke();

  // H{R}(t) quadrature trace
  hilbertCtx.strokeStyle = '#e840f8';
  hilbertCtx.lineWidth = 1.3;
  hilbertCtx.beginPath();
  let hPenDown = false;
  prevTid = -1;
  for (let i = 0; i < SCOPE_HIST; i++) {
    const idx = (scopeWriteIdx + i) % SCOPE_HIST;
    const tidHere = histPipTid[idx];
    if (!histPhaseValid[idx] || tidHere < 0) {
      hPenDown = false;
      prevTid = -1;
      continue;
    }
    const wx = waveX0 + (i / (SCOPE_HIST - 1)) * waveW;
    const wy = waveMidY - Math.max(-1, Math.min(1, histTremorH[idx] * invWaveScale)) * waveAmp;
    if (!hPenDown || tidHere !== prevTid) {
      hilbertCtx.moveTo(wx, wy);
      hPenDown = true;
    } else {
      hilbertCtx.lineTo(wx, wy);
    }
    prevTid = tidHere;
  }
  hilbertCtx.stroke();

  const pSlot = Math.round(telemView[8]);
  const cohPct = pSlot >= 0 ? Math.round(telemView[TELEM_EXCUR + pSlot * 8 + 6] * 100) : 0;
  hilbertCtx.fillStyle = '#9db4d4';
  hilbertCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  const phaseValid = telemView[26] > 0.5;
  const freqValid = telemView[TELEM_GLOBAL + 12] > 0.5 && telemView[13] > 0.02;
  if (phaseValid) {
    const phiDeg = (telemView[11] * 180 / Math.PI).toFixed(0);
    const fTxt = freqValid ? `f=${telemView[13].toFixed(2)}Hz` : 'APERIODIC';
    hilbertCtx.fillText(
      `HILBERT φ=${phiDeg}° | ${fTxt} | COH ${cohPct}% | R CYAN · H{R} MAGENTA`,
      10,
      14
    );
  } else {
    const effK = Math.round(telemView[1]);
    hilbertCtx.fillText(
      effK < 7 ? 'HILBERT QUADRATURE GATED (REQUIRES K >= 7 FUTURE SUPPORT)' : 'HILBERT WINDOW WARMING / AWAITING OSCILLATION',
      10,
      14
    );
  }
}

function renderSpectralAndRingScope() {
  if (!spectralCtx) return;
  const W = spectralCanvas.width;
  const H = spectralCanvas.height;

  spectralCtx.fillStyle = '#080c14';
  spectralCtx.fillRect(0, 0, W, H);

  const midH = Math.floor(H * 0.52);
  spectralCtx.fillStyle = '#9db4d4';
  spectralCtx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  spectralCtx.fillText(
    `PRE-ASSIGNMENT FIEDLER λ₂=${telemView[7].toFixed(4)} (Δλ=${telemView[24].toFixed(3)})`,
    10,
    14
  );

  const barBaseY = Math.floor(midH * 0.58);
  spectralCtx.strokeStyle = '#1d293d';
  spectralCtx.beginPath();
  spectralCtx.moveTo(10, barBaseY);
  spectralCtx.lineTo(W * 0.56, barBaseY);
  spectralCtx.stroke();

  const barAreaW = W * 0.54;
  const barSlotW = barAreaW / MAX_TARGETS;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    const bx = 12 + m * barSlotW;
    if (telemView[base] < 0.5) {
      spectralCtx.fillStyle = '#151e2e';
      spectralCtx.fillRect(bx, barBaseY - 2, barSlotW - 4, 4);
      continue;
    }
    const v2 = fiedlerView[m];
    const bh = Math.max(-28, Math.min(28, v2 * 36));
    spectralCtx.fillStyle = v2 >= 0 ? '#00f0ff' : '#ff4d6d';
    if (bh >= 0) spectralCtx.fillRect(bx, barBaseY - bh, barSlotW - 4, Math.max(2, bh));
    else spectralCtx.fillRect(bx, barBaseY, barSlotW - 4, Math.max(2, -bh));
  }

  const spX0 = W * 0.60;
  const spW = W - spX0 - 10;
  spectralCtx.strokeStyle = '#ffd166';
  spectralCtx.lineWidth = 1.4;
  spectralCtx.beginPath();
  for (let i = 0; i < SCOPE_HIST; i++) {
    const idx = (scopeWriteIdx + i) % SCOPE_HIST;
    const x = spX0 + (i / (SCOPE_HIST - 1)) * spW;
    const y = midH - 8 - Math.min(1.0, histLambda2[idx] * 0.35) * (midH - 28);
    if (i === 0) spectralCtx.moveTo(x, y);
    else spectralCtx.lineTo(x, y);
  }
  spectralCtx.stroke();

  // Chronological 12-Slot Non-Causal Timeline: u = 0 (t-11) .. uDisp [t-K VIEW] .. 11 [t NOW]
  const ringTop = midH + 20;
  const effK = Math.round(telemView[1]);
  const uDisp = (RING_LEN - 1) - effK;
  spectralCtx.fillStyle = '#9db4d4';
  spectralCtx.fillText(`NON-CAUSAL TIMELINE: PAST [t−11] → VIEW [t−${effK}] → FUTURE [t NOW]`, 10, midH + 12);

  const fIdx = currentCausalEngineFrameU32();
  const slotOldest = wave1CoreAbiValid && wave1TelemView && (wave1TelemView[2] >>> 0) < RING_LEN
    ? ((wave1TelemView[2] >>> 0) + 1) % RING_LEN
    : (fIdx + 1) % RING_LEN;
  const cellW = (W - 20) / RING_LEN;
  const cellH = H - ringTop - 8;
  const pIdx = Math.round(telemView[8]);
  const pBridged = pIdx >= 0 && telemView[32 + pIdx * 56 + 11] > 0.5;

  for (let u = 0; u < RING_LEN; u++) {
    const cx = 10 + u * cellW;
    const isNow = u === RING_LEN - 1;
    const isDisp = u === uDisp;
    const isFuture = u > uDisp;
    const relLag = u - (RING_LEN - 1); // -11 .. 0
    const ringSlot = (slotOldest + u) % RING_LEN;

    let visState = 1;
    if (pIdx >= 0) {
      visState = telemView[32 + pIdx * 56 + 44 + u] > 0.5 ? 1 : 0;
    }

    spectralCtx.fillStyle = isDisp
      ? 'rgba(0, 240, 255, 0.30)'
      : isNow
      ? 'rgba(56, 242, 122, 0.28)'
      : visState === 0
      ? 'rgba(255, 209, 102, 0.24)'
      : isFuture
      ? 'rgba(0, 180, 216, 0.12)'
      : '#101826';
    spectralCtx.fillRect(cx + 1, ringTop, cellW - 3, cellH);

    spectralCtx.strokeStyle = isDisp
      ? '#00f0ff'
      : isNow
      ? '#38f27a'
      : visState === 0
      ? '#ffd166'
      : isFuture
      ? '#1d4e68'
      : '#26344d';
    spectralCtx.lineWidth = isDisp || isNow ? 1.8 : 1.0;
    spectralCtx.strokeRect(cx + 1.5, ringTop + 0.5, cellW - 4, cellH - 1);

    spectralCtx.fillStyle = isDisp ? '#ffffff' : '#c6d8f0';
    spectralCtx.font = '600 8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    const topLbl = relLag === 0 ? 't' : `t${relLag}`;
    spectralCtx.fillText(topLbl, cx + 3, ringTop + 12);

    spectralCtx.fillStyle = '#5c7599';
    spectralCtx.font = '500 7.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    spectralCtx.fillText(`s${ringSlot}`, cx + 4, ringTop + 23);

    spectralCtx.font = '700 8.5px ui-monospace, SFMono-Regular, Menlo, monospace';
    if (isDisp) {
      spectralCtx.fillStyle = '#00f0ff';
      spectralCtx.fillText('VIEW', cx + 3, ringTop + cellH - 5);
    } else if (isNow) {
      spectralCtx.fillStyle = '#38f27a';
      spectralCtx.fillText('NOW', cx + 4, ringTop + cellH - 5);
    } else if (visState === 0) {
      spectralCtx.fillStyle = '#ffd166';
      spectralCtx.fillText(pBridged ? 'BRG' : 'OCC', cx + 4, ringTop + cellH - 5);
    } else if (isFuture) {
      spectralCtx.fillStyle = '#6ec8e8';
      spectralCtx.fillText('+LK', cx + 4, ringTop + cellH - 5);
    }
  }
}

function pushPhysicsPhaseSample() {
  if (!telemView) return;
  const pIdx = resolvePrimarySlot();
  const w = phaseHistWrite;
  if (pIdx >= 0 && telemView[32 + pIdx * 56] > 0.5) {
    const b = 32 + pIdx * 56;
    const kb = TELEM_KINETIC + pIdx * 8;
    const pb = TELEM_PHYSICS_START + pIdx * TELEM_PHYSICS_STRIDE;
    const spd = safeFinite(telemView[kb + 1]);
    const hRad = safeFinite(telemView[kb + 0]) * (Math.PI / 180.0);
    phaseHistTid[w] = Math.round(safeFinite(telemView[b + 1]));
    phaseHistX[w] = safeFinite(telemView[b + 2]);
    phaseHistY[w] = safeFinite(telemView[b + 3]);
    phaseHistVx[w] = spd * Math.cos(hRad);
    phaseHistVy[w] = spd * Math.sin(hRad);
    phaseHistE[w] = safeFinite(telemView[pb + 12]);
    phaseHistRegime[w] = Math.max(0, Math.min(8, Math.round(safeFinite(telemView[pb + 4]))));
    phaseHistJump[w] = Math.max(0, Math.min(4, Math.round(safeFinite(telemView[pb + 15]))));
  } else {
    phaseHistTid[w] = -1;
    phaseHistX[w] = 0.0;
    phaseHistY[w] = 0.0;
    phaseHistVx[w] = 0.0;
    phaseHistVy[w] = 0.0;
    phaseHistE[w] = 0.0;
    phaseHistRegime[w] = 0;
    phaseHistJump[w] = 0;
  }
  const eb = TELEM_ENSEMBLE_START;
  phaseHistEnsK[w] = safeFinite(telemView[eb + 3]);
  phaseHistKurR[w] = safeFinite(telemView[eb + 0]);
  phaseHistKurN[w] = Math.max(0, Math.min(255, Math.round(safeFinite(telemView[eb + 2]))));
  phaseHistWrite = (phaseHistWrite + 1) % PHASE_HIST;
  if (phaseHistCount < PHASE_HIST) phaseHistCount++;
}

function renderPhysicsPhaseCanvas(force = false) {
  if (!physicsPhaseCtx || !physicsPhaseCanvas) return;
  const nowMs = performance.now();
  if (!force && nowMs - lastPhysicsPhaseRenderMs < 66.0) return;
  lastPhysicsPhaseRenderMs = nowMs;

  const W = physicsPhaseCanvas.width;
  const H = physicsPhaseCanvas.height;
  const ctx = physicsPhaseCtx;

  ctx.fillStyle = '#070b13';
  ctx.fillRect(0, 0, W, H);

  const lx = 6;
  const ly = 6;
  const lw = Math.max(96, Math.floor(W * 0.49) - 8);
  const lh = Math.max(60, H - 12);
  const rx = lx + lw + 6;
  const ry = 6;
  const rw = Math.max(96, W - rx - 6);
  const rh = lh;

  ctx.fillStyle = '#0b1220';
  ctx.fillRect(lx, ly, lw, lh);
  ctx.fillRect(rx, ry, rw, rh);
  ctx.strokeStyle = '#192438';
  ctx.lineWidth = 1;
  ctx.strokeRect(lx + 0.5, ly + 0.5, lw - 1, lh - 1);
  ctx.strokeRect(rx + 0.5, ry + 0.5, rw - 1, rh - 1);

  const px0 = lx + 6;
  const py0 = ly + 20;
  const pw = Math.max(20, lw - 12);
  const ph = Math.max(20, lh - 26);

  const ex0 = rx + 6;
  const ey0 = ry + 20;
  const ew = Math.max(20, rw - 12);
  const eh = Math.max(24, Math.floor((rh - 44) * 0.62));
  const kx0 = rx + 6;
  const ky0 = ey0 + eh + 18;
  const kw = ew;
  const kh = Math.max(18, ry + rh - ky0 - 6);

  ctx.font = '600 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
  if (phaseHistCount <= 0 || !telemView) {
    ctx.fillStyle = '#9db4d4';
    ctx.fillText('PHASE PORTRAIT (q, qdot)', lx + 6, ly + 13);
    ctx.fillText('CONSERVATION E(t) & R_sync', rx + 6, ry + 13);
    ctx.fillStyle = '#5c7394';
    ctx.fillText('AWAITING TRACK DYNAMICS...', px0 + 8, py0 + ph * 0.5);
    ctx.fillText('AWAITING ENERGY & R_sync...', ex0 + 8, ey0 + eh * 0.55);
    return;
  }

  const lastIdx = (phaseHistWrite + PHASE_HIST - 1) % PHASE_HIST;
  const curTid = phaseHistTid[lastIdx];
  const curRegime = phaseHistRegime[lastIdx];
  const curR = phaseHistKurR[lastIdx];
  const curN = phaseHistKurN[lastIdx];

  // Left pane: contiguous phase portrait for current primary track
  let segLen = 0;
  if (curTid > 0) {
    for (let k = 0; k < phaseHistCount; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - 1 - k) % PHASE_HIST;
      if (phaseHistTid[idx] !== curTid) break;
      segLen++;
    }
  }

  ctx.strokeStyle = '#182438';
  ctx.lineWidth = 1;
  ctx.strokeRect(px0 + 0.5, py0 + 0.5, pw - 1, ph - 1);
  const pcx = px0 + pw * 0.5;
  const pcy = py0 + ph * 0.5;
  ctx.strokeStyle = 'rgba(100, 128, 168, 0.25)';
  ctx.beginPath();
  ctx.moveTo(px0, pcy);
  ctx.lineTo(px0 + pw, pcy);
  ctx.moveTo(pcx, py0);
  ctx.lineTo(pcx, py0 + ph);
  ctx.stroke();

  if (segLen >= 2) {
    let sumX = 0.0;
    let sumY = 0.0;
    for (let k = 0; k < segLen; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - segLen + k) % PHASE_HIST;
      sumX += phaseHistX[idx];
      sumY += phaseHistY[idx];
    }
    const meanX = sumX / segLen;
    const meanY = sumY / segLen;

    let varX = 0.0;
    let varY = 0.0;
    for (let k = 0; k < segLen; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - segLen + k) % PHASE_HIST;
      const dx = phaseHistX[idx] - meanX;
      const dy = phaseHistY[idx] - meanY;
      varX += dx * dx;
      varY += dy * dy;
    }
    const useY = varY > varX * 1.15;

    let maxQ = 4.0;
    let maxV = 15.0;
    for (let k = 0; k < segLen; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - segLen + k) % PHASE_HIST;
      const q = useY ? -(phaseHistY[idx] - meanY) : (phaseHistX[idx] - meanX);
      const v = useY ? phaseHistVy[idx] : phaseHistVx[idx];
      if (Math.abs(q) > maxQ) maxQ = Math.abs(q);
      if (Math.abs(v) > maxV) maxV = Math.abs(v);
    }
    maxQ *= 1.15;
    maxV *= 1.15;

    const col = TARGET_COLORS[Math.max(0, curTid - 1) % TARGET_COLORS.length];
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let k = 0; k < segLen; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - segLen + k) % PHASE_HIST;
      const q = useY ? -(phaseHistY[idx] - meanY) : (phaseHistX[idx] - meanX);
      const v = useY ? phaseHistVy[idx] : phaseHistVx[idx];
      const gx = pcx + (q / maxQ) * (pw * 0.45);
      const gy = pcy - (v / maxV) * (ph * 0.45);
      if (k === 0) ctx.moveTo(gx, gy);
      else ctx.lineTo(gx, gy);
    }
    ctx.stroke();

    // Jump markers & live state dot
    for (let k = 0; k < segLen; k++) {
      const idx = (phaseHistWrite + PHASE_HIST - segLen + k) % PHASE_HIST;
      if (phaseHistJump[idx] > 0) {
        const q = useY ? -(phaseHistY[idx] - meanY) : (phaseHistX[idx] - meanX);
        const v = useY ? phaseHistVy[idx] : phaseHistVx[idx];
        const gx = pcx + (q / maxQ) * (pw * 0.45);
        const gy = pcy - (v / maxV) * (ph * 0.45);
        ctx.fillStyle = '#ff4d6d';
        ctx.fillRect(gx - 2.5, gy - 2.5, 5, 5);
      }
    }
    const qNow = useY ? -(phaseHistY[lastIdx] - meanY) : (phaseHistX[lastIdx] - meanX);
    const vNow = useY ? phaseHistVy[lastIdx] : phaseHistVx[lastIdx];
    const gxNow = pcx + (qNow / maxQ) * (pw * 0.45);
    const gyNow = pcy - (vNow / maxV) * (ph * 0.45);
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(gxNow, gyNow, 2.8, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#9db4d4';
    ctx.fillText(`PHASE (${useY ? 'y,vy' : 'x,vx'}) T${curTid}`, lx + 6, ly + 13);
    ctx.fillStyle = col;
    ctx.fillText(PHYS_REGIME_NAMES[curRegime].slice(0, 12), lx + lw - 80, ly + 13);
  } else {
    ctx.fillStyle = '#9db4d4';
    ctx.fillText('PHASE PORTRAIT (q, qdot)', lx + 6, ly + 13);
    ctx.fillStyle = '#5c7394';
    ctx.fillText(curTid > 0 ? 'WARMING TRACE...' : 'NO PRIMARY TARGET', px0 + 8, pcy + 14);
  }

  // Right pane top: Energy Conservation E(t) & Ensemble K_ens(t)
  ctx.fillStyle = '#9db4d4';
  ctx.fillText('E(t) CYAN · K_ens AMBER', rx + 6, ry + 13);
  ctx.strokeStyle = '#182438';
  ctx.strokeRect(ex0 + 0.5, ey0 + 0.5, ew - 1, eh - 1);

  let maxE = 1.0;
  for (let i = 0; i < phaseHistCount; i++) {
    const idx = (phaseHistWrite + PHASE_HIST - phaseHistCount + i) % PHASE_HIST;
    if (phaseHistE[idx] > maxE) maxE = phaseHistE[idx];
    if (phaseHistEnsK[idx] > maxE) maxE = phaseHistEnsK[idx];
  }
  maxE *= 1.12;

  const denom = Math.max(1, phaseHistCount - 1);
  // Jump vertical markers
  ctx.fillStyle = 'rgba(255, 77, 109, 0.55)';
  for (let i = 0; i < phaseHistCount; i++) {
    const idx = (phaseHistWrite + PHASE_HIST - phaseHistCount + i) % PHASE_HIST;
    if (phaseHistJump[idx] > 0) {
      const gx = ex0 + 2 + (i / denom) * (ew - 4);
      ctx.fillRect(gx, ey0 + 1, 1.5, eh - 2);
    }
  }

  // K_ens curve
  ctx.strokeStyle = '#ffb703';
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  for (let i = 0; i < phaseHistCount; i++) {
    const idx = (phaseHistWrite + PHASE_HIST - phaseHistCount + i) % PHASE_HIST;
    const gx = ex0 + 2 + (i / denom) * (ew - 4);
    const gy = ey0 + eh - 3 - Math.min(1.0, phaseHistEnsK[idx] / maxE) * (eh - 6);
    if (i === 0) ctx.moveTo(gx, gy);
    else ctx.lineTo(gx, gy);
  }
  ctx.stroke();

  // Specific Energy E(t) curve
  ctx.strokeStyle = '#00f0ff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < phaseHistCount; i++) {
    const idx = (phaseHistWrite + PHASE_HIST - phaseHistCount + i) % PHASE_HIST;
    const gx = ex0 + 2 + (i / denom) * (ew - 4);
    const gy = ey0 + eh - 3 - Math.min(1.0, phaseHistE[idx] / maxE) * (eh - 6);
    if (i === 0) ctx.moveTo(gx, gy);
    else ctx.lineTo(gx, gy);
  }
  ctx.stroke();

  // Right pane bottom: Kuramoto R_sync [0..1]
  ctx.fillStyle = '#9db4d4';
  ctx.fillText(`KURAMOTO R=${curR.toFixed(2)} (N=${curN})`, rx + 6, ky0 - 5);
  ctx.strokeStyle = '#182438';
  ctx.strokeRect(kx0 + 0.5, ky0 + 0.5, kw - 1, kh - 1);
  ctx.strokeStyle = 'rgba(100, 128, 168, 0.25)';
  ctx.beginPath();
  ctx.moveTo(kx0, ky0 + kh * 0.5);
  ctx.lineTo(kx0 + kw, ky0 + kh * 0.5);
  ctx.stroke();

  ctx.strokeStyle = '#38f27a';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < phaseHistCount; i++) {
    const idx = (phaseHistWrite + PHASE_HIST - phaseHistCount + i) % PHASE_HIST;
    const gx = kx0 + 2 + (i / denom) * (kw - 4);
    const rClamp = Math.max(0.0, Math.min(1.0, phaseHistKurR[idx]));
    const gy = ky0 + kh - 2 - rClamp * (kh - 4);
    if (i === 0) ctx.moveTo(gx, gy);
    else ctx.lineTo(gx, gy);
  }
  ctx.stroke();
}

function buildPhysicsIdSnapshot(pIdx, hasPrimary, fxPx, fyPx, speedPhys, speedUnit, pMetricValid, angVelDegS) {
  const pb = hasPrimary ? (TELEM_PHYSICS_START + pIdx * TELEM_PHYSICS_STRIDE) : -1;
  const eb = TELEM_ENSEMBLE_START;

  const immWcv = hasPrimary ? safeFinite(telemView[pb + 0]) : 0.0;
  const immWbal = hasPrimary ? safeFinite(telemView[pb + 1]) : 0.0;
  const immWturnOsc = hasPrimary ? safeFinite(telemView[pb + 2]) : 0.0;
  const fistaSharpKnots = hasPrimary ? Math.max(0, Math.min(10, Math.round(safeFinite(telemView[pb + 3])))) : 0;
  const physRegime = hasPrimary ? Math.max(0, Math.min(8, Math.round(safeFinite(telemView[pb + 4])))) : 0;
  const physRegimeConf = hasPrimary ? safeFinite(telemView[pb + 5]) : 0.0;
  const odeR2 = hasPrimary ? safeFinite(telemView[pb + 6]) : 0.0;
  const aeroDragBeta = hasPrimary ? Math.max(0.0, safeFinite(telemView[pb + 7])) : 0.0;
  const thrustAccel = hasPrimary ? Math.max(0.0, safeFinite(telemView[pb + 8])) : 0.0;
  const ellipticPeriodS = hasPrimary ? Math.max(0.0, safeFinite(telemView[pb + 9])) : 0.0;
  const dampingZeta = hasPrimary ? safeFinite(telemView[pb + 10]) : 0.0;
  const qualityFactorQ = hasPrimary ? safeFinite(telemView[pb + 11]) : 0.0;
  const specificEnergy = hasPrimary ? Math.max(0.0, safeFinite(telemView[pb + 12])) : 0.0;
  const specificAngMom = hasPrimary ? safeFinite(telemView[pb + 13]) : 0.0;
  const normalRestitutionE = hasPrimary ? safeFinite(telemView[pb + 14]) : 0.0;
  const constraintJumpCode = hasPrimary ? Math.max(0, Math.min(4, Math.round(safeFinite(telemView[pb + 15])))) : 0;

  const kuramotoOrderR = safeFinite(telemView[eb + 0]);
  const kuramotoMeanPsi = safeFinite(telemView[eb + 1]);
  const kuramotoOscCount = Math.max(0, Math.round(safeFinite(telemView[eb + 2])));
  const ensembleKinetic = safeFinite(telemView[eb + 3]);
  const ensembleMechEnergy = safeFinite(telemView[eb + 4]);
  const lastPairMassRatio = safeFinite(telemView[eb + 5]);
  const lastPairMomResid = safeFinite(telemView[eb + 6]);
  const lastPairEnergyRetention = safeFinite(telemView[eb + 7]);
  const lastPairTidA = Math.round(safeFinite(telemView[eb + 8]));
  const lastPairTidB = Math.round(safeFinite(telemView[eb + 9]));
  const lastPairFrame = Math.round(safeFinite(telemView[eb + 10]));
  const lyapunovDivRate = Math.max(0.0, safeFinite(telemView[eb + 11]));
  const taubinVsKasaDeltaPx = Math.max(0.0, safeFinite(telemView[eb + 12]));
  const taubinCurvatureInvPx = Math.max(0.0, safeFinite(telemView[eb + 13]));

  const camSusp = isCameraMotionSuspendingCal();
  let unitMode = 'pixel';
  let pxPerMeter = 0.0;
  let localPxPerMeter = 0.0;
  if (camSusp) {
    unitMode = 'out_of_plane';
  } else if (calMode === 'homography' && calHomographyValid) {
    const hX = hasPrimary ? fxPx : frameW * 0.5;
    const hY = hasPrimary ? fyPx : frameH * 0.5;
    const lPpm = computeLocalHomographyPpm(hX, hY);
    if (lPpm > 0.5 && pMetricValid >= 0) {
      unitMode = 'metric';
      localPxPerMeter = lPpm;
    } else if (lPpm < 0 || pMetricValid < -0.5) {
      unitMode = 'out_of_plane';
    }
  } else {
    const ppm = getEffectivePxPerMeter();
    pxPerMeter = ppm;
    if (ppm > 0.5) {
      unitMode = 'metric';
      localPxPerMeter = ppm;
    }
  }

  const metricOk = unitMode === 'metric' && localPxPerMeter > 0.5;
  const ppm = metricOk ? localPxPerMeter : 1.0;
  const ppm2 = ppm * ppm;

  const betaDisp = metricOk ? aeroDragBeta * ppm : aeroDragBeta;
  const betaUnit = metricOk ? 'm^-1' : 'px^-1';
  const thrustDisp = metricOk ? thrustAccel / ppm : thrustAccel;
  const thrustUnit = metricOk ? 'm/s^2' : 'px/s^2';
  const energyDisp = metricOk ? specificEnergy / ppm2 : specificEnergy;
  const energyUnit = metricOk ? 'J/kg' : 'px^2/s^2';
  const angMomDisp = metricOk ? specificAngMom / ppm2 : specificAngMom;
  const angMomUnit = metricOk ? 'm^2/s' : 'px^2/s';
  const ensKineticDisp = metricOk ? ensembleKinetic / ppm2 : ensembleKinetic;
  const ensMechEnergyDisp = metricOk ? ensembleMechEnergy / ppm2 : ensembleMechEnergy;
  const taubinDeltaDisp = metricOk ? taubinVsKasaDeltaPx / ppm : taubinVsKasaDeltaPx;
  const taubinDeltaUnit = metricOk ? 'm' : 'px';
  const curvatureDisp = metricOk ? taubinCurvatureInvPx * ppm : taubinCurvatureInvPx;
  const curvatureUnit = metricOk ? 'm^-1' : 'px^-1';

  const sceneGravPx = safeFinite(telemView[TELEM_GLOBAL + 0]);
  const gStr = metricOk
    ? `${(calMode === 'auto_g' ? GRAVITY_M_S2 : (sceneGravPx > 50.0 ? sceneGravPx / ppm : GRAVITY_M_S2)).toFixed(2)} m/s^2`
    : (sceneGravPx > 10.0 ? `${sceneGravPx.toFixed(0)} px/s^2` : 'g');
  const speedStr = speedUnit === 'OUT OF PLANE'
    ? 'OOP'
    : (metricOk ? `${speedPhys.toFixed(2)} m/s` : `${speedPhys.toFixed(1)} px/s`);

  const symbolicOde = formatSymbolicOdeString({
    hasPrimary,
    physRegime,
    odeR2,
    betaDisp,
    betaUnit,
    thrustDisp,
    thrustUnit,
    curvatureDisp,
    curvatureUnit,
    dampingZeta,
    qualityFactorQ,
    ellipticPeriodS,
    normalRestitutionE,
    lastPairMassRatio,
    lastPairFrame,
    kuramotoOrderR,
    kuramotoOscCount,
    tremorFreqHz: safeFinite(telemView[13]),
    angVelDegS,
    speedStr,
    gStr,
  });

  return {
    hasPrimary,
    physRegime,
    physRegimeName: PHYS_REGIME_NAMES[physRegime] || 'INERTIAL CV',
    physRegimeConf,
    odeR2,
    symbolicOde,
    immWcv,
    immWbal,
    immWturnOsc,
    fistaSharpKnots,
    aeroDragBeta,
    thrustAccel,
    ellipticPeriodS,
    dampingZeta,
    qualityFactorQ,
    specificEnergy,
    specificAngMom,
    normalRestitutionE,
    constraintJumpCode,
    constraintJumpName: JUMP_CODE_NAMES[constraintJumpCode] || 'NONE',
    kuramotoOrderR,
    kuramotoMeanPsi,
    kuramotoOscCount,
    ensembleKinetic,
    ensembleMechEnergy,
    lastPairMassRatio,
    lastPairMomResid,
    lastPairEnergyRetention,
    lastPairTidA,
    lastPairTidB,
    lastPairFrame,
    lyapunovDivRate,
    taubinVsKasaDeltaPx,
    taubinCurvatureInvPx,
    pxPerMeter,
    localPxPerMeter,
    unitMode,
    display: {
      beta: betaDisp,
      betaUnit,
      thrust: thrustDisp,
      thrustUnit,
      energy: energyDisp,
      energyUnit,
      angMom: angMomDisp,
      angMomUnit,
      ensKinetic: ensKineticDisp,
      ensMechEnergy: ensMechEnergyDisp,
      taubinDelta: taubinDeltaDisp,
      taubinDeltaUnit,
      curvature: curvatureDisp,
      curvatureUnit,
    },
  };
}

// ============================================================================
// Core Per-Frame Execution Pipeline (Pre-Tick -> Lanczos Fiedler -> Post-Tick)
// ============================================================================

function computeMedianTickMs(count) {
  if (count <= 0) return 0.0;
  for (let i = 0; i < count; i++) {
    wasmTimeSortBuf[i] = wasmTimeHist[i];
  }
  for (let i = 1; i < count; i++) {
    const v = wasmTimeSortBuf[i];
    let j = i - 1;
    while (j >= 0 && wasmTimeSortBuf[j] > v) {
      wasmTimeSortBuf[j + 1] = wasmTimeSortBuf[j];
      j--;
    }
    wasmTimeSortBuf[j + 1] = v;
  }
  const mid = count >> 1;
  return (count & 1) ? wasmTimeSortBuf[mid] : 0.5 * (wasmTimeSortBuf[mid - 1] + wasmTimeSortBuf[mid]);
}

function orientedExtent(cw, ch, orientation) {
  return (orientation & 1) === 0 ? [cw, ch] : [ch, cw];
}

function applyOrientationTransform(ctx, cw, ch, orientation, dx, dy, dw, dh) {
  const [ow, oh] = orientedExtent(cw, ch, orientation);
  const sx = dw / Math.max(1, ow);
  const sy = dh / Math.max(1, oh);
  const o = orientation & 7;
  let m11 = 1, m12 = 0, m21 = 0, m22 = 1, ox = 0, oy = 0;
  switch (o) {
    case 0: m11 = 1;  m12 = 0;  m21 = 0;  m22 = 1;  ox = 0;  oy = 0;  break;
    case 1: m11 = 0;  m12 = 1;  m21 = -1; m22 = 0;  ox = ch; oy = 0;  break;
    case 2: m11 = -1; m12 = 0;  m21 = 0;  m22 = -1; ox = cw; oy = ch; break;
    case 3: m11 = 0;  m12 = -1; m21 = 1;  m22 = 0;  ox = 0;  oy = cw; break;
    case 4: m11 = -1; m12 = 0;  m21 = 0;  m22 = 1;  ox = cw; oy = 0;  break;
    case 5: m11 = 0;  m12 = 1;  m21 = 1;  m22 = 0;  ox = 0;  oy = 0;  break;
    case 6: m11 = 1;  m12 = 0;  m21 = 0;  m22 = -1; ox = 0;  oy = ch; break;
    case 7: m11 = 0;  m12 = -1; m21 = -1; m22 = 0;  ox = ch; oy = cw; break;
  }
  ctx.setTransform(m11 * sx, m12 * sy, m21 * sx, m22 * sy, dx + ox * sx, dy + oy * sy);
}

function unrotateBoxFromOriented(dispW, dispH, dispX0, dispY0, dispCw, dispCh, orientation) {
  const o = orientation & 7;
  const [rawSrcW, rawSrcH] = orientedExtent(dispW, dispH, o);
  const [rawCw, rawCh] = orientedExtent(dispCw, dispCh, o);
  const xFlip = (o & 4) !== 0;
  const rotK = o & 3;
  const rxMin = xFlip ? (dispW - dispX0 - dispCw) : dispX0;
  const ryMin = dispY0;
  let rawX0 = 0;
  let rawY0 = 0;
  if (rotK === 0) {
    rawX0 = rxMin;
    rawY0 = ryMin;
  } else if (rotK === 1) {
    rawX0 = ryMin;
    rawY0 = rawSrcH - rxMin - dispCw;
  } else if (rotK === 2) {
    rawX0 = rawSrcW - rxMin - dispCw;
    rawY0 = rawSrcH - ryMin - dispCh;
  } else {
    rawX0 = rawSrcW - ryMin - dispCh;
    rawY0 = rxMin;
  }
  return {
    srcW: rawSrcW,
    srcH: rawSrcH,
    cropX0: Math.max(0, Math.min(rawSrcW - rawCw, rawX0)),
    cropY0: Math.max(0, Math.min(rawSrcH - rawCh, rawY0)),
    cropW: rawCw,
    cropH: rawCh,
  };
}

function currentEffectiveRoiProjection() {
  if (sourceMode === 'testbench') {
    return {
      sx0: 0,
      sy0: 0,
      sw: 1,
      sh: 1,
      orientation: 0,
      dx0: 0,
      dy0: 0,
      dw: 1,
      dh: 1,
    };
  }
  const srcW = Math.max(1, lastIngestGeom.srcW || frameW || 1);
  const srcH = Math.max(1, lastIngestGeom.srcH || frameH || 1);
  const g = resolveIngestGeomSpec(srcW, srcH, null);
  const gSrcW = Math.max(1, g.srcW || srcW);
  const gSrcH = Math.max(1, g.srcH || srcH);
  const gEngW = Math.max(1, g.engineW || frameW || 1);
  const gEngH = Math.max(1, g.engineH || frameH || 1);
  return {
    sx0: g.cropX0 / gSrcW,
    sy0: g.cropY0 / gSrcH,
    sw: g.cropW / gSrcW,
    sh: g.cropH / gSrcH,
    orientation: g.orientation & 7,
    dx0: g.dstX / gEngW,
    dy0: g.dstY / gEngH,
    dw: g.dstW / gEngW,
    dh: g.dstH / gEngH,
  };
}

function roiProjectionChanged(a, b) {
  if (!a || !b) return false;
  return (
    a.orientation !== b.orientation ||
    Math.abs(a.sx0 - b.sx0) > 1e-6 ||
    Math.abs(a.sy0 - b.sy0) > 1e-6 ||
    Math.abs(a.sw - b.sw) > 1e-6 ||
    Math.abs(a.sh - b.sh) > 1e-6 ||
    Math.abs(a.dx0 - b.dx0) > 1e-6 ||
    Math.abs(a.dy0 - b.dy0) > 1e-6 ||
    Math.abs(a.dw - b.dw) > 1e-6 ||
    Math.abs(a.dh - b.dh) > 1e-6
  );
}

function resolveIngestGeomSpec(bw, bh, infoGeom = null) {
  const g = infoGeom || {};
  const orientation =
    typeof g.orientation === 'number' && Number.isInteger(g.orientation) && g.orientation >= 0 && g.orientation <= 7
      ? g.orientation
      : (roiOrientation & 7);
  const scanDir =
    typeof g.scanDir === 'number' && Number.isInteger(g.scanDir) && (g.scanDir === 0xffffffff || (g.scanDir >= 0 && g.scanDir <= 3))
      ? (g.scanDir >>> 0)
      : (roiScanDir >>> 0);
  const alreadyOriented = Boolean(g.alreadyOriented);
  const drawOrient = alreadyOriented ? 0 : orientation;
  const reportedOrient = orientation;
  const preScaled = Boolean(g.preScaled);

  const vr = g.visibleRect && typeof g.visibleRect === 'object' ? g.visibleRect : null;
  const visX0 = vr ? Math.max(0, Math.min(bw - 1, Math.trunc(vr.x || 0))) : 0;
  const visY0 = vr ? Math.max(0, Math.min(bh - 1, Math.trunc(vr.y || 0))) : 0;
  const visW = vr ? Math.max(1, Math.min(bw - visX0, Math.trunc(vr.width || (bw - visX0)))) : bw;
  const visH = vr ? Math.max(1, Math.min(bh - visY0, Math.trunc(vr.height || (bh - visY0)))) : bh;

  const [reqDrawCw, reqDrawCh] = alreadyOriented ? [frameW, frameH] : orientedExtent(frameW, frameH, reportedOrient);
  const cropFitsReq =
    (typeof g.cropW !== 'number' || g.cropW === reqDrawCw) &&
    (typeof g.cropH !== 'number' || g.cropH === reqDrawCh);

  if (nativeRoiEnabled && !preScaled && cropFitsReq && visW >= reqDrawCw && visH >= reqDrawCh) {
    const maxX0 = visW - reqDrawCw;
    const maxY0 = visH - reqDrawCh;
    const hasExplicitCrop =
      typeof g.cropX0 === 'number' && Number.isInteger(g.cropX0) &&
      typeof g.cropY0 === 'number' && Number.isInteger(g.cropY0) &&
      (typeof g.cropW !== 'number' || g.cropW === reqDrawCw) &&
      (typeof g.cropH !== 'number' || g.cropH === reqDrawCh);
    const relX0 = hasExplicitCrop
      ? Math.max(0, Math.min(maxX0, g.cropX0))
      : roiUseExplicitPx
      ? Math.max(0, Math.min(maxX0, Math.round(roiExplicitX0)))
      : Math.max(0, Math.min(maxX0, Math.round(Math.max(0.0, Math.min(1.0, roiOffsetNormX)) * maxX0)));
    const relY0 = hasExplicitCrop
      ? Math.max(0, Math.min(maxY0, g.cropY0))
      : roiUseExplicitPx
      ? Math.max(0, Math.min(maxY0, Math.round(roiExplicitY0)))
      : Math.max(0, Math.min(maxY0, Math.round(Math.max(0.0, Math.min(1.0, roiOffsetNormY)) * maxY0)));
    const metaBox = alreadyOriented
      ? unrotateBoxFromOriented(visW, visH, relX0, relY0, reqDrawCw, reqDrawCh, reportedOrient)
      : { srcW: visW, srcH: visH, cropX0: relX0, cropY0: relY0, cropW: reqDrawCw, cropH: reqDrawCh };
    return {
      srcW: metaBox.srcW,
      srcH: metaBox.srcH,
      cropX0: metaBox.cropX0,
      cropY0: metaBox.cropY0,
      codedCropX0: visX0 + relX0,
      codedCropY0: visY0 + relY0,
      cropW: metaBox.cropW,
      cropH: metaBox.cropH,
      drawCropW: reqDrawCw,
      drawCropH: reqDrawCh,
      orientation: reportedOrient,
      drawOrient,
      scanDir,
      engineW: frameW,
      engineH: frameH,
      dstX: 0,
      dstY: 0,
      dstW: frameW,
      dstH: frameH,
      native1to1: 1,
      fallbackScale: false,
      allowSmoothing: false,
      fullFrameFit: false,
    };
  }

  let relX0 = 0;
  let relY0 = 0;
  let sw = visW;
  let sh = visH;
  if (
    typeof g.cropX0 === 'number' && Number.isInteger(g.cropX0) &&
    typeof g.cropY0 === 'number' && Number.isInteger(g.cropY0) &&
    typeof g.cropW === 'number' && Number.isInteger(g.cropW) && g.cropW >= 1 &&
    typeof g.cropH === 'number' && Number.isInteger(g.cropH) && g.cropH >= 1
  ) {
    relX0 = Math.max(0, Math.min(visW - 1, g.cropX0));
    relY0 = Math.max(0, Math.min(visH - 1, g.cropY0));
    sw = Math.max(1, Math.min(visW - relX0, g.cropW));
    sh = Math.max(1, Math.min(visH - relY0, g.cropH));
  } else {
    const cx = Math.max(0.0, Math.min(0.98, cropNorm[0]));
    const cy = Math.max(0.0, Math.min(0.98, cropNorm[1]));
    const cwNorm = Math.max(0.02, Math.min(1.0 - cx, cropNorm[2]));
    const chNorm = Math.max(0.02, Math.min(1.0 - cy, cropNorm[3]));
    relX0 = Math.max(0, Math.min(visW - 1, Math.round(cx * visW)));
    relY0 = Math.max(0, Math.min(visH - 1, Math.round(cy * visH)));
    sw = Math.max(1, Math.min(visW - relX0, Math.round(cwNorm * visW)));
    sh = Math.max(1, Math.min(visH - relY0, Math.round(chNorm * visH)));
  }
  const codedCropX0 = visX0 + relX0;
  const codedCropY0 = visY0 + relY0;

  const [ow, oh] = orientedExtent(sw, sh, drawOrient);
  const srcAspect = ow / oh;
  const dstAspect = frameW / frameH;
  const fullVis = relX0 === 0 && relY0 === 0 && sw === visW && sh === visH;
  const failedNative = Boolean(nativeRoiEnabled);
  let dx = 0;
  let dy = 0;
  let dw = frameW;
  let dh = frameH;
  if (fullVis && Math.abs(srcAspect - dstAspect) / dstAspect <= 0.02) {
    dx = 0;
    dy = 0;
    dw = frameW;
    dh = frameH;
  } else {
    if (srcAspect > dstAspect) {
      dh = Math.max(1, Math.round(frameW / srcAspect));
      dy = (frameH - dh) >> 1;
    } else {
      dw = Math.max(1, Math.round(frameH * srcAspect));
      dx = (frameW - dw) >> 1;
    }
  }
  const exact1to1Unflagged = !preScaled && ow === dw && oh === dh;
  const metaBox = alreadyOriented
    ? unrotateBoxFromOriented(visW, visH, relX0, relY0, sw, sh, reportedOrient)
    : { srcW: visW, srcH: visH, cropX0: relX0, cropY0: relY0, cropW: sw, cropH: sh };
  return {
    srcW: metaBox.srcW,
    srcH: metaBox.srcH,
    cropX0: metaBox.cropX0,
    cropY0: metaBox.cropY0,
    codedCropX0,
    codedCropY0,
    cropW: metaBox.cropW,
    cropH: metaBox.cropH,
    drawCropW: sw,
    drawCropH: sh,
    orientation: reportedOrient,
    drawOrient,
    scanDir,
    engineW: frameW,
    engineH: frameH,
    dstX: dx,
    dstY: dy,
    dstW: dw,
    dstH: dh,
    native1to1: 0,
    fallbackScale: failedNative,
    allowSmoothing: !failedNative && !exact1to1Unflagged,
    fullFrameFit: dx === 0 && dy === 0 && dw === frameW && dh === frameH,
  };
}

function drawCroppedLetterboxedFrame(bmp, infoGeom = null) {
  const bw = Math.max(1, Math.trunc(bmp.width || bmp.displayWidth || bmp.codedWidth || frameW));
  const bh = Math.max(1, Math.trunc(bmp.height || bmp.displayHeight || bmp.codedHeight || frameH));
  const geom = resolveIngestGeomSpec(bw, bh, infoGeom);
  const drawCw = typeof geom.drawCropW === 'number' ? geom.drawCropW : geom.cropW;
  const drawCh = typeof geom.drawCropH === 'number' ? geom.drawCropH : geom.cropH;
  simCtx.save();
  simCtx.setTransform(1, 0, 0, 1, 0, 0);
  if (!geom.fullFrameFit) {
    simCtx.fillStyle = '#000000';
    simCtx.fillRect(0, 0, frameW, frameH);
  }
  simCtx.imageSmoothingEnabled = geom.allowSmoothing;
  applyOrientationTransform(
    simCtx,
    drawCw,
    drawCh,
    geom.drawOrient,
    geom.dstX,
    geom.dstY,
    geom.dstW,
    geom.dstH
  );
  const srcReadX = typeof geom.codedCropX0 === 'number' ? geom.codedCropX0 : geom.cropX0;
  const srcReadY = typeof geom.codedCropY0 === 'number' ? geom.codedCropY0 : geom.cropY0;
  simCtx.drawImage(
    bmp,
    srcReadX,
    srcReadY,
    drawCw,
    drawCh,
    0,
    0,
    drawCw,
    drawCh
  );
  simCtx.restore();
  lastIngestGeom = geom;
  return geom;
}

function extractAndOrientRgba(srcRgba, bw, bh, infoGeom = null) {
  const geom = resolveIngestGeomSpec(bw, bh, infoGeom);
  const out = rgbaStageScratch.subarray(0, frameBytes);
  const fullRect = geom.dstX === 0 && geom.dstY === 0 && geom.dstW === frameW && geom.dstH === frameH;
  if (!fullRect) {
    for (let i = 0; i < frameBytes; i += 4) {
      out[i] = 0;
      out[i + 1] = 0;
      out[i + 2] = 0;
      out[i + 3] = 255;
    }
  }
  const cw = typeof geom.drawCropW === 'number' ? geom.drawCropW : geom.cropW;
  const ch = typeof geom.drawCropH === 'number' ? geom.drawCropH : geom.cropH;
  const [ow, oh] = orientedExtent(cw, ch, geom.drawOrient);
  const orient = geom.drawOrient & 7;
  const xFlip = (orient & 4) !== 0;
  const rotK = orient & 3;
  const sxBase = typeof geom.codedCropX0 === 'number' ? geom.codedCropX0 : geom.cropX0;
  const syBase = typeof geom.codedCropY0 === 'number' ? geom.codedCropY0 : geom.cropY0;
  const dw = geom.dstW;
  const dh = geom.dstH;
  const dx = geom.dstX;
  const dy = geom.dstY;
  const exact1to1 = dw === ow && dh === oh;

  for (let py = 0; py < dh; py++) {
    const oy = exact1to1
      ? py
      : (geom.allowSmoothing
          ? Math.max(0, Math.min(oh - 1, ((py + 0.5) * oh) / dh - 0.5))
          : Math.max(0, Math.min(oh - 1, Math.floor(((py + 0.5) * oh) / dh))));
    const dstRow = (dy + py) * frameW + dx;
    if (exact1to1 || !geom.allowSmoothing) {
      const oyi = oy | 0;
      for (let px = 0; px < dw; px++) {
        const oxi = exact1to1
          ? px
          : Math.max(0, Math.min(ow - 1, Math.floor(((px + 0.5) * ow) / dw)));
        const rx = xFlip ? (ow - 1 - oxi) : oxi;
        const ry = oyi;
        let u = 0;
        let v = 0;
        if (rotK === 0) {
          u = rx;
          v = ry;
        } else if (rotK === 1) {
          u = ry;
          v = ch - 1 - rx;
        } else if (rotK === 2) {
          u = cw - 1 - rx;
          v = ch - 1 - ry;
        } else {
          u = cw - 1 - ry;
          v = rx;
        }
        const sOff = ((syBase + v) * bw + (sxBase + u)) * 4;
        const dOff = (dstRow + px) * 4;
        out[dOff] = srcRgba[sOff];
        out[dOff + 1] = srcRgba[sOff + 1];
        out[dOff + 2] = srcRgba[sOff + 2];
        out[dOff + 3] = srcRgba[sOff + 3];
      }
    } else {
      const y0 = Math.floor(oy);
      const y1 = Math.min(oh - 1, y0 + 1);
      const fy = oy - y0;
      for (let px = 0; px < dw; px++) {
        const ox = Math.max(0, Math.min(ow - 1, ((px + 0.5) * ow) / dw - 0.5));
        const x0 = Math.floor(ox);
        const x1 = Math.min(ow - 1, x0 + 1);
        const fx = ox - x0;
        const sampleAt = (sx0, sy0) => {
          const rx = xFlip ? (ow - 1 - sx0) : sx0;
          const ry = sy0;
          let u = 0, v = 0;
          if (rotK === 0) { u = rx; v = ry; }
          else if (rotK === 1) { u = ry; v = ch - 1 - rx; }
          else if (rotK === 2) { u = cw - 1 - rx; v = ch - 1 - ry; }
          else { u = cw - 1 - ry; v = rx; }
          return ((syBase + v) * bw + (sxBase + u)) * 4;
        };
        const o00 = sampleAt(x0, y0);
        const o10 = sampleAt(x1, y0);
        const o01 = sampleAt(x0, y1);
        const o11 = sampleAt(x1, y1);
        const w00 = (1 - fx) * (1 - fy);
        const w10 = fx * (1 - fy);
        const w01 = (1 - fx) * fy;
        const w11 = fx * fy;
        const dOff = (dstRow + px) * 4;
        out[dOff] = Math.round(srcRgba[o00] * w00 + srcRgba[o10] * w10 + srcRgba[o01] * w01 + srcRgba[o11] * w11);
        out[dOff + 1] = Math.round(srcRgba[o00 + 1] * w00 + srcRgba[o10 + 1] * w10 + srcRgba[o01 + 1] * w01 + srcRgba[o11 + 1] * w11);
        out[dOff + 2] = Math.round(srcRgba[o00 + 2] * w00 + srcRgba[o10 + 2] * w10 + srcRgba[o01 + 2] * w01 + srcRgba[o11 + 2] * w11);
        out[dOff + 3] = Math.round(srcRgba[o00 + 3] * w00 + srcRgba[o10 + 3] * w10 + srcRgba[o01 + 3] * w01 + srcRgba[o11 + 3] * w11);
      }
    }
  }
  lastIngestGeom = geom;
  return { pixels: out, geom };
}

function finiteOrNull(v) {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function intOrNull(v, min = 0, max = 16777215) {
  if (typeof v !== 'number' || !Number.isFinite(v) || !Number.isInteger(v)) return null;
  return v >= min && v <= max ? v : null;
}

function parseRigidBodyDescriptor(buf, bo) {
  if (!buf) return null;
  const bodyId = intOrNull(buf[bo], 1, MAX_RIGID_BODIES);
  const memberCount = intOrNull(buf[bo + 1], 1, MAX_MICRO);
  const cx = finiteOrNull(buf[bo + 2]);
  const cy = finiteOrNull(buf[bo + 3]);
  const vx = finiteOrNull(buf[bo + 4]);
  const vy = finiteOrNull(buf[bo + 5]);
  const omegaRadS = finiteOrNull(buf[bo + 6]);
  if (
    bodyId === null ||
    memberCount === null ||
    cx === null ||
    cy === null ||
    vx === null ||
    vy === null ||
    omegaRadS === null
  ) {
    return null;
  }
  return {
    bodyId,
    memberCount,
    cx,
    cy,
    vx,
    vy,
    omegaRadS,
    se2RmsePx: finiteOrNull(buf[bo + 7]),
  };
}

function computeLookaheadTiming(effK, dispSlot, fIdxNow) {
  const k = intOrNull(effK, 0, RING_LEN - 1) ?? (Number.isFinite(effK) ? Math.max(0, Math.round(effK)) : 0);
  const curF = fIdxNow >>> 0;
  if (k === 0) {
    return {
      lookaheadWindowMs: 0.0,
      lookaheadSource: 'zero_lookahead',
      displayedFrameIdx: curF,
      causalFrameIdx: curF,
      renderedDelayFrames: 0,
    };
  }
  const targetDispFrame = (curF - k) >>> 0;
  const validDispSlot = dispSlot >= 0 && dispSlot < RING_LEN;
  const slotMetaOff = validDispSlot ? dispSlot * RING_META_STRIDE : -1;
  const slotEngineFrame = validDispSlot && ringMetaView
    ? (ringMetaView[slotMetaOff + 3] >>> 0)
    : (validDispSlot && ringIngestFrameValid[dispSlot] === 1 ? (ringIngestFrameId[dispSlot] >>> 0) : -1);
  const dispCommitted = validDispSlot && (
    ringMetaView
      ? ((ringMetaView[slotMetaOff] & 1) !== 0)
      : ringIngestFrameValid[dispSlot] === 1
  );
  const sameEpoch = !wave1CoreAbiValid || !wave1TelemView || !ringMetaView || slotMetaOff < 0
    ? true
    : ((ringMetaView[slotMetaOff + 23] >>> 0) === (wave1TelemView[6] >>> 0));
  const ringMatch = validDispSlot && dispCommitted && sameEpoch && slotEngineFrame === targetDispFrame;
  if (
    ringMatch &&
    latestCausalPtsMs >= 0.0 &&
    ringIngestPtsMs[dispSlot] >= 0.0 &&
    latestCausalPtsMs >= ringIngestPtsMs[dispSlot]
  ) {
    return {
      lookaheadWindowMs: Math.max(0.0, latestCausalPtsMs - ringIngestPtsMs[dispSlot]),
      lookaheadSource: 'source_pts',
      displayedFrameIdx: targetDispFrame,
      causalFrameIdx: curF,
      renderedDelayFrames: k,
    };
  }
  if (
    ringMatch &&
    ringIngestTimeMs[dispSlot] >= 0.0 &&
    latestCausalTimeMs >= ringIngestTimeMs[dispSlot]
  ) {
    return {
      lookaheadWindowMs: Math.max(0.0, latestCausalTimeMs - ringIngestTimeMs[dispSlot]),
      lookaheadSource: 'accumulated_dt',
      displayedFrameIdx: targetDispFrame,
      causalFrameIdx: curF,
      renderedDelayFrames: k,
    };
  }
  const dtSec = lastDtSecUsed > 0.001 && lastDtSecUsed < 0.25 ? lastDtSecUsed : (1.0 / 30.0);
  return {
    lookaheadWindowMs: k * dtSec * 1000.0,
    lookaheadSource: 'dt_estimate',
    displayedFrameIdx: targetDispFrame,
    causalFrameIdx: curF,
    renderedDelayFrames: k,
  };
}

function buildChronos1000Snapshot(medianMs, effK, fIdxNow = 0, dispSlot = -1) {
  const dtSec = lastDtSecUsed > 0.001 && lastDtSecUsed < 0.25 ? lastDtSecUsed : (1.0 / 30.0);
  const timing = computeLookaheadTiming(effK, dispSlot, fIdxNow);
  const lookaheadWindowMs = timing.lookaheadWindowMs;
  const partialPipelineLatencyMs = lookaheadWindowMs + medianMs + lastSwarmCopyMs + lastFieldRenderMs;
  const st = swarmTelemView;
  const avail = Boolean(st && hasChronos1000Exports());
  const stInspTid = avail ? intOrNull(st[11], 0, 16777215) : null;
  const inspectExplicitClear =
    lastInspectStatus === 'expired' ||
    lastInspectStatus === 'miss' ||
    lastInspectStatus === 'unavailable';
  const activeInspTid = inspectExplicitClear
    ? 0
    : (inspectedMicroTid > 0 ? inspectedMicroTid : (stInspTid !== null && stInspTid > 0 ? stInspTid : 0));
  let inspectedTrack = null;
  if (avail && activeInspTid > 0 && microTracksView) {
    for (let i = 0; i < MAX_MICRO; i++) {
      const b = i * MICRO_STRIDE;
      const trackSt = intOrNull(microTracksView[b + 5], 1, 2);
      const trackTid = intOrNull(microTracksView[b], 1, 16777215);
      if (trackSt !== null && trackTid === activeInspTid) {
        const rawTrackFtle = finiteOrNull(microTracksView[b + 7]);
        inspectedTrack = {
          tid: activeInspTid,
          x: finiteOrNull(microTracksView[b + 1]),
          y: finiteOrNull(microTracksView[b + 2]),
          vx: finiteOrNull(microTracksView[b + 3]),
          vy: finiteOrNull(microTracksView[b + 4]),
          state: trackSt,
          bodyId: intOrNull(microTracksView[b + 6], 0, MAX_RIGID_BODIES),
          ftle: rawTrackFtle !== null && rawTrackFtle > -900.0 ? rawTrackFtle : null,
        };
        break;
      }
    }
  }

  const swarmActiveFlag = avail ? intOrNull(st[1], 0, 1) : null;
  const validityMaskVal = avail ? intOrNull(st[16], 0, 31) : null;
  const rigidBodyCountVal = avail ? intOrNull(st[23], 0, MAX_RIGID_BODIES) : 0;
  const topRigidMembersVal = avail ? intOrNull(st[25], 0, MAX_MICRO) : 0;
  const topRigidOmegaVal = avail ? finiteOrNull(st[24]) : 0.0;
  let validRigidBodyCount = 0;
  if (avail && rigidBodiesView && validityMaskVal !== null && (validityMaskVal & 4) !== 0) {
    for (let b = 0; b < MAX_RIGID_BODIES; b++) {
      if (parseRigidBodyDescriptor(rigidBodiesView, b * BODY_STRIDE)) {
        validRigidBodyCount++;
      }
    }
  }
  const topRigidBodyValid = Boolean(
    avail &&
    validityMaskVal !== null &&
    (validityMaskVal & 4) !== 0 &&
    validRigidBodyCount > 0 &&
    rigidBodyCountVal !== null &&
    rigidBodyCountVal > 0 &&
    topRigidMembersVal !== null &&
    topRigidMembersVal >= 1 &&
    topRigidOmegaVal !== null
  );

  const out = {
    available: avail,
    diagnosticsValid: false,
    delayFrames: 0,
    causalFrameIdx: timing.causalFrameIdx,
    displayedFrameIdx: timing.displayedFrameIdx,
    renderedDelayFrames: timing.renderedDelayFrames,
    inspectStatus: lastInspectStatus,
    inspectSeq: lastInspectResSeq,
    inspectReqSeq: lastInspectReqSeq,
    inspectResSeq: lastInspectResSeq,
    inspectReqPxX: lastInspectReqX,
    inspectReqPxY: lastInspectReqY,
    inspectReqX: lastInspectReqX,
    inspectReqY: lastInspectReqY,
    inspectReqNormX: lastInspectReqNormX,
    inspectReqNormY: lastInspectReqNormY,
    swarmMode: avail ? intOrNull(st[0], 0, 2) : swarmMode,
    userSwarmMode,
    denseSwarmOverrideActive,
    swarmActive: avail ? swarmActiveFlag === 1 : false,
    validRigidBodyCount,
    topRigidBodyValid,
    fieldOverlayMode,
    fieldBackend: lastFieldBackend,
    fieldRenderMs: lastFieldRenderMs,
    snapshotCopyMs: lastSwarmCopyMs,
    peakCount: avail ? intOrNull(st[2], 0, 2048) : 0,
    activeCount: avail ? intOrNull(st[3], 0, MAX_MICRO) : 0,
    confirmedCount: avail ? intOrNull(st[4], 0, MAX_MICRO) : 0,
    peakSimultaneous: avail ? intOrNull(st[5], 0, MAX_MICRO) : 0,
    uniqueTracks: avail ? intOrNull(st[6], 0, 16777215) : 0,
    coastingCount: avail ? intOrNull(st[7], 0, MAX_MICRO) : 0,
    auctionSteps: avail ? intOrNull(st[8], 0, 16777215) : 0,
    reacqEvents: avail ? intOrNull(st[9], 0, 16777215) : 0,
    meanIcLkRes: avail ? finiteOrNull(st[10]) : 0.0,
    inspectedTid: activeInspTid,
    inspectedBodyId: avail ? intOrNull(st[12], 0, MAX_RIGID_BODIES) : 0,
    births: avail ? intOrNull(st[13], 0, MAX_MICRO) : 0,
    tentativeCount: avail ? intOrNull(st[14], 0, MAX_MICRO) : 0,
    ambiguousCount: avail ? intOrNull(st[15], 0, MAX_MICRO) : 0,
    validityMask: avail ? intOrNull(st[16], 0, 31) : 0,
    validCells: avail ? intOrNull(st[17], 0, FIELD_CELLS) : 0,
    maxVorticityS1: avail ? finiteOrNull(st[18]) : 0.0,
    rmsVorticityS1: avail ? finiteOrNull(st[19]) : 0.0,
    rmsDivergenceS1: avail ? finiteOrNull(st[20]) : 0.0,
    maxFtleS1: avail ? finiteOrNull(st[21]) : 0.0,
    ftleCells: avail ? intOrNull(st[22], 0, FIELD_CELLS) : 0,
    rigidBodyCount: avail ? intOrNull(st[23], 0, MAX_RIGID_BODIES) : 0,
    topRigidOmegaRadS: avail ? finiteOrNull(st[24]) : 0.0,
    topRigidMembers: avail ? intOrNull(st[25], 0, MAX_MICRO) : 0,
    topRigidSe2RmsePx: avail ? finiteOrNull(st[26]) : 0.0,
    modalDomFreqHz: avail ? finiteOrNull(st[27]) : 0.0,
    modalCoherence: avail ? finiteOrNull(st[28]) : 0.0,
    modalValidSensors: avail ? intOrNull(st[29], 0, MAX_MICRO) : 0,
    kuramotoR: avail ? finiteOrNull(st[30]) : 0.0,
    kuramotoPsi: avail ? finiteOrNull(st[31]) : 0.0,
    comX: avail ? finiteOrNull(st[32]) : 0.0,
    comY: avail ? finiteOrNull(st[33]) : 0.0,
    comVx: avail ? finiteOrNull(st[34]) : 0.0,
    comVy: avail ? finiteOrNull(st[35]) : 0.0,
    velDispersion: avail ? finiteOrNull(st[36]) : 0.0,
    meanStrain: avail ? finiteOrNull(st[37]) : 0.0,
    knnEdgeCount: avail ? intOrNull(st[38], 0, MAX_MICRO * 6) : 0,
    ftleMatureTracks: avail ? intOrNull(st[39], 0, MAX_MICRO) : 0,
    topRigidThetaRad: avail ? finiteOrNull(st[40]) : 0.0,
    modalR2: avail ? finiteOrNull(st[41]) : 0.0,
    kernelComputeMedianMs: medianMs,
    lookaheadWindowMs,
    lookaheadSource: timing.lookaheadSource,
    estimatedPartialPipelineLatencyMs: partialPipelineLatencyMs,
    partialPipelineLatencyMs,
    dtSec,
    inspectedTrack,
  };

  out.diagnosticsValid = Boolean(
    avail &&
    out.swarmMode !== null &&
    swarmActiveFlag !== null &&
    out.peakCount !== null &&
    out.activeCount !== null &&
    out.confirmedCount !== null &&
    out.peakSimultaneous !== null &&
    out.uniqueTracks !== null &&
    out.coastingCount !== null &&
    out.auctionSteps !== null &&
    out.reacqEvents !== null &&
    out.meanIcLkRes !== null &&
    stInspTid !== null &&
    out.inspectedBodyId !== null &&
    out.births !== null &&
    out.tentativeCount !== null &&
    out.ambiguousCount !== null &&
    out.validityMask !== null &&
    out.validCells !== null &&
    out.maxVorticityS1 !== null &&
    out.rmsVorticityS1 !== null &&
    out.rmsDivergenceS1 !== null &&
    out.maxFtleS1 !== null &&
    out.ftleCells !== null &&
    out.rigidBodyCount !== null &&
    out.topRigidOmegaRadS !== null &&
    out.topRigidMembers !== null &&
    out.topRigidSe2RmsePx !== null &&
    out.modalDomFreqHz !== null &&
    out.modalCoherence !== null &&
    out.modalValidSensors !== null &&
    out.kuramotoR !== null &&
    out.kuramotoPsi !== null &&
    out.comX !== null &&
    out.comY !== null &&
    out.comVx !== null &&
    out.comVy !== null &&
    out.velDispersion !== null &&
    out.meanStrain !== null &&
    out.knnEdgeCount !== null &&
    out.ftleMatureTracks !== null &&
    out.topRigidThetaRad !== null &&
    out.modalR2 !== null
  );
  return out;
}

function decodeStampFrom(u32View, f32View, base) {
  if (!u32View || !f32View) return null;
  const validity = u32View[base] >>> 0;
  const tLo = u32View[base + 4] >>> 0;
  const tHi = u32View[base + 5] >>> 0;
  const ageRaw = f32View[base + 6];
  return {
    validity,
    valid: (validity & 1) !== 0,
    held: (validity & 2) !== 0,
    ptsKnown: (validity & 4) !== 0,
    captureAge: (validity & 8) !== 0,
    handoffAge: (validity & 16) !== 0,
    continuityEpoch: u32View[base + 1] >>> 0,
    sourceEpoch: u32View[base + 2] >>> 0,
    sourceSeq: u32View[base + 3] >>> 0,
    ptsUs: tHi * 4294967296 + tLo,
    ageMs: Number.isFinite(ageRaw) ? ageRaw : -1.0,
    engineFrameIdx: u32View[base + 7] >>> 0,
  };
}

function decodeStamp(base) {
  return decodeStampFrom(wave1TelemView, wave1TelemF32View, base);
}

function decodeWave2Stamp(base) {
  return decodeStampFrom(wave2TelemView, wave2TelemF32View, base);
}

function readRingMetaSlot(slot) {
  if (!ringMetaView || !ringMetaF32View || slot < 0 || slot >= RING_LEN) return null;
  const o = slot * RING_META_STRIDE;
  const flags = ringMetaView[o] >>> 0;
  if ((flags & 1) === 0) return null;
  const tLo = ringMetaView[o + 4] >>> 0;
  const tHi = ringMetaView[o + 5] >>> 0;
  const ageRaw = ringMetaF32View[o + 6];
  const gapFlag = ringMetaView[o + 21] >>> 0;
  const gaps = [];
  for (let b = 0; b < GAP_FLAG_NAMES.length; b++) {
    if (gapFlag & (1 << b)) gaps.push(GAP_FLAG_NAMES[b]);
  }
  return {
    flags,
    committed: true,
    ptsKnown: (flags & 2) !== 0,
    captureAge: (flags & 4) !== 0,
    handoffAge: (flags & 8) !== 0,
    geomKnown: (flags & 16) !== 0,
    sourceEpoch: ringMetaView[o + 1] >>> 0,
    sourceSeq: ringMetaView[o + 2] >>> 0,
    engineFrameIdx: ringMetaView[o + 3] >>> 0,
    tSrcUs: tHi * 4294967296 + tLo,
    ageMs: Number.isFinite(ageRaw) ? ageRaw : -1.0,
    srcW: ringMetaView[o + 7] >>> 0,
    srcH: ringMetaView[o + 8] >>> 0,
    cropX0: ringMetaView[o + 9] >>> 0,
    cropY0: ringMetaView[o + 10] >>> 0,
    cropW: ringMetaView[o + 11] >>> 0,
    cropH: ringMetaView[o + 12] >>> 0,
    orientation: ringMetaView[o + 13] >>> 0,
    scanDir: ringMetaView[o + 14] >>> 0,
    engineW: ringMetaView[o + 15] >>> 0,
    engineH: ringMetaView[o + 16] >>> 0,
    dstX: ringMetaView[o + 17] >>> 0,
    dstY: ringMetaView[o + 18] >>> 0,
    dstW: ringMetaView[o + 19] >>> 0,
    dstH: ringMetaView[o + 20] >>> 0,
    frameGapFlag: gapFlag,
    gapLabels: gaps,
    native1to1Active: ringMetaView[o + 22] === 1,
    continuityEpoch: ringMetaView[o + 23] >>> 0,
  };
}

function buildWave1Snapshot(dispSlot) {
  const w1 = wave1CoreAbiValid && wave1TelemView ? {
    available: true,
    abi: wave1TelemView[0] >>> 0,
    arenaBytes: wave1TelemView[1] >>> 0,
    lastCommittedSlot: wave1TelemView[2] >>> 0,
    displaySlot: wave1TelemView[3] >>> 0,
    totalCommits: (wave1TelemView[5] >>> 0) * 4294967296 + (wave1TelemView[4] >>> 0),
    continuityEpoch: wave1TelemView[6] >>> 0,
    enabledBits: wave1TelemView[7] >>> 0,
    lastIngestStatus: wave1TelemView[8] >>> 0,
    currentCommittedFlags: wave1TelemView[9] >>> 0,
    validRowCount: wave1TelemView[10] >>> 0,
    ringStride: wave1TelemView[11] >>> 0,
    linkedStaticEnd: wave1TelemView[12] >>> 0,
    dispMeta: readRingMetaSlot(dispSlot),
    stamps: {
      micro: decodeStamp(16),
      grad: decodeStamp(24),
      ftle: decodeStamp(32),
      modalField: decodeStamp(40),
      heroPhysics: decodeStamp(48),
      bos: decodeStamp(56),
    },
  } : { available: false };

  if (w1.available && rowLumaView && dispSlot >= 0 && dispSlot < RING_LEN) {
    const validRows = Math.min(ROW_LUMA_H, w1.validRowCount || frameH);
    let minL = 1.0;
    let maxL = 0.0;
    let sumL = 0.0;
    const base = dispSlot * ROW_LUMA_H;
    for (let r = 0; r < validRows; r++) {
      const v = rowLumaView[base + r];
      if (Number.isFinite(v)) {
        if (v < minL) minL = v;
        if (v > maxL) maxL = v;
        sumL += v;
      }
    }
    w1.dispRowLumaStats = validRows > 0
      ? { min: minL, max: maxL, mean: sumL / validRows, validRows }
      : null;
  }

  let metrology = {
    available: wave1MetrologyAbiValid,
    mode: microMetrologyMode,
    showCovEllipses,
    promotedCandidates: 0,
    activeAutoHeroes: 0,
    usableCovCount: 0,
    loomRejectedCount: 0,
    meanD2Bench: null,
  };
  if (wave1MetrologyAbiValid && heroCandidatesView && microQualityView && microResidualView) {
    metrology.promotedCandidates = heroCandidatesView[0] >>> 0;
    metrology.promoEpoch = heroCandidatesView[1] >>> 0;
    metrology.activeAutoHeroes = heroCandidatesView[2] >>> 0;
    let covUsable = 0;
    let loomRejected = 0;
    let d2Sum = 0.0;
    let d2Cnt = 0;
    for (let i = 0; i < MAX_MICRO; i++) {
      const q = microQualityView[i];
      if ((q & (1 << 3)) !== 0) covUsable++;
      const d2 = microResidualView[2 * MAX_MICRO + i];
      const benchUsable = (q & (1 << 4)) !== 0;
      const loomUsable = (q & (1 << 17)) !== 0;
      const isPromo = (q & (1 << 18)) !== 0;
      if (benchUsable && Number.isFinite(d2) && d2 >= 0.0) {
        d2Sum += d2;
        d2Cnt++;
        if (
          loomUsable &&
          d2 >= 18.42 &&
          !isPromo &&
          microLoomView &&
          Number.isFinite(microLoomView[i]) &&
          microLoomView[i] < 13.82
        ) {
          loomRejected++;
        }
      }
    }
    metrology.usableCovCount = covUsable;
    metrology.loomRejectedCount = loomRejected;
    metrology.meanD2Bench = d2Cnt > 0 ? d2Sum / d2Cnt : null;
  }

  let bos = {
    available: wave1BosAbiValid,
    reqMode: bosReqMode,
    overlayKind: bosOverlayKind,
    gaugeRoi: [...bosGaugeRoi],
    state: 0,
    stateLabel: 'OFF',
    reasonMask: 0,
    reasonFlags: [],
    validDisp: 0,
    validCurl: 0,
    components: 0,
    validGauge: 0,
    pcgIters: 0,
    path: 0,
    warmupCnt: 0,
    refEpoch: 0,
    trueRelRes: 0.0,
    curlRms: 0.0,
    maxCurl: 0.0,
    shiftX: 0.0,
    shiftY: 0.0,
    anchorMse: 0.0,
    validFrac: 0.0,
    rmsDisp: 0.0,
  };
  const bt = (dispSlot >= 0 && dispSlot < RING_LEN && bosRingValid[dispSlot] === 1)
    ? bosRingTelem[dispSlot]
    : null;
  if (bt) {
    const rMask = bt[3] | 0;
    const rList = [];
    for (let b = 0; b < BOS_REASON_NAMES.length; b++) {
      if (rMask & (1 << b)) rList.push(BOS_REASON_NAMES[b]);
    }
    bos.reqMode = bt[1] | 0;
    bos.state = bt[2] | 0;
    bos.stateLabel = BOS_STATE_LABELS[Math.max(0, Math.min(3, bos.state))] || 'OFF';
    bos.reasonMask = rMask;
    bos.reasonFlags = rList;
    bos.validDisp = bt[4] | 0;
    bos.validCurl = bt[5] | 0;
    bos.components = bt[6] | 0;
    bos.validGauge = bt[7] | 0;
    bos.pcgIters = bt[8] | 0;
    bos.path = bt[9] | 0;
    bos.warmupCnt = bt[10] | 0;
    bos.refEpoch = bt[11] | 0;
    bos.trueRelRes = finiteOrNull(bt[12]) ?? 0.0;
    bos.curlRms = finiteOrNull(bt[13]) ?? 0.0;
    bos.maxCurl = finiteOrNull(bt[14]) ?? 0.0;
    bos.shiftX = finiteOrNull(bt[15]) ?? 0.0;
    bos.shiftY = finiteOrNull(bt[16]) ?? 0.0;
    bos.anchorMse = finiteOrNull(bt[17]) ?? 0.0;
    bos.validFrac = finiteOrNull(bt[18]) ?? 0.0;
    bos.rmsDisp = finiteOrNull(bt[19]) ?? 0.0;
  } else if (wave1BosAbiValid && bosTelemView && bosTelemF32View) {
    const rMask = bosTelemView[3] >>> 0;
    const rList = [];
    for (let b = 0; b < BOS_REASON_NAMES.length; b++) {
      if (rMask & (1 << b)) rList.push(BOS_REASON_NAMES[b]);
    }
    bos.reqMode = bosTelemView[1] >>> 0;
    bos.state = bosTelemView[2] >>> 0;
    bos.stateLabel = BOS_STATE_LABELS[Math.max(0, Math.min(3, bos.state))] || 'OFF';
    bos.reasonMask = rMask;
    bos.reasonFlags = rList;
    bos.validDisp = bosTelemView[4] >>> 0;
    bos.validCurl = bosTelemView[5] >>> 0;
    bos.components = bosTelemView[6] >>> 0;
    bos.validGauge = bosTelemView[7] >>> 0;
    bos.pcgIters = bosTelemView[8] >>> 0;
    bos.path = bosTelemView[9] >>> 0;
    bos.warmupCnt = bosTelemView[10] >>> 0;
    bos.refEpoch = bosTelemView[11] >>> 0;
    bos.trueRelRes = finiteOrNull(bosTelemF32View[16]) ?? 0.0;
    bos.curlRms = finiteOrNull(bosTelemF32View[17]) ?? 0.0;
    bos.maxCurl = finiteOrNull(bosTelemF32View[18]) ?? 0.0;
    bos.shiftX = finiteOrNull(bosTelemF32View[19]) ?? 0.0;
    bos.shiftY = finiteOrNull(bosTelemF32View[20]) ?? 0.0;
    bos.anchorMse = finiteOrNull(bosTelemF32View[21]) ?? 0.0;
    bos.validFrac = finiteOrNull(bosTelemF32View[22]) ?? 0.0;
    bos.rmsDisp = finiteOrNull(bosTelemF32View[23]) ?? 0.0;
  }

  const dispNativeActive = Boolean(w1.dispMeta ? w1.dispMeta.native1to1Active : lastIngestGeom.native1to1);
  const roi = {
    native1to1: nativeRoiEnabled,
    native1to1Active: dispNativeActive,
    fallbackScale: Boolean(w1.dispMeta ? (nativeRoiEnabled && !dispNativeActive) : lastIngestGeom.fallbackScale),
    normX: roiOffsetNormX,
    normY: roiOffsetNormY,
    cropX0: w1.dispMeta ? w1.dispMeta.cropX0 : lastIngestGeom.cropX0,
    cropY0: w1.dispMeta ? w1.dispMeta.cropY0 : lastIngestGeom.cropY0,
    cropW: w1.dispMeta ? w1.dispMeta.cropW : lastIngestGeom.cropW,
    cropH: w1.dispMeta ? w1.dispMeta.cropH : lastIngestGeom.cropH,
    srcW: w1.dispMeta ? w1.dispMeta.srcW : lastIngestGeom.srcW,
    srcH: w1.dispMeta ? w1.dispMeta.srcH : lastIngestGeom.srcH,
    orientation: w1.dispMeta ? w1.dispMeta.orientation : lastIngestGeom.orientation,
    scanDir: w1.dispMeta ? w1.dispMeta.scanDir : lastIngestGeom.scanDir,
    dstX: w1.dispMeta ? w1.dispMeta.dstX : lastIngestGeom.dstX,
    dstY: w1.dispMeta ? w1.dispMeta.dstY : lastIngestGeom.dstY,
    dstW: w1.dispMeta ? w1.dispMeta.dstW : lastIngestGeom.dstW,
    dstH: w1.dispMeta ? w1.dispMeta.dstH : lastIngestGeom.dstH,
    engineW: frameW,
    engineH: frameH,
  };

  return { wave1: w1, roi, bos, metrology };
}

function decodeConservationArr(ca) {
  const cStatus = Math.max(0, Math.min(3, Math.round(ca[0]) | 0));
  const cMask = Math.round(ca[1]) >>> 0;
  const hyps = [];
  for (let b = 0; b < OCCL_HYP_NAMES.length; b++) {
    if (cMask & (1 << b)) hyps.push(OCCL_HYP_NAMES[b]);
  }
  return {
    available: true,
    status: cStatus,
    statusLabel: OCCL_STATUS_LABELS[cStatus] || 'NONE',
    ambiguityMask: cMask,
    hypotheses: hyps,
    admissibleCount: Math.round(ca[2]) >>> 0,
    jNetMag: finiteOrNull(ca[3]) ?? 0.0,
    jEnvMax: finiteOrNull(ca[4]) ?? 0.0,
    momResidNorm: finiteOrNull(ca[5]) ?? 0.0,
    angMomResidNorm: finiteOrNull(ca[6]) ?? 0.0,
    deltaKeRatio: finiteOrNull(ca[7]) ?? 0.0,
    chi2: [
      finiteOrNull(ca[8]) ?? 0.0,
      finiteOrNull(ca[9]) ?? 0.0,
      finiteOrNull(ca[10]) ?? 0.0,
      finiteOrNull(ca[11]) ?? 0.0,
    ],
    estTauSec: finiteOrNull(ca[12]) ?? 0.0,
    estRestitutionE: finiteOrNull(ca[13]) ?? 0.0,
    tidA: Math.round(ca[14]) | 0,
    tidB: Math.round(ca[15]) | 0,
    occlDtSec: finiteOrNull(ca[16]) ?? 0.0,
    eventFrame: Math.round(ca[17]) >>> 0,
    impResids: [
      finiteOrNull(ca[18]) ?? 0.0,
      finiteOrNull(ca[19]) ?? 0.0,
      finiteOrNull(ca[20]) ?? 0.0,
      finiteOrNull(ca[21]) ?? 0.0,
    ],
    gravityYUsed: finiteOrNull(ca[22]) ?? 0.0,
    massRatio: finiteOrNull(ca[23]) ?? 1.0,
  };
}

function buildWave2Snapshot(dispSlot) {
  const effK = telemView ? Math.round(telemView[1]) : 0;
  const delayedRing = hasAnyRenderedFrame() && effK > 0;
  const hasSlotCore = dispSlot >= 0 && dispSlot < RING_LEN && w2CoreRingValid[dispSlot] === 1;
  const core = hasSlotCore ? (() => {
    const cr = w2CoreRing[dispSlot];
    const crF32 = w2CoreRingF32[dispSlot];
    return {
      available: true,
      abiVersion: cr[0] >>> 0,
      arenaSizeBytes: cr[1] >>> 0,
      enabledBits: cr[2] >>> 0,
      continuityEpoch: cr[3] >>> 0,
      stamps: {
        vibro: decodeStampFrom(cr, crF32, 4),
        hodge: decodeStampFrom(cr, crF32, 12),
        conservation: manualConsTestActive && wave2CoreAbiValid
          ? decodeWave2Stamp(32)
          : decodeStampFrom(cr, crF32, 20),
        depth: decodeStampFrom(cr, crF32, 28),
        wind: decodeStampFrom(cr, crF32, 36),
        elasto: decodeStampFrom(cr, crF32, 44),
      },
    };
  })() : (!delayedRing && wave2CoreAbiValid && wave2TelemView && wave2TelemF32View ? {
    available: true,
    abiVersion: wave2TelemView[0] >>> 0,
    arenaSizeBytes: wave2TelemView[1] >>> 0,
    enabledBits: wave2TelemView[2] >>> 0,
    continuityEpoch: wave2TelemView[3] >>> 0,
    stamps: {
      vibro: decodeWave2Stamp(16),
      hodge: decodeWave2Stamp(24),
      conservation: decodeWave2Stamp(32),
      depth: decodeWave2Stamp(40),
      wind: decodeWave2Stamp(48),
      elasto: decodeWave2Stamp(56),
    },
  } : {
    available: wave2CoreAbiValid,
    abiVersion: wave2CoreAbiValid && wave2TelemView ? (wave2TelemView[0] >>> 0) : 0,
    arenaSizeBytes: wave2CoreAbiValid && wave2TelemView ? (wave2TelemView[1] >>> 0) : 0,
    enabledBits: 0,
    continuityEpoch: wave2CoreAbiValid && wave2TelemView ? (wave2TelemView[3] >>> 0) : 0,
    stamps: {
      vibro: { validity: 0, valid: false, held: false },
      hodge: { validity: 0, valid: false, held: false },
      conservation: manualConsTestActive && wave2CoreAbiValid
        ? decodeWave2Stamp(32)
        : { validity: 0, valid: false, held: false },
      depth: { validity: 0, valid: false, held: false },
      wind: { validity: 0, valid: false, held: false },
      elasto: { validity: 0, valid: false, held: false },
    },
  });

  const vibroSrc = resolveWave2RingOrLive(
    dispSlot, effK, vibroRingValid, vibroRingTelem, vibroRingPlanes,
    wave2VibroAbiValid, vibroTelemView, vibroPlanesView
  );
  const liveVibroNonReady = rsCalibConfig.mode > 0 && wave2VibroAbiValid && vibroTelemView && vibroTelemView[0] !== 3;
  const vt = vibroSrc
    ? vibroSrc.t
    : (wave2VibroAbiValid && (!delayedRing || liveVibroNonReady) ? vibroTelemView : null);
  const vibro = vt ? (() => {
    const vState = Math.max(0, Math.min(4, vt[0] >>> 0));
    const rMask = vt[2] >>> 0;
    const rList = [];
    for (let b = 0; b < RS_REASON_NAMES.length; b++) {
      if (rMask & (1 << b)) rList.push(RS_REASON_NAMES[b]);
    }
    return {
      available: true,
      reqMode: rsCalibConfig.mode,
      state: vState,
      stateLabel: VIBRO_STATE_LABELS[vState] || 'OFF',
      mode: vt[1] >>> 0,
      reasonMask: rMask,
      reasonFlags: rList,
      welchK: vt[3] >>> 0,
      validCells: vt[4] >>> 0,
      domFreqHz: finiteOrNull(u32BitsToF32(vt[5])) ?? 0.0,
      maxCohGamma2: finiteOrNull(u32BitsToF32(vt[6])) ?? 0.0,
      gamma2Crit: finiteOrNull(u32BitsToF32(vt[7])) ?? 0.0,
      sincH: finiteOrNull(u32BitsToF32(vt[8])) ?? 1.0,
      k1TrapCoh: finiteOrNull(u32BitsToF32(vt[9])) ?? 0.0,
      clockSkewUs: finiteOrNull(u32BitsToF32(vt[10])) ?? 0.0,
      domPhaseRad: finiteOrNull(u32BitsToF32(vt[11])) ?? 0.0,
      domAmpPx: finiteOrNull(u32BitsToF32(vt[12])) ?? 0.0,
      freqBinHzStep: finiteOrNull(u32BitsToF32(vt[13])) ?? 0.0,
      dtRowUs: finiteOrNull(u32BitsToF32(vt[14])) ?? 0.0,
      exposureUs: finiteOrNull(u32BitsToF32(vt[15])) ?? 0.0,
      audioRateHz: finiteOrNull(u32BitsToF32(vt[16])) ?? 0.0,
      maxSkewUs: finiteOrNull(u32BitsToF32(vt[17])) ?? 0.0,
      audioTotalSamples: vt[18] >>> 0,
      activeFramesInRun: vt[19] >>> 0,
      domBandIdx: vt[20] >>> 0,
      coupledCellsCount: vt[21] >>> 0,
      exposureNullFlag: (rMask & (1 << 6)) !== 0,
      calib: { ...rsCalibConfig },
    };
  })() : {
    available: wave2VibroAbiValid,
    reqMode: rsCalibConfig.mode,
    state: 0,
    stateLabel: 'OFF',
    mode: 0,
    reasonMask: 0,
    reasonFlags: [],
  };

  const hodgeSrc = resolveWave2RingOrLive(
    dispSlot, effK, hodgeRingValid, hodgeRingTelem, hodgeRingPlanes,
    wave2HodgeAbiValid, hodgeTelemView, hodgePlanesView
  );
  const liveHodgeNonReady = hodgeReqMode > 0 && wave2HodgeAbiValid && hodgeTelemView && hodgeTelemView[0] !== 2;
  const ht = hodgeSrc
    ? hodgeSrc.t
    : (wave2HodgeAbiValid && (!delayedRing || liveHodgeNonReady) ? hodgeTelemView : null);
  const hodge = ht ? (() => {
    const hState = Math.max(0, Math.min(3, ht[0] >>> 0));
    const wakeX0 = finiteOrNull(u32BitsToF32(ht[25])) ?? 0.0;
    const wakeY0 = finiteOrNull(u32BitsToF32(ht[26])) ?? 0.0;
    const wakeX1 = finiteOrNull(u32BitsToF32(ht[27])) ?? 0.0;
    const wakeY1 = finiteOrNull(u32BitsToF32(ht[28])) ?? 0.0;
    return {
      available: true,
      reqMode: hodgeReqMode,
      state: hState,
      stateLabel: HODGE_STATE_LABELS[hState] || 'OFF',
      mode: ht[1] >>> 0,
      converged: (ht[2] >>> 0) !== 0,
      vAct: ht[3] >>> 0,
      eAct: ht[4] >>> 0,
      fAct: ht[5] >>> 0,
      compCnt: ht[6] >>> 0,
      beta1: ht[7] >>> 0,
      internalVoids: ht[8] >>> 0,
      openPorts: ht[9] >>> 0,
      wakeHyps: ht[10] >>> 0,
      totalEnergy: finiteOrNull(u32BitsToF32(ht[11])) ?? 0.0,
      exactFrac: finiteOrNull(u32BitsToF32(ht[12])) ?? 0.0,
      coexactFrac: finiteOrNull(u32BitsToF32(ht[13])) ?? 0.0,
      harmFrac: finiteOrNull(u32BitsToF32(ht[14])) ?? 0.0,
      orthEcRel: finiteOrNull(u32BitsToF32(ht[15])) ?? 0.0,
      orthEhRel: finiteOrNull(u32BitsToF32(ht[16])) ?? 0.0,
      orthChRel: finiteOrNull(u32BitsToF32(ht[17])) ?? 0.0,
      energyPartErr: finiteOrNull(u32BitsToF32(ht[18])) ?? 0.0,
      exactCgRes: finiteOrNull(u32BitsToF32(ht[19])) ?? 0.0,
      coexactCgRes: finiteOrNull(u32BitsToF32(ht[20])) ?? 0.0,
      primaryVoidCirc: finiteOrNull(u32BitsToF32(ht[21])) ?? 0.0,
      primaryWakeDeficit: finiteOrNull(u32BitsToF32(ht[22])) ?? 0.0,
      voidCx: finiteOrNull(u32BitsToF32(ht[23])) ?? 0.0,
      voidCy: finiteOrNull(u32BitsToF32(ht[24])) ?? 0.0,
      wakeX0,
      wakeY0,
      wakeX1,
      wakeY1,
      wakeBbox: [wakeX0, wakeY0, wakeX1, wakeY1],
      exactIters: ht[29] >>> 0,
      coexactIters: ht[30] >>> 0,
    };
  })() : {
    available: wave2HodgeAbiValid,
    reqMode: hodgeReqMode,
    state: 0,
    stateLabel: 'OFF',
    mode: 0,
    converged: false,
  };

  const ca = resolveWave2ConsArr(dispSlot, effK);
  const conservation = ca
    ? decodeConservationArr(ca)
    : { available: wave2ConsAbiValid, status: 0, statusLabel: 'NONE', ambiguityMask: 0, hypotheses: [] };

  const hasSlotDepth = dispSlot >= 0 && dispSlot < RING_LEN && depthRingValid[dispSlot] === 1;
  const dtU = hasSlotDepth
    ? depthRingTelem[dispSlot]
    : (!delayedRing && depthAbiValid && depthTelemView ? depthTelemView : null);
  const dtF = hasSlotDepth
    ? depthRingTelemF32[dispSlot]
    : (!delayedRing && depthAbiValid && depthTelemF32View ? depthTelemF32View : null);
  const depth = dtU && dtF
    ? (() => {
        const st = Math.max(0, Math.min(7, dtU[2] >>> 0));
        const pv = Math.max(0, Math.min(6, dtU[4] >>> 0));
        return {
          available: true,
          reqMode: depthReqMode,
          mode: dtU[1] >>> 0,
          state: st,
          stateLabel: DEPTH_STATE_LABELS[st] || 'OFF',
          reasonMask: dtU[3] >>> 0,
          provenance: pv,
          provenanceLabel: DEPTH_PROV_LABELS[pv] || 'NONE',
          validTracks: dtU[5] >>> 0,
          movingRejected: dtU[6] >>> 0,
          ttcValidTracks: dtU[7] >>> 0,
          foeInliers: dtU[8] >>> 0,
          validCells: dtU[9] >>> 0,
          extSeq: dtU[12] >>> 0,
          stereoValidNodes: dtU[13] >>> 0,
          heroMask: dtU[14] >>> 0,
          medRotResid: finiteOrNull(dtF[16]) ?? 0.0,
          rotRatio: finiteOrNull(dtF[17]) ?? 0.0,
          foeX: finiteOrNull(dtF[22]) ?? 0.0,
          foeY: finiteOrNull(dtF[23]) ?? 0.0,
          minTtcSec: finiteOrNull(dtF[24]) ?? 0.0,
          bestTtcR2: finiteOrNull(dtF[25]) ?? 0.0,
          epiRmsPx: finiteOrNull(dtF[28]) ?? 0.0,
          medDepthM: finiteOrNull(dtF[29]) ?? 0.0,
          minDepthM: finiteOrNull(dtF[30]) ?? 0.0,
          maxDepthM: finiteOrNull(dtF[31]) ?? 0.0,
        };
      })()
    : { available: depthAbiValid, reqMode: depthReqMode, state: 0, stateLabel: 'OFF', provenance: 0, provenanceLabel: 'NONE', extSeq: 0 };

  const hasSlotWind = dispSlot >= 0 && dispSlot < RING_LEN && windRingValid[dispSlot] === 1;
  const wtU = hasSlotWind
    ? windRingTelem[dispSlot]
    : (!delayedRing && windAbiValid && windTelemView ? windTelemView : null);
  const wtF = hasSlotWind
    ? windRingTelemF32[dispSlot]
    : (!delayedRing && windAbiValid && windTelemF32View ? windTelemF32View : null);
  const wind = wtU && wtF
    ? (() => {
        const st = Math.max(0, Math.min(4, wtU[0] >>> 0));
        return {
          available: true,
          reqMode: windReqMode,
          mode: wtU[1] >>> 0,
          state: st,
          stateLabel: WIND_STATE_LABELS[st] || 'OFF',
          reasonMask: wtU[2] >>> 0,
          provenance: wtU[3] >>> 0,
          validCells: wtU[4] >>> 0,
          beta1: wtU[5] >>> 0,
          uInf: finiteOrNull(wtF[8]) ?? 0.0,
          dHolePx: finiteOrNull(wtF[10]) ?? 0.0,
          nu: finiteOrNull(wtF[11]) ?? 0.0,
          nuR2: finiteOrNull(wtF[12]) ?? 0.0,
          reNum: finiteOrNull(wtF[13]) ?? 0.0,
          cpMin: finiteOrNull(wtF[14]) ?? 0.0,
          cpMax: finiteOrNull(wtF[15]) ?? 0.0,
          clBnd: finiteOrNull(wtF[16]) ?? 0.0,
          cdBnd: finiteOrNull(wtF[17]) ?? 0.0,
          cl: finiteOrNull(wtF[18]) ?? 0.0,
          cd: finiteOrNull(wtF[19]) ?? 0.0,
          clCv: finiteOrNull(wtF[18]) ?? 0.0,
          cdCv: finiteOrNull(wtF[19]) ?? 0.0,
          clKj: finiteOrNull(wtF[20]) ?? 0.0,
          etaNs: finiteOrNull(wtF[21]) ?? 0.0,
          divFrac: finiteOrNull(wtF[22]) ?? 0.0,
        };
      })()
    : { available: windAbiValid, reqMode: windReqMode, state: 0, stateLabel: 'OFF' };

  const hasSlotElasto = dispSlot >= 0 && dispSlot < RING_LEN && elastoRingValid[dispSlot] === 1;
  const etU = hasSlotElasto
    ? elastoRingTelem[dispSlot]
    : (!delayedRing && elastoAbiValid && elastoTelemView ? elastoTelemView : null);
  const etF = hasSlotElasto
    ? elastoRingTelemF32[dispSlot]
    : (!delayedRing && elastoAbiValid && elastoTelemF32View ? elastoTelemF32View : null);
  const elasto = etU && etF
    ? (() => {
        const st = Math.max(0, Math.min(4, etU[0] >>> 0));
        const cls = Math.max(0, Math.min(3, etU[3] >>> 0));
        return {
          available: true,
          reqMode: elastoReqMode,
          mode: etU[1] >>> 0,
          state: st,
          stateLabel: ELASTO_STATE_LABELS[st] || 'OFF',
          reasonMask: etU[2] >>> 0,
          dispClass: cls,
          classLabel: ELASTO_CLASS_LABELS[cls] || 'NONE',
          validCells: etU[4] >>> 0,
          lockedFreqs: etU[5] >>> 0,
          welchSegs: etU[6] >>> 0,
          beta1: etU[7] >>> 0,
          f0: finiteOrNull(etF[8]) ?? 0.0,
          f1: finiteOrNull(etF[9]) ?? 0.0,
          medC: finiteOrNull(etF[12]) ?? 0.0,
          alpha: finiteOrNull(etF[13]) ?? 0.0,
          anisotropy: finiteOrNull(etF[14]) ?? 1.0,
          meanCoh: finiteOrNull(etF[15]) ?? 0.0,
          srcX: finiteOrNull(etF[16]) ?? 0.0,
          srcY: finiteOrNull(etF[17]) ?? 0.0,
          sinkX: finiteOrNull(etF[18]) ?? 0.0,
          sinkY: finiteOrNull(etF[19]) ?? 0.0,
          abosCurlRms: finiteOrNull(etF[24]) ?? 0.0,
          abosPhiRms: finiteOrNull(etF[25]) ?? 0.0,
        };
      })()
    : { available: elastoAbiValid, reqMode: elastoReqMode, state: 0, stateLabel: 'OFF' };

  return { ...core, vibro, hodge, conservation, depth, wind, elasto };
}

function emitTelemetrySnapshot() {
  if (!wasmExports || !telemView) return;
  let sumMs = 0.0;
  let maxMs = 0.0;
  const count = Math.min(Math.max(1, tickCounter), TIME_HIST_LEN);
  for (let i = 0; i < count; i++) {
    sumMs += wasmTimeHist[i];
    if (wasmTimeHist[i] > maxMs) maxMs = wasmTimeHist[i];
  }
  const meanMs = sumMs / Math.max(1, count);
  const medianMs = computeMedianTickMs(count);
  const effSampleRate = sourceMode === 'testbench' ? 60.0 : Math.max(8.0, sourceFps);
  const fIdxNow = currentCausalEngineFrameU32();

  const pIdx = resolvePrimarySlot();
  const focusTid = wasmExports ? wasmExports.chronos_get_focus_target() : 0;
  const metricOn = isMetricCalibrated();
  let pTid = 0;
  let pSigHex = '0x00000000';
  let pVal = 0;
  let apexMs = 0;
  let primaryCoherence = 0;
  let headingDeg = 0.0;
  let speedPhys = 0.0;
  let pMetricValid = metricOn ? 1.0 : 0.0;
  let angVelDegS = 0.0;
  let pivotValid = false;
  let pendSwingDeg = 0.0;
  let pendRadiusPhys = 0.0;
  let pFxPx = frameW * 0.5;
  let pFyPx = frameH * 0.5;
  const hasPrimary = pIdx >= 0 && telemView[32 + pIdx * 56] > 0.5;
  if (hasPrimary) {
    const b = 32 + pIdx * 56;
    const kb = TELEM_KINETIC + pIdx * 8;
    const fxPx = telemView[b + 2];
    const fyPx = telemView[b + 3];
    pFxPx = fxPx;
    pFyPx = fyPx;
    pTid = Math.round(telemView[b + 1]);
    pSigHex = toHex8(telemView[b + 12], telemView[b + 13]);
    pVal = Math.round(telemView[b + 14]);
    apexMs = telemView[b + 17] * (1000.0 / effSampleRate);
    primaryCoherence = telemView[TELEM_EXCUR + pIdx * 8 + 6];
    const physVel = mapVelocityToPhysical(fxPx, fyPx, telemView[kb + 1], telemView[kb + 0]);
    speedPhys = physVel[0];
    headingDeg = physVel[1];
    pMetricValid = physVel[2];
    angVelDegS = (calMode === 'homography' && calHomographyValid && physVel[2] > 0.5)
      ? slotRectAngVel[pIdx]
      : telemView[kb + 2];
    if (telemView[kb + 6] > -9.0e5 && telemView[kb + 7] > -9.0e5) {
      pivotValid = true;
      const pend = computePendulumKinematics(fxPx, fyPx, telemView[kb + 6], telemView[kb + 7]);
      pendSwingDeg = pend[0];
      pendRadiusPhys = pend[1];
    }
  }
  if (apexMs <= 0.0) {
    for (let m = 0; m < MAX_TARGETS; m++) {
      const mb = 32 + m * 56;
      if (telemView[mb] > 0.5 && telemView[mb + 17] > 0.35) {
        const candApexMs = telemView[mb + 17] * (1000.0 / effSampleRate);
        if (candApexMs > apexMs) apexMs = candApexMs;
      }
    }
  }

  const speedUnit = pMetricValid < -0.5
    ? 'OUT OF PLANE'
    : (pMetricValid > 0.5 ? 'm/s' : 'px/s');
  const distUnit = pMetricValid < -0.5
    ? 'OUT OF PLANE'
    : (metricOn ? 'm' : 'px');
  let lastStrikeAgeFrames = -1;
  if (lastStrikeValid && lastStrikeFrame >= 0) {
    const d = (fIdxNow - (lastStrikeFrame >>> 0)) >>> 0;
    lastStrikeAgeFrames = d < 0x80000000 ? d : 0;
  }
  const scatterFrameWasm = Math.round(telemView[TELEM_GLOBAL + 7]);
  if (scatterFrameWasm > 0) lastScatterFrame = scatterFrameWasm;
  let scatterAgeFrames = -1;
  if (lastScatterFrame > 0) {
    const dSc = (fIdxNow - (lastScatterFrame >>> 0)) >>> 0;
    scatterAgeFrames = dSc < 0x80000000 ? dSc : 0;
  }

  const slotRoster = new Array(MAX_TARGETS);
  for (let m = 0; m < MAX_TARGETS; m++) {
    const b = 32 + m * 56;
    const active = telemView[b] > 0.5;
    if (active) {
      const tid = Math.round(telemView[b + 1]);
      const dispVis = telemView[b + 10] > 0.5;
      const isBridged = telemView[b + 11] > 0.5;
      const isPinned = focusTid !== 0 && tid === focusTid;
      const isPrimary = m === pIdx;
      const mv = slotLastMetricValid[m];
      const isPiv = slotLastPivotValid[m] > 0;
      const state = isBridged
        ? 'BRIDGED'
        : !dispVis
        ? 'COAST'
        : isPinned
        ? 'PINNED'
        : isPrimary
        ? 'FOCUS'
        : 'LOCKED';
      slotRoster[m] = {
        slot: m,
        state,
        tid,
        color: TARGET_COLORS[Math.max(0, tid - 1) % TARGET_COLORS.length],
        sigHex: toHex8(slotLastSigHi[m], slotLastSigLo[m]),
        padicVal: slotLastVal[m],
        reacq: slotLastReacq[m],
        coastFrames: slotLastCoast[m],
        ageSec: slotLastAgeFrames[m] / effSampleRate,
        speed: slotSmoothedSpeed[m],
        speedUnit: mv < -0.5 ? 'OOP' : (mv > 0.5 ? 'm/s' : 'px/s'),
        isPivot: isPiv,
        angleDeg: isPiv ? slotLastPendSwing[m] : slotSmoothedHeading[m],
        rateOrRad: isPiv ? slotLastPendRad[m] : slotSmoothedOmega[m],
        rateUnit: isPiv ? (metricOn ? 'm' : 'px') : '°/s',
        freqHz: slotLastFreqHz[m],
        ecaPct: slotSmoothedEca[m],
        isPinned,
        isPrimary,
        physRegime: Math.max(0, Math.min(8, Math.round(safeFinite(telemView[TELEM_PHYSICS_START + m * TELEM_PHYSICS_STRIDE + 4])))),
        odeR2: safeFinite(telemView[TELEM_PHYSICS_START + m * TELEM_PHYSICS_STRIDE + 6]),
        jumpCode: Math.max(0, Math.min(4, Math.round(safeFinite(telemView[TELEM_PHYSICS_START + m * TELEM_PHYSICS_STRIDE + 15])))),
      };
    } else if (slotHoldLeft[m] > 0 && slotHoldTid[m] > 0) {
      const tid = slotHoldTid[m];
      const mv = slotLastMetricValid[m];
      const isPiv = slotLastPivotValid[m] > 0;
      slotRoster[m] = {
        slot: m,
        state: 'HOLD',
        tid,
        color: TARGET_COLORS[Math.max(0, tid - 1) % TARGET_COLORS.length],
        sigHex: toHex8(slotLastSigHi[m], slotLastSigLo[m]),
        padicVal: slotLastVal[m],
        reacq: slotLastReacq[m],
        coastFrames: slotLastCoast[m],
        ageSec: slotLastAgeFrames[m] / effSampleRate,
        speed: slotSmoothedSpeed[m],
        speedUnit: mv < -0.5 ? 'OOP' : (mv > 0.5 ? 'm/s' : 'px/s'),
        isPivot: isPiv,
        angleDeg: isPiv ? slotLastPendSwing[m] : slotSmoothedHeading[m],
        rateOrRad: isPiv ? slotLastPendRad[m] : slotSmoothedOmega[m],
        rateUnit: isPiv ? (metricOn ? 'm' : 'px') : '°/s',
        freqHz: slotLastFreqHz[m],
        ecaPct: slotSmoothedEca[m],
        isPinned: focusTid !== 0 && tid === focusTid,
        isPrimary: false,
        physRegime: 0,
        odeR2: 0.0,
        jumpCode: 0,
      };
    } else {
      slotRoster[m] = {
        slot: m,
        state: 'IDLE',
        tid: 0,
        color: '#64748b',
        sigHex: '0x00000000',
        padicVal: 0,
        reacq: 0,
        coastFrames: 0,
        ageSec: 0.0,
        speed: 0.0,
        speedUnit: metricOn ? 'm/s' : 'px/s',
        isPivot: false,
        angleDeg: 0.0,
        rateOrRad: 0.0,
        rateUnit: '°/s',
        freqHz: 0.0,
        ecaPct: 0.0,
        isPinned: false,
        isPrimary: false,
        physRegime: 0,
        odeR2: 0.0,
        jumpCode: 0,
      };
    }
  }

  const dispSlotNow = currentDisplaySlot();
  const effKNow = Math.round(telemView[1]);
  const c1kSnap = buildChronos1000Snapshot(medianMs, effKNow, fIdxNow, dispSlotNow);
  const w1Snap = buildWave1Snapshot(dispSlotNow);
  const oracleValid = sourceMode === 'testbench' && !isChronos1000Scenario(benchScenario);

  self.postMessage({
    type: 'telemetry',
    frameIdx: fIdxNow,
    lookaheadK: effKNow,
    fps: effSampleRate,
    wasmMeanMs: meanMs,
    wasmMedianMs: medianMs,
    wasmMaxMs: maxMs,
    lookaheadWindowMs: c1kSnap.lookaheadWindowMs,
    lookaheadSource: c1kSnap.lookaheadSource,
    partialPipelineLatencyMs: c1kSnap.partialPipelineLatencyMs,
    estimatedPartialPipelineLatencyMs: c1kSnap.estimatedPartialPipelineLatencyMs,
    snapshotCopyMs: lastSwarmCopyMs,
    fieldRenderMs: lastFieldRenderMs,
    activeTargets: Math.round(telemView[3]),
    clutterDb: telemView[4],
    clutterDbValid: telemView[22] > 0.5,
    clutterEvalRatio: telemView[23],
    padicSwaps: Math.round(telemView[5]),
    oracleSwaps: oracleValid ? oraclePadicSwaps : null,
    oracleSwapsAvailable: oracleValid,
    sourceMode,
    benchScenario,
    kalmanSwaps: Math.round(telemView[6]),
    inPillarCollisions,
    bridgedFrames: totalBridgedEvents,
    strikeEvents: totalStrikeEvents,
    lambda2: lastLambda2,
    spectralGap: telemView[24],
    fistaDualityGap: telemView[25],
    phaseValid: telemView[26] > 0.5,
    freqValid: telemView[TELEM_GLOBAL + 12] > 0.5,
    tremorPhaseDeg: telemView[11] * (180.0 / Math.PI),
    tremorFreqHz: telemView[13],
    tremorAmp: telemView[12],
    tremorCoherence: primaryCoherence,
    saccadeLatencyMs: telemView[19],
    stepSettleMs: telemView[28],
    stimulusArmed: telemView[29] > 0.5,
    primaryTrackId: pTid,
    focusTrackId: focusTid,
    primarySigHex: pSigHex,
    primaryPadicVal: pVal,
    apexMs,
    apexErrPx: lastScoredApexErrPx,
    headingDeg,
    speedPhys,
    metricValid: pMetricValid,
    speedUnit,
    distUnit,
    angVelDegS,
    pivotValid,
    pendSwingDeg,
    pendRadiusPhys,
    sceneGravityPxS2: telemView[TELEM_GLOBAL + 0],
    sceneGravityConf: telemView[TELEM_GLOBAL + 1],
    sceneGravityArcs: Math.round(telemView[TELEM_GLOBAL + 11]),
    dispLagMs: telemView[TELEM_GLOBAL + 8],
    camMotionState: Math.round(telemView[TELEM_GLOBAL + 9]),
    camPanSpeedPxS: telemView[TELEM_GLOBAL + 10],
    discontCount: Math.round(telemView[TELEM_GLOBAL + 15]),
    lastDeflectDeg: lastStrikeDeflectDeg,
    lastRestitution: lastStrikeRestitution,
    lastStrikeTid,
    lastStrikeAgeFrames,
    lastStrikeValid,
    scatterDeg: lastScatterDeg,
    scatterAgeFrames,
    calSegmentId,
    calMode,
    calProvenance,
    calArmed,
    calibrated: metricOn,
    showHeroTargets,
    showAllOverlays,
    overlaySyncSeq: lastOverlaySyncSeq,
    rawWipeRatio,
    showcaseMinimalScopes,
    swarmMode,
    cropNorm: [cropNorm[0], cropNorm[1], cropNorm[2], cropNorm[3]],
    arenaBytes: Math.round(telemView[20]),
    memoryPages: wasmMemory.buffer.byteLength / 65536,
    resMode,
    frameW,
    frameH,
    pipDim,
    recActive,
    recFrames: recCount,
    recDurationSec: recElapsedSec,
    sessionPersistence: {
      uniqueIds: Math.round(telemView[TELEM_PERSIST + 0]),
      coastReacqEvents: Math.round(telemView[TELEM_PERSIST + 1]),
      dormantReidEvents: Math.round(telemView[TELEM_PERSIST + 2]),
      totalRelocks: Math.round(telemView[TELEM_PERSIST + 1]) + Math.round(telemView[TELEM_PERSIST + 2]),
      peakSimultaneous: Math.round(telemView[TELEM_PERSIST + 3]),
      longestTenureFrames: Math.round(telemView[TELEM_PERSIST + 4]),
      longestTenureSec: Math.round(telemView[TELEM_PERSIST + 4]) / effSampleRate,
      longestTenureTid: Math.round(telemView[TELEM_PERSIST + 5]),
      sustained10s: Math.round(telemView[TELEM_PERSIST + 6]),
      sustained60s: Math.round(telemView[TELEM_PERSIST + 7]),
      dormantActive: Math.round(telemView[TELEM_PERSIST + 8]),
      lastReidGapFrames: Math.round(telemView[TELEM_PERSIST + 9]),
      sessionElapsedSec,
    },
    physicsId: buildPhysicsIdSnapshot(pIdx, hasPrimary, pFxPx, pFyPx, speedPhys, speedUnit, pMetricValid, angVelDegS),
    slotRoster,
    chronos1000: c1kSnap,
    wave1: w1Snap.wave1,
    roi: w1Snap.roi,
    bos: w1Snap.bos,
    metrology: w1Snap.metrology,
    wave2: buildWave2Snapshot(dispSlotNow),
  });
}

function processIngestedFrame(rawPixels, dtSec, mediaTimeSec, ingestInfo = null) {
  if (!wasmExports || !rawPixels || rawPixels.byteLength !== frameBytes) {
    return { committed: false, status: 2, slotIdx: -1, dispSlot: -1, error: 'byteLength_mismatch' };
  }

  const ptr = wasmExports.chronos_get_ingest_rgba_ptr();
  const slotIdx = ringSlotByteOffsets.indexOf(ptr);
  if (slotIdx < 0 || !ringSlotU8[slotIdx]) {
    throw new Error(`Unmapped Wasm ingest pointer 0x${(ptr >>> 0).toString(16)} outside ringSlotByteOffsets`);
  }

  const rawDt = Math.max(0.001, Number.isFinite(dtSec) && dtSec > 0 ? dtSec : 1.0 / 60.0);
  const fpsDt = Math.min(0.25, Math.max(0.002, rawDt));
  const instSourceFps = 1.0 / fpsDt;
  sourceFps = 0.90 * sourceFps + 0.10 * instSourceFps;

  workerSimTimeSec += rawDt;
  const info = ingestInfo || {};
  const ingestEpoch = typeof info.epoch === 'number' && Number.isInteger(info.epoch) ? (info.epoch >>> 0) : workerLocalEpoch;
  const ingestSeq = typeof info.seq === 'number' && Number.isInteger(info.seq) ? (info.seq >>> 0) : (++workerLocalSeq >>> 0);
  const hasExplicitTSrcUs =
    typeof info.tSrcUs === 'number' &&
    Number.isFinite(info.tSrcUs) &&
    info.tSrcUs >= 0 &&
    info.ptsKnown !== false;
  const hasMediaPts =
    typeof mediaTimeSec === 'number' &&
    Number.isFinite(mediaTimeSec) &&
    mediaTimeSec >= 0.0;
  const hasPtsNow = info.ptsKnown !== false && (hasMediaPts || hasExplicitTSrcUs);
  const tUs = hasPtsNow
    ? (hasExplicitTSrcUs
        ? Math.round(info.tSrcUs)
        : Math.max(0, Math.round(mediaTimeSec * 1_000_000)))
    : 0;
  const tLo = tUs >>> 0;
  const tHi = Math.floor(tUs / 4294967296) >>> 0;

  let ageMs = -1.0;
  let ageFlags = 0;
  if (typeof info.captureAgeMs === 'number' && Number.isFinite(info.captureAgeMs) && info.captureAgeMs >= 0.0) {
    ageMs = info.captureAgeMs;
    ageFlags = 4; // INGEST_FLAG_CAPTURE_AGE
  } else if (typeof info.enqueuedEpochMs === 'number' && Number.isFinite(info.enqueuedEpochMs)) {
    ageMs = Math.max(0.0, performance.timeOrigin + performance.now() - info.enqueuedEpochMs);
    ageFlags = 8; // INGEST_FLAG_HANDOFF_AGE
  }

  const preScaled = Boolean(info.fallbackScaled || (info.geom && info.geom.preScaled));
  const geom = info.geom || {
    srcW: frameW,
    srcH: frameH,
    cropX0: 0,
    cropY0: 0,
    cropW: frameW,
    cropH: frameH,
    orientation: roiOrientation & 7,
    scanDir: roiScanDir >>> 0,
    dstX: 0,
    dstY: 0,
    dstW: frameW,
    dstH: frameH,
    native1to1: !preScaled && nativeRoiEnabled && (roiOrientation & 1) === 0 ? 1 : 0,
  };
  if (preScaled) {
    geom.native1to1 = 0;
    geom.fallbackScale = Boolean(nativeRoiEnabled);
  }
  if (!info.geom) {
    lastIngestGeom = {
      ...geom,
      engineW: frameW,
      engineH: frameH,
      fallbackScale: Boolean(nativeRoiEnabled && !geom.native1to1),
    };
  }
  const slewFlag = roiSlewPending ? 64 : 0; // INGEST_FLAG_ROI_SLEW = 1 << 6
  roiSlewPending = false;
  const metaFlags = (hasPtsNow ? 2 : 0) | ageFlags | 16 | slewFlag; // PTS_KNOWN | AGE | GEOM_KNOWN | ROI_SLEW
  const gapMask = (typeof info.gapMask === 'number' ? info.gapMask : 0) & 0x3f;

  const hasMetaSetter = typeof wasmExports.chronos_set_ingest_meta === 'function';
  if (hasMetaSetter) {
    const stagedOk = wasmExports.chronos_set_ingest_meta(
      ingestEpoch,
      ingestSeq,
      tLo,
      tHi,
      ageMs,
      metaFlags,
      geom.srcW >>> 0,
      geom.srcH >>> 0,
      geom.cropX0 >>> 0,
      geom.cropY0 >>> 0,
      geom.cropW >>> 0,
      geom.cropH >>> 0,
      (geom.orientation ?? 0) >>> 0,
      (geom.scanDir ?? 0xffffffff) >>> 0,
      geom.dstX >>> 0,
      geom.dstY >>> 0,
      geom.dstW >>> 0,
      geom.dstH >>> 0,
      gapMask,
      geom.native1to1 ? 1 : 0
    );
    if (!stagedOk) {
      const rejectStatus = wave1TelemView ? (wave1TelemView[8] >>> 0) : 2;
      return { committed: false, status: rejectStatus, slotIdx, dispSlot: -1 };
    }
  } else if (
    typeof wasmExports.chronos_set_ingest_pts === 'function' &&
    hasPtsNow
  ) {
    wasmExports.chronos_set_ingest_pts(tUs * 1e-6);
  }

  if (rawPixels !== ringSlotU8[slotIdx]) {
    ringSlotU8[slotIdx].set(rawPixels);
  }

  const prevCommits = wave1CoreAbiValid && wave1TelemView
    ? ((wave1TelemView[5] >>> 0) * 4294967296 + (wave1TelemView[4] >>> 0))
    : -1;
  const expEngIdx = workerExpectedEngineFrame >>> 0;

  // Measure pure kernel compute path: Wasm Pre-Tick + Shifted Lanczos Fiedler + Wasm Post-Tick
  const t0 = performance.now();
  let preTickStarted = false;
  let postTickFinished = false;
  let lambda2 = 0.0;
  let dispSlot = 0;
  try {
    wasmExports.chronos_pre_tick(rawDt);
    preTickStarted = true;
    lambda2 = wasmExports.chronos_solve_fiedler_lanczos();
    dispSlot = wasmExports.chronos_post_tick(rawDt);
    postTickFinished = true;
  } catch (tickErr) {
    if (preTickStarted && !postTickFinished) {
      wasmExports.chronos_reset_tracks();
      clearVisualQueues();
      workerExpectedEngineFrame = 0;
    }
    return {
      committed: false,
      status: 4,
      slotIdx: -1,
      dispSlot: -1,
      error: tickErr && tickErr.message ? tickErr.message : 'mid_tick_error',
    };
  }
  const wasmMs = performance.now() - t0;

  const commitsNow = wave1CoreAbiValid && wave1TelemView
    ? ((wave1TelemView[5] >>> 0) * 4294967296 + (wave1TelemView[4] >>> 0))
    : -1;
  const mOff = slotIdx * RING_META_STRIDE;
  const commitOk = Boolean(
    wave1CoreAbiValid &&
    wave1TelemView &&
    ringMetaView &&
    postTickFinished &&
    commitsNow === prevCommits + 1 &&
    (wave1TelemView[8] >>> 0) === 1 &&
    (wave1TelemView[9] & 1) !== 0 &&
    (wave1TelemView[13] >>> 0) === 0 &&
    (wave1TelemView[2] >>> 0) === (slotIdx >>> 0) &&
    (ringMetaView[mOff] & 1) !== 0 &&
    (ringMetaView[mOff + 1] >>> 0) === (ingestEpoch >>> 0) &&
    (ringMetaView[mOff + 2] >>> 0) === (ingestSeq >>> 0) &&
    (ringMetaView[mOff + 3] >>> 0) === expEngIdx
  );
  if (!commitOk) {
    const rejectStatus = wave1TelemView ? (wave1TelemView[8] >>> 0) : 2;
    return {
      committed: false,
      status: rejectStatus === 1 ? 2 : rejectStatus,
      slotIdx,
      dispSlot: -1,
    };
  }
  workerExpectedEngineFrame = (expEngIdx + 1) >>> 0;

  // Virtual-PTZ Auto-Slew: smoothly pan 1:1 native ROI window when focus Hero target drifts toward edges
  if (
    autoTrackRoiEnabled &&
    nativeRoiEnabled &&
    sourceMode !== 'testbench' &&
    telemView &&
    lastIngestGeom &&
    lastIngestGeom.srcW > frameW &&
    lastIngestGeom.srcH > frameH
  ) {
    const pSlot = resolvePrimarySlot();
    if (pSlot >= 0 && telemView[32 + pSlot * 56] > 0.5) {
      const fx = telemView[32 + pSlot * 56 + 2];
      const fy = telemView[32 + pSlot * 56 + 3];
      let stepX = 0;
      let stepY = 0;
      if (fx < frameW * 0.25) stepX = Math.max(-16, Math.round(fx - frameW * 0.35));
      else if (fx > frameW * 0.75) stepX = Math.min(16, Math.round(fx - frameW * 0.65));
      if (fy < frameH * 0.25) stepY = Math.max(-16, Math.round(fy - frameH * 0.35));
      else if (fy > frameH * 0.75) stepY = Math.min(16, Math.round(fy - frameH * 0.65));
      if (stepX !== 0 || stepY !== 0) {
        const maxX0 = Math.max(0, lastIngestGeom.srcW - frameW);
        const maxY0 = Math.max(0, lastIngestGeom.srcH - frameH);
        const curX0 = roiUseExplicitPx ? roiExplicitX0 : lastIngestGeom.cropX0;
        const curY0 = roiUseExplicitPx ? roiExplicitY0 : lastIngestGeom.cropY0;
        const nx0 = Math.max(0, Math.min(maxX0, curX0 + stepX));
        const ny0 = Math.max(0, Math.min(maxY0, curY0 + stepY));
        if (nx0 !== curX0 || ny0 !== curY0) {
          roiUseExplicitPx = true;
          roiExplicitX0 = nx0;
          roiExplicitY0 = ny0;
          roiSlewPending = true;
        }
      }
    }
  }

  lastDtSecUsed = rawDt;
  ingestElapsedMs += rawDt * 1000.0;
  latestCausalTimeMs = ingestElapsedMs;
  const causalFrameNow = currentCausalEngineFrameU32();
  if (hasPtsNow) {
    const ptsMsNow = tUs * 0.001;
    if (latestCausalPtsMs >= 0.0 && ptsMsNow < latestCausalPtsMs) {
      ringIngestPtsMs.fill(-1.0);
    }
    latestCausalPtsMs = ptsMsNow;
    ringIngestPtsMs[slotIdx] = ptsMsNow;
  } else {
    latestCausalPtsMs = -1.0;
    ringIngestPtsMs.fill(-1.0);
  }
  ringIngestFrameId[slotIdx] = causalFrameNow;
  ringIngestFrameValid[slotIdx] = 1;
  ringIngestTimeMs[slotIdx] = ingestElapsedMs;

  const tCopy0 = performance.now();
  if (hasChronos1000Exports()) {
    if (swarmTelemView && (intOrNull(swarmTelemView[1], 0, 1) === 1 || swarmMode === 2)) {
      swarmRingMicro[slotIdx].set(microTracksView);
      swarmRingField[slotIdx].set(continuumFieldView);
      swarmRingBodies[slotIdx].set(rigidBodiesView);
      swarmRingModal[slotIdx].set(modalFieldView);
      swarmRingTelem[slotIdx].set(swarmTelemView);
      swarmRingValid[slotIdx] = 1;
      if (wave1MetrologyAbiValid && microCovView && microQualityView) {
        covRingBuf[slotIdx].set(microCovView);
        qualRingBuf[slotIdx].set(microQualityView);
        covRingValid[slotIdx] = 1;
      } else {
        covRingValid[slotIdx] = 0;
      }
    } else {
      swarmRingValid[slotIdx] = 0;
      covRingValid[slotIdx] = 0;
    }
  }
  if (wave1BosAbiValid && bosRgbaView && bosTelemView && bosTelemF32View && bosReqMode !== 0) {
    bosRingRgba[slotIdx].set(bosRgbaView);
    const bt = bosRingTelem[slotIdx];
    bt[0] = 1;
    bt[1] = bosTelemView[1];
    bt[2] = bosTelemView[2];
    bt[3] = bosTelemView[3];
    bt[4] = bosTelemView[4];
    bt[5] = bosTelemView[5];
    bt[6] = bosTelemView[6];
    bt[7] = bosTelemView[7];
    bt[8] = bosTelemView[8];
    bt[9] = bosTelemView[9];
    bt[10] = bosTelemView[10];
    bt[11] = bosTelemView[11];
    bt[12] = bosTelemF32View[16];
    bt[13] = bosTelemF32View[17];
    bt[14] = bosTelemF32View[18];
    bt[15] = bosTelemF32View[19];
    bt[16] = bosTelemF32View[20];
    bt[17] = bosTelemF32View[21];
    bt[18] = bosTelemF32View[22];
    bt[19] = bosTelemF32View[23];
    bosRingValid[slotIdx] = 1;
  } else {
    bosRingValid[slotIdx] = 0;
  }
  if (wave2CoreAbiValid && wave2TelemView) {
    const cr = w2CoreRing[slotIdx];
    cr[0] = wave2TelemView[0];
    cr[1] = wave2TelemView[1];
    cr[2] = wave2TelemView[2];
    cr[3] = wave2TelemView[3];
    cr.set(wave2TelemView.subarray(16, 64), 4);
    w2CoreRingValid[slotIdx] = 1;
  } else {
    w2CoreRingValid[slotIdx] = 0;
  }
  if (wave2VibroAbiValid && vibroTelemView && vibroPlanesView && vibroTelemView[0] > 0) {
    vibroRingTelem[slotIdx].set(vibroTelemView);
    vibroRingPlanes[slotIdx].set(vibroPlanesView);
    vibroRingValid[slotIdx] = 1;
  } else {
    vibroRingPlanes[slotIdx].fill(0.0);
    vibroRingValid[slotIdx] = 0;
  }
  if (wave2HodgeAbiValid && hodgeTelemView && hodgePlanesView && hodgeTelemView[0] > 0) {
    hodgeRingTelem[slotIdx].set(hodgeTelemView);
    hodgeRingPlanes[slotIdx].set(hodgePlanesView);
    hodgeRingValid[slotIdx] = 1;
  } else {
    hodgeRingPlanes[slotIdx].fill(0.0);
    hodgeRingValid[slotIdx] = 0;
  }
  const liveConsStamp = wave2CoreAbiValid ? decodeWave2Stamp(32) : null;
  if (wave2ConsAbiValid && conservationTelemView && liveConsStamp && liveConsStamp.valid) {
    consRingTelem[slotIdx].set(conservationTelemView);
    consRingValid[slotIdx] = 1;
    manualConsTestActive = false;
  } else {
    consRingValid[slotIdx] = 0;
  }
  if (
    depthAbiValid &&
    depthTelemView &&
    depthFieldView &&
    (depthReqMode !== 0 || (depthTelemView[1] >>> 0) !== 0)
  ) {
    depthRingTelem[slotIdx].set(depthTelemView);
    depthRingField[slotIdx].set(depthFieldView);
    if ((depthTelemView[2] >>> 0) === 6 && bosRgbaView) {
      depthRingStereoRgba[slotIdx].set(bosRgbaView);
    }
    depthRingValid[slotIdx] = 1;
  } else {
    depthRingField[slotIdx].fill(0.0);
    depthRingValid[slotIdx] = 0;
  }
  if (
    windAbiValid &&
    windTelemView &&
    windPlanesView &&
    (windReqMode !== 0 || (windTelemView[1] >>> 0) !== 0)
  ) {
    windRingTelem[slotIdx].set(windTelemView);
    windRingPlanes[slotIdx].set(windPlanesView);
    windRingValid[slotIdx] = 1;
  } else {
    windRingValid[slotIdx] = 0;
  }
  if (
    elastoAbiValid &&
    elastoTelemView &&
    elastoPlanesView &&
    elastoValidView &&
    (elastoReqMode !== 0 || (elastoTelemView[1] >>> 0) !== 0)
  ) {
    elastoRingTelem[slotIdx].set(elastoTelemView);
    if ((elastoTelemView[1] >>> 0) === 4) {
      if ((elastoTelemView[0] >>> 0) === 3) {
        colorizeAbosToRgba(elastoPlanesView, elastoValidView, elastoRingAbosRgba[slotIdx]);
      } else {
        elastoRingAbosRgba[slotIdx].fill(0);
      }
    } else {
      elastoRingPlanes[slotIdx].set(elastoPlanesView.subarray(0, ELASTO_SURF_PLANES_LEN));
      elastoRingValidMask[slotIdx].set(elastoValidView.subarray(0, FIELD_CELLS));
    }
    elastoRingValid[slotIdx] = 1;
  } else {
    elastoRingValid[slotIdx] = 0;
  }
  lastSwarmCopyMs = performance.now() - tCopy0;
  lastLambda2 = lambda2;

  wasmTimeHist[timeHistIdx] = wasmMs;
  timeHistIdx = (timeHistIdx + 1) % TIME_HIST_LEN;

  const camMotionState = Math.round(telemView[TELEM_GLOBAL + 9]);
  const isFastPass = Boolean(info.fastPass || fastPassActive);
  const fastPassPresent = Boolean(info.fastPassPresent);
  const evTimeSec = hasMediaPts
    ? mediaTimeSec
    : (hasExplicitTSrcUs ? (tUs * 1e-6) : sessionElapsedSec);
  if (camMotionState === 2) {
    if (causalFrameNow !== lastCutEventFrame) {
      lastCutEventFrame = causalFrameNow;
      pushRecEvent('scene_cut', evTimeSec, causalFrameNow, 0, 'SCENE CUT', 0, 0);
    }
    camPanStreakWorker = 0;
    clearVisualQueues();
    if (sourceMode !== 'testbench' && calMode !== 'none') {
      clearCalibrationState();
      recordCalibrationSegmentTransition('scene_cut_reset');
      emitCalibrationState('SCENE CUT · ECA & IMAGE-PLANE CALIBRATION RESET');
    }
  } else if (camMotionState === 1 && sourceMode !== 'testbench') {
    camPanStreakWorker++;
    if (
      camPanStreakWorker >= 4 &&
      (calMode === 'homography' || calMode === 'ruler' || calArmed !== 'none')
    ) {
      clearCalibrationState();
      recordCalibrationSegmentTransition('camera_ptz_pan_reset');
      emitCalibrationState('CAMERA PTZ PAN · IMAGE-PLANE CALIBRATION INVALIDATED (PX UNITS)');
    }
  } else {
    camPanStreakWorker = 0;
  }

  if (sourceMode === 'testbench' && !isChronos1000Scenario(benchScenario)) {
    auditOracleSwaps();
  }

  const fIdx = causalFrameNow;
  let frameHasBridge = false;
  for (let m = 0; m < MAX_TARGETS; m++) {
    const b = 32 + m * 56;
    if (telemView[b] > 0.5) {
      if (telemView[b + 11] > 0.5) frameHasBridge = true;
      if (calMode === 'homography' && calHomographyValid) {
        const kb = TELEM_KINETIC + m * 8;
        const tidM = Math.round(telemView[b + 1]);
        const pv = mapVelocityToPhysical(telemView[b + 2], telemView[b + 3], telemView[kb + 1], telemView[kb + 0]);
        if (pv[2] > 0.5 && pv[0] > 0.08) {
          const fDelta = (fIdx - prevRectFrame[m]) >>> 0;
          if (prevRectTid[m] === tidM && prevRectValid[m] === 1 && fDelta >= 1 && fDelta <= 4) {
            const dth = wrapDeg180(pv[1] - prevRectHeading[m]);
            const spanDt = Math.max(0.002, fDelta * fpsDt);
            const instW = Math.max(-720.0, Math.min(720.0, dth / spanDt));
            slotRectAngVel[m] = 0.62 * slotRectAngVel[m] + 0.38 * instW;
          } else {
            slotRectAngVel[m] = 0.0;
          }
          prevRectHeading[m] = pv[1];
          prevRectTid[m] = tidM;
          prevRectFrame[m] = fIdx;
          prevRectValid[m] = 1;
        } else {
          slotRectAngVel[m] *= 0.78;
        }
      }
    } else {
      prevRectTid[m] = 0;
      prevRectFrame[m] = 0;
      prevRectValid[m] = 0;
      slotRectAngVel[m] = 0.0;
    }
  }
  if (frameHasBridge) totalBridgedEvents++;
  updateSlotRosterEma();

  // Record scope traces + primary track ID for seamless discontinuity breaks
  const pIdxNow = Math.round(telemView[8]);
  const curPipTid = pIdxNow >= 0 && telemView[32 + pIdxNow * 56] > 0.5
    ? Math.round(telemView[32 + pIdxNow * 56 + 1])
    : -1;
  histTremorR[scopeWriteIdx] = telemView[9];
  histTremorH[scopeWriteIdx] = telemView[10];
  histPhaseValid[scopeWriteIdx] = telemView[26] > 0.5 ? 1 : 0;
  histStrobe[scopeWriteIdx] = telemView[14] > 0.5 ? 1 : 0;
  histPipTid[scopeWriteIdx] = curPipTid;
  histLambda2[scopeWriteIdx] = lambda2;
  scopeWriteIdx = (scopeWriteIdx + 1) % SCOPE_HIST;

  // Present-time phase strobe cue
  if (telemView[14] > 0.5 && strobeEnabled && !isFastPass) {
    strobeFlashAlpha = 1.0;
    self.postMessage({
      type: 'audio_strobe',
      resonant: phaseOffsetRad > 0,
      freqHz: telemView[13],
      amp: telemView[12],
    });
  }

  // Multi-arc consensus scene gravity lock (>= 2 consistent free-fall arcs required)
  const gConf = telemView[TELEM_GLOBAL + 1];
  const gPxS2 = telemView[TELEM_GLOBAL + 0];
  if (gConf >= AUTO_G_LOCK_CONF && gPxS2 > 50.0) {
    const hadLock = calAutoGLocked;
    calAutoGNormH = gPxS2 / frameH;
    calAutoGLocked = true;
    if (calMode === 'auto_g' && !hadLock) {
      recordCalibrationSegmentTransition('auto_g_locked');
      emitCalibrationState();
    }
  } else if (calAutoGLocked && gConf < 0.35) {
    calAutoGLocked = false;
    if (calMode === 'auto_g') {
      recordCalibrationSegmentTransition('auto_g_provisional');
      emitCalibrationState();
    }
  } else if (calMode === 'auto_g' && !calAutoGLocked && gPxS2 > 50.0 && Math.round(telemView[TELEM_GLOBAL + 11]) === 1 && calAutoGNormH === 0.0) {
    calAutoGNormH = gPxS2 / frameH;
    emitCalibrationState();
  }

  // Emit per-target deduplicated Mode C strikes and schedule synchronized t - K shockwave rings!
  const percMask = Math.round(telemView[15]);
  if (percMask !== 0) {
    const effSampleRate = Math.max(8.0, sourceMode === 'testbench' ? smoothFps : sourceFps);
    const nowEpochMs = performance.timeOrigin + performance.now();
    for (let m = 0; m < MAX_TARGETS; m++) {
      const ebase = TELEM_EXCUR + m * 8;
      if (telemView[ebase] > 0.5) {
        totalStrikeEvents++;
        const kbase = TELEM_KINETIC + m * 8;
        const zone = Math.round(telemView[ebase + 1]) & 3;
        const leadFrames = telemView[ebase + 2];
        const vel = telemView[ebase + 3];
        const sxPos = telemView[ebase + 4];
        const syPos = telemView[ebase + 5];
        const rawDefl = telemView[kbase + 3];
        const rawRest = telemView[kbase + 4];
        const rawIn = telemView[kbase + 5];
        slotLastStrikeX[m] = sxPos;
        slotLastStrikeY[m] = syPos;
        pushRecEvent(
          'strike',
          evTimeSec,
          fIdx,
          Math.round(telemView[32 + m * 56 + 1]),
          'L1 STRIKE',
          sxPos,
          syPos
        );

        const maxValidRest = sourceMode === 'testbench' ? 1.65 : 1.18;
        if (rawRest >= 0.06 && rawRest <= maxValidRest && Math.abs(rawDefl) >= 3.0) {
          const rectSt = rectifyStrikeKinematics(sxPos, syPos, rawIn, rawDefl, rawRest);
          if (rectSt[2] <= maxValidRest) {
            lastStrikeDeflectDeg = rectSt[1];
            lastStrikeRestitution = rectSt[2];
            lastStrikeTid = Math.round(telemView[32 + m * 56 + 1]);
            lastStrikeFrame = (fIdx + Math.round(leadFrames)) >>> 0;
            lastStrikeValid = true;
          }
        }

        if (labMode === 2) {
          shockTriggerFrame[shockWriteHead] = (fIdx + Math.round(leadFrames)) >>> 0;
          shockTriggerArmed[shockWriteHead] = 1;
          shockX[shockWriteHead] = sxPos;
          shockY[shockWriteHead] = syPos;
          shockZone[shockWriteHead] = zone;
          shockVel[shockWriteHead] = vel;
          shockInDeg[shockWriteHead] = rawIn;
          shockDeflectDeg[shockWriteHead] = rawDefl;
          shockRestitution[shockWriteHead] = rawRest;
          shockAge[shockWriteHead] = -1;
          shockWriteHead = (shockWriteHead + 1) % SHOCK_POOL;

          if (!isFastPass) {
            self.postMessage({
              type: 'audio_percussion',
              zone,
              leadSec: leadFrames / effSampleRate,
              velocity: vel,
              sentEpochMs: nowEpochMs,
            });
          }
        }
      }
    }
  }

  if (telemView[TELEM_GLOBAL + 2] > 2.0) {
    const rectSc = rectifyScatterKinematics(
      telemView[TELEM_GLOBAL + 3],
      telemView[TELEM_GLOBAL + 4],
      telemView[TELEM_GLOBAL + 5],
      telemView[TELEM_GLOBAL + 6],
      telemView[TELEM_GLOBAL + 2]
    );
    lastScatterDeg = rectSc[2];
    lastScatterFrame = Math.round(telemView[TELEM_GLOBAL + 7]);
  }

  updatePersistenceState(fpsDt);
  pushPhysicsPhaseSample();

  // Capture apex, constraint-jump, and BOS-disturbed timeline markers
  for (let m = 0; m < MAX_TARGETS; m++) {
    const b = 32 + m * 56;
    if (telemView[b] > 0.5) {
      const tidM = Math.round(telemView[b + 1]);
      const apexF = safeFinite(telemView[b + 17]);
      if (
        apexF > 0.0 &&
        apexF <= 1.2 &&
        prevSlotApexVal[m] > 1.2 &&
        ((fIdx - prevSlotApexEventFrame[m]) >>> 0) >= 8
      ) {
        prevSlotApexEventFrame[m] = fIdx;
        pushRecEvent('apex', evTimeSec, fIdx, tidM, `T${tidM} APEX`, telemView[b + 15], telemView[b + 16]);
      }
      prevSlotApexVal[m] = apexF;

      const pb = TELEM_PHYSICS_START + m * TELEM_PHYSICS_STRIDE;
      const jCode = Math.max(0, Math.min(4, Math.round(safeFinite(telemView[pb + 15]))));
      if (jCode >= 1 && jCode <= 4 && jCode !== prevSlotJumpEventCode[m]) {
        pushRecEvent(
          'jump',
          evTimeSec,
          fIdx,
          tidM,
          `T${tidM} ${JUMP_EVENT_LABELS[jCode] || 'JUMP'}`,
          telemView[b + 2],
          telemView[b + 3]
        );
      }
      prevSlotJumpEventCode[m] = jCode;
    } else {
      prevSlotApexVal[m] = 0.0;
      prevSlotJumpEventCode[m] = 0;
    }
  }
  if (wave1BosAbiValid && bosTelemView && bosReqMode !== 0) {
    const bosSt = bosTelemView[2] >>> 0;
    if (bosSt === 3 && prevBosEventState !== 3) {
      pushRecEvent('bos_disturbed', evTimeSec, fIdx, 0, 'BOS DISTURBED', 0, 0);
    }
    prevBosEventState = bosSt;
  } else {
    prevBosEventState = 0;
  }
  if (!isFastPass || fastPassPresent) {
    emitTimelineEventsIfNeeded(false);
  }

  // Render all OffscreenCanvases inside Worker isolate (isolated from committed frame ack)
  if (!isFastPass || fastPassPresent) {
    try {
      scheduleOffscreenHudPresent(dispSlot, true, false);
    } catch (_hudErr) {
      // Non-fatal HUD presentation fault must never invert an already committed frame into ingest_rejected
    }
  }

  // Zero-allocation per-frame scientific telemetry recording (1:1 frame fidelity, 56 columns)
  if (recActive) {
    const effRateRec = sourceMode === 'testbench' ? 60.0 : Math.max(8.0, sourceFps);
    const pSlot = Math.round(telemView[8]);
    const ro = recHead * REC_COLS;
    recElapsedSec += Math.min(1.0, rawDt);

    recSlab[ro + 0] = fIdx;
    recSlab[ro + 1] = recElapsedSec;
    recSlab[ro + 2] = frameW;
    recSlab[ro + 3] = frameH;
    recSlab[ro + 4] = Math.round(telemView[1]);
    recSlab[ro + 5] = labMode;
    recSlab[ro + 6] = wasmMs;
    recSlab[ro + 7] = Math.round(telemView[3]);
    recSlab[ro + 8] = telemView[4];
    recSlab[ro + 9] = telemView[23];
    const oracleAvailRec = sourceMode === 'testbench' && !isChronos1000Scenario(benchScenario);
    recSlab[ro + 10] = Math.round(telemView[5]);
    recSlab[ro + 11] = oracleAvailRec ? oraclePadicSwaps : Number.NaN;
    recSlab[ro + 12] = Math.round(telemView[6]);
    recSlab[ro + 13] = inPillarCollisions;
    recSlab[ro + 14] = lambda2;
    recSlab[ro + 15] = telemView[24];
    recSlab[ro + 16] = telemView[25];

    if (pSlot >= 0 && telemView[32 + pSlot * 56] > 0.5) {
      const b = 32 + pSlot * 56;
      const kb = TELEM_KINETIC + pSlot * 8;
      const fxPx = telemView[b + 2];
      const fyPx = telemView[b + 3];
      const hi = Math.round(telemView[b + 12]) & 0xffff;
      const lo = Math.round(telemView[b + 13]) & 0xffff;
      recSlab[ro + 17] = Math.round(telemView[b + 1]);
      recSlab[ro + 18] = hi * 65536.0 + lo;
      recSlab[ro + 19] = Math.round(telemView[b + 14]);
      recSlab[ro + 20] = telemView[b + 4];
      recSlab[ro + 21] = telemView[b + 5];
      recSlab[ro + 22] = fxPx;
      recSlab[ro + 23] = fyPx;
      recSlab[ro + 24] = telemView[b + 6];
      recSlab[ro + 25] = telemView[b + 7];
      recSlab[ro + 26] = telemView[b + 10];
      recSlab[ro + 27] = telemView[b + 11];
      recSlab[ro + 36] = telemView[b + 17] * (1000.0 / effRateRec);

      const physVel = mapVelocityToPhysical(fxPx, fyPx, telemView[kb + 1], telemView[kb + 0]);
      recSlab[ro + 38] = physVel[1];
      recSlab[ro + 39] = physVel[0];
      recSlab[ro + 40] = (calMode === 'homography' && calHomographyValid && physVel[2] > 0.5)
        ? slotRectAngVel[pSlot]
        : telemView[kb + 2];
      if (telemView[kb + 6] > -9.0e5 && telemView[kb + 7] > -9.0e5) {
        const pend = computePendulumKinematics(fxPx, fyPx, telemView[kb + 6], telemView[kb + 7]);
        recSlab[ro + 41] = pend[0];
      } else {
        recSlab[ro + 41] = 0.0;
      }
      const sxRec = slotLastStrikeX[pSlot] > 0.0 ? slotLastStrikeX[pSlot] : fxPx;
      const syRec = slotLastStrikeY[pSlot] > 0.0 ? slotLastStrikeY[pSlot] : fyPx;
      const stRect = rectifyStrikeKinematics(sxRec, syRec, telemView[kb + 5], telemView[kb + 3], telemView[kb + 4]);
      recSlab[ro + 42] = stRect[1];
      recSlab[ro + 43] = stRect[2];
      const mPt = mapPixelToMetric(fxPx, fyPx);
      recSlab[ro + 44] = mPt[0];
      recSlab[ro + 45] = mPt[1];
      recSlab[ro + 46] = calSegmentId;
      recSlab[ro + 47] = physVel[2];
      const pb = TELEM_PHYSICS_START + pSlot * TELEM_PHYSICS_STRIDE;
      recSlab[ro + 48] = Math.max(0, Math.min(8, Math.round(safeFinite(telemView[pb + 4]))));
      recSlab[ro + 49] = safeFinite(telemView[pb + 5]);
      recSlab[ro + 50] = safeFinite(telemView[pb + 6]);
      recSlab[ro + 51] = safeFinite(telemView[pb + 7]);
      recSlab[ro + 52] = safeFinite(telemView[pb + 8]);
      recSlab[ro + 53] = safeFinite(telemView[pb + 10]);
    } else {
      for (let c = 17; c <= 27; c++) recSlab[ro + c] = 0.0;
      recSlab[ro + 36] = 0.0;
      for (let c = 38; c <= 45; c++) recSlab[ro + c] = 0.0;
      recSlab[ro + 46] = calSegmentId;
      recSlab[ro + 47] = isMetricCalibrated() ? 1.0 : 0.0;
      for (let c = 48; c <= 53; c++) recSlab[ro + c] = 0.0;
    }

    recSlab[ro + 54] = safeFinite(telemView[TELEM_ENSEMBLE_START + 0]);
    recSlab[ro + 55] = safeFinite(telemView[TELEM_ENSEMBLE_START + 11]);
    recSlab[ro + 28] = telemView[9];
    recSlab[ro + 29] = telemView[10];
    recSlab[ro + 30] = telemView[11] * (180.0 / Math.PI);
    recSlab[ro + 31] = telemView[13];
    recSlab[ro + 32] = telemView[12];
    recSlab[ro + 33] = telemView[26] > 0.5 ? 1 : 0;
    recSlab[ro + 34] = telemView[14] > 0.5 ? 1 : 0;
    recSlab[ro + 35] = Math.round(telemView[15]);
    recSlab[ro + 37] = telemView[19];

    recHead = (recHead + 1) % REC_MAX_FRAMES;
    if (recCount < REC_MAX_FRAMES) recCount++;
    recTotalWritten++;
  }

  tickCounter++;
  if ((!isFastPass && (tickCounter % 3 === 0 || sourceMode !== 'testbench')) || (isFastPass && fastPassPresent)) {
    try {
      emitTelemetrySnapshot();
    } catch (_telemErr) {
      // Telemetry post failure must not override an already committed frame ack
    }
  }
  return {
    committed: Boolean(commitOk),
    status: wave1TelemView ? (wave1TelemView[8] >>> 0) : 1,
    slotIdx,
    dispSlot,
  };
}

function configureResolutionViews(mode, forceReset = false) {
  refreshChronos1000Views();
  refreshWave1Views();
  const prevResMode = resMode;
  const targetMode = mode ? 1 : 0;
  wasmExports.chronos_set_resolution(targetMode);
  resMode = wasmExports.chronos_get_resolution();
  const didReset = Boolean(forceReset) || resMode !== prevResMode;
  if (didReset) {
    workerExpectedEngineFrame = 0;
  }
  frameW = wasmExports.chronos_get_frame_w();
  frameH = wasmExports.chronos_get_frame_h();
  pipDim = wasmExports.chronos_get_pip_dim();
  resScale = resMode === 1 ? 2 : 1;
  framePixels = frameW * frameH;
  frameBytes = framePixels * 4;

  if (didReset || dispStageCanvas.width !== frameW || dispStageCanvas.height !== frameH) {
    dispStageCanvas.width = frameW;
    dispStageCanvas.height = frameH;
  }
  if (didReset || pipStageCanvas.width !== pipDim || pipStageCanvas.height !== pipDim) {
    pipStageCanvas.width = pipDim;
    pipStageCanvas.height = pipDim;
  }
  if (didReset || simCanvas.width !== frameW || simCanvas.height !== frameH) {
    simCanvas.width = frameW;
    simCanvas.height = frameH;
  }
  if (didReset || ecaStageCanvas.width !== frameW || ecaStageCanvas.height !== frameH) {
    ecaStageCanvas.width = frameW;
    ecaStageCanvas.height = frameH;
  }

  ecaOverlayBuf = ecaOverlayBackingBuf.subarray(0, frameBytes);
  ecaOverlayImgData = new ImageData(ecaOverlayBuf, frameW, frameH);

  for (let s = 0; s < RING_LEN; s++) {
    const byteOff = wasmExports.chronos_get_ring_slot_ptr(s);
    ringSlotByteOffsets[s] = byteOff;
    ringSlotU8[s] = new Uint8Array(wasmMemory.buffer, byteOff, frameBytes);
    ringSlotImgData[s] = new ImageData(
      new Uint8ClampedArray(wasmMemory.buffer, byteOff, frameBytes),
      frameW,
      frameH
    );
  }

  ecaResView = new Float32Array(wasmMemory.buffer, wasmExports.chronos_get_eca_res_ptr(), framePixels);
  const pipBytes = pipDim * pipDim * 4;
  pipImgData = new ImageData(
    new Uint8ClampedArray(wasmMemory.buffer, wasmExports.chronos_get_pip_rgba_ptr(), pipBytes),
    pipDim,
    pipDim
  );
  if (typeof wasmExports.chronos_get_slot_meta_ptr === 'function') {
    slotMetaView = new Float32Array(
      wasmMemory.buffer,
      wasmExports.chronos_get_slot_meta_ptr(),
      MAX_TARGETS * 8
    );
  }
  if (typeof wasmExports.chronos_get_all_pip_rgba_ptr === 'function') {
    const allPipBase = wasmExports.chronos_get_all_pip_rgba_ptr();
    for (let m = 0; m < MAX_TARGETS; m++) {
      slotPipImgData[m] = new ImageData(
        new Uint8ClampedArray(wasmMemory.buffer, allPipBase + m * MAX_PIP_BYTES, pipBytes),
        pipDim,
        pipDim
      );
    }
  }
  if (didReset) {
    slotIdleDrawn.fill(0);
    wasmTimeHist.fill(0.0);
    timeHistIdx = 0;
    tickCounter = 0;
    clearVisualQueues();
    resetTestbenchState();
  }
}

function renderSecondaryScopes(slotFast, fullHist) {
  renderEulerianPip();
  renderAllSlotPips(slotFast);
  renderHilbertScope();
  renderSpectralAndRingScope();
  renderPersistenceTimeline(fullHist);
  renderPhysicsPhaseCanvas(fullHist);
}

function cancelPendingHudRender() {
  clearTimeout(hudPresentWatchdogId);
  hudPresentWatchdogId = 0;
  clearTimeout(hudSettleTimerId);
  hudSettleTimerId = 0;
  if (hudPresentRafId && typeof self.cancelAnimationFrame === 'function') {
    self.cancelAnimationFrame(hudPresentRafId);
  }
  hudPresentRafId = 0;
  pendingHudPresentDirty = false;
  pendingHudPresentSlot = -1;
  lastCommittedHudSlot = -1;
  pendingHudSlotFast = true;
  pendingHudFullHist = false;
}

function isCommittedRingSlot(slot) {
  if (!Number.isInteger(slot) || slot < 0 || slot >= RING_LEN) return false;
  if (wave1CoreAbiValid && ringMetaView) {
    return (ringMetaView[slot * RING_META_STRIDE] & 1) !== 0;
  }
  return true;
}

function currentCausalEngineFrameU32() {
  if (wave1CoreAbiValid && wave1TelemView && ringMetaView) {
    const lastSlot = wave1TelemView[2] >>> 0;
    if (lastSlot < RING_LEN && (ringMetaView[lastSlot * RING_META_STRIDE] & 1) !== 0) {
      return ringMetaView[lastSlot * RING_META_STRIDE + 3] >>> 0;
    }
    return 0;
  }
  return telemView ? (Math.round(telemView[0]) >>> 0) : 0;
}

function currentDisplaySlot() {
  if (wave1CoreAbiValid && wave1TelemView && ringMetaView) {
    const w1Disp = wave1TelemView[3] >>> 0;
    if (w1Disp < RING_LEN && isCommittedRingSlot(w1Disp)) {
      return w1Disp;
    }
    return -1;
  }
  return telemView ? (intOrNull(telemView[2], 0, RING_LEN - 1) ?? -1) : -1;
}

function hasAnyRenderedFrame() {
  if (!wasmExports || !telemView) return false;
  if (wave1CoreAbiValid && wave1TelemView && ringMetaView) {
    const w1Disp = wave1TelemView[3] >>> 0;
    return w1Disp < RING_LEN && isCommittedRingSlot(w1Disp);
  }
  return currentCausalEngineFrameU32() > 0 || tickCounter > 0;
}

function flushPendingHudRender() {
  clearTimeout(hudPresentWatchdogId);
  hudPresentWatchdogId = 0;
  clearTimeout(hudSettleTimerId);
  hudSettleTimerId = 0;
  if (hudPresentRafId && typeof self.cancelAnimationFrame === 'function') {
    self.cancelAnimationFrame(hudPresentRafId);
  }
  hudPresentRafId = 0;
  if (!pendingHudPresentDirty || !wasmExports || !telemView || !hasAnyRenderedFrame()) {
    pendingHudPresentDirty = false;
    pendingHudPresentSlot = -1;
    return;
  }
  const w1Disp = wave1CoreAbiValid && wave1TelemView ? (wave1TelemView[3] >>> 0) : -1;
  const dSlot = wave1CoreAbiValid && wave1TelemView && ringMetaView && isCommittedRingSlot(w1Disp)
    ? w1Disp
    : (pendingHudPresentSlot >= 0 && pendingHudPresentSlot < RING_LEN && isCommittedRingSlot(pendingHudPresentSlot)
        ? pendingHudPresentSlot
        : Math.round(telemView[2]));
  if (!isCommittedRingSlot(dSlot)) {
    pendingHudPresentDirty = false;
    pendingHudPresentSlot = -1;
    return;
  }
  const slotFast = pendingHudSlotFast;
  const fullHist = pendingHudFullHist;
  pendingHudPresentDirty = false;
  pendingHudPresentSlot = -1;
  pendingHudSlotFast = true;
  pendingHudFullHist = false;
  renderMainHud(dSlot);
  if (!showcaseMinimalScopes) {
    renderSecondaryScopes(slotFast, fullHist);
  }
  lastCommittedHudSlot = dSlot;
  if (!inWorkerRafTick && sourceMode !== 'testbench') {
    // Software 2D OffscreenCanvas in headless Chromium can drop a compositor push under burst load;
    // re-present only after stream/UI activity settles (>90 ms) with zero steady-state overhead.
    hudSettleTimerId = setTimeout(() => {
      hudSettleTimerId = 0;
      if (!wasmExports || !telemView || sourceMode === 'testbench' || pendingHudPresentDirty) return;
      if (!isCommittedRingSlot(lastCommittedHudSlot) || !ringSlotImgData[lastCommittedHudSlot]) return;
      if (!hasAnyRenderedFrame()) return;
      renderMainHud(lastCommittedHudSlot);
    }, 90);
  }
}

function scheduleOffscreenHudPresent(dispSlot, slotFast, fullHist) {
  clearTimeout(hudSettleTimerId);
  hudSettleTimerId = 0;
  if (inWorkerRafTick || typeof self.requestAnimationFrame !== 'function') {
    if (inWorkerRafTick) cancelPendingHudRender();
    renderMainHud(dispSlot);
    if (!showcaseMinimalScopes) {
      renderSecondaryScopes(slotFast, fullHist);
    }
    return;
  }
  pendingHudPresentSlot = dispSlot;
  pendingHudSlotFast = pendingHudPresentDirty ? (pendingHudSlotFast && slotFast) : slotFast;
  pendingHudFullHist = pendingHudFullHist || fullHist;
  pendingHudPresentDirty = true;
  if (!hudPresentRafId) {
    hudPresentRafId = self.requestAnimationFrame(flushPendingHudRender);
    clearTimeout(hudPresentWatchdogId);
    hudPresentWatchdogId = setTimeout(flushPendingHudRender, 80);
  }
}

function redrawPausedHudIfReady() {
  if (hasAnyRenderedFrame()) {
    const dispSlot = Math.round(telemView[2]);
    scheduleOffscreenHudPresent(dispSlot, false, true);
    emitTelemetrySnapshot();
  }
}

function commitSurveyedCalibrationFromDraft(targetMode, reason) {
  if (targetMode === 'ruler') {
    const spanPx = Math.hypot(
      (calDraftPtsNormX[1] - calDraftPtsNormX[0]) * frameW,
      (calDraftPtsNormY[1] - calDraftPtsNormY[0]) * frameH
    );
    if (spanPx < 4.0) {
      calArmed = 'none';
      calDraftCount = 0;
      emitCalibrationState('RULER REJECTED (< 4 px SPAN) · PRIOR CALIBRATION PRESERVED');
      return false;
    }
    calRulerMeters = calDraftRulerMeters;
    calPtsNormX[0] = calDraftPtsNormX[0];
    calPtsNormY[0] = calDraftPtsNormY[0];
    calPtsNormX[1] = calDraftPtsNormX[1];
    calPtsNormY[1] = calDraftPtsNormY[1];
    calPtCount = 2;
    calDraftCount = 0;
    calArmed = 'none';
    calMode = 'ruler';
    calProvenance = 'surveyed';
    calPresetName = '';
    calHomographyValid = false;
    recordCalibrationSegmentTransition(reason || 'surveyed_ruler');
    emitCalibrationState();
    return true;
  }

  if (targetMode === 'homography') {
    const solved = solveHomography4Pt(calDraftPlaneW, calDraftPlaneH, calDraftPtsNormX, calDraftPtsNormY, true);
    if (!solved) {
      calArmed = 'none';
      calDraftCount = 0;
      emitCalibrationState('INVALID QUAD (NON-CONVEX / HORIZON) · PRIOR CALIBRATION PRESERVED');
      return false;
    }
    calPlaneW = calDraftPlaneW;
    calPlaneH = calDraftPlaneH;
    for (let i = 0; i < 4; i++) {
      calPtsNormX[i] = calDraftPtsNormX[i];
      calPtsNormY[i] = calDraftPtsNormY[i];
    }
    calPtCount = 4;
    calDraftCount = 0;
    calArmed = 'none';
    calHomographyValid = true;
    calMode = 'homography';
    calProvenance = 'surveyed';
    calPresetName = '';
    recordCalibrationSegmentTransition(reason || 'surveyed_homography');
    emitCalibrationState();
    return true;
  }
  return false;
}

function syncSwarmConfigToMain(sc) {
  self.postMessage({
    type: 'swarm_config_sync',
    scenario: sc,
    resMode,
    frameW,
    frameH,
    pipDim,
    swarmMode,
    userSwarmMode,
    denseSwarmOverrideActive,
    swarmActive: Boolean(hasChronos1000Exports() && swarmTelemView && swarmTelemView[1] > 0.5),
    fieldOverlayMode,
    showHeroTargets,
    overlaySyncSeq: lastOverlaySyncSeq,
  });
}

function applyScenarioResolutionAndSwarm(sc) {
  if (sourceMode === 'testbench' && isChronos1000Scenario(sc)) {
    if (resMode !== 1 && wasmExports) {
      configureResolutionViews(1);
      self.postMessage({
        type: 'resolution_changed',
        resMode,
        frameW,
        frameH,
        pipDim,
      });
      emitCalibrationState();
    }
    denseSwarmOverrideActive = true;
    if (swarmMode !== 2) {
      swarmMode = 2;
      if (hasChronos1000Exports()) {
        wasmExports.chronos_set_swarm_mode(2);
      }
    }
    syncSwarmConfigToMain(sc);
    return;
  }

  if (denseSwarmOverrideActive) {
    denseSwarmOverrideActive = false;
    if (swarmMode !== userSwarmMode) {
      swarmMode = userSwarmMode;
      swarmRingValid.fill(0);
      clearActiveInspectionSelection();
      if (hasChronos1000Exports()) {
        wasmExports.chronos_set_swarm_mode(swarmMode);
      }
    }
    syncSwarmConfigToMain(sc);
  }
}

function applyTestbenchScenarioSwitch(sc) {
  benchScenario = sc || 'pillar_crossing';
  wasmTimeHist.fill(0.0);
  timeHistIdx = 0;
  tickCounter = 0;
  clearVisualQueues();
  applyScenarioResolutionAndSwarm(benchScenario);
  if (wasmExports) wasmExports.chronos_reset_tracks();
  resetTestbenchState();
  redrawPausedHudIfReady();
  emitTelemetrySnapshot();
}

function findLiveMicroTrackIndexByTid(tid, confirmedOnly = false) {
  if (!microTracksView || !Number.isInteger(tid) || tid < 1 || tid > 16777215) return -1;
  const maxState = confirmedOnly ? 1 : 2;
  for (let i = 0; i < MAX_MICRO; i++) {
    const b = i * MICRO_STRIDE;
    const st = intOrNull(microTracksView[b + 5], 1, maxState);
    const curTid = intOrNull(microTracksView[b], 1, 16777215);
    if (st !== null && curTid === tid) {
      return i;
    }
  }
  return -1;
}

function inspectFieldPointAt(rawX, rawY, reqSeq = null, rawNormX = null, rawNormY = null) {
  const maxX = Math.max(0.0, frameW - 1e-3);
  const maxY = Math.max(0.0, frameH - 1e-3);
  const x = Math.max(0.0, Math.min(maxX, Number.isFinite(rawX) ? rawX : frameW * 0.5));
  const y = Math.max(0.0, Math.min(maxY, Number.isFinite(rawY) ? rawY : frameH * 0.5));
  const abiOk = hasChronos1000Exports();

  lastInspectResSeq = reqSeq !== null ? Math.max(lastInspectResSeq + 1, reqSeq) : lastInspectResSeq + 1;
  lastInspectReqSeq = reqSeq !== null ? reqSeq : lastInspectResSeq;
  lastInspectReqX = x;
  lastInspectReqY = y;
  lastInspectReqNormX = frameW > 0 ? x / frameW : (Number.isFinite(rawNormX) ? Math.max(0.0, Math.min(1.0, rawNormX)) : 0.0);
  lastInspectReqNormY = frameH > 0 ? y / frameH : (Number.isFinite(rawNormY) ? Math.max(0.0, Math.min(1.0, rawNormY)) : 0.0);

  // Map click from displayed ring slot (t-effK) onto exact live confirmed track identity
  let bestHistTid = 0;
  const qScale = Math.max(0.25, frameW / 640.0);
  const dispSlot = telemView ? Math.round(telemView[2]) : -1;
  if (dispSlot >= 0 && dispSlot < RING_LEN && swarmRingValid[dispSlot] === 1) {
    const dispMicro = swarmRingMicro[dispSlot];
    let bestD2 = (16.0 * qScale) * (16.0 * qScale);
    for (let i = 0; i < MAX_MICRO; i++) {
      const b = i * MICRO_STRIDE;
      const st = intOrNull(dispMicro[b + 5], 1, 1);
      const candTid = intOrNull(dispMicro[b], 1, 16777215);
      if (st === null || candTid === null) continue;
      const px = dispMicro[b + 1];
      const py = dispMicro[b + 2];
      if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      const dx = px - x;
      const dy = py - y;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD2) {
        bestD2 = d2;
        bestHistTid = candTid;
      }
    }
  }

  if (!abiOk) {
    inspectedMicroTid = 0;
    lastInspectStatus = 'unavailable';
    redrawPausedHudIfReady();
    emitTelemetrySnapshot();
    return;
  }

  if (bestHistTid > 0) {
    const liveIdx = findLiveMicroTrackIndexByTid(bestHistTid, true);
    let pinnedOk = false;
    if (liveIdx >= 0) {
      let ret = 0;
      if (typeof wasmExports.chronos_inspect_micro_track === 'function') {
        ret = wasmExports.chronos_inspect_micro_track(bestHistTid);
      } else {
        const lb = liveIdx * MICRO_STRIDE;
        const lx = Math.max(0.0, Math.min(maxX, microTracksView[lb + 1]));
        const ly = Math.max(0.0, Math.min(maxY, microTracksView[lb + 2]));
        ret = wasmExports.chronos_inspect_field_point(lx, ly);
      }
      const retTid = intOrNull(ret, 1, 16777215);
      if (retTid === bestHistTid) {
        inspectedMicroTid = bestHistTid;
        lastInspectStatus = 'pinned';
        wasmExports.chronos_set_focus_target(bestHistTid);
        pinnedOk = true;
      }
    }
    if (!pinnedOk) {
      inspectedMicroTid = 0;
      lastInspectStatus = 'expired';
      wasmExports.chronos_set_focus_target(0);
    }
  } else {
    const ret = wasmExports.chronos_inspect_field_point(x, y);
    const retTid = intOrNull(ret, 1, 16777215);
    const candTid = retTid ?? 0;
    if (candTid > 0 && findLiveMicroTrackIndexByTid(candTid, true) >= 0) {
      inspectedMicroTid = candTid;
      lastInspectStatus = 'live';
      wasmExports.chronos_set_focus_target(candTid);
    } else {
      inspectedMicroTid = 0;
      lastInspectStatus = 'miss';
      wasmExports.chronos_set_focus_target(0);
    }
  }

  redrawPausedHudIfReady();
  emitTelemetrySnapshot();
}

function handleHudCanvasClick(normX, normY, intent = '', expectedCalArmed = '') {
  if (!wasmExports || !telemView) return;

  if (calArmed !== 'none' && expectedCalArmed && expectedCalArmed !== calArmed) {
    return;
  }

  // Stage interactive 2-Pt Ruler & 4-Pt Plane Homography clicks in draft buffers so Esc/invalid preserves prior state!
  if (calArmed === 'ruler') {
    calDraftPtsNormX[calDraftCount] = normX;
    calDraftPtsNormY[calDraftCount] = normY;
    calDraftCount++;
    if (calDraftCount >= 2) {
      commitSurveyedCalibrationFromDraft('ruler', 'canvas_ruler');
    } else {
      emitCalibrationState();
    }
    redrawPausedHudIfReady();
    return;
  }

  if (calArmed === 'homography') {
    calDraftPtsNormX[calDraftCount] = normX;
    calDraftPtsNormY[calDraftCount] = normY;
    calDraftCount++;
    if (calDraftCount >= 4) {
      commitSurveyedCalibrationFromDraft('homography', 'canvas_homography');
    } else {
      emitCalibrationState();
    }
    redrawPausedHudIfReady();
    return;
  }

  // Never let a stale queued calibration click fall through into Hero target focus pinning.
  if (intent === 'calibration') {
    return;
  }

  const cx = normX * frameW;
  const cy = normY * frameH;
  const hitRadius = 28.0 * resScale;
  let bestDist = hitRadius;
  let bestTid = 0;

  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] < 0.5) continue;
    const fx = telemView[base + 2];
    const fy = telemView[base + 3];
    const d = Math.hypot(cx - fx, cy - fy);
    if (d <= bestDist) {
      bestDist = d;
      bestTid = Math.round(telemView[base + 1]);
    }
  }

  const curFocus = wasmExports.chronos_get_focus_target();
  if (bestTid > 0 && bestTid !== curFocus) {
    wasmExports.chronos_set_focus_target(bestTid);
  } else {
    wasmExports.chronos_set_focus_target(0);
  }

  // Re-render canvases & KPI snapshot immediately if video is paused
  redrawPausedHudIfReady();
}

function cycleFocusTarget() {
  if (!wasmExports || !telemView) return;
  const activeIds = [];
  for (let m = 0; m < MAX_TARGETS; m++) {
    const base = 32 + m * 56;
    if (telemView[base] > 0.5) {
      activeIds.push(Math.round(telemView[base + 1]));
    }
  }
  activeIds.sort((a, b) => a - b);
  const curFocus = wasmExports.chronos_get_focus_target();
  if (activeIds.length === 0) {
    wasmExports.chronos_set_focus_target(0);
    return;
  }
  if (curFocus === 0) {
    wasmExports.chronos_set_focus_target(activeIds[0]);
  } else {
    const idx = activeIds.indexOf(curFocus);
    if (idx < 0 || idx + 1 >= activeIds.length) {
      wasmExports.chronos_set_focus_target(0);
    } else {
      wasmExports.chronos_set_focus_target(activeIds[idx + 1]);
    }
  }
  redrawPausedHudIfReady();
}

function resetTestbenchRsState() {
  tbRsFrameSeq = 0;
  tbRsAudioSampleCursor = 0;
  tbRsWasActive = false;
}

function isTestbenchRsStimulusActive() {
  return (
    sourceMode === 'testbench' &&
    !isChronos1000Scenario(benchScenario) &&
    rsCalibConfig.mode > 0 &&
    Number.isFinite(rsCalibConfig.dtRowUs) &&
    rsCalibConfig.dtRowUs > 0 &&
    Number.isFinite(rsCalibConfig.exposureUs) &&
    rsCalibConfig.exposureUs > 0
  );
}

function ensureTestbenchRsTexTables() {
  const npix = frameW * frameH;
  if (tbRsTexW === frameW && tbRsTexH === frameH && tbRsSinA && tbRsFrameBuf) {
    return;
  }
  tbRsTexW = frameW;
  tbRsTexH = frameH;
  tbRsSinA = new Float32Array(npix);
  tbRsCosA = new Float32Array(npix);
  tbRsCosB = new Float32Array(npix);
  tbRsSinB = new Float32Array(npix);
  tbRsSinC = new Float32Array(npix);
  tbRsCosC = new Float32Array(npix);
  tbRsFrameBuf = new Uint8ClampedArray(npix * 4);
  const ax = 0.27, ay = 0.13, bx = 0.19, by = -0.31, cx = 0.37, cy = 0.29;
  for (let y = 0; y < frameH; y++) {
    const row = y * frameW;
    for (let x = 0; x < frameW; x++) {
      const idx = row + x;
      const pA = ax * x + ay * y;
      const pB = bx * x + by * y;
      const pC = cx * x + cy * y;
      tbRsSinA[idx] = Math.sin(pA);
      tbRsCosA[idx] = Math.cos(pA);
      tbRsCosB[idx] = Math.cos(pB);
      tbRsSinB[idx] = Math.sin(pB);
      tbRsSinC[idx] = Math.sin(pC);
      tbRsCosC[idx] = Math.cos(pC);
    }
  }
}

function stepAndPushTestbenchRs(simDt) {
  ensureTestbenchRsTexTables();
  if (!tbRsWasActive) {
    tbRsWasActive = true;
    tbRsFrameSeq = 0;
    tbRsAudioSampleCursor = 0;
    if (wasmExports && typeof wasmExports.chronos_reset_tracks === 'function') {
      wasmExports.chronos_reset_tracks();
    }
    workerExpectedEngineFrame = 0;
  }
  const fps = 60.0;
  const f = tbRsFrameSeq++;
  const t0Us = 1_000_000;
  const tSrcUs = t0Us + Math.round((f * 1_000_000) / fps);
  const fTone = 1125.0;
  const ampPx = 0.45;
  const dtRowUs = rsCalibConfig.dtRowUs;
  const exposureUs = rsCalibConfig.exposureUs;
  const taps = exposureUs >= 500.0 ? 7 : 5;
  const invT = 1.0 / taps;
  const expSec = exposureUs * 1e-6;
  const w2piF = 2.0 * Math.PI * fTone;
  const ax = 0.27, bx = 0.19, cx = 0.37;

  if (
    rsCalibConfig.mode === 2 &&
    wave2VibroAbiValid &&
    audioStagingView &&
    typeof wasmExports.chronos_push_audio_block === 'function'
  ) {
    const sRate =
      Number.isFinite(rsCalibConfig.audioRateHz) && rsCalibConfig.audioRateHz >= 4000.0
        ? rsCalibConfig.audioRateHz
        : 48000.0;
    const endSample = Math.round(((f + 1) / fps) * sRate);
    const rawCount = endSample - tbRsAudioSampleCursor;
    if (rawCount > 0) {
      const nAudio = Math.min(AUDIO_STAGING_LEN, rawCount);
      for (let i = 0; i < nAudio; i++) {
        const tS = t0Us * 1e-6 + (tbRsAudioSampleCursor + i) / sRate;
        audioStagingView[i] = 0.5 * Math.cos(w2piF * tS + 0.1);
      }
      const pushedEnd = tbRsAudioSampleCursor + nAudio;
      const tEndUs = t0Us + Math.round((pushedEnd * 1_000_000) / sRate);
      const tLo = tEndUs >>> 0;
      const tHi = Math.floor(tEndUs / 4294967296) >>> 0;
      wasmExports.chronos_push_audio_block(nAudio >>> 0, tLo, tHi, sRate, 0.0);
      tbRsAudioSampleCursor = pushedEnd;
    }
  }

  const sd = roiScanDir >>> 0;
  const scanAlongEx = sd === 2 || sd === 3;
  const scanLen = scanAlongEx ? frameW : frameH;
  const mScA = new Float32Array(scanLen);
  const mCcA = new Float32Array(scanLen);
  const mScB = new Float32Array(scanLen);
  const mCcB = new Float32Array(scanLen);
  const mScC = new Float32Array(scanLen);
  const mCcC = new Float32Array(scanLen);
  for (let qIdx = 0; qIdx < scanLen; qIdx++) {
    const q = (sd === 1 || sd === 3) ? (scanLen - 1 - qIdx) : qIdx;
    const tMid = (tSrcUs + q * dtRowUs + 0.5 * exposureUs) * 1e-6;
    let sA = 0.0, cA = 0.0, sB = 0.0, cB = 0.0, sC = 0.0, cC = 0.0;
    for (let t = 0; t < taps; t++) {
      const tS = taps === 1 ? tMid : (tMid + ((t + 0.5) * invT - 0.5) * expSec);
      const ux = ampPx * Math.cos(w2piF * tS + 0.4);
      sA += Math.sin(ax * ux);
      cA += Math.cos(ax * ux);
      sB += Math.sin(bx * ux);
      cB += Math.cos(bx * ux);
      sC += Math.sin(cx * ux);
      cC += Math.cos(cx * ux);
    }
    mScA[qIdx] = sA * invT;
    mCcA[qIdx] = cA * invT;
    mScB[qIdx] = sB * invT;
    mCcB[qIdx] = cB * invT;
    mScC[qIdx] = sC * invT;
    mCcC[qIdx] = cC * invT;
  }

  const x0Vib = frameW >> 2;
  const x1Vib = frameW - x0Vib;
  const y0Vib = frameH >> 2;
  const y1Vib = frameH - y0Vib;

  for (let y = 0; y < frameH; y++) {
    const row = y * frameW;
    if (!scanAlongEx) {
      const sA = mScA[y], cA = mCcA[y];
      const sB = mScB[y], cB = mCcB[y];
      const sC = mScC[y], cC = mCcC[y];
      for (let x = 0; x < frameW; x++) {
        const idx = row + x;
        let l;
        if (x < x0Vib || x >= x1Vib) {
          l = Math.round(128.0 + 38.0 * tbRsSinA[idx] + 34.0 * tbRsCosB[idx] + 22.0 * tbRsSinC[idx]);
        } else {
          const vA = tbRsSinA[idx] * cA - tbRsCosA[idx] * sA;
          const vB = tbRsCosB[idx] * cB + tbRsSinB[idx] * sB;
          const vC = tbRsSinC[idx] * cC - tbRsCosC[idx] * sC;
          l = Math.round(128.0 + 38.0 * vA + 34.0 * vB + 22.0 * vC);
        }
        const v = l < 8 ? 8 : (l > 247 ? 247 : l);
        const o = idx * 4;
        tbRsFrameBuf[o] = v;
        tbRsFrameBuf[o + 1] = v;
        tbRsFrameBuf[o + 2] = v;
        tbRsFrameBuf[o + 3] = 255;
      }
    } else {
      const staticBandY = y < y0Vib || y >= y1Vib;
      for (let x = 0; x < frameW; x++) {
        const idx = row + x;
        let l;
        if (staticBandY) {
          l = Math.round(128.0 + 38.0 * tbRsSinA[idx] + 34.0 * tbRsCosB[idx] + 22.0 * tbRsSinC[idx]);
        } else {
          const vA = tbRsSinA[idx] * mCcA[x] - tbRsCosA[idx] * mScA[x];
          const vB = tbRsCosB[idx] * mCcB[x] + tbRsSinB[idx] * mScB[x];
          const vC = tbRsSinC[idx] * mCcC[x] - tbRsCosC[idx] * mScC[x];
          l = Math.round(128.0 + 38.0 * vA + 34.0 * vB + 22.0 * vC);
        }
        const v = l < 8 ? 8 : (l > 247 ? 247 : l);
        const o = idx * 4;
        tbRsFrameBuf[o] = v;
        tbRsFrameBuf[o + 1] = v;
        tbRsFrameBuf[o + 2] = v;
        tbRsFrameBuf[o + 3] = 255;
      }
    }
  }

  return {
    pixels: tbRsFrameBuf,
    mediaTimeSec: tSrcUs * 1e-6,
    ingestInfo: { tSrcUs, ptsKnown: true },
  };
}

let rafHandle = 0;
let rafWatchdogId = 0;

function cancelWorkerTickLoop() {
  clearTimeout(rafWatchdogId);
  rafWatchdogId = 0;
  if (rafHandle && typeof self.cancelAnimationFrame === 'function') {
    self.cancelAnimationFrame(rafHandle);
  }
  rafHandle = 0;
}

function scheduleNextWorkerTick() {
  if (!running || sourceMode !== 'testbench') return;
  if (typeof self.requestAnimationFrame === 'function') {
    clearTimeout(rafWatchdogId);
    const prevTick = tickCounter;
    rafHandle = self.requestAnimationFrame(workerRafLoop);
    rafWatchdogId = setTimeout(() => {
      if (running && sourceMode === 'testbench' && tickCounter === prevTick) {
        if (typeof self.cancelAnimationFrame === 'function') {
          self.cancelAnimationFrame(rafHandle);
        }
        workerRafLoop(performance.now());
      }
    }, 18);
  } else {
    rafWatchdogId = setTimeout(() => workerRafLoop(performance.now()), 16);
  }
}

function workerRafLoop(now) {
  if (!running || sourceMode !== 'testbench') return;
  clearTimeout(rafWatchdogId);
  const dtMs = lastRafTs > 0 ? Math.min(50.0, Math.max(4.0, now - lastRafTs)) : 16.667;
  lastRafTs = now;
  const instFps = 1000.0 / dtMs;
  smoothFps = 0.92 * smoothFps + 0.08 * instFps;

  inWorkerRafTick = true;
  try {
    if (wasmExports) {
      const simDt = 1.0 / 60.0;
      if (isTestbenchRsStimulusActive()) {
        const rsStep = stepAndPushTestbenchRs(simDt);
        processIngestedFrame(rsStep.pixels, simDt, rsStep.mediaTimeSec, rsStep.ingestInfo);
      } else {
        if (tbRsWasActive) {
          resetTestbenchRsState();
          if (typeof wasmExports.chronos_reset_tracks === 'function') {
            wasmExports.chronos_reset_tracks();
          }
          workerExpectedEngineFrame = 0;
        }
        const rawPixels = stepAndRenderBlindPillarScene(simDt);
        processIngestedFrame(rawPixels, simDt, -1.0);
      }
    }
  } finally {
    inWorkerRafTick = false;
    scheduleNextWorkerTick();
  }
}

// ============================================================================
// Worker Message Control Plane
// ============================================================================

self.onmessage = async (ev) => {
  const msg = ev.data;

  if (msg.type === 'init') {
    hudCanvas = msg.hudCanvas;
    pipCanvas = msg.pipCanvas;
    hilbertCanvas = msg.hilbertCanvas;
    spectralCanvas = msg.spectralCanvas;
    persistenceCanvas = msg.persistenceCanvas || null;
    physicsPhaseCanvas = msg.physicsPhaseCanvas || null;

    hudCtx = hudCanvas.getContext('2d', { alpha: false });
    pipCtx = pipCanvas.getContext('2d', { alpha: false });
    hilbertCtx = hilbertCanvas.getContext('2d', { alpha: false });
    spectralCtx = spectralCanvas.getContext('2d', { alpha: false });
    persistenceCtx = persistenceCanvas ? persistenceCanvas.getContext('2d', { alpha: false }) : null;
    physicsPhaseCtx = physicsPhaseCanvas ? physicsPhaseCanvas.getContext('2d', { alpha: false }) : null;

    if (Array.isArray(msg.slotPipCanvases)) {
      for (let m = 0; m < Math.min(MAX_TARGETS, msg.slotPipCanvases.length); m++) {
        const c = msg.slotPipCanvases[m];
        if (c && typeof c.getContext === 'function') {
          slotPipCanvases[m] = c;
          slotPipCtxs[m] = c.getContext('2d', { alpha: false });
        }
      }
    }

    dispStageCanvas = new OffscreenCanvas(frameW, frameH);
    dispStageCtx = dispStageCanvas.getContext('2d');
    pipStageCanvas = new OffscreenCanvas(pipDim, pipDim);
    pipStageCtx = pipStageCanvas.getContext('2d');
    simCanvas = new OffscreenCanvas(frameW, frameH);
    simCtx = simCanvas.getContext('2d', { willReadFrequently: true });

    ecaStageCanvas = new OffscreenCanvas(frameW, frameH);
    ecaStageCtx = ecaStageCanvas.getContext('2d');
    bosOverlayCanvas = new OffscreenCanvas(BOS_W, BOS_H);
    bosOverlayCtx = bosOverlayCanvas.getContext('2d');
    bosOverlayImgData = bosOverlayCtx.createImageData(BOS_W, BOS_H);
    w2OverlayCanvas = new OffscreenCanvas(FIELD_GRID_W, FIELD_GRID_H);
    w2OverlayCtx = w2OverlayCanvas.getContext('2d');
    w2OverlayImgData = w2OverlayCtx.createImageData(FIELD_GRID_W, FIELD_GRID_H);

    try {
      const resp = await fetch(msg.wasmUrl || './chronos_spectral.wasm?v=20261115c1k');
      if (!resp.ok) {
        throw new Error(`Wasm fetch failed (HTTP ${resp.status})`);
      }
      const wasmBytes = await resp.arrayBuffer();
      const { instance } = await WebAssembly.instantiate(wasmBytes, {});

      wasmExports = instance.exports;
      wasmMemory = wasmExports.memory;
      wasmExports.chronos_init();
      workerExpectedEngineFrame = 0;
      wasmExports.chronos_set_params(
        lookaheadK,
        ecaAdaptRate,
        fistaLambda,
        eulerianGain,
        phaseOffsetRad,
        labMode
      );

      fiedlerView = new Float32Array(
        wasmMemory.buffer,
        wasmExports.chronos_get_fiedler_ptr(),
        GRAPH_NODES
      );
      telemView = new Float32Array(
        wasmMemory.buffer,
        wasmExports.chronos_get_telemetry_ptr(),
        TELEMETRY_LEN
      );

      if (typeof msg.showcaseMinimalScopes === 'boolean') {
        showcaseMinimalScopes = msg.showcaseMinimalScopes;
      }
      if (typeof msg.rawWipeRatio === 'number' && Number.isFinite(msg.rawWipeRatio)) {
        rawWipeRatio = Math.max(0.0, Math.min(1.0, msg.rawWipeRatio));
      }
      if (typeof msg.showHeroTargets === 'boolean') {
        showHeroTargets = msg.showHeroTargets;
      }
      configureResolutionViews(msg.resMode || 0, true);
      applySceneCalibrationPreset('testbench');
      running = true;
      self.postMessage({
        type: 'ready',
        memoryBytes: wasmMemory.buffer.byteLength,
        pages: wasmMemory.buffer.byteLength / 65536,
        resMode,
        frameW,
        frameH,
        pipDim,
        showHeroTargets,
        overlaySyncSeq: lastOverlaySyncSeq,
      });
      scheduleNextWorkerTick();
    } catch (initErr) {
      running = false;
      self.postMessage({
        type: 'init_error',
        message: initErr && initErr.message ? initErr.message : String(initErr),
      });
    }
  } else if (msg.type === 'set_resolution') {
    if (wasmExports) {
      const prevRoiProj = currentEffectiveRoiProjection();
      const reqMode = msg.resMode !== undefined ? msg.resMode : msg.mode;
      configureResolutionViews(reqMode ? 1 : 0, false);
      const nextRoiProj = currentEffectiveRoiProjection();
      const roiInvalidated =
        roiProjectionChanged(prevRoiProj, nextRoiProj) &&
        (calMode === 'ruler' || calMode === 'homography' || calArmed !== 'none');
      if (roiInvalidated) {
        clearCalibrationState();
        recordCalibrationSegmentTransition('roi_changed');
      }
      self.postMessage({
        type: 'resolution_changed',
        resMode,
        frameW,
        frameH,
        pipDim,
      });
      emitCalibrationState(
        roiInvalidated
          ? 'ROI CROP CHANGED · IMAGE-PLANE CALIBRATION RESET TO PX UNITS'
          : ''
      );
      if (roiInvalidated) {
        redrawPausedHudIfReady();
      }
    }
  } else if (msg.type === 'set_scenario') {
    applyTestbenchScenarioSwitch(msg.scenario);
  } else if (msg.type === 'set_crop') {
    if (Array.isArray(msg.cropNorm) && msg.cropNorm.length === 4) {
      const x = Math.max(0.0, Math.min(0.95, Number(msg.cropNorm[0]) || 0.0));
      const y = Math.max(0.0, Math.min(0.95, Number(msg.cropNorm[1]) || 0.0));
      const w = Math.max(0.05, Math.min(1.0 - x, Number(msg.cropNorm[2]) || 1.0));
      const h = Math.max(0.05, Math.min(1.0 - y, Number(msg.cropNorm[3]) || 1.0));
      const cropChanged =
        Math.abs(cropNorm[0] - x) > 1e-6 ||
        Math.abs(cropNorm[1] - y) > 1e-6 ||
        Math.abs(cropNorm[2] - w) > 1e-6 ||
        Math.abs(cropNorm[3] - h) > 1e-6;
      cropNorm[0] = x;
      cropNorm[1] = y;
      cropNorm[2] = w;
      cropNorm[3] = h;
      if (cropChanged && (calMode === 'ruler' || calMode === 'homography' || calArmed !== 'none')) {
        clearCalibrationState();
        recordCalibrationSegmentTransition('crop_changed');
        emitCalibrationState('ROI CROP CHANGED · IMAGE-PLANE CALIBRATION RESET TO PX UNITS');
        redrawPausedHudIfReady();
      }
    }
  } else if (msg.type === 'cal_control') {
    if (typeof msg.syncSeq === 'number' && Number.isFinite(msg.syncSeq) && msg.syncSeq > lastCalSyncSeq) {
      lastCalSyncSeq = Math.trunc(msg.syncSeq);
    }
    const inRulerM = typeof msg.rulerMeters === 'number' && msg.rulerMeters >= 0.01 ? msg.rulerMeters : null;
    const inPlaneW = typeof msg.planeW === 'number' && msg.planeW >= 0.01 ? msg.planeW : null;
    const inPlaneH = typeof msg.planeH === 'number' && msg.planeH >= 0.01 ? msg.planeH : null;

    if (msg.action === 'arm_ruler') {
      if (calArmed === 'ruler') {
        calArmed = 'none';
        calDraftCount = 0;
      } else {
        calDraftRulerMeters = inRulerM ?? calRulerMeters;
        calArmed = 'ruler';
        calDraftCount = 0;
      }
      emitCalibrationState();
    } else if (msg.action === 'arm_homography') {
      if (calArmed === 'homography') {
        calArmed = 'none';
        calDraftCount = 0;
      } else {
        calDraftPlaneW = inPlaneW ?? calPlaneW;
        calDraftPlaneH = inPlaneH ?? calPlaneH;
        calArmed = 'homography';
        calDraftCount = 0;
      }
      emitCalibrationState();
    } else if (msg.action === 'cancel_arm') {
      if (calArmed !== 'none') {
        calArmed = 'none';
        calDraftCount = 0;
      }
      emitCalibrationState();
    } else if (msg.action === 'stage_dims') {
      if (inRulerM !== null) calDraftRulerMeters = inRulerM;
      if (inPlaneW !== null) calDraftPlaneW = inPlaneW;
      if (inPlaneH !== null) calDraftPlaneH = inPlaneH;
    } else if (msg.action === 'set_coords' && Array.isArray(msg.pts)) {
      const targetMode = msg.coordMode === 'homography' ? 'homography' : 'ruler';
      const reqN = targetMode === 'homography' ? 4 : 2;
      if (msg.pts.length >= reqN) {
        const tmpX = new Float64Array(4);
        const tmpY = new Float64Array(4);
        let validNums = true;
        for (let i = 0; i < reqN; i++) {
          const px = Number(msg.pts[i] && msg.pts[i][0]);
          const py = Number(msg.pts[i] && msg.pts[i][1]);
          if (!Number.isFinite(px) || !Number.isFinite(py) || px < 0.0 || px > 1.0 || py < 0.0 || py > 1.0) {
            validNums = false;
            break;
          }
          tmpX[i] = px;
          tmpY[i] = py;
        }
        if (validNums) {
          calDraftRulerMeters = inRulerM ?? calRulerMeters;
          calDraftPlaneW = inPlaneW ?? calPlaneW;
          calDraftPlaneH = inPlaneH ?? calPlaneH;
          for (let i = 0; i < reqN; i++) {
            calDraftPtsNormX[i] = tmpX[i];
            calDraftPtsNormY[i] = tmpY[i];
          }
          calDraftCount = reqN;
          commitSurveyedCalibrationFromDraft(targetMode, `keyboard_${targetMode}`);
        } else {
          emitCalibrationState('COORDINATES MUST BE NORMALIZED IN [0.00, 1.00] · PRIOR CALIBRATION PRESERVED');
        }
      }
    } else if (msg.action === 'set_auto_g') {
      calArmed = 'none';
      calDraftCount = 0;
      calPtCount = 0;
      calPresetName = '';
      calHomographyValid = false;
      calMode = 'auto_g';
      calProvenance = 'assumed';
      calAutoGLocked = false;
      calAutoGNormH = 0.0;
      if (telemView && telemView[TELEM_GLOBAL + 1] >= AUTO_G_LOCK_CONF && telemView[TELEM_GLOBAL + 0] > 50.0) {
        calAutoGNormH = telemView[TELEM_GLOBAL + 0] / frameH;
        calAutoGLocked = true;
      } else if (telemView && telemView[TELEM_GLOBAL + 0] > 50.0) {
        calAutoGNormH = telemView[TELEM_GLOBAL + 0] / frameH;
      }
      recordCalibrationSegmentTransition('set_auto_g');
      emitCalibrationState();
    } else if (msg.action === 'apply_preset') {
      applySceneCalibrationPreset(msg.preset || 'testbench');
    } else if (msg.action === 'update_dims') {
      if (inRulerM !== null) calDraftRulerMeters = inRulerM;
      if (inPlaneW !== null) calDraftPlaneW = inPlaneW;
      if (inPlaneH !== null) calDraftPlaneH = inPlaneH;
      if (calArmed !== 'none') {
        emitCalibrationState();
      } else if (calMode === 'homography' && calPtCount >= 4) {
        const nextW = inPlaneW ?? calPlaneW;
        const nextH = inPlaneH ?? calPlaneH;
        const changed = Math.abs(nextW - calPlaneW) > 1e-6 || Math.abs(nextH - calPlaneH) > 1e-6;
        if (inRulerM !== null) calRulerMeters = inRulerM;
        if (changed) {
          const ok = solveHomography4Pt(nextW, nextH, calPtsNormX, calPtsNormY, true);
          if (ok) {
            calPlaneW = nextW;
            calPlaneH = nextH;
            calHomographyValid = true;
            recordCalibrationSegmentTransition('update_dims');
          }
        }
        emitCalibrationState();
      } else {
        const rulerChanged = inRulerM !== null && Math.abs(inRulerM - calRulerMeters) > 1e-6;
        if (inRulerM !== null) calRulerMeters = inRulerM;
        if (inPlaneW !== null) calPlaneW = inPlaneW;
        if (inPlaneH !== null) calPlaneH = inPlaneH;
        if (calMode === 'ruler' && calPtCount >= 2 && rulerChanged) {
          recordCalibrationSegmentTransition('update_dims');
        }
        emitCalibrationState();
      }
    } else if (msg.action === 'clear') {
      clearCalibrationState();
      recordCalibrationSegmentTransition('clear');
      emitCalibrationState();
    }
    redrawPausedHudIfReady();
  } else if (msg.type === 'canvas_click') {
    if (typeof msg.normX === 'number' && typeof msg.normY === 'number') {
      const intent = typeof msg.intent === 'string' ? msg.intent : '';
      const expectedCalArmed = typeof msg.expectedCalArmed === 'string' ? msg.expectedCalArmed : '';
      handleHudCanvasClick(msg.normX, msg.normY, intent, expectedCalArmed);
    }
  } else if (msg.type === 'cycle_focus') {
    cycleFocusTarget();
  } else if (msg.type === 'pin_target') {
    if (wasmExports) {
      const reqTid = Math.max(0, Math.round(Number(msg.trackId) || 0));
      const curFocus = wasmExports.chronos_get_focus_target();
      wasmExports.chronos_set_focus_target(reqTid > 0 && reqTid !== curFocus ? reqTid : 0);
      redrawPausedHudIfReady();
    }
  } else if (msg.type === 'params') {
    if (typeof msg.overlaySyncSeq === 'number' && msg.overlaySyncSeq > lastOverlaySyncSeq) {
      lastOverlaySyncSeq = msg.overlaySyncSeq;
    }
    let visualFlagChanged = false;
    let labModeChanged = false;
    if (typeof msg.lookaheadK === 'number') lookaheadK = msg.lookaheadK;
    if (typeof msg.ecaAdaptRate === 'number') ecaAdaptRate = msg.ecaAdaptRate;
    if (typeof msg.fistaLambda === 'number') fistaLambda = msg.fistaLambda;
    if (typeof msg.eulerianGain === 'number') eulerianGain = msg.eulerianGain;
    if (typeof msg.phaseOffsetRad === 'number') phaseOffsetRad = msg.phaseOffsetRad;
    if (typeof msg.labMode === 'number') {
      if (msg.labMode !== labMode) {
        manualConsTestActive = false;
        labModeChanged = true;
      }
      labMode = msg.labMode;
      if (labMode !== 2) {
        consRingValid.fill(0);
        invalidateWave2RingStamp(16);
      }
    }
    if (typeof msg.showEcaOverlay === 'boolean') {
      visualFlagChanged = visualFlagChanged || showEcaOverlay !== msg.showEcaOverlay;
      showEcaOverlay = msg.showEcaOverlay;
    }
    if (typeof msg.showKalmanBaseline === 'boolean') {
      visualFlagChanged = visualFlagChanged || showKalmanBaseline !== msg.showKalmanBaseline;
      showKalmanBaseline = msg.showKalmanBaseline;
    }
    if (typeof msg.showHeroTargets === 'boolean') {
      visualFlagChanged = visualFlagChanged || showHeroTargets !== msg.showHeroTargets;
      showHeroTargets = msg.showHeroTargets;
    }
    if (typeof msg.showAllOverlays === 'boolean') {
      visualFlagChanged = visualFlagChanged || showAllOverlays !== msg.showAllOverlays;
      showAllOverlays = msg.showAllOverlays;
    }
    if (typeof msg.strobeEnabled === 'boolean') {
      visualFlagChanged = visualFlagChanged || strobeEnabled !== msg.strobeEnabled;
      strobeEnabled = msg.strobeEnabled;
    }
    if (typeof msg.rawWipeRatio === 'number' && Number.isFinite(msg.rawWipeRatio)) {
      const clamped = Math.max(0.0, Math.min(1.0, msg.rawWipeRatio));
      visualFlagChanged = visualFlagChanged || Math.abs(rawWipeRatio - clamped) > 1e-4;
      rawWipeRatio = clamped;
    }
    if (typeof msg.showcaseMinimalScopes === 'boolean') {
      const wakingScopes = showcaseMinimalScopes && !msg.showcaseMinimalScopes;
      showcaseMinimalScopes = msg.showcaseMinimalScopes;
      if (wakingScopes) visualFlagChanged = true;
    }
    if (typeof msg.audioOutLatencySec === 'number') {
      audioOutLatencySec = msg.audioOutLatencySec;
      if (wasmExports) wasmExports.chronos_set_audio_latency(audioOutLatencySec);
    }

    if (wasmExports) {
      wasmExports.chronos_set_params(
        lookaheadK,
        ecaAdaptRate,
        fistaLambda,
        eulerianGain,
        phaseOffsetRad,
        labMode
      );
      if (sourceMode !== 'testbench' || visualFlagChanged || labModeChanged || !running) {
        redrawPausedHudIfReady();
        if ((visualFlagChanged || labModeChanged || !running) && (!telemView || !hasAnyRenderedFrame())) {
          emitTelemetrySnapshot();
        }
      }
    }
  } else if (msg.type === 'set_swarm_mode') {
    const rawM = Number(msg.swarmMode);
    const nextMode = Number.isFinite(rawM) ? Math.max(0, Math.min(2, Math.round(rawM))) : 0;
    userSwarmMode = nextMode;
    denseSwarmOverrideActive = false;
    swarmMode = nextMode;
    if (hasChronos1000Exports()) {
      wasmExports.chronos_set_swarm_mode(nextMode);
    }
    syncWave2RingEnabledBits();
    if (nextMode === 0) {
      swarmRingValid.fill(0);
      if (hodgeReqMode === 0 || !hodgeTelemView) {
        hodgeRingValid.fill(0);
      } else {
        propagateWave2FaultToRing(hodgeRingValid, hodgeRingTelem, hodgeRingPlanes, hodgeTelemView);
      }
      invalidateWave2RingStamp(8);
      clearActiveInspectionSelection();
    }
    syncSwarmConfigToMain(benchScenario);
    redrawPausedHudIfReady();
    if (!telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'set_field_overlay_mode') {
    if (typeof msg.overlaySyncSeq === 'number' && msg.overlaySyncSeq > lastOverlaySyncSeq) {
      lastOverlaySyncSeq = msg.overlaySyncSeq;
    }
    const rawO = Number(msg.fieldOverlayMode);
    fieldOverlayMode = Number.isFinite(rawO) ? Math.max(0, Math.min(15, Math.round(rawO))) : 0;
    if (fieldOverlayMode >= 5 && fieldOverlayMode <= 7) {
      bosOverlayKind = fieldOverlayMode - 5;
      if (wave1BosAbiValid && typeof wasmExports.chronos_set_bos_overlay_kind === 'function') {
        wasmExports.chronos_set_bos_overlay_kind(bosOverlayKind);
      }
    }
    syncSwarmConfigToMain(benchScenario);
    redrawPausedHudIfReady();
    if (!telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'set_roi') {
    const prevRoiProj = currentEffectiveRoiProjection();
    let rsGeomChanged = false;
    if (typeof msg.native1to1 === 'boolean') {
      rsGeomChanged = rsGeomChanged || nativeRoiEnabled !== msg.native1to1;
      nativeRoiEnabled = msg.native1to1;
    }
    if (typeof msg.normX === 'number' && Number.isFinite(msg.normX)) {
      roiOffsetNormX = Math.max(0.0, Math.min(1.0, msg.normX));
      roiUseExplicitPx = false;
    }
    if (typeof msg.normY === 'number' && Number.isFinite(msg.normY)) {
      roiOffsetNormY = Math.max(0.0, Math.min(1.0, msg.normY));
      roiUseExplicitPx = false;
    }
    if (
      typeof msg.x0 === 'number' &&
      Number.isFinite(msg.x0) &&
      typeof msg.y0 === 'number' &&
      Number.isFinite(msg.y0)
    ) {
      roiUseExplicitPx = true;
      roiExplicitX0 = Math.max(0, Math.round(msg.x0));
      roiExplicitY0 = Math.max(0, Math.round(msg.y0));
    }
    if (
      typeof msg.orientation === 'number' &&
      Number.isInteger(msg.orientation) &&
      msg.orientation >= 0 &&
      msg.orientation <= 7
    ) {
      rsGeomChanged = rsGeomChanged || roiOrientation !== msg.orientation;
      roiOrientation = msg.orientation;
    }
    if (
      typeof msg.scanDir === 'number' &&
      Number.isInteger(msg.scanDir) &&
      (msg.scanDir === 0xffffffff || msg.scanDir === -1 || (msg.scanDir >= 0 && msg.scanDir <= 3))
    ) {
      const nextSd = msg.scanDir >>> 0;
      rsGeomChanged = rsGeomChanged || roiScanDir !== nextSd;
      roiScanDir = nextSd;
    }
    const nextRoiProj = currentEffectiveRoiProjection();
    if (
      roiProjectionChanged(prevRoiProj, nextRoiProj) &&
      (calMode === 'ruler' || calMode === 'homography' || calArmed !== 'none')
    ) {
      clearCalibrationState();
      recordCalibrationSegmentTransition('roi_changed');
      emitCalibrationState('ROI CROP CHANGED · IMAGE-PLANE CALIBRATION RESET TO PX UNITS');
    }
    if (rsGeomChanged) {
      resetTestbenchRsState();
      if (rsCalibConfig.mode === 0) {
        vibroRingValid.fill(0);
        invalidateWave2RingStamp(0);
      } else if (!nativeRoiEnabled || roiScanDir > 3) {
        if (vibroTelemView && vibroTelemView[0] !== 3) {
          propagateWave2FaultToRing(vibroRingValid, vibroRingTelem, vibroRingPlanes, vibroTelemView);
        } else {
          vibroRingValid.fill(0);
        }
        invalidateWave2RingStamp(0);
      }
    }
    redrawPausedHudIfReady();
    if (!telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'set_bos_config') {
    if (typeof msg.bosMode === 'number' && Number.isFinite(msg.bosMode)) {
      bosReqMode = Math.max(0, Math.min(2, Math.round(msg.bosMode)));
      if (wave1BosAbiValid && typeof wasmExports.chronos_set_bos_mode === 'function') {
        wasmExports.chronos_set_bos_mode(bosReqMode);
      }
      if (bosReqMode === 0) bosRingValid.fill(0);
    }
    if (typeof msg.overlayKind === 'number' && Number.isFinite(msg.overlayKind)) {
      bosOverlayKind = Math.max(0, Math.min(2, Math.round(msg.overlayKind)));
      if (wave1BosAbiValid && typeof wasmExports.chronos_set_bos_overlay_kind === 'function') {
        wasmExports.chronos_set_bos_overlay_kind(bosOverlayKind);
      }
    }
    if (Array.isArray(msg.gaugeRoi) && msg.gaugeRoi.length === 4) {
      const gx0 = Math.max(0, Math.min(BOS_W, Math.round(Number(msg.gaugeRoi[0]) || 0)));
      const gy0 = Math.max(0, Math.min(BOS_H, Math.round(Number(msg.gaugeRoi[1]) || 0)));
      const gx1 = Math.max(0, Math.min(BOS_W, Math.round(Number(msg.gaugeRoi[2]) || 0)));
      const gy1 = Math.max(0, Math.min(BOS_H, Math.round(Number(msg.gaugeRoi[3]) || 0)));
      bosGaugeRoi = [gx0, gy0, gx1, gy1];
      if (wave1BosAbiValid && typeof wasmExports.chronos_set_bos_gauge_roi === 'function') {
        wasmExports.chronos_set_bos_gauge_roi(gx0, gy0, gx1, gy1);
      }
    }
    if (msg.resetRef && wave1BosAbiValid && typeof wasmExports.chronos_reset_bos_ref === 'function') {
      wasmExports.chronos_reset_bos_ref();
    }
    redrawPausedHudIfReady();
    if (!telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'set_metrology_config') {
    if (typeof msg.metrologyMode === 'number' && Number.isFinite(msg.metrologyMode)) {
      microMetrologyMode = Math.max(0, Math.min(2, Math.round(msg.metrologyMode)));
      if (wave1MetrologyAbiValid && typeof wasmExports.chronos_set_micro_metrology_mode === 'function') {
        wasmExports.chronos_set_micro_metrology_mode(microMetrologyMode);
      }
      if (microMetrologyMode === 0) covRingValid.fill(0);
    }
    if (typeof msg.showCovEllipses === 'boolean') {
      showCovEllipses = msg.showCovEllipses;
    }
    redrawPausedHudIfReady();
    if (!telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'set_rs_calib') {
    const reqM = msg.rsMode !== undefined ? msg.rsMode : msg.mode;
    if (typeof reqM === 'number' && Number.isFinite(reqM)) {
      rsCalibConfig.mode = Math.max(0, Math.min(2, Math.round(reqM)));
    }
    let dtArg = rsCalibConfig.dtRowUs;
    if (msg.dtRowUs !== undefined) {
      dtArg = typeof msg.dtRowUs === 'number' ? msg.dtRowUs : NaN;
      rsCalibConfig.dtRowUs = Number.isFinite(dtArg) && dtArg >= 0.0 ? dtArg : 0.0;
    }
    let expArg = rsCalibConfig.exposureUs;
    if (msg.exposureUs !== undefined) {
      expArg = typeof msg.exposureUs === 'number' ? msg.exposureUs : NaN;
      rsCalibConfig.exposureUs = Number.isFinite(expArg) && expArg >= 0.0 ? expArg : 0.0;
    }
    let rateArg = rsCalibConfig.audioRateHz;
    if (msg.audioRateHz !== undefined) {
      rateArg = typeof msg.audioRateHz === 'number' ? msg.audioRateHz : NaN;
      rsCalibConfig.audioRateHz = Number.isFinite(rateArg) && rateArg >= 0.0 ? rateArg : 0.0;
    }
    let skewArg = rsCalibConfig.maxSkewUs;
    if (msg.maxSkewUs !== undefined) {
      skewArg = typeof msg.maxSkewUs === 'number' ? msg.maxSkewUs : NaN;
      rsCalibConfig.maxSkewUs = Number.isFinite(skewArg) && skewArg >= 0.0 ? skewArg : 0.0;
    }
    resetTestbenchRsState();
    if (wave2VibroAbiValid && typeof wasmExports.chronos_set_rs_calib === 'function') {
      wasmExports.chronos_set_rs_calib(
        rsCalibConfig.mode,
        dtArg,
        expArg,
        rateArg,
        skewArg
      );
    }
    syncWave2RingEnabledBits();
    if (rsCalibConfig.mode === 0) {
      vibroRingValid.fill(0);
      invalidateWave2RingStamp(0);
    } else if (!vibroTelemView || vibroTelemView[0] !== 3) {
      if (vibroTelemView) {
        propagateWave2FaultToRing(vibroRingValid, vibroRingTelem, vibroRingPlanes, vibroTelemView);
      } else {
        vibroRingValid.fill(0);
      }
      invalidateWave2RingStamp(0);
    }
    redrawPausedHudIfReady();
    if (!telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'push_audio_pcm') {
    let accepted = 0;
    let pushed = 0;
    let ringFaulted = false;
    if (wave2VibroAbiValid && audioStagingView && typeof wasmExports.chronos_push_audio_block === 'function') {
      const hasExplicitRate = msg.sampleRateHz !== undefined && msg.sampleRateHz !== null;
      const sRate = Math.fround(hasExplicitRate ? Number(msg.sampleRateHz) : (rsCalibConfig.audioRateHz || 48000.0));
      const hasExplicitSkew = msg.estSkewUs !== undefined && msg.estSkewUs !== null;
      const skew = Math.fround(hasExplicitSkew ? Number(msg.estSkewUs) : 0.0);
      const hasExplicitTime = msg.tEndUs !== undefined && msg.tEndUs !== null;
      const tEnd = hasExplicitTime ? Number(msg.tEndUs) : NaN;
      const src = msg.samples;
      const totalLen = src && typeof src.length === 'number' ? Math.trunc(src.length) : 0;

      const clearOrFaultVibroRing = () => {
        if (rsCalibConfig.mode === 1) return;
        ringFaulted = true;
        if (rsCalibConfig.mode === 0) {
          vibroRingValid.fill(0);
          invalidateWave2RingStamp(0);
        } else if (vibroTelemView && vibroTelemView[0] !== 3) {
          propagateWave2FaultToRing(vibroRingValid, vibroRingTelem, vibroRingPlanes, vibroTelemView);
          invalidateWave2RingStamp(0);
        }
      };

      if (!Number.isFinite(sRate) || sRate <= 0.0 || !Number.isFinite(skew)) {
        audioStagingView[0] = 0.0;
        wasmExports.chronos_push_audio_block(1, 0, 0, sRate, skew);
        clearOrFaultVibroRing();
      } else if (totalLen <= 0 || !Number.isFinite(tEnd) || tEnd < 0.0 || tEnd > Number.MAX_SAFE_INTEGER) {
        wasmExports.chronos_push_audio_block(0, 0, 0, sRate, skew);
        clearOrFaultVibroRing();
      } else {
        let allFinite = true;
        for (let i = 0; i < totalLen; i++) {
          if (!Number.isFinite(Math.fround(Number(src[i])))) {
            allFinite = false;
            break;
          }
        }
        const firstChunkLen = Math.min(AUDIO_STAGING_LEN, totalLen);
        const firstRem = totalLen - firstChunkLen;
        const firstTEnd = tEnd - (firstRem * 1_000_000.0) / sRate;
        if (!allFinite || !Number.isFinite(firstTEnd) || firstTEnd < 0.0 || firstTEnd > Number.MAX_SAFE_INTEGER) {
          wasmExports.chronos_push_audio_block(0, 0, 0, sRate, skew);
          clearOrFaultVibroRing();
        } else {
          let allOk = true;
          let offset = 0;
          while (offset < totalLen) {
            const chunkLen = Math.min(AUDIO_STAGING_LEN, totalLen - offset);
            for (let i = 0; i < chunkLen; i++) {
              audioStagingView[i] = Math.fround(Number(src[offset + i]));
            }
            const rem = totalLen - (offset + chunkLen);
            const chunkTEnd = Math.round(tEnd - (rem * 1_000_000.0) / sRate);
            const tLo = chunkTEnd >>> 0;
            const tHi = Math.floor(chunkTEnd / 4294967296) >>> 0;
            const ok = wasmExports.chronos_push_audio_block(chunkLen >>> 0, tLo, tHi, sRate, skew) >>> 0;
            if (ok !== 1) {
              allOk = false;
              break;
            }
            pushed += chunkLen;
            offset += chunkLen;
          }
          accepted = allOk && pushed === totalLen ? 1 : 0;
          if (rsCalibConfig.mode !== 1 && (!accepted || (vibroTelemView && vibroTelemView[0] !== 3))) {
            clearOrFaultVibroRing();
          }
        }
      }
    }
    if (msg.wantAck) {
      self.postMessage({
        type: 'wave2_audio_ack',
        reqId: msg.reqId ?? null,
        accepted,
        count: pushed,
        totalSamples: vibroTelemView ? (vibroTelemView[18] >>> 0) : 0,
      });
    }
    if (ringFaulted && hasAnyRenderedFrame()) {
      redrawPausedHudIfReady();
    } else if (msg.wantAck || ringFaulted || !telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'set_depth_config') {
    if (typeof msg.depthMode === 'number' && Number.isFinite(msg.depthMode)) {
      depthReqMode = Math.max(0, Math.min(4, Math.round(msg.depthMode)));
      if (depthAbiValid && typeof wasmExports.chronos_set_depth_mode === 'function') {
        wasmExports.chronos_set_depth_mode(depthReqMode);
      }
      syncWave2RingEnabledBits();
      if (depthReqMode === 0) {
        depthRingValid.fill(0);
        invalidateWave2RingStamp(24);
      }
    }
    if (
      depthAbiValid &&
      typeof wasmExports.chronos_set_depth_calib === 'function' &&
      typeof msg.fPx === 'number'
    ) {
      const calibOk = wasmExports.chronos_set_depth_calib(
        Number(msg.fPx),
        msg.speedMS !== undefined ? Number(msg.speedMS) : (msg.scaleMS !== undefined ? Number(msg.scaleMS) : 0.0),
        msg.baselineM !== undefined ? Number(msg.baselineM) : 0.0,
        msg.maxDispPx !== undefined ? Number(msg.maxDispPx) : 32.0,
        (msg.stereoSign | 0) || 0
      ) >>> 0;
      if (calibOk === 0) {
        self.postMessage({ type: 'depth_calib_rejected', reqId: msg.reqId ?? null });
      }
    }
    redrawPausedHudIfReady();
    if (!telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'set_hodge_config') {
    if (typeof msg.hodgeMode === 'number' && Number.isFinite(msg.hodgeMode)) {
      hodgeReqMode = Math.max(0, Math.min(2, Math.round(msg.hodgeMode)));
      if (wave2HodgeAbiValid && typeof wasmExports.chronos_set_hodge_mode === 'function') {
        wasmExports.chronos_set_hodge_mode(hodgeReqMode);
      }
      syncWave2RingEnabledBits();
      if (hodgeReqMode === 0) {
        hodgeRingValid.fill(0);
        invalidateWave2RingStamp(8);
      } else if (!hodgeTelemView || hodgeTelemView[0] !== 2) {
        if (hodgeTelemView) {
          propagateWave2FaultToRing(hodgeRingValid, hodgeRingTelem, hodgeRingPlanes, hodgeTelemView);
        } else {
          hodgeRingValid.fill(0);
        }
        invalidateWave2RingStamp(8);
      } else {
        for (let s = 0; s < RING_LEN; s++) {
          if (hodgeRingValid[s] === 1) {
            hodgeRingTelem[s][1] = hodgeReqMode;
          }
        }
      }
    }
    redrawPausedHudIfReady();
    if (!telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'set_wind_config') {
    if (typeof msg.windMode === 'number' && Number.isFinite(msg.windMode)) {
      windReqMode = Math.max(0, Math.min(2, Math.round(msg.windMode)));
      if (windAbiValid && typeof wasmExports.chronos_set_wind_mode === 'function') {
        wasmExports.chronos_set_wind_mode(windReqMode);
      }
      syncWave2RingEnabledBits();
      if (windReqMode === 0) {
        windRingValid.fill(0);
        invalidateWave2RingStamp(32);
      }
    }
    if (typeof msg.fluidDensity === 'number' && Number.isFinite(msg.fluidDensity) && msg.fluidDensity >= 0.0) {
      windFluidDensity = msg.fluidDensity;
      if (windAbiValid && typeof wasmExports.chronos_set_wind_fluid_density === 'function') {
        wasmExports.chronos_set_wind_fluid_density(windFluidDensity);
      }
    }
    redrawPausedHudIfReady();
    if (!telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'set_elasto_config') {
    if (typeof msg.elastoMode === 'number' && Number.isFinite(msg.elastoMode)) {
      elastoReqMode = Math.max(0, Math.min(4, Math.round(msg.elastoMode)));
      if (elastoAbiValid && typeof wasmExports.chronos_set_elasto_mode === 'function') {
        wasmExports.chronos_set_elasto_mode(elastoReqMode);
      }
      syncWave2RingEnabledBits();
      if (elastoReqMode === 0) {
        elastoRingValid.fill(0);
        invalidateWave2RingStamp(40);
      }
    }
    if (Array.isArray(msg.lockFreqs)) {
      elastoLockFreqs = msg.lockFreqs
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v) && v > 0.1 && v <= 50000.0)
        .slice(0, 4);
      if (elastoAbiValid && typeof wasmExports.chronos_set_elasto_lock_freqs === 'function') {
        wasmExports.chronos_set_elasto_lock_freqs(
          elastoLockFreqs.length,
          elastoLockFreqs[0] || 0,
          elastoLockFreqs[1] || 0,
          elastoLockFreqs[2] || 0,
          elastoLockFreqs[3] || 0
        );
      }
    }
    redrawPausedHudIfReady();
    if (!telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'eval_occluded_pair_test') {
    let ok = 0;
    let decoded = null;
    if (wave2ConsAbiValid && conservationInputView && conservationTelemView && typeof wasmExports.chronos_eval_occluded_pair === 'function') {
      conservationInputView.fill(0.0);
      const inp = msg.input;
      if (inp && typeof inp.length === 'number') {
        const n = Math.min(CONSERVATION_LEN, inp.length | 0);
        for (let i = 0; i < n; i++) {
          conservationInputView[i] = Number(inp[i]) || 0.0;
        }
      }
      ok = wasmExports.chronos_eval_occluded_pair() >>> 0;
      manualConsTestActive = true;
      decoded = decodeConservationArr(conservationTelemView);
    }
    if (msg.wantAck !== false) {
      self.postMessage({
        type: 'wave2_cons_test_ack',
        reqId: msg.reqId ?? null,
        ok,
        conservation: decoded,
      });
    }
    redrawPausedHudIfReady();
    if (!telemView || !hasAnyRenderedFrame()) {
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'inspect_point') {
    if (calArmed === 'none') {
      const parseNorm = (val, fallback = 0.5) => {
        if (val === undefined || val === null || val === '') return fallback;
        const n = Number(val);
        return Number.isFinite(n) ? Math.max(0.0, Math.min(1.0, n)) : fallback;
      };
      const hasAbsX = typeof msg.x === 'number' && Number.isFinite(msg.x);
      const hasAbsY = typeof msg.y === 'number' && Number.isFinite(msg.y);
      const reqNormX = hasAbsX ? (frameW > 0 ? Math.max(0.0, Math.min(1.0, msg.x / frameW)) : 0.0) : parseNorm(msg.normX, 0.5);
      const reqNormY = hasAbsY ? (frameH > 0 ? Math.max(0.0, Math.min(1.0, msg.y / frameH)) : 0.0) : parseNorm(msg.normY, 0.5);
      const qx = hasAbsX ? msg.x : reqNormX * frameW;
      const qy = hasAbsY ? msg.y : reqNormY * frameH;
      const reqSeq = typeof msg.inspectSeq === 'number' && Number.isFinite(msg.inspectSeq)
        ? Math.max(1, Math.trunc(msg.inspectSeq))
        : null;
      inspectFieldPointAt(qx, qy, reqSeq, reqNormX, reqNormY);
    }
  } else if (msg.type === 'set_bench_scenario') {
    applyTestbenchScenarioSwitch(msg.scenario);
  } else if (msg.type === 'set_source') {
    if (typeof msg.scenario === 'string' && msg.scenario) {
      benchScenario = msg.scenario;
    }
    sourceMode = msg.sourceMode;
    let syncSwarmUi = false;
    if (typeof msg.swarmMode === 'number' && Number.isFinite(msg.swarmMode)) {
      const explicitMode = Math.max(0, Math.min(2, Math.round(msg.swarmMode)));
      userSwarmMode = explicitMode;
      denseSwarmOverrideActive = false;
      swarmMode = explicitMode;
      if (hasChronos1000Exports()) {
        wasmExports.chronos_set_swarm_mode(swarmMode);
      }
      syncSwarmUi = true;
    } else if (typeof msg.swarmOverride === 'number' && Number.isFinite(msg.swarmOverride)) {
      const ovMode = Math.max(0, Math.min(2, Math.round(msg.swarmOverride)));
      denseSwarmOverrideActive = true;
      swarmMode = ovMode;
      if (hasChronos1000Exports()) {
        wasmExports.chronos_set_swarm_mode(swarmMode);
      }
      syncSwarmUi = true;
    } else if (denseSwarmOverrideActive) {
      denseSwarmOverrideActive = false;
      swarmMode = userSwarmMode;
      if (hasChronos1000Exports()) {
        wasmExports.chronos_set_swarm_mode(swarmMode);
      }
      syncSwarmUi = true;
    }

    if (sourceMode === 'testbench') {
      applyScenarioResolutionAndSwarm(benchScenario);
    }
    if (syncSwarmUi && !(sourceMode === 'testbench' && isChronos1000Scenario(benchScenario))) {
      syncSwarmConfigToMain(benchScenario);
    }

    clearVisualQueues();
    resetRecEvents();
    seedBoxPreview = null;
    fastPassActive = false;
    if (typeof msg.fpsHint === 'number' && msg.fpsHint >= 8.0) {
      sourceFps = msg.fpsHint;
    } else if (sourceMode !== 'testbench') {
      sourceFps = 30.0;
    }
    cropNorm[0] = 0.0;
    cropNorm[1] = 0.0;
    cropNorm[2] = 1.0;
    cropNorm[3] = 1.0;
    wasmTimeHist.fill(0.0);
    timeHistIdx = 0;
    tickCounter = 0;
    lastRafTs = 0;
    clearCalibrationState();
    recordCalibrationSegmentTransition(`source:${sourceMode}`);
    emitCalibrationState();
    if (wasmExports) {
      wasmExports.chronos_reset_tracks();
    }
    resetTestbenchState();
    if (sourceMode === 'testbench') {
      applySceneCalibrationPreset('testbench');
      scheduleNextWorkerTick();
    } else {
      cancelWorkerTickLoop();
    }
  } else if (msg.type === 'video_frame') {
    const bmp = msg.bitmap;
    let ackPayload = null;
    let rejectPayload = null;
    try {
      if (wasmExports && sourceMode !== 'testbench' && bmp) {
        const preScaledReq = Boolean(msg.preScaled || msg.fallbackScaled || (msg.geom && msg.geom.preScaled));
        const infoGeom = preScaledReq ? { ...(msg.geom || {}), preScaled: true } : (msg.geom || null);
        const geom = drawCroppedLetterboxedFrame(bmp, infoGeom);
        stageFoveaTileFromDrawable(bmp, bmp.width || geom.srcW, bmp.height || geom.srcH);
        const rawPixels = simCtx.getImageData(0, 0, frameW, frameH).data;
        const ingestRes = processIngestedFrame(
          rawPixels,
          msg.dtSec || 1.0 / 30.0,
          msg.mediaTimeSec,
          {
            epoch: msg.epoch,
            seq: msg.seq,
            ptsKnown: msg.ptsKnown,
            tSrcUs: msg.tSrcUs,
            captureAgeMs: msg.captureAgeMs,
            enqueuedEpochMs: msg.enqueuedEpochMs,
            gapMask: msg.gapMask,
            fallbackScaled: preScaledReq,
            fastPass: Boolean(msg.fastPass),
            fastPassPresent: Boolean(msg.fastPassPresent),
            geom,
          }
        );
        if (ingestRes && ingestRes.committed && ingestRes.status === 1) {
          ackPayload = {
            type: 'frame_ack',
            epoch: msg.epoch || 0,
            seq: msg.seq || 0,
            status: 1,
            committed: true,
          };
        } else {
          rejectPayload = {
            type: 'ingest_rejected',
            epoch: msg.epoch || 0,
            seq: msg.seq || 0,
            status: ingestRes ? ingestRes.status : 2,
            error: ingestRes && ingestRes.error ? ingestRes.error : undefined,
          };
        }
      } else {
        rejectPayload = {
          type: 'ingest_rejected',
          epoch: msg.epoch || 0,
          seq: msg.seq || 0,
          status: 2,
        };
      }
    } catch (err) {
      const errSt = wave1TelemView ? (wave1TelemView[8] >>> 0) : 2;
      rejectPayload = {
        type: 'ingest_rejected',
        epoch: msg.epoch || 0,
        seq: msg.seq || 0,
        status: errSt !== 1 ? errSt : 2,
        error: err && err.message ? err.message : 'video_frame_error',
      };
    } finally {
      if (bmp && typeof bmp.close === 'function') bmp.close();
      if (ackPayload) {
        self.postMessage(ackPayload);
      } else {
        self.postMessage(
          rejectPayload || {
            type: 'ingest_rejected',
            epoch: msg.epoch || 0,
            seq: msg.seq || 0,
            status: 2,
          }
        );
      }
    }
  } else if (msg.type === 'video_rgba') {
    let ackPayload = null;
    let rejectPayload = null;
    try {
      if (wasmExports && sourceMode !== 'testbench' && msg.buffer) {
        const srcPixels = new Uint8Array(msg.buffer);
        const preScaledReq = Boolean(msg.preScaled || msg.fallbackScaled || (msg.geom && msg.geom.preScaled));
        const hasDimSpec =
          Boolean(msg.geom) ||
          (typeof msg.width === 'number' && typeof msg.height === 'number');
        const baseGeom =
          msg.geom ||
          (typeof msg.width === 'number' && typeof msg.height === 'number'
            ? { srcW: msg.width, srcH: msg.height }
            : { srcW: frameW, srcH: frameH });
        const infoGeom = {
          ...baseGeom,
          ...(preScaledReq ? { preScaled: true } : {}),
        };
        const sw =
          typeof infoGeom.srcW === 'number' && infoGeom.srcW > 0
            ? Math.trunc(infoGeom.srcW)
            : (typeof msg.width === 'number' ? Math.trunc(msg.width) : frameW);
        const sh =
          typeof infoGeom.srcH === 'number' && infoGeom.srcH > 0
            ? Math.trunc(infoGeom.srcH)
            : (typeof msg.height === 'number' ? Math.trunc(msg.height) : frameH);
        let finalPixels = srcPixels;
        let finalGeom = null;
        if (sw <= 0 || sh <= 0 || srcPixels.byteLength !== sw * sh * 4) {
          rejectPayload = {
            type: 'ingest_rejected',
            epoch: msg.epoch || 0,
            seq: msg.seq || 0,
            status: 2,
            error: 'byteLength_mismatch',
          };
        } else {
          const effOrient =
            typeof infoGeom.orientation === 'number' &&
            Number.isInteger(infoGeom.orientation) &&
            infoGeom.orientation >= 0 &&
            infoGeom.orientation <= 7
              ? infoGeom.orientation
              : (roiOrientation & 7);
          const hasCropSpec =
            Boolean(infoGeom.visibleRect) ||
            typeof infoGeom.cropX0 === 'number' ||
            cropNorm[0] !== 0 ||
            cropNorm[1] !== 0 ||
            cropNorm[2] !== 1 ||
            cropNorm[3] !== 1;
          const needXform =
            hasDimSpec ||
            effOrient !== 0 ||
            hasCropSpec ||
            sw !== frameW ||
            sh !== frameH;
          if (needXform) {
            const xform = extractAndOrientRgba(srcPixels, sw, sh, infoGeom);
            finalPixels = xform.pixels;
            finalGeom = xform.geom;
          } else {
            finalGeom = resolveIngestGeomSpec(sw, sh, infoGeom);
            lastIngestGeom = finalGeom;
          }
        }
        if (!rejectPayload) {
          const ingestRes = processIngestedFrame(
            finalPixels,
            msg.dtSec || 1.0 / 30.0,
            msg.mediaTimeSec,
            {
              epoch: msg.epoch,
              seq: msg.seq,
              ptsKnown: msg.ptsKnown,
              tSrcUs: msg.tSrcUs,
              captureAgeMs: msg.captureAgeMs,
              enqueuedEpochMs: msg.enqueuedEpochMs,
              gapMask: msg.gapMask,
              fallbackScaled: preScaledReq,
              fastPass: Boolean(msg.fastPass),
              fastPassPresent: Boolean(msg.fastPassPresent),
              geom: finalGeom,
            }
          );
          if (ingestRes && ingestRes.committed && ingestRes.status === 1) {
            ackPayload = {
              type: 'frame_ack',
              epoch: msg.epoch || 0,
              seq: msg.seq || 0,
              status: 1,
              committed: true,
            };
          } else {
            rejectPayload = {
              type: 'ingest_rejected',
              epoch: msg.epoch || 0,
              seq: msg.seq || 0,
              status: ingestRes ? ingestRes.status : 2,
              error: ingestRes && ingestRes.error ? ingestRes.error : undefined,
            };
          }
        }
      } else {
        rejectPayload = {
          type: 'ingest_rejected',
          epoch: msg.epoch || 0,
          seq: msg.seq || 0,
          status: 2,
        };
      }
    } catch (err) {
      const errSt = wave1TelemView ? (wave1TelemView[8] >>> 0) : 2;
      rejectPayload = {
        type: 'ingest_rejected',
        epoch: msg.epoch || 0,
        seq: msg.seq || 0,
        status: errSt !== 1 ? errSt : 2,
        error: err && err.message ? err.message : 'video_rgba_error',
      };
    } finally {
      if (ackPayload) {
        self.postMessage(ackPayload);
      } else {
        self.postMessage(
          rejectPayload || {
            type: 'ingest_rejected',
            epoch: msg.epoch || 0,
            seq: msg.seq || 0,
            status: 2,
          }
        );
      }
    }
  } else if (msg.type === 'lab_action') {
    if (msg.action === 'saccade_step') {
      if (wasmExports) wasmExports.chronos_trigger_stimulus();
      if (sourceMode === 'testbench') {
        pendingSaccadeFrames = 9;
      }
    } else if (msg.action === 'ballistic_toss' && sourceMode === 'testbench') {
      if (benchScenario === 'model_rocket_flight') {
        simDiscs[0].x = simDiscs[0].baseX;
        simDiscs[0].y = 152 * resScale;
        simDiscs[0].vx = 0.35 * resScale;
        simDiscs[0].vy = -1.6 * resScale;
        simDiscs[0].ballistic = true;
        rocketPhase[0] = 0;
        rocketBoostTicks[0] = 0;
      } else {
        simDiscs[0].x = 24 * resScale;
        simDiscs[0].y = 148 * resScale;
        simDiscs[0].vx = 3.8 * resScale;
        simDiscs[0].vy = -6.8 * resScale;
        simDiscs[0].ballistic = true;
      }
    } else if (msg.action === 'air_strike' && sourceMode === 'testbench') {
      const nActiveStrike = getActiveSimDiscCount();
      for (let i = 0; i < nActiveStrike; i++) {
        simDiscs[i].vx = -simDiscs[i].vx * 1.45;
      }
      pendingAirSnapFrames = 7;
    } else if (msg.action === 'reset') {
      if (wasmExports) wasmExports.chronos_reset_tracks();
      resetTestbenchState();
    }
  } else if (msg.type === 'rec_control') {
    if (msg.action === 'start') {
      if (msg.clear || recCalSegments.length === 0) {
        recHead = 0;
        recCount = 0;
        recTotalWritten = 0;
        recElapsedSec = 0.0;
        recCalSegments = [
          {
            segmentId: calSegmentId,
            absStartFrame: 0,
            startRow: 0,
            elapsedSec: 0.0,
            reason: 'rec_start',
            sourceMode,
            calibration: getCalibrationSnapshot(),
          },
        ];
      } else {
        const lastSeg = recCalSegments[recCalSegments.length - 1];
        if (!lastSeg || lastSeg.segmentId !== calSegmentId || lastSeg.sourceMode !== sourceMode) {
          recCalSegments.push({
            segmentId: calSegmentId,
            absStartFrame: recTotalWritten,
            startRow: recCount,
            elapsedSec: Number(recElapsedSec.toFixed(4)),
            reason: 'rec_resume_sync',
            sourceMode,
            calibration: getCalibrationSnapshot(),
          });
        }
      }
      recActive = true;
    } else if (msg.action === 'stop') {
      recActive = false;
    } else if (msg.action === 'clear') {
      recActive = false;
      recHead = 0;
      recCount = 0;
      recTotalWritten = 0;
      recElapsedSec = 0.0;
      recCalSegments = [];
      resetRecEvents();
    }
    self.postMessage({
      type: 'rec_state',
      recActive,
      recFrames: recCount,
      recDurationSec: recElapsedSec,
    });
  } else if (msg.type === 'seed_box_preview') {
    if (msg.box && Number.isFinite(msg.box.x0) && Number.isFinite(msg.box.y0) && Number.isFinite(msg.box.x1) && Number.isFinite(msg.box.y1)) {
      seedBoxPreview = {
        x0: Math.max(0, Math.min(1, msg.box.x0)),
        y0: Math.max(0, Math.min(1, msg.box.y0)),
        x1: Math.max(0, Math.min(1, msg.box.x1)),
        y1: Math.max(0, Math.min(1, msg.box.y1)),
      };
    } else {
      seedBoxPreview = null;
    }
    if (hasAnyRenderedFrame()) {
      renderMainHud(currentDisplaySlot());
    }
  } else if (msg.type === 'seed_target') {
    seedBoxPreview = null;
    let tid = 0;
    let px = 0.0;
    let py = 0.0;
    let pw = 0.0;
    let ph = 0.0;
    if (wasmExports && typeof wasmExports.chronos_seed_target === 'function') {
      if (typeof msg.normX === 'number' && typeof msg.normY === 'number') {
        px = Math.max(0, Math.min(1, msg.normX)) * frameW;
        py = Math.max(0, Math.min(1, msg.normY)) * frameH;
        pw = Math.max(0, Math.min(1, Number(msg.normW) || 0)) * frameW;
        ph = Math.max(0, Math.min(1, Number(msg.normH) || 0)) * frameH;
      } else {
        px = Number(msg.x) || 0.0;
        py = Number(msg.y) || 0.0;
        pw = Number(msg.w) || 0.0;
        ph = Number(msg.h) || 0.0;
      }
      tid = wasmExports.chronos_seed_target(px, py, pw, ph) >>> 0;
    }
    self.postMessage({
      type: 'seed_target_ack',
      tid,
      x: px,
      y: py,
      w: pw,
      h: ph,
    });
    redrawPausedHudIfReady();
  } else if (msg.type === 'roi_box_preview') {
    if (msg.box && Number.isFinite(msg.box.x0) && Number.isFinite(msg.box.y0) && Number.isFinite(msg.box.x1) && Number.isFinite(msg.box.y1)) {
      roiBoxPreview = {
        x0: Math.max(0, Math.min(1, msg.box.x0)),
        y0: Math.max(0, Math.min(1, msg.box.y0)),
        x1: Math.max(0, Math.min(1, msg.box.x1)),
        y1: Math.max(0, Math.min(1, msg.box.y1)),
      };
    } else {
      roiBoxPreview = null;
    }
    if (hasAnyRenderedFrame()) {
      renderMainHud(currentDisplaySlot());
    }
  } else if (msg.type === 'set_auto_track_roi') {
    autoTrackRoiEnabled = Boolean(msg.enabled);
    if (hasAnyRenderedFrame()) {
      renderMainHud(currentDisplaySlot());
    }
  } else if (msg.type === 'worker_fast_pass_start') {
    runWorkerWebCodecsFastPass(msg);
  } else if (msg.type === 'worker_fast_pass_cancel') {
    workerFastPassCancel = true;
  } else if (msg.type === 'fast_pass_start') {
    fastPassActive = true;
    seedBoxPreview = null;
    roiBoxPreview = null;
    resetPersistenceState();
    if (wasmExports) {
      wasmExports.chronos_reset_tracks();
    }
    workerExpectedEngineFrame = 0;
  } else if (msg.type === 'fast_pass_end') {
    fastPassActive = false;
    emitTimelineEventsIfNeeded(true);
    if (hasAnyRenderedFrame()) {
      scheduleOffscreenHudPresent(currentDisplaySlot(), false, true);
      emitTelemetrySnapshot();
    }
  } else if (msg.type === 'export_telemetry') {
    const rows = recCount;
    const ordered = new Float64Array(rows * REC_COLS);
    if (rows > 0) {
      const oldest = rows < REC_MAX_FRAMES ? 0 : recHead;
      for (let i = 0; i < rows; i++) {
        const srcOff = ((oldest + i) % REC_MAX_FRAMES) * REC_COLS;
        ordered.set(recSlab.subarray(srcOff, srcOff + REC_COLS), i * REC_COLS);
      }
    }
    const winStartAbs = Math.max(0, recTotalWritten - rows);
    const exportedSegments = [];
    for (let i = 0; i < recCalSegments.length; i++) {
      const seg = recCalSegments[i];
      const nextAbs = i + 1 < recCalSegments.length ? recCalSegments[i + 1].absStartFrame : recTotalWritten;
      if (rows > 0 && nextAbs <= winStartAbs && i + 1 < recCalSegments.length) {
        continue;
      }
      const absStart = typeof seg.absStartFrame === 'number' ? seg.absStartFrame : seg.startRow;
      const startRow = Math.min(rows, Math.max(0, absStart - winStartAbs));
      exportedSegments.push({
        segmentId: seg.segmentId,
        startRow,
        elapsedSec: seg.elapsedSec,
        reason: seg.reason,
        sourceMode: seg.sourceMode,
        calibration: seg.calibration,
      });
    }
    self.postMessage(
      {
        type: 'telemetry_export_data',
        format: msg.format || 'csv',
        rows,
        cols: REC_COLS,
        columns: REC_COLUMNS,
        buffer: ordered.buffer,
        events: recEvents.slice(),
        sessionConfig: {
          resMode,
          frameW,
          frameH,
          lookaheadK,
          ecaAdaptRate,
          fistaLambda,
          eulerianGain,
          phaseOffsetRad,
          labMode,
          swarmMode,
          bosMode: bosReqMode,
          hodgeMode: hodgeReqMode,
          sourceMode,
          benchScenario,
          cropNorm: Array.from(cropNorm),
          nativeRoiEnabled,
        },
        meta: {
          schemaVersion: '1.5.0',
          sourceMode,
          benchScenario,
          resMode,
          frameW,
          frameH,
          pipDim,
          lookaheadK,
          ecaAdaptRate,
          fistaLambda,
          eulerianGain,
          phaseOffsetRad,
          labMode,
          calibration: getCalibrationSnapshot(),
          calibrationSegments: exportedSegments,
          chronos1000: (() => {
            const medCount = Math.min(Math.max(1, tickCounter), TIME_HIST_LEN);
            const medMs = tickCounter > 0 ? computeMedianTickMs(medCount) : 0.0;
            const effK = telemView ? intOrNull(telemView[1], 0, RING_LEN - 1) ?? lookaheadK : lookaheadK;
            const fIdxExp = (wave1CoreAbiValid && wave1TelemView && ringMetaView)
              ? currentCausalEngineFrameU32()
              : (telemView ? (intOrNull(telemView[0], 0, 0xffffffff) ?? 0) : 0);
            const dispSlotExp = currentDisplaySlot();
            const snap = buildChronos1000Snapshot(medMs, effK, fIdxExp, dispSlotExp);
            const bodies = [];
            const validMaskExp = Number.isInteger(snap.validityMask) ? snap.validityMask : 0;
            if (rigidBodiesView && hasChronos1000Exports() && (validMaskExp & 4) !== 0) {
              for (let b = 0; b < MAX_RIGID_BODIES; b++) {
                const rb = parseRigidBodyDescriptor(rigidBodiesView, b * BODY_STRIDE);
                if (rb) {
                  bodies.push(rb);
                }
              }
            }
            return {
              ...snap,
              rigidBodies: bodies,
            };
          })(),
        },
      },
      [ordered.buffer]
    );
  } else if (msg.type === 'audit_init') {
    try {
      const resp = await fetch(msg.wasmUrl || './chronos_spectral.wasm?v=20261115c1k');
      if (!resp.ok) throw new Error(`Wasm fetch failed (HTTP ${resp.status})`);
      const wasmBytes = await resp.arrayBuffer();
      const { instance } = await WebAssembly.instantiate(wasmBytes, {});
      wasmExports = instance.exports;
      wasmMemory = wasmExports.memory;
      wasmExports.chronos_init();
      telemView = new Float32Array(wasmMemory.buffer, wasmExports.chronos_get_telemetry_ptr(), TELEMETRY_LEN);
      fiedlerView = new Float32Array(wasmMemory.buffer, wasmExports.chronos_get_fiedler_ptr(), GRAPH_NODES);
      refreshWave1Views();
      refreshWave2Views();
      simCanvas = new OffscreenCanvas(320, 180);
      simCtx = simCanvas.getContext('2d', { willReadFrequently: true });
      running = false;
      self.postMessage({ type: 'audit_ready', version: '3.4.0', abi: 4 });
    } catch (err) {
      self.postMessage({ type: 'audit_error', slot: msg.slot || 'A', message: String(err && err.message ? err.message : err) });
    }
  } else if (msg.type === 'audit_cancel') {
    auditCancelSeq++;
  } else if (msg.type === 'audit_run') {
    await runWorkerPhysicsAudit(msg);
  }
};

// ============================================================================
// v3.4.0 Lazy Physics Integrity Audit Worker (`chronos-audit/1`, Zero Upload)
// ============================================================================

let auditCancelSeq = 0;

const AUDIT_AXIS_NAMES = [
  'trajectory',
  'collision',
  'depth_parallax',
  'fluid_div',
  'fluid_momentum',
  'bg_shimmer',
  'temporal_cadence',
  'av_sync',
];

function auditComputeLetterbox(srcW, srcH, dstW, dstH) {
  if (srcW <= 0 || srcH <= 0) return { sw: dstW, sh: dstH, dx: 0, dy: 0 };
  let sw = dstW;
  let sh = dstH;
  if (srcW * dstH >= dstW * srcH) {
    sw = dstW;
    sh = Math.max(1, Math.min(dstH, Math.round((srcH * dstW) / srcW)));
  } else {
    sh = dstH;
    sw = Math.max(1, Math.min(dstW, Math.round((srcW * dstH) / srcH)));
  }
  return { sw, sh, dx: Math.floor((dstW - sw) / 2), dy: Math.floor((dstH - sh) / 2) };
}

function auditSampleLumaSig(rgba, w, h) {
  const sig = new Float32Array(64);
  let idx = 0;
  for (let gy = 0; gy < 8; gy++) {
    const y0 = Math.floor((gy * h) / 8);
    const y1 = Math.max(y0 + 1, Math.floor(((gy + 1) * h) / 8));
    for (let gx = 0; gx < 8; gx++) {
      const x0 = Math.floor((gx * w) / 8);
      const x1 = Math.max(x0 + 1, Math.floor(((gx + 1) * w) / 8));
      let sum = 0;
      let cnt = 0;
      for (let py = y0; py < y1; py += 2) {
        const row = py * w;
        for (let px = x0; px < x1; px += 2) {
          const off = (row + px) * 4;
          sum += (rgba[off] * 77 + rgba[off + 1] * 150 + rgba[off + 2] * 29) >> 8;
          cnt++;
        }
      }
      sig[idx++] = cnt > 0 ? sum / cnt : 0;
    }
  }
  return sig;
}

function makeAuditSyntheticFrames(kind, nFrames = 48, w = 320, h = 180) {
  const frames = [];
  if (kind === 'static_decoy') {
    const black = new Uint8Array(w * h * 4);
    for (let i = 3; i < black.length; i += 4) black[i] = 255;
    for (let f = 0; f < 16; f++) frames.push(black);
    return { frames, fps: 30.0, profileFlags: 1 << 5, requiredMask: 1 << 0 };
  }
  const freezeAt = kind === 'stutter_freeze' ? 22 : -1;
  let x = 42.0;
  let y = 92.0;
  let vx = 4.2;
  let vy = -3.4;
  let prevBuf = null;
  for (let f = 0; f < nFrames; f++) {
    if (freezeAt >= 0 && f >= freezeAt && f < freezeAt + 5 && prevBuf) {
      frames.push(new Uint8Array(prevBuf));
      continue;
    }
    x += vx;
    y += vy;
    if (x < 22 || x > w - 22) { vx = -vx; x = Math.max(22, Math.min(w - 22, x)); }
    if (y < 22 || y > h - 22) { vy = -vy; y = Math.max(22, Math.min(h - 22, y)); }
    const buf = new Uint8Array(w * h * 4);
    for (let i = 0; i < buf.length; i += 4) {
      buf[i] = 22; buf[i + 1] = 26; buf[i + 2] = 34; buf[i + 3] = 255;
    }
    const cx = Math.round(x);
    const cy = Math.round(y);
    for (let dy = -11; dy <= 11; dy++) {
      const py = cy + dy;
      if (py < 0 || py >= h) continue;
      for (let dx = -11; dx <= 11; dx++) {
        const px = cx + dx;
        if (px < 0 || px >= w || dx * dx + dy * dy > 100) continue;
        const off = (py * w + px) * 4;
        buf[off] = 244; buf[off + 1] = 230; buf[off + 2] = 185; buf[off + 3] = 255;
      }
    }
    prevBuf = buf;
    frames.push(buf);
  }
  return { frames, fps: 30.0, profileFlags: freezeAt >= 0 ? (1 << 5) : 0, requiredMask: 0 };
}

async function runWorkerPhysicsAudit(msg) {
  const slot = msg.slot === 'B' ? 'B' : 'A';
  const mySeq = ++auditCancelSeq;
  if (!wasmExports || !wasmMemory) {
    self.postMessage({ type: 'audit_error', slot, message: 'Audit Wasm engine not initialized' });
    return;
  }

  const profileName = String(msg.profile || 'quick');
  const useDenseFluid = profileName === 'fluid_dense' || profileName === 'standard';
  const resModeReq = profileName === 'standard' ? 0 : 0;
  const ew = resModeReq === 1 ? 640 : 320;
  const eh = resModeReq === 1 ? 360 : 180;
  const maxFrames = Math.max(12, Math.min(360, Number(msg.maxFrames) || 120));
  const tStartMs = performance.now();

  try {
    wasmExports.chronos_init();
    wasmExports.chronos_set_resolution(resModeReq);
    wasmExports.chronos_set_swarm_mode(0);
    wasmExports.chronos_set_bos_mode(0);
    wasmExports.chronos_set_depth_mode(profileName === 'standard' ? 1 : 0);
    wasmExports.chronos_set_hodge_mode(useDenseFluid ? 1 : 0);
    wasmExports.chronos_set_wind_mode(useDenseFluid ? 3 : 0);
    telemView = new Float32Array(wasmMemory.buffer, wasmExports.chronos_get_telemetry_ptr(), TELEMETRY_LEN);
    refreshWave1Views();
    refreshWave2Views();
    if (!simCanvas || simCanvas.width !== ew || simCanvas.height !== eh) {
      simCanvas = new OffscreenCanvas(ew, eh);
      simCtx = simCanvas.getContext('2d', { willReadFrequently: true });
    }

    const applFrames = new Uint32Array(8);
    const warnFrames = new Uint32Array(8);
    const violFrames = new Uint32Array(8);
    const worstResid = new Float64Array(8);
    const hasResid = new Uint8Array(8);
    const violations = [];
    const timeline = [];

    let profileFlags = Number(msg.profileFlags || 0) >>> 0;
    let requiredMask = Number(msg.requiredMask || 0) >>> 0;
    let framesDecoded = 0;
    let framesCommitted = 0;
    let shots = 0;
    let lastEpoch = 0;
    let lastPtsUs = -1;
    let prevSig = null;
    let frozenRun = 0;
    let odeLowStreak = 0;
    let fluidDivStreak = 0;
    let fluidNsStreak = 0;
    let odeR2Sum = 0;
    let odeR2Cnt = 0;
    let gSignPos = 0;
    let gSignNeg = 0;
    let srcW = ew;
    let srcH = eh;
    let lbRect = [0, 0, ew, eh];
    let audioPresent = false;
    const prevTracks = Array.from({ length: 16 }, () => ({ tid: 0, x: 0, y: 0, spd: 0, f: 0 }));

    const updateResid = (ax, v) => {
      if (Number.isFinite(v) && (!hasResid[ax] || v > worstResid[ax])) {
        worstResid[ax] = v;
        hasResid[ax] = 1;
      }
    };
    const pushSpan = (axisId, code, frame, tUs, dtUs, severity, measured, expected) => {
      const sev = Math.max(0, Math.min(1, Number.isFinite(severity) ? severity : 0.5));
      const t1 = tUs + Math.max(1, dtUs);
      for (let i = violations.length - 1; i >= 0; i--) {
        const s = violations[i];
        if (s.axis_id === axisId && s.code === code && frame <= s.frames[1] + 4) {
          s.frames[1] = frame;
          s.t_us[1] = Math.max(s.t_us[1], t1);
          if (sev > s.severity) {
            s.severity = Number(sev.toFixed(4));
            s.measured = Number(measured.toFixed(4));
            s.expected = Number(expected.toFixed(4));
          }
          return;
        }
      }
      if (violations.length < 128) {
        violations.push({
          axis: AUDIT_AXIS_NAMES[axisId],
          axis_id: axisId,
          code,
          frames: [frame, frame],
          t_us: [tUs, t1],
          severity: Number(sev.toFixed(4)),
          measured: Number(measured.toFixed(4)),
          expected: Number(expected.toFixed(4)),
        });
      }
    };

    const stepOne = (rgba, dtSec, ptsUs, seq, sw, sh, lb) => {
      const sig = auditSampleLumaSig(rgba, ew, eh);
      const ptr = wasmExports.chronos_get_ingest_rgba_ptr();
      new Uint8Array(wasmMemory.buffer, ptr, ew * eh * 4).set(rgba);
      const ptsLo = ptsUs >>> 0;
      const ptsHi = Math.floor(ptsUs / 4294967296) >>> 0;
      wasmExports.chronos_set_ingest_meta(
        1, seq, ptsLo, ptsHi, -1.0,
        (1 << 1) | (1 << 4), sw, sh, 0, 0, sw, sh, 0, 0xFFFFFFFF,
        lb[0], lb[1], lb[2], lb[3], 0, (sw === ew && sh === eh) ? 1 : 0
      );
      wasmExports.chronos_pre_tick(dtSec);
      wasmExports.chronos_post_tick(dtSec);

      const frameIdx = framesDecoded++;
      const status = wave1TelemView ? (wave1TelemView[8] >>> 0) : 0;
      if (status === 2 || status === 3 || status === 4) {
        timeline.push(0);
        return;
      }
      framesCommitted++;
      const ep = wave1TelemView ? Math.max(1, wave1TelemView[6] >>> 0) : 1;
      const epChanged = lastEpoch !== 0 && ep !== lastEpoch;
      if (shots === 0 || epChanged) {
        shots++;
        frozenRun = 0;
        odeLowStreak = 0;
        fluidDivStreak = 0;
        fluidNsStreak = 0;
        prevSig = null;
        for (let m = 0; m < 16; m++) prevTracks[m].tid = 0;
      }
      lastEpoch = ep;

      const dtUs = Math.round(Math.max(0.001, Math.min(1.0, dtSec)) * 1_000_000);
      let frameSev = 0;
      const camMoving = Math.round(telemView[1257]) !== 0;
      const activeN = Math.round(telemView[3]);
      let maxSpdF = 0;
      for (let m = 0; m < 16; m++) {
        if (telemView[32 + m * 56] > 0.5) {
          const sp = telemView[1120 + m * 8 + 1] * dtSec;
          if (Number.isFinite(sp) && sp > maxSpdF) maxSpdF = sp;
        }
      }
      let lumaMad = 1.0;
      let maxTileDiff = 255.0;
      if (prevSig) {
        let acc = 0;
        maxTileDiff = 0;
        for (let k = 0; k < 64; k++) {
          const d = Math.abs(prevSig[k] - sig[k]);
          acc += d;
          if (d > maxTileDiff) maxTileDiff = d;
        }
        lumaMad = acc / (64.0 * 255.0);
      }
      const hasMotion = activeN > 0 && maxSpdF >= 1.5;
      if (prevSig && !epChanged) {
        if (lumaMad <= 0.0006 && maxTileDiff <= 0.35 && hasMotion) {
          frozenRun++;
          if (frozenRun >= 2) {
            applFrames[6]++;
            updateResid(6, frozenRun);
            if ((profileFlags & (1 << 5)) !== 0) {
              violFrames[6]++;
              frameSev = Math.max(frameSev, 0.8);
              pushSpan(6, 'FROZEN_MID_MOTION', frameIdx, ptsUs, dtUs, 0.8, frozenRun, 0);
            } else {
              warnFrames[6]++;
              frameSev = Math.max(frameSev, 0.35);
            }
          }
        } else {
          frozenRun = 0;
          if (hasMotion) {
            applFrames[6]++;
            updateResid(6, lumaMad);
          }
        }
      }
      prevSig = sig;
      lastPtsUs = ptsUs;

      // Trajectory & Collision
      let hadTraj = false;
      let bestR2 = -1;
      for (let m = 0; m < 16; m++) {
        const base = 32 + m * 56;
        if (telemView[base] < 0.5) { prevTracks[m].tid = 0; continue; }
        const tid = Math.round(telemView[base + 1]);
        const px = telemView[base + 8];
        const py = telemView[base + 9];
        const kbase = 1120 + m * 8;
        const spd = telemView[kbase + 1];
        const strike = telemView[kbase + 4] > 0.5;
        const restE = telemView[kbase + 5];
        const pbase = 1280 + m * 16;
        const fitR2 = telemView[pbase + 9];
        const lawClass = Math.round(telemView[pbase + 10]);
        const c0y = telemView[pbase + 3];
        if (lawClass >= 1 && Number.isFinite(fitR2) && fitR2 >= 0) {
          hadTraj = true;
          if (fitR2 > bestR2) bestR2 = fitR2;
          if (lawClass === 1 && !camMoving && Math.abs(c0y) > 40) {
            if (c0y > 0) gSignPos++; else gSignNeg++;
          }
        }
        const pt = prevTracks[m];
        if (pt.tid === tid && frameIdx === pt.f + 1) {
          const jump = Math.hypot(px - pt.x, py - pt.y);
          const reach = Math.max(pt.spd, spd) * dtSec * 3.5 + 36.0;
          if (jump > reach && jump > 48.0) {
            applFrames[6]++;
            violFrames[6]++;
            frameSev = Math.max(frameSev, 0.8);
            updateResid(6, jump / reach);
            pushSpan(6, 'TELEPORT_JUMP', frameIdx, ptsUs, dtUs, 0.8, jump, reach);
          }
        }
        prevTracks[m] = { tid, x: px, y: py, spd, f: frameIdx };
        if (strike && Number.isFinite(restE) && restE > 0.05 && restE < 5.0) {
          applFrames[1]++;
          updateResid(1, Math.max(0, restE - 1.0));
          if (restE > 1.18) {
            violFrames[1]++;
            frameSev = Math.max(frameSev, 0.85);
            pushSpan(1, 'SUPERELASTIC_RESTITUTION', frameIdx, ptsUs, dtUs, 0.85, restE, 1.0);
          }
        }
      }
      if (hadTraj && !camMoving) {
        applFrames[0]++;
        if (bestR2 >= 0) {
          odeR2Sum += bestR2;
          odeR2Cnt++;
          updateResid(0, Math.max(0, 1.0 - bestR2));
          if (bestR2 < 0.72) {
            odeLowStreak++;
            if (odeLowStreak >= 4) {
              violFrames[0]++;
              frameSev = Math.max(frameSev, 0.75);
              pushSpan(0, 'ODE_LAW_BREAK', frameIdx, ptsUs, dtUs, 0.75, bestR2, 0.85);
            }
          } else {
            odeLowStreak = 0;
          }
        }
      }

      // Fluid axes (dense LK or sparse wind tunnel)
      if (useDenseFluid && windTelemView && windTelemF32View) {
        const wst = windTelemView[0] >>> 0;
        const wreason = windTelemView[2] >>> 0;
        const lowSup = (wreason & (1 << 9)) !== 0;
        const highDiv = (wreason & (1 << 3)) !== 0;
        const highNs = (wreason & (1 << 4)) !== 0;
        const divFrac = windTelemF32View[22];
        const nsResid = windTelemF32View[21];
        if ((wst === 2 || wst === 3 || (wst === 4 && highDiv)) && !lowSup && Number.isFinite(divFrac)) {
          applFrames[3]++;
          updateResid(3, divFrac);
          if (highDiv || divFrac > 0.30) {
            fluidDivStreak++;
            if (fluidDivStreak >= 3) {
              if ((profileFlags & (1 << 2)) !== 0) {
                violFrames[3]++;
                frameSev = Math.max(frameSev, 0.8);
                pushSpan(3, 'FLUID_MASS_LEAK', frameIdx, ptsUs, dtUs, 0.8, divFrac, 0.30);
              } else {
                warnFrames[3]++;
                frameSev = Math.max(frameSev, 0.35);
              }
            }
          } else {
            fluidDivStreak = 0;
          }
        }
        if ((wst === 2 || wst === 3 || (wst === 4 && highNs)) && !lowSup && Number.isFinite(nsResid)) {
          applFrames[4]++;
          updateResid(4, nsResid);
          if (highNs || nsResid > 0.35) {
            fluidNsStreak++;
            if (fluidNsStreak >= 3) {
              violFrames[4]++;
              frameSev = Math.max(frameSev, 0.8);
              pushSpan(4, 'FLUID_NONCONSERVATIVE', frameIdx, ptsUs, dtUs, 0.8, nsResid, 0.35);
            }
          } else {
            fluidNsStreak = 0;
          }
        }
      }
      timeline.push(frameSev);
    };

    if (msg.syntheticKind) {
      const syn = makeAuditSyntheticFrames(msg.syntheticKind, maxFrames, ew, eh);
      profileFlags = (profileFlags | syn.profileFlags) >>> 0;
      requiredMask = (requiredMask | syn.requiredMask) >>> 0;
      const dtSec = 1.0 / syn.fps;
      for (let i = 0; i < syn.frames.length; i++) {
        if (auditCancelSeq !== mySeq) return;
        stepOne(syn.frames[i], dtSec, Math.round(i * dtSec * 1_000_000), i + 1, ew, eh, [0, 0, ew, eh]);
        if ((i % 10) === 0 || i + 1 === syn.frames.length) {
          self.postMessage({ type: 'audit_progress', slot, frame: i + 1, total: syn.frames.length });
        }
      }
    } else {
      let rawBuf = null;
      if (msg.arrayBuffer instanceof ArrayBuffer) {
        rawBuf = msg.arrayBuffer;
      } else if (msg.file && typeof msg.file.arrayBuffer === 'function') {
        rawBuf = await msg.file.arrayBuffer();
      } else if (typeof msg.url === 'string' && msg.url.length > 0) {
        const resp = await fetch(msg.url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        rawBuf = await resp.arrayBuffer();
      } else {
        throw new Error('No clip source provided');
      }

      const track = parseMp4TrackMultiCodec(rawBuf);
      if (!track || track.samples.length === 0) throw new Error('Unsupported MP4 container or video codec');
      const sampleCount = Math.min(track.samples.length, maxFrames);
      const decConfig = { codec: track.codec };
      if (track.description) decConfig.description = track.description;
      if (track.codedWidth > 0) decConfig.codedWidth = track.codedWidth;
      if (track.codedHeight > 0) decConfig.codedHeight = track.codedHeight;

      const audioPcm = msg.audioPcm instanceof Float32Array ? msg.audioPcm : null;
      const audioRate = Number(msg.audioRateHz) || 48000;
      audioPresent = Boolean(audioPcm && audioPcm.length > 0);
      let audioCursor = 0;

      let decErr = null;
      let chain = Promise.resolve();
      let prevTsUs = -1;
      let seq = 0;
      const decoder = new VideoDecoder({
        output: (vf) => {
          if (auditCancelSeq !== mySeq || decErr) { vf.close(); return; }
          chain = chain.then(() => {
            try {
              if (auditCancelSeq !== mySeq || decErr || seq >= maxFrames) return;
              srcW = Math.max(1, Math.trunc(vf.displayWidth || vf.codedWidth || ew));
              srcH = Math.max(1, Math.trunc(vf.displayHeight || vf.codedHeight || eh));
              const lb = auditComputeLetterbox(srcW, srcH, ew, eh);
              lbRect = [lb.dx, lb.dy, lb.sw, lb.sh];
              const ptsUs = typeof vf.timestamp === 'number' && Number.isFinite(vf.timestamp)
                ? Math.max(0, Math.round(vf.timestamp))
                : Math.round((seq / 30.0) * 1_000_000);
              const dtSec = prevTsUs >= 0 && ptsUs > prevTsUs
                ? Math.max(0.005, Math.min(0.25, (ptsUs - prevTsUs) * 1e-6))
                : (1.0 / 30.0);
              prevTsUs = ptsUs;
              seq++;

              if (audioPcm && audioStagingView) {
                const sNext = Math.min(audioPcm.length, Math.round((ptsUs * 1e-6 + dtSec) * audioRate));
                while (audioCursor < sNext) {
                  const chunk = Math.min(4096, sNext - audioCursor);
                  audioStagingView.set(audioPcm.subarray(audioCursor, audioCursor + chunk), 0);
                  audioCursor += chunk;
                  const tEndUs = Math.round((audioCursor / audioRate) * 1_000_000);
                  wasmExports.chronos_push_audio_block(chunk, tEndUs >>> 0, Math.floor(tEndUs / 4294967296) >>> 0, audioRate, 0.0);
                }
              }

              simCtx.fillStyle = '#000';
              simCtx.fillRect(0, 0, ew, eh);
              simCtx.drawImage(vf, 0, 0, srcW, srcH, lb.dx, lb.dy, lb.sw, lb.sh);
              const px = simCtx.getImageData(0, 0, ew, eh).data;
              stepOne(px, dtSec, ptsUs, seq, srcW, srcH, lbRect);
              if ((seq % 10) === 0 || seq === sampleCount) {
                self.postMessage({ type: 'audit_progress', slot, frame: seq, total: sampleCount });
              }
            } finally {
              vf.close();
            }
          }).catch((e) => { decErr = e; });
        },
        error: (e) => { decErr = e || new Error('VideoDecoder error'); },
      });
      decoder.configure(decConfig);
      for (let i = 0; i < sampleCount; i++) {
        if (auditCancelSeq !== mySeq || decErr) break;
        while (decoder.decodeQueueSize > 8 && auditCancelSeq === mySeq && !decErr) {
          await new Promise((r) => setTimeout(r, 1));
        }
        const s = track.samples[i];
        decoder.decode(new EncodedVideoChunk({
          type: s.isKey ? 'key' : 'delta',
          timestamp: s.timestampUs,
          duration: s.durationUs,
          data: track.bytes.subarray(s.offset, s.offset + s.size),
        }));
      }
      if (auditCancelSeq === mySeq && !decErr && decoder.state === 'configured') {
        await decoder.flush();
      }
      await chain;
      if (decoder.state !== 'closed') { try { decoder.close(); } catch (_e) {} }
      if (decErr) throw decErr;
      if (auditCancelSeq !== mySeq) return;
    }

    if (gSignPos >= 6 && gSignNeg >= 6) {
      violFrames[0] += 4;
      pushSpan(0, 'G_SIGN_FLIP', 0, 0, 33333, 0.9, -1.0, 1.0);
    }

    const minAppl = 6;
    const enabledMask = (1 << 0) | (1 << 1) | (1 << 6) | (useDenseFluid ? ((1 << 3) | (1 << 4)) : 0) | (profileName === 'standard' ? (1 << 2) : 0);
    const axes = [];
    let applicableAxes = 0;
    let violatedAxes = 0;
    let warnedAxes = 0;
    let scoreSum = 0;

    for (let ax = 0; ax < 8; ax++) {
      const enabled = (enabledMask & (1 << ax)) !== 0;
      const af = applFrames[ax];
      const wf = warnFrames[ax];
      const vf = violFrames[ax];
      let applicability = 'unknown';
      let verdict = 'none';
      let score = null;
      const naReasons = [];
      if (!enabled) {
        applicability = 'absent';
        naReasons.push('DISABLED');
      } else if (ax === 7 && !audioPresent) {
        applicability = 'absent';
        naReasons.push('NO_AUDIO');
      } else if (af === 0) {
        applicability = 'unknown';
        naReasons.push('NO_EVIDENCE');
      } else if (af < (ax === 1 ? 1 : minAppl) && vf === 0) {
        applicability = 'insufficient';
        naReasons.push('TOO_SHORT');
      } else {
        applicability = 'eligible';
        applicableAxes++;
        const vfFrac = vf / Math.max(1, af);
        const wfFrac = wf / Math.max(1, af);
        let sc = Math.max(0, Math.min(1, 1.0 - vfFrac - 0.35 * wfFrac));
        if (ax === 0 && odeR2Cnt > 0) sc = Math.min(sc, Math.max(0, Math.min(1, odeR2Sum / odeR2Cnt)));
        score = Number(sc.toFixed(4));
        if (vf > 0) { verdict = 'violated'; violatedAxes++; }
        else if (wf > 0) { verdict = 'warn'; warnedAxes++; }
        else { verdict = 'pass'; }
        scoreSum += sc;
      }
      axes.push({
        id: AUDIT_AXIS_NAMES[ax],
        axis_id: ax,
        applicability,
        applicable: applicability === 'eligible',
        verdict,
        pass: verdict === 'none' ? null : verdict !== 'violated',
        na_reasons: naReasons,
        applicable_frames: af,
        warned_frames: wf,
        violated_frames: vf,
        score,
        worst_residual: hasResid[ax] ? Number(worstResid[ax].toFixed(4)) : null,
      });
    }

    const integrity = applicableAxes > 0 ? Number((scoreSum / applicableAxes).toFixed(4)) : null;
    const coverage = Number((framesCommitted > 0 ? Math.min(1, Math.max(...Array.from(applFrames)) / Math.max(1, framesCommitted)) : 0).toFixed(4));
    let grade = 'NA';
    if (applicableAxes > 0) {
      grade = violatedAxes > 0 ? 'FAIL' : (warnedAxes > 0 ? 'WARN' : 'PASS');
    }
    let rewardMilli = null;
    if (requiredMask !== 0) {
      let missingReq = false;
      let acc = 0;
      let cnt = 0;
      for (let ax = 0; ax < 8; ax++) {
        if ((requiredMask & (1 << ax)) === 0) continue;
        cnt++;
        if (axes[ax].applicability !== 'eligible') { missingReq = true; acc -= 1; }
        else if (axes[ax].verdict === 'violated') acc -= 1;
        else if (axes[ax].verdict === 'pass') acc += 1;
      }
      rewardMilli = missingReq || coverage < 0.15 ? -1000 : Math.round((acc / Math.max(1, cnt)) * 1000);
    } else if (applicableAxes > 0) {
      rewardMilli = Math.round(((integrity ?? 0) * 2 - 1) * 1000);
    }

    const record = {
      schema: 'chronos-audit/1',
      clip: { path: String(msg.clipName || msg.url || msg.syntheticKind || '<clip>') },
      status: 'ok',
      error: null,
      verdict: {
        grade,
        integrity,
        coverage,
        applicable_axes: applicableAxes,
        violated_axes: violatedAxes,
        warned_axes: warnedAxes,
        violations: violations.length,
      },
      rlvr: {
        reward: rewardMilli === null ? null : Number((rewardMilli / 1000).toFixed(4)),
        reward_milli: rewardMilli,
      },
      axes,
      violations,
      shots,
      ingest: {
        decoder: msg.syntheticKind ? 'synthetic' : 'webcodecs',
        src_w: srcW,
        src_h: srcH,
        frames_decoded: framesDecoded,
        frames_committed: framesCommitted,
        letterbox: lbRect,
        audio: { present: audioPresent },
      },
      engine: { version: '3.4.0', abi: 4, law_set: 'core-8' },
      perf: { wall_s: Number(((performance.now() - tStartMs) / 1000).toFixed(3)) },
      timeline,
    };
    self.postMessage({ type: 'audit_done', slot, record });
  } catch (err) {
    self.postMessage({ type: 'audit_error', slot, message: String(err && err.message ? err.message : err) });
  }
}

// ============================================================================
// Dual-Resolution 1:1 Fovea Staging & Worker-Resident WebCodecs Fast-Pass
// ============================================================================

function stageFoveaTileFromDrawable(drawable, srcW, srcH) {
  if (!foveaAbiValid || !foveaRgbaView || !foveaReqView || !wasmExports || !drawable) return false;
  if (typeof wasmExports.chronos_set_fovea_meta !== 'function') return false;
  const active = foveaReqView[0] >>> 0;
  const sw = Math.max(0, Math.trunc(srcW || 0));
  const sh = Math.max(0, Math.trunc(srcH || 0));
  if (active === 0 || sw <= frameW || sh <= frameH) {
    wasmExports.chronos_set_fovea_meta(0, 0, 0, 0, 0, 0, 0, 0);
    return false;
  }
  const fx0 = foveaReqView[1] >>> 0;
  const fy0 = foveaReqView[2] >>> 0;
  const fw = Math.min(FOVEA_DIM, foveaReqView[3] >>> 0);
  const fh = Math.min(FOVEA_DIM, foveaReqView[4] >>> 0);
  const tid = foveaReqView[5] >>> 0;
  if (fw === 0 || fh === 0 || tid === 0 || fx0 + fw > sw || fy0 + fh > sh) {
    wasmExports.chronos_set_fovea_meta(0, 0, 0, 0, 0, 0, 0, 0);
    return false;
  }
  if (!foveaStageCanvas || !foveaStageCtx) {
    if (typeof OffscreenCanvas !== 'function') {
      wasmExports.chronos_set_fovea_meta(0, 0, 0, 0, 0, 0, 0, 0);
      return false;
    }
    foveaStageCanvas = new OffscreenCanvas(FOVEA_DIM, FOVEA_DIM);
    foveaStageCtx = foveaStageCanvas.getContext('2d', { willReadFrequently: true });
  }
  if (!foveaStageCtx) {
    wasmExports.chronos_set_fovea_meta(0, 0, 0, 0, 0, 0, 0, 0);
    return false;
  }
  foveaStageCtx.imageSmoothingEnabled = false;
  foveaStageCtx.drawImage(drawable, fx0, fy0, fw, fh, 0, 0, fw, fh);
  const img = foveaStageCtx.getImageData(0, 0, fw, fh).data;
  foveaRgbaView.set(img.subarray(0, fw * fh * 4), 0);
  return (wasmExports.chronos_set_fovea_meta(0, sw, sh, fx0, fy0, fw, fh, tid) >>> 0) === 1;
}

let webCodecsCopyToRgbaState = 0; // 0=unknown, 1=supported, -1=fallback to OffscreenCanvas

async function stageFoveaTileFromVideoFrame(vf, srcW, srcH) {
  if (!foveaAbiValid || !foveaRgbaView || !foveaReqView || !wasmExports || !vf) return false;
  if (typeof wasmExports.chronos_set_fovea_meta !== 'function') return false;
  const active = foveaReqView[0] >>> 0;
  const sw = Math.max(0, Math.trunc(srcW || 0));
  const sh = Math.max(0, Math.trunc(srcH || 0));
  if (active === 0 || sw <= frameW || sh <= frameH) {
    wasmExports.chronos_set_fovea_meta(0, 0, 0, 0, 0, 0, 0, 0);
    return false;
  }
  const fx0 = foveaReqView[1] >>> 0;
  const fy0 = foveaReqView[2] >>> 0;
  const fw = Math.min(FOVEA_DIM, foveaReqView[3] >>> 0);
  const fh = Math.min(FOVEA_DIM, foveaReqView[4] >>> 0);
  const tid = foveaReqView[5] >>> 0;
  if (fw === 0 || fh === 0 || tid === 0 || fx0 + fw > sw || fy0 + fh > sh) {
    wasmExports.chronos_set_fovea_meta(0, 0, 0, 0, 0, 0, 0, 0);
    return false;
  }
  if (webCodecsCopyToRgbaState >= 0 && typeof vf.copyTo === 'function') {
    try {
      const dst = foveaRgbaView.subarray(0, fw * fh * 4);
      await vf.copyTo(dst, {
        rect: { x: fx0, y: fy0, width: fw, height: fh },
        format: 'RGBA',
      });
      webCodecsCopyToRgbaState = 1;
      return (wasmExports.chronos_set_fovea_meta(0, sw, sh, fx0, fy0, fw, fh, tid) >>> 0) === 1;
    } catch (_err) {
      webCodecsCopyToRgbaState = -1;
    }
  }
  return stageFoveaTileFromDrawable(vf, sw, sh);
}

function parseMp4TrackMultiCodec(buf) {
  const bytes = new Uint8Array(buf);
  const dv = new DataView(buf);
  const len = bytes.byteLength;

  function boxType(off) {
    return String.fromCharCode(bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]);
  }
  function u64(off) {
    return dv.getUint32(off) * 4294967296 + dv.getUint32(off + 4);
  }
  function iterBoxes(start, end, cb) {
    let off = start;
    while (off + 8 <= end) {
      let size = dv.getUint32(off);
      const type = boxType(off + 4);
      let hdr = 8;
      if (size === 1) {
        if (off + 16 > end) break;
        size = u64(off + 8);
        hdr = 16;
      } else if (size === 0) {
        size = end - off;
      }
      if (size < hdr || off + size > end) break;
      cb(type, off, off + hdr, off + size);
      off += size;
    }
  }
  function findChild(start, end, target) {
    let found = null;
    iterBoxes(start, end, (type, boxStart, dataStart, boxEnd) => {
      if (!found && type === target) {
        found = { type, boxStart, dataStart, boxEnd };
      }
    });
    return found;
  }

  function hex2(v) {
    return (v & 0xff).toString(16).toUpperCase().padStart(2, '0');
  }
  function buildHevcCodecString(cfg, fourcc) {
    if (cfg.byteLength < 13) return `${fourcc}.1.6.L93.B0`;
    const b1 = cfg[1];
    const spaceCode = (b1 >> 6) & 3;
    const tierFlag = (b1 >> 5) & 1;
    const profileIdc = b1 & 0x1f;
    const spaceStr = spaceCode === 1 ? 'A' : (spaceCode === 2 ? 'B' : (spaceCode === 3 ? 'C' : ''));
    const compat = ((cfg[2] << 24) | (cfg[3] << 16) | (cfg[4] << 8) | cfg[5]) >>> 0;
    let revCompat = 0;
    for (let i = 0; i < 32; i++) {
      if ((compat >>> (31 - i)) & 1) revCompat |= (1 << i);
    }
    const compatHex = (revCompat >>> 0).toString(16).toUpperCase();
    const levelIdc = cfg[12];
    const tierChar = tierFlag ? 'H' : 'L';
    const parts = [fourcc, `${spaceStr}${profileIdc}`, compatHex, `${tierChar}${levelIdc}`];
    let lastNonZero = -1;
    for (let i = 0; i < 6; i++) {
      if (cfg[6 + i] !== 0) lastNonZero = i;
    }
    if (lastNonZero >= 0) {
      for (let i = 0; i <= lastNonZero; i++) {
        parts.push(cfg[6 + i].toString(16).toUpperCase());
      }
    }
    return parts.join('.');
  }
  function buildVp9CodecString(cfg) {
    // vpcC is a FullBox (4-byte version+flags header)
    if (cfg.byteLength < 8) return 'vp09.00.10.08';
    const profile = String(cfg[4]).padStart(2, '0');
    const level = String(cfg[5]).padStart(2, '0');
    const bitDepth = String((cfg[6] >> 4) & 0x0f).padStart(2, '0');
    return `vp09.${profile}.${level}.${bitDepth}`;
  }
  function buildAv1CodecString(cfg) {
    if (cfg.byteLength < 4) return 'av01.0.04M.08';
    const profile = (cfg[1] >> 5) & 0x07;
    const level = cfg[1] & 0x1f;
    const tier = (cfg[2] & 0x80) ? 'H' : 'M';
    const highBd = (cfg[2] >> 6) & 1;
    const twelve = (cfg[2] >> 5) & 1;
    const bitDepth = highBd ? (twelve ? 12 : 10) : 8;
    return `av01.${profile}.${String(level).padStart(2, '0')}${tier}.${String(bitDepth).padStart(2, '0')}`;
  }

  const moov = findChild(0, len, 'moov');
  if (!moov) return null;

  let movieTimescale = 1000;
  const mvhd = findChild(moov.dataStart, moov.boxEnd, 'mvhd');
  if (mvhd && mvhd.dataStart + 20 <= mvhd.boxEnd) {
    const mVer = bytes[mvhd.dataStart];
    const mts = mVer === 1 && mvhd.dataStart + 28 <= mvhd.boxEnd
      ? dv.getUint32(mvhd.dataStart + 20)
      : dv.getUint32(mvhd.dataStart + 12);
    if (mts > 0) movieTimescale = mts;
  }

  function i64(off) {
    return dv.getInt32(off) * 4294967296 + dv.getUint32(off + 4);
  }

  // Parse mvex/trex defaults for fragmented fMP4
  const trexByTrack = new Map();
  const mvex = findChild(moov.dataStart, moov.boxEnd, 'mvex');
  if (mvex) {
    iterBoxes(mvex.dataStart, mvex.boxEnd, (type, _bs, ds, be) => {
      if (type === 'trex' && ds + 24 <= be) {
        const trackId = dv.getUint32(ds + 4);
        trexByTrack.set(trackId, {
          defaultSampleDescIdx: dv.getUint32(ds + 8),
          defaultSampleDuration: dv.getUint32(ds + 12),
          defaultSampleSize: dv.getUint32(ds + 16),
          defaultSampleFlags: dv.getUint32(ds + 20),
        });
      }
    });
  }

  let selected = null;
  iterBoxes(moov.dataStart, moov.boxEnd, (type, _bs, ds, be) => {
    if (selected || type !== 'trak') return;
    const mdia = findChild(ds, be, 'mdia');
    if (!mdia) return;
    const hdlr = findChild(mdia.dataStart, mdia.boxEnd, 'hdlr');
    if (!hdlr || hdlr.dataStart + 12 > hdlr.boxEnd) return;
    if (boxType(hdlr.dataStart + 8) !== 'vide') return;

    let trackId = 1;
    const tkhd = findChild(ds, be, 'tkhd');
    if (tkhd && tkhd.dataStart + 24 <= tkhd.boxEnd) {
      const ver = bytes[tkhd.dataStart];
      trackId = ver === 1 && tkhd.dataStart + 32 <= tkhd.boxEnd
        ? dv.getUint32(tkhd.dataStart + 20)
        : dv.getUint32(tkhd.dataStart + 12);
    }

    let timescale = 90000;
    const mdhd = findChild(mdia.dataStart, mdia.boxEnd, 'mdhd');
    if (mdhd && mdhd.dataStart + 20 <= mdhd.boxEnd) {
      const ver = bytes[mdhd.dataStart];
      timescale = ver === 1 && mdhd.dataStart + 28 <= mdhd.boxEnd
        ? dv.getUint32(mdhd.dataStart + 20)
        : dv.getUint32(mdhd.dataStart + 12);
    }
    if (timescale <= 0) timescale = 90000;

    // Parse optional edts/elst edit list (ISO/IEC 14496-12 8.6.6) so B-frame priming offsets and
    // trimmed preroll/postroll presentation windows match FFmpeg / ISO BMFF timeline semantics.
    let editTimelineOffsetUs = 0;
    let editMediaStartTicks = 0;
    let hasEditWindow = false;
    let editMediaDurUs = Infinity;
    let editUnsupported = false;
    const edts = findChild(ds, be, 'edts');
    if (edts) {
      const elst = findChild(edts.dataStart, edts.boxEnd, 'elst');
      if (elst && elst.dataStart + 8 <= elst.boxEnd) {
        const eVer = bytes[elst.dataStart];
        const eCnt = dv.getUint32(elst.dataStart + 4);
        const stride = eVer === 1 ? 20 : 12;
        let sawMediaEdit = false;
        let ep = elst.dataStart + 8;
        for (let ei = 0; ei < eCnt && ep + stride <= elst.boxEnd; ei++) {
          const segDur = eVer === 1 ? u64(ep) : dv.getUint32(ep);
          const mediaTime = eVer === 1 ? i64(ep + 8) : dv.getInt32(ep + 4);
          const rateInt = dv.getInt16(ep + (eVer === 1 ? 16 : 8));
          const rateFrac = dv.getInt16(ep + (eVer === 1 ? 18 : 10));
          ep += stride;
          if (mediaTime === -1) {
            if (sawMediaEdit) {
              editUnsupported = true;
              break;
            }
            editTimelineOffsetUs += (segDur / movieTimescale) * 1_000_000;
          } else {
            if (sawMediaEdit || rateInt !== 1 || rateFrac !== 0 || mediaTime < 0) {
              editUnsupported = true;
              break;
            }
            sawMediaEdit = true;
            hasEditWindow = true;
            editMediaStartTicks = mediaTime;
            if (segDur > 0) {
              editMediaDurUs = (segDur / movieTimescale) * 1_000_000;
            }
          }
        }
      }
    }
    if (editUnsupported) return;

    const minf = findChild(mdia.dataStart, mdia.boxEnd, 'minf');
    if (!minf) return;
    const stbl = findChild(minf.dataStart, minf.boxEnd, 'stbl');
    if (!stbl) return;
    const stsd = findChild(stbl.dataStart, stbl.boxEnd, 'stsd');
    if (!stsd || stsd.dataStart + 8 > stsd.boxEnd) return;

    let codec = null;
    let codecTag = '';
    let description = null;
    let codedWidth = 0;
    let codedHeight = 0;

    iterBoxes(stsd.dataStart + 8, stsd.boxEnd, (eType, _ebs, eds, ebe) => {
      if (codec) return;
      if (eds + 78 > ebe) return;
      const w = dv.getUint16(eds + 24);
      const h = dv.getUint16(eds + 26);
      const childStart = eds + 78;
      if (eType === 'avc1' || eType === 'avc3') {
        const avcC = findChild(childStart, ebe, 'avcC');
        if (avcC && avcC.dataStart + 4 <= avcC.boxEnd) {
          const cfg = bytes.slice(avcC.dataStart, avcC.boxEnd);
          codec = `avc1.${hex2(cfg[1])}${hex2(cfg[2])}${hex2(cfg[3])}`;
          codecTag = 'H.264';
          description = cfg;
          codedWidth = w;
          codedHeight = h;
        }
      } else if (eType === 'hvc1' || eType === 'hev1') {
        const hvcC = findChild(childStart, ebe, 'hvcC');
        if (hvcC && hvcC.dataStart + 13 <= hvcC.boxEnd) {
          const cfg = bytes.slice(hvcC.dataStart, hvcC.boxEnd);
          codec = buildHevcCodecString(cfg, eType);
          codecTag = 'H.265';
          description = cfg;
          codedWidth = w;
          codedHeight = h;
        }
      } else if (eType === 'vp09') {
        const vpcC = findChild(childStart, ebe, 'vpcC');
        if (vpcC && vpcC.dataStart + 8 <= vpcC.boxEnd) {
          const cfg = bytes.slice(vpcC.dataStart, vpcC.boxEnd);
          codec = buildVp9CodecString(cfg);
          codecTag = 'VP9';
          codedWidth = w;
          codedHeight = h;
        }
      } else if (eType === 'av01') {
        const av1C = findChild(childStart, ebe, 'av1C');
        if (av1C && av1C.dataStart + 4 <= av1C.boxEnd) {
          const cfg = bytes.slice(av1C.dataStart, av1C.boxEnd);
          codec = buildAv1CodecString(cfg);
          codecTag = 'AV1';
          codedWidth = w;
          codedHeight = h;
        }
      }
    });

    if (!codec) return;

    // 1. Parse classic stbl samples
    const samples = [];
    const stsz = findChild(stbl.dataStart, stbl.boxEnd, 'stsz');
    const stsc = findChild(stbl.dataStart, stbl.boxEnd, 'stsc');
    const stco = findChild(stbl.dataStart, stbl.boxEnd, 'stco');
    const co64 = findChild(stbl.dataStart, stbl.boxEnd, 'co64');
    const stts = findChild(stbl.dataStart, stbl.boxEnd, 'stts');
    const stss = findChild(stbl.dataStart, stbl.boxEnd, 'stss');
    const ctts = findChild(stbl.dataStart, stbl.boxEnd, 'ctts');

    if (stsz && stsc && (stco || co64) && stts && stsz.dataStart + 12 <= stsz.boxEnd) {
      const defSize = dv.getUint32(stsz.dataStart + 4);
      const sampleCount = dv.getUint32(stsz.dataStart + 8);
      if (sampleCount > 0) {
        const sizes = new Uint32Array(sampleCount);
        if (defSize > 0) {
          sizes.fill(defSize);
        } else {
          for (let i = 0; i < sampleCount && stsz.dataStart + 12 + i * 4 + 4 <= stsz.boxEnd; i++) {
            sizes[i] = dv.getUint32(stsz.dataStart + 12 + i * 4);
          }
        }
        const chunkOffsets = [];
        if (stco && stco.dataStart + 8 <= stco.boxEnd) {
          const cnt = dv.getUint32(stco.dataStart + 4);
          for (let i = 0; i < cnt && stco.dataStart + 8 + i * 4 + 4 <= stco.boxEnd; i++) {
            chunkOffsets.push(dv.getUint32(stco.dataStart + 8 + i * 4));
          }
        } else if (co64 && co64.dataStart + 8 <= co64.boxEnd) {
          const cnt = dv.getUint32(co64.dataStart + 4);
          for (let i = 0; i < cnt && co64.dataStart + 8 + i * 8 + 8 <= co64.boxEnd; i++) {
            chunkOffsets.push(u64(co64.dataStart + 8 + i * 8));
          }
        }
        const stscEntries = [];
        if (stsc.dataStart + 8 <= stsc.boxEnd) {
          const cnt = dv.getUint32(stsc.dataStart + 4);
          for (let i = 0; i < cnt && stsc.dataStart + 8 + i * 12 + 12 <= stsc.boxEnd; i++) {
            stscEntries.push({
              firstChunk: dv.getUint32(stsc.dataStart + 8 + i * 12),
              samplesPerChunk: dv.getUint32(stsc.dataStart + 12 + i * 12),
            });
          }
        }
        const offsets = new Float64Array(sampleCount);
        let sIdx = 0;
        for (let c = 0; c < chunkOffsets.length && sIdx < sampleCount; c++) {
          const chunkNum = c + 1;
          let spc = 1;
          for (let e = 0; e < stscEntries.length; e++) {
            if (stscEntries[e].firstChunk <= chunkNum) spc = stscEntries[e].samplesPerChunk;
            else break;
          }
          let curOff = chunkOffsets[c];
          for (let k = 0; k < spc && sIdx < sampleCount; k++) {
            offsets[sIdx] = curOff;
            curOff += sizes[sIdx];
            sIdx++;
          }
        }
        const dts = new Float64Array(sampleCount);
        const dur = new Uint32Array(sampleCount);
        if (stts.dataStart + 8 <= stts.boxEnd) {
          const cnt = dv.getUint32(stts.dataStart + 4);
          let idx = 0;
          let acc = 0;
          for (let i = 0; i < cnt && stts.dataStart + 8 + i * 8 + 8 <= stts.boxEnd; i++) {
            const sc = dv.getUint32(stts.dataStart + 8 + i * 8);
            const sd = dv.getUint32(stts.dataStart + 12 + i * 8);
            for (let k = 0; k < sc && idx < sampleCount; k++) {
              dts[idx] = acc;
              dur[idx] = sd;
              acc += sd;
              idx++;
            }
          }
        }
        const ctsOff = new Int32Array(sampleCount);
        if (ctts && ctts.dataStart + 8 <= ctts.boxEnd) {
          const ver = bytes[ctts.dataStart];
          const cnt = dv.getUint32(ctts.dataStart + 4);
          let idx = 0;
          for (let i = 0; i < cnt && ctts.dataStart + 8 + i * 8 + 8 <= ctts.boxEnd; i++) {
            const sc = dv.getUint32(ctts.dataStart + 8 + i * 8);
            const so = ver === 1 ? dv.getInt32(ctts.dataStart + 12 + i * 8) : dv.getUint32(ctts.dataStart + 12 + i * 8);
            for (let k = 0; k < sc && idx < sampleCount; k++) {
              ctsOff[idx++] = so;
            }
          }
        }
        const keySet = new Uint8Array(sampleCount);
        if (stss && stss.dataStart + 8 <= stss.boxEnd) {
          const cnt = dv.getUint32(stss.dataStart + 4);
          for (let i = 0; i < cnt && stss.dataStart + 8 + i * 4 + 4 <= stss.boxEnd; i++) {
            const k1 = dv.getUint32(stss.dataStart + 8 + i * 4) - 1;
            if (k1 >= 0 && k1 < sampleCount) keySet[k1] = 1;
          }
        } else {
          keySet.fill(1);
        }
        for (let i = 0; i < sampleCount; i++) {
          const sz = sizes[i];
          const off = offsets[i];
          if (sz === 0 || off + sz > len) continue;
          const ptsTicks = dts[i] + ctsOff[i];
          samples.push({
            offset: off,
            size: sz,
            isKey: keySet[i] === 1 || i === 0,
            timestampUs: Math.round(editTimelineOffsetUs + ((ptsTicks - editMediaStartTicks) / timescale) * 1_000_000),
            durationUs: Math.max(1000, Math.round((dur[i] / timescale) * 1_000_000)),
          });
        }
      }
    }

    // 2. Append fragmented fMP4 (moof/traf/tfhd/tfdt/trun) samples honoring ISO/IEC 14496-12 8.8.7
    //    multi-traf implicit base_data_offset chaining across ALL tracks (including audio-first moofs).
    let runningDecodeTicks = 0;
    let fmp4Unsupported = false;
    iterBoxes(0, len, (topType, moofStart, moofDataStart, moofEnd) => {
      if (topType !== 'moof' || fmp4Unsupported) return;
      let trafIdx = 0;
      let prevTrafDataEnd = -1;
      iterBoxes(moofDataStart, moofEnd, (cType, _tbs, trafDataStart, trafEnd) => {
        if (cType !== 'traf' || fmp4Unsupported) return;
        const tfhd = findChild(trafDataStart, trafEnd, 'tfhd');
        if (!tfhd || tfhd.dataStart + 8 > tfhd.boxEnd) {
          prevTrafDataEnd = -1;
          trafIdx++;
          return;
        }
        const tfFlags = (bytes[tfhd.dataStart + 1] << 16) | (bytes[tfhd.dataStart + 2] << 8) | bytes[tfhd.dataStart + 3];
        const trafTrackId = dv.getUint32(tfhd.dataStart + 4);
        const isVideoTraf = trafTrackId === trackId;
        const trafTrex = trexByTrack.get(trafTrackId) || {
          defaultSampleDuration: isVideoTraf ? Math.round(timescale / 30) : 0,
          defaultSampleSize: 0,
          defaultSampleFlags: 0,
        };
        let p = tfhd.dataStart + 8;
        let baseDataOffset = -1;
        if (tfFlags & 0x000001) {
          if (p + 8 > tfhd.boxEnd) { fmp4Unsupported = true; return; }
          baseDataOffset = u64(p);
          p += 8;
        } else if ((tfFlags & 0x020000) !== 0 || trafIdx === 0) {
          baseDataOffset = moofStart;
        } else if (prevTrafDataEnd >= 0) {
          baseDataOffset = prevTrafDataEnd;
        } else {
          if (isVideoTraf) fmp4Unsupported = true;
          prevTrafDataEnd = -1;
          trafIdx++;
          return;
        }
        if (tfFlags & 0x000002) { p += 4; } // sample_description_index
        let defDur = trafTrex.defaultSampleDuration;
        if (tfFlags & 0x000008) { defDur = dv.getUint32(p); p += 4; }
        let defSize = trafTrex.defaultSampleSize;
        if (tfFlags & 0x000010) { defSize = dv.getUint32(p); p += 4; }
        let defFlags = trafTrex.defaultSampleFlags;
        if (tfFlags & 0x000020) { defFlags = dv.getUint32(p); p += 4; }

        if (isVideoTraf) {
          const tfdt = findChild(trafDataStart, trafEnd, 'tfdt');
          if (tfdt && tfdt.dataStart + 8 <= tfdt.boxEnd) {
            const ver = bytes[tfdt.dataStart];
            runningDecodeTicks = ver === 1 && tfdt.dataStart + 12 <= tfdt.boxEnd
              ? u64(tfdt.dataStart + 4)
              : dv.getUint32(tfdt.dataStart + 4);
          }
        }

        let runningOffset = baseDataOffset;
        let trafMaxEnd = baseDataOffset;
        let trafSizeKnown = true;
        iterBoxes(trafDataStart, trafEnd, (tType, _rbs, rds, rbe) => {
          if (tType !== 'trun' || rds + 8 > rbe || fmp4Unsupported) return;
          const trVer = bytes[rds];
          const trFlags = (bytes[rds + 1] << 16) | (bytes[rds + 2] << 8) | bytes[rds + 3];
          const sCount = dv.getUint32(rds + 4);
          let rp = rds + 8;
          let curOff = runningOffset;
          if (trFlags & 0x000001) {
            if (rp + 4 > rbe) { fmp4Unsupported = true; return; }
            const dataOffset = dv.getInt32(rp);
            rp += 4;
            curOff = baseDataOffset + dataOffset;
          }
          let firstSampleFlags = defFlags;
          const hasFirstFlags = (trFlags & 0x000004) !== 0;
          if (hasFirstFlags) {
            if (rp + 4 > rbe) { fmp4Unsupported = true; return; }
            firstSampleFlags = dv.getUint32(rp);
            rp += 4;
          }
          const hasDur = (trFlags & 0x000100) !== 0;
          const hasSize = (trFlags & 0x000200) !== 0;
          const hasFlags = (trFlags & 0x000400) !== 0;
          const hasCto = (trFlags & 0x000800) !== 0;
          for (let i = 0; i < sCount; i++) {
            let sDur = defDur;
            let sSize = defSize;
            let sFlags = (i === 0 && hasFirstFlags) ? firstSampleFlags : defFlags;
            let sCto = 0;
            if (hasDur && rp + 4 <= rbe) { sDur = dv.getUint32(rp); rp += 4; }
            if (hasSize && rp + 4 <= rbe) { sSize = dv.getUint32(rp); rp += 4; }
            if (hasFlags && rp + 4 <= rbe) { sFlags = dv.getUint32(rp); rp += 4; }
            if (hasCto && rp + 4 <= rbe) {
              sCto = trVer === 1 ? dv.getInt32(rp) : dv.getUint32(rp);
              rp += 4;
            }
            if (!hasSize && defSize === 0) {
              trafSizeKnown = false;
            }
            if (isVideoTraf) {
              if (sSize > 0 && (curOff < moofEnd || curOff + sSize > len)) {
                fmp4Unsupported = true;
                return;
              }
              const nonSync = (sFlags >>> 16) & 1;
              const depOn = (sFlags >>> 24) & 3;
              const isKey = samples.length === 0 || (nonSync === 0 && (depOn === 2 || (i === 0 && (hasFirstFlags || !hasFlags))));
              const ptsTicks = runningDecodeTicks + sCto;
              if (sSize > 0) {
                samples.push({
                  offset: curOff,
                  size: sSize,
                  isKey,
                  timestampUs: Math.round(editTimelineOffsetUs + ((ptsTicks - editMediaStartTicks) / timescale) * 1_000_000),
                  durationUs: Math.max(1000, Math.round(((sDur || 1) / timescale) * 1_000_000)),
                });
              }
              runningDecodeTicks += sDur;
            }
            curOff += sSize;
          }
          runningOffset = curOff;
          if (curOff > trafMaxEnd) trafMaxEnd = curOff;
        });
        prevTrafDataEnd = trafSizeKnown && trafMaxEnd >= 0 ? trafMaxEnd : -1;
        trafIdx++;
      });
    });

    if (fmp4Unsupported) return;

    if (samples.length > 0) {
      selected = {
        codec,
        codecTag,
        description,
        codedWidth,
        codedHeight,
        timescale,
        editStartUs: hasEditWindow ? Math.round(editTimelineOffsetUs) : -Infinity,
        editEndUs: hasEditWindow && Number.isFinite(editMediaDurUs)
          ? Math.round(editTimelineOffsetUs + editMediaDurUs)
          : Infinity,
        samples,
        bytes,
      };
    }
  });

  return selected;
}

async function runWorkerWebCodecsFastPass(msg) {
  if (workerFastPassRunning) return;
  if (!wasmExports) {
    self.postMessage({ type: 'fast_pass_done', processed: 0, label: 'WASM NOT READY' });
    return;
  }
  if (typeof VideoDecoder !== 'function' || typeof EncodedVideoChunk !== 'function') {
    self.postMessage({ type: 'fast_pass_done', processed: 0, label: 'WEBCODECS UNAVAILABLE' });
    return;
  }

  workerFastPassRunning = true;
  workerFastPassCancel = false;
  fastPassActive = true;
  seedBoxPreview = null;
  roiBoxPreview = null;

  const epoch = typeof msg.epoch === 'number' && Number.isInteger(msg.epoch) ? (msg.epoch >>> 0) : ++workerLocalEpoch;
  const fallbackFps = typeof msg.fps === 'number' && msg.fps > 1.0 ? msg.fps : 30.0;
  let processed = 0;
  let rejected = 0;
  let totalSamples = 0;
  let codecTag = 'WC';
  let decoder = null;
  let frameChain = Promise.resolve();
  let lastPtsUs = -1;
  let passError = null;
  const tStartMs = performance.now();

  try {
    let rawBuf = null;
    if (msg.file && typeof msg.file.arrayBuffer === 'function') {
      rawBuf = await msg.file.arrayBuffer();
    } else if (typeof msg.url === 'string' && msg.url.length > 0) {
      const resp = await fetch(msg.url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      rawBuf = await resp.arrayBuffer();
    } else {
      throw new Error('No MP4 source provided');
    }

    const track = parseMp4TrackMultiCodec(rawBuf);
    if (!track || track.samples.length === 0) {
      throw new Error('Unsupported container or video codec');
    }
    const editStartUs = typeof track.editStartUs === 'number' ? track.editStartUs : -Infinity;
    const editEndUs = typeof track.editEndUs === 'number' ? track.editEndUs : Infinity;
    const hasEditBounds = Number.isFinite(editStartUs) || Number.isFinite(editEndUs);
    const inEditWindow = (tsUs) => tsUs >= editStartUs && tsUs < editEndUs;
    const decodeSampleCount = track.samples.length;
    totalSamples = hasEditBounds
      ? track.samples.filter((s) => inEditWindow(s.timestampUs)).length
      : decodeSampleCount;
    codecTag = track.codecTag || track.codec;

    const decConfig = { codec: track.codec };
    if (track.description) decConfig.description = track.description;
    if (track.codedWidth > 0) decConfig.codedWidth = track.codedWidth;
    if (track.codedHeight > 0) decConfig.codedHeight = track.codedHeight;

    if (typeof VideoDecoder.isConfigSupported === 'function') {
      const sup = await VideoDecoder.isConfigSupported(decConfig);
      if (!sup || !sup.supported) {
        throw new Error(`Codec not supported: ${track.codec}`);
      }
    }

    resetPersistenceState();
    clearVisualQueues();
    wasmExports.chronos_reset_tracks();
    workerExpectedEngineFrame = 0;
    let passSeq = 0;

    decoder = new VideoDecoder({
      output: (vf) => {
        if (workerFastPassCancel || passError) {
          vf.close();
          return;
        }
        frameChain = frameChain.then(async () => {
          try {
            if (workerFastPassCancel || passError) return;
            const rawTs = typeof vf.timestamp === 'number' && Number.isFinite(vf.timestamp) ? vf.timestamp : null;
            if (hasEditBounds && rawTs !== null && !inEditWindow(rawTs)) {
              return;
            }
            const sw = Math.max(1, Math.trunc(vf.displayWidth || vf.codedWidth || frameW));
            const sh = Math.max(1, Math.trunc(vf.displayHeight || vf.codedHeight || frameH));
            const ptsUs = rawTs !== null
              ? Math.max(0, rawTs)
              : Math.round((passSeq / fallbackFps) * 1_000_000);
            const dtSec = lastPtsUs >= 0 && ptsUs > lastPtsUs
              ? Math.max(0.001, Math.min(0.25, (ptsUs - lastPtsUs) * 1e-6))
              : (1.0 / fallbackFps);
            lastPtsUs = ptsUs;
            passSeq++;

            await stageFoveaTileFromVideoFrame(vf, sw, sh);

            const geom = resolveIngestGeomSpec(sw, sh, null);
            let rawPixels = null;
            const canDirectCopy =
              webCodecsCopyToRgbaState >= 0 &&
              typeof vf.copyTo === 'function' &&
              geom.native1to1 === 1 &&
              geom.drawOrient === 0 &&
              geom.fullFrameFit &&
              geom.cropW === frameW &&
              geom.cropH === frameH;
            if (canDirectCopy) {
              const ptr = wasmExports.chronos_get_ingest_rgba_ptr();
              const slotIdx = ringSlotByteOffsets.indexOf(ptr);
              if (slotIdx >= 0 && ringSlotU8[slotIdx]) {
                try {
                  const srcReadX = typeof geom.codedCropX0 === 'number' ? geom.codedCropX0 : geom.cropX0;
                  const srcReadY = typeof geom.codedCropY0 === 'number' ? geom.codedCropY0 : geom.cropY0;
                  await vf.copyTo(ringSlotU8[slotIdx], {
                    rect: { x: srcReadX, y: srcReadY, width: frameW, height: frameH },
                    format: 'RGBA',
                  });
                  webCodecsCopyToRgbaState = 1;
                  lastIngestGeom = geom;
                  rawPixels = ringSlotU8[slotIdx];
                } catch (_copyErr) {
                  webCodecsCopyToRgbaState = -1;
                }
              }
            }
            if (!rawPixels) {
              drawCroppedLetterboxedFrame(vf, null);
              rawPixels = simCtx.getImageData(0, 0, frameW, frameH).data;
            }

            const attemptCount = processed + rejected;
            const fastPassPresent = (attemptCount % 12) === 0 || (attemptCount + 1) === totalSamples;
            const ingestRes = processIngestedFrame(rawPixels, dtSec, ptsUs * 1e-6, {
              epoch,
              seq: passSeq,
              ptsKnown: true,
              tSrcUs: ptsUs,
              fastPass: true,
              fastPassPresent,
              geom: lastIngestGeom,
            });
            if (ingestRes && ingestRes.committed) {
              processed++;
            } else {
              rejected++;
            }
            if (fastPassPresent) {
              const elapsedSec = Math.max(0.001, (performance.now() - tStartMs) * 1e-3);
              self.postMessage({
                type: 'fast_pass_progress',
                processed,
                rejected,
                total: totalSamples,
                fps: Math.round(processed / elapsedSec),
                codec: codecTag,
              });
            }
          } finally {
            vf.close();
          }
        }).catch((err) => {
          passError = err;
        });
      },
      error: (err) => {
        passError = err || new Error('VideoDecoder error');
      },
    });
    decoder.configure(decConfig);

    for (let i = 0; i < decodeSampleCount; i++) {
      if (workerFastPassCancel || passError) break;
      while (decoder.decodeQueueSize > 8 && !workerFastPassCancel && !passError) {
        await new Promise((r) => setTimeout(r, 1));
      }
      if (workerFastPassCancel || passError) break;
      const s = track.samples[i];
      const chunk = new EncodedVideoChunk({
        type: s.isKey ? 'key' : 'delta',
        timestamp: s.timestampUs,
        duration: s.durationUs,
        data: track.bytes.subarray(s.offset, s.offset + s.size),
      });
      decoder.decode(chunk);
    }

    if (!workerFastPassCancel && !passError && decoder.state === 'configured') {
      await decoder.flush();
    }
    await frameChain;

    const elapsedSec = Math.max(0.001, (performance.now() - tStartMs) * 1e-3);
    const fps = Math.round(processed / elapsedSec);
    const label = passError
      ? `FAST-PASS ERR: ${passError.message || 'decode failed'}`
      : (workerFastPassCancel
          ? `CANCELLED ${processed}/${totalSamples}f (${fps} fps)`
          : (rejected > 0
              ? `FAST-PASS ERR: ${rejected}/${totalSamples} rejected (${processed} committed)`
              : `INDEXED ${processed}f · ${fps} fps (${codecTag})`));
    self.postMessage({ type: 'fast_pass_done', processed, rejected, total: totalSamples, fps, codec: codecTag, label });
  } catch (err) {
    self.postMessage({
      type: 'fast_pass_done',
      processed,
      rejected,
      total: totalSamples,
      label: `FAST-PASS ERR: ${err && err.message ? err.message : 'failed'}`,
    });
  } finally {
    if (decoder && decoder.state !== 'closed') {
      try { decoder.close(); } catch (_e) {}
    }
    fastPassActive = false;
    workerFastPassRunning = false;
    workerFastPassCancel = false;
    emitTimelineEventsIfNeeded(true);
    if (hasAnyRenderedFrame()) {
      scheduleOffscreenHudPresent(currentDisplaySlot(), false, true);
      emitTelemetrySnapshot();
    }
  }
}
