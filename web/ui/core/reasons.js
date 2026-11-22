// ============================================================================
// CHRONOS-SPECTRAL v3.5 // Kernel State & Reason Bitmask Decoders
// Mirrors `src/{bos,depth,vibrometry,wind_tunnel,elastography,hodge}.rs`.
// Replaces raw hex reason masks (`0x...`) with structured human-readable chips.
// ============================================================================

export const BOS_STATE_LABELS = Object.freeze({
  0: 'OFF',
  1: 'WARMING',
  2: 'READY',
  3: 'DISTURBED',
});

export const BOS_REASON_BITS = Object.freeze([
  { bit: 1 << 0, code: 'ANCHOR_SUPPORT', label: 'Low anchor support', severity: 'fault' },
  { bit: 1 << 1, code: 'EXT_PAN_CUT', label: 'Camera pan or cut', severity: 'fault' },
  { bit: 1 << 2, code: 'INT_SHIFT', label: 'Tripod shift detected', severity: 'fault' },
  { bit: 1 << 3, code: 'PHOTOM_MSE', label: 'Photometric shift', severity: 'fault' },
  { bit: 1 << 4, code: 'DISCONT', label: 'Frame discontinuity', severity: 'warn' },
  { bit: 1 << 5, code: 'NO_GAUGE', label: 'Gauge ROI empty', severity: 'warn' },
  { bit: 1 << 6, code: 'NONCONV', label: 'Poisson solver unconverged', severity: 'fault' },
  { bit: 1 << 7, code: 'LOW_TEX', label: 'Low background texture', severity: 'warn' },
]);

export const DEPTH_STATE_LABELS = Object.freeze({
  0: 'OFF',
  1: 'WARMING',
  2: 'READY_PARALLAX',
  3: 'READY_TTC_ONLY',
  4: 'REFUSED_PURE_ROTATION',
  5: 'REFUSED_NO_BASELINE',
  6: 'READY_STEREO',
  7: 'READY_EXTERNAL',
});

export const DEPTH_REASON_BITS = Object.freeze([
  { bit: 1 << 0, code: 'PURE_ROTATION', label: 'Pure camera yaw (no parallax)', severity: 'fault' },
  { bit: 1 << 1, code: 'NO_BASELINE', label: 'Insufficient translation baseline', severity: 'warn' },
  { bit: 1 << 2, code: 'LOW_TRACKS', label: 'Too few tracks for depth', severity: 'warn' },
  { bit: 1 << 3, code: 'NO_INTRINSICS', label: 'Unset focal/scale intrinsics', severity: 'warn' },
  { bit: 1 << 4, code: 'EXT_STALE', label: 'External depth plane stale', severity: 'warn' },
  { bit: 1 << 5, code: 'ASSUMED_PRIOR', label: 'Assumed external prior', severity: 'warn' },
  { bit: 1 << 6, code: 'STEREO_NOT_READY', label: 'Waiting for stereo reference', severity: 'warn' },
  { bit: 1 << 7, code: 'MIXED_ROT_TRANS', label: 'Mixed rotation + translation', severity: 'warn' },
  { bit: 1 << 8, code: 'BAD_DT', label: 'Invalid frame interval', severity: 'fault' },
]);

export const RS_STATE_LABELS = Object.freeze({
  0: 'OFF',
  1: 'UNCALIBRATED',
  2: 'WARMING',
  3: 'READY',
  4: 'DISTURBED',
});

export const RS_REASON_BITS = Object.freeze([
  { bit: 1 << 0, code: 'BAD_CALIB', label: 'Invalid RS timing', severity: 'fault' },
  { bit: 1 << 1, code: 'NO_SCAN_DIR', label: 'Scan axis unset', severity: 'fault' },
  { bit: 1 << 2, code: 'NOT_NATIVE_1TO1', label: 'Requires 1:1 native ROI', severity: 'fault' },
  { bit: 1 << 3, code: 'FRAME_GAP', label: 'Frame cadence gap', severity: 'warn' },
  { bit: 1 << 4, code: 'CAMERA_MOTION', label: 'Camera moving', severity: 'fault' },
  { bit: 1 << 5, code: 'CLOCK_SKEW', label: 'Audio/video clock skew', severity: 'fault' },
  { bit: 1 << 6, code: 'EXPOSURE_NULL', label: 'Exposure integration null', severity: 'warn' },
  { bit: 1 << 7, code: 'LOW_GRAD', label: 'Low vertical gradient', severity: 'warn' },
  { bit: 1 << 8, code: 'AUDIO_UNDERRUN', label: 'Waiting for 48 kHz audio', severity: 'warn' },
  { bit: 1 << 9, code: 'AXIS_TRANSPOSED', label: 'Scan axis transposed', severity: 'fault' },
  { bit: 1 << 10, code: 'NO_PTS', label: 'Presentation timestamps missing', severity: 'fault' },
]);

export const WIND_STATE_LABELS = Object.freeze({
  0: 'OFF',
  1: 'WARMING',
  2: 'READY_FIELD_ONLY',
  3: 'READY_FULL_LOADS',
  4: 'REFUSED',
});

export const WIND_REASON_BITS = Object.freeze([
  { bit: 1 << 0, code: 'NO_CONTINUUM', label: 'No continuum velocity field', severity: 'warn' },
  { bit: 1 << 1, code: 'HODGE_UNCONV', label: 'Hodge solver unconverged', severity: 'fault' },
  { bit: 1 << 2, code: 'MUTEX_CONFLICT', label: 'Scratch arena held by higher owner', severity: 'fault' },
  { bit: 1 << 3, code: 'HIGH_DIV', label: 'High 3D / compressible divergence', severity: 'fault' },
  { bit: 1 << 4, code: 'HIGH_NS_RESIDUAL', label: 'Non-conservative NS residual', severity: 'fault' },
  { bit: 1 << 5, code: 'NO_OBSTACLE_VOID', label: 'No closed obstacle void (beta_1=0)', severity: 'warn' },
  { bit: 1 << 6, code: 'NO_FREESTREAM', label: 'No uniform freestream band', severity: 'warn' },
  { bit: 1 << 7, code: 'LOW_VORT_COND', label: 'Low vorticity conditioning (nu NA)', severity: 'warn' },
  { bit: 1 << 8, code: 'PCG_UNCONV', label: 'Pressure Poisson unconverged', severity: 'fault' },
  { bit: 1 << 9, code: 'LOW_SUPPORT', label: 'Insufficient stencil support', severity: 'warn' },
]);

export const ELASTO_STATE_LABELS = Object.freeze({
  0: 'OFF',
  1: 'UNCALIBRATED',
  2: 'WARMING',
  3: 'READY',
  4: 'REFUSED',
});

// Elastography packs RS low bits (0..10) alongside Elastography high bits (16..20).
export const ELASTO_REASON_BITS = Object.freeze([
  ...RS_REASON_BITS,
  { bit: 1 << 16, code: 'SPATIAL_ALIASING', label: 'Spatial wave aliasing', severity: 'fault' },
  { bit: 1 << 17, code: 'WAVELENGTH_TOO_LONG', label: 'Wavelength exceeds aperture', severity: 'warn' },
  { bit: 1 << 18, code: 'STANDING_WAVE_DOMINANT', label: 'Standing wave dominant', severity: 'warn' },
  { bit: 1 << 19, code: 'MUTEX_CONFLICT', label: 'Scratch arena held by higher owner', severity: 'fault' },
  { bit: 1 << 20, code: 'LOW_COH_COVERAGE', label: 'Low coherent cell coverage', severity: 'warn' },
]);

export const HODGE_STATE_LABELS = Object.freeze({
  0: 'OFF',
  1: 'STANDBY',
  2: 'READY',
  3: 'UNCONVERGED',
});

export function decodeReasonMask(mask, table) {
  const m = (mask >>> 0) || 0;
  if (m === 0) return [];
  const out = [];
  for (let i = 0; i < table.length; i++) {
    const entry = table[i];
    if ((m & entry.bit) !== 0) out.push(entry);
  }
  return out;
}

export function formatReasonSummary(mask, table, maxChips = 2) {
  const items = decodeReasonMask(mask, table);
  if (items.length === 0) return { text: '', overflow: 0, items };
  const shown = items.slice(0, maxChips).map((x) => x.label).join(' · ');
  const overflow = Math.max(0, items.length - maxChips);
  return {
    text: overflow > 0 ? `${shown} (+${overflow})` : shown,
    overflow,
    items,
  };
}
