// ============================================================================
// CHRONOS-SPECTRAL v3.5 // Pure Constraint Matrix, Capability Derivation &
// Atomic Resolution Planner (`deriveControlStates`, `planIntent`, `keepControl`)
//
// Zero DOM dependencies. Imported identically by the browser UI and
// `scripts/verify_ui_fsm.mjs` (exhaustive 194,400-combination verifier).
// ============================================================================

import { BOS_OWNERS, computeBosOwner, listBosClaimants } from './fsm.js';
import {
  BOS_REASON_BITS,
  DEPTH_REASON_BITS,
  ELASTO_REASON_BITS,
  RS_REASON_BITS,
  WIND_REASON_BITS,
  decodeReasonMask,
} from './reasons.js';

export const AVAILABILITY = Object.freeze({
  ACTIVE: 'ACTIVE',
  AVAILABLE: 'AVAILABLE',
  CONFLICT_PREEMPTABLE: 'CONFLICT_PREEMPTABLE',
  REQUIRES_PREREQ: 'REQUIRES_PREREQ',
  UNSUPPORTED_FOR_SOURCE: 'UNSUPPORTED_FOR_SOURCE',
  LOCKED_BY_RUNTIME: 'LOCKED_BY_RUNTIME',
});

// Unified 7-state visual lamp / chip contract (`data-ctl-state`).
export const CTL_STATES = Object.freeze({
  IDLE: 'idle',
  ACTIVE: 'active',
  PENDING: 'pending',
  CONSTRAINED: 'constrained',
  REFUSED: 'refused',
  NA: 'na',
  UNAVAILABLE: 'unavailable',
});

export const OVERLAY_SPECS = Object.freeze([
  { id: 0, label: '0 · Baseline + Micro-Tracks / Hulls', producer: null },
  { id: 1, label: '1 · Velocity |v| Field (64x36)', producer: 'swarm' },
  { id: 2, label: '2 · Vorticity Curl (dv/dx - du/dy)', producer: 'swarm' },
  { id: 3, label: '3 · FTLE Coherent Ridges (16f)', producer: 'swarm' },
  { id: 4, label: '4 · Modal Vibration Phase [-pi, pi]', producer: 'swarm' },
  { id: 5, label: '5 · BOS Refraction |eps| (256x128)', producer: 'bos' },
  { id: 6, label: '6 · BOS Neumann Potential Phi (256x128)', producer: 'bos' },
  { id: 7, label: '7 · BOS Curl Diagnostic (256x128)', producer: 'bos' },
  { id: 8, label: '8 · Sparse Hodge Exact |grad phi| (64x36)', producer: 'hodge' },
  { id: 9, label: '9 · Sparse Hodge Coexact |curl* psi| (64x36)', producer: 'hodge' },
  { id: 10, label: '10 · Sparse Hodge Harmonic |h| (64x36)', producer: 'hodge' },
  { id: 11, label: '11 · RS Amplitude / Welch Coherence', producer: 'rs' },
  { id: 12, label: '12 · Depth / Parallax / Stereo', producer: 'depth' },
  { id: 13, label: '13 · Wind Tunnel Cp + Lift/Drag (64x36)', producer: 'wind' },
  { id: 14, label: '14 · Surface Elastography c(x,y) + Power', producer: 'elasto_surface' },
  { id: 15, label: '15 · Lock-In Acoustic BOS |Phi| (256x128)', producer: 'elasto_abos' },
]);

export function dynamicOverlayLabel(overlayId, organs) {
  if (overlayId === 11) {
    return organs.rs === 2
      ? '11 · RS Welch Coherence gamma^2 (64x36)'
      : '11 · RS Sub-Frame Amplitude (64x36)';
  }
  if (overlayId === 13 && organs.wind === 3) {
    return '13 · Wind Dense Audit Field (64x36 · FIELD ONLY)';
  }
  const spec = OVERLAY_SPECS[overlayId];
  return spec ? spec.label : `Overlay ${overlayId}`;
}

export function bosOwnerDisplayName(owner) {
  switch (owner) {
    case BOS_OWNERS.DEPTH_STEREO: return 'Depth Stereo (P1)';
    case BOS_OWNERS.BOS_THERMAL: return 'Thermal BOS (P2)';
    case BOS_OWNERS.WIND_TUNNEL: return 'Wind Tunnel (P3)';
    case BOS_OWNERS.ELASTO_ACOUSTIC_BOS: return 'Acoustic BOS (P4)';
    case BOS_OWNERS.ELASTO_SURFACE: return 'Surface Elasto (P5)';
    default: return 'Free';
  }
}

function clearCompetingBosOwners(organs, targetOwner) {
  const next = { ...organs };
  if (targetOwner !== BOS_OWNERS.DEPTH_STEREO && (next.depth === 2 || next.depth === 4)) {
    next.depth = 0;
  }
  if (targetOwner !== BOS_OWNERS.BOS_THERMAL && next.bos !== 0) {
    next.bos = 0;
  }
  if (targetOwner !== BOS_OWNERS.WIND_TUNNEL && next.wind !== 0) {
    next.wind = 0;
  }
  if (targetOwner !== BOS_OWNERS.ELASTO_ACOUSTIC_BOS && next.elasto === 4) {
    next.elasto = 0;
  }
  if (targetOwner !== BOS_OWNERS.ELASTO_SURFACE && (next.elasto >= 1 && next.elasto <= 3)) {
    next.elasto = 0;
  }
  return next;
}

function sanitizeOrphanedField(organs, field) {
  if (field >= 1 && field <= 4 && organs.swarm === 0) return 0;
  if (field >= 5 && field <= 7 && organs.bos === 0) return 0;
  if (field >= 8 && field <= 10 && (organs.hodge === 0 || organs.swarm === 0)) return 0;
  if (field === 11 && organs.rs === 0) return 0;
  if (field === 12 && organs.depth === 0) return 0;
  if (field === 13 && organs.wind === 0) return 0;
  if (field === 14 && !(organs.elasto >= 1 && organs.elasto <= 3)) return 0;
  if (field === 15 && organs.elasto !== 4) return 0;
  return field;
}

/**
 * Deterministic 1-click atomic repair / handoff planner.
 * Guarantees the resulting candidate satisfies:
 *   1. `listBosClaimants(candidate.organs).length <= 1` (exclusive BOS scratch)
 *   2. Every active downstream organ has its required upstream organ enabled
 *   3. No orphaned overlay left pointing at a retired producer
 */
export function planIntent(state, intent) {
  const curOrg = state.organ_config;
  let nextOrg = {
    ...curOrg,
    bosGauge: curOrg.bosGauge.slice(),
    rsCal: { ...curOrg.rsCal },
    depthCal: { ...curOrg.depthCal },
    lockHz: curOrg.lockHz.slice(),
  };
  let nextOv = { ...state.overlays };
  let nextLab = state.lab_mode;
  const retires = [];
  const enables = [];

  const setBosOwner = (owner) => {
    const prevClaimants = listBosClaimants(nextOrg).filter((o) => o !== owner);
    if (prevClaimants.length > 0) {
      for (const p of prevClaimants) retires.push(bosOwnerDisplayName(p));
      nextOrg = clearCompetingBosOwners(nextOrg, owner);
    }
  };

  if (intent.kind === 'SET_SWARM') {
    nextOrg.swarm = intent.mode;
    if (intent.mode === 0) {
      if (nextOrg.hodge !== 0) { nextOrg.hodge = 0; retires.push('Sparse Hodge'); }
      if (nextOrg.wind === 1 || nextOrg.wind === 2) { nextOrg.wind = 0; retires.push('Sparse Wind'); }
      if (nextOrg.metrology !== 0) { nextOrg.metrology = 0; nextOv.covEllipses = false; retires.push('Metrology'); }
    }
  } else if (intent.kind === 'SET_HODGE') {
    nextOrg.hodge = intent.mode;
    if (intent.mode > 0) {
      if (nextOrg.swarm === 0) { nextOrg.swarm = 2; enables.push('Swarm=Forced'); }
      if (nextOrg.wind === 3) { nextOrg.wind = 0; retires.push('Dense Wind'); }
    } else if (nextOrg.wind === 1 || nextOrg.wind === 2) {
      nextOrg.wind = 0;
      retires.push('Sparse Wind');
    }
  } else if (intent.kind === 'SET_METROLOGY') {
    nextOrg.metrology = intent.mode;
    if (intent.mode > 0 && nextOrg.swarm === 0) {
      nextOrg.swarm = 2;
      enables.push('Swarm=Forced');
    }
    if (intent.mode === 0) nextOv.covEllipses = false;
  } else if (intent.kind === 'SET_BOS') {
    if (intent.mode > 0) setBosOwner(BOS_OWNERS.BOS_THERMAL);
    nextOrg.bos = intent.mode;
  } else if (intent.kind === 'SET_DEPTH') {
    if (intent.mode === 2 || intent.mode === 4) setBosOwner(BOS_OWNERS.DEPTH_STEREO);
    if (intent.mode === 1 && nextOrg.swarm === 0) { nextOrg.swarm = 2; enables.push('Swarm=Forced'); }
    nextOrg.depth = intent.mode;
  } else if (intent.kind === 'SET_WIND') {
    if (intent.mode > 0) setBosOwner(BOS_OWNERS.WIND_TUNNEL);
    nextOrg.wind = intent.mode;
    if (intent.mode === 1 || intent.mode === 2) {
      if (nextOrg.swarm === 0) { nextOrg.swarm = 2; enables.push('Swarm=Forced'); }
      if (nextOrg.hodge === 0) { nextOrg.hodge = 1; enables.push('Hodge=2f'); }
    } else if (intent.mode === 3) {
      // Dense Audit runs its own 64x36 optical-flow Hodge; retire conflicting sparse Hodge.
      if (nextOrg.hodge !== 0) { nextOrg.hodge = 0; retires.push('Sparse Hodge'); }
    }
  } else if (intent.kind === 'SET_RS') {
    nextOrg.rs = intent.mode;
    if (intent.mode === 0) {
      if (nextOrg.elasto === 1 || nextOrg.elasto === 2 || nextOrg.elasto === 4) {
        nextOrg.elasto = 0;
        retires.push('RS Elastography');
      }
    } else if (intent.mode === 1 && (nextOrg.elasto === 1 || nextOrg.elasto === 4)) {
      nextOrg.elasto = 2;
      retires.push('Audio-Locked Elasto');
    }
  } else if (intent.kind === 'SET_ELASTO') {
    if (intent.mode === 4) {
      setBosOwner(BOS_OWNERS.ELASTO_ACOUSTIC_BOS);
      if (nextOrg.rs !== 2) { nextOrg.rs = 2; enables.push('RS+Welch Audio'); }
    } else if (intent.mode >= 1 && intent.mode <= 3) {
      setBosOwner(BOS_OWNERS.ELASTO_SURFACE);
      if (intent.mode === 1 && nextOrg.rs !== 2) { nextOrg.rs = 2; enables.push('RS+Welch Audio'); }
      if (intent.mode === 2 && nextOrg.rs === 0) { nextOrg.rs = 1; enables.push('RS Calib'); }
    }
    nextOrg.elasto = intent.mode;
  } else if (intent.kind === 'SET_STROBE_READY') {
    if (nextOrg.lookahead < 7) { nextOrg.lookahead = 7; enables.push('K=7'); }
    if (nextLab !== 1) { nextLab = 1; enables.push('Lab B'); }
    nextOv.strobe = true;
  } else if (intent.kind === 'SET_LOOKAHEAD') {
    nextOrg.lookahead = intent.k;
    if (intent.k < 7) nextOv.strobe = false;
  } else if (intent.kind === 'SET_FIELD') {
    const f = intent.field;
    nextOv.field = f;
    if (f >= 1 && f <= 4 && nextOrg.swarm === 0) { nextOrg.swarm = 2; enables.push('Swarm=Forced'); }
    if (f >= 5 && f <= 7) {
      setBosOwner(BOS_OWNERS.BOS_THERMAL);
      if (nextOrg.bos === 0) { nextOrg.bos = 1; enables.push('BOS=Static'); }
    }
    if (f >= 8 && f <= 10) {
      if (nextOrg.swarm === 0) { nextOrg.swarm = 2; enables.push('Swarm=Forced'); }
      if (nextOrg.hodge === 0) { nextOrg.hodge = 1; enables.push('Hodge=2f'); }
      if (nextOrg.wind === 3) { nextOrg.wind = 0; retires.push('Dense Wind'); }
    }
    if (f === 11 && nextOrg.rs === 0) { nextOrg.rs = 2; enables.push('RS+Welch'); }
    if (f === 12) {
      if (nextOrg.depth === 2 || nextOrg.depth === 4) setBosOwner(BOS_OWNERS.DEPTH_STEREO);
      if (nextOrg.depth === 0) {
        if (nextOrg.swarm === 0) { nextOrg.swarm = 2; enables.push('Swarm=Forced'); }
        nextOrg.depth = 1;
        enables.push('Depth=Mono');
      }
    }
    if (f === 13) {
      setBosOwner(BOS_OWNERS.WIND_TUNNEL);
      if (nextOrg.wind === 0) {
        if (nextOrg.swarm === 0) { nextOrg.swarm = 2; enables.push('Swarm=Forced'); }
        if (nextOrg.hodge === 0) { nextOrg.hodge = 1; enables.push('Hodge=2f'); }
        nextOrg.wind = 1;
        enables.push('Wind=2f');
      } else if (nextOrg.wind === 1 || nextOrg.wind === 2) {
        if (nextOrg.swarm === 0) { nextOrg.swarm = 2; enables.push('Swarm=Forced'); }
        if (nextOrg.hodge === 0) { nextOrg.hodge = 1; enables.push('Hodge=2f'); }
      }
    }
    if (f === 14) {
      setBosOwner(BOS_OWNERS.ELASTO_SURFACE);
      if (!(nextOrg.elasto >= 1 && nextOrg.elasto <= 3)) {
        nextOrg.elasto = 3;
        enables.push('Elasto=Surface');
      }
    }
    if (f === 15) {
      setBosOwner(BOS_OWNERS.ELASTO_ACOUSTIC_BOS);
      if (nextOrg.rs !== 2) { nextOrg.rs = 2; enables.push('RS+Welch Audio'); }
      if (nextOrg.elasto !== 4) {
        nextOrg.elasto = 4;
        enables.push('Acoustic BOS');
      }
    }
  } else if (intent.kind === 'TAKE_BOS_OWNER') {
    setBosOwner(intent.owner);
    if (intent.owner === BOS_OWNERS.DEPTH_STEREO && nextOrg.depth !== 2 && nextOrg.depth !== 4) nextOrg.depth = 2;
    if (intent.owner === BOS_OWNERS.BOS_THERMAL && nextOrg.bos === 0) nextOrg.bos = 1;
    if (intent.owner === BOS_OWNERS.WIND_TUNNEL) {
      if (nextOrg.wind === 0) {
        nextOrg.wind = nextOrg.swarm > 0 ? 1 : 3;
      }
      if (nextOrg.wind === 1 || nextOrg.wind === 2) {
        if (nextOrg.swarm === 0) { nextOrg.swarm = 2; enables.push('Swarm=Forced'); }
        if (nextOrg.hodge === 0) { nextOrg.hodge = 1; enables.push('Hodge=2f'); }
      }
    }
    if (intent.owner === BOS_OWNERS.ELASTO_SURFACE && !(nextOrg.elasto >= 1 && nextOrg.elasto <= 3)) nextOrg.elasto = 3;
    if (intent.owner === BOS_OWNERS.ELASTO_ACOUSTIC_BOS && nextOrg.elasto !== 4) {
      nextOrg.elasto = 4;
      if (nextOrg.rs !== 2) nextOrg.rs = 2;
    }
  }

  // Self-healing exclusive BOS scratch invariant: even if a raw probe created a
  // multi-claimant starting state, any planned transaction converges to <= 1 owner.
  const remainingClaims = listBosClaimants(nextOrg);
  if (remainingClaims.length > 1) {
    const winner = computeBosOwner(nextOrg);
    for (const loser of remainingClaims) {
      if (loser !== winner) retires.push(bosOwnerDisplayName(loser));
    }
    nextOrg = clearCompetingBosOwners(nextOrg, winner);
  }

  // Final orphan cleanup: never leave an active field pointing at a retired organ.
  if (intent.kind !== 'SET_FIELD') {
    nextOv.field = sanitizeOrphanedField(nextOrg, nextOv.field);
  }

  const labelParts = [];
  if (retires.length > 0) labelParts.push(`retires ${retires.join(', ')}`);
  if (enables.length > 0) labelParts.push(`enables ${enables.join(', ')}`);
  return {
    organs: nextOrg,
    overlays: nextOv,
    lab_mode: nextLab,
    retires,
    enables,
    summary: labelParts.length > 0 ? labelParts.join(' · ') : 'Apply',
    legal: listBosClaimants(nextOrg).length <= 1,
  };
}

/**
 * Pure capability derivation for every instrument organ, overlay, tool, and control.
 * Returns a `Map<string, ControlCapability>`.
 */
export function deriveControlStates(state) {
  const org = state.organ_config;
  const ov = state.overlays;
  const rt = state.runtime_health;
  const src = state.source.kind;
  const host = rt.host;
  const bosOwner = computeBosOwner(org);
  const bosClaimants = listBosClaimants(org);
  const camMoving = rt.camMotionState !== 0 || rt.roiSlewActive;
  const isMedia = src === 'FILE' || src === 'SAMPLE';
  const isLive = src === 'LIVE_TAB' || src === 'LIVE_URL' || src === 'LIVE_MSE';
  const isBench = src === 'TESTBENCH';
  const hasSource = src !== 'IDLE';
  const out = new Map();

  const put = (id, entry) => out.set(id, entry);

  // ---- 1. Swarm (continuum substrate) ----
  put('swarmModeSelect', {
    id: 'swarmModeSelect',
    cluster: 'engines',
    selected: org.swarm > 0,
    availability: org.swarm > 0 ? AVAILABILITY.ACTIVE : AVAILABILITY.AVAILABLE,
    ctlState: org.swarm === 0
      ? CTL_STATES.IDLE
      : (rt.swarmActive ? CTL_STATES.ACTIVE : CTL_STATES.PENDING),
    reasons: org.swarm === 1 && !rt.swarmActive
      ? [{ code: 'SWARM_STANDBY', label: 'Auto-density standby' }]
      : [],
    conflictWith: [],
    fixAction: org.swarm === 1 && !rt.swarmActive
      ? { label: 'Force On', intent: { kind: 'SET_SWARM', mode: 2 } }
      : null,
  });

  // ---- 2. Sparse Hodge ----
  {
    const active = org.hodge > 0;
    const needsSwarm = org.swarm === 0;
    const denseConflict = org.wind === 3;
    const p = rt.products.hodge;
    let ctlState = CTL_STATES.IDLE;
    let availability = AVAILABILITY.AVAILABLE;
    const reasons = [];
    let fixAction = null;
    if (active) {
      availability = AVAILABILITY.ACTIVE;
      if (needsSwarm) {
        ctlState = CTL_STATES.CONSTRAINED;
        reasons.push({ code: 'NEEDS_SWARM', label: 'Needs Swarm 1/2' });
        fixAction = { label: '+Swarm', intent: { kind: 'SET_HODGE', mode: org.hodge } };
      } else if (denseConflict) {
        ctlState = CTL_STATES.CONSTRAINED;
        reasons.push({ code: 'DENSE_WIND_CONFLICT', label: 'Dense Wind owns Hodge' });
        fixAction = { label: 'Switch', intent: { kind: 'SET_HODGE', mode: org.hodge } };
      } else if (p.state === 3) {
        ctlState = CTL_STATES.REFUSED;
        reasons.push({ code: 'HODGE_UNCONV', label: 'Solver unconverged' });
      } else if (p.state === 2) {
        ctlState = CTL_STATES.ACTIVE;
      } else {
        ctlState = CTL_STATES.PENDING;
      }
    } else if (needsSwarm) {
      availability = AVAILABILITY.REQUIRES_PREREQ;
      reasons.push({ code: 'NEEDS_SWARM', label: 'Needs Swarm' });
      fixAction = { label: '+Swarm', intent: { kind: 'SET_HODGE', mode: 1 } };
    }
    put('hodgeModeSelect', {
      id: 'hodgeModeSelect',
      cluster: 'engines',
      selected: active,
      availability,
      ctlState,
      reasons,
      conflictWith: denseConflict ? ['windModeSelect'] : [],
      fixAction,
    });
  }

  // ---- 3. Metrology 2x2 Cov ----
  {
    const active = org.metrology > 0;
    const needsSwarm = org.swarm === 0;
    put('metrologyModeSelect', {
      id: 'metrologyModeSelect',
      cluster: 'engines',
      selected: active,
      availability: active
        ? AVAILABILITY.ACTIVE
        : (needsSwarm ? AVAILABILITY.REQUIRES_PREREQ : AVAILABILITY.AVAILABLE),
      ctlState: !active
        ? CTL_STATES.IDLE
        : (needsSwarm ? CTL_STATES.CONSTRAINED : (rt.swarmActive ? CTL_STATES.ACTIVE : CTL_STATES.PENDING)),
      reasons: needsSwarm ? [{ code: 'NEEDS_SWARM', label: 'Needs Swarm' }] : [],
      conflictWith: [],
      fixAction: needsSwarm ? { label: '+Swarm', intent: { kind: 'SET_METROLOGY', mode: active ? org.metrology : 2 } } : null,
    });
    put('chkCovEllipses', {
      id: 'chkCovEllipses',
      cluster: 'display',
      selected: ov.covEllipses,
      availability: ov.covEllipses
        ? AVAILABILITY.ACTIVE
        : (org.metrology === 0 ? AVAILABILITY.REQUIRES_PREREQ : AVAILABILITY.AVAILABLE),
      ctlState: !ov.covEllipses
        ? CTL_STATES.IDLE
        : (org.metrology === 0 || needsSwarm ? CTL_STATES.CONSTRAINED : CTL_STATES.ACTIVE),
      reasons: org.metrology === 0 ? [{ code: 'NEEDS_METROLOGY', label: 'Needs Metrology' }] : [],
      conflictWith: [],
      fixAction: org.metrology === 0 ? { label: 'Enable', intent: { kind: 'SET_METROLOGY', mode: 2 } } : null,
    });
  }

  // ---- 4. BOS Thermal (P2 scratch owner) ----
  {
    const active = org.bos > 0;
    const blockedBy = bosOwner !== BOS_OWNERS.NONE && bosOwner !== BOS_OWNERS.BOS_THERMAL ? bosOwner : null;
    const p = rt.products.bos;
    const kReasons = decodeReasonMask(p.reasonMask, BOS_REASON_BITS);
    const reasons = [];
    let ctlState = CTL_STATES.IDLE;
    let availability = active ? AVAILABILITY.ACTIVE : (blockedBy ? AVAILABILITY.CONFLICT_PREEMPTABLE : AVAILABILITY.AVAILABLE);
    let fixAction = null;

    if (blockedBy) {
      reasons.push({ code: 'MUTEX', label: `Held: ${bosOwnerDisplayName(blockedBy)}` });
      fixAction = { label: 'Take P2', intent: { kind: 'SET_BOS', mode: active ? org.bos : 1 } };
    }
    if (active) {
      if (blockedBy) {
        ctlState = CTL_STATES.REFUSED;
      } else if (p.state === 3 || camMoving) {
        ctlState = CTL_STATES.REFUSED;
        if (kReasons.length > 0) reasons.push(...kReasons);
        else if (camMoving) reasons.push({ code: 'TRIPOD', label: 'Waiting for static camera' });
      } else if (p.state === 2) {
        ctlState = CTL_STATES.ACTIVE;
      } else {
        ctlState = CTL_STATES.PENDING;
      }
    }
    put('bosModeSelect', {
      id: 'bosModeSelect',
      cluster: 'engines',
      selected: active,
      availability,
      ctlState,
      reasons,
      conflictWith: blockedBy ? [blockedBy] : [],
      fixAction,
    });
  }

  // ---- 5. Pocket Wind Tunnel (P3 scratch owner; mode 1/2 sparse, mode 3 dense standalone) ----
  {
    const active = org.wind > 0;
    const isDense = org.wind === 3;
    const blockedBy = bosOwner !== BOS_OWNERS.NONE && bosOwner !== BOS_OWNERS.WIND_TUNNEL ? bosOwner : null;
    const needsSwarm = (org.wind === 1 || org.wind === 2) && org.swarm === 0;
    const needsHodge = (org.wind === 1 || org.wind === 2) && org.hodge === 0;
    const p = rt.products.wind;
    const kReasons = decodeReasonMask(p.reasonMask, WIND_REASON_BITS);
    const reasons = [];
    let ctlState = CTL_STATES.IDLE;
    let availability = active ? AVAILABILITY.ACTIVE : (blockedBy ? AVAILABILITY.CONFLICT_PREEMPTABLE : AVAILABILITY.AVAILABLE);
    let fixAction = null;

    if (blockedBy) {
      reasons.push({ code: 'MUTEX', label: `Held: ${bosOwnerDisplayName(blockedBy)}` });
      fixAction = { label: 'Take P3', intent: { kind: 'SET_WIND', mode: active ? org.wind : (org.swarm > 0 ? 1 : 3) } };
    }
    if (needsSwarm || needsHodge) {
      reasons.push({ code: 'NEEDS_CONTINUUM', label: `Needs ${[needsSwarm && 'Swarm', needsHodge && 'Hodge'].filter(Boolean).join('+')}` });
      fixAction = { label: '+Prereq', intent: { kind: 'SET_WIND', mode: org.wind } };
    }
    if (isDense) {
      reasons.push({ code: 'FIELD_ONLY', label: 'FIELD ONLY (no Cp/CL/CD)' });
    }
    if (active) {
      if (blockedBy || p.state === 4) {
        ctlState = CTL_STATES.REFUSED;
        if (kReasons.length > 0) reasons.push(...kReasons);
      } else if (needsSwarm || needsHodge) {
        ctlState = CTL_STATES.CONSTRAINED;
      } else if (camMoving) {
        ctlState = CTL_STATES.REFUSED;
        reasons.push({ code: 'TRIPOD', label: 'Waiting for static camera' });
      } else if (p.state === 2 || p.state === 3) {
        ctlState = isDense ? CTL_STATES.PENDING : CTL_STATES.ACTIVE;
      } else {
        ctlState = CTL_STATES.PENDING;
      }
    }
    put('windModeSelect', {
      id: 'windModeSelect',
      cluster: 'engines',
      selected: active,
      availability,
      ctlState,
      reasons,
      conflictWith: blockedBy ? [blockedBy] : [],
      fixAction,
    });
  }

  // ---- 6. Depth Sensing (mode 1 mono, mode 2/4 reserve P1 scratch; 2..4 SDK input) ----
  {
    const active = org.depth > 0;
    const isSdkOnly = org.depth >= 2;
    const p = rt.products.depth;
    const kReasons = decodeReasonMask(p.reasonMask, DEPTH_REASON_BITS);
    const reasons = [];
    let ctlState = CTL_STATES.IDLE;
    if (org.depth === 4 && !rt.stereoRefFresh) {
      reasons.push({ code: 'RESERVED_AUTO_FUSED', label: 'Reserved scratch · waiting for stereo ref' });
    }
    if (active) {
      if (p.state === 4 || p.state === 5) {
        ctlState = CTL_STATES.REFUSED;
        reasons.push(...kReasons);
      } else if (p.state === 2 || p.state === 3 || p.state === 6 || p.state === 7) {
        ctlState = CTL_STATES.ACTIVE;
      } else {
        ctlState = CTL_STATES.PENDING;
      }
    }
    put('depthModeSelect', {
      id: 'depthModeSelect',
      cluster: 'engines',
      selected: active,
      availability: active ? AVAILABILITY.ACTIVE : AVAILABILITY.AVAILABLE,
      ctlState,
      reasons,
      conflictWith: [],
      fixAction: isSdkOnly ? null : (org.depth === 1 && org.swarm === 0 ? { label: '+Swarm', intent: { kind: 'SET_DEPTH', mode: 1 } } : null),
    });
  }

  // ---- 7. Rolling-Shutter Vibrometry ----
  {
    const active = org.rs > 0;
    const needsAudio = org.rs === 2 && !rt.audioPresent;
    const needsNative = active && !state.resolution_roi.native1to1;
    const p = rt.products.rs;
    const kReasons = decodeReasonMask(p.reasonMask, RS_REASON_BITS);
    const reasons = [];
    let ctlState = CTL_STATES.IDLE;
    if (active) {
      if (p.state === 4) {
        ctlState = CTL_STATES.REFUSED;
        reasons.push(...kReasons);
      } else if (needsAudio || needsNative || p.state === 1) {
        ctlState = CTL_STATES.CONSTRAINED;
        if (needsAudio) reasons.push({ code: 'NEEDS_AUDIO', label: 'Needs 48 kHz audio' });
        if (needsNative) reasons.push({ code: 'NEEDS_NATIVE', label: 'Needs 1:1 native ROI' });
        if (kReasons.length > 0) reasons.push(...kReasons);
      } else if (p.state === 3) {
        ctlState = CTL_STATES.ACTIVE;
      } else {
        ctlState = CTL_STATES.PENDING;
      }
    }
    put('rsVibroModeSelect', {
      id: 'rsVibroModeSelect',
      cluster: 'engines',
      selected: active,
      availability: active ? AVAILABILITY.ACTIVE : AVAILABILITY.AVAILABLE,
      ctlState,
      reasons,
      conflictWith: [],
      fixAction: needsAudio ? { label: 'Probe', intent: { kind: 'START_PROBE' } } : null,
    });
  }

  // ---- 8. kHz Optical Elastography & Lock-In Acoustic BOS (P4 / P5 scratch) ----
  {
    const active = org.elasto > 0;
    const myOwner = org.elasto === 4 ? BOS_OWNERS.ELASTO_ACOUSTIC_BOS : BOS_OWNERS.ELASTO_SURFACE;
    const blockedBy = bosOwner !== BOS_OWNERS.NONE && bosOwner !== myOwner ? bosOwner : null;
    const needsAudio = (org.elasto === 1 || org.elasto === 4) && !rt.audioPresent;
    const needsRs2 = (org.elasto === 1 || org.elasto === 4) && org.rs !== 2;
    const needsRs1 = org.elasto === 2 && org.rs === 0;
    const p = rt.products.elasto;
    const kReasons = decodeReasonMask(p.reasonMask, ELASTO_REASON_BITS);
    const reasons = [];
    let ctlState = CTL_STATES.IDLE;
    let availability = active ? AVAILABILITY.ACTIVE : (blockedBy ? AVAILABILITY.CONFLICT_PREEMPTABLE : AVAILABILITY.AVAILABLE);
    let fixAction = null;

    if (blockedBy) {
      reasons.push({ code: 'MUTEX', label: `Held: ${bosOwnerDisplayName(blockedBy)}` });
      fixAction = { label: org.elasto === 4 ? 'Take P4' : 'Take P5', intent: { kind: 'SET_ELASTO', mode: active ? org.elasto : 3 } };
    }
    if (needsRs2 || needsRs1) {
      reasons.push({ code: 'NEEDS_RS', label: needsRs2 ? 'Needs RS+Welch' : 'Needs RS Calib' });
      fixAction = { label: '+RS', intent: { kind: 'SET_ELASTO', mode: org.elasto } };
    }
    if (needsAudio) {
      reasons.push({ code: 'NEEDS_AUDIO', label: 'Needs 48 kHz audio' });
      if (!fixAction) fixAction = { label: 'Probe', intent: { kind: 'START_PROBE' } };
    }
    if (active) {
      if (blockedBy || p.state === 4 || camMoving) {
        ctlState = CTL_STATES.REFUSED;
        if (kReasons.length > 0) reasons.push(...kReasons);
        else if (camMoving) reasons.push({ code: 'TRIPOD', label: 'Waiting for static camera' });
      } else if (needsAudio || needsRs2 || needsRs1 || p.state === 1) {
        ctlState = CTL_STATES.CONSTRAINED;
      } else if (p.state === 3) {
        ctlState = CTL_STATES.ACTIVE;
      } else {
        ctlState = CTL_STATES.PENDING;
      }
    }
    put('elastoModeSelect', {
      id: 'elastoModeSelect',
      cluster: 'engines',
      selected: active,
      availability,
      ctlState,
      reasons,
      conflictWith: blockedBy ? [blockedBy] : [],
      fixAction,
    });
  }

  // ---- 9. Lookahead K >= 7 -> Hilbert / Phase Strobe / Hermite ----
  {
    const kOk = org.lookahead >= 7;
    const labBOk = state.lab_mode === 1;
    const strobeConstrained = !kOk || !labBOk;
    const reasons = [];
    if (!kOk) reasons.push({ code: 'NEEDS_K7', label: `K=${org.lookahead}, needs K>=7` });
    if (!labBOk) reasons.push({ code: 'NEEDS_LAB_B', label: 'Needs Lab B (Phase)' });
    put('chkStrobe', {
      id: 'chkStrobe',
      cluster: 'phase',
      selected: ov.strobe,
      availability: ov.strobe
        ? AVAILABILITY.ACTIVE
        : (strobeConstrained ? AVAILABILITY.REQUIRES_PREREQ : AVAILABILITY.AVAILABLE),
      ctlState: ov.strobe
        ? (strobeConstrained ? CTL_STATES.CONSTRAINED : CTL_STATES.ACTIVE)
        : (strobeConstrained ? CTL_STATES.CONSTRAINED : CTL_STATES.IDLE),
      reasons,
      conflictWith: [],
      fixAction: strobeConstrained ? { label: 'K=7+B', intent: { kind: 'SET_STROBE_READY' } } : null,
    });
  }

  // ---- 10. Source-specific controls ----
  put('btnLabStimulus', {
    id: 'btnLabStimulus',
    cluster: 'engines',
    selected: false,
    availability: (isBench || state.lab_mode === 1) ? AVAILABILITY.AVAILABLE : AVAILABILITY.UNSUPPORTED_FOR_SOURCE,
    ctlState: (isBench || state.lab_mode === 1) ? CTL_STATES.IDLE : CTL_STATES.NA,
    reasons: (isBench || state.lab_mode === 1) ? [] : [{ code: 'BENCH_OR_LABB', label: 'Signal Generator or Lab B only' }],
    conflictWith: [],
    fixAction: null,
  });

  put('btnFastPass', {
    id: 'btnFastPass',
    cluster: 'transport',
    selected: state.transport.fastPassRunning,
    availability: !isMedia
      ? AVAILABILITY.UNSUPPORTED_FOR_SOURCE
      : (!host.webCodecs ? AVAILABILITY.LOCKED_BY_RUNTIME : AVAILABILITY.AVAILABLE),
    ctlState: state.transport.fastPassRunning
      ? CTL_STATES.PENDING
      : (!isMedia ? CTL_STATES.NA : (!host.webCodecs ? CTL_STATES.UNAVAILABLE : CTL_STATES.IDLE)),
    reasons: !isMedia
      ? [{ code: 'FILE_ONLY', label: 'MP4/Sample files only' }]
      : (!host.webCodecs ? [{ code: 'NO_WEBCODECS', label: 'WebCodecs unavailable' }] : []),
    conflictWith: [],
    fixAction: null,
  });

  put('btnSrcLiveStream', {
    id: 'btnSrcLiveStream',
    cluster: 'acquire',
    selected: isLive,
    availability: isLive
      ? AVAILABILITY.ACTIVE
      : (host.staticHost && !host.displayCaptureCapable ? AVAILABILITY.LOCKED_BY_RUNTIME : AVAILABILITY.AVAILABLE),
    ctlState: isLive
      ? CTL_STATES.ACTIVE
      : (host.staticHost ? CTL_STATES.PENDING : CTL_STATES.IDLE),
    reasons: host.staticHost
      ? [{ code: 'LOCAL_RELAY_T3', label: 'Tab/URL live ready · Tier-3 MSE needs local dev_server' }]
      : [],
    conflictWith: [],
    fixAction: null,
  });

  put('chkNativeRoi', {
    id: 'chkNativeRoi',
    cluster: 'acquire',
    selected: state.resolution_roi.native1to1,
    availability: !hasSource
      ? AVAILABILITY.UNSUPPORTED_FOR_SOURCE
      : (!rt.nativeEligible && rt.srcW > 0 ? AVAILABILITY.LOCKED_BY_RUNTIME : AVAILABILITY.AVAILABLE),
    ctlState: state.resolution_roi.native1to1
      ? (rt.nativeEligible || rt.srcW === 0 ? CTL_STATES.ACTIVE : CTL_STATES.CONSTRAINED)
      : (!hasSource ? CTL_STATES.NA : (!rt.nativeEligible && rt.srcW > 0 ? CTL_STATES.CONSTRAINED : CTL_STATES.IDLE)),
    reasons: hasSource && rt.srcW > 0 && !rt.nativeEligible
      ? [{ code: 'SRC_SMALLER', label: `Source ${rt.srcW}x${rt.srcH} < engine dims` }]
      : [],
    conflictWith: [],
    fixAction: hasSource && rt.srcW > 0 && !rt.nativeEligible && state.resolution_roi.resMode === 1
      ? { label: 'Use SD', intent: { kind: 'SET_RES', resMode: 0 } }
      : null,
  });

  // ---- 11. Field Overlay Selector (0..15) ----
  {
    const f = ov.field;
    const spec = OVERLAY_SPECS[f] || OVERLAY_SPECS[0];
    let missingProducer = null;
    if (spec.producer === 'swarm' && org.swarm === 0) missingProducer = 'Swarm';
    else if (spec.producer === 'bos' && org.bos === 0) missingProducer = 'Thermal BOS';
    else if (spec.producer === 'hodge' && (org.hodge === 0 || org.swarm === 0)) missingProducer = 'Swarm + Hodge';
    else if (spec.producer === 'rs' && org.rs === 0) missingProducer = 'RS Vibrometry';
    else if (spec.producer === 'depth' && org.depth === 0) missingProducer = 'Depth';
    else if (spec.producer === 'wind' && org.wind === 0) missingProducer = 'Wind Tunnel';
    else if (spec.producer === 'elasto_surface' && !(org.elasto >= 1 && org.elasto <= 3)) missingProducer = 'Surface Elasto';
    else if (spec.producer === 'elasto_abos' && org.elasto !== 4) missingProducer = 'Acoustic BOS';

    put('fieldOverlayModeSelect', {
      id: 'fieldOverlayModeSelect',
      cluster: 'viewport',
      selected: f > 0,
      availability: missingProducer ? AVAILABILITY.REQUIRES_PREREQ : (f > 0 ? AVAILABILITY.ACTIVE : AVAILABILITY.AVAILABLE),
      ctlState: missingProducer ? CTL_STATES.CONSTRAINED : (f > 0 ? CTL_STATES.ACTIVE : CTL_STATES.IDLE),
      reasons: missingProducer ? [{ code: 'NO_PRODUCER', label: `Needs ${missingProducer}` }] : [],
      conflictWith: [],
      fixAction: missingProducer ? { label: '+Engine', intent: { kind: 'SET_FIELD', field: f } } : null,
      displayLabel: dynamicOverlayLabel(f, org),
    });
  }

  // ---- 12. Scratch Arena Bus Summary (meta-entry for UI + verifier) ----
  put('bosBusStrip', {
    id: 'bosBusStrip',
    cluster: 'engines',
    selected: bosOwner !== BOS_OWNERS.NONE,
    availability: AVAILABILITY.ACTIVE,
    ctlState: bosClaimants.length > 1
      ? CTL_STATES.REFUSED
      : (bosOwner !== BOS_OWNERS.NONE ? CTL_STATES.ACTIVE : CTL_STATES.IDLE),
    owner: bosOwner,
    claimants: bosClaimants,
    reasons: bosClaimants.length > 1
      ? [{ code: 'MULTI_CLAIM', label: `${bosClaimants.length} organs claim scratch` }]
      : [],
    conflictWith: bosClaimants.filter((c) => c !== bosOwner),
    fixAction: null,
  });

  return out;
}

/**
 * Pure visibility projection (`ALL_CONSTRAINED` vs `APPLICABLE_ONLY`).
 * Never removes DOM or shifts layout; returns `true` for full-strength controls
 * and `false` for quieted-in-place controls under `APPLICABLE_ONLY`.
 */
export function keepControlFullStrength(ctl, state) {
  if (!ctl) return true;
  if (state.visibility_policy === 'ALL_CONSTRAINED') return true;
  if (ctl.selected) return true;
  if (ctl.availability !== AVAILABILITY.UNSUPPORTED_FOR_SOURCE) return true;
  return false;
}
