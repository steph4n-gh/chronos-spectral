// ============================================================================
// CHRONOS-SPECTRAL v3.5 // Gates, Constraint Chips, Segmented Switches &
// 5-Slot Scratch Arena Priority Bus View
//
// Pure DOM projection from `InstrumentState` + `deriveControlStates(state)`.
// Every interactive `[data-ctl]` element receives an explicit `data-ctl-state`
// (`idle|active|pending|constrained|refused|na|unavailable`) and `data-ctl-quiet`.
// ============================================================================

import { BOS_OWNERS, computeBosOwner, listBosClaimants } from '../core/fsm.js';
import {
  CTL_STATES,
  OVERLAY_SPECS,
  bosOwnerDisplayName,
  deriveControlStates,
  dynamicOverlayLabel,
  keepControlFullStrength,
} from '../core/constraints.js';

const LAMP_GLYPHS = Object.freeze({
  [CTL_STATES.IDLE]: '○',
  [CTL_STATES.ACTIVE]: '◆',
  [CTL_STATES.PENDING]: '◇',
  [CTL_STATES.CONSTRAINED]: '◐',
  [CTL_STATES.REFUSED]: '✕',
  [CTL_STATES.NA]: '⊘',
  [CTL_STATES.UNAVAILABLE]: '⊘',
});

const BUS_SLOTS = Object.freeze([
  { owner: BOS_OWNERS.DEPTH_STEREO, slotElId: 'busSlotStereo', short: 'P1', label: 'STEREO' },
  { owner: BOS_OWNERS.BOS_THERMAL, slotElId: 'busSlotBos', short: 'P2', label: 'THERMAL' },
  { owner: BOS_OWNERS.WIND_TUNNEL, slotElId: 'busSlotWind', short: 'P3', label: 'WIND' },
  { owner: BOS_OWNERS.ELASTO_ACOUSTIC_BOS, slotElId: 'busSlotAbos', short: 'P4', label: 'A-BOS' },
  { owner: BOS_OWNERS.ELASTO_SURFACE, slotElId: 'busSlotElasto', short: 'P5', label: 'ELASTO' },
]);

const ENGINE_ROW_BINDINGS = Object.freeze([
  { selectId: 'swarmModeSelect', rowId: 'rowSwarm', intentKind: 'SET_SWARM', statusId: 'statusSwarm' },
  { selectId: 'hodgeModeSelect', rowId: 'rowHodge', intentKind: 'SET_HODGE', statusId: 'statusHodge' },
  { selectId: 'metrologyModeSelect', rowId: 'rowMetrology', intentKind: 'SET_METROLOGY', statusId: 'statusMetrology' },
  { selectId: 'depthModeSelect', rowId: 'rowDepth', intentKind: 'SET_DEPTH', statusId: 'statusDepth' },
  { selectId: 'bosModeSelect', rowId: 'rowBos', intentKind: 'SET_BOS', statusId: 'statusBos' },
  { selectId: 'windModeSelect', rowId: 'rowWind', intentKind: 'SET_WIND', statusId: 'statusWind' },
  { selectId: 'rsVibroModeSelect', rowId: 'rowRs', intentKind: 'SET_RS', statusId: 'statusRs' },
  { selectId: 'elastoModeSelect', rowId: 'rowElasto', intentKind: 'SET_ELASTO', statusId: 'statusElasto' },
]);

function toneForCtlState(ctlState) {
  if (ctlState === CTL_STATES.ACTIVE) return 'lock';
  if (ctlState === CTL_STATES.PENDING || ctlState === CTL_STATES.CONSTRAINED) return 'warn';
  if (ctlState === CTL_STATES.REFUSED) return 'fault';
  return 'idle';
}

function setText(el, next) {
  if (el && el.textContent !== next) el.textContent = next;
}

export function createGatesView({ onApplyIntent, onTakeBosOwner, onUndo, onStartProbe }) {
  const rowRefs = new Map();
  for (const spec of ENGINE_ROW_BINDINGS) {
    const row = document.getElementById(spec.rowId);
    const select = document.getElementById(spec.selectId);
    const statusSlot = document.getElementById(spec.statusId);
    const lamp = row ? row.querySelector('.lamp') : null;
    const segButtons = row ? Array.from(row.querySelectorAll('.seg-opt')) : [];
    rowRefs.set(spec.selectId, { spec, row, select, statusSlot, lamp, segButtons });

    // Hide native engine selects from keyboard focus / assistive tree since segmented switches represent them.
    if (select) {
      select.tabIndex = -1;
      select.setAttribute('aria-hidden', 'true');
    }
    segButtons.forEach((btn, idx) => {
      btn.addEventListener('click', () => {
        if (!select) return;
        const nextVal = String(btn.dataset.val ?? '0');
        if (select.value !== nextVal) {
          select.value = nextVal;
          select.dispatchEvent(new Event('change', { bubbles: true }));
        }
      });
      btn.addEventListener('keydown', (ev) => {
        let nextIdx = -1;
        if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') nextIdx = (idx + 1) % segButtons.length;
        else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') nextIdx = (idx - 1 + segButtons.length) % segButtons.length;
        if (nextIdx >= 0) {
          ev.preventDefault();
          segButtons[nextIdx].focus();
          segButtons[nextIdx].click();
        }
      });
    });
  }

  // Bind 5-slot Scratch Arena Priority Bus buttons (1-click atomic handoff!)
  const busSlotEls = new Map();
  for (const slot of BUS_SLOTS) {
    const el = document.getElementById(slot.slotElId);
    if (el) {
      busSlotEls.set(slot.owner, el);
      el.addEventListener('click', () => {
        if (typeof onTakeBosOwner === 'function') {
          onTakeBosOwner(slot.owner);
        }
      });
    }
  }

  const busOwnerLabel = document.getElementById('bosBusOwnerLabel');
  const busStripEl = document.getElementById('bosBusStrip');
  const btnUndo = document.getElementById('btnUndoConfig');
  const fieldOverlaySel = document.getElementById('fieldOverlayModeSelect');
  const fieldOverlayStatus = document.getElementById('fieldOverlayStatus');
  const kUnlockStrip = document.getElementById('kUnlockStrip');
  const strobeStatusSlot = document.getElementById('statusStrobe');

  if (btnUndo) {
    btnUndo.addEventListener('click', () => {
      if (btnUndo.getAttribute('aria-disabled') === 'true') return;
      if (typeof onUndo === 'function') onUndo();
    });
  }

  // Track last rendered fixAction per statusSlot to avoid needless DOM thrash at 60 Hz
  const statusCache = new Map();

  function renderStatusSlot(slotEl, cap, confirmed) {
    if (!slotEl) return;
    const reasonTxt = cap.reasons && cap.reasons.length > 0 ? cap.reasons[0].label : '';
    const fixLabel = cap.fixAction ? cap.fixAction.label : '';
    const fixIntentKey = cap.fixAction ? JSON.stringify(cap.fixAction.intent) : '';
    const tone = toneForCtlState(cap.ctlState);
    const cacheKey = `${cap.ctlState}|${confirmed ? 1 : 0}|${reasonTxt}|${fixLabel}|${fixIntentKey}`;
    if (statusCache.get(slotEl) === cacheKey) return;
    statusCache.set(slotEl, cacheKey);

    slotEl.textContent = '';
    if (reasonTxt) {
      const chip = document.createElement('span');
      chip.className = 'sig-chip';
      chip.dataset.tone = tone;
      chip.dataset.confirmed = String(Boolean(confirmed));
      chip.textContent = reasonTxt;
      chip.title = cap.reasons.map((r) => r.label).join(' · ');
      slotEl.appendChild(chip);
    } else if (cap.ctlState === CTL_STATES.ACTIVE) {
      const chip = document.createElement('span');
      chip.className = 'sig-chip';
      chip.dataset.tone = 'lock';
      chip.dataset.confirmed = String(Boolean(confirmed));
      chip.textContent = 'LOCK';
      slotEl.appendChild(chip);
    } else if (cap.ctlState === CTL_STATES.PENDING) {
      const chip = document.createElement('span');
      chip.className = 'sig-chip';
      chip.dataset.tone = 'warn';
      chip.dataset.confirmed = 'false';
      chip.textContent = 'WARM';
      slotEl.appendChild(chip);
    }

    if (cap.fixAction) {
      const fixBtn = document.createElement('button');
      fixBtn.type = 'button';
      fixBtn.className = 'ctl-btn accent-fix';
      fixBtn.textContent = cap.fixAction.label;
      fixBtn.title = `Atomic 1-click fix: ${cap.fixAction.label}`;
      fixBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        if (cap.fixAction.intent?.kind === 'START_PROBE') {
          if (typeof onStartProbe === 'function') onStartProbe();
        } else if (typeof onApplyIntent === 'function') {
          onApplyIntent(cap.fixAction.intent, cap.fixAction.label);
        }
      });
      slotEl.appendChild(fixBtn);
    }
  }

  function render(state) {
    const caps = deriveControlStates(state);
    const org = state.organ_config;
    const rt = state.runtime_health;
    const bosOwner = computeBosOwner(org);
    const claimants = listBosClaimants(org);
    const kernelConfirmed = rt.workerState === 'READY' && (!state.pending || rt.appliedRevision >= state.revision);

    // 1. Update body visibility policy attribute
    if (document.body.dataset.visibilityPolicy !== state.visibility_policy) {
      document.body.dataset.visibilityPolicy = state.visibility_policy;
    }

    // 2. Update 5-Slot Scratch Arena Bus
    if (busStripEl) {
      const busCap = caps.get('bosBusStrip');
      busStripEl.dataset.ctlState = busCap ? busCap.ctlState : CTL_STATES.IDLE;
    }
    setText(busOwnerLabel, bosOwner === BOS_OWNERS.NONE ? 'FREE' : bosOwnerDisplayName(bosOwner).toUpperCase());
    for (const slot of BUS_SLOTS) {
      const el = busSlotEls.get(slot.owner);
      if (!el) continue;
      let busState = 'free';
      if (bosOwner === slot.owner) busState = 'owner';
      else if (claimants.includes(slot.owner)) busState = 'blocked';
      if (el.dataset.busState !== busState) el.dataset.busState = busState;
      el.setAttribute('aria-pressed', String(busState === 'owner'));
      const tagSpan = el.firstElementChild;
      if (tagSpan) {
        const prefix = busState === 'owner' ? '●' : (busState === 'blocked' ? '!' : '');
        const want = prefix ? `${prefix}${slot.short}` : slot.short;
        if (tagSpan.textContent !== want) tagSpan.textContent = want;
      }
      el.title = busState === 'owner'
        ? `${slot.label} owns shared BOS scratch arena`
        : busState === 'blocked'
        ? `${slot.label} blocked by ${bosOwnerDisplayName(bosOwner)}. Click to hand off scratch (atomic, undoable).`
        : `Click to hand BOS scratch arena to ${slot.label} (atomic, undoable)`;
    }

    // 3. Update every engine row: lamp, segmented switch active state, reason chip & fix button
    for (const [selectId, ref] of rowRefs.entries()) {
      const cap = caps.get(selectId);
      const ctlState = cap ? cap.ctlState : CTL_STATES.IDLE;
      const curVal = ref.select ? String(ref.select.value) : '0';
      if (ref.row) {
        ref.row.dataset.ctlState = ctlState;
        ref.row.dataset.ctlQuiet = String(!keepControlFullStrength(cap, state));
      }
      if (ref.select) {
        ref.select.dataset.ctlState = ctlState;
      }
      if (ref.lamp) {
        setText(ref.lamp, LAMP_GLYPHS[ctlState] || '○');
      }
      for (const btn of ref.segButtons) {
        const checked = String(btn.dataset.val) === curVal;
        if (btn.getAttribute('aria-checked') !== String(checked)) {
          btn.setAttribute('aria-checked', String(checked));
        }
        btn.tabIndex = checked ? 0 : -1;
      }
      if (cap) {
        renderStatusSlot(ref.statusSlot, cap, kernelConfirmed);
      }
    }

    // 4. Dynamic Overlay 11 (RS Amplitude vs Welch Coherence) & Overlay 13 (FIELD ONLY) labels
    if (fieldOverlaySel) {
      const opt11 = fieldOverlaySel.querySelector('option[value="11"]');
      const lbl11 = dynamicOverlayLabel(11, org);
      if (opt11 && opt11.textContent !== lbl11) opt11.textContent = lbl11;

      const opt13 = fieldOverlaySel.querySelector('option[value="13"]');
      const lbl13 = dynamicOverlayLabel(13, org);
      if (opt13 && opt13.textContent !== lbl13) opt13.textContent = lbl13;

      const fCap = caps.get('fieldOverlayModeSelect');
      if (fCap) {
        fieldOverlaySel.dataset.ctlState = fCap.ctlState;
        renderStatusSlot(fieldOverlayStatus, fCap, kernelConfirmed);
      }
    }

    // 5. Lookahead K >= 7 unlock indicator & Strobe chip
    if (kUnlockStrip) {
      const kOk = org.lookahead >= 7;
      const nextTxt = kOk
        ? `K=${org.lookahead} · Hilbert, Strobe & Hermite UNLOCKED`
        : `K=${org.lookahead} · Raise K >= 7 to unlock Hilbert, Strobe & Hermite`;
      setText(kUnlockStrip, nextTxt);
      kUnlockStrip.dataset.signal = kOk ? 'lock' : 'warn';
    }
    const strobeCap = caps.get('chkStrobe');
    if (strobeCap) {
      renderStatusSlot(strobeStatusSlot, strobeCap, kernelConfirmed);
    }

    // 6. Undo button visibility/enablement (fixed slot, never shifts rail!)
    if (btnUndo) {
      const hasUndo = Boolean(state.undoConfig);
      btnUndo.setAttribute('aria-disabled', String(!hasUndo));
      btnUndo.dataset.ctlState = hasUndo ? CTL_STATES.PENDING : CTL_STATES.IDLE;
      btnUndo.title = hasUndo
        ? `Restore: ${state.undoConfig.label}`
        : 'Undo last atomic engine handoff or fix';
    }

    // 7. Stamp 100% of `[data-ctl]` elements with `data-ctl-state` & `data-ctl-quiet`
    const allCtls = document.querySelectorAll('[data-ctl]');
    const hasSource = state.source.kind !== 'IDLE';
    const isMedia = state.source.kind === 'FILE' || state.source.kind === 'SAMPLE';
    const isLive = state.source.kind === 'LIVE_TAB' || state.source.kind === 'LIVE_URL' || state.source.kind === 'LIVE_MSE';
    const isBench = state.source.kind === 'TESTBENCH';

    for (let i = 0; i < allCtls.length; i++) {
      const el = allCtls[i];
      const id = el.id || el.dataset.ctl;
      const cap = caps.get(id);
      if (cap) {
        if (el.dataset.ctlState !== cap.ctlState) el.dataset.ctlState = cap.ctlState;
        const quiet = !keepControlFullStrength(cap, state);
        if (el.dataset.ctlQuiet !== String(quiet)) el.dataset.ctlQuiet = String(quiet);
        continue;
      }

      // Deterministic derivation for non-organ controls so 0 controls ever lack `data-ctl-state`.
      let st = CTL_STATES.IDLE;
      let quiet = false;
      const pressed = el.getAttribute('aria-pressed') === 'true' || el.getAttribute('aria-checked') === 'true';
      const checked = el.tagName === 'INPUT' && el.type === 'checkbox' ? el.checked : false;
      const disabled = el.getAttribute('aria-disabled') === 'true' || el.disabled === true;

      if (id === 'btnVidPlayPause' || id === 'btnVidStepBack' || id === 'btnVidStepFwd' || id === 'btnVidReplay' || id === 'videoScrubSlider') {
        st = isMedia ? (pressed ? CTL_STATES.ACTIVE : CTL_STATES.IDLE) : CTL_STATES.NA;
        quiet = state.visibility_policy === 'APPLICABLE_ONLY' && !isMedia;
      } else if (id === 'liveCropSelect' || id === 'btnReopenLiveModal' || id === 'btnStopLiveFeed') {
        st = isLive ? CTL_STATES.ACTIVE : CTL_STATES.NA;
        quiet = state.visibility_policy === 'APPLICABLE_ONLY' && !isLive;
      } else if (id === 'benchScenarioSelect') {
        st = isBench ? CTL_STATES.ACTIVE : CTL_STATES.NA;
        quiet = state.visibility_policy === 'APPLICABLE_ONLY' && !isBench;
      } else if (id === 'btnStartTier3Mse' || id === 'btnRefreshTier3Presets' || id === 'tier3PresetSelect' || id === 'tier3CustomHlsUrl') {
        st = rt.host.staticHost ? CTL_STATES.UNAVAILABLE : CTL_STATES.IDLE;
        quiet = state.visibility_policy === 'APPLICABLE_ONLY' && rt.host.staticHost;
      } else if (id === 'btnProbeExciter' || id === 'probeExciterPreset' || id === 'inpRsDtRowUs' || id === 'inpRsExposureUs' || id === 'selRsScanDir') {
        st = pressed ? CTL_STATES.ACTIVE : CTL_STATES.IDLE;
      } else if (id === 'btnRecToggle' || id === 'btnRailRec') {
        st = state.recorder.active ? CTL_STATES.ACTIVE : CTL_STATES.IDLE;
      } else if (id === 'btnVisibilityPolicy') {
        st = state.visibility_policy === 'APPLICABLE_ONLY' ? CTL_STATES.ACTIVE : CTL_STATES.IDLE;
      } else if (id === 'btnOpenPresets') {
        st = state.workspace_layout.dialog === 'PRESETS' ? CTL_STATES.ACTIVE : CTL_STATES.IDLE;
      } else if (id === 'btnShortcuts') {
        st = state.workspace_layout.dialog === 'SHORTCUTS' ? CTL_STATES.ACTIVE : CTL_STATES.IDLE;
      } else if (id === 'btnAuditCurrent') {
        st = isMedia ? CTL_STATES.IDLE : CTL_STATES.NA;
        quiet = state.visibility_policy === 'APPLICABLE_ONLY' && !isMedia;
      } else if (id === 'chkAutoTrackRoi' || id === 'rangeRoiX0' || id === 'rangeRoiY0' || id === 'btnResetRoiCrop') {
        st = !hasSource ? CTL_STATES.NA : (checked ? CTL_STATES.ACTIVE : CTL_STATES.IDLE);
        quiet = state.visibility_policy === 'APPLICABLE_ONLY' && !hasSource;
      } else if (disabled) {
        st = CTL_STATES.NA;
      } else if (pressed || checked) {
        st = CTL_STATES.ACTIVE;
      }

      if (el.dataset.ctlState !== st) el.dataset.ctlState = st;
      if (el.dataset.ctlQuiet !== String(quiet)) el.dataset.ctlQuiet = String(quiet);
    }

    return caps;
  }

  return {
    render,
    overlaySpecs: OVERLAY_SPECS,
  };
}
