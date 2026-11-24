// ============================================================================
// CHRONOS-SPECTRAL v3.5 // Bench Shell View: 4-Tab Rack, 5-Meter Scope Bay,
// Presets Drawer (`Load` + `Settings Only`), Welcome/Drop Card, 6-Tool Strip,
// Shortcuts Dialog & `window.__chronosUi` Contract Surface
// ============================================================================

import { BAY_TABS, RACK_TABS } from '../core/fsm.js';
import { BAY_PANEL_OWNERSHIP, CONTROL_BY_ID } from '../core/registry.js';
import { RECIPES, RECIPE_BY_KEY, RECIPE_DOMAINS } from '../recipes/registry.js';

// Scope visibility bitmask matching `web/worker.js` `renderSecondaryScopes`:
// bit 0 (0x01) = Track (PiP + Persistence), bit 1 (0x02) = Phase (Hilbert + Spectral),
// bit 2 (0x04) = Physics (physicsPhase), bit 3 (0x08) = 16-Ch Roster (always visible).
const BAY_SCOPE_MASK = Object.freeze({
  track: 0x09,
  phase: 0x0a,
  physics: 0x0c,
  field: 0x08,
  audit: 0x08,
});

export function createBenchShell({
  getState,
  dispatch,
  onScopeVisibilityMask,
  onLoadRecipe,
  onOpenFile,
  onStartWebcam,
  onOpenLiveStream,
  onStartSignalGen,
  onSelectPointerTool,
  onTakeBosScratch,
  onApplyIntent,
  onPlanIntent,
  onUndo,
  onExportRecipe,
  onToggleHero,
  onToggleFullscreen,
  onCycleFocus,
  onToggleRec,
  onPlayPause,
  onStepFrame,
  onEscape,
}) {
  const rackTabs = Array.from(document.querySelectorAll('.rack-tab[data-rack-tab]'));
  const rackPages = Array.from(document.querySelectorAll('.rack-page[data-rack-page]'));
  const bayTabs = Array.from(document.querySelectorAll('.bay-tab[data-bay-tab]'));
  const bayPages = Array.from(document.querySelectorAll('.bay-page[data-bay-page]'));

  const presetsDrawer = document.getElementById('presetsDrawer');
  const presetsSearch = document.getElementById('presetsSearchInput');
  const presetsList = document.getElementById('presetsDrawerList');
  const btnOpenPresets = document.getElementById('btnOpenPresets');
  const btnClosePresets = document.getElementById('btnClosePresets');
  const btnVisPolicy = document.getElementById('btnVisibilityPolicy');
  const btnShortcuts = document.getElementById('btnShortcuts');
  const shortcutsDialog = document.getElementById('shortcutsDialog');
  const welcomeOverlay = document.getElementById('welcomeOverlay');
  const hudWipeTrack = document.getElementById('hudWipeTrack');
  const btnRailRec = document.getElementById('btnRailRec');
  const btnExportRecipe = document.getElementById('btnExportRecipe');

  const toolButtons = {
    INSPECT_TARGET: document.getElementById('toolInspect'),
    SEED_BOX: document.getElementById('toolSeedBox'),
    CAL_RULER: document.getElementById('btnCalRuler'),
    CAL_QUAD: document.getElementById('btnCalQuad'),
    CROP_ROI: document.getElementById('btnToolCropRoi'),
    WIPE_DRAG: document.getElementById('toolWipe'),
  };
  const toolHintEl = document.getElementById('toolOptionsHint');
  const activeRecipeBadge = document.getElementById('activeRecipeBadge');

  // ---- Populate Presets Drawer (all 53 declarative recipes, Load + Settings Only) ----
  const recipeRowRefs = new Map();
  function buildPresetsDrawer() {
    if (!presetsList) return;
    presetsList.textContent = '';
    const isStatic = document.body.dataset.staticHost === 'true';

    for (const domain of RECIPE_DOMAINS) {
      const items = RECIPES.filter((r) => r.domain === domain.id);
      if (items.length === 0) continue;

      const sec = document.createElement('section');
      sec.className = 'recipe-domain-group';
      sec.dataset.domain = domain.id;

      const hdr = document.createElement('div');
      hdr.className = 'recipe-domain-title';
      hdr.textContent = `${domain.label} (${items.length})`;
      sec.appendChild(hdr);

      for (const r of items) {
        const row = document.createElement('div');
        row.className = 'recipe-row';
        row.dataset.recipeKey = r.key;
        row.dataset.searchText = `${r.title} ${r.key} ${domain.label} ${r.requires.join(' ')}`.toLowerCase();

        const info = document.createElement('div');
        const title = document.createElement('div');
        title.className = 'recipe-row-title';
        title.textContent = r.title;
        info.appendChild(title);

        const meta = document.createElement('div');
        meta.className = 'recipe-row-meta';
        const tags = [];
        tags.push(r.resMode === 1 ? '640x360 HD' : '320x180 SD');
        if (r.durationS > 0) tags.push(`${r.durationS}s`);
        if (r.organs.bos) tags.push('BOS');
        if (r.organs.wind === 3) tags.push('Dense Wind');
        else if (r.organs.wind) tags.push('Wind');
        if (r.organs.hodge) tags.push('Hodge');
        if (r.organs.elasto === 4) tags.push('Acoustic BOS');
        else if (r.organs.elasto) tags.push('Elasto');
        if (r.organs.depth) tags.push('Depth');
        if (r.organs.rs) tags.push('RS');
        if (r.organs.swarm) tags.push('Swarm');
        if (r.requires.includes('relay')) tags.push(isStatic ? 'Local relay req.' : 'Live relay');
        if (r.requires.includes('audio')) tags.push('48 kHz audio');
        meta.textContent = tags.join(' · ');
        info.appendChild(meta);
        row.appendChild(info);

        const actions = document.createElement('div');
        actions.className = 'recipe-row-actions';

        // Primary Load button keeps `.showcase-btn` + `data-showcase` for 100% E2E selector continuity!
        const btnLoad = document.createElement('button');
        btnLoad.type = 'button';
        btnLoad.className = 'ctl-btn sm showcase-btn';
        btnLoad.dataset.showcase = r.key;
        btnLoad.dataset.recipe = r.key;
        if (r.requires.includes('relay')) btnLoad.dataset.localServerOnly = 'true';
        btnLoad.setAttribute('aria-pressed', 'false');
        btnLoad.textContent = 'Load';
        btnLoad.addEventListener('click', () => {
          onLoadRecipe(r.key, { settingsOnly: false });
          setPresetsOpen(false);
        });
        actions.appendChild(btnLoad);

        const btnSettings = document.createElement('button');
        btnSettings.type = 'button';
        btnSettings.className = 'ctl-btn sm';
        btnSettings.dataset.recipeSettingsOnly = r.key;
        btnSettings.textContent = 'Settings Only';
        btnSettings.title = 'Apply this recipe engine, overlay, and calibration setup to your currently loaded video';
        btnSettings.addEventListener('click', () => {
          onLoadRecipe(r.key, { settingsOnly: true });
          setPresetsOpen(false);
        });
        actions.appendChild(btnSettings);

        row.appendChild(actions);
        sec.appendChild(row);
        recipeRowRefs.set(r.key, { row, btnLoad, btnSettings });
      }
      presetsList.appendChild(sec);
    }
  }
  buildPresetsDrawer();

  if (presetsSearch && presetsList) {
    presetsSearch.addEventListener('input', () => {
      const q = presetsSearch.value.trim().toLowerCase();
      presetsList.querySelectorAll('.recipe-domain-group').forEach((grp) => {
        let visibleInGrp = 0;
        grp.querySelectorAll('.recipe-row').forEach((row) => {
          const match = !q || (row.dataset.searchText || '').includes(q);
          row.hidden = !match;
          if (match) visibleInGrp++;
        });
        grp.hidden = visibleInGrp === 0;
      });
    });
  }

  // ---- Rack & Bay Tabs (visibility + inert inside fixed grid cell -> 0.000000 CLS) ----
  function setRackTab(tab) {
    if (!RACK_TABS.includes(tab)) return;
    dispatch({ type: 'SET_RACK_TAB', tab });
  }

  function setBayTab(tab) {
    if (!BAY_TABS.includes(tab)) return;
    dispatch({ type: 'SET_BAY_TAB', tab });
    if (typeof onScopeVisibilityMask === 'function') {
      onScopeVisibilityMask(BAY_SCOPE_MASK[tab] ?? 0x0f);
    }
  }

  function setPresetsOpen(open, restoreFocus = false) {
    dispatch({ type: 'SET_DIALOG', dialog: open ? 'PRESETS' : 'NONE' });
    if (open && presetsSearch) {
      presetsSearch.value = '';
      presetsSearch.dispatchEvent(new Event('input'));
      setTimeout(() => presetsSearch.focus(), 0);
    } else if (!open && restoreFocus && btnOpenPresets) {
      btnOpenPresets.focus();
    }
  }

  function setShortcutsOpen(open) {
    if (!shortcutsDialog) return;
    if (open) {
      dispatch({ type: 'SET_DIALOG', dialog: 'SHORTCUTS' });
      if (!shortcutsDialog.open) shortcutsDialog.showModal();
    } else {
      if (shortcutsDialog.open) shortcutsDialog.close();
      dispatch({ type: 'SET_DIALOG', dialog: 'NONE' });
    }
  }

  function wireTablistKeyboard(tabs, onSelect) {
    tabs.forEach((btn, idx) => {
      btn.addEventListener('keydown', (ev) => {
        let nextIdx = -1;
        if (ev.key === 'ArrowRight') nextIdx = (idx + 1) % tabs.length;
        else if (ev.key === 'ArrowLeft') nextIdx = (idx - 1 + tabs.length) % tabs.length;
        else if (ev.key === 'Home') nextIdx = 0;
        else if (ev.key === 'End') nextIdx = tabs.length - 1;
        if (nextIdx >= 0) {
          ev.preventDefault();
          onSelect(tabs[nextIdx]);
          tabs[nextIdx].focus();
        }
      });
    });
  }

  rackTabs.forEach((btn) => {
    btn.addEventListener('click', () => setRackTab(btn.dataset.rackTab));
  });
  wireTablistKeyboard(rackTabs, (b) => setRackTab(b.dataset.rackTab));
  bayTabs.forEach((btn) => {
    btn.addEventListener('click', () => setBayTab(btn.dataset.bayTab));
  });
  wireTablistKeyboard(bayTabs, (b) => setBayTab(b.dataset.bayTab));

  if (btnOpenPresets) {
    btnOpenPresets.addEventListener('click', () => {
      const cur = getState().workspace_layout.dialog;
      setPresetsOpen(cur !== 'PRESETS', cur === 'PRESETS');
    });
  }
  if (btnClosePresets) {
    btnClosePresets.addEventListener('click', () => setPresetsOpen(false, true));
  }
  if (btnVisPolicy) {
    btnVisPolicy.addEventListener('click', () => {
      const cur = getState().visibility_policy;
      dispatch({
        type: 'SET_VISIBILITY_POLICY',
        policy: cur === 'APPLICABLE_ONLY' ? 'ALL_CONSTRAINED' : 'APPLICABLE_ONLY',
      });
    });
  }
  if (btnShortcuts) {
    btnShortcuts.addEventListener('click', () => setShortcutsOpen(true));
  }
  if (shortcutsDialog) {
    shortcutsDialog.addEventListener('close', () => {
      if (getState().workspace_layout.dialog === 'SHORTCUTS') {
        dispatch({ type: 'SET_DIALOG', dialog: 'NONE' });
      }
    });
  }
  if (btnRailRec) {
    btnRailRec.addEventListener('click', () => {
      if (typeof onToggleRec === 'function') onToggleRec();
    });
  }
  if (btnExportRecipe) {
    btnExportRecipe.addEventListener('click', () => {
      if (typeof onExportRecipe === 'function') onExportRecipe();
    });
  }

  // ---- Welcome / Instrument-First Drop Overlay Buttons ----
  const btnWelcomeOpen = document.getElementById('btnWelcomeOpenVideo');
  const btnWelcomeWebcam = document.getElementById('btnWelcomeWebcam');
  const btnWelcomeLive = document.getElementById('btnWelcomeLiveStream');
  const btnWelcomeAllPresets = document.getElementById('btnWelcomeAllPresets');
  const btnWelcomeSignalGen = document.getElementById('btnWelcomeSignalGen');

  if (btnWelcomeOpen) btnWelcomeOpen.addEventListener('click', () => onOpenFile());
  if (btnWelcomeWebcam) btnWelcomeWebcam.addEventListener('click', () => onStartWebcam());
  if (btnWelcomeLive) btnWelcomeLive.addEventListener('click', () => onOpenLiveStream());
  if (btnWelcomeAllPresets) btnWelcomeAllPresets.addEventListener('click', () => setPresetsOpen(true));
  if (btnWelcomeSignalGen) btnWelcomeSignalGen.addEventListener('click', () => onStartSignalGen());

  document.querySelectorAll('[data-welcome-preset]').forEach((btn) => {
    btn.addEventListener('click', () => {
      onLoadRecipe(btn.dataset.welcomePreset, { settingsOnly: false });
    });
  });

  // ---- 6-Tool Radio Strip (Inspect, Seed Box, Ruler, Quad, Crop ROI, Wipe) ----
  if (toolButtons.INSPECT_TARGET) {
    toolButtons.INSPECT_TARGET.addEventListener('click', () => onSelectPointerTool('INSPECT_TARGET'));
  }
  if (toolButtons.SEED_BOX) {
    toolButtons.SEED_BOX.addEventListener('click', () => onSelectPointerTool('SEED_BOX'));
  }
  if (toolButtons.WIPE_DRAG) {
    toolButtons.WIPE_DRAG.addEventListener('click', () => onSelectPointerTool('WIPE_DRAG'));
  }

  // ---- Reveal(id): programmatically switches owning Rack/Bay tab with 0 CLS ----
  function reveal(id) {
    const reg = CONTROL_BY_ID[id];
    if (reg) {
      if (reg.rackTab) setRackTab(reg.rackTab);
      if (reg.bayTab) setBayTab(reg.bayTab);
      if (reg.cluster === 'live_dialog') onOpenLiveStream();
      return true;
    }
    const bayOwner = BAY_PANEL_OWNERSHIP[id];
    if (bayOwner) {
      setBayTab(bayOwner);
      return true;
    }
    if (id === 'presetsDrawer') {
      setPresetsOpen(true);
      return true;
    }
    const el = document.getElementById(id);
    if (!el) return false;
    const rackPage = el.closest('[data-rack-page]');
    if (rackPage) setRackTab(rackPage.dataset.rackPage);
    const bayPage = el.closest('[data-bay-page]');
    if (bayPage) setBayTab(bayPage.dataset.bayPage);
    return true;
  }

  // ---- Unified Keyboard Shortcuts ----
  document.addEventListener('keydown', (ev) => {
    const liveDialog = document.getElementById('liveStreamDialog');
    if (ev.key === 'Escape') {
      if (liveDialog && liveDialog.open) return;
      if (shortcutsDialog && shortcutsDialog.open) {
        shortcutsDialog.close();
        return;
      }
      if (getState().workspace_layout.dialog === 'PRESETS') {
        ev.preventDefault();
        setPresetsOpen(false, true);
        return;
      }
      if (typeof onEscape === 'function') onEscape(ev);
      return;
    }
    const k = ev.key;
    const isStepKey = k === ',' || k === '.';
    if (ev.ctrlKey || ev.metaKey || ev.altKey || (ev.repeat && !isStepKey)) return;
    if ((liveDialog && liveDialog.open) || (shortcutsDialog && shortcutsDialog.open)) return;

    const t = ev.target;
    const tag = t && t.tagName ? t.tagName.toUpperCase() : '';
    const inputType = tag === 'INPUT' ? String(t.type || 'text').toLowerCase() : '';
    const isTextInput =
      tag === 'TEXTAREA' ||
      tag === 'SELECT' ||
      (t && t.isContentEditable) ||
      (tag === 'INPUT' && inputType !== 'range' && inputType !== 'checkbox' && inputType !== 'radio' && inputType !== 'button');
    if (isTextInput) return;

    if (k === 'p' || k === 'P') {
      ev.preventDefault();
      const open = getState().workspace_layout.dialog !== 'PRESETS';
      setPresetsOpen(open, !open);
    } else if (k === '?') {
      ev.preventDefault();
      setShortcutsOpen(true);
    } else if (k === '1') {
      ev.preventDefault();
      onSelectPointerTool('INSPECT_TARGET');
    } else if (k === '2') {
      ev.preventDefault();
      onSelectPointerTool('SEED_BOX');
    } else if (k === '3') {
      ev.preventDefault();
      document.getElementById('btnCalRuler')?.click();
    } else if (k === '4') {
      ev.preventDefault();
      document.getElementById('btnCalQuad')?.click();
    } else if (k === '5') {
      ev.preventDefault();
      document.getElementById('btnToolCropRoi')?.click();
    } else if (k === '6') {
      ev.preventDefault();
      onSelectPointerTool('WIPE_DRAG');
    } else if (k === 'o' || k === 'O') {
      ev.preventDefault();
      onOpenFile();
    } else if (k === 'r' || k === 'R') {
      ev.preventDefault();
      onToggleRec();
    } else if (k === 'h' || k === 'H') {
      ev.preventDefault();
      onToggleHero();
    } else if (k === 'f' || k === 'F') {
      ev.preventDefault();
      onToggleFullscreen();
    } else if (k === ' ') {
      const role = t && typeof t.getAttribute === 'function' ? t.getAttribute('role') : '';
      const isInteractiveFocus =
        tag === 'BUTTON' ||
        tag === 'A' ||
        tag === 'SUMMARY' ||
        inputType === 'checkbox' ||
        inputType === 'radio' ||
        role === 'tab' ||
        role === 'radio' ||
        role === 'button';
      if (isInteractiveFocus) return;
      const st = getState();
      if (st.source.kind === 'FILE' || st.source.kind === 'SAMPLE') {
        ev.preventDefault();
        onPlayPause();
      }
    } else if (k === ',') {
      ev.preventDefault();
      onStepFrame(-1);
    } else if (k === '.') {
      ev.preventDefault();
      onStepFrame(1);
    } else if (k === ']') {
      ev.preventDefault();
      onCycleFocus();
    }
  });

  // ---- Pure Render from InstrumentState ----
  function render(state) {
    const wl = state.workspace_layout;

    // 1. Rack Tabs + stacked inert pages
    for (const btn of rackTabs) {
      const on = btn.dataset.rackTab === wl.rackTab;
      if (btn.getAttribute('aria-selected') !== String(on)) btn.setAttribute('aria-selected', String(on));
      btn.tabIndex = on ? 0 : -1;
    }
    for (const page of rackPages) {
      const on = page.dataset.rackPage === wl.rackTab;
      if (page.dataset.active !== String(on)) page.dataset.active = String(on);
      if (page.inert !== !on) page.inert = !on;
    }

    // 2. Scope Bay Tabs + stacked inert pages
    for (const btn of bayTabs) {
      const on = btn.dataset.bayTab === wl.bayTab;
      if (btn.getAttribute('aria-selected') !== String(on)) btn.setAttribute('aria-selected', String(on));
      btn.tabIndex = on ? 0 : -1;
    }
    for (const page of bayPages) {
      const on = page.dataset.bayPage === wl.bayTab;
      if (page.dataset.active !== String(on)) page.dataset.active = String(on);
      if (page.inert !== !on) page.inert = !on;
    }

    // 3. Top-layer Presets Drawer & Welcome Overlay
    if (presetsDrawer) {
      const showDrawer = wl.dialog === 'PRESETS';
      if (presetsDrawer.hidden !== !showDrawer) presetsDrawer.hidden = !showDrawer;
      if (btnOpenPresets && btnOpenPresets.getAttribute('aria-pressed') !== String(showDrawer)) {
        btnOpenPresets.setAttribute('aria-pressed', String(showDrawer));
      }
    }
    if (welcomeOverlay) {
      const showWelcome = wl.welcomeVisible;
      if (welcomeOverlay.hidden !== !showWelcome) welcomeOverlay.hidden = !showWelcome;
      if (hudWipeTrack && hudWipeTrack.hidden !== showWelcome) hudWipeTrack.hidden = showWelcome;
    }

    // 4. Visibility policy rail toggle
    if (btnVisPolicy) {
      const appOnly = state.visibility_policy === 'APPLICABLE_ONLY';
      btnVisPolicy.setAttribute('aria-pressed', String(appOnly));
      btnVisPolicy.textContent = appOnly ? 'View: Applicable' : 'View: All';
    }

    // 5. Rail Rec lamp
    if (btnRailRec) {
      btnRailRec.setAttribute('aria-pressed', String(state.recorder.active));
      const f = state.recorder.frames;
      btnRailRec.textContent = state.recorder.active ? `■ ${f > 9999 ? '9999+' : f}f` : '● Rec';
      btnRailRec.title = state.recorder.active
        ? `Recording active (${f} frames). Click or press R to stop.`
        : 'Start or stop session telemetry recording (R)';
    }

    // 6. 6-Tool Toggle Strip + Contextual Tool Options hint
    const pt = state.pointer_tool;
    const isRuler = pt === 'CAL_RULER_P1' || pt === 'CAL_RULER_P2';
    const isQuad = pt.startsWith('CAL_QUAD_');
    const setTool = (el, on) => {
      if (!el) return;
      if (el.getAttribute('aria-pressed') !== String(on)) el.setAttribute('aria-pressed', String(on));
      el.classList.toggle('active', on);
    };
    const committedCalKind = state.calibration?.committed?.kind || 'PX';
    const rulerPressed = isRuler || (!isQuad && committedCalKind === 'RULER');
    const quadPressed = isQuad || (!isRuler && committedCalKind === 'QUAD');
    setTool(toolButtons.INSPECT_TARGET, pt === 'INSPECT_TARGET');
    setTool(toolButtons.SEED_BOX, pt === 'SEED_BOX');
    setTool(toolButtons.CAL_RULER, rulerPressed);
    setTool(toolButtons.CAL_QUAD, quadPressed);
    setTool(toolButtons.CROP_ROI, pt === 'CROP_ROI');
    setTool(toolButtons.WIPE_DRAG, pt === 'WIPE_DRAG');

    if (toolHintEl) {
      let hint = 'Inspect: click track/field · drag box to seed target';
      if (pt === 'SEED_BOX') hint = 'Seed Box: drag rectangle on canvas to acquire manual target';
      else if (pt === 'CAL_RULER_P1') hint = 'Ruler (1/2): click first metric reference point (Esc cancels)';
      else if (pt === 'CAL_RULER_P2') hint = 'Ruler (2/2): click second metric reference point';
      else if (isQuad) hint = `4-Pt Plane (${pt.slice(-1)}/4): click TL -> TR -> BR -> BL corners`;
      else if (pt === 'CROP_ROI') hint = 'Crop ROI: drag zoom rectangle on viewport';
      else if (pt === 'WIPE_DRAG') hint = 'Split Wipe: drag handle or use Left/Right arrows';
      if (toolHintEl.textContent !== hint) toolHintEl.textContent = hint;
    }

    // 7. Active recipe badge & drawer row highlights
    const activeKey = state.source.originPreset;
    if (activeRecipeBadge) {
      const rec = activeKey ? RECIPE_BY_KEY[activeKey] : null;
      const shortTitle = rec ? rec.title.split('·')[0].split('(')[0].trim() : '';
      const txt = rec
        ? `${shortTitle}${state.source.presetModified ? ' *' : ''}`
        : (state.source.kind === 'IDLE' ? 'IDLE BENCH' : state.source.kind);
      const fullTip = rec
        ? `${rec.title} (${rec.key})${state.source.presetModified ? ' [modified]' : ''}`
        : (state.source.label || txt);
      if (activeRecipeBadge.textContent !== txt) activeRecipeBadge.textContent = txt;
      if (activeRecipeBadge.title !== fullTip) activeRecipeBadge.title = fullTip;
    }
    for (const [key, ref] of recipeRowRefs.entries()) {
      const on = key === activeKey;
      if (ref.row.dataset.active !== String(on)) ref.row.dataset.active = String(on);
      if (ref.btnLoad.getAttribute('aria-pressed') !== String(on)) ref.btnLoad.setAttribute('aria-pressed', String(on));
    }
  }

  // Expose unified public instrument UI controller for tests, keyboard, and tooling
  window.__chronosUi = {
    getState,
    reveal,
    setRackTab,
    setBayTab,
    setVisibilityPolicy: (policy) => dispatch({ type: 'SET_VISIBILITY_POLICY', policy }),
    openPresets: (open = true) => setPresetsOpen(Boolean(open)),
    loadRecipe: (key, opts = {}) => onLoadRecipe(key, opts),
    takeBosScratch: (owner) => onTakeBosScratch(owner),
    planIntent: (intent) => onPlanIntent(intent),
    applyIntent: (intent, label) => onApplyIntent(intent, label),
    undo: () => onUndo(),
    recipes: RECIPES,
  };

  return {
    render,
    reveal,
    setRackTab,
    setBayTab,
    setPresetsOpen,
  };
}
