// ============================================================================
// CHRONOS-SPECTRAL v3.5 // Complete UI Control & Reveal Ownership Registry
// Maps every interactive DOM control to its owning cluster, tab, and role so
// `scripts/check_ui_contract.mjs` and `window.__chronosUi.reveal(id)` stay 100% in sync.
// ============================================================================

export const CONTROL_REGISTRY = Object.freeze([
  // Status Rail (always visible)
  { id: 'btnOpenPresets', cluster: 'rail', rackTab: null, bayTab: null, key: 'P' },
  { id: 'btnVisibilityPolicy', cluster: 'rail', rackTab: null, bayTab: null, key: null },
  { id: 'btnShortcuts', cluster: 'rail', rackTab: null, bayTab: null, key: '?' },
  { id: 'btnRailRec', cluster: 'rail', rackTab: null, bayTab: null, key: 'R' },
  { id: 'btnUndoConfig', cluster: 'rail', rackTab: null, bayTab: null, key: null },

  // Viewport Toolstrip (always visible)
  { id: 'toolInspect', cluster: 'viewport', rackTab: null, bayTab: null, key: '1' },
  { id: 'toolSeedBox', cluster: 'viewport', rackTab: null, bayTab: null, key: '2' },
  { id: 'btnCalRuler', cluster: 'viewport', rackTab: null, bayTab: null, key: '3' },
  { id: 'btnCalQuad', cluster: 'viewport', rackTab: null, bayTab: null, key: '4' },
  { id: 'btnToolCropRoi', cluster: 'viewport', rackTab: null, bayTab: null, key: '5' },
  { id: 'toolWipe', cluster: 'viewport', rackTab: null, bayTab: null, key: '6' },
  { id: 'fieldOverlayModeSelect', cluster: 'viewport', rackTab: null, bayTab: null, key: null },
  { id: 'btnHudHeroToggle', cluster: 'viewport', rackTab: null, bayTab: null, key: 'H' },
  { id: 'btnHudFullscreen', cluster: 'viewport', rackTab: null, bayTab: null, key: 'F' },
  { id: 'hudWipeHandle', cluster: 'viewport', rackTab: null, bayTab: null, key: null },
  { id: 'showcaseWipeSlider', cluster: 'viewport', rackTab: null, bayTab: null, key: null },

  // Viewport Source Transport (always visible slot)
  { id: 'btnVidPlayPause', cluster: 'transport', rackTab: null, bayTab: null, key: 'Space' },
  { id: 'btnVidStepBack', cluster: 'transport', rackTab: null, bayTab: null, key: ',' },
  { id: 'btnVidStepFwd', cluster: 'transport', rackTab: null, bayTab: null, key: '.' },
  { id: 'btnVidReplay', cluster: 'transport', rackTab: null, bayTab: null, key: null },
  { id: 'videoScrubSlider', cluster: 'transport', rackTab: null, bayTab: null, key: null },
  { id: 'btnFastPass', cluster: 'transport', rackTab: null, bayTab: null, key: null },
  { id: 'liveCropSelect', cluster: 'transport', rackTab: null, bayTab: null, key: null },
  { id: 'btnReopenLiveModal', cluster: 'transport', rackTab: null, bayTab: null, key: null },
  { id: 'btnStopLiveFeed', cluster: 'transport', rackTab: null, bayTab: null, key: null },
  { id: 'benchScenarioSelect', cluster: 'transport', rackTab: null, bayTab: null, key: null },

  // Left Rack: ENGINES (default active rack tab)
  { id: 'sliderK', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'btnLabStimulus', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'swarmModeSelect', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'hodgeModeSelect', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'metrologyModeSelect', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'windModeSelect', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'bosModeSelect', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'depthModeSelect', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'rsVibroModeSelect', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'elastoModeSelect', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'probeExciterPreset', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'btnProbeExciter', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'inpRsDtRowUs', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'inpRsExposureUs', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },
  { id: 'selRsScanDir', cluster: 'engines', rackTab: 'engines', bayTab: null, key: null },

  // Left Rack: ACQUIRE
  { id: 'btnResSd', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: null },
  { id: 'btnResHd', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: null },
  { id: 'btnSrcTestbench', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: null },
  { id: 'btnSrcWebcam', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: null },
  { id: 'btnSrcLiveStream', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: null },
  { id: 'mp4Input', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: 'O' },
  { id: 'sampleVideoSelect', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: null },
  { id: 'chkNativeRoi', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: null },
  { id: 'chkAutoTrackRoi', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: null },
  { id: 'rangeRoiX0', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: null },
  { id: 'rangeRoiY0', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: null },
  { id: 'btnResetRoiCrop', cluster: 'acquire', rackTab: 'acquire', bayTab: null, key: null },

  // Left Rack: CALIBRATE
  { id: 'btnCalAutoG', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'btnCalPreset', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'btnCalClear', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'calRulerM', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'calPlaneW', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'calPlaneH', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'btnCalApplyDims', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'calCoordMode', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'coordX1', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'coordY1', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'coordX2', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'coordY2', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'coordX3', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'coordY3', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'coordX4', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'coordY4', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },
  { id: 'btnCommitCoords', cluster: 'calibrate', rackTab: 'calibrate', bayTab: null, key: null },

  // Left Rack: RECORD & DISPLAY TUNING
  { id: 'btnRecToggle', cluster: 'record', rackTab: 'record', bayTab: null, key: 'R' },
  { id: 'btnRecClear', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'btnExportCsv', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'btnExportJson', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'btnExportRecipe', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'chkHeroTargets', cluster: 'record', rackTab: 'record', bayTab: null, key: 'H' },
  { id: 'chkKalman', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'chkAllOverlays', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'chkEcaOverlay', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'chkCovEllipses', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'btnAudioToggle', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'btnResetArena', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'sliderEca', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'sliderLambda', cluster: 'record', rackTab: 'record', bayTab: null, key: null },
  { id: 'sliderGain', cluster: 'record', rackTab: 'record', bayTab: null, key: null },

  // Right Roster (always visible at >=1280px)
  { id: 'btnFocusCycle', cluster: 'roster', rackTab: null, bayTab: null, key: ']' },

  // Scope Bay: PHASE
  { id: 'chkStrobe', cluster: 'phase', rackTab: null, bayTab: 'phase', key: null },

  // Scope Bay: AUDIT A/B
  { id: 'btnAuditSampleRef', cluster: 'audit', rackTab: null, bayTab: 'audit', key: null },
  { id: 'btnAuditSampleStutter', cluster: 'audit', rackTab: null, bayTab: 'audit', key: null },
  { id: 'btnAuditCurrent', cluster: 'audit', rackTab: null, bayTab: 'audit', key: null },
  { id: 'btnAuditLiveToggle', cluster: 'audit', rackTab: null, bayTab: 'audit', key: null },
  { id: 'btnAuditExportJson', cluster: 'audit', rackTab: null, bayTab: 'audit', key: null },
  { id: 'btnAuditChooseA', cluster: 'audit', rackTab: null, bayTab: 'audit', key: null },
  { id: 'btnAuditChooseB', cluster: 'audit', rackTab: null, bayTab: 'audit', key: null },
  { id: 'auditFileA', cluster: 'audit', rackTab: null, bayTab: 'audit', key: null },
  { id: 'auditFileB', cluster: 'audit', rackTab: null, bayTab: 'audit', key: null },

  // Live Stream Modal
  { id: 'btnStartTier1Capture', cluster: 'live_dialog', rackTab: null, bayTab: null, key: null },
  { id: 'tier3PresetSelect', cluster: 'live_dialog', rackTab: null, bayTab: null, key: null },
  { id: 'tier3CustomHlsUrl', cluster: 'live_dialog', rackTab: null, bayTab: null, key: null },
  { id: 'btnStartTier3Mse', cluster: 'live_dialog', rackTab: null, bayTab: null, key: null },
  { id: 'btnRefreshTier3Presets', cluster: 'live_dialog', rackTab: null, bayTab: null, key: null },
  { id: 'tier2StreamUrl', cluster: 'live_dialog', rackTab: null, bayTab: null, key: null },
  { id: 'btnStartTier2Url', cluster: 'live_dialog', rackTab: null, bayTab: null, key: null },
  { id: 'btnFallbackTier2ToTier3', cluster: 'live_dialog', rackTab: null, bayTab: null, key: null },
]);

export const CONTROL_BY_ID = Object.freeze(
  Object.fromEntries(CONTROL_REGISTRY.map((c) => [c.id, c]))
);

// Panel/canvas IDs owned by specific Scope Bay tabs for `window.__chronosUi.reveal(id)`.
export const BAY_PANEL_OWNERSHIP = Object.freeze({
  pipCanvas: 'track',
  persistencePanel: 'track',
  persistenceCanvas: 'track',
  hilbertCanvas: 'phase',
  spectralCanvas: 'phase',
  physicsIdPanel: 'physics',
  physicsPhaseCanvas: 'physics',
  continuumFieldPanel: 'field',
  expertDiagnosticsDrawer: 'field',
  physicsAuditCard: 'audit',
  auditTimelineCanvas: 'audit',
});
