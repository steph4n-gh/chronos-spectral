"""Zero-dependency Python ctypes wrapper for the native multi-instance CHRONOS-SPECTRAL engine (v3.4.0, ABI 4).

Supports:
- Zero-copy writable RGBA8 ring-buffer views (`ingest_buffer()` + `step_in_place()`)
- Deterministic strided RGBA resample + letterbox (`ingest_rgba_strided()`)
- Per-engine mono f32 audio ingest (`audio_staging()`, `push_audio_block()`, `push_audio()`)
- Explicit 96-byte `ChronosIngestMeta` staging (`set_ingest_meta()`)
- Zero-copy read-only field views via `chronos_field_layout()` (with or without NumPy)
- 8-axis deterministic Physics Integrity Audit (`audit_begin/end/finalize/summary/spans`)
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CHRONOS_ABI_VERSION = 4
CHRONOS_AUDIO_STAGING_LEN = 4096
CHRONOS_AUDIT_AXIS_COUNT = 8

FIELD_TELEMETRY = 0
FIELD_ECA_RES = 1
FIELD_MICRO_TRACKS = 2
FIELD_CONTINUUM = 3
FIELD_RIGID_BODIES = 4
FIELD_SWARM_TELEMETRY = 5
FIELD_BOS_PLANES = 6
FIELD_BOS_VALID = 7
FIELD_BOS_TELEMETRY = 8
FIELD_HODGE_PLANES = 9
FIELD_HODGE_TELEMETRY = 10
FIELD_VIBRO_PLANES = 11
FIELD_VIBRO_TELEMETRY = 12
FIELD_WAVE1_TELEMETRY = 13
FIELD_WAVE2_TELEMETRY = 14
FIELD_DEPTH_TELEMETRY = 15
FIELD_DEPTH_FIELD = 16
FIELD_DEPTH_TRACKS = 17
FIELD_DEPTH_FLAGS = 18
FIELD_DEPTH_HERO = 19
FIELD_WIND_TELEMETRY = 20
FIELD_WIND_PLANES = 21
FIELD_ELASTO_TELEMETRY = 22
FIELD_ELASTO_PLANES = 23
FIELD_ELASTO_VALID = 24

INGEST_PTS_KNOWN = 1 << 1
INGEST_CAPTURE_AGE = 1 << 2
INGEST_HANDOFF_AGE = 1 << 3
INGEST_GEOM_KNOWN = 1 << 4
INGEST_ROI_SLEW = 1 << 6

WIND_MODE_OFF = 0
WIND_MODE_STAGGERED = 1
WIND_MODE_EVERY_FRAME = 2
WIND_MODE_DENSE_AUDIT = 3

AXIS_TRAJECTORY = 0
AXIS_COLLISION = 1
AXIS_DEPTH_PARALLAX = 2
AXIS_FLUID_DIV = 3
AXIS_FLUID_MOMENTUM = 4
AXIS_BG_SHIMMER = 5
AXIS_TEMPORAL_CADENCE = 6
AXIS_AV_SYNC = 7

AXIS_SLUGS = (
    "trajectory",
    "collision",
    "depth_parallax",
    "fluid_div",
    "fluid_momentum",
    "bg_shimmer",
    "temporal_cadence",
    "av_sync",
)

APPL_NAMES = ("unknown", "absent", "insufficient", "eligible")
VERDICT_NAMES = ("none", "pass", "warn", "violated")
GRADE_NAMES = ("NA", "PASS", "WARN", "FAIL")

AUDIT_CLAIM_3D_TRANSLATION = 1 << 0
AUDIT_CLAIM_FIXED_FOCAL = 1 << 1
AUDIT_CLAIM_PLANAR_FLUID = 1 << 2
AUDIT_CLAIM_STATIC_BG = 1 << 3
AUDIT_CLAIM_DIRECT_AUDIO = 1 << 4
AUDIT_CLAIM_CONTINUOUS_CAPTURE = 1 << 5

NA_REASON_BITS = (
    (1 << 0, "NO_EVIDENCE"),
    (1 << 1, "TOO_SHORT"),
    (1 << 2, "CAMERA_MOTION"),
    (1 << 3, "HARD_CUT"),
    (1 << 4, "LOW_TEXTURE"),
    (1 << 5, "AMBIGUOUS"),
    (1 << 6, "NO_AUDIO"),
    (1 << 7, "NO_CALIB"),
    (1 << 8, "DISABLED"),
)

VIOLATION_CODE_NAMES = {
    101: "G_SIGN_FLIP",
    102: "G_ACCEL_DRIFT",
    103: "ODE_LAW_BREAK",
    201: "SUPERELASTIC_RESTITUTION",
    202: "CONSERVATION_VIOLATED",
    301: "FAKE_3D_PURE_YAW",
    302: "STEREO_EPIPOLAR_BREAK",
    303: "TTC_COLLAPSE",
    401: "FLUID_MASS_LEAK",
    501: "FLUID_NONCONSERVATIVE",
    601: "BG_SHIMMER_CURL",
    602: "BG_SOLVER_NONCONV",
    701: "FROZEN_MID_MOTION",
    702: "PTS_CADENCE_BREAK",
    703: "TELEPORT_JUMP",
    801: "AV_IMPACT_LAG",
    803: "AV_RS_CLOCK_SKEW",
}

_FIELD_NAME_MAP = {
    "telemetry": FIELD_TELEMETRY,
    "eca_res": FIELD_ECA_RES,
    "micro_tracks": FIELD_MICRO_TRACKS,
    "continuum": FIELD_CONTINUUM,
    "continuum_field": FIELD_CONTINUUM,
    "rigid_bodies": FIELD_RIGID_BODIES,
    "swarm_telemetry": FIELD_SWARM_TELEMETRY,
    "bos_planes": FIELD_BOS_PLANES,
    "bos_valid": FIELD_BOS_VALID,
    "bos_telemetry": FIELD_BOS_TELEMETRY,
    "hodge_planes": FIELD_HODGE_PLANES,
    "hodge_telemetry": FIELD_HODGE_TELEMETRY,
    "vibro_planes": FIELD_VIBRO_PLANES,
    "vibro_telemetry": FIELD_VIBRO_TELEMETRY,
    "wave1_telemetry": FIELD_WAVE1_TELEMETRY,
    "wave2_telemetry": FIELD_WAVE2_TELEMETRY,
    "depth_telemetry": FIELD_DEPTH_TELEMETRY,
    "depth_field": FIELD_DEPTH_FIELD,
    "depth_tracks": FIELD_DEPTH_TRACKS,
    "depth_flags": FIELD_DEPTH_FLAGS,
    "depth_hero": FIELD_DEPTH_HERO,
    "wind_telemetry": FIELD_WIND_TELEMETRY,
    "wind_planes": FIELD_WIND_PLANES,
    "elasto_telemetry": FIELD_ELASTO_TELEMETRY,
    "elasto_planes": FIELD_ELASTO_PLANES,
    "elasto_valid": FIELD_ELASTO_VALID,
}


class _ChronosFieldView(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
        ("elem_size", ctypes.c_uint32),
        ("elem_kind", ctypes.c_uint32),
        ("rows", ctypes.c_uint32),
        ("cols", ctypes.c_uint32),
        ("stride", ctypes.c_uint32),
    ]


class ChronosIngestMeta(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("pts_us", ctypes.c_uint64),
        ("epoch", ctypes.c_uint32),
        ("seq", ctypes.c_uint32),
        ("age_ms", ctypes.c_float),
        ("src_w", ctypes.c_uint32),
        ("src_h", ctypes.c_uint32),
        ("crop_x", ctypes.c_uint32),
        ("crop_y", ctypes.c_uint32),
        ("crop_w", ctypes.c_uint32),
        ("crop_h", ctypes.c_uint32),
        ("dst_x", ctypes.c_uint32),
        ("dst_y", ctypes.c_uint32),
        ("dst_w", ctypes.c_uint32),
        ("dst_h", ctypes.c_uint32),
        ("orientation", ctypes.c_uint32),
        ("scan_dir", ctypes.c_uint32),
        ("gap_mask", ctypes.c_uint32),
        ("native_1to1", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class ChronosAuditConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("profile_flags", ctypes.c_uint32),
        ("required_axes_mask", ctypes.c_uint32),
        ("expected_frames", ctypes.c_uint32),
        ("min_applicable_frames", ctypes.c_uint32),
        ("span_merge_gap_frames", ctypes.c_uint32),
        ("fail_severity", ctypes.c_float),
        ("min_coverage", ctypes.c_float),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class ChronosAuditAxis(ctypes.Structure):
    _fields_ = [
        ("axis_id", ctypes.c_uint32),
        ("applicability", ctypes.c_uint32),
        ("verdict", ctypes.c_uint32),
        ("na_reason_mask", ctypes.c_uint32),
        ("applicable_frames", ctypes.c_uint32),
        ("warned_frames", ctypes.c_uint32),
        ("violated_frames", ctypes.c_uint32),
        ("span_count", ctypes.c_uint32),
        ("score", ctypes.c_float),
        ("worst_residual", ctypes.c_float),
    ]


class ChronosAuditSpan(ctypes.Structure):
    _fields_ = [
        ("axis_id", ctypes.c_uint32),
        ("code", ctypes.c_uint32),
        ("f0", ctypes.c_uint32),
        ("f1", ctypes.c_uint32),
        ("t0_us", ctypes.c_uint64),
        ("t1_us", ctypes.c_uint64),
        ("track_id", ctypes.c_int32),
        ("severity", ctypes.c_float),
        ("bbox", ctypes.c_float * 4),
        ("measured", ctypes.c_float),
        ("expected", ctypes.c_float),
    ]


class ChronosAuditSummary(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("schema_version", ctypes.c_uint32),
        ("profile_flags", ctypes.c_uint32),
        ("required_axes_mask", ctypes.c_uint32),
        ("known_modalities", ctypes.c_uint32),
        ("present_modalities", ctypes.c_uint32),
        ("frames_seen", ctypes.c_uint32),
        ("frames_committed", ctypes.c_uint32),
        ("frames_rejected", ctypes.c_uint32),
        ("shots", ctypes.c_uint32),
        ("audio_blocks", ctypes.c_uint32),
        ("audio_underruns", ctypes.c_uint32),
        ("axis_count", ctypes.c_uint32),
        ("applicable_axes", ctypes.c_uint32),
        ("violated_axes", ctypes.c_uint32),
        ("warned_axes", ctypes.c_uint32),
        ("span_count", ctypes.c_uint32),
        ("spans_dropped", ctypes.c_uint32),
        ("overall_grade", ctypes.c_uint32),
        ("coverage_permille", ctypes.c_uint32),
        ("reward_milli", ctypes.c_int32),
        ("integrity", ctypes.c_float),
        ("reserved", ctypes.c_uint32),
        ("axes", ChronosAuditAxis * CHRONOS_AUDIT_AXIS_COUNT),
    ]


# Import-time ABI layout verification against `native/include/chronos.h`
assert ctypes.sizeof(ChronosIngestMeta) == 96, f"ChronosIngestMeta={ctypes.sizeof(ChronosIngestMeta)}"
assert ctypes.sizeof(ChronosAuditConfig) == 40, f"ChronosAuditConfig={ctypes.sizeof(ChronosAuditConfig)}"
assert ctypes.sizeof(ChronosAuditAxis) == 40, f"ChronosAuditAxis={ctypes.sizeof(ChronosAuditAxis)}"
assert ctypes.sizeof(ChronosAuditSpan) == 64, f"ChronosAuditSpan={ctypes.sizeof(ChronosAuditSpan)}"
assert ctypes.sizeof(ChronosAuditSummary) == 416, f"ChronosAuditSummary={ctypes.sizeof(ChronosAuditSummary)}"


def decode_na_reasons(mask: int) -> List[str]:
    return [name for bit, name in NA_REASON_BITS if (mask & bit) != 0]


def _default_lib_path() -> Path:
    env_p = os.environ.get("CHRONOS_CAPI_LIB")
    if env_p:
        return Path(env_p)
    native_root = Path(__file__).resolve().parents[1]
    ext = {"Darwin": "dylib", "Windows": "dll"}.get(platform.system(), "so")
    prefix = "" if platform.system() == "Windows" else "lib"
    fname = f"{prefix}chronos_capi.{ext}"
    candidates = []
    for base in (native_root / "target", native_root / "capi" / "target"):
        for sub in (
            "release",
            "aarch64-apple-darwin/release",
            "x86_64-apple-darwin/release",
            "x86_64-unknown-linux-gnu/release",
            "aarch64-unknown-linux-gnu/release",
            "debug",
        ):
            candidates.append(base / sub / fname)
    for cand in candidates:
        if cand.exists():
            return cand
    return native_root / "target" / "release" / fname


_LIB_CACHE: Dict[str, ctypes.CDLL] = {}


def _load_lib(lib_path: Optional[os.PathLike | str] = None) -> ctypes.CDLL:
    path = Path(lib_path) if lib_path is not None else _default_lib_path()
    key = str(path.resolve()) if path.exists() else str(path)
    if key in _LIB_CACHE:
        return _LIB_CACHE[key]
    if not path.exists():
        raise FileNotFoundError(
            f"chronos_capi library not found at {path}. "
            "Build it with: cargo build --release --manifest-path native/capi/Cargo.toml "
            "--target host-tuple --target-dir native/target"
        )
    lib = ctypes.CDLL(str(path))
    lib.chronos_abi_version.restype = ctypes.c_uint32
    lib.chronos_version_string.restype = ctypes.c_char_p
    lib.chronos_field_layout.argtypes = [ctypes.c_uint32]
    lib.chronos_field_layout.restype = ctypes.c_uint32

    abi = int(lib.chronos_abi_version())
    if abi != CHRONOS_ABI_VERSION:
        raise RuntimeError(
            f"chronos_capi ABI mismatch at {path}: got ABI {abi}, expected {CHRONOS_ABI_VERSION}"
        )

    lib.chronos_engine_create.restype = ctypes.c_void_p
    lib.chronos_engine_reset.argtypes = [ctypes.c_void_p]
    lib.chronos_engine_destroy.argtypes = [ctypes.c_void_p]
    lib.chronos_engine_set_resolution.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.chronos_engine_set_resolution_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.chronos_engine_set_params.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint32,
    ]
    lib.chronos_engine_set_swarm_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.chronos_engine_set_bos_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.chronos_engine_set_bos_mode.restype = ctypes.c_uint32
    lib.chronos_engine_set_bos_gauge_roi.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.chronos_engine_set_bos_gauge_roi.restype = ctypes.c_uint32
    lib.chronos_engine_set_micro_metrology_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.chronos_engine_set_micro_metrology_mode.restype = ctypes.c_uint32
    lib.chronos_engine_set_rs_calib.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.chronos_engine_set_rs_calib.restype = ctypes.c_uint32
    lib.chronos_engine_set_hodge_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.chronos_engine_set_hodge_mode.restype = ctypes.c_uint32
    lib.chronos_engine_set_wind_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.chronos_engine_set_wind_mode.restype = ctypes.c_uint32
    lib.chronos_engine_set_wind_fluid_density.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.chronos_engine_set_wind_fluid_density.restype = ctypes.c_uint32
    lib.chronos_engine_set_elasto_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.chronos_engine_set_elasto_mode.restype = ctypes.c_uint32
    lib.chronos_engine_set_elasto_lock_freqs.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.chronos_engine_set_elasto_lock_freqs.restype = ctypes.c_uint32
    lib.chronos_engine_set_depth_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.chronos_engine_set_depth_mode.restype = ctypes.c_uint32
    lib.chronos_engine_set_depth_calib.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int32,
    ]
    lib.chronos_engine_set_depth_calib.restype = ctypes.c_uint32
    lib.chronos_engine_depth_capture_stereo_left.argtypes = [ctypes.c_void_p]
    lib.chronos_engine_depth_capture_stereo_left.restype = ctypes.c_uint32
    lib.chronos_engine_set_depth_external_meta.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.chronos_engine_set_depth_external_meta.restype = ctypes.c_uint32
    lib.chronos_engine_depth_ingest_mut_ptr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.chronos_engine_depth_ingest_mut_ptr.restype = ctypes.POINTER(ctypes.c_float)

    lib.chronos_engine_audio_staging_mut_ptr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.chronos_engine_audio_staging_mut_ptr.restype = ctypes.POINTER(ctypes.c_float)
    lib.chronos_engine_push_audio_block.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.chronos_engine_push_audio_block.restype = ctypes.c_uint32
    lib.chronos_engine_push_audio.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_float,
    ]
    lib.chronos_engine_push_audio.restype = ctypes.c_int32
    lib.chronos_engine_set_ingest_meta.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ChronosIngestMeta),
    ]
    lib.chronos_engine_set_ingest_meta.restype = ctypes.c_int32
    lib.chronos_engine_ingest_rgba_strided.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ChronosIngestMeta),
    ]
    lib.chronos_engine_ingest_rgba_strided.restype = ctypes.c_int32

    lib.chronos_engine_seed_target.argtypes = [
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.chronos_engine_seed_target.restype = ctypes.c_uint32
    lib.chronos_engine_reset_tracks.argtypes = [ctypes.c_void_p]
    lib.chronos_engine_frame_w.argtypes = [ctypes.c_void_p]
    lib.chronos_engine_frame_w.restype = ctypes.c_uint32
    lib.chronos_engine_frame_h.argtypes = [ctypes.c_void_p]
    lib.chronos_engine_frame_h.restype = ctypes.c_uint32
    lib.chronos_engine_ingest_rgba.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.chronos_engine_ingest_rgba.restype = ctypes.c_int32
    lib.chronos_engine_ingest_rgba_mut_ptr.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.chronos_engine_ingest_rgba_mut_ptr.restype = ctypes.POINTER(ctypes.c_uint8)
    lib.chronos_engine_step.argtypes = [ctypes.c_void_p, ctypes.c_float]
    lib.chronos_engine_step.restype = ctypes.c_uint32
    lib.chronos_engine_step_with_pts.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_float,
    ]
    lib.chronos_engine_step_with_pts.restype = ctypes.c_uint32
    lib.chronos_engine_telemetry_ptr.argtypes = [ctypes.c_void_p]
    lib.chronos_engine_telemetry_ptr.restype = ctypes.POINTER(ctypes.c_float)
    lib.chronos_engine_telemetry_len.restype = ctypes.c_size_t
    lib.chronos_engine_field_view.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.chronos_engine_field_view.restype = _ChronosFieldView

    lib.chronos_engine_audit_begin.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ChronosAuditConfig),
    ]
    lib.chronos_engine_audit_begin.restype = ctypes.c_int32
    lib.chronos_engine_audit_end.argtypes = [ctypes.c_void_p]
    lib.chronos_engine_audit_finalize.argtypes = [ctypes.c_void_p]
    lib.chronos_engine_audit_finalize.restype = ctypes.c_int32
    lib.chronos_engine_audit_summary.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ChronosAuditSummary),
    ]
    lib.chronos_engine_audit_summary.restype = ctypes.c_int32
    lib.chronos_engine_audit_spans.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ChronosAuditSpan),
        ctypes.c_size_t,
    ]
    lib.chronos_engine_audit_spans.restype = ctypes.c_size_t

    _LIB_CACHE[key] = lib
    return lib


def _try_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError:
        return None


def _qfloat(x: float) -> Optional[float]:
    if not math.isfinite(x):
        return None
    return float(format(x, ".6g"))


class ChronosEngine:
    """Multi-instance handle around native `libchronos_capi` (ABI 4)."""

    def __init__(
        self,
        lib_path: Optional[os.PathLike | str] = None,
        *,
        hd: bool = False,
        mode: Optional[int] = None,
    ) -> None:
        lib = _load_lib(lib_path)
        self._lib = lib
        self._handle = lib.chronos_engine_create()
        if not self._handle:
            raise MemoryError("chronos_engine_create returned NULL")
        self._telem_len = int(lib.chronos_engine_telemetry_len())
        self._last_dt_sec = 1.0 / 30.0
        if mode is not None:
            self.set_resolution_mode(mode)
        elif hd:
            self.set_resolution(True)

    @property
    def abi_version(self) -> int:
        return int(self._lib.chronos_abi_version())

    @property
    def version_string(self) -> str:
        raw = self._lib.chronos_version_string()
        return raw.decode("utf-8") if raw else "3.4.0"

    def reset(self) -> None:
        """Reset engine in place to a state bit-identical to a fresh `ChronosEngine()`."""
        self._lib.chronos_engine_reset(self._handle)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.chronos_engine_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "ChronosEngine":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @property
    def frame_dims(self) -> Tuple[int, int]:
        return (
            int(self._lib.chronos_engine_frame_w(self._handle)),
            int(self._lib.chronos_engine_frame_h(self._handle)),
        )

    def set_resolution(self, hd: bool) -> None:
        self._lib.chronos_engine_set_resolution(self._handle, 1 if hd else 0)

    def set_resolution_mode(self, mode: int) -> None:
        """0 = 320x180 SD, 1 = 640x360 HD, 2 = 1280x720 (requires hires-720p build)."""
        self._lib.chronos_engine_set_resolution_mode(self._handle, int(mode))

    def set_params(
        self,
        *,
        lookahead_k: int = 8,
        eca_adapt_rate: float = 0.035,
        fista_lambda: float = 1.45,
        eulerian_gain: float = 25.0,
        phase_offset_rad: float = -1.5707963,
        lab_mode: int = 0,
    ) -> None:
        self._lib.chronos_engine_set_params(
            self._handle,
            int(lookahead_k),
            float(eca_adapt_rate),
            float(fista_lambda),
            float(eulerian_gain),
            float(phase_offset_rad),
            int(lab_mode),
        )

    def set_swarm_mode(self, mode: int) -> None:
        self._lib.chronos_engine_set_swarm_mode(self._handle, int(mode))

    def set_bos_mode(self, mode: int) -> int:
        return int(self._lib.chronos_engine_set_bos_mode(self._handle, int(mode)))

    def set_bos_gauge_roi(self, x0: int, y0: int, x1: int, y1: int) -> int:
        return int(
            self._lib.chronos_engine_set_bos_gauge_roi(
                self._handle, int(x0), int(y0), int(x1), int(y1)
            )
        )

    def set_micro_metrology_mode(self, mode: int) -> int:
        return int(self._lib.chronos_engine_set_micro_metrology_mode(self._handle, int(mode)))

    def set_rs_calib(
        self,
        mode: int,
        dt_row_us: float,
        exposure_us: float,
        audio_rate_hz: float = 48000.0,
        max_skew_us: float = 500.0,
    ) -> int:
        return int(
            self._lib.chronos_engine_set_rs_calib(
                self._handle,
                int(mode),
                float(dt_row_us),
                float(exposure_us),
                float(audio_rate_hz),
                float(max_skew_us),
            )
        )

    def set_hodge_mode(self, mode: int) -> int:
        return int(self._lib.chronos_engine_set_hodge_mode(self._handle, int(mode)))

    def set_wind_mode(self, mode: int) -> int:
        """0 = Off, 1 = Staggered PIV, 2 = Every-Frame PIV, 3 = Dense Optical-Flow Audit."""
        return int(self._lib.chronos_engine_set_wind_mode(self._handle, int(mode)))

    def set_wind_fluid_density(self, rho_kg_m3: float) -> int:
        return int(self._lib.chronos_engine_set_wind_fluid_density(self._handle, float(rho_kg_m3)))

    def set_elasto_mode(self, mode: int) -> int:
        return int(self._lib.chronos_engine_set_elasto_mode(self._handle, int(mode)))

    def set_elasto_lock_freqs(self, *freqs: float) -> int:
        if len(freqs) == 1 and isinstance(freqs[0], (list, tuple)):
            freqs = tuple(freqs[0])
        if len(freqs) > 4:
            raise ValueError("set_elasto_lock_freqs accepts at most 4 frequencies")
        vals = [float(f) for f in freqs] + [0.0] * (4 - len(freqs))
        return int(
            self._lib.chronos_engine_set_elasto_lock_freqs(
                self._handle, len(freqs), vals[0], vals[1], vals[2], vals[3]
            )
        )

    def set_depth_mode(self, mode: int) -> int:
        return int(self._lib.chronos_engine_set_depth_mode(self._handle, int(mode)))

    def set_depth_calib(
        self,
        f_px: float,
        speed_m_s: float = 0.0,
        baseline_m: float = 0.0,
        max_disp_px: float = 32.0,
        stereo_sign: int = 0,
    ) -> int:
        return int(
            self._lib.chronos_engine_set_depth_calib(
                self._handle,
                float(f_px),
                float(speed_m_s),
                float(baseline_m),
                float(max_disp_px),
                int(stereo_sign),
            )
        )

    def capture_stereo_left(self) -> int:
        return int(self._lib.chronos_engine_depth_capture_stereo_left(self._handle))

    def set_depth_external_meta(
        self,
        w: int,
        h: int,
        f_px: float = 0.0,
        provenance: int = 1,
        seq: int = 1,
    ) -> int:
        return int(
            self._lib.chronos_engine_set_depth_external_meta(
                self._handle, int(w), int(h), float(f_px), int(provenance), int(seq)
            )
        )

    def depth_ingest_buffer(self, w: Optional[int] = None, h: Optional[int] = None):
        out_len = ctypes.c_size_t(0)
        ptr = self._lib.chronos_engine_depth_ingest_mut_ptr(self._handle, ctypes.byref(out_len))
        if not ptr or out_len.value == 0:
            raise RuntimeError("chronos_engine_depth_ingest_mut_ptr returned NULL")
        raw = (ctypes.c_float * out_len.value).from_address(ctypes.addressof(ptr.contents))
        raw._owner = self
        np = _try_numpy()
        if np is not None:
            arr = np.ctypeslib.as_array(raw)
            if w is not None and h is not None:
                wi, hi = int(w), int(h)
                if wi <= 0 or hi <= 0 or wi * hi > int(out_len.value):
                    raise ValueError(f"Requested shape ({hi}, {wi}) exceeds capacity {int(out_len.value)}")
                return arr[: wi * hi].reshape((hi, wi))
            return arr
        raw_b = memoryview(raw).cast("B")
        if w is not None and h is not None:
            wi, hi = int(w), int(h)
            if wi <= 0 or hi <= 0 or wi * hi > int(out_len.value):
                raise ValueError(f"Requested shape ({hi}, {wi}) exceeds capacity {int(out_len.value)}")
            return raw_b[: wi * hi * 4].cast("f")
        return raw_b.cast("f")

    def audio_staging_buffer(self, *, raw_bytes: bool = False):
        """Zero-copy writable view into the 4,096-float mono audio staging buffer."""
        out_len = ctypes.c_size_t(0)
        ptr = self._lib.chronos_engine_audio_staging_mut_ptr(self._handle, ctypes.byref(out_len))
        if not ptr or out_len.value == 0:
            raise RuntimeError("chronos_engine_audio_staging_mut_ptr returned NULL")
        raw = (ctypes.c_float * out_len.value).from_address(ctypes.addressof(ptr.contents))
        raw._owner = self
        if raw_bytes:
            return memoryview(raw).cast("B")
        np = _try_numpy()
        if np is not None:
            return np.ctypeslib.as_array(raw)
        return memoryview(raw).cast("B").cast("f")

    def push_audio_block(
        self,
        num_samples: int,
        t_end_us: int,
        sample_rate_hz: float = 48000.0,
        est_skew_us: float = 0.0,
    ) -> int:
        return int(
            self._lib.chronos_engine_push_audio_block(
                self._handle,
                int(num_samples),
                int(t_end_us),
                float(sample_rate_hz),
                float(est_skew_us),
            )
        )

    def push_audio(
        self,
        samples: object,
        t_end_us: int,
        sample_rate_hz: float = 48000.0,
    ) -> int:
        if hasattr(samples, "ctypes") and hasattr(samples, "size"):
            arr = samples
            if str(getattr(arr, "dtype", "")) != "float32" or not arr.flags.c_contiguous:
                import numpy as np  # type: ignore

                arr = np.ascontiguousarray(arr, dtype=np.float32)
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            n = int(arr.size)
        else:
            mv = memoryview(samples)
            if mv.format == "f" and mv.c_contiguous:
                n = len(mv)
                arr_t = (ctypes.c_float * n).from_buffer_copy(mv)
            else:
                seq = list(samples)  # type: ignore[arg-type]
                n = len(seq)
                arr_t = (ctypes.c_float * n)(*seq)
            ptr = ctypes.cast(arr_t, ctypes.POINTER(ctypes.c_float))
        return int(
            self._lib.chronos_engine_push_audio(
                self._handle, ptr, n, int(t_end_us), float(sample_rate_hz)
            )
        )

    def set_ingest_meta(self, meta: ChronosIngestMeta) -> int:
        if meta.struct_size == 0:
            meta.struct_size = ctypes.sizeof(ChronosIngestMeta)
        return int(self._lib.chronos_engine_set_ingest_meta(self._handle, ctypes.byref(meta)))

    def ingest_rgba_strided(
        self,
        rgba: object,
        src_w: int,
        src_h: int,
        src_stride_bytes: int = 0,
        fit: int = 0,
        meta: Optional[ChronosIngestMeta] = None,
    ) -> int:
        if hasattr(rgba, "ctypes") and hasattr(rgba, "nbytes"):
            arr = rgba
            if not getattr(arr, "flags", None) or not arr.flags.c_contiguous:
                arr = arr.copy(order="C")
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        else:
            mv = memoryview(rgba).cast("B")
            if mv.readonly:
                arr_t = (ctypes.c_uint8 * len(mv)).from_buffer_copy(mv)
            else:
                arr_t = (ctypes.c_uint8 * len(mv)).from_buffer(mv)
            ptr = ctypes.cast(arr_t, ctypes.POINTER(ctypes.c_uint8))
        m_ptr = ctypes.byref(meta) if meta is not None else None
        return int(
            self._lib.chronos_engine_ingest_rgba_strided(
                self._handle,
                ptr,
                int(src_w),
                int(src_h),
                int(src_stride_bytes),
                int(fit),
                m_ptr,
            )
        )

    def seed_target(self, x: float, y: float, w: float = 0.0, h: float = 0.0) -> int:
        return int(self._lib.chronos_engine_seed_target(self._handle, float(x), float(y), float(w), float(h)))

    def reset_tracks(self) -> None:
        self._lib.chronos_engine_reset_tracks(self._handle)

    def ingest_buffer(self, *, raw: bool = False):
        """Return a zero-copy writable view into the next RGBA8 ring slot.

        When `raw=True`, always returns a flat 1D writable `memoryview` of bytes (ideal for
        `file.readinto(buf)` zero-copy streaming without NumPy). Otherwise returns a writable
        NumPy `(H, W, 4)` uint8 array when NumPy is installed, or a byte `memoryview`.
        """
        out_len = ctypes.c_size_t(0)
        ptr = self._lib.chronos_engine_ingest_rgba_mut_ptr(self._handle, ctypes.byref(out_len))
        if not ptr or out_len.value == 0:
            raise RuntimeError("chronos_engine_ingest_rgba_mut_ptr returned NULL")
        raw_arr = (ctypes.c_uint8 * out_len.value).from_address(ctypes.addressof(ptr.contents))
        raw_arr._owner = self
        if raw:
            return memoryview(raw_arr).cast("B")
        np = _try_numpy()
        if np is not None:
            w, h = self.frame_dims
            return np.ctypeslib.as_array(raw_arr).reshape((h, w, 4))
        return memoryview(raw_arr).cast("B")

    def step_in_place(self, dt_sec: float = 1.0 / 30.0, pts_sec: Optional[float] = None) -> int:
        self._last_dt_sec = float(dt_sec)
        if pts_sec is not None:
            return int(self._lib.chronos_engine_step_with_pts(self._handle, float(pts_sec), float(dt_sec)))
        return int(self._lib.chronos_engine_step(self._handle, float(dt_sec)))

    def step_rgba(self, rgba_frame: object, dt_sec: float = 1.0 / 30.0, pts_sec: Optional[float] = None) -> int:
        self._last_dt_sec = float(dt_sec)
        if hasattr(rgba_frame, "ctypes") and hasattr(rgba_frame, "nbytes"):
            arr = rgba_frame
            if not getattr(arr, "flags", None) or not arr.flags.c_contiguous:
                arr = arr.copy(order="C")
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            nbytes = int(arr.nbytes)
        else:
            mv = memoryview(rgba_frame).cast("B")
            nbytes = len(mv)
            if mv.readonly:
                arr_t = (ctypes.c_uint8 * nbytes).from_buffer_copy(mv)
            else:
                arr_t = (ctypes.c_uint8 * nbytes).from_buffer(mv)
            ptr = ctypes.cast(arr_t, ctypes.POINTER(ctypes.c_uint8))

        rc = int(self._lib.chronos_engine_ingest_rgba(self._handle, ptr, nbytes))
        if rc != 0:
            w, h = self.frame_dims
            raise ValueError(f"chronos_engine_ingest_rgba failed (rc={rc}); expected {w * h * 4} bytes, got {nbytes}")
        if pts_sec is not None:
            return int(self._lib.chronos_engine_step_with_pts(self._handle, float(pts_sec), float(dt_sec)))
        return int(self._lib.chronos_engine_step(self._handle, float(dt_sec)))

    def field_view(self, kind: int | str):
        kind_id = _FIELD_NAME_MAP[kind] if isinstance(kind, str) else int(kind)
        view = self._lib.chronos_engine_field_view(self._handle, kind_id)
        if not view.ptr or view.len == 0:
            raise ValueError(f"Invalid or empty field view kind={kind!r}")
        ctype_map = {1: ctypes.c_uint8, 2: ctypes.c_uint32, 3: ctypes.c_float}
        ct = ctype_map.get(int(view.elem_kind))
        if ct is None:
            raise ValueError(f"Unknown elem_kind={view.elem_kind}")
        raw = (ct * int(view.len)).from_address(int(view.ptr))
        raw._owner = self
        np = _try_numpy()
        if np is None:
            fmt_map = {1: "B", 2: "I", 3: "f"}
            return memoryview(raw).cast("B").cast(fmt_map[int(view.elem_kind)]).toreadonly()
        arr = np.ctypeslib.as_array(raw)
        rows, cols, stride = int(view.rows), int(view.cols), int(view.stride)
        layout = int(self._lib.chronos_field_layout(kind_id))
        if layout == 2:
            arr = arr.reshape((stride, rows, cols))
        elif layout == 3:
            arr = arr.reshape((rows, cols, stride))
        elif layout == 1 and cols > 1:
            arr = arr.reshape((rows, cols))
        arr.setflags(write=False)
        return arr

    def telemetry_view(self):
        return self.field_view(FIELD_TELEMETRY)

    def telemetry(self) -> List[float]:
        ptr = self._lib.chronos_engine_telemetry_ptr(self._handle)
        return [float(ptr[i]) for i in range(self._telem_len)]

    def active_tracks(self) -> List[Dict[str, float]]:
        ptr = self._lib.chronos_engine_telemetry_ptr(self._handle)
        dt = self._last_dt_sec
        tracks: List[Dict[str, float]] = []
        for m in range(16):
            base = 32 + m * 56
            if ptr[base] > 0.5:
                kbase = 1120 + m * 8
                pbase = 1280 + m * 16
                heading_rad = math.radians(float(ptr[kbase]))
                speed_px_s = float(ptr[kbase + 1])
                vx_s = math.cos(heading_rad) * speed_px_s
                vy_s = math.sin(heading_rad) * speed_px_s
                tracks.append(
                    {
                        "slot": m,
                        "id": int(round(ptr[base + 1])),
                        "x": float(ptr[base + 2]),
                        "y": float(ptr[base + 3]),
                        "vx": vx_s,
                        "vy": vy_s,
                        "vx_px_f": vx_s * dt,
                        "vy_px_f": vy_s * dt,
                        "vis": float(ptr[base + 10]),
                        "bridged": float(ptr[base + 11]),
                        "regime": int(round(ptr[pbase + 4])),
                        "ode_r2": float(ptr[pbase + 6]),
                    }
                )
        return tracks

    # ---- v3.4.0 Physics Integrity Audit API ----
    def audit_begin(
        self,
        *,
        profile_flags: int = 0,
        required_axes_mask: int = 0,
        expected_frames: int = 0,
        min_applicable_frames: int = 8,
        span_merge_gap_frames: int = 4,
        fail_severity: float = 0.5,
        min_coverage: float = 0.15,
    ) -> int:
        cfg = ChronosAuditConfig(
            struct_size=ctypes.sizeof(ChronosAuditConfig),
            profile_flags=int(profile_flags),
            required_axes_mask=int(required_axes_mask),
            expected_frames=int(expected_frames),
            min_applicable_frames=int(min_applicable_frames),
            span_merge_gap_frames=int(span_merge_gap_frames),
            fail_severity=float(fail_severity),
            min_coverage=float(min_coverage),
        )
        return int(self._lib.chronos_engine_audit_begin(self._handle, ctypes.byref(cfg)))

    def audit_end(self) -> None:
        self._lib.chronos_engine_audit_end(self._handle)

    def audit_finalize(self) -> int:
        return int(self._lib.chronos_engine_audit_finalize(self._handle))

    def audit_summary_struct(self) -> ChronosAuditSummary:
        out = ChronosAuditSummary()
        rc = int(self._lib.chronos_engine_audit_summary(self._handle, ctypes.byref(out)))
        if rc != 0:
            raise RuntimeError(f"chronos_engine_audit_summary failed (rc={rc})")
        return out

    def audit_summary(self) -> Dict[str, Any]:
        s = self.audit_summary_struct()
        axes_out = []
        for i in range(int(s.axis_count)):
            ax = s.axes[i]
            appl = int(ax.applicability)
            verd = int(ax.verdict)
            eligible = appl == 3
            axes_out.append(
                {
                    "id": AXIS_SLUGS[i] if i < len(AXIS_SLUGS) else f"axis_{i}",
                    "axis_id": int(ax.axis_id),
                    "applicability": APPL_NAMES[appl] if appl < len(APPL_NAMES) else "unknown",
                    "applicable": eligible,
                    "verdict": VERDICT_NAMES[verd] if verd < len(VERDICT_NAMES) else "none",
                    "pass": (verd == 1) if eligible else None,
                    "na_reasons": decode_na_reasons(int(ax.na_reason_mask)),
                    "applicable_frames": int(ax.applicable_frames),
                    "warned_frames": int(ax.warned_frames),
                    "violated_frames": int(ax.violated_frames),
                    "span_count": int(ax.span_count),
                    "score": _qfloat(float(ax.score)) if eligible and ax.score >= 0.0 else None,
                    "worst_residual": _qfloat(float(ax.worst_residual)) if ax.worst_residual >= 0.0 else None,
                }
            )
        grade_i = int(s.overall_grade)
        reward_i = int(s.reward_milli)
        return {
            "abi_version": int(s.abi_version),
            "schema_version": int(s.schema_version),
            "profile_flags": int(s.profile_flags),
            "required_axes_mask": int(s.required_axes_mask),
            "known_modalities": int(s.known_modalities),
            "present_modalities": int(s.present_modalities),
            "frames_seen": int(s.frames_seen),
            "frames_committed": int(s.frames_committed),
            "frames_rejected": int(s.frames_rejected),
            "shots": int(s.shots),
            "audio_blocks": int(s.audio_blocks),
            "audio_underruns": int(s.audio_underruns),
            "applicable_axes": int(s.applicable_axes),
            "violated_axes": int(s.violated_axes),
            "warned_axes": int(s.warned_axes),
            "span_count": int(s.span_count),
            "spans_dropped": int(s.spans_dropped),
            "grade": GRADE_NAMES[grade_i] if grade_i < len(GRADE_NAMES) else "NA",
            "coverage": _qfloat(float(s.coverage_permille) / 1000.0),
            "integrity": _qfloat(float(s.integrity)) if s.applicable_axes > 0 and s.integrity >= 0.0 else None,
            "reward_milli": None if reward_i == -2147483648 else reward_i,
            "reward": None if reward_i == -2147483648 else _qfloat(reward_i / 1000.0),
            "axes": axes_out,
        }

    def audit_spans(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        buf = (ChronosAuditSpan * 128)()
        start = 0
        while True:
            got = int(self._lib.chronos_engine_audit_spans(self._handle, start, buf, 128))
            if got <= 0:
                break
            for k in range(got):
                sp = buf[k]
                aid = int(sp.axis_id)
                code = int(sp.code)
                out.append(
                    {
                        "axis": AXIS_SLUGS[aid] if aid < len(AXIS_SLUGS) else f"axis_{aid}",
                        "axis_id": aid,
                        "code": VIOLATION_CODE_NAMES.get(code, f"CODE_{code}"),
                        "code_id": code,
                        "f0": int(sp.f0),
                        "f1": int(sp.f1),
                        "t0_us": int(sp.t0_us),
                        "t1_us": int(sp.t1_us),
                        "track_id": int(sp.track_id),
                        "severity": _qfloat(float(sp.severity)),
                        "bbox": [_qfloat(float(sp.bbox[j])) for j in range(4)],
                        "measured": _qfloat(float(sp.measured)),
                        "expected": _qfloat(float(sp.expected)),
                    }
                )
            start += got
            if got < 128:
                break
        return out
