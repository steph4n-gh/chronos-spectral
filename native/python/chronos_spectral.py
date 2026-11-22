"""Zero-dependency Python ctypes wrapper for the native multi-instance CHRONOS-SPECTRAL engine.

Supports zero-copy writable RGBA8 ring-buffer views (`ingest_buffer()` + `step_in_place()`)
and zero-copy read-only NumPy field views (`field_view()`, `telemetry_view()`) with no hard
NumPy dependency.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def _default_lib_path() -> Path:
    root = Path(__file__).resolve().parents[1] / "capi" / "target"
    ext = {"Darwin": "dylib", "Windows": "dll"}.get(platform.system(), "so")
    prefix = "" if platform.system() == "Windows" else "lib"
    fname = f"{prefix}chronos_capi.{ext}"
    for cand in [
        root / "release" / fname,
        root / "aarch64-apple-darwin" / "release" / fname,
        root / "x86_64-apple-darwin" / "release" / fname,
        root / "x86_64-unknown-linux-gnu" / "release" / fname,
        root / "debug" / fname,
    ]:
        if cand.exists():
            return cand
    return root / "release" / fname


def _try_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError:
        return None


class ChronosEngine:
    """Multi-instance handle around native `libchronos_capi`."""

    def __init__(
        self,
        lib_path: Optional[os.PathLike | str] = None,
        *,
        hd: bool = False,
        mode: Optional[int] = None,
    ) -> None:
        path = Path(lib_path) if lib_path is not None else _default_lib_path()
        if not path.exists():
            raise FileNotFoundError(
                f"chronos_capi library not found at {path}. "
                "Build it with: cargo build --release --manifest-path native/capi/Cargo.toml --target host-tuple"
            )
        lib = ctypes.CDLL(str(path))
        lib.chronos_engine_create.restype = ctypes.c_void_p
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
        return int(self._lib.chronos_engine_set_bos_gauge_roi(self._handle, int(x0), int(y0), int(x1), int(y1)))

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
        """0 = Off, 1 = Staggered Pocket Wind Tunnel, 2 = Every-Frame Pocket Wind Tunnel."""
        return int(self._lib.chronos_engine_set_wind_mode(self._handle, int(mode)))

    def set_wind_fluid_density(self, rho_kg_m3: float) -> int:
        """Fluid density in kg/m^3 (0.0 = dimensionless Cp/CL/CD/Re only)."""
        return int(self._lib.chronos_engine_set_wind_fluid_density(self._handle, float(rho_kg_m3)))

    def set_elasto_mode(self, mode: int) -> int:
        """0 = Off, 1 = RS Audio-Locked, 2 = RS Self-Locked, 3 = Frame-Rate Surface, 4 = Acoustic BOS."""
        return int(self._lib.chronos_engine_set_elasto_mode(self._handle, int(mode)))

    def set_elasto_lock_freqs(self, *freqs: float) -> int:
        """Set 0..4 explicit lock-in frequencies in Hz (0 clears explicit overrides)."""
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
        """0 = Off, 1 = Mono Parallax+TTC, 2 = Calibrated Stereo BOS, 3 = External Depth Plane, 4 = Auto-Fused."""
        return int(self._lib.chronos_engine_set_depth_mode(self._handle, int(mode)))

    def set_depth_calib(
        self,
        f_px: float,
        speed_m_s: float = 0.0,
        baseline_m: float = 0.0,
        max_disp_px: float = 32.0,
        stereo_sign: int = 0,
    ) -> int:
        """Configure depth calibration with strictly separated mono speed (m/s) and stereo baseline (m)."""
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
        """Capture the RGBA8 frame staged in `ingest_buffer()` as a single-use Left rectified stereo reference."""
        return int(self._lib.chronos_engine_depth_capture_stereo_left(self._handle))

    def set_depth_external_meta(
        self,
        w: int,
        h: int,
        f_px: float = 0.0,
        provenance: int = 1,
        seq: int = 1,
    ) -> int:
        """Stage single-frame external depth metadata. provenance: 1=SURVEYED_LIDAR, 2=SURVEYED_STEREO, 3=ASSUMED_NEURAL_PRIOR."""
        return int(
            self._lib.chronos_engine_set_depth_external_meta(
                self._handle,
                int(w),
                int(h),
                float(f_px),
                int(provenance),
                int(seq),
            )
        )

    def depth_ingest_buffer(self, w: Optional[int] = None, h: Optional[int] = None):
        """Return a zero-copy writable view into the 8,192-float external depth-plane staging buffer.

        Returns a writable NumPy `float32` array shaped `(h, w)` (or `(8192,)`) when NumPy is installed,
        or a flat 1D writable native-endian `'f'` `memoryview` of length `w * h` (or `8192`) otherwise
        so both scalar indexing and row/slice assignment work in CPython without NumPy.
        """
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
                    raise ValueError(f"Requested shape ({hi}, {wi}) exceeds depth ingest capacity {int(out_len.value)}")
                return arr[: wi * hi].reshape((hi, wi))
            return arr
        raw_b = memoryview(raw).cast("B")
        if w is not None and h is not None:
            wi, hi = int(w), int(h)
            if wi <= 0 or hi <= 0 or wi * hi > int(out_len.value):
                raise ValueError(f"Requested shape ({hi}, {wi}) exceeds depth ingest capacity {int(out_len.value)}")
            return raw_b[: wi * hi * 4].cast("f")
        return raw_b.cast("f")

    def seed_target(self, x: float, y: float, w: float = 0.0, h: float = 0.0) -> int:
        return int(self._lib.chronos_engine_seed_target(self._handle, float(x), float(y), float(w), float(h)))

    def reset_tracks(self) -> None:
        self._lib.chronos_engine_reset_tracks(self._handle)

    def ingest_buffer(self):
        """Return a zero-copy writable view into the next RGBA8 ring slot.

        Returns a writable NumPy `uint8` array shaped `(H, W, 4)` when NumPy is installed,
        or a writable `memoryview` of length `H * W * 4` otherwise. Write frame pixels directly
        into this view and call `step_in_place()`.
        """
        out_len = ctypes.c_size_t(0)
        ptr = self._lib.chronos_engine_ingest_rgba_mut_ptr(self._handle, ctypes.byref(out_len))
        if not ptr or out_len.value == 0:
            raise RuntimeError("chronos_engine_ingest_rgba_mut_ptr returned NULL")
        w, h = self.frame_dims
        raw = (ctypes.c_uint8 * out_len.value).from_address(ctypes.addressof(ptr.contents))
        raw._owner = self  # keep engine alive while view exists
        np = _try_numpy()
        if np is not None:
            return np.ctypeslib.as_array(raw).reshape((h, w, 4))
        return memoryview(raw).cast("B")

    def step_in_place(self, dt_sec: float = 1.0 / 30.0, pts_sec: Optional[float] = None) -> int:
        """Advance the engine using pixels already written into `ingest_buffer()`."""
        self._last_dt_sec = float(dt_sec)
        if pts_sec is not None:
            return int(self._lib.chronos_engine_step_with_pts(self._handle, float(pts_sec), float(dt_sec)))
        return int(self._lib.chronos_engine_step(self._handle, float(dt_sec)))

    def step_rgba(self, rgba_frame: object, dt_sec: float = 1.0 / 30.0, pts_sec: Optional[float] = None) -> int:
        """Ingest one tightly-packed RGBA8 frame (`bytes`, `bytearray`, or NumPy `(H,W,4)` uint8 array) and step."""
        self._last_dt_sec = float(dt_sec)
        if hasattr(rgba_frame, "ctypes") and hasattr(rgba_frame, "nbytes"):
            arr = rgba_frame
            if not getattr(arr, "flags", None) or not arr.flags.c_contiguous:
                arr = arr.copy(order="C")
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            nbytes = int(arr.nbytes)
        else:
            buf = memoryview(rgba_frame).cast("B")
            nbytes = len(buf)
            arr_t = (ctypes.c_uint8 * nbytes).from_buffer_copy(buf)
            ptr = ctypes.cast(arr_t, ctypes.POINTER(ctypes.c_uint8))

        rc = int(self._lib.chronos_engine_ingest_rgba(self._handle, ptr, nbytes))
        if rc != 0:
            w, h = self.frame_dims
            raise ValueError(f"chronos_engine_ingest_rgba failed (rc={rc}); expected {w * h * 4} bytes, got {nbytes}")
        if pts_sec is not None:
            return int(self._lib.chronos_engine_step_with_pts(self._handle, float(pts_sec), float(dt_sec)))
        return int(self._lib.chronos_engine_step(self._handle, float(dt_sec)))

    def field_view(self, kind: int | str):
        """Return a zero-copy read-only NumPy array (or `memoryview`) for any `FIELD_*` buffer."""
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
        if kind_id in (FIELD_BOS_PLANES, FIELD_HODGE_PLANES, FIELD_WIND_PLANES, FIELD_ELASTO_PLANES):
            arr = arr.reshape((stride, rows, cols))
        elif kind_id in (FIELD_CONTINUUM, FIELD_DEPTH_FIELD):
            arr = arr.reshape((rows, cols, stride))
        elif cols > 1:
            arr = arr.reshape((rows, cols))
        arr.setflags(write=False)
        return arr

    def telemetry_view(self):
        """Zero-copy read-only view of the 1,568-float telemetry array."""
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
