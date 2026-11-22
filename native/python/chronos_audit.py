#!/usr/bin/env python3
"""CHRONOS-SPECTRAL v3.4.0 Deterministic Physics Integrity Auditor (SDK & Streaming CLI).

Zero hard third-party dependencies (standard library + native `libchronos_capi`).
Optional `pyarrow` enables Parquet export; external `ffmpeg`/`ffprobe` enable MP4/MOV/WebM streaming.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import multiprocessing
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chronos_spectral import (  # noqa: E402
    AUDIT_CLAIM_3D_TRANSLATION,
    AUDIT_CLAIM_CONTINUOUS_CAPTURE,
    AUDIT_CLAIM_DIRECT_AUDIO,
    AUDIT_CLAIM_FIXED_FOCAL,
    AUDIT_CLAIM_PLANAR_FLUID,
    AUDIT_CLAIM_STATIC_BG,
    AXIS_AV_SYNC,
    AXIS_BG_SHIMMER,
    AXIS_COLLISION,
    AXIS_DEPTH_PARALLAX,
    AXIS_FLUID_DIV,
    AXIS_FLUID_MOMENTUM,
    AXIS_SLUGS,
    AXIS_TEMPORAL_CADENCE,
    AXIS_TRAJECTORY,
    CHRONOS_ABI_VERSION,
    CHRONOS_AUDIO_STAGING_LEN,
    ChronosEngine,
    ChronosIngestMeta,
    INGEST_GEOM_KNOWN,
    INGEST_PTS_KNOWN,
    WIND_MODE_DENSE_AUDIT,
    WIND_MODE_STAGGERED,
)

SCHEMA_RECORD = "chronos-audit/1"
SCHEMA_COMPARE = "chronos-audit-compare/1"
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}

LAW_REGISTRY: Tuple[Dict[str, Any], ...] = (
    {"axis_id": AXIS_TRAJECTORY, "id": "trajectory", "name": "Trajectory & Gravity ODE", "unit": "rel_drift"},
    {"axis_id": AXIS_COLLISION, "id": "collision", "name": "Collision Conservation & Restitution", "unit": "e_n_excess"},
    {"axis_id": AXIS_DEPTH_PARALLAX, "id": "depth_parallax", "name": "3D Parallax, Pure-Yaw & TTC", "unit": "1-R2_or_px"},
    {"axis_id": AXIS_FLUID_DIV, "id": "fluid_div", "name": "Planar Fluid Mass Conservation (div u)", "unit": "div_frac"},
    {"axis_id": AXIS_FLUID_MOMENTUM, "id": "fluid_momentum", "name": "Navier-Stokes Curl Integrability", "unit": "eta_NS"},
    {"axis_id": AXIS_BG_SHIMMER, "id": "bg_shimmer", "name": "Static Background Curl / Shimmer", "unit": "px"},
    {"axis_id": AXIS_TEMPORAL_CADENCE, "id": "temporal_cadence", "name": "Temporal Cadence & Stutter", "unit": "ratio_or_frames"},
    {"axis_id": AXIS_AV_SYNC, "id": "av_sync", "name": "Audio-Video Physical Sync", "unit": "ms_or_1-coh"},
)


@dataclass(frozen=True)
class AuditProfile:
    name: str
    res_mode: int
    swarm_mode: int
    depth_mode: int
    bos_mode: int
    hodge_mode: int
    wind_mode: int
    profile_flags: int
    required_axes_mask: int
    min_applicable_frames: int = 8
    fail_severity: float = 0.5
    min_coverage: float = 0.15


PROFILES: Dict[str, AuditProfile] = {
    "quick": AuditProfile(
        name="quick",
        res_mode=0,
        swarm_mode=0,
        depth_mode=0,
        bos_mode=0,
        hodge_mode=0,
        wind_mode=0,
        profile_flags=0,
        required_axes_mask=0,
    ),
    "standard": AuditProfile(
        name="standard",
        res_mode=1,
        swarm_mode=2,
        depth_mode=1,
        bos_mode=0,
        hodge_mode=1,
        wind_mode=WIND_MODE_STAGGERED,
        profile_flags=0,
        required_axes_mask=0,
    ),
    "full": AuditProfile(
        name="full",
        res_mode=1,
        swarm_mode=2,
        depth_mode=1,
        bos_mode=0,
        hodge_mode=1,
        wind_mode=WIND_MODE_STAGGERED,
        profile_flags=AUDIT_CLAIM_CONTINUOUS_CAPTURE,
        required_axes_mask=0,
    ),
    "bos_static": AuditProfile(
        name="bos_static",
        res_mode=1,
        swarm_mode=0,
        depth_mode=0,
        bos_mode=1,
        hodge_mode=0,
        wind_mode=0,
        profile_flags=AUDIT_CLAIM_STATIC_BG | AUDIT_CLAIM_CONTINUOUS_CAPTURE,
        required_axes_mask=(1 << AXIS_BG_SHIMMER),
    ),
    "planar_fluid": AuditProfile(
        name="planar_fluid",
        res_mode=1,
        swarm_mode=2,
        depth_mode=0,
        bos_mode=0,
        hodge_mode=1,
        wind_mode=WIND_MODE_STAGGERED,
        profile_flags=AUDIT_CLAIM_PLANAR_FLUID,
        required_axes_mask=(1 << AXIS_FLUID_DIV) | (1 << AXIS_FLUID_MOMENTUM),
    ),
    "dense_fluid": AuditProfile(
        name="dense_fluid",
        res_mode=1,
        swarm_mode=0,
        depth_mode=0,
        bos_mode=0,
        hodge_mode=1,
        wind_mode=WIND_MODE_DENSE_AUDIT,
        profile_flags=AUDIT_CLAIM_PLANAR_FLUID,
        required_axes_mask=(1 << AXIS_FLUID_DIV),
    ),
    "claimed_3d": AuditProfile(
        name="claimed_3d",
        res_mode=1,
        swarm_mode=2,
        depth_mode=1,
        bos_mode=0,
        hodge_mode=0,
        wind_mode=0,
        profile_flags=AUDIT_CLAIM_3D_TRANSLATION | AUDIT_CLAIM_FIXED_FOCAL,
        required_axes_mask=(1 << AXIS_DEPTH_PARALLAX),
    ),
    "av_impact": AuditProfile(
        name="av_impact",
        res_mode=1,
        swarm_mode=0,
        depth_mode=0,
        bos_mode=0,
        hodge_mode=0,
        wind_mode=0,
        profile_flags=AUDIT_CLAIM_DIRECT_AUDIO,
        required_axes_mask=(1 << AXIS_AV_SYNC),
    ),
}


def _qfloat(x: Optional[float]) -> Optional[float]:
    if x is None or not math.isfinite(x):
        return None
    return float(format(x, ".6g"))


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _record_hash(rec: Dict[str, Any]) -> str:
    body = {k: v for k, v in rec.items() if k not in ("record_sha256", "provenance", "perf")}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_letterbox_rect(src_w: int, src_h: int, dst_w: int, dst_h: int) -> Tuple[int, int, int, int]:
    """Exact integer aspect-preserving letterbox rectangle `(sw, sh, dx, dy)` inside `(dst_w, dst_h)`."""
    if src_w <= 0 or src_h <= 0:
        return dst_w, dst_h, 0, 0
    if src_w * dst_h >= dst_w * src_h:
        sw = dst_w
        sh = max(1, min(dst_h, (src_h * dst_w + src_w // 2) // src_w))
    else:
        sh = dst_h
        sw = max(1, min(dst_w, (src_w * dst_h + src_h // 2) // src_h))
    dx = (dst_w - sw) // 2
    dy = (dst_h - sh) // 2
    return sw, sh, dx, dy


def _parse_fraction(val: Optional[str], default: Tuple[int, int] = (30, 1)) -> Fraction:
    if not val or val in ("0/0", "N/A"):
        return Fraction(*default)
    try:
        fr = Fraction(val)
        if fr <= 0:
            return Fraction(*default)
        return fr
    except Exception:
        return Fraction(*default)


def probe_media(path: Path, ffprobe: str = "ffprobe") -> Dict[str, Any]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.PIPE)
    data = json.loads(out.decode("utf-8", "replace"))
    streams = data.get("streams", [])
    v_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not v_streams:
        raise ValueError("NO_VIDEO_STREAM")
    vs = v_streams[0]
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]
    aud = a_streams[0] if a_streams else None

    w = int(vs.get("width") or 0)
    h = int(vs.get("height") or 0)
    sar_str = vs.get("sample_aspect_ratio")
    if sar_str and ":" in sar_str and sar_str != "0:1":
        sn, sd = sar_str.split(":", 1)
        try:
            sn_i, sd_i = int(sn), int(sd)
            if sn_i > 0 and sd_i > 0 and sn_i != sd_i:
                w = max(1, (w * sn_i + sd_i // 2) // sd_i)
        except ValueError:
            pass

    rot = 0
    for sd in vs.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rot = int(round(float(sd["rotation"]))) % 360
            except Exception:
                pass
    if rot in (90, 270):
        w, h = h, w

    avg_fr = _parse_fraction(vs.get("avg_frame_rate"))
    r_fr = _parse_fraction(vs.get("r_frame_rate"), (avg_fr.numerator, avg_fr.denominator))
    vfr = abs(float(avg_fr) - float(r_fr)) / max(float(avg_fr), 1e-6) > 0.03

    v_start = float(vs.get("start_time") or 0.0)
    a_start = float(aud.get("start_time") or 0.0) if aud else 0.0
    dur = float(vs.get("duration") or data.get("format", {}).get("duration") or 0.0)
    return {
        "src_w": w,
        "src_h": h,
        "rotation": rot,
        "fps_frac": avg_fr,
        "vfr_detected": vfr,
        "v_start_s": v_start,
        "a_start_s": a_start,
        "has_audio": aud is not None,
        "duration_s": dur,
    }


def _configure_engine(
    eng: ChronosEngine,
    prof: AuditProfile,
    *,
    profile_flags: Optional[int] = None,
    required_axes_mask: Optional[int] = None,
    expected_frames: int = 0,
    depth_calib: Optional[Tuple[float, float]] = None,
) -> None:
    eng.reset()
    eng.set_resolution_mode(prof.res_mode)
    eng.set_swarm_mode(prof.swarm_mode)
    if prof.bos_mode:
        eng.set_bos_mode(prof.bos_mode)
        eng.set_bos_gauge_roi(8, 8, 36, 120)
    if prof.hodge_mode:
        eng.set_hodge_mode(prof.hodge_mode)
    if prof.wind_mode:
        eng.set_wind_mode(prof.wind_mode)
    if prof.depth_mode:
        eng.set_depth_mode(prof.depth_mode)
        if depth_calib is not None:
            eng.set_depth_calib(depth_calib[0], depth_calib[1], 0.0, 32.0, 0)
    eng.audit_begin(
        profile_flags=prof.profile_flags if profile_flags is None else profile_flags,
        required_axes_mask=prof.required_axes_mask if required_axes_mask is None else required_axes_mask,
        expected_frames=expected_frames,
        min_applicable_frames=prof.min_applicable_frames,
        fail_severity=prof.fail_severity,
        min_coverage=prof.min_coverage,
    )


def _build_record(
    *,
    clip_path: str,
    clip_sha256: Optional[str],
    clip_bytes: int,
    label: Optional[str],
    status: str,
    error: Optional[Dict[str, Any]],
    summary: Optional[Dict[str, Any]],
    spans: List[Dict[str, Any]],
    ingest_info: Dict[str, Any],
    profile_name: str,
    res_mode: int,
    max_seconds: Optional[float],
    engine_ver: str,
    ffmpeg_ver: Optional[str],
    include_perf: bool,
    wall_s: float,
) -> Dict[str, Any]:
    cfg_obj = {"profile": profile_name, "res_mode": res_mode, "max_seconds": max_seconds}
    cfg_hash = hashlib.sha256(_canonical_json(cfg_obj).encode("utf-8")).hexdigest()[:16]
    verdict = {
        "grade": summary["grade"] if summary else "ERROR",
        "integrity": summary["integrity"] if summary else None,
        "coverage": summary["coverage"] if summary else 0.0,
        "applicable_axes": summary["applicable_axes"] if summary else 0,
        "violated_axes": summary["violated_axes"] if summary else 0,
        "warned_axes": summary["warned_axes"] if summary else 0,
        "violations": summary["span_count"] if summary else 0,
    }
    rlvr = {
        "reward": summary["reward"] if summary else None,
        "reward_milli": summary["reward_milli"] if summary else None,
        "gate": (
            None
            if (summary and summary["reward_milli"] is not None)
            else ("NO_APPLICABLE_LAW" if (summary and summary["applicable_axes"] == 0) else "COVERAGE_LOW")
        ),
    }
    rec: Dict[str, Any] = {
        "schema": SCHEMA_RECORD,
        "clip": {
            "path": clip_path,
            "sha256": clip_sha256,
            "bytes": clip_bytes,
            "label": label,
        },
        "status": status,
        "error": error,
        "verdict": verdict,
        "rlvr": rlvr,
        "axes": summary["axes"] if summary else [],
        "violations": spans,
        "violations_truncated": summary["spans_dropped"] if summary else 0,
        "shots": summary["shots"] if summary else 0,
        "ingest": ingest_info,
        "engine": {"version": engine_ver, "abi": CHRONOS_ABI_VERSION, "law_set": "core-8"},
        "config": {**cfg_obj, "config_sha256": cfg_hash},
        "provenance": {
            "ffmpeg": ffmpeg_ver,
            "python": platform.python_version(),
            "platform": f"{platform.system().lower()}-{platform.machine().lower()}",
        },
    }
    if include_perf:
        frames_c = ingest_info.get("frames_committed", 0)
        rec["perf"] = {
            "wall_s": _qfloat(wall_s),
            "fps": _qfloat(frames_c / max(wall_s, 1e-6)),
        }
    rec["record_sha256"] = _record_hash(rec)
    return rec


class ChronosAuditor:
    """Reusable deterministic physics auditor backed by one persistent `ChronosEngine` handle."""

    def __init__(
        self,
        profile: str = "standard",
        *,
        lib_path: Optional[os.PathLike | str] = None,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        audio: bool = True,
        max_seconds: Optional[float] = None,
        clip_timeout: float = 600.0,
        hash_mode: str = "sha256",
        label: Optional[str] = None,
        include_perf: bool = False,
        profile_flags: Optional[int] = None,
        required_axes_mask: Optional[int] = None,
        depth_calib: Optional[Tuple[float, float]] = None,
        ffmpeg_threads: int = 0,
    ) -> None:
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(PROFILES)}")
        self.profile = PROFILES[profile]
        self.lib_path = str(lib_path) if lib_path is not None else None
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.audio = audio
        self.max_seconds = max_seconds
        self.clip_timeout = clip_timeout
        self.hash_mode = hash_mode
        self.label = label
        self.include_perf = include_perf
        self.profile_flags = profile_flags
        self.required_axes_mask = required_axes_mask
        self.depth_calib = depth_calib
        self.ffmpeg_threads = ffmpeg_threads
        self._engine = ChronosEngine(lib_path=lib_path, mode=self.profile.res_mode)
        self._ffmpeg_ver: Optional[str] = None

    def close(self) -> None:
        if getattr(self, "_engine", None) is not None:
            self._engine.close()
            self._engine = None

    def __enter__(self) -> "ChronosAuditor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get_ffmpeg_version(self) -> Optional[str]:
        if self._ffmpeg_ver is None:
            exe = shutil.which(self.ffmpeg)
            if exe:
                try:
                    line = subprocess.check_output([exe, "-version"], stderr=subprocess.DEVNULL, text=True).splitlines()[0]
                    self._ffmpeg_ver = line.split()[2] if len(line.split()) >= 3 else line
                except Exception:
                    self._ffmpeg_ver = "unknown"
        return self._ffmpeg_ver

    def audit_frames(
        self,
        frames: Iterable[Any],
        *,
        src_w: int,
        src_h: int,
        fps: float = 30.0,
        audio_pcm: Optional[Any] = None,
        audio_rate_hz: float = 48000.0,
        pts_us_list: Optional[Sequence[int]] = None,
        scan_dir: int = 0xFFFFFFFF,
        native_1to1: int = 0,
        clip_name: str = "<frames>",
        fit: int = 0,
        expected_frames: int = 0,
    ) -> Dict[str, Any]:
        """Audit an in-memory sequence of RGBA8 frames (RLDS / LeRobot / NumPy / synthetic fixtures)."""
        t0 = time.perf_counter()
        eng = self._engine
        _configure_engine(
            eng,
            self.profile,
            profile_flags=self.profile_flags,
            required_axes_mask=self.required_axes_mask,
            expected_frames=expected_frames,
            depth_calib=self.depth_calib,
        )
        ew, eh = eng.frame_dims
        dt_sec = 1.0 / max(float(fps), 1.0)
        if audio_pcm is not None:
            _mv = memoryview(audio_pcm)
            audio_mv = _mv if _mv.format == "f" else _mv.cast("B").cast("f")
        else:
            audio_mv = None
        a_cursor = 0
        k = 0
        last_dst = [0, 0, ew, eh]
        for frame in frames:
            pts_us = int(pts_us_list[k]) if (pts_us_list is not None and k < len(pts_us_list)) else int(round(k * dt_sec * 1_000_000.0))
            if audio_mv is not None:
                s_next = min(len(audio_mv), int(round((k + 1) * dt_sec * audio_rate_hz)))
                while a_cursor < s_next:
                    chunk = min(CHRONOS_AUDIO_STAGING_LEN, s_next - a_cursor)
                    end_s = a_cursor + chunk
                    t_end_us = int(round((end_s * 1_000_000.0) / audio_rate_hz))
                    stg = eng.audio_staging_buffer(raw_bytes=True)
                    src_b = audio_mv[a_cursor:end_s].cast("B")
                    stg[: len(src_b)] = src_b
                    eng.push_audio_block(chunk, t_end_us, audio_rate_hz, 0.0)
                    a_cursor = end_s

            meta = ChronosIngestMeta(
                struct_size=ctypes.sizeof(ChronosIngestMeta),
                flags=INGEST_PTS_KNOWN | INGEST_GEOM_KNOWN,
                pts_us=pts_us,
                epoch=1,
                seq=k + 1,
                age_ms=-1.0,
                orientation=0,
                scan_dir=scan_dir,
                gap_mask=0,
                native_1to1=native_1to1,
            )
            rc = eng.ingest_rgba_strided(frame, src_w, src_h, 0, fit, meta)
            if rc != 0:
                raise RuntimeError(f"ingest_rgba_strided failed with rc={rc} at frame {k}")
            last_dst = [int(meta.dst_x), int(meta.dst_y), int(meta.dst_w), int(meta.dst_h)]
            eng.set_ingest_meta(meta)
            eng.step_in_place(dt_sec)
            k += 1

        eng.audit_finalize()
        summary = eng.audit_summary()
        spans = eng.audit_spans()
        wall = time.perf_counter() - t0
        ingest_info = {
            "decoder": "memory",
            "src_w": src_w,
            "src_h": src_h,
            "rotation": 0,
            "fps": f"{fps:.6g}",
            "vfr_detected": False,
            "duration_us": int(round(k * dt_sec * 1_000_000.0)),
            "frames_decoded": k,
            "frames_committed": summary["frames_committed"],
            "frames_rejected": summary["frames_rejected"],
            "letterbox": last_dst,
            "audio": {
                "present": audio_mv is not None,
                "rate_hz": int(audio_rate_hz) if audio_mv is not None else 0,
                "samples": a_cursor,
                "underruns": summary["audio_underruns"],
            },
        }
        return _build_record(
            clip_path=clip_name,
            clip_sha256=None,
            clip_bytes=0,
            label=self.label,
            status="ok",
            error=None,
            summary=summary,
            spans=spans,
            ingest_info=ingest_info,
            profile_name=self.profile.name,
            res_mode=self.profile.res_mode,
            max_seconds=self.max_seconds,
            engine_ver=eng.version_string,
            ffmpeg_ver=None,
            include_perf=self.include_perf,
            wall_s=wall,
        )

    def audit_video(self, path: os.PathLike | str, *, rel_path: Optional[str] = None) -> Dict[str, Any]:
        """Stream one video file (plus optional muxed audio) in constant RAM through `ChronosEngine`."""
        p = Path(path)
        disp_path = rel_path if rel_path is not None else str(p)
        t0 = time.perf_counter()
        clip_bytes = p.stat().st_size if p.exists() else 0
        clip_sha = _sha256_file(p) if (self.hash_mode == "sha256" and p.exists()) else None
        eng = self._engine

        ffprobe_exe = shutil.which(self.ffprobe)
        ffmpeg_exe = shutil.which(self.ffmpeg)
        if not ffprobe_exe or not ffmpeg_exe:
            raise EnvironmentError("ffmpeg and ffprobe are required on PATH to audit media files")

        try:
            meta_p = probe_media(p, ffprobe_exe)
        except Exception as exc:
            return _build_record(
                clip_path=disp_path,
                clip_sha256=clip_sha,
                clip_bytes=clip_bytes,
                label=self.label,
                status="error",
                error={"code": "PROBE_FAILED", "message": str(exc), "stderr_tail": ""},
                summary=None,
                spans=[],
                ingest_info={},
                profile_name=self.profile.name,
                res_mode=self.profile.res_mode,
                max_seconds=self.max_seconds,
                engine_ver=eng.version_string,
                ffmpeg_ver=self._get_ffmpeg_version(),
                include_perf=self.include_perf,
                wall_s=time.perf_counter() - t0,
            )

        fps_fr: Fraction = meta_p["fps_frac"]
        num, den = fps_fr.numerator, fps_fr.denominator
        dt_sec = float(den) / float(num)
        est_frames = int(round(min(meta_p["duration_s"], self.max_seconds or 1e9) * float(fps_fr))) if meta_p["duration_s"] > 0 else 0

        _configure_engine(
            eng,
            self.profile,
            profile_flags=self.profile_flags,
            required_axes_mask=self.required_axes_mask,
            expected_frames=est_frames,
            depth_calib=self.depth_calib,
        )
        ew, eh = eng.frame_dims
        sw, sh, dx, dy = compute_letterbox_rect(meta_p["src_w"], meta_p["src_h"], ew, eh)

        vf = (
            f"scale={sw}:{sh}:flags=area+accurate_rnd+full_chroma_int+bitexact,"
            f"format=rgba,pad={ew}:{eh}:{dx}:{dy}:black,setsar=1"
        )
        v_cmd = [ffmpeg_exe, "-nostdin", "-hide_banner", "-v", "error"]
        if self.ffmpeg_threads > 0:
            v_cmd += ["-threads", str(self.ffmpeg_threads)]
        if self.max_seconds is not None and self.max_seconds > 0:
            v_cmd += ["-t", f"{self.max_seconds:.6f}"]
        v_cmd += [
            "-i",
            str(p),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            vf,
            "-fps_mode",
            "cfr",
            "-r",
            f"{num}/{den}",
            "-f",
            "rawvideo",
            "pipe:1",
        ]

        want_audio = bool(self.audio and meta_p["has_audio"])
        a_cmd = [ffmpeg_exe, "-nostdin", "-hide_banner", "-v", "error"]
        if self.ffmpeg_threads > 0:
            a_cmd += ["-threads", "1"]
        if self.max_seconds is not None and self.max_seconds > 0:
            a_cmd += ["-t", f"{self.max_seconds:.6f}"]
        a_cmd += ["-i", str(p), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1"]

        v_err = tempfile.TemporaryFile()
        a_err = tempfile.TemporaryFile() if want_audio else None
        v_proc = None
        a_proc = None
        frames_decoded = 0
        audio_samples_pushed = 0
        a_offset_us = int(round((meta_p["a_start_s"] - meta_p["v_start_s"]) * 1_000_000.0))

        try:
            v_proc = subprocess.Popen(v_cmd, stdout=subprocess.PIPE, stderr=v_err, bufsize=ew * eh * 4)
            if want_audio:
                a_proc = subprocess.Popen(a_cmd, stdout=subprocess.PIPE, stderr=a_err, bufsize=4096 * 4)

            assert v_proc.stdout is not None
            frame_bytes = ew * eh * 4
            a_eof = not want_audio
            a_rate = 48000

            while True:
                if time.perf_counter() - t0 > self.clip_timeout:
                    raise TimeoutError(f"clip timeout after {self.clip_timeout:.1f}s")
                buf = eng.ingest_buffer(raw=True)
                got = _read_exact_into(v_proc.stdout, buf, frame_bytes)
                if got < frame_bytes:
                    break

                if a_proc is not None and a_proc.stdout is not None and not a_eof:
                    s_next = ((frames_decoded + 1) * den * a_rate) // num
                    while audio_samples_pushed < s_next:
                        need = min(CHRONOS_AUDIO_STAGING_LEN, s_next - audio_samples_pushed)
                        stg = eng.audio_staging_buffer(raw_bytes=True)
                        got_a = _read_exact_into(a_proc.stdout, stg[: need * 4], need * 4)
                        got_s = got_a // 4
                        if got_s <= 0:
                            a_eof = True
                            break
                        audio_samples_pushed += got_s
                        t_end_us = max(0, a_offset_us + (audio_samples_pushed * 1_000_000) // a_rate)
                        eng.push_audio_block(got_s, t_end_us, float(a_rate), 0.0)
                        if got_s < need:
                            a_eof = True
                            break

                pts_us = (frames_decoded * den * 1_000_000 + num // 2) // num
                imeta = ChronosIngestMeta(
                    struct_size=ctypes.sizeof(ChronosIngestMeta),
                    flags=INGEST_PTS_KNOWN | INGEST_GEOM_KNOWN,
                    pts_us=pts_us,
                    epoch=1,
                    seq=frames_decoded + 1,
                    age_ms=-1.0,
                    src_w=meta_p["src_w"],
                    src_h=meta_p["src_h"],
                    crop_x=0,
                    crop_y=0,
                    crop_w=meta_p["src_w"],
                    crop_h=meta_p["src_h"],
                    dst_x=dx,
                    dst_y=dy,
                    dst_w=sw,
                    dst_h=sh,
                    orientation=0,
                    scan_dir=0xFFFFFFFF,
                    gap_mask=0,
                    native_1to1=0,
                )
                eng.set_ingest_meta(imeta)
                eng.step_in_place(dt_sec)
                frames_decoded += 1

            eng.audit_finalize()
            summary = eng.audit_summary()
            spans = eng.audit_spans()
            wall = time.perf_counter() - t0
            ingest_info = {
                "decoder": "ffmpeg",
                "src_w": meta_p["src_w"],
                "src_h": meta_p["src_h"],
                "rotation": meta_p["rotation"],
                "fps": f"{num}/{den}",
                "vfr_detected": meta_p["vfr_detected"],
                "duration_us": (frames_decoded * den * 1_000_000 + num // 2) // num,
                "frames_decoded": frames_decoded,
                "frames_committed": summary["frames_committed"],
                "frames_rejected": summary["frames_rejected"],
                "letterbox": [dx, dy, sw, sh],
                "audio": {
                    "present": want_audio,
                    "rate_hz": a_rate if want_audio else 0,
                    "samples": audio_samples_pushed,
                    "underruns": summary["audio_underruns"],
                },
            }
            err_obj = None
            if frames_decoded == 0:
                v_err.seek(0)
                v_tail = v_err.read()[-1024:].decode("utf-8", "replace")
                err_obj = {"code": "DECODE_FAILED", "message": "0 frames decoded", "stderr_tail": v_tail}
            return _build_record(
                clip_path=disp_path,
                clip_sha256=clip_sha,
                clip_bytes=clip_bytes,
                label=self.label,
                status="ok" if frames_decoded > 0 else "error",
                error=err_obj,
                summary=summary if frames_decoded > 0 else None,
                spans=spans if frames_decoded > 0 else [],
                ingest_info=ingest_info,
                profile_name=self.profile.name,
                res_mode=self.profile.res_mode,
                max_seconds=self.max_seconds,
                engine_ver=eng.version_string,
                ffmpeg_ver=self._get_ffmpeg_version(),
                include_perf=self.include_perf,
                wall_s=wall,
            )
        except Exception as exc:
            v_err.seek(0)
            tail = v_err.read()[-1024:].decode("utf-8", "replace")
            code = "TIMEOUT" if isinstance(exc, TimeoutError) else "DECODE_FAILED"
            return _build_record(
                clip_path=disp_path,
                clip_sha256=clip_sha,
                clip_bytes=clip_bytes,
                label=self.label,
                status="error",
                error={"code": code, "message": str(exc), "stderr_tail": tail},
                summary=None,
                spans=[],
                ingest_info={"frames_decoded": frames_decoded},
                profile_name=self.profile.name,
                res_mode=self.profile.res_mode,
                max_seconds=self.max_seconds,
                engine_ver=eng.version_string,
                ffmpeg_ver=self._get_ffmpeg_version(),
                include_perf=self.include_perf,
                wall_s=time.perf_counter() - t0,
            )
        finally:
            for proc in (v_proc, a_proc):
                if proc is not None:
                    try:
                        if proc.stdout:
                            proc.stdout.close()
                    except Exception:
                        pass
                    if proc.poll() is None:
                        proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass
            v_err.close()
            if a_err is not None:
                a_err.close()


def _read_exact_into(stream, mv: memoryview, nbytes: int) -> int:
    total = 0
    while total < nbytes:
        got = stream.readinto(mv[total:nbytes])
        if not got:
            break
        total += got
    return total


def discover_videos(paths: Sequence[os.PathLike | str]) -> List[Tuple[Path, str]]:
    found: List[Tuple[Path, str]] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            found.append((p, p.name))
        elif p.is_dir():
            for root, dirs, files in os.walk(p, followlinks=False):
                dirs.sort()
                for fn in sorted(files):
                    fp = Path(root) / fn
                    if fp.is_symlink():
                        continue
                    if fp.suffix.lower() in VIDEO_EXTS:
                        rel = fp.relative_to(p).as_posix()
                        found.append((fp, rel))
    found.sort(key=lambda item: item[1].encode("utf-8"))
    return found


_WORKER_AUDITOR: Optional[ChronosAuditor] = None


def _init_worker(cfg: Dict[str, Any]) -> None:
    global _WORKER_AUDITOR
    _WORKER_AUDITOR = ChronosAuditor(**cfg)


def _worker_run_one(item: Tuple[str, str]) -> Dict[str, Any]:
    assert _WORKER_AUDITOR is not None
    abs_p, rel_p = item
    return _WORKER_AUDITOR.audit_video(abs_p, rel_path=rel_p)


def resolve_workers(workers: int | str) -> int:
    if isinstance(workers, int):
        return max(1, workers)
    ncpu = os.cpu_count() or 2
    return max(1, ncpu - 1)


def audit_video(path: os.PathLike | str, profile: str = "standard", **kwargs: Any) -> Dict[str, Any]:
    with ChronosAuditor(profile=profile, **kwargs) as aud:
        return aud.audit_video(path)


def iter_directory(
    paths: Sequence[os.PathLike | str],
    *,
    profile: str = "standard",
    workers: int | str = "auto",
    skip_paths: Optional[set[str]] = None,
    **kwargs: Any,
) -> Iterator[Dict[str, Any]]:
    items = discover_videos(paths)
    if skip_paths:
        items = [it for it in items if it[1] not in skip_paths]
    w = resolve_workers(workers)
    cfg = dict(kwargs)
    cfg["profile"] = profile
    cfg["ffmpeg_threads"] = 1 if w > 1 else 0
    if w <= 1 or len(items) <= 1:
        with ChronosAuditor(**cfg) as aud:
            for abs_p, rel_p in items:
                yield aud.audit_video(abs_p, rel_path=rel_p)
        return
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=w, initializer=_init_worker, initargs=(cfg,), maxtasksperchild=256) as pool:
        for rec in pool.imap(_worker_run_one, [(str(a), r) for a, r in items], chunksize=1):
            yield rec


def _csv_clip_headers() -> List[str]:
    cols = [
        "path",
        "sha256",
        "label",
        "status",
        "grade",
        "integrity",
        "coverage",
        "reward",
        "applicable_axes",
        "violated_axes",
        "violations",
        "duration_s",
        "frames",
        "audio",
    ]
    for slug in AXIS_SLUGS:
        cols.extend([f"{slug}_appl", f"{slug}_verdict", f"{slug}_score"])
    cols.append("record_sha256")
    return cols


def _csv_clip_row(rec: Dict[str, Any]) -> List[Any]:
    v = rec["verdict"]
    ing = rec.get("ingest", {})
    dur_s = (ing.get("duration_us", 0) or 0) / 1_000_000.0
    row: List[Any] = [
        rec["clip"]["path"],
        rec["clip"]["sha256"] or "",
        rec["clip"]["label"] or "",
        rec["status"],
        v["grade"],
        "" if v["integrity"] is None else v["integrity"],
        v["coverage"],
        "" if rec["rlvr"]["reward"] is None else rec["rlvr"]["reward"],
        v["applicable_axes"],
        v["violated_axes"],
        v["violations"],
        f"{dur_s:.3f}",
        ing.get("frames_committed", 0),
        int(bool(ing.get("audio", {}).get("present", False))),
    ]
    ax_map = {a["id"]: a for a in rec.get("axes", [])}
    for slug in AXIS_SLUGS:
        a = ax_map.get(slug)
        if a:
            row.extend([a["applicability"], a["verdict"], "" if a["score"] is None else a["score"]])
        else:
            row.extend(["", "", ""])
    row.append(rec["record_sha256"])
    return row


def _write_parquet(records: List[Dict[str, Any]], out_path: Path) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Parquet output requires optional `pyarrow` (`pip install pyarrow`)") from exc
    headers = _csv_clip_headers()
    cols: Dict[str, List[Any]] = {h: [] for h in headers}
    for rec in records:
        row = _csv_clip_row(rec)
        for h, val in zip(headers, row):
            cols[h].append(val if val != "" else None)
    table = pa.table(cols)
    pq.write_table(table, str(out_path))


def audit_directory(
    paths: Sequence[os.PathLike | str] | os.PathLike | str,
    *,
    profile: str = "standard",
    workers: int | str = "auto",
    out: Optional[os.PathLike | str] = None,
    fmt: str = "jsonl",
    violations_out: Optional[os.PathLike | str] = None,
    resume: bool = False,
    **kwargs: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path_list = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)
    skip_paths: set[str] = set()
    existing_records: List[Dict[str, Any]] = []
    if resume and out and str(out) != "-" and fmt == "jsonl" and Path(out).exists():
        good_lines: List[str] = []
        with Path(out).open("r", encoding="utf-8") as fh:
            for line in fh:
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    rec = json.loads(line_s)
                    skip_paths.add(rec["clip"]["path"])
                    existing_records.append(rec)
                    good_lines.append(line_s)
                except json.JSONDecodeError:
                    break
        with Path(out).open("w", encoding="utf-8") as fh:
            for gl in good_lines:
                fh.write(gl + "\n")

    records = list(existing_records)
    out_fh = None
    if out and fmt == "jsonl":
        if str(out) == "-":
            out_fh = sys.stdout
        else:
            mode = "a" if (resume and existing_records) else "w"
            out_fh = Path(out).open(mode, encoding="utf-8")

    try:
        for rec in iter_directory(path_list, profile=profile, workers=workers, skip_paths=skip_paths, **kwargs):
            records.append(rec)
            if out_fh is not None:
                out_fh.write(_canonical_json(rec) + "\n")
                out_fh.flush()
    finally:
        if out_fh is not None and out_fh is not sys.stdout:
            out_fh.close()

    if out and fmt != "jsonl":
        out_p = Path(out)
        if fmt == "json":
            txt = json.dumps(records, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            if str(out) == "-":
                sys.stdout.write(txt)
            else:
                out_p.write_text(txt, encoding="utf-8")
        elif fmt == "csv":
            fh = sys.stdout if str(out) == "-" else out_p.open("w", encoding="utf-8", newline="")
            try:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(_csv_clip_headers())
                for rec in records:
                    w.writerow(_csv_clip_row(rec))
            finally:
                if fh is not sys.stdout:
                    fh.close()
        elif fmt == "parquet":
            _write_parquet(records, out_p)

    if violations_out:
        with Path(violations_out).open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(
                [
                    "path",
                    "sha256",
                    "axis",
                    "code",
                    "f0",
                    "f1",
                    "t0_s",
                    "t1_s",
                    "t0_us",
                    "t1_us",
                    "severity",
                    "track_id",
                    "bx",
                    "by",
                    "bw",
                    "bh",
                    "measured",
                    "expected",
                ]
            )
            for rec in records:
                for sp in rec.get("violations", []):
                    bb = sp.get("bbox") or [0, 0, 1, 1]
                    w.writerow(
                        [
                            rec["clip"]["path"],
                            rec["clip"]["sha256"] or "",
                            sp["axis"],
                            sp["code"],
                            sp["f0"],
                            sp["f1"],
                            f"{sp['t0_us'] / 1e6:.6f}",
                            f"{sp['t1_us'] / 1e6:.6f}",
                            sp["t0_us"],
                            sp["t1_us"],
                            sp["severity"],
                            sp["track_id"],
                            bb[0],
                            bb[1],
                            bb[2],
                            bb[3],
                            sp["measured"],
                            sp["expected"],
                        ]
                    )

    grades: Dict[str, int] = {}
    for rec in records:
        g = rec["verdict"]["grade"]
        grades[g] = grades.get(g, 0) + 1
    return records, {"total": len(records), "grades": grades}


def _wilson_ci95(k: int, n: int) -> Tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return _qfloat(max(0.0, center - half)), _qfloat(min(1.0, center + half))


def _mann_whitney_auroc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    wins = 0.0
    for x in pos:
        for y in neg:
            if x > y:
                wins += 1.0
            elif x == y:
                wins += 0.5
    return _qfloat(wins / (len(pos) * len(neg)))


def _ks_stat(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if not a or not b:
        return None
    sa = sorted(a)
    sb = sorted(b)
    pts = sorted(set(sa) | set(sb))
    ia = ib = 0
    na, nb = len(sa), len(sb)
    max_d = 0.0
    for x in pts:
        while ia < na and sa[ia] <= x:
            ia += 1
        while ib < nb and sb[ib] <= x:
            ib += 1
        d = abs(ia / na - ib / nb)
        if d > max_d:
            max_d = d
    return _qfloat(max_d)


def _cohens_d(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(1, len(a) + len(b) - 2))
    if pooled < 1e-12:
        return 0.0
    return _qfloat((ma - mb) / pooled)


def _quantiles(xs: Sequence[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {"mean": None, "p10": None, "p50": None, "p90": None}
    s = sorted(xs)
    n = len(s)
    return {
        "mean": _qfloat(sum(s) / n),
        "p10": _qfloat(s[min(n - 1, int(0.10 * (n - 1)))]),
        "p50": _qfloat(s[n // 2]),
        "p90": _qfloat(s[min(n - 1, int(0.90 * (n - 1)))]),
    }


def compare_records(
    real_records: Sequence[Dict[str, Any]],
    gen_records: Sequence[Dict[str, Any]],
    *,
    profile: str = "standard",
) -> Dict[str, Any]:
    """Closed-form corpus separation statistics (`real` vs `generated`) with zero randomness."""
    def _corpus_stats(recs: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[float]]:
        ok = sum(1 for r in recs if r["status"] == "ok")
        err = sum(1 for r in recs if r["status"] != "ok")
        na = sum(1 for r in recs if r["verdict"]["grade"] == "NA")
        vals = [float(r["verdict"]["integrity"]) for r in recs if r["verdict"]["integrity"] is not None]
        false_viol = sum(1 for r in recs if r["verdict"]["grade"] == "FAIL")
        return {
            "n": len(recs),
            "ok": ok,
            "error": err,
            "na": na,
            "fail_rate": _qfloat(false_viol / max(1, ok)),
            "integrity": _quantiles(vals),
        }, vals

    real_s, real_vals = _corpus_stats(real_records)
    gen_s, gen_vals = _corpus_stats(gen_records)

    laws_out = []
    top_offenders: Dict[str, List[Dict[str, Any]]] = {}
    for idx, slug in enumerate(AXIS_SLUGS):
        r_appl = r_pass = g_appl = g_pass = 0
        r_scores: List[float] = []
        g_scores: List[float] = []
        top_codes: Dict[str, int] = {}
        offenders: List[Tuple[float, str, Dict[str, Any]]] = []

        for recs, is_real in ((real_records, True), (gen_records, False)):
            for rec in recs:
                axes = rec.get("axes", [])
                if idx >= len(axes):
                    continue
                ax = axes[idx]
                if ax["applicable"]:
                    if is_real:
                        r_appl += 1
                        if ax["pass"]:
                            r_pass += 1
                        if ax["score"] is not None:
                            r_scores.append(float(ax["score"]))
                    else:
                        g_appl += 1
                        if ax["pass"]:
                            g_pass += 1
                        if ax["score"] is not None:
                            g_scores.append(float(ax["score"]))
                if not is_real:
                    for sp in rec.get("violations", []):
                        if sp["axis"] == slug:
                            top_codes[sp["code"]] = top_codes.get(sp["code"], 0) + 1
                            offenders.append(
                                (
                                    float(sp["severity"] or 0.0),
                                    rec["clip"]["path"],
                                    {
                                        "path": rec["clip"]["path"],
                                        "code": sp["code"],
                                        "severity": sp["severity"],
                                        "f0": sp["f0"],
                                        "f1": sp["f1"],
                                    },
                                )
                            )

        r_pr = _qfloat(r_pass / r_appl) if r_appl > 0 else None
        g_pr = _qfloat(g_pass / g_appl) if g_appl > 0 else None
        delta_pr = _qfloat(g_pr - r_pr) if (r_pr is not None and g_pr is not None) else None
        laws_out.append(
            {
                "id": slug,
                "real": {
                    "applicable_rate": _qfloat(r_appl / max(1, len(real_records))),
                    "pass_rate": r_pr,
                    "ci95": list(_wilson_ci95(r_pass, r_appl)),
                    "n": r_appl,
                },
                "generated": {
                    "applicable_rate": _qfloat(g_appl / max(1, len(gen_records))),
                    "pass_rate": g_pr,
                    "ci95": list(_wilson_ci95(g_pass, g_appl)),
                    "n": g_appl,
                },
                "delta_pass_rate": delta_pr,
                "auroc": _mann_whitney_auroc(r_scores, g_scores),
                "top_codes": dict(sorted(top_codes.items(), key=lambda kv: (-kv[1], kv[0]))[:8]),
            }
        )
        offenders.sort(key=lambda t: (-t[0], t[1]))
        top_offenders[slug] = [o[2] for o in offenders[:20]]

    return {
        "schema": SCHEMA_COMPARE,
        "law_set": "core-8",
        "profile": profile,
        "corpora": {"real": real_s, "generated": gen_s},
        "separation": {
            "auroc": _mann_whitney_auroc(real_vals, gen_vals),
            "ks": _ks_stat(real_vals, gen_vals),
            "cohens_d": _cohens_d(real_vals, gen_vals),
        },
        "axes": laws_out,
        "top_offenders": top_offenders,
    }


def compare_corpora(
    real_dir: os.PathLike | str,
    gen_dir: os.PathLike | str,
    *,
    profile: str = "standard",
    workers: int | str = "auto",
    out: Optional[os.PathLike | str] = None,
    records_out: Optional[os.PathLike | str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    real_recs, _ = audit_directory(real_dir, profile=profile, workers=workers, label="real", **kwargs)
    gen_recs, _ = audit_directory(gen_dir, profile=profile, workers=workers, label="generated", **kwargs)
    report = compare_records(real_recs, gen_recs, profile=profile)
    if records_out:
        with Path(records_out).open("w", encoding="utf-8") as fh:
            for r in real_recs + gen_recs:
                fh.write(_canonical_json(r) + "\n")
    if out:
        txt = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        if str(out) == "-":
            sys.stdout.write(txt)
        else:
            Path(out).write_text(txt, encoding="utf-8")
    return report


def run_selftest(*, lib_path: Optional[str] = None) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("selftest: ffmpeg missing on PATH", file=sys.stderr)
        return 3
    with tempfile.TemporaryDirectory(prefix="chronos_audit_selftest_") as td:
        clip = Path(td) / "selftest.mp4"
        subprocess.check_call(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=320x180:rate=30:duration=2",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(clip),
            ]
        )
        rec = audit_video(clip, profile="quick", lib_path=lib_path, hash_mode="sha256")
        assert rec["status"] == "ok", f"selftest failed: {rec['error']}"
        assert rec["ingest"]["frames_committed"] >= 50
        assert rec["ingest"]["audio"]["present"] is True
        print(_canonical_json(rec))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="chronos-audit", description="CHRONOS-SPECTRAL v3.4.0 Physics Integrity Auditor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Audit one or more video files or directories")
    p_scan.add_argument("paths", nargs="+")
    p_scan.add_argument("--profile", default="standard", choices=sorted(PROFILES))
    p_scan.add_argument("--workers", default="auto")
    p_scan.add_argument("--out", default="-")
    p_scan.add_argument("--format", dest="fmt", default="jsonl", choices=["jsonl", "json", "csv", "parquet"])
    p_scan.add_argument("--violations", default=None)
    p_scan.add_argument("--resume", action="store_true")
    p_scan.add_argument("--max-seconds", type=float, default=None)
    p_scan.add_argument("--no-audio", action="store_true")
    p_scan.add_argument("--hash", dest="hash_mode", default="sha256", choices=["sha256", "none"])
    p_scan.add_argument("--label", default=None)
    p_scan.add_argument("--fail-on", default=None, choices=["FAIL", "WARN"])
    p_scan.add_argument("--perf", action="store_true")
    p_scan.add_argument("--lib", default=None)

    p_cmp = sub.add_parser("compare", help="Compare real vs generated video corpora")
    p_cmp.add_argument("real_dir")
    p_cmp.add_argument("gen_dir")
    p_cmp.add_argument("--profile", default="standard", choices=sorted(PROFILES))
    p_cmp.add_argument("--workers", default="auto")
    p_cmp.add_argument("--out", default="-")
    p_cmp.add_argument("--records-out", default=None)
    p_cmp.add_argument("--max-seconds", type=float, default=None)
    p_cmp.add_argument("--no-audio", action="store_true")
    p_cmp.add_argument("--lib", default=None)

    sub.add_parser("laws", help="Print the 8-axis physics law registry as JSON")
    p_st = sub.add_parser("selftest", help="Run a self-contained lavfi streaming audit smoke test")
    p_st.add_argument("--lib", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "laws":
        print(json.dumps({"schema": SCHEMA_RECORD, "abi": CHRONOS_ABI_VERSION, "axes": LAW_REGISTRY, "profiles": sorted(PROFILES)}, indent=2))
        return 0
    if args.cmd == "selftest":
        return run_selftest(lib_path=args.lib)

    workers: int | str = "auto" if args.workers == "auto" else int(args.workers)
    if args.cmd == "scan":
        try:
            recs, _ = audit_directory(
                args.paths,
                profile=args.profile,
                workers=workers,
                out=args.out,
                fmt=args.fmt,
                violations_out=args.violations,
                resume=args.resume,
                lib_path=args.lib,
                audio=not args.no_audio,
                max_seconds=args.max_seconds,
                hash_mode=args.hash_mode,
                label=args.label,
                include_perf=args.perf,
            )
        except (EnvironmentError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        if any(r["status"] != "ok" for r in recs):
            return 4
        if args.fail_on == "FAIL" and any(r["verdict"]["grade"] == "FAIL" for r in recs):
            return 1
        if args.fail_on == "WARN" and any(r["verdict"]["grade"] in ("WARN", "FAIL") for r in recs):
            return 1
        return 0

    if args.cmd == "compare":
        try:
            compare_corpora(
                args.real_dir,
                args.gen_dir,
                profile=args.profile,
                workers=workers,
                out=args.out,
                records_out=args.records_out,
                lib_path=args.lib,
                audio=not args.no_audio,
                max_seconds=args.max_seconds,
            )
        except (EnvironmentError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
