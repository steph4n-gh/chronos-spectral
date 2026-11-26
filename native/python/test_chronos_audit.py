"""Zero-dependency unittest suite for CHRONOS-SPECTRAL v3.5.0 Native C ABI + chronos-audit SDK.

Runs cleanly on machines with or without NumPy / ffmpeg / pyarrow (ffmpeg-dependent tests
explicitly report `[SKIP]` when ffmpeg is absent, matching Gate N's macOS convention).
"""

from __future__ import annotations

import array
import ctypes
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chronos_audit as ca  # noqa: E402
import chronos_spectral as cs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "web" / "samples"
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _make_solid_frame(w: int, h: int, r: int = 0, g: int = 0, b: int = 0) -> bytearray:
    row = bytes([r, g, b, 255]) * w
    return bytearray(row * h)


def _make_textured_frame(w: int, h: int, dx: float = 0.0, dy: float = 0.0, gain: float = 1.0) -> bytearray:
    """Smooth multi-frequency 2D texture shifted by `(dx, dy)` (advected material field)."""
    out = bytearray(w * h * 4)
    idx = 0
    for y in range(h):
        yy = y - dy
        sy1 = math.sin(yy * 0.21)
        cy2 = math.cos(yy * 0.37)
        sy3 = math.sin(yy * 0.13)
        for x in range(w):
            xx = x - dx
            v = (
                0.5
                + 0.18 * math.sin(xx * 0.19) * cy2
                + 0.16 * math.cos(xx * 0.31) * sy1
                + 0.12 * math.sin((xx + yy) * 0.27)
                + 0.08 * math.cos((xx - yy) * 0.17) * sy3
            )
            c = max(16, min(239, int(round(v * 255.0 * gain))))
            out[idx] = c
            out[idx + 1] = c
            out[idx + 2] = c
            out[idx + 3] = 255
            idx += 4
    return out


def _make_moving_ball_frames(
    n: int,
    w: int = 320,
    h: int = 180,
    *,
    freeze_at: Optional[int] = None,
    bounce_frames: Sequence[int] = (),
) -> list[bytearray]:
    frames: list[bytearray] = []
    bg = _make_solid_frame(w, h, 24, 28, 36)
    x, y = 40.0, 90.0
    vx, vy = 4.2, -3.5
    last_frame = None
    for f in range(n):
        if freeze_at is not None and freeze_at <= f < freeze_at + 4 and last_frame is not None:
            frames.append(bytearray(last_frame))
            continue
        if f in bounce_frames:
            vy = -vy
        x += vx
        y += vy
        if x < 20 or x > w - 20:
            vx = -vx
            x = max(20.0, min(w - 20.0, x))
        if y < 20 or y > h - 20:
            vy = -vy
            y = max(20.0, min(h - 20.0, y))
        buf = bytearray(bg)
        cx, cy = int(round(x)), int(round(y))
        for dy in range(-8, 9):
            yy = cy + dy
            if 0 <= yy < h:
                for dx in range(-8, 9):
                    xx = cx + dx
                    if 0 <= xx < w and dx * dx + dy * dy <= 64:
                        off = (yy * w + xx) * 4
                        buf[off] = 240
                        buf[off + 1] = 220
                        buf[off + 2] = 60
                        buf[off + 3] = 255
        last_frame = buf
        frames.append(buf)
    return frames


class TestChronosAuditV34(unittest.TestCase):
    def test_abi_and_struct_sizes(self) -> None:
        self.assertEqual(ctypes.sizeof(cs.ChronosIngestMeta), 96)
        self.assertEqual(ctypes.sizeof(cs.ChronosAuditConfig), 40)
        self.assertEqual(ctypes.sizeof(cs.ChronosAuditAxis), 40)
        self.assertEqual(ctypes.sizeof(cs.ChronosAuditSpan), 64)
        self.assertEqual(ctypes.sizeof(cs.ChronosAuditSummary), 416)
        with cs.ChronosEngine() as eng:
            self.assertEqual(eng.abi_version, cs.CHRONOS_ABI_VERSION)
            self.assertTrue(eng.version_string.startswith("3.5.0"))

    def test_reset_and_multi_handle_isolation(self) -> None:
        frames = _make_moving_ball_frames(20, 320, 180)
        with cs.ChronosEngine(hd=False) as e1, cs.ChronosEngine(hd=False) as e2:
            # Dirty e1 first, then reset and verify bit-exact parity with fresh e2
            for fr in frames[:8]:
                e1.step_rgba(fr, 1.0 / 30.0)
            e1.reset()

            # Interleave steps on e1 and e2 with different audio pushes to prove zero global arena bleed
            a1 = array.array("f", [0.1 * math.sin(i * 0.05) for i in range(1600)])
            for k, fr in enumerate(frames):
                t_us = (k + 1) * 33333
                self.assertEqual(e1.push_audio(a1, t_us, 48000.0), 0)
                e1.step_rgba(fr, 1.0 / 30.0, pts_sec=k / 30.0)
                e2.step_rgba(fr, 1.0 / 30.0, pts_sec=k / 30.0)

            self.assertEqual(e1.telemetry()[:128], e2.telemetry()[:128])
            # e1 received audio, e2 did not (zero cross-handle bleed)
            v1 = e1.field_view(cs.FIELD_VIBRO_TELEMETRY)
            v2 = e2.field_view(cs.FIELD_VIBRO_TELEMETRY)
            self.assertGreater(int(v1[18]), 0)
            self.assertEqual(int(v2[18]), 0)

    def test_audio_edge_cases_and_u64_timestamps(self) -> None:
        with cs.ChronosEngine(hd=False) as eng:
            stg = eng.audio_staging_buffer(raw_bytes=True)
            self.assertEqual(len(stg), 4096 * 4)
            # Invalid lengths 0 and 4097 rejected
            self.assertEqual(eng.push_audio_block(0, 10000, 48000.0), 0)
            self.assertEqual(eng.push_audio_block(4097, 10000, 48000.0), 0)
            # Invalid sample rate rejected
            struct.pack_into("<f", stg, 0, 0.25)
            self.assertEqual(eng.push_audio_block(1, 10000, 100.0), 0)
            # Non-finite sample rejected
            struct.pack_into("<f", stg, 0, float("nan"))
            self.assertEqual(eng.push_audio_block(1, 10000, 48000.0), 0)
            # Valid 1-sample and 4096-sample blocks crossing 2^32 us (> 4295s)
            struct.pack_into("<f", stg, 0, 0.1)
            t_hi = 5_000_000_000
            self.assertEqual(eng.push_audio_block(1, t_hi, 48000.0), 1)
            ones = array.array("f", [0.05] * 4096)
            stg[:] = ones.tobytes()
            t_hi2 = t_hi + int(round(4096 * 1_000_000 / 48000.0))
            self.assertEqual(eng.push_audio_block(4096, t_hi2, 48000.0), 1)
            # Backward timestamp rejected
            self.assertEqual(eng.push_audio_block(16, t_hi, 48000.0), 0)

    def test_anti_gaming_decoys_and_portrait_letterbox(self) -> None:
        # Static black, static texture, and exposure-only flicker decoys must yield NA / reward_milli=-1000 when profiled
        black = [_make_solid_frame(108, 192, 0, 0, 0) for _ in range(18)]
        static_tex = [_make_textured_frame(160, 160, 0.0, 0.0) for _ in range(18)]
        flicker = [_make_textured_frame(160, 160, 0.0, 0.0, gain=0.85 if (i & 1) else 1.15) for i in range(18)]

        for name, seq, w, h in [
            ("black_portrait", black, 108, 192),
            ("static_square", static_tex, 160, 160),
            ("exposure_flicker", flicker, 160, 160),
        ]:
            with ca.ChronosAuditor(profile="quick", required_axes_mask=(1 << cs.AXIS_TRAJECTORY)) as aud:
                rec = aud.audit_frames(seq, src_w=w, src_h=h, fps=30.0, clip_name=name)
                self.assertEqual(rec["verdict"]["grade"], "NA", msg=f"{name} got {rec['verdict']}")
                self.assertEqual(rec["rlvr"]["reward_milli"], -1000, msg=f"{name} must not game RLVR reward")
                if name == "black_portrait":
                    self.assertEqual(rec["ingest"]["letterbox"], [109, 0, 101, 180])

    def test_dense_flow_fallback_analytic_and_no_pressure_loads(self) -> None:
        # Pure uniform advection of continuous unseeded texture (dx = 1.5 px/frame -> 45 px/s @ 30fps)
        # Sparse micro-swarm is OFF; WIND_MODE_DENSE_AUDIT (3) must solve READY_FIELD_ONLY with
        # provenance=DENSE_LK(3), zero Cp/CL/CD, low div_frac!
        w, h = 320, 180
        frames = [_make_textured_frame(w, h, dx=1.5 * f, dy=0.5 * f) for f in range(16)]
        with cs.ChronosEngine(hd=False) as eng:
            eng.set_swarm_mode(0)
            eng.set_hodge_mode(1)
            self.assertEqual(eng.set_wind_mode(cs.WIND_MODE_DENSE_AUDIT), 1)
            wt = eng.field_view(cs.FIELD_WIND_TELEMETRY)
            wp = eng.field_view(cs.FIELD_WIND_PLANES)
            ready_field = 0
            full_loads = 0
            last_div = 1.0
            for f, fr in enumerate(frames):
                eng.step_rgba(fr, 1.0 / 30.0, pts_sec=f / 30.0)
                st = int(wt[0])
                prov = int(wt[3])
                if st == 2:
                    ready_field += 1
                    self.assertEqual(prov, 3)  # WIND_PROV_DENSE_LK
                    last_div = struct.unpack("<f", struct.pack("<I", int(wt[22])))[0]
                    cl_bnd = struct.unpack("<f", struct.pack("<I", int(wt[16])))[0]
                    cl_cv = struct.unpack("<f", struct.pack("<I", int(wt[18])))[0]
                    self.assertEqual(cl_bnd, 0.0)
                    self.assertEqual(cl_cv, 0.0)
                elif st == 3:
                    full_loads += 1
            self.assertGreaterEqual(ready_field, 8, "Dense fallback should reach READY_FIELD_ONLY on advected texture")
            self.assertEqual(full_loads, 0, "Dense fallback must NEVER emit READY_FULL_LOADS")
            self.assertLess(last_div, 0.08, f"Uniform advection should have low divergence, got {last_div}")
            # Cp plane (first 2304 floats) must stay strictly zero on dense fallback
            cp_slice = wp[0] if hasattr(wp, "ndim") and wp.ndim == 3 else wp[:2304]
            self.assertEqual(max(abs(float(x)) for x in (cp_slice.flat if hasattr(cp_slice, "flat") else cp_slice)), 0.0)

    def test_frozen_mid_motion_stutter_and_av_lag_detection(self) -> None:
        # 1. Mid-motion duplicate-frame freeze under continuous-capture claim -> CODE_FROZEN_MID_MOTION
        stutter = _make_moving_ball_frames(45, 320, 180, freeze_at=22)
        with ca.ChronosAuditor(
            profile="quick",
            profile_flags=cs.AUDIT_CLAIM_CONTINUOUS_CAPTURE,
            required_axes_mask=(1 << cs.AXIS_TEMPORAL_CADENCE),
        ) as aud:
            rec = aud.audit_frames(stutter, src_w=320, src_h=180, fps=30.0, clip_name="stutter")
            codes = {sp["code"] for sp in rec["violations"]}
            self.assertIn("FROZEN_MID_MOTION", codes, f"Expected FROZEN_MID_MOTION in {rec['violations']}")

        # 2. AV impact sync: 3 unequal visual bounces with aligned vs +140ms delayed PCM clicks
        bounces = (18, 37, 59)
        clean_ball = _make_moving_ball_frames(75, 320, 180, bounce_frames=bounces)
        rate = 48000
        n_samp = int(round((75 / 30.0) * rate))

        def _make_click_pcm(offset_ms: float) -> array.array:
            pcm = array.array("f", [0.001 * math.sin(i * 0.02) for i in range(n_samp)])
            for bf in bounces:
                t_s = bf / 30.0 + offset_ms / 1000.0
                center = int(round(t_s * rate))
                for k in range(max(0, center - 120), min(n_samp, center + 360)):
                    env = math.exp(-abs(k - center) / 80.0)
                    pcm[k] = float(0.85 * env * math.sin((k - center) * 0.25))
            return pcm

        with ca.ChronosAuditor(profile="av_impact") as aud:
            rec_ok = aud.audit_frames(
                clean_ball, src_w=320, src_h=180, fps=30.0, audio_pcm=_make_click_pcm(0.0), clip_name="av_aligned"
            )
            rec_lag = aud.audit_frames(
                clean_ball, src_w=320, src_h=180, fps=30.0, audio_pcm=_make_click_pcm(140.0), clip_name="av_delayed_140ms"
            )
            self.assertNotIn("AV_IMPACT_LAG", {sp["code"] for sp in rec_ok["violations"]})
            self.assertIn("AV_IMPACT_LAG", {sp["code"] for sp in rec_lag["violations"]})

    def test_streaming_ffmpeg_real_clips_and_determinism(self) -> None:
        if not HAS_FFMPEG:
            print("  [SKIP] ffmpeg not available; skipping streaming MP4 + multiprocessing tests")
            return

        # 1. Selftest lavfi pipeline
        self.assertEqual(ca.run_selftest(), 0)

        # 2. Real-clip negative controls: real Daphnia laser PIV under wild `standard` profile MUST NOT false-fail
        daphnia = SAMPLES / "real_daphnia_laser_piv.mp4"
        juggling = SAMPLES / "juggling_ballistic.mp4"
        crack = SAMPLES / "elasto_hidden_crack_panel.mp4"
        if daphnia.exists() and juggling.exists():
            rec_d = ca.audit_video(daphnia, profile="standard", max_seconds=3.0)
            self.assertEqual(rec_d["status"], "ok")
            self.assertNotEqual(
                rec_d["verdict"]["grade"],
                "FAIL",
                f"Real 3D laser PIV must not false-FAIL under wild standard profile: {rec_d['verdict']}",
            )
            rec_j1 = ca.audit_video(juggling, profile="quick", max_seconds=3.0)
            rec_j2 = ca.audit_video(juggling, profile="quick", max_seconds=3.0)
            self.assertEqual(rec_j1["record_sha256"], rec_j2["record_sha256"])

            # 3. workers=1 vs workers=2 must produce byte-identical JSONL
            with tempfile.TemporaryDirectory(prefix="chronos_mp_det_") as td:
                in_dir = Path(td) / "in"
                in_dir.mkdir()
                shutil.copy2(juggling, in_dir / "a_juggling.mp4")
                shutil.copy2(SAMPLES / "carom_billiards_impulses.mp4", in_dir / "b_carom.mp4")
                out1 = Path(td) / "w1.jsonl"
                out2 = Path(td) / "w2.jsonl"
                ca.audit_directory(in_dir, profile="quick", workers=1, out=out1, max_seconds=2.0)
                ca.audit_directory(in_dir, profile="quick", workers=2, out=out2, max_seconds=2.0)
                self.assertEqual(out1.read_bytes(), out2.read_bytes(), "workers=1 vs workers=2 JSONL must be byte-identical")

        # 4. Native calibrated RS elastography via new per-engine audio + scan_dir metadata!
        if crack.exists():
            raw_v = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(crack),
                    "-vf",
                    "scale=640:360",
                    "-frames:v",
                    "140",
                    "-pix_fmt",
                    "rgba",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ]
            )
            raw_a = subprocess.check_output(
                ["ffmpeg", "-v", "error", "-i", str(crack), "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1"]
            )
            a_arr = array.array("f")
            a_arr.frombytes(raw_a)
            with cs.ChronosEngine(hd=True) as eng:
                eng.set_swarm_mode(0)
                self.assertEqual(eng.set_rs_calib(2, 35.0, 200.0, 48000.0, 5000.0), 1)
                self.assertEqual(eng.set_elasto_mode(1), 1)
                et = eng.field_view(cs.FIELD_ELASTO_TELEMETRY)
                fb = 640 * 360 * 4
                a_cur = 0
                t0_us = 1_000_000
                ready = 0
                best_coh = 0.0
                f0 = f1 = 0.0
                for f in range(140):
                    t_src_us = t0_us + int(round(f * 1_000_000 / 30.0))
                    s_tgt = min(len(a_arr), int(round(((f + 1) / 30.0 + 0.025) * 48000.0)))
                    while a_cur < s_tgt:
                        ch = min(2048, s_tgt - a_cur)
                        end_s = a_cur + ch
                        t_end_us = t0_us + int(round(end_s * 1_000_000 / 48000.0))
                        stg = eng.audio_staging_buffer(raw_bytes=True)
                        b_chunk = a_arr[a_cur:end_s].tobytes()
                        stg[: len(b_chunk)] = b_chunk
                        self.assertEqual(eng.push_audio_block(ch, t_end_us, 48000.0, 0.0), 1)
                        a_cur = end_s
                    imeta = cs.ChronosIngestMeta(
                        struct_size=ctypes.sizeof(cs.ChronosIngestMeta),
                        flags=cs.INGEST_PTS_KNOWN | cs.INGEST_GEOM_KNOWN,
                        pts_us=t_src_us,
                        epoch=1,
                        seq=f + 1,
                        age_ms=-1.0,
                        src_w=640,
                        src_h=360,
                        crop_x=0,
                        crop_y=0,
                        crop_w=640,
                        crop_h=360,
                        dst_x=0,
                        dst_y=0,
                        dst_w=640,
                        dst_h=360,
                        orientation=0,
                        scan_dir=0,
                        gap_mask=0,
                        native_1to1=1,
                    )
                    self.assertEqual(eng.set_ingest_meta(imeta), 0)
                    buf = eng.ingest_buffer(raw=True)
                    buf[:] = raw_v[f * fb : (f + 1) * fb]
                    eng.step_in_place(1.0 / 30.0)
                    if int(et[0]) == 3:
                        ready += 1
                        coh = struct.unpack("<f", struct.pack("<I", int(et[15])))[0]
                        if coh >= best_coh:
                            best_coh = coh
                            f0 = struct.unpack("<f", struct.pack("<I", int(et[8])))[0]
                            f1 = struct.unpack("<f", struct.pack("<I", int(et[9])))[0]
                self.assertGreaterEqual(ready, 75, f"Native RS elastography ready={ready}")
                self.assertLess(abs(f0 - 380.0), 3.0)
                self.assertLess(abs(f1 - 860.0), 3.0)
                self.assertGreaterEqual(best_coh, 0.85)

    def test_compare_corpora_closed_form(self) -> None:
        real_rec = {
            "clip": {"path": "r1.mp4", "sha256": "a", "bytes": 1, "label": "real"},
            "status": "ok",
            "verdict": {"grade": "PASS", "integrity": 0.96, "coverage": 0.9, "applicable_axes": 2, "violated_axes": 0, "warned_axes": 0, "violations": 0},
            "axes": [
                {"id": s, "applicable": i < 2, "pass": True if i < 2 else None, "score": 0.96 if i < 2 else None}
                for i, s in enumerate(cs.AXIS_SLUGS)
            ],
            "violations": [],
        }
        gen_rec = {
            "clip": {"path": "g1.mp4", "sha256": "b", "bytes": 1, "label": "generated"},
            "status": "ok",
            "verdict": {"grade": "FAIL", "integrity": 0.35, "coverage": 0.9, "applicable_axes": 2, "violated_axes": 1, "warned_axes": 0, "violations": 1},
            "axes": [
                {"id": s, "applicable": i < 2, "pass": (i == 1) if i < 2 else None, "score": (0.2 if i == 0 else 0.5) if i < 2 else None}
                for i, s in enumerate(cs.AXIS_SLUGS)
            ],
            "violations": [{"axis": "trajectory", "code": "G_SIGN_FLIP", "severity": 0.95, "f0": 10, "f1": 15}],
        }
        rep = ca.compare_records([real_rec], [gen_rec])
        self.assertEqual(rep["schema"], ca.SCHEMA_COMPARE)
        self.assertEqual(rep["separation"]["auroc"], 1.0)
        self.assertEqual(rep["separation"]["ks"], 1.0)
        self.assertEqual(rep["top_offenders"]["trajectory"][0]["code"], "G_SIGN_FLIP")


if __name__ == "__main__":
    unittest.main(verbosity=2)
