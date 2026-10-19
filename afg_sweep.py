#!/usr/bin/env python3
"""
afg_sweep.py - ECE266B Lab 2
Steps a Tektronix AFG3022C through a log-spaced list of (near-)coherent
frequencies over SCPI (PyVISA) and preserves one completed PLABS .bin
record per frequency.  PLABS runs in continuous capture on its own; this
script never triggers the ADC.

Usage
  python afg_sweep.py --mode idn                    # connect, print *IDN?
  python afg_sweep.py --mode single --freq 2000     # one (snapped) frequency
  python afg_sweep.py --mode sweep                  # full 0.1-250 kHz sweep
  python afg_sweep.py --mode sweep --subset sfdr    # only f <= 20 kHz
  add --dry-run to run everything with a simulated AFG + simulated PLABS

Things to edit at the bench: VISA_ADDRESS, EXPORT_FILE, SETTLE_S.
"""
import argparse
import csv
import os
import shutil
import threading
import time
from datetime import datetime

import numpy as np

# ----------------------------- CONFIG ---------------------------------------
FS = 1_000_000            # ADC sampling rate set in PLABS (Hz)
N_SAMPLES = 262_144       # samples per record (2^18)
ADC_BITS = 16
AMPLITUDE_VPP = 4.9       # generator programmed amplitude (keep fixed all sweep)
OFFSET_V = 2.5            # generator DC offset (keep fixed all sweep)

VISA_ADDRESS = None       # e.g. "USB0::0x0699::0x0353::C0XXXXX::INSTR"; None = auto-find
EXPORT_FILE = r"C:\PLABS\export\data.bin"   # file PLABS keeps rewriting (BENCH: fix path)
OUT_DIR = "sweep_data"    # preserved per-frequency records + log go here

SETTLE_S = 1.0            # must exceed record time (N/FS = 0.26 s) + AFG/filter settling
FILE_TIMEOUT_S = 30.0     # max wait for a fresh PLABS file per attempt
POLL_S = 0.2
MAX_ATTEMPTS = 3
TONE_TOL_BINS = 5         # tone-check tolerance (bins) + 1 % of f, covers clock mismatch
# -----------------------------------------------------------------------------


# ------------------------- frequency planning --------------------------------
def coherent_freq(f_target, fs=FS, n=N_SAMPLES):
    """Snap f_target to M*fs/N with M odd (odd M is coprime with N = 2^18)."""
    m_ideal = f_target * n / fs
    m = max(int(round(m_ideal)), 1)
    if m % 2 == 0:
        m = m - 1 if abs(m - 1 - m_ideal) <= abs(m + 1 - m_ideal) and m > 1 else m + 1
    return m, m * fs / n


def build_sweep(subset="full"):
    """10 pts/decade from 100 Hz; full list ends with a 250 kHz endpoint."""
    targets = [10 ** (2 + k / 10) for k in range(34)]      # 100 Hz ... ~199.5 kHz
    if subset == "full":
        targets.append(250e3)
    elif subset == "sfdr":
        targets = [t for t in targets if t <= 20e3 * 1.001]
    plan, seen = [], set()
    for t in targets:
        m, f = coherent_freq(t)
        if m not in seen:
            seen.add(m)
            plan.append((t, m, f))
    return plan


# ----------------------------- instruments -----------------------------------
class MockAFG:
    """Stand-in for the AFG: prints commands, tracks frequency for MockPLABS."""
    def __init__(self):
        self.freq = 0.0
        self.log = []

    def write(self, cmd):
        print(f"  [mock AFG] <- {cmd}")
        if "FREQuency:FIXed" in cmd and not cmd.endswith("?"):
            self.freq = float(cmd.split()[-1].upper().replace("HZ", ""))

    def query(self, cmd):
        if cmd == "*IDN?":
            return "TEKTRONIX,AFG3022C,MOCK,SCPI:99.0"
        if cmd.startswith("SYSTem:ERRor"):
            return '0,"No error"'
        if cmd.startswith("SOURce1:FREQuency:FIXed?"):
            return f"{self.freq}"
        return "0"

    def close(self):
        pass


class MockPLABS(threading.Thread):
    """Continuously rewrites a synthetic .bin, like PLABS continuous capture."""
    def __init__(self, afg, path, period=0.5):
        super().__init__(daemon=True)
        self.afg, self.path, self.period = afg, path, period
        self.stop = threading.Event()

    def run(self):
        rng = np.random.default_rng(0)
        while not self.stop.is_set():
            f = self.afg.freq
            if f > 0:
                t = np.arange(N_SAMPLES) / FS
                gain = 1 / np.sqrt(1 + (f / 16e3) ** 2)            # ~16 kHz input filter
                peak = (AMPLITUDE_VPP / 2) / 5.0 * 2 ** ADC_BITS * gain
                x = 2 ** (ADC_BITS - 1) + peak * np.sin(2 * np.pi * f * t) + rng.normal(0, 3, N_SAMPLES)
                codes = np.clip(np.round(x), 0, 2 ** ADC_BITS - 1).astype("<u2")
                raw = np.zeros(N_SAMPLES * 4, dtype="<u2")
                raw[::4] = codes
                raw.tofile(self.path)
            self.stop.wait(self.period)


def open_afg(dry_run):
    if dry_run:
        return MockAFG()
    import pyvisa
    rm = pyvisa.ResourceManager()
    addr = VISA_ADDRESS
    if addr is None:
        found = [r for r in rm.list_resources() if "0x0699" in r or "AFG" in r.upper()]
        print("VISA resources:", rm.list_resources())
        if not found:
            raise RuntimeError("AFG not found - check cable/VISA driver, or set VISA_ADDRESS")
        addr = found[0]
    afg = rm.open_resource(addr)
    afg.timeout = 10_000
    afg.write_termination = "\n"
    afg.read_termination = "\n"
    return afg


def send(afg, cmd):
    """Write a command, then check the instrument error queue."""
    afg.write(cmd)
    err = afg.query("SYSTem:ERRor?").strip()
    if not err.startswith("0"):
        raise RuntimeError(f"AFG error after '{cmd}': {err}")


def setup_afg(afg):
    send(afg, "*CLS")
    send(afg, "OUTPut1:STATe OFF")
    send(afg, "OUTPut1:IMPedance INFinity")          # high-Z load (verify in programmer manual)
    send(afg, "SOURce1:FUNCtion:SHAPe SINusoid")
    send(afg, f"SOURce1:VOLTage:LEVel:IMMediate:AMPLitude {AMPLITUDE_VPP}VPP")
    send(afg, f"SOURce1:VOLTage:LEVel:IMMediate:OFFSet {OFFSET_V}V")


def set_frequency(afg, f):
    send(afg, f"SOURce1:FREQuency:FIXed {f:.6f}HZ")
    readback = float(afg.query("SOURce1:FREQuency:FIXed?"))
    if abs(readback - f) > 1e-6 * f + 1e-6:
        raise RuntimeError(f"Frequency readback {readback} != requested {f}")


# --------------------------- capture handling --------------------------------
def wait_for_fresh_file(path, t_after, timeout=FILE_TIMEOUT_S):
    """Return once `path` exists, is nonempty, has mtime >= t_after, and has
    stopped changing (same size + mtime across two polls)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(path):
            s1 = os.stat(path)
            if s1.st_size > 0 and s1.st_mtime >= t_after:
                time.sleep(POLL_S)
                s2 = os.stat(path)
                if (s1.st_size, s1.st_mtime_ns) == (s2.st_size, s2.st_mtime_ns):
                    return True
        time.sleep(POLL_S)
    return False


def load_codes(path):
    raw = np.fromfile(path, dtype="<u2")
    return raw[::4]                                  # one ADC code per 4 words


def check_tone(codes, f_expected):
    """Quick sanity check of an extracted record. Not the final analysis."""
    n = len(codes)
    info = {"n": n, "ok": False, "f_peak": np.nan, "min": int(codes.min()),
            "max": int(codes.max()), "clipped": False, "amp_dbfs": np.nan}
    if n < 1024:
        return info
    x = codes.astype(float)
    full = 2 ** ADC_BITS
    info["clipped"] = bool(codes.min() <= 0 or codes.max() >= full - 1)
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.blackman(n)))
    k = int(np.argmax(spec[1:]) + 1)
    info["f_peak"] = k * FS / n
    info["amp_dbfs"] = 20 * np.log10(np.sqrt(2) * x.std() / (full / 2))   # rough quick-look
    tol = TONE_TOL_BINS * FS / n + 0.01 * f_expected
    info["ok"] = abs(info["f_peak"] - f_expected) <= tol and n == N_SAMPLES
    return info


def capture_point(afg, f, export_file, out_dir):
    """Set frequency, wait, grab a fresh verified record, save it. Returns row dict."""
    fname = f"sweep_{f:012.3f}Hz.bin".replace(".", "p", 1)
    dest = os.path.join(out_dir, fname)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        set_frequency(afg, f)
        t_set = time.time()
        t_after = t_set + SETTLE_S            # file must have been written after settling
        if not wait_for_fresh_file(export_file, t_after):
            print(f"  attempt {attempt}: no fresh file within timeout")
            continue
        tmp = dest + ".tmp"
        shutil.copy2(export_file, tmp)
        codes = load_codes(tmp)
        info = check_tone(codes, f)
        if info["ok"] and not info["clipped"]:
            os.replace(tmp, dest)
            return {"status": "OK", "file": fname, **info, "attempts": attempt}
        print(f"  attempt {attempt}: tone check failed "
              f"(n={info['n']}, f_peak={info['f_peak']:.1f}, clipped={info['clipped']})")
        os.remove(tmp)
    return {"status": "FAILED", "file": "", "n": 0, "f_peak": np.nan, "min": 0, "max": 0,
            "clipped": False, "amp_dbfs": np.nan, "attempts": MAX_ATTEMPTS}


# --------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["idn", "single", "sweep"], default="sweep")
    ap.add_argument("--freq", type=float, default=2000.0, help="target Hz for --mode single")
    ap.add_argument("--subset", choices=["full", "sfdr"], default="full")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    export_file = EXPORT_FILE
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.dry_run:
        export_file = os.path.join(OUT_DIR, "_mock_plabs_export.bin")
        if os.path.exists(export_file):
            os.remove(export_file)

    afg = open_afg(args.dry_run)
    plabs = None
    try:
        print("IDN:", afg.query("*IDN?").strip())
        if args.mode == "idn":
            return
        if args.dry_run:
            plabs = MockPLABS(afg, export_file)
            plabs.start()

        setup_afg(afg)
        if args.mode == "single":
            m, f = coherent_freq(args.freq)
            plan = [(args.freq, m, f)]
        else:
            plan = build_sweep(args.subset)
        print(f"{len(plan)} frequencies; amplitude {AMPLITUDE_VPP} Vpp, offset {OFFSET_V} V")

        first = True
        log_path = os.path.join(OUT_DIR, f"sweep_log_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(log_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["idx", "target_hz", "M", "f_hz", "status", "file", "n_samples",
                        "f_peak_hz", "code_min", "code_max", "clipped",
                        "amp_dbfs_quicklook", "attempts", "timestamp"])
            for i, (target, m, f) in enumerate(plan):
                print(f"[{i + 1}/{len(plan)}] target {target:.2f} Hz -> M={m}, f={f:.3f} Hz")
                if first:
                    set_frequency(afg, f)
                    send(afg, "OUTPut1:STATe ON")
                    first = False
                row = capture_point(afg, f, export_file, OUT_DIR)
                print(f"  {row['status']}  {row['file']}  amp~{row['amp_dbfs']:.2f} dBFS")
                w.writerow([i, f"{target:.4f}", m, f"{f:.6f}", row["status"], row["file"],
                            row["n"], f"{row['f_peak']:.2f}", row["min"], row["max"],
                            row["clipped"], f"{row['amp_dbfs']:.3f}", row["attempts"],
                            datetime.now().isoformat(timespec="seconds")])
                fh.flush()
        print("Done. Log:", log_path)
    finally:
        try:
            if args.mode != "idn":
                afg.write("OUTPut1:STATe OFF")
        except Exception:
            pass
        if plabs:
            plabs.stop.set()
        afg.close()


if __name__ == "__main__":
    main()