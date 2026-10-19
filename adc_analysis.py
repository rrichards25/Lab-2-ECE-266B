#!/usr/bin/env python3
"""
adc_analysis.py - ECE266B Lab 2 post-processing

  analyze_spectrum()      SNR, SNDR, THD, SFDR, ENOB, amplitude (dBFS), harmonics 2..H
                          with aliased locations
  plot_spectrum()         annotated spectrum plot (harmonics 2-5) for the report
  histogram_inl_dnl()     sine-histogram INL/DNL (cumulative-histogram method)
  process_sweep()         run the spectral analysis over every file in a sweep log
                          and plot SNR/SNDR/amplitude and SFDR vs frequency
  selftest()              synthetic data with KNOWN answers to validate all of the above

Usage
  python adc_analysis.py selftest
  python adc_analysis.py spectrum --file data.bin --fs 1e6 [--window bh7|rect] [--plot out.png]
  python adc_analysis.py hist     --file big.bin  --bits 16 [--fit best|endpoint]
  python adc_analysis.py sweep    --log sweep_data/sweep_log_XXXX.csv
"""
import argparse
import csv
import os

import numpy as np

ADC_BITS = 16
FS_DEFAULT = 1_000_000

# 7-term Blackman-Harris (Albrecht) - same family PLABS uses ("7 Term B-Harris")
_BH7 = [0.27105140069342, 0.43329793923448, 0.21812299954311, 0.06592544638803,
        0.01081174209837, 0.00077658482522, 0.00001388721735]


# ------------------------------- I/O -----------------------------------------
def load_codes(path):
    """PLABS .bin: little-endian uint16, one ADC code per 4 words."""
    return np.fromfile(path, dtype="<u2")[::4]


# ----------------------------- spectral --------------------------------------
def make_window(name, n):
    """Returns (window, main-lobe half-width in bins used for integration)."""
    if name == "rect":
        return np.ones(n), 0
    if name == "bh7":
        i = np.arange(n)
        w = sum(((-1) ** k) * a * np.cos(2 * np.pi * k * i / n) for k, a in enumerate(_BH7))
        return w, 7
    raise ValueError("window must be 'rect' or 'bh7'")


def fold_bin(k, n):
    """Fold an (unfolded, possibly fractional) bin index into [0, n/2]."""
    k = k % n
    return n - k if k > n / 2 else k


def analyze_spectrum(codes, fs=FS_DEFAULT, n_harm=9, window="bh7", bits=ADC_BITS):
    """
    Definitions (all from one-sided power spectrum, ratios are consistent):
      signal   = power integrated over the fundamental main lobe (+-hw bins)
      harmonic = power integrated over each harmonic main lobe, h = 2..n_harm,
                 located at fold(h*f0) so aliased harmonics are found
      noise    = everything else except the DC lobe, fundamental lobe, harmonic lobes
      SNR  = S/N          SNDR = S/(N + sum of harmonics)      THD = sum(H)/S
      SFDR = fundamental peak bin / largest other bin (DC lobe excluded, harmonics INCLUDED)
      ENOB = (SNDR - 1.76)/6.02 ; enob_fs additionally corrects for amplitude below full scale
    """
    x = np.asarray(codes, dtype=float)
    n = len(x)
    x = x - x.mean()
    w, hw = make_window(window, n)
    X = np.fft.rfft(x * w)
    P = np.abs(X) ** 2
    nb = len(P)
    sw = w.sum()
    sw2 = (w ** 2).sum()

    # fundamental: strongest bin outside the DC lobe, fractional bin by power centroid
    lo = hw + 1
    k0 = lo + int(np.argmax(P[lo:]))
    sl = slice(max(k0 - hw, 0), min(k0 + hw, nb - 1) + 1)
    kc = float((np.arange(nb)[sl] * P[sl]).sum() / P[sl].sum())
    f0 = kc * fs / n
    p_fund = P[sl].sum()
    amp = np.sqrt(4 * p_fund / (n * sw2))              # peak amplitude in codes
    amp_dbfs = 20 * np.log10(amp / 2 ** (bits - 1))

    claimed = np.zeros(nb, dtype=bool)
    claimed[: hw + 1] = True                            # DC lobe
    claimed[sl] = True                                  # fundamental lobe
    excl_spur = claimed.copy()                          # for SFDR (harmonics NOT excluded)

    harmonics = []
    search = 3 if hw else 1
    for h in range(2, n_harm + 1):
        kh_pred = fold_bin(h * kc, n)
        kh = int(round(kh_pred))
        kh = min(max(kh, 0), nb - 1)
        a, b = max(kh - search, 0), min(kh + search, nb - 1) + 1
        kh = a + int(np.argmax(P[a:b]))
        rec = {"h": h, "f_unfolded": h * f0, "f_folded": kh * fs / n, "bin": kh,
               "aliased": h * f0 > fs / 2, "valid": True, "p": 0.0, "dbc": np.nan}
        if claimed[kh]:                                 # sits on DC / fundamental / earlier harmonic
            rec["valid"] = False
        else:
            lobe = np.arange(max(kh - hw, 0), min(kh + hw, nb - 1) + 1)
            lobe = lobe[~claimed[lobe]]
            rec["p"] = P[lobe].sum()
            rec["dbc"] = 10 * np.log10(rec["p"] / p_fund)
            claimed[lobe] = True
        harmonics.append(rec)

    p_harm = sum(r["p"] for r in harmonics if r["valid"])
    p_noise = P[~claimed].sum()
    snr = 10 * np.log10(p_fund / p_noise)
    sndr = 10 * np.log10(p_fund / (p_noise + p_harm))
    thd = 10 * np.log10(p_harm / p_fund) if p_harm > 0 else -np.inf

    # SFDR: largest bin outside DC and fundamental lobes (harmonics count as spurs)
    Pm = np.where(excl_spur, 0.0, P)
    ks = int(np.argmax(Pm))
    sfdr = 10 * np.log10(P[k0] / Pm[ks])
    spur_is_harm = any(r["valid"] and abs(ks - r["bin"]) <= hw for r in harmonics)

    dbfs_spec = 20 * np.log10(np.maximum(2 * np.abs(X) / sw, 1e-12) / 2 ** (bits - 1))
    return {
        "n": n, "fs": fs, "window": window, "f0": f0, "amp_dbfs": amp_dbfs,
        "snr": snr, "sndr": sndr, "thd": thd, "sfdr_dbc": sfdr,
        "sfdr_dbfs": amp_dbfs - sfdr,                   # spur level in dBFS
        "spur_freq": ks * fs / n, "spur_is_harmonic": spur_is_harm,
        "enob": (sndr - 1.76) / 6.02, "enob_fs": (sndr - 1.76 - amp_dbfs) / 6.02,
        "harmonics": harmonics, "freqs": np.arange(nb) * fs / n, "spec_dbfs": dbfs_spec,
    }


def plot_spectrum(res, path=None, title=None, n_annot=5):
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, s = res["freqs"] / 1e3, res["spec_dbfs"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(f, s, lw=0.5, color="tab:blue")
    k0 = int(round(res["f0"] / res["fs"] * res["n"]))
    ax.annotate(f"H1 {res['f0'] / 1e3:.3f} kHz\n{res['amp_dbfs']:.2f} dBFS",
                (res["f0"] / 1e3, s[k0]), xytext=(8, -4), textcoords="offset points", fontsize=8)
    for r in res["harmonics"]:
        if r["h"] > n_annot:
            break
        y = s[r["bin"]]
        tag = f"H{r['h']}" + (f" (alias of {r['f_unfolded'] / 1e3:.1f} kHz)" if r["aliased"] else "")
        ax.plot(r["f_folded"] / 1e3, y, "rv", ms=5)
        ax.annotate(f"{tag}\n{r['f_folded'] / 1e3:.2f} kHz, {r['dbc']:.1f} dBc",
                    (r["f_folded"] / 1e3, y), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Amplitude (dBFS)")
    ax.set_ylim(-160, 10)
    ax.grid(alpha=0.3)
    ax.set_title(title or (f"SNR {res['snr']:.2f} dB  SNDR {res['sndr']:.2f} dB  "
                           f"THD {res['thd']:.2f} dB  SFDR {res['sfdr_dbc']:.2f} dBc"), fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


# --------------------------- histogram INL / DNL -----------------------------
def histogram_inl_dnl(codes, bits=ADC_BITS, fit="best"):
    """
    Cumulative-histogram method for a sine-wave input (offset and amplitude cancel):
      p_k = (#samples with code < k) / N          fraction below transition k
      T_k = C - A cos(pi * p_k)                   transition level into code k
    Because T_k is linear in u_k = -cos(pi p_k), fit u_k vs k (best-fit line or endpoints):
      LSB   = 1/slope of the fit (gain removed), offset removed by the intercept
      DNL_k = (u_{k+1} - u_k)/slope - 1           code width error
      INL_k = (u_k - line(k))/slope               transition-level deviation from the line
    Tested range: transitions from the first to the last populated code. The first and
    last populated codes are only partly covered by the sine peaks, so their widths are
    not reported. Zero-count codes INSIDE the range are kept (they give DNL = -1).
    """
    codes = np.asarray(codes).astype(np.int64)
    ntot = len(codes)
    hist = np.bincount(codes, minlength=2 ** bits)
    pop = np.nonzero(hist)[0]
    kmin, kmax = int(pop[0]), int(pop[-1])
    ks = np.arange(kmin + 1, kmax + 1)                  # transitions into code k
    p = np.cumsum(hist)[ks - 1] / ntot
    u = -np.cos(np.pi * p)
    if fit == "best":
        slope, icpt = np.polyfit(ks, u, 1)
    elif fit == "endpoint":
        slope = (u[-1] - u[0]) / (ks[-1] - ks[0])
        icpt = u[0] - slope * ks[0]
    else:
        raise ValueError("fit must be 'best' or 'endpoint'")
    inl = (u - (slope * ks + icpt)) / slope
    dnl = np.diff(u) / slope - 1
    return {
        "ks": ks, "inl": inl, "codes_dnl": ks[:-1], "dnl": dnl,
        "n_samples": ntot, "code_range": (kmin, kmax), "fit": fit,
        "empty_codes_in_range": int((hist[kmin + 1:kmax] == 0).sum()),
        "dnl_min": float(dnl.min()), "dnl_max": float(dnl.max()),
        "inl_min": float(inl.min()), "inl_max": float(inl.max()),
        "amp_est_codes": 1 / slope, "offset_est_codes": -icpt / slope,
        "min_counts_in_range": int(hist[kmin + 1:kmax].min()),
    }


def plot_inl_dnl(res, path=None, title=None):
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    a1.plot(res["codes_dnl"], res["dnl"], lw=0.5)
    a1.set_ylabel("DNL (LSB)")
    a2.plot(res["ks"], res["inl"], lw=0.7, color="tab:orange")
    a2.set_ylabel(f"INL (LSB, {res['fit']} fit)")
    a2.set_xlabel("Code")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    a1.set_title(title or (f"{res['n_samples']:,} samples, codes {res['code_range'][0]}-"
                           f"{res['code_range'][1]}  DNL [{res['dnl_min']:.2f}, {res['dnl_max']:.2f}]  "
                           f"INL [{res['inl_min']:.2f}, {res['inl_max']:.2f}]"), fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


# ------------------------------- sweep ---------------------------------------
def process_sweep(log_csv, fs=FS_DEFAULT, window="bh7", n_harm=9, bits=ADC_BITS):
    """Analyze every OK row of the afg_sweep.py log; write results CSV and two plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = os.path.dirname(os.path.abspath(log_csv))
    rows = []
    with open(log_csv) as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "OK":
                print("skipping", r["f_hz"], r["status"])
                continue
            res = analyze_spectrum(load_codes(os.path.join(folder, r["file"])),
                                   fs, n_harm, window, bits)
            rows.append((float(r["f_hz"]), res))
            print(f"{r['f_hz']:>12} Hz  amp {res['amp_dbfs']:7.2f} dBFS  SNR {res['snr']:6.2f}  "
                  f"SNDR {res['sndr']:6.2f}  SFDR {res['sfdr_dbc']:6.2f}")
    rows.sort(key=lambda t: t[0])
    out_csv = os.path.join(folder, "sweep_results.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f_hz", "f0_measured_hz", "amp_dbfs", "snr_db", "sndr_db", "thd_db",
                    "sfdr_dbc", "enob"])
        for f, r in rows:
            w.writerow([f, f"{r['f0']:.3f}", f"{r['amp_dbfs']:.3f}", f"{r['snr']:.3f}",
                        f"{r['sndr']:.3f}", f"{r['thd']:.3f}", f"{r['sfdr_dbc']:.3f}",
                        f"{r['enob']:.3f}"])
    f = np.array([t[0] for t in rows]) / 1e3
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.semilogx(f, [t[1]["snr"] for t in rows], "o-", label="SNR (dB)")
    ax.semilogx(f, [t[1]["sndr"] for t in rows], "s-", label="SNDR (dB)")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("dB")
    ax2 = ax.twinx()
    ax2.semilogx(f, [t[1]["amp_dbfs"] for t in rows], "k--", label="ADC input amplitude (dBFS)")
    ax2.set_ylabel("Amplitude (dBFS)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "sweep_snr_sndr.png"), dpi=150)
    plt.close(fig)
    sub = [t for t in rows if t[0] <= 20.5e3]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx([t[0] / 1e3 for t in sub], [t[1]["sfdr_dbc"] for t in sub], "o-")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("SFDR (dBc)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "sweep_sfdr.png"), dpi=150)
    plt.close(fig)
    print("wrote", out_csv, "+ 2 plots in", folder)


# ------------------------------ self test ------------------------------------
def _synth_tone(f, n=262_144, fs=FS_DEFAULT, amp_dbfs=-0.5, snr_db=93.0, hd_dbc=None,
                bits=ADC_BITS, seed=1):
    """Ideal ADC + white noise + injected harmonics with KNOWN levels."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    a = 10 ** (amp_dbfs / 20) * 2 ** (bits - 1)
    x = a * np.sin(2 * np.pi * f * t + 0.3)
    for h, dbc in (hd_dbc or {}).items():
        x += a * 10 ** (dbc / 20) * np.sin(2 * np.pi * h * f * t + rng.uniform(0, 6.28))
    sigma = np.sqrt((a ** 2 / 2) / 10 ** (snr_db / 10))
    x += rng.normal(0, sigma, n) + 2 ** (bits - 1)
    codes = np.clip(np.round(x), 0, 2 ** bits - 1).astype(np.int64)
    expected_snr = 10 * np.log10((a ** 2 / 2) / (sigma ** 2 + 1 / 12))      # incl. quantization
    return codes, a, expected_snr


def selftest(plot_dir=None):
    ok = True

    def check(name, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:<28} got {got:9.3f}  want {want:9.3f}  (tol {tol})")

    hd = {2: -105.0, 3: -112.0, 4: -120.0, 5: -118.0}
    thd_true = 10 * np.log10(sum(10 ** (v / 10) for v in hd.values()))
    for label, f, window in (("coherent f (M=523), rect", 523 * FS_DEFAULT / 262144, "rect"),
                             ("coherent f (M=523), BH7", 523 * FS_DEFAULT / 262144, "bh7"),
                             ("NON-coherent 2000.0 Hz, BH7", 2000.0, "bh7")):
        print(f"[spectrum] {label}")
        codes, a, snr_true = _synth_tone(f, hd_dbc=hd)
        r = analyze_spectrum(codes, window=window)
        sndr_true = 10 * np.log10(10 ** (snr_true / 10) * 1 / (1 + 10 ** (snr_true / 10) *
                                                                10 ** (thd_true / 10)))
        check("f0 (Hz)", r["f0"], f, 0.5)
        check("amplitude (dBFS)", r["amp_dbfs"], -0.5, 0.05)
        check("SNR (dB)", r["snr"], snr_true, 0.3)
        check("THD (dB)", r["thd"], thd_true, 0.3)
        check("SNDR (dB)", r["sndr"], sndr_true, 0.3)
        check("SFDR (dBc)", r["sfdr_dbc"], 105.0, 0.5)
        if window == "bh7" and f == 2000.0 and plot_dir:
            plot_spectrum(r, os.path.join(plot_dir, "selftest_spectrum.png"))

    print("[spectrum] aliased harmonics: tone near 330 kHz, fs = 1 MHz")
    f = 84_501 * FS_DEFAULT / 262144                    # ~322 kHz, M odd
    codes, _, _ = _synth_tone(f, hd_dbc={2: -100.0, 3: -105.0, 4: -110.0, 5: -115.0}, snr_db=95)
    r = analyze_spectrum(codes, window="bh7")
    for h in r["harmonics"][:4]:
        want_f = abs(((h["h"] * f + FS_DEFAULT / 2) % FS_DEFAULT) - FS_DEFAULT / 2)
        check(f"H{h['h']} folded f (Hz)", h["f_folded"], want_f, 10.0)
        check(f"H{h['h']} level (dBc)", h["dbc"], [-100, -105, -110, -115][h["h"] - 2], 0.5)

    print("[histogram] 10-bit ADC, injected DNL errors + one missing code, 10M samples")
    rng = np.random.default_rng(7)
    bits, ncodes = 10, 1024
    delta = rng.uniform(-0.25, 0.25, ncodes - 1)        # delta[k-1] = shift of transition k
    delta[500 - 1], delta[501 - 1] = 1.0, 0.0           # code 500 has zero width
    thr = np.arange(1, ncodes) + delta                  # true transition levels T_k
    n = 10_000_000
    t = np.arange(n)
    x = 511.5 + 508 * np.sin(2 * np.pi * 0.2360679775 * t + 0.4) + rng.normal(0, 0.05, n)
    codes = np.searchsorted(thr, x, side="right")
    h = histogram_inl_dnl(codes, bits=bits, fit="best")
    dnl_true = np.diff(thr)[h["codes_dnl"] - 1] * 1.0 - 1       # width of code k = T_{k+1}-T_k
    err = h["dnl"] - dnl_true
    k = np.arange(len(err))
    print(f"  range {h['code_range']}, empty codes {h['empty_codes_in_range']}, "
          f"min counts/code {h['min_counts_in_range']}")
    check("DNL rms error (LSB)", float(np.sqrt((err ** 2).mean())), 0.0, 0.03)
    check("DNL of code 500 (missing)", float(h["dnl"][list(h["codes_dnl"]).index(500)]), -1.0, 0.1)
    kk = h["ks"]
    inl_true = thr[kk - 1] - np.polyval(np.polyfit(kk, thr[kk - 1], 1), kk)
    check("INL max |error| (LSB)", float(np.abs(h["inl"] - inl_true).max()), 0.0, 0.1)
    check("amplitude estimate (codes)", h["amp_est_codes"], 508.0, 2.0)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


# --------------------------------- CLI ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "spectrum", "hist", "sweep"])
    ap.add_argument("--file")
    ap.add_argument("--log")
    ap.add_argument("--fs", type=float, default=FS_DEFAULT)
    ap.add_argument("--bits", type=int, default=ADC_BITS)
    ap.add_argument("--window", choices=["rect", "bh7"], default="bh7")
    ap.add_argument("--fit", choices=["best", "endpoint"], default="best")
    ap.add_argument("--harmonics", type=int, default=9)
    ap.add_argument("--plot")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest(a.plot or ".")
    elif a.cmd == "spectrum":
        r = analyze_spectrum(load_codes(a.file), a.fs, a.harmonics, a.window, a.bits)
        print(f"f0 {r['f0']:.3f} Hz   amplitude {r['amp_dbfs']:.3f} dBFS\n"
              f"SNR {r['snr']:.3f} dB   SNDR {r['sndr']:.3f} dB   THD {r['thd']:.3f} dB\n"
              f"SFDR {r['sfdr_dbc']:.3f} dBc  (spur at {r['spur_freq']:.1f} Hz, "
              f"harmonic={r['spur_is_harmonic']})   ENOB {r['enob']:.3f}")
        for h in r["harmonics"]:
            note = "" if h["valid"] else "  (overlaps fundamental/DC - excluded)"
            print(f"  H{h['h']}: {h['f_unfolded']:.1f} Hz -> {h['f_folded']:.1f} Hz"
                  f"{' [aliased]' if h['aliased'] else ''}  {h['dbc']:.2f} dBc{note}")
        if a.plot:
            plot_spectrum(r, a.plot)
    elif a.cmd == "hist":
        h = histogram_inl_dnl(load_codes(a.file), a.bits, a.fit)
        print(f"{h['n_samples']:,} samples, codes {h['code_range']}, fit={h['fit']}, "
              f"empty codes in range: {h['empty_codes_in_range']}\n"
              f"DNL [{h['dnl_min']:.3f}, {h['dnl_max']:.3f}] LSB   "
              f"INL [{h['inl_min']:.3f}, {h['inl_max']:.3f}] LSB")
        if a.plot:
            plot_inl_dnl(h, a.plot)
    elif a.cmd == "sweep":
        process_sweep(a.log, a.fs, a.window, a.harmonics, a.bits)


if __name__ == "__main__":
    main()