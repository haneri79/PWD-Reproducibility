#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_pwd.py  —  PWD 'hundreds-x' verifier (final)"""
import argparse, json, math, re, sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def _gmean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0: return np.nan
    return float(np.exp(np.log(x).mean()))

def _ci95_logratio(logr, alpha=0.05):
    n = len(logr)
    if n == 0: return (np.nan, np.nan)
    m = float(np.mean(logr))
    se = float(np.std(logr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    z = float(stats.t.ppf(1-alpha/2, df=n-1)) if n > 1 else 1.96
    return float(np.exp(m - z*se)), float(np.exp(m + z*se))

def _bootstrap_ci_gm(ratios, B=2000, seed=1234, alpha=0.05):
    rng = np.random.default_rng(seed)
    r = np.asarray(ratios, dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0: return (np.nan, np.nan)
    idx = np.arange(r.size)
    boots = []
    for _ in range(B):
        s = rng.choice(idx, size=idx.size, replace=True)
        boots.append(_gmean(r[s]))
    return (float(np.percentile(boots, 100*alpha/2)),
            float(np.percentile(boots, 100*(1-alpha/2))))

def _tost_paired(x, y, delta, alpha=0.05):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return False, np.nan, np.nan, n
    md = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return True, 0.0, 0.0, n
    se = sd / math.sqrt(n)
    t1 = (md + delta)/se
    p1 = 1 - stats.t.cdf(t1, df=n-1)
    t2 = (md - delta)/se
    p2 = stats.t.cdf(t2, df=n-1)
    return (p1 < alpha) and (p2 < alpha), float(p1), float(p2), n

P_PWD  = re.compile(r"(pwd|ours?)", re.I)
P_BASE = re.compile(r"(base|baseline|ref)", re.I)

METRIC_ALIASES = {
    "time":        [r"time", r"wall[_ ]?time[_ ]?s", r"runtime[_ ]?s", r"elapsed[_ ]?s"],
    "core_hours":  [r"core[_ ]?hours", r"corehrs", r"cpu[_ ]?hours"],
    "peak_ram_gb": [r"peak[_ ]?ram[_ ]?gb", r"ram[_ ]?gb", r"mem[_ ]?gb"],
    "energy_wh":   [r"energy[_ ]?wh", r"wh", r"energy[_ ]?watt[_ ]?hours?"],
    "error_mha":   [r"error[_ ]?mha", r"err[_ ]?mha", r"abs[_ ]?error[_ ]?mha"]
}

def _find_col(df, patterns):
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in df.columns:
            if rx.fullmatch(str(c)) or rx.search(str(c)):
                return c
    return None

def _autodetect_pairs(df, metric_key):
    metric_col = _find_col(df, METRIC_ALIASES.get(metric_key, [metric_key]))
    def _pick(regex_str):
        regs = re.compile(regex_str, re.I)
        hits = [c for c in df.columns if regs.search(str(c))]
        return hits[0] if hits else None
    col_pwd  = _pick(fr"{metric_key}.*(_pwd|pwd$)") or (_pick(fr"{metric_col}.*(_pwd|pwd$)") if metric_col else None)
    col_base = _pick(fr"{metric_key}.*(_base|_baseline|base$)") or (_pick(fr"{metric_col}.*(_base|_baseline|base$)") if metric_col else None)
    if col_pwd and col_base:
        return pd.to_numeric(df[col_pwd], errors="coerce"), pd.to_numeric(df[col_base], errors="coerce")
    if metric_col:
        cand_pwd  = _pick(fr"{metric_col}.*pwd") or _pick(fr"pwd.*{metric_col}")
        cand_base = _pick(fr"{metric_col}.*base") or _pick(fr"base.*{metric_col}")
        if cand_pwd and cand_base:
            return pd.to_numeric(df[cand_pwd], errors="coerce"), pd.to_numeric(df[cand_base], errors="coerce")
    method_col = _find_col(df, [r"method", r"algo", r"approach"])
    if method_col and (metric_col or metric_key):
        case_col = _find_col(df, [r"case[_ ]?id", r"case", r"id", r"sample", r"uid"])  # optional
        d2 = df.copy()
        d2["__role__"] = d2[method_col].astype(str).str.lower().map(lambda s: "pwd" if P_PWD.search(s or "") else ("base" if P_BASE.search(s or "") else "other"))
        value_col = metric_col if metric_col else _find_col(df, METRIC_ALIASES.get(metric_key, [metric_key]))
        if value_col is None:
            raise KeyError(f"metric '{metric_key}' column not found")
        if case_col:
            pivot = d2.pivot_table(index=case_col, columns="__role__", values=value_col, aggfunc="first")
            if {"pwd","base"}.issubset(pivot.columns):
                return pd.to_numeric(pivot["pwd"], errors="coerce"), pd.to_numeric(pivot["base"], errors="coerce")
        else:
            pwd_vals  = d2.loc[d2["__role__"]=="pwd",  value_col].reset_index(drop=True)
            base_vals = d2.loc[d2["__role__"]=="base", value_col].reset_index(drop=True)
            n = min(len(pwd_vals), len(base_vals))
            return pd.to_numeric(pwd_vals.iloc[:n], errors="coerce"), pd.to_numeric(base_vals.iloc[:n], errors="coerce")
    raise KeyError(f"Could not autodetect columns for metric '{metric_key}'. Use wide columns like '{metric_key}_pwd/_base' or tall with [method,{metric_key}].")

@dataclass
class MetricResult:
    name: str
    n: int
    gm_speedup: float
    ci95_low: float
    ci95_high: float
    t_p_one_sided: float
    wilcoxon_p_one_sided: float
    cliffs_delta: float

def _analyze_ratio(pwd_series, base_series, alpha=0.05, bootstrap=0):
    pwd = pd.to_numeric(pwd_series, errors="coerce").to_numpy(dtype=float)
    base= pd.to_numeric(base_series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(pwd) & np.isfinite(base) & (pwd>0) & (base>0)
    pwd, base = pwd[mask], base[mask]
    n = len(pwd)
    if n == 0: return None
    ratios = base / pwd
    logr = np.log(ratios)
    gm = _gmean(ratios)
    ciL, ciH = _ci95_logratio(logr, alpha=alpha)
    if bootstrap and bootstrap>0:
        try:
            bL, bH = _bootstrap_ci_gm(ratios, B=bootstrap)
            ciL, ciH = bL, bH
        except Exception:
            pass
    tstat, p_t = stats.ttest_1samp(logr, popmean=0.0, alternative="greater")
    try:
        wstat, p_w = stats.wilcoxon(logr, alternative="greater", zero_method="zsplit")
    except Exception:
        p_w = np.nan
    cd = _cliffs_delta(logr, np.zeros_like(logr))
    return n, gm, ciL, ciH, float(p_t), float(p_w), float(cd)

def _cliffs_delta(x, y):
    d = np.asarray(x) - np.asarray(y)
    n_pos = np.sum(d > 0); n_neg = np.sum(d < 0)
    den = n_pos + n_neg
    return float((n_pos - n_neg) / den) if den > 0 else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with PWD/base paired results")
    ap.add_argument("--metrics", nargs="+", default=["time","core_hours","peak_ram_gb","energy_wh"])
    ap.add_argument("--primary-metric", default="time")
    ap.add_argument("--delta-mha", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--speedup-threshold", type=float, default=100.0)
    ap.add_argument("--ci-lower-threshold", type=float, default=50.0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--figdir", default="docs/img")
    args = ap.parse_args()
    np.random.seed(args.seed)

    df = pd.read_csv(args.input)

    acc = {}
    try:
        pwd_e, base_e = _autodetect_pairs(df, "error_mha")
        eq, p1, p2, n_eq = _tost_paired(pwd_e, base_e, delta=args.delta_mha, alpha=args.alpha)
        acc = {"equivalent": bool(eq), "p1": p1, "p2": p2, "margin_mha": args.delta_mha, "n": int(n_eq)}
    except Exception as e:
        acc = {"equivalent": None, "note": f"accuracy columns not found: {e}"}

    results = []
    gm_for_bar, labels = [], []
    for m in args.metrics:
        try:
            pwd_v, base_v = _autodetect_pairs(df, m)
        except Exception:
            continue
        out = _analyze_ratio(pwd_v, base_v, alpha=args.alpha, bootstrap=args.bootstrap)
        if out is None: 
            continue
        n, gm, ciL, ciH, p_t, p_w, cd = out
        results.append(MetricResult(m, n, gm, ciL, ciH, p_t, p_w, cd).__dict__)
        labels.append(m); gm_for_bar.append(gm)

    verdict = {"accuracy_tost": acc, "metrics": results,
               "thresholds": {"gm_speedup": args.speedup_threshold, "ci_lower": args.ci_lower_threshold},
               "primary_metric": args.primary_metric}
    primary = next((r for r in results if r["name"] == args.primary_metric), None)
    ok_hx = (primary is not None and (primary["gm_speedup"] >= args.speedup_threshold) and (primary["ci95_low"]   >= args.ci_lower_threshold))
    verdict["hundreds_x_on_primary"] = bool(ok_hx)
    verdict["all_checks_pass"] = bool(ok_hx)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    with open(outdir/"verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    lines = []
    lines += ["# PWD Verification Summary", ""]
    lines += [f"- Input: `{args.input}`"]
    lines += [f"- TOST (accuracy, Δ={args.delta_mha} mHa): {acc.get('equivalent')}"]
    if acc.get("equivalent") is None and acc.get("note"):
        lines += [f"  - note: {acc.get('note')}"]
    lines += [f"- Hundreds-x on `{args.primary_metric}`: {verdict.get('hundreds_x_on_primary')}",""]
    lines += ["## Head-to-head (PWD vs Baseline) — geometric mean speedup (×)"]
    lines += ["| Metric | n | GM× | 95% CI | t one-sided p | Wilcoxon p | Cliff’s δ |",              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        ci = f"{r['ci95_low']:.2f}–{r['ci95_high']:.2f}"
        lines += [f"| {r['name']} | {r['n']} | {r['gm_speedup']:.2f} | {ci} | {r['t_p_one_sided']:.2e} | {r['wilcoxon_p_one_sided']:.2e} | {r['cliffs_delta']:.2f} |"]
    with open(outdir/"summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")

    if labels and gm_for_bar:
        plt.figure(figsize=(6,4))
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.bar(labels, gm_for_bar)
        plt.ylabel("Geometric mean speedup (×)")
        plt.title("PWD vs Baseline — Head-to-Head")
        plt.tight_layout()
        plt.savefig(figdir/"h2h_speedup.png", dpi=150)

        try:
            pwd_v, base_v = _autodetect_pairs(df, args.primary_metric)
            logr = np.log(pd.to_numeric(base_v, errors="coerce") / pd.to_numeric(pwd_v, errors="coerce"))
            logr = logr[np.isfinite(logr)]
            if len(logr) > 0:
                plt.figure(figsize=(6,4))
                plt.violinplot(logr, showmeans=True)
                plt.axhline(0.0, linestyle="--", linewidth=1)
                plt.ylabel("log(speedup)  (0 → parity, >0 → PWD faster)")
                plt.title(f"Distribution of log speedups ({args.primary_metric})")
                plt.tight_layout()
                plt.savefig(figdir/"ratios_violin.png", dpi=150)
        except Exception:
            pass

    if primary:
        gm, lo, hi = primary["gm_speedup"], primary["ci95_low"], primary["ci95_high"]
        print(f"[SUMMARY] {args.primary_metric}: GM×={gm:.2f}, 95%CI=[{lo:.2f},{hi:.2f}], TOST={acc.get('equivalent')}, hundreds-x={verdict['hundreds_x_on_primary']}")
    print(f"[OK] Wrote: {outdir/'verdict.json'}, {outdir/'summary.md'}")
    print(f"[OK] Plots: {figdir/'h2h_speedup.png'}, {figdir/'ratios_violin.png'} (if available)")

if __name__ == "__main__":
    main()
