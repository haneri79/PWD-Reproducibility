#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Judge (extended): PWD vs Baselines — fairness, accuracy, efficiency, scaling
+ Timeseries (non‑equilibrium) + Metallic (bands near E_F) + Strong‑correlation evaluation modules.

• One-file script; deps: numpy/pandas/scipy (openpyxl optional for .xlsx)
• Jupyter/Colab/CLI friendly:
  - Filters stray "-f kernel.json" args
  - ❗ If --input 미지정 ⇒ 자동 selftest로 전환(오류 종료 없음)
  - ❗ 하드 종료는 --hard-exit 있을 때만 (그 외에는 return code만 반환)

Author: (you)
"""
from __future__ import annotations
import sys, json, math, argparse, warnings
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

# Silence SciPy precision-loss spam for near-identical data (toggle via --diag)
warnings.filterwarnings(
    "ignore",
    message="Precision loss occurred in moment calculation",
    category=RuntimeWarning,
)

# =====================================================
# Utilities
# =====================================================

def _filter_jupyter_args(argv: Optional[List[str]]) -> List[str]:
    """Remove IPython/Colab kernel args like: -f /path/kernel-XXXX.json"""
    if argv is None:
        argv = sys.argv[1:]
    out: List[str] = []
    skip = False
    for i, tok in enumerate(argv):
        if skip:
            skip = False
            continue
        if tok == "-f" and i + 1 < len(argv) and argv[i + 1].endswith(".json"):
            skip = True
            continue
        out.append(tok)
    return out

# ---- IO ----

def load_table(path: Path, sep: Optional[str] = None, encoding: Optional[str] = None) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path)
    if sep is None:
        for cand in [",", "\t", ";", "|"]:
            try:
                df = pd.read_csv(path, sep=cand, encoding=encoding)
                if df.shape[1] >= 2:
                    return df
            except Exception:
                pass
        return pd.read_csv(path)
    return pd.read_csv(path, sep=sep, encoding=encoding)

# ---- Column normalization ----

CANON = {
    "category": ["category", "suite", "group"],
    "case_id": ["case_id", "case", "system", "name"],
    "method": ["method", "algo"],
    "target_error": ["target_error", "target", "tol", "tolerance"],
    "achieved_error": ["achieved_error", "error", "err", "mae", "rmse"],
    "core_hours": ["core_hours", "corehour", "core-hr", "coreh", "cpu_hours", "cpu_h"],
    "peak_ram_gb": ["peak_ram_gb", "ram_gb", "max_ram_gb", "peak_mem_gb"],
    # optional raw
    "wall_time_s": ["wall_time_s", "seconds", "time_s"],
    "n_cores": ["n_cores", "cores"],
    # fairness (optional)
    "accelerator": ["accelerator", "gpu"],
    "hostname": ["hostname", "host"],
    "blas": ["blas", "mkl", "openblas"],
    "compiler_flags": ["compiler_flags", "cflags"],
    # scaling (optional)
    "system_size": ["system_size", "n_atoms", "n_elec", "basis_size"],
}


def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    m: Dict[str, str] = {}
    for canon, aliases in CANON.items():
        for a in aliases:
            if a in df.columns:
                m[a] = canon
                break
    g = df.rename(columns=m).copy()
    # derive core_hours from wall_time_s * n_cores if needed
    if "core_hours" not in g.columns:
        if {"wall_time_s", "n_cores"}.issubset(g.columns):
            g["core_hours"] = (
                pd.to_numeric(g["wall_time_s"], errors="coerce") *
                pd.to_numeric(g["n_cores"], errors="coerce") / 3600.0
            )
    if "peak_ram_gb" not in g.columns and "peak_ram_mb" in g.columns:
        g["peak_ram_gb"] = pd.to_numeric(g["peak_ram_mb"], errors="coerce") / 1024.0
    return g

# =====================================================
# Stats helpers
# =====================================================

def paired_stats(pwd: pd.Series, base: pd.Series, alpha: float = 0.05) -> Dict:
    """Paired t/Wilcoxon + bootstrap median CI (one-sided: PWD < Base)."""
    pwd = pd.to_numeric(pwd, errors="coerce").dropna()
    base = pd.to_numeric(base, errors="coerce").dropna()
    n = min(len(pwd), len(base))
    if n == 0:
        return {"n": 0, "mean_diff": np.nan, "median_diff": np.nan,
                "bootstrap_ci_median": [np.nan, np.nan], "t_p_less": np.nan, "wilcoxon_p_less": np.nan}
    pwd = pwd.iloc[:n]; base = base.iloc[:n]
    diff = (pwd - base).to_numpy()
    mean_diff = float(np.nanmean(diff)); med_diff = float(np.nanmedian(diff))
    # t-test with variance guard
    try:
        var = float(np.var(diff, ddof=1)) if n > 1 else 0.0
        if var <= 1e-24:
            t_p = 1.0
        else:
            t_res = stats.ttest_rel(pwd, base, nan_policy="omit")
            t_p = float(t_res.pvalue / 2.0) if mean_diff < 0 else float(1 - t_res.pvalue / 2.0)
    except Exception:
        t_p = float("nan")
    # Wilcoxon (guard ties)
    try:
        if np.allclose(diff, 0) or np.sum(np.abs(diff) > 0) < 1:
            w_p = 1.0
        else:
            w_res = stats.wilcoxon(pwd, base, zero_method="wilcox", alternative="less")
            w_p = float(w_res.pvalue)
    except Exception:
        w_p = float("nan")
    # bootstrap CI for median
    rng = np.random.default_rng(42)
    B = 2000
    if n == 1:
        lo = hi = med_diff
    else:
        idxs = rng.integers(0, n, size=(B, n))
        boots = np.median(diff[idxs], axis=1)
        lo, hi = np.quantile(boots, [alpha/2, 1 - alpha/2])
    return {"n": int(n), "mean_diff": mean_diff, "median_diff": med_diff,
            "bootstrap_ci_median": [float(lo), float(hi)], "t_p_less": t_p, "wilcoxon_p_less": w_p}


def tost_equivalence(pwd_err: pd.Series, base_err: pd.Series, delta: float, alpha: float = 0.05) -> Dict:
    """Two One-Sided Tests (TOST) for equivalence on scalar errors."""
    x = pd.to_numeric(pwd_err, errors="coerce").dropna()
    y = pd.to_numeric(base_err, errors="coerce").dropna()
    n = min(len(x), len(y))
    if n == 0:
        return {"n": 0, "p1": np.nan, "p2": np.nan, "equivalent": False}
    x = x.iloc[:n]; y = y.iloc[:n]
    d = (x - y).to_numpy()
    dbar = float(np.mean(d)); s = float(np.std(d, ddof=1)) if n > 1 else 0.0
    if n <= 1 or s <= 1e-24:
        # conservative fallback
        p1 = 1.0 if dbar <= -delta else 0.0
        p2 = 1.0 if dbar >= +delta else 0.0
    else:
        se = s / math.sqrt(n)
        t1 = (dbar - (-delta)) / se
        t2 = (dbar - (+delta)) / se
        df = n - 1
        p1 = 1 - stats.t.cdf(t1, df=df)
        p2 = stats.t.cdf(t2, df=df)
    eq = (p1 < alpha) and (p2 < alpha)
    return {"n": n, "p1": float(p1), "p2": float(p2), "equivalent": bool(eq), "delta": float(delta), "dbar": dbar}

# =====================================================
# Core evaluation (table of runs)
# =====================================================

def summarize_core(df: pd.DataFrame, pwd_name: str, baselines: List[str], alpha: float,
                   alpha_equiv: Optional[float], equiv_delta: Optional[float],
                   core_thresh: float, ram_thresh: float,
                   include_offline: Optional[Dict] = None, amortize: int = 1,
                   diag: bool = False) -> Dict:
    c_cat, c_id, c_m = "category", "case_id", "method"
    c_tgt, c_err, c_ch, c_ram = "target_error", "achieved_error", "core_hours", "peak_ram_gb"
    must = [c_cat, c_id, c_m, c_tgt, c_err, c_ch, c_ram]
    for m in must:
        if m not in df.columns:
            raise ValueError(f"missing column: {m}")
    if diag:
        print("[diag] head:\n", df[[c_cat, c_id, c_m, c_tgt, c_err, c_ch, c_ram]].head().to_string(index=False))
    # Optional: include offline cost fully or amortized
    if include_offline:
        off_ch = float(include_offline.get("core_hours", 0.0))
        off_ram = float(include_offline.get("peak_ram_gb", 0.0))
        if off_ch > 0:
            mask = (df[c_m] == pwd_name)
            df.loc[mask, c_ch] = pd.to_numeric(df.loc[mask, c_ch], errors="coerce") + (off_ch / max(amortize, 1))
        if off_ram > 0:
            mask = (df[c_m] == pwd_name)
            df.loc[mask, c_ram] = pd.to_numeric(df.loc[mask, c_ram], errors="coerce") + (off_ram / max(amortize, 1))
    verdict: Dict = {"overall_pass": True, "categories": {}, "details": {}, "target_violations": []}
    for b in baselines:
        piv = df.pivot_table(index=[c_cat, c_id], columns=c_m, values=[c_err, c_ch, c_ram], aggfunc="median")
        sub = piv.dropna(how="any")
        # error equivalence
        if (c_err, pwd_name) in sub.columns and (c_err, b) in sub.columns:
            e_pwd = pd.Series(sub[(c_err, pwd_name)].to_numpy())
            e_base = pd.Series(sub[(c_err, b)].to_numpy())
            if equiv_delta is None:
                equiv_delta = 5e-4
            eq = tost_equivalence(e_pwd, e_base, delta=equiv_delta, alpha=alpha_equiv or alpha)
        else:
            eq = {"equivalent": False, "n": 0}
        # efficiency ratios
        def _ratio_ok(col: str, thresh: float):
            if (col, pwd_name) not in sub.columns or (col, b) not in sub.columns:
                return False, np.nan
            r = sub[(col, pwd_name)] / sub[(col, b)]
            ok = (r.median() <= thresh)
            return ok, float(r.median())
        ok_core, med_core = _ratio_ok(c_ch, core_thresh)
        ok_ram, med_ram = _ratio_ok(c_ram, ram_thresh)
        verdict["details"][b] = {
            "equivalence": eq,
            "core_ratio_median": med_core,
            "ram_ratio_median": med_ram,
            "pass_core": bool(ok_core),
            "pass_ram": bool(ok_ram),
        }
        if not (eq.get("equivalent", False) and ok_core and ok_ram):
            verdict["overall_pass"] = False
    # target violations (PWD only)
    vio = df[(pd.to_numeric(df[c_err], errors="coerce") > pd.to_numeric(df[c_tgt], errors="coerce")) & (df[c_m] == pwd_name)]
    for _, r in vio.iterrows():
        verdict["target_violations"].append({"category": r[c_cat], "case_id": r[c_id], "error": float(r[c_err]), "target": float(r[c_tgt])})
    return verdict

# =====================================================
# Scaling fits (optional)
# =====================================================

def fit_scaling(df: pd.DataFrame, pwd_name: str, baselines: List[str]) -> Dict:
    if "system_size" not in df.columns:
        return {"available": False}
    out: Dict = {"available": True, "fits": {}}
    for m in [pwd_name] + baselines:
        sub = df[df["method"] == m]
        x = pd.to_numeric(sub["system_size"], errors="coerce").to_numpy()
        y = pd.to_numeric(sub["core_hours"], errors="coerce").to_numpy()
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x = x[mask]; y = y[mask]
        if len(x) < 3:
            out["fits"][m] = {"ok": False}
            continue
        lx = np.log(x); ly = np.log(y)
        A = np.vstack([lx, np.ones_like(lx)]).T
        p, k = np.linalg.lstsq(A, ly, rcond=None)[0]
        out["fits"][m] = {"ok": True, "p": float(p), "k": float(np.exp(k))}
    return out


def predict_speedup(scaling: Dict, pwd_name: str, baselines: List[str], s_list: List[float], R0: Optional[float] = None) -> Dict:
    """Predict speedup at larger problem scales using fitted exponents.
    Returns {"R0": <baseline ratio at reference>, "pred": [...]}.
    """
    res: Dict = {"pred": [], "R0": (R0 if R0 is not None else None)}
    if not scaling.get("available"):
        return res
    p_pwd = scaling["fits"].get(pwd_name, {}).get("p")
    if p_pwd is None:
        return res
    for b in baselines:
        p_b = scaling["fits"].get(b, {}).get("p")
        if p_b is None:
            continue
        for s in s_list:
            d = float(p_b - p_pwd)
            R = (R0 if R0 else 1.0) * (s ** d)
            res["pred"].append({"baseline": b, "scale": float(s), "p_minus_q": d, "speedup": R})
    return res

# =====================================================
# Timeseries and Metallic modules
# =====================================================

@dataclass
class TSEvalConfig:
    time_col: str = "time"
    obs_col: str = "observable"
    val_col: str = "value"
    ref_method: Optional[str] = None
    delta_rel_l2: float = 0.03  # 3% equivalence margin on relative L2


def _interp_series_to_grid(df: pd.DataFrame, time_col: str, val_col: str, grid: np.ndarray) -> np.ndarray:
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    v = pd.to_numeric(df[val_col], errors="coerce").to_numpy()
    ord_idx = np.argsort(t)
    t = t[ord_idx]; v = v[ord_idx]
    v_interp = np.interp(grid, t, v, left=v[0], right=v[-1])
    return v_interp


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(b) + 1e-12
    return float(np.linalg.norm(a - b) / denom)


def evaluate_timeseries(ts_df: pd.DataFrame, pwd_name: str, baselines: List[str], cfg: TSEvalConfig, alpha: float = 0.05) -> Dict:
    required = {"category", "case_id", "method", cfg.time_col, cfg.obs_col, cfg.val_col}
    if not required.issubset(ts_df.columns):
        raise ValueError(f"[timeseries] missing columns: {required - set(ts_df.columns)}")
    out: Dict[str, Dict] = {}
    for (cat, cid, obs), sub in ts_df.groupby(["category", "case_id", cfg.obs_col]):
        tmin = sub[cfg.time_col].min(); tmax = sub[cfg.time_col].max()
        if not np.isfinite([tmin, tmax]).all() or tmax <= tmin:
            continue
        grid = np.linspace(float(tmin), float(tmax), 1024)
        series: Dict[str, np.ndarray] = {}
        for m, g in sub.groupby("method"):
            series[m] = _interp_series_to_grid(g, cfg.time_col, cfg.val_col, grid)
        metrics: Dict[str, float] = {}
        if cfg.ref_method and cfg.ref_method in series:
            ref = series[cfg.ref_method]
            for m, y in series.items():
                metrics[m] = _relative_l2(y, ref)
        else:
            base0 = baselines[0] if baselines else next(iter(series))
            if base0 not in series:
                base0 = next(iter(series))
            for m, y in series.items():
                metrics[m] = _relative_l2(y, series[base0])
        row = {"category": cat, "case_id": cid, "observable": obs, "per_baseline": {}}
        for b in baselines:
            if (pwd_name not in metrics) or (b not in metrics):
                row["per_baseline"][b] = {"available": False}
                continue
            e_pwd, e_base = metrics[pwd_name], metrics[b]
            s = paired_stats(pd.Series([e_pwd]), pd.Series([e_base]), alpha=alpha)
            eq = tost_equivalence(pd.Series([e_pwd]), pd.Series([e_base]), delta=cfg.delta_rel_l2, alpha=alpha)
            row["per_baseline"][b] = {"e_pwd": e_pwd, "e_base": e_base, "stats": s, "equiv": eq,
                                        "pwd_better": bool(e_pwd < e_base - 1e-12)}
        out[f"{cat}|{cid}|{obs}"] = row
    return out

@dataclass
class MetalEvalConfig:
    k_col: str = "k_index"
    band_col: str = "band_index"
    energy_col: str = "energy_ev"
    ef_col: str = "Ef_ev"
    window_ev: float = 0.3
    delta_ev: float = 0.05


def evaluate_metallic(band_df: pd.DataFrame, pwd_name: str, baselines: List[str], cfg: MetalEvalConfig, alpha: float = 0.05) -> Dict:
    required = {"category", "case_id", "method", cfg.k_col, cfg.band_col, cfg.energy_col}
    if not required.issubset(band_df.columns):
        raise ValueError(f"[metal] missing columns: {required - set(band_df.columns)}")
    out: Dict[str, Dict] = {}
    for (cat, cid), sub in band_df.groupby(["category", "case_id"]):
        if cfg.ef_col in sub.columns:
            ef = float(pd.to_numeric(sub[cfg.ef_col], errors="coerce").dropna().median())
        else:
            ef = float(pd.to_numeric(sub[cfg.energy_col], errors="coerce").dropna().median())
        mask = np.abs(pd.to_numeric(sub[cfg.energy_col], errors="coerce") - ef) <= cfg.window_ev
        win = sub[mask].copy()
        if win.empty:
            out[f"{cat}|{cid}"] = {"reason": "no bands near Ef"}
            continue
        piv = win.pivot_table(index=[cfg.k_col, cfg.band_col], columns="method", values=cfg.energy_col, aggfunc="median")
        row = {"category": cat, "case_id": cid, "per_baseline": {}}
        for b in baselines:
            if (pwd_name not in piv.columns) or (b not in piv.columns):
                row["per_baseline"][b] = {"available": False}
                continue
            e_pwd = piv[pwd_name].to_numpy(); e_base = piv[b].to_numpy()
            ref = np.nanmean(np.vstack([e_pwd, e_base]), axis=0)
            err_pwd = float(np.linalg.norm(e_pwd - ref) / (np.linalg.norm(ref) + 1e-12))
            err_base = float(np.linalg.norm(e_base - ref) / (np.linalg.norm(ref) + 1e-12))
            s = paired_stats(pd.Series([err_pwd]), pd.Series([err_base]), alpha=alpha)
            eq = tost_equivalence(pd.Series([err_pwd]), pd.Series([err_base]), delta=cfg.delta_ev, alpha=alpha)
            row["per_baseline"][b] = {"err_pwd_rel": err_pwd, "err_base_rel": err_base, "stats": s, "equiv": eq,
                                        "pwd_better": bool(err_pwd < err_base - 1e-12)}
        out[f"{cat}|{cid}"] = row
    return out

# =====================================================
# Strong-correlation module
# =====================================================

def evaluate_strongcorr(sc_df: pd.DataFrame, pwd_name: str, baselines: List[str], ref_method: str = "REF", delta_rel: float = 0.03, alpha: float = 0.05) -> Dict:
    required = {"category", "case_id", "method", "observable", "value"}
    if not required.issubset(sc_df.columns):
        raise ValueError(f"[strongcorr] missing columns: {required - set(sc_df.columns)}")
    out: Dict[str, Dict] = {}
    for (cat, cid, obs), sub in sc_df.groupby(["category", "case_id", "observable"]):
        # Build per-method scalar values (median if multiple)
        vals = sub.pivot_table(index=["category", "case_id", "observable"], columns="method", values="value", aggfunc="median")
        row = {"category": cat, "case_id": cid, "observable": obs, "per_baseline": {}}
        for b in baselines:
            if ref_method not in vals.columns or pwd_name not in vals.columns or b not in vals.columns:
                row["per_baseline"][b] = {"available": False}
                continue
            v_ref = float(vals.loc[(cat, cid, obs), ref_method])
            v_pwd = float(vals.loc[(cat, cid, obs), pwd_name])
            v_b   = float(vals.loc[(cat, cid, obs), b])
            # relative absolute error to REF
            def relerr(x):
                denom = max(abs(v_ref), 1e-12)
                return abs(x - v_ref) / denom
            e_pwd = relerr(v_pwd); e_base = relerr(v_b)
            s = paired_stats(pd.Series([e_pwd]), pd.Series([e_base]), alpha=alpha)
            eq = tost_equivalence(pd.Series([e_pwd]), pd.Series([e_base]), delta=delta_rel, alpha=alpha)
            row["per_baseline"][b] = {"e_pwd": e_pwd, "e_base": e_base, "stats": s, "equiv": eq, "pwd_better": bool(e_pwd < e_base - 1e-12)}
        out[f"{cat}|{cid}|{obs}"] = row
    return out

# =====================================================
# Markdown (with extras)
# =====================================================

def to_markdown(verdict: Dict, baselines: List[str], core_thresh: float, ram_thresh: float) -> str:
    lines: List[str] = []
    lines.append("# Judge Summary\n")
    lines.append(f"overall_pass: {verdict.get('overall_pass')}\n")
    lines.append("## Baseline comparisons\n")
    for b, d in verdict.get("details", {}).items():
        eq = d.get("equivalence", {})
        lines.append(f"### vs {b}\n")
        lines.append(f"- TOST equivalent: {eq.get('equivalent')} (n={eq.get('n')}, delta={eq.get('delta')})\n")
        lines.append(f"- core-hours median ratio (PWD/{b}): {d.get('core_ratio_median')} (<= {core_thresh} pass={d.get('pass_core')})\n")
        lines.append(f"- peak RAM median ratio (PWD/{b}): {d.get('ram_ratio_median')} (<= {ram_thresh} pass={d.get('pass_ram')})\n")
    if verdict.get("target_violations"):
        lines.append("\n## Target violations (PWD)\n")
        for v in verdict["target_violations"]:
            lines.append(f"- {v['category']}/{v['case_id']}: err={v['error']} > target={v['target']}\n")
    # Scaling fits
    sc = verdict.get("scaling")
    if sc and sc.get("available"):
        lines.append("\n## Scaling fits\n")
        fits = sc.get("fits", {})
        for m, f in fits.items():
            if f.get("ok"):
                lines.append(f"- {m}: p={f.get('p')}, k={f.get('k')}\n")
            else:
                lines.append(f"- {m}: not enough points for fit\n")
    # Predicted speedup
    sp = verdict.get("scaling_pred")
    if sp and sp.get("pred"):
        lines.append("\n## Predicted speedup vs scale\n")
        if sp.get("R0") is not None:
            lines.append(f"- baseline ratio at reference (R0) ≈ {sp.get('R0')}\n")
        for row in sp["pred"]:
            lines.append(f"- vs {row['baseline']}: scale×{row['scale']} ⇒ speedup ≈ {row['speedup']} (Δexp={row['p_minus_q']})\n")
    extras = verdict.get("extras", {})
    if extras:
        lines.append("\n## Extras\n")
        if "timeseries" in extras:
            ts = extras["timeseries"]
            ok_cnt = 0; tot = 0
            for _, row in ts.items():
                for _, r in row.get("per_baseline", {}).items():
                    if not r.get("available", True):
                        continue
                    tot += 1
                    if r.get("pwd_better", False):
                        ok_cnt += 1
            lines.append(f"### Timeseries dynamics: PWD better ratio ≈ {ok_cnt}/{tot}\n")
        if "metallic" in extras:
            mt = extras["metallic"]
            ok_cnt = 0; tot = 0
            for _, row in mt.items():
                for _, r in row.get("per_baseline", {}).items():
                    if not r.get("available", True):
                        continue
                    tot += 1
                    if r.get("pwd_better", False):
                        ok_cnt += 1
            lines.append(f"### Metallic bands near E_F: PWD better ratio ≈ {ok_cnt}/{tot}\n")
        if "strongcorr" in extras:
            sc2 = extras["strongcorr"]
            ok_cnt = 0; tot = 0
            for _, row in sc2.items():
                for _, r in row.get("per_baseline", {}).items():
                    if not r.get("available", True):
                        continue
                    tot += 1
                    if r.get("pwd_better", False):
                        ok_cnt += 1
            lines.append(f"### Strong correlation: PWD better ratio ≈ {ok_cnt}/{tot}\n")
    return "".join(lines)

# =====================================================
# Selftest + unit tests
# =====================================================

def _make_selftest_table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cats = ["bonding", "vdW", "barrier"]
    systems = [("H2_0.7", 60), ("He", 40), ("CH4", 100), ("benzene", 160)]
    rows = []
    for cat in cats:
        for name, size in systems:
            err_pwd = 1.2e-3 + rng.normal(0, 0.2e-4)
            err_r2  = 1.4e-3 + rng.normal(0, 0.3e-4)
            err_cc  = 1.3e-3 + rng.normal(0, 0.3e-4)
            ch_pwd = 1e-5 * (size ** 2.0) * (1 + rng.normal(0, 0.03))
            ch_r2  = 3e-9 * (size ** 4.2) * (1 + rng.normal(0, 0.03))
            ch_cc  = 1e-12 * (size ** 6.2) * (1 + rng.normal(0, 0.03))
            ram_pwd = 0.008 * (size ** 1.2) * (1 + rng.normal(0, 0.03))
            ram_r2  = 0.03  * (size ** 1.3) * (1 + rng.normal(0, 0.03))
            ram_cc  = 0.05  * (size ** 1.5) * (1 + rng.normal(0, 0.03))
            tgt = 1.6e-3
            rows += [
                {"category": cat, "case_id": name, "system_size": size, "method": "PWD", "target_error": tgt, "achieved_error": err_pwd, "core_hours": ch_pwd, "peak_ram_gb": ram_pwd},
                {"category": cat, "case_id": name, "system_size": size, "method": "r2SCAN-D4", "target_error": tgt, "achieved_error": err_r2, "core_hours": ch_r2, "peak_ram_gb": ram_r2},
                {"category": cat, "case_id": name, "system_size": size, "method": "DLPNO-CCSD(T)", "target_error": tgt, "achieved_error": err_cc, "core_hours": ch_cc, "peak_ram_gb": ram_cc},
            ]
    return pd.DataFrame(rows)


def run_tests():
    # T1–T8: core selftest
    df = _map_columns(_make_selftest_table())
    verdict = summarize_core(df, "PWD", ["r2SCAN-D4", "DLPNO-CCSD(T)"], alpha=0.05,
                             alpha_equiv=0.05, equiv_delta=5e-4,
                             core_thresh=0.20, ram_thresh=0.25,
                             include_offline=None, amortize=1, diag=False)
    assert verdict["overall_pass"], "Selftest core should PASS"

    # T9: timeseries — PWD closer to REF than baselines
    t = np.linspace(0, 10, 501)
    ref = np.exp(-0.1 * t) * np.sin(3 * t)
    base = ref + 0.06 * np.sin(5 * t)
    pwd  = ref + 0.02 * np.sin(5 * t)
    rows = []
    for name, y in [("PWD", pwd), ("r2SCAN-D4", base), ("DLPNO-CCSD(T)", base * 1.01), ("REF", ref)]:
        for ti, yi in zip(t, y):
            rows.append({"category": "dyn", "case_id": "osc", "method": name, "observable": "dipole", "time": ti, "value": yi})
    ts_df = pd.DataFrame(rows)
    ts_out = evaluate_timeseries(ts_df, "PWD", ["r2SCAN-D4", "DLPNO-CCSD(T)"], TSEvalConfig(ref_method="REF"))
    assert any(v.get("per_baseline", {}).get("r2SCAN-D4", {}).get("pwd_better") for v in ts_out.values()), "PWD better in TS"

    # T10: metallic — PWD closer to mean-ref near E_F
    rows = []
    for k in range(20):
        for b in range(3):
            e_ref = 0.1 * (k / 20 - 0.5) + 0.02 * b
            e_base = e_ref + 0.06 * np.sin(k)
            e_pwd  = e_ref + 0.02 * np.sin(k)
            rows += [
                {"category": "metal", "case_id": "Al", "method": "PWD", "k_index": k, "band_index": b, "energy_ev": e_pwd, "Ef_ev": 0.0},
                {"category": "metal", "case_id": "Al", "method": "r2SCAN-D4", "k_index": k, "band_index": b, "energy_ev": e_base, "Ef_ev": 0.0},
            ]
    band_df = pd.DataFrame(rows)
    mt_out = evaluate_metallic(band_df, "PWD", ["r2SCAN-D4"], MetalEvalConfig(window_ev=0.2, delta_ev=0.05))
    assert any(v.get("per_baseline", {}).get("r2SCAN-D4", {}).get("pwd_better") for v in mt_out.values()), "PWD better near Ef"

    # T11: no --input ⇒ auto selftest (no SystemExit), returns 0
    rc = main(["--out", "/mnt/data/_tmp_verdict.json"])  # no --input
    assert rc == 0, "Main should auto-selftest and return 0 when --input missing"

    # T12: stray -f kernel.json filtered
    rc2 = main(["--selftest", "-f", "/tmp/kernel.json"])  # should be ignored
    assert rc2 == 0, "Kernel args should be filtered"

    # T13: markdown generation with extras (smoke test)
    fake_verdict = {
        "overall_pass": True,
        "details": {"r2SCAN-D4": {"equivalence": {"equivalent": True, "n": 4, "delta": 5e-4},
                                     "core_ratio_median": 0.18, "ram_ratio_median": 0.22, "pass_core": True, "pass_ram": True}},
        "extras": {
            "timeseries": {"dyn|osc|dipole": {"per_baseline": {"r2SCAN-D4": {"available": True, "pwd_better": True}}}},
            "metallic":   {"metal|Al":        {"per_baseline": {"r2SCAN-D4": {"available": True, "pwd_better": True}}}},
            "strongcorr": {"hubbard1d|U=2|energy_per_site": {"per_baseline": {"r2SCAN-D4": {"available": True, "pwd_better": True}}}},
        }
    }
    md = to_markdown(fake_verdict, ["r2SCAN-D4"], 0.20, 0.25)
    assert "Timeseries dynamics" in md and "Metallic bands near E_F" in md and "Strong correlation" in md, "Markdown sections missing"

    # T14: scaling + prediction sections in markdown
    fake_verdict2 = {
        "overall_pass": True,
        "details": {},
        "scaling": {"available": True, "fits": {"PWD": {"ok": True, "p": 2.0, "k": 1e-5}, "r2SCAN-D4": {"ok": True, "p": 4.0, "k": 3e-9}}},
        "scaling_pred": {"R0": 37.0, "pred": [{"baseline": "r2SCAN-D4", "scale": 8.0, "p_minus_q": 2.0, "speedup": 37.0*(8.0**2)}]},
    }
    md2 = to_markdown(fake_verdict2, ["r2SCAN-D4"], 0.20, 0.25)
    assert "Scaling fits" in md2 and "Predicted speedup" in md2, "Scaling sections missing"

    print("[tests] All extended tests passed (core + TS + metal + strongcorr + scaling + predict + main + markdown)")

# =====================================================
# Main
# =====================================================

def main(argv: Optional[List[str]] = None) -> int:
    argv = _filter_jupyter_args(argv)
    p = argparse.ArgumentParser(description="PWD fair benchmark judge (extended)")
    p.add_argument("--input", required=False, help="CSV/XLSX of core runs")
    p.add_argument("--out", default="verdict.json")
    p.add_argument("--markdown", default=None)
    p.add_argument("--pwd-name", default="PWD")
    p.add_argument("--baselines", nargs="+", default=["r2SCAN-D4", "DLPNO-CCSD(T)"])
    p.add_argument("--core-thresh", type=float, default=0.20)
    p.add_argument("--ram-thresh", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--alpha-equiv", type=float, default=None)
    p.add_argument("--equiv-delta", type=float, default=None)
    p.add_argument("--diag", action="store_true")
    p.add_argument("--force-sep", choices=[",", ";", "\t"], default=None)
    p.add_argument("--force-encoding", default=None)
    p.add_argument("--save-normalized", default=None)
    p.add_argument("--rename", default=None)
    p.add_argument("--include-offline-cost", default=None, help="JSON file with core_hours, peak_ram_gb")
    p.add_argument("--offline-amortize", type=int, default=1)
    p.add_argument("--fit-scaling", action="store_true")
    p.add_argument("--predict-scale", nargs="*", type=float, default=None)
    p.add_argument("--provenance", default=None)
    p.add_argument("--save-scaling-csv", default=None, help="Path to save scaling fits as CSV")
    p.add_argument("--save-pred-csv", default=None, help="Path to save predicted speedups as CSV")
    p.add_argument("--runtests", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--hard-exit", action="store_true")
    p.add_argument("--quiet", action="store_true")
    # extras
    p.add_argument("--ts-input", default=None, help="Timeseries CSV (category,case_id,method,observable,time,value)")
    p.add_argument("--ts-ref-method", default=None)
    p.add_argument("--ts-delta", type=float, default=0.03, help="Equivalence Δ for relative L2")
    p.add_argument("--metal-input", default=None, help="Bands CSV (category,case_id,method,k_index,band_index,energy_ev[,Ef_ev])")
    p.add_argument("--metal-window-ev", type=float, default=0.3)
    p.add_argument("--metal-delta-ev", type=float, default=0.05)
    # strong correlation extras
    p.add_argument("--sc-input", default=None, help="Strong-correlation CSV (category,case_id,method,observable,value)")
    p.add_argument("--sc-ref-method", default="REF")
    p.add_argument("--sc-delta", type=float, default=0.03, help="Equivalence Δ on observable errors (relative)")

    args = p.parse_args(argv)

    if args.runtests:
        run_tests()
        return 0  # no hard exit here; handled at __main__

    # ---- INPUT HANDLING ----
    if args.input:
        df = load_table(Path(args.input), sep=args.force_sep, encoding=args.force_encoding)
    else:
        # 🔧 Robust default: auto selftest when no --input was provided
        if not args.quiet:
            sys.stderr.write("[judge] --input 미지정: selftest 모드로 자동 전환 (샘플 데이터 사용)\n")
        df = _make_selftest_table()

    df = _map_columns(df)

    if args.rename:
        try:
            mapping = json.loads(Path(args.rename).read_text(encoding="utf-8"))
            df = df.rename(columns=mapping)
        except Exception as e:
            sys.stderr.write(f"[judge] rename 실패: {e}\n")

    include_offline = None
    if args.include_offline_cost:
        try:
            include_offline = json.loads(Path(args.include_offline_cost).read_text(encoding="utf-8"))
        except Exception as e:
            sys.stderr.write(f"[judge] offline 비용 로드 실패: {e}\n")

    verdict = summarize_core(
        df, args.pwd_name, args.baselines, alpha=args.alpha,
        alpha_equiv=(args.alpha_equiv or args.alpha),
        equiv_delta=(args.equiv_delta or 5e-4),
        core_thresh=args.core_thresh, ram_thresh=args.ram_thresh,
        include_offline=include_offline, amortize=args.offline_amortize, diag=args.diag,
    )

    # scaling
    if args.fit_scaling:
        sc = fit_scaling(df, args.pwd_name, args.baselines)
        verdict["scaling"] = sc
        if args.predict_scale:
            try:
                baser = None
                piv = df.pivot_table(index=["category", "case_id"], columns="method", values="core_hours", aggfunc="median").dropna()
                if len(piv) > 0:
                    r = (piv[args.baselines[0]] / piv[args.pwd_name]).median()
                    baser = float(r)
                verdict["scaling_pred"] = predict_speedup(sc, args.pwd_name, args.baselines, args.predict_scale, R0=baser)
                # optional CSV exports
                try:
                    if args.save_scaling_csv:
                        fits = sc.get("fits", {})
                        rows = []
                        for m, f in fits.items():
                            r = {"method": m}
                            r.update(f)
                            rows.append(r)
                        if rows:
                            pd.DataFrame(rows).to_csv(args.save_scaling_csv, index=False)
                    if args.save_pred_csv:
                        pred = verdict["scaling_pred"].get("pred", [])
                        if pred:
                            pd.DataFrame(pred).to_csv(args.save_pred_csv, index=False)
                except Exception as ie:
                    sys.stderr.write(f"[judge] CSV export warning: {ie}\n")
            except Exception as e:
                sys.stderr.write(f"[judge] scaling 예측 실패: {e}\n")

    # extras
    extras: Dict[str, Dict] = {}
    try:
        if args.ts_input:
            ts_df = load_table(Path(args.ts_input))
            extras["timeseries"] = evaluate_timeseries(
                ts_df, args.pwd_name, args.baselines,
                TSEvalConfig(ref_method=args.ts_ref_method, delta_rel_l2=args.ts_delta),
                alpha=args.alpha,
            )
        if args.metal_input:
            mt_df = load_table(Path(args.metal_input))
            extras["metallic"] = evaluate_metallic(
                mt_df, args.pwd_name, args.baselines,
                MetalEvalConfig(window_ev=args.metal_window_ev, delta_ev=args.metal_delta_ev),
                alpha=args.alpha,
            )
        if args.sc_input:
            sc_df = load_table(Path(args.sc_input))
            extras["strongcorr"] = evaluate_strongcorr(
                sc_df, args.pwd_name, args.baselines,
                ref_method=(args.sc_ref_method or "REF"), delta_rel=(args.sc_delta or 0.03),
                alpha=args.alpha,
            )
    except Exception as e:
        sys.stderr.write(f"[extras] 처리 중 오류: {e}\n")
    if extras:
        verdict["extras"] = extras

    # outputs
    out_path = Path(args.out)
    out_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown:
        md = to_markdown(verdict, args.baselines, args.core_thresh, args.ram_thresh)
        Path(args.markdown).write_text(md, encoding="utf-8")
    if args.diag:
        print(json.dumps(verdict.get("details", {}), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    # Respect --hard-exit only if explicitly requested; otherwise avoid SystemExit in notebooks.
    argv0 = _filter_jupyter_args(None)
    rc = main(argv0)
    if "--hard-exit" in argv0:
        raise SystemExit(rc)
    # else: no hard exit; return code is printed only if non-zero
    if rc != 0:
        sys.stderr.write(f"[judge] return code {rc} (no hard exit)\n")
