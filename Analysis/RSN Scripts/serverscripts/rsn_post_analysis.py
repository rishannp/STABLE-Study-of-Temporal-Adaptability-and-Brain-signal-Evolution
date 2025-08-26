#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rsn_post_analysis_fixed.py
Produce bandpower / criticality plots and FC GIFs from RSN pickles.

Usage:
    python rsn_post_analysis_fixed.py RSN_LS.pkl
    python rsn_post_analysis_fixed.py RSN_LS.pkl RSN_GH.pkl

Notes:
- Accepts legacy pickles of form (S, config) or new unified dict {'sessions', 'channels', ...}
- Saves outputs to ./outputs/<pkl_basename>/...
"""
import os
import sys
import argparse
import pickle
import numpy as np
import math
import warnings
from pathlib import Path
from tqdm import tqdm
import matplotlib as mpl
mpl.use("Agg")   # headless backend
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import imageio.v2 as imageio

# pastel palette
PASTEL = {
    'blue':  '#A3C4F3',
    'green': '#B9E3C6',
    'red':   '#F7B2AD',
    'purple':'#CDB4DB',
    'orange':'#FFD6A5',
    'grey':  '#CED4DA'
}

# -------------------------
# Utility & robust helpers
# -------------------------
def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def is_unified(rsn):
    return isinstance(rsn, dict) and 'sessions' in rsn and 'channels' in rsn

def get_sessions(rsn):
    if is_unified(rsn):
        return rsn['sessions']
    # legacy: assume tuple/list (S, config)
    if isinstance(rsn, (list, tuple)) and len(rsn) >= 1:
        return rsn[0]
    raise ValueError("Unrecognised RSN structure")

def get_channels(rsn):
    if is_unified(rsn):
        return rsn['channels']['Channel']
    # legacy: assume config at index 1
    if isinstance(rsn, (list, tuple)) and len(rsn) >= 2:
        cfg = rsn[1]
        if isinstance(cfg, dict):
            return cfg.get('channels', {}).get('Channel')
    return None

def safe_mean(a, axis=None):
    a = np.asarray(a)
    if a.size == 0:
        return np.nan
    return np.nanmean(a, axis=axis)

def safe_std(a, axis=None):
    a = np.asarray(a)
    if a.size == 0:
        return np.nan
    return np.nanstd(a, axis=axis)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def _smooth_interp(x, y, n_out=200, kind='cubic'):
    """
    Smooth/interpolate (x,y) to a nicely-gridded curve.
    If not enough points for 'cubic', falls back to 'linear'.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return np.array([]), np.array([])
    x0 = x[mask]; y0 = y[mask]
    if x0.size < 2:
        return x0, y0
    kind_try = kind
    if x0.size < 4 and kind == 'cubic':
        kind_try = 'linear'
    try:
        f = interp1d(x0, y0, kind=kind_try, bounds_error=False, fill_value="extrapolate")
        xs = np.linspace(x0.min(), x0.max(), n_out)
        ys = f(xs)
        return xs, ys
    except Exception:
        # fallback: linear
        f = interp1d(x0, y0, kind='linear', bounds_error=False, fill_value="extrapolate")
        xs = np.linspace(x0.min(), x0.max(), max(n_out, len(x0)))
        ys = f(xs)
        return xs, ys

def plot_series_with_shade(xs_raw, ys_raw_mean, ys_raw_std, outpath, title, xlabel="Window / Session", ylabel="", color=PASTEL['blue']):
    """
    xs_raw: array-like x coordinates (e.g., window indices or session indices)
    ys_raw_mean: array-like of means (same length)
    ys_raw_std: array-like of stds (same length)
    """
    ensure_dir(Path(outpath).parent)
    if len(xs_raw) == 0:
        with warnings.catch_warnings():
            warnings.warn(f"Empty series for {title} - skipping {outpath}")
        return False

    xs = np.asarray(xs_raw)
    ys_mean = np.asarray(ys_raw_mean)
    ys_std = np.asarray(ys_raw_std) if ys_raw_std is not None else np.zeros_like(ys_mean)

    xs_s, ys_s = _smooth_interp(xs, ys_mean, n_out=max(200, len(xs)*8), kind='cubic')
    if xs_s.size == 0:
        xs_s, ys_s = xs, ys_mean
    try:
        fstd = interp1d(xs, ys_std, kind='linear', bounds_error=False, fill_value="extrapolate")
        std_s = fstd(xs_s)
    except Exception:
        std_s = np.interp(xs_s, xs, ys_std, left=np.nan, right=np.nan)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(xs_s, ys_s, lw=2.0, color=color)
    finite = np.isfinite(std_s) & np.isfinite(ys_s)
    if finite.any():
        ax.fill_between(xs_s[finite], ys_s[finite] - std_s[finite], ys_s[finite] + std_s[finite], alpha=0.25, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    try:
        plt.savefig(outpath, dpi=200)
    except Exception as e:
        warnings.warn(f"Failed saving {outpath}: {e}")
        plt.close()
        return False
    plt.close()
    return True

# -------------------------
# Aggregation helpers (similar to previous notebook)
# -------------------------
def reconstruct_features_from_sessions(sessions):
    """
    Robustly build aggregated features from a list of session dicts.
    Returns a dict:
      {
        'BANDPOWER': {'absolute': {band: (S,C)}, 'relative': {band: (S,C)}},
        'CRITICALITY': {'LZC': {band: (S,C)}, 'PLE': (S,C)}
      }
    Handles many session-level storage formats:
      - BANDPOWER['absolute'] as dict band->(E,W) or band->(E,)
      - BANDPOWER['absolute'] as ndarray (E,B,W) or (E,B) or (B,E) and an accompanying 'bands' list
      - CRITICALITY['LZC'] as dict or ndarray (E,B,W)/(E,B)
      - CRITICALITY['PLE'] as (E,W) or (E,)
    """
    S = len(sessions)
    first = sessions[0]
    bands = None

    bp0 = first.get('BANDPOWER', None)
    if isinstance(bp0, dict) and 'bands' in bp0:
        try:
            bands = list(bp0['bands'])
        except Exception:
            bands = None

    if bands is None and isinstance(bp0, dict) and 'absolute' in bp0 and isinstance(bp0['absolute'], dict):
        bands = list(bp0['absolute'].keys())

    if bands is None and isinstance(first.get('CRITICALITY', None), dict) and 'LZC' in first['CRITICALITY'] and isinstance(first['CRITICALITY']['LZC'], dict):
        bands = list(first['CRITICALITY']['LZC'].keys())

    if bands is None or len(bands) == 0:
        bands = ['2-4','4-7','7-13','13-30','30-47','53-97']

    C = None
    for sess in sessions:
        bp = sess.get('BANDPOWER', None)
        if isinstance(bp, dict) and 'absolute' in bp:
            a = bp['absolute']
            try:
                arr = np.asarray(a)
                if arr.ndim >= 1:
                    if arr.ndim == 3:  # (E,B,W)
                        C = arr.shape[0]; break
                    if arr.ndim == 2:
                        if arr.shape[1] == len(bands):
                            C = arr.shape[0]; break
                        if arr.shape[0] == len(bands):
                            C = arr.shape[1]; break
                        C = arr.shape[0]; break
                    if arr.ndim == 1:
                        C = arr.shape[0]; break
            except Exception:
                pass
        cr = sess.get('CRITICALITY', None)
        if isinstance(cr, dict) and 'LZC' in cr:
            l = cr['LZC']
            try:
                arr = np.asarray(l)
                if arr.ndim >= 1:
                    C = arr.shape[0]; break
            except Exception:
                pass
    if C is None:
        C = 1

    bp_abs = {b: np.full((S, C), np.nan) for b in bands}
    bp_rel = {b: np.full((S, C), np.nan) for b in bands}
    lzc    = {b: np.full((S, C), np.nan) for b in bands}
    ple    = np.full((S, C), np.nan)

    def _write_band_matrix(dest_dict, bname, s_idx, vec):
        if vec is None:
            return
        vec = np.asarray(vec)
        if vec.size == 0:
            return
        L = vec.shape[0]
        dest_dict[bname][s_idx, :L] = np.nanmean(vec, axis=-1) if vec.ndim > 1 else vec

    for s_idx, sess in enumerate(sessions):
        bp = sess.get('BANDPOWER', None)
        if isinstance(bp, dict) and 'absolute' in bp:
            a = bp['absolute']
            if isinstance(a, dict):
                for b in bands:
                    if b in a:
                        arr = np.asarray(a[b])
                        if arr.ndim == 2:
                            _write_band_matrix(bp_abs, b, s_idx, np.nanmean(arr, axis=1))
                        elif arr.ndim == 1:
                            _write_band_matrix(bp_abs, b, s_idx, arr)
            else:
                arr = np.asarray(a)
                if arr.ndim == 3:
                    E, B, W = arr.shape
                    for b_i, bname in enumerate(bands[:B]):
                        vec = np.nanmean(arr[:, b_i, :], axis=1)
                        _write_band_matrix(bp_abs, bname, s_idx, vec)
                elif arr.ndim == 2:
                    if arr.shape[1] == len(bands):
                        E, B = arr.shape
                        for b_i, bname in enumerate(bands[:B]):
                            vec = arr[:, b_i]
                            _write_band_matrix(bp_abs, bname, s_idx, vec)
                    elif arr.shape[0] == len(bands):
                        B, E = arr.shape
                        for b_i, bname in enumerate(bands[:B]):
                            vec = arr[b_i, :]
                            _write_band_matrix(bp_abs, bname, s_idx, vec)
                    else:
                        E, B = arr.shape
                        for b_i, bname in enumerate(bands[:B]):
                            vec = arr[:, b_i]
                            _write_band_matrix(bp_abs, bname, s_idx, vec)
                elif arr.ndim == 1:
                    _write_band_matrix(bp_abs, bands[0], s_idx, arr)

        if isinstance(bp, dict) and 'relative' in bp:
            a = bp['relative']
            if isinstance(a, dict):
                for b in bands:
                    if b in a:
                        arr = np.asarray(a[b])
                        if arr.ndim == 2:
                            _write_band_matrix(bp_rel, b, s_idx, np.nanmean(arr, axis=1))
                        elif arr.ndim == 1:
                            _write_band_matrix(bp_rel, b, s_idx, arr)
            else:
                arr = np.asarray(a)
                if arr.ndim == 3:
                    E, B, W = arr.shape
                    for b_i, bname in enumerate(bands[:B]):
                        vec = np.nanmean(arr[:, b_i, :], axis=1)
                        _write_band_matrix(bp_rel, bname, s_idx, vec)
                elif arr.ndim == 2:
                    if arr.shape[1] == len(bands):
                        E, B = arr.shape
                        for b_i, bname in enumerate(bands[:B]):
                            _write_band_matrix(bp_rel, bname, s_idx, arr[:, b_i])
                    elif arr.shape[0] == len(bands):
                        B, E = arr.shape
                        for b_i, bname in enumerate(bands[:B]):
                            _write_band_matrix(bp_rel, bname, s_idx, arr[b_i, :])
                    else:
                        E, B = arr.shape
                        for b_i, bname in enumerate(bands[:B]):
                            _write_band_matrix(bp_rel, bname, s_idx, arr[:, b_i])
                elif arr.ndim == 1:
                    _write_band_matrix(bp_rel, bands[0], s_idx, arr)

        cr = sess.get('CRITICALITY', None)
        if isinstance(cr, dict) and 'LZC' in cr:
            l = cr['LZC']
            if isinstance(l, dict):
                for b in bands:
                    if b in l:
                        arr = np.asarray(l[b])
                        if arr.ndim == 2:
                            _write_band_matrix(lzc, b, s_idx, np.nanmean(arr, axis=1))
                        elif arr.ndim == 1:
                            _write_band_matrix(lzc, b, s_idx, arr)
            else:
                arr = np.asarray(l)
                if arr.ndim == 3:
                    E, B, W = arr.shape
                    for b_i, bname in enumerate(bands[:B]):
                        _write_band_matrix(lzc, bname, s_idx, np.nanmean(arr[:, b_i, :], axis=1))
                elif arr.ndim == 2:
                    if arr.shape[1] == len(bands):
                        E, B = arr.shape
                        for b_i, bname in enumerate(bands[:B]):
                            _write_band_matrix(lzc, bname, s_idx, arr[:, b_i])
                    else:
                        _write_band_matrix(lzc, bands[0], s_idx, np.nanmean(arr, axis=1))
                elif arr.ndim == 1:
                    _write_band_matrix(lzc, bands[0], s_idx, arr)

        if isinstance(cr, dict) and 'PLE' in cr:
            p = cr['PLE']
            try:
                arr = np.asarray(p)
                if arr.ndim == 2:
                    ple[s_idx, :arr.shape[0]] = np.nanmean(arr, axis=1)
                elif arr.ndim == 1:
                    ple[s_idx, :arr.shape[0]] = arr
            except Exception:
                pass

    return {
        'BANDPOWER': {'absolute': bp_abs, 'relative': bp_rel},
        'CRITICALITY': {'LZC': lzc, 'PLE': ple}
    }

# -------------------------
# FC GIF generation
# -------------------------
def make_fc_gif(matrix_list, outpath, cmap='magma', vmin=None, vmax=None, titles=None, fps=2):
    """
    matrix_list: list of 2D numpy arrays (square)
    outpath: gif path (str / Path)
    """
    ensure_dir(Path(outpath).parent)
    vmin = np.nanmin(matrix_list) if vmin is None else vmin
    vmax = np.nanmax(matrix_list) if vmax is None else vmax
    tmp_pngs = []
    for idx, mat in enumerate(matrix_list):
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.axis('off')
        tit = titles[idx] if (titles is not None and idx < len(titles)) else f"S{idx+1}"
        ax.set_title(tit, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        tmp = f"{outpath}.frame_{idx}.png"
        try:
            plt.savefig(tmp, dpi=150, bbox_inches='tight')
        except Exception as e:
            warnings.warn(f"Failed save frame {tmp}: {e}")
        plt.close()
        tmp_pngs.append(tmp)

    imgs = []
    for p in tmp_pngs:
        try:
            imgs.append(imageio.imread(p))
        except Exception:
            pass
    if imgs:
        # duration in ms = 1000 / fps
        duration = int(1000 / fps)
        imageio.mimsave(outpath, imgs, duration=duration)

    for p in tmp_pngs:
        try:
            os.remove(p)
        except Exception:
            pass
    return True if imgs else False

# -------------------------
# MAIN routine for one pickle
# -------------------------
def analyze_rsn(pkl_path: Path, out_base: Path, make_gifs=True):
    print(f"\nProcessing: {pkl_path} -> outputs in {out_base}")
    ensure_dir(out_base)
    rsn = load_pickle(pkl_path)
    sessions = get_sessions(rsn)
    channels = get_channels(rsn) or [f"Ch{i}" for i in range(1, 129)]
    S = len(sessions)
    print(f"Sessions: {S}, Channels (len): {len(channels)}")

    # reconstruct aggregates if needed
    try:
        feats = rsn.get('features') if is_unified(rsn) else None
    except Exception:
        feats = None
    if feats is None:
        feats_agg = reconstruct_features_from_sessions(sessions)
    else:
        feats_agg = feats

    # -----------------------------
    # BANDPOWER per-session plotting
    # (robust to dict/ndarray formats)
    # -----------------------------
    bp_folder = ensure_dir(out_base / "bandpower_per_session")
    for s_idx, sess in enumerate(sessions):
        s_name = f"S{s_idx+1:02d}"
        sess_dir = ensure_dir(bp_folder / s_name)
        if not isinstance(sess, dict) or 'BANDPOWER' not in sess:
            continue

        bp_dict = sess['BANDPOWER']
        abs_obj = bp_dict.get('absolute', None)
        if abs_obj is None:
            continue

        bands_fallback = list(feats_agg['BANDPOWER']['absolute'].keys())
        bands_local = list(bp_dict.get('bands', bands_fallback))

        def _plot_series_for_band(bname: str, arr_like):
            arr = np.asarray(arr_like)
            if arr.size == 0:
                return
            if arr.ndim == 1:
                arr2 = arr.reshape(arr.shape[0], 1)  # (E,1)
            else:
                arr2 = arr  # (E,W)
            perwin_mean = np.nanmean(arr2, axis=0)  # (W,)
            perwin_std  = np.nanstd(arr2, axis=0)
            xs = np.arange(1, perwin_mean.size + 1)
            title = f"{pkl_path.stem} {s_name} BandPower Abs {bname}"
            outpng = sess_dir / f"{s_name}_band_{bname}_bandpower_abs.png"
            plot_series_with_shade(xs, perwin_mean, perwin_std, outpng,
                                   title, xlabel="Window",
                                   ylabel="Absolute Power", color=PASTEL['green'])

        if isinstance(abs_obj, dict):
            for bname in bands_local:
                if bname in abs_obj:
                    _plot_series_for_band(bname, abs_obj[bname])
        else:
            arr = np.asarray(abs_obj)
            if arr.ndim == 3:
                E, B, W = arr.shape
                for b_i, bname in enumerate(bands_local[:B]):
                    _plot_series_for_band(bname, arr[:, b_i, :])  # (E,W)
            elif arr.ndim == 2:
                if arr.shape[1] == len(bands_local):
                    E, B = arr.shape
                    for b_i, bname in enumerate(bands_local[:B]):
                        _plot_series_for_band(bname, arr[:, b_i])  # (E,)
                elif arr.shape[0] == len(bands_local):
                    B, E = arr.shape
                    for b_i, bname in enumerate(bands_local[:B]):
                        _plot_series_for_band(bname, arr[b_i, :])  # (E,)
                else:
                    E, B = arr.shape
                    for b_i, bname in enumerate(bands_local[:B]):
                        _plot_series_for_band(bname, arr[:, b_i])
            elif arr.ndim == 1:
                _plot_series_for_band(bands_local[0], arr)

    # Global per-band across sessions plots
    gp = ensure_dir(out_base / "bandpower_global")
    bp_abs = feats_agg['BANDPOWER']['absolute']   # dict band -> (S, C)
    bp_rel = feats_agg['BANDPOWER']['relative']   # dict band -> (S, C)

    for mode_name, bp_dict, col in [('abs', bp_abs, PASTEL['green']), ('rel', bp_rel, PASTEL['orange'])]:
        for bname, mat in bp_dict.items():
            means = np.nanmean(mat, axis=1)  # (S,)
            stds = np.nanstd(mat, axis=1)
            sessions_idx = np.arange(1, len(means)+1)
            title = f"{pkl_path.stem} BandPower ({mode_name.upper()}) {bname} across sessions"
            outpng = Path(gp) / f"BandPower_{mode_name}_{bname}_sessions.png"
            plot_series_with_shade(sessions_idx, means, stds, outpng, title, xlabel="Session", ylabel=f"{mode_name} bandpower", color=col)

    # Criticality: LZC per band & PLE
    crit_dir = ensure_dir(out_base / "criticality")
    lzc = feats_agg['CRITICALITY']['LZC']  # dict band -> (S, C)
    ple = feats_agg['CRITICALITY']['PLE']  # (S, C)

    for bname, mat in lzc.items():
        means = np.nanmean(mat, axis=1)
        stds = np.nanstd(mat, axis=1)
        xs = np.arange(1, len(means)+1)
        outpng = Path(crit_dir) / f"LZC_sessions_{bname}.png"
        plot_series_with_shade(xs, means, stds, outpng, title=f"LZC across sessions ({bname})", xlabel="Session", ylabel="LZC", color=PASTEL['purple'])

    try:
        ple_means = np.nanmean(ple, axis=1)
        ple_stds = np.nanstd(ple, axis=1)
        xs = np.arange(1, len(ple_means)+1)
        outpng = Path(crit_dir) / "PLE_sessions.png"
        plot_series_with_shade(xs, ple_means, ple_stds, outpng, title="PLE (2-40 Hz) across sessions", xlabel="Session", ylabel="PLE (beta)", color=PASTEL['blue'])
    except Exception:
        warnings.warn("PLE plotting failed (no data?)")

    # Build FC GIFs
    if make_gifs:
        fc_out = ensure_dir(out_base / "fc_gifs")
        features_try = ['PLV', 'iCoh', 'CSD', 'MSC']
        for feat in features_try:
            present = False
            for sess in sessions:
                if isinstance(sess, dict) and (feat in sess.get('FC', {}) or feat in sess):
                    present = True
                    break
            if not present:
                continue

            first = sessions[0]
            band_list = None
            if isinstance(first, dict) and 'FC' in first and feat in first['FC']:
                try:
                    band_list = first['FC'][feat]['bands']
                except Exception:
                    band_list = None
            else:
                if isinstance(first, dict) and feat in first:
                    try:
                        band_list = list(first[feat].keys())
                    except Exception:
                        band_list = None
            if not band_list:
                band_list = list(feats_agg['BANDPOWER']['absolute'].keys())

            for bname in band_list:
                mats_per_session = []
                titles = []
                for s_idx, sess in enumerate(sessions):
                    mat = None
                    if isinstance(sess, dict) and 'FC' in sess and feat in sess['FC']:
                        try:
                            data = sess['FC'][feat]['data']  # (E,E,W,B)
                            bands_ = sess['FC'][feat]['bands']
                            if bname in bands_:
                                b_i = bands_.index(bname)
                                if data.size == 0:
                                    mat = None
                                else:
                                    if data.ndim == 4:
                                        m = np.nanmean(data[:, :, :, b_i], axis=2)
                                    else:
                                        m = np.nanmean(data, axis=2)
                                    mat = np.asarray(m)
                        except Exception:
                            pass
                    if mat is None and isinstance(sess, dict) and feat in sess:
                        try:
                            entry = sess[feat].get(bname)
                            if entry is None:
                                entry = sess[feat]
                            arr = np.asarray(entry)
                            if arr.ndim == 3:
                                mat = np.nanmean(arr, axis=2)
                            elif arr.ndim == 2:
                                mat = arr
                        except Exception:
                            pass
                    if mat is None:
                        continue
                    mats_per_session.append(np.nan_to_num(mat, nan=0.0))
                    titles.append(f"S{s_idx+1:02d}")
                if len(mats_per_session) >= 1:
                    outgif = Path(fc_out) / f"{feat}_{bname}.gif"
                    try:
                        vmin = np.nanmin(mats_per_session)
                        vmax = np.nanmax(mats_per_session)
                    except Exception:
                        vmin, vmax = None, None
                    ok = make_fc_gif(mats_per_session, str(outgif), cmap='magma', vmin=vmin, vmax=vmax, titles=titles, fps=2)
                    if ok:
                        print(f"Saved GIF {outgif}")
                    else:
                        warnings.warn(f"No frames made for {feat} {bname} (maybe missing data)")

    print(f"Finished writing outputs to {out_base}")

# -------------------------
# CLI
# -------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Post-analysis for RSN pickle(s): bandpower, criticality, FC GIFs.")
    p.add_argument("pkls", nargs="+", help="One or more RSN pickle files (.pkl)")
    p.add_argument("--outdir", default="outputs", help="Base output folder")
    p.add_argument("--no-gifs", action="store_true", help="Do not generate FC GIFs (faster)")
    args = p.parse_args(argv)

    for pkl in args.pkls:
        pklp = Path(pkl)
        if not pklp.exists():
            print("Missing:", pkl); continue
        out_base = Path(args.outdir) / pklp.stem
        analyze_rsn(pklp, out_base, make_gifs=not args.no_gifs)

if __name__ == "__main__":
    main()
