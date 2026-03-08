import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Zerve design system ───────────────────────────────────────────────────
_BG     = "#1D1D20"
_TEXT   = "#fbfbff"
_MUTED  = "#909094"
_BLUE   = "#A1C9F4"
_ORANGE = "#FFB482"
_GREEN  = "#8DE5A1"
_CORAL  = "#FF9F9B"
_LAVEND = "#D0BBFF"
_YELLOW = "#ffd400"
_SUCC   = "#17b26a"
_WARN   = "#f04438"

_PALETTE = [_BLUE, _ORANGE, _GREEN, _CORAL, _LAVEND,
            _YELLOW, _SUCC, _WARN, "#9467BD", "#8C564B",
            "#C49C94", "#E377C2", "#F7B6D2"]

plt.rcParams.update({
    "figure.facecolor": _BG, "axes.facecolor": _BG,
    "text.color": _TEXT, "axes.labelcolor": _TEXT,
    "xtick.color": _TEXT, "ytick.color": _TEXT,
    "axes.edgecolor": _MUTED, "grid.color": "#333338",
    "font.family": "sans-serif",
})

# ── Feature columns & readable labels ────────────────────────────────────
_FEAT_COLS = [c for c in user_feature_matrix.columns if c.startswith("feat_")]

_FEAT_LABELS = {
    "feat_total_early_events":     "Total Early Events (7d)",
    "feat_unique_event_types":     "Unique Event Types (7d)",
    "feat_unique_tools_used":      "Unique Tools Used (7d)",
    "feat_unique_days_early":      "Active Days (7d)",
    "feat_tool_event_rate":        "Tool Event Rate",
    "feat_early_credit_total":     "Early Credit Total",
    "feat_has_early_credit":       "Has Early Credit (0/1)",
    "feat_session_count":          "Session Count (7d)",
    "feat_avg_events_per_sess":    "Avg Events / Session",
    "feat_median_sess_dur_min":    "Median Session Duration (min)",
    "feat_max_events_in_sess":     "Max Events in Session",
    "feat_tool_session_count":     "Tool Sessions (7d)",
    "feat_time_to_2nd_sess_min":   "Time to 2nd Session (min)",
}

_data = user_feature_matrix[_FEAT_COLS].copy()

print("=" * 75)
print("  DESCRIPTIVE STATISTICS — USER-LEVEL BEHAVIORAL FEATURE MATRIX")
print(f"  {len(_data):,} users  ×  {len(_FEAT_COLS)} features  (first 7 days of activity)")
print("=" * 75)

# ── 1. Compute full stats ─────────────────────────────────────────────────
_stats_rows = []
for _fc in _FEAT_COLS:
    _s = _data[_fc].dropna()
    _mode_res = stats.mode(_s, keepdims=True)
    _row = {
        "feature":   _fc,
        "label":     _FEAT_LABELS.get(_fc, _fc),
        "n":         len(_s),
        "mean":      _s.mean(),
        "median":    _s.median(),
        "mode":      float(_mode_res.mode[0]),
        "std":       _s.std(ddof=1),
        "variance":  _s.var(ddof=1),
        "min":       _s.min(),
        "max":       _s.max(),
        "range":     _s.max() - _s.min(),
        "q1":        _s.quantile(0.25),
        "q3":        _s.quantile(0.75),
        "iqr":       _s.quantile(0.75) - _s.quantile(0.25),
        "skewness":  float(stats.skew(_s)),
        "kurtosis":  float(stats.kurtosis(_s)),   # excess kurtosis (Fisher def.)
        "p5":        _s.quantile(0.05),
        "p95":       _s.quantile(0.95),
    }
    _stats_rows.append(_row)

descriptive_stats_results = pd.DataFrame(_stats_rows).set_index("feature")

# ── 2. Pretty-print results table ────────────────────────────────────────
_print_cols = ["mean", "median", "mode", "std", "variance",
               "range", "skewness", "kurtosis"]

_display_df = descriptive_stats_results[_print_cols].copy()
_display_df.index = [_FEAT_LABELS.get(i, i) for i in _display_df.index]

print("\n┌─ CENTRAL TENDENCY & DISPERSION ─────────────────────────────────────────────────────────────────────┐")
print(f"{'Feature':<35} {'Mean':>10} {'Median':>10} {'Mode':>10} {'Std Dev':>10} {'Variance':>12} {'Range':>10} {'Skew':>8} {'Kurt':>8}")
print("├" + "─" * 109 + "┤")
for _idx, _r in _display_df.iterrows():
    print(f"  {_idx:<33} {_r['mean']:>10.3f} {_r['median']:>10.3f} {_r['mode']:>10.3f} "
          f"{_r['std']:>10.3f} {_r['variance']:>12.3f} {_r['range']:>10.3f} "
          f"{_r['skewness']:>8.3f} {_r['kurtosis']:>8.3f}")
print("└" + "─" * 109 + "┘")

print("\n┌─ PERCENTILE BREAKDOWN ───────────────────────────────────────────────────────────┐")
print(f"{'Feature':<35} {'Min':>10} {'P5':>10} {'Q1 (P25)':>10} {'Q3 (P75)':>10} {'P95':>10} {'Max':>10} {'IQR':>10}")
print("├" + "─" * 87 + "┤")
for _fc in _FEAT_COLS:
    _r = descriptive_stats_results.loc[_fc]
    _lbl = _FEAT_LABELS.get(_fc, _fc)
    print(f"  {_lbl:<33} {_r['min']:>10.3f} {_r['p5']:>10.3f} {_r['q1']:>10.3f} "
          f"{_r['q3']:>10.3f} {_r['p95']:>10.3f} {_r['max']:>10.3f} {_r['iqr']:>10.3f}")
print("└" + "─" * 87 + "┘")

# ── 3. Interpretation flags ───────────────────────────────────────────────
print("\n┌─ DISTRIBUTION SHAPE ASSESSMENT ────────────────────────────────────────────────────────────────────────────────────┐")
print(f"{'Feature':<35} {'Skewness':>10} {'Shape':>18}   {'Kurtosis':>10} {'Tail Weight':>14}   {'Normality Note'}")
print("├" + "─" * 120 + "┤")
for _fc in _FEAT_COLS:
    _r = descriptive_stats_results.loc[_fc]
    _sk = _r["skewness"]
    _ku = _r["kurtosis"]
    _lbl = _FEAT_LABELS.get(_fc, _fc)
    _sk_desc = ("heavy right skew" if _sk > 2 else
                "moderate right skew" if _sk > 0.5 else
                "heavy left skew" if _sk < -2 else
                "moderate left skew" if _sk < -0.5 else
                "symmetric")
    _ku_desc = ("leptokurtic (heavy tails)" if _ku > 1 else
                "platykurtic (light tails)" if _ku < -1 else
                "mesokurtic (near-normal)")
    _norm_note = "⚠ Non-normal" if abs(_sk) > 1 or _ku > 3 else "≈ Near-normal"
    print(f"  {_lbl:<33} {_sk:>10.3f} {_sk_desc:>18}   {_ku:>10.3f} {_ku_desc:>14}   {_norm_note}")
print("└" + "─" * 120 + "┘")

print(f"\n📦 descriptive_stats_results stored as DataFrame — {descriptive_stats_results.shape[0]} features × {descriptive_stats_results.shape[1]} statistics")

# ═══════════════════════════════════════════════════════════════════════════
# ── VISUALISATIONS ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# ── CHART 1: Histograms (all 13 features, 4×4 grid minus last 3) ─────────
_n_feats = len(_FEAT_COLS)
_ncols = 4
_nrows = (_n_feats + _ncols - 1) // _ncols   # ceil div

fig_histograms, _hist_axes = plt.subplots(_nrows, _ncols, figsize=(20, 4 * _nrows))
fig_histograms.patch.set_facecolor(_BG)
fig_histograms.suptitle(
    "Histogram & Frequency Distribution — Behavioral Feature Matrix\n(4,774 Users · First 7 Days of Activity)",
    color=_TEXT, fontsize=15, fontweight="bold", y=1.01
)

for _i, (_fc, _ax) in enumerate(zip(_FEAT_COLS, _hist_axes.flat)):
    _s = _data[_fc].dropna()
    _cap99 = np.percentile(_s, 99)
    _s_clipped = _s.clip(upper=_cap99)
    _color = _PALETTE[_i % len(_PALETTE)]
    _lbl = _FEAT_LABELS.get(_fc, _fc)

    # Stats for annotation
    _mn  = _s.mean()
    _med = _s.median()
    _sk  = descriptive_stats_results.loc[_fc, "skewness"]

    _bins = 35
    _n_hist, _bin_edges, _patches = _ax.hist(
        _s_clipped, bins=_bins, color=_color, alpha=0.82, edgecolor="none"
    )

    # Mean & median lines
    _cap_mn  = min(_mn,  _cap99)
    _cap_med = min(_med, _cap99)
    _ax.axvline(_cap_mn,  color=_YELLOW, linewidth=1.6, linestyle="--", label=f"Mean={_mn:.2f}")
    _ax.axvline(_cap_med, color=_WARN,   linewidth=1.6, linestyle=":",  label=f"Median={_med:.2f}")

    _ax.set_title(_lbl, color=_TEXT, fontsize=9.5, fontweight="bold", pad=4)
    _ax.set_xlabel("Value (99th pct cap)", color=_MUTED, fontsize=7.5)
    _ax.set_ylabel("Count", color=_MUTED, fontsize=7.5)
    _ax.tick_params(labelsize=7.5, colors=_TEXT)
    _ax.spines[["top", "right"]].set_visible(False)

    # Skewness annotation
    _ax.text(0.97, 0.95, f"skew={_sk:.2f}", transform=_ax.transAxes,
             ha="right", va="top", fontsize=7.5, color=_MUTED)

    _ax.legend(fontsize=6.5, facecolor=_BG, edgecolor=_MUTED,
               labelcolor=_TEXT, loc="upper right", handlelength=1.5)
    _ax.set_facecolor(_BG)

# Hide any unused subplots
for _j in range(_n_feats, len(_hist_axes.flat)):
    _hist_axes.flat[_j].set_visible(False)

plt.tight_layout()
plt.savefig("behavioral_feature_histograms.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

# ── CHART 2: Skewness & Kurtosis comparison (side-by-side bars) ──────────
_sk_vals = [descriptive_stats_results.loc[fc, "skewness"] for fc in _FEAT_COLS]
_ku_vals = [descriptive_stats_results.loc[fc, "kurtosis"] for fc in _FEAT_COLS]
_feat_short = [_FEAT_LABELS.get(fc, fc).replace(" (7d)", "").replace(" (min)", "")
               for fc in _FEAT_COLS]

fig_skew_kurt, (_ax_sk, _ax_ku) = plt.subplots(1, 2, figsize=(18, 6))
fig_skew_kurt.patch.set_facecolor(_BG)
fig_skew_kurt.suptitle(
    "Distribution Shape: Skewness & Excess Kurtosis — All Behavioral Features",
    color=_TEXT, fontsize=13, fontweight="bold"
)

# Skewness
_sk_colors = [_CORAL if s > 1 else _SUCC if s < -1 else _BLUE for s in _sk_vals]
_bars_sk = _ax_sk.barh(_feat_short[::-1], _sk_vals[::-1], color=_sk_colors[::-1],
                        edgecolor="none", height=0.65)
_ax_sk.axvline(0, color=_MUTED, linewidth=1.2, linestyle="-")
_ax_sk.axvline(1, color=_YELLOW, linewidth=0.9, linestyle="--", alpha=0.6, label="|skew|=1 threshold")
_ax_sk.axvline(-1, color=_YELLOW, linewidth=0.9, linestyle="--", alpha=0.6)
for _bar, _sv in zip(_bars_sk, _sk_vals[::-1]):
    _ax_sk.text(_sv + (0.05 if _sv >= 0 else -0.05), _bar.get_y() + _bar.get_height() / 2,
                f"{_sv:.2f}", va="center", ha="left" if _sv >= 0 else "right",
                fontsize=8.5, color=_TEXT)
_ax_sk.set_title("Skewness (0 = symmetric)", color=_TEXT, fontsize=11, fontweight="bold")
_ax_sk.set_xlabel("Skewness", color=_MUTED, fontsize=10)
_ax_sk.spines[["top", "right"]].set_visible(False)
_ax_sk.tick_params(labelsize=9)
_ax_sk.set_facecolor(_BG)
_ax_sk.legend(facecolor=_BG, edgecolor=_MUTED, labelcolor=_TEXT, fontsize=8, loc="lower right")

# Kurtosis
_ku_colors = [_CORAL if k > 3 else _LAVEND if k > 1 else _BLUE for k in _ku_vals]
_bars_ku = _ax_ku.barh(_feat_short[::-1], _ku_vals[::-1], color=_ku_colors[::-1],
                        edgecolor="none", height=0.65)
_ax_ku.axvline(0, color=_MUTED, linewidth=1.2, linestyle="-")
_ax_ku.axvline(3, color=_YELLOW, linewidth=0.9, linestyle="--", alpha=0.6, label="excess kurt=3")
for _bar, _kv in zip(_bars_ku, _ku_vals[::-1]):
    _ax_ku.text(_kv + (0.2 if _kv >= 0 else -0.2), _bar.get_y() + _bar.get_height() / 2,
                f"{_kv:.1f}", va="center", ha="left" if _kv >= 0 else "right",
                fontsize=8.5, color=_TEXT)
_ax_ku.set_title("Excess Kurtosis (0 = normal)", color=_TEXT, fontsize=11, fontweight="bold")
_ax_ku.set_xlabel("Excess Kurtosis (Fisher)", color=_MUTED, fontsize=10)
_ax_ku.spines[["top", "right"]].set_visible(False)
_ax_ku.tick_params(labelsize=9)
_ax_ku.set_facecolor(_BG)
_ax_ku.legend(facecolor=_BG, edgecolor=_MUTED, labelcolor=_TEXT, fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("behavioral_skew_kurtosis.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

# ── CHART 3: Box-plots (variance & outliers across all features) ──────────
# Normalise to [0,1] so all features fit on one axis
_df_norm = _data[_FEAT_COLS].copy()
for _fc in _FEAT_COLS:
    _mn_v, _mx_v = _df_norm[_fc].min(), _df_norm[_fc].max()
    if _mx_v > _mn_v:
        _df_norm[_fc] = (_df_norm[_fc] - _mn_v) / (_mx_v - _mn_v)

fig_boxplots, _ax_bp = plt.subplots(figsize=(16, 7))
fig_boxplots.patch.set_facecolor(_BG)
_ax_bp.set_facecolor(_BG)

_bp_data = [_df_norm[_fc].dropna().values for _fc in _FEAT_COLS]
_bp = _ax_bp.boxplot(
    _bp_data,
    patch_artist=True,
    notch=False,
    vert=True,
    widths=0.5,
    flierprops=dict(marker="o", markersize=2.5, alpha=0.4, markeredgewidth=0),
    medianprops=dict(color=_YELLOW, linewidth=2),
    whiskerprops=dict(color=_MUTED, linewidth=1.2),
    capprops=dict(color=_MUTED, linewidth=1.2),
    boxprops=dict(linewidth=1.2),
)

for _patch, _color in zip(_bp["boxes"], _PALETTE):
    _patch.set_facecolor(_color)
    _patch.set_alpha(0.75)
for _flier, _color in zip(_bp["fliers"], _PALETTE):
    _flier.set(markerfacecolor=_color, markeredgecolor=_color)

_ax_bp.set_xticks(range(1, len(_FEAT_COLS) + 1))
_ax_bp.set_xticklabels(
    [_FEAT_LABELS.get(fc, fc).replace(" (7d)", "").replace(" (min)", "")
     for fc in _FEAT_COLS],
    rotation=35, ha="right", fontsize=9
)
_ax_bp.set_ylabel("Normalised Value [0–1]", color=_MUTED, fontsize=10)
_ax_bp.set_title(
    "Box Plots — All Behavioral Features (Normalised)\nMedian (yellow line) · IQR box · 1.5×IQR whiskers · Outliers as dots",
    color=_TEXT, fontsize=13, fontweight="bold", pad=10
)
_ax_bp.spines[["top", "right"]].set_visible(False)
_ax_bp.tick_params(axis="x", colors=_TEXT, labelsize=9)
_ax_bp.tick_params(axis="y", colors=_MUTED, labelsize=9)
_ax_bp.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig("behavioral_boxplots.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

# ── CHART 4: Mean vs Median divergence (central tendency comparison) ──────
_means_v  = [descriptive_stats_results.loc[fc, "mean"]   for fc in _FEAT_COLS]
_medians_v = [descriptive_stats_results.loc[fc, "median"] for fc in _FEAT_COLS]
_short_lbls = [_FEAT_LABELS.get(fc, fc).replace(" (7d)", "").replace(" (min)", "")
               for fc in _FEAT_COLS]

# Normalise for comparability within each feature
_ratios = [
    (m - med) / max(abs(med), 0.01)
    for m, med in zip(_means_v, _medians_v)
]

fig_central, _ax_ct = plt.subplots(figsize=(14, 6))
fig_central.patch.set_facecolor(_BG)
_ax_ct.set_facecolor(_BG)

_ct_colors = [_CORAL if r > 0.5 else _SUCC if r < -0.5 else _BLUE for r in _ratios]
_ct_bars = _ax_ct.bar(_short_lbls, _ratios, color=_ct_colors, edgecolor="none", width=0.65)
_ax_ct.axhline(0, color=_MUTED, linewidth=1.2)
_ax_ct.axhline(0.5,  color=_YELLOW, linewidth=0.9, linestyle="--", alpha=0.6, label="±50% divergence")
_ax_ct.axhline(-0.5, color=_YELLOW, linewidth=0.9, linestyle="--", alpha=0.6)

for _bar, _r in zip(_ct_bars, _ratios):
    _ax_ct.text(_bar.get_x() + _bar.get_width() / 2,
                _r + (0.03 if _r >= 0 else -0.06),
                f"{_r:.2f}", ha="center", va="bottom" if _r >= 0 else "top",
                fontsize=8, color=_TEXT)

_ax_ct.set_ylabel("(Mean − Median) / |Median|", color=_MUTED, fontsize=10)
_ax_ct.set_title(
    "Mean vs. Median Divergence — Central Tendency Comparison\n"
    "Positive = right-skewed (outliers pull mean up) · Negative = left-skewed",
    color=_TEXT, fontsize=12, fontweight="bold", pad=10
)
_ax_ct.spines[["top", "right"]].set_visible(False)
_ax_ct.tick_params(axis="x", rotation=35, labelsize=8.5, colors=_TEXT)
_ax_ct.tick_params(axis="y", colors=_MUTED, labelsize=9)
_ax_ct.legend(facecolor=_BG, edgecolor=_MUTED, labelcolor=_TEXT, fontsize=9)

plt.tight_layout()
plt.savefig("behavioral_central_tendency.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

print("\n✅ All visualisations saved:")
print("   • behavioral_feature_histograms.png — individual histograms for all 13 features")
print("   • behavioral_skew_kurtosis.png      — skewness & kurtosis shape comparison")
print("   • behavioral_boxplots.png           — normalised box plots across all features")
print("   • behavioral_central_tendency.png   — mean vs. median divergence chart")
print(f"\n📦 descriptive_stats_results: {descriptive_stats_results.shape[0]} features × {descriptive_stats_results.shape[1]} statistics stored for downstream use")
