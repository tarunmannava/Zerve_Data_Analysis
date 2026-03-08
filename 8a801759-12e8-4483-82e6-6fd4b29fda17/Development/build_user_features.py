import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Color palette ─────────────────────────────────────────────────────────
BG_F = "#1D1D20"; TEXT_F = "#fbfbff"; MUTED_F = "#909094"
CORAL_F = "#FF9F9B"; SUCCESS_F = "#17b26a"
BLUE_F = "#A1C9F4"; YELLOW_F = "#ffd400"

# ── 1. Prepare raw events ─────────────────────────────────────────────────
_raw = df[["person_id", "distinct_id", "timestamp", "event",
           "prop_tool_name", "prop_session_id",
           "prop_credit_amount", "prop_credits_used"]].copy()

_raw["timestamp"]  = pd.to_datetime(_raw["timestamp"], utc=True, errors="coerce")
_raw["user_key"]   = _raw["person_id"].fillna(_raw["distinct_id"])
_raw = _raw.dropna(subset=["timestamp", "user_key"])
_raw = _raw.sort_values(["user_key", "timestamp"]).reset_index(drop=True)

print("=" * 60)
print("FEATURE ENGINEERING — EARLY PERIOD EVENTS (≤7 days)")
print("=" * 60)
print(f"Columns in _raw : {list(_raw.columns)}")
print(f"Total events    : {len(_raw):,}")

# ── 2. Compute per-user first timestamp (vectorized) ─────────────────────
_first_ts = _raw.groupby("user_key")["timestamp"].transform("min")
_raw["first_event_ts"]   = _first_ts
_raw["days_since_first"] = (_raw["timestamp"] - _raw["first_event_ts"]).dt.total_seconds() / 86400

# ── 3. Restrict to early period ───────────────────────────────────────────
EARLY_DAYS = 7
_early = _raw[_raw["days_since_first"] <= EARLY_DAYS].copy()
print(f"Early-period rows: {len(_early):,}  ({len(_early)/len(_raw)*100:.1f}%)")

# ── 4. Vectorized session assignment (30-min gap rule) ───────────────────
_SESSION_GAP_MIN = 30
_early = _early.sort_values(["user_key", "timestamp"]).reset_index(drop=True)
_time_diff_min = _early["timestamp"].diff().dt.total_seconds() / 60
_new_user       = _early["user_key"] != _early["user_key"].shift(1)
_new_session    = _new_user | (_time_diff_min > _SESSION_GAP_MIN) | _time_diff_min.isna()
_early["session_num"] = _new_session.cumsum()   # globally unique session IDs

print(f"\nSample _early columns: {list(_early.columns)}")
print(f"Unique users in early: {_early['user_key'].nunique():,}")
print(f"Total sessions       : {_early['session_num'].nunique():,}")

# ── 5. Tool events ────────────────────────────────────────────────────────
_early["tool_clean"]    = _early["prop_tool_name"].fillna("").str.strip()
_early["is_tool_event"] = (_early["tool_clean"] != "").astype(int)

print("\nBuilding features...")

# ── 6. Session-level table ────────────────────────────────────────────────
_sess = (
    _early.groupby(["user_key", "session_num"])
    .agg(
        sess_events    = ("timestamp", "count"),
        sess_start     = ("timestamp", "min"),
        sess_end       = ("timestamp", "max"),
        sess_has_tool  = ("is_tool_event", "max"),
    )
    .reset_index()
)
_sess["sess_dur_min"] = (
    (_sess["sess_end"] - _sess["sess_start"]).dt.total_seconds() / 60
)

# ── 6a. User-level session features ──────────────────────────────────────
_user_sess = (
    _sess.groupby("user_key")
    .agg(
        feat_session_count       = ("session_num", "count"),
        feat_avg_events_per_sess = ("sess_events", "mean"),
        feat_median_sess_dur_min = ("sess_dur_min", "median"),
        feat_max_events_in_sess  = ("sess_events", "max"),
        feat_tool_session_count  = ("sess_has_tool", "sum"),
    )
    .reset_index()
)

# ── 6b. Time-to-second-session (vectorized) ───────────────────────────────
_sess_sorted = _sess.sort_values(["user_key", "sess_start"])
_sess_sorted["_rank"] = _sess_sorted.groupby("user_key").cumcount()
_first_sess  = _sess_sorted[_sess_sorted["_rank"] == 0][["user_key", "sess_start"]].rename(columns={"sess_start": "_t1"})
_second_sess = _sess_sorted[_sess_sorted["_rank"] == 1][["user_key", "sess_start"]].rename(columns={"sess_start": "_t2"})
_t2s = _first_sess.merge(_second_sess, on="user_key", how="left")
_t2s["feat_time_to_2nd_sess_min"] = (
    (_t2s["_t2"] - _t2s["_t1"]).dt.total_seconds() / 60
)
_t2s = _t2s[["user_key", "feat_time_to_2nd_sess_min"]]

# ── 6c. User-level event & tool features ─────────────────────────────────
_user_events = (
    _early.groupby("user_key")
    .agg(
        feat_total_early_events = ("timestamp", "count"),
        feat_unique_event_types = ("event", "nunique"),
        feat_unique_tools_used  = ("tool_clean", lambda x: (x[x != ""]).nunique()),
        feat_early_credit_raw   = ("prop_credit_amount", "sum"),
        feat_early_credits_used = ("prop_credits_used", "sum"),
        feat_unique_days_early  = ("timestamp", lambda x: x.dt.normalize().nunique()),
        feat_tool_event_rate    = ("is_tool_event", "mean"),
    )
    .reset_index()
)
_user_events["feat_early_credit_total"] = (
    _user_events[["feat_early_credit_raw", "feat_early_credits_used"]]
    .max(axis=1).fillna(0)
)
_user_events["feat_has_early_credit"] = (
    (_user_events["feat_early_credit_total"] > 0).astype(int)
)
_user_events.drop(columns=["feat_early_credit_raw", "feat_early_credits_used"], inplace=True)

# ── 7. Merge all feature tables ───────────────────────────────────────────
_feat = (
    _user_events
    .merge(_user_sess, on="user_key", how="left")
    .merge(_t2s,       on="user_key", how="left")
)

print(f"\n_feat shape: {_feat.shape}, columns: {list(_feat.columns)}")

# ── 8. Merge with success labels ─────────────────────────────────────────
user_feature_matrix = (
    user_success_df[[
        "user_key", "first_event", "long_term_success",
        "lifespan_days", "lifespan_bucket"
    ]]
    .merge(_feat, on="user_key", how="left")
)

_feat_cols = [c for c in user_feature_matrix.columns if c.startswith("feat_")]
user_feature_matrix[_feat_cols] = user_feature_matrix[_feat_cols].fillna(0)

print(f"\n✅ user_feature_matrix: {user_feature_matrix.shape[0]:,} × {user_feature_matrix.shape[1]}")
print(f"   Features ({len(_feat_cols)}): {_feat_cols}")

_vc = user_feature_matrix["long_term_success"].value_counts().sort_index()
print(f"\nLabel distribution:")
print(f"  Success (1)    : {_vc.get(1,0):,}  ({_vc.get(1,0)/len(user_feature_matrix)*100:.1f}%)")
print(f"  No success (0) : {_vc.get(0,0):,}  ({_vc.get(0,0)/len(user_feature_matrix)*100:.1f}%)")

_nulls = user_feature_matrix[_feat_cols].isnull().sum()
print(f"\nNull values in features:")
print(_nulls[_nulls > 0].to_string() if _nulls.sum() > 0 else "  ✅ None — clean!")

print(f"\nFeature summary statistics:")
print(user_feature_matrix[_feat_cols].describe().round(3).to_string())

# ── 9. Feature distributions (success vs. not) ───────────────────────────
plt.rcParams.update({
    "figure.facecolor": BG_F, "axes.facecolor": BG_F,
    "text.color": TEXT_F, "axes.labelcolor": TEXT_F,
    "xtick.color": TEXT_F, "ytick.color": TEXT_F,
    "axes.edgecolor": MUTED_F, "grid.color": "#333338",
    "font.family": "sans-serif"
})

_success_mask = user_feature_matrix["long_term_success"] == 1

_compare_feats = [
    ("feat_session_count",          "Session Count (7d)"),
    ("feat_unique_event_types",     "Unique Event Types (7d)"),
    ("feat_unique_tools_used",      "Unique Tools Used (7d)"),
    ("feat_early_credit_total",     "Early Credit Consumed (7d)"),
    ("feat_avg_events_per_sess",    "Avg Events / Session (7d)"),
    ("feat_time_to_2nd_sess_min",   "Time to 2nd Session (min)"),
    ("feat_total_early_events",     "Total Early Events (7d)"),
    ("feat_unique_days_early",      "Active Days (7d)"),
]

fig_dist, axes = plt.subplots(2, 4, figsize=(18, 9))
fig_dist.patch.set_facecolor(BG_F)
fig_dist.suptitle(
    "Feature Distributions: Success vs. No Success (First 7 Days)",
    color=TEXT_F, fontsize=15, fontweight="bold", y=1.01
)

for _ax, (feat, label) in zip(axes.flat, _compare_feats):
    _ax.set_facecolor(BG_F)
    _s0 = user_feature_matrix.loc[~_success_mask, feat].dropna()
    _s1 = user_feature_matrix.loc[_success_mask,  feat].dropna()
    _cap = np.percentile(user_feature_matrix[feat].dropna(), 99)
    _s0 = _s0.clip(upper=_cap); _s1 = _s1.clip(upper=_cap)
    _bins = np.linspace(0, max(_cap, 1), 30)
    _ax.hist(_s0, bins=_bins, color=CORAL_F,  alpha=0.7, label="No Success", density=True)
    _ax.hist(_s1, bins=_bins, color=SUCCESS_F, alpha=0.8, label="Success",    density=True)
    _ax.set_title(label, color=TEXT_F, fontsize=10, fontweight="bold")
    _ax.set_xlabel("Value", color=MUTED_F, fontsize=8)
    _ax.set_ylabel("Density", color=MUTED_F, fontsize=8)
    _ax.tick_params(labelsize=8)
    _ax.spines[["top", "right"]].set_visible(False)

axes.flat[0].legend(facecolor=BG_F, edgecolor=MUTED_F, labelcolor=TEXT_F, fontsize=9)
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=150, bbox_inches="tight", facecolor=BG_F)
plt.show()

# ── 10. Feature means comparison ─────────────────────────────────────────
print("\n" + "=" * 70)
print("FEATURE MEANS: SUCCESS vs. NO SUCCESS")
print("=" * 70)
_means = user_feature_matrix.groupby("long_term_success")[_feat_cols].mean().T
_means.columns = ["No Success (0)", "Success (1)"]
_means["Ratio (S/NS)"] = (
    _means["Success (1)"] / _means["No Success (0)"].replace(0, np.nan)
).round(2)
print(_means.round(4).to_string())

print(f"\n🎉 Feature matrix ready for modelling!")
print(f"   → Variable : user_feature_matrix  ({user_feature_matrix.shape[0]:,} × {user_feature_matrix.shape[1]})")
print(f"   → Label    : long_term_success (0/1)")
print(f"   → Features : {len(_feat_cols)} early-period behavioral signals")
print(f"   → No leakage: all features computed from first {EARLY_DAYS} days only")
