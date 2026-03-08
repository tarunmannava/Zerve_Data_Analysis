import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

# ── Setup & Constants ──────────────────────────────────────────────────────
ACTIVITY_THRESHOLD_DAYS = 30   # Must return after 30 days
CREDIT_THRESHOLD = 0           # Must have consumed any credits (> 0)

BG = "#1D1D20"
TEXT = "#fbfbff"
MUTED = "#909094"
BLUE = "#A1C9F4"
GREEN = "#8DE5A1"
ORANGE = "#FFB482"
CORAL = "#FF9F9B"
LAVENDER = "#D0BBFF"
YELLOW = "#ffd400"
SUCCESS_GREEN = "#17b26a"
WARN_RED = "#f04438"

# ── 1. Parse timestamps & identify relevant columns ───────────────────────
_raw = df.copy()
_raw["timestamp"] = pd.to_datetime(_raw["timestamp"], utc=True, errors="coerce")
_raw["created_at"] = pd.to_datetime(_raw["created_at"], utc=True, errors="coerce")

# Use person_id as stable user identifier (prefer over distinct_id which can merge)
_raw["user_key"] = _raw["person_id"].fillna(_raw["distinct_id"])

# ── 2. Build user-level aggregates ────────────────────────────────────────
print("=" * 60)
print("BUILDING USER-LEVEL AGGREGATES")
print("=" * 60)

_user_agg = (
    _raw.groupby("user_key")
    .agg(
        first_event       = ("timestamp", "min"),
        last_event        = ("timestamp", "max"),
        total_events      = ("timestamp", "count"),
        total_credits     = ("prop_credit_amount", "sum"),
        total_credits_used= ("prop_credits_used", "sum"),
        unique_event_types= ("event", "nunique"),
        unique_days       = ("timestamp", lambda x: x.dt.normalize().nunique()),
    )
    .reset_index()
)

# Days since first event to last event (lifespan in the dataset)
_user_agg["lifespan_days"] = (
    (_user_agg["last_event"] - _user_agg["first_event"])
    .dt.total_seconds() / 86400
)

# Credits combined (some may be in credit_amount, some in credits_used)
# Use the max of either to capture actual usage
_user_agg["credits_consumed"] = _user_agg[["total_credits", "total_credits_used"]].max(axis=1).fillna(0)

print(f"Total unique users: {len(_user_agg):,}")
print(f"\nLifespan stats (days):")
print(_user_agg["lifespan_days"].describe().round(2).to_string())
print(f"\nCredits consumed stats:")
print(_user_agg["credits_consumed"].describe().round(4).to_string())

# ── 3. Define 'long-term success' criteria ────────────────────────────────
# SUCCESS = returned 30+ days after first event AND consumed any credits
_user_agg["returned_after_30d"] = _user_agg["lifespan_days"] >= ACTIVITY_THRESHOLD_DAYS
_user_agg["has_credit_consumption"] = _user_agg["credits_consumed"] > CREDIT_THRESHOLD

_user_agg["long_term_success"] = (
    _user_agg["returned_after_30d"] & _user_agg["has_credit_consumption"]
).astype(int)

# ── 4. Print key stats ────────────────────────────────────────────────────
total_users = len(_user_agg)
success_count = _user_agg["long_term_success"].sum()
success_rate = success_count / total_users * 100

returned_30d = _user_agg["returned_after_30d"].sum()
has_credits  = _user_agg["has_credit_consumption"].sum()
both_but_no = returned_30d + has_credits - success_count

print("\n" + "=" * 60)
print("LONG-TERM SUCCESS LABEL — KEY STATS")
print("=" * 60)
print(f"  Total users                : {total_users:,}")
print(f"  Success (label = 1)        : {success_count:,}  ({success_rate:.1f}%)")
print(f"  Not successful (label = 0) : {total_users - success_count:,}  ({100 - success_rate:.1f}%)")
print(f"\n  — Breakdown of criteria —")
print(f"  Returned 30+ days          : {returned_30d:,}  ({returned_30d/total_users*100:.1f}%)")
print(f"  Has credit consumption     : {has_credits:,}  ({has_credits/total_users*100:.1f}%)")
print(f"\n  Thresholds used:")
print(f"    Activity threshold       : {ACTIVITY_THRESHOLD_DAYS} days")
print(f"    Credit threshold         : > {CREDIT_THRESHOLD}")

# ── 5. Cohort sizes by lifespan bucket ────────────────────────────────────
_bins = [0, 1, 7, 14, 30, 60, 90, 180, np.inf]
_labels = ["<1d", "1-7d", "7-14d", "14-30d", "30-60d", "60-90d", "90-180d", "180d+"]
_user_agg["lifespan_bucket"] = pd.cut(
    _user_agg["lifespan_days"], bins=_bins, labels=_labels, right=False
)
_cohort = (
    _user_agg.groupby("lifespan_bucket", observed=False)
    .agg(
        n_users        = ("user_key", "count"),
        n_success      = ("long_term_success", "sum"),
    )
    .assign(success_rate=lambda x: (x["n_success"] / x["n_users"] * 100).round(1))
    .reset_index()
)

print("\n" + "=" * 60)
print("ACTIVE COHORT SIZES BY USER LIFESPAN")
print("=" * 60)
print(_cohort.to_string(index=False))

# ── 6. Retention curve: % of users active at each day threshold ───────────
_day_thresholds = [1, 3, 7, 14, 21, 30, 45, 60, 90]
_retention = []
for d in _day_thresholds:
    n = (_user_agg["lifespan_days"] >= d).sum()
    _retention.append({"days": d, "n_users": n, "retention_pct": n / total_users * 100})
_retention_df = pd.DataFrame(_retention)

print("\n" + "=" * 60)
print("USER RETENTION CURVE")
print("=" * 60)
print(_retention_df.to_string(index=False))

# ── 7. Expose the clean user-level DataFrame ──────────────────────────────
user_success_df = _user_agg[[
    "user_key", "first_event", "last_event", "lifespan_days",
    "total_events", "unique_days", "unique_event_types",
    "credits_consumed", "returned_after_30d", "has_credit_consumption",
    "long_term_success", "lifespan_bucket"
]].copy()

print(f"\n✅ user_success_df created: {user_success_df.shape[0]:,} users × {user_success_df.shape[1]} features")
print(f"   Success label column: 'long_term_success' (binary: 0/1)")

# ── 8. Visualizations ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.edgecolor": MUTED, "grid.color": "#333338",
    "font.family": "sans-serif"
})

# ── Plot 1: Success label distribution ────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))
fig1.patch.set_facecolor(BG)
labels_bar = ["Not Successful\n(label = 0)", "Long-Term Success\n(label = 1)"]
counts_bar = [total_users - success_count, success_count]
colors_bar = [CORAL, SUCCESS_GREEN]
bars = ax1.bar(labels_bar, counts_bar, color=colors_bar, edgecolor=BG, linewidth=1.5, width=0.55)
for bar, count in zip(bars, counts_bar):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total_users * 0.01,
        f"{count:,}\n({count/total_users*100:.1f}%)",
        ha="center", va="bottom", color=TEXT, fontsize=12, fontweight="bold"
    )
ax1.set_title("Long-Term Success Label Distribution", color=TEXT, fontsize=15, fontweight="bold", pad=14)
ax1.set_ylabel("Number of Users", color=TEXT, fontsize=11)
ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
ax1.set_ylim(0, max(counts_bar) * 1.18)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(labelsize=11)
plt.tight_layout()
plt.savefig("success_label_distribution.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()

# ── Plot 2: Retention curve ────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))
fig2.patch.set_facecolor(BG)
ax2.plot(
    _retention_df["days"], _retention_df["retention_pct"],
    color=BLUE, linewidth=2.5, marker="o", markersize=7, markerfacecolor=YELLOW
)
ax2.fill_between(_retention_df["days"], _retention_df["retention_pct"], alpha=0.18, color=BLUE)
ax2.axvline(x=30, color=YELLOW, linestyle="--", linewidth=1.5, label="30-day threshold")
for _, row in _retention_df.iterrows():
    ax2.annotate(
        f"{row['retention_pct']:.1f}%",
        (row["days"], row["retention_pct"]),
        textcoords="offset points", xytext=(0, 10),
        ha="center", fontsize=9, color=TEXT
    )
ax2.set_title("User Retention Curve (% Active at Each Day Threshold)", color=TEXT, fontsize=14, fontweight="bold", pad=14)
ax2.set_xlabel("Days Since First Event", color=TEXT, fontsize=11)
ax2.set_ylabel("% of Users Still Active", color=TEXT, fontsize=11)
ax2.legend(facecolor=BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-2, _retention_df["days"].max() + 5)
ax2.set_ylim(0, 105)
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("retention_curve.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()

# ── Plot 3: Cohort sizes (lifespan buckets) ────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))
fig3.patch.set_facecolor(BG)
_x = range(len(_cohort))
_bar_w = 0.38
_bars_total = ax3.bar(
    [x - _bar_w/2 for x in _x], _cohort["n_users"],
    width=_bar_w, color=BLUE, label="Total Users", edgecolor=BG
)
_bars_success = ax3.bar(
    [x + _bar_w/2 for x in _x], _cohort["n_success"],
    width=_bar_w, color=SUCCESS_GREEN, label="Successful Users", edgecolor=BG
)
ax3.set_xticks(list(_x))
ax3.set_xticklabels(_cohort["lifespan_bucket"].astype(str), fontsize=10, rotation=20, ha="right")
ax3.set_title("User Cohort Sizes by Lifespan Bucket", color=TEXT, fontsize=14, fontweight="bold", pad=14)
ax3.set_ylabel("Number of Users", color=TEXT, fontsize=11)
ax3.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
ax3.legend(facecolor=BG, edgecolor=MUTED, labelcolor=TEXT, fontsize=10)
ax3.spines[["top", "right"]].set_visible(False)
ax3.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("cohort_sizes.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()

print("\n🎉 All visualizations rendered successfully.")
