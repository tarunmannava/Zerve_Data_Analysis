
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ── Color palette ─────────────────────────────────────────────────────────
_BG    = "#1D1D20"; _TEXT  = "#fbfbff"; _MUTED = "#909094"
_BLUE  = "#A1C9F4"; _ORANGE = "#FFB482"; _GREEN = "#8DE5A1"
_CORAL = "#FF9F9B"; _LAVEN = "#D0BBFF"; _YELL  = "#ffd400"
_SUCC  = "#17b26a"; _WARN  = "#f04438"

_ARCHETYPE_COLORS = [_BLUE, _ORANGE, _GREEN, _CORAL, _LAVEN, _YELL]

plt.rcParams.update({
    "figure.facecolor": _BG, "axes.facecolor": _BG,
    "text.color": _TEXT, "axes.labelcolor": _TEXT,
    "xtick.color": _TEXT, "ytick.color": _TEXT,
    "axes.edgecolor": _MUTED, "grid.color": "#333338",
    "font.family": "sans-serif"
})

# ═══════════════════════════════════════════════════════════════════════════
# 1. Prepare raw event data
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("CONTENT ANALYSIS — EVENT SEQUENCE & WORKFLOW ARCHETYPES")
print("=" * 65)

_raw = df[["person_id", "distinct_id", "timestamp", "event", "prop_tool_name"]].copy()
_raw["timestamp"] = pd.to_datetime(_raw["timestamp"], utc=True, errors="coerce")
_raw["user_key"]  = _raw["person_id"].fillna(_raw["distinct_id"])
_raw = _raw.dropna(subset=["timestamp", "user_key"]).sort_values(["user_key", "timestamp"])

# Clean event / tool labels
_raw["event_clean"]      = _raw["event"].str.strip().str.lower()
_raw["tool_clean"]       = _raw["prop_tool_name"].fillna("").str.strip().str.lower()
_raw["event_with_tool"]  = np.where(
    _raw["tool_clean"] != "",
    "use_tool:" + _raw["tool_clean"],
    _raw["event_clean"]
)

print(f"Total events  : {len(_raw):,}")
print(f"Unique users  : {_raw['user_key'].nunique():,}")
print(f"Unique events : {_raw['event_clean'].nunique():,}")
print(f"Unique tools  : {(_raw['tool_clean'][_raw['tool_clean'] != '']).nunique():,}")

# ── Top raw events ─────────────────────────────────────────────────────────
_top_events = _raw["event_clean"].value_counts().head(20)
print("\nTop-20 events:")
print(_top_events.to_string())

_top_tools = _raw.loc[_raw["tool_clean"] != "", "tool_clean"].value_counts().head(10)
print("\nTop-10 tools:")
print(_top_tools.to_string())

# ═══════════════════════════════════════════════════════════════════════════
# 2. Build per-user ordered event sequences (first 7 days, up to 30 events)
# ═══════════════════════════════════════════════════════════════════════════
_first_ts = _raw.groupby("user_key")["timestamp"].transform("min")
_raw["days_since_first"] = (_raw["timestamp"] - _first_ts).dt.total_seconds() / 86400
_early = _raw[_raw["days_since_first"] <= 7].copy()

# Build sequence per user: ordered list of event tokens
_user_seqs = (
    _early.groupby("user_key")["event_with_tool"]
    .apply(lambda x: list(x.iloc[:30]))   # cap at 30 events per user
    .reset_index()
    .rename(columns={"event_with_tool": "sequence"})
)

print(f"\nUsers with event sequences: {len(_user_seqs):,}")
print(f"Median sequence length    : {_user_seqs['sequence'].apply(len).median():.0f}")
print(f"Max sequence length       : {_user_seqs['sequence'].apply(len).max():,}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Top-10 most common workflow sequences (n-gram approach: first 5 events)
# ═══════════════════════════════════════════════════════════════════════════
def _condense_seq(seq, n=5):
    """Take first n events, replacing rare events with their category prefix."""
    return tuple(seq[:n])

_user_seqs["seq_5gram"] = _user_seqs["sequence"].apply(_condense_seq)
_seq_counts = Counter(_user_seqs["seq_5gram"])
_top10_seqs = _seq_counts.most_common(10)

print("\n" + "=" * 65)
print("TOP-10 MOST COMMON WORKFLOW SEQUENCES (first 5 events)")
print("=" * 65)
for _rank, (_seq, _cnt) in enumerate(_top10_seqs, 1):
    _pct = _cnt / len(_user_seqs) * 100
    _seq_str = " → ".join(_seq) if _seq else "(empty)"
    print(f"  {_rank:>2}. [{_cnt:>4} users, {_pct:4.1f}%] {_seq_str}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Feature engineering for clustering (event-type frequency profile)
# ═══════════════════════════════════════════════════════════════════════════
# Use top N event tokens as features
_top_n_events = 30
_vocab = [e for e, _ in
          Counter([tok for seq in _user_seqs["sequence"] for tok in seq]).most_common(_top_n_events)]

print(f"\nVocabulary size for clustering: {len(_vocab)}")

def _seq_to_vec(seq):
    _c = Counter(seq)
    return [_c.get(e, 0) for e in _vocab]

_X = np.array([_seq_to_vec(s) for s in _user_seqs["sequence"]], dtype=float)

# Additional signals: sequence length, tool rate
_seq_lens   = np.array([len(s) for s in _user_seqs["sequence"]], dtype=float).reshape(-1, 1)
_tool_rates = np.array(
    [sum(1 for e in s if e.startswith("use_tool:")) / max(len(s), 1)
     for s in _user_seqs["sequence"]], dtype=float
).reshape(-1, 1)
_X_aug = np.hstack([_X, _seq_lens, _tool_rates])

_scaler = StandardScaler()
_X_scaled = _scaler.fit_transform(_X_aug)

# ── Choose k via silhouette (k = 3..6) ────────────────────────────────────
_sil_scores = {}
for _k in range(3, 7):
    _km = KMeans(n_clusters=_k, random_state=42, n_init=10)
    _labels_tmp = _km.fit_predict(_X_scaled)
    _sil_scores[_k] = silhouette_score(_X_scaled, _labels_tmp, sample_size=min(2000, len(_X_scaled)))

_best_k = max(_sil_scores, key=_sil_scores.get)
print(f"\nSilhouette scores: { {k: round(v, 3) for k, v in _sil_scores.items()} }")
print(f"Best k (clusters): {_best_k}")

_km_final = KMeans(n_clusters=_best_k, random_state=42, n_init=10)
_cluster_labels = _km_final.fit_predict(_X_scaled)
_user_seqs["cluster"] = _cluster_labels

# ═══════════════════════════════════════════════════════════════════════════
# 5. Inductive coding — assign archetype names per cluster
# ═══════════════════════════════════════════════════════════════════════════
def _cluster_profile(cluster_id):
    _mask = _user_seqs["cluster"] == cluster_id
    _seqs = _user_seqs.loc[_mask, "sequence"]
    _all_events = [e for seq in _seqs for e in seq]
    _top_events_cl = Counter(_all_events).most_common(8)
    _tool_rate = sum(1 for e in _all_events if e.startswith("use_tool:")) / max(len(_all_events), 1)
    _avg_len = _seqs.apply(len).mean()
    return {
        "n_users": _mask.sum(),
        "avg_len": _avg_len,
        "tool_rate": _tool_rate,
        "top_events": _top_events_cl,
    }

_profiles = {c: _cluster_profile(c) for c in range(_best_k)}

print("\n" + "=" * 65)
print("CLUSTER PROFILES")
print("=" * 65)
for _c, _p in _profiles.items():
    print(f"\nCluster {_c}: {_p['n_users']} users | avg_len={_p['avg_len']:.1f} | tool_rate={_p['tool_rate']:.2%}")
    print(f"  Top events: {[e for e, _ in _p['top_events'][:5]]}")

# ── Inductive archetype naming rules ──────────────────────────────────────
def _assign_archetype(profile):
    _tl   = profile["tool_rate"]
    _len  = profile["avg_len"]
    _top  = [e for e, _ in profile["top_events"][:6]]
    _has_onboarding = any("onboarding" in e for e in _top)
    _has_tool       = any("use_tool:" in e for e in _top)
    _has_agent      = any("agent" in e for e in _top)
    _has_block      = any("block" in e for e in _top)
    _has_canvas     = any("canvas" in e for e in _top)

    if _tl > 0.25 and _has_agent:
        return "AI Power User"
    elif _tl > 0.15 or _has_tool:
        return "Builder"
    elif _has_onboarding and _len < 10:
        return "Explorer"
    elif _has_block and _len >= 8:
        return "Analyst"
    elif _len <= 5:
        return "Casual Visitor"
    else:
        return "Active Learner"

_archetype_map = {c: _assign_archetype(_profiles[c]) for c in range(_best_k)}
print("\nArchetype assignments:")
for _c, _name in _archetype_map.items():
    print(f"  Cluster {_c} → '{_name}'  ({_profiles[_c]['n_users']} users)")

_user_seqs["archetype"] = _user_seqs["cluster"].map(_archetype_map)

# ═══════════════════════════════════════════════════════════════════════════
# 6. Merge with success labels
# ═══════════════════════════════════════════════════════════════════════════
_seq_success = _user_seqs.merge(
    user_success_df[["user_key", "long_term_success"]],
    on="user_key", how="left"
)
_seq_success["long_term_success"] = _seq_success["long_term_success"].fillna(0).astype(int)

# Archetype distribution
_arch_dist = (
    _seq_success.groupby("archetype")
    .agg(
        n_users      = ("user_key", "count"),
        n_success    = ("long_term_success", "sum"),
    )
    .assign(success_rate=lambda x: (x["n_success"] / x["n_users"] * 100).round(2))
    .sort_values("n_users", ascending=False)
    .reset_index()
)

print("\n" + "=" * 65)
print("ARCHETYPE DISTRIBUTION & SUCCESS CORRELATION")
print("=" * 65)
print(_arch_dist.to_string(index=False))
print()

_best_arch  = _arch_dist.sort_values("success_rate", ascending=False).iloc[0]
print(f"  ✅ Highest success rate archetype: '{_best_arch['archetype']}' ({_best_arch['success_rate']:.1f}%)")
_most_arch  = _arch_dist.iloc[0]
print(f"  📊 Largest archetype: '{_most_arch['archetype']}' ({_most_arch['n_users']:,} users, {_most_arch['n_users']/len(_seq_success)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# 7. Visualisation 1 — Workflow frequency bar chart
# ═══════════════════════════════════════════════════════════════════════════
fig_arch_bar, ax_bar = plt.subplots(figsize=(11, 6))
fig_arch_bar.patch.set_facecolor(_BG)
ax_bar.set_facecolor(_BG)

_sorted = _arch_dist.sort_values("n_users", ascending=True)
_bar_colors = [_ARCHETYPE_COLORS[i % len(_ARCHETYPE_COLORS)] for i in range(len(_sorted))]
_bars = ax_bar.barh(_sorted["archetype"], _sorted["n_users"], color=_bar_colors,
                    edgecolor=_BG, height=0.6)

for _b, (_nu, _sr) in zip(_bars, zip(_sorted["n_users"], _sorted["success_rate"])):
    ax_bar.text(
        _nu + max(_sorted["n_users"]) * 0.01,
        _b.get_y() + _b.get_height() / 2,
        f"{_nu:,}  ({_sr:.1f}% success)",
        va="center", ha="left", color=_TEXT, fontsize=10
    )

ax_bar.set_xlabel("Number of Users", fontsize=11, color=_TEXT)
ax_bar.set_title("Workflow Archetypes — User Distribution & Success Rate",
                 fontsize=14, fontweight="bold", color=_TEXT, pad=14)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.tick_params(labelsize=11)
ax_bar.set_xlim(0, max(_sorted["n_users"]) * 1.35)
plt.tight_layout()
plt.savefig("workflow_archetype_distribution.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# 8. Visualisation 2 — Sankey-style flow (step 1 → 2 → 3 event transitions)
# ═══════════════════════════════════════════════════════════════════════════
# Build transition pairs from step-1 → step-2 → step-3
def _get_transitions(seqs, n_steps=4, top_k=8):
    """Get top transition pairs for a Sankey-like flow plot."""
    _transitions = {}
    for _step in range(n_steps - 1):
        _pairs = Counter()
        for _seq in seqs:
            if len(_seq) > _step + 1:
                _src = _seq[_step][:28]   # truncate long labels
                _tgt = _seq[_step + 1][:28]
                _pairs[(_src, _tgt)] += 1
        # Keep top-k transitions at each step
        _transitions[_step] = _pairs.most_common(top_k)
    return _transitions

_transitions = _get_transitions(_user_seqs["sequence"].tolist(), n_steps=4, top_k=7)

fig_flow, ax_flow = plt.subplots(figsize=(14, 7))
fig_flow.patch.set_facecolor(_BG)
ax_flow.set_facecolor(_BG)

# Draw each step as a column of node boxes
_step_labels = ["Step 1\n(Entry)", "Step 2", "Step 3", "Step 4"]
_step_x = [0.05, 0.38, 0.65, 0.92]
_node_positions = {}   # (step, label) → (x_center, y_center)

for _si, (_step_x_pos, _step_label) in enumerate(zip(_step_x, _step_labels)):
    # Collect nodes for this step
    _nodes_at_step = []
    for _s, _pairs in _transitions.items():
        if _s == _si:
            _nodes_at_step += [p[0] for p, _ in _pairs]
        if _s + 1 == _si:
            _nodes_at_step += [p[1] for p, _ in _pairs]
    _unique_nodes = list(dict.fromkeys(_nodes_at_step))[:8]  # ordered unique

    _n = len(_unique_nodes)
    _y_positions = np.linspace(0.92, 0.08, _n) if _n > 1 else [0.5]
    for _node, _y in zip(_unique_nodes, _y_positions):
        _key = (_si, _node)
        _node_positions[_key] = (_step_x_pos, _y)
        _node_label = _node if len(_node) <= 22 else _node[:20] + "…"
        _box_w, _box_h = 0.28, 0.08
        _rect = plt.Rectangle(
            (_step_x_pos - _box_w / 2, _y - _box_h / 2),
            _box_w, _box_h,
            linewidth=0, facecolor="#2a2a30", transform=ax_flow.transAxes, zorder=3
        )
        ax_flow.add_patch(_rect)
        ax_flow.text(_step_x_pos, _y, _node_label,
                     ha="center", va="center", fontsize=7.5, color=_TEXT,
                     transform=ax_flow.transAxes, zorder=4, wrap=True)

    # Step header
    ax_flow.text(_step_x_pos, 1.03, _step_label,
                 ha="center", va="bottom", fontsize=10, color=_YELL,
                 fontweight="bold", transform=ax_flow.transAxes)

# Draw flow arrows between steps
_max_count = max(cnt for _pairs in _transitions.values() for _, cnt in _pairs)
for _s, _pairs in _transitions.items():
    for (_src, _tgt), _cnt in _pairs:
        _k_src = (_s, _src);  _k_tgt = (_s + 1, _tgt)
        if _k_src in _node_positions and _k_tgt in _node_positions:
            _x0, _y0 = _node_positions[_k_src]
            _x1, _y1 = _node_positions[_k_tgt]
            _alpha = 0.15 + 0.55 * (_cnt / _max_count)
            _lw    = 1.0  + 5.0  * (_cnt / _max_count)
            _color = _ARCHETYPE_COLORS[_s % len(_ARCHETYPE_COLORS)]
            # Bezier-style curve via annotate
            ax_flow.annotate(
                "", xy=(_x1 - 0.14, _y1), xytext=(_x0 + 0.14, _y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-|>", color=_color,
                    lw=_lw, alpha=_alpha,
                    connectionstyle="arc3,rad=0.05"
                )
            )

ax_flow.set_xlim(0, 1); ax_flow.set_ylim(0, 1)
ax_flow.axis("off")
ax_flow.set_title("User Workflow Flow — Top Event Transitions (Steps 1→2→3→4)",
                  fontsize=13, fontweight="bold", color=_TEXT,
                  transform=ax_flow.transAxes, y=1.07)
plt.tight_layout()
plt.savefig("workflow_event_flow.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# 9. Visualisation 3 — Success rate per archetype (bar)
# ═══════════════════════════════════════════════════════════════════════════
fig_succ_rate, ax_sr = plt.subplots(figsize=(10, 5))
fig_succ_rate.patch.set_facecolor(_BG)
ax_sr.set_facecolor(_BG)

_sr_sorted = _arch_dist.sort_values("success_rate", ascending=False)
_sr_colors = [_SUCC if v >= _best_arch["success_rate"] else _BLUE for v in _sr_sorted["success_rate"]]
_sr_bars = ax_sr.bar(_sr_sorted["archetype"], _sr_sorted["success_rate"],
                     color=_sr_colors, edgecolor=_BG, width=0.55)

for _b, _v in zip(_sr_bars, _sr_sorted["success_rate"]):
    ax_sr.text(
        _b.get_x() + _b.get_width() / 2,
        _v + 0.1,
        f"{_v:.1f}%",
        ha="center", va="bottom", fontsize=11, color=_TEXT, fontweight="bold"
    )

ax_sr.set_ylabel("Success Rate (%)", fontsize=11)
ax_sr.set_title("Long-Term Success Rate by Workflow Archetype",
                fontsize=14, fontweight="bold", color=_TEXT, pad=14)
ax_sr.spines[["top", "right"]].set_visible(False)
ax_sr.tick_params(axis="x", labelsize=10, rotation=10)
ax_sr.set_ylim(0, max(_sr_sorted["success_rate"]) * 1.3)
plt.tight_layout()
plt.savefig("workflow_success_by_archetype.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# 10. Store results
# ═══════════════════════════════════════════════════════════════════════════
content_analysis_results = {
    "archetype_distribution": _arch_dist,
    "user_archetype_map": _user_seqs[["user_key", "cluster", "archetype"]],
    "top10_sequences": _top10_seqs,
    "cluster_profiles": _profiles,
    "archetype_map": _archetype_map,
    "silhouette_scores": _sil_scores,
    "best_k": _best_k,
    "seq_with_success": _seq_success[["user_key", "archetype", "long_term_success"]],
}

print("\n" + "=" * 65)
print("✅ CONTENT ANALYSIS COMPLETE")
print("=" * 65)
print(f"  Archetypes identified  : {_best_k}")
print(f"  Users classified       : {len(_user_seqs):,}")
print(f"  Top-10 sequences       : captured")
print(f"  Success correlation    : computed per archetype")
print(f"  Results stored in      : content_analysis_results")
print(f"\n  ARCHETYPE SUMMARY:")
for _, _row in _arch_dist.iterrows():
    _mark = "⭐" if _row["archetype"] == _best_arch["archetype"] else "  "
    print(f"  {_mark} {_row['archetype']:<20} {_row['n_users']:>5} users  |  {_row['success_rate']:.1f}% success")
