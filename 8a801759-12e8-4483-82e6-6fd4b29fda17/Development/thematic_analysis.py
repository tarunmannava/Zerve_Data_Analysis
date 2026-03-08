
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

# ── Zerve Design System ───────────────────────────────────────────────────
_BG      = "#1D1D20"
_TEXT    = "#fbfbff"
_MUTED   = "#909094"
_BLUE    = "#A1C9F4"
_ORANGE  = "#FFB482"
_GREEN   = "#8DE5A1"
_CORAL   = "#FF9F9B"
_LAVEND  = "#D0BBFF"
_YELLOW  = "#ffd400"
_SUCCESS = "#17b26a"
_WARN    = "#f04438"

_THEME_COLORS = [_BLUE, _ORANGE, _GREEN, _CORAL, _LAVEND, _YELLOW]

plt.rcParams.update({
    "figure.facecolor": _BG, "axes.facecolor": _BG,
    "text.color": _TEXT, "axes.labelcolor": _TEXT,
    "xtick.color": _TEXT, "ytick.color": _TEXT,
    "axes.edgecolor": _MUTED, "grid.color": "#333338",
    "font.family": "sans-serif"
})

print("=" * 70)
print("THEMATIC ANALYSIS — TF-IDF + LDA TOPIC MODELING")
print("Following Braun & Clarke (2006) 6-Step Thematic Analysis")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — FAMILIARIZATION WITH THE DATA
# Prepare user-level event text corpora from event names + tool calls
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 1: Familiarization with the Data ──")

_raw = df[["person_id", "distinct_id", "event", "prop_tool_name"]].copy()
_raw["user_key"] = _raw["person_id"].fillna(_raw["distinct_id"])

# Clean text tokens
def _clean_token(s):
    """Normalise event/tool text into space-separated words."""
    if pd.isna(s) or str(s).strip() == "":
        return ""
    s = str(s).strip().lower()
    # Split on common separators
    for sep in ["_", ":", "-", "."]:
        s = s.replace(sep, " ")
    return s.strip()

_raw["event_tokens"]    = _raw["event"].apply(_clean_token)
_raw["tool_tokens"]     = _raw["prop_tool_name"].apply(_clean_token)
_raw["combined_tokens"] = (_raw["event_tokens"] + " " + _raw["tool_tokens"]).str.strip()

# Build per-user document = all their events concatenated
_user_docs = (
    _raw.groupby("user_key")["combined_tokens"]
    .apply(lambda x: " ".join(t for t in x if t))
    .reset_index()
    .rename(columns={"combined_tokens": "document"})
)
_user_docs = _user_docs[_user_docs["document"].str.len() > 0]

# Merge success labels
_user_docs = _user_docs.merge(
    user_success_df[["user_key", "long_term_success"]],
    on="user_key", how="left"
)
_user_docs["long_term_success"] = _user_docs["long_term_success"].fillna(0).astype(int)

print(f"  Users with text documents : {len(_user_docs):,}")
print(f"  Success users in corpus   : {_user_docs['long_term_success'].sum():,}")
print(f"  Vocabulary sample (first 20 event tokens):")
_vocab_sample = list(set(" ".join(_raw["event_tokens"].dropna().head(200)).split()))[:20]
print(f"  {_vocab_sample}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — GENERATING INITIAL CODES
# TF-IDF vectorization — extract weighted term features per user
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 2: Generating Initial Codes (TF-IDF) ──")

# Custom stop words: low-signal platform noise terms
_stop_words = [
    "prop", "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "to", "of", "for", "in", "on", "at", "by", "with", "and", "or",
    "tool", "call", "create", "get", "run", "set", "use", "new",
    "null", "none", "true", "false", "0", "1", "2", "3", "4", "5",
    "sdk", "debug", "session", "replay", "internal", "buffer", "server",
    "side", "enabled", "disabled", "status", "trigger", "linked", "flag",
    "size", "length", "queue", "retry", "below", "above", "current",
    "start", "end", "time", "type", "id", "user", "event", "item",
    "initial", "entry", "referred", "domain", "window", "lib", "version",
    "os", "browser", "screen", "viewport", "device", "geoip", "country",
    "code", "name", "url", "referrer", "pathname", "host"
]

_tfidf = TfidfVectorizer(
    max_features=200,
    min_df=5,           # term must appear in at least 5 docs
    max_df=0.85,        # ignore terms in >85% docs
    ngram_range=(1, 2), # unigrams + bigrams
    stop_words=_stop_words,
    sublinear_tf=True
)

_X_tfidf = _tfidf.fit_transform(_user_docs["document"])
_vocab    = np.array(_tfidf.get_feature_names_out())

print(f"  TF-IDF matrix shape : {_X_tfidf.shape}")
print(f"  Vocabulary size     : {len(_vocab)} terms")
print(f"  Top TF-IDF terms    : {list(_vocab[:30])}")

# Global term weights (mean TF-IDF across users)
_term_weights = np.array(_X_tfidf.mean(axis=0)).flatten()
_top_global   = sorted(zip(_vocab, _term_weights), key=lambda x: -x[1])[:30]
print(f"\n  Top-30 globally weighted terms (codes):")
for _t, _w in _top_global:
    print(f"    {_t:<40} {_w:.5f}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — SEARCHING FOR THEMES
# LDA topic modeling with N_TOPICS = 5 (target 4-6 themes)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 3: Searching for Themes (LDA) ──")

N_THEMES = 5  # 4-6 meaningful themes
_lda = LatentDirichletAllocation(
    n_components=N_THEMES,
    random_state=42,
    learning_method="online",
    max_iter=20,
    doc_topic_prior=0.1,    # sparse: each user has few themes
    topic_word_prior=0.01   # sparse: each theme has few core terms
)
_doc_topic_matrix = _lda.fit_transform(_X_tfidf)   # shape: (users, N_THEMES)
_topic_term_matrix = _lda.components_               # shape: (N_THEMES, vocab)

print(f"  LDA perplexity: {_lda.perplexity(_X_tfidf):.1f}")

# Top terms per raw topic
print(f"\n  Raw LDA topics (top-15 terms each):")
for _t in range(N_THEMES):
    _top_idx  = _topic_term_matrix[_t].argsort()[::-1][:15]
    _top_terms = [_vocab[i] for i in _top_idx]
    print(f"  Topic {_t}: {_top_terms}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — REVIEWING THEMES
# Analyse topic-success correlations and validate coherence
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 4: Reviewing Themes ──")

_dominant_theme = _doc_topic_matrix.argmax(axis=1)
_user_docs["dominant_theme"] = _dominant_theme
_user_docs["theme_confidence"] = _doc_topic_matrix.max(axis=1)

_theme_stats = []
for _t in range(N_THEMES):
    _mask    = _dominant_theme == _t
    _n       = _mask.sum()
    _succ    = _user_docs.loc[_mask, "long_term_success"].mean() * 100
    _conf    = _user_docs.loc[_mask, "theme_confidence"].mean()
    _top_idx = _topic_term_matrix[_t].argsort()[::-1][:10]
    _terms   = [_vocab[i] for i in _top_idx]
    _theme_stats.append({
        "topic_id":     _t,
        "n_users":      _n,
        "success_rate": _succ,
        "avg_confidence": _conf,
        "top_terms":    _terms
    })

print(f"\n  Theme validation (success rates & coherence):")
for _ts in _theme_stats:
    print(f"  Topic {_ts['topic_id']}: {_ts['n_users']:>5} users | "
          f"success={_ts['success_rate']:.1f}% | conf={_ts['avg_confidence']:.2f} | "
          f"terms={_ts['top_terms'][:5]}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — DEFINING AND NAMING THEMES
# Assign human-readable names based on dominant term semantics
# ═══════════════════════════════════════════════════════════════════════════
print("\n── STEP 5: Defining & Naming Themes ──")

def _name_theme(top_terms, success_rate, n_users):
    """
    Infer a meaningful theme name from the top LDA terms.
    Uses keyword matching on Zerve-specific semantics.
    """
    _t = " ".join(top_terms)

    # Priority hierarchy: most specific first
    if any(w in _t for w in ["agent", "worker", "coder", "chat"]):
        if any(w in _t for w in ["tool", "block", "refactor", "canvas"]):
            return "AI-Assisted Canvas Building"
        return "AI Agent Engagement"
    if any(w in _t for w in ["credit", "credits used", "addon", "payment"]):
        return "Credit Consumption & Billing"
    if any(w in _t for w in ["block", "run block", "canvas", "variable"]):
        if any(w in _t for w in ["refactor", "engineer", "build"]):
            return "Active Development & Iteration"
        return "Canvas Exploration & Block Usage"
    if any(w in _t for w in ["sign", "sign in", "onboard", "fullscreen"]):
        return "Onboarding & Session Start"
    if any(w in _t for w in ["preview", "summary", "variable", "inspect"]):
        return "Data Inspection & Analysis"
    if any(w in _t for w in ["finish", "ticket", "complete", "deploy"]):
        return "Task Completion & Delivery"

    # Fallback: use top 2 terms
    return f"Mixed: {top_terms[0].title()} + {top_terms[1].title()}"

_theme_names = {}
for _ts in _theme_stats:
    _name = _name_theme(_ts["top_terms"], _ts["success_rate"], _ts["n_users"])
    _theme_names[_ts["topic_id"]] = _name
    _ts["theme_name"] = _name

print(f"\n  Theme name assignments:")
for _ts in _theme_stats:
    _succ_icon = "⭐" if _ts["success_rate"] > 3.0 else ("🔷" if _ts["success_rate"] > 1.5 else "○")
    print(f"  {_succ_icon} Topic {_ts['topic_id']} → \"{_ts['theme_name']}\"")
    print(f"          Users: {_ts['n_users']:,} | Success: {_ts['success_rate']:.1f}% | "
          f"Top terms: {_ts['top_terms'][:6]}")

_user_docs["theme_name"] = _user_docs["dominant_theme"].map(_theme_names)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — REPORT & EVIDENCE
# Print named themes with supporting evidence
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Thematic Analysis Report — Named Themes with Evidence")
print("=" * 70)

for _ts in sorted(_theme_stats, key=lambda x: -x["n_users"]):
    _tid  = _ts["topic_id"]
    _name = _ts["theme_name"]
    _n    = _ts["n_users"]
    _sr   = _ts["success_rate"]
    _conf = _ts["avg_confidence"]
    _tms  = _ts["top_terms"]

    # Example events: get most common raw events for users in this theme
    _theme_users  = _user_docs.loc[_user_docs["dominant_theme"] == _tid, "user_key"]
    _theme_events = df[df["person_id"].isin(_theme_users) |
                       df["distinct_id"].isin(_theme_users)]["event"].value_counts().head(5)

    print(f"\n┌─ Theme {_tid + 1}: \"{_name}\"")
    print(f"│  Users:          {_n:,} ({_n/len(_user_docs)*100:.1f}% of corpus)")
    print(f"│  Success Rate:   {_sr:.1f}%  (overall avg: {_user_docs['long_term_success'].mean()*100:.1f}%)")
    print(f"│  Avg Confidence: {_conf:.3f}")
    print(f"│  Top LDA Terms:  {', '.join(_tms)}")
    print(f"│  Example Events: {list(_theme_events.index)}")
    print(f"└─────────────────────────────────────────────────────────────")

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 1 — Word Cloud Grid (one per theme)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Generating Word Cloud Visualizations ──")

_CLOUD_CMAP_COLORS = [
    ["#A1C9F4", "#D0BBFF", "#fbfbff"],  # blue-lavender
    ["#FFB482", "#ffd400", "#fbfbff"],  # orange-yellow
    ["#8DE5A1", "#17b26a", "#fbfbff"],  # green
    ["#FF9F9B", "#f04438", "#fbfbff"],  # coral-red
    ["#D0BBFF", "#A1C9F4", "#fbfbff"],  # lavender-blue
]

fig_wordclouds, _axes_wc = plt.subplots(1, N_THEMES, figsize=(22, 5))
fig_wordclouds.patch.set_facecolor(_BG)

for _t in range(N_THEMES):
    _ax = _axes_wc[_t]
    _ax.set_facecolor(_BG)

    # Build word frequency dict from topic-term weights
    _top_n  = 40
    _top_idx   = _topic_term_matrix[_t].argsort()[::-1][:_top_n]
    _word_freq = {_vocab[i]: float(_topic_term_matrix[_t, i]) for i in _top_idx}

    # Word cloud with theme palette
    _cmap = LinearSegmentedColormap.from_list("theme", _CLOUD_CMAP_COLORS[_t % len(_CLOUD_CMAP_COLORS)])
    _wc   = WordCloud(
        width=600, height=300,
        background_color="#1D1D20",
        colormap=_cmap,
        max_words=35,
        prefer_horizontal=0.8,
        min_font_size=9,
        max_font_size=60,
        collocations=False
    ).generate_from_frequencies(_word_freq)

    _ax.imshow(_wc, interpolation="bilinear")
    _ax.axis("off")

    _ts       = _theme_stats[_t]
    _succ_clr = _SUCCESS if _ts["success_rate"] > 3.0 else (_ORANGE if _ts["success_rate"] > 1.5 else _MUTED)
    _ax.set_title(
        f'Theme {_t+1}\n"{_theme_names[_t]}"\n{_ts["n_users"]:,} users · {_ts["success_rate"]:.1f}% success',
        fontsize=9.5, color=_TEXT, pad=8, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=_succ_clr + "33", edgecolor=_succ_clr, lw=1)
    )

fig_wordclouds.suptitle(
    "Thematic Analysis — LDA Topic Word Clouds (TF-IDF weighted terms)",
    fontsize=15, fontweight="bold", color=_TEXT, y=1.04
)
plt.tight_layout()
plt.savefig("thematic_wordclouds.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 2 — Topic Distribution (users per theme + success rate)
# ═══════════════════════════════════════════════════════════════════════════
fig_topic_dist, (_ax_users, _ax_succ) = plt.subplots(1, 2, figsize=(16, 6))
fig_topic_dist.patch.set_facecolor(_BG)

_ordered = sorted(_theme_stats, key=lambda x: -x["n_users"])
_names_ordered = [f'T{s["topic_id"]+1}: {s["theme_name"]}' for s in _ordered]
_n_users_ordered = [s["n_users"] for s in _ordered]
_succ_ordered    = [s["success_rate"] for s in _ordered]
_colors_ordered  = [_THEME_COLORS[s["topic_id"] % len(_THEME_COLORS)] for s in _ordered]

# — Left: user distribution bars ——
_ax_users.set_facecolor(_BG)
_bars_u = _ax_users.barh(
    _names_ordered[::-1], _n_users_ordered[::-1],
    color=_colors_ordered[::-1], edgecolor=_BG, height=0.6
)
for _b, _n in zip(_bars_u, _n_users_ordered[::-1]):
    _ax_users.text(
        _n + max(_n_users_ordered) * 0.01,
        _b.get_y() + _b.get_height() / 2,
        f"{_n:,} users ({_n/len(_user_docs)*100:.1f}%)",
        va="center", ha="left", color=_TEXT, fontsize=9.5
    )
_ax_users.set_xlabel("Number of Users", fontsize=11)
_ax_users.set_title("Theme Distribution — User Volume", fontsize=13, fontweight="bold", pad=12)
_ax_users.set_xlim(0, max(_n_users_ordered) * 1.4)
_ax_users.spines[["top", "right"]].set_visible(False)
_ax_users.tick_params(labelsize=9.5)

# — Right: success rate bars ——
_ax_succ.set_facecolor(_BG)
_overall_avg = _user_docs["long_term_success"].mean() * 100
_names_succ  = [f'T{s["topic_id"]+1}: {s["theme_name"]}' for s in _ordered]
_succ_colors = [
    _SUCCESS if s["success_rate"] > _overall_avg * 1.5 else
    (_ORANGE  if s["success_rate"] > _overall_avg else _CORAL)
    for s in _ordered
]
_bars_s = _ax_succ.bar(
    range(len(_ordered)), _succ_ordered,
    color=_succ_colors, edgecolor=_BG, width=0.55
)
# Overall average line
_ax_succ.axhline(_overall_avg, color=_YELLOW, linestyle="--", lw=1.5, label=f"Overall avg: {_overall_avg:.1f}%")
for _b, _v in zip(_bars_s, _succ_ordered):
    _ax_succ.text(
        _b.get_x() + _b.get_width() / 2,
        _v + 0.05,
        f"{_v:.1f}%",
        ha="center", va="bottom", color=_TEXT, fontsize=10, fontweight="bold"
    )
_ax_succ.set_xticks(range(len(_ordered)))
_ax_succ.set_xticklabels(
    [f'T{s["topic_id"]+1}' for s in _ordered],
    fontsize=11, rotation=0
)
_ax_succ.set_ylabel("Success Rate (%)", fontsize=11)
_ax_succ.set_title("Theme × Success Rate", fontsize=13, fontweight="bold", pad=12)
_ax_succ.set_ylim(0, max(_succ_ordered) * 1.4 + 0.5)
_ax_succ.spines[["top", "right"]].set_visible(False)
_ax_succ.legend(facecolor=_BG, edgecolor=_MUTED, labelcolor=_TEXT, fontsize=10)

# Add theme name legend at bottom
_patches = [mpatches.Patch(color=_THEME_COLORS[s["topic_id"] % len(_THEME_COLORS)],
                            label=f'T{s["topic_id"]+1}: {s["theme_name"]}')
            for s in _ordered]
fig_topic_dist.legend(
    handles=_patches, loc="lower center", ncol=min(3, len(_ordered)),
    fontsize=8.5, facecolor=_BG, edgecolor=_MUTED, labelcolor=_TEXT,
    bbox_to_anchor=(0.5, -0.06)
)

fig_topic_dist.suptitle(
    "Thematic Analysis — Topic Distribution & Success Correlation",
    fontsize=15, fontweight="bold", color=_TEXT, y=1.03
)
plt.tight_layout()
plt.savefig("thematic_topic_distribution.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 3 — Theme → Success mapping heatmap (topic weight × outcome)
# ═══════════════════════════════════════════════════════════════════════════
fig_heatmap, _ax_heat = plt.subplots(figsize=(14, 5))
fig_heatmap.patch.set_facecolor(_BG)
_ax_heat.set_facecolor(_BG)

# Average topic probability per success group
_success_groups   = [0, 1]
_success_labels   = ["Not Successful", "Long-Term Success"]
_theme_by_success = np.zeros((2, N_THEMES))
for _gi, _g in enumerate(_success_groups):
    _mask = _user_docs["long_term_success"] == _g
    _theme_by_success[_gi] = _doc_topic_matrix[_mask].mean(axis=0)

# Normalise each group row so values sum to 1
_theme_by_success = _theme_by_success / _theme_by_success.sum(axis=1, keepdims=True)

_theme_labels = [f'T{_t+1}: {_theme_names[_t]}' for _t in range(N_THEMES)]

_heatmap_cmap = LinearSegmentedColormap.from_list("zerve_heat", ["#1D1D20", "#A1C9F4", "#ffd400"])
_im = _ax_heat.imshow(_theme_by_success, cmap=_heatmap_cmap, aspect="auto", vmin=0)

# Annotate cells
for _gi in range(2):
    for _tj in range(N_THEMES):
        _val = _theme_by_success[_gi, _tj]
        _ax_heat.text(_tj, _gi, f"{_val:.3f}", ha="center", va="center",
                      color=_TEXT, fontsize=11, fontweight="bold")

_ax_heat.set_xticks(range(N_THEMES))
_ax_heat.set_xticklabels(_theme_labels, rotation=15, ha="right", fontsize=9.5)
_ax_heat.set_yticks([0, 1])
_ax_heat.set_yticklabels(_success_labels, fontsize=11)
_ax_heat.set_title(
    "Theme Probability Distribution by Success Outcome\n(normalised avg topic weight per group)",
    fontsize=13, fontweight="bold", color=_TEXT, pad=12
)

_cbar = plt.colorbar(_im, ax=_ax_heat, fraction=0.04, pad=0.02)
_cbar.ax.yaxis.set_tick_params(color=_TEXT)
_cbar.ax.set_ylabel("Relative Theme Weight", color=_TEXT, fontsize=9)
plt.setp(_cbar.ax.yaxis.get_ticklabels(), color=_TEXT)

plt.tight_layout()
plt.savefig("thematic_success_heatmap.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# STORE RESULTS
# ═══════════════════════════════════════════════════════════════════════════
thematic_analysis_results = {
    "n_themes": N_THEMES,
    "theme_names": _theme_names,                      # {topic_id: name}
    "theme_stats": _theme_stats,                      # list of dicts per theme
    "theme_by_success": _theme_by_success,            # (2, N_THEMES) heatmap data
    "doc_topic_matrix": _doc_topic_matrix,            # (n_users, N_THEMES) LDA weights
    "tfidf_vocab": _vocab,                            # feature names array
    "lda_model": _lda,                                # fitted LDA model
    "user_themes": _user_docs[["user_key", "dominant_theme", "theme_name",
                                "theme_confidence", "long_term_success"]],
    "top_global_terms": _top_global,                  # global TF-IDF codes
}

print("\n" + "=" * 70)
print("✅ THEMATIC ANALYSIS COMPLETE")
print("=" * 70)
print(f"  6-step process completed: Braun & Clarke (2006)")
print(f"  Themes identified       : {N_THEMES}")
print(f"  Users analysed          : {len(_user_docs):,}")
print(f"  Results stored in       : thematic_analysis_results")
print(f"\n  THEME SUMMARY:")
_overall_sr = _user_docs['long_term_success'].mean() * 100
for _ts in sorted(_theme_stats, key=lambda x: -x['success_rate']):
    _diff = _ts['success_rate'] - _overall_sr
    _arrow = "▲" if _diff > 0 else "▼"
    print(f"    {_arrow} Theme {_ts['topic_id']+1}: \"{_ts['theme_name']}\"")
    print(f"       {_ts['n_users']:,} users | {_ts['success_rate']:.1f}% success ({_diff:+.1f}pp vs avg) | top: {_ts['top_terms'][:4]}")
