
# ── EFA & PCA on Behavioral Feature Matrix ─────────────────────────────────
# Full EFA pipeline: KMO/Bartlett, parallel analysis scree, varimax rotation,
# loadings/communalities/variance, factor labeling, PCA comparison, heatmap,
# per-user factor scores, factor_analysis_results summary object
#
# Uses: factor_analyzer, sklearn, scipy, matplotlib, seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats

# ── Try factor_analyzer, fall back to custom implementation ─────────────────
import importlib
_fa_available = importlib.util.find_spec("factor_analyzer") is not None
if _fa_available:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    print("factor_analyzer: available ✅")
else:
    print("factor_analyzer: not installed — using sklearn/scipy EFA implementation")

# ── Zerve color palette ──────────────────────────────────────────────────────
_BG      = "#1D1D20"
_TEXT    = "#fbfbff"
_MUTED   = "#909094"
_BLUE    = "#A1C9F4"
_ORANGE  = "#FFB482"
_GREEN   = "#8DE5A1"
_CORAL   = "#FF9F9B"
_LAVEND  = "#D0BBFF"
_YELLOW  = "#ffd400"
_PINK    = "#F7B6D2"

plt.rcParams.update({
    "figure.facecolor": _BG, "axes.facecolor": _BG,
    "text.color": _TEXT, "axes.labelcolor": _TEXT,
    "xtick.color": _TEXT, "ytick.color": _TEXT,
    "axes.edgecolor": _MUTED, "grid.color": "#333338",
    "font.family": "sans-serif", "axes.titlecolor": _TEXT,
})

# ── 1. Feature matrix ────────────────────────────────────────────────────────
_feat_cols = [
    "feat_total_early_events",
    "feat_unique_event_types",
    "feat_unique_tools_used",
    "feat_unique_days_early",
    "feat_tool_event_rate",
    "feat_early_credit_total",
    "feat_has_early_credit",
    "feat_session_count",
    "feat_avg_events_per_sess",
    "feat_median_sess_dur_min",
    "feat_max_events_in_sess",
    "feat_tool_session_count",
    "feat_time_to_2nd_sess_min",
]

_nice = [
    "Total Events", "Unique Event Types", "Unique Tools Used",
    "Active Days", "Tool Event Rate", "Credit Consumed",
    "Has Early Credit", "Session Count", "Avg Events/Session",
    "Median Sess Duration", "Max Events in Session",
    "Tool Sessions", "Time to 2nd Session",
]

_X_raw = user_feature_matrix[_feat_cols].fillna(0).values.astype(float)
_user_keys = user_feature_matrix["user_key"].values
n_users, n_feats = _X_raw.shape

# Standardize
_scaler = StandardScaler()
_X = _scaler.fit_transform(_X_raw)

print("=" * 65)
print("  EXPLORATORY FACTOR ANALYSIS — BEHAVIORAL FEATURES")
print("=" * 65)
print(f"\n  Feature matrix : {n_users:,} users × {n_feats} features")

# ── 2. KMO & Bartlett ────────────────────────────────────────────────────────
_cor_mat = np.corrcoef(_X.T)

if _fa_available:
    _kmo_all, _kmo_model = calculate_kmo(pd.DataFrame(_X, columns=_feat_cols))
    _bart_chi2, _bart_p   = calculate_bartlett_sphericity(pd.DataFrame(_X, columns=_feat_cols))
else:
    # Manual KMO (Kaiser-Meyer-Olkin)
    def _compute_kmo(X):
        """Manual KMO computation via partial correlations."""
        R = np.corrcoef(X.T)
        R_inv = np.linalg.pinv(R)
        n = R.shape[0]
        kmo_num = kmo_den = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_ij  = R[i, j]
                    p_ij  = -R_inv[i, j] / np.sqrt(R_inv[i, i] * R_inv[j, j])
                    kmo_num += r_ij ** 2
                    kmo_den += r_ij ** 2 + p_ij ** 2
        return kmo_num / kmo_den if kmo_den > 0 else 0.0

    _kmo_model = _compute_kmo(_X)
    # Bartlett: chi2 = -(n-1-(2p+5)/6) * ln(|R|)
    _n, _p = _X.shape
    _det = np.linalg.det(_cor_mat)
    _det = max(_det, 1e-300)
    _bart_chi2 = -(_n - 1 - (2 * _p + 5) / 6) * np.log(_det)
    _bart_df   = _p * (_p - 1) / 2
    _bart_p    = 1 - stats.chi2.cdf(_bart_chi2, _bart_df)

_kmo_label = "Meritorious" if _kmo_model >= 0.8 else ("Acceptable" if _kmo_model >= 0.6 else "Poor")
print(f"\n  KMO Overall MSA : {_kmo_model:.4f}  ({_kmo_label})")
print(f"  Bartlett χ²     : {_bart_chi2:.1f}  p={_bart_p:.2e}  → factor analysis {'appropriate ✅' if _bart_p < 0.001 else 'marginal ⚠️'}")

# ── 3. Parallel Analysis ─────────────────────────────────────────────────────
print("\nRunning parallel analysis (200 random simulations)...")
_n_iter   = 200
_n_rand   = min(n_users, 5000)
_rand_eigenvals = []
for _ in range(_n_iter):
    _rand_data = np.random.normal(size=(_n_rand, n_feats))
    _rand_data = StandardScaler().fit_transform(_rand_data)
    _eig = np.linalg.eigvalsh(np.corrcoef(_rand_data.T))[::-1]
    _rand_eigenvals.append(_eig)

_rand_eig_95 = np.percentile(_rand_eigenvals, 95, axis=0)  # 95th percentile threshold

# Actual eigenvalues from PCA
_actual_eigs = np.linalg.eigvalsh(_cor_mat)[::-1]
_actual_eigs = np.maximum(_actual_eigs, 0)

# Number of factors where actual > simulated 95th percentile
_n_factors = max(np.sum(_actual_eigs > _rand_eig_95), 3)
_n_components = max(np.sum(_actual_eigs > 1.0), 3)   # Kaiser criterion for PCA

print(f"  Parallel analysis → EFA factors: {_n_factors}  |  PCA components: {_n_components}")

# ── 4. Scree Plot ────────────────────────────────────────────────────────────
# NOTE: using _ax_scree to avoid collision with ax_sc from train_rf_classifier
fig_scree, _ax_scree = plt.subplots(figsize=(11, 6))
fig_scree.patch.set_facecolor(_BG)
_ax_scree.set_facecolor(_BG)

_x_vals = np.arange(1, n_feats + 1)
_ax_scree.plot(_x_vals, _actual_eigs, "o-", color=_BLUE, lw=2.5, ms=7, label="Actual Eigenvalues", zorder=5)
_ax_scree.plot(_x_vals, _rand_eig_95, "--", color=_CORAL, lw=1.8, alpha=0.9, label="Simulated 95th pct (parallel)")
_ax_scree.axhline(1.0, color=_YELLOW, lw=1.4, ls=":", alpha=0.7, label="Kaiser criterion (λ=1)")
_ax_scree.axvline(_n_factors + 0.5, color=_GREEN, lw=1.5, ls="--", alpha=0.8, label=f"Suggested # factors ({_n_factors})")

_ax_scree.fill_between(_x_vals[:_n_factors], _actual_eigs[:_n_factors],
                   alpha=0.12, color=_BLUE)

_ax_scree.set_xlabel("Factor / Component", fontsize=12)
_ax_scree.set_ylabel("Eigenvalue", fontsize=12)
_ax_scree.set_title("Parallel Analysis Scree Plot — Behavioral Features", fontsize=14, fontweight="bold", pad=12)
_ax_scree.legend(facecolor=_BG, edgecolor=_MUTED, labelcolor=_TEXT, fontsize=10)
_ax_scree.spines[["top", "right"]].set_visible(False)
_ax_scree.set_xticks(_x_vals)
_ax_scree.set_xticklabels(_nice, rotation=45, ha="right", fontsize=9)
_ax_scree.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig("efa_parallel_scree.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

# ── 5. EFA with varimax rotation ─────────────────────────────────────────────
print(f"\nFitting EFA with {_n_factors} factors (varimax rotation)...")

if _fa_available:
    _fa = FactorAnalyzer(n_factors=_n_factors, rotation="varimax", method="ml")
    _fa.fit(pd.DataFrame(_X, columns=_feat_cols))
    _loadings_arr = _fa.loadings_  # shape: (n_feats, n_factors)
    _communalities_arr = _fa.get_communalities()
    _uniqueness_arr    = 1 - _communalities_arr
    _var_explained     = _fa.get_factor_variance()
    # factor scores
    _scores_arr = _fa.transform(pd.DataFrame(_X, columns=_feat_cols))

else:
    # ── Custom EFA via PCA + varimax rotation ──────────────────────────────
    # Initial loadings from PCA
    _pca_init = PCA(n_components=_n_factors, random_state=42)
    _pca_init.fit(_X)
    _init_loadings = _pca_init.components_.T * np.sqrt(_pca_init.explained_variance_)

    # Varimax rotation (raw criterion)
    def _varimax(Phi, gamma=1.0, q=20, tol=1e-6):
        p, k = Phi.shape
        R = np.eye(k)
        d_old = 0
        for _ in range(q):
            Lambda = Phi @ R
            u, s, vt = np.linalg.svd(
                Phi.T @ (Lambda ** 3 - (gamma / p) * Lambda @ np.diag(np.sum(Lambda ** 2, axis=0)))
            )
            R = u @ vt
            d = np.sum(s)
            if abs(d - d_old) < tol:
                break
            d_old = d
        return Phi @ R, R

    _loadings_arr, _rot_mat = _varimax(_init_loadings)

    _communalities_arr = np.sum(_loadings_arr ** 2, axis=1)
    _uniqueness_arr    = 1 - _communalities_arr
    _ss_loadings       = np.sum(_loadings_arr ** 2, axis=0)
    _prop_var          = _ss_loadings / n_feats
    _cum_var           = np.cumsum(_prop_var)

    # Factor scores via regression
    _R_inv = np.linalg.pinv(_cor_mat)
    _scores_arr = _X @ _R_inv @ _loadings_arr

# ── 6. Loadings matrix ───────────────────────────────────────────────────────
_factor_names_raw = [f"ML{i+1}" for i in range(_n_factors)]
_loadings_df = pd.DataFrame(_loadings_arr, index=_nice, columns=_factor_names_raw)

print("\n" + "-" * 65)
print("FACTOR LOADINGS MATRIX (varimax-rotated)")
print("-" * 65)
print(_loadings_df.round(3).to_string())

# ── 7. Communalities ─────────────────────────────────────────────────────────
_communalities_df = pd.DataFrame({
    "Feature":     _nice,
    "Communality": np.round(_communalities_arr, 4),
    "Uniqueness":  np.round(_uniqueness_arr, 4),
})

print("\n" + "-" * 65)
print("COMMUNALITIES (h²) & UNIQUENESS (u²)")
print("-" * 65)
print(_communalities_df.to_string(index=False))

# ── 8. Variance explained ────────────────────────────────────────────────────
if _fa_available:
    _ss_load, _prop_var, _cum_var = _var_explained
else:
    _ss_load = np.sum(_loadings_arr ** 2, axis=0)
    _prop_var = _ss_load / n_feats
    _cum_var  = np.cumsum(_prop_var)

_var_df = pd.DataFrame({
    "Factor":          _factor_names_raw,
    "SS Loadings":     np.round(_ss_load, 4),
    "Proportion Var":  np.round(_prop_var, 4),
    "Cumulative Var":  np.round(_cum_var, 4),
})

print("\n" + "-" * 65)
print("VARIANCE EXPLAINED PER FACTOR")
print("-" * 65)
print(_var_df.to_string(index=False))

# ── 9. Label latent factors ──────────────────────────────────────────────────
_label_map = {
    "Total Events":           "Engagement",
    "Unique Event Types":     "Engagement",
    "Session Count":          "Engagement",
    "Active Days":            "Engagement",
    "Time to 2nd Session":    "Engagement",
    "Avg Events/Session":     "Intensity",
    "Max Events in Session":  "Intensity",
    "Median Sess Duration":   "Intensity",
    "Unique Tools Used":      "Adoption",
    "Tool Event Rate":        "Adoption",
    "Tool Sessions":          "Adoption",
    "Credit Consumed":        "Adoption",
    "Has Early Credit":       "Adoption",
}

_named_factors = []
for f_idx in range(_n_factors):
    _top3 = list(np.array(_nice)[np.argsort(np.abs(_loadings_arr[:, f_idx]))[::-1][:3]])
    _votes = [_label_map[n] for n in _top3]
    _winner = max(set(_votes), key=_votes.count)
    _suffix = _named_factors.count(_winner)
    _named_factors.append(_winner if _suffix == 0 else f"{_winner}_{_suffix + 1}")

# Rename columns
_loadings_df.columns = _named_factors

print("\n" + "-" * 65)
print("LATENT FACTOR LABELS (domain interpretation)")
print("-" * 65)
for i, fname in enumerate(_named_factors):
    _top3 = list(np.array(_nice)[np.argsort(np.abs(_loadings_arr[:, i]))[::-1][:3]])
    print(f"  Factor {i+1} → {fname:<22}  (top features: {', '.join(_top3)})")

# ── 10. PCA comparison ───────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("PCA COMPARISON (sklearn, varimax-like)")
print("-" * 65)
_pca_comp = PCA(n_components=_n_components, random_state=42)
_pca_comp.fit(_X)
_pca_scores = _pca_comp.transform(_X)

_pca_var_df = pd.DataFrame({
    "Component":     [f"PC{i+1}" for i in range(_n_components)],
    "Proportion Var": np.round(_pca_comp.explained_variance_ratio_, 4),
    "Cumulative Var": np.round(np.cumsum(_pca_comp.explained_variance_ratio_), 4),
})
print("\nPCA Variance Explained:")
print(_pca_var_df.to_string(index=False))

_efa_cum = _cum_var[-1]
_pca_cum = np.cumsum(_pca_comp.explained_variance_ratio_)[-1]
print(f"\n  EFA ({_n_factors} factors, varimax) cumulative variance : {_efa_cum*100:.1f}%")
print(f"  PCA ({_n_components} components)     cumulative variance : {_pca_cum*100:.1f}%")

# ── 11. Factor scores per user ───────────────────────────────────────────────
factor_scores_df = pd.DataFrame(_scores_arr, columns=_named_factors)
factor_scores_df["user_key"] = _user_keys

print(f"\nFactor scores (first 5 users):")
print(factor_scores_df[["user_key"] + _named_factors].head(5).to_string(index=False))

# ── 12. Factor loadings heatmap ──────────────────────────────────────────────
print("\nGenerating factor loadings heatmap...")

_hm_data = _loadings_df.values  # (n_feats, n_factors)
_row_labels = _nice[::-1]       # reverse for top-to-bottom display
_col_labels = _named_factors
_hm_display = _hm_data[::-1, :]

fig_efa_heatmap, ax_hm = plt.subplots(figsize=(max(7, _n_factors * 2 + 2), 8))
fig_efa_heatmap.patch.set_facecolor(_BG)
ax_hm.set_facecolor(_BG)

_cmap = mcolors.LinearSegmentedColormap.from_list(
    "zerve_div", [_CORAL, _BG, _BLUE], N=256
)
_im = ax_hm.imshow(_hm_display, aspect="auto", cmap=_cmap, vmin=-1, vmax=1)

# Annotations
for r in range(n_feats):
    for c in range(_n_factors):
        val = _hm_display[r, c]
        if abs(val) >= 0.3:
            _txt_col = _BG if abs(val) >= 0.55 else _TEXT
            ax_hm.text(c, r, f"{val:.2f}", ha="center", va="center",
                       fontsize=9, fontweight="bold", color=_txt_col)

ax_hm.set_xticks(range(_n_factors))
ax_hm.set_xticklabels(_col_labels, fontsize=11, fontweight="bold", color=_TEXT)
ax_hm.set_yticks(range(n_feats))
ax_hm.set_yticklabels(_row_labels, fontsize=10, color=_TEXT)
ax_hm.tick_params(length=0)
ax_hm.spines[:].set_visible(False)

_cb = fig_efa_heatmap.colorbar(_im, ax=ax_hm, fraction=0.025, pad=0.03)
_cb.ax.yaxis.set_tick_params(color=_TEXT, labelsize=9)
_cb.outline.set_visible(False)
_cb.set_label("Factor Loading", color=_TEXT, fontsize=10)
plt.setp(plt.getp(_cb.ax.axes, "yticklabels"), color=_TEXT)

ax_hm.set_title(
    "EFA Factor Loadings Heatmap (Varimax-Rotated)",
    fontsize=14, fontweight="bold", color=_TEXT, pad=14
)
ax_hm.set_xlabel("Latent Factor", fontsize=11, color=_MUTED, labelpad=8)

# Add subtitle annotation
fig_efa_heatmap.text(0.5, 0.94,
    f"{n_feats} behavioral features × {_n_factors} latent factors  |  n={n_users:,} users",
    ha="center", va="top", fontsize=10, color=_MUTED
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("efa_loadings_heatmap.png", dpi=150, bbox_inches="tight", facecolor=_BG)
plt.show()

# ── 13. Summary results object ───────────────────────────────────────────────
factor_analysis_results = {
    "n_users":            n_users,
    "n_features":         n_feats,
    "n_factors_efa":      _n_factors,
    "n_factors_pca":      _n_components,
    "kmo_msa":            round(float(_kmo_model), 4),
    "bartlett_chi2":      round(float(_bart_chi2), 2),
    "bartlett_p":         float(_bart_p),
    "factor_labels":      _named_factors,
    "loadings_matrix":    _loadings_df.round(4),
    "communalities":      _communalities_df,
    "variance_explained": _var_df,
    "pca_variance":       _pca_var_df,
    "factor_scores":      factor_scores_df,
}

print("\n" + "=" * 65)
print("EFA COMPLETE — factor_analysis_results saved")
print("=" * 65)
print(f"   KMO MSA           : {factor_analysis_results['kmo_msa']:.4f}")
print(f"   Factors extracted : {_n_factors}")
print(f"   Factor labels     : {' | '.join(_named_factors)}")
print(f"   Cumul. var (EFA)  : {_efa_cum*100:.1f}%")
print(f"   Cumul. var (PCA)  : {_pca_cum*100:.1f}%")
print(f"   Users with scores : {len(factor_scores_df):,}")
print(f"   Files saved       : efa_parallel_scree.png, efa_loadings_heatmap.png")
