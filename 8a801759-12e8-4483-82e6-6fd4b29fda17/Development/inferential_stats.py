import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")

# ── Data Prep ─────────────────────────────────────────────────────────────
_LABEL  = "long_term_success"
_FEAT_CONTINUOUS = [
    "feat_total_early_events", "feat_unique_event_types", "feat_unique_tools_used",
    "feat_unique_days_early", "feat_tool_event_rate", "feat_early_credit_total",
    "feat_session_count", "feat_avg_events_per_sess", "feat_median_sess_dur_min",
    "feat_max_events_in_sess", "feat_tool_session_count", "feat_time_to_2nd_sess_min",
]
_FEAT_BINARY = ["feat_has_early_credit"]

_df = user_feature_matrix.copy()
_df[_FEAT_CONTINUOUS] = _df[_FEAT_CONTINUOUS].fillna(0)
_df[_FEAT_BINARY]     = _df[_FEAT_BINARY].fillna(0)

_success_mask = _df[_LABEL] == 1
_grp_yes = _df[_success_mask]
_grp_no  = _df[~_success_mask]

print("=" * 72)
print("   INFERENTIAL STATISTICAL ANALYSIS — USER SUCCESS COMPARISON")
print("=" * 72)
print(f"   Total users : {len(_df):,}   |   Success: {_success_mask.sum():,}   |   No-Success: {(~_success_mask).sum():,}")
print()


# ─────────────────────────────────────────────────────────────────────────
# HELPER: Cohen's d effect size
# ─────────────────────────────────────────────────────────────────────────
def _cohens_d(a, b):
    _na, _nb = len(a), len(b)
    _pooled_std = np.sqrt(
        ((_na - 1) * np.var(a, ddof=1) + (_nb - 1) * np.var(b, ddof=1)) /
        (_na + _nb - 2)
    )
    return 0.0 if _pooled_std == 0 else (np.mean(a) - np.mean(b)) / _pooled_std


# ─────────────────────────────────────────────────────────────────────────
# 1. INDEPENDENT SAMPLES T-TESTS (Welch's, continuous features)
# ─────────────────────────────────────────────────────────────────────────
print("─" * 72)
print("1. INDEPENDENT SAMPLES T-TESTS  (Success vs No-Success, Welch's)")
print("─" * 72)
print(f"{'Feature':<32} {'Mean(S)':>8} {'Mean(NS)':>9} {'t-stat':>9} {'p-value':>11} {'Cohen d':>8} {'Effect':>10} {'Sig':>5}")
print("-" * 106)

ttest_records = []
for _col in _FEAT_CONTINUOUS:
    _a = _grp_yes[_col].values
    _b = _grp_no[_col].values
    _t, _p = ttest_ind(_a, _b, equal_var=False)  # Welch's

    _d = _cohens_d(_a, _b)
    _d_mag = "large" if abs(_d) >= 0.8 else ("medium" if abs(_d) >= 0.5 else ("small" if abs(_d) >= 0.2 else "negligible"))

    # 95% CI on the difference of means
    _se  = np.sqrt(np.var(_a, ddof=1)/len(_a) + np.var(_b, ddof=1)/len(_b))
    _dfw = (np.var(_a, ddof=1)/len(_a) + np.var(_b, ddof=1)/len(_b))**2 / \
           ((np.var(_a, ddof=1)/len(_a))**2/(len(_a)-1) + (np.var(_b, ddof=1)/len(_b))**2/(len(_b)-1))
    _tc  = stats.t.ppf(0.975, _dfw)
    _md  = np.mean(_a) - np.mean(_b)
    _ci_lo, _ci_hi = _md - _tc * _se, _md + _tc * _se

    _sig = "✅ *" if _p < 0.05 else "  "
    _lbl = _col.replace("feat_", "").replace("_", " ").title()[:30]

    ttest_records.append({
        "feature": _col, "label": _lbl,
        "mean_success": float(np.mean(_a)), "mean_no_success": float(np.mean(_b)),
        "t_stat": float(_t), "p_value": float(_p), "cohens_d": float(_d),
        "effect_size": _d_mag, "ci_95_lo": float(_ci_lo), "ci_95_hi": float(_ci_hi),
        "significant": bool(_p < 0.05)
    })
    print(f"{_lbl:<32} {np.mean(_a):>8.3f} {np.mean(_b):>9.3f} {_t:>9.3f} {_p:>11.4e} {_d:>8.3f} {_d_mag:>10}  {_sig}")

ttest_df = pd.DataFrame(ttest_records)
print(f"\n   Significant (p<0.05): {ttest_df['significant'].sum()}/{len(ttest_df)} features")
print(f"   95% CIs are on the difference in means (Success − No-Success)")


# ─────────────────────────────────────────────────────────────────────────
# 2. CHI-SQUARE TESTS (binary + categorical features)
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("2. CHI-SQUARE TESTS  (Binary / Categorical Features)")
print("─" * 72)
print(f"{'Feature':<28} {'χ²':>10} {'p-value':>11} {'df':>4} {'Cramér V':>10} {'Min Exp N':>11} {'Sig':>5}")
print("-" * 80)

chitest_records = []
for _col in _FEAT_BINARY:
    _ct = pd.crosstab(_df[_col], _df[_LABEL])
    _chi2, _p_chi, _dof, _exp = chi2_contingency(_ct)
    _n_chi = _ct.sum().sum()
    _phi_c = np.sqrt(_chi2 / (_n_chi * (min(_ct.shape) - 1))) if min(_ct.shape) > 1 else 0.0
    _sig = "✅ *" if _p_chi < 0.05 else "  "
    _lbl = _col.replace("feat_", "").replace("_", " ").title()[:26]
    chitest_records.append({
        "feature": _col, "label": _lbl,
        "chi2": float(_chi2), "p_value": float(_p_chi), "df": int(_dof),
        "cramers_v": float(_phi_c), "min_expected": float(_exp.min()), "significant": bool(_p_chi < 0.05)
    })
    print(f"{_lbl:<28} {_chi2:>10.3f} {_p_chi:>11.4e} {_dof:>4d} {_phi_c:>10.4f} {_exp.min():>11.2f}  {_sig}")

# Categorical: lifespan_bucket (drop NaN rows)
_df_no_nan_bucket = _df.dropna(subset=["lifespan_bucket"])
_ct2 = pd.crosstab(_df_no_nan_bucket["lifespan_bucket"], _df_no_nan_bucket[_LABEL])
_chi2_b, _p_b, _dof_b, _exp_b = chi2_contingency(_ct2)
_phi_b = np.sqrt(_chi2_b / (_ct2.sum().sum() * (min(_ct2.shape) - 1)))
_sig_b = "✅ *" if _p_b < 0.05 else "  "
chitest_records.append({
    "feature": "lifespan_bucket", "label": "Lifespan Bucket",
    "chi2": float(_chi2_b), "p_value": float(_p_b), "df": int(_dof_b),
    "cramers_v": float(_phi_b), "min_expected": float(_exp_b.min()), "significant": bool(_p_b < 0.05)
})
print(f"{'Lifespan Bucket':<28} {_chi2_b:>10.3f} {_p_b:>11.4e} {_dof_b:>4d} {_phi_b:>10.4f} {_exp_b.min():>11.2f}  {_sig_b}")

chitest_df = pd.DataFrame(chitest_records)
print(f"\n   Significant (p<0.05): {chitest_df['significant'].sum()}/{len(chitest_df)} features")
print(f"   Effect sizes — Cramér V: 0.1=small, 0.3=medium, 0.5=large")


# ─────────────────────────────────────────────────────────────────────────
# 3. ONE-WAY ANOVA: Lifespan Cohort Groups on key metrics
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("3. ONE-WAY ANOVA  (Lifespan Cohort × Credit & Session Metrics)")
print("─" * 72)

_ANOVA_TARGETS = [
    "feat_early_credit_total", "feat_session_count",
    "feat_total_early_events", "feat_avg_events_per_sess",
    "feat_tool_session_count", "feat_median_sess_dur_min"
]
# Drop NaN buckets before ANOVA
_df_anova = _df.dropna(subset=["lifespan_bucket"]).copy()
_df_anova["lifespan_bucket"] = _df_anova["lifespan_bucket"].cat.remove_unused_categories()
_buckets = sorted(_df_anova["lifespan_bucket"].unique().tolist(), key=str)
print(f"   Cohort groups : {_buckets}  (k={len(_buckets)},  n={len(_df_anova):,} after dropping NaN)")
print()
print(f"{'Metric':<30} {'F-stat':>10} {'p-value':>11} {'η² (eta²)':>11} {'Sig':>5}")
print("-" * 68)

anova_records = []
for _col in _ANOVA_TARGETS:
    _groups = [_df_anova[_df_anova["lifespan_bucket"] == _b][_col].values for _b in _buckets]
    _groups = [_g for _g in _groups if len(_g) > 1]  # need ≥2 obs per group
    _f, _p_f = f_oneway(*_groups)
    # Eta-squared (SS_between / SS_total)
    _gm  = _df_anova[_col].mean()
    _ssb = sum(len(_g) * (np.mean(_g) - _gm)**2 for _g in _groups)
    _sst = sum(np.sum((_g - _gm)**2) for _g in _groups)
    _eta2 = _ssb / _sst if _sst > 0 else 0.0
    _sig = "✅ *" if _p_f < 0.05 else "  "
    _lbl = _col.replace("feat_", "").replace("_", " ").title()[:28]

    anova_records.append({
        "metric": _col, "label": _lbl,
        "F_stat": float(_f), "p_value": float(_p_f), "eta_squared": float(_eta2),
        "significant": bool(_p_f < 0.05),
        "group_means": {str(_b): float(np.mean(_df_anova[_df_anova["lifespan_bucket"] == _b][_col].values))
                        for _b in _buckets}
    })
    print(f"{_lbl:<30} {_f:>10.3f} {_p_f:>11.4e} {_eta2:>11.4f}  {_sig}")

anova_df = pd.DataFrame(anova_records)
print(f"\n   Significant (p<0.05): {anova_df['significant'].sum()}/{len(anova_df)} metrics")
print(f"   Effect sizes — η²: 0.01=small, 0.06=medium, 0.14=large")

print("\n   Group Mean Values by Cohort Bucket:")
_gm_df = pd.DataFrame({r["label"]: r["group_means"] for r in anova_records}).T
print(_gm_df.to_string(float_format="{:.3f}".format))


# ─────────────────────────────────────────────────────────────────────────
# 4. LOGISTIC REGRESSION — coefficients, p-values (Wald), odds ratios
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("4. LOGISTIC REGRESSION  (Predicting Long-Term Success)")
print("─" * 72)

_ALL_FEATS = _FEAT_CONTINUOUS + _FEAT_BINARY
_X_raw = _df[_ALL_FEATS].fillna(0).values
_y_lr  = _df[_LABEL].values

# Standardise for comparable coefficients
_scaler = StandardScaler()
_X_std  = _scaler.fit_transform(_X_raw)

# Use stronger L2 regularisation (C=0.1) to avoid numerical issues with imbalanced classes
_lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=0.1,
                         class_weight="balanced", random_state=42)
_lr.fit(_X_std, _y_lr)

# ── Wald test p-values via observed Fisher information ───────────────────
_p_hat   = _lr.predict_proba(_X_std)[:, 1]
_X_int   = np.hstack([np.ones((_X_std.shape[0], 1)), _X_std])  # add intercept
_W_diag  = _p_hat * (1 - _p_hat)

# Fisher information / Hessian: X^T W X
_XtW = (_X_int * _W_diag[:, None]).T   # shape (p+1, n)
_H   = _XtW @ _X_int                   # shape (p+1, p+1)
_cov = np.linalg.pinv(_H)
_coef_full = np.concatenate([[_lr.intercept_[0]], _lr.coef_[0]])
_se_full   = np.sqrt(np.clip(np.diag(_cov), 1e-12, None))

_z_vals  = _coef_full / _se_full
_pvals   = 2 * (1 - stats.norm.cdf(np.abs(_z_vals)))

_z95     = 1.96
_ci_lo   = _coef_full - _z95 * _se_full
_ci_hi   = _coef_full + _z95 * _se_full

# McFadden Pseudo-R²
_ll_full = -log_loss(_y_lr, _p_hat, normalize=False)
_p_null  = np.full_like(_p_hat, _y_lr.mean())
_ll_null = -log_loss(_y_lr, _p_null, normalize=False)
_pseudo_r2 = max(0.0, 1 - (_ll_full / _ll_null)) if _ll_null != 0 else 0.0

_feat_names_lr = ["const"] + _ALL_FEATS
print(f"\n   Pseudo R² (McFadden) : {_pseudo_r2:.4f}")
print(f"   Log-Likelihood — full: {_ll_full:.2f}   null: {_ll_null:.2f}")
print(f"   Note: L2 regularised (C=0.1); coefficients on standardised features")
print()
print(f"{'Feature':<32} {'Coef':>8} {'OR':>8} {'OR CI 95%':>20} {'p-value':>11} {'Sig':>5}")
print("-" * 90)

logreg_records = []
for _i, _fname in enumerate(_feat_names_lr):
    _c     = _coef_full[_i]
    _p     = _pvals[_i]
    _od    = np.exp(_c)
    _od_lo = np.exp(_ci_lo[_i])
    _od_hi = np.exp(_ci_hi[_i])
    _sig   = "✅ *" if _p < 0.05 else "  "
    _lbl   = str(_fname).replace("feat_", "").replace("_", " ").title()[:30]

    logreg_records.append({
        "feature": _fname, "label": _lbl,
        "coefficient": float(_c), "p_value": float(_p),
        "odds_ratio": float(_od), "or_ci_lo": float(_od_lo), "or_ci_hi": float(_od_hi),
        "significant": bool(_p < 0.05)
    })
    print(f"{_lbl:<32} {_c:>8.4f} {_od:>8.4f} [{_od_lo:.4f}, {_od_hi:.4f}]  {_p:>11.4e}  {_sig}")

logreg_df = pd.DataFrame(logreg_records)
_n_sig_lr = int(logreg_df[logreg_df["feature"] != "const"]["significant"].sum())
print(f"\n   Significant (p<0.05): {_n_sig_lr}/{len(logreg_df)-1} features")
print(f"   OR > 1 ↑ increases success odds  |  OR < 1 ↓ decreases success odds")


# ─────────────────────────────────────────────────────────────────────────
# STORE RESULTS
# ─────────────────────────────────────────────────────────────────────────
inferential_stats_results = {
    "ttest":   ttest_df,
    "chitest": chitest_df,
    "anova":   anova_df,
    "logreg":  logreg_df,
    "logreg_model_summary": {
        "pseudo_r2_mcfadden": float(_pseudo_r2),
        "log_likelihood_full": float(_ll_full),
        "log_likelihood_null": float(_ll_null),
    }
}

print("\n" + "=" * 72)
print("✅ All four inferential test types completed successfully.")
print(f"   T-tests     : {len(ttest_df)} features  | {ttest_df['significant'].sum()} significant")
print(f"   Chi-square  : {len(chitest_df)} features  | {chitest_df['significant'].sum()} significant")
print(f"   ANOVA       : {len(anova_df)} metrics   | {anova_df['significant'].sum()} significant")
print(f"   Logistic Reg: {len(logreg_df)-1} features  | {_n_sig_lr} significant")
print(f"\n   inferential_stats_results dict stored ✔")
print("=" * 72)
