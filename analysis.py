#!/usr/bin/env python3
"""
analyze_experiments.py

Reads a text file containing lines like:
  1) acc=85.33% | lr=0.00033 wd=0.07 bs=128 dpr=0.2

Parses the values, writes a CSV, produces:
 - correlation matrix
 - single-variable linear regressions (acc ~ lr, acc ~ wd, ...)
 - multiple linear regression (standardized inputs)
 - polynomial (degree=2) regression with interaction terms (regularized with RidgeCV)
 - scatter plots (with regression line)
 - detection of high-correlation variable pairs
 - joint visualizations: 3D scatter + fitted surface and 2D contour/heatmap for acc vs (param1, param2)
 - full text report and JSON indexes for plots
Outputs are written to `output_dir` (default: ./analysis_output).
"""
import re
import sys
from pathlib import Path
import argparse
import json
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold

# Config
MIN_CORR = 0.3      # keep pairs with |corr| >= MIN_CORR
GRID_RES = 60       # mesh resolution for surface/contour plots
LOG_TRANSFORM = True  # allow automatic log10 transform for lr/wd when appropriate

# Optional libs
try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import seaborn as sns
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

try:
    import statsmodels.api as sm
    HAVE_STATSM = True
except Exception:
    HAVE_STATSM = False


# ---------------------------
# Helper: robust file resolution
# ---------------------------
def resolve_input_path(path_str):
    """Try the given path; if not found, try same directory as script; finally use cwd.
    Returns a Path object that exists or raises FileNotFoundError.
    """
    p = Path(path_str)
    if p.exists():
        return p.resolve()
    # try script dir if __file__ exists
    try:
        script_dir = Path(__file__).parent.resolve()
        candidate = script_dir / path_str
        if candidate.exists():
            return candidate.resolve()
    except Exception:
        pass
    # try cwd
    candidate = Path.cwd() / path_str
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not find file '{path_str}' (checked provided path, script dir, and cwd).")


# ---------------------------
# Helper: parse file contents
# ---------------------------
def parse_experiments_text(text):
    pattern = re.compile(
        r'^\s*(\d+)\)\s*acc=([\d.]+)%\s*\|\s*lr=([0-9.eE+-]+)\s+wd=([0-9.eE+-]+)\s+bs=(\d+)\s+dpr=([0-9.eE+-]+)',
        re.MULTILINE
    )
    rows = []
    for m in pattern.finditer(text):
        idx = int(m.group(1))
        acc = float(m.group(2))
        lr = float(m.group(3))
        wd = float(m.group(4))
        bs = int(m.group(5))
        dpr = float(m.group(6))
        rows.append((idx, acc, lr, wd, bs, dpr))

    if not rows:
        raise ValueError("No matching lines found. Check file format.")
    df = pd.DataFrame(rows, columns=['idx', 'acc', 'lr', 'wd', 'bs', 'dpr'])
    df = df.sort_values('idx').reset_index(drop=True)
    return df


# ------------------------------------
# Helper: single-variable regression
# ------------------------------------
def single_var_regression(df, var):
    X = df[[var]].values.reshape(-1, 1)
    y = df['acc'].values
    reg = LinearRegression().fit(X, y)

    slope = float(reg.coef_[0])
    intercept = float(reg.intercept_)
    r2 = float(reg.score(X, y))

    if HAVE_SCIPY:
        r, p = stats.pearsonr(df[var], df['acc'])
    else:
        r = float(np.corrcoef(df[var], df['acc'])[0, 1])
        p = None

    return {
        'model': reg,
        'slope': slope,
        'intercept': intercept,
        'r2': r2,
        'pearson_r': r,
        'p_value': p
    }


# -------------------------
# Helper: plotting function
# -------------------------
def plot_acc_vs(df, var, out_path):
    X = df[[var]].values.reshape(-1, 1)
    y = df['acc'].values
    reg = LinearRegression().fit(X, y)

    xs = np.linspace(df[var].min(), df[var].max(), 200)
    ys = reg.predict(xs.reshape(-1, 1))

    plt.figure(figsize=(6, 4))
    plt.scatter(df[var], df['acc'], alpha=0.8)
    plt.plot(xs, ys, linewidth=2)
    plt.xlabel(var)
    plt.ylabel('acc (%)')
    plt.title(f'acc vs {var}  (slope={reg.coef_[0]:.4g}, R²={reg.score(X,y):.3f})')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


# -------------------------
# Helpers for joint visualizations
# -------------------------
def maybe_log_transform_series(s: pd.Series, name: str):
    """Decide whether to log10-transform this parameter (useful for lr/wd)"""
    if not LOG_TRANSFORM:
        return s, False
    # only transform strictly positive values
    if (s <= 0).any():
        return s, False
    # transform if dynamic range > 100 (two orders of magnitude) or if median < 1e-2
    rng = s.max() / max(s.min(), 1e-12)
    if rng > 100 or s.median() < 1e-2:
        return np.log10(s), True
    return s, False


def fit_plane_and_surface(x, y, z, grid_res=GRID_RES):
    """
    Fit a linear model z ~ [x, y]. Return fitted model and a grid (Xg, Yg, Zg_pred)
    where Xg/Yg are 2D mesh grids and Zg_pred the predicted z values.
    """
    XY = np.column_stack([x, y])
    model = LinearRegression().fit(XY, z)
    # grid
    xlin = np.linspace(x.min(), x.max(), grid_res)
    ylin = np.linspace(y.min(), y.max(), grid_res)
    Xg, Yg = np.meshgrid(xlin, ylin)
    XYg = np.column_stack([Xg.ravel(), Yg.ravel()])
    Zg = model.predict(XYg).reshape(Xg.shape)
    return model, Xg, Yg, Zg


def save_3d_scatter_with_surface(df, v1, v2, out_png_3d, out_png_contour, transform_flags):
    """
    Create:
      - 3D scatter of (v1, v2, acc) and overlay fitted plane surface
      - 2D contour/heatmap of predicted acc on grid
    transform_flags: dict indicating if v1 or v2 was log-transformed (for axis labels)
    """
    x_series, x_log = maybe_log_transform_series(df[v1], v1)
    y_series, y_log = maybe_log_transform_series(df[v2], v2)
    z = df['acc'].values
    x = x_series.values if isinstance(x_series, pd.Series) else np.asarray(x_series)
    y = y_series.values if isinstance(y_series, pd.Series) else np.asarray(y_series)

    # Fit plane and predict grid
    model, Xg, Yg, Zg = fit_plane_and_surface(x, y, z, grid_res=GRID_RES)

    # 3D scatter + surface
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=40, alpha=0.8, label='observed')
    # surface (transparent)
    surf = ax.plot_surface(Xg, Yg, Zg, cmap=cm.viridis, alpha=0.45, linewidth=0, antialiased=True)
    ax.set_xlabel(f"{v1}{' (log10)' if x_log else ''}")
    ax.set_ylabel(f"{v2}{' (log10)' if y_log else ''}")
    ax.set_zlabel('acc (%)')
    ax.set_title(f"Accuracy vs {v1} & {v2} (linear surface fit)")
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    plt.tight_layout()
    fig.savefig(out_png_3d)
    plt.close(fig)

    # 2D contour / heatmap: show predicted Zg
    fig2, ax2 = plt.subplots(figsize=(6.5, 5))
    # contourf with colorbar
    cf = ax2.contourf(Xg, Yg, Zg, levels=40, cmap='viridis')
    # overlay the raw points
    ax2.scatter(x, y, c='k', s=18, alpha=0.6, linewidths=0.2)
    ax2.set_xlabel(f"{v1}{' (log10)' if x_log else ''}")
    ax2.set_ylabel(f"{v2}{' (log10)' if y_log else ''}")
    ax2.set_title(f"Predicted accuracy (linear fit) on {v1} vs {v2}")
    fig2.colorbar(cf, ax=ax2)
    plt.tight_layout()
    fig2.savefig(out_png_contour)
    plt.close(fig2)

    # return some metadata
    return {
        'model_coef': model.coef_.tolist(),
        'model_intercept': float(model.intercept_),
        'v1_log': x_log,
        'v2_log': y_log
    }


# -------------------------
# Main analysis function
# -------------------------
def analyze_file(file_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = resolve_input_path(file_path)
    text = file_path.read_text()
    df = parse_experiments_text(text)

    # Save parsed CSV
    csv_path = output_dir / 'experiments_parsed.csv'
    df.to_csv(csv_path, index=False)

    # Basic stats & correlations
    desc = df.describe()
    corr = df[['acc', 'lr', 'wd', 'bs', 'dpr']].corr()

    # Single-variable regressions and plots
    vars_to_check = ['lr', 'wd', 'bs', 'dpr']
    single_results = {}
    plots = {}
    for var in vars_to_check:
        res = single_var_regression(df, var)
        single_results[var] = res
        png = output_dir / f'plot_acc_vs_{var}.png'
        plots[var] = str(plot_acc_vs(df, var, png))

    # Multiple linear regression (standardized inputs)
    X = df[['lr', 'wd', 'bs', 'dpr']].values
    y = df['acc'].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    multi = LinearRegression().fit(Xs, y)
    multi_r2 = float(multi.score(Xs, y))
    multi_coefs = dict(zip(['lr', 'wd', 'bs', 'dpr'], [float(c) for c in multi.coef_]))
    multi_intercept = float(multi.intercept_)

    # Polynomial (degree=2) with RidgeCV
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_feature_names = poly.get_feature_names_out(['lr', 'wd', 'bs', 'dpr'])
    poly_scaler = StandardScaler()
    X_poly_s = poly_scaler.fit_transform(X_poly)
    ridge_alphas = np.logspace(-6, 3, 30)
    ridge = RidgeCV(alphas=ridge_alphas, cv=5, scoring='r2').fit(X_poly_s, y)
    ridge_r2_train = float(ridge.score(X_poly_s, y))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ridge, X_poly_s, y, cv=cv, scoring='r2')
    cv_r2_mean = float(np.mean(cv_scores))
    cv_r2_std = float(np.std(cv_scores))
    poly_coefs = dict(zip(poly_feature_names, ridge.coef_.tolist()))
    sorted_poly_coefs = sorted(poly_coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)

    # Optional statsmodels OLS
    ols_summary_str = None
    if HAVE_STATSM:
        try:
            X_poly_sm = sm.add_constant(X_poly)
            ols = sm.OLS(y, X_poly_sm).fit()
            ols_summary_str = ols.summary().as_text()
        except Exception as e:
            ols_summary_str = f"statsmodels OLS failed: {e}"

    # Detect highly correlated pairs (just hyperparams and acc)
    variables = ['acc', 'lr', 'wd', 'bs', 'dpr']
    high_corr_pairs = []
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            v1 = variables[i]
            v2 = variables[j]
            cval = float(corr.loc[v1, v2])
            if abs(cval) >= MIN_CORR:
                high_corr_pairs.append((v1, v2, cval))

    with open(output_dir / "high_correlation_pairs.txt", "w") as f:
        f.write(f"Pairs with |corr| >= {MIN_CORR}\n")
        f.write("=================================\n")
        for v1, v2, c in high_corr_pairs:
            f.write(f"{v1} — {v2}: corr = {c:.4f}\n")

    # ------------------------
    # Joint visualizations: Accuracy as function of (param1, param2)
    # we create for every pair of hyperparameters (lr, wd, bs, dpr)
    # ------------------------
    from itertools import combinations
    hyperparams = ['lr', 'wd', 'bs', 'dpr']
    joint3d_dir = output_dir / "joint_3d_plots"
    joint3d_dir.mkdir(exist_ok=True)
    heatmap_dir = output_dir / "joint_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    joint_index = {}  # store metadata about produced plots

    for (v1, v2) in combinations(hyperparams, 2):
        # produce 3D scatter + fitted surface + 2D contour
        png3d = joint3d_dir / f"3d_acc_{v1}_{v2}.png"
        pngcont = heatmap_dir / f"contour_acc_{v1}_{v2}.png"

        meta = save_3d_scatter_with_surface(df, v1, v2, png3d, pngcont, transform_flags=None)
        joint_index[f"{v1}__{v2}"] = {
            '3d_png': str(png3d),
            'contour_png': str(pngcont),
            'model_coef': meta['model_coef'],
            'model_intercept': meta['model_intercept'],
            'v1_log_applied': meta['v1_log'],
            'v2_log_applied': meta['v2_log']
        }

    # Save joint_index JSON
    with open(output_dir / 'joint_plots_index.json', 'w') as jf:
        json.dump(joint_index, jf, indent=2)

    # Write report
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("Experiment analysis report\n")
        f.write("==========================\n\n")
        f.write("Parsed CSV: " + str(csv_path) + "\n\n")
        f.write("Basic description:\n")
        f.write(str(desc) + "\n\n")
        f.write("Correlation matrix (acc, lr, wd, bs, dpr):\n")
        f.write(str(corr) + "\n\n")

        f.write("Highly correlated pairs (|corr| >= %.3f):\n" % MIN_CORR)
        for v1, v2, c in high_corr_pairs:
            f.write(f"  {v1} — {v2}: {c:.4f}\n")
        f.write("\n")

        f.write("Single-variable regressions:\n")
        for var, r in single_results.items():
            f.write(f"\nacc ~ {var}\n")
            f.write(f"  slope: {r['slope']}\n")
            f.write(f"  intercept: {r['intercept']}\n")
            f.write(f"  R^2: {r['r2']}\n")
            f.write(f"  Pearson r: {r['pearson_r']}\n")
            f.write(f"  p-value: {r['p_value']}\n")
            f.write(f"  plot: {plots[var]}\n")

        f.write("\nMultiple linear regression (standardized inputs):\n")
        for name, coef in multi_coefs.items():
            f.write(f"  {name}: standardized coef = {coef}\n")
        f.write(f"Intercept: {multi_intercept}\n")
        f.write(f"R^2: {multi_r2}\n")

        f.write("\n\nPolynomial (degree=2) regression with interactions (RidgeCV regularized):\n")
        f.write(f"Ridge alpha chosen: {ridge.alpha_}\n")
        f.write(f"Train R^2 (on full data): {ridge_r2_train}\n")
        f.write(f"CV R^2 (5-fold) mean: {cv_r2_mean:.4f}  std: {cv_r2_std:.4f}\n\n")
        f.write("Top polynomial features by absolute coefficient:\n")
        for name, coef in sorted_poly_coefs[:20]:
            f.write(f"  {name}: {coef}\n")

        if ols_summary_str is not None:
            f.write("\n\nOptional OLS (no regularization) summary from statsmodels:\n")
            f.write(ols_summary_str + "\n")

        f.write("\nJoint plots produced (3D + contour):\n")
        for pair, info in joint_index.items():
            f.write(f"  {pair}: 3d -> {info['3d_png']}, contour -> {info['contour_png']}, v1_log={info['v1_log_applied']}, v2_log={info['v2_log_applied']}\n")

    # Save polynomial coefs
    with open(output_dir / 'poly_coefficients.json', 'w') as jf:
        json.dump(poly_coefs, jf, indent=2)

    # Console summary
    print("Wrote CSV to:", csv_path)
    print("Wrote report to:", report_path)
    print("Joint plots index:", output_dir / 'joint_plots_index.json')
    print("Ridge alpha selected:", ridge.alpha_)
    print(f"Ridge train R^2: {ridge_r2_train:.4f}; CV R^2 (5-fold) mean: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")

    return {
        'df': df,
        'csv': csv_path,
        'report': report_path,
        'joint_index': output_dir / 'joint_plots_index.json',
        'poly': {
            'feature_names': poly_feature_names.tolist(),
            'coefs': poly_coefs,
            'train_r2': ridge_r2_train,
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std,
        }
    }


# -------------------------
# CLI
# -------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Parse and analyze experiments text file.")
    p.add_argument('--file', '-f', type=str, default='base_training_results.txt',
                   help='Path to the experiments text file (default: base_training_results.txt)')
    p.add_argument('--out', '-o', type=str, default='analysis_output',
                   help='Output directory for CSV, plots, report (default: ./analysis_output)')
    args = p.parse_args(argv)

    try:
        analyze_file(args.file, args.out)
    except Exception as e:
        print("ERROR during analysis:", e, file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
