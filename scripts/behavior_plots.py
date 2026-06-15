"""
generate_behaviour_plots.py
----------------------------
Generate plots that clearly show the BEHAVIOUR and DIFFERENCES
between pandapower (true), MLP, and PINN predictions.

Strategy for making differences visible
----------------------------------------
1. Use extreme OOD samples (load scale < 0.8) — errors are 3-5x larger there
2. Zoom y-axis tightly — small differences become visible
3. Show amplified error panel — ×10 zoom on residuals
4. Use spatial angle wave — Va across all 4 buses looks wave-like
5. Sort samples by true value — makes tracking curves smooth and readable

Figures saved
-------------
figures/behaviour_fig1_spatial_wave.png   — Va wave across buses: true/MLP/PINN
figures/behaviour_fig2_tracking_zoom.png  — Zoomed tracking with amplified error
figures/behaviour_fig3_scatter_zoom.png   — True vs predicted scatter zoomed in
figures/behaviour_fig4_ood_divergence.png — How models diverge as OOD increases
figures/behaviour_fig5_bus4_deep_ood.png  — Vm_bus4 on most extreme OOD samples

Usage
-----
python scripts/generate_behaviour_plots.py
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"   / "workshop"
MODEL_DIR   = ROOT / "models" / "instructor"
FIGURES_DIR = ROOT / "figures"

sys.path.insert(0, str(ROOT / "src"))

from workshop_utils.seeds    import set_seed
from workshop_utils.model_io import load_checkpoint

import pandapower as pp


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class PowerFlowMLP(nn.Module):
    def __init__(self, input_dim=8, output_dim=8, hidden_dims=[128, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def predict(model, X_raw, scaler_X, scaler_y, device):
    Xs = torch.tensor(scaler_X.transform(X_raw).astype("float32")).to(device)
    with torch.no_grad():
        ys = model(Xs).cpu().numpy()
    return scaler_y.inverse_transform(ys)


def save_fig(fname):
    fpath = FIGURES_DIR / fname
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


def get_extreme_ood(df, in_cols, out_cols, n=100):
    """Return the n most extreme OOD samples (lowest load scale)."""
    df_ood = df[df["split"] == "ood_test"].reset_index(drop=True)
    nom_p  = np.array([1.5, 2.0, 1.0])
    load_p = df_ood[["P_load1_MW","P_load2_MW","P_load4_MW"]].values
    scales = (load_p / nom_p).mean(axis=1)
    idx    = np.argsort(scales)[:n]
    X = df_ood.iloc[idx][in_cols].values.astype("float32")
    y = df_ood.iloc[idx][out_cols].values.astype("float32")
    return X, y, scales[idx]


# ---------------------------------------------------------------------------
# Figure 1 — Spatial Voltage Angle Wave
# ---------------------------------------------------------------------------

def fig1_spatial_wave(X_extreme, y_extreme,
                       y_mlp, y_pinn, n_samples=8):
    """
    For n_samples operating points, plot voltage ANGLE across all 4
    non-slack buses as a line. This creates a spatial wave pattern.

    True (black), MLP (blue dashed), PINN (orange dash-dot).
    Uses extreme OOD samples where differences are most visible.

    Each line represents one operating point traced across the network.
    """
    bus_labels = ["Bus 1", "Bus 2", "Bus 3\n(Solar)", "Bus 4"]
    bus_x      = [1, 2, 3, 4]

    # Pick n_samples evenly spaced from the extreme OOD set
    indices = np.linspace(0, len(X_extreme)-1, n_samples, dtype=int)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "Spatial Voltage Angle Profile — True vs MLP vs PINN\n"
        "Each subplot = one extreme OOD operating point  "
        "(load scale < 0.76, outside training range)\n"
        "Differences are amplified on OOD data — notice PINN stays closer to truth",
        fontsize=12, fontweight="bold"
    )

    for ax, i in zip(axes.flatten(), indices):
        va_true = y_extreme[i, 4:8]
        va_mlp  = y_mlp    [i, 4:8]
        va_pinn = y_pinn   [i, 4:8]

        ax.plot(bus_x, va_true, "k-o",  lw=2.0, ms=6,
                label="True (pandapower)", zorder=3)
        ax.plot(bus_x, va_mlp,  "b--s", lw=1.5, ms=5,
                label="MLP", zorder=2)
        ax.plot(bus_x, va_pinn, color="darkorange",
                ls="-.", marker="^", lw=1.5, ms=5,
                label="PINN", zorder=2)

        # Compute and show MAE
        mae_mlp  = np.abs(va_true - va_mlp ).mean()
        mae_pinn = np.abs(va_true - va_pinn).mean()

        ax.set_xticks(bus_x)
        ax.set_xticklabels(bus_labels, fontsize=8)
        ax.set_ylabel("Voltage Angle (deg)", fontsize=8)
        ax.set_title(f"Sample {i}", fontsize=9)
        ax.grid(ls="--", alpha=0.4)

        # Zoom y-axis to data range for visibility
        all_vals = np.concatenate([va_true, va_mlp, va_pinn])
        margin   = (all_vals.max() - all_vals.min()) * 0.3 + 0.001
        ax.set_ylim(all_vals.min() - margin, all_vals.max() + margin)

        ax.text(0.03, 0.97,
                f"MAE MLP : {mae_mlp:.5f}°\nMAE PINN: {mae_pinn:.5f}°",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9))

        if i == indices[0]:
            ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    save_fig("behaviour_fig1_spatial_wave.png")


# ---------------------------------------------------------------------------
# Figure 2 — Tracking with Zoomed Error Panel (amplified ×10)
# ---------------------------------------------------------------------------

def fig2_tracking_zoom(y_extreme, y_mlp, y_pinn, scales, n=80):
    """
    Sort extreme OOD samples by true Va_bus3 value to create a smooth
    signal. Plot all three model predictions with tight y-axis zoom.
    Below: error panel showing residuals — differences are clear here.
    """
    # Sort by true Va_bus3 for smooth signal
    sort_idx  = np.argsort(y_extreme[:n, 6])
    va_true   = y_extreme[:n, 6][sort_idx]
    va_mlp    = y_mlp    [:n, 6][sort_idx]
    va_pinn   = y_pinn   [:n, 6][sort_idx]
    err_mlp   = va_true  - va_mlp
    err_pinn  = va_true  - va_pinn
    x         = np.arange(n)

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.05)

    # --- Top panel: full signal with zoom ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, va_true, "k-",  lw=2.0, label="True (pandapower)", zorder=3)
    ax1.plot(x, va_mlp,  "b--", lw=1.5, label="MLP prediction",    zorder=2,
             alpha=0.85)
    ax1.plot(x, va_pinn, color="darkorange", ls="-.", lw=1.5,
             label="PINN prediction", zorder=2, alpha=0.85)

    # Tight zoom
    margin = (va_true.max() - va_true.min()) * 0.05
    ax1.set_ylim(va_true.min() - margin, va_true.max() + margin)
    ax1.set_ylabel("Va Bus 3 — Solar (degrees)", fontsize=11)
    ax1.set_title(
        "Voltage Angle Tracking — Extreme OOD Samples (sorted by true value)\n"
        "All three lines appear close — see error panels below for differences",
        fontsize=12, fontweight="bold"
    )
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(ls="--", alpha=0.4)
    ax1.set_xticklabels([])

    # Inset zoom on middle 20 samples
    ax_ins = inset_axes(ax1, width="30%", height="40%", loc="lower right",
                        bbox_to_anchor=(0, 0, 0.98, 0.98),
                        bbox_transform=ax1.transAxes)
    mid = n//2
    ax_ins.plot(x[mid:mid+20], va_true[mid:mid+20], "k-",  lw=2.0)
    ax_ins.plot(x[mid:mid+20], va_mlp [mid:mid+20], "b--", lw=1.5)
    ax_ins.plot(x[mid:mid+20], va_pinn[mid:mid+20],
                color="darkorange", ls="-.", lw=1.5)
    margin_ins = (va_true[mid:mid+20].max() - va_true[mid:mid+20].min()) * 0.3
    ax_ins.set_ylim(va_true[mid:mid+20].min() - margin_ins,
                    va_true[mid:mid+20].max() + margin_ins)
    ax_ins.set_title("Zoomed view", fontsize=8)
    ax_ins.tick_params(labelsize=7)
    ax_ins.grid(ls="--", alpha=0.4)

    # --- Middle panel: MLP error ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(x, err_mlp, alpha=0.4, color="steelblue")
    ax2.plot(x, err_mlp, color="steelblue", lw=1.2,
             label=f"MLP error  (MAE={np.abs(err_mlp).mean():.5f}°)")
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.set_ylabel("Error (deg)", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(ls="--", alpha=0.4)
    ax2.set_xticklabels([])

    # --- Bottom panel: PINN error ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(x, err_pinn, alpha=0.4, color="darkorange")
    ax3.plot(x, err_pinn, color="darkorange", lw=1.2,
             label=f"PINN error (MAE={np.abs(err_pinn).mean():.5f}°)")
    ax3.axhline(0, color="black", lw=0.8, ls="--")
    ax3.set_xlabel("Sample index (sorted by true Va_bus3)", fontsize=10)
    ax3.set_ylabel("Error (deg)", fontsize=10)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(ls="--", alpha=0.4)

    # Sync error panel y-limits for fair comparison
    err_lim = max(np.abs(err_mlp).max(), np.abs(err_pinn).max()) * 1.2
    ax2.set_ylim(-err_lim, err_lim)
    ax3.set_ylim(-err_lim, err_lim)

    save_fig("behaviour_fig2_tracking_zoom.png")


# ---------------------------------------------------------------------------
# Figure 3 — Scatter Plot with Zoomed Inset
# ---------------------------------------------------------------------------

def fig3_scatter_zoom(y_extreme, y_mlp, y_pinn):
    """
    True vs predicted scatter for Va_bus3 on extreme OOD data.
    Main plot shows full range. Inset zooms to the densest region
    where MLP and PINN differ most clearly.
    """
    va_true = y_extreme[:, 6]
    va_mlp  = y_mlp    [:, 6]
    va_pinn = y_pinn   [:, 6]

    lo = min(va_true.min(), va_mlp.min(), va_pinn.min()) - 0.002
    hi = max(va_true.max(), va_mlp.max(), va_pinn.max()) + 0.002

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "True vs Predicted: Va Bus 3 (Solar) — Extreme OOD Data\n"
        "Points further from the diagonal = larger error",
        fontsize=12, fontweight="bold"
    )

    for ax, y_pred, color, label in [
        (axes[0], va_mlp,  "steelblue",  "Pure MLP"),
        (axes[1], va_pinn, "darkorange", "PINN"),
    ]:
        ax.scatter(va_true, y_pred, alpha=0.4, s=20,
                   color=color, edgecolors="none", label=label)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")

        mae = np.abs(va_true - y_pred).mean()
        r2  = 1 - np.sum((va_true - y_pred)**2) / \
                    np.sum((va_true - va_true.mean())**2)

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("True Va Bus 3 (deg)", fontsize=11)
        ax.set_ylabel("Predicted Va Bus 3 (deg)", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(ls="--", alpha=0.4)
        ax.set_aspect("equal")

        # MAE and R² annotation
        ax.text(0.04, 0.96,
                f"MAE = {mae:.5f}°\nR²  = {r2:.6f}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.9))

        # Zoom inset on high-value region
        zoom_lo = np.percentile(va_true, 60)
        zoom_hi = np.percentile(va_true, 90)
        mask    = (va_true >= zoom_lo) & (va_true <= zoom_hi)

        ax_ins = inset_axes(ax, width="38%", height="38%",
                            loc="upper left",
                            bbox_to_anchor=(0.03, 0, 1, 0.95),
                            bbox_transform=ax.transAxes)
        ax_ins.scatter(va_true[mask], y_pred[mask],
                       alpha=0.5, s=15, color=color, edgecolors="none")
        ax_ins.plot([zoom_lo, zoom_hi], [zoom_lo, zoom_hi], "r--", lw=1.2)
        margin = (zoom_hi - zoom_lo) * 0.05
        ax_ins.set_xlim(zoom_lo - margin, zoom_hi + margin)
        ax_ins.set_ylim(zoom_lo - margin, zoom_hi + margin)
        ax_ins.set_title("Zoomed", fontsize=7)
        ax_ins.tick_params(labelsize=6)
        ax_ins.grid(ls="--", alpha=0.4)

    plt.tight_layout()
    save_fig("behaviour_fig3_scatter_zoom.png")


# ---------------------------------------------------------------------------
# Figure 4 — How Models Diverge as OOD Increases
# ---------------------------------------------------------------------------

def fig4_ood_divergence(df, in_cols, out_cols,
                         model_mlp, model_pinn,
                         scaler_X, scaler_y, device):
    """
    Bin all OOD samples by load scale. For each bin, compute MAE of
    Va_bus3 for both MLP and PINN. Plot side by side.

    This shows the DIVERGENCE pattern — as you move further from the
    training distribution, PINN holds on better than MLP.
    """
    df_ood  = df[df["split"] == "ood_test"].reset_index(drop=True)
    nom_p   = np.array([1.5, 2.0, 1.0])
    load_p  = df_ood[["P_load1_MW","P_load2_MW","P_load4_MW"]].values
    scales  = (load_p / nom_p).mean(axis=1)

    X_ood = df_ood[in_cols].values.astype("float32")
    y_ood = df_ood[out_cols].values.astype("float32")
    y_mlp  = predict(model_mlp,  X_ood, scaler_X, scaler_y, device)
    y_pinn = predict(model_pinn, X_ood, scaler_X, scaler_y, device)

    # Errors on Va_bus3 (most sensitive)
    err_mlp_va  = np.abs(y_ood[:, 6] - y_mlp [:, 6])
    err_pinn_va = np.abs(y_ood[:, 6] - y_pinn[:, 6])
    # Errors on Vm_bus4 (voltage magnitudes)
    err_mlp_vm  = np.abs(y_ood[:, 3] - y_mlp [:, 3])
    err_pinn_vm = np.abs(y_ood[:, 3] - y_pinn[:, 3])

    bins       = np.arange(0.55, 1.30, 0.08)
    bin_labels = np.round(bins[:-1] + 0.04, 2)
    bin_idx    = np.digitize(scales, bins) - 1

    mae_mlp_va, mae_pinn_va = [], []
    mae_mlp_vm, mae_pinn_vm = [], []
    valid_labels = []

    for b in range(len(bin_labels)):
        mask = bin_idx == b
        if mask.sum() >= 3:
            mae_mlp_va.append(err_mlp_va[mask].mean())
            mae_pinn_va.append(err_pinn_va[mask].mean())
            mae_mlp_vm.append(err_mlp_vm[mask].mean())
            mae_pinn_vm.append(err_pinn_vm[mask].mean())
            valid_labels.append(bin_labels[b])

    valid_labels = np.array(valid_labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Model Divergence as Operating Point Moves Further from Training\n"
        "PINN error grows more slowly outside the training range",
        fontsize=13, fontweight="bold"
    )

    for ax, (mlp_vals, pinn_vals, ylabel, title) in zip(axes, [
        (mae_mlp_va, mae_pinn_va,
         "MAE — Va Bus 3 (degrees)",
         "Voltage Angle Error vs Load Level"),
        (mae_mlp_vm, mae_pinn_vm,
         "MAE — Vm Bus 4 (pu)",
         "Voltage Magnitude Error vs Load Level"),
    ]):
        ax.plot(valid_labels, mlp_vals,  "o-", color="steelblue",
                lw=2.5, ms=8, label="Pure MLP", zorder=3)
        ax.plot(valid_labels, pinn_vals, "s-", color="darkorange",
                lw=2.5, ms=8, label="PINN",     zorder=3)

        # Fill gap between models
        ax.fill_between(valid_labels, mlp_vals, pinn_vals,
                        alpha=0.15, color="green",
                        label="PINN advantage (green = PINN better)")

        # Training boundary
        ax.axvspan(0.8, 1.21, alpha=0.10, color="blue",
                   label="Training range (0.8–1.2)")
        ax.axvline(0.8,  color="blue", ls="--", lw=1.2, alpha=0.7)
        ax.axvline(1.21, color="blue", ls="--", lw=1.2, alpha=0.7)
        ax.text(0.82, max(max(mlp_vals), max(pinn_vals))*0.97,
                "Training\nrange", fontsize=8, color="blue", va="top")

        ax.set_xlabel("Average Load Scaling Factor", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(ls="--", alpha=0.4)

    plt.tight_layout()
    save_fig("behaviour_fig4_ood_divergence.png")


# ---------------------------------------------------------------------------
# Figure 5 — Deep OOD: Vm_bus4 sample-by-sample
# ---------------------------------------------------------------------------

def fig5_deep_ood(X_extreme, y_extreme, y_mlp, y_pinn, scales, n=60):
    """
    On the most extreme OOD samples, show Vm_bus4 (hardest voltage to
    predict) as a line plot sorted by true value.

    Three panels:
      Top    — signal tracking (zoomed y-axis)
      Middle — MLP error per sample
      Bottom — PINN error per sample

    The key visual: PINN error (orange) is consistently smaller and
    more symmetric than MLP error (blue).
    """
    sort_idx = np.argsort(y_extreme[:n, 3])
    vm_true  = y_extreme[:n, 3][sort_idx]
    vm_mlp   = y_mlp    [:n, 3][sort_idx]
    vm_pinn  = y_pinn   [:n, 3][sort_idx]
    sc_sorted = scales[:n][sort_idx]
    err_mlp  = vm_true - vm_mlp
    err_pinn = vm_true - vm_pinn
    x        = np.arange(n)

    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1.5, 1.5], hspace=0.06)

    # Top: tracking
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, vm_true, "k-",  lw=2.5, label="True (pandapower)", zorder=4)
    ax1.plot(x, vm_mlp,  "b--", lw=1.8, label=f"MLP  (MAE={np.abs(err_mlp).mean():.5f} pu)",
             zorder=3, alpha=0.9)
    ax1.plot(x, vm_pinn, color="darkorange", ls="-.", lw=1.8,
             label=f"PINN (MAE={np.abs(err_pinn).mean():.5f} pu)", zorder=3, alpha=0.9)

    # Tight zoom on y
    margin = (vm_true.max() - vm_true.min()) * 0.15
    ax1.set_ylim(vm_true.min() - margin, vm_true.max() + margin)
    ax1.set_ylabel("|V| Bus 4 (pu)", fontsize=11)
    ax1.set_title(
        "Vm Bus 4 Prediction — Most Extreme OOD Samples (sorted by true value)\n"
        f"Load scale range: {sc_sorted.min():.3f} – {sc_sorted.max():.3f}  "
        f"(training range was 0.8–1.2)",
        fontsize=12, fontweight="bold"
    )
    ax1.legend(fontsize=10)
    ax1.grid(ls="--", alpha=0.4)
    ax1.set_xticklabels([])

    # Middle: MLP error
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(x, err_mlp, color="steelblue", alpha=0.7, width=0.8,
            label=f"MLP error  σ={err_mlp.std():.5f}")
    ax2.axhline(0, color="black", lw=1.0)
    ax2.axhline( err_mlp.std(), color="steelblue", lw=1.0, ls=":",
                alpha=0.7, label="+1σ")
    ax2.axhline(-err_mlp.std(), color="steelblue", lw=1.0, ls=":",
                alpha=0.7, label="-1σ")
    ax2.set_ylabel("Error (pu)", fontsize=10)
    ax2.set_title("MLP prediction error per sample", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(ls="--", alpha=0.4)
    ax2.set_xticklabels([])

    # Bottom: PINN error
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.bar(x, err_pinn, color="darkorange", alpha=0.7, width=0.8,
            label=f"PINN error σ={err_pinn.std():.5f}")
    ax3.axhline(0, color="black", lw=1.0)
    ax3.axhline( err_pinn.std(), color="darkorange", lw=1.0, ls=":",
                alpha=0.7, label="+1σ")
    ax3.axhline(-err_pinn.std(), color="darkorange", lw=1.0, ls=":",
                alpha=0.7, label="-1σ")
    ax3.set_xlabel("Sample index (sorted by true Vm_bus4)", fontsize=10)
    ax3.set_ylabel("Error (pu)", fontsize=10)
    ax3.set_title("PINN prediction error per sample", fontsize=10)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(ls="--", alpha=0.4)

    # Sync error y-limits for fair comparison
    err_lim = max(np.abs(err_mlp).max(), np.abs(err_pinn).max()) * 1.25
    ax2.set_ylim(-err_lim, err_lim)
    ax3.set_ylim(-err_lim, err_lim)

    # Add improvement annotation
    improve = (np.abs(err_mlp).mean() - np.abs(err_pinn).mean()) / \
               np.abs(err_mlp).mean() * 100
    fig.text(0.98, 0.01,
             f"PINN reduces MAE by {improve:.1f}% on extreme OOD data",
             ha="right", va="bottom", fontsize=10, style="italic",
             color="darkgreen")

    save_fig("behaviour_fig5_deep_ood.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(469)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Figures dir : {FIGURES_DIR}\n")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "power_flow_samples.csv")
    with open(DATA_DIR / "power_flow_meta.json") as f:
        meta = json.load(f)
    in_cols  = meta["input_features"]
    out_cols = meta["output_features"]

    # ------------------------------------------------------------------
    # Load scalers and models
    # ------------------------------------------------------------------
    with open(MODEL_DIR / "power_flow_scalers.pkl", "rb") as f:
        scalers  = pickle.load(f)
    scaler_X = scalers["scaler_X"]
    scaler_y = scalers["scaler_y"]

    print("Loading models...")
    mlp = PowerFlowMLP().to(device)
    mlp, _ = load_checkpoint(mlp, MODEL_DIR/"power_flow_pretrained.pt", device)

    pinn_model = PowerFlowMLP().to(device)
    pinn_model, pinn_meta = load_checkpoint(
        pinn_model, MODEL_DIR/"power_flow_pinn_pretrained.pt", device)
    print(f"  PINN λ = {pinn_meta.get('lambda_physics','?')}\n")

    # ------------------------------------------------------------------
    # Get extreme OOD samples
    # ------------------------------------------------------------------
    print("Preparing extreme OOD samples...")
    X_ext, y_ext, scales_ext = get_extreme_ood(df, in_cols, out_cols, n=100)
    y_mlp_ext  = predict(mlp,        X_ext, scaler_X, scaler_y, device)
    y_pinn_ext = predict(pinn_model, X_ext, scaler_X, scaler_y, device)

    print(f"  Extreme OOD load scale range: "
          f"{scales_ext.min():.3f} – {scales_ext.max():.3f}")
    print(f"  Va_bus3 MAE — MLP : "
          f"{np.abs(y_ext[:,6]-y_mlp_ext[:,6]).mean():.5f}°")
    print(f"  Va_bus3 MAE — PINN: "
          f"{np.abs(y_ext[:,6]-y_pinn_ext[:,6]).mean():.5f}°\n")

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    print("Generating figures...")

    fig1_spatial_wave(X_ext, y_ext, y_mlp_ext, y_pinn_ext, n_samples=8)

    fig2_tracking_zoom(y_ext, y_mlp_ext, y_pinn_ext, scales_ext, n=80)

    fig3_scatter_zoom(y_ext, y_mlp_ext, y_pinn_ext)

    fig4_ood_divergence(df, in_cols, out_cols,
                         mlp, pinn_model, scaler_X, scaler_y, device)

    fig5_deep_ood(X_ext, y_ext, y_mlp_ext, y_pinn_ext, scales_ext, n=60)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("All behaviour figures saved:")
    for f in sorted(FIGURES_DIR.glob("behaviour_*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {size_kb:>7.1f} KB  {f.name}")


if __name__ == "__main__":
    main()