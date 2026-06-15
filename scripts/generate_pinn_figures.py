"""
generate_pinn_figures.py
-------------------------
Compare PINN surrogate vs pure MLP surrogate and save figures.

Figures saved
-------------
figures/pinn_fig1_loss_components.png  — PINN data loss vs physics loss over epochs
figures/pinn_fig2_mae_inDist.png       — In-distribution MAE: PINN vs MLP
figures/pinn_fig3_mae_ood.png          — OOD MAE: PINN vs MLP
figures/pinn_fig4_mae_comparison.png   — Side-by-side grouped bar: all conditions
figures/pinn_fig5_physics_residual.png — Physics violation: PINN vs MLP on OOD data
figures/pinn_fig6_error_vs_loading.png — Error vs load level: PINN vs MLP

Usage
-----
Run from the project root:
    python scripts/generate_pinn_figures.py

Requires:
    models/instructor/power_flow_pretrained.pt        (pure MLP)
    models/instructor/power_flow_pinn_pretrained.pt   (PINN)
    models/instructor/power_flow_scalers.pkl
    models/instructor/power_flow_training_curves.json
    models/instructor/power_flow_pinn_training_curves.json
    data/workshop/power_flow_samples.csv
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
import seaborn as sns
from matplotlib.patches import Patch

ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"   / "workshop"
MODEL_DIR   = ROOT / "models" / "instructor"
FIGURES_DIR = ROOT / "figures"

sys.path.insert(0, str(ROOT / "src"))

from workshop_utils.seeds    import set_seed
from workshop_utils.model_io import load_checkpoint
from workshop_utils.power_flow import build_network
from workshop_utils.pinn     import build_ybus, physics_loss

import pandapower as pp


# ---------------------------------------------------------------------------
# Model definition (same as training scripts)
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
    X_sc = torch.tensor(scaler_X.transform(X_raw).astype("float32")).to(device)
    with torch.no_grad():
        y_sc = model(X_sc).cpu().numpy()
    return scaler_y.inverse_transform(y_sc)


def compute_physics_residual(y_pred_raw, X_raw, G, B, base_mva, device):
    """
    Compute mean physics residual (MW) for a set of predictions.
    Returns scalar: average power balance violation across all buses and samples.
    """
    y_t = torch.tensor(y_pred_raw.astype("float32")).to(device)
    X_t = torch.tensor(X_raw.astype("float32")).to(device)
    with torch.no_grad():
        res = physics_loss(y_t, X_t, G, B, base_mva)
    return res.item()


def save_fig(fname):
    fpath = FIGURES_DIR / fname
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")


# ---------------------------------------------------------------------------
# Figure 1 — PINN loss components over training
# ---------------------------------------------------------------------------

def fig1_loss_components(pinn_curves):
    epochs     = range(1, len(pinn_curves["train_total"]) + 1)
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PINN Training — Loss Components", fontsize=14,
                 fontweight="bold")

    # Left: all three loss components on log scale
    ax = axes[0]
    ax.plot(epochs, pinn_curves["train_total"], color="black",
            lw=1.8, label="Total loss (data + λ·physics)")
    ax.plot(epochs, pinn_curves["train_data"],  color="steelblue",
            lw=1.5, ls="--", label="Data loss (MSE)")
    ax.plot(epochs, pinn_curves["train_phys"],  color="darkorange",
            lw=1.5, ls="-.", label="Physics residual")
    ax.axvline(20, color="gray", ls=":", lw=1.2,
               label="Phase 1 end (epoch 20)")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training loss components (log scale)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    # Right: val total vs train total
    ax = axes[1]
    ax.plot(epochs, pinn_curves["train_total"], color="steelblue",
            lw=1.8, label="Train total loss")
    ax.plot(epochs, pinn_curves["val_total"],   color="darkorange",
            lw=1.8, ls="--", label="Val total loss")
    best = int(np.argmin(pinn_curves["val_total"])) + 1
    ax.axvline(best, color="green", ls=":", lw=1.5,
               label=f"Best val epoch {best}")
    ax.axvline(20, color="gray", ls="-.", lw=1.2,
               label="Phase 1 end")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("Train vs. Val total loss (log scale)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    save_fig("pinn_fig1_loss_components.png")


# ---------------------------------------------------------------------------
# Figure 2 — In-distribution MAE: PINN vs MLP
# ---------------------------------------------------------------------------

def fig2_mae_inDist(mae_mlp_id, mae_pinn_id, feature_names):
    display = {
        "Vm_bus1_pu": "|V| Bus 1 (pu)", "Vm_bus2_pu": "|V| Bus 2 (pu)",
        "Vm_bus3_pu": "|V| Bus 3 Solar (pu)", "Vm_bus4_pu": "|V| Bus 4 (pu)",
        "Va_bus1_deg": "δ Bus 1 (deg)", "Va_bus2_deg": "δ Bus 2 (deg)",
        "Va_bus3_deg": "δ Bus 3 Solar (deg)", "Va_bus4_deg": "δ Bus 4 (deg)",
    }
    labels = [display.get(f, f) for f in feature_names]
    n      = len(labels)
    y_pos  = np.arange(n)
    h      = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.barh(y_pos - h/2, mae_mlp_id,  height=h,
                 color="steelblue", edgecolor="white", label="Pure MLP")
    b2 = ax.barh(y_pos + h/2, mae_pinn_id, height=h,
                 color="darkorange", edgecolor="white", label="PINN")
    ax.bar_label(b1, fmt="%.5f", padding=3, fontsize=8, color="steelblue")
    ax.bar_label(b2, fmt="%.5f", padding=3, fontsize=8, color="darkorange")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute Error", fontsize=11)
    ax.set_title("In-Distribution MAE: PINN vs Pure MLP  (load ±20%)\n"
                 "Both models perform similarly within training range",
                 fontsize=12, pad=10)
    ax.set_xlim(0, max(mae_mlp_id.max(), mae_pinn_id.max()) * 1.3)
    ax.legend(fontsize=10)
    ax.grid(axis="x", ls="--", alpha=0.4)
    plt.tight_layout()
    save_fig("pinn_fig2_mae_inDist.png")


# ---------------------------------------------------------------------------
# Figure 3 — OOD MAE: PINN vs MLP
# ---------------------------------------------------------------------------

def fig3_mae_ood(mae_mlp_ood, mae_pinn_ood, feature_names):
    display = {
        "Vm_bus1_pu": "|V| Bus 1 (pu)", "Vm_bus2_pu": "|V| Bus 2 (pu)",
        "Vm_bus3_pu": "|V| Bus 3 Solar (pu)", "Vm_bus4_pu": "|V| Bus 4 (pu)",
        "Va_bus1_deg": "δ Bus 1 (deg)", "Va_bus2_deg": "δ Bus 2 (deg)",
        "Va_bus3_deg": "δ Bus 3 Solar (deg)", "Va_bus4_deg": "δ Bus 4 (deg)",
    }
    labels = [display.get(f, f) for f in feature_names]
    n      = len(labels)
    y_pos  = np.arange(n)
    h      = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.barh(y_pos - h/2, mae_mlp_ood,  height=h,
                 color="steelblue", edgecolor="white", label="Pure MLP")
    b2 = ax.barh(y_pos + h/2, mae_pinn_ood, height=h,
                 color="darkorange", edgecolor="white", label="PINN")
    ax.bar_label(b1, fmt="%.5f", padding=3, fontsize=8, color="steelblue")
    ax.bar_label(b2, fmt="%.5f", padding=3, fontsize=8, color="darkorange")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute Error", fontsize=11)
    ax.set_title("OOD MAE: PINN vs Pure MLP  (load ±40%)\n"
                 "PINN degrades more gracefully outside training range",
                 fontsize=12, pad=10)
    ax.set_xlim(0, max(mae_mlp_ood.max(), mae_pinn_ood.max()) * 1.3)
    ax.legend(fontsize=10)
    ax.grid(axis="x", ls="--", alpha=0.4)
    plt.tight_layout()
    save_fig("pinn_fig3_mae_ood.png")


# ---------------------------------------------------------------------------
# Figure 4 — Full comparison: all 4 conditions in one plot
# ---------------------------------------------------------------------------

def fig4_full_comparison(mae_mlp_id, mae_mlp_ood,
                          mae_pinn_id, mae_pinn_ood, feature_names):
    display = {
        "Vm_bus1_pu": "|V| Bus1", "Vm_bus2_pu": "|V| Bus2",
        "Vm_bus3_pu": "|V| Bus3", "Vm_bus4_pu": "|V| Bus4",
        "Va_bus1_deg": "δ Bus1",  "Va_bus2_deg": "δ Bus2",
        "Va_bus3_deg": "δ Bus3",  "Va_bus4_deg": "δ Bus4",
    }
    labels = [display.get(f, f) for f in feature_names]
    x      = np.arange(len(labels))
    w      = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 1.5*w, mae_mlp_id,  width=w, color="steelblue",
           alpha=1.0,  label="MLP  in-dist")
    ax.bar(x - 0.5*w, mae_pinn_id, width=w, color="darkorange",
           alpha=1.0,  label="PINN in-dist")
    ax.bar(x + 0.5*w, mae_mlp_ood, width=w, color="steelblue",
           alpha=0.45, label="MLP  OOD",   hatch="//")
    ax.bar(x + 1.5*w, mae_pinn_ood,width=w, color="darkorange",
           alpha=0.45, label="PINN OOD",   hatch="//")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, rotation=15)
    ax.set_ylabel("Mean Absolute Error", fontsize=11)
    ax.set_title("PINN vs MLP — All Conditions\n"
                 "Solid = in-distribution (±20%)  |  "
                 "Hatched = OOD (±40%)",
                 fontsize=12, pad=10)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_yscale("log")
    plt.tight_layout()
    save_fig("pinn_fig4_mae_comparison.png")


# ---------------------------------------------------------------------------
# Figure 5 — Physics residual violation: PINN vs MLP
# ---------------------------------------------------------------------------

def fig5_physics_residual(res_mlp_id, res_mlp_ood,
                           res_pinn_id, res_pinn_ood):
    """
    Bar chart showing mean power balance violation (physics residual)
    for MLP and PINN on in-distribution and OOD data.

    Lower = more physically consistent.
    """
    categories = ["In-dist (±20%)", "OOD (±40%)"]
    mlp_vals   = [res_mlp_id,  res_mlp_ood]
    pinn_vals  = [res_pinn_id, res_pinn_ood]

    x = np.arange(len(categories))
    w = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, mlp_vals,  width=w, color="steelblue",
                edgecolor="white", label="Pure MLP")
    b2 = ax.bar(x + w/2, pinn_vals, width=w, color="darkorange",
                edgecolor="white", label="PINN")

    ax.bar_label(b1, fmt="%.6f", padding=4, fontsize=9, color="steelblue")
    ax.bar_label(b2, fmt="%.6f", padding=4, fontsize=9, color="darkorange")

    ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Mean Squared Power Balance Residual (pu²)", fontsize=10)
    ax.set_title("Physics Constraint Violation: PINN vs Pure MLP\n"
                 "Lower = more physically consistent predictions",
                 fontsize=12, pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", ls="--", alpha=0.4)

    # Annotation explaining significance
    ax.text(0.5, 0.95,
            "PINN explicitly penalises physics violations during training\n"
            "→ predictions stay closer to valid power flow solutions",
            transform=ax.transAxes, fontsize=9, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8))

    plt.tight_layout()
    save_fig("pinn_fig5_physics_residual.png")


# ---------------------------------------------------------------------------
# Figure 6 — Error vs load scaling: PINN vs MLP
# ---------------------------------------------------------------------------

def fig6_error_vs_loading(model_mlp, model_pinn,
                           df, in_cols, out_cols,
                           scaler_X, scaler_y, device):
    df_ood = df[df["split"] == "ood_test"].copy()

    nom_p      = np.array([1.5, 2.0, 1.0])
    load_p     = df_ood[["P_load1_MW","P_load2_MW","P_load4_MW"]].values
    scales     = (load_p / nom_p).mean(axis=1)

    bins       = np.arange(0.55, 1.45, 0.1)
    bin_labels = np.round(bins[:-1] + 0.05, 2)
    bin_idx    = np.digitize(scales, bins) - 1

    X_ood = df_ood[in_cols].values.astype("float32")
    y_ood = df_ood[out_cols].values.astype("float32")

    y_pred_mlp  = predict(model_mlp,  X_ood, scaler_X, scaler_y, device)
    y_pred_pinn = predict(model_pinn, X_ood, scaler_X, scaler_y, device)

    mae_mlp_vm  = np.abs(y_ood[:,:4] - y_pred_mlp [:,:4]).mean(axis=1)
    mae_pinn_vm = np.abs(y_ood[:,:4] - y_pred_pinn[:,:4]).mean(axis=1)

    bin_mlp, bin_pinn, valid_labels = [], [], []
    for b in range(len(bin_labels)):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_mlp.append(mae_mlp_vm[mask].mean())
            bin_pinn.append(mae_pinn_vm[mask].mean())
            valid_labels.append(bin_labels[b])

    bin_mlp  = np.array(bin_mlp)
    bin_pinn = np.array(bin_pinn)
    valid_labels = np.array(valid_labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(valid_labels, bin_mlp,  "o-", color="steelblue",
            lw=2, ms=7, label="Pure MLP")
    ax.plot(valid_labels, bin_pinn, "s-", color="darkorange",
            lw=2, ms=7, label="PINN")

    ax.axvspan(0.8, 1.2, alpha=0.12, color="green",
               label="Training range (±20%)")
    ax.axvline(0.8, color="green", ls="--", lw=1)
    ax.axvline(1.2, color="green", ls="--", lw=1)

    ax.set_xlabel("Load scaling factor (pu)", fontsize=11)
    ax.set_ylabel("Mean Absolute Error — |V| (pu)", fontsize=11)
    ax.set_title("Surrogate Error vs. Load Level: PINN vs Pure MLP\n"
                 "Green band = training range  |  "
                 "PINN degrades more slowly outside it",
                 fontsize=12, pad=10)
    ax.legend(fontsize=10)
    ax.grid(ls="--", alpha=0.4)
    plt.tight_layout()
    save_fig("pinn_fig6_error_vs_loading.png")


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
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "power_flow_samples.csv")
    with open(DATA_DIR / "power_flow_meta.json") as f:
        meta = json.load(f)
    in_cols  = meta["input_features"]
    out_cols = meta["output_features"]

    df_id  = (df[df["split"] == "train"]
              .sample(500, random_state=469)
              .reset_index(drop=True))
    df_ood = df[df["split"] == "ood_test"].reset_index(drop=True)

    X_id  = df_id[in_cols].values.astype("float32")
    y_id  = df_id[out_cols].values.astype("float32")
    X_ood = df_ood[in_cols].values.astype("float32")
    y_ood = df_ood[out_cols].values.astype("float32")

    # ------------------------------------------------------------------
    # 2. Load scalers
    # ------------------------------------------------------------------
    with open(MODEL_DIR / "power_flow_scalers.pkl", "rb") as f:
        scalers  = pickle.load(f)
    scaler_X = scalers["scaler_X"]
    scaler_y = scalers["scaler_y"]

    # ------------------------------------------------------------------
    # 3. Load both models
    # ------------------------------------------------------------------
    print("Loading Pure MLP...")
    mlp = PowerFlowMLP().to(device)
    mlp, _ = load_checkpoint(mlp, MODEL_DIR / "power_flow_pretrained.pt", device)

    print("Loading PINN...")
    pinn = PowerFlowMLP().to(device)
    pinn, pinn_meta = load_checkpoint(
        pinn, MODEL_DIR / "power_flow_pinn_pretrained.pt", device)
    print(f"  PINN lambda : {pinn_meta.get('lambda_physics', '?')}\n")

    # ------------------------------------------------------------------
    # 4. Build Y-bus for physics residual calculation
    # ------------------------------------------------------------------
    print("Building Y-bus...")
    net = build_network()
    pp.runpp(net, numba=False)
    G, B, base_mva = build_ybus(net)
    G, B = G.to(device), B.to(device)

    # ------------------------------------------------------------------
    # 5. Predictions
    # ------------------------------------------------------------------
    print("Running predictions...")
    y_pred_mlp_id   = predict(mlp,  X_id,  scaler_X, scaler_y, device)
    y_pred_mlp_ood  = predict(mlp,  X_ood, scaler_X, scaler_y, device)
    y_pred_pinn_id  = predict(pinn, X_id,  scaler_X, scaler_y, device)
    y_pred_pinn_ood = predict(pinn, X_ood, scaler_X, scaler_y, device)

    mae_mlp_id   = np.abs(y_id  - y_pred_mlp_id ).mean(axis=0)
    mae_mlp_ood  = np.abs(y_ood - y_pred_mlp_ood ).mean(axis=0)
    mae_pinn_id  = np.abs(y_id  - y_pred_pinn_id ).mean(axis=0)
    mae_pinn_ood = np.abs(y_ood - y_pred_pinn_ood).mean(axis=0)

    # ------------------------------------------------------------------
    # 6. Physics residuals
    # ------------------------------------------------------------------
    print("Computing physics residuals...")
    res_mlp_id   = compute_physics_residual(y_pred_mlp_id,   X_id,  G, B, base_mva, device)
    res_mlp_ood  = compute_physics_residual(y_pred_mlp_ood,  X_ood, G, B, base_mva, device)
    res_pinn_id  = compute_physics_residual(y_pred_pinn_id,  X_id,  G, B, base_mva, device)
    res_pinn_ood = compute_physics_residual(y_pred_pinn_ood, X_ood, G, B, base_mva, device)

    # ------------------------------------------------------------------
    # 7. Print comparison table
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"  {'Feature':<22} {'MLP_ID':>10} {'PINN_ID':>10} "
          f"{'MLP_OOD':>10} {'PINN_OOD':>10}")
    print("  " + "-"*64)
    for name, m_id, p_id, m_ood, p_ood in zip(
            out_cols, mae_mlp_id, mae_pinn_id, mae_mlp_ood, mae_pinn_ood):
        print(f"  {name:<22} {m_id:>10.6f} {p_id:>10.6f} "
              f"{m_ood:>10.6f} {p_ood:>10.6f}")
    print(f"\n  Physics residual (pu²):")
    print(f"    MLP  in-dist : {res_mlp_id:.8f}")
    print(f"    PINN in-dist : {res_pinn_id:.8f}")
    print(f"    MLP  OOD     : {res_mlp_ood:.8f}")
    print(f"    PINN OOD     : {res_pinn_ood:.8f}")
    print(f"\n  Physics residual reduction (OOD):")
    if res_mlp_ood > 0:
        reduction = (res_mlp_ood - res_pinn_ood) / res_mlp_ood * 100
        print(f"    PINN reduces physics violation by {reduction:.1f}% on OOD data")

    # ------------------------------------------------------------------
    # 8. Load training curves
    # ------------------------------------------------------------------
    with open(MODEL_DIR / "power_flow_pinn_training_curves.json") as f:
        pinn_curves = json.load(f)

    # ------------------------------------------------------------------
    # 9. Generate all figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")

    fig1_loss_components(pinn_curves)
    fig2_mae_inDist(mae_mlp_id,  mae_pinn_id,  out_cols)
    fig3_mae_ood(mae_mlp_ood, mae_pinn_ood, out_cols)
    fig4_full_comparison(mae_mlp_id, mae_mlp_ood,
                         mae_pinn_id, mae_pinn_ood, out_cols)
    fig5_physics_residual(res_mlp_id, res_mlp_ood,
                          res_pinn_id, res_pinn_ood)
    fig6_error_vs_loading(mlp, pinn, df, in_cols, out_cols,
                          scaler_X, scaler_y, device)

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("All PINN figures saved:")
    for f in sorted(FIGURES_DIR.glob("pinn_*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {size_kb:>7.1f} KB  {f.name}")


if __name__ == "__main__":
    main()