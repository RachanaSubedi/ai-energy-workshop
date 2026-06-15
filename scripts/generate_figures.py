"""
generate_figures.py
--------------------
Generate and save all workshop figures using the trained surrogate model
and the real power-flow dataset.

Figures saved
-------------
figures/fig1_vm_inDist.png        — True vs. predicted |V|, in-distribution
figures/fig2_va_inDist.png        — True vs. predicted δ,  in-distribution
figures/fig3_mae_inDist.png       — MAE bar chart, in-distribution
figures/fig4_training_curves.png  — Train/val loss over 200 epochs
figures/fig5_vm_ood.png           — True vs. predicted |V|, OOD test
figures/fig6_va_ood.png           — True vs. predicted δ,  OOD test
figures/fig7_mae_comparison.png   — In-dist vs. OOD MAE side-by-side
figures/fig8_error_vs_loading.png — Surrogate MAE vs. load scaling factor

Usage
-----
Run from the project root:
    python scripts/generate_figures.py
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
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"   / "workshop"
MODEL_DIR   = ROOT / "models" / "instructor"
FIGURES_DIR = ROOT / "figures"

sys.path.insert(0, str(ROOT / "src"))

from workshop_utils.seeds    import set_seed
from workshop_utils.model_io import load_checkpoint
from workshop_utils.plotting import (
    plot_true_vs_predicted,
    plot_error_by_bus,
    plot_training_curve,
    plot_error_vs_loading,
)


# ---------------------------------------------------------------------------
# Model definition  (must match train_power_flow_surrogate.py exactly)
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
# Helper: run model on a numpy array, return predictions in original units
# ---------------------------------------------------------------------------

def predict(model, X_raw, scaler_X, scaler_y, device):
    """Scale inputs → run model → inverse-scale outputs."""
    X_scaled = torch.tensor(
        scaler_X.transform(X_raw).astype("float32")
    ).to(device)
    with torch.no_grad():
        y_scaled = model(X_scaled).cpu().numpy()
    return scaler_y.inverse_transform(y_scaled)


# ---------------------------------------------------------------------------
# Figure 7 — In-dist vs. OOD MAE comparison bar chart
# ---------------------------------------------------------------------------

def plot_mae_comparison(mae_id, mae_ood, feature_names,
                        save_dir=None):
    """
    Grouped horizontal bar chart comparing in-distribution vs OOD MAE
    for every output feature.

    Blue  = in-distribution (training range ±20%)
    Red   = OOD             (wider range     ±40%)
    """
    display = {
        "Vm_bus1_pu"  : "|V| Bus 1 (pu)",
        "Vm_bus2_pu"  : "|V| Bus 2 (pu)",
        "Vm_bus3_pu"  : "|V| Bus 3 Solar (pu)",
        "Vm_bus4_pu"  : "|V| Bus 4 (pu)",
        "Va_bus1_deg" : "δ Bus 1 (deg)",
        "Va_bus2_deg" : "δ Bus 2 (deg)",
        "Va_bus3_deg" : "δ Bus 3 Solar (deg)",
        "Va_bus4_deg" : "δ Bus 4 (deg)",
    }
    labels = [display.get(f, f) for f in feature_names]
    n      = len(labels)
    y_pos  = np.arange(n)
    h      = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_id  = ax.barh(y_pos - h/2, mae_id,  height=h,
                       color="steelblue",  edgecolor="white",
                       label="In-distribution (load ±20%)")
    bars_ood = ax.barh(y_pos + h/2, mae_ood, height=h,
                       color="crimson",    edgecolor="white",
                       label="OOD test (load ±40%)")

    ax.bar_label(bars_id,  fmt="%.5f", padding=3, fontsize=8,
                 color="steelblue")
    ax.bar_label(bars_ood, fmt="%.5f", padding=3, fontsize=8,
                 color="crimson")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Absolute Error", fontsize=11)
    ax.set_title(
        "Surrogate MAE: In-Distribution vs. Out-of-Distribution\n"
        "OOD error reveals where the surrogate becomes unreliable",
        fontsize=12, pad=10
    )
    ax.set_xlim(0, max(mae_ood.max(), mae_id.max()) * 1.3)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", ls="--", alpha=0.4)

    # Draw a vertical line at the max in-dist MAE to highlight OOD degradation
    ax.axvline(mae_id.max(), color="steelblue", ls=":", lw=1.2,
               alpha=0.6, label="Max in-dist MAE")

    plt.tight_layout()

    if save_dir is not None:
        fpath = Path(save_dir) / "fig7_mae_comparison.png"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")

    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Figure 8 — Error vs. load scaling factor
# ---------------------------------------------------------------------------

def plot_error_vs_loading_real(model, df, in_cols, out_cols,
                                scaler_X, scaler_y, device,
                                save_dir=None):
    """
    Compute MAE of voltage magnitude predictions at each load scaling
    level across the full ±40% range. Uses OOD samples bucketed by
    their actual load scaling factor.

    Shows clearly where surrogate degrades outside training range.
    """
    # Nominal load values from the dataset
    # (recover approximate scale from the OOD samples)
    df_ood = df[df["split"] == "ood_test"].copy()

    # Compute average load scaling per sample
    # P_load1 nominal = 1.5, P_load2 = 2.0, P_load4 = 1.0
    nom_p = np.array([1.5, 2.0, 1.0])
    load_p = df_ood[["P_load1_MW", "P_load2_MW", "P_load4_MW"]].values
    scales = (load_p / nom_p).mean(axis=1)   # average scale across 3 buses

    # Bucket into 0.1-wide bins from 0.6 to 1.4
    bins       = np.arange(0.55, 1.45, 0.1)
    bin_labels = np.round(bins[:-1] + 0.05, 2)
    bin_idx    = np.digitize(scales, bins) - 1

    X_ood = df_ood[in_cols].values.astype("float32")
    y_ood = df_ood[out_cols].values.astype("float32")
    y_pred_ood = predict(model, X_ood, scaler_X, scaler_y, device)

    # MAE of voltage magnitudes only (first 4 outputs) per bin
    mae_vm = np.abs(y_ood[:, :4] - y_pred_ood[:, :4]).mean(axis=1)

    bin_mae, bin_count = [], []
    for b in range(len(bin_labels)):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_mae.append(mae_vm[mask].mean())
            bin_count.append(mask.sum())
        else:
            bin_mae.append(np.nan)
            bin_count.append(0)

    bin_mae    = np.array(bin_mae)
    bin_labels = np.array(bin_labels)

    # Remove empty bins
    valid      = ~np.isnan(bin_mae)
    bin_labels = bin_labels[valid]
    bin_mae    = bin_mae[valid]

    # Use the existing plot_error_vs_loading from plotting.py
    plot_error_vs_loading(
        load_scales = bin_labels,
        errors      = bin_mae,
        train_range = (0.8, 1.2),
        xlabel      = "Load scaling factor (pu)",
        ylabel      = "Mean Absolute Error — |V| (pu)",
        title       = "Surrogate Error vs. Load Level\n"
                      "Green band = training range  |  "
                      "Outside = OOD degradation",
    )

    if save_dir is not None:
        fpath = Path(save_dir) / "fig8_error_vs_loading.png"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")

    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(469)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Figures dir : {FIGURES_DIR}\n")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("Loading dataset...")
    df = pd.read_csv(DATA_DIR / "power_flow_samples.csv")
    with open(DATA_DIR / "power_flow_meta.json") as f:
        meta = json.load(f)

    in_cols  = meta["input_features"]
    out_cols = meta["output_features"]

    # In-distribution: 500 samples from training split
    df_id = (df[df["split"] == "train"]
             .sample(500, random_state=469)
             .reset_index(drop=True))
    X_id  = df_id[in_cols].values.astype("float32")
    y_id  = df_id[out_cols].values.astype("float32")

    # OOD: all 1000 samples from ood_test split
    df_ood = df[df["split"] == "ood_test"].reset_index(drop=True)
    X_ood  = df_ood[in_cols].values.astype("float32")
    y_ood  = df_ood[out_cols].values.astype("float32")

    print(f"  In-dist samples : {len(X_id)}")
    print(f"  OOD samples     : {len(X_ood)}\n")

    # ------------------------------------------------------------------
    # 2. Load scalers
    # ------------------------------------------------------------------
    print("Loading scalers...")
    with open(MODEL_DIR / "power_flow_scalers.pkl", "rb") as f:
        scalers  = pickle.load(f)
    scaler_X = scalers["scaler_X"]
    scaler_y = scalers["scaler_y"]

    # ------------------------------------------------------------------
    # 3. Load pretrained model
    # ------------------------------------------------------------------
    print("Loading pretrained model...")
    model = PowerFlowMLP().to(device)
    model, metadata = load_checkpoint(
        model,
        MODEL_DIR / "power_flow_pretrained.pt",
        device,
    )
    print(f"  Trained for    : {metadata.get('epochs_trained', '?')} epochs")
    print(f"  Final val loss : {metadata.get('final_val_loss', '?')}\n")

    # ------------------------------------------------------------------
    # 4. Run predictions on both splits
    # ------------------------------------------------------------------
    y_pred_id  = predict(model, X_id,  scaler_X, scaler_y, device)
    y_pred_ood = predict(model, X_ood, scaler_X, scaler_y, device)

    mae_id  = np.abs(y_id  - y_pred_id ).mean(axis=0)
    mae_ood = np.abs(y_ood - y_pred_ood).mean(axis=0)

    print("MAE summary:")
    print(f"  {'Feature':<22}  {'In-dist':>10}  {'OOD':>10}  {'OOD/InDist':>12}")
    print("  " + "-" * 58)
    for name, e_id, e_ood in zip(out_cols, mae_id, mae_ood):
        ratio = e_ood / e_id if e_id > 0 else float("inf")
        print(f"  {name:<22}  {e_id:>10.6f}  {e_ood:>10.6f}  {ratio:>10.1f}×")
    print()

    # ------------------------------------------------------------------
    # 5. Fig 1 + 2 — In-distribution True vs. Predicted
    # ------------------------------------------------------------------
    print("Generating Fig 1 & 2 — In-distribution True vs. Predicted...")
    plot_true_vs_predicted(
        y_id, y_pred_id,
        feature_names = out_cols,
        save_dir      = FIGURES_DIR,
        title         = "In-Distribution (load ±20%)",
    )
    # rename to clarify they are in-dist
    (FIGURES_DIR / "fig1_voltage_magnitudes.png").rename(
        FIGURES_DIR / "fig1_vm_inDist.png")
    (FIGURES_DIR / "fig2_voltage_angles.png").rename(
        FIGURES_DIR / "fig2_va_inDist.png")

    # ------------------------------------------------------------------
    # 6. Fig 3 — In-distribution MAE bar chart
    # ------------------------------------------------------------------
    print("Generating Fig 3 — In-distribution MAE by feature...")
    plot_error_by_bus(
        y_id, y_pred_id,
        feature_names = out_cols,
        save_dir      = FIGURES_DIR,
        title         = "In-Distribution MAE by Output Feature  (load ±20%)\n"
                        "(blue = voltage magnitude | orange = voltage angle)",
    )
    (FIGURES_DIR / "fig3_mae_by_feature.png").rename(
        FIGURES_DIR / "fig3_mae_inDist.png")

    # ------------------------------------------------------------------
    # 7. Fig 4 — Training curves
    # ------------------------------------------------------------------
    print("Generating Fig 4 — Training curves...")
    with open(MODEL_DIR / "power_flow_training_curves.json") as f:
        curves = json.load(f)

    plt.ioff()
    plot_training_curve(
        train_losses  = curves["train_losses"],
        val_losses    = curves["val_losses"],
        phase1_epochs = 20,
        title         = "Power Flow Surrogate — Training Loss",
    )
    fpath = FIGURES_DIR / "fig4_training_curves.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fpath}")

    # ------------------------------------------------------------------
    # 8. Fig 5 + 6 — OOD True vs. Predicted
    # ------------------------------------------------------------------
    print("\nGenerating Fig 5 & 6 — OOD True vs. Predicted...")
    plot_true_vs_predicted(
        y_ood, y_pred_ood,
        feature_names = out_cols,
        save_dir      = FIGURES_DIR,
        title         = "OOD Test (load ±40%) — Surrogate Generalisation",
    )
    (FIGURES_DIR / "fig1_voltage_magnitudes.png").rename(
        FIGURES_DIR / "fig5_vm_ood.png")
    (FIGURES_DIR / "fig2_voltage_angles.png").rename(
        FIGURES_DIR / "fig6_va_ood.png")

    # ------------------------------------------------------------------
    # 9. Fig 7 — In-dist vs. OOD MAE comparison
    # ------------------------------------------------------------------
    print("Generating Fig 7 — In-dist vs. OOD MAE comparison...")
    plot_mae_comparison(
        mae_id, mae_ood,
        feature_names = out_cols,
        save_dir      = FIGURES_DIR,
    )

    # ------------------------------------------------------------------
    # 10. Fig 8 — Error vs. load scaling factor
    # ------------------------------------------------------------------
    print("Generating Fig 8 — Error vs. load scaling factor...")
    plot_error_vs_loading_real(
        model, df, in_cols, out_cols,
        scaler_X, scaler_y, device,
        save_dir=FIGURES_DIR,
    )

    # ------------------------------------------------------------------
    # 11. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("All figures saved:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  {size_kb:>7.1f} KB  {f.name}")
    print(f"\nDirectory: {FIGURES_DIR}")

    # Key insight summary
    print("\n" + "=" * 60)
    print("KEY INSIGHT — OOD degradation ratio (OOD MAE / In-dist MAE):")
    for name, e_id, e_ood in zip(out_cols, mae_id, mae_ood):
        ratio = e_ood / e_id if e_id > 0 else float("inf")
        flag  = " ← significant degradation" if ratio > 5 else ""
        print(f"  {name:<22}: {ratio:>6.1f}×{flag}")


if __name__ == "__main__":
    main()