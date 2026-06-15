"""
train_pinn_surrogate.py
------------------------
Train the PINN surrogate for AC power flow.

Same architecture as the pure MLP but trained with:
    L_total = L_data + λ * L_physics

Saves
-----
models/instructor/power_flow_pinn_pretrained.pt
models/starting_points/power_flow_pinn_starting_point.pt

Usage
-----
python scripts/train_pinn_surrogate.py

Requires:
  data/workshop/power_flow_samples.csv
  data/workshop/power_flow_meta.json
  models/instructor/power_flow_scalers.pkl   (reuse from MLP training)
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workshop_utils.seeds    import set_seed
from workshop_utils.model_io import save_checkpoint
from workshop_utils.power_flow import build_network
from workshop_utils.pinn     import build_ybus, pinn_loss
from sklearn.preprocessing import StandardScaler

import pandapower as pp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH        = ROOT / "data"   / "workshop" / "power_flow_samples.csv"
META_PATH        = ROOT / "data"   / "workshop" / "power_flow_meta.json"
SCALERS_PATH     = ROOT / "models" / "instructor" / "power_flow_scalers.pkl"
MODEL_INSTRUCTOR = ROOT / "models" / "instructor"      / "power_flow_pinn_pretrained.pt"
MODEL_START      = ROOT / "models" / "starting_points" / "power_flow_pinn_starting_point.pt"
CURVES_PATH      = ROOT / "models" / "instructor"      / "power_flow_pinn_training_curves.json"

# ---------------------------------------------------------------------------
# Hyperparameters  (same architecture as pure MLP for fair comparison)
# ---------------------------------------------------------------------------
INPUT_DIM    = 8
OUTPUT_DIM   = 8
HIDDEN_DIMS  = [128, 128, 64]
BATCH_SIZE   = 256
LR           = 1e-3
LAMBDA_PHYS  = 0.1      # physics loss weight
FULL_EPOCHS  = 200
START_EPOCHS = 20
VAL_FRAC     = 0.15
TEST_FRAC    = 0.10


# ---------------------------------------------------------------------------
# Model  (identical architecture to pure MLP — fair comparison)
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
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path, meta_path, scalers_path):
    """
    Load CSV and reuse the SAME scalers as the pure MLP.
    This is critical — both models must operate in the same scaled space
    so that results are directly comparable.

    Returns DataLoaders with BOTH scaled inputs (for model) and
    raw inputs (for physics loss — physics equations need physical units).
    """
    df = pd.read_csv(data_path)
    with open(meta_path) as f:
        meta = json.load(f)
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    scaler_X = scalers["scaler_X"]
    scaler_y = scalers["scaler_y"]

    in_cols  = meta["input_features"]
    out_cols = meta["output_features"]

    df_train_full = df[df["split"] == "train"].drop(columns=["split"])
    df_ood        = df[df["split"] == "ood_test"].drop(columns=["split"])

    X_all_raw = df_train_full[in_cols].values.astype(np.float32)
    y_all_raw = df_train_full[out_cols].values.astype(np.float32)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X_all_raw, y_all_raw, test_size=TEST_FRAC, random_state=469
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=VAL_FRAC / (1.0 - TEST_FRAC),
        random_state=469,
    )

    def make_loader(X_raw, y_raw, shuffle=False):
        # Scaled inputs/outputs for model forward pass and data loss
        Xs = torch.tensor(scaler_X.transform(X_raw))
        ys = torch.tensor(scaler_y.transform(y_raw))
        # Raw inputs for physics loss (physical units MW/MVAr)
        Xr = torch.tensor(X_raw)
        return DataLoader(
            TensorDataset(Xs, ys, Xr),
            batch_size=BATCH_SIZE, shuffle=shuffle
        )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)

    X_ood_raw = df_ood[in_cols].values.astype(np.float32)
    y_ood_raw = df_ood[out_cols].values.astype(np.float32)
    ood_loader = make_loader(X_ood_raw, y_ood_raw)

    print(f"  Train : {len(X_train)} | Val : {len(X_val)} | "
          f"Test : {len(X_test)} | OOD : {len(X_ood_raw)}")

    return (train_loader, val_loader, test_loader, ood_loader,
            scaler_X, scaler_y, meta)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, n_epochs, device, G, B, base_mva, scaler_y):
    """
    PINN training loop.
    Each batch provides (X_scaled, y_scaled, X_raw).
    Physics loss uses X_raw (physical units) and y_pred (unscaled back).

    Note: physics loss is computed in physical units because the Y-bus
    equations need actual MW/MVAr/pu values, not normalised ones.
    We inverse-transform y_pred inside the loop before passing to physics_loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    train_total, train_data, train_phys = [], [], []
    val_total,   val_data,   val_phys   = [], [], []

    for epoch in range(1, n_epochs + 1):

        # Train
        model.train()
        bt, bd, bp = [], [], []
        for Xs, ys, Xr in train_loader:
            Xs, ys, Xr = Xs.to(device), ys.to(device), Xr.to(device)
            optimizer.zero_grad()
            y_pred_scaled = model(Xs)

            loss, l_d, l_p = pinn_loss(
                y_pred_scaled, ys, Xr, G, B, scaler_y, base_mva, lam=LAMBDA_PHYS
            )
            loss.backward()
            optimizer.step()
            bt.append(loss.item())
            bd.append(l_d.item())
            bp.append(l_p.item())

        train_total.append(np.mean(bt))
        train_data.append(np.mean(bd))
        train_phys.append(np.mean(bp))

        # Validate
        model.eval()
        vt, vd, vp = [], [], []
        with torch.no_grad():
            for Xs, ys, Xr in val_loader:
                Xs, ys, Xr = Xs.to(device), ys.to(device), Xr.to(device)
                y_pred_scaled = model(Xs)
                loss, l_d, l_p = pinn_loss(
                    y_pred_scaled, ys, Xr, G, B, scaler_y, base_mva, lam=LAMBDA_PHYS
                )
                vt.append(loss.item())
                vd.append(l_d.item())
                vp.append(l_p.item())

        val_total.append(np.mean(vt))
        val_data.append(np.mean(vd))
        val_phys.append(np.mean(vp))

        scheduler.step(np.mean(vt))

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{n_epochs} | "
                  f"Total: {train_total[-1]:.5f} | "
                  f"Data: {train_data[-1]:.5f} | "
                  f"Physics: {train_phys[-1]:.5f} | "
                  f"Val: {val_total[-1]:.5f}")

    return {
        "train_total" : train_total,
        "train_data"  : train_data,
        "train_phys"  : train_phys,
        "val_total"   : val_total,
        "val_data"    : val_data,
        "val_phys"    : val_phys,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, scaler_y, feature_names, label=""):
    """MAE per output feature in physical units."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for Xs, ys, _ in loader:
            preds.append(model(Xs.to(device)).cpu().numpy())
            trues.append(ys.numpy())

    pred = scaler_y.inverse_transform(np.vstack(preds))
    true = scaler_y.inverse_transform(np.vstack(trues))
    mae  = np.abs(pred - true).mean(axis=0)

    print(f"\n  MAE — {label}:")
    for name, e in zip(feature_names, mae):
        print(f"    {name:<22}: {e:.6f}")
    return mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(469)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # Check data exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run: python scripts/generate_power_flow_data.py"
        )

    # ------------------------------------------------------------------
    # 1. Build Y-bus from network
    # ------------------------------------------------------------------
    print("Building network and extracting Y-bus...")
    net = build_network()
    pp.runpp(net, numba=False)
    G, B, base_mva = build_ybus(net)
    G, B = G.to(device), B.to(device)
    print(f"  Y-bus shape : {G.shape}")
    print(f"  base_mva    : {base_mva}\n")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    (train_loader, val_loader, test_loader,
     ood_loader, scaler_X, scaler_y, meta) = load_data(
        DATA_PATH, META_PATH, SCALERS_PATH
    )
    out_names = meta["output_features"]

    # ------------------------------------------------------------------
    # Phase 1 — 20 epochs → starting-point model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training PINN starting-point ({START_EPOCHS} epochs)...")
    print(f"         λ = {LAMBDA_PHYS}  (data + physics loss)\n")

    model = PowerFlowMLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS).to(device)
    curves = train(model, train_loader, val_loader,
                   START_EPOCHS, device, G, B, base_mva, scaler_y)

    evaluate(model, test_loader, device, scaler_y, out_names,
             label="PINN starting-point, in-distribution")

    save_checkpoint(model, MODEL_START, metadata={
        "epochs_trained"  : START_EPOCHS,
        "final_val_loss"  : curves["val_total"][-1],
        "architecture"    : HIDDEN_DIMS,
        "lambda_physics"  : LAMBDA_PHYS,
        "input_features"  : meta["input_features"],
        "output_features" : meta["output_features"],
        "note"            : "PINN partially trained — participants fine-tune",
    })

    # ------------------------------------------------------------------
    # Phase 2 — continue to 200 epochs → instructor model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    remaining = FULL_EPOCHS - START_EPOCHS
    print(f"PHASE 2: Continuing PINN to full model ({remaining} more epochs)...\n")

    more = train(model, train_loader, val_loader,
                 remaining, device, G, B, base_mva, scaler_y)

    # Combine curves
    all_curves = {k: curves[k] + more[k] for k in curves}

    print("\nFull PINN model evaluation:")
    mae_test = evaluate(model, test_loader, device, scaler_y, out_names,
                        label="PINN instructor, in-distribution")
    mae_ood  = evaluate(model, ood_loader,  device, scaler_y, out_names,
                        label="PINN instructor, OOD (±40% load)")

    save_checkpoint(model, MODEL_INSTRUCTOR, metadata={
        "epochs_trained"  : FULL_EPOCHS,
        "final_val_loss"  : all_curves["val_total"][-1],
        "architecture"    : HIDDEN_DIMS,
        "lambda_physics"  : LAMBDA_PHYS,
        "input_features"  : meta["input_features"],
        "output_features" : meta["output_features"],
        "note"            : "Fully trained PINN instructor model",
    })

    # ------------------------------------------------------------------
    # Save training curves
    # ------------------------------------------------------------------
    with open(CURVES_PATH, "w") as f:
        json.dump(all_curves, f)
    print(f"\nPINN training curves saved → {CURVES_PATH}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Files saved:")
    for p in [MODEL_INSTRUCTOR, MODEL_START, CURVES_PATH]:
        size_kb = p.stat().st_size / 1024
        print(f"  {size_kb:>7.1f} KB  {p.relative_to(ROOT)}")

    print(f"\nλ used : {LAMBDA_PHYS}")
    print("Next step: run scripts/generate_pinn_figures.py to compare PINN vs MLP")


if __name__ == "__main__":
    main()