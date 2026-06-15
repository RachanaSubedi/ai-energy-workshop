"""
train_power_flow_surrogate.py
------------------------------
Train the MLP surrogate for AC power-flow mapping.

Saves
-----
models/instructor/power_flow_pretrained.pt          — fully trained (200 epochs)
models/starting_points/power_flow_starting_point.pt — partially trained (20 epochs)
models/instructor/power_flow_scalers.pkl            — fitted StandardScalers
models/instructor/power_flow_training_curves.json   — loss history for notebook plot

Input / Output dimensions (Option A)
--------------------------------------
  INPUT_DIM  = 8   [P_load1, Q_load1, P_load2, Q_load2,
                     P_load4, Q_load4, P_solar, Q_solar]
  OUTPUT_DIM = 8   [Vm_bus1..4, Va_bus1..4]

Usage
-----
python scripts/train_power_flow_surrogate.py

Run generate_power_flow_data.py first if the CSV does not exist.
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workshop_utils.seeds import set_seed
from workshop_utils.model_io import save_checkpoint

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH        = ROOT / "data"   / "workshop" / "power_flow_samples.csv"
META_PATH        = ROOT / "data"   / "workshop" / "power_flow_meta.json"
MODEL_INSTRUCTOR = ROOT / "models" / "instructor"     / "power_flow_pretrained.pt"
MODEL_START      = ROOT / "models" / "starting_points"/ "power_flow_starting_point.pt"
SCALERS_PATH     = ROOT / "models" / "instructor"     / "power_flow_scalers.pkl"
CURVES_PATH      = ROOT / "models" / "instructor"     / "power_flow_training_curves.json"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
INPUT_DIM    = 8          # 8 input features  (Option A)
OUTPUT_DIM   = 8          # 8 output features (Option A)
HIDDEN_DIMS  = [128, 128, 64]
BATCH_SIZE   = 256
LR           = 1e-3
FULL_EPOCHS  = 200        # instructor model
START_EPOCHS = 20         # starting-point model (participants fine-tune)
VAL_FRAC     = 0.15
TEST_FRAC    = 0.10


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PowerFlowMLP(nn.Module):
    """
    Feedforward MLP surrogate for AC power flow.

    Architecture:  Input → [Linear → BatchNorm → ReLU] × N → Linear → Output

    BatchNorm stabilises training when input features (MW, MVAr) and output
    features (pu voltages, degrees) are on very different scales.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list):
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

def load_data(data_path: Path, meta_path: Path):
    """
    Read CSV, split into train/val/test/ood, fit scalers on train only.

    Returns DataLoaders and the fitted scalers.
    """
    df = pd.read_csv(data_path)

    with open(meta_path) as f:
        meta = json.load(f)

    in_cols  = meta["input_features"]    # 8 columns
    out_cols = meta["output_features"]   # 8 columns

    # Use only in-distribution rows for training
    df_train_full = df[df["split"] == "train"].drop(columns=["split"])
    df_ood        = df[df["split"] == "ood_test"].drop(columns=["split"])

    X_all = df_train_full[in_cols].values.astype(np.float32)
    y_all = df_train_full[out_cols].values.astype(np.float32)

    # Train / val / test split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_all, y_all, test_size=TEST_FRAC, random_state=469
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=VAL_FRAC / (1.0 - TEST_FRAC),
        random_state=469,
    )

    # Fit scalers on training set ONLY — never on val, test, or OOD
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    def make_loader(X, y, shuffle=False):
        Xt = torch.tensor(scaler_X.transform(X), dtype=torch.float32)
        yt = torch.tensor(scaler_y.transform(y), dtype=torch.float32)
        return DataLoader(TensorDataset(Xt, yt),
                          batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)

    X_ood = df_ood[in_cols].values.astype(np.float32)
    y_ood = df_ood[out_cols].values.astype(np.float32)
    ood_loader = make_loader(X_ood, y_ood)

    print(f"  Train : {len(X_train)} | Val : {len(X_val)} | "
          f"Test : {len(X_test)} | OOD : {len(X_ood)}")

    return train_loader, val_loader, test_loader, ood_loader, scaler_X, scaler_y, meta


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, n_epochs, device):
    """MSE training loop. Returns train and val loss lists."""
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(1, n_epochs + 1):

        # Train
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        t_loss = float(np.mean(batch_losses))
        train_losses.append(t_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            v_losses = [criterion(model(xb.to(device)), yb.to(device)).item()
                        for xb, yb in val_loader]
        v_loss = float(np.mean(v_losses))
        val_losses.append(v_loss)

        scheduler.step(v_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d}/{n_epochs} | "
                  f"Train MSE: {t_loss:.5f} | Val MSE: {v_loss:.5f}")

    return train_losses, val_losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, scaler_y, feature_names, label=""):
    """Compute MAE per output feature in original physical units."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy())

    pred = scaler_y.inverse_transform(np.vstack(preds))
    true = scaler_y.inverse_transform(np.vstack(trues))
    mae  = np.abs(pred - true).mean(axis=0)

    header = f"MAE per output feature — {label}" if label else "MAE per output feature"
    print(f"\n  {header}:")
    for name, e in zip(feature_names, mae):
        print(f"    {name:<22}: {e:.6f}")
    return mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(469)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Check data exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run:  python scripts/generate_power_flow_data.py"
        )

    # Load data
    print("Loading data...")
    (train_loader, val_loader, test_loader,
     ood_loader, scaler_X, scaler_y, meta) = load_data(DATA_PATH, META_PATH)

    out_names = meta["output_features"]

    # Verify dimensions match expectations
    assert meta["input_dim"]  == INPUT_DIM,  \
        f"Expected INPUT_DIM={INPUT_DIM}, got {meta['input_dim']}"
    assert meta["output_dim"] == OUTPUT_DIM, \
        f"Expected OUTPUT_DIM={OUTPUT_DIM}, got {meta['output_dim']}"

    # -----------------------------------------------------------------------
    # Phase 1 — train 20 epochs → starting-point model
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training starting-point model ({START_EPOCHS} epochs)...")
    model = PowerFlowMLP(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIMS).to(device)
    train_losses, val_losses = train(
        model, train_loader, val_loader, START_EPOCHS, device
    )

    evaluate(model, test_loader, device, scaler_y, out_names,
             label="starting-point, in-distribution test")

    save_checkpoint(model, MODEL_START, metadata={
        "epochs_trained"        : START_EPOCHS,
        "final_val_loss"        : val_losses[-1],
        "architecture"          : HIDDEN_DIMS,
        "input_features"        : meta["input_features"],
        "output_features"       : meta["output_features"],
        "input_dim"             : INPUT_DIM,
        "output_dim"            : OUTPUT_DIM,
        "load_range"            : meta["load_range"],
        "solar_p_range"         : meta["solar_p_range"],
        "solar_q_range"         : meta["solar_q_range"],
        "note"                  : "Partially trained — participants fine-tune during workshop",
    })

    # -----------------------------------------------------------------------
    # Phase 2 — continue training to 200 epochs → instructor model
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    remaining = FULL_EPOCHS - START_EPOCHS
    print(f"PHASE 2: Continuing to full instructor model ({remaining} more epochs)...")
    more_train, more_val = train(
        model, train_loader, val_loader, remaining, device
    )

    all_train = train_losses + more_train
    all_val   = val_losses   + more_val

    evaluate(model, test_loader, device, scaler_y, out_names,
             label="instructor, in-distribution test")
    evaluate(model, ood_loader, device, scaler_y, out_names,
             label="instructor, OOD test (±40% load)")

    save_checkpoint(model, MODEL_INSTRUCTOR, metadata={
        "epochs_trained"        : FULL_EPOCHS,
        "final_train_loss"      : all_train[-1],
        "final_val_loss"        : all_val[-1],
        "architecture"          : HIDDEN_DIMS,
        "input_features"        : meta["input_features"],
        "output_features"       : meta["output_features"],
        "input_dim"             : INPUT_DIM,
        "output_dim"            : OUTPUT_DIM,
        "load_range"            : meta["load_range"],
        "solar_p_range"         : meta["solar_p_range"],
        "solar_q_range"         : meta["solar_q_range"],
        "note"                  : "Fully trained instructor model",
    })

    # -----------------------------------------------------------------------
    # Save scalers
    # -----------------------------------------------------------------------
    SCALERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALERS_PATH, "wb") as f:
        pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)
    print(f"\nScalers saved      → {SCALERS_PATH}")

    # -----------------------------------------------------------------------
    # Save training curves
    # -----------------------------------------------------------------------
    with open(CURVES_PATH, "w") as f:
        json.dump({"train_losses": all_train, "val_losses": all_val}, f)
    print(f"Training curves    → {CURVES_PATH}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training complete. Files saved:")
    for p in [MODEL_INSTRUCTOR, MODEL_START, SCALERS_PATH, CURVES_PATH]:
        size_kb = p.stat().st_size / 1024
        print(f"  {size_kb:>8.1f} KB  {p.relative_to(ROOT)}")
    print("\nNext step: upload .pt and .pkl files to Google Drive")
    print("and record the file IDs in models/README.md")


if __name__ == "__main__":
    main()