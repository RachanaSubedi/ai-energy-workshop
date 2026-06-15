"""
train_forecasting_model.py
==========================
Standalone training script for day-ahead solar irradiance forecasting.
Designed to run on Saidie HPC (DGX H200) at CSU.

Trains three models:
  1. Persistence baseline (no training needed)
  2. Random Forest
  3. PyTorch MLP with early stopping

Saves:
  models/instructor/forecasting_mlp.pt        — fully trained MLP
  models/starting_points/forecasting_mlp_start.pt — checkpoint at epoch 10
  models/instructor/forecasting_rf.pkl        — trained random forest
  outputs/training_results.json               — metrics summary

Usage
-----
On Saidie (interactive):
    source ~/envs/ai-energy-workshop/bin/activate
    cd ~/ai-energy-workshop
    python scripts/train_forecasting_model.py

Via SLURM:
    sbatch scripts/train_forecasting_model.slurm
"""

import sys
import json
import time
import random
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_PATH  = ROOT / "data" / "workshop" / "solar_forecasting.csv"
INST_DIR   = ROOT / "models" / "instructor"
START_DIR  = ROOT / "models" / "starting_points"
OUT_DIR    = ROOT / "outputs"

for d in [INST_DIR, START_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEED            = 469
BATCH_SIZE      = 256
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
EPOCHS          = 300       # more epochs than notebook — DGX is fast
PATIENCE        = 20
START_EPOCH     = 10        # save starting-point model at this epoch
HIDDEN          = 64
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH    = 14

FEATURE_COLS = [
    "sin_hour", "cos_hour", "sin_doy", "cos_doy",
    "temp_forecast", "cloud_forecast", "humidity_forecast", "precip_forecast",
    "clearsky_ghi",
    "solar_rad_t24", "solar_rad_t48",
]
TARGET_COL = "target_t24"


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Model ─────────────────────────────────────────────────────────────────────
class SolarMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),

            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, name="Model"):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (y_true.mean() + 1e-8) * 100
    return {"model": name, "mae": round(mae, 4),
            "rmse": round(rmse, 4), "nrmse": round(nrmse, 4)}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    print(f"\n[Data] Loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/Denver")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    n       = len(df)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    splits = {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_val"  : X[n_train:n_train+n_val],
        "y_val"  : y[n_train:n_train+n_val],
        "X_test" : X[n_train+n_val:],
        "y_test" : y[n_train+n_val:],
        "clearsky_test": df["clearsky_ghi"].values[n_train+n_val:],
    }

    print(f"  Train : {len(splits['X_train']):,} rows")
    print(f"  Val   : {len(splits['X_val']):,} rows")
    print(f"  Test  : {len(splits['X_test']):,} rows")
    return splits, df


# ── Main training ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",  type=int, default=EPOCHS)
    parser.add_argument("--patience",type=int, default=PATIENCE)
    parser.add_argument("--hidden",  type=int, default=HIDDEN)
    parser.add_argument("--seed",    type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Solar Forecasting — Training Script")
    print(f"  Device  : {device}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Seed    : {args.seed}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    splits, df = load_data()
    X_train = splits["X_train"];  y_train = splits["y_train"]
    X_val   = splits["X_val"];    y_val   = splits["y_val"]
    X_test  = splits["X_test"];   y_test  = splits["y_test"]
    cs_test = splits["clearsky_test"]

    # ── Persistence baseline ──────────────────────────────────────────────────
    print("\n[1/3] Persistence baseline")
    feat_idx  = FEATURE_COLS.index("solar_rad_t24")
    y_pers    = X_test[:, feat_idx]
    res_pers  = evaluate(y_test, y_pers, "Persistence")
    print(f"  RMSE = {res_pers['rmse']:.2f} W/m²")

    # ── Feature scaling ───────────────────────────────────────────────────────
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s   = scaler_X.transform(X_val)
    X_test_s  = scaler_X.transform(X_test)
    y_train_s = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
    y_val_s   = scaler_y.transform(y_val.reshape(-1,1)).ravel()

    # ── Random Forest ─────────────────────────────────────────────────────────
    print(f"\n[2/3] Random Forest (n={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH})")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=args.seed,
    )
    rf.fit(X_train, y_train)
    y_rf_test = rf.predict(X_test)
    res_rf    = evaluate(y_test, y_rf_test, "Random Forest")
    print(f"  RMSE = {res_rf['rmse']:.2f} W/m²  ({time.time()-t0:.1f}s)")

    rf_path = INST_DIR / "forecasting_rf.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"  Saved → {rf_path}")

    # ── MLP ───────────────────────────────────────────────────────────────────
    print(f"\n[3/3] MLP (hidden={args.hidden}, epochs={args.epochs})")

    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(1),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    train_loader = make_loader(X_train_s, y_train_s, shuffle=True)
    val_loader   = make_loader(X_val_s,   y_val_s)

    model     = SolarMLP(n_features=len(FEATURE_COLS), hidden=args.hidden).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=7, factor=0.5)

    best_val      = float("inf")
    best_state    = None
    no_improve    = 0
    start_saved   = False
    train_losses  = []
    val_losses    = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):

        # Train
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(Xb)
        train_loss = running / len(train_loader.dataset)

        # Validate
        model.eval()
        running = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                running += criterion(model(Xb), yb).item() * len(Xb)
        val_loss = running / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Save starting-point model
        if epoch == START_EPOCH and not start_saved:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, START_DIR / "forecasting_mlp_start.pt")
            print(f"  Starting-point model saved at epoch {epoch}")
            start_saved = True

        # Early stopping
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  patience={no_improve}/{args.patience}"
                  f"  ({elapsed:.0f}s)")

        if no_improve >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Restore best weights and evaluate
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        y_mlp_s  = model(X_test_t).cpu().numpy().ravel()

    y_mlp_test = scaler_y.inverse_transform(y_mlp_s.reshape(-1,1)).ravel()
    y_mlp_test = np.where(cs_test <= 0, 0,
                          np.clip(y_mlp_test, 0, None))

    res_mlp = evaluate(y_test, y_mlp_test, "MLP")
    print(f"\n  Best val loss : {best_val:.4f}")
    print(f"  Test RMSE     : {res_mlp['rmse']:.2f} W/m²")
    print(f"  Training time : {time.time()-t0:.1f}s")

    # ── Save instructor MLP ───────────────────────────────────────────────────
    mlp_path = INST_DIR / "forecasting_mlp.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_X_mean"   : scaler_X.mean_,
        "scaler_X_scale"  : scaler_X.scale_,
        "scaler_y_mean"   : scaler_y.mean_,
        "scaler_y_scale"  : scaler_y.scale_,
        "feature_cols"    : FEATURE_COLS,
        "train_losses"    : train_losses,
        "val_losses"      : val_losses,
        "test_rmse"       : res_mlp["rmse"],
        "metadata": {
            "location" : "Fort Collins CO",
            "period"   : "Jun 2025 - May 2026",
            "seed"     : args.seed,
            "hidden"   : args.hidden,
            "device"   : str(device),
        }
    }, mlp_path)
    print(f"  Saved → {mlp_path}")

    # ── Save scalers separately for notebook use ──────────────────────────────
    scaler_path = INST_DIR / "forecasting_scalers.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)
    print(f"  Saved → {scaler_path}")

    # ── Results summary ───────────────────────────────────────────────────────
    rmse_pers = res_pers["rmse"]
    for r in [res_pers, res_rf, res_mlp]:
        r["skill"] = round(1 - r["rmse"] / rmse_pers, 3)

    results = {
        "models"      : [res_pers, res_rf, res_mlp],
        "training_args": vars(args),
        "device"      : str(device),
    }

    results_path = OUT_DIR / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved → {results_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Final Results — Test Set")
    print("=" * 60)
    print(f"  {'Model':<15} {'RMSE':>8}  {'Skill':>6}")
    print(f"  {'-'*35}")
    for r in [res_pers, res_rf, res_mlp]:
        print(f"  {r['model']:<15} {r['rmse']:>8.2f}  {r['skill']:>6.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
