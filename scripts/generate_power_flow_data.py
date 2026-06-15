"""
generate_power_flow_data.py
---------------------------
Run once to produce the workshop dataset.

Output files
------------
data/workshop/power_flow_samples.csv   — 6000 simulated operating points
data/workshop/power_flow_meta.json     — feature names and generation config

Usage
-----
python scripts/generate_power_flow_data.py

This script is run by the developer (Student 2) only.
Workshop participants load the pre-generated CSV — they never run this.
"""

import sys
import json
from pathlib import Path

import pandas as pd
import pandapower as pp

# Allow imports from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workshop_utils.seeds import set_seed
from workshop_utils.power_flow import (
    build_network,
    generate_samples,
    samples_to_dataframe,
)


def main():
    set_seed(469)

    # ------------------------------------------------------------------
    # 1. Build network and verify base case
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Building 5-bus radial distribution network...")
    net = build_network()
    pp.runpp(net, numba=False)
    assert net.converged, "Base-case power flow did not converge!"
    print("Base-case power flow: CONVERGED")
    print(net.res_bus[["vm_pu", "va_degree"]].to_string())

    # ------------------------------------------------------------------
    # 2. Training samples — moderate perturbation range (±20% load)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating TRAINING samples...")
    print("  load_range    : (0.8, 1.2)   — ±20% of nominal")
    print("  solar_p_range : (0.3, 1.5)   — MW")
    print("  solar_q_range : (-0.3, 0.3)  — MVAr")

    X_train, y_train, meta = generate_samples(
        net,
        n_samples=5000,
        load_range=(0.8, 1.2),
        solar_p_range=(0.3, 1.5),
        solar_q_range=(-0.3, 0.3),
        seed=469,
    )
    df_train = samples_to_dataframe(X_train, y_train, meta)
    df_train["split"] = "train"

    # ------------------------------------------------------------------
    # 3. OOD test samples — wider perturbation range (±40% load)
    #    Used in the participant exercise to show where surrogate fails
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Generating OOD TEST samples...")
    print("  load_range    : (0.6, 1.4)   — ±40% of nominal")
    print("  solar_p_range : (0.3, 1.5)   — same as training")
    print("  solar_q_range : (-0.3, 0.3)  — same as training")

    X_ood, y_ood, _ = generate_samples(
        net,
        n_samples=1000,
        load_range=(0.6, 1.4),
        solar_p_range=(0.3, 1.5),
        solar_q_range=(-0.3, 0.3),
        seed=999,
    )
    df_ood = samples_to_dataframe(X_ood, y_ood, meta)
    df_ood["split"] = "ood_test"

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    out_dir = ROOT / "data" / "workshop"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combined CSV
    df_all = pd.concat([df_train, df_ood], ignore_index=True)
    csv_path = out_dir / "power_flow_samples.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nSaved CSV  : {csv_path}  ({len(df_all)} rows)")

    # Metadata JSON
    meta_path = out_dir / "power_flow_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved meta : {meta_path}")

    # ------------------------------------------------------------------
    # 5. Verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Dataset summary:")
    print(df_all.groupby("split").size().rename("count").to_string())

    print("\nInput feature ranges (verify Q_solar_MVAr is non-zero):")
    in_cols = meta["input_features"]
    for col in in_cols:
        print(f"  {col:<22}: "
              f"min={df_all[col].min():.4f}  "
              f"max={df_all[col].max():.4f}  "
              f"std={df_all[col].std():.4f}")

    print("\nOutput feature ranges:")
    out_cols = meta["output_features"]
    for col in out_cols:
        print(f"  {col:<22}: "
              f"min={df_all[col].min():.4f}  "
              f"max={df_all[col].max():.4f}  "
              f"std={df_all[col].std():.4f}")

    print("\nData generation complete.")


if __name__ == "__main__":
    main()