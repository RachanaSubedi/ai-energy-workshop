"""
power_flow.py
-------------
Power-system network building and supervised-learning data generation
using pandapower AC power flow.

Functions
---------
build_network()        — 5-bus radial distribution feeder
generate_samples()     — randomly perturbs loads and solar (P and Q),
                         runs AC power flow, returns (X, y) arrays
samples_to_dataframe() — converts arrays to a labelled pandas DataFrame
compute_line_flows()   — derives line flows from predicted complex voltages

Input / Output structure (Option A)
-------------------------------------
  Inputs  (8 features — all known quantities):
      P_load1, Q_load1   — active and reactive load at Bus 1 (PQ)
      P_load2, Q_load2   — active and reactive load at Bus 2 (PQ)
      P_load4, Q_load4   — active and reactive load at Bus 4 (PQ)
      P_solar, Q_solar   — active and reactive injection at Bus 3 (sgen)

  Outputs (8 features — solved by AC power flow):
      Vm_bus1..4         — voltage magnitudes at non-slack buses (pu)
      Va_bus1..4         — voltage angles   at non-slack buses (degrees)

  Bus 0 is the slack bus — voltage fixed at 1.0 pu / 0 deg, never predicted.

  Line flows are derived from predicted complex voltages after the fact
  using compute_line_flows() — no need to predict them directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandapower as pp
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. Network builder
# ---------------------------------------------------------------------------

def build_network() -> pp.pandapowerNet:
    """
    Build a 5-bus radial distribution feeder.

    Topology
    --------
    Bus 0 (slack)  — external grid / substation, |V|=1.0 pu fixed
      └─ Bus 1 (PQ load)
           └─ Bus 2 (PQ load)
                ├─ Bus 3 (solar PV — sgen, P and Q both controllable)
                └─ Bus 4 (PQ load)

    Nominal voltage : 20 kV
    Line parameters : typical MV overhead / cable mix
    """
    net = pp.create_empty_network()

    # Buses
    b0 = pp.create_bus(net, vn_kv=20.0, name="Bus 0 (Slack)")
    b1 = pp.create_bus(net, vn_kv=20.0, name="Bus 1 (PQ)")
    b2 = pp.create_bus(net, vn_kv=20.0, name="Bus 2 (PQ)")
    b3 = pp.create_bus(net, vn_kv=20.0, name="Bus 3 (Solar)")
    b4 = pp.create_bus(net, vn_kv=20.0, name="Bus 4 (PQ)")

    # External grid at slack bus
    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, name="Grid")

    # Lines
    line_params = dict(
        r_ohm_per_km=0.642,
        x_ohm_per_km=0.083,
        c_nf_per_km=210,
        max_i_ka=0.142,
    )
    pp.create_line_from_parameters(net, from_bus=b0, to_bus=b1,
                                   length_km=1.0, name="Line 0-1", **line_params)
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2,
                                   length_km=1.2, name="Line 1-2", **line_params)
    pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3,
                                   length_km=0.8, name="Line 2-3", **line_params)
    pp.create_line_from_parameters(net, from_bus=b2, to_bus=b4,
                                   length_km=1.5, name="Line 2-4", **line_params)

    # Nominal loads at PQ buses
    pp.create_load(net, bus=b1, p_mw=1.5, q_mvar=0.5, name="Load 1")
    pp.create_load(net, bus=b2, p_mw=2.0, q_mvar=0.7, name="Load 2")
    pp.create_load(net, bus=b4, p_mw=1.0, q_mvar=0.3, name="Load 4")

    # Solar PV as static generator (sgen)
    # Both p_mw and q_mvar are perturbed independently during data generation
    pp.create_sgen(net, bus=b3, p_mw=1.2, q_mvar=0.0, name="Solar PV")

    return net


# ---------------------------------------------------------------------------
# 2. Sample generator
# ---------------------------------------------------------------------------

def generate_samples(
    net: pp.pandapowerNet,
    n_samples: int = 5000,
    load_range: tuple[float, float] = (0.8, 1.2),
    solar_p_range: tuple[float, float] = (0.3, 1.5),
    solar_q_range: tuple[float, float] = (-0.3, 0.3),
    seed: int = 469,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate supervised-learning pairs via random AC power-flow simulation.

    For each sample:
      1. Randomly scale loads (P and Q together) within load_range.
      2. Randomly sample solar P within solar_p_range.
      3. Randomly sample solar Q within solar_q_range.
      4. Run AC power flow.
      5. Record inputs (X) and outputs (y) if converged.

    Parameters
    ----------
    net           : pandapowerNet  — base network from build_network()
    n_samples     : int            — number of operating points to attempt
    load_range    : (lo, hi)       — multiplicative range for load scaling
    solar_p_range : (lo, hi)       — absolute MW range for solar P injection
    solar_q_range : (lo, hi)       — absolute MVAr range for solar Q injection
    seed          : int            — numpy random seed

    Returns
    -------
    X    : np.ndarray (n_valid, 8)  — input features
    y    : np.ndarray (n_valid, 8)  — output features
    meta : dict                     — feature names, dims, generation config
    """
    rng = np.random.default_rng(seed)

    # Nominal load values from the network
    nom_loads = net.load[["p_mw", "q_mvar"]].values.copy()  # shape (3, 2)

    X_list, y_list = [], []
    n_failed = 0

    for _ in tqdm(range(n_samples), desc="Simulating power flow"):

        # Perturb loads (same scale factor for P and Q at each bus)
        scale = rng.uniform(*load_range, size=len(nom_loads))
        net.load["p_mw"]   = nom_loads[:, 0] * scale
        net.load["q_mvar"] = nom_loads[:, 1] * scale

        # Perturb solar P and Q independently
        p_solar = rng.uniform(*solar_p_range)
        q_solar = rng.uniform(*solar_q_range)
        net.sgen.loc[0, "p_mw"]   = p_solar
        net.sgen.loc[0, "q_mvar"] = q_solar

        # Run AC power flow
        try:
            pp.runpp(net, algorithm="nr", numba=False, verbose=False)
        except Exception:
            n_failed += 1
            continue

        if not net.converged:
            n_failed += 1
            continue

        # Build input vector (8 features)
        p1 = float(net.load.loc[0, "p_mw"]);   q1 = float(net.load.loc[0, "q_mvar"])
        p2 = float(net.load.loc[1, "p_mw"]);   q2 = float(net.load.loc[1, "q_mvar"])
        p4 = float(net.load.loc[2, "p_mw"]);   q4 = float(net.load.loc[2, "q_mvar"])
        x = np.array([p1, q1, p2, q2, p4, q4, p_solar, q_solar],
                     dtype=np.float32)

        # Build output vector (8 features)
        vm = net.res_bus["vm_pu"].values     # voltage magnitudes, all buses
        va = net.res_bus["va_degree"].values  # voltage angles, all buses
        # Bus 0 is slack (fixed) — exclude it; keep buses 1-4
        y = np.array([
            vm[1], vm[2], vm[3], vm[4],   # |V| at buses 1-4
            va[1], va[2], va[3], va[4],   # δ  at buses 1-4
        ], dtype=np.float32)

        X_list.append(x)
        y_list.append(y)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(f"\nGenerated {len(X)} valid samples  ({n_failed} failed to converge).")

    meta = {
        "n_samples"       : len(X),
        "n_failed"        : n_failed,
        "load_range"      : list(load_range),
        "solar_p_range"   : list(solar_p_range),
        "solar_q_range"   : list(solar_q_range),
        "input_features"  : [
            "P_load1_MW", "Q_load1_MVAr",
            "P_load2_MW", "Q_load2_MVAr",
            "P_load4_MW", "Q_load4_MVAr",
            "P_solar_MW", "Q_solar_MVAr",
        ],
        "output_features" : [
            "Vm_bus1_pu", "Vm_bus2_pu", "Vm_bus3_pu", "Vm_bus4_pu",
            "Va_bus1_deg", "Va_bus2_deg", "Va_bus3_deg", "Va_bus4_deg",
        ],
        "input_dim"  : X.shape[1],   # 8
        "output_dim" : y.shape[1],   # 8
    }

    return X, y, meta


# ---------------------------------------------------------------------------
# 3. Convert to DataFrame
# ---------------------------------------------------------------------------

def samples_to_dataframe(
    X: np.ndarray,
    y: np.ndarray,
    meta: dict,
) -> pd.DataFrame:
    """
    Combine X and y into a single labelled DataFrame ready to save as CSV.

    Parameters
    ----------
    X    : (n_samples, 8) input array
    y    : (n_samples, 8) output array
    meta : dict from generate_samples()

    Returns
    -------
    pd.DataFrame with columns = input_features + output_features
    """
    cols = meta["input_features"] + meta["output_features"]
    return pd.DataFrame(np.hstack([X, y]), columns=cols)


# ---------------------------------------------------------------------------
# 4. Derive line flows from predicted voltages
# ---------------------------------------------------------------------------

def compute_line_flows(
    net: pp.pandapowerNet,
    vm_all_pu: np.ndarray,
    va_all_deg: np.ndarray,
) -> dict:
    """
    Derive branch active/reactive power flows from predicted complex voltages.

    This is pure network math — no ML needed. Once you have complex bus
    voltages V_i = |V_i| * exp(j * delta_i), the current on branch i→k is:

        I_ik = (V_i - V_k) / Z_ik

    and the apparent power injected at bus i into the branch is:

        S_ik = V_i * conj(I_ik)

    Parameters
    ----------
    net         : pandapowerNet — the same network used for data generation
    vm_all_pu   : (n_buses,) array — voltage magnitudes in pu, ALL buses
                  including slack (index 0 = 1.0 pu)
    va_all_deg  : (n_buses,) array — voltage angles in degrees, ALL buses
                  including slack (index 0 = 0.0 deg)

    Returns
    -------
    dict:
        "P_flow_MW"    : active power flow per branch (MW)
        "Q_flow_MVAr"  : reactive power flow per branch (MVAr)
        "branch_names" : list of branch names
    """
    va_rad = np.deg2rad(va_all_deg)
    V = vm_all_pu * np.exp(1j * va_rad)   # complex voltage at each bus

    base_mva = net.sn_mva   # system base (default 1 MVA in pandapower)

    P_flows, Q_flows, names = [], [], []

    for _, line in net.line.iterrows():
        fb = int(line["from_bus"])
        tb = int(line["to_bus"])

        # Convert line impedance from physical Ohms to per-unit
        vn_kv  = net.bus.loc[fb, "vn_kv"]
        z_base = vn_kv ** 2 / base_mva          # Ohm
        r_pu   = line["r_ohm_per_km"] * line["length_km"] / z_base
        x_pu   = line["x_ohm_per_km"] * line["length_km"] / z_base
        z_pu   = complex(r_pu, x_pu)

        I_pu = (V[fb] - V[tb]) / z_pu
        S_pu = V[fb] * np.conj(I_pu)            # complex power (base = sn_mva)

        P_flows.append(S_pu.real * base_mva)
        Q_flows.append(S_pu.imag * base_mva)
        names.append(line["name"])

    return {
        "P_flow_MW"    : np.array(P_flows),
        "Q_flow_MVAr"  : np.array(Q_flows),
        "branch_names" : names,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("Step 1 — Build network and run base-case power flow")
    print("=" * 55)
    net = build_network()
    pp.runpp(net, numba=False)
    print("Converged:", net.converged)
    print(net.res_bus[["vm_pu", "va_degree"]].to_string())

    print("\n" + "=" * 55)
    print("Step 2 — Generate 200 samples (quick sanity check)")
    print("=" * 55)
    X, y, meta = generate_samples(
        net,
        n_samples=200,
        load_range=(0.8, 1.2),
        solar_p_range=(0.3, 1.5),
        solar_q_range=(-0.3, 0.3),
        seed=469,
    )
    print(f"X shape : {X.shape}   (expected: (n_valid, 8))")
    print(f"y shape : {y.shape}   (expected: (n_valid, 8))")
    print(f"Input features  : {meta['input_features']}")
    print(f"Output features : {meta['output_features']}")

    print("\nInput ranges (should all be non-zero and varied):")
    for i, name in enumerate(meta["input_features"]):
        print(f"  {name:<22}: min={X[:,i].min():.4f}  max={X[:,i].max():.4f}")

    print("\nOutput ranges:")
    for i, name in enumerate(meta["output_features"]):
        print(f"  {name:<22}: min={y[:,i].min():.4f}  max={y[:,i].max():.4f}")

    print("\n" + "=" * 55)
    print("Step 3 — DataFrame conversion")
    print("=" * 55)
    df = samples_to_dataframe(X, y, meta)
    print(df.head(3).to_string())

    print("\n" + "=" * 55)
    print("Step 4 — Line flow derivation from sample 0")
    print("=" * 55)
    vm_pred = np.concatenate([[1.0], y[0, :4]])   # prepend slack bus
    va_pred = np.concatenate([[0.0], y[0, 4:8]])
    flows   = compute_line_flows(net, vm_pred, va_pred)
    for name, p, q in zip(flows["branch_names"],
                           flows["P_flow_MW"],
                           flows["Q_flow_MVAr"]):
        print(f"  {name}: P = {p:.3f} MW,  Q = {q:.3f} MVAr")

    print("\npower_flow.py self-test PASSED.")