"""
pinn.py
-------
Physics-Informed Neural Network (PINN) loss functions for AC power flow.

Key design note
---------------
The model works in SCALED space (StandardScaler applied to inputs and outputs).
The physics equations (Y-bus power balance) work in PHYSICAL units.

Therefore pinn_loss() accepts a scaler_y argument and inverse-transforms
the model's scaled predictions back to physical units before computing
the physics residual. This is the correct approach.

PINN combined loss:
    L_total = L_data + λ * L_physics

    L_data    = MSE(y_pred_scaled, y_true_scaled)   — standard data loss
    L_physics = power balance residual using physical-unit predictions
    λ         = 0.1 (default) — physics weight
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import pandapower as pp
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1. Build G and B matrices from pandapower network
# ---------------------------------------------------------------------------

def build_ybus(net: pp.pandapowerNet) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Extract Y-bus admittance matrix from pandapower and return
    conductance (G) and susceptance (B) matrices as tensors.

    Must be called AFTER pp.runpp() so internal ppc structure exists.

    Returns
    -------
    G        : (n_bus, n_bus) conductance matrix
    B        : (n_bus, n_bus) susceptance matrix
    base_mva : float — system base (1.0 MVA default in pandapower)
    """
    from pandapower.pypower.makeYbus import makeYbus

    ppc        = net._ppc
    base_mva   = float(ppc["baseMVA"])
    Ybus, _, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    Ybus_dense = Ybus.toarray()

    G = torch.tensor(Ybus_dense.real, dtype=torch.float32)
    B = torch.tensor(Ybus_dense.imag, dtype=torch.float32)

    return G, B, base_mva


# ---------------------------------------------------------------------------
# 2. Physics loss — operates in physical units
# ---------------------------------------------------------------------------

def physics_loss(
    y_pred_physical : torch.Tensor,
    X_raw           : torch.Tensor,
    G               : torch.Tensor,
    B               : torch.Tensor,
    base_mva        : float = 1.0,
) -> torch.Tensor:
    """
    Compute AC power balance residual using PHYSICAL-UNIT predictions.

    For each bus i (non-slack):
        P_i = |V_i| * Σ_k |V_k| * (G_ik*cos(δi-δk) + B_ik*sin(δi-δk))
        Q_i = |V_i| * Σ_k |V_k| * (G_ik*sin(δi-δk) - B_ik*cos(δi-δk))

    The residual is: known_injection - calculated_injection
    Perfect predictions → residual = 0.

    Parameters
    ----------
    y_pred_physical : (batch, 8) — predictions in PHYSICAL units
                      [Vm_bus1..4 (pu), Va_bus1..4 (degrees)]
    X_raw           : (batch, 8) — inputs in physical units [MW/MVAr]
    G, B            : admittance matrices (n_bus, n_bus)
    base_mva        : system base

    Returns
    -------
    loss : scalar tensor — mean squared power balance residual
    """
    batch = y_pred_physical.shape[0]
    G = G.to(y_pred_physical.device)
    B = B.to(y_pred_physical.device)

    # Full voltage vectors including slack bus 0 (fixed)
    slack_vm = torch.ones (batch, 1, device=y_pred_physical.device)
    slack_va = torch.zeros(batch, 1, device=y_pred_physical.device)

    Vm_all = torch.cat([slack_vm, y_pred_physical[:, :4]], dim=1)   # (batch,5)
    Va_all = torch.cat([slack_va,
                        torch.deg2rad(y_pred_physical[:, 4:8])],
                       dim=1)                                          # (batch,5)

    # Angle difference matrix (batch, 5, 5)
    dVa     = Va_all.unsqueeze(2) - Va_all.unsqueeze(1)
    cos_dVa = torch.cos(dVa)
    sin_dVa = torch.sin(dVa)

    # Outer product of magnitudes (batch, 5, 5)
    VmVm = Vm_all.unsqueeze(2) * Vm_all.unsqueeze(1)

    # Calculated injections at all buses (batch, 5)
    P_calc = (VmVm * (G * cos_dVa + B * sin_dVa)).sum(dim=2)
    Q_calc = (VmVm * (G * sin_dVa - B * cos_dVa)).sum(dim=2)

    # Non-slack buses 1-4 only
    P_calc_ns = P_calc[:, 1:]   # (batch, 4)
    Q_calc_ns = Q_calc[:, 1:]

    # Known injections from inputs (per-unit)
    # Bus 1: load only    Bus 2: load only
    # Bus 3: solar inject Bus 4: load only
    P_load1=X_raw[:,0]; Q_load1=X_raw[:,1]
    P_load2=X_raw[:,2]; Q_load2=X_raw[:,3]
    P_load4=X_raw[:,4]; Q_load4=X_raw[:,5]
    P_solar=X_raw[:,6]; Q_solar=X_raw[:,7]

    P_known = torch.stack([-P_load1, -P_load2,  P_solar, -P_load4],
                           dim=1) / base_mva
    Q_known = torch.stack([-Q_load1, -Q_load2,  Q_solar, -Q_load4],
                           dim=1) / base_mva

    P_res = P_known - P_calc_ns
    Q_res = Q_known - Q_calc_ns

    return (P_res**2).mean() + (Q_res**2).mean()


# ---------------------------------------------------------------------------
# 3. Combined PINN loss — handles scaling internally
# ---------------------------------------------------------------------------

def pinn_loss(
    y_pred_scaled : torch.Tensor,
    y_true_scaled : torch.Tensor,
    X_raw         : torch.Tensor,
    G             : torch.Tensor,
    B             : torch.Tensor,
    scaler_y      : StandardScaler,
    base_mva      : float = 1.0,
    lam           : float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined data + physics loss.

        L_total = L_data + λ * L_physics

    Handles the scaling mismatch internally:
      - L_data    uses scaled predictions (same space as training)
      - L_physics inverse-transforms predictions to physical units first

    Parameters
    ----------
    y_pred_scaled : (batch, 8) — model output (StandardScaler space)
    y_true_scaled : (batch, 8) — true outputs (StandardScaler space)
    X_raw         : (batch, 8) — inputs in physical units [MW/MVAr]
    G, B          : admittance matrices
    scaler_y      : fitted StandardScaler for outputs
    base_mva      : float
    lam           : float — physics weight (default 0.1)

    Returns
    -------
    total, l_data, l_physics : scalar tensors
    """
    # Data loss in scaled space
    l_data = nn.functional.mse_loss(y_pred_scaled, y_true_scaled)

    # Inverse-transform predictions to physical units for physics loss
    mean = torch.tensor(scaler_y.mean_,  dtype=torch.float32,
                        device=y_pred_scaled.device)
    std  = torch.tensor(scaler_y.scale_, dtype=torch.float32,
                        device=y_pred_scaled.device)
    y_pred_physical = y_pred_scaled * std + mean

    l_physics = physics_loss(y_pred_physical, X_raw, G, B, base_mva)
    total     = l_data + lam * l_physics

    return total, l_data, l_physics


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from workshop_utils.power_flow import build_network

    print("Building network...")
    net = build_network()
    pp.runpp(net, numba=False)

    print("Extracting Y-bus...")
    G, B, base_mva = build_ybus(net)
    print(f"  G shape   : {G.shape}")
    print(f"  base_mva  : {base_mva}")

    # Ground truth voltages — physics residual should be ~0
    print("\nTesting physics_loss with ground-truth voltages...")
    vm = net.res_bus["vm_pu"].values[1:]
    va = net.res_bus["va_degree"].values[1:]
    y_gt = torch.tensor(
        np.concatenate([vm, va]).astype("float32")
    ).unsqueeze(0).repeat(4, 1)

    p1=float(net.load.loc[0,"p_mw"]); q1=float(net.load.loc[0,"q_mvar"])
    p2=float(net.load.loc[1,"p_mw"]); q2=float(net.load.loc[1,"q_mvar"])
    p4=float(net.load.loc[2,"p_mw"]); q4=float(net.load.loc[2,"q_mvar"])
    ps=float(net.sgen.loc[0,"p_mw"]); qs=float(net.sgen.loc[0,"q_mvar"])
    X = torch.tensor([p1,q1,p2,q2,p4,q4,ps,qs],
                     dtype=torch.float32).unsqueeze(0).repeat(4,1)

    loss_gt = physics_loss(y_gt, X, G, B, base_mva)
    print(f"  Physics loss (ground truth) : {loss_gt.item():.10f}  (expect ~0)")

    loss_bad = physics_loss(y_gt + 0.05, X, G, B, base_mva)
    print(f"  Physics loss (perturbed+0.05): {loss_bad.item():.6f}  (expect > 0)")

    # Test pinn_loss with mock scaler
    from sklearn.preprocessing import StandardScaler
    mock_scaler = StandardScaler()
    mock_scaler.fit(y_gt.numpy())
    y_sc = torch.tensor(mock_scaler.transform(y_gt.numpy()).astype("float32"))

    total, l_d, l_p = pinn_loss(y_sc, y_sc, X, G, B, mock_scaler,
                                  base_mva, lam=0.1)
    print(f"\n  pinn_loss (perfect pred): total={total.item():.8f}  "
          f"data={l_d.item():.8f}  phys={l_p.item():.8f}")

    print("\npinn.py self-test PASSED.")