"""
plotting.py
-----------
Reusable plotting helpers for Notebook 2: Power-Flow Surrogate.

Functions
---------
plot_true_vs_predicted()   — scatter plot of true vs. predicted outputs
plot_error_by_bus()        — bar chart of MAE per bus / output feature
plot_error_vs_loading()    — key exercise plot: surrogate error vs load level
plot_voltage_profile()     — bus voltage magnitudes: true vs predicted
plot_training_curve()      — loss vs epoch for the training run
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Shared style defaults (call once at notebook top, or they apply here)
# ---------------------------------------------------------------------------

def set_plot_style():
    import seaborn as sns
    sns.set_context("notebook")
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["font.size"] = 11


# ---------------------------------------------------------------------------
# 1. True vs. Predicted scatter
# ---------------------------------------------------------------------------

def plot_true_vs_predicted(y_true, y_pred, feature_names,
                           save_dir=None, title=None):
    """
    Scatter plots of true vs. predicted outputs.

    Splits into TWO separate figures to avoid label overlap:
      Figure 1 — Voltage magnitudes |V| (pu)       [buses 1-4]
      Figure 2 — Voltage angles     δ   (degrees)  [buses 1-4]

    Each subplot includes an MAE annotation box so participants can
    immediately see which bus the surrogate struggles with most.

    Parameters
    ----------
    y_true        : np.ndarray (n_samples, 8)
    y_pred        : np.ndarray (n_samples, 8)
    feature_names : list[str] — must follow the order
                    [Vm_bus1..4, Va_bus1..4]
    save_dir      : str | Path | None — if provided, saves PNGs there
    title         : str | None — base title (appended with Vm / Va)
    """
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_context("notebook")

    # Human-readable axis labels
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

    groups = [
        {
            "indices" : [0, 1, 2, 3],
            "color"   : "steelblue",
            "unit"    : "pu",
            "suffix"  : "Voltage Magnitudes |V| (pu)",
            "fname"   : "fig1_voltage_magnitudes.png",
        },
        {
            "indices" : [4, 5, 6, 7],
            "color"   : "darkorange",
            "unit"    : "deg",
            "suffix"  : "Voltage Angles δ (degrees)",
            "fname"   : "fig2_voltage_angles.png",
        },
    ]

    for grp in groups:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        base = title or "True vs. Predicted"
        fig.suptitle(f"{base} — {grp['suffix']}",
                     fontsize=14, fontweight="bold", y=1.01)

        for ax, i in zip(axes.flatten(), grp["indices"]):
            name  = feature_names[i]
            label = display.get(name, name)
            lo = min(y_true[:, i].min(), y_pred[:, i].min()) - 0.002
            hi = max(y_true[:, i].max(), y_pred[:, i].max()) + 0.002

            ax.scatter(y_true[:, i], y_pred[:, i],
                       alpha=0.4, s=18,
                       color=grp["color"], edgecolors="none",
                       label="Surrogate prediction")
            ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")

            mae = np.abs(y_true[:, i] - y_pred[:, i]).mean()
            ax.text(0.04, 0.93,
                    f"MAE = {mae:.5f} {grp['unit']}",
                    transform=ax.transAxes, fontsize=9,
                    color="darkred", va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="lightyellow",
                              edgecolor="gray", alpha=0.8))
            ax.set_title(label, fontsize=12, pad=8)
            ax.set_xlabel(f"True {label}", fontsize=10)
            ax.set_ylabel(f"Predicted {label}", fontsize=10)
            ax.legend(fontsize=9)
            ax.tick_params(labelsize=9)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        plt.tight_layout()

        if save_dir is not None:
            from pathlib import Path
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            fpath = Path(save_dir) / grp["fname"]
            plt.savefig(fpath, dpi=150, bbox_inches="tight")
            print(f"Saved: {fpath}")

        plt.show()


# ---------------------------------------------------------------------------
# 2. MAE per output feature (bar chart)
# ---------------------------------------------------------------------------

def plot_error_by_bus(y_true, y_pred, feature_names,
                      save_dir=None, title=None):
    """
    Horizontal bar chart showing MAE per output feature.

    Voltage magnitudes (blue) and voltage angles (orange) are
    color-coded so participants can compare error types at a glance.

    Parameters
    ----------
    y_true        : np.ndarray (n_samples, 8)
    y_pred        : np.ndarray (n_samples, 8)
    feature_names : list[str]
    save_dir      : str | Path | None
    title         : str | None
    """
    from matplotlib.patches import Patch

    mae    = np.abs(y_true - y_pred).mean(axis=0)
    # First 4 = voltage magnitudes (blue), last 4 = angles (orange)
    colors = ["steelblue"] * 4 + ["darkorange"] * 4

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

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, mae, color=colors, edgecolor="white", height=0.6)
    ax.bar_label(bars, fmt="%.5f", padding=4, fontsize=9)
    ax.set_xlabel("Mean Absolute Error", fontsize=11)
    ax.set_title(
        title or "Surrogate MAE by Output Feature\n"
                 "(blue = voltage magnitude | orange = voltage angle)",
        fontsize=12, pad=10,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, mae.max() * 1.25)
    ax.tick_params(labelsize=10)
    ax.grid(axis="x", ls="--", alpha=0.4)

    legend_elements = [
        Patch(facecolor="steelblue",  label="Voltage magnitude |V| (pu)"),
        Patch(facecolor="darkorange", label="Voltage angle δ (deg)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()

    if save_dir is not None:
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fpath = Path(save_dir) / "fig3_mae_by_feature.png"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"Saved: {fpath}")

    plt.show()
    return dict(zip(feature_names, mae))


# ---------------------------------------------------------------------------
# 3. Error vs. loading level  (the key participant-exercise plot)
# ---------------------------------------------------------------------------

def plot_error_vs_loading(
    load_scales,
    errors,
    train_range=(0.8, 1.2),
    xlabel="Load scaling factor (pu)",
    ylabel="Mean Absolute Error (pu)",
    title="Surrogate Error vs. Operating Point",
):
    """
    Line/scatter plot showing how surrogate MAE changes with load level.

    Used in the participant exercise:
      - Train on load_range=(0.8, 1.2)
      - Test across a wider range e.g. (0.6, 1.4)
      - Plot shows where the model breaks down outside training distribution

    Parameters
    ----------
    load_scales : array-like — x-axis values (e.g. 0.6, 0.7, ... 1.4)
    errors      : array-like — mean absolute error at each load scale
    train_range : (lo, hi)  — shaded region indicating the training range
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(load_scales, errors, "o-", color="steelblue",
            lw=2, ms=6, label="Surrogate MAE")

    # Shade the training region
    ax.axvspan(train_range[0], train_range[1],
               alpha=0.15, color="green", label=f"Training range {train_range}")

    # Mark boundaries
    ax.axvline(train_range[0], color="green", ls="--", lw=1)
    ax.axvline(train_range[1], color="green", ls="--", lw=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Bus voltage profile  (true vs. predicted magnitudes)
# ---------------------------------------------------------------------------

def plot_voltage_profile(
    vm_true,
    vm_pred,
    bus_names=None,
    title="Bus Voltage Profile: True vs. Predicted",
):
    """
    Side-by-side bar chart comparing true and predicted voltage magnitudes
    across all non-slack buses.

    Parameters
    ----------
    vm_true   : array-like — true |V| values in pu  (one sample)
    vm_pred   : array-like — predicted |V| values in pu (same sample)
    bus_names : list[str] | None
    """
    vm_true = np.array(vm_true)
    vm_pred = np.array(vm_pred)
    n = len(vm_true)
    bus_names = bus_names or [f"Bus {i+1}" for i in range(n)]
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, vm_true, width=w, label="True (pandapower)",
           color="steelblue", edgecolor="white")
    ax.bar(x + w/2, vm_pred, width=w, label="Predicted (surrogate)",
           color="darkorange", edgecolor="white")

    ax.axhline(1.0, color="gray", ls=":", lw=1.2, label="|V|=1.0 pu (nominal)")
    ax.set_xticks(x)
    ax.set_xticklabels(bus_names)
    ax.set_ylabel("Voltage magnitude (pu)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Training curve
# ---------------------------------------------------------------------------

def plot_training_curve(
    train_losses,
    val_losses=None,
    phase1_epochs=None,
    title="Training Loss Curve",
):
    """
    Plot training and validation loss vs. epoch.

    Two side-by-side panels:
      Left  — log scale  : shows the full dynamic range of loss drop
      Right — linear scale: shows fine-grained behaviour after initial drop

    Parameters
    ----------
    train_losses  : list[float] — MSE loss per epoch (training set)
    val_losses    : list[float] | None — MSE loss per epoch (val set)
    phase1_epochs : int | None  — if provided, draws a vertical dashed line
                                  marking where Phase 1 ended and Phase 2 began
                                  (e.g. phase1_epochs=20 for the workshop)
    title         : str
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(14, 4))

    for ax, scale in [(ax_log, "log"), (ax_lin, "linear")]:

        ax.plot(epochs, train_losses,
                label="Train loss", color="steelblue", lw=1.8)

        if val_losses is not None:
            ax.plot(epochs, val_losses,
                    label="Val loss", color="darkorange",
                    ls="--", lw=1.8)

            # Mark best validation epoch
            best_epoch = int(np.argmin(val_losses)) + 1
            best_val   = min(val_losses)
            ax.axvline(best_epoch, color="green", ls=":", lw=1.2,
                       label=f"Best val epoch {best_epoch}")
            ax.scatter([best_epoch], [best_val],
                       color="green", zorder=5, s=40)

        # Phase boundary marker
        if phase1_epochs is not None:
            ax.axvline(phase1_epochs, color="gray", ls="-.", lw=1.5,
                       label=f"Phase 1 end (epoch {phase1_epochs})")
            ax.text(phase1_epochs + 1,
                    ax.get_ylim()[1] if scale == "linear"
                    else train_losses[0],
                    "← Phase 2", fontsize=8, color="gray", va="top")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_yscale(scale)
        ax.set_title(f"{title} ({'log scale' if scale == 'log' else 'linear scale'})")
        ax.legend(fontsize=8)
        ax.grid(True, which="both" if scale == "log" else "major",
                ls="--", alpha=0.4)

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"  Total epochs      : {len(train_losses)}")
    print(f"  Final train loss  : {train_losses[-1]:.6f}")
    if val_losses:
        best_epoch = int(np.argmin(val_losses)) + 1
        print(f"  Best val loss     : {min(val_losses):.6f}  (epoch {best_epoch})")
        print(f"  Final val loss    : {val_losses[-1]:.6f}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(469)
    n = 200

    # 8 output features (Option A — Q_solar is now an INPUT not output)
    feature_names = [
        "Vm_bus1_pu", "Vm_bus2_pu", "Vm_bus3_pu", "Vm_bus4_pu",
        "Va_bus1_deg", "Va_bus2_deg", "Va_bus3_deg", "Va_bus4_deg",
    ]

    y_true = rng.uniform(0.95, 1.05, size=(n, len(feature_names)))
    y_pred = y_true + rng.normal(0, 0.005, size=y_true.shape)

    print("Testing plot_true_vs_predicted...")
    plot_true_vs_predicted(y_true, y_pred, feature_names)

    print("Testing plot_error_by_bus...")
    plot_error_by_bus(y_true, y_pred, feature_names)

    print("Testing plot_error_vs_loading...")
    scales = np.linspace(0.6, 1.4, 9)
    errors = 0.002 + 0.01 * np.abs(scales - 1.0) ** 2
    plot_error_vs_loading(scales, errors)

    print("Testing plot_voltage_profile...")
    plot_voltage_profile(
        vm_true=[1.00, 0.985, 0.972, 0.968],
        vm_pred=[1.001, 0.984, 0.974, 0.969],
    )

    print("Testing plot_training_curve...")
    train_l = [0.5 / (1 + 0.3*i) for i in range(200)]
    val_l   = [0.55 / (1 + 0.28*i) for i in range(200)]
    plot_training_curve(
        train_l, val_l,
        phase1_epochs=20,
        title="Power Flow Surrogate Training"
    )

    print("plotting.py self-test complete.")