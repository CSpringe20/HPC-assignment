import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Load data
# -----------------------------
strong = pd.read_csv("csv/strong_scaling.csv")
weak = pd.read_csv("csv/weak_scaling.csv")
openmp = pd.read_csv("csv/openmp_scaling.csv")

for df in (strong, weak):
    df["TOTAL_CORES"] = df["MPI_TASKS"] * df["OMP_THREADS"]

# -----------------------------
# Plot styling (your rules)
#   - AVG: green
#   - WORST: red
#   - TOTAL: solid line
#   - COMP: dotted line
# -----------------------------
STYLES = {
    ("AVG", "TOTAL"): {"color": "tab:green", "linestyle": "-", "marker": "o"},
    ("AVG", "COMP"):  {"color": "tab:green", "linestyle": ":", "marker": "s"},
    ("WORST", "TOTAL"): {"color": "tab:red", "linestyle": "-", "marker": "o"},
    ("WORST", "COMP"):  {"color": "tab:red", "linestyle": ":", "marker": "s"},
}


def _agg_with_std(df: pd.DataFrame, group_col: str, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby(group_col)
    mean_df = grouped[cols].mean()
    std_df = grouped[cols].std()
    return mean_df, std_df


def _plot_band(ax, x, y_mean, y_std, label: str, style: dict, band_alpha: float = 0.18):
    ax.plot(x, y_mean, label=label, **style)
    # If only one repeat, std can be NaN; avoid fill_between issues
    if y_std is not None and not np.all(np.isnan(y_std)):
        ax.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            color=style["color"],
            alpha=band_alpha,
        )


# -----------------------------
# OpenMP scaling (2 subplots)
#   - Speedup line: blue
#   - Efficiency line: orange
# -----------------------------
def plot_openmp_scaling(save_path="images/openmp_scaling.png"):
    df = openmp.copy()
    avg_df = df.groupby("OMP_THREADS", as_index=False).mean(numeric_only=True)

    threads = avg_df["OMP_THREADS"].astype(int)
    avg_comp = avg_df["AVG_COMP"]

    t1 = avg_comp.iloc[0]
    speedup = t1 / avg_comp
    efficiency = speedup / threads

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Speedup (BLUE)
    axes[0].plot(threads, speedup, marker="o", color="tab:blue", label="Speedup")
    axes[0].plot(threads, threads / threads.iloc[0], linestyle="--", color="gray", label="Ideal Speedup")
    axes[0].set_xlabel("Number of Threads", fontsize=12)
    axes[0].set_ylabel("Speedup (T1 / Tn)", fontsize=12)
    axes[0].set_title("OpenMP Speedup vs Threads", fontsize=14)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].set_xticks(threads)
    axes[0].set_xticklabels(threads)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Efficiency (ORANGE)
    axes[1].plot(threads, efficiency, marker="o", color="tab:orange", label="Efficiency")
    axes[1].axhline(1.0, linestyle="--", color="gray", linewidth=1, label="Ideal Efficiency")
    axes[1].set_xlabel("Number of Threads", fontsize=12)
    axes[1].set_ylabel("Efficiency (Speedup / Threads)", fontsize=12)
    axes[1].set_title("OpenMP Efficiency vs Threads", fontsize=14)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(threads)
    axes[1].set_xticklabels(threads)
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Plot saved to {save_path}")


# -----------------------------
# Strong scaling (speedup + efficiency, separate figures)
# -----------------------------
def plot_strong_scaling(save_dir="images/"):
    df = strong.copy()
    min_cores = df["TOTAL_CORES"].min()

    # Baselines at min cores (AVG and WORST baselines separately)
    base_avg_total = df.loc[df["TOTAL_CORES"] == min_cores, "AVG_TOTAL"].mean()
    base_avg_comp  = df.loc[df["TOTAL_CORES"] == min_cores, "AVG_COMP"].mean()
    base_max_total = df.loc[df["TOTAL_CORES"] == min_cores, "MAX_TOTAL"].mean()
    base_max_comp  = df.loc[df["TOTAL_CORES"] == min_cores, "MAX_COMP"].mean()

    # Speedup (AVG and WORST)
    df["SPEEDUP_TOTAL"] = base_avg_total / df["AVG_TOTAL"]
    df["SPEEDUP_COMP"]  = base_avg_comp  / df["AVG_COMP"]
    df["WORST_SPEEDUP_TOTAL"] = base_max_total / df["MAX_TOTAL"]
    df["WORST_SPEEDUP_COMP"]  = base_max_comp  / df["MAX_COMP"]

    # Efficiency = speedup / (p/p0)
    scale = df["TOTAL_CORES"] / min_cores
    df["EFF_TOTAL"] = df["SPEEDUP_TOTAL"] / scale
    df["EFF_COMP"]  = df["SPEEDUP_COMP"]  / scale
    df["WORST_EFF_TOTAL"] = df["WORST_SPEEDUP_TOTAL"] / scale
    df["WORST_EFF_COMP"]  = df["WORST_SPEEDUP_COMP"]  / scale

    cols = [
        "SPEEDUP_TOTAL", "SPEEDUP_COMP", "WORST_SPEEDUP_TOTAL", "WORST_SPEEDUP_COMP",
        "EFF_TOTAL", "EFF_COMP", "WORST_EFF_TOTAL", "WORST_EFF_COMP",
    ]
    mean_df, std_df = _agg_with_std(df, "TOTAL_CORES", cols)
    x = mean_df.index

    # --- Strong Scaling Speedup ---
    fig, ax = plt.subplots(figsize=(8, 6))

    _plot_band(ax, x, mean_df["SPEEDUP_TOTAL"], std_df["SPEEDUP_TOTAL"],
               "AVG Total Speedup", STYLES[("AVG", "TOTAL")])
    _plot_band(ax, x, mean_df["SPEEDUP_COMP"], std_df["SPEEDUP_COMP"],
               "AVG Computation Speedup", STYLES[("AVG", "COMP")])

    _plot_band(ax, x, mean_df["WORST_SPEEDUP_TOTAL"], std_df["WORST_SPEEDUP_TOTAL"],
               "WORST Total Speedup", STYLES[("WORST", "TOTAL")])
    _plot_band(ax, x, mean_df["WORST_SPEEDUP_COMP"], std_df["WORST_SPEEDUP_COMP"],
               "WORST Computation Speedup", STYLES[("WORST", "COMP")])

    ax.plot(x, x / x.min(), "k--", label="Ideal Speedup")
    ax.set_xlabel("Total Cores")
    ax.set_ylabel("Speedup")
    ax.set_title("Strong Scaling Speedup")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()

    out_speedup = f"{save_dir}strong_scaling_speedup.png"
    plt.savefig(out_speedup, dpi=300)
    plt.show()
    print(f"Plot saved to {out_speedup}")

    # --- Strong Scaling Efficiency ---
    fig, ax = plt.subplots(figsize=(8, 6))

    _plot_band(ax, x, mean_df["EFF_TOTAL"], std_df["EFF_TOTAL"],
               "AVG Total Efficiency", STYLES[("AVG", "TOTAL")])
    _plot_band(ax, x, mean_df["EFF_COMP"], std_df["EFF_COMP"],
               "AVG Computation Efficiency", STYLES[("AVG", "COMP")])

    _plot_band(ax, x, mean_df["WORST_EFF_TOTAL"], std_df["WORST_EFF_TOTAL"],
               "WORST Total Efficiency", STYLES[("WORST", "TOTAL")])
    _plot_band(ax, x, mean_df["WORST_EFF_COMP"], std_df["WORST_EFF_COMP"],
               "WORST Computation Efficiency", STYLES[("WORST", "COMP")])

    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="Ideal Efficiency")
    ax.set_xlabel("Total Cores")
    ax.set_ylabel("Parallel Efficiency")
    ax.set_title("Strong Scaling Efficiency")
    ax.set_ylim(0.70, 1.10)
    ax.set_yticks(np.arange(0.70, 1.11, 0.10))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()

    out_eff = f"{save_dir}strong_scaling_efficiency.png"
    plt.savefig(out_eff, dpi=300)
    plt.show()
    print(f"Plot saved to {out_eff}")


# -----------------------------
# Weak scaling (ONLY efficiency, single plot)
# -----------------------------
def plot_weak_scaling(save_path="images/weak_scaling.png"):
    df = weak.copy()
    min_cores = df["TOTAL_CORES"].min()

    # Baselines at min cores (AVG and WORST baselines separately)
    base_avg_total = df.loc[df["TOTAL_CORES"] == min_cores, "AVG_TOTAL"].mean()
    base_avg_comp  = df.loc[df["TOTAL_CORES"] == min_cores, "AVG_COMP"].mean()
    base_max_total = df.loc[df["TOTAL_CORES"] == min_cores, "MAX_TOTAL"].mean()
    base_max_comp  = df.loc[df["TOTAL_CORES"] == min_cores, "MAX_COMP"].mean()

    # Weak efficiency (standard) = T1/Tp
    df["EFF_TOTAL"] = base_avg_total / df["AVG_TOTAL"]
    df["EFF_COMP"]  = base_avg_comp  / df["AVG_COMP"]
    df["WORST_EFF_TOTAL"] = base_max_total / df["MAX_TOTAL"]
    df["WORST_EFF_COMP"]  = base_max_comp  / df["MAX_COMP"]

    cols = ["EFF_TOTAL", "EFF_COMP", "WORST_EFF_TOTAL", "WORST_EFF_COMP"]
    mean_df, std_df = _agg_with_std(df, "TOTAL_CORES", cols)
    x = mean_df.index

    fig, ax = plt.subplots(figsize=(8, 6))

    _plot_band(ax, x, mean_df["EFF_TOTAL"], std_df["EFF_TOTAL"],
               "AVG Total Efficiency", STYLES[("AVG", "TOTAL")])
    _plot_band(ax, x, mean_df["EFF_COMP"], std_df["EFF_COMP"],
               "AVG Computation Efficiency", STYLES[("AVG", "COMP")])

    _plot_band(ax, x, mean_df["WORST_EFF_TOTAL"], std_df["WORST_EFF_TOTAL"],
               "WORST Total Efficiency", STYLES[("WORST", "TOTAL")])
    _plot_band(ax, x, mean_df["WORST_EFF_COMP"], std_df["WORST_EFF_COMP"],
               "WORST Computation Efficiency", STYLES[("WORST", "COMP")])

    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="Ideal Efficiency")
    ax.set_xlabel("Total Cores")
    ax.set_ylabel("Parallel Efficiency (T1 / Tp)")
    ax.set_title("Weak Scaling Efficiency")
    ax.set_ylim(0.70, 1.10)
    ax.set_yticks(np.arange(0.70, 1.11, 0.10))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Plot saved to {save_path}")


def main():
    plot_openmp_scaling()
    plot_strong_scaling()
    plot_weak_scaling()


if __name__ == "__main__":
    main()
