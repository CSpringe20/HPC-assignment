import pandas as pd
import matplotlib.pyplot as plt


strong = pd.read_csv("csv/strong_scaling.csv")
weak = pd.read_csv("csv/weak_scaling.csv")
openmp = pd.read_csv("csv/timing_summary.csv")

for df in [strong, weak]:
    df["TOTAL_CORES"] = df["MPI_TASKS"] * df["OMP_THREADS"]
    
colors = {"TOTAL": "lightgreen", "COMP": "tab:red"}

def plot_openmp_scaling(save_path="images/openmp_scaling.png"):
    df = openmp
    avg_df = df.groupby("OMP_THREADS", as_index=False).mean()

    threads = avg_df["OMP_THREADS"]
    avg_comp = avg_df["AVG_COMP"]

    # Compute speedup
    t1 = avg_comp.iloc[0]
    speedup = t1 / avg_comp

    # Compute efficiency = speedup / number_of_threads
    efficiency = speedup / threads

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left Plot: Speedup ---
    axes[0].plot(threads, speedup, marker="o", label="Speedup")
    axes[0].plot(
        threads,
        threads / threads.iloc[0],
        linestyle="--",
        color="gray",
        label="Ideal Speedup",
    )
    axes[0].set_xlabel("Number of Threads", fontsize=12)
    axes[0].set_ylabel("Speedup (T1 / Tn)", fontsize=12)
    axes[0].set_title("Speedup vs Threads", fontsize=14)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=2)
    axes[0].set_xticks(threads)
    axes[0].set_xticklabels(threads)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # --- Right Plot: Efficiency ---
    axes[1].plot(threads, efficiency, marker="o", color="tab:orange", label="Efficiency")
    axes[1].axhline(1.0, linestyle="--", color="gray", linewidth=1, label="Ideal Efficiency")
    axes[1].set_xlabel("Number of Threads", fontsize=12)
    axes[1].set_ylabel("Efficiency (Speedup / Threads)", fontsize=12)
    axes[1].set_title("Parallel Efficiency vs Threads", fontsize=14)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(threads)
    axes[1].set_xticklabels(threads)
    axes[1].set_ylim(0, 1.1)  # Efficiency max is 1
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Plot saved to {save_path}")


def plot_strong_scaling(save_path="images/"):
    strong_repeats = strong.copy()

    # Baselines
    baseline_total = strong_repeats[strong_repeats["TOTAL_CORES"] == strong_repeats["TOTAL_CORES"].min()]["AVG_TOTAL"].mean()
    baseline_comp  = strong_repeats[strong_repeats["TOTAL_CORES"] == strong_repeats["TOTAL_CORES"].min()]["AVG_COMP"].mean()
    m_baseline_total = strong_repeats[strong_repeats["TOTAL_CORES"] == strong_repeats["TOTAL_CORES"].min()]["MAX_TOTAL"].mean()
    m_baseline_comp  = strong_repeats[strong_repeats["TOTAL_CORES"] == strong_repeats["TOTAL_CORES"].min()]["MAX_COMP"].mean()

    # Speedup
    strong_repeats["SPEEDUP_TOTAL"] = baseline_total / strong_repeats["AVG_TOTAL"]
    strong_repeats["SPEEDUP_COMP"]  = baseline_comp / strong_repeats["AVG_COMP"]
    strong_repeats["MAX_SPEEDUP_TOTAL"] = m_baseline_total / strong_repeats["MAX_TOTAL"]
    strong_repeats["MAX_SPEEDUP_COMP"]  = m_baseline_comp / strong_repeats["MAX_COMP"]

    # Efficiency
    strong_repeats["EFF_TOTAL"] = strong_repeats["SPEEDUP_TOTAL"] / (strong_repeats["TOTAL_CORES"] / strong_repeats["TOTAL_CORES"].min())
    strong_repeats["EFF_COMP"]  = strong_repeats["SPEEDUP_COMP"] / (strong_repeats["TOTAL_CORES"] / strong_repeats["TOTAL_CORES"].min())
    strong_repeats["MAX_EFF_TOTAL"] = strong_repeats["MAX_SPEEDUP_TOTAL"] / (strong_repeats["TOTAL_CORES"] / strong_repeats["TOTAL_CORES"].min())
    strong_repeats["MAX_EFF_COMP"]  = strong_repeats["MAX_SPEEDUP_COMP"] / (strong_repeats["TOTAL_CORES"] / strong_repeats["TOTAL_CORES"].min())

    # Aggregate
    strong_grouped = strong_repeats.groupby("TOTAL_CORES")
    strong_mean = strong_grouped[["SPEEDUP_TOTAL", "SPEEDUP_COMP", "EFF_TOTAL", "EFF_COMP",
                                "MAX_SPEEDUP_TOTAL", "MAX_SPEEDUP_COMP", "MAX_EFF_TOTAL", "MAX_EFF_COMP"]].mean()
    strong_std  = strong_grouped[["SPEEDUP_TOTAL", "SPEEDUP_COMP", "EFF_TOTAL", "EFF_COMP",
                                "MAX_SPEEDUP_TOTAL", "MAX_SPEEDUP_COMP", "MAX_EFF_TOTAL", "MAX_EFF_COMP"]].std()

    # --- Plot Strong Scaling Speedup ---
    plt.figure(figsize=(8,6))
    x = strong_mean.index
    plt.plot(x, strong_mean["SPEEDUP_TOTAL"], 'o-', color=colors["TOTAL"], label="AVG Total Speedup")
    plt.fill_between(x, strong_mean["SPEEDUP_TOTAL"] - strong_std["SPEEDUP_TOTAL"],
                        strong_mean["SPEEDUP_TOTAL"] + strong_std["SPEEDUP_TOTAL"], color=colors["TOTAL"], alpha=0.2)
    plt.plot(x, strong_mean["MAX_SPEEDUP_TOTAL"], 'o--', color=colors["TOTAL"], label="WORST Total Speedup")
    plt.fill_between(x, strong_mean["MAX_SPEEDUP_TOTAL"] - strong_std["MAX_SPEEDUP_TOTAL"],
                        strong_mean["MAX_SPEEDUP_TOTAL"] + strong_std["MAX_SPEEDUP_TOTAL"], color=colors["TOTAL"], alpha=0.2)

    plt.plot(x, strong_mean["SPEEDUP_COMP"], 's-', color=colors["COMP"], label="AVG Computation Speedup")
    plt.fill_between(x, strong_mean["SPEEDUP_COMP"] - strong_std["SPEEDUP_COMP"],
                        strong_mean["SPEEDUP_COMP"] + strong_std["SPEEDUP_COMP"], color=colors["COMP"], alpha=0.2)
    plt.plot(x, strong_mean["MAX_SPEEDUP_COMP"], 's--', color=colors["COMP"], label="WORST Computation Speedup")
    plt.fill_between(x, strong_mean["MAX_SPEEDUP_COMP"] - strong_std["MAX_SPEEDUP_COMP"],
                        strong_mean["MAX_SPEEDUP_COMP"] + strong_std["MAX_SPEEDUP_COMP"], color=colors["COMP"], alpha=0.2)

    plt.plot(x, x / x.min(), 'k--', label="Ideal Speedup")
    plt.xlabel("Total Cores")
    plt.ylabel("Speedup")
    plt.title("Strong Scaling Speedup")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path+"strong_scaling_speedup.png", dpi=300)
    plt.show()

    print(f"Plot saved to {save_path}strong_scaling_speedup.png")


    # --- Plot Strong Scaling Efficiency ---
    plt.figure(figsize=(8,6))
    plt.plot(x, strong_mean["EFF_TOTAL"], 'o-', color=colors["TOTAL"], label="AVG Total Efficiency")
    plt.fill_between(x, strong_mean["EFF_TOTAL"] - strong_std["EFF_TOTAL"],
                        strong_mean["EFF_TOTAL"] + strong_std["EFF_TOTAL"], color=colors["TOTAL"], alpha=0.2)
    plt.plot(x, strong_mean["MAX_EFF_TOTAL"], 'o--', color=colors["TOTAL"], label="WORST Total Efficiency")
    plt.fill_between(x, strong_mean["MAX_EFF_TOTAL"] - strong_std["MAX_EFF_TOTAL"],
                        strong_mean["MAX_EFF_TOTAL"] + strong_std["MAX_EFF_TOTAL"], color=colors["TOTAL"], alpha=0.2)

    plt.plot(x, strong_mean["EFF_COMP"], 's-', color=colors["COMP"], label="AVG Computation Efficiency")
    plt.fill_between(x, strong_mean["EFF_COMP"] - strong_std["EFF_COMP"],
                        strong_mean["EFF_COMP"] + strong_std["EFF_COMP"], color=colors["COMP"], alpha=0.2)
    plt.plot(x, strong_mean["MAX_EFF_COMP"], 's--', color=colors["COMP"], label="WORST Computation Efficiency")
    plt.fill_between(x, strong_mean["MAX_EFF_COMP"] - strong_std["MAX_EFF_COMP"],
                        strong_mean["MAX_EFF_COMP"] + strong_std["MAX_EFF_COMP"], color=colors["COMP"], alpha=0.2)

    plt.plot(x, [1]*len(x), 'k--', label="Ideal Efficiency")
    plt.xlabel("Total Cores")
    plt.ylabel("Parallel Efficiency")
    plt.title("Strong Scaling Efficiency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path+"strong_scaling_efficiency.png", dpi=300)
    plt.show()

    print(f"Plot saved to {save_path}strong_scaling_efficiency.png")

def plot_weak_scaling(save_path="images/"):
    weak_repeats = weak.copy()

    # Baselines
    baseline_weak_total = weak_repeats[weak_repeats["TOTAL_CORES"] == weak_repeats["TOTAL_CORES"].min()]["AVG_TOTAL"].mean()
    baseline_weak_comp  = weak_repeats[weak_repeats["TOTAL_CORES"] == weak_repeats["TOTAL_CORES"].min()]["AVG_COMP"].mean()

    # Efficiency
    weak_repeats["EFF_TOTAL"] = baseline_weak_total / weak_repeats["AVG_TOTAL"]
    weak_repeats["EFF_COMP"]  = baseline_weak_comp / weak_repeats["AVG_COMP"]
    weak_repeats["MAX_EFF_TOTAL"] = baseline_weak_total / weak_repeats["MAX_TOTAL"]
    weak_repeats["MAX_EFF_COMP"]  = baseline_weak_comp / weak_repeats["MAX_COMP"]

    # Aggregate
    weak_grouped = weak_repeats.groupby("TOTAL_CORES")
    weak_mean = weak_grouped[["EFF_TOTAL", "EFF_COMP", "MAX_EFF_TOTAL", "MAX_EFF_COMP"]].mean()
    weak_std  = weak_grouped[["EFF_TOTAL", "EFF_COMP", "MAX_EFF_TOTAL", "MAX_EFF_COMP"]].std()

    # Define colors
    save_path = save_path + "weak_scaling.png"

    # --- Plot Weak Scaling Efficiency ---
    plt.figure(figsize=(8,6))
    xw = weak_mean.index
    plt.plot(xw, weak_mean["EFF_TOTAL"], 'o-', color=colors["TOTAL"], label="AVG Total Efficiency")
    plt.fill_between(xw, weak_mean["EFF_TOTAL"] - weak_std["EFF_TOTAL"],
                        weak_mean["EFF_TOTAL"] + weak_std["EFF_TOTAL"], color=colors["TOTAL"], alpha=0.2)
    plt.plot(xw, weak_mean["MAX_EFF_TOTAL"], 'o--', color=colors["TOTAL"], label="WORST Total Efficiency")
    plt.fill_between(xw, weak_mean["MAX_EFF_TOTAL"] - weak_std["MAX_EFF_TOTAL"],
                        weak_mean["MAX_EFF_TOTAL"] + weak_std["MAX_EFF_TOTAL"], color=colors["TOTAL"], alpha=0.2)

    plt.plot(xw, weak_mean["EFF_COMP"], 's-', color=colors["COMP"], label="AVG Computation Efficiency")
    plt.fill_between(xw, weak_mean["EFF_COMP"] - weak_std["EFF_COMP"],
                        weak_mean["EFF_COMP"] + weak_std["EFF_COMP"], color=colors["COMP"], alpha=0.2)
    plt.plot(xw, weak_mean["MAX_EFF_COMP"], 's--', color=colors["COMP"], label="WORST Computation Efficiency")
    plt.fill_between(xw, weak_mean["MAX_EFF_COMP"] - weak_std["MAX_EFF_COMP"],
                        weak_mean["MAX_EFF_COMP"] + weak_std["MAX_EFF_COMP"], color=colors["COMP"], alpha=0.2)

    plt.plot(xw, [1]*len(xw), 'k--', label="Ideal Efficiency")
    plt.xlabel("Total Cores")
    plt.ylabel("Parallel Efficiency")
    plt.title("Weak Scaling Efficiency")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Plot saved to {save_path}")

def main():
    plot_openmp_scaling()
    plot_strong_scaling()
    plot_weak_scaling()

if __name__=="__main__":
    main()