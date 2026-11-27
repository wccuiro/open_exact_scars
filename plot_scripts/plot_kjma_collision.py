"""Plot the local damage and independent-front prediction from the kjma_collision Rust example."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def plot_depolarizing(path, data):
    # Projective measurement opposite the noiseless Bloch direction: P = (1 - Re c)/2.
    data = data.assign(p_opposite=data.damage / 2)
    profiles = {
        run: rows.pivot_table(index="t", columns="site", values="p_opposite", aggfunc="mean")
        for run, rows in data.groupby("run")
    }
    fig = plt.figure(figsize=(8.0, 6.4), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=[1.45, 0.08, 1.05], hspace=0.12)
    for col, (run, title) in enumerate([("A", "Kick at site 3"), ("B", "Kick at site 10"),
                                         ("AB", "Both kicks")]):
        ax = fig.add_subplot(grid[0, col])
        d = profiles[run]
        heat = ax.pcolormesh(d.columns, d.index, d, shading="nearest", cmap="inferno", vmin=0, vmax=1)
        events = data.loc[(data.run == run) & (data.kick == 1), ["site", "t"]].drop_duplicates()
        ax.scatter(events.site, events.t, s=65, marker="*", c="white", edgecolors="#222222",
                   linewidths=0.8, zorder=3)
        ax.set(title=title, xlabel="Spin site", xticks=[0, 3, 7, 10, 13], ylim=(0, d.index.max()))
        if col == 0:
            ax.set_ylabel("Time")
            ax.text(.04, .95, "★  Applied kick", transform=ax.transAxes, va="top", color="white", fontsize=9)
        else:
            ax.tick_params(labelleft=False)
    colorbar = fig.colorbar(heat, cax=fig.add_subplot(grid[1, :]), orientation="horizontal",
                            ticks=[0, 0.25, 0.5, 0.75, 1], format=PercentFormatter(xmax=1, decimals=0))
    colorbar.set_label("Probability of measuring the spin opposite to the unperturbed scar")

    def first_crossing(d):
        above = d >= 0.175  # The 0.35 local-damage threshold used in the z-kick plot.
        return above.idxmax().where(above.any())

    pred = first_crossing(np.maximum(profiles["A"], profiles["B"]))
    actual = first_crossing(profiles["AB"])
    ax = fig.add_subplot(grid[2, :])
    ax.plot(actual.index, actual, "o-", color="#0c6465", label="Both kicks together", lw=2, ms=5)
    ax.plot(pred.index, pred, "s--", color="#b84716", mfc="white", ms=4,
            label="Earlier of the two single-kick crossings")
    missing = pred.index[pred.isna()]
    if len(missing):
        end = profiles["AB"].index.max()
        ax.scatter(missing, np.full(len(missing), end), marker="^", s=45,
                   edgecolor="#b84716", facecolor="white")
        ax.annotate("Neither single kick crossed by the end", (missing[-1], end),
                    xytext=(-8, -5), textcoords="offset points", ha="right", va="top",
                    color="#994114", fontsize=8)
    ax.set(title="When does each spin first reach 17.5%?", xlabel="Spin site", ylabel="First crossing time",
           xticks=actual.index, ylim=(0, max(3, actual.max(), pred.max()) + 0.1))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.34), ncols=2, frameon=False, fontsize=9)
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path.with_suffix(".png"), dpi=190)
    plt.close(fig)


def main(path):
    data = pd.read_csv(path)
    if "axis_a" in data.columns:
        plot_depolarizing(path, data)
        return
    profiles = {
        run: rows.pivot(index="t", columns="site", values="damage").sort_index()
        for run, rows in data.groupby("run")
    }
    threshold = 0.35  # Same front-body level as spreading.py.
    predicted = np.maximum(profiles["A"], profiles["B"])
    reached = predicted >= threshold
    arrival = reached.idxmax().where(reached.any())
    measured = profiles["AB"] >= threshold
    pair_arrival = measured.idxmax().where(measured.any())

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8.2, 4.1), layout="constrained")
    for ax, run, title in zip(axes, ("A", "AB"), ("One kick", "Two separated kicks")):
        d = profiles[run]
        heat = ax.pcolormesh(d.columns, d.index, d, shading="nearest", cmap="magma", vmin=0, vmax=2,
                             rasterized=True)
        events = data[(data.run == run) & (data.kick == 1)]
        ax.scatter(events.site, events.t, marker="*", s=90, color="white", edgecolor="#222222",
                   linewidth=0.8, zorder=5, label="Applied kick")
        ax.set(title=title, xlabel="Site on the ring", xticks=[0, 3, 6, 10, 13], ylim=(0, d.index.max()))
    axes[0].set_ylabel("Time")
    axes[1].plot(pair_arrival.index, pair_arrival, "o-", color="white", ms=3, lw=1.4,
                 label="Measured first arrival")
    axes[1].plot(arrival.index, arrival, "--", color="#66e4e2", lw=1.6,
                 label="Predicted from isolated kicks")
    axes[0].legend(loc="upper left", framealpha=0.94, fontsize=8)
    axes[1].legend(handles=axes[1].lines, loc="upper left", framealpha=0.94, fontsize=8)
    fig.colorbar(heat, ax=axes, shrink=0.87, ticks=[0, 0.35, 1, 2], label="Local scar damage")
    fig.savefig(path.with_suffix(".png"), dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "data" / "kjma_collision.csv"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
