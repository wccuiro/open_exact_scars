"""Compare isolated Floquet seeds with held-out noisy quenches on the same ring."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def trajectories(root, rows):
    for row in rows:
        for seed in range(int(row["first"]), int(row["first"]) + int(row["count"])):
            path = root / row["directory"] / f"seed_{seed}.csv.gz"
            data = pd.read_csv(path)
            cycles = int(row["cycles"])
            if (len(data) != cycles + 1 or
                    not np.array_equal(data.cycle, np.arange(cycles + 1)) or
                    not (data.seed == seed).all() or not (data.L == int(row["L"])).all() or
                    not np.allclose(data.gamma, float(row["gamma"])) or
                    not np.allclose(data.dt, float(row["dt"]))):
                raise ValueError(f"Inconsistent trajectory: {path}")
            yield seed, data


def columns(prefix, size):
    return [f"{prefix}{j}" for j in range(size)]


def calibrate(root, rows, threshold, output):
    row = rows[0]
    size, phases, cycles = (int(row[k]) for k in ("L", "phases", "cycles"))
    # Each entry counts the sources (site and axis) that can reach this target bond.
    kernel = np.zeros((phases, cycles + 1, size))
    opposite = np.zeros((phases, cycles + 1))
    z_profile = np.zeros((cycles + 1, size))
    seen = set()
    for seed, data in trajectories(root, rows):
        phase, mark = divmod(seed - 1, 3 * size)
        site, axis = divmod(mark, 3)
        kicks = data[columns("kick", size)].to_numpy()
        if (seed in seen or np.count_nonzero(kicks >= 0) != 1 or kicks[0, site] != axis or
                not np.isclose(data.phase.iloc[0], 2*np.pi*phase/phases)):
            raise ValueError(f"Unexpected isolated kick in trajectory {seed}")
        seen.add(seed)
        singlet = data[columns("r", size)].to_numpy()
        reached = np.maximum.accumulate(singlet >= threshold, axis=0)
        if phase == 0 and axis == 2:
            z_profile += np.roll(singlet, size // 2 - site, axis=1) / size
        kernel[phase] += reached
        far = (site + size // 2) % size
        opposite[phase] += reached[:, far] | reached[:, (far - 1) % size]
    if seen != set(range(1, 3 * size * phases + 1)):
        raise ValueError("The calibration must cover every site, Pauli axis and phase")
    # Sum over birth sites, average over the three equally likely axes.
    kernel /= 3
    width = kernel.mean(axis=2)
    opposite /= 3 * size
    time = np.arange(cycles + 1) * float(row["dt"])
    table = pd.DataFrame({"age": time, "mean_reached_bonds": width.mean(axis=0),
                          "fraction_reaching_opposite_side": opposite.mean(axis=0)})
    for phase in range(phases):
        table[f"width_phase_{phase}"] = width[phase]
    table.to_csv(output / (row["directory"] + ".csv"), index=False)
    fig, (profile_ax, width_ax) = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    half_step = float(row["dt"]) / 2
    image = profile_ax.imshow(z_profile, origin="lower", aspect="auto", interpolation="nearest",
                              extent=(-size/2, size/2, -half_step, time[-1]+half_step),
                              vmin=0, cmap="magma")
    fig.colorbar(image, ax=profile_ax, label="Bond singlet weight")
    profile_ax.set(xlabel="Bond midpoint relative to kick", ylabel="Age of kick",
                   title="One Z kick at phase zero")
    width_ax.plot(time, width.mean(axis=0))
    width_ax.set(xlabel="Age of kick", ylabel="Mean number of bonds ever reached",
                 title=f"All kick sites, axes and phases (L={size})")
    fig.savefig(output / (row["directory"] + ".png"), dpi=170)
    plt.close(fig)
    return kernel


def predict(kernel, time, p):
    phases, ages, size = kernel.shape
    if ages < len(time):
        raise ValueError("Quench exceeds single-seed calibration horizon")
    # h=1. Interpolate the growth kernel at the clean tower phase at birth.
    position = (time / (2*np.pi) % 1) * phases
    left = np.floor(position).astype(int) % phases
    weight = position - np.floor(position)
    exposure = np.zeros((len(time), size))
    for phase in range(phases):
        births = p * ((left == phase)*(1-weight) + ((left+1) % phases == phase)*weight)
        births[0] = 0  # Initial state is clean. First kick follows cycle 1.
        for bond in range(size):
            exposure[:, bond] += np.convolve(births, kernel[phase, :len(time), bond])[:len(time)]
    return (-np.expm1(-exposure)).mean(axis=1)


def compare(root, rows, threshold, kernel, output):
    row = rows[0]
    size, cycles = int(row["L"]), int(row["cycles"])
    dt, gamma = float(row["dt"]), float(row["gamma"])
    time = np.arange(cycles + 1) * dt
    p = -0.75 * np.expm1(-4 * gamma * dt)
    prediction = predict(kernel, time, p)
    totals = np.zeros((7, len(time)))
    count = 0
    for _, data in trajectories(root, rows):
        r = data[columns("r", size)].to_numpy()
        above = r >= threshold
        fraction = np.maximum.accumulate(above, axis=0).mean(axis=1)
        kicks = (data[columns("kick", size)].to_numpy() >= 0).sum(axis=1)
        totals += np.stack((fraction, fraction**2, above.mean(axis=1), r.mean(axis=1),
                            data.mx, data.my, kicks))
        count += 1
    f, f2, current, r_mean, mx, my, kicks = totals / count
    error = np.sqrt(np.maximum(0, f2 - f*f)/(count-1)) if count > 1 else np.full_like(f, np.nan)
    table = pd.DataFrame(dict(t=time, f_observed=f, f_sem=error, f_kjma=prediction,
                              f_current=current, mean_singlet=r_mean, mx=mx, my=my,
                              m_perp=np.hypot(mx, my), kicks_per_site_per_step=kicks/size,
                              expected_kicks_per_site_per_step=np.where(time > 0, p, 0)))
    name = row["directory"]
    table.to_csv(output / (name + ".csv"), index=False)
    fig, ax = plt.subplots(figsize=(6.2, 4), constrained_layout=True)
    ax.plot(time, f, label="Quench trajectories")
    ax.fill_between(time, f-error, f+error, alpha=.22, linewidth=0, label="±1 trajectory SEM")
    ax.plot(time, prediction, "--", label="KJMA from isolated kicks")
    ax.set(xlabel="Time", ylabel="Fraction of bonds ever reached", ylim=(-.02, 1.02),
           title=rf"$L={size},\ \gamma={gamma:g},\ r_\ast={threshold:g}$")
    ax.legend(frameon=False)
    fig.savefig(output / (name + ".png"), dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--threshold", type=float, default=0.125)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("threshold must lie strictly between 0 and 1")
    root = args.manifest.resolve().parent
    with args.manifest.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    output = root / (args.manifest.stem + "_r" + str(args.threshold))
    output.mkdir(exist_ok=True)
    for size in dict.fromkeys(row["L"] for row in rows):
        group = [row for row in rows if row["L"] == size]
        controls = [row for row in group if row["mode"] == "single"]
        kernel = calibrate(root, controls, args.threshold, output)
        quenches = [row for row in group if row["mode"] == "quench"]
        for directory in dict.fromkeys(row["directory"] for row in quenches):
            compare(root, [row for row in quenches if row["directory"] == directory],
                    args.threshold, kernel, output)
    print(f"Comparison tables and figures: {output}")


if __name__ == "__main__":
    main()
