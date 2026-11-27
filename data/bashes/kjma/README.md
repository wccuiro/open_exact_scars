# Single-seed growth and the noisy scar quench on Leonardo

This experiment adapts the isolated-seed and full-quench comparison of [Gribben, Lesanovsky and Gutiérrez, Phys. Rev. A 97, 011603(R)](https://doi.org/10.1103/PhysRevA.97.011603) to the Floquet scar chain. Both arms use the same periodic Hamiltonian realization at each size, with coupling seed `1000 + L`, field $h=1$, and time step `DT`.

## Run

Extract the archive on a filesystem shared by the login and compute nodes. The archive contains the Rust source and its locked dependencies. The submission host needs Cargo and a C linker. From the extracted `leonardo_kjma` directory, run

```bash
bash data/bashes/kjma/submit.sh plan
bash data/bashes/kjma/submit.sh submit
```

The submit command builds the example, writes a fresh manifest, prints its path, and submits one Slurm array. By default files go under `data/kjma_sweep/`. Set `OUT` to an absolute shared scratch path to write there. A repeat submission with the same parameters and output directory reuses finished trajectories. Each gzip file is published only when the trajectory exits successfully. An interrupted trajectory restarts from its initial state.

| Setting | Default |
| --- | --- |
| `SIZES` | `12 16 20 24` |
| `GAMMAS` | `0.0001 0.0005 0.001 0.0025 0.005 0.01 0.025 0.05 0.1` |
| `DT` | `0.1` |
| `PHASES` | 8 evenly spaced initial azimuths on the clean scar orbit |
| `HISTORIES` | 224 independent noisy trajectories per size and rate |
| `CPUS` | 112 single-threaded trajectories per full array element |
| `ACCOUNT`, `PARTITION` | `sis26_collura`, `dcgp_usr_prod` |

`CYCLES` overrides the horizon for both experiments. By default each quench runs for the ceiling of `max(2/sqrt(gamma), 3/(gamma*L))/DT` cycles. Each size's controls cover its longest quench. This choice sets the duration of data collection. The prediction uses the measured growth throughout. No trajectory stops at a threshold.

## What a trajectory records

- **Single kick.** Start on the clean precessing orbit, apply one Pauli operator on one site at age zero, then evolve unitarily. Run every site, each of X, Y and Z, and each initial phase. A kick along the initial spin can have zero footprint and contributes its actual weight to the average.
- **Quench.** Start from the clean minus-X product state. After each Floquet step, kick each site independently with probability `p = 3(1-exp(-4 gamma DT))/4`, choosing X, Y or Z uniformly. Here gamma is the Lindblad coefficient of each Pauli. The mean recorded event rate per site is `p/DT`. It approaches `3 gamma` as the step becomes short.
- Both arms save the singlet weight `rj = <R_{j,j+1}>` on every bond, the kick axis on every site (`-1` none, `0,1,2` for X,Y,Z), and the spatial means of the Pauli X and Y operators. Row zero is immediately after the isolated kick or immediately before the noisy quench starts. Output is a gzip CSV per trajectory. Each size uses one disorder realization.

## Analyze

Run the following with the path printed by **submit**, once the trajectories listed there have finished. Analysis needs Python with NumPy, pandas and Matplotlib. It can run locally after copying the manifest and trajectory directories together.

```bash
python3 data/bashes/kjma/analyze.py /path/to/plan.XXXXXXXX.tsv --threshold 0.125
```

A bond is *reached* at its first sampled crossing of the specified singlet weight. The default 1/8 is halfway between the clean value zero and the infinite-temperature value 1/4. This trajectory diagnostic assumes the Pauli-kick unraveling and involves no intermediate projective measurements. Repeating analysis with another threshold reuses the raw trajectories and writes a separate result directory.

The single-kick figure shows a space–time map of the actual singlet weight after a Z kick, aligned at the kicked site and averaged over birth sites. Its second panel shows the mean number of distinct bonds ever reached by an isolated kick, including both propagation directions. The calibration table also records this width by initial phase and the fraction of seeds reaching either diametrically opposite bond. Once fronts reach the opposite side the measured width is affected by the ring size.

For target bond $j$, the analysis computes a growth kernel $K_j(a,\phi)$ at age $a$ by summing its single-seed reach indicator over all source sites and averaging over the three axes. Its spatial mean is the full seed width. At quench step $n$ the one-dimensional KJMA prediction is

$$
Y_j(n)=p\sum_{m=1}^{n} K_j((n-m)\,\Delta t,\,h m\Delta t),
\qquad
f_{\rm KJMA}(n)=\frac{1}{L}\sum_j [1-e^{-Y_j(n)}].
$$

The kernel is interpolated periodically in the birth phase. The exponential is the Poisson independent-growth approximation to the discrete noisy quench. It allows footprints to overlap and uses no fitted velocity, nucleation rate or growth exponent. Later kicks can alter a front in the actual quench, which is precisely what the comparison tests.

Each quench figure overlays this prediction and the measured ever-reached fraction with one standard error over trajectories. Its CSV also contains the instantaneous fraction above threshold, mean singlet weight, actual and expected kicks per site per step, and transverse magnetization computed from the ensemble-averaged X and Y components. Comparing that magnetization with the independently measured coverage tests whether the two decay on the same timescale.
