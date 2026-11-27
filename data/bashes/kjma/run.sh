#!/usr/bin/env bash
#SBATCH --job-name=kjma
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
set -euo pipefail
export LC_ALL=C
repo=$1; manifest=$2; out=$3
row=$(sed -n "$((SLURM_ARRAY_TASK_ID+2))p" "$manifest")
[[ -n $row ]]
IFS=$'\t' read -r kind l g dt cycles first count phases subdir <<< "$row"
bin="$repo/rust/target/release/examples/floquet_traj_bond"
dir="$out/$subdir"
mkdir -p "$dir"
export bin dir kind l g dt cycles phases RAYON_NUM_THREADS=1
run_seed() (
    set -euo pipefail
    seed=$1
    final="$dir/seed_$seed.csv.gz"
    [[ ! -f $final ]] || exit 0
    temp=$(mktemp "$dir/.seed_${seed}.XXXXXXXX")
    trap 'rm -f "$temp" "$temp.log"' EXIT
    args=("--$kind" "$l" "$g" "$dt" "$cycles" "$seed")
    if [[ $kind == single ]]; then
        index=$((seed-1))
        phase=$(awk -v i="$((index/(3*l)))" -v n="$phases" 'BEGIN {printf "%.17g", 8*atan2(1,1)*i/n}')
        args+=("$((index/3%l))" "$((index%3))" "$phase")
    fi
    "$bin" "${args[@]}" 2>"$temp.log" | gzip -1 > "$temp" || { cat "$temp.log" >&2; exit 1; }
    mv -f "$temp" "$final"
)
export -f run_seed
seq "$first" "$((first+count-1))" | xargs -P "${SLURM_CPUS_PER_TASK:-1}" -n 1 bash -c 'run_seed "$1"' _
echo "Completed $count trajectories in $dir starting at seed $first"
