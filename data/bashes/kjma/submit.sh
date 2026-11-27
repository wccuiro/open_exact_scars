#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
mode=${1:-plan}
case "$mode" in plan|submit) ;; *) echo "usage: bash submit.sh [plan|submit]" >&2; exit 2;; esac
sizes=${SIZES:-"12 16 20 24"}
gammas=${GAMMAS:-"0.0001 0.0005 0.001 0.0025 0.005 0.01 0.025 0.05 0.1"}
dt=${DT:-0.1}
histories=${HISTORIES:-224}
phases=${PHASES:-8}
cpus=${CPUS:-112}
out=${OUT:-"$repo/data/kjma_sweep"}
positive='^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
for n in "$histories" "$phases" "$cpus" "${CYCLES:-1}"; do
    [[ $n =~ ^[1-9][0-9]*$ ]] || { echo "expected positive integer: $n" >&2; exit 2; }
done
(( cpus <= 112 ))
[[ $dt =~ $positive ]] && awk -v d="$dt" 'BEGIN {exit !(d>0)}' || exit 2
mkdir -p "$out"
out=$(cd "$out" && pwd)
manifest=$(mktemp "$out/plan.XXXXXXXX.tsv")
printf 'mode\tL\tgamma\tdt\tcycles\tfirst\tcount\tphases\tdirectory\n' > "$manifest"
batches() {
    local kind=$1 g=$2 cycles=$3 total=$4 dir=$5 start count
    for ((start=1; start<=total; start+=cpus)); do
        count=$((total-start+1))
        if ((count > cpus)); then count=$cpus; fi
        printf '%s\t%d\t%s\t%s\t%d\t%d\t%d\t%d\t%s\n' \
            "$kind" "$l" "$g" "$dt" "$cycles" "$start" "$count" "$phases" "$dir" >> "$manifest"
    done
}
for l in $sizes; do
    [[ $l =~ ^[0-9]+$ ]] && (( l >= 4 && l <= 24 && l % 4 == 0 )) || { echo "sizes: multiples of 4 from 4 to 24" >&2; exit 2; }
    longest=0
    for g in $gammas; do
        [[ $g =~ $positive ]] || exit 2
        cycles=$(awk -v l="$l" -v g="$g" -v dt="$dt" 'BEGIN {
            if (g<=0) exit 2
            t=2/sqrt(g); if (3/(g*l)>t) t=3/(g*l)
            n=t/dt; print int(n)+(n>int(n))
        }')
        cycles=${CYCLES:-$cycles}
        if ((cycles > longest)); then longest=$cycles; fi
        batches quench "$g" "$cycles" "$histories" "quench_L${l}_g${g}_dt${dt}_n${cycles}"
    done
    ((longest > 0))
    batches single 0 "$longest" "$((3*l*phases))" "single_L${l}_dt${dt}_n${longest}_p${phases}"
done
tasks=$(($(wc -l < "$manifest") - 1))
(( tasks > 0 ))
echo "Manifest $manifest"
echo "$tasks array elements, up to $cpus trajectories per element"
if [[ $mode == submit ]]; then
    # Use a portable binary even when the login and compute CPUs differ.
    (cd "$repo/rust" && RUSTFLAGS="${RUSTFLAGS:--C target-cpu=x86-64}" cargo build --locked --release --example floquet_traj_bond)
    sbatch --array="0-$((tasks-1))" --cpus-per-task="$cpus" \
        --account="${ACCOUNT:-sis26_collura}" --partition="${PARTITION:-dcgp_usr_prod}" \
        --output="$out/slurm-%A_%a.out" --error="$out/slurm-%A_%a.err" \
        "$here/run.sh" "$repo" "$manifest" "$out"
fi
