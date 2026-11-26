r"""TikZ drawings of one period of the Floquet circuit (floquet-circuit.md, rust/src/floquet.rs).

    python plot_scripts/floquet_circuit_tikz.py [out_dir] [N ...]     default: latex/figures, N = 8 10 12 14

Writes floquet_circuit_N<N>.tex (a tikzpicture, \input by latex/sections/floquet.tex; needs the colors caRA, caAR,
caARA, caB, caE and the TikZ libraries of latex/preamble.tex). Standard library only.

Schedule (the same rule as `floquet::layer_count`, `floquet::layer_starts`): d = smallest divisor of N that is
>= 4; layer U_l (l = 1 ... d) has blocks starting at j = l, l + d, ... (sites 1 ... N), block (j, j+1, j+2, j+3).
Time runs upward: U_d at the bottom (acts first), U_1 above it, then the field row (B on every site), then the noise
row (E on every site, acts last). Inside a layer the rows are, from the top: RA (sites j+1 ... j+3), AR (j ... j+2),
ARA (j ... j+3), the three summands of one block exponential. Capsules that cross the boundary N | 1 are cut at the
two edges of the drawing (dashed ends).
"""

import sys
from pathlib import Path

ROW = 0.42  # row height (TikZ units; one unit = one wire spacing)
GAP = 0.55  # vertical gap between layers
HALF = 0.16  # half height of a capsule
PAD = 0.36  # horizontal padding of a capsule around its outer wires
TERMS = [("RA", 1, 3, "caRA"), ("AR", 0, 2, "caAR"), ("ARA", 0, 3, "caARA")]  # name, first, last offset, color


def layer_count(n):
    return next(d for d in range(4, n + 1) if n % d == 0)


def layer_starts(n, layer):
    d = layer_count(n)
    return list(range(layer, n + 1, d))


def segments(n, first, last):
    """Sites first ... last (may exceed n) as contiguous pieces in 1 ... n: (a, b, cut_left, cut_right)."""
    sites = [(s - 1) % n + 1 for s in range(first, last + 1)]
    pieces, start = [], 0
    for i in range(1, len(sites) + 1):
        if i == len(sites) or sites[i] != sites[i - 1] + 1:
            pieces.append((sites[start], sites[i - 1]))
            start = i
    return [(a, b, k > 0, k < len(pieces) - 1) for k, (a, b) in enumerate(pieces)]


def capsule(lines, a, b, cut_l, cut_r, y, color, label, n):
    x0 = 0.45 if cut_l else a - PAD
    x1 = n + 0.55 if cut_r else b + PAD
    y0, y1 = y - HALF, y + HALF
    if not cut_l and not cut_r:
        lines.append(rf"\draw[{color}, fill={color}!22, rounded corners=3pt, line width=0.6pt] "
                     rf"({x0:.2f},{y0:.2f}) rectangle ({x1:.2f},{y1:.2f});")
    else:
        lines.append(rf"\fill[{color}!22] ({x0:.2f},{y0:.2f}) rectangle ({x1:.2f},{y1:.2f});")
        lines.append(rf"\draw[{color}, line width=0.6pt] ({x0:.2f},{y0:.2f}) -- ({x1:.2f},{y0:.2f}) "
                     rf"({x0:.2f},{y1:.2f}) -- ({x1:.2f},{y1:.2f});")
        for x, cut in ((x0, cut_l), (x1, cut_r)):
            style = ", dashed" if cut else ""
            lines.append(rf"\draw[{color}, line width=0.6pt{style}] ({x:.2f},{y0:.2f}) -- ({x:.2f},{y1:.2f});")
    lines.append(rf"\node[font=\scriptsize\sffamily, text={color}!70!black] at ({(x0 + x1) / 2:.2f},{y:.2f}) {{{label}}};")


def circuit(n):
    d = layer_count(n)
    height = d * 3 * ROW + (d - 1) * GAP
    y_field = height + GAP + ROW / 2
    y_noise = y_field + 1.6 * ROW
    top = y_noise + ROW
    lines = [rf"% One period of the Floquet circuit, N = {n}, d = {d} interaction layers, then the uniform field and"
             rf" the noise (plot_scripts/floquet_circuit_tikz.py).",
             r"\begin{tikzpicture}[x=0.78cm, y=1cm]"]
    # wires and site numbers
    for s in range(1, n + 1):
        lines.append(rf"\draw[black!35, line width=0.5pt] ({s},-0.25) -- ({s},{top:.2f});")
        lines.append(rf"\node[font=\scriptsize, below] at ({s},-0.25) {{{s}}};")
    lines.append(rf"\node[font=\scriptsize, below] at ({(n + 1) / 2:.2f},-0.6) {{site}};")
    # time arrow
    lines.append(rf"\draw[-latex, line width=0.6pt] (-2.35,0) -- (-2.35,{top:.2f}) "
                 rf"node[midway, rotate=90, above, font=\scriptsize] {{time}};")
    for layer in range(d, 0, -1):
        base = (d - layer) * (3 * ROW + GAP)
        y_row = {"ARA": base + 0.5 * ROW, "AR": base + 1.5 * ROW, "RA": base + 2.5 * ROW}
        lines.append(rf"\draw[decorate, decoration={{brace, amplitude=4pt}}, line width=0.6pt] "
                     rf"(-0.95,{base + 0.05:.2f}) -- (-0.95,{base + 3 * ROW - 0.05:.2f}) "
                     rf"node[midway, left=5pt, font=\small] {{$U_{{{layer}}}$}};")
        for name, y in y_row.items():
            lines.append(rf"\node[font=\tiny\sffamily, anchor=east, text=black!60] at (0.45,{y:.2f}) {{{name}}};")
        covered = set()
        for j in layer_starts(n, layer):
            covered.update((j - 1 + q) % n + 1 for q in range(4))
            for name, first, last, color in TERMS:
                for a, b, cut_l, cut_r in segments(n, j + first, j + last):
                    capsule(lines, a, b, cut_l, cut_r, y_row[name], color, name, n)
        for s in sorted(set(range(1, n + 1)) - covered):
            lines.append(rf"\draw[black!55, dashed, rounded corners=2pt, line width=0.5pt] "
                         rf"({s - 0.3:.2f},{base + 0.03:.2f}) rectangle ({s + 0.3:.2f},{base + 3 * ROW - 0.03:.2f});")
            lines.append(rf"\node[font=\tiny\sffamily, rotate=90, text=black!60] at ({s},{base + 1.5 * ROW:.2f}) {{idle}};")
    # field row: B on every site, after U_1
    for s in range(1, n + 1):
        lines.append(rf"\node[circle, draw=caB, fill=caB!25, line width=0.6pt, inner sep=0pt, minimum size=9pt, "
                     rf"font=\tiny\sffamily, text=caB!70!black] at ({s},{y_field:.2f}) {{B}};")
    lines.append(rf"\node[font=\small, anchor=east] at (-1.05,{y_field:.2f}) {{$U_B$}};")
    lines.append(rf"\node[font=\scriptsize, anchor=east] at (0.45,{y_field:.2f}) {{field}};")
    # noise row: E on every site, last
    for s in range(1, n + 1):
        lines.append(rf"\node[draw=caE, fill=caE!18, line width=0.6pt, rounded corners=1pt, inner sep=1.5pt, "
                     rf"minimum width=13pt, font=\tiny] at ({s},{y_noise:.2f}) {{$\mathcal{{E}}_{{{s}}}$}};")
    lines.append(rf"\node[font=\small, anchor=east] at (-1.05,{y_noise:.2f}) {{$\mathcal{{E}}$}};")
    lines.append(rf"\node[font=\scriptsize, anchor=east] at (0.45,{y_noise:.2f}) {{noise}};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def main():
    args = sys.argv[1:]
    default = Path(__file__).resolve().parent.parent / "latex/figures"
    out = Path(args.pop(0)) if args and not args[0].isdigit() else default
    sizes = [int(a) for a in args] or [8, 10, 12, 14]
    for n in sizes:
        d = layer_count(n)
        # every site starts exactly one block per period, so every bond is a middle bond exactly once
        assert sorted(j for layer in range(1, d + 1) for j in layer_starts(n, layer)) == list(range(1, n + 1))
        path = out / f"floquet_circuit_N{n}.tex"
        path.write_text(circuit(n), encoding="utf-8")
        print(f"N = {n}: d = {d}, starts " + ", ".join(f"U{l}: {layer_starts(n, l)}" for l in range(1, d + 1)),
              f"-> {path}")


if __name__ == "__main__":
    main()
