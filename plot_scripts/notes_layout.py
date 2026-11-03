"""
Layout rules for the figures of latex/notes.tex (shared by the plot scripts).

  MAX_PANELS   at most 4 panels per figure: a longer list of panels is split over several files
               <stem>_1.png, <stem>_2.png, ...
  GAMMA_MIN    the smallest γ > 0 shown or fitted in the notes (no γ < 0.01: no size L ≤ 22 reaches the
               thermodynamic rate there); γ = 0 is kept where a script shows it
"""

MAX_PANELS = 4
GAMMA_MIN = 0.01


def keep_gamma(g, zero=True):
    """True for γ ≥ GAMMA_MIN, and for γ = 0 if zero."""
    return (zero and g == 0) or g >= GAMMA_MIN * (1 - 1e-9)


def pages(items, n=MAX_PANELS):
    """Split a list of panels into consecutive pages of at most n."""
    items = list(items)
    return [items[i:i + n] for i in range(0, len(items), n)]


def page_name(stem, k, n_pages, ext=".png"):
    """File name of page k (1-based) of n_pages: <stem>.png for a single page, else <stem>_<k>.png."""
    return f"{stem}{ext}" if n_pages == 1 else f"{stem}_{k}{ext}"


def grid_shape(n):
    """Rows and columns for n ≤ 4 panels: 1×1, 1×2, 1×3, 2×2."""
    return {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2)}[n]
