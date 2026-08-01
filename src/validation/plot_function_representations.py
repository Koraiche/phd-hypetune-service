"""Generate Table-3-style 3D surface plot "representations" of the synthetic
validation functions (article_draft.md Section 3.8), mirroring [11]'s own
Table 3 (Sphere/Goldstein-Price/FindValues, with a Representation column of
3D surface plots).

FindValues has no plot here, exactly as in [11]'s Table 3: it's a 5-D
squared-error function against a fixed target vector, not a 2-variable
surface, and there is no honest 2D/3D "representation" of it to draw.

Usage:
    python src/validation/plot_function_representations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3d projection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validation.synthetic_functions import goldstein_price, sphere  # noqa: E402

CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"


def _plot_surface(fn, lb: float, ub: float, title: str, out_path: Path, n: int = 80) -> None:
    x = np.linspace(lb, ub, n)
    y = np.linspace(lb, ub, n)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[fn(np.array([xi, yi])) for xi in x] for yi in y])

    fig = plt.figure(figsize=(5, 4.2), dpi=150)
    fig.patch.set_facecolor(CHART_SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(CHART_SURFACE)
    # Rainbow colormap (matches [11]'s own Figure/Table 3 surface plots),
    # by request — deliberately overriding this project's usual single-hue
    # sequential convention for these two figures specifically.
    surf = ax.plot_surface(X, Y, Z, cmap="jet", linewidth=0, antialiased=True, rcount=n, ccount=n)
    ax.set_xlabel("x1", color=SECONDARY_INK, fontsize=9)
    ax.set_ylabel("x2", color=SECONDARY_INK, fontsize=9)
    ax.set_zlabel("f(x1, x2)", color=SECONDARY_INK, fontsize=9)
    ax.tick_params(colors=SECONDARY_INK, labelsize=7)
    ax.set_title(title, fontsize=11, color=PRIMARY_INK)
    fig.colorbar(surf, shrink=0.55, aspect=12, pad=0.12)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    figures_dir = repo_root / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_surface(sphere, -10, 10, "Sphere", figures_dir / "function_repr_sphere.png")
    _plot_surface(goldstein_price, -2, 2, "Goldstein-Price", figures_dir / "function_repr_goldstein_price.png")
    print("[repr] wrote function_repr_sphere.png, function_repr_goldstein_price.png")


if __name__ == "__main__":
    main()
