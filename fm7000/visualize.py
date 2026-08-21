"""Grafici della simulazione: mappa altezze, sezioni, copertura per strato."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fm7000.agent.hybrid_agent import CubeResult, HybridAgent
from fm7000.config.constants import CUBE
from fm7000.cube.state import CubeState

TYPE_COLORS = {
    "HIGH_QUALITY": "#c0392b",
    "MEDIUM_QUALITY": "#e67e22",
    "FAT": "#f1c40f",
}


def render_cube(
    cube_state: CubeState,
    result: CubeResult,
    out_path: Path,
    title: str = "FM 7000 - riempimento cubo",
) -> Path:
    """Scrive una tavola con altezze, sezioni, copertura e colonne di grasso."""
    res = cube_state.res
    height = cube_state.height_map
    extent = (0.0, cube_state.l * res, 0.0, cube_state.w * res)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    grid = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # --- vista dall'alto: altezza raggiunta cella per cella
    ax = fig.add_subplot(grid[0, 0])
    img = ax.imshow(
        height.T,
        origin="lower",
        extent=extent,
        cmap="turbo",
        vmin=0.0,
        vmax=CUBE.height_mm,
    )
    ax.set_title(f"altezza (vista dall'alto) - media {result.mean_height_mm:.0f} mm")
    ax.set_xlabel("Y [mm]")
    ax.set_ylabel("X [mm]")
    fig.colorbar(img, ax=ax, label="mm")

    # --- sezioni verticali al centro del cubo
    ax = fig.add_subplot(grid[0, 1])
    mid_x = cube_state.w // 2
    mid_y = cube_state.l // 2
    axis_x = np.arange(cube_state.l) * res
    axis_y = np.arange(cube_state.w) * res
    ax.fill_between(axis_x, height[mid_x, :], color="#c0392b", alpha=0.55, label="sezione X")
    ax.plot(axis_y, height[:, mid_y], color="#2c3e50", lw=1.8, label="sezione Y")
    ax.axhline(CUBE.height_mm, color="k", ls="--", lw=1.2, label="bordo cubo")
    ax.set_ylim(0, CUBE.height_mm * 1.05)
    ax.set_title(f"sezione verticale - planarita {result.flatness_mm:.0f} mm")
    ax.set_xlabel("mm")
    ax.set_ylabel("altezza [mm]")
    ax.legend(fontsize=8)

    # --- colonne di grasso
    ax = fig.add_subplot(grid[0, 2])
    fat = cube_state.get_fat_column_map()
    img = ax.imshow(fat.T, origin="lower", extent=extent, cmap="YlOrBr", vmin=0.0, vmax=1.0)
    ax.set_title(f"grasso per colonna - indice {result.fat_column_index * 100:.1f}%")
    ax.set_xlabel("Y [mm]")
    ax.set_ylabel("X [mm]")
    fig.colorbar(img, ax=ax, label="densita")

    # --- copertura per strato
    ax = fig.add_subplot(grid[1, 0:2])
    idx = [layer.index for layer in result.layers]
    cov = [layer.coverage * 100 for layer in result.layers]
    colors = [TYPE_COLORS.get(layer.meat_type, "#7f8c8d") for layer in result.layers]
    ax.bar(idx, cov, color=colors, edgecolor="#2c3e50")
    ax.axhline(
        CUBE.layer_coverage_threshold * 100,
        color="#27ae60",
        ls="--",
        label=f"soglia {CUBE.layer_coverage_threshold * 100:.0f}%",
    )
    for layer, value in zip(result.layers, cov, strict=False):
        ax.text(
            layer.index,
            value + 1.5,
            f"{value:.0f}%\n{layer.slices}f",
            ha="center",
            fontsize=8,
        )
    ax.set_ylim(0, 112)
    ax.set_title(
        f"copertura per strato - media {result.mean_layer_coverage * 100:.1f}%, "
        f"{result.total_slices} fette"
    )
    ax.set_xlabel("strato")
    ax.set_ylabel("copertura [%]")
    ax.legend(fontsize=8, loc="lower left")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=name)
        for name, color in TYPE_COLORS.items()
    ]
    ax.add_artist(ax.legend(handles=handles, fontsize=8, loc="lower right"))

    # --- riepilogo numerico
    ax = fig.add_subplot(grid[1, 2])
    ax.axis("off")
    lines = [
        f"strati pianificati: {result.planned_layers}",
        f"strati eseguiti:    {result.layer_count}",
        f"fette nel cubo:     {result.total_slices}",
        f"riempimento:        {result.fill_percentage * 100:.1f}%",
        f"altezza media:      {result.mean_height_mm:.0f} mm",
        f"altezza max:        {result.max_height_mm:.0f} mm",
        f"planarita:          {result.flatness_mm:.0f} mm",
        f"sovrapp. tollerate: {result.overlap_tolerated}",
        f"ciclo medio:        {result.avg_cycle_time:.2f} s",
        f"tempo cubo:         {result.total_time_sec:.0f} s",
        f"throughput:         {result.cubes_per_hour:.1f} cubi/ora",
        f"ricetta rispettata: {'si' if result.recipe_followed else 'no'}",
        "fette solo appoggiate: nessuna pressatura",
    ]
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        family="monospace",
        fontsize=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run(n_cubes: int = 1, seed: int = 0, out_dir: Path = Path("simulazione")) -> list[Path]:
    np.random.seed(seed)
    agent = HybridAgent(simulation=True)
    if not agent.initialize():
        raise RuntimeError("inizializzazione simulazione fallita")

    paths: list[Path] = []
    for i in range(n_cubes):
        result = agent.fill_cube()
        path = render_cube(
            agent.cube_state,
            result,
            out_dir / f"cubo_{i + 1}.png",
            title=f"FM 7000 - cubo {i + 1}",
        )
        paths.append(path)
        print(f"{path}: copertura {result.mean_layer_coverage * 100:.1f}%")

    agent.shutdown()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Grafici simulazione FM 7000")
    parser.add_argument("-n", "--cubes", type=int, default=1)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-o", "--out", type=Path, default=Path("simulazione"))
    args = parser.parse_args()
    run(n_cubes=args.cubes, seed=args.seed, out_dir=args.out)


if __name__ == "__main__":
    main()
