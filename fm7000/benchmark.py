"""Banco di prova: riempie N cubi in simulazione e misura le prestazioni."""

import argparse
import time

import numpy as np

from fm7000.agent.hybrid_agent import CubeResult, HybridAgent


def run_benchmark(n_cubes: int = 5, seed: int = 0, verbose: bool = True) -> list[CubeResult]:
    np.random.seed(seed)
    agent = HybridAgent(simulation=True)
    if not agent.initialize():
        raise RuntimeError("inizializzazione simulazione fallita")

    results: list[CubeResult] = []
    for i in range(n_cubes):
        started = time.time()
        result = agent.fill_cube()
        results.append(result)
        if verbose:
            _print_cube(i, result, time.time() - started)

    agent.shutdown()
    if verbose:
        _print_summary(results)
    return results


def _print_cube(index: int, r: CubeResult, wall_sec: float) -> None:
    print(f"\n=== CUBO {index + 1} ===")
    print(
        f"strati pianificati: {r.planned_layers} | eseguiti: {r.layer_count} | "
        f"fette: {r.total_slices}"
    )
    print("strato  tipo            copertura  fette  spessore  altezza")
    for layer in r.layers:
        print(
            f"{layer.index:>6}  {layer.meat_type:<14} "
            f"{layer.coverage * 100:>8.1f}%  {layer.slices:>5}  "
            f"{layer.thickness_mm:>7.1f}mm  {layer.height_mm:>6.1f}mm"
        )
    print(
        f"copertura media strati: {r.mean_layer_coverage * 100:.1f}% | "
        f"riempimento: {r.fill_percentage * 100:.1f}% | "
        f"altezza media: {r.mean_height_mm:.1f}mm (max {r.max_height_mm:.1f}mm)"
    )
    print(
        f"planarita (dev.std): {r.flatness_mm:.1f}mm | "
        f"colonne di grasso: {r.fat_column_index * 100:.1f}% | "
        f"sovrapposizioni tollerate: {r.overlap_tolerated}"
    )
    print(
        f"ciclo medio: {r.avg_cycle_time:.2f}s | tempo cubo: {r.total_time_sec:.1f}s | "
        f"{r.cubes_per_hour:.1f} cubi/ora | ricetta rispettata: "
        f"{'si' if r.recipe_followed else f'no ({r.recipe_violations} deroghe)'}"
    )
    print(f"(calcolo simulazione: {wall_sec:.1f}s)")


def _print_summary(results: list[CubeResult]) -> None:
    coverage = [r.mean_layer_coverage for r in results]
    per_layer = [layer.coverage for r in results for layer in r.layers]
    print("\n=== RIEPILOGO ===")
    print(f"cubi simulati: {len(results)}")
    print(
        f"copertura per strato: media {np.mean(per_layer) * 100:.1f}% | "
        f"min {np.min(per_layer) * 100:.1f}% | max {np.max(per_layer) * 100:.1f}%"
    )
    print(f"copertura media per cubo: {np.mean(coverage) * 100:.1f}%")
    print(f"riempimento: {np.mean([r.fill_percentage for r in results]) * 100:.1f}%")
    print(f"strati per cubo: {np.mean([r.layer_count for r in results]):.1f}")
    print(f"fette per cubo: {np.mean([r.total_slices for r in results]):.1f}")
    print(f"planarita: {np.mean([r.flatness_mm for r in results]):.1f}mm")
    print(
        f"colonne di grasso: {np.mean([r.fat_column_index for r in results]) * 100:.1f}%"
    )
    print(f"ciclo medio: {np.mean([r.avg_cycle_time for r in results]):.2f}s")
    print(f"throughput: {np.mean([r.cubes_per_hour for r in results]):.1f} cubi/ora")
    print(
        f"ricetta rispettata: {sum(1 for r in results if r.recipe_followed)}/{len(results)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Banco di prova FM 7000")
    parser.add_argument("-n", "--cubes", type=int, default=5)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()
    run_benchmark(n_cubes=args.cubes, seed=args.seed)


if __name__ == "__main__":
    main()
