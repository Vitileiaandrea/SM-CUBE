"""
Test script for filling 100 cubes with slices from the 1000-slice database.

Collects statistics:
- Fill percentage per cube
- Slices skipped (too large for remaining space)
- Slices recovered (skipped slices used later)
- Average fill percentage
- Total slices used
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from meat_packing_agent.env.cube_environment import CubeState, MeatSlice
from meat_packing_agent.data.slice_database import SliceDatabase


@dataclass
class CubeStats:
    cube_id: int
    fill_percentage: float
    slices_placed: int
    slices_skipped: int
    slices_recovered: int
    layers_completed: int
    final_layer_coverage: float


def create_slice_from_db(slice_data: Dict, slice_id: int) -> MeatSlice:
    """Create a MeatSlice object from database entry."""
    slice_obj = MeatSlice(
        width=slice_data['width'],
        length=slice_data['length'],
        thickness_min=slice_data['thickness_min'],
        thickness_max=slice_data['thickness_max'],
        slice_id=slice_id
    )
    
    if 'shape_points' in slice_data:
        h = int(slice_data['width'] / 5)
        w = int(slice_data['length'] / 5)
        mask = np.zeros((h, w), dtype=np.float32)
        
        points = slice_data['shape_points']
        center_x, center_y = h / 2, w / 2
        
        for i in range(h):
            for j in range(w):
                px = (i - center_x) / center_x if center_x > 0 else 0
                py = (j - center_y) / center_y if center_y > 0 else 0
                
                inside = True
                for k in range(len(points)):
                    p1 = points[k]
                    p2 = points[(k + 1) % len(points)]
                    
                    if ((p1[1] > py) != (p2[1] > py)) and \
                       (px < (p2[0] - p1[0]) * (py - p1[1]) / (p2[1] - p1[1] + 1e-10) + p1[0]):
                        inside = not inside
                
                if inside:
                    mask[i, j] = 1.0
        
        if np.sum(mask) < 10:
            slice_obj.shape_mask = slice_obj._generate_irregular_shape(0.2)
        else:
            slice_obj.shape_mask = mask
    else:
        slice_obj.shape_mask = slice_obj._generate_irregular_shape(0.2)
    
    slice_obj.thickness_map = slice_obj._generate_thickness_map()
    return slice_obj


def fill_cube_with_slices(cube: CubeState, slices: List[Dict], max_attempts: int = 500) -> CubeStats:
    """
    Fill a single cube using slices from the database.
    
    Uses circular conveyor logic: skipped slices go back to the queue.
    """
    cube.reset()
    
    slice_queue = list(range(len(slices)))
    skipped_queue = []
    
    slices_placed = 0
    slices_skipped = 0
    slices_recovered = 0
    attempts = 0
    consecutive_skips = 0
    max_consecutive_skips = 50
    
    while attempts < max_attempts and consecutive_skips < max_consecutive_skips:
        if cube.get_fill_percentage() >= 95:
            break
        
        if not slice_queue and not skipped_queue:
            break
        
        if not slice_queue:
            slice_queue = skipped_queue
            skipped_queue = []
            if not slice_queue:
                break
        
        slice_idx = slice_queue.pop(0)
        slice_data = slices[slice_idx]
        slice_obj = create_slice_from_db(slice_data, slice_idx)
        
        layer_coverage = cube.get_layer_coverage()
        if layer_coverage >= cube.LAYER_COVERAGE_THRESHOLD:
            cube.press_layer()
        
        best_pos = None
        best_rotation = 0
        best_score = float('inf')
        is_floor_placement = False
        
        for rotation in range(4):
            rotated = slice_obj.rotate(rotation * 90)
            result = cube.find_perimeter_first_position(rotated, floor_only=True)
            x, y, height, is_floor = result[0], result[1], result[2], result[3]
            
            if x >= 0 and y >= 0 and is_floor:
                can_place_result, _ = cube.can_place(rotated, x, y)
                if can_place_result:
                    score = cube._calculate_gap_score(rotated, x, y, height)
                    if score < best_score:
                        best_score = score
                        best_pos = (x, y)
                        best_rotation = rotation
                        is_floor_placement = True
        
        if best_pos is None:
            skipped_queue.append(slice_idx)
            slices_skipped += 1
            consecutive_skips += 1
            attempts += 1
            continue
        
        if slice_idx in [s for s in range(len(slices)) if s not in slice_queue]:
            pass
        
        rotated_slice = slice_obj.rotate(best_rotation * 90)
        success, metrics = cube.place_slice(rotated_slice, best_pos[0], best_pos[1])
        
        if success:
            slices_placed += 1
            consecutive_skips = 0
            
            if slice_idx in skipped_queue:
                slices_recovered += 1
        else:
            skipped_queue.append(slice_idx)
            slices_skipped += 1
            consecutive_skips += 1
        
        attempts += 1
    
    return CubeStats(
        cube_id=0,
        fill_percentage=cube.get_fill_percentage(),
        slices_placed=slices_placed,
        slices_skipped=slices_skipped,
        slices_recovered=slices_recovered,
        layers_completed=cube.layers_completed,
        final_layer_coverage=cube.get_layer_coverage()
    )


def run_100_cube_test():
    """Run the full test: fill 100 cubes and collect statistics."""
    print("=" * 60)
    print("SM-CUBE Test: Filling 100 cubes with 1000-slice database")
    print("=" * 60)
    
    db = SliceDatabase()
    try:
        db.load()
    except FileNotFoundError:
        print("Database not found, generating new one...")
        db.generate_database(num_slices=1000, seed=42)
        db.save()
    
    all_slices = [s.to_dict() for s in db.slices]
    print(f"\nLoaded {len(all_slices)} slices from database")
    
    all_stats: List[CubeStats] = []
    
    for cube_id in range(100):
        cube = CubeState(width=210.0, length=210.0, height=250.0, resolution=5.0)
        
        np.random.seed(cube_id * 42)
        shuffled_indices = np.random.permutation(len(all_slices))
        shuffled_slices = [all_slices[i] for i in shuffled_indices]
        
        stats = fill_cube_with_slices(cube, shuffled_slices)
        stats.cube_id = cube_id
        all_stats.append(stats)
        
        if (cube_id + 1) % 10 == 0:
            avg_fill = np.mean([s.fill_percentage for s in all_stats])
            print(f"  Cubes {cube_id + 1}/100 completed - Avg fill: {avg_fill:.1f}%")
    
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    fill_percentages = [s.fill_percentage for s in all_stats]
    slices_placed_list = [s.slices_placed for s in all_stats]
    slices_skipped_list = [s.slices_skipped for s in all_stats]
    slices_recovered_list = [s.slices_recovered for s in all_stats]
    layers_completed_list = [s.layers_completed for s in all_stats]
    
    print(f"\n--- PERCENTUALE DI RIEMPIMENTO ---")
    print(f"  Media: {np.mean(fill_percentages):.2f}%")
    print(f"  Min: {np.min(fill_percentages):.2f}%")
    print(f"  Max: {np.max(fill_percentages):.2f}%")
    print(f"  Std Dev: {np.std(fill_percentages):.2f}%")
    
    print(f"\n--- FETTINE POSIZIONATE ---")
    print(f"  Media per cubo: {np.mean(slices_placed_list):.1f}")
    print(f"  Totale: {np.sum(slices_placed_list)}")
    
    print(f"\n--- FETTINE SCARTATE (troppo grandi) ---")
    print(f"  Media per cubo: {np.mean(slices_skipped_list):.1f}")
    print(f"  Totale: {np.sum(slices_skipped_list)}")
    
    print(f"\n--- FETTINE RECUPERATE (dal nastro circolare) ---")
    print(f"  Media per cubo: {np.mean(slices_recovered_list):.1f}")
    print(f"  Totale: {np.sum(slices_recovered_list)}")
    
    print(f"\n--- STRATI COMPLETATI ---")
    print(f"  Media per cubo: {np.mean(layers_completed_list):.1f}")
    print(f"  Max: {np.max(layers_completed_list)}")
    
    results = {
        "test_summary": {
            "total_cubes": 100,
            "total_slices_in_database": len(all_slices),
        },
        "fill_percentage": {
            "mean": float(np.mean(fill_percentages)),
            "min": float(np.min(fill_percentages)),
            "max": float(np.max(fill_percentages)),
            "std": float(np.std(fill_percentages)),
        },
        "slices_placed": {
            "mean_per_cube": float(np.mean(slices_placed_list)),
            "total": int(np.sum(slices_placed_list)),
        },
        "slices_skipped": {
            "mean_per_cube": float(np.mean(slices_skipped_list)),
            "total": int(np.sum(slices_skipped_list)),
        },
        "slices_recovered": {
            "mean_per_cube": float(np.mean(slices_recovered_list)),
            "total": int(np.sum(slices_recovered_list)),
        },
        "layers_completed": {
            "mean_per_cube": float(np.mean(layers_completed_list)),
            "max": int(np.max(layers_completed_list)),
        },
        "per_cube_stats": [
            {
                "cube_id": s.cube_id,
                "fill_percentage": s.fill_percentage,
                "slices_placed": s.slices_placed,
                "slices_skipped": s.slices_skipped,
                "slices_recovered": s.slices_recovered,
                "layers_completed": s.layers_completed,
            }
            for s in all_stats
        ]
    }
    
    output_path = Path(__file__).parent / "test_results_100_cubes.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    results = run_100_cube_test()
