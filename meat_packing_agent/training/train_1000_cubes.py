#!/usr/bin/env python3
"""
Training script for SM-CUBE meat packing agent.
Trains on 1000 cubes with analysis every 10 cubes.
Saves results to file for persistence.
"""

import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from datetime import datetime
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from meat_packing_agent.env.cube_environment import CubeState, MeatSlice


@dataclass
class SliceData:
    id: int
    cut_type: str
    width: float
    length: float
    thickness_min: float
    thickness_max: float
    shape: str
    irregularity: float


@dataclass
class CubeResult:
    cube_id: int
    fill_percentage: float
    slices_used: int
    slices_discarded: int
    layers_completed: int
    avg_height_mm: float
    max_height_mm: float
    time_seconds: float


@dataclass
class BatchReport:
    batch_id: int
    cubes_start: int
    cubes_end: int
    avg_fill: float
    min_fill: float
    max_fill: float
    total_slices_used: int
    total_slices_discarded: int
    discard_rate: float
    cubes_above_95: int
    cubes_90_to_95: int
    cubes_below_90: int
    timestamp: str


class MeatPackingTrainer:
    def __init__(self):
        self.database_path = Path(__file__).parent.parent / "data" / "slices_10000.json"
        self.results_path = Path(__file__).parent / "training_results.json"
        self.slices: List[SliceData] = []
        self.all_results: List[CubeResult] = []
        self.batch_reports: List[BatchReport] = []
        self.slice_index = 0
        
        self.load_database()
        self.load_previous_results()
    
    def load_database(self):
        """Load the 10,000 slice database."""
        print(f"Caricamento database da {self.database_path}...")
        with open(self.database_path, 'r') as f:
            data = json.load(f)
        self.slices = [SliceData(**s) for s in data]
        print(f"Caricate {len(self.slices)} fettine")
    
    def load_previous_results(self):
        """Load any previous training results."""
        if self.results_path.exists():
            print(f"Caricamento risultati precedenti da {self.results_path}...")
            with open(self.results_path, 'r') as f:
                data = json.load(f)
            self.all_results = [CubeResult(**r) for r in data.get('cube_results', [])]
            self.batch_reports = [BatchReport(**b) for b in data.get('batch_reports', [])]
            self.slice_index = data.get('slice_index', 0)
            print(f"Caricati {len(self.all_results)} risultati precedenti")
    
    def save_results(self):
        """Save current results to file."""
        data = {
            'cube_results': [asdict(r) for r in self.all_results],
            'batch_reports': [asdict(b) for b in self.batch_reports],
            'slice_index': self.slice_index,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.results_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def fill_single_cube(self, cube_id: int, return_cube: bool = False):
        """
        Fill a single cube using a PER-CUBE slice set.
        
        Each cube gets its own set of slices (e.g., 60 slices).
        The algorithm tries to use ALL of them for this cube.
        Slices that don't fit are truly discarded (not passed to next cube).
        
        Args:
            cube_id: The cube ID (used for reproducible random seed)
            return_cube: If True, returns (CubeResult, CubeState) tuple
                        If False, returns just CubeResult (backward compatible)
        """
        start_time = time.time()
        
        cube = CubeState(width=210.0, length=210.0, height=250.0, resolution=5.0)
        cube.reset()
        
        # PER-CUBE SLICE SET: Each cube gets 60 random slices
        # This matches the user's requirement: "le fettine le deve usare per il cubo che sta facendo"
        np.random.seed(cube_id * 42)  # Reproducible but different for each cube
        cube_slice_indices = np.random.choice(len(self.slices), size=60, replace=False)
        cube_slices = [self.slices[i] for i in cube_slice_indices]
        
        slices_used = 0
        slices_discarded = 0
        retry_list = []  # (slice_data, retry_count)
        max_retries = 5  # More retries since we want to use all slices
        slice_idx = 0
        max_iterations = 500
        consecutive_failures = 0
        
        for iteration in range(max_iterations):
            # Check if cube is full
            avg_height = np.mean(cube.height_map) * cube.resolution
            if avg_height >= 245:
                break
            
            # Press layer if needed
            if cube.should_press_layer():
                cube.press_layer()
                # Reset retry counts when layer changes (slices get another chance)
                new_retry_list = [(s, 0) for s, _ in retry_list]
                retry_list = new_retry_list
                consecutive_failures = 0
            
            # Get next slice (prefer retry list, then new slices)
            current_slice = None
            retries = 0
            
            if retry_list:
                current_slice, retries = retry_list.pop(0)
            elif slice_idx < len(cube_slices):
                current_slice = cube_slices[slice_idx]
                slice_idx += 1
                retries = 0
            else:
                # No more slices available for this cube
                break
            
            if current_slice is None:
                break
            
            # Try to place the slice
            slice_obj = MeatSlice(
                width=current_slice.width,
                length=current_slice.length,
                thickness_min=current_slice.thickness_min,
                thickness_max=current_slice.thickness_max,
                slice_id=current_slice.id
            )
            
            placed = False
            
            # AGGRESSIVE OVERLAP MODE: Try floor_only first, then allow overlap
            coverage = cube.get_layer_coverage()
            floor_only = coverage < 0.7  # Only strict floor for first 70% coverage
            
            for rot in range(4):
                rotated = slice_obj.rotate(rot * 90)
                result = cube.find_perimeter_first_position(rotated, floor_only=floor_only)
                x, y, height, is_floor = result
                
                if x >= 0 and y >= 0:
                    success, _ = cube.place_slice(rotated, x, y)
                    if success:
                        slices_used += 1
                        placed = True
                        consecutive_failures = 0
                        break
            
            if not placed:
                consecutive_failures += 1
                if retries < max_retries:
                    retry_list.append((current_slice, retries + 1))
                else:
                    slices_discarded += 1
                
                # If too many consecutive failures, force press layer
                if consecutive_failures > 10 and cube.get_layer_coverage() > 0.8:
                    cube.press_layer()
                    consecutive_failures = 0
        
        # Count remaining retries as discarded
        slices_discarded += len(retry_list)
        
        # Calculate final metrics using ACTUAL VOLUME (not heightmap which is inflated by press)
        # The heightmap-based calculation gives 100% because press_layer sets all cells to floor level
        # The correct calculation uses total_volume_filled which tracks actual slice volume
        fill_pct = cube.get_fill_percentage()
        
        elapsed = time.time() - start_time
        
        result = CubeResult(
            cube_id=cube_id,
            fill_percentage=fill_pct,
            slices_used=slices_used,
            slices_discarded=slices_discarded,
            layers_completed=cube.layers_completed,
            avg_height_mm=np.mean(cube.height_map) * cube.resolution,
            max_height_mm=np.max(cube.height_map) * cube.resolution,
            time_seconds=elapsed
        )
        
        if return_cube:
            return result, cube
        return result
    
    def analyze_batch(self, batch_id: int, results: List[CubeResult]) -> BatchReport:
        """Analyze a batch of 10 cubes."""
        fills = [r.fill_percentage for r in results]
        total_used = sum(r.slices_used for r in results)
        total_discarded = sum(r.slices_discarded for r in results)
        
        return BatchReport(
            batch_id=batch_id,
            cubes_start=batch_id * 10 + 1,
            cubes_end=(batch_id + 1) * 10,
            avg_fill=np.mean(fills),
            min_fill=np.min(fills),
            max_fill=np.max(fills),
            total_slices_used=total_used,
            total_slices_discarded=total_discarded,
            discard_rate=total_discarded / (total_used + total_discarded) * 100 if (total_used + total_discarded) > 0 else 0,
            cubes_above_95=sum(1 for f in fills if f >= 95),
            cubes_90_to_95=sum(1 for f in fills if 90 <= f < 95),
            cubes_below_90=sum(1 for f in fills if f < 90),
            timestamp=datetime.now().isoformat()
        )
    
    def print_batch_report(self, report: BatchReport):
        """Print a batch report."""
        print(f"\n{'='*70}")
        print(f"BATCH {report.batch_id + 1} (Cubi {report.cubes_start}-{report.cubes_end})")
        print(f"{'='*70}")
        print(f"Riempimento: Media {report.avg_fill:.1f}% | Min {report.min_fill:.1f}% | Max {report.max_fill:.1f}%")
        print(f"Fettine: Usate {report.total_slices_used} | Scartate {report.total_slices_discarded} | Scarto {report.discard_rate:.1f}%")
        print(f"Distribuzione: >=95%: {report.cubes_above_95} | 90-95%: {report.cubes_90_to_95} | <90%: {report.cubes_below_90}")
    
    def train(self, total_cubes: int = 1000, batch_size: int = 10):
        """Train on specified number of cubes."""
        start_cube = len(self.all_results)
        
        print(f"\n{'#'*70}")
        print(f"ADDESTRAMENTO SM-CUBE: {total_cubes} cubi totali")
        print(f"Inizio da cubo {start_cube + 1}")
        print(f"{'#'*70}")
        
        num_batches = total_cubes // batch_size
        start_batch = start_cube // batch_size
        
        for batch_id in range(start_batch, num_batches):
            batch_results = []
            
            for i in range(batch_size):
                cube_id = batch_id * batch_size + i
                
                # Skip if already done
                if cube_id < len(self.all_results):
                    batch_results.append(self.all_results[cube_id])
                    continue
                
                result = self.fill_single_cube(cube_id)
                batch_results.append(result)
                self.all_results.append(result)
                
                # Print progress
                print(f"  Cubo {cube_id + 1}: {result.fill_percentage:.1f}% | {result.slices_used} fettine | {result.time_seconds:.1f}s")
            
            # Analyze and report batch
            report = self.analyze_batch(batch_id, batch_results)
            self.batch_reports.append(report)
            self.print_batch_report(report)
            
            # Save after each batch
            self.save_results()
            print(f"Risultati salvati in {self.results_path}")
        
        # Final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print final training summary."""
        print(f"\n{'#'*70}")
        print(f"RIEPILOGO FINALE ADDESTRAMENTO")
        print(f"{'#'*70}")
        
        all_fills = [r.fill_percentage for r in self.all_results]
        total_used = sum(r.slices_used for r in self.all_results)
        total_discarded = sum(r.slices_discarded for r in self.all_results)
        
        print(f"\nStatistiche globali ({len(self.all_results)} cubi):")
        print(f"  Riempimento medio: {np.mean(all_fills):.1f}%")
        print(f"  Riempimento min/max: {np.min(all_fills):.1f}% / {np.max(all_fills):.1f}%")
        print(f"  Deviazione standard: {np.std(all_fills):.2f}%")
        print(f"  Fettine totali usate: {total_used}")
        print(f"  Fettine totali scartate: {total_discarded}")
        print(f"  Tasso scarto: {total_discarded/(total_used+total_discarded)*100:.1f}%")
        
        excellent = sum(1 for f in all_fills if f >= 95)
        good = sum(1 for f in all_fills if 90 <= f < 95)
        poor = sum(1 for f in all_fills if f < 90)
        
        print(f"\nDistribuzione riempimento:")
        print(f"  Eccellente (>=95%): {excellent} cubi ({excellent/len(all_fills)*100:.1f}%)")
        print(f"  Buono (90-95%): {good} cubi ({good/len(all_fills)*100:.1f}%)")
        print(f"  Scarso (<90%): {poor} cubi ({poor/len(all_fills)*100:.1f}%)")


if __name__ == "__main__":
    trainer = MeatPackingTrainer()
    trainer.train(total_cubes=1000, batch_size=10)
