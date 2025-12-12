"""
Database of 1000 diverse meat slices for AI training.

This module generates and stores a variety of meat slice configurations
with different shapes, sizes, and thicknesses to train the AI operator
to handle real-world variability in meat slices.
"""

import numpy as np
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class SliceTemplate:
    """Template for a meat slice with all its properties."""
    slice_id: int
    width: float
    length: float
    thickness_min: float
    thickness_max: float
    shape_type: str
    irregularity: float
    shape_seed: int
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class SliceDatabase:
    """Database of diverse meat slice templates for AI training."""
    
    SHAPE_TYPES = [
        'oval',
        'irregular_oval',
        'elongated',
        'wide',
        'triangular',
        'asymmetric',
        'curved',
        'natural'
    ]
    
    def __init__(self, db_path: Optional[str] = None):
        self.slices: List[SliceTemplate] = []
        self.db_path = db_path or self._default_db_path()
        
    def _default_db_path(self) -> str:
        """Get default database path."""
        module_dir = Path(__file__).parent
        return str(module_dir / "slices_1000.json")
    
    def generate_database(self, num_slices: int = 1000, seed: int = 42):
        """
        Generate a database of diverse meat slices.
        
        Creates slices with realistic distributions of:
        - Sizes (80-200mm width/length)
        - Thicknesses (15-25mm typical, some 10-40mm extreme)
        - Shapes (various irregular patterns)
        - Irregularity levels (0.05 to 0.35)
        """
        np.random.seed(seed)
        self.slices = []
        
        for i in range(num_slices):
            slice_template = self._generate_slice(i)
            self.slices.append(slice_template)
        
        print(f"Generated {len(self.slices)} diverse meat slice templates")
        return self.slices
    
    def _generate_slice(self, slice_id: int) -> SliceTemplate:
        """Generate a single slice with realistic properties."""
        
        category = np.random.choice([
            'small', 'medium', 'large', 'extra_large',
            'thin', 'thick', 'wedge', 'irregular'
        ], p=[0.15, 0.35, 0.25, 0.10, 0.05, 0.03, 0.04, 0.03])
        
        if category == 'small':
            width = np.random.uniform(80, 110)
            length = np.random.uniform(80, 110)
            thickness_min = np.random.uniform(15, 20)
            thickness_max = np.random.uniform(18, 25)
            irregularity = np.random.uniform(0.1, 0.2)
            
        elif category == 'medium':
            width = np.random.uniform(100, 150)
            length = np.random.uniform(100, 150)
            thickness_min = np.random.uniform(15, 20)
            thickness_max = np.random.uniform(20, 25)
            irregularity = np.random.uniform(0.1, 0.25)
            
        elif category == 'large':
            width = np.random.uniform(140, 180)
            length = np.random.uniform(140, 180)
            thickness_min = np.random.uniform(18, 22)
            thickness_max = np.random.uniform(22, 28)
            irregularity = np.random.uniform(0.15, 0.3)
            
        elif category == 'extra_large':
            width = np.random.uniform(170, 200)
            length = np.random.uniform(170, 200)
            thickness_min = np.random.uniform(20, 25)
            thickness_max = np.random.uniform(25, 35)
            irregularity = np.random.uniform(0.2, 0.35)
            
        elif category == 'thin':
            width = np.random.uniform(100, 160)
            length = np.random.uniform(100, 160)
            thickness_min = np.random.uniform(10, 15)
            thickness_max = np.random.uniform(12, 18)
            irregularity = np.random.uniform(0.1, 0.2)
            
        elif category == 'thick':
            width = np.random.uniform(100, 150)
            length = np.random.uniform(100, 150)
            thickness_min = np.random.uniform(25, 30)
            thickness_max = np.random.uniform(30, 40)
            irregularity = np.random.uniform(0.1, 0.2)
            
        elif category == 'wedge':
            width = np.random.uniform(100, 160)
            length = np.random.uniform(100, 160)
            thickness_min = np.random.uniform(5, 12)
            thickness_max = np.random.uniform(20, 35)
            irregularity = np.random.uniform(0.15, 0.25)
            
        else:
            width = np.random.uniform(80, 200)
            length = np.random.uniform(80, 200)
            thickness_min = np.random.uniform(10, 25)
            thickness_max = np.random.uniform(15, 35)
            irregularity = np.random.uniform(0.2, 0.35)
        
        aspect_ratio = np.random.uniform(0.6, 1.4)
        if aspect_ratio > 1:
            length = width * aspect_ratio
        else:
            width = length / aspect_ratio
        
        width = np.clip(width, 80, 200)
        length = np.clip(length, 80, 200)
        
        if thickness_max < thickness_min:
            thickness_min, thickness_max = thickness_max, thickness_min
        
        shape_type = np.random.choice(self.SHAPE_TYPES, p=[
            0.20,
            0.25,
            0.15,
            0.10,
            0.08,
            0.10,
            0.07,
            0.05
        ])
        
        shape_seed = np.random.randint(0, 1000000)
        
        return SliceTemplate(
            slice_id=slice_id,
            width=float(width),
            length=float(length),
            thickness_min=float(thickness_min),
            thickness_max=float(thickness_max),
            shape_type=shape_type,
            irregularity=float(irregularity),
            shape_seed=shape_seed
        )
    
    def save(self, path: Optional[str] = None):
        """Save database to JSON file."""
        save_path = path or self.db_path
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            'version': '1.0',
            'num_slices': len(self.slices),
            'slices': [s.to_dict() for s in self.slices]
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.slices)} slices to {save_path}")
    
    def load(self, path: Optional[str] = None):
        """Load database from JSON file."""
        load_path = path or self.db_path
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        self.slices = [SliceTemplate.from_dict(s) for s in data['slices']]
        print(f"Loaded {len(self.slices)} slices from {load_path}")
        return self.slices
    
    def get_slice(self, slice_id: int) -> Optional[SliceTemplate]:
        """Get a specific slice by ID."""
        if 0 <= slice_id < len(self.slices):
            return self.slices[slice_id]
        return None
    
    def get_random_slice(self) -> SliceTemplate:
        """Get a random slice from the database."""
        return np.random.choice(self.slices)
    
    def get_slices_by_size(self, min_width: float, max_width: float) -> List[SliceTemplate]:
        """Get slices within a size range."""
        return [s for s in self.slices if min_width <= s.width <= max_width]
    
    def get_slices_by_shape(self, shape_type: str) -> List[SliceTemplate]:
        """Get slices of a specific shape type."""
        return [s for s in self.slices if s.shape_type == shape_type]
    
    def get_statistics(self) -> dict:
        """Get statistics about the database."""
        if not self.slices:
            return {}
        
        widths = [s.width for s in self.slices]
        lengths = [s.length for s in self.slices]
        thicknesses_min = [s.thickness_min for s in self.slices]
        thicknesses_max = [s.thickness_max for s in self.slices]
        irregularities = [s.irregularity for s in self.slices]
        
        shape_counts = {}
        for s in self.slices:
            shape_counts[s.shape_type] = shape_counts.get(s.shape_type, 0) + 1
        
        return {
            'total_slices': len(self.slices),
            'width': {
                'min': min(widths),
                'max': max(widths),
                'mean': np.mean(widths),
                'std': np.std(widths)
            },
            'length': {
                'min': min(lengths),
                'max': max(lengths),
                'mean': np.mean(lengths),
                'std': np.std(lengths)
            },
            'thickness_min': {
                'min': min(thicknesses_min),
                'max': max(thicknesses_min),
                'mean': np.mean(thicknesses_min)
            },
            'thickness_max': {
                'min': min(thicknesses_max),
                'max': max(thicknesses_max),
                'mean': np.mean(thicknesses_max)
            },
            'irregularity': {
                'min': min(irregularities),
                'max': max(irregularities),
                'mean': np.mean(irregularities)
            },
            'shape_distribution': shape_counts
        }


def create_default_database():
    """Create and save the default 1000-slice database."""
    db = SliceDatabase()
    db.generate_database(num_slices=1000, seed=42)
    db.save()
    
    stats = db.get_statistics()
    print("\nDatabase Statistics:")
    print(f"  Total slices: {stats['total_slices']}")
    print(f"  Width range: {stats['width']['min']:.1f} - {stats['width']['max']:.1f} mm")
    print(f"  Length range: {stats['length']['min']:.1f} - {stats['length']['max']:.1f} mm")
    print(f"  Thickness range: {stats['thickness_min']['min']:.1f} - {stats['thickness_max']['max']:.1f} mm")
    print(f"  Shape distribution: {stats['shape_distribution']}")
    
    return db


if __name__ == "__main__":
    create_default_database()
