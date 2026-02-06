"""Recipe manager - enforces the arrosticini composition sequence."""

from typing import List, Optional

from fm7000.config.constants import MeatType, RECIPE_SEQUENCE


class RecipeManager:
    """
    Manages the layer-by-layer recipe for cube filling.

    The recipe defines which meat type goes into each layer:
    HIGH_QUALITY -> FAT -> MEDIUM -> FAT -> MEDIUM -> FAT -> ... -> HIGH_QUALITY

    First and last layers are always high quality (first/last bite strategy).
    """

    def __init__(self, sequence: Optional[List[MeatType]] = None) -> None:
        self.sequence = sequence or list(RECIPE_SEQUENCE)
        self.current_layer: int = 0

    @property
    def current_meat_type(self) -> MeatType:
        if self.current_layer < len(self.sequence):
            return self.sequence[self.current_layer]
        idx = (self.current_layer - 1) % (len(self.sequence) - 2) + 1
        return self.sequence[idx]

    @property
    def is_first_layer(self) -> bool:
        return self.current_layer == 0

    @property
    def remaining_layers(self) -> int:
        return max(0, len(self.sequence) - self.current_layer)

    def advance_layer(self) -> MeatType:
        self.current_layer += 1
        return self.current_meat_type

    def get_required_type_for_layer(self, layer_index: int) -> MeatType:
        if layer_index < len(self.sequence):
            return self.sequence[layer_index]
        idx = (layer_index - 1) % (len(self.sequence) - 2) + 1
        return self.sequence[idx]

    def is_final_layer(self, estimated_height_mm: float, cube_height_mm: float = 250.0) -> bool:
        remaining_space = cube_height_mm - estimated_height_mm
        return remaining_space < 30.0

    def get_conveyor_lane(self) -> int:
        meat_type = self.current_meat_type
        if meat_type == MeatType.HIGH_QUALITY:
            return 0
        elif meat_type == MeatType.MEDIUM_QUALITY:
            return 1
        else:
            return 2

    def should_switch_to_final(self, current_height_mm: float, cube_height_mm: float = 250.0) -> bool:
        if self.current_meat_type == MeatType.HIGH_QUALITY:
            return False
        return self.is_final_layer(current_height_mm, cube_height_mm)

    def reset(self) -> None:
        self.current_layer = 0
